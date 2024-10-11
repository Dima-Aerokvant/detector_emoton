import cv2                                              #импорт библиотеки OpenCV
import numpy as np                                      #импорт библиотеки numpy

def stackImages(scale,imgArray):                        #функция стыковки изображений
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):                                                                       #функция выделения контуров и классификации геометрических фигур
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)      #функция нахождения контуров, подробнее: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0,
                                                                                            #запись всех контуров в объект contours
    for cnt in contours:                                                                    #цикл, проходящий по всем найденным контурам
        area = cv2.contourArea(cnt)                                                         #вычисление площади контура. Зачастую полезно, так как позволяет отсечь шумы, случайно найденные контуры.
        print(area)                                                                         #вывод площади текущего контура
        if area>500:                                                                        #если площадь более 500 пикселей, то считаем контур "полезным"
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)                           #imgContours - изображение, на котором рисуем, cnt - текущий контур, -1 - рисуем все контуры, (255, 0, 0) - цвет, 3 - толщина. Подробнее: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
            peri = cv2.arcLength(cnt,True)                                                  #вычисление длины (периметра) текущего контура, True - контур замкнут
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)                                   #аппроксимация текущего контура до контура, с меньшим количеством вершин. 0.02*peri - параметр точности, максимальное расстояние от контура до аппроксимируемого контура, нужно выбирать с умом.
                                                                                            #подробнее: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            print(len(approx))                                                              #количество элементов аппроксимированного контура, указывает на количество вершин
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)                                           #параметры обводящего аппроксимированный контур прямоугольника

            if objCor ==3: objectType ="Tri"                                                #определение типа контура, если 3 вершины - то треугольник
            elif objCor == 4:                                                               #если 4 вершины
                aspRatio = w/float(h)                                                       #то ищем соотношение сторон обводящего прямоугольника
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"                  #если оно близко к 1, то квадрат
                else:objectType="Rectangle"                                                 #если нет - то прямоугольник
            elif objCor>4: objectType= "Circle"                                             #если вершин более 4, то круг
            else:objectType="None"                                                          #в остальных случаях присваиваем None, так как не можем классифицировать контур



            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)                           #рисуем обводящий прямоугольник
            cv2.putText(imgContour,objectType,                                              #а также подписываем тип контура
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)




path = 'Resources/shapes.png'                                       #путь к файлу
img = cv2.imread(path)                                              #создание объекта img с использованием команды чтения изображения
imgContour = img.copy()                                             #создание копии изображения, на которой можно рисовать

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                      #перевод изображения в оттенки серого
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)                         #размытие изображения
imgCanny = cv2.Canny(imgBlur,50,50)                                 #применение фильтра Кэнни для выделения границ
getContours(imgCanny)                                               #применение функции получения и рисования контуров

imgBlank = np.zeros_like(img)                                       #создание черного изображения, по размерам равного img
imgStack = stackImages(0.7,([img,imgGray,imgBlur],                  #стыковка нескольких изебражений
                            [imgCanny,imgContour,imgBlank]))

cv2.imshow("Stack", imgStack)                                       #ывод состыкованных изображений

cv2.waitKey(0)                                                      #команда для того, чтобы изображение не закрывалось
