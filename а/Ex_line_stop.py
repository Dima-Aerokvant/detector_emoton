import sim  # импорт модуля sim

# --------------
import sys  # импорт модуля sys
import cv2  # импорт библиотеки OpenCV
import numpy as np  # импорт библиотеки numpy

# --------------

sim.simxFinish(-1)  # на всякий случай закрываем все открытые соединения с Коппелией

clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # строки инициализации подключения к CoppeliaSim

if clientID != -1:  # check if client connection successful             #строки инициализации подключения к CoppeliaSim
    print('Connected to remote API server')  # строки инициализации подключения к CoppeliaSim

else:  # строки инициализации подключения к CoppeliaSim
    print('Connection not successful')  # строки инициализации подключения к CoppeliaSim
    sys.exit('Could not connect')  # строки инициализации подключения к CoppeliaSim

error, Left_wheel = sim.simxGetObjectHandle(clientID, 'Left_joint',
                                            sim.simx_opmode_oneshot_wait)  # создаем объект левого колеса
error, Right_wheel = sim.simxGetObjectHandle(clientID, 'Right_joint',
                                             sim.simx_opmode_oneshot_wait)  # создаем объект правого колеса

# -------------------
error, vis_cam = sim.simxGetObjectHandle(clientID, 'cam', sim.simx_opmode_oneshot_wait)  # создаем объект камеры
# -------------------

def getContours(img):                                                                       #функция выделения контуров и классификации геометрических фигур
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)      #функция нахождения контуров, подробнее: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0,
                                                                                            #запись всех контуров в объект contours
    #print('Контуры', contours)
    counter = 1
    for cnt in contours:                                                                    #цикл, проходящий по всем найденным контурам
        area = cv2.contourArea(cnt)                                                         #вычисление площади контура. Зачастую полезно, так как позволяет отсечь шумы, случайно найденные контуры.
        #print('Contour Area = ',area)                                                                         #вывод площади текущего контура
        if area>500:                                                                        #если площадь более 500 пикселей, то считаем контур "полезным"
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 1)                           #imgContours - изображение, на котором рисуем, cnt - текущий контур, -1 - рисуем все контуры, (255, 0, 0) - цвет, 3 - толщина. Подробнее: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
            peri = cv2.arcLength(cnt,True)                                                  #вычисление длины (периметра) текущего контура, True - контур замкнут
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)                                   #аппроксимация текущего контура до контура, с меньшим количеством вершин. 0.02*peri - параметр точности, максимальное расстояние от контура до аппроксимируемого контура, нужно выбирать с умом.
                                                                                            #подробнее: https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
            #print('Number of pts = ', len(approx))                                                              #количество элементов аппроксимированного контура, указывает на количество вершин
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)                                           #параметры обводящего аппроксимированный контур прямоугольника

            #print('x = ', x , 'y = ' , y , 'w = ' , w , 'h = ' , h)

            if objCor ==3: objectType ="Tri"                                                #определение типа контура, если 3 вершины - то треугольник
            elif objCor == 4:                                                               #если 4 вершины
                aspRatio = w/float(h)                                                       #то ищем соотношение сторон обводящего прямоугольника
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"                  #если оно близко к 1, то квадрат
                else:objectType="Rectangle"                                                 #если нет - то прямоугольник
            elif objCor>4: objectType= "Circle"                                            #если вершин более 4, то круг
            else:objectType="None"                                                          #в остальных случаях присваиваем None, так как не можем классифицировать контур
            objColor = imgContour[y+(h//2), x+(w//2)]
            # print('img = ', img)
            #print('Color = ', objColor)

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),1)                           #рисуем обводящий прямоугольник
            cv2.putText(imgContour,objectType+str(counter),                                              #а также подписываем тип контура
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.2,
                        (255,255,255),1)
            counter = counter+1
            cv2.putText(imgContour, str(objColor),                                               # а также подписываем цвет контура
                        (x + (w // 2) - 10, y + (h // 2) + 10), cv2.FONT_HERSHEY_COMPLEX, 0.2,
                        (255, 255, 255), 1)
            return objectType, objColor
    return 0, [0, 0, 0]





w1 = 0.5  # скорость левого колеса
w2 = 0.5  # скорость правого колеса

while 1:  # вечный цикл
    err, resolution, image = sim.simxGetVisionSensorImage(clientID, vis_cam, 0,
                                                          sim.simx_opmode_oneshot_wait)  # считываем изображение в объект image
    # print(err, resolution, type(image))
    img = np.array(image, ndmin=1, dtype=np.uint8)  # преобразуем считанное изображение в фармат OpenCV
    # img = img.reshape(resolution[0], resolution[1], 3)
    img = np.reshape(img, (resolution[0], resolution[1], 3))  # преобразуем считанное изображение в фармат OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # меняем цветовую палитру с RGB на BGR для OpenCV

    # --------------
    img = cv2.rotate(img, cv2.ROTATE_180)  # переворачиваем изображение для корректного отображения
    flip_image = cv2.flip(img, 1)  # переворачиваем изображение для корректного отображения
    cv2.imshow('Original Image', flip_image)  # выводим изображение в окно
    gray = cv2.cvtColor(flip_image, cv2.COLOR_BGR2GRAY)  # получаем изображение в оттенках серого

    if cv2.waitKey(1) == 27:  # если нажата клавиша esc - выходим из цикла
        break
    ret, threshold_image = cv2.threshold(gray, 127, 255,
                                         0)  # бинаризуем изображение. Все что выше 127 становится равно 255, все что ниже - 0
    cv2.imshow('Image', threshold_image)  # выводим бинаризованное изображение
    print('Изображение = ', threshold_image)

    string_search = threshold_image[250]  # будем отслеживать линию на 250 строке
    pixel_last = 255  # вспомогательная переменная
    counter = 0  # переменная-счетчик
    for pixel in string_search:  # в цикле находим левую и правую границы черной линии
        if (pixel_last == 255) and (pixel == 0):
            left_border = counter
        if (pixel_last == 0) and (pixel == 255):
            right_border = counter - 1
        pixel_last = pixel
        counter = counter + 1
    middle_point = (right_border + left_border) / 2  # находим среднюю точку как среднее между левой и правой границами

    if (middle_point > 117) and (middle_point < 137):  # если точка примерно посередине
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, w1,
                                               sim.simx_opmode_oneshot_wait)  # едем прямо, скорости колес равны
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, w2, sim.simx_opmode_oneshot_wait)
    elif (middle_point < 117):  # если точка ушла влево
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, 0, sim.simx_opmode_oneshot_wait)
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, w2 * 2,
                                               sim.simx_opmode_oneshot_wait)  # подворачиваем влево
    elif (middle_point > 137):  # если точка ушла вправо
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, w1 * 2,
                                               sim.simx_opmode_oneshot_wait)  # подворачиваем вправо
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, 0, sim.simx_opmode_oneshot_wait)

    imgContour = flip_image.copy()  # создание копии изображения, на которой можно рисовать
    # print('Image Shape = ', imgContour.shape)

    imgCut = imgContour[:, 1:127]  # смотрим только верхнюю половину изображения
    imgGray = cv2.cvtColor(imgCut, cv2.COLOR_BGR2GRAY)  # перевод изображения в оттенки серого
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)  # размытие изображения
    imgCanny = cv2.Canny(imgBlur, 50, 50)  # применение фильтра Кэнни для выделения границ
    objectT, objectC = getContours(imgCanny)  # применение функции получения и рисования контуров
    cv2.imshow('Image Contours', imgContour)  # выводим контур на экран
    print("ЦВЕТА:", objectC)  # выводим цвет на экран
    if (objectT == 'Circle') and (
            (objectC[0] < 20) and (objectC[1] < 20) and (objectC[2] > 120)):  # Если объект круг, и цвет его красный
        print('УРАААА')  # пишем ура
        w1 = 0  # И останавливаем колеса
        w2 = 0


cv2.destroyAllWindows()
# --------------