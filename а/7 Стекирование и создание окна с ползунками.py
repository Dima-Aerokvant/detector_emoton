import cv2                                              #импорт библиотеки OpenCV
import numpy as np                                      #импорт библиотеки numpy


def nothing(args):                                      #создание пустой функции, которая ничего не делает
    pass

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


# создаем окно для отображения результата и бегунки
cv2.namedWindow("TrackBars")                                            #создание окна с именем "TrackBars", которое будет содержать бегунки для изменения параметров RGB
cv2.resizeWindow("TrackBars",640,240)                                   #изменение его размера
cv2.createTrackbar("BlueLow", "TrackBars", 0, 255, nothing)             #создание бегунка с нижним значением синей составляющей, обязательно указание функции, которая будет вызываться при изменении
                                                                        #значения бегунка. Поэтому пришлось создать пустую функцию nothing
cv2.createTrackbar("BlueHigh", "TrackBars", 255, 255, nothing)          #создание бегунка с верхним значением синей составляющей
cv2.createTrackbar("GreenLow", "TrackBars", 0, 255, nothing)            #создание бегунка с нижним значением зеленой составляющей
cv2.createTrackbar("GreenHigh", "TrackBars", 255, 255, nothing)         #создание бегунка с верхним значением зеленой составляющей
cv2.createTrackbar("RedLow", "TrackBars", 0, 255, nothing)              #создание бегунка с нижним значением красной составляющей
cv2.createTrackbar("RedHigh", "TrackBars", 255, 255, nothing)           #создание бегунка с верхним значением красной составляющей

path = 'Resources/lena.png'                                            #путь к файлу

img = cv2.imread(path)                                                  #создание объекта img с использованием команды чтения изображения из папки Resources

while True:                                                             #вечный цикл
    BlueLow = cv2.getTrackbarPos('BlueLow', 'TrackBars')                #получение текущего значения нижней границы синего
    BlueHigh = cv2.getTrackbarPos('BlueHigh', 'TrackBars')              #получение текущего значения верхней границы синего
    GreenLow = cv2.getTrackbarPos('GreenLow', 'TrackBars')              #получение текущего значения нижней границы зеленого
    GreenHigh = cv2.getTrackbarPos('GreenHigh', 'TrackBars')            #получение текущего значения верхней границы зеленого
    RedLow = cv2.getTrackbarPos('RedLow', 'TrackBars')                  #получение текущего значения нижней границы красного
    RedHigh = cv2.getTrackbarPos('RedHigh', 'TrackBars')                #получение текущего значения верхней границы красного
    min_p = (GreenLow, BlueLow, RedLow)                                 #создание кортежа минимальных значений GBR
    max_p = (GreenHigh, BlueHigh, RedHigh)                              #создание кортежа минимальных значений GBR
    masking = cv2.inRange(img, min_p, max_p)                            #создание маски, при этом маска бинарна
    imgResult = cv2.bitwise_and(img, img, mask=masking)                 #создание результирующего изображения, которое пропускает маска

    imgStack = stackImages(0.6, ([img, masking, imgResult]))            #сшитие исходного изображения, маски и результирующего изображения
    cv2.imshow("Stacked Images", imgStack)                              #вывод сшитых изображений на экран


    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()