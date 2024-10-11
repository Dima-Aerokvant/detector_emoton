import cv2                                                                                                                                  #импорт библиотеки OpenCV
import numpy as np

def stackImages(scale,imgArray):                                                                                                            #функция умной сшивки, на вход которой подаются масштаб и двумерный массив изображений
    rows = len(imgArray)                                                                                                                    #определение количества строк итогового изображения
    cols = len(imgArray[0])                                                                                                                 #определение количества столбцов итогового изображения
    rowsAvailable = isinstance(imgArray[0], list)                                                                                           #определение, больше ли одной строки в массиве изображений
    width = imgArray[0][0].shape[1]                                                                                                         #определение ширины первого из передаваемых изображений
    height = imgArray[0][0].shape[0]                                                                                                        #определение высоты первого из передаваемых изображений
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

img = cv2.imread('Resources/lena.png')                                                                          #создание объекта img с использованием команды чтения изображения из папки Resources
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                                                  #создание объекта imgGray как img в оттенках серого

imgStack = stackImages(0.5,([img,imgGray,img],[img,img,img]))                                                   #применение функции стыковки

cv2.imshow("ImageStack",imgStack)                                                                               #вывод результата - состыкованных изображений


# imgHor = np.hstack((img,img))
# imgVer = np.vstack((img,img))
#
# cv2.imshow("Horizontal",imgHor)
# cv2.imshow("Vertical",imgVer)

cv2.waitKey(0)