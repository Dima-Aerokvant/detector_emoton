import cv2                                                      #импорт библиотеки OpenCV
import numpy as np                                              #импорт библиотеки numpy

img = cv2.imread("Resources/lena.png")                          #создание объекта img с использованием команды чтения изображения из папки Resources
kernel = np.ones((5,5),np.uint8)                                #создание единичной матрицы размером 5 на 5 с элементами типа uint8 (беззнаковый int восьмибитный)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                  #конвертация изображения в оттенки серого
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)                     #размытие (сглаживание) изображения (фильтр Гаусса), используется для избавления от "шумов".
                                                                # "7 на 7" - размер ядра (значения положительные, целые, нечетные. 0 - стандартное отклонение
                                                                # (при значении 0 вычисляется из размера ядра), подробнее: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
imgCanny = cv2.Canny(img,150,200)                               #применение фильтра Кэнни, выделяющего границы (на основе градиента, т.е. степени различия соседних пикселей)
                                                                #150 и 200 - это пороговые значения, настраивают "чувствительность", подробнее: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)         #расширение границ, полученных фильтром Кэнни (для получения линии в случае, если фильтр Кэнни дает отдельные точки)
                                                                #чем больше размер kernel, тем больше будет расширение. iterations - это количество итераций (чем больше, тем больше будет расширение)
imgEroded = cv2.erode(imgDialation,kernel,iterations=1)         #эррозия расширенного изображения, чтоб уменьшить толщину линий, подробнее: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html

cv2.imshow("Original Image",img)                                #вывод оригинального изображения
cv2.imshow("Gray Image",imgGray)                                #вывод изображения в оттенках серого
cv2.imshow("Blur Image",imgBlur)                                #вывод размытого изображения
cv2.imshow("Canny Image",imgCanny)                              #вывод изображения, обработанного фильтром Кэнни (границ)
cv2.imshow("Dialation Image",imgDialation)                      #вывод расширенного изображения границ
cv2.imshow("Eroded Image",imgEroded)                            #вывод эродированного изображения расширенных границ
cv2.waitKey(0)                                                  #команда для того, чтобы изображения не закрывались