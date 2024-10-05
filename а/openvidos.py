# import cv2                                                #импорт библиотеки OpenCV
# print("OpenCV version : {0}".format(cv2.__version__))     #проверка работоспособности библиотеки
# # LOAD AN IMAGE USING 'IMREAD'
# img = cv2.imread("Resources/lena.png")                    #создание объекта img с использованием команды чтения изображения из папки Resources
# # DISPLAY
# cv2.imshow("Lena Soderberg",img)                          #вывод изображения в окно с именем "Lena Soderberg"
# cv2.waitKey(0)                                            #команда для того, чтобы изображение не закрывалось



# import cv2                                                #импорт библиотеки OpenCV
# frameWidth = 640                                          #переменная, отвечающая за ширину в пкс
# frameHeight = 480                                         #переменная, отвечающая за высоту в пкс
# cap = cv2.VideoCapture("Resources/test_video.mp4")        #создание объекта cap с использованием команды захвата видео из папки Resources
# while True:                                               #вечный цикл
#     success, img = cap.read()                             #создание объекта img с использованием команды чтения изображения из объекта cap
#     print(success)                                        #если чтение успешно, т.е. видео не закончилось, вывести True
#     img = cv2.resize(img, (frameWidth, frameHeight))      #масштабирование видео
#     cv2.imshow("Result", img)                             #вывод изображения в окно с именем "Result"
#     if cv2.waitKey(1) & 0xFF == ord('q'):                 #досрочное закрытие видео по нажатию на кнопку q
#         break                                             #осуществляемое выходом из цикла с помощью команды break
# cap.release()
# cv2.destroyAllWindows()

import cv2                                                #импорт библиотеки OpenCV
frameWidth = 640                                          #переменная, отвечающая за ширину в пкс
frameHeight = 480                                         #переменная, отвечающая за высоту в пкс
cap = cv2.VideoCapture(0)                                 #создание объекта cap с использованием команды захвата видео с вебкамеры
cap.set(3, frameWidth)                                    #установка параметра ширины для объекта, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
cap.set(4, frameHeight)                                   #установка параметра высоты для объекта, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# cap.set(10,100)                                           #опциональная установка яркости, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
while True:                                               #вечный цикл
    success, img = cap.read()                             #создание объекта img с использованием команды чтения изображения из объекта cap
    cv2.imshow("Result", img)                             #вывод изображения в окно с именем "Result"
    if cv2.waitKey(1) & 0xFF == ord('q'):                 #досрочное закрытие видео по нажатию на кнопку q
        cap.release()
        cv2.destroyAllWindows()
        break                                             #осуществляемое выходом из цикла с помощью команды break
cap.release()

