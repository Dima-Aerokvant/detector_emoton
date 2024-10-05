import sim                                                              #импорт модуля sim

# --------------
import sys                                                              #импорт модуля sys
import cv2                                                              #импорт библиотеки OpenCV
import numpy as np                                                      #импорт библиотеки numpy
# --------------

sim.simxFinish(-1)                                                      #на всякий случай закрываем все открытые соединения с Коппелией

clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)       #строки инициализации подключения к CoppeliaSim

if clientID != -1:  # check if client connection successful             #строки инициализации подключения к CoppeliaSim
    print('Connected to remote API server')                             #строки инициализации подключения к CoppeliaSim

else:                                                                   #строки инициализации подключения к CoppeliaSim
    print('Connection not successful')                                  #строки инициализации подключения к CoppeliaSim
    sys.exit('Could not connect')                                       #строки инициализации подключения к CoppeliaSim


error,Left_wheel=sim.simxGetObjectHandle(clientID,'Left_joint', sim.simx_opmode_oneshot_wait)       #создаем объект левого колеса
error,Right_wheel=sim.simxGetObjectHandle(clientID,'Right_joint', sim.simx_opmode_oneshot_wait)     #создаем объект правого колеса

#-------------------
error,vis_cam = sim.simxGetObjectHandle(clientID,'cam', sim.simx_opmode_oneshot_wait)               #создаем объект камеры
#-------------------

w1=0.5                                                                                              #скорость левого колеса
w2=0.5                                                                                              #скорость правого колеса


while 1:                                                                                                        #вечный цикл
    err, resolution, image = sim.simxGetVisionSensorImage(clientID, vis_cam, 0, sim.simx_opmode_oneshot_wait)   #считываем изображение в объект image
    #print(err, resolution, type(image))
    img = np.array(image, ndmin=1, dtype=np.uint8)                                                              #преобразуем считанное изображение в фармат OpenCV
    #img = img.reshape(resolution[0], resolution[1], 3)
    img = np.reshape(img, (resolution[0], resolution[1], 3))                                                    #преобразуем считанное изображение в фармат OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)                                                                  #меняем цветовую палитру с RGB на BGR для OpenCV

    # --------------
    img = cv2.rotate(img, cv2.ROTATE_180)                                                                       #переворачиваем изображение для корректного отображения
    flip_image = cv2.flip(img, 1)                                                                               #переворачиваем изображение для корректного отображения
    cv2.imshow('Original Image', flip_image)                                                                    #выводим изображение в окно
    gray = cv2.cvtColor(flip_image, cv2.COLOR_BGR2GRAY)                                                         #получаем изображение в оттенках серого
    
    if cv2.waitKey(1) == 27:                                                                                    #если нажата клавиша esc - выходим из цикла
        break
    ret, threshold_image = cv2.threshold(gray, 127, 255, 0)                                                     #бинаризуем изображение. Все что выше 127 становится равно 255, все что ниже - 0
    cv2.imshow('Image', threshold_image)                                                                        #выводим бинаризованное изображение
    print('Изображение = ', threshold_image)

    string_search = threshold_image[250]                                                                        #будем отслеживать линию на 250 строке
    pixel_last = 255                                                                                            #вспомогательная переменная
    counter = 0                                                                                                 #переменная-счетчик
    for pixel in string_search:                                                                                 #в цикле находим левую и правую границы черной линии
        if (pixel_last == 255) and (pixel == 0):
            left_border = counter
        if (pixel_last == 0) and (pixel == 255):
            right_border = counter - 1
        pixel_last = pixel
        counter = counter + 1
    middle_point = (right_border + left_border) / 2                                                             #находим среднюю точку как среднее между левой и правой границами

    if (middle_point > 117) and (middle_point < 137):                                                           #если точка примерно посередине
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, w1, sim.simx_opmode_oneshot_wait)          #едем прямо, скорости колес равны
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, w2, sim.simx_opmode_oneshot_wait)
    elif (middle_point < 117):                                                                                  #если точка ушла влево
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, 0, sim.simx_opmode_oneshot_wait)
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, w2*2, sim.simx_opmode_oneshot_wait)       #подворачиваем влево
    elif (middle_point > 137):                                                                                  #если точка ушла вправо
        error = sim.simxSetJointTargetVelocity(clientID, Left_wheel, w1*2, sim.simx_opmode_oneshot_wait)        #подворачиваем вправо
        error = sim.simxSetJointTargetVelocity(clientID, Right_wheel, 0, sim.simx_opmode_oneshot_wait)

cv2.destroyAllWindows()
# --------------