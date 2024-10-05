import cv2                                                #импорт библиотеки OpenCV
import time
from deepface import DeepFace
from rich.console import Console
frameWidth = 640                                          #переменная, отвечающая за ширину в пкс
frameHeight = 480                                         #переменная, отвечающая за высоту в пкс
cap = cv2.VideoCapture(0)                                 #создание объекта cap с использованием команды захвата видео с вебкамеры
cap.set(3, frameWidth)                                    #установка параметра ширины для объекта, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
cap.set(4, frameHeight)                                   #установка параметра высоты для объекта, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# cap.set(10,100)  
faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")     
emotions = ['angry','disgust','fear','happy','sad','surprise','neutral']    
mas = [0,0,0,0,0,0,0]
count =0
while sum(mas)<10:  
    start_time = time.time()                                             #вечный цикл
    success, img = cap.read() 
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              #создание объекта img с использованием команды чтения изображения из объекта cap
    faces = faceCascade.detectMultiScale(imgGray,1.1,7)


    for (x,y,w,h) in faces:                                                               
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,250,0),4)
        img2 = img[y:y+h, x:x+w]

        objs = DeepFace.analyze(                                                                #вывод результата
            img_path = img2,
            actions = ['emotion'],
            enforce_detection=False
        )
        dom_em = (objs[0])['dominant_emotion']
        mas[emotions.index(dom_em)]+=1
        count+=1
    cv2.imshow("Result", img)                             #вывод изображения в окно с именем "Result"
    if cv2.waitKey(1) & 0xFF == ord('q'):                 #досрочное закрытие видео по нажатию на кнопку q
        cap.release()
        cv2.destroyAllWindows()
        break     
    time.sleep(1.0)  

print(emotions[mas.index(max(mas))])    
mas2 =[]
for elem in mas:
    elem = elem / count
    mas.append(elem)
                       #осуществляемое выходом из цикла с помощью команды break
                       
print(mas2)
cap.release()