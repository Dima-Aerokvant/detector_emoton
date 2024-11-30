from deepface import DeepFace
import cv2

def frame_detection(img):
    faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              
    faces = faceCascade.detectMultiScale(imgGray,1.1,7)

    for (x,y,w,h) in faces:                                                               
            cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),4)
            img2 = img[y:y+h, x:x+w]
            objs = DeepFace.analyze(                                                                #вывод результата
                img_path = img2,
                actions = ['emotion'],
                enforce_detection=False
            )  
            emotions = (objs[0])['emotion']
            return emotions, img 
