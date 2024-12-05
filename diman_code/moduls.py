from aniemore.recognizers.text import TextRecognizer
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel# есть прикол что библеотеки нужно подключать в определеннном порядке иначе на длл будет ругаться
from deepface import DeepFace
import cv2
import torch
import yarppg

rppg = yarppg.Rppg(hr_calc= yarppg.PeakBasedHrCalculator(
    fs=30, window_seconds=5, distance=0.6, update_interval=1
))#уменьшаем буффер

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
    return None, img 

def audio_recognition_text(file):
    model = HuggingFaceModel.Voice.WavLM
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vr = VoiceRecognizer(model=model, device=device)
    res = vr.recognize(file, return_single_label=True)
    return res