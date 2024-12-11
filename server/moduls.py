from aniemore.recognizers.multimodal import VoiceTextRecognizer
from aniemore.recognizers.voice import VoiceRecognizer
from aniemore.models import HuggingFaceModel# есть прикол что библеотеки нужно подключать в определеннном порядке иначе на длл будет ругаться
from aniemore.utils.speech2text import SmallSpeech2Text

from deepface import DeepFace
import cv2
import torch
import yarppg

rppg = yarppg.Rppg()

def file_pulse_detect(file_path):
  fps = yarppg.get_video_fps(file_path)
  filter_cfg = yarppg.digital_filter.FilterConfig(fps, 0.5, 1.5, btype="bandpass")
  livefilter = yarppg.digital_filter.make_digital_filter(filter_cfg)
  processor = yarppg.FilteredProcessor(yarppg.Processor(), livefilter=livefilter)
  results = rppg.process_video(file_path)
  res = []
  for elem in results:
    if(60 * fps / elem.hr > 60):
        res.append(int(60 * fps / elem.hr))
  return res

def frame_detection(img):
    faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              
    faces = faceCascade.detectMultiScale(imgGray,1.1,7)
    for (x,y,w,h) in faces:                                                               
        cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),4)
        img2 = img[y:y+h, x:x+w]
        objs = DeepFace.analyze(                                                                #
            img_path = img2,
            actions = ['emotion'],
            enforce_detection=False
        )  
        emotions = (objs[0])['emotion']
        return emotions, img 
    return None, img 


def multimodal_audio_recognition(file):
    model = HuggingFaceModel.MultiModal.WavLMBertFusion
    s2t_model = SmallSpeech2Text()

    text = s2t_model.recognize(file).text
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vtr = VoiceTextRecognizer(model=model, device=device)
    res = vtr.recognize((file, text), return_single_label=False)
    return res , text