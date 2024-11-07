from PyQt5 import QtWidgets, uic, QtGui#для интерфейса
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *
import sys
import pyaudio
import numpy as np
from scipy.io import wavfile#для аудио
import time as time
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
print("transformers ready")
import torch
import torchaudio
import moviepy.editor as mp
import sounddevice as sd
import whisper
from transformers import pipeline
from cv2_enumerate_cameras import enumerate_cameras
import cv2 #для видео
from deepface import DeepFace
print("deepface ready")
from threading import Thread
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

# export TF_ENABLE_ONEDNN_OPTS=0

class Emo_model():
    def __init__(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.model_hubert = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
        # self.faceCascade= cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml") # cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.faceCascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.cap = cv2.VideoCapture(0)
        self.index_video = 0
        frameWidth = 640                                          #переменная, отвечающая за ширину в пкс
        frameHeight = 480                                         #переменная, отвечающая за высоту в пкс                          #создание объекта cap с использованием команды захвата видео с вебкамеры
        self.cap.set(3, frameWidth)                               #установка параметра ширины для объекта, подробнее: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        self.cap.set(4, frameHeight)   
        self.p = pyaudio.PyAudio()   
        self.index_audio = 1

        self.model_txt_whisp= whisper.load_model("medium")
        self.model_txt_2 = pipeline(model="seara/rubert-base-cased-ru-go-emotions")

def list_ports():
        cameras = []
        for camera_info in enumerate_cameras(cv2.CAP_MSMF):#cv.CAP_MSMF   CAP_GSTREAMER
            print(str(camera_info.index)+':'+str(camera_info.name))
            if camera_info.index % 2 == 0:
                cameras.append([int(camera_info.index),str(camera_info.name)])
        return cameras
def list_audio_ports():
    p = sd.query_devices()
    print(p)
    mics = []
    for i in range(len(p)):
        if p[i]['name'] == 'Microsoft Sound Mapper - Input':
            pass
        elif p[i]['name'] == 'Microsoft Sound Mapper - Output':
            break
        else:
            mics.append([p[i]['index'],p[i]['name']])
    return mics


class Ui(QtWidgets.QMainWindow):
    
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv) 
        # super(Ui, self).__init__() 
        super().__init__() 
        uic.loadUi('medium.ui', self) 
        self.cams = list_ports()
        self.mics = list_audio_ports()
        self.models_init = Emo_model()
        self.flag_test_start = False
        self.delay = 0.5
        self.file_audio = "mic.wav" # пофиксил
        self.num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}
        self.text_probabilities ={}
        self.audio_probabilities = {'neutral': 0, 'angry': 0, 'positive': 0, 'sad': 0, 'other': 0}
        self.video_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

        self.setWindowTitle("detector_emotion")
        self.show() 
        self.button_start.clicked.connect(self.button_mute_check)
        self.chose_file.clicked.connect(self.chose_folder)
        self.start_test_file.clicked.connect(self.start_thread)
        self.apply_settings.clicked.connect(self.chance_cam_audio)
        self.pixmap = QPixmap('no_singal.jpg')
        self.label.setPixmap(self.pixmap)
        self.pulse_box.setPixmap(self.pixmap)
        self.cam_device.clear()
        self.audio_device.clear()
        self.update_combo_box()       
        layout = QVBoxLayout()
        self.graph_widget.setLayout(layout)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.table_probabilities_video.resizeColumnsToContents()

        
    def start_thread(self):
        self.start_test_file.setEnabled(False)
        self.chose_file.setEnabled(False)
        print("start thread analyse file")
        file_recognition = Thread(target=self.start_file_test)
        file_recognition.start()

    def record(self):
        print("start record")
        audio_format = pyaudio.paInt16
        channels = 1
        sample_rate = 16000
        chunk = 1024 # int(0.03*sample_rate)
        filename = 'mic.wav'

        recorded_data = []
        stream = self.models_init.p.open(format=audio_format, channels=channels,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk,
                        input_device_index=self.models_init.index_audio)
        print (self.flag_test_start, " ", self.models_init.index_audio)
        while self.flag_test_start:
            data = stream.read(chunk, exception_on_overflow = False)
            recorded_data.append(data)
        stream.stop_stream()
        stream.close()
        recorded_data = [np.frombuffer(frame, dtype=np.int16) for frame in recorded_data]
        wav = np.concatenate(recorded_data, axis=0)
        wavfile.write(filename, sample_rate, wav)
        print("Succed record")  
        
    def audio_processor(self):
        waveform, sample_rate = torchaudio.load(self.file_audio, normalize=True)
        transform = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = transform(waveform)
        inputs = self.models_init.feature_extractor(
                waveform, 
                sampling_rate=self.models_init.feature_extractor.sampling_rate, 
                return_tensors="pt",
                padding=True,
                max_length=16000 * 10,
                truncation=True
            )
        logits = self.models_init.model_hubert(inputs['input_values'][0]).logits
        softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
        probabilities = softmax_probs[0].detach().numpy()
        # global audio_probabilities
        self.audio_probabilities = {self.num2emotion[i]: probabilities[i] for i in range(len(probabilities))}
        print("audio probab",self.audio_probabilities)
        self.update_table(self.table_probabilities_audio, self.audio_probabilities)       

    def audio_processing(self):
        print('audio file for analyse: ', self.file_audio)
        potok1 = Thread(target=self.text_recognition)
        potok2 = Thread(target=self.audio_processor)
        potok1.start()
        potok2.start()
        potok1.join()
        potok2.join()
        print("end threads_audio")

    def text_recognition(self):
        print ("start text rec")
        result = self.models_init.model_txt_whisp.transcribe(self.file_audio)
        print("analyse_text ", self.file_audio)
        stroka = result["text"]
        print(stroka)
        result = self.models_init.model_txt_2(stroka, return_all_scores = True)
        result = result[0]
        probab = {}
        for elem in result:
            probab[str(elem['label'])] = float(elem['score'])
        self.text_probabilities = {}
        self.text_probabilities['positive']= probab['amusement'] + probab['joy'] + probab['optimism'] + probab['relief']
        self.text_probabilities['neutral']= probab['neutral'] + probab['embarrassment'] 
        self.text_probabilities['sad']= probab['sadness'] + probab['remorse'] + probab['grief'] 
        self.text_probabilities['disgust']= probab['disappointment'] + probab['disapproval'] + probab['nervousness']
        self.text_probabilities['surprise']= probab['admiration'] + probab['realization']
        self.text_probabilities['angry']= probab['anger'] + probab['annoyance'] 
        self.text_probabilities['fear']= probab['fear'] 
        self.update_table(self.table_probabilities_text, self.text_probabilities)       
        print("Text probab",self.text_probabilities)

    def video_processing(self, arg):
        self.emotions = {'angry':0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad':0, 'surprise': 0, 'neutral': 0}
        self.emotions_help = ['angry','disgust','fear','happy','sad','surprise','neutral']   
        print(str(arg), " arg cam")# выдает это  <function Ui.start_file_test.<locals>.<lambda> at 0x00000179061CC360>  arg cam
        if str(arg) == 'camera':
            while(self.flag_test_start): 
                success, img = self.models_init.cap.read() 
                if not success:
                    print("error: success, img = models_init.cap.read(), no image to read")
                self.update_picture_label_1(img)
                imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              
                faces = self.models_init.faceCascade.detectMultiScale(imgGray,1.1,7)
                img_dsd = self.frame_processing(faces, img)

                # self.main_puls(img)
                # stra = 0
                # for emotion in self.emotions:
                #     interface.table_probabilities_video.setItem(0,stra, QTableWidgetItem(str(self.emotions[emotion])))#srtoka column
                #     stra+=1
                # time.sleep(self.delay)
            print("live camera:  ",self.video_probabilities) 
        elif str(arg) == 'file':
            print('video file recognition')
            success = True
            while success:
                success, img = self.models_init.cap.read() 
                imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                              
                faces = self.models_init.faceCascade.detectMultiScale(imgGray,1.1,7)
                img = self.frame_processing(faces, img)
                for i in range(5):
                    success, img = self.models_init.cap.read() #skip frames
                    if not success:
                        break
            print("file:  ",self.video_probabilities)   
            self.models_init.cap = cv2.VideoCapture(self.models_init.index_video)  
            print("end tedt file. cap cahanged to ", self.models_init.index_video)     
        self.update_table(self.table_probabilities_video, self.video_probabilities)       
    def frame_processing(self, faces, img):
        for (x,y,w,h) in faces:                                                               
            cv2.rectangle(img,(x,y),(x+w,y+h),(36,255,12),4)
            self.update_picture_label_2(img)
            img2 = img[y:y+h, x:x+w]
            objs = DeepFace.analyze(                                                                #вывод результата
                img_path = img2,
                actions = ['emotion'],
                enforce_detection=False
            )  

            self.emotions = (objs[0])['emotion']
            ind = 0
            for emotion in self.emotions:
                self.video_probabilities[self.emotions_help[ind]]=(self.emotions[emotion]+self.video_probabilities[self.emotions_help[ind]])/2
                ind+=1
            
            result_text = str((objs[0])["dominant_emotion"]) +" "+str(self.emotions[(objs[0])["dominant_emotion"]])
            cv2.putText(img, result_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 2)
            return img

    
    def button_mute_check(self):
        if not self.flag_test_start:
            self.button_start.setText("Остановить тестирование")
            self.button_start.setStyleSheet("background-color : red")
            self.update()
            self.flag_test_start = True
            self.p1 = Thread(target=self.record)#, args =(lambda : self.flag_test_start, )
            arg = "camera"
            self.p2 = Thread(target=self.video_processing,args =(arg, ))#, args =(lambda : self.flag_test_start, ) 
            self.p1.start()
            self.p2.start()
        else:
            self.button_start.setText("Начать тестирование")
            self.button_start.setStyleSheet("background-color : lightgreen")
            self.update()
            self.flag_test_start = False
            self.p1.join()
            self.p2.join()
            # возможно сделать задержку
            self.audio_processing()
            res = self.predominant_emotion([self.audio_probabilities, self.video_probabilities, self.text_probabilities])
            self.resullt.setText(str(res[0]))
            print("результат работы", res)
                
    def start_file_test(self):
        self.file_audio = "mic_from_file.wav"
        arg = "file"
        p_video = Thread(target=self.video_processing,args =(arg, ))
        print('p_audio start')
        p_video.start()
        self.audio_processing()
        p_video.join()
        self.file_audio = "mic.wav"  
        print("file_audio is changed to ",self.file_audio)
        res = self.predominant_emotion([self.audio_probabilities, self.video_probabilities, self.text_probabilities])
        self.resullt.setText(str(res[0]))
        print("результат работы", res)
    
    def chose_folder(self):
        global file_audio, filetype, models_init, prev_cap
        fname, filetype = QFileDialog.getOpenFileName(self)
        print(filetype)
        if fname:
            print('file to analyse: ',fname)
            self.models_init.cap = cv2.VideoCapture(fname)
            self.file_result.setItem(0,0, QTableWidgetItem(str(fname)))
            clip = mp.VideoFileClip(fname)
            clip.audio.write_audiofile(r"mic_from_file.wav")
            self.start_test_file.setEnabled(True)

    def update_picture_label_1(self, image_cv2):
        image = QtGui.QImage(image_cv2, image_cv2.shape[1],\
                            image_cv2.shape[0], image_cv2.shape[1] * 3,QtGui.QImage.Format_BGR888)
        self.pixmap = QPixmap(image)
        self.label.setPixmap(self.pixmap)
        self.update()

    def update_picture_label_2(self, image_cv2):
        image = QtGui.QImage(image_cv2, image_cv2.shape[1],\
                            image_cv2.shape[0], image_cv2.shape[1] * 3,QtGui.QImage.Format_BGR888)
        self.pixmap = QPixmap(image)
        self.pulse_box.setPixmap(self.pixmap)
        self.update()

    def update_picture_label_3(self, image_cv2):
        image = QtGui.QImage(image_cv2, image_cv2.shape[1],\
                            image_cv2.shape[0], image_cv2.shape[1] * 3,QtGui.QImage.Format_BGR888)
        self.pixmap = QPixmap(image)
        self.pulse_box.setPixmap(self.pixmap)
        self.update()
    def update_combo_box(self):
        for elem in self.cams:
            self.cam_device.addItem(str(elem[1]))
        for elem in self.mics:
            self.audio_device.addItem(str(elem[1]))
        self.update()

    def chance_cam_audio(self):
        cam = self.cam_device.currentText()
        audio = self.audio_device.currentText()
        index_camera = 0
        for elem in self.cams:
            if(elem[1] == cam):
                index_camera = elem[0]
        audio_index = 0
        for elem in self.mics:
            if(elem[1] == audio):
                audio_index = elem[0]

        self.models_init.index_audio = audio_index
        self.models_init.cap = cv2.VideoCapture(0)#index_camera)
        self.models_init.index = index_camera
        frameWidth = 640                                          
        frameHeight = 480                                        
        self.models_init.cap.set(3, frameWidth)                   
        self.models_init.cap.set(4, frameHeight)
        print("index_camera changed on: ",index_camera) 
        print("index_audio changed on: ",audio_index) 

    def plot_graph(self, values):

        self.ax.plot(values, 'g')
        self.ax.set_title('График пульса')
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.canvas.draw()

    def update_table(self, object_table, list):
        i = 0
        for key in list:
            object_table.setItem(0, i , QTableWidgetItem(str(list[key])))
            i+=1
        object_table.resizeColumnsToContents()
    def predominant_emotion(self, list):
        self.result = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0,'positive': 0,'other' : 0}
        for elem in list:
            for key in elem:
                if self.result[key] == 0:
                    self.result[key] = elem[key]
                else:
                    self.result[key] = (self.result[key] + elem[key])/2
        self.result['positive'] = (self.result['positive']+ self.result['happy'])/2
        self.result.pop('happy')

        predict = ['other', 0]
        for key in self.result:
            if self.result[key] > self.result[predict[0]]:
                predict[0] = key
                predict[1] = self.result[key]

        return predict
    

if __name__ == "__main__":

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # models_init = models()
    
    print("models_init ready...")

    interface = Ui()
    print("UI ready...")


    # puls = puls_detection()
    print("puls ready...")
    print("plot ready...")
    interface.app.exec()
    # emomodels.cap.release()
    # emomodels.p.terminate()