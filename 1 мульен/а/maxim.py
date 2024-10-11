from multiprocessing import Process
from rich import print
import sounddevice as sd
from scipy.io.wavfile import write


from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch


import cv2                                                #импорт библиотеки OpenCV
import time
from deepface import DeepFace
from rich.console import Console


from transformers import pipeline
from rich.console import Console
from rich import inspect

model = pipeline(model="seara/rubert-base-cased-ru-go-emotions")
console = Console()

audio_ver =[]#вероятности по интонации
mas2 =[]#вероятности по видео

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
hubert = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")

result = model("Я опечален, потому что хожу в школу к 8 утра.")
def record():
    frequency = 44400
    duration = 2#сколько будеи идти запись
    recording = sd.rec(int(duration * frequency),
                    samplerate = frequency, channels = 2)
    sd.wait()
    write("record.mp3", frequency, recording)#запись голоса


def raspoznanie_audio():
    global feature_extractor
    global hubert 
    num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}


    waveform, sample_rate = torchaudio.load("record.mp3", normalize=True)
    transform = torchaudio.transforms.Resample(sample_rate, 16000)
    waveform = transform(waveform)

    inputs = feature_extractor(
            waveform, 
            sampling_rate=feature_extractor.sampling_rate, 
            return_tensors="pt",
            padding=True,
            max_length=16000 * 10,
            truncation=True
        )

    logits = hubert(inputs['input_values'][0]).logits
    predictions = torch.argmax(logits, dim=-1)
    predicted_emotion = num2emotion[predictions.numpy()[0]]
    

    softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
    probabilities = softmax_probs[0].detach().numpy()
    emotion_probabilities = {num2emotion[i]: probabilities[i] for i in range(len(probabilities))}
    #Вывод вероятностей с которыми были определены эмоции
    maxx = 0
    for emotion, probability in emotion_probabilities.items():
        maxx = max(maxx, probability)
        audio_ver.append(probability)

    print("Эмоция в аудиофайле:",predicted_emotion," Точность: ", maxx)
        

def video():
    global mas2
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
    ind = mas.index(max(mas))
    res = emotions[ind]

    for elem in mas:
        zxc = elem / count
        mas2.append(zxc)
    print("Эмоция видос:",str(res)+";","Точность:",mas2[ind])
    print(mas2)
                        #осуществляемое выходом из цикла с помощью команды break
    cap.release()

def text_recognition():

    result = model("Я опечален, потому что хожу в школу к 8 утра.")
    console.print (result)

    joy = ['admiration', 'amusement', 'joy', 'optimism', 'relief']
    neutral = ['neutral', 'embarrassment']
    sadness = ['sadness', 'remorse', 'grief']
    discontent = ['disappointment', 'disapproval', 'nervousness', 'fear']
    suprise = ['realization', 'suprise']
    concern = ['caring', 'confusion', 'curiosity']
    anger = ['anger', 'annoyance', 'disgust']
    trust = ['approval', 'gratitude']

    if result[0]['label'] in joy:
        print('joy', result[0]['score'])
    elif result[0]['label'] in neutral:
        print('neutral', result[0]['score']) 
    elif result[0]['label'] in sadness:
        print('sadness', result[0]['score'])
    elif result[0]['label'] in discontent:
        print('discontent', result[0]['score']) 
    elif result[0]['label'] in suprise:
        print('surprise', result[0]['score'])
    elif result[0]['label'] in concern:
        print('concern', result[0]['score'])
    elif result[0]['label'] in anger:
        print('anger', result[0]['score'])
    elif result[0]['label'] in trust:
        print('trust', result[0]['score']) 
    elif result[0]['label'] == 'desire':
        print('desire', result[0]['score'])  
    elif result[0]['label'] == 'love':
        print('love', result[0]['score'])
    elif result[0]['label'] == 'excitement':
        print('excitement', result[0]['score'])
    else:
        print('pride', result[0]['score'])

if __name__ == '__main__':
    p1 = Process(target=video, daemon=False)
    p2 = Process(target=record, daemon=False)
    n = input()
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    raspoznanie_audio()

    print(audio_ver)
    