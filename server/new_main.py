import moduls
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from pydub import AudioSegment
from threading import Thread
import os


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class data_template():
    video_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    audio_probabilities = {}
    pulse_line = []
    text = ''

now_frame_pulse = 0
data_now = data_template()

def result_moduls(data_now):
    video_probab =[]
    audio_probab = []
    series_pulse = []

    for emotion in data_now.video_probabilities:
        video_probab.append([str(emotion), int(data_now.video_probabilities[emotion])]) # переделываем словарь в массив нужна для вывода в диограммы

    for emotion in data_now.audio_probabilities:
        audio_probab.append([str(emotion), int(data_now.audio_probabilities[emotion]*100)])

    for i in range(len(data_now.pulse_line)):
        series_pulse.append([i, data_now.pulse_line[i]])

    send = {'series_vid' : video_probab,'series_aud' : audio_probab,'series_pulse':series_pulse, "text_from_audio" :data_now.text}
    print(send)
    data_now = data_template() # чистим для запуска нового теста
    return jsonify({'data':send})

# эти функции будут одновременно
#проблема в том что функция запущенная в потоке не може вернуть результат, поэтому пишем так и сохраняем данные в глобальную переменную

def multimodal_audio_recognition(file_path):    
    global data_now
    data_now.audio_probabilities, data_now.text = moduls.multimodal_audio_recognition(file_path)
    print(data_now.audio_probabilities)

def processing_pulse(file_path):
    global data_now
    data_now.pulse_line = moduls.file_pulse_detect(file_path)

@app.route('/')
def index():
    return redirect(url_for('online'))

@app.route('/online')
def online():
    return render_template('online.html')
@app.route('/file')
def file():
    return render_template('file.html')
@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/upload_frame_deepface', methods=["GET", "POST"])
def upload_frame():
    data = request.get_json()
    global data_now # без глобала не видит дата_нау
    encoded_data = data["data"].split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)#Раскодировали дату

    emotions, img_rec = moduls.frame_detection(img)
    if(emotions):
      for emotion in emotions:
          data_now.video_probabilities[emotion] = (data_now.video_probabilities[emotion]+emotions[emotion])/2

    jpg_img = cv2.imencode('.jpg', img_rec)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    b64_string = "data:image/jpg;base64," + b64_string
    return jsonify({'data':b64_string}) # отправляем картинку с выделенной зоной интереса
    
@app.route('/upload_frame_yarppg', methods=["GET", "POST"])
def upload_frame_yarppg(): 
    global now_frame_pulse, data_now
    data = request.get_json()
    encoded_data = data["data"].split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)#Раскодировали дату
    result = moduls.rppg.process_frame(img)
    bpm =  60 * 30 / result.hr
    now_frame_pulse+=1
    now_pulse = {}
    if(bpm > 60):
        now_pulse = {"x": now_frame_pulse, "value": bpm}
        data_now.pulse_line.append(int(bpm))
    print(now_pulse)
    return jsonify({'data':now_pulse}) 

@app.route('/upload_audio', methods=["POST"])
def upload_audio():
    global data_now

    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    audio_file = request.files['audio']  # Получаем файл из запроса
    audio_io = io.BytesIO(audio_file.read())  # Читаем файл в BytesIO
    audio_segment = AudioSegment.from_file(audio_io, format='webm')  # Декодируем файл
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)     # Сохраняем файл
    audio_segment.export(file_path , format="wav")

    th_txt = Thread(target=multimodal_audio_recognition, args=(file_path,))#поток с текстовой обработкой

    th_txt.start()
    th_txt.join()

    send = result_moduls(data_now)
    return send

@app.route('/processing_file', methods=["POST"])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    video_file = request.files['video']  # Получаем файл из запроса
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(file_path)  # Сохраняем файл на сервере

    audio = AudioSegment.from_file(file_path, format="mp4")
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], "audio.wav")# достаем аудио дорожку из видео файла
    audio.export(audio_path, format="wav")

    th_txt = Thread(target=multimodal_audio_recognition, args=(audio_path,))#поток с текстовой обработкой
    th_video = Thread(target=processing_frames, args=(file_path,))#поток с видео обработкой
    th_pulse = Thread(target=processing_pulse, args=(file_path,))
    th_txt.start()
    th_video.start()
    th_pulse.start()

    th_txt.join()# пока потоки не завершат свою работу программа не будет дальше работать
    th_video.join()
    th_pulse.join()

    send = result_moduls(data_now)#send уже json
    return send


def processing_frames(file_path):
    global data_now
    video = cv2.VideoCapture(file_path)
    success = True
    now_frame = 0
    while success:
        success, img = video.read() 
        now_frame+=1
        try:
            if(now_frame == 5):
                emotions, img_rec = moduls.frame_detection(img)
                if(emotions):
                    for emotion in emotions:
                        data_now.video_probabilities[emotion] = (data_now.video_probabilities[emotion]+emotions[emotion])/2
                        now_frame = 0
        except:
            pass
    return data_now


if __name__ == '__main__':
    socketio.run(app, debug=True)

