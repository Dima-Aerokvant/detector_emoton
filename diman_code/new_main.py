import moduls
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from pydub import AudioSegment
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

class data_template():
    video_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    audio_probabilities = {}
    text_probabilities = {}
    pulse_line = []

data_now = data_template()


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

@app.route('/upload_frame', methods=["GET", "POST"])
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

    result = moduls.rppg.process_frame(img)
    bpm =  60 * 30 / result.hr
    if(bpm > 60):
        data_now.pulse_line.append(result.hr)
    print(bpm)# первые 300 кадров он пишет нан. нужно как то изменить размер буферра https://samproell.github.io/yarppg/deepdive/ 

    jpg_img = cv2.imencode('.jpg', img_rec)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    b64_string = "data:image/jpg;base64," + b64_string
    return jsonify({'data':b64_string}) # отправляем картинку с выделенной зоной интереса
    
    
    

def result_moduls(data_now):
    video_probab =[],audio_probab = [],text_probab = [],series_pulse = []

    for emotion in data_now.video_probabilities:
        video_probab.append([str(emotion), int(data_now.video_probabilities[emotion])]) # переделываем словарь в массив нужна для вывода в диограммы

    for emotion in data_now.audio_probabilities:
        audio_probab.append([str(emotion), int(data_now.audio_probabilities[emotion]*100)]) # на выхоже получаем дробь а не проценты нопример 0,215  поэтому *100

    for emotion in data_now.text_probabilities:
        text_probab.append([str(emotion), int(data_now.text_probabilities[emotion]*100)])

    for i in range(len(data_now.pulse_line)):
        series_pulse.append([data_now.pulse_line[i], i+1])

    send = {'series_aud' : audio_probab, 'series_vid' : video_probab,'series_txt' : text_probab,'series_pulse':series_pulse}
    print(send)
    data_now = data_template # чистим для запуска нового теста
    return jsonify({'data':send})
    


@app.route('/upload_audio', methods=["POST"])
def upload_audio():
    global data_now
    audio_file = request.files['audio']  # Получаем файл из запроса
    audio_io = io.BytesIO(audio_file.read())  # Читаем файл в BytesIO
    audio_segment = AudioSegment.from_file(audio_io, format='webm')  # Декодируем файл
    audio_segment.export("mic.wav", format="wav")  # Сохраняем файл как WAV

    def res_th1(x):
        global data_now
        data_now.audio_probabilities = x
    #это костыль? ДА это костыль да еще какой! получть return из функции находящийся в потоке нельзя и приходится выкручиваться
    def res_th2(x):
        global data_now
        data_now.text_probabilities = x

    thread1 = Thread(target= lambda x : res_th1(moduls.audio_recognition_text(x)), args =('mic.wav'))
    thread2 = Thread(target= lambda x : res_th2(moduls.text_recognition(x)), args =('mic.wav')) # парралельно обрабатываем аудио с помощью 2ух модулей

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    send = result_moduls(data_now)
    return send



if __name__ == '__main__':
    socketio.run(app, debug=True)

