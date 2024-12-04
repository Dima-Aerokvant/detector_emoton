import moduls
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

class data_template:
    video_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    audio_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}# данные обработки всех модулей остаются на сервере
    text_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
    pulse_line = []

data_now = data_template


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

@socketio.on('uploadFrame')
def uploadFrame(data):
    global data_now # без глобала не видит дата_нау
    encoded_data = data["data"].split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    emotions, img_rec = moduls.frame_detection(img)
    fps = data['pulse']
    result = moduls.rppg.process_frame(img)
    if(result.hr > 60):
        data_now.pulse_line.append(result.hr)
    print(result.hr)# первые 300 кадров он пишет нан. нужно как то изменить размер буферра https://samproell.github.io/yarppg/deepdive/ 
    jpg_img = cv2.imencode('.jpg', img_rec)# и по идее тут есть казус  с частотой кадров ведь мы берем каждый какой то там кадр
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    b64_string = "data:image/jpg;base64," + b64_string
    if(emotions):
        for emotion in emotions:
            data_now.video_probabilities[emotion] = (data_now.video_probabilities[emotion]+emotions[emotion])/2
    emit('my_response', b64_string)
    data_now = data_template

@socketio.on('uploadResult')
def emit_result(data):
    global data_now# без глобала не видит дата_нау
    video_probab =[]
    audio_probab = []
    text_probab = []
    series_pulse = []
    for emotion in data_now.video_probabilities:
        video_probab.append([str(emotion), int(data_now.video_probabilities[emotion])]) # переделываем словарь в массив нужна для вывода в диограммы
    for emotion in data_now.audio_probabilities:
        audio_probab.append([str(emotion), int(data_now.audio_probabilities[emotion])])  
    for emotion in data_now.text_probabilities:
        text_probab.append([str(emotion), int(data_now.text_probabilities[emotion])])
    for i in range(len(data_now.pulse_line)):
        series_pulse.append([data_now.pulse_line[i],i+1])

    send = {'series_aud' : audio_probab, 'series_vid' : video_probab,'series_txt' : text_probab,'series_pulse':series_pulse}
    emit('chart_update', send)
    data_now = data_template # чистим для запуска нового теста



@socketio.on('uploadAudio')
def uploadAudio(data):
    print(data)

@socketio.on('startRec')
def startRec(data):
    if data["data"]:
        print ("START")
    else:
        print ("SEND") #sahdahfva
        emit('send_list', ["123", "234", "345"])


if __name__ == '__main__':
    socketio.run(app, debug=True)

