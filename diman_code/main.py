from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import moduls

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

video_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
audio_probabilities = {'angry': 120, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}
text_probabilities = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 150}
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
    encoded_data = data["data"].split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    emotions, img_rec = moduls.frame_detection(img)
    jpg_img = cv2.imencode('.jpg', img_rec)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    b64_string = "data:image/jpg;base64," + b64_string
    global video_probabilities
    for emotion in emotions:
        video_probabilities[emotion] = (video_probabilities[emotion]+emotions[emotion])/2
    emit('my_response', b64_string)

@socketio.on('uploadResult')
def emit_result(data):
    video_probab =[]
    audio_probab = []
    text_probab = []

    for emotion in video_probabilities:
        video_probab.append([str(emotion), int(video_probabilities[emotion])]) # переделываем словарь в массив нужна для вывода в диограммы
    for emotion in audio_probabilities:
        audio_probab.append([str(emotion), int(audio_probabilities[emotion])])  
    for emotion in text_probabilities:
        text_probab.append([str(emotion), int(text_probabilities[emotion])])

    send = {'series_aud' : audio_probab, 'series_vid' : video_probab,'series_txt' : text_probab}
    
    emit("chart_update", send)

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

