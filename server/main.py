from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import base64
import video_rec

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

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
@socketio.on('upload')
def upload(data):
    # print(data)
    encoded_data = data["data"].split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = video_rec.video_recognize(img)
    # text = "2342342"
    img = cv2.putText(img, text, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255, 255, 255), 1, cv2.LINE_AA)

    jpg_img = cv2.imencode('.jpg', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
    b64_string = "data:image/jpg;base64," + b64_string
    emit('my_response', b64_string)
    # testf()

@socketio.on('uploadAudio')
def uploadAudio(data):
    print (len(data))

@socketio.on('startRec')
def startRec(data):
    if data["data"]:
        print ("START")
    else:
        print ("SEND") #sahdahfva
        emit('send_list', ["123", "234", "345"])

def testf():
    for _ in range (10):
        time.sleep(2)
        emit('my_response', "superdata")

if __name__ == '__main__':
    socketio.run(app, debug=True)

