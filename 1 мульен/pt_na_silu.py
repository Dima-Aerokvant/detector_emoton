

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice

from multiprocessing import Process


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys

import pyaudio
import keyboard
import numpy as np
from scipy.io import wavfile
import time as time


from multiprocessing import Process



    
def pr():
    print("work")
    
p3 = Process(target=pr,daemon=False)


class Ui(QtWidgets.QMainWindow):
    def __init__(self):
        self.flag_test_start = False
        app = QtWidgets.QApplication(sys.argv) 
        super(Ui, self).__init__() 
        uic.loadUi('min.ui', self) 
        self.setWindowTitle("detector_emotin")
        self.show() 
        self.button_start.clicked.connect(self.button_mute_check)

    
    def button_mute_check(self):
       
        if not self.flag_test_start:
            self.button_start.setText("Остановить тестирование")
            self.button_start.setStyleSheet("background-color : red")
            print("Остановить тестирование")
            self.update()
            p3.start()
            self.flag_test_start = True      
        else:
            self.button_start.setText("Начать тестирование")
            self.button_start.setStyleSheet("background-color : lightgreen")
            print("Начать тестирование")
            self.update()   
            self.flag_test_start = False
            
            


class Recorder():
    def __init__(self):
        super(Recorder, self).__init__() 
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = int(0.03*self.sample_rate)
        self.filename = "mic.wav"
        self.START_KEY = 's'
        self.STOP_KEY = 'q'
        self.listen()


    def record(self):
        recorded_data = []
        p = pyaudio.PyAudio()

        stream = p.open(format=self.audio_format, channels=self.channels,
                        rate=self.sample_rate, input=True,
                        frames_per_buffer=self.chunk)
        while(True):
            data = stream.read(self.chunk)
            recorded_data.append(data)
            if keyboard.is_pressed(self.STOP_KEY):
                print("Stop recording")
                # stop and close the stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                #convert recorded data to numpy array
                recorded_data = [np.frombuffer(frame, dtype=np.int16) for frame in recorded_data]
                wav = np.concatenate(recorded_data, axis=0)
                wavfile.write(self.filename, self.sample_rate, wav)
                print("You should have a wav file in the current directory")
                break


    def listen(self):
        global flag_test_start
        while True:
            time.sleep(0.5)
            print(flag_test_start," potok2")
            if flag_test_start:
                print("pass")
                self.record()
                break  

    
if __name__ == "__main__":
    p1 = Process(target=Ui, daemon=False)
    p1.start()
    p1.join()
