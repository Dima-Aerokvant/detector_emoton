from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QPushButton
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QIODevice

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
 

app = QtWidgets.QApplication(sys.argv)

ui = uic.loadUi("min.ui")
ui.setWindowTitle("detector_emotin")
ui.show()

button_start = QPushButton("button_start")

flag_test_start = False

def button_mute_check():
    global flag_test_start
    if not flag_test_start:
        ui.button_start.setText("Остановить тестирование")
        ui.button_start.setStyleSheet("background-color : red")
        print("Остановить тестирование")
        ui.update()
        flag_test_start = True
    else:
        ui.button_start.setText("Начать тестирование")
        ui.button_start.setStyleSheet("background-color : lightgreen")
        print("Начать тестирование")
        ui.update()
        flag_test_start = False
    




ui.button_start.clicked.connect(button_mute_check)


app.exec()