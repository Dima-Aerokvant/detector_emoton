from aniemore.recognizers.multimodal import VoiceTextRecognizer
from aniemore.utils.speech2text import SmallSpeech2Text
from aniemore.models import HuggingFaceModel# есть прикол что библеотеки нужно подключать в определеннном порядке иначе на длл будет ругаться
from deepface import DeepFace
import cv2
import torch
import yarppg
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

hrcalc = yarppg.PeakBasedHrCalculator(
    fs=30, window_seconds=1, distance=0.6, update_interval=30
)
rppg = yarppg.Rppg(hr_calc=hrcalc)

# while running:
#     # frame = ...  # get an image array of shape h x w x 3
#     result = rppg.process_frame(frame)
#     print(f"Current rPPG signal value: {result.value} (HR: {result.hr})")

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
 
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    result = rppg.process_frame(frame)
    # print(f"Current rPPG signal value: {result.value} (HR: {result.hr})")
    print(60 * fps / result.hr)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print (fps)