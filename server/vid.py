from deepface import DeepFace
import cv2

img = cv2.imread("a2s.png")
res = DeepFace.analyze(img,("emotion"), enforce_detection=False, detector_backend="ssd")
print (res[0]['dominant_emotion'])