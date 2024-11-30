from deepface import DeepFace

def video_recognize(img):
    res = DeepFace.analyze(img,("emotion"), enforce_detection=False, detector_backend="ssd")
    return res[0]['dominant_emotion'] 