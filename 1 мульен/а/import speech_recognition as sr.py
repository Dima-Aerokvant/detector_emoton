import speech_recognition as sr
from transformers import pipeline
from rich.console import Console
from rich import inspect
# model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
model = pipeline(model="seara/rubert-base-cased-ru-go-emotions")
console = Console()
# Создаем объект Recognizer
recognizer = sr.Recognizer()

# Загружаем аудиофайл
audio_file = "output.wav"
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)


text = recognizer.recognize_google(audio_data, language="ru-RU")
result = model(text)

