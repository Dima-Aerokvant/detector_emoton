from rich import print
import sounddevice as sd
from scipy.io.wavfile import write


from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}


dfwgrvgv

frequency = 44400
duration = 2#сколько будеи идти запись
recording = sd.rec(int(duration * frequency),
				samplerate = frequency, channels = 2)
sd.wait()
write("record.mp3", frequency, recording)#запись голоса


waveform, sample_rate = torchaudio.load("record.mp3", normalize=True)
transform = torchaudio.transforms.Resample(sample_rate, 16000)
waveform = transform(waveform)

inputs = feature_extractor(
        waveform, 
        sampling_rate=feature_extractor.sampling_rate, 
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

logits = model(inputs['input_values'][0]).logits
predictions = torch.argmax(logits, dim=-1)
predicted_emotion = num2emotion[predictions.numpy()[0]]
print("Эмоция в аудиофайле:",predicted_emotion)

softmax_probs = torch.nn.functional.softmax(logits, dim=-1)
probabilities = softmax_probs[0].detach().numpy()
emotion_probabilities = {num2emotion[i]: probabilities[i] for i in range(len(probabilities))}
#Вывод вероятностей с которыми были определены эмоции
for emotion, probability in emotion_probabilities.items():
    print(f"{emotion}: {probability}")
