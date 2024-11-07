from transformers import pipeline
from rich.console import Console
from rich import inspect
import whisper
model = pipeline(model="seara/rubert-base-cased-ru-go-emotions")

console = Console()



stroka = "Храни тебя господь. Ирина шейк мейби бейби баба нина гадалка"

result = model(stroka, return_all_scores = True)
result = result[0]
probab = {}

for elem in result:
    probab[str(elem['label'])] = float(elem['score'])
text_probabilities = {}
joy = ['amusement', 'joy', 'optimism', 'relief']
neutral = ['neutral', 'embarrassment']
sadness = ['sadness', 'remorse', 'grief']
discontent = ['disappointment', 'disapproval', 'nervousness', 'disgust']
suprise = ['realization', 'suprise']
anger = ['anger', 'annoyance']


text_probabilities['positive']= probab['amusement'] + probab['joy'] + probab['optimism'] + probab['relief']
text_probabilities['neutral']= probab['neutral'] + probab['embarrassment'] 
text_probabilities['sad']= probab['sadness'] + probab['remorse'] + probab['grief'] 
text_probabilities['disgust']= probab['disappointment'] + probab['disapproval'] + probab['nervousness']
text_probabilities['surprise']= probab['admiration'] + probab['realization']
text_probabilities['angry']= probab['anger'] + probab['annoyance'] 
text_probabilities['fear']= probab['fear'] 
print("text_probabilities ", text_probabilities)
