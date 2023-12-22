# %%
import warnings
import numpy as np
import pandas as pd

import os

# %%
df = pd.read_csv("archive/train.txt",sep=";",
    names=["Description","Emotion"])
df.head(5)

# %%
df['Emotion'].value_counts()

# %%
df['label_num'] = df['Emotion'].map({
    'joy' : 0, 
    'sadness': 1, 
    'anger': 2, 
    'fear': 3,
    'love': 4,
    'surprise':5
})

df.head(5)

# %%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.Description,df.label_num,test_size=0.2)

# %%
import spacy
nlp = spacy.load("en_core_web_sm")

# %%
def preprocess(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        else:
            filtered_tokens.append(token.lemma_)
    return " ".join(filtered_tokens)

# %%
df['processed_text'] = df["Description"].apply(preprocess)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


clf = Pipeline([
     ('vectorizer_tfidf',TfidfVectorizer()), 
     ('Random Forest', RandomForestClassifier())         
])

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred_rf = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# %%
from matplotlib import pyplot as plt
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Prediction')
plt.ylabel('Truth')

# %%
import whisper
import sounddevice as sd
import numpy as np
from pydub import AudioSegment

def record_audio(duration, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording complete.")
    return audio_data.flatten()

def save_audio_mp3(data, filename='collection/output.mp3', samplerate=44100):
    print(f"Saving audio to {filename}...")
    # Normalize the audio data to the range [-1, 1]
    normalized_data = data / np.max(np.abs(data), axis=0)
    # Convert to 16-bit PCM format
    pcm_data = (normalized_data * 32767).astype(np.int16)
    # Create an AudioSegment from the PCM data
    audio_segment = AudioSegment(pcm_data.tobytes(), frame_rate=samplerate, sample_width=2, channels=1)
    # Export as MP3
    audio_segment.export(filename, format="mp3")
    print("Audio saved.")

def transcribe():
    model = whisper.load_model("base")
    result = model.transcribe("collection/output.mp3")
    return result

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


X = df['processed_text']
y = df['label_num']

clf = Pipeline([
    ('vectorizer_tfidf', TfidfVectorizer()),
    ('Random Forest', RandomForestClassifier())
])

clf.fit(X, y)

def predict_emotion(sentence):
    processed_sentence = preprocess(sentence)

    emotion_label = clf.predict([processed_sentence])[0]
    emotion_mapping = {0: 'joy', 1: 'sadness', 2: 'anger', 3: 'fear', 4: 'love', 5: 'surprise'}
    predicted_emotion = emotion_mapping[emotion_label]

    return predicted_emotion

def get_user_input():
    duration = int(input("put the length of your speech in seconds (e.g. 5): "))
    audio_data = record_audio(duration)
    save_audio_mp3(audio_data)
    text = transcribe()
    return str(text['text'])
    

user_input = get_user_input()
predicted_emotion = predict_emotion(user_input)
print(user_input)
print(f"The predicted emotion for the sentence is: {predicted_emotion}")
#i was ready to meet mom in the airport and feel her ever supportive arms around me;


