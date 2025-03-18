import io
import json
import pickle
import random
import sys

import nltk
import numpy as np
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior

with open("intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


# Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    print(bag)
    return np.array(bag)


# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category


# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            response_text = random.choice(intent["responses"])
            if "image_url" in intent:
                return {"tipo": "imagen", "texto": response_text, "image_url": intent["image_url"]}
            else:
                return {"tipo": "texto", "texto": response_text}
    return {"tipo": "texto", "texto": "No entiendo."}



def respuesta(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res


while True:
    message = input()
    print(respuesta(message))
