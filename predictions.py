import numpy as np
import PIL
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.utils import pad_sequences
from models import loadModels

def featExtract(img, model):

    # convolution layer input size: 4D
    img = np.expand_dims(img, axis=0)

    # preprocessing image
    img = preprocess_input(img)

    # predciting the features
    feat = model.predict(img)

    return feat

def generateSeq(tokenizer, maxLength, feat, model):
    start = 'START'

    for i in range(maxLength):
        seq = tokenizer.texts_to_sequences([start])[0]
        seq = pad_sequences([seq], maxlen=maxLength)

        pred = model.predict([feat, seq], verbose=0)
        word = np.argmax(pred)
        word = tokenizer.index_word[word]

        if word is None:
            break
        if 'end' in word:
            break

        start += ' ' + word

    return start[6:]

