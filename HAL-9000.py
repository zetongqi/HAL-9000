import tensorflow as tf
import random
import json
import fileinput
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
from pymagnitude import *
import pandas as pd
import re
from sklearn import preprocessing
import os
from colorama import Fore, Back, Style
from constants import (
    DIR,
    HAL_model_file,
    HAL_responses_file,
    MAG_FILE,
    DATA_FILE,
    MAX_SEQ_LEN,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HAL_9000 = tf.keras.models.load_model(HAL_model_file)
outputs = json.load(open(HAL_responses_file))
vectors = Magnitude(MAG_FILE)

y = []
with open(DATA_FILE, mode="r", encoding="ascii", errors="ignore") as csvfile:
    intents = pd.read_csv(csvfile)
    y = list(intents["labels"])

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


def classify(model, utterance):
    def _process_string(x):

        # x is numpy array
        def _pad_zeros(x, MAX_SEQ_LEN):
            if x.shape[0] >= MAX_SEQ_LEN:
                return x[0:MAX_SEQ_LEN, :]
            else:
                return np.concatenate(
                    (x, np.zeros((MAX_SEQ_LEN - x.shape[0], x.shape[1]))), axis=0
                )

        stemmer = LancasterStemmer()
        x = word_tokenize(x)
        x = [stemmer.stem(i) for i in x]
        if len(x) != 0:
            x = vectors.query(x)
            x = _pad_zeros(x, MAX_SEQ_LEN)
        else:
            x = np.zeros((MAX_SEQ_LEN, vectors.dim))
        return x

    return str(
        le.inverse_transform(
            [
                np.argmax(
                    model.predict(np.expand_dims(_process_string(utterance), axis=0))
                )
            ]
        )[0]
    )


if __name__ == "__main__":
    print("Hi! I am HAL!\n")
    for line in fileinput.input():
        print()
        if re.match("exit|Exit|quit|Quit", line):
            print("Goodbye!")
            break
        intent = classify(HAL_9000, line)
        if intent in outputs:
            print(Fore.RED + "HAL: " + random.choice(outputs[intent]) + "\n")
