import pandas as pd
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize
import nltk
from pymagnitude import *
from nltk.stem.lancaster import LancasterStemmer
from sklearn.model_selection import train_test_split
from constants import (
    DIR,
    HAL_model_file,
    HAL_responses_file,
    MAG_FILE,
    DATA_FILE,
    MAX_SEQ_LEN,
    batch_size,
)

# load the intent dataset
DIR = os.path.dirname(os.path.abspath(__file__))
HAL_model_file = os.path.join(DIR, "intents.csv")
X = []
y = []
with open(DATA_FILE, mode="r", encoding="ascii", errors="ignore") as csvfile:
    intents = pd.read_csv(csvfile)
    X = list(intents["utterances"])
    y = list(intents["labels"])
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
X = np.asarray(X)

# train val test set split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)
vectors = Magnitude(MAG_FILE)

# construct models
i = tf.keras.layers.Input(shape=(MAX_SEQ_LEN, vectors.dim))
Bidir_LSTM = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32, activation="tanh", return_sequences=True),
    merge_mode="concat",
)(i)
maxpool = tf.keras.layers.GlobalMaxPooling1D()(Bidir_LSTM)
hidden = tf.keras.layers.Dense(32)(maxpool)
dropout = tf.keras.layers.Dropout(0.3)(hidden)
output = tf.keras.layers.Dense(le.classes_.shape[0], activation="softmax")(dropout)
model = tf.keras.Model(inputs=i, outputs=output)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
model.summary()


def make_dataset(xarr, yarr):
    dataset = tf.data.Dataset.from_tensor_slices((xarr, yarr)).repeat()

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
        x = x.numpy().decode()
        x = word_tokenize(x)
        x = [stemmer.stem(i) for i in x]
        if len(x) != 0:
            x = vectors.query(x)
            x = _pad_zeros(x, MAX_SEQ_LEN)
        else:
            x = np.zeros((MAX_SEQ_LEN, vectors.dim))
        return x

    def _process_datapair(X, y):
        X = tf.py_function(_process_string, [X], tf.float32)
        X.set_shape([MAX_SEQ_LEN, vectors.dim])
        y.set_shape([])
        return X, y

    dataset = dataset.map(_process_datapair)
    return dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(batch_size)


train = make_dataset(X_train, y_train)
val = make_dataset(X_val, y_val)
test = make_dataset(X_test, y_test)

stopping_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
filename = "HAL-9000.h5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filename, monitor="val_loss", save_best_only=True, mode="min"
)
model.fit(
    train,
    validation_data=val,
    callbacks=[stopping_early, checkpoint],
    validation_steps=X_val.shape[0] / batch_size,
    steps_per_epoch=X_train.shape[0] / batch_size,
    epochs=100,
)
print("training complete!")
eval_results = model.evaluate(test, steps=X_test.shape[0] / batch_size)
print("HAL's accuracy on the test set is " + str(eval_results[1]))
