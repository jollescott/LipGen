from tensorflow.keras.layers import Normalization
import keras
from keras import layers
import os
import json
from os.path import join
import numpy as np

from constants import OUTPUT_DIR


def normalize_data(data):
    norm = Normalization()
    norm.adapt(data)

    return norm(data)


def import_data(type):
    input_path = join(OUTPUT_DIR, type)

    files = [join(input_path, f) for f in os.listdir(input_path)]
    inputs = []

    for file in files:
        with open(file) as f:
            input = json.load(f)
            inputs.append(input)

    return np.array(inputs)


def build_model():
    inputs = keras.Input(shape=(10))
    x = layers.Flatten()
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def train_model(model, input, output):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(input, output, batch_size=64, epochs=1)


if __name__ == "__main__":
    input = import_data("input")
    output = import_data("output")

    input = normalize_data(input)
    output = normalize_data(output)

    model = build_model()
    train_model(model, input, output)
