from tensorflow.keras.layers import Normalization
import keras
from keras import layers
import os
import json
from os.path import join
import numpy as np

from constants import INPUTS, OUTPUT_DIR, OUTPUTS, EPOCHS


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
    inputs = keras.Input(shape=(INPUTS))
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(OUTPUTS, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def train_model(model, input, output):
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    model.fit(input, output, batch_size=64, epochs=EPOCHS)


if __name__ == "__main__":
    train_input = import_data("input/train")
    train_output = import_data("output/train")

    train_input = normalize_data(train_input)
    train_output = normalize_data(train_output)

    validate_input = import_data("input/validate")
    validate_output = import_data("output/validate")

    validate_input = normalize_data(validate_input)
    validate_output = normalize_data(validate_output)

    model = build_model()
    train_model(model, train_input, train_output)
