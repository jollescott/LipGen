import keras
from keras import layers
import os
import json
from os.path import join
import numpy as np
import tensorflow as tf
from sklearn import preprocessing

from constants import BATCH_SIZE, INPUTS, OUTPUT_DIR, OUTPUTS, EPOCHS


def import_data(type):
    input_path = join(OUTPUT_DIR, "input", type)

    files = [f for f in os.listdir(input_path)]
    inputs = []
    outputs = []

    for file in files:
        with open(join(OUTPUT_DIR, "input", file)) as f:
            input = json.load(f)
            inputs.append(input)

        with open(join(OUTPUT_DIR, "output", file)) as f:
            output = json.load(f)
            outputs.append(output)

    return (np.array(inputs), np.array(outputs))


def build_simple_model():
    inputs = keras.Input(shape=(INPUTS))
    x = layers.Dense(INPUTS, activation="relu")(inputs)
    x = layers.Dense(INPUTS, activation="relu")(x)
    x = layers.Dense(OUTPUTS, activation="relu")(x)

    outputs = layers.Dense(OUTPUTS, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model


def build_rnn_model():
    inputs = keras.Input(shape=(INPUTS))
    x = layers.Dense(128)(inputs)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64)(x)
    x = layers.Dropout(0.8)(x)
    x = layers.Dense(64)(x)
    outputs = layers.Dense(OUTPUTS)(x)

    model = keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    train_input, train_output = import_data("train")
    validate_input, validate_output = import_data("validate")

    #train_input = preprocessing.normalize(train_input, "l1")
    #train_output = preprocessing.normalize(train_output, "l1")

    #validate_input = preprocessing.normalize(validate_input, "l1")
    #validate_output = preprocessing.normalize(validate_output, "l1")

    model = build_rnn_model()

    model.compile(
        optimizer="sgd",
        loss=tf.keras.losses.MeanSquaredLogarithmicError(),
        metrics=["accuracy"],
    )

    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output)).batch(
        BATCH_SIZE
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (validate_input, validate_output)
    ).batch(BATCH_SIZE)

    history = model.fit(dataset, validation_data=val_dataset, epochs=EPOCHS)
