import keras
from keras import layers
import os
import json
from os.path import join
import numpy as np
import tensorflow as tf
from sklearn import preprocessing
from datetime import datetime

from constants import (
    BATCH_SIZE,
    DATETIME_FORMAT,
    INPUTS,
    MODELS_DIR,
    TEMP_DIR,
    OUTPUTS,
    EPOCHS,
)


def import_data(subtype):
    input_path = join(TEMP_DIR, "input", subtype)

    files = [f for f in os.listdir(input_path)]
    inputs = []
    outputs = []

    for file in files:
        with open(join(TEMP_DIR, "input", subtype, file)) as f:
            input = json.load(f)
            inputs.append(input)

        with open(join(TEMP_DIR, "output", subtype, file)) as f:
            output = json.load(f)
            outputs.append(output)

    return (np.array(inputs), np.array(outputs))


def build_simple_model():
    inputs = keras.Input(shape=(INPUTS))
    x = layers.Dense(32)(inputs)
    x = layers.Dense(64)(x)
    x = layers.Dense(16)(x)
    outputs = layers.Dense(OUTPUTS, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    train_input, train_output = import_data("train")
    validate_input, validate_output = import_data("validate")

    model = build_simple_model()

    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["accuracy"],
    )

    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output)).batch(
        BATCH_SIZE
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (validate_input, validate_output)
    ).batch(BATCH_SIZE)

    history = model.fit(dataset, validation_data=val_dataset, epochs=EPOCHS)
    model.save("{}/{}.h5".format(MODELS_DIR, datetime.now().strftime(DATETIME_FORMAT)))
