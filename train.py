from keras.layers.core.dense import Dense
from tensorflow.keras.layers import Normalization
import keras
import os
import json
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.core import Dropout
from sklearn import preprocessing

from constants import BATCH_SIZE, INPUTS, OUTPUT_DIR, OUTPUTS, EPOCHS


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
    x = Dense(INPUTS, activation='relu')(inputs)
    x = Dense(INPUTS, activation='relu')(x)
    x = Dense(OUTPUTS, activation='relu')(x)

    outputs = Dense(OUTPUTS, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    train_input = import_data("input/train")
    train_output = import_data("output/train")

    validate_input = import_data("input/validate")
    validate_output = import_data("output/validate")

    train_input = preprocessing.normalize(train_input, 'l1')
    train_output = preprocessing.normalize(train_output, 'l1')

    validate_input = preprocessing.normalize(validate_input, 'l1')
    validate_output = preprocessing.normalize(validate_output, 'l1')

    model = build_model()

    model.compile(
        optimizer="sgd", loss=tf.keras.losses.MeanSquaredLogarithmicError(), metrics=["accuracy"]
    )

    dataset = tf.data.Dataset.from_tensor_slices((train_input, train_output)).batch(
        BATCH_SIZE
    )

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (validate_input, validate_output)
    ).batch(BATCH_SIZE)

    history = model.fit(dataset, validation_data=val_dataset, epochs=EPOCHS)
