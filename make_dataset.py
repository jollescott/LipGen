import os
from os.path import join, isfile, exists
from random import randint
import shutil

from constants import OUTPUT_DIR, TRAIN_VALIDATE_RATIO

DIRS = [
    join(OUTPUT_DIR, "input", "train"),
    join(OUTPUT_DIR, "input", "validate"),
    join(OUTPUT_DIR, "output", "train"),
    join(OUTPUT_DIR, "output", "validate"),
]

for dir in DIRS:
    if exists(dir):
        shutil.rmtree(dir)

    os.mkdir(dir)


def split_type(type):
    dir = join(OUTPUT_DIR, type)

    files = [f for f in os.listdir(dir) if isfile(join(dir, f))]
    orig_len = len(files)

    while len(files) > orig_len * TRAIN_VALIDATE_RATIO:
        file = files.pop(randint(0, len(files) - 1))
        shutil.copy(join(dir, file), join(dir, "train", file))

    for file in files:
        shutil.copy(join(dir, file), join(dir, "validate", file))


split_type("input")
split_type("output")
