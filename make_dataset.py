import os
from os.path import join, isfile, exists
from random import randint
import shutil

from constants import TEMP_DIR, TRAIN_VALIDATE_RATIO

DIRS = [
    join(TEMP_DIR, "input", "train"),
    join(TEMP_DIR, "input", "validate"),
    join(TEMP_DIR, "output", "train"),
    join(TEMP_DIR, "output", "validate"),
]

for input_files in DIRS:
    if exists(input_files):
        shutil.rmtree(input_files)

    os.mkdir(input_files)


input_files = join(TEMP_DIR, "input")

files = [f for f in os.listdir(input_files) if isfile(join(input_files, f))]
orig_len = len(files)

while len(files) > orig_len * TRAIN_VALIDATE_RATIO:
    file = files.pop(randint(0, len(files) - 1))
    shutil.copy(join(TEMP_DIR, 'input', file), join(TEMP_DIR, "input", "train", file))
    shutil.copy(join(TEMP_DIR, 'output', file), join(TEMP_DIR, "output", "train", file))


for file in files:
    shutil.copy(
        join(TEMP_DIR, "input", file),
        join(TEMP_DIR, "input", "validate", file),
    )
    shutil.copy(
        join(TEMP_DIR, "output", file),
        join(TEMP_DIR, "output", "validate", file),
    )
