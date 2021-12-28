import os
from os.path import join, isfile, exists
from random import randint
import shutil

from constants import INPUT_DIR, OUTPUT_DIR, TRAIN_VALIDATE_RATIO

DIRS = [
    join(INPUT_DIR, "train"),
    join(INPUT_DIR, "validate"),
    join(OUTPUT_DIR, "train"),
    join(OUTPUT_DIR, "validate"),
]

for input_files in DIRS:
    if exists(input_files):
        shutil.rmtree(input_files)

    os.mkdir(input_files)


files = [f for f in os.listdir(INPUT_DIR) if isfile(join(INPUT_DIR, f))]
orig_len = len(files)

while len(files) > orig_len * TRAIN_VALIDATE_RATIO:
    file = files.pop(randint(0, len(files) - 1))
    shutil.copy(join(INPUT_DIR, file), join(INPUT_DIR, "train", file))
    shutil.copy(join(OUTPUT_DIR, file), join(OUTPUT_DIR, "train", file))


for file in files:
    shutil.copy(
        join(INPUT_DIR, file),
        join(INPUT_DIR, "validate", file),
    )
    shutil.copy(
        join(OUTPUT_DIR, file),
        join(OUTPUT_DIR, "validate", file),
    )
