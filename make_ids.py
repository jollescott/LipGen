import os
from os.path import join, isfile

from constants import DATA_DIR

files = [f for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]

for i, f in enumerate(files):
    os.rename(join(DATA_DIR, f), join(DATA_DIR, "{}.{}".format(i, f.split(".")[-1])))
