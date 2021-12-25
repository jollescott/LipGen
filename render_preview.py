from pathlib import Path
from os.path import isfile, join
from os import listdir
from scipy.io import wavfile

from constants import MODELS_DIR


def query_models():
    files = [f for f in listdir(MODELS_DIR) if isfile(join(MODELS_DIR, f))]
    return files


if __name__ == "__main__":
    files = query_models()
