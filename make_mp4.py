import ffmpeg
import os
from os.path import isfile, join

from constants import DATA_DIR

files = [
    f
    for f in os.listdir(DATA_DIR)
    if isfile(join(DATA_DIR, f)) and "mp4" not in f.split(".")
]

for file in files:
    stream = ffmpeg.input(join(DATA_DIR, file))
    stream = ffmpeg.output(
        stream, "{}/{}.mp4".format(DATA_DIR, file.split(".")[0]), loglevel="quiet"
    )
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

    os.remove("{}/{}".format(DATA_DIR, file))
