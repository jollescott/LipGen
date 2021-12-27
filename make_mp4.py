import ffmpeg
import os
from os.path import isfile, join
import argparse
import numpy as np
import concurrent.futures

from constants import DATA_DIR


def query_non_mp4():
    files = [
        f
        for f in os.listdir(DATA_DIR)
        if isfile(join(DATA_DIR, f)) and "mp4" not in f.split(".")
    ]

    return files


def convert_to_mp4(file):
    stream = ffmpeg.input(join(DATA_DIR, file))
    stream = ffmpeg.output(
        stream, "{}/{}.mp4".format(DATA_DIR, file.split(".")[0]), loglevel="quiet"
    )
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

    os.remove("{}/{}".format(DATA_DIR, file))

def convert_files(files):
    for file in files:
        convert_to_mp4(file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-w",
        "--workers",
        default=1,
        type=int,
        help="Amount of concurrent worker threads.",
        nargs="?",
    )

    args = ap.parse_args()

    files = query_non_mp4()
    dist_work = np.array_split(files, args.workers)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(convert_files, dist_work)
