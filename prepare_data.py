import os
import shutil
import argparse
from os.path import isfile, join, exists
import ffmpeg

from analyze_audio import extract_frequencies_magnitudes
from analyze_frame import process_frame
from constants import DATA_DIR, OUTPUT_DIR, STAGING_DIR

SPLIT_DELTA = 0.25


def split_and_import(index, path):

    probe = ffmpeg.probe(path)

    audio_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "audio"), None
    )

    if audio_stream is None:
        print("{} is missing audio stream, skipping...".format(path))
        return

    duration = float(audio_stream["duration"])
    points = [x * SPLIT_DELTA for x in range(0, round(duration / SPLIT_DELTA))]

    for i, point in enumerate(points):

        # Extract frame
        stream = ffmpeg.input(
            path,
            ss=point,
        )
        stream = ffmpeg.output(
            stream, "{}/{}{}.png".format(STAGING_DIR, index, i), vframes=1
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)

        # Extract audio
        stream = ffmpeg.input(path, ss=point, t=SPLIT_DELTA)
        stream = ffmpeg.output(stream, "{}/{}{}.wav".format(STAGING_DIR, index, i))
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)


def stage_data():
    items = [
        join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))
    ]

    for i, item in enumerate(items):
        split_and_import(i, item)


def process_data():
    items = [
        join(STAGING_DIR, f)
        for f in os.listdir(STAGING_DIR)
        if isfile(join(STAGING_DIR, f))
    ]

    for a in [f for f in items if "wav" in f.split(".")]:
        extract_frequencies_magnitudes(a)

    for p in [f for f in items if "png" in f.split(".")]:
        process_frame(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clean", dest="clean", action="store_true")
    args = ap.parse_args()

    if not exists(OUTPUT_DIR) or args.clean:
        if exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)

        os.mkdir(OUTPUT_DIR)
        os.mkdir(STAGING_DIR)

        stage_data()

    process_data()