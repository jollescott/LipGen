import os
import shutil
import argparse
from os.path import isfile, join, exists
import ffmpeg
from pathlib import Path

from analyze_audio import prepare_audio
from analyze_frame import prepare_frame
from constants import DATA_DIR, STAGING_DIR

SPLIT_DELTA = 0.25


def split_and_import(path, output_dir=STAGING_DIR):

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

        base_name = Path(path).stem

        # Extract frame
        stream = ffmpeg.input(
            path,
            ss=point,
        )
        stream = ffmpeg.output(
            stream,
            "{}/{}-{}.png".format(output_dir, base_name, i),
            vframes=1,
            loglevel="quiet",
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)

        # Extract audio
        stream = ffmpeg.input(path, ss=point, t=SPLIT_DELTA)
        stream = ffmpeg.output(
            stream, "{}/{}-{}.wav".format(output_dir, base_name, i), loglevel="quiet"
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)


def stage_data():
    items = [
        join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))
    ]

    for item in items:
        split_and_import(item)


def process_data():
    items = [
        join(STAGING_DIR, f)
        for f in os.listdir(STAGING_DIR)
        if isfile(join(STAGING_DIR, f))
    ]

    failed = []

    for p in [f for f in items if "png" in f.split(".")]:
        if prepare_frame(p) is False:
            failed.append(Path(p).stem)

    for a in [f for f in items if "wav" in f.split(".")]:
        if Path(a).stem not in failed:
            prepare_audio(a)
        else:
            print("Skipping {}, face landmark recognition failed earlier...".format(a))


if __name__ == "__main__":
    stage_data()
    process_data()
