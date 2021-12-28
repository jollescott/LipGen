import os
from os.path import isfile, join, exists
import ffmpeg
import argparse
from pathlib import Path
import numpy as np
import concurrent.futures
import shutil

from analyze_audio import process_audio
from analyze_frame import process_frame
from constants import DATA_DIR, INPUT_DIR, OUTPUT_DIR, STAGING_DIR

REQUIRED_DIRS = [OUTPUT_DIR, INPUT_DIR, STAGING_DIR]


def process_worker_entry(audio_files):
    for audio_file in audio_files:
        img_file = str.replace(audio_file, ".wav", ".png")

        if process_frame(img_file) == False:
            print("{} failed to process frame, skipping...".format(img_file))
            continue

        process_audio(audio_file)


def process_data(worker_count=1):
    audio_files = [
        join(STAGING_DIR, f)
        for f in os.listdir(STAGING_DIR)
        if isfile(join(STAGING_DIR, f)) and "wav" in f.split(".")
    ]

    work_dist = np.array_split(audio_files, worker_count)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(process_worker_entry, work_dist)


def split_and_import(path, output_dir=STAGING_DIR):

    probe = ffmpeg.probe(path)

    stream_of_type = lambda t: next(
        (s for s in probe["streams"] if s["codec_type"] == t), None
    )

    audio_stream = stream_of_type("audio")
    video_stream = stream_of_type("video")

    if audio_stream is None or video_stream is None:
        print("{} is missing audio stream, skipping...".format(path))
        return

    audio_duration = float(audio_stream["duration"])
    video_duration = float(video_stream["duration"])
    nb_video_frames = int(video_stream["nb_frames"])

    video_frame_rate = nb_video_frames / video_duration
    delta_time = 1 / video_frame_rate

    i = 0
    step = 0

    while step < audio_duration:
        base_name = Path(path).stem

        # Extract frame
        stream = ffmpeg.input(path, ss=step)
        stream = ffmpeg.output(
            stream,
            "{}/{}-{}.png".format(output_dir, base_name, i),
            vframes=1,
            loglevel="quiet",
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)

        # Extract audio
        stream = ffmpeg.input(path, ss=step, t=delta_time)
        stream = ffmpeg.output(
            stream, "{}/{}-{}.wav".format(output_dir, base_name, i), loglevel="quiet"
        )
        stream = ffmpeg.overwrite_output(stream)
        ffmpeg.run(stream)

        i += 1
        step += delta_time


def import_worker_entry(files):
    for file in files:
        split_and_import(file)


def import_data(worker_count=1):
    import_files = [
        join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if isfile(join(DATA_DIR, f))
    ]

    dist_work = np.array_split(import_files, worker_count)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(import_worker_entry, dist_work)


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

    for dir in REQUIRED_DIRS:
        if exists(dir):
            shutil.rmtree(dir)

        os.makedirs(dir)

    import_data(args.workers)
    process_data(args.workers)
