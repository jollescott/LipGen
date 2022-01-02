from genericpath import exists
from pathlib import Path
from os.path import isfile, join
from os import listdir, makedirs
from keras.models import load_model
import argparse
import shutil
import cairo
from analyze_audio import analyze_audio
import math
import ffmpeg

from constants import MODELS_DIR, PREVIEW_DIR
from build_data import split_and_import


def process_frame(path, model):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB30, 366, 288)
    ctx = cairo.Context(surface)

    model_input = analyze_audio(path)

    predict = model.predict([model_input])[0]
    ctx.set_source_rgb(0, 1, 0)

    for i in range(0, len(predict), 2):
        x = predict[i] * surface.get_width()
        y = predict[i + 1] * surface.get_height()

        ctx.arc(x, y, 1, 0, 2 * math.pi)
        ctx.fill()

    surface.write_to_png(str.replace(path, ".wav", ".png"))


def choose_model():
    models = [f for f in listdir(MODELS_DIR) if isfile(join(MODELS_DIR, f))]
    print("\n")

    for i, model in enumerate(models):
        print("{}. {} \n".format(i + 1, Path(model).stem))

    selected_model = int(input("Use model: ")) - 1
    model = load_model(join(MODELS_DIR, models[selected_model]))
    return model


def build_frames(output_dir, model):
    frames = [f for f in listdir(output_dir) if "wav" in f.split(".")]

    for frame in frames:
        process_frame(join(output_dir, frame), model)


def build_result_sequence(frame_dir, original_path):
    video_input = ffmpeg.input(
        "{}/*.png".format(frame_dir), pattern_type="glob", framerate=25
    )
    original_input = ffmpeg.input(
        original_path
    )
    stream = ffmpeg.output(video_input, original_input.audio, "temp/output.mp4", loglevel='quiet')
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    args = ap.parse_args()

    if exists(PREVIEW_DIR):
        shutil.rmtree(PREVIEW_DIR)

    preview_stage = join(PREVIEW_DIR, "staging")
    makedirs(preview_stage)

    split_and_import(args.mp4, preview_stage)

    model = choose_model()

    build_frames(preview_stage, model)
    build_result_sequence(preview_stage, args.mp4)
