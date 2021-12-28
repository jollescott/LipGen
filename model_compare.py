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
from analyze_frame import analyze_frame

from constants import MODELS_DIR, PREVIEW_DIR
from build_data import split_and_import


def process_frame(path, model):
    surface = cairo.ImageSurface.create_from_png(path)
    ctx = cairo.Context(surface)

    model_input = analyze_audio(str.replace(path, ".png", ".wav"))

    predict = model.predict([model_input])[0]
    ctx.set_source_rgb(0, 1, 0)

    for i in range(0, len(predict), 2):
        x = predict[i] * surface.get_width()
        y = predict[i + 1] * surface.get_height()

        ctx.arc(x, y, 1, 0, 2 * math.pi)
        ctx.fill()

    dlib_truth = analyze_frame(path)
    ctx.set_source_rgb(0, 0, 1)
    for i in range(0, len(predict), 2):
        x = dlib_truth[i] * surface.get_width()
        y = dlib_truth[i + 1] * surface.get_height()

        ctx.arc(x, y, 1, 0, 2 * math.pi)
        ctx.fill()

    surface.write_to_png(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4")
    args = ap.parse_args()

    if exists(PREVIEW_DIR):
        shutil.rmtree(PREVIEW_DIR)

    preview_stage = join(PREVIEW_DIR, "staging")
    makedirs(preview_stage)

    split_and_import(args.mp4, preview_stage)

    models = [f for f in listdir(MODELS_DIR) if isfile(join(MODELS_DIR, f))]
    print("\n")

    for i, model in enumerate(models):
        print("{}. {} \n".format(i + 1, Path(model).stem))

    selected_model = int(input("Use model: ")) - 1
    model = load_model(join(MODELS_DIR, models[selected_model]))

    frames = [f for f in listdir(preview_stage) if "png" in f.split(".")]

    for frame in frames:
        process_frame(join(preview_stage, frame), model)
