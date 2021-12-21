import cv2
import dlib
import argparse
from imutils import face_utils
import json
from os.path import join
from pathlib import Path

from constants import OUTPUT_DIR, STAGING_DIR

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")


def process_frame(path):
    image = cv2.imread(path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(grayscale, 1)

    if len(rects) == 0:
        print('"{}" Failed to recognize any faces, skipping...'.format(path))
        return False

    rect = rects[0]

    shape = predictor(grayscale, rect)
    values = []

    for i in range(0, shape.num_parts):
        values.append(shape.part(i).x)
        values.append(shape.part(i).y)

    with open(
        join(
            OUTPUT_DIR,
            "output",
            Path(path).stem + ".json",
        ),
        "w",
    ) as f:
        json.dump(values, f)

    return True
