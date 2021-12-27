import cv2
import dlib
import json
from os.path import join
from pathlib import Path

from constants import TEMP_DIR, STAGING_DIR

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def analyze_frame(path):
    image = cv2.imread(path)
    height, width, channels = image.shape
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(grayscale, 1)

    if len(rects) == 0:
        print('"{}" Failed to recognize any faces, skipping...'.format(path))
        return None

    rect = rects[0]

    shape = predictor(grayscale, rect)
    values = []

    for i in range(0, shape.num_parts):
        
        values.append(shape.part(i).x / width)
        values.append(shape.part(i).y / height)

    return values

def prepare_frame(path):
    values = analyze_frame(path)

    if values is None:
        return False

    with open(
        join(
            TEMP_DIR,
            "output",
            Path(path).stem + ".json",
        ),
        "w",
    ) as f:
        json.dump(values, f)

    return True
