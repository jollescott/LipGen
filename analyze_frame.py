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


def process_frame(path, preview=False):
    image = cv2.imread(path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(grayscale, 1)

    if len(rects) == 0:
        print('"{}" Failed to recognize any faces, skipping...'.format(path))
        return

    rect = rects[0]

    shape = predictor(grayscale, rect)
    shape = face_utils.shape_to_np(shape)

    if preview:
        (x, y, b, h) = face_utils.rect_to_bb(rect)

        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        cv2.imshow("Preview", image)
        cv2.waitKey()
    else:
        with open(
            join(
                OUTPUT_DIR,
                "output",
                Path(path).stem + ".json",
            ),
            "w",
        ) as f:
            json.dump(shape.tolist(), f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()

    process_frame(join(STAGING_DIR, args.image))
