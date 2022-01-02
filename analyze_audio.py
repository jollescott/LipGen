from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse
from os.path import join

from constants import AMPLITUDE_THRESHOLD, INPUT_DIR, INPUTS


def wavfile_fft(path):
    sr, y = wavfile.read(path)
    signal = y.T[0]

    fft_amp = fft(signal)
    freq = fftfreq(signal.size, d=1.0 / sr)

    freq_n = int(len(freq) / 2)
    amp_n = int(len(fft_amp) / 2)

    return np.abs(freq[freq_n:]), np.abs(fft_amp[amp_n:])


def analyze_audio(path):
    freq, amp = wavfile_fft(path)

    values = ((freq[i], f) for i, f in enumerate(amp))
    values = filter(lambda x: 125 < x[0] < 8000, values)

    values = sorted(values, key=lambda t: t[1], reverse=True)
    result = []

    for i in range(0, INPUTS):
        result.append(values[i][0])

    return result


def process_audio(path):
    result = analyze_audio(path)

    with open(join(INPUT_DIR, Path(path).stem + ".json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wavfile")
    args = ap.parse_args()

    fx, fy = wavfile_fft(args.wavfile)

    print(fy)

    for i, f in enumerate(fy):
        if f > AMPLITUDE_THRESHOLD:
            print(
                "frequency = {} Hz with amplitude {} ".format(
                    np.round(fx[i], 1), np.round(f)
                )
            )

    plt.plot(fx, fy)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.show()
