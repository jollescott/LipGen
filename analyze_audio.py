from pathlib import Path
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np
import json

import argparse
from os.path import join

from constants import AMPLITUDE_THRESHOLD, INPUTS, OUTPUT_DIR, STAGING_DIR


def wavfile_fft(path):
    sr, y = wavfile.read(path)

    y = y / 2.0 ** 15
    signal = y.T[0]

    fft_spectrum = fft(signal)
    freq = fftfreq(signal.size, d=1.0 / sr)

    fft_spectrum_abs = np.abs(fft_spectrum)
    freq_abs = np.abs(freq)

    return freq_abs, fft_spectrum_abs


def process_audio(path):
    freq, spectrum = wavfile_fft(path)

    values = []

    for i, f in enumerate(spectrum):
        if f > AMPLITUDE_THRESHOLD:
            values.append((np.round(freq[i]), np.round(f)))

    values = sorted(values, key=lambda t: t[1], reverse=True)
    result = []

    for i in range(0, INPUTS):
        result.append(values[i][0] if i < len(values) else 0)

    with open(join(OUTPUT_DIR, "input", Path(path).stem + ".json"), "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wavfile")
    args = ap.parse_args()

    fx, fy = wavfile_fft(join(STAGING_DIR, args.wavfile))

    for i, f in enumerate(fy):
        if f > 50:
            print(
                "frequency = {} Hz with amplitude {} ".format(
                    np.round(fx[i], 1), np.round(f)
                )
            )

    plt.plot(fx, fy)
    plt.xlabel("frequency, Hz")
    plt.ylabel("Amplitude, units")
    plt.show()
