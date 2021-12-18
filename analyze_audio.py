from scipy.io import wavfile
from scipy.fft import fft


def extract_frequencies_magnitudes(path):
    y, sr = wavfile.read(path)
    t = fft(sr)
    print(t)
