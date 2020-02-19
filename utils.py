import librosa

import matplotlib.pyplot as plt

import numpy as np

samplerate = 33100


def load_audio(path, samplerate = samplerate):
    data, sr =  librosa.load(path, samplerate)

    length = len(data) / sr #Length of audio in seconds.

    return (data, length)


def hamming_window_signal(signal):

    return signal * np.hamming(len(signal))



def fft(signal, samplerate = samplerate):
    converted = np.fft.rfft(signal)


    print("Shape: {}".format(converted.shape))


    plt.specgram(signal, Fs=samplerate)
    
    plt.savefig("spectrogram.png")
    plt.close()

    return converted


def plt_values(values, path = "temp"):


    plt.plot(values)

    plt.savefig(path + ".png")
    plt.close()