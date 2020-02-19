import librosa

import matplotlib.pyplot as plt

import numpy as np

import simpleaudio as sa

samplerate = 33100


def load_audio(path, samplerate = samplerate):
    data, sr =  librosa.load(path, samplerate)

    length = len(data) / sr * 1000 #Length of audio in ms.

    return data, length


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

    plt.show()

    plt.savefig(path + ".png")
    plt.close()

def plt_spec(values):
    plt.imshow(values, cmap='hot',interpolation='nearest', origin='lower')
    plt.show()



def play_sound(audio, samplerate):
    sound = audio * ( 32767 / max(abs(audio)))
    sound = sound.astype(np.int16)
    play_obj = sa.play_buffer(sound,1,2,samplerate)
    play_obj.wait_done()