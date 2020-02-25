import librosa

import matplotlib.pyplot as plt

import numpy as np

import simpleaudio as sa

samplerate = 33100


def load_audio(path, samplerate = samplerate, crop = True):
    data, sr =  librosa.load(path, None)


    if crop:
        #Find first sample above threshold
        start = 0
        while abs(data[start]) < 0.001:
            start += 1
        start = max(start - 10, 0)

        #Find last sample above threshold
        stop = len(data) - 1
        while abs(data[stop]) < 0.001:
            stop -= 1
        stop = min(stop + 10, len(data)-2)

        if start >= stop:
            print(path, start, stop)

        data = data[start:stop]

    data = librosa.resample(data, sr, samplerate)

    length = len(data) / samplerate * 1000 #Length of audio in ms.

    if crop:
        if length > 3300:
            print(path, length)

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


def plt_spec_ones(x, y):
    '''
    Input:
        x = spectrogram
        y = result of ones
    '''

    plt.subplot(2,1,0)

    
    


def play_sound(audio, samplerate):
    sound = audio * ( 32767 / max(abs(audio)))
    sound = sound.astype(np.int16)
    play_obj = sa.play_buffer(sound,1,2,samplerate)
    play_obj.wait_done()