import librosa

import matplotlib.pyplot as plt

import numpy as np

import simpleaudio as sa

samplerate = 33100


loaded_audio = {}

def load_audio(path, samplerate = samplerate, crop = True, length = None):

    if path in loaded_audio:
        return loaded_audio[path]
    else:
        data, sr =  librosa.load(path, None)
                
        if "background" in path:
            pass
            #data = librosa.util.normalize(data) * 0.001
            data = np.ones(data.shape) * 0.001
        else:
            data = librosa.util.normalize(data)



        if crop:
            data, _ = librosa.effects.trim(data)
            
            '''
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
            '''

        data = librosa.resample(data, sr, samplerate)

        if length:
            if len(data) < samplerate * length:
                #print("Added in the beginning")
                data = np.concatenate((np.zeros(int(samplerate*length - len(data))),data),axis=0)
                #print(data.shape)
            elif len(data) > samplerate*length:
                #print("subtracted beginning")
                data = data[len(data)-samplerate*length:]

        length = len(data) / samplerate * 1000 #Length of audio in ms.

        if crop:
            if length > 3300:
                pass
                #print(path, length)

        loaded_audio[path] = (data,length)
        if length > 1200:
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


def plt_values(values, path = "temp", store=False):
    if not store:
        plt.close()
    plt.plot(values)

    #plt.show()

    plt.savefig(path + ".png")
    if not store:
        plt.close()


def plt_history(history, path):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/" + path + "_accuracy.png")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("plots/" + path + "_loss.png")
    plt.close()


def plt_spec(values, path):
    plt.close()
    plt.imshow(np.swapaxes(values,0,1), cmap='hot',interpolation='nearest', origin='lower', aspect='auto')
    plt.savefig(path + "_spec.png")
    plt.close()


def plt_spec_res(x, y):
    np.swapaxes(x,0,1)
    number = np.random.randint(2)
    plt.imshow(x)
    plt.savefig("plots/Spectogram_{}.png".format(number))
    plt.close()

    number += 1


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