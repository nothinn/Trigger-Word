import numpy as np

import keras

import utils
import random

import math

import time

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, backgrounds, activations, negatives, Ty, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 samplerate=33100, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.backgrounds = backgrounds
        self.samplerate = samplerate
        self.activations = activations
        self.negatives = negatives

        self.Ty = Ty
        
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.activations) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.activations[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.activations))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.Ty, 1))

        # Generate data, make a 10 second clip with 3 words from which one is the activation word
        for i, ID in enumerate(list_IDs_temp):

            # Randomly choose background
            background = random.choice(self.backgrounds)

            background, length = utils.load_audio(background,self.samplerate, crop = False)

            # Randomly choose two negative words
            neg1 = random.choice(self.negatives)
            neg1, neg1_length = utils.load_audio(neg1,self.samplerate)
            while neg1_length > 3300:
                neg1 = random.choice(self.negatives)
                neg1, neg1_length = utils.load_audio(neg1,self.samplerate)

            #neg2 = random.choice(self.negatives)
            #neg2, neg2_length = utils.load_audio(neg2,self.samplerate)

            # Load the activation word
            act, act_length = utils.load_audio(ID,self.samplerate)



            previous_segments = []
            y[i] = np.zeros((self.Ty,1)) #Fill out result vector
            #Insert activation word randomly
            background, segment_time = insert_audio_clip(background, act, act_length, self.samplerate, previous_segments)
            segment_start, segment_end = segment_time
            y[i] = insert_ones(y[i], segment_end, self.Ty)

            #Insert the two negatives randomly
            background, segment_time = insert_audio_clip(background, neg1, neg1_length, self.samplerate, previous_segments)
            #background, segment_time = insert_audio_clip(background, neg2, neg2_length, previous_segments)

            
            # calculate spectrum, should be of shape (5511,101) where 5511 is the timesteps and 101 are the fft results
            spectrum = get_spectrum(background)

            X[i, ] = spectrum

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)


def get_spectrum(signal):
    #We want to have 128 frequency bins so we take twice the size in blocks.

    #We also use 50% overlap, which approx. doubles the number of timesteps

    entries = math.ceil(len(signal)/256)

    result = []

    start = 0
    stop = 256

    while start < len(signal):

        if len(signal) < start + 256:
            block = np.concatenate(( signal[start:], np.zeros(stop - len(signal))))
        else:
            block = signal[start:stop]

        assert(len(block) == 256)

        result.append(get_fft(block))
        start += 128
        stop += 128
        


    for i in range(entries):
        if (i+1) * 256 >= len(signal):
            tmp = np.zeros(256)
            for i in range((i+1)*256,len(signal)):
                tmp[i] = signal[(i+1)*256]
            result[i] = get_fft(tmp)
        else:
            result[i] = get_fft(signal[i*256:(i+1) * 256])

    return result



def get_fft(block):
    '''
    Should have 2^n samples for speed
    '''
    np.seterr(divide = 'ignore') 
    fft = np.fft.fft(block * np.hamming(len(block)))
    return np.log2( abs(fft)[0:int(len(fft)/2)])


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)
    
def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.
    
    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
    
    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """
    
    segment_start, segment_end = segment_time
    
    ### START CODE HERE ### (≈ 4 line)
    # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
    overlap = False
    
    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True
    ### END CODE HERE ###

    return overlap



def insert_audio_clip(background, audio_clip, duration, samplerate, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the 
    audio segment does not overlap with existing segments.
    
    Arguments:
    background -- a 10 second background audio recording.  
    audio_clip -- the audio clip to be inserted/overlaid. 
    previous_segments -- times where audio segments have already been placed
    
    Returns:
    new_background -- the updated background audio
    """
    
    # Get the duration of the audio clip in ms
    segment_ms = duration
    
    ### START CODE HERE ### 
    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
    # the new audio clip. (≈ 1 line)
    segment_time = get_random_time_segment(segment_ms)
    
    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
    # picking new segment_time at random until it doesn't overlap. (≈ 2 lines)
    t_end = time.time() + 5 #may only run for 5 seconds
    while is_overlapping(segment_time, previous_segments):
        if time.time() > t_end:
            raise Exception("Too long to find new segment: " + str(previous_segments) + " Duration: " + str(duration))
        segment_time = get_random_time_segment(segment_ms)

    # Step 3: Add the new segment_time to the list of previous_segments (≈ 1 line)
    previous_segments.append(segment_time)
    ### END CODE HERE ###
    
    # Step 4: Superpose audio segment and background
    for i in range(len(audio_clip)):
        background[i + (segment_time[0] * int((samplerate / 1000)))] = background[i + int(segment_time[0]*(samplerate / 1000))] + audio_clip[i]
    
    return background, segment_time


def insert_ones(y, segment_end_ms, Ty):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.
    
    
    Arguments:
    y -- numpy array of shape (Ty,1), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels
    """
    
    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(segment_end_y + 1, segment_end_y + 1 + Ty//30): #Ty//30 gives ~ 45 ones for 1375 outputs
        if i < Ty:
            y[i, 0] = 1
    ### END CODE HERE ###
    
    return y