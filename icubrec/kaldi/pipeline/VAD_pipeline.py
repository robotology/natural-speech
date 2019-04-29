#!/usr/bin/env python

'''
Module that implement the VAD pipeline. The pipeline continuously listen a yarp port where the audio captured by
microphone is streamed, then selects the parts where speech signal is detected. Each selected segment is put in a queue
consumed by the CommandRecognizer pipeline.
'''
__author__ = "Luca Pasa, Bertrand Higy"


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import yarp
import time
import numpy as np
from utils.IndexableQueue import IndexableQueue
from utils.logfbank_fun import logfbank
from python_speech_features.base import delta
from threading import Thread
from VAD_DNN import VAD
from scipy.io.wavfile import write

np.set_printoptions(threshold=np.nan)

import struct
import argparse
from signal_handler import SignalHandler



# data structure
received_data = []
audio_buffer = IndexableQueue()
audio_buffer.init_isFirst()
delta_buffer = IndexableQueue()
delta_buffer.init_isFirst()
delta_delta_buffer = IndexableQueue()
delta_delta_buffer.init_isFirst()
VAD_out_buffer = IndexableQueue()
VAD_out_buffer.init_isFirst()
smoother_buffer = IndexableQueue()
smoother_buffer.init_isFirst()

# DEBUG
debug_buffer = []

# parameters
fbanks_and_delta_window_size = 2
VAD_window_size = 5
smoother_window_size = 2
min_num_frame_speech_recognizer = 200 # 200 for sound_player, 300 for microphone
speech_recognizer_threshold = 0.5 # 0.1 for sound player ,0.7 for microphone
destination_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./save/")
print os.path.dirname(os.path.abspath(__file__))
downsampling_factor = 3
sampling_rate = 48000  #The command recognizer model is trained to manage 16kHz input but depending on the sound source
#the downsampling parameter may have to change in a way that sampling rate/downsampling factor= 16k
output_delay = 15

# VAD_Module
vad_obj = VAD.VADModule()

# Downsample var
counter = 0






def set_counter(val):
    counter = val


def get_counter():
    return counter


def toS16(uint):
    if (uint >> 15):  # is the sign bit set?
        return (-0x8000 + (uint & 0x7fff))  # "cast" it to signed
    return uint


class DisplayUpdater(yarp.TypedReaderCallbackSound):

    def __init__(self):
        yarp.TypedReaderCallbackSound.__init__(self)

    def onRead(self, *argv):
        '''
        # Function that receive the audio stream and compute the corresponding fbanks representation and put it in a
         buffer called audio_buffer. the function push in the queue a tuple (fbanks representation, wave_form)
        :param argv: argv[0] is the audio stream
        :return: True
        '''
        sound = argv[0]
        # Downgrade sample rate to 16k
        samples = [toS16(sound.get(i)) for i in range(0, sound.getSamples(), downsampling_factor)]
        received_data.extend(samples)
        debug_buffer.extend(samples)
        nb_samples = int(0.025 * sampling_rate/downsampling_factor)
        while len(received_data) >= (nb_samples):  # store 25 ms seconds
            current_frame = received_data[0:nb_samples]
            fbanks_input = computeFbanks(current_frame, sampling_rate=sampling_rate/downsampling_factor)[0]
            # The first is a numpy array of size (NUMFRAMES, nfilt) that contains the features. Each row holds 1 feature
            # vector. The second returned value is the energy in each frame
            audio_buffer.put((fbanks_input, current_frame))

            if audio_buffer.isFirst:
                for _ in range(
                        fbanks_and_delta_window_size):
                    audio_buffer.put((fbanks_input, current_frame))
                audio_buffer.isFirst = False
            del received_data[0:int(0.010 * sampling_rate/downsampling_factor)]  # delete the oldest 10 ms

        return True


def computeFbanks(sample, sampling_rate=sampling_rate, nfilt=40):

    '''
    Function that compute the mel-fbanks representation of an audio sample
    :param sample: the audio sample
    :param sampling_rate: the audio sample sampling rate (in Hz)
    :param nfilt: number of mel filter in the mel-fbanks representation
    :return: the mel-fbanks representation, composed of nfilt filter value and the energy
    '''
    log_fbank_array_input = np.squeeze(np.asarray(sample))
    fbank, energy = logfbank(np.asarray(log_fbank_array_input), sampling_rate, nfilt=nfilt)
    return np.concatenate((fbank, np.reshape(energy, (energy.shape[0], 1))), axis=1)

def computeDelta():
    '''
    function that compute the first derivative of the mel-fbanks representation. The function read the input data from
    the audio_buffer queue and put the results in delta_buffer queue. the results for each frame is a tuple
    (fbanks+delta, wave_form)
    Note: not used in this version of the pipeline
    '''
    num_frame = (fbanks_and_delta_window_size * 2 + 1)
    while True:
        if audio_buffer.qsize() >= num_frame:
            last_N_frames = np.asarray(audio_buffer.get_last_n_frame(num_frame))
            last_N_frames = [frame[0] for frame in last_N_frames]
            fbanks = np.squeeze(np.asarray((audio_buffer[fbanks_and_delta_window_size])[0]))
            wave_form = (audio_buffer[fbanks_and_delta_window_size])[1]
            audio_buffer.pull_last_n_frame(1)
            frame_delta = delta(last_N_frames, fbanks_and_delta_window_size)[fbanks_and_delta_window_size]
            delta_frame = np.concatenate((fbanks, frame_delta))
            delta_buffer.put((delta_frame, wave_form))
            if delta_buffer.isFirst:
                for _ in range(fbanks_and_delta_window_size):
                    delta_buffer.put((delta_frame, wave_form))
                delta_buffer.isFirst = False


def computeDeltaDelta(nfilt=41):
    '''
    function that compute the second derivative of the mel-fbanks representation. The function read the input data from
    the delta_buffer queue and put the results in delta_delta_buffer queue. he results for each frame is a tuple
    (fbanks+delta+deltaDelta, wave_form)
    Note: not used in this version of the pipeline
    '''
    num_frame = (fbanks_and_delta_window_size * 2 + 1)
    num_frame_VAD_window = (VAD_window_size * 2 + 1)
    while True:
        if delta_buffer.qsize() >= num_frame:
            last_N_frames = delta_buffer.get_last_n_frame(num_frame)
            last_N_frames = [frame[0] for frame in last_N_frames]
            last_N_frames = [d[:-nfilt] for d in last_N_frames]
            frame_delta = np.squeeze(np.asarray((delta_buffer[fbanks_and_delta_window_size])[0]))
            wave_form = (delta_buffer[fbanks_and_delta_window_size])[1]
            delta_buffer.pull_last_n_frame(1)
            frame_delta_delta = delta(last_N_frames, fbanks_and_delta_window_size)[fbanks_and_delta_window_size]
            delta_delta_frame = np.concatenate((frame_delta, frame_delta_delta))
            delta_delta_buffer.put((delta_delta_frame, wave_form))
            if delta_delta_buffer.isFirst:
                delta_delta_buffer.isFirst = False
                # add n times the first frame in order to allow th VAD to compute teh result even for the first frame
                for _ in range(num_frame_VAD_window - 1):
                    delta_delta_buffer.put((delta_delta_frame, wave_form))


def blackBoxVAD_withdelta():
    '''
    VAD method that consider a sequence of fbanks+delta+delta-delta representations and check if there is a speech or
    not. The check is performed frame by frame using a context window of  VAD_window_size frames. The function read the
    data from delta_delta_buffer and put the results (a real values that represent the probability that the  frame
    belongs to an audio command) to the VAD_out_buffer queue. the result is a tuple (frame,VAD_result,wave_form)
    Note: not used in this version of the pipeline
    '''
    num_frame_VAD_window = (VAD_window_size * 2 + 1)
    while True:
        if delta_delta_buffer.qsize() >= num_frame_VAD_window:
            current_window = delta_delta_buffer.get_last_n_frame(num_frame_VAD_window)
            current_window = [frame[0] for frame in current_window]
            delta_delta_buffer.pull_last_n_frame(1)
            current_frame = (delta_delta_buffer[VAD_window_size])[
                0]  # the current frame is the one in the center of the window
            wave_form = (delta_delta_buffer[VAD_window_size])[1]
            if current_window is not None:
                print current_window
                vad_result = int(vad_obj.GetOutput(current_window))
                print(vad_result)
                if VAD_out_buffer.isFirst:
                    VAD_out_buffer.isFirst = False
                    for _ in range(smoother_window_size):
                        VAD_out_buffer.put((current_frame, vad_result, wave_form))
                VAD_out_buffer.put((current_frame, vad_result, wave_form))


def blackBoxVAD():
    '''
        VAD method that consider a sequence of fbanks frame and check if there is a command or not. The check is
        performed frame by frame using a context window of VAD_window_size frames. The function read the data from
        audio_buffer and put the results (a real values that represent the probability that the frame belongs to an
        audio command) to the VAD_out_buffer queue. the result is a tuple (frame,VAD_result,wave_form)
    '''
    num_frame_VAD_window = (VAD_window_size * 2 + 1)
    while not SignalHandler.should_stop:
        if audio_buffer.qsize() >= num_frame_VAD_window:
            current_window = audio_buffer.get_last_n_frame(num_frame_VAD_window)
            current_window = [frame[0] for frame in current_window]
            audio_buffer.pull_last_n_frame(1)
            current_frame = (audio_buffer[VAD_window_size])[
                0]  # the current frame is the one in the center of the window
            wave_form = (audio_buffer[VAD_window_size])[1]
            if current_window is not None:
                vad_result = abs(1 - vad_obj.GetOutput(current_window))
                if VAD_out_buffer.isFirst:
                    VAD_out_buffer.isFirst = False
                    for _ in range(smoother_window_size):
                        VAD_out_buffer.put((current_frame, vad_result, wave_form))
                VAD_out_buffer.put((current_frame, vad_result, wave_form))  # store a tuple (frame,VAD_result,wave_form)


def smoother():
    '''
    Given the VAD results of all frames, the function selects the sequences of them where a command is detected. the
    sequences are selected by considering the moving average of the VAD results, and by using the
    speech_recognizer_threshold parameter as a threshold. The other parameter that influence the results is
    min_num_frame_speech_recognizer that define the minimum number of frames which a command is composed of. The
    waveform of the selected segments are save in destination_folder.
    '''
    index = 0
    n_verified_frame = 0
    smoother_inner_queue = IndexableQueue()
    smoother_inner_queue.init_isFirst()
    N_frames_window = smoother_window_size * 2 + 1
    while not SignalHandler.should_stop:
        # For each element in the VAD_out_buffer we extract a window of smoother_window_size * 2 +1 elements
        # the VAD put in the buffer smoother_window_size padding elements before the first element
        if VAD_out_buffer.qsize() >= N_frames_window:
            # get the window
            window = VAD_out_buffer.get_last_n_frame(N_frames_window)
            window = [int(frame[1]) for frame in window]
            # compute the moving avg of the window
            running_mean = np.convolve(window, np.ones((N_frames_window,)) / N_frames_window, mode='valid')
            print "running mean:", running_mean
            # check if the running avg is higher than the threshold
            if running_mean >= speech_recognizer_threshold:
                # if the queue is empty it is the first frame of a segment, so also the left part of the corresponding
                # window has to be saved
                if smoother_inner_queue.isFirst:
                    window = VAD_out_buffer.get_last_n_frame(N_frames_window)
                    frames = window[:-smoother_window_size]
                    for f in frames:
                        # put in the queue the waveform data
                        smoother_inner_queue.put(f[2])
                    smoother_inner_queue.isFirst = False
                # otherwise just the current frame (it is part of an ongoing segment) has to be saved
                else:
                    # put in the queue the waveform data
                    smoother_inner_queue.put((VAD_out_buffer[smoother_window_size])[2])
                n_verified_frame += 1
            else:
                # if the running avg is lower than the threshold:
                #   if the queue is not empty, check if the number of considered frames is higher then
                #   min_num_frame_speech_recognizer:
                #       if so:  save the segment, notice that also the right part to the window is part of the segment
                if not smoother_inner_queue.isFirst:
                    if n_verified_frame >= min_num_frame_speech_recognizer:
                        window = VAD_out_buffer.get_last_n_frame(N_frames_window)
                        frames = window[:smoother_window_size + 1]
                        for f in frames:
                            # save the waveform data
                            smoother_inner_queue.put(f[2])
                        # save file
                        save_file(smoother_inner_queue.toList(), str(index)[:3]+"_"+str(index)[-3:])
                        # reset the queue and the frame counter
                        smoother_inner_queue.empty_queue()
                        n_verified_frame = 0
                    else:
                        # if the number of considered frames is lower than the min_num_frame_speech_recognizer
                        # the frame could be put in the queue or discard the whole current segment
                        # in order to act in more conservative way, we decided to go for the first option
                        smoother_inner_queue.put((VAD_out_buffer[smoother_window_size])[2])
                        n_verified_frame += 1
                else:
                    # in this case the avg is lower then the threshold and the queue is empty:
                    #   just ignore the current frame
                    pass
            VAD_out_buffer.pull_last_n_frame(1)
        index += 1


def audio_reconstruction(segment):
    '''
    Given a list of the frame (waveform) function reconstruct the waveform of the whole segment
    :param segment: list of frame (waveform) that are part of the audio segment
    :return: the waveform of the audio segment
    '''
    rec = segment[0]

    for sample in segment[1::]:
        rec.extend(sample[:(sampling_rate/downsampling_factor) / 100])
    return rec



def save_file(segment, index):
    '''
    Function that save an audio segment and notify the Command Recognizer pipeline.
    :param segment: the audio segment
    :param index: index that wil be added to the name of the wav file in order to recognize the segment
    '''
    waveform = np.array(audio_reconstruction(segment))
    fname = os.path.join(destination_folder, str(index) + ".wav")
    waveform = waveform.astype(np.int16)
    write(fname, sampling_rate/downsampling_factor, waveform)
    # notify the CR_pipeline
    file_output_port.write(yarp.Value(fname))



if __name__ == '__main__':
    # Initializing parameters
    parser = argparse.ArgumentParser(description='Voice Activity Detection system.')
    parser.add_argument('-m', '--min_num_frame_speech_recognizer', type=int, default=200, help='200 for sound_player (DEFAULT), 300 for mic')
    parser.add_argument('-t', '--speech_recognizer_threshold', type=float, default=0.5, help='0.1 for sound player, 0.7 for mic, 0.5 DEFAULT')
    args = parser.parse_args()
    min_num_frame_speech_recognizer = args.min_num_frame_speech_recognizer
    print('min_num_frame_speech_recognizer:', min_num_frame_speech_recognizer)
    speech_recognizer_threshold = args.speech_recognizer_threshold
    print('speech_recognizer_threshold:', speech_recognizer_threshold)

    yarp.Network.init()
    p = yarp.BufferedPortSound()
    p.setStrict()
    updater = DisplayUpdater()
    p.useCallback(updater)
    p.open("/reader:i")

    # open yarp port to notify the CR_pipeline when a new file is create
    file_output_port = yarp.Port()
    file_output_port.open("/file_writer:o")

    thread_VAD = Thread(target=blackBoxVAD)
    thread_VAD.start()

    thread_smoother = Thread(target=smoother)
    thread_smoother.start()

    # Test code-----------------#
    output_timer = 0
    muted = False
    while not SignalHandler.should_stop:
        time.sleep(1)
        if output_timer % output_delay == 0:
            print("audio:", audio_buffer.qsize())
            print("delta:", delta_buffer.qsize())
            print("delta_delta", delta_delta_buffer.qsize())
            print("VAD_out_buffer", VAD_out_buffer.qsize())
        output_timer += 1
    # --------------------------#
    file_output_port.close()
    p.close()
