import numpy as np
from utils.logfbank_fun import logfbank
from python_speech_features.base import delta
from scipy.io.wavfile import read
import glob
import os


win_size=2

def compute_fbanks_dataset(path="",nfilt=40):
    for filename in glob.glob(os.path.join(path, '*.wav')):
        sample_rate, audio_data = read(filename)
        fbanks,energy=logfbank(signal=audio_data,samplerate=sample_rate,nfilt=nfilt)
        fbanks = np.concatenate((fbanks, np.reshape(energy,(energy.shape[0],1))),axis=1)

        fbanks_delta=delta(feat=fbanks, N=win_size)
        fbanks_delta_delta=delta(feat=fbanks_delta, N=win_size)
        audio_features= np.concatenate((fbanks,fbanks_delta,fbanks_delta_delta),axis=1)
        filename, _=os.path.splitext(filename)
        print filename
        np.save(filename, audio_features)




if __name__ == '__main__':
    compute_fbanks_dataset('/home/storage/Data/vocub_icub/*/*/')
