# iCubrec

This folder contains code to train, test and run an automatic speech
recognition (ASR) system. Even though the scripts are quite generic and can be
used to trained models for other purposes, our ultimate goal is to provide
tools to perform command recognition on the iCub plateform.

The code is split into two subfolders:
* `htk` contains the first version of the code which was based on the [Hidden
Markov Model ToolKit (HTK)](http://htk.eng.cam.ac.uk/). As HTK doesn't allow
live recognition with deep neural networks (DNNs), it is based on Gaussian
mixture models (GMMs) instead.
* `kaldi` contains a more recent version of the pipeline based on
[kaldi](http://kaldi-asr.org). It uses a DNN-based acoustic model and
incorporates a voice activity detection (VAD) system to allow hand-free online
detection of commands.

## License

The code is released under GPLv3 license.
