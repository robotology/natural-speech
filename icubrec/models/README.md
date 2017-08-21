# Pretrained models

As the examples provided here use mainly non freely available datasets, and to ease the use of the speech recognition system for non expert users, we offer pretrained models (available [here](https://zenodo.org/record/836692)). This file provide a detailed descriptions of the those models.

## GMM models

Three different GMMs are proposed, trained for WSJ0, Chime4 and VoCub datasets respectively. This last model is the model of main interest for iCub users, allowing to recognize some basic commands out of the box (see [VoCub's website](https://robotology.github.io/natural-speech/vocub/index.html) for the complete list of commands).

More details about the training procedure are given in the table below.

|Archive name|Training dataset(s)|Monophone starting models|Context expansion|
|-|
|[gmm_wsj0](https://zenodo.org/record/836692/files/gmm_wsj0.tar.gz)|WSJ0|Trained from TIMIT|Cross word|
|[gmm_chime4](https://zenodo.org/record/836692/files/gmm_chime4.tar.gz)|Chime4|Flat start|Word internal|
|[gmm_vocub](https://zenodo.org/record/836692/files/gmm_vocub.tar.gz)|Chime4 + VoCub|Flat start|Word internal|

## DNN models

We also proposed pretrained DNN-based models for TIMIT and WSJ0.

More details about the training procedure are given in the table below.

|Archive name|Training dataset(s)|Context expansion|Optimizer|Learning rate|Decay factor|
|-|
|[dnn_timit](https://zenodo.org/record/836692/files/dnn_timit.tar.gz)|TIMIT|Word internal|Newbob|3e-5|0.75|
|[dnn_wsj0](https://zenodo.org/record/836692/files/dnn_wsj0.tar.gz)|WSJ0|Word internal|Newbob|3e-5|0.75|
