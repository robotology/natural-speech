# Tutorial - Training a speech recognition system for WSJ dataset

## Overview

In this tutorial, we will see how to train and test a DNN-based model for WSJ0.
We assume here that you have access to the dataset and that the environment
variable $WSJ0_CORPORA (set in sys_dpdt.env) points to it. The main steps to
obtain a trained system are:
* extract features from the wavfiles.
* train a GMM-HMM model.
* compute the alignments for the training and validation data using the GMM-HMM
  model.
* train a DNN model to perform phoneme recognition using these alignments.
* test the DNN-HMM system offline.


We will now go through each step in detail.

## Prerequisites

* [Download](http://htk.eng.cam.ac.uk/) and install HTK.
* [Download and prepare CMU's dictionary](README.md#preparing-cmu-dictionary).
* Create a folder were all the data (feature files, models, ...) will be
  stored. We will suppose hereafter that the environment variable
  `$WSJ0_OTHER` points to this folder (file `sys_dpdt.env` should be changed
  accordingly). When using the environment file wsj0.env, `$CORPORA_OTHER` will
  point to the same folder.
* Build the dictionary and the word net using `build_lm_wsj.sh`.

      ./build_lm_wsj.sh -e gmm_training/wsj0.env

  This will create the files `dict_5k` and `wdnet_bigram` under
  `$CORPORA_OTHER` folder.

## Extracting the features

To train and test the system, we will use the 5k vocabulary,
speaker-independent part of WSJ0. The means:
* `si_tr_s` subset for training.
* `si_dt_05` subset for validation.
* `si_et_05` subset for testing.

Lists of the wavfiles we will use can be created with the commands:

    find $WSJ0_CORPORA/wsj0/si_tr_s -iname "*.wv1" >$WSJ0_OTHER/tr_wav.lst
    find $WSJ0_CORPORA/wsj0/si_dt_05 -iname "*.wv1" >$WSJ0_OTHER/dt_wav.lst
    find $WSJ0_CORPORA/wsj0/si_et_05 -iname "*.wv1" >$WSJ0_OTHER/et_wav.lst

From these wavfiles, we then need to extract two kind of features:
* Mel-Frequency Cepstral Coefficient (MFCC) for the GMM.
* Filter banks for the DNN model.

The feature extraction is done with the script `extract_feat.sh`. To extract
the features for both kind of models, run:

    mkdir -p $WSJ0_OTHER/feat/mfcc
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/tr_wav.lst $WSJ0_OTHER/feat/mfcc
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/dt_wav.lst $WSJ0_OTHER/feat/mfcc
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/et_wav.lst $WSJ0_OTHER/feat/mfcc
    mkdir -p $WSJ0_OTHER/feat/fbanks
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/tr_wav.lst $WSJ0_OTHER/feat/fbanks
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/dt_wav.lst $WSJ0_OTHER/feat/fbanks
    ./extract_feat.sh -e gmm_training/wsj0.env -f $WSJ0_OTHER/et_wav.lst $WSJ0_OTHER/feat/fbanks

The folders `$WSJ0/mfcc` and `$WSJ0/fbanks` should now contain the same folder
hierarchy as `$WSJ0_CORPORA/wsj0` for our 3 subsets, with a `.feat` file in place
of each `.wv1` file.

Finally, similarly to what we did for the wavfiles, we will create lists
containing all the feature files. However, in order to be able to switch easily
from one kind of feature to the other, we will decouple the features location
(e.g.  `$WSJ0_OTHER/feat/mfcc`) from the subsequent path (e.g.
`si_tr_s/011/011c0201.feat`). The features root folder is defined by an
environment variable while the feature list will only contain the relative path
under this root folder. The lists of feature files for our 3 subsets can by
running:

    cd $WSJ0_CORPORA
    find si_tr_s -iname "*.wv1" >$WSJ0_OTHER/tr_feat.lst
    find si_dt_05 -iname "*.wv1" >$WSJ0_OTHER/dt_feat.lst
    find si_et_05 -iname "*.wv1" >$WSJ0_OTHER/et_feat.lst
    cd -

## Training the GMM-HMM model

From `gmm_training` subfolder (in this repo), simply launch the training script:

    mkdir $WSJ0_OTHER/gmm
    ./train.sh -e wsj0.env $WSJ0_OTHER/gmm

## Computing alignments

Once the model is trained, we can use it to compute the phones alignments (or
transcriptions) on the training and validation sets. These is done using
`HVite` program from HTK in the following way (still from `gmm_training`):

    source wsj0.env
    HVite -T 1 -C $HTK_COMMON/wi.htkc \
          -H $WSJ0_OTHER/gmm/hmm42/macros -H $WSJ0_OTHER/gmm/hmm42/hmmdefs \
          -S $WSJ0_OTHER/gmm/train.scp -I $WSJ0_OTHER/gmm/words.mlf \
          -f -b silence -y lab -o SW $HTK_DATA/cmu/cmu6spsil \
          $WSJ0_OTHER/gmm/tiedlist >$WSJ0_OTHER/hvite_alignment.log

    # Create word-level MLF file for validation set
    mkdir $WSJ0_OTHER/gmm/dt
    ./scripts/make_mlf.sh $WSJ0_OTHER/gmm/dt $WSJ0_OTHER/dt_feat.lst validation.scp words.mlf test
    HVite -T 1 -C $HTK_COMMON/wi.htkc \
          -H $WSJ0_OTHER/gmm/hmm42/macros -H $WSJ0_OTHER/gmm/hmm42/hmmdefs \
          -S $WSJ0_OTHER/gmm/dt/validation.scp -I $WSJ0_OTHER/gmm/dt/words.mlf \
          -f -b silence -y lab -o SW $HTK_DATA/cmu/cmu6spsil \
          $WSJ0_OTHER/gmm/dt/tiedlist >$WSJ0_OTHER/dthvite_alignment.log

The folder `$WSJ0_OTHER/feat/mfcc` should now contain `.lab` files for each of
the `.feat` files of the training and validation subsets.

## Training the DNN model

Once we have the alignments, it's again quite straightforward to train a DNN to perform framebased state recognition. Simply go to the `dnn_training` subfolder and run:

    mkdir $WSJ0_OTHER/dnn
    ./train.sh -e wsj0.env $WSJ0_OTHER/dnn

The best Frame Error Rate (FER) for the validation set should be around 61.88%.

## Test the system

Now that the DNN acoustic model is trained, we can test it. For that, use the script `test.sh` available under `offline_decoding` folder. Supposing you're in this later folder, you can simply type:

    mkdir $WSJ0_OTHER/dnn/results
    ./test.sh -e ../dnn/wsj0.env $WSJ0_OTHER/dnn $WSJ0_OTHER/et_feat.lst $WSJ0_OTHER/dnn/results

The results will be stored in the file `hresults_et_prune350.0_wi.log` under `$WSJ0_OTHER/dnn/results`. As a comparison point, we got an accuracy of 91.28% with this training procedure.
