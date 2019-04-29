# Demo - Offline decoding with a pretrained model for the VoCub dataset

This page describes how to test the performance of the pretrained model for the
VoCub dataset. This serves as a starting point for people interested in our
framework and demonstrates its capabilities.

## Prerequisites

1. [Download](http://htk.eng.cam.ac.uk/) and install HTK. We will suppose
   hereafter that HTK is installed under `$HTK_DIR`.
1. Download the [VoCub
   dataset](https://zenodo.org/record/834934/files/vocub.tar.gz). We will
suppose hereafter that the dataset is saved under `$VOCUB_DIR`.
1. Download the [pretrained
   model](https://zenodo.org/record/836692/files/gmm_vocub.tar.gz) for VoCub
and extract it. We will suppose hereafter that the model is saved under
`$MODEL_DIR`.

## Testing the model

From the command line:
1. Move to `$MODEL_DIR` folder.
1. Generate the list of test utterances:

        find $VOCUB_DIR -iname "*_1_*.wav" -o -iname "*_2_*.wav" >et_wav.lst

1. Additionally, a file containing the transcriptions of all test utterances is
   required. This file can be generated manually (see
[test.sh](offline_decoding/test.sh) for example) but here, for conveniency, it
is provided under `icubrec` folder ([icubrec/words_vocub.mlf](words_vocub.mlf))
and should be copied to `$MODEL_DIR`.
1. Then, to perform decoding on the test set using our pretrained model, simply
   execute following command (still from `$MODEL_DIR`):

        $HTK_DIR/HTKTools/HVite -C config_on -H mmf -i recout.mlf -w wdnet \ 
        -p 4.0 -s 15 -S et_wav.lst dict tiedlist

1. Finally, the results can be agregated using the command:

        $HTK_DIR/HTKTools/HResults -n -I words_vocub.mlf tiedlist recout.mlf \
        >hresults.log

   You should get a word error rate of 93.12, as shown from the file
`hresults.log`:

        ====================== HTK Results Analysis =======================
          Date: Mon Nov 20 11:56:47 2017
          Ref : words_vocub.mlf
          Rec : recout.mlf
        ------------------------ Overall Results --------------------------
        SENT: %Correct=83.47 [H=394, S=78, N=472]
        WORD: %Corr=93.66, Acc=93.12 [H=2083, D=12, S=129, I=12, N=2224]
        ===================================================================

## Going further

This demo shows how the pretrained model for VoCub dataset can be used for
offline decoding. [Pretrained models](models/README.md) for several other
datasets are available to replace the model we have used here. Also, the same
models can be used for [online recognition within
yarp](yarp_decoding/README.md). Finally, [a tutorial](TUTORIAL.md) explains in
detail how to train your own model, using Wall Street Journal dataset as an
example.
