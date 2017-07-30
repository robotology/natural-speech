#!/bin/bash

# Aligns a new MLF based on the best monophone models.
#
# Parameters:
#  1 - "flat" if we are flat starting from monophone models living
#      in hmm5 in this directory.

# Cleanup old files
rm -f $MODEL_FOLDER/hvite_align.log $MODEL_FOLDER/hled_sp_sil.log

# Do alignment using our best monophone models to create a phone-level MLF
# HVite parameters
#  -l       Path to use in the names in the output MLF
#  -o SWT   How to output labels, S remove scores,
#           W do not include words, T do not include times
#  -b       Use this word as the sentence boundary during alignment
#  -C       Config files
#  -a       Perform alignment
#  -H       HMM macro definition files
#  -i       Output to this MLF file
#  -m       During recognition keep track of model boundaries
#  -t       Enable beam searching
#  -y       Extension for output label files
#  -I       Word level MLF file
#  -S       File contain the list of MFC files

if [[ $1 != "flat" ]]
then
    HVite -A -T 1 -o SWT -b silence -C $HTK_COMMON/$FEAT_CONFIG_FILE -a -H $MODEL_FOLDER/timit/hmm8/macros -H $MODEL_FOLDER/timit/hmm8/hmmdefs -i $MODEL_FOLDER/aligned.mlf -m -t 250.0 -I $MODEL_FOLDER/words.mlf -S $MODEL_FOLDER/train.scp $HTK_DATA/cmu/cmu6spsil $MODEL_FOLDER/timit/monophones1 >$MODEL_FOLDER/hvite_align.log
else
    HVite -A -T 1 -o SWT -b silence -C $HTK_COMMON/$FEAT_CONFIG_FILE -a -H $MODEL_FOLDER/hmm5/macros -H $MODEL_FOLDER/hmm5/hmmdefs -i $MODEL_FOLDER/aligned.mlf -m -t 250.0 -I $MODEL_FOLDER/words.mlf -S $MODEL_FOLDER/train.scp $HTK_DATA/cmu/cmu6spsil $MODEL_FOLDER/monophones1 >$MODEL_FOLDER/hvite_align.log
fi

# We'll get a "sp sil" sequence at the end of each sentence.  Merge these
# into a single sil phone.  Also might get "sil sil", we'll merge anything
# combination of sp and sil into a single sil.
HLEd -A -T 1 -i $MODEL_FOLDER/aligned2.mlf $HTK_COMMON/merge_sp_sil.led $MODEL_FOLDER/aligned.mlf >$MODEL_FOLDER/hled_sp_sil.log

# Forced alignment might fail for a few files (why?), these will be missing
# from the MLF, so we need to prune these out of the script so we don't try
# and train on them.
cp $MODEL_FOLDER/train.scp $MODEL_FOLDER/train_temp.scp
perl $SCRIPTS_PATH/RemovePrunedFiles.pl $MODEL_FOLDER/aligned2.mlf $MODEL_FOLDER/train_temp.scp >$MODEL_FOLDER/train.scp
rm -f $MODEL_FOLDER/train_temp.scp

