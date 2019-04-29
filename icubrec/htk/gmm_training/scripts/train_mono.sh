#!/bin/bash

# Take the best TIMIT monophone models and reestimate using the
# forced aligned phone transcriptions of WSJ0.
#
# Parameters:
#  $1 - "flat" if we are flat starting from monophone models living
#       in hmm5 in this directory.

cd $MODEL_FOLDER

# Copy our lists of monophones over from TIMIT directory
cp $HTK_COMMON/timit/monophones0 .
cp $HTK_COMMON/timit/monophones1 .

# Cleanup old files and directories
rm -f -r $MODEL_FOLDER/hmm6 $MODEL_FOLDER/hmm7 $MODEL_FOLDER/hmm8 $MODEL_FOLDER/hmm9
mkdir $MODEL_FOLDER/hmm6 $MODEL_FOLDER/hmm7 $MODEL_FOLDER/hmm8 $MODEL_FOLDER/hmm9
rm -f $MODEL_FOLDER/hmm6.log $MODEL_FOLDER/hmm7.log $MODEL_FOLDER/hmm8.log $MODEL_FOLDER/hmm9.log

# Now do three rounds of Baum-Welch reestimation of the monophone models
# using the phone-level transcriptions.
if [[ $1 != "flat" ]]; then
    # Copy over the TIMIT monophones to the same directory that a
    # flat-start would use.
    mkdir -p $MODEL_FOLDER/hmm5
    cp -f $MODEL_FOLDER/timit/hmm8/* $MODEL_FOLDER/hmm5
fi

# We'll create a new variance floor macro that reflects 1% of the
# global variance over our WSJ0 + WSJ1 training data.

# First convert to text format so we can edit the macro file
mkdir -p $MODEL_FOLDER/hmm5_text
HHEd -H $MODEL_FOLDER/hmm5/hmmdefs -H $MODEL_FOLDER/hmm5/macros -M $MODEL_FOLDER/hmm5_text /dev/null $MODEL_FOLDER/monophones1

# HCompV parameters:
#  -C   Config file to load, gets us the TARGETKIND = MFCC_0_D_A_Z
#  -f   Create variance floor equal to value times global variance
#  -m   Update the means as well (not needed?)
#  -S   File listing all the feature vector files
#  -M   Where to store the output files
#  -I   MLF containg phone labels of feature vector files
HCompV -A -T 1 -C $HTK_COMMON/$FEAT_CONF_FILE -f 0.01 -m -S $MODEL_FOLDER/train.scp -M $MODEL_FOLDER/hmm5_text -I $MODEL_FOLDER/aligned2.mlf $HTK_COMMON/proto >$MODEL_FOLDER/hcompv.log
cp $HTK_COMMON/macros $MODEL_FOLDER/hmm5_text/macros
cat $MODEL_FOLDER/hmm5_text/vFloors >> $MODEL_FOLDER/hmm5_text/macros

train_iter.sh $MODEL_FOLDER hmm5_text hmm6 monophones1 aligned2.mlf 3
train_iter.sh $MODEL_FOLDER hmm6 hmm7 monophones1 aligned2.mlf 3
train_iter.sh $MODEL_FOLDER hmm7 hmm8 monophones1 aligned2.mlf 3

# Do an extra round just so we end up with hmm9 and synched with the tutorial
train_iter.sh $MODEL_FOLDER hmm8 hmm9 monophones1 aligned2.mlf 3
