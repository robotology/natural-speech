#!/bin/bash

# Evaluate the TIMIT monophone models by testing them on their
# own training data (unfair but tests for fundamental brokenness). 
# Uses a simple word loop grammar where each word is a monophone.
# We use monophones0 since sp is allowed in the grammar since 
# it has a transition with no output.

cd $MODEL_FOLDER/timit

rm -f $MODEL_FOLDER/timit/hbuild.log $MODEL_FOLDER/timit/hresults.log $MODEL_FOLDER/timit/hvite.log

# Create a dictionary where each word is a monophone 
perl $SCRIPTS_PATH/DuplicateLine.pl $MODEL_FOLDER/timit/monophones0 >$MODEL_FOLDER/timit/dict_monophones0

# Build the word network
HBuild -A -T 1 $MODEL_FOLDER/timit/monophones0 $MODEL_FOLDER/timit/wdnet_monophones0 >$MODEL_FOLDER/timit/hbuild.log

# Recognize the data on the final monophone models
rm -f $MODEL_FOLDER/timit/hresults.log

# HVite parameters:
#  -H    HMM macro definition files to load
#  -S    List of feature vector files to recognize
#  -i    Where to output the recognition MLF file
#  -w    Word network to you as language model
#  -p    Insertion penalty
#  -s    Language model scale factor
HVite -A -T 1 -H $MODEL_FOLDER/timit/hmm8/macros -H $MODEL_FOLDER/timit/hmm8/hmmdefs -S $MODEL_FOLDER/timit/train.scp -i $MODEL_FOLDER/timit/recout.mlf -w $MODEL_FOLDER/timit/wdnet_monophones0 -p -1.0 -s 4.0 $MODEL_FOLDER/timit/dict_monophones0 $MODEL_FOLDER/timit/monophones0 >$MODEL_FOLDER/timit/hvite.log

# Now lets see how we did!
HResults -A -T 1 -I $MODEL_FOLDER/timit/phone.mlf $MODEL_FOLDER/timit/monophones1 $MODEL_FOLDER/timit/recout.mlf >$MODEL_FOLDER/timit/hresults.log

