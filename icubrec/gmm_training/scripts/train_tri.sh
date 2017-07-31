#!/bin/bash
# Train the triphone models

cd $MODEL_FOLDER

rm -f -r $MODEL_FOLDER/hmm11 $MODEL_FOLDER/hmm12 $MODEL_FOLDER/hmm11.log $MODEL_FOLDER/hmm12.log
mkdir $MODEL_FOLDER/hmm11 $MODEL_FOLDER/hmm12

# HERest -B -A -T 1 -m 1 -d $TRAIN_WSJ0/hmm10 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1500.0 -S train.scp -H $TRAIN_WSJ0/hmm10/macros -H $TRAIN_WSJ0/hmm10/hmmdefs -M $TRAIN_WSJ0/hmm11 $TRAIN_WSJ0/triphones1 >$TRAIN_WSJ0/hmm11.log
train_iter.sh $MODEL_FOLDER hmm10 hmm11 triphones1 wintri.mlf 1

# Second round, also generate stats file we use for state tying
#HERest -B -A -T 1 -m 1 -d $TRAIN_WSJ0/hmm11 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1500.0 -s $TRAIN_WSJ0/stats -S train.scp -H $TRAIN_WSJ0/hmm11/macros -H $TRAIN_WSJ0/hmm11/hmmdefs -M $TRAIN_WSJ0/hmm12 $TRAIN_WSJ0/triphones1 >$TRAIN_WSJ0/hmm12.log
train_iter.sh $MODEL_FOLDER hmm11 hmm12 triphones1 wintri.mlf 1

# Copy the stats file off to the main directory for use in state tying
cp $MODEL_FOLDER/hmm12/stats_hmm12 $MODEL_FOLDER/stats
