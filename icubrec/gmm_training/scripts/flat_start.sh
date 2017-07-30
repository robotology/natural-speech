#!/bin/bash

# If previous monophone models aren't available (say from TIMIT), then
# this script can be used to flat start the models using the word 
# level MLF of WSJ0.

rm -f -r $MODEL_FOLDER/hmm0 $MODEL_FOLDER/hhed_flat.log $MODEL_FOLDER/hcompv_flat.log $MODEL_FOLDER/hmm1 $MODEL_FOLDER/hmm2 $MODEL_FOLDER/hmm3 $MODEL_FOLDER/hmm4 $MODEL_FOLDER/hmm5
mkdir $MODEL_FOLDER/hmm0 $MODEL_FOLDER/hmm1 $MODEL_FOLDER/hmm2 $MODEL_FOLDER/hmm3 $MODEL_FOLDER/hmm4 $MODEL_FOLDER/hmm5
cp $HTK_COMMON/timit/monophones0 $MODEL_FOLDER
cp $HTK_COMMON/timit/monophones1 $MODEL_FOLDER

# First convert the word level MLF into a phone MLF
HLEd -A -T 1 -l '*' -d $HTK_DATA/cmu/cmu6 -i $MODEL_FOLDER/phones0.mlf $HTK_COMMON/mkphones0.led $MODEL_FOLDER/words.mlf >$MODEL_FOLDER/hhed_flat.log

# Compute the global mean and variance and set all Gaussians in the given
# HMM to have the same mean and variance

# HCompV parameters:
#  -C   Config file to load, gets us the TARGETKIND = MFCC_0_D_A_Z
#  -f   Create variance floor equal to value times global variance
#  -m   Update the means as well
#  -S   File listing all the feature vector files
#  -M   Where to store the output files
HCompV -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -f 0.01 -m -S $MODEL_FOLDER/train.scp -M $MODEL_FOLDER/hmm0 $HTK_COMMON/proto >$MODEL_FOLDER/hcompv_flat.log

# Create the master model definition and macros file
cd $MODEL_FOLDER
cp $HTK_COMMON/macros $MODEL_FOLDER/hmm0
cat $MODEL_FOLDER/hmm0/vFloors >> $MODEL_FOLDER/hmm0/macros
perl $HTK_SCRIPTS/CreateHMMDefs.pl $MODEL_FOLDER/hmm0/proto $MODEL_FOLDER/monophones0 >$MODEL_FOLDER/hmm0/hmmdefs

# Okay now to train up the models
#
# HERest parameters:
#  -d    Where to look for the monophone defintions in
#  -C    Config file to load
#  -I    MLF containing the phone-level transcriptions
#  -t    Set pruning threshold (3.2.1)
#  -S    List of feature vector files
#  -H    Load this HMM macro definition file
#  -M    Store output in this directory
train_iter.sh $MODEL_FOLDER hmm0 hmm1 monophones0 phones0.mlf 3 text
train_iter.sh $MODEL_FOLDER hmm1 hmm2 monophones0 phones0.mlf 3 text
train_iter.sh $MODEL_FOLDER hmm2 hmm3 monophones0 phones0.mlf 3 text

cd $MODEL_FOLDER

# Finally we'll fix the silence model and add in our short pause sp 
# See HTKBook 3.2.2.
perl $HTK_SCRIPTS/DuplicateSilence.pl $MODEL_FOLDER/hmm3/hmmdefs >$MODEL_FOLDER/hmm4/hmmdefs
cp $MODEL_FOLDER/hmm3/macros $MODEL_FOLDER/hmm4/macros

HHEd -A -T 1 -H $MODEL_FOLDER/hmm4/macros -H $MODEL_FOLDER/hmm4/hmmdefs -M $MODEL_FOLDER/hmm5 $HTK_COMMON/sil.hed $MODEL_FOLDER/monophones1 >$MODEL_FOLDER/hhed_flat_sil.log
