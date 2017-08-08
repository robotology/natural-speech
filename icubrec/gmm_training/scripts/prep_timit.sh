#!/bin/bash

# Prepare the files needed to train on the TIMIT corpus
#
# This version only uses the CMU dictionary and uses
# both the training and test files from TIMIT.
#
# We are using the CMU set of 39 phonemes

# First we need to generate a file that contains all the
# filenames of all the TIMIT format phone level
# transcriptions.
cd $TIMIT_OTHER/transcr_keith
find -iname S*.PHN >$MODEL_FOLDER/timit/phone_files.txt

# Create new phone labels that also have sp between words.  We'll
# use these transcriptions once we add in the sp model after the
# intial training is complete.
perl $SCRIPTS_PATH/AddSpToTimit.pl $MODEL_FOLDER/timit/phone_files.txt PHN_SP
find -iname S*.PHN_SP >$MODEL_FOLDER/timit/phone_sp_files.txt

# Convert all the TIMIT phone labels to our smaller set and
# put them into a big MLF file.
HLEd -A -T 1 -D -n $MODEL_FOLDER/timit/tlist -i $MODEL_FOLDER/timit/phone.mlf -G TIMIT -S $MODEL_FOLDER/timit/phone_files.txt $HTK_COMMON/timit.led >$MODEL_FOLDER/timit/hhed_convert.log
sed -i "s:^\"\./:\"$TIMIT_OTHER/feat/$FEATURE_FOLDER/:g" $MODEL_FOLDER/timit/phone.mlf

# Same thing but for the version that has sp in it
HLEd -A -T 1 -D -n $MODEL_FOLDER/timit/tlist -i $MODEL_FOLDER/timit/temp.mlf -G TIMIT -S $MODEL_FOLDER/timit/phone_sp_files.txt $HTK_COMMON/timit.led >$MODEL_FOLDER/timit/hhed_convert_sp.log

# We could get several sp's in a row in the above due to sp being added
# between words and deletion of epi symbol in TIMIT transcription.
# We'll merge them back into a single sp phone.
HLEd -A -T 1 -i $MODEL_FOLDER/timit/phone_sp.mlf $HTK_COMMON/merge_sp.led $MODEL_FOLDER/timit/temp.mlf >$MODEL_FOLDER/timit/hled_sp.log
sed -i "s:^\"\./:\"$TIMIT_OTHER/feat/$FEATURE_FOLDER/:g" $MODEL_FOLDER/timit/phone_sp.mlf
rm -f $MODEL_FOLDER/timit/temp.mlf
