#!/bin/bash
# Init the HMM models based on TIMIT phonetic transcriptions.
# Then train up the monophone models.

cd $TIMIT_OTHER
sed "s:^:$TIMIT_OTHER/feat/$FEATURE_FOLDER/:g" $TIMIT_OTHER/lists/train_feat.lst >$MODEL_FOLDER/timit/train.scp

cd $MODEL_FOLDER/timit
# Create monophones0 from monophones1, eliminating the sp model
cp $HTK_COMMON/timit/monophones1 monophones1
grep -v "^sp" monophones1 > monophones0

# Clean out any old directories or log files
rm -f -r phone0 phone1 hmm1 hmm2 hmm3 hmm4 hmm5 hmm6 hmm7 hmm8
mkdir phone0 phone1 hmm1 hmm2 hmm3 hmm4 hmm5 hmm6 hmm7 hmm8
rm -f hrest.log hinit.log hmm1.log hmm2.log hmm3.log hmm4.log hmm5.log hmm6.log hmm7.log hmm8.log hhed_sil.log

# We need to run HInit for each phone in the monophones0 file
cd $TIMIT_DIR
perl $HTK_SCRIPTS/ProcessNums.pl $MODEL_FOLDER/timit/monophones0 $SCRIPTS_PATH/template_hinit
cd $MODEL_FOLDER/timit

# Train using HRest
cd $TIMIT_DIR
perl $HTK_SCRIPTS/ProcessNums.pl $MODEL_FOLDER/timit/monophones0 $SCRIPTS_PATH/template_hrest
cd $MODEL_FOLDER/timit

# At this point, we should have decent set of monophones stored in
# $MODEL_FOLDER/timit/phone1/hmmdefs, but we'll carry on and do BW
# reestimation using the phone labeled data.

# Figure out the global variance, we do this so we can have a floor
# on the variances in further re-estimation steps.
cd $TIMIT_DIR

# HCompV parameters:
#  -C   Config file to load, gets us the TARGETKIND = MFCC_0_D_A_Z
#  -f   Create variance floor equal to value times global variance
#  -m   Update the means as well (not needed?)
#  -S   File listing all the feature vector files
#  -M   Where to store the output files
#  -I   MLF containg phone labels of feature vector files
HCompV -A -T 1 -C $HTK_COMMON/$FEAT_CONF_FILE -f 0.01 -m -S $MODEL_FOLDER/timit/train.scp -M $MODEL_FOLDER/timit/phone1 -I $MODEL_FOLDER/timit/phone.mlf $HTK_COMMON/proto >$MODEL_FOLDER/timit/hcompv.log
cd $MODEL_FOLDER/timit
cp $HTK_COMMON/macros phone1
cat phone1/vFloors >> phone1/macros

# We don't actually want to use the global means since we went through
# the trouble of not flat starting.
rm -f ./phone1/proto

# Now do three rounds of Baum-Welch reesimtation of the monophone models
# using the phone-level transcriptions.
cd $TIMIT_DIR

# HERest parameters:
#  -d    Where to look for the monophone defintions in
#  -C    Config file to load
#  -I    MLF containing the phone-level transcriptions
#  -t    Set pruning threshold (3.2.1)
#  -S    List of feature vector files
#  -H    Load this HMM macro definition file
#  -M    Store output in this directory
HERest -A -T 1 -d $MODEL_FOLDER/timit/phone1 -C $HTK_COMMON/$FEAT_CONF_FILE -I $MODEL_FOLDER/timit/phone.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/phone1/macros -H $MODEL_FOLDER/timit/phone1/hmmdefs -M $MODEL_FOLDER/timit/hmm1 $MODEL_FOLDER/timit/monophones0 >$MODEL_FOLDER/timit/hmm1.log

#HERest -A -T 1 -C $HTK_COMMON/config -I $MODEL_FOLDER/timit/phone.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/hmm1/macros -H $MODEL_FOLDER/timit/hmm1/hmmdefs -M $MODEL_FOLDER/timit/hmm2 $MODEL_FOLDER/timit/monophones0 >$MODEL_FOLDER/timit/hmm2.log
train_iter.sh $MODEL_FOLDER/timit hmm1 hmm2 monophones0 phone.mlf 3

#HERest -A -T 1 -C $HTK_COMMON/config -I $MODEL_FOLDER/timit/phone.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/hmm2/macros -H $MODEL_FOLDER/timit/hmm2/hmmdefs -M $MODEL_FOLDER/timit/hmm3 $MODEL_FOLDER/timit/monophones0 >$MODEL_FOLDER/timit/hmm3.log
train_iter.sh $MODEL_FOLDER/timit hmm2 hmm3 monophones0 phone.mlf 3 text

cd $MODEL_FOLDER/timit

# We'll fix the silence model and add in our short pause sp.
# This form of silence is different from the tutorial as sp will
# have three states, all tied to sil.  sp will allow transition
# without any output and both will have transitions from 2 to 4.
perl $HTK_SCRIPTS/DuplicateSilence.pl hmm3/hmmdefs >hmm4/hmmdefs
cp hmm3/macros hmm4/macros

HHEd -A -T 1 -H hmm4/macros -H hmm4/hmmdefs -M hmm5 $HTK_COMMON/sil.hed monophones1 >hhed_sil.log

cd $TIMIT_DIR

# Now do more training of the new sp model using an MLF that has
# the sp between words and sil just before and after sentences.
#HERest -A -T 1 -C $HTK_COMMON/config -I $MODEL_FOLDER/timit/phone_sp.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/hmm5/macros -H $MODEL_FOLDER/timit/hmm5/hmmdefs -M $MODEL_FOLDER/timit/hmm6 $MODEL_FOLDER/timit/monophones1 >$MODEL_FOLDER/timit/hmm6.log
train_iter.sh $MODEL_FOLDER/timit hmm5 hmm6 monophones1 phone_sp.mlf 3

#HERest -A -T 1 -C $HTK_COMMON/config -I $MODEL_FOLDER/timit/phone_sp.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/hmm6/macros -H $MODEL_FOLDER/timit/hmm6/hmmdefs -M $MODEL_FOLDER/timit/hmm7 $MODEL_FOLDER/timit/monophones1 >$MODEL_FOLDER/timit/hmm7.log
train_iter.sh $MODEL_FOLDER/timit hmm6 hmm7 monophones1 phone_sp.mlf 3

#HERest -A -T 1 -C $HTK_COMMON/config -I $MODEL_FOLDER/timit/phone_sp.mlf -t 250.0 150.0 1000.0 -S $MODEL_FOLDER/timit/train.scp -H $MODEL_FOLDER/timit/hmm7/macros -H $MODEL_FOLDER/timit/hmm7/hmmdefs -M $MODEL_FOLDER/timit/hmm8 $MODEL_FOLDER/timit/monophones1 >$MODEL_FOLDER/timit/hmm8.log
train_iter.sh $MODEL_FOLDER/timit hmm7 hmm8 monophones1 phone_sp.mlf 3
 
