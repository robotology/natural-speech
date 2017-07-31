
# Mixup the number of Gaussians per state, from 1 up to 8.
# We do this in 4 steps, with 4 rounds of reestimation 
# each time.  We mix to 8 to match paper "Large Vocabulary
# Continuous Speech Recognition Using HTK"
#
# Also per Phil Woodland's comment in the mailing list, we
# will let the sp/sil model have double the number of 
# Gaussians.
#
# This version does sil mixup to 2 first, then from 2->4->6->8 for
# normal and double for sil.

cd $MODEL_FOLDER

# Prepare new directories for all our model files
rm -f -r $MODEL_FOLDER/hmm18 $MODEL_FOLDER/hmm19 $MODEL_FOLDER/hmm20 $MODEL_FOLDER/hmm21 $MODEL_FOLDER/hmm22 $MODEL_FOLDER/hmm23 $MODEL_FOLDER/hmm24 $MODEL_FOLDER/hmm25 $MODEL_FOLDER/hmm26 $MODEL_FOLDER/hmm27 $MODEL_FOLDER/hmm28 $MODEL_FOLDER/hmm29 $MODEL_FOLDER/hmm30 $MODEL_FOLDER/hmm31 $MODEL_FOLDER/hmm32 $MODEL_FOLDER/hmm33 $MODEL_FOLDER/hmm34 $MODEL_FOLDER/hmm35 $MODEL_FOLDER/hmm36 $MODEL_FOLDER/hmm37 $MODEL_FOLDER/hmm38 $MODEL_FOLDER/hmm39 $MODEL_FOLDER/hmm40 $MODEL_FOLDER/hmm41 $MODEL_FOLDER/hmm42
mkdir $MODEL_FOLDER/hmm18 $MODEL_FOLDER/hmm19 $MODEL_FOLDER/hmm20 $MODEL_FOLDER/hmm21 $MODEL_FOLDER/hmm22 $MODEL_FOLDER/hmm23 $MODEL_FOLDER/hmm24 $MODEL_FOLDER/hmm25 $MODEL_FOLDER/hmm26 $MODEL_FOLDER/hmm27 $MODEL_FOLDER/hmm28 $MODEL_FOLDER/hmm29 $MODEL_FOLDER/hmm30 $MODEL_FOLDER/hmm31 $MODEL_FOLDER/hmm32 $MODEL_FOLDER/hmm33 $MODEL_FOLDER/hmm34 $MODEL_FOLDER/hmm35 $MODEL_FOLDER/hmm36 $MODEL_FOLDER/hmm37 $MODEL_FOLDER/hmm38 $MODEL_FOLDER/hmm39 $MODEL_FOLDER/hmm40 $MODEL_FOLDER/hmm41 $MODEL_FOLDER/hmm42
rm -f $MODEL_FOLDER/hmm18.log $MODEL_FOLDER/hmm19.log $MODEL_FOLDER/hmm20.log $MODEL_FOLDER/hmm21.log $MODEL_FOLDER/hmm22.log $MODEL_FOLDER/hmm23.log $MODEL_FOLDER/hmm24.log $MODEL_FOLDER/hmm25.log $MODEL_FOLDER/hmm26.log $MODEL_FOLDER/hmm27.log $MODEL_FOLDER/hmm28.log $MODEL_FOLDER/hmm29.log $MODEL_FOLDER/hmm30.log $MODEL_FOLDER/hmm31.log $MODEL_FOLDER/hmm32.log $MODEL_FOLDER/hmm33.log $MODEL_FOLDER/hmm34.log $MODEL_FOLDER/hmm35.log $MODEL_FOLDER/hmm36.log $MODEL_FOLDER/hmm37.log $MODEL_FOLDER/hmm38.log $MODEL_FOLDER/hmm39.log $MODEL_FOLDER/hmm40.log $MODEL_FOLDER/hmm41.log $MODEL_FOLDER/hmm42.log $MODEL_FOLDER/hhed_mixup2.log $MODEL_FOLDER/hhed_mixup3.log $MODEL_FOLDER/hhed_mixup4.log $MODEL_FOLDER/hhed_mixup5.log $MODEL_FOLDER/hhed_mixup8.log $MODEL_FOLDER/hhed_mixup12.log $MODEL_FOLDER/hhed_mixup16.log

#cd $WSJ1_DIR

# HERest parameters:
#  -d    Where to look for the monophone defintions in
#  -C    Config file to load
#  -I    MLF containing the phone-level transcriptions
#  -t    Set pruning threshold (3.2.1)
#  -S    List of feature vector files
#  -H    Load this HMM macro definition file
#  -M    Store output in this directory
#  -m    Minimum examples needed to update model

# As per the CSTIT notes, do four rounds of reestimation (more than
# in the tutorial).

#######################################################
# Mixup sil from 1->2
HHEd -B -H $MODEL_FOLDER/hmm17/macros -H $MODEL_FOLDER/hmm17/hmmdefs -M $MODEL_FOLDER/hmm18 $HTK_COMMON/mix1.hed $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hhed_mix1.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm18/macros -H $MODEL_FOLDER/hmm18/hmmdefs -M $MODEL_FOLDER/hmm19 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm19.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm19/macros -H $MODEL_FOLDER/hmm19/hmmdefs -M $MODEL_FOLDER/hmm20 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm20.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm20/macros -H $MODEL_FOLDER/hmm20/hmmdefs -M $MODEL_FOLDER/hmm21 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm21.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm21/macros -H $MODEL_FOLDER/hmm21/hmmdefs -M $MODEL_FOLDER/hmm22 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm22.log

train_iter.sh $MODEL_FOLDER hmm18 hmm19 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm19 hmm20 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm20 hmm21 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm21 hmm22 tiedlist wintri.mlf 0

#######################################################
# Mixup 1->2, sil 2->4
HHEd -B -H $MODEL_FOLDER/hmm22/macros -H $MODEL_FOLDER/hmm22/hmmdefs -M $MODEL_FOLDER/hmm23 $HTK_COMMON/mix2.hed $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hhed_mix2.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm23/macros -H $MODEL_FOLDER/hmm23/hmmdefs -M $MODEL_FOLDER/hmm24 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm24.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm24/macros -H $MODEL_FOLDER/hmm24/hmmdefs -M $MODEL_FOLDER/hmm25 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm25.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm25/macros -H $MODEL_FOLDER/hmm25/hmmdefs -M $MODEL_FOLDER/hmm26 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm26.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm26/macros -H $MODEL_FOLDER/hmm26/hmmdefs -M $MODEL_FOLDER/hmm27 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm27.log

train_iter.sh $MODEL_FOLDER hmm23 hmm24 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm24 hmm25 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm25 hmm26 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm26 hmm27 tiedlist wintri.mlf 0

#######################################################
# Mixup 2->4, sil from 4->8
HHEd -B -H $MODEL_FOLDER/hmm27/macros -H $MODEL_FOLDER/hmm27/hmmdefs -M $MODEL_FOLDER/hmm28 $HTK_COMMON/mix4.hed $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hhed_mix4.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm28/macros -H $MODEL_FOLDER/hmm28/hmmdefs -M $MODEL_FOLDER/hmm29 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm29.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm29/macros -H $MODEL_FOLDER/hmm29/hmmdefs -M $MODEL_FOLDER/hmm30 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm30.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm30/macros -H $MODEL_FOLDER/hmm30/hmmdefs -M $MODEL_FOLDER/hmm31 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm31.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm31/macros -H $MODEL_FOLDER/hmm31/hmmdefs -M $MODEL_FOLDER/hmm32 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm32.log

train_iter.sh $MODEL_FOLDER hmm28 hmm29 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm29 hmm30 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm30 hmm31 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm31 hmm32 tiedlist wintri.mlf 0

#######################################################
# Mixup 4->6, sil 8->12
HHEd -B -H $MODEL_FOLDER/hmm32/macros -H $MODEL_FOLDER/hmm32/hmmdefs -M $MODEL_FOLDER/hmm33 $HTK_COMMON/mix6.hed $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hhed_mix6.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm33/macros -H $MODEL_FOLDER/hmm33/hmmdefs -M $MODEL_FOLDER/hmm34 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm34.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm34/macros -H $MODEL_FOLDER/hmm34/hmmdefs -M $MODEL_FOLDER/hmm35 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm35.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm35/macros -H $MODEL_FOLDER/hmm35/hmmdefs -M $MODEL_FOLDER/hmm36 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm36.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm36/macros -H $MODEL_FOLDER/hmm36/hmmdefs -M $MODEL_FOLDER/hmm37 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm37.log

train_iter.sh $MODEL_FOLDER hmm33 hmm34 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm34 hmm35 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm35 hmm36 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm36 hmm37 tiedlist wintri.mlf 0

#######################################################
# Mixup 6->8, sil 12->16
HHEd -B -H $MODEL_FOLDER/hmm37/macros -H $MODEL_FOLDER/hmm37/hmmdefs -M $MODEL_FOLDER/hmm38 $HTK_COMMON/mix8.hed $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hhed_mix8.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm38/macros -H $MODEL_FOLDER/hmm38/hmmdefs -M $MODEL_FOLDER/hmm39 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm39.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm39/macros -H $MODEL_FOLDER/hmm39/hmmdefs -M $MODEL_FOLDER/hmm40 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm40.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm40/macros -H $MODEL_FOLDER/hmm40/hmmdefs -M $MODEL_FOLDER/hmm41 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm41.log

#HERest -B -m 0 -A -T 1 -C $HTK_COMMON/$FEAT_CONFIG_FILE -I $MODEL_FOLDER/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $MODEL_FOLDER/hmm41/macros -H $MODEL_FOLDER/hmm41/hmmdefs -M $MODEL_FOLDER/hmm42 $MODEL_FOLDER/tiedlist >$MODEL_FOLDER/hmm42.log

train_iter.sh $MODEL_FOLDER hmm38 hmm39 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm39 hmm40 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm40 hmm41 tiedlist wintri.mlf 0
train_iter.sh $MODEL_FOLDER hmm41 hmm42 tiedlist wintri.mlf 0
