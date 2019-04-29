
# Train the word internal phonetic decision tree state tied models

cd $MODEL_FOLDER

# Cleanup old files and create new directories for model files
rm -f -r $MODEL_FOLDER/hmm14 $MODEL_FOLDER/hmm15 $MODEL_FOLDER/hmm16 $MODEL_FOLDER/hmm17
mkdir $MODEL_FOLDER/hmm14 $MODEL_FOLDER/hmm15 $MODEL_FOLDER/hmm16 $MODEL_FOLDER/hmm17
rm -f $MODEL_FOLDER/hmm14.log $MODEL_FOLDER/hmm15.log $MODEL_FOLDER/hmm16.log $MODEL_FOLDER/hmm17.log

# HERest parameters:
#  -d    Where to look for the monophone defintions in
#  -C    Config file to load
#  -I    MLF containing the phone-level transcriptions
#  -t    Set pruning threshold (3.2.1)
#  -S    List of feature vector files
#  -H    Load this HMM macro definition file
#  -M    Store output in this directory
#  -m    Sets the minimum number of examples for training, by setting 
#        to 0 we stop suprious warnings about no examples for the 
#        sythensized triphones
#
# As per the CSTIT notes, do four rounds of reestimation (more than
# in the tutorial).

#HERest -B -A -T 1 -m 0 -C $HTK_COMMON/$FEAT_CONF_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $TRAIN_WSJ0/hmm13/macros -H $TRAIN_WSJ0/hmm13/hmmdefs -M $TRAIN_WSJ0/hmm14 $TRAIN_WSJ0/tiedlist >$TRAIN_WSJ0/hmm14.log
train_iter.sh $MODEL_FOLDER hmm13 hmm14 tiedlist wintri.mlf 0

#HERest -B -A -T 1 -m 0 -C $HTK_COMMON/$FEAT_CONF_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $TRAIN_WSJ0/hmm14/macros -H $TRAIN_WSJ0/hmm14/hmmdefs -M $TRAIN_WSJ0/hmm15 $TRAIN_WSJ0/tiedlist >$TRAIN_WSJ0/hmm15.log
train_iter.sh $MODEL_FOLDER hmm14 hmm15 tiedlist wintri.mlf 0

#HERest -B -A -T 1 -m 0 -C $HTK_COMMON/$FEAT_CONF_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $TRAIN_WSJ0/hmm15/macros -H $TRAIN_WSJ0/hmm15/hmmdefs -M $TRAIN_WSJ0/hmm16 $TRAIN_WSJ0/tiedlist >$TRAIN_WSJ0/hmm16.log
train_iter.sh $MODEL_FOLDER hmm15 hmm16 tiedlist wintri.mlf 0

#HERest -B -A -T 1 -m 0 -C $HTK_COMMON/$FEAT_CONF_FILE -I $TRAIN_WSJ0/wintri.mlf -t 250.0 150.0 1000.0 -S train.scp -H $TRAIN_WSJ0/hmm16/macros -H $TRAIN_WSJ0/hmm16/hmmdefs -M $TRAIN_WSJ0/hmm17 $TRAIN_WSJ0/tiedlist >$TRAIN_WSJ0/hmm17.log
train_iter.sh $MODEL_FOLDER hmm16 hmm17 tiedlist wintri.mlf 0
