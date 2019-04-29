#!/bin/bash

# Evaluate on the November 92 ARPA test set.
#
# This version doesn't produce the lattice and can be
# used for a final evaluation using a larger pruning
# value and previous tuned penalty and scale factor.
# Doesn't consume as much time or resources to run as
# the lattice producing version of the script.
#
# Parameters:
#  1 - Directory name of model to test
#  2 - Distinguishing name for this test run.
#  3 - HVite pruning value
#  4 - Insertion penalty
#  5 - Language model scale factor

cd $RESULT_FOLDER

rm -f $RESULT_FOLDER/hvite_et$2.log $RESULT_FOLDER/hresults_et$2.log

# HVite parameters:
#  -H    HMM macro definition files to load
#  -S    List of feature vector files to recognize
#  -i    Where to output the recognition MLF file
#  -w    Word network to you as language model
#  -p    Insertion penalty
#  -s    Language model scale factor
#  -z    Extension for lattice output files
#  -n    Number of tokens in a state (bigger number means bigger lattices)

# We'll run with some reasonable values for insertion penalty and LM scale,
# but these will need to be tuned.

# We need to send in a different config file depending on whether
# we are doing cross word triphones or not.
if [[ $MODEL_TYPE == "GMM" ]]; then
    HVite -A -T 1 -t $3 -C $HTK_COMMON/$CONFIG_CTXEXP -H $MODEL_FOLDER/$1/macros -H $MODEL_FOLDER/$1/$GMM_HMMDEFS -S $RESULT_FOLDER/dataset.scp -i $RESULT_FOLDER/recout_et$2.mlf -w $GRAM_FILE -p $4 -s $5 $DICT_FILE $MODEL_FOLDER/$GMM_HMMLIST >$RESULT_FOLDER/hvite_et$2.log
else
    cp -rf $MODEL_FOLDER/cvn $MODEL_FOLDER/ident_cvn $RESULT_FOLDER
    HVite -A -T 1 -t $3 -C $HTK_COMMON/$CONFIG_CTXEXP -C $MODEL_FOLDER/basic.htkc -H $MODEL_FOLDER/$1/$GMM_HMMDEFS -S $RESULT_FOLDER/dataset.scp -i $RESULT_FOLDER/recout_et$2.mlf -w $GRAM_FILE -p $4 -s $5 $DICT_FILE $MODEL_FOLDER/$GMM_HMMLIST >$RESULT_FOLDER/hvite_et$2.log
fi


# Now lets see how we did!
cd $RESULT_FOLDER
HResults -n -A -T 1 -I $RESULT_FOLDER/words.mlf $MODEL_FOLDER/$GMM_HMMLIST $RESULT_FOLDER/recout_et$2.mlf >$RESULT_FOLDER/hresults_et$2.log

# Add on a NIST style output result for good measure
HResults -n -h -A -T 1 -I $RESULT_FOLDER/words.mlf $MODEL_FOLDER/$GMM_HMMLIST $RESULT_FOLDER/recout_et$2.mlf >>$RESULT_FOLDER/hresults_et$2.log
