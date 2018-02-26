#!/bin/bash

# Prepare things for the tied state triphones
#
# Parameters:
#   1 - RO value for clustering
#   2 - TB value for clustering
#   3 - "cross" if we are doing cross word triphones
#
# We need to create a list of all the triphone contexts we might
# see based on the whole dictionary (not just what we see in
# the training data).

cd $MODEL_FOLDER

rm -f -r $MODEL_FOLDER/hmm13 $MODEL_FOLDER/hhed_cluster.log $MODEL_FOLDER/fullist $MODEL_FOLDER/tree.hed
mkdir $MODEL_FOLDER/hmm13

# We have our own script which generate all possible monophone,
# left and right biphones, and triphones.  It will also add
# an entry for sp and sil
if [[ $3 != "cross" ]]
then
perl $SCRIPTS_PATH/CreateFullListWI.pl $HTK_DATA/cmu/cmu6 >$MODEL_FOLDER/fulllist
else
perl $SCRIPTS_PATH/CreateFullList.pl $MODEL_FOLDER/monophones0 >$MODEL_FOLDER/fulllist
fi

# Now create the instructions for doing the decision tree clustering

# RO sets the outlier threshold and load the stats file from the
# last round of training
echo "RO $1 stats" >$MODEL_FOLDER/tree.hed

# Add the phoenetic questions used in the decision tree
echo "TR 0" >>$MODEL_FOLDER/tree.hed
cat $HTK_COMMON/tree_ques.hed >>$MODEL_FOLDER/tree.hed

# Now the commands that cluster each output state
echo "TR 12" >>$MODEL_FOLDER/tree.hed
perl $SCRIPTS_PATH/MakeClusteredTri.pl TB $2 $MODEL_FOLDER/monophones1 >> $MODEL_FOLDER/tree.hed

echo "TR 1" >>$MODEL_FOLDER/tree.hed
echo "AU \"fulllist\"" >>$MODEL_FOLDER/tree.hed

echo "CO \"tiedlist\"" >>$MODEL_FOLDER/tree.hed
echo "ST \"trees\"" >>$MODEL_FOLDER/tree.hed

# Do the clustering
HHEd -A -T 1 -H $MODEL_FOLDER/hmm12/macros -H $MODEL_FOLDER/hmm12/hmmdefs -M $MODEL_FOLDER/hmm13 $MODEL_FOLDER/tree.hed $MODEL_FOLDER/triphones1 >$MODEL_FOLDER/hhed_cluster.log

