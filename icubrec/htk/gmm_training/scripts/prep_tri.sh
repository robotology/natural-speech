# Convert our monophone models and MLFs into triphones.  If a parameter
# "cross" is passed to script, we'll build cross word triphones, otherwise
# they will be word internal.
#
# Parameters:
#  1 - "cross" for cross word triphones, anything else means word internal

cd $MODEL_FOLDER

rm -f -r $MODEL_FOLDER/hled_make_tri.log $MODEL_FOLDER/mktri.hed $MODEL_FOLDER/hhed_clone_mono.log $MODEL_FOLDER/hmm10
mkdir $MODEL_FOLDER/hmm10

# Keep a copy of the monophones around in this directory for convience.
cp $HTK_COMMON/timit/monophones0 .
cp $HTK_COMMON/timit/monophones1 .

# Check to see if we are doing cross word triphones or not
if [[ $1 != "cross" ]]
then
# This converts the monophone MLF into a word internal triphone MLF
HLEd -A -T 1 -n $MODEL_FOLDER/triphones1 -i $MODEL_FOLDER/wintri.mlf $HTK_COMMON/mktri.led $MODEL_FOLDER/aligned2.mlf >$MODEL_FOLDER/hled_make_tri.log
else
# This version makes it into a cross word triphone MLF, the short pause
# phone will not block context across words.
HLEd -A -T 1 -n $MODEL_FOLDER/triphones1 -i $MODEL_FOLDER/wintri.mlf $HTK_COMMON/mktri_cross.led $MODEL_FOLDER/aligned2.mlf >$MODEL_FOLDER/hled_make_tri.log
fi

# Prepare the script that will be used to clone the monophones into
# their cooresponding triphones.  The script will also tie the transition
# matrices of all triphones with the same central phone together.
perl $SCRIPTS_PATH/MakeClonedMono.pl $MODEL_FOLDER/monophones1 $MODEL_FOLDER/triphones1 >$MODEL_FOLDER/mktri.hed

# Go go gadget clone monophones and tie transition matricies
HHEd -A -T 1 -B -H $MODEL_FOLDER/hmm9/macros -H $MODEL_FOLDER/hmm9/hmmdefs -M $MODEL_FOLDER/hmm10 $MODEL_FOLDER/mktri.hed $MODEL_FOLDER/monophones1 >$MODEL_FOLDER/hhed_clone_mono.log
