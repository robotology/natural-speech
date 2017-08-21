#!/bin/bash
# Given the CMU 0.6 pronounciation dictionary, convert it
# into the form we'll be using with the HTK.
#
# Also adds in extra words we discovered we needed for
# coverage in the WSJ1 training data
#

DESCRIPTION="Prepare CMU dictionary"
USAGE="Usage: $(basename $0) [-h] [-e envt_file]

Optional arguments:
    -e              environment file
    -h              help"

# ":" for options that require a string argument
# "#" for options that require a int argument
while getopts "e:h" opt; do
    case $opt in
    e)
        ENVT_FILE=$OPTARG;;
    h)
        echo -e "$DESCRIPTION\n";
        echo -e "$USAGE";
        exit 0;;
    \?)
        echo -e "$USAGE" >&2;
        exit 1;;
    esac
done

# shifting the options index to the next parameter we didn't take care of
shift $((OPTIND - 1));

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

echo "Environment variables:"
echo "ENVT_FILE       = $ENVT_FILE"
echo ""

perl $HTK_SCRIPTS/FixCMUDict.pl $HTK_DATA/cmu/c0.6 >$HTK_DATA/cmu/cmu6

#perl MergeDict.pl $HTK_DATA/cmu/cmu6temp $HTK_COMMON/wsj1_extra_dict >$HTK_DATA/cmu/cmu6

# Create a dictionary with a sp short pause after each word, this is
# so when we do the phone alignment from the word level MLF, we get
# the sp phone inbetween the words.  This version duplicates each
# entry and uses a long pause sil after each word as well.  By doing
# this we get about a 0.5% abs increase on Nov92 test set.
perl $HTK_SCRIPTS/AddSp.pl $HTK_DATA/cmu/cmu6 1 >$HTK_DATA/cmu/cmu6sp

# We need a dictionary that has the word "silence" with the mapping to the sil phone
cat $HTK_DATA/cmu/cmu6sp >$HTK_DATA/cmu/cmu6temp
echo "silence sil" >>$HTK_DATA/cmu/cmu6temp
sort $HTK_DATA/cmu/cmu6temp >$HTK_DATA/cmu/cmu6spsil
rm -f $HTK_DATA/cmu/cmu6temp
