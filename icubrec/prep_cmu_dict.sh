#!/bin/bash
# Given the CMU 0.6 pronounciation dictionary, convert it
# into the form we'll be using with the HTK.
#
# Also adds in extra words we discovered we needed for
# coverage in the WSJ1 training data
#

USAGE="Prepare CMU dictionary\n"
USAGE=$USAGE"Usage: $0 [-e envt_file] dict_folder";

# ":" for options that require a string argument
# "#" for options that require a int argument
while getopts "e:h" opt; do
    case $opt in
    e)
        ENVT_FILE=$OPTARG;;
    h)
        echo -e $USAGE >&2;
        exit 0;;
    \?)
        echo -e $USAGE >&2;
        exit 1;;
    esac
done

# shifting the options index to the next parameter we didn't take care of
shift $((OPTIND - 1));

DICT_FOLDER=${1%%/}; export DICT_FOLDER

# Check mandatory arguments
if test -z $DICT_FOLDER; then
    echo -e $USAGE >&2;
    exit 1;
fi

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

echo "Environment variables:"
echo "ENVT_FILE       = $ENVT_FILE"
echo "DICT_FOLDER     = $DICT_FOLDER"
echo ""

perl $HTK_SCRIPTS/FixCMUDict.pl $DICT_FOLDER/c0.6 >$DICT_FOLDER/cmu6

#perl MergeDict.pl $DICT_FOLDER/cmu6temp $HTK_COMMON/wsj1_extra_dict >$DICT_FOLDER/cmu6

# Create a dictionary with a sp short pause after each word, this is
# so when we do the phone alignment from the word level MLF, we get
# the sp phone inbetween the words.  This version duplicates each
# entry and uses a long pause sil after each word as well.  By doing
# this we get about a 0.5% abs increase on Nov92 test set.
perl $HTK_SCRIPTS/AddSp.pl $DICT_FOLDER/cmu6 1 >$DICT_FOLDER/cmu6sp

# We need a dictionary that has the word "silence" with the mapping to the sil phone
cat $DICT_FOLDER/cmu6sp >$DICT_FOLDER/cmu6temp
echo "silence sil" >>$DICT_FOLDER/cmu6temp
sort $DICT_FOLDER/cmu6temp >$DICT_FOLDER/cmu6spsil
rm -f $DICT_FOLDER/cmu6temp
