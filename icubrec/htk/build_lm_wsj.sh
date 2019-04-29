#!/bin/bash

DESCRIPTION="Builds the dictionary and the word network needed for recognition"
USAGE="Usage: $(basename $0) [-h] [-e envt_file]

Optional arguments:
    -e              environment file
    -h              help"
# Set default values
USAGE="Usage: $0 [-e envt_file]";

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

# Get rid of the MIT comment lines from the top
grep -v "#" $VOCAB_FILE >dict_temp

# We need sentence start and end symbols which match the WSJ
# standard language model and produce no output symbols.
echo "<s> [] sil" >$DICT_FILE
echo "</s> [] sil" >>$DICT_FILE

# Add pronunciations for each word
perl $HTK_SCRIPTS/WordsToDictionary.pl dict_temp $HTK_DATA/cmu/cmu6sp dict_temp2
cat dict_temp2 >>$DICT_FILE
rm -f dict_temp dict_temp2

# Decompress the WSJ standard language model and build the word network
gunzip -d -c $LNG_MODEL >lm_temp
HBuild -A -T 1 -C $HTK_COMMON/rawmit.htkc -n lm_temp -u '<UNK>' -s '<s>' '</s>' -z $DICT_FILE $GRAM_FILE >hbuild.log
rm -f lm_temp
