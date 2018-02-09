#!/bin/bash

DESCRIPTION="Creates a word level MLF for all the files."
USAGE="Usage: $(basename $0) [-h] [-s] [-e envt_file] [-t transcr_list] model_folder
           feat_list scp_filename mlf_filename set

Positional arguments:
    model_folder    folder where the model is stored
    feat_list       list of feature files to use for training
    scp_filename    name for the produced SCP file
    mlf_filename    name for the produced word level MLF file
    set             'train' for the training set, 'test' for validation and
                    testing sets

Optional arguments:
    -e              environment file
    -h              help
    -s              silent mode
    -t              list of transcriptions"

SILENT_MODE=0

# ":" for options that require a string argument
# "#" for options that require a int argument
while getopts "e:hst:" opt; do
    case $opt in
    e)
        ENVT_FILE=$OPTARG;;
    h)
        echo -e "$DESCRIPTION\n";
        echo -e "$USAGE";
        exit 0;;
    s)
        SILENT_MODE=1;;
    t)
        TRANSCR_LIST=$OPTARG;;
    \?)
        echo -e "$USAGE" >&2;
        exit 1;;
    esac
done

# shifting the options index to the next parameter we didn't take care of
shift $((OPTIND - 1));

export MODEL_FOLDER=${1%%/};
export FEAT_LIST=${2%%/};
export SCP_FILE=${3%%/};
export MLF_FILE=${4%%/};
export SET=${5%%/};

# Check mandatory arguments
if test -z $MODEL_FOLDER; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $FEAT_LIST; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $SCP_FILE; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $MLF_FILE; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $SET; then
    echo -e "$USAGE" >&2;
    exit 1;
fi

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

if [ $SILENT_MODE -eq 0 ]; then
    echo "Environment variables:"
    echo "ENVT_FILE       = $ENVT_FILE"
    echo "MODEL_FOLDER    = $MODEL_FOLDER"
    echo "FEAT_LIST       = $FEAT_LIST"
    echo "TRANSCR_LIST    = $TRANSCR_LIST"
    echo "SCP_FILE        = $SCP_FILE"
    echo "MLF_FILE        = $MLF_FILE"
    echo "SET             = $SET"
    echo ""
fi

cd $MODEL_FOLDER

# Cleanup old files
rm -f prune.log missing.log missing.txt dot_files.txt $SCP_FILE $MLF_FILE

# Create a file listing all the FEAT files
sed "s:^*:$CORPORA_OTHER/feat/$FEATURE_FOLDER/:g" $FEAT_LIST >feat_files.txt

# Create a file that contains the filename of all the transcription files
if test -z $TRANSCR_LIST; then
    find -L $TRANS_FOLDER -iname '*.dot' >dot_files.txt
else
    cat $TRANSCR_LIST >dot_files.txt
fi

# Now create the MLF file using a script, we prune out anything that
# has words that aren't in our dictionary, producing a MLF with only
# these files and a corresponding script file.
perl $HTK_SCRIPTS/$MLF_CREATION_SCRIPT $SET feat_files.txt dot_files.txt $DICT_FILE $MLF_FILE $SCP_FILE 1 "" missing.txt 1 >missing.log
