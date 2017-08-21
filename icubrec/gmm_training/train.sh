#!/bin/bash

DESCRIPTION="Trains GMM models."
USAGE="Usage: $(basename $0) [-h] [-e envt_file] [-t transcr_list] model_folder
           feat_list

Positional arguments:
    model_folder    folder where the model is stored
    feat_list       list of feature files to use for testing

Optional arguments:
    -e              environment file
    -h              help
    -t              list of transcriptions"

# ":" for options that require a string argument
# "#" for options that require a int argument
while getopts "e:ht:" opt; do
    case $opt in
    e)
        ENVT_FILE=$OPTARG;;
    h)
        echo -e "$DESCRIPTION\n";
        echo -e "$USAGE";
        exit 0;;
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

# Check mandatory arguments
if test -z $MODEL_FOLDER; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $FEAT_LIST; then
    echo -e "$USAGE" >&2;
    exit 1;
fi

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

if test -z $CTXEXP; then
    export CTXEXP='wi'
fi
case $CTXEXP in
    wi)
        export CONFIG_CTXEXP=configwi;;
    cross)
        export CONFIG_CTXEXP=configcross;;
esac

if test -z $MODEL_START; then
    export MODEL_START='flat'
fi

echo "Environment variables:"
echo "ENVT_FILE       = $ENVT_FILE"
echo "MODEL_FOLDER    = $MODEL_FOLDER"
echo "FEAT_LIST       = $FEAT_LIST"
echo "TRANSCR_LIST    = $TRANSCR_LIST"
echo "CTXEXP          = $CTXEXP"
echo "MODEL_START     = $MODEL_START"
echo ""

cd $MODEL_FOLDER

# Intial setup of training MLFs
echo "Building training MLF..."
make_mlf.sh -s -t "$TRANSCR_LIST" $MODEL_FOLDER $FEAT_LIST train.scp words.mlf train

if [[ $MODEL_START == "flat" ]]; then
    # Get the basic monophone models trained
    echo "Flat starting monophones..."
    $SCRIPTS_PATH/flat_start.sh
else
    mkdir -p $MODEL_FOLDER/timit
    # Adapting TIMIT phone transcription files
    echo "Preparing TIMIT..."
    $SCRIPTS_PATH/prep_timit.sh

    # Use the transcriptions to train up the monophone models
    mkdir -p timit
    echo "Training TIMIT monophones..."
    $SCRIPTS_PATH/train_mono_timit.sh

    # As a sanity check, we'll evaluate the monophone models
    # doing just phone recognition on the training data.
    echo "Evaluating TIMIT monophones..."
    $SCRIPTS_PATH/eval_mono.sh

    cd $MODEL_FOLDER
fi

# Create a new MLF that is aligned based on our monophone model
echo "Aligning with monophones..."
$SCRIPTS_PATH/align_mlf.sh $MODEL_START

# More training for the monophones, create triphones, train
# triphones, tie the triphones, train tied triphones, then
# mixup the number of Gaussians per state.
echo "Training monophones..."
$SCRIPTS_PATH/train_mono.sh $MODEL_START
echo "Prepping triphones..."
$SCRIPTS_PATH/prep_tri.sh $CTXEXP
echo "Training triphones..."
$SCRIPTS_PATH/train_tri.sh
# These values of RO and TB seem to work fairly well, but
# there may be more optimal values.
echo "Prepping state-tied triphones..."
$SCRIPTS_PATH/prep_tied.sh 200 750 $CTXEXP
echo "Training state-tied triphones..."
$SCRIPTS_PATH/train_tied.sh
echo "Mixing up..."
$SCRIPTS_PATH/train_mixup.sh

echo ""
echo "Finished."
