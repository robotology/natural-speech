#!/bin/bash
# Trains up GMM models.

# Set default values
USAGE="Usage: $0 [-e envt_file] model_folder [feat_list] [dot_list]";

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

export MODEL_FOLDER=${1%%/};
export FEAT_LIST=${2%%/};
export DOT_FILES=${3%%/};

# Check mandatory arguments
if test -z $MODEL_FOLDER
then
    echo -e $USAGE >&2;
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
echo "DOT_FILES       = $DOT_FILES"
echo "CTXEXP          = $CTXEXP"
echo "MODEL_START     = $MODEL_START"
echo ""

cd $MODEL_FOLDER

if ! test -z $FEAT_LIST; then
    sed "s:^:$CORPORA_OTHER/feat/$FEATURE_FOLDER/:g" $FEAT_LIST >train.scp
    export FEAT_FILES=train.scp
fi

# Intial setup of training and test MLFs
echo "Building training MLF..."
make_mlf.sh $MODEL_FOLDER train
mv $MODEL_FOLDER/dataset.scp train.scp

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

