#!/bin/bash

DESCRIPTION="Tests models"
USAGE="Usage: $(basename $0) [-h] [-e envt_file] [-t transcr_list] model_folder
           feat_list result_folder

Positional arguments:
    model_folder    folder where the model is stored
    feat_list       list of feature files to use for testing
    result_folder   folder where the results should be stored

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
export RESULT_FOLDER=${3%%/};

# Check mandatory arguments
if test -z $MODEL_FOLDER; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $FEAT_LIST; then
    echo -e "$USAGE" >&2;
    exit 1;
fi
if test -z $RESULT_FOLDER; then
    echo -e "$USAGE" >&2;
    exit 1;
fi

# Default values
if test -z $PRUNING_BEAM; then
    export PRUNING_BEAM=250.0;
fi
if test -z $CTXEXP; then
    export CTXEXP='wi'
fi
case $CTXEXP in
    wi)
        export CONFIG_CTXEXP=wi.htkc;;
    cross)
        export CONFIG_CTXEXP=cross.htkc;;
esac

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

echo "Environment variables:"
echo "ENVT_FILE       = $ENVT_FILE"
echo "MODEL_FOLDER    = $MODEL_FOLDER"
echo "FEAT_LIST       = $FEAT_LIST"
echo "TRANSCR_LIST    = $TRANSCR_LIST"
echo "RESULT_FOLDER   = $RESULT_FOLDER"
echo ""

# Intial setup of test MLFs
echo "Building test MLF..."
make_mlf.sh -s -t "$TRANSCR_LIST" $RESULT_FOLDER $FEAT_LIST dataset.scp words.mlf test

# You can probably now increase results slightly by running
# the best penalty and scale factor with a higher beam width,
# say 350.0.  Then relax and have a beer- you've earned it.
echo "Evaluating on test set..."
HMMDEFS_DIR=${MODEL_TYPE}_HMMDEFS_DIR
eval_no_lat.sh ${!HMMDEFS_DIR} _prune${PRUNING_BEAM}_$CTXEXP $PRUNING_BEAM -4.0 15.0

echo ""
echo "Finished."
