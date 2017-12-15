#!/bin/bash

DESCRIPTION="Integrate DNN and HMM definitions

The output folder should already contain a file called htkdef which contains
the definition of the network (extracted from TensorFlow).
The GMM model under \$GMM_MODEL_DIR (normally defined in the environment file)
is used for the HMM definition."
USAGE="Usage: $(basename $0) [-h] [-e envt_file] output_folder

Positional arguments:
    output_folder   folder where the output should be stored

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

OUTPUT_FOLDER=${1%%/}; export OUTPUT_FOLDER

# Check mandatory arguments
if test -z $OUTPUT_FOLDER; then
    echo -e "$USAGE" >&2;
    exit 1;
fi

# Source envt file
if ! test -z $ENVT_FILE; then
    source $ENVT_FILE
fi

echo "Environment variables:"
echo "ENVT_FILE         = $ENVT_FILE"
echo "OUTPUT_FOLDER     = $OUTPUT_FOLDER"
echo ""

cd $OUTPUT_FOLDER

### Writing config files ####################
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/cvn.cfg)\"" >cvn.cfg
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/connect.tf.hed)\"" >connect.hed
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/basic.htkc)\"" >basic.htkc
# ident_cvn
echo "<VARSCALE> $FEATURE_SIZE" >ident_cvn
eval printf '"1.0 "%.0s' {1..$FEATURE_SIZE} >>ident_cvn
# Transcriptions
merge_mlf.sh $TRANSCR_FILES | sed "s:*:$CORPORA_OTHER/feat/$FEATURE_FOLDER:g" >alignment.mlf

### Preparing prototype ####################
if [ ! -d "proto" ]; then
    mkdir -p proto
fi
cp htkdef proto/dnn
if [ ! -d "dnn6.trained/init" ]; then
    mkdir -p dnn6.trained/init
fi
cp -f $GMM_MODEL_DIR/$GMM_HMMLIST .
touch proto/foolist
HHEd -H $GMM_MODEL_DIR/$GMM_HMMDEFS_DIR/$GMM_HMMDEFS -M dnn6.trained/init connect.hed $GMM_HMMLIST
if [ "$MODEL_UNIT" = "TRI" ]; then
    $REPO/icubrec/dnn_training/tf/connect_HMM.py --senones dnn6.trained/init/$GMM_HMMDEFS $GMM_MODEL_DIR/senlist
else
    $REPO/icubrec/dnn_training/tf/connect_HMM.py dnn6.trained/init/$GMM_HMMDEFS $GMM_HMMLIST
fi
if [ ! -d "cvn" ]; then
    mkdir -p cvn
fi
sed "s:^*:$CORPORA_OTHER/feat/$FEATURE_FOLDER:g" <$DNN_TR_LIST >./train.scp
sed "s:^*:$CORPORA_OTHER/feat/$FEATURE_FOLDER:g" <$DNN_DT_LIST >./hv.scp
HCompV -p *%%%% -k *.%%%% -C cvn.cfg -q mv -c cvn -S train.scp

cp dnn6.trained/init/$GMM_HMMDEFS dnn6.trained
# HNTrainSGD -C basic.htkc -C $HTK_COMMON/train_tf.htkc -H dnn6.trained/init/$GMM_HMMDEFS -M dnn6.trained -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST
