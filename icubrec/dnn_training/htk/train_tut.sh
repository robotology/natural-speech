#!/bin/bash

DESCRIPTION="Train DNN following HTK tutorial"
USAGE="Usage: $(basename $0) [-h] [-e envt_file] output_folder

Positional arguments:
    output_folder   folder where the model is stored

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
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/HTE.dnn.tut.am)\"" >HTE.dnn.am
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/cvn.cfg)\"" >cvn.cfg
eval "echo \"$(sed "s:\":\\\\\":g" $HTK_COMMON/connect.tut.hed)\"" >connect.hed
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
$REPO/icubrec/dnn_training/htk/GenInitDNN.py HTE.dnn.am proto/dnn
if [ ! -d "dnn3/init" ]; then
  mkdir -p dnn3/init
fi
cp -f $GMM_MODEL_DIR/$GMM_HMMLIST .
touch proto/foolist
HHEd -H $GMM_MODEL_DIR/$GMM_HMMDEFS_DIR/$GMM_HMMDEFS -M dnn3/init connect.hed $GMM_HMMLIST
if [ ! -d "cvn" ]; then
  mkdir -p cvn
fi
sed "s:^:$CORPORA_OTHER/feat/$FEATURE_FOLDER/:g" <$DNN_TR_LIST >./train.scp
sed "s:^:$CORPORA_OTHER/feat/$FEATURE_FOLDER/:g" <$DNN_DT_LIST >./hv.scp
HCompV -p *%%%% -k *.%%%% -C cvn.cfg -q mv -c cvn -S train.scp

### Pre-training ####################
HNTrainSGD -C basic.htkc -C $HTK_COMMON/pretrain.htkc -H dnn3/init/$GMM_HMMDEFS -M dnn3 -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST

if [ ! -d "dnn4/init" ]; then
  mkdir -p dnn4/init
fi
HHEd -H dnn3/$GMM_HMMDEFS -M dnn4/init $HTK_COMMON/addlayer4.hed $GMM_HMMLIST
HNTrainSGD -C basic.htkc -C $HTK_COMMON/pretrain.htkc -H dnn4/init/$GMM_HMMDEFS -M dnn4 -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST

if [ ! -d "dnn5/init" ]; then
  mkdir -p dnn5/init
fi
HHEd -H dnn4/$GMM_HMMDEFS -M dnn5/init $HTK_COMMON/addlayer5.hed $GMM_HMMLIST
HNTrainSGD -C basic.htkc -C $HTK_COMMON/pretrain.htkc -H dnn5/init/$GMM_HMMDEFS -M dnn5 -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST

if [ ! -d "dnn6/init" ]; then
  mkdir -p dnn6/init
fi
HHEd -H dnn5/$GMM_HMMDEFS -M dnn6/init $HTK_COMMON/addlayer6.hed $GMM_HMMLIST
HNTrainSGD -C basic.htkc -C $HTK_COMMON/pretrain.htkc -H dnn6/init/$GMM_HMMDEFS -M dnn6 -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST

### Fine-tuning ####################
if [ ! -d "dnn7/init" ]; then
  mkdir -p dnn7/init
fi
HHEd -H dnn6/$GMM_HMMDEFS -M dnn7/init $HTK_COMMON/addlayer7.hed $GMM_HMMLIST
if [ ! -d "dnn7.trained.tut" ]; then
  mkdir -p dnn7.trained.tut
fi
date
HNTrainSGD -C basic.htkc -C $HTK_COMMON/finetune.htkc -H dnn7/init/$GMM_HMMDEFS -M dnn7.trained.tut -S train.scp -N hv.scp -l LABEL -I alignment.mlf $GMM_HMMLIST
date
