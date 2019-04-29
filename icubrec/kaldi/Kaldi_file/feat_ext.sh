#!/bin/bash

script_folder=$(dirname "${BASH_SOURCE[0]}")

. $script_folder/path.sh
. $script_folder/cmd.sh

s_wav_scp=$1
global_cmvn=$script_folder/fbank/global_train_cmvn.mat
splice_opts=$script_folder/conf/splice_opts
fbank_opts=$script_folder/conf/fbank.conf
tmp_delta=$script_folder/tmp_delta.ark

#compute-fbank-feats --config=$fbank_opts --verbose=2 scp:$s_wav_scp ark:- | apply-cmvn $global_cmvn ark:- ark:- | \
 #           add-deltas ark:- ark:- splice-feats --config=$splice_opts ark:- ark,t:- | grep -v '\[' | sed -e s/]//g > $2


# questo e' il commento che ho provato io dalla mia home ~/kaldi/egs/chime4_vocub/s5_1ch e tutto funziona
compute-fbank-feats --config=$fbank_opts --verbose=2 scp:$s_wav_scp ark:- | apply-cmvn $global_cmvn ark:- ark:- | add-deltas ark:- ark:- | splice-feats --config=$splice_opts ark:- ark,t:- | grep -v '\[' | sed -e s/]//g > $2