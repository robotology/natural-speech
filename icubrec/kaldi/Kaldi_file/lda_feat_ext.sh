#!/bin/bash

script_folder=$(dirname "${BASH_SOURCE[0]}")

. $script_folder/path.sh
. $script_folder/cmd.sh

s_wav_scp=$1
#global_cmvn=$script_folder/fbank/global_train_cmvn.mat
splice_opts=$script_folder/conf/splice_opts
splice_lda_opts=$script_folder/conf/splice_lda_opts
#fbank_opts=$script_folder/conf/fbank.conf
mfcc_opts=$script_folder/conf/mfcc.conf
#tmp_delta=$script_folder/tmp_delta.ark
tmp_mfcc_ark=$script_folder/tmp/mfcc.ark
tmp_mfcc_scp=$script_folder/tmp/mfcc.scp
tmp_cmvn_scp=$script_folder/tmp/cmvn.scp
tmp_cmvn_ark=$script_folder/tmp/cmvn.ark
lda_mat=$script_folder/lda.mat

#compute-fbank-feats --config=$fbank_opts --verbose=2 scp:$s_wav_scp ark:- | apply-cmvn $global_cmvn ark:- ark:- | \
#           add-deltas ark:- ark:- splice-feats --config=$splice_opts ark:- ark,t:- | grep -v '\[' | sed -e s/]//g > $2


# questo e' il commento che ho provato io dalla mia home ~/kaldi/egs/chime4_vocub/s5_1ch e tutto funziona
#compute-fbank-feats --config=$fbank_opts --verbose=2 scp:$s_wav_scp ark:- | apply-cmvn $global_cmvn ark:- ark:- | add-deltas ark:- ark:- | splice-feats --config=$splice_opts ark:- ark,t:- | grep -v '\[' | sed -e s/]//g > $2

mkdir -p $script_folder/tmp
compute-mfcc-feats --config=$mfcc_opts --verbose=2 scp:$s_wav_scp ark,scp:$tmp_mfcc_ark,$tmp_mfcc_scp
compute-cmvn-stats scp:$tmp_mfcc_scp ark,scp:$tmp_cmvn_ark,$tmp_cmvn_scp
apply-cmvn scp:$tmp_cmvn_scp scp:$tmp_mfcc_scp ark:- | splice-feats --config=$splice_lda_opts ark:- ark:- | transform-feats $lda_mat ark:- ark:- | splice-feats --config=$splice_opts ark:- ark,t:- | grep -v '\[' | sed -e s/]//g > $2