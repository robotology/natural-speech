#!/bin/bash

script_folder=$(dirname "${BASH_SOURCE[0]}")

#echo $script_folder

. $script_folder/path.sh
. $script_folder/cmd.sh 

# Kaldi files/directories created during training
mfcc_config=$script_folder/mfcc.conf
cmvn_scp=$script_folder/data/vocub_tr/data/cmvn_vocub_tr.scp
srcdir=$script_folder/exp/tri3b_vocub_tr #directory af the model
graphdir=$script_folder/exp/tri3b_vocub_tr/graph_tgpr_vocub #directory of the graph

# once a wav file is created the scp file must be created
# Format is:
# utterance_id path_to_wav_file
# in our case we have to set a fake utterance id

mkdir -p $script_folder/exp/single_eg
s_wav_scp=$1
#exp/single_eg/single_wav.scp

#files that will be created
s_mfcc=$script_folder/exp/single_eg/single_mfcc
s_nmfcc=$script_folder/exp/single_eg/single_mfcc_norm
global_cmvn=$script_folder/exp/single_eg/global_cmvn.mat
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
decode_dir=$script_folder/exp/single_eg/decoded
lattice_out=$decode_dir/latt1
test_cmvn=$script_folder/data/vocub_et/cmvn.scp

# options for decoding
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
decode_extra_opts=

compute-mfcc-feats --config=$script_folder/conf/mfcc.conf --verbose=2 scp:$s_wav_scp ark:$s_mfcc.ark

#compute-cmvn-stats ark:$s_mfcc.ark ark:$s_nmfcc.ark

#feats="ark,s,cs:apply-cmvn --utt2spk=ark:data/vocub_et/utt2spk scp:$test_cmvn  ark:$s_mfcc.ark ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

feats="ark,s,cs:apply-cmvn $global_cmvn ark:$s_mfcc.ark ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

#feats="ark,s,cs:apply-cmvn scp:$test_cmvn ark:$s_mfcc.ark ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"

model=$srcdir/final.alimdl
mkdir -p $decode_dir
gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model $graphdir/HCLG.fst "$feats" ark:$lattice_out
    

lattice-best-path --word-symbol-table=$graphdir/words.txt ark:$lattice_out ark,t:-> $2 2>&1 
