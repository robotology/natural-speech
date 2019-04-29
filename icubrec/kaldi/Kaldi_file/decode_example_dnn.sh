#!/bin/bash
# this script takes a text file of posteriors in the kaldi format and returns a transcription
script_folder=$(dirname "${BASH_SOURCE[0]}")


. $script_folder/path.sh
. $script_folder/cmd.sh


# Kaldi files/directories created during training
srcdir=$script_folder
#srcdir=$script_folder/exp/tri3b_vocub_tr #directory af the model
graphdir=$script_folder/graph
#graphdir=$script_folder/exp/tri3b_vocub_tr/graph_tgpr_vocub #directory of the graph

#input file
posts_txt=$1

#files that will be created

dir=$script_folder/exp/single_eg_dnn
decode_dir=$dir/decoded
lattice_out=$decode_dir/latt1

# options for decoding
min_active=200
max_active=7000
beam=13.0
lattice_beam=8.0
acwt=1.0 # note: only really affects pruning (scoring is on lattices).
decode_extra_opts=

mkdir -p $dir
mkdir -p $decode_dir 

copy-matrix ark:$posts_txt ark:$decode_dir/logposts.ark

model=$srcdir/final.mdl

latgen-faster-mapped-parallel --num-threads=3 --num-threads-total=5 --acoustic-scale=$acwt --min-active=$min_active --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam --allow-partial=true $model $graphdir/HCLG.fst ark:$decode_dir/logposts.ark  ark:$decode_dir/lat.1.ark >$3 2>&1
#latgen-faster-mapped --acoustic-scale=$acwt --min-active=$min_active --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam --allow-partial=true $model $graphdir/HCLG.fst ark:$decode_dir/logposts.ark  ark:$decode_dir/lat.1.ark >$3 2>&1


#questo e' quello che ritorna la trascrizione... tra le altre cose
lattice-best-path --word-symbol-table=$graphdir/words.txt ark:$decode_dir/lat.1.ark ark,t:-> $2 2>&1

#lattice-to-nbest --n=10 ark:$decode_dir/lat.1.ark ark:- | nbest-to-linear ark:- ark:/dev/null ark:/dev/null ark:/dev/null ark,t:- >$3 2>&1
