% demo script for phoneme classification using acoustic-articultory data
clear all;
close all;
%%First run: baseline only audio, no motor
plosclassify('audio1_motor0_rec0');

%%Second run: raw motor reconstructed from raw audio
plosclassify('audio1_motor3_rec1');
