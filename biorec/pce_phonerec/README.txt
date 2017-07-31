Copywigth 2016 Istituto Italiano di Tecnologia 

Author: Leonardo Badino
                 
	   pce_phonerec

The current folder contains scripts and function to train and build the phonetic context embedding-based articulatory phone recognition system decribed in: 

Badino, L., "Phonetic Context Embeddings for DNN-HMM Phone Recognition", in Proc. of Interspeech, San Francisco, CA, USA, 2016. 

The main scripts in the folder are:

- mtkpr_pce.m: main script. It builds a phonetic context embedding and train and test a phone recognition system that uses such embedding. Training is based on multi-task learning (MTL) 

- mtkpr_baseline: trains and tests the alternative (baseline) MTL-based system described in: Seltzer, M. and Droppo, J. (2013). Multi-task learning in deep neural networks for improved phoneme recognition. In Proc. of ICASSP (Vancouver, Canada).

- extractDFeatures.m: extracts the linguistic articulatory/distinctive features of each phoneme by using a look-up table defined for each datatset.

Given the current scripts the phonetic context embedding-based phone recognition systems can be trained and test on TIMIT and mngu0. However only the extractDFeatures is dataset dependent.
 
