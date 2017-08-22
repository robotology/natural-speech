Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
Netutils

The current folder contains utilities for phone recognition, 
unsupervised word classificitation and other utilities needed by phonrec, pce_phonerec and 
zerorchallenge.

the following is a list of the main functions:

- BinEncData.m: binarizes hidden unit values

- computeBestPath_new.m: computes the most probable phone sequence (after calling viterbi_path_wlm)

- computePhoneRecognitionError: computes PER and provives detailed information
  concerning the recognition errors (e.g, No. of insertions, substitutions, etc..)

- computeReconstructionError.m: computes the reconstruction error in the 
  acoustic-to-articulatory mapping (aka articulatory inversion)

- createContext.m: creates the input vector by adding left and rigth context to
  the centrale frame vector

- fnormData.m: normalizes data. 

- fwsback.m: forward-backward algorithm adapted from Kevin P. Murphy's function
 
- loadAndNormData.m: load and normalizes input data from datasets with articulatory data (presented in our format)

- loadAndNormData.m: load and normalizes input data from TIMIT and other audio-only datasets (presented in our format)

- minimize_pct.m: Carl Edward Rasmussen's conjugate gradient method

- viterbi_path_wlm: viterbi algorithm used to compute the most probable 
  phone state sequence




