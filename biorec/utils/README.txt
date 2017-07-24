Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
Netutils

The current folder contains the utilities for phone recognition, 
unsupervised word classificitation and other stuff needed by phonrec and 
zerorchallenge.
(e.g., training, forward pass, etc...)
the follwoing is a list of the main functions:

- BinEncData.m: binarizes hiine unit values

- computeBestPath_new.m: computes the most probable phone sequence 

- computePhoneRecognitionError: computes PER and provives details information
  concerning the recognition errors (e.g, No. of insertions, substitutions, etc..)

- computeReconstructionError.m: computes the reconstruction error in the 
  acoustic-to-articulatory mapping (aka articulatory inversion)

- createContext.m: create input vector by adding left and rigth context to
  the centrale frame vector

- fnormData.m: normalize data. 2 options: 1. values in 0-1 range; 
  2. 0-mean unit variance

- fwsback.m: forward-backward algorithm adapted from Kevin P. Murphy's function
 
- loadAndNormData.m: load and normalizes input data from (presented in our format)

- minimize_pct.m: Carl Edward Rasmussen's conjugate gradient method

- viterbi_path_wlm: viterbi algorithm used to compute the most probable 
  phone state sequence

- windowData.m: function called by createContext

- mk_stochastic.m: function taken from Kevin P. Murphy's FullBNT toolbox



