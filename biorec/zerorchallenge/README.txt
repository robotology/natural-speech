Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
zerorchallenge

The current folder contains the scripts/functions for training and testing
the autoencoders and the hmm-encoder proposed for the 
Zero Resource Speech Challenge (ZRSC).
For details please refer to:
Badino, L., Mereta, A. Rosasco, L. "Discovering discrete subword units with 
Binarized Autoencoders and Hidden-Markov-Model Encoders", 
Proc. of Interspeech, Dresden, Germany, 2015.

The following is the list of scripts/functions in the folder:

- zeromain.m: this is used to generate the ZRSC files with binarized autoencoders

- mainaehmm.m: this is used to generate the ZRSC files with hmm-encoders

- hmmauto_learn.m: learns the hmm-encoder parameters using Expectation Maximization
  It's adapted from the mhmm_em function from K. Murphy's BayesNet toolbox

- ae_prob and ess_aehmm.m are used for the expectation step of the Expectation-Maximization (EM) algorithm. They are 
  called by hmmauto_learn

- getexplogs.m: used for the maximization  step of EM. It is called by 
  hmmauto_learn

- hmmauto_decode.m: decodes speech using the hmm-encoder

3 different datasets are available for training at: https://zenodo.org/record/836692

Testing tools are provided by the ZRSC Task1 evaluation tools 

 
