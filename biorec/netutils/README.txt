Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
Netutils

The current folder contains functions for DNN and Autoencoder (AE) training, forward pass, etc...

-ae_hyperparameters.m: only used for description purposes to show and describe all the possible autoencoder hyper-parameters. Should not be modified

- dnn_hyperparameters: only used for description purposes to show and describe all the possible dnn hyper-parameters. Shoud not be modified

- aeTrain.m: trains an AE after it has been inizialized or pretrained

- aeCheckNumericalGradient.m: checks the computation of the gradient (for AEs)

- aeCost.m: computes the gradient (for AEs)

- computeNumericalGradient.m: called by aeCheckNumericalGradient.m and 
  nnCheckNumericalGradient.m

- dbnTrain.m: deep belief network training (for DNNcacc pretraining)
 Attention: at present dropoout only works with relu units

- nnBackprop.m: function that actually trains an dnn by updating 
  the DNN parameters after each minibatch

- dnnCheckNumericalGradient.m: checks the computation of the gradient (for DNNs)

- dnnCost.m: computes the gradient (for DNNs)

- nnFwd.m: forward pass (for either DNN or AE)

- nnRandInit.m: random inizialization of the DNN parameters

- nnTrain.m:  trains a DNN after it has been inizialized or pretrained

- nnTrain_mtk.m: trains a DNN after it has been inizialized or pretrained using multi-task learning. 
It requires a secondary target cost to be defined.

- rbmTrain.m: restricted Boltzmann machine training 

