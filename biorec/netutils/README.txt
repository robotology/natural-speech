Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
Netutils

The current folder contains the functions for DNNs and Autoencoders 
(e.g., training, forward pass, etc...)

- aeBackprop.m: function that actually trains an autoencoder (AE) by updating 
  the AE parameters after each minibatch

- aeCheckNumericalGradient.m: checks the computation of the gradient (for AEs)

- aeCost.m: computes the gradient (for AEs)

- aeTrain.m: inizializes or pretraines the AE and then call aeBackprop.m to 
  train it (or pretrain it)

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

- nnTrain.m: inizializes or pretraines the DNN and then call nnBackprop.m to 
  train it (or pretrain it)

- rbmTrain.m: restricted Boltzmann machine training 

