# natural-speech
Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Alessio Mereta
         Claudia Canevari
         
         
                          IITspeech

The current folder contains the following sub-folders:

- phonerec: script(s) for articulatory phone recognition. This code was used 
  to run some of the experiments described in
  Badino, L., Canevari, C., Fadiga, L., Metta, G., "Integrating Articulatory 
  Data in Deep Neural Network-based Acoustic Modeling", 
  Computer Speech and Language, vol 36, pp. 173�195, 2016.
  
Please cite the above paper if you use phonerec.
  
- zerorchallenge: zero-resource automatic speech recognition 
  Contains code of our entry to the Zero Resource Speech Challenge at Interspeech 2015.
  
If you use this code please cite:
  Badino, L., Mereta, A. Rosasco, L. "Discovering discrete subword units 
  with Binarized Autoencoders and Hidden-Markov-Model Encoders", 
  Proc. of Interspeech, Dresden, Germany, 2015.  

- netutils: deep neural network functions (e.g., training, forward pass)

- utils: functions/scripts called by functions/scripts in phonrec and netutils 

A brief description of each function/script is provided within the README file of each sub-folder.

This toolbox uses the Parallel Computing Matlab toolbox. It can either run with GPUs or not.

Acknowledgment. This work was funded by the European Commision project Poeticon++ (grant agreement 288382). 
