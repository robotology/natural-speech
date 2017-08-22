Copywigth 2016 Istituto Italiano di Tecnologia 

Authors: Leonardo Badino
         Claudia Canevari
	 Alessio Mereta
         
         
	   phonerec

The current folder contains the scripts to train and test DNN-HMM articulatory phone recognition systems as well as standard (i.e., purely acoustic) DNN-HMM phone recognition systems.

For details please refer to:
Badino, L., Canevari, C., Fadiga, L., Metta, G., "Integrating Articulatory Data in Deep Neural Network-based Acoustic Modeling", Computer Speech and Language, vol 36, pp. 173–195, 2016.

The folder contains:

- demo:  contains 2 examples to build and evaluate a baseline (audio1_motor0_rec0) and an  articulatory phone recognition system (audio1_motor3_rec1) on the mngu0 dataset. The preprocessed mngu0 dataset is available at:  https://zenodo.org/record/836692

- plosclassify.m: train and tests articulatory phone recognition systems. It needs the inivar.m configuration file.

- inivar.m: configuration firl to define, e.g.: input and resource files, the type of articulatory features, the hyperparameters of the different deep neural networks.
 
