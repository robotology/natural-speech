# Environment files

To make the scripts more flexibles, we make a heavy use of environment variables. Those mainly fall into to groups:
* default locations that will vary from one computer to the other
* options defining the way to train the model

We use to define those environment variables in configuration files whose file extension is `.env`. Grouping the environment variables in a file makes in it easy to configure experiments once for all and switch between them easily. Those environment files can then be used in two different ways:
* all the main scripts offered here accept a `-e` parameter corresponding to an environment file. This file will be automatically loaded at the beginning of the script (leaving the shell environment clean after the execution of the script).
* alternatively, one can load all environment variables in the shell's environment before running the script, using the command `source {file}.env`. This allows the user to easily modify some parameters without modifying the default environment file.

We offer default environment files for WSJ, Chime4 and VoCub datasets. They all depend on the file `sys-dpdt.env` that contains the system dependent variables (files locations and computer configuration). The user will have to check adapt this file for his own configuration before running the scripts.

HOME=... -> to modify
CTX_EXP = {wi | cross}
MODEL_START = {flat | timit}

# Preparation

In order to train or test
# Feature extraction

# Training

The first step, before training the GMM, is to create the dictionary and the
word net.
To do so, run the script `build_lm_wsj.sh`.

train.sh

[-e env] or "source env" (-> possible to modify some parameters then)
feat and dot files

