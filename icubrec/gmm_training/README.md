# GMM-based model training

The main script for GMM-based model training is `train.sh`.

For example, to train such a model for wsj0, you can run the command:

    ./train.sh -e wsj0.env model_folder

## Environment files

We offer default environment files to train models for WSJ, CHiME4 and VoCub
datasets, all stored in this folder and named after their respective dataset.

## Main training parameters

We descrive here a few options available for the training procedure.

| Parameter name | Possible values (default value in bold) | Description |
|-|-|-|
| CTX_EXP | **wi** \| cross| Defines the triphones extention behavior: across words (cross) or word internal only (wi)|
| MODEL_START | **flat** \| timit | Specifies wether the monophones should be flat started (flat) or initialized with timit dataset (timit)|

TODO: ref Woodland
