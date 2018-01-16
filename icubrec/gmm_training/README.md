# GMM-based model training

The main script for GMM-based model training is `train.sh`.

For example, to train such a model for WSJ, you can run the command:

    ./train.sh -e wsj0_5k.env model_folder

The training procedures follows the gender independent SI-84 systems described
in [1](#ref1) [1](#references).

## Environment files

We offer default environment files to train models for WSJ, chime4 and VoCub
datasets, all stored in this folder and named after their respective dataset.

## Main training parameters

We descrive here a few options available for the training procedure.

| Parameter name | Possible values (default value in bold) | Description |
|-|-|-|
| CTX_EXP | **wi** \| cross| Defines the triphones extention behavior: across words (cross) or word internal only (wi)|
| MODEL_START | **flat** \| timit | Specifies wether the monophones should be flat started (flat) or initialized with timit dataset (timit)|

## Credits

The starting point of most of the scripts provided here is the [HTK Wall Street
Journal Training Recipe](http://www.keithv.com/software/htk/) written by Keith
Vertanen. His code is released under the new BSD licence, except for the file
`tree_ques.hed` which he didn't write (even though no mention of its origin is
made). This is compatible with the GPLv3 license we use here.

## References

[1] Woodland, P.C., J.J. Odell, V. Valtchev, and S.J. Young.  “Large Vocabulary
Continuous Speech Recognition Using HTK.” In IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), 2:125–28. Adelaide, South
Australia: IEEE, 1994.  doi:10.1109/ICASSP.1994.389562.
