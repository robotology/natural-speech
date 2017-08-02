# Training

The training is performed by calling the script `train.sh`.

## Main training parameters

We descrive here a few options regulating the training procedure.

| Parameter name | Possible values | Description |
|-|-|-|
| CTX_EXP | wi \| cross| Defines the triphones extention behavior: across words (cross) or word internal only (wi)|
| MODEL_START | flat \| timit | Specifies wether the monophones should be flat started (flat) or initialized with timit dataset (timit)|
