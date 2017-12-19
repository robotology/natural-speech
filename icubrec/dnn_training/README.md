# DNN-based acoustic model training

The easiest way to train a DNN-based acoustic model is to use HTK tools. The
drawback is that HTK is quite limited in his support of deep networks training.
That's why we propose an alternative method which is to train a network with
Tensorflow and convert it to HTK format. Let's see how both methods work.

## HTK training

The folder `htk` contains two different scripts to train a network:
* `train_tut.sh` that follows HTK tutorial. The network uses the sigmoid
  activation layer and is trained in two steps: a layer-by-layer pretraining
  phase followed by a finetuning phase.
* `train.sh` uses RELU units that don't require the pretraining phase. The
  network is trained in one step with all layers together.

## Tensorflow training

**Dependency**: the scripts are based on Tensorflow 1.0.

The file `tf/train.py` gives an example of how to train a net with Tensorflow.
Once the net is trained, it can be used with HTK in a few step:
* first, the net parameters should be exported in a format HTK can handle. To
  do that, it is enough to add following lines just after the net has finished
  training (or alternatively after it is loaded from a previous save):

        net = nndef.parse_from_tf_graph(tf.get_default_graph(), "output")
        htk_def = nndef.convert_for_htk(net, "FBANK_D_A_Z", 11)
        file = open('htkdef', 'w')
        file.write('\n'.join(htk_def))
        file.close()

  The paramaters are exported in a file called `htkdef`. We assume here that
  the output layer is named `output` in TF's graph, that mean-normalised fbanks
  are used with first and second derivatives, and that a context of 11 frames
  is used.
* once the parameters are exported, they need to be integrated with an existing
  HMM definition. We provide the script `integrate.sh` for that purpose, which
  can be used as following:

        integrate.sh -e <envt_file> <output_folder>

  where `<output_folder>` will be used to store the new model and should
  already contain the definition of the DNN (the `htkdef` file created above).
  The GMM model under `$GMM_MODEL_DIR` (normally defined in the environment
  file) is used for the HMM definition.

The model is then ready to be used within HTK.
