import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import CommandRecognizer.net_mtl as net

#class dnn_config(object):
#    input_dim = 1320
#    target_dim = 681
#    target2_dim = 300
#    netH = [2000, 2000, 2000, 2000]
class dnn_config(object):
    input_dim = 440
    target_dim = 1808
    netH = [2000, 2000, 2000, 2000]

class AMModule():

    def __init__(self, bPriors=False):

        net_parms = dnn_config()
        recipe_folder = '/home/storage/projects/kaldi/egs/vochime/s5_1ch/'
        exp_folder = '../'
        net_folder = recipe_folder + 'netmodel/'
        net_model = net_folder + 'model'
        lpriors_file = recipe_folder + 'log_priors.npy'

        net_dims = [net_parms.input_dim] + net_parms.netH + [net_parms.target_dim]

        if bPriors:
            self.lpriors = np.load(lpriors_file)

        self.bPiors = bPriors

        with tf.Graph().as_default() as self.graph:
            self.Cnet = net.RECNet(net_dims, net_parms, )

            with tf.variable_scope('myRecNet', reuse=None) as scope:
                print('Building  model graph:')
                self.Cnet.build_graph()


            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            saverCNet = tf.train.Saver(self.Cnet.W + self.Cnet.B)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)

        self.session.run(init_op)

        saverCNet.restore(self.session, net_model)

    #La funzione deve salavre le log post nel formato di kaldi e mandare indietro un bool
    def GetOutput(self, input, file_id, output_file_path):
        logposts = self.session.run(self.Cnet.logy, feed_dict={self.Cnet.x: input})
        if self.bPiors:

            write_kaldi_tmatrix (file_id=file_id, matrix=logposts - self.lpriors, fileout=output_file_path)
            return True
        else:
            write_kaldi_tmatrix(file_id=file_id, matrix=logposts, fileout=output_file_path)
            return True

def write_kaldi_tmatrix(file_id, matrix, fileout):
    n = matrix.shape[0]
    with open(fileout, 'w') as fo:
        fo.write(file_id + ' ' + '[' + '\n')
        l = 0
        for row in matrix:
            if len(row) > 200:
                line = ''
                for e in row:
                    line = line + "%.4f" % e + ' '
                if l < n - 1:
                    line = line.rstrip() + '\n'
                else:
                    line = line + ']' + '\n'
            else:
                line = np.array2string(row, formatter={'float_kind': lambda x: "%.5f" % x})
                line = line.replace('\n', '')
                line = line.replace('[', '')
                if l < n - 1:
                    line = line.replace(']', '') + '\n'
                else:
                    line = line.replace(']', ' ]') + '\n'
            fo.write(line)
            l += 1
        fo.close()

if __name__ == '__main__':
    model = AMModule()
