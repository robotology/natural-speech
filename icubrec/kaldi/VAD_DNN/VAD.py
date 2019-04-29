import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import VAD_DNN
import time


class Configuration(object):

    def __init__(self, WindowSize, feat_dim, threshold=0.5):
        self.win_size = WindowSize
        self.feat_dim = feat_dim
        self.threshold = threshold

        self.audio_feat_dimension = feat_dim + 2 * WindowSize * feat_dim

        self.num_classes = 2
        self.n_hidden = 2000
        self.num_layers = 4


class VADModule(object):

    def __init__(self, WindowSize=5, FeaturesDimension=41):

        # features normalization values
        self.mean_vect = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TrainingSetMean.npy'))
        self.stdev_vect = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/TrainingSetStDev.npy'))

        # TF graph initialization
        self.config = Configuration(WindowSize, FeaturesDimension)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.feat = tf.placeholder(dtype=tf.float32, shape=[1, self.config.audio_feat_dimension])
            with tf.variable_scope('model'):
                model = VAD_DNN.Model(self.feat, self.config)

            logits_prob = model.softmax

            # the probability of speech is given by the first dimension in the softmax
            # so we slice the output accordingly
            self.speech_prob = tf.slice(logits_prob, [0, 0], [-1, 1])

            init_op = tf.local_variables_initializer()
            saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True


        self.session = tf.Session(graph=self.graph, config=config)

        self.session.run(init_op)

        saver.restore(self.session,
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/datamean_nodeltas_model_epoch13.ckpt"))

    def GetOutput(self, features_list):

        # normalize features
        features_norm_list=[]
        for frame in features_list:
            norm_frame = np.subtract(frame, self.mean_vect) / self.stdev_vect
            features_norm_list.append(norm_frame)


        features_list=features_norm_list

        # concatenate features list
        features = np.concatenate(features_list)
        features = np.expand_dims(features, axis=0)

        # get posteriors
        posteriors = self.session.run(self.speech_prob,
                                      feed_dict={self.feat: features})

        #if posteriors >= self.config.threshold:
        #    return 1.0
        #else:
        #    return 0.0
	return posteriors

    def MakeRandomInput(self):
        rand_input = []

        for _ in range(2 * self.config.win_size + 1):
            randomfeat = np.random.rand(self.config.feat_dim)
            rand_input.append(randomfeat)

        return rand_input
