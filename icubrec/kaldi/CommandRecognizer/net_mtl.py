#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from abc import abstractmethod


def add_summary(filewriter, value, tag, step):
    summary = tf.Summary()
    if type(value) == tf.Tensor:
        v = tf.get_default_session().run(value)
    else:
        v = value
    summary.value.add(tag=tag, simple_value=v)
    filewriter.add_summary(summary, step)


class BaseNet:

    @staticmethod
    def bias_variable(shape, name):
        #initial = tf.constant(0.1, shape=shape)
        initial = tf.random_uniform(shape=shape, maxval=0.1)
        # return tf.Variable(initial, name)
        return tf.get_variable(name, initializer=initial)

    @staticmethod
    def weight_variable_relu(shape, name):
        initial = 2 * (tf.random_uniform(shape=shape, maxval=1) - 0.5) * 0.01
        return tf.get_variable(name, initializer=initial)

    @staticmethod
    def weight_variable_out(shape, name):
        initial = 2 * (tf.random_uniform(shape=shape, maxval=1) - 0.5)/tf.sqrt(float(shape[0]))
        return tf.get_variable(name, initializer=initial)

    @staticmethod
    def weight_variable_xavier(shape, name):
        initer = tf.contrib.layers.xavier_initializer()
        initial = initer(shape)
        return tf.get_variable(name, initializer=initial)

    @staticmethod
    def init_params(dims, variable_scope_name, initype=None):
        nb_layers = len(dims) - 1
        weights = [None] * nb_layers
        biases = [None] * nb_layers
        with tf.variable_scope(variable_scope_name):
            for i in range(nb_layers):
                name_w = "W" + str(i+1)
                name_b = "b" + str(i+1)
                print("Len", nb_layers)
                print("Dims", dims[i], dims[i+1])
                biases[i] = BaseNet.bias_variable([dims[i + 1]], name_b)
                if initype is "no_xavier" and i == nb_layers-1:
                    weights[i] = BaseNet.weight_variable_out([dims[i], dims[i + 1]], name_w)
                elif initype is "no_xavier":
                    weights[i] = BaseNet.weight_variable_relu([dims[i], dims[i + 1]], name_w)
                else:
                    weights[i] = BaseNet.weight_variable_xavier([dims[i], dims[i+1]], name_w)
                print(weights[i].get_shape())
        return weights, biases

    def __init__(self, netDims, config):
        self.dims = netDims
        self.W, self.B = [], []
        self.x, self.t, self.y = None, None, None
        self.logits, self.loss = None, None
        self.train_step = None
        self.learning_rate, self.global_step = None, None
        self.step = 0

        self.starter_learning_rate, self.momentum = 0.1, 0.9
        self.decay_rate = 0.95
        self.l2 = 0.0001
        self.batch_size_tr, self.batch_size_tst = 100, 100000
        self.ops_assign = []
        self.nepochs = 30
        self.early_stop = 3
        self.targettype = 'sparse'
        self.inittype = 'xavier'

        if config is not None:
            if hasattr(config, 'starter_learning_rate'):
                self.starter_learning_rate = config.starter_learning_rate
            if hasattr(config, 'momentum'):
                self.momentum = config.momentum
            if hasattr(config, 'decay_rate'):
                self.decay_rate = config.decay_rate
            if hasattr(config, 'l2'):
                self.l2 = config.l2
            if hasattr(config, 'batch_size_tr'):
                self.batch_size_tr = config.batch_size_tr
            if hasattr(config, 'batch_size_tst'):
                self.batch_size_tst = config.batch_size_tst
            if hasattr(config, 'nepochs'):
                self.nepochs = config.nepochs
            if hasattr(config, 'early_stop'):
                self.early_stop = config.early_stop
            if hasattr(config, 'targettype'):
                self.targettype = config.targettype
            if hasattr(config, 'inittype'):
                self.inittype = config.inittype

    def save_hyperparams(self, filewriter):
        add_summary(filewriter, self.batch_size_tr, "Batch size", 0)
        add_summary(filewriter, self.nepochs, "# training epochs", 0)
        add_summary(filewriter, self.early_stop, "Early stopping", 0)
        add_summary(filewriter, self.momentum, "Momentum", 0)

    def build_graph(self, x=None, t=None, t2=None, name='BaseNet'):
        self.nhidden_layers = len(self.dims) - 2
        self.W, self.B = BaseNet.init_params(self.dims, initype=self.inittype, variable_scope_name=name)
        self.define_placeholders(x, t, t2)

    def define_placeholders(self, x=None, t=None, t2=None):
        self.x = tf.placeholder(tf.float32, [None, self.dims[0]], name='x-input')

    def adam_optimizer(self):
        optimizer = tf.train.AdamOptimizer()
        return optimizer

    def momentum_optimizer(self):
        optimizer = tf.train.MomentumOptimizer(self.starter_learning_rate,
                                               self.momentum)
        return optimizer

    def momentum_optimizer_with_decay(self):
        if self.global_step is None:
            self.global_step = tf.placeholder(tf.int32, [], name="global_step")
        if self.learning_rate is None:
            self.learning_rate = tf.train.exponential_decay(
                self.starter_learning_rate, self.global_step, 1,
                self.decay_rate)
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.momentum)
        return optimizer

    def delegate(self):
        self.define_loss()
    @abstractmethod
    def define_loss(self):
        pass

    def train_batches(self, xs, ts, fd_opt=None, summary=True):
        sess = tf.get_default_session()
        nb_samples = xs.shape[0]
        permS = np.random.permutation(nb_samples)
        for iter in range(0, nb_samples, self.batch_size_tr):
            end = min(iter + self.batch_size_tr, nb_samples)
            batch_xs = xs[permS[iter:end], :]
            batch_ts = ts[permS[iter:end]]
            fd = {self.x: batch_xs, self.t: batch_ts,
                  self.global_step: self.step}
            if fd_opt is not None:
                fd.update(fd_opt)
            else:
                sess.run(self.train_step, feed_dict=fd)
                merged = None
        self.step += 1


class RECNet(BaseNet):
    def __init__(self, netDims, config, t2size=2):
        BaseNet.__init__(self, netDims, config)
        self.t2, self.y2, self.sels = None, None, None
        self.lmtl, self.tf_lmtl = 0, None
        self.t2size = t2size
        self.W2 = []
        # 0.075, 0.5, 0.75
        #self.starter_learning_rate, self.momentum = 0.075, 0.5
        #self.decay_rate = 0.75
        #self.l2 = 0.001

    def define_placeholders(self, x=None, t=None, t2=None):
        BaseNet.define_placeholders(self)
        self.t = tf.placeholder(tf.int32, [None], name='primary_targets')
        self.t2 = tf.placeholder(tf.float32, [None, self.t2size], name='secondary_targets')
        self.lmtl = tf.placeholder(tf.float32, [], name='mtl_lambda')
        self.sels = tf.placeholder(tf.float32, [None], name='selected_t2s')

    def define_loss(self):
        self.loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.t, logits=self.logits))
        #self.loss2 = self.lmtl * tf.reduce_mean(tf.multiply(tf.reduce_mean(tf.squared_difference(self.y2, self.t2),1),self.sels))
        self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.t2, self.y2), 1))
        #self.loss2 = tf.reduce_mean(-tf.losses.mean_squared_error(self.t2, self.y2))

        #self.loss2 = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(self.sigy2, self.sigt2), 1))
        #self.loss2 = tf.reduce_mean(tf.squared_difference(self.y2, self.t2))

        self.l2loss1 = self.l2 * tf.add_n([tf.nn.l2_loss(w) for w in self.W])
        self.l2loss2 = self.l2 * tf.nn.l2_loss(self.W2)


        #self.loss = tf.cond(tf.equal(self.lmtl, 0), lambda: self.loss1, lambda:  self.loss1 + self.lmtl * self.loss2)
        self.loss = tf.cond(tf.equal(self.lmtl, 0), lambda: self.loss1 + self.l2loss1,
                            lambda: self.l2loss1 + self.l2loss2 + self.loss1 + self.lmtl * self.loss2)



    def save_hyperparams(self, filewriter):
        BaseNet.save_hyperparams(self, filewriter)
        add_summary(filewriter, self.lmtl, "Lambda", 0)

    def build_graph(self, x=None, t=None, t2=None, name='myRecNet'):
        BaseNet.build_graph(self, x, t, t2, name)
        with tf.variable_scope(name):
            if self.inittype == 'no_xavier':
                self.W2 = BaseNet.weight_variable_relu([self.dims[-2], self.t2size], "Secondary_Ws")
            else:
                self.W2 = BaseNet.weight_variable_xavier([self.dims[-2], self.t2size], "Secondary_Ws")
            self.B2 = BaseNet.bias_variable([self.t2size], "Secondary_Bs")
        inp = self.x
        for i in range(self.nhidden_layers):
            z = tf.matmul(inp, self.W[i]) + self.B[i]
            inp = tf.nn.relu(z)

        #self.y2 = tf.nn.relu(tf.matmul(inp, self.W2) + self.B2)
        self.y2 = tf.matmul(inp, self.W2) + self.B2
        #self.sigy2 = tf.nn.sigmoid(self.y2)
        #self.sigt2 = tf.nn.sigmoid(self.t2)
        #cy2 = tf.concat([self.y2, inp], 1, name="concat")
        self.logits = tf.add(tf.matmul(inp, self.W[-1]), self.B[-1], name="logits")
        self.y = tf.nn.softmax(self.logits, name="output_ACUNet")
        self.logy = tf.log(self.y)

        self.define_loss()

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "myRecNet")

        optimizer = BaseNet.momentum_optimizer_with_decay(self)
        self.train_step = optimizer.minimize(self.loss, var_list=train_vars)

        cpred = tf.equal(tf.argmax(self.y, 1), tf.cast(self.t, tf.int64))
        self.fer = 1 - tf.reduce_mean(tf.cast(cpred, tf.float32))

    def compute_fer(self, xs, ts):
        sess = tf.get_default_session()
        nb_samples = xs.shape[0]
        fer = 0
        for iter in range(0, nb_samples, self.batch_size_tst):
            end = min(iter + self.batch_size_tst, nb_samples)
            size = end - iter
            batch_xs = xs[iter:end, :]
            batch_ts = ts[iter:end]
            fd = {self.x: batch_xs, self.t: batch_ts}
            f = sess.run(self.fer, feed_dict=fd)
            fer += f * size
        return fer / nb_samples

    def compute_loss(self, xs, ts, ts2, lmtl):
        sess = tf.get_default_session()
        nb_samples = xs.shape[0]
        L, L1, L2 = 0, 0, 0
        for iter in range(0, nb_samples, self.batch_size_tst):
            end = min(iter + self.batch_size_tst, nb_samples)
            size = end - iter
            batch_xs = xs[iter:end, :]
            batch_ts = ts[iter:end]
            batch_ts2 = ts2[iter:end, :]
            fd = {self.x: batch_xs, self.t: batch_ts, self.t2: batch_ts2, self.lmtl: lmtl}
            L, L1, L2 = sess.run([self.loss, self.loss1, self.loss2], feed_dict=fd)
            L += L * size
            L1 += L1 * size
            L2 += L2 * size
        return L / nb_samples, L1 / nb_samples, L2 / nb_samples

    def train_batches_mtl(self, xs, ts, ts2, lmtl, sels, fd_opt=None, summary=True):
        sess = tf.get_default_session()
        nb_samples = xs.shape[0]
        permS = np.random.permutation(nb_samples)
        for iter in range(0, nb_samples, self.batch_size_tr):
            end = min(iter + self.batch_size_tr, nb_samples)
            batch_xs = xs[permS[iter:end], :]
            batch_ts = ts[permS[iter:end]]
            batch_ts2 = ts2[permS[iter:end], :]
            batch_sels = sels[permS[iter:end]]
            fd = {self.x: batch_xs, self.t: batch_ts,
                  self.t2: batch_ts2, self.lmtl: lmtl,
                  self.sels: batch_sels,
                  self.global_step: self.step}
            if (fd_opt is not None):
                fd.update(fd_opt)
            else:
                sess.run(self.train_step, feed_dict=fd)
                merged = None
        self.step += 1

class RECNet_Q(RECNet):
    def __init__(self, netDims, config, t2size):
        RECNet.__init__(self, netDims, config, t2size)
        self.input_dim = config.input_dim
        self.target_dim = config.target_dim
        self.num_files = config.num_files
        self.num_examples = config.num_examples

    def define_placeholders(self, x=None, t=None, t2=None):
        if(x is None or t is None):
            raise NameError('At least one between x, t and t2 is not defined')
        else:
            self.x = x
            self.t = t
            self.t2 = t2
            self.lmtl = tf.placeholder(tf.float32, [], name='mtl_lambda')
            self.sels = tf.placeholder(tf.float32, [None], name='selected_t2s')
            self.iter = tf.placeholder(tf.int32, [], name='iter_prova')

    def build_graph(self, x=None, t=None, t2=None, name='myRecNet'):
        RECNet.build_graph(self, x, t, t2, name=name)

