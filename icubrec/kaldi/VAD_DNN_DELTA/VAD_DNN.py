from __future__ import division
import tensorflow as tf
import sys


class Model(object):

  def __init__(self,features,config):

    batch_size=tf.shape(features)[0]
    features = tf.cast(features,tf.float32)
    
    with tf.variable_scope('layer_1'):
      layer_1_weights = tf.get_variable('weights',[config.audio_feat_dimension,config.n_hidden],initializer=tf.contrib.layers.xavier_initializer())
      layer_1_biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[config.n_hidden]))
      
      layer_1= tf.nn.relu( tf.add(tf.matmul(features,layer_1_weights),layer_1_biases) )

    with tf.variable_scope('layer_2'):
      layer_2_weights = tf.get_variable('weights',[config.n_hidden,config.n_hidden],initializer=tf.contrib.layers.xavier_initializer())
      layer_2_biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[config.n_hidden]))
      
      layer_2= tf.nn.relu( tf.add(tf.matmul(layer_1,layer_2_weights),layer_2_biases) )

    with tf.variable_scope('layer_3'):
      layer_3_weights = tf.get_variable('weights',[config.n_hidden,config.n_hidden],initializer=tf.contrib.layers.xavier_initializer())
      layer_3_biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[config.n_hidden]))
      
      layer_3= tf.nn.relu( tf.add(tf.matmul(layer_2,layer_3_weights),layer_3_biases) )

    with tf.variable_scope('layer_4'):
      layer_4_weights = tf.get_variable('weights',[config.n_hidden,config.n_hidden],initializer=tf.contrib.layers.xavier_initializer())
      layer_4_biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[config.n_hidden]))
      
      layer_4= tf.nn.relu( tf.add(tf.matmul(layer_3,layer_4_weights),layer_4_biases) )

    with tf.variable_scope('output'):
      output_weights = tf.get_variable('weights',[config.n_hidden,config.num_classes],initializer=tf.contrib.layers.xavier_initializer())
      output_biases = tf.get_variable('biases',initializer=tf.constant(0.1,shape=[config.num_classes]))
      output=tf.add(tf.matmul(layer_4, output_weights),output_biases)


    prediction=tf.argmax(output,axis=1)

    self._logits=output
    self._prediction = prediction
    self._softmax = tf.nn.softmax(output)

  @property
  def prediction(self):
    return self._prediction

  @property
  def logits(self):
    return self._logits

  @property
  def softmax(self):
    return self._softmax