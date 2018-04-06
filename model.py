from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import tensorflow as tf

_OPERATIONS=[
  'identity',
  'sep_conv 3x3',
  'sep_conv 5x5',
  'sep_conv 7x7',
  'avg_pool 2x2',
  'avg_pool 3x3',
  'avg_pool 5x5',
  'max_pool 2x2',
  'max_pool 3x3',
  'max_pool 5x5',
  'max_pool 7x7',
  'min_pool 2x2',
  'conv 1x1',
  'conv 3x3',
  'conv 1x3+3x1',
  'conv 1x7+7x1',
  'dil_sep_conv 3x3',
  'dil_sep_conv 5x5',
  'dil_sep_conv 7x7',
#  'dil_conv 3x3 2',
#  'dil_conv 3x3 4',
#  'dil_conv 3x3 6',
]

def encoder(x, params, is_training):

  #num_layers = params['num_layers']
  hidden_size = params['hidden_size']
  batch_size = params['batch_size']
  length = params['length']
  vocab_size = params.get('vocab_size', len(list(_OPERATIONS)))
  
  assert x.shape.ndims == 3, '[batch_size, length, 1]'
  static_shape = x.get_shape()
  #assert length == static_shape[1].value
  #assert 1 == static_shape[2].value

  with tf.name_scope('embedding'):
    x = tf.squeeze(x, axis=2)
    emb = tf.get_variable('embedding', [vocab_size, hidden_size],
    	initializer=tf.random_normal_initializer(0.0, hidden_size**-0.5))
    x = tf.gather(emb, x)
  
  assert x.shape.ndims == 3, '[batch_size, length, hidden_dim]'
  static_shape = x.get_shape()
  #assert length == static_shape[1].value
  assert hidden_size == static_shape[2].value

  with tf.variable_scope('body'):
    lstm_cell = tf.contrib.rnn.LSTMCell(
      hidden_size,
      initializer=tf.orthogonal_initializer(),
      activation=tf.sigmoid)
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    x, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

    assert x.shape.ndims == 3 #and \
      #x.get_shape()[1].value == length and \
      #x.get_shape()[2].value == hidden_size, '[batch_size, hidden_dim]'
    x = tf.reduce_mean(x, axis=1)
  
    x = tf.layers.dense(x, 1, activation=tf.sigmoid, name='regression')

    assert x.shape.ndims == 2 and x.get_shape()[1].value == 1
  
  return x
