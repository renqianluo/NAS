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

def encoder(inputs, params, is_training):

  num_layers = params['num_layers']
  hidden_size = params['hidden_size']
  batch_size = params['batch_size']
  length = params['length']
  
  assert inputs.ndim == 3, '[batch_size, length, dim]'
  static_shape = inputs.shape
  assert batch_size == static_shape[0]
  assert length == static_shape[1]
  assert hidden_size == static_shape[2]


  with tf.variable_scope('body'):
    lstm_cell = tf.contrib.rnn.LSTMCell(
      hidden_size,
      initializer=tf.orthogonal_initializer(),
      activation=tf.sigmoid,
      name='lstm')
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float332)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)

    assert output.ndim == 3 and \
      output.shape[0] == batch_size and \
      output.shape[1] == length and \
      output.shape[2] == hidden_size, '[batch_size, dim]'
    output = tf.reduce_mean(outputs, axis=1)
  
    output = tf.layers.dense(output, 1, activation=tf.sigmoid, name='regression')

    assert output.ndim == 2 and output.shape[0] == batch_size and output.shape[1] == 1
  
  return output