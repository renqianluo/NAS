from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

_NODES=[
  'node1',
  'node2',
  'node3',
  'node4',
  'node5',
  'node6'
]

def encoder(x, params, is_training):

  #num_layers = params['num_layers']
  hidden_size = params['hidden_size']
  batch_size = tf.shape(x)[0]
  length = params['length']
  vocab_size = params.get('vocab_size', len(list(_OPERATIONS)) + len(list(_NODES)))
  
  assert x.shape.ndims == 3, '[batch_size, length, 1]'

  with tf.name_scope('embedding'):
    x = tf.squeeze(x, axis=2)
    emb = tf.get_variable('W_emb', [vocab_size, hidden_size],
    	initializer=tf.random_normal_initializer(0.0, hidden_size**-0.5))
    x = tf.gather(emb, x)

  with tf.variable_scope('body'):
    lstm_cell = tf.contrib.rnn.LSTMCell(
      hidden_size,
      initializer=tf.orthogonal_initializer())
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    x, state = tf.nn.dynamic_rnn(lstm_cell, x, initial_state=initial_state, dtype=tf.float32)

    structure_emb = tf.reduce_mean(x, axis=1)
  
    predict_value = tf.layers.dense(structure_emb, 1, activation=tf.sigmoid, name='regression')
  
  return {
    'structure_emb' : structure_emb,
    'predict_value' : predict_value,
  }

