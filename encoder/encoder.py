from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def encoder(x, params, is_training):

  num_layers = params['num_layers']
  hidden_size = params['hidden_size']
  batch_size = tf.shape(x)[0]
  length = params['length']
  vocab_size = params['vocab_size']
  input_keep_prob = params['input_keep_prob']
  output_keep_prob = params['output_keep_prob']
  
  assert x.shape.ndims == 3, '[batch_size, length, 1]'

  with tf.name_scope('embedding'):
    x = tf.squeeze(x, axis=2)
    emb = tf.get_variable('W_emb', [vocab_size, hidden_size],
    	initializer=tf.orthogonal_initializer())
    x = tf.gather(emb, x)

  with tf.variable_scope('body'):
    cell_list = []
    for i in range(num_layers):
      lstm_cell = tf.contrib.rnn.LSTMCell(
        hidden_size,
        initializer=tf.orthogonal_initializer())
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
      cell_list.append(lstm_cell)
    #initial_state = cell_list[0].zero_state(batch_size, dtype=tf.float32)
    if len(cell_list) == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    cell_list = []
    for i in range(num_layers):
      lstm_cell = tf.contrib.rnn.LSTMCell(
        hidden_size,
        initializer=tf.orthogonal_initializer())
      lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
      cell_list.append(lstm_cell)
    #initial_state = cell_list[0].zero_state(batch_size, dtype=tf.float32)
    if len(cell_list) == 1:
      cell_b = cell_list[0]
    else:
      cell_b = tf.contrib.rnn.MultiRNNCell(cell_list)
    #x, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)#initial_state=initial_state, dtype=tf.float32)
    x, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_b, x, dtype=tf.float32)
    x = x[0] + x[1]
    structure_emb = tf.reduce_mean(x, axis=1)
    #structure_emb = x[:, -1, :]
  
    predict_value = tf.layers.dense(structure_emb, 1, activation=tf.sigmoid, name='regression')
  
  return {
    'structure_emb' : structure_emb,
    'predict_value' : predict_value,
  }

