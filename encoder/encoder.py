from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5


class Encoder(object):
  def __init__(self, params, mode, W_emb):
    self.num_layers = params['encoder_num_layers']
    self.hidden_size = params['encoder_hidden_size']
    self.emb_size = params['encoder_emb_size']
    self.mlp_num_layers = params['mlp_num_layers']
    self.mlp_hidden_size = params['mlp_hidden_size']
    self.length = params['length']
    self.vocab_size = params['vocab_size']
    self.input_keep_prob = params['input_keep_prob']
    self.output_keep_prob = params['output_keep_prob']
    self.W_emb = W_emb
    self.mode = mode
  def build_encoder(self, x, batch_size, is_training):
    self.batch_size = batch_size
    assert x.shape.ndims == 2, '[batch_size, length]'
    x = tf.gather(self.W_emb, x)
    x = tf.reshape(x, [batch_size, self.length//3, 3*self.emb_size])
    cell_list = []
    for i in range(self.num_layers):
      lstm_cell = tf.contrib.rnn.LSTMCell(
        self.hidden_size)
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, 
        input_keep_prob=self.input_keep_prob, 
        output_keep_prob=self.output_keep_prob)
      cell_list.append(lstm_cell)
    #initial_state = cell_list[0].zero_state(batch_size, dtype=tf.float32)
    if len(cell_list) == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    x, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)#initial_state=initial_state, dtype=tf.float32)
    
    self.encoder_output = x
    self.encoder_state = state
    x = tf.reduce_mean(x, axis=1)
    x = tf.nn.l2_normalize(x, dim=1)

    self.arch_emb = x
    
    for i in range(self.mlp_num_layers):
      name = 'mlp_{}'.format(i)
      x = tf.layers.dense(x, self.mlp_hidden_size, activation=tf.nn.relu, name=name)
      x = tf.layers.batch_normalization(
        x, axis=1,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training, fused=True
        )
      x = tf.layers.dropout(x, 1 - self.output_keep_prob)
    self.predict_value = tf.layers.dense(x, 1, activation=tf.sigmoid, name='regression')
    return {
      'arch_emb' : self.arch_emb,
      'predict_value' : self.predict_value,
      'encoder_output' : self.encoder_output,
      'encoder_state' : self.encoder_state,
    }


class Model(object):
  def __init__(self, x, y, params, mode, scope=None):
    self.x = x
    self.y = y
    self.params = params
    self.batch_size = tf.shape(x)[0]
    self.vocab_size = params['vocab_size']
    self.emb_size = params['encoder_emb_size']
    self.weight_decay = params['weight_decay']
    self.mode = mode
    self.is_training = self.mode == tf.estimator.ModeKeys.TRAIN
    if self.is_training:
      self.params['input_keep_prob'] = 1.0
      self.params['output_keep_prob'] = 1.0

    initializer = tf.orthogonal_initializer()
    tf.get_variable_scope().set_initializer(initializer)
    self.build_graph(scope=scope)

  
  def build_graph(self, scope=None):
    tf.logging.info("# creating %s graph ..." % self.mode)
    # Encoder
    with tf.variable_scope(scope):
      self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.emb_size])
      self.arch_emb, self.predict_value, self.encoder_output, self.encoder_state = self.build_encoder()
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        self.compute_loss()
      else:
        self.loss = None
        self.total_loss = None

  def build_encoder(self):
    encoder = Encoder(self.params, self.mode, self.W_emb)
    res = encoder.build_encoder(self.x, self.batch_size, self.is_training)
    return res['arch_emb'], res['predict_value'], res['encoder_output'], res['encoder_state']

  def compute_loss(self):
    mean_squared_error = tf.losses.mean_squared_error(labels=self.y, predictions=self.predict_value)
    #mean_squared_error = tf.losses.huber_loss(labels=self.y, predictions=self.predict_value)
    tf.identity(mean_squared_error, name='squared_error')
    tf.summary.scalar('mean_squared_error', mean_squared_error)
    # Add weight decay to the loss.
    self.loss = mean_squared_error
    total_loss = mean_squared_error + self.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    self.total_loss = total_loss

  def train(self):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = tf.constant(self.params['lr'])
    if self.params['optimizer'] == "sgd":
      self.learning_rate = tf.cond(
        self.global_step < self.params['start_decay_step'],
        lambda: self.learning_rate,
        lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - self.params['start_decay_step']),
                self.params['decay_steps'],
                self.params['decay_factor'],
                staircase=True),
            name="learning_rate")
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    elif self.params['optimizer'] == "adam":
      assert float(
        self.params['lr']
      ) <= 0.001, "! High Adam learning rate %g" % self.params['lr']
      opt = tf.train.AdamOptimizer(self.learning_rate)
    elif self.params['optimizer'] == 'adadelta':
      opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      gradients, variables = zip(*opt.compute_gradients(self.total_loss))
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params['max_gradient_norm'])
      self.train_op = opt.apply_gradients(
        zip(clipped_gradients, variables), global_step=self.global_step)


    tf.identity(self.learning_rate, 'learning_rate')
    tf.summary.scalar("learning_rate", self.learning_rate),
    tf.summary.scalar("total_loss", self.total_loss),
    
    return {
      'train_op' : self.train_op,
      'loss' : self.total_loss,
    }

  def eval(self):
    assert self.mode == tf.estimator.ModeKeys.EVAL
    return {
      'loss': self.total_loss,
    }

  def infer(self):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    return {
      'arch_emb' : self.arch_emb,
      'predict_value' : self.predict_value,
    }
