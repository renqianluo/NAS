from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.layers import core as layers_core

class Decoder():
  def __init__(self, params, mode, embedding_decoder, output_layer):
    self.num_layers = params['decoder_num_layers']
    self.hidden_size = params['decoder_hidden_size']
    self.length = params['decode_length']
    self.vocab_size = params['vocab_size']
    self.embedding_decoder = embedding_decoder
    self.output_layer = output_layer
    self.time_major = params['time_major']
    self.beam_width = params.get('beam_width', 1)
    self.mode = mode

  def build_decoder(self, decoder_init_state, target_input, batch_size):
    tgt_sos_id = tf.constant(0)
    tgt_eos_id = tf.constant(0)

    self.batch_size = batch_size
 
    decoder_init_state = tf.concat([decoder_init_state, decoder_init_state], axis=-1)
    cell, decoder_initial_state = self.build_decoder_cell(decoder_init_state)

    if self.mode != tf.estimator.ModeKeys.PREDICT:
      if self.time_major:
        target_input = tf.transpose(target_input)
      decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
      #concat_inp_and_context = tf.concat([decoder_emb_inp, target_input], axis=0 if self.time_major else 1)
      #Helper
      helper = tf.contrib.seq2seq.TrainingHelper(
        decoder_emb_inp, tf.tile([self.length], [self.batch_size]),
        time_major=self.time_major)

      #Decoder
      my_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        decoder_initial_state)

      #Dynamic decoding
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
        my_decoder,
        output_time_major=self.time_major,
        swap_memory=False,
        scope=decoder_scope)

      sample_id = outputs.sample_id

      logits = self.output_layer(outputs.rnn_output)

    else:
      beam_width = self.beam_width
      start_tokens = tf.fill([self.batch_size], tgt_sos_id)
      end_token = tgt_eos_id

      if beam_width > 0:
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=cell,
            embedding=self.embedding_decoder,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
            beam_width=beam_width,
            output_layer=self.output_layer)
      else:
        # Helper
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            self.embedding_decoder, start_tokens, end_token)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,
            output_layer=self.output_layer  # applied per timestep
        )

      # Dynamic decoding
      outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          my_decoder,
          maximum_iterations=self.length,
          output_time_major=self.time_major,
          swap_memory=True,
          scope=decoder_scope)

      if beam_width > 0:
        logits = tf.no_op()
        sample_id = outputs.predicted_ids
      else:
        logits = outputs.rnn_output
        sample_id = outputs.sample_id

    return logits, sample_id, final_context_state


  def build_decoder_cell(self, decoder_init_state):
    cell_list = []
    for i in range(self.num_layers):
      lstm_cell = tf.contrib.rnn.LSTMCell(
        self.hidden_size,
        state_is_tuple=False)
      cell_list.append(lstm_cell)
    if len(cell_list) == 1:
      cell = cell_list[0]
    else:
      cell = tf.contrib.rnn.MultiRNNCell(cell_list)
      decoder_init_state = (decoder_init_state,) * self.num_layers
    # For beam search, we need to replicate encoder infos beam_width times
    if self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
      decoder_init_state = tf.contrib.seq2seq.tile_batch(
          decoder_init_state, multiplier=self.beam_width)

    return cell, decoder_init_state

class Model(object):
  def __init__(self,
               init_decoder_state,
               target_input,
               target,
               params,
               mode,
               scope=None):
    """Create the model."""
    self.params = params
    self.init_decoder_state = init_decoder_state
    self.batch_size = tf.shape(self.init_decoder_state)[0]
    self.target_input = target_input
    self.target = target
    self.mode = mode

    self.vocab_size = params['vocab_size']
    self.num_layers = params['decoder_num_layers']
    self.time_major = params['time_major']
    self.hidden_size = params['decoder_hidden_size']

    # Initializer
    initializer = tf.orthogonal_initializer()
    tf.get_variable_scope().set_initializer(initializer)

    ## Build graph
    self.build_graph(scope=scope)

  def build_graph(self, scope=None):
    tf.logging.info("# creating %s graph ..." % self.mode)
    ## Decoder
    with tf.variable_scope(scope):
      # Embeddings
      self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.hidden_size])
      # Projection
      with tf.variable_scope("output_projection"):
        self.output_layer = layers_core.Dense(
            self.vocab_size, use_bias=False, name="output_projection")
      self.logits, self.sample_id, self.final_context_state = self.build_decoder()

      ## Loss
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        self.compute_loss(logits)
      else:
        self.loss = None
        self.total_loss = None
  
  def build_decoder(self):
    decoder = Decoder(self.params, self.mode, self.W_emb, self.output_layer)
    logits, sample_id, final_context_state = decoder.build_decoder(self.init_decoder_state, self.target_input, self.batch_size)
    return logits, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]
 
  def compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.target
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.losses.sparse_softmax_cross_entropy(
        labels=target_output, logits=logits)
    tf.identity(crossent, 'cross_entropy')
    self.loss = crossent
    total_loss = mean_squared_error + self.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    self.total_loss = total_loss
  
  def train(self):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = tf.constant(params['lr'])
    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
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


    tf.summary.scalar("lr", self.learning_rate),
    tf.summary.scalar("train_loss", self.total_loss),
    tf.identity(self.learning_rate, 'learning_rate')

    return {
      'train_op': self.train_op, 
      'loss' : self.total_loss}

  def eval(self):
    assert self.mode == tf.estimator.ModeKeys.EVAL
    return {
      'loss' : self.total_loss,
    }

  def infer(self):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    return {
      'logits' : self.infer_logits,
      'sample_id' : self.sample_id,
    }

  def decode(self):
    res = self.infer()
    sample_id = res['sample_id']
    # make sure outputs is of shape [batch_size, time, 1]
    if self.time_major:
      sample_id = tf.transpose(sample_id, [1,0,2])
    return {
      'sample_id' : sample_id
    }
