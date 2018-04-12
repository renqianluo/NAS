from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


"""
class Decoder(object):
  def __init__(self, params):
    self.num_layers = params.get('num_layers', 1)
    self.hidden_size = params['hidden_size']
    self.batch_size = params['batch_size']
    self.length = params['length']
    self.vocab_size = params.get('vocab_size', len(list(_OPERATIONS)) + len(list(_NODES)))
    self.initializer = tf.random_normal_initializer(0.0, hidden_size**-0.5)
    self.init_decoder()

  def init_decoder(self):
    self.ff_state_W = self.get_variable('ff_state_W', [self.hidden_size, self.hidden_size],
      initializer=self.initializer)
    self.ff_state_b = self.get_variable('ff_state_b', initializer=tf.zeros(self.hidden_size))
    self.Wemb = self.get_variable('Wemb', [self.vocab_size, self.hidden_size],
      initializer=self.initializer)
    self.W = {}
    for layer_id in range(self.num_layers):
      name = 'Wx_{}'.format(layer_id)
      self.W[name] = tf.get_variable(name, [self.hidden_size, self.hidden_size*4],
        initializer=self.initializer)
      name = 'Wh_{}'.format(layer_id)
      self.W[name] = tf.get_variable(name, [self.hidden_size, self.hidden_size*4],
        initializer=self.initializer)
      name = 'b_{}'.format(layer_id)
      self.b[name] = tf.get_variable(name, initializer=tf.zeros([4*self.hidden_size]))
      name = 'Wc_{}'.format(layer_id)
      self.W[name] = tf.get_variable(name, [self.hidden_size, self.hidden_size*4],
        initializer=self.initializer)
    self.ff_logit_lstm_W = tf.get_variable('ff_logit_lstm_W', [self.hidden_size, self.hidden_size],
      initializer=self.initializer)
    self.ff_logit_lstm_b = tf.get_variable('ff_logit_lstm_b', initializer=tf.zeros(self.hidden_size))
    self.ff_logit_prev_W = tf.get_variable('ff_logit_prev_W', [self.hidden_size, self.hidden_size],
      initializer=self.initializer)
    self.ff_logit_prev_b = tf.get_variable('ff_logit_prev_b', initializer=tf.zeros(self.hidden_size))
    self.ff_logit_prev_W = tf.get_variable('ff_logit_ctx_W', [self.hidden_size, self.hidden_size],
      initializer=self.initializer)
    self.ff_logit_prev_b = tf.get_variable('ff_logit_ctx_b', initializer=tf.zeros(self.hidden_size))
    self.ff_logit_W = tf.get_variable('ff_logit_W', [self.hidden_size, self.vocab_size],
      initializer=self.initializer)
    self.ff_logit_b = tf.get_variable('ff_logit_b', initializer=tf.zeros(self.vocab_size))

  def lstm(state_below, context, init_state, init_memory, layer_id, one_step):
    n_steps = xstate_below.shape[0].value if state_below.shape.ndims == 3 else 1
    n_samples = state_below.shape[1].value if state_below.shape.ndims == 3 else state_below.shape[0].value
    state_below = tf.matmul(state_below, self.W['Wx_{}'.format(layer_id)]) + self.b['b_{}'.format(layer_id)]


    def _step_slice(x_, h_, c_, U):
      h_tmp = h_
      c_tmp = c_
      h, c = _lstm_step_slice(x, h_tmp, c_tmp, U[j])




  def decoder(self, tgt_embedding, init_state, context, one_step, init_memory=None):
    if not one_step:
      init_state = [init_state for _ in range(self.num_layers)]
      init_memory = [init_memory for _ in range(self.num_layers)]

    x = tgt_embedding

    for layer_id in range(self.num_layers):
      x = self.lstm(x, init_state=init_state[layer_id], context=context, 
        layer_id, init_memory=init_memory[layer_id], one_step=one_step)

    return x


  def get_probs(self, hidden_deocder, context_decoder, tgt_embedding):
    logit_lstm = tf.matmul(hidden_deocder, self.ff_logit_lstm_W) + self.ff_logit_lstm_b
    logit_prev = tf.matmul(tgt_embedding, self.ff_logit_prev_W) + self.ff_logit_prev_b
    logit_ctx = tf.matmul(context_decoder, self.ff_logit_ctx_W) + self.ff_logit_ctx_b
    logit = tf.tanh(logit_lstm + logit_prev + logit_ctx)
    logit = tf.matmul(logit, self.ff_logit_W) + self.ff_logit_b
    logit_shp = logit.shape
    probs = tf.nn.softmax(tf.reshape(logit, [logit_shp[0].value * logit_shp[1].value, logit_shp[2].value]))

    return probs

  def build_model(self, x, y, emb):
    init_decoder_state = tf.tanh(tf.matmul(emb, self.ff_state_W) + self.ff_logit_b)

    n_timestep, n_samples = x.shape[0].value, x.shape[1].value
    n_timestep_tgt = y.shape[0].value

    emb_shifted = tf.zeros([1, n_samples, self.vocab_size])
    emb_shifted = tf.concat([emb_shifted, y[:-1]])
    tgt_embedding = emb_shifted

    hidden_deocder, context_decoder = self.decoder(
      tgt_embedding, init_state=init_decoder_state, context=emb, one_step=False)
    
    probs = self.get_probs(hidden_deocder, context_decoder, tgt_embedding)

    self.cost = self.build_cost(y, probs)

def decoder(x, params, is_training):

  #num_layers = params['num_layers']
  hidden_size = params['hidden_size']
  batch_size = params['batch_size']
  length = params['length']
  vocab_size = params.get('vocab_size', len(list(_OPERATIONS)) + len(list(_NODES)))
  
  assert x.shape.ndims == 3, '[batch_size, 1, hidden_size]'

  w_s = tf.get_variable('softmax', [hidden_size, vocab_size],
      initializer=tf.random_normal_initializer(0.0, hidden_size**-0.5))

  logits = tf.zeros([batch_size, 0, hidden_size])

  with tf.variable_scope('body'):
    lstm_cell = tf.contrib.rnn.LSTMCell(
      hidden_size,
      initializer=tf.orthogonal_initializer())

    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    x, state = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)

  #logits = 
  
  return x
  """
class Decoder():
  def __init__(self, params, mode, embedding_decoder, output_layer):
    self.num_layers = params.get('num_layers', 1)
    self.hidden_size = params['hidden_size']
    self.batch_size = params['batch_size']
    self.length = params['length']
    self.vocab_size = params['vocab_size']
    self.embedding_decoder = embedding_decoder
    self.output_layer = output_layer
    self.time_major = params['time_major']
    self.beam_width = params.get('beam_width', 1)
    self.mode = mode

  def build_decoder(self, decoder_init_state, target_input, batch_size):
    self.batch_size = batch_size
    
    with tf.variable_scope('Decoder') as decoder_scope:
      cell, decoder_initial_state = self.build_decoder_cell(decoder_init_state)

      if self.mode != tf.estimator.ModeKeys.PREDICT:
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)

        #concat_inp_and_context = tf.concat([decoder_emb_inp, target_input], axis=0 if self.time_major else 1)

        #Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
          decoder_emb_inp, self.length,
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
          swap_memory=True,
          scop=decoder_scope)

        sample_id = outputs.sample_id

        logits = self.output_layer(outputs.rnn_output)

      else:
        beam_width = self.beam_width
        start_tokens = tf.fill([self.batch_size], 0)
        end_token = 0

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
    cell = tf.contrib.rnn.LSTMCell(
      hidden_size,
      initializer=tf.orthogonal_initializer())

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
    """Create the model.

    Args:
      params: Hyperparameter configurations.
      mode: TRAIN | EVAL | PREDICT
      init_decoder_state, target_input, target: Dataset Iterator that feeds data.
      scope: scope of the model.
    """
    self.params = params
    self.init_decoder_state = init_decoder_state
    self.target_input = target_input
    self.target = target
    self.mode = mode

    self.vocab_size = params['vocab_size']
    self.num_layers = params.get('num_layers',1)
    self.time_major = params['time_major']
    self.hidden_size = params['hidden_size']

    # Initializer
    initializer = tf.random_normal_initializer(0.0, self.hidden_size**-0.5)
    tf.get_variable_scope().set_initializer(initializer)

    # Embeddings
    self.batch_size = tf.size(self.target_input.shape[0].value)
    self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.hidden_size])
    # Projection
    with tf.variable_scope("output_projection"):
      self.output_layer = layers_core.Dense(
          self.vocab_size, use_bias=False, name="output_projection")
    ## Train graph
    res = self.build_graph(scope=scope)

    if self.mode == tf.estimator.ModeKeys.TRAIN:
      self.train_loss = res[1]
    elif self.mode == tf.estimator.ModeKeys.EVAL:
      self.eval_loss = res[1]
    elif self.mode == tf.estimator.ModeKeys.PREDICT:
      self.infer_logits, _, self.final_context_state, self.sample_id = res
    
    self.global_step = tf.get_or_create_global_step()

    net_params = tf.trainable_variables()

    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.mode == tf.estimator.ModeKeys.TRAIN:
      self.learning_rate = tf.constant(params['lr'])

      if params['optimizer'] == "sgd":
        self.learning_rate = tf.cond(
            self.global_step < params['start_decay_step'],
            lambda: self.learning_rate,
            lambda: tf.train.exponential_decay(
                self.learning_rate,
                (self.global_step - params['start_decay_step']),
                params['decay_steps'],
                params['decay_factor'],
                staircase=True),
            name="learning_rate")
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      elif params['optimizer'] == "adam":
        assert float(
            self.learning_rate
        ) <= 0.001, "! High Adam learning rate %g" % self.learning_rate
        opt = tf.train.AdamOptimizer(self.learning_rate)

      gradients = tf.gradients(self.train_loss, net_params)

      clipped_gradients = tf.clip_by_global_norm(
        gradients, params['max_gradient_norm'])

      self.update = opt.apply_gradients(
          zip(clipped_gradients, net_params), global_step=self.global_step)


      tf.summary.scalar("lr", self.learning_rate),
      tf.summary.scalar("train_loss", self.train_loss),
      tf.identity(self.learning, 'learning_rate')

    # Print trainable variables
    print("# Trainable variables")
    for param in all_params:
      print("  %s, %s, %s" % (param.name, str(param.get_shape()),
                                        param.op.device))
    
  def train(self):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    return {
      'train_op': self.update, 
      'loss' : self.train_loss}

  def eval(self):
    assert self.mode == tf.estimator.ModeKeys.EVAL
    return {
      'loss' : self.eval_loss,
    }

  def build_graph(self, scope=None):
    """Subclass must implement this method.
    Creates a sequence-to-sequence model with dynamic RNN decoder API.
    Args:
      scope: VariableScope for the created subgraph; default "dynamic_seq2seq".

    Returns:
      A tuple of the form (logits, loss, final_context_state),
      where:
        logits: float32 Tensor [batch_size x num_decoder_symbols].
        loss: the total loss / batch_size.
        final_context_state: The final state of decoder RNN.

    Raises:
      ValueError: if encoder_type differs from mono and bi, or
        attention_option is not (luong | scaled_luong |
        bahdanau | normed_bahdanau).
    """
    print("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      ## Decoder
      logits, sample_id, final_context_state = self.build_decoder()

      ## Loss
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        loss = self._compute_loss(logits)
      else:
        loss = None
      
      return logits, loss, final_context_state, sample_id

  
  def build_decoder(self):
    """Build and run a RNN decoder with a final projection layer.
    Subclass must implement this.
    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    decoder = Decoder(self.params, self.mode, self.embedding_decoder, self.output_layer, self.W_emb)
    logits, sample_id, final_context_state = decoder.build_decoder(self.init_decoder_state, self.target_input, self.batch_size)
    return logits, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]
 
  def _compute_loss(self, logits):
    """Compute optimization loss."""
    target_output = self.target
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.losses.sparse_softmax_cross_entropy_with_logits(
        labels=target_output, logits=logits)
    tf.identity(crossent, 'cross_entropy')
    return crossent

  def infer(self):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    return {
      'logits' : self.infer_logits,
      'sample_id' : self.sample_id,
    }

  def decode(self):
    """Decode a batch.

    Args:
      sess: tensorflow session to use.

    Returns:
      A tuple consiting of outputs, infer_summary.
        outputs: of size [batch_size, time]
    """
    res = self.infer()
    sample_id = res['sample_id']
    # make sure outputs is of shape [batch_size, time]
    if self.time_major:
      sample_id = sample_id.transpose()
    return {
      'sample_id' : sample_id
    }
