from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import model
import six
import random
import json

_NUM_SAMPLES = {
  'train' : 500,
  'test' : 100,
}

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test', 'predict'],
                    help='Train, eval or infer.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--restore', action='store_true', default=False,
                    help='Restore from a configuration params.')

parser.add_argument('--hidden_size', type=int, default=32,
                    help='The number of nodes in a cell.')

parser.add_argument('--B', type=int, default=5,
                    help='The number of non-input-nodes in a cell.')

parser.add_argument('--weight_decay', type=int, default=1e-4,
                    help='Weight decay.')

parser.add_argument('--vocab_size', type=float, default=19,
                    help='Vocabulary size.')

parser.add_argument('--train_epochs', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--eval_frequency', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--lr_schedule', type=str, default='decay',
                    choices=['constant', 'decay'],
                    help='Learning rate schedule schema.')

parser.add_argument('--lr', type=float, default=1.0,
                    help='Learning rate when learning rate schedule is constant.')

# Below are arguments for predicting
parser.add_argument('--predict_from_file', type=str, default=None,
                    help='File to predict from.')

parser.add_argument('--predict_to_file', type=str, default=None,
                    help='File to store predictions.')


def get_filenames(mode, data_dir):
  """Returns a list of filenames."""
  if mode == 'train':
    return [os.path.join(data_dir, 'train.tfrecords')]
  else:
    return [os.path.join(data_dir, 'test.tfrecords')]

def input_fn(mode, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.TFRecordDataset(get_filenames(mode, data_dir))
  
  is_training = mode == 'train'

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_SAMPLES['train'])

  def cast_int64_to_int32(features):
    f = {}
    for k, v in six.iteritems(features):
      if v.dtype == tf.int64:
        v = tf.to_int32(v)
      f[k] = v
    return f

  def decode_record(record):
    """Serialized Example to dict of <feature name, Tensor>."""
    data_fields = {
      'inputs' : tf.FixedLenFeature([4*FLAGS.B*2], tf.int64),
      'targets' : tf.FixedLenFeature([1], tf.float32)
    }
    data_items_to_decoders = {
        field: tf.contrib.slim.tfexample_decoder.Tensor(field)
        for field in data_fields
    }
    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(
        data_fields, data_items_to_decoders)

    decode_items = list(data_items_to_decoders)
    decoded = decoder.decode(record, items=decode_items)
    return dict(zip(decode_items, decoded))

  dataset = dataset.map(decode_record)
  dataset = dataset.map(cast_int64_to_int32)

  #dataset = dataset.prefetch(2 * batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  batched_examples = iterator.get_next()

  inputs = batched_examples['inputs']
  targets = batched_examples['targets']

  while inputs.shape.ndims < 3:
    inputs = tf.expand_dims(inputs, axis=-1)
  while targets.shape.ndims < 2:
    targets = tf.expand_dims(targets, axis=-1)
  return inputs, targets

def _log_variable_sizes(var_list, tag):
  """Log the sizes and shapes of variables, and the total size.

    Args:
      var_list: a list of varaibles
      tag: a string
  """
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
      v.name[:-2].ljust(80),
      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)

def model_fn(features, labels, mode, params):
  inputs = features
  
  predict_value = model.encoder(inputs, params, mode == tf.estimator.ModeKeys.TRAIN)
  tf.identity(predict_value, name='predict_value')  

  predictions = {
    'predict_value': predict_value,
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=predict_value)
  tf.identity(mean_squared_error, name='MSE')
  tf.summary.scalar('mean_squared_error', mean_squared_error)
  # Add weight decay to the loss.
  loss = mean_squared_error + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.EVAL:
    #TODO: pearson
    mean_squared_error = tf.metrics.mean_squared_error(labels, predict_value)
    metrics = {
      'mean_squared_error': mean_squared_error,}

    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN

  global_step = tf.train.get_or_create_global_step()

  if params['lr_schedule'] == 'decay':
    batches_per_epoch = _NUM_SAMPLES['train'] / params['batch_size']
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 500, 1000]]
    values = [params['lr'] * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)
  else:
    learning_rate = params['lr']

  # Create a tensor named learning_rate for logging purposes
  tf.identity(learning_rate, name='learning_rate')
  tf.summary.scalar('learning_rate', learning_rate)
 
  optimizer = tf.train.AdadeltaOptimizer(
    learning_rate=learning_rate)

  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss, global_step)
    #gradients, variables = zip(*optimizer.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op)

def get_inputs(filename, delimiter='\n'):
  tf.logging.info('Getting inputs')
  with tf.gfile.Open(filename) as f:
    text = f.read()
    records = text.split(delimiter)
    inputs = [record.strip().split() for record in records]
    if not inputs[-1]:
      inputs.pop()
  return inputs

def predict_from_file(estimator, batch_size, filename, decode_to_file=None, delimiter='\n'):
  inputs = get_inputs(filename)
  num_batches = (len(inputs) - 1) // batch_size + 1
  
  def batch_input_fn(num_batches, x, batch_size):
    tf.logging.info('batch %d' % num_batches)
    batch_inputs = []
    for b in range(num_batches):
      tf.logging.info('Predicting batch %d' % b)
      for i in x[b * batch_size:(b+1) * batch_size]:
        batch_inputs.append(i)
      yield np.array(batch_inputs).astype(np.int32)

  def make_input_fn_from_generator(gen):
    first_ex = six.next(gen)
    flattened = tf.contrib.framework.nest.flatten(first_ex)
    types = [t.dtype for t in flattened]
    shapes = [[None] * len(t.shape) for t in flattened]
    first_ex_list = [first_ex]

    def py_func():
      if first_ex_list:
        example = first_ex_list.pop()
      else:
        example = six.next(gen)
      return tf.contrib.framework.nest.flatten(example)

    def input_fn():
      flat_example = tf.py_func(py_func, [], types)
      _ = [t.set_shape(shape) for t, shape in zip(flat_example, shapes)]
      example = tf.contrib.framework.nest.pack_sequence_as(first_ex, flat_example)
      return example

    return input_fn

  def input_fn():
    input_gen = batch_input_fn(num_batches, inputs, batch_size)
    gen_fn = make_input_fn_from_generator(input_gen)
    example = gen_fn()
    example = tf.convert_to_tensor(example)
    while example.shape.ndims < 3:
      example = tf.expand_dims(example, axis=-1)
    example = tf.to_int32(example)
    return example

  results = []
  result_iter = estimator.predict(input_fn)
  for result in result_iter:
    result = result['predict_value']
    result = ' '.join(map(str, result.flatten()))
    tf.logging.info('Inference results OUTPUT: %s' % result)
    results.append(result)
  
  if decode_to_file:
    output_filename = decode_to_file
  else:
    output_filename = '%s.result' % filename

  tf.logging.info('Writing results into %s' % output_filename)
  with tf.gfile.Open(output_filename, 'w') as f:
    for result in results:
      f.write('%s%s' % (result, delimiter))

def get_params():
  params = vars(FLAGS)
  params['length'] = 4*FLAGS.B*2

  if FLAGS.restore:
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)
    params.update(old_params)

  return params 


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.mode == 'train':
    params = get_params()

    #model_fn(tf.zeros([128,40,1], dtype=tf.int32),tf.zeros([128,1]),tf.estimator.ModeKeys.TRAIN, params)

    #_log_variable_sizes(tf.trainable_variables(), "Trainable Variables")

    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
      json.dump(params, f)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=params['model_dir'], config=run_config,
      params=params)
    for _ in range(params['train_epochs'] // params['eval_frequency']):
      tensors_to_log = {
          'learning_rate': 'learning_rate',
          'mean_squared_error': 'MSE',#'mean_squared_error'
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      estimator.train(
          input_fn=lambda: input_fn(
              'train', params['data_dir'], params['batch_size'], params['eval_frequency']),
          hooks=[logging_hook])
      
      # Evaluate the model and print results
      eval_results = estimator.evaluate(
          input_fn=lambda: input_fn('test', params['data_dir'], _NUM_SAMPLES['test']))
      tf.logging.info('Evaluation on test data set')
      print(eval_results)

  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
  
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    eval_results = estimator.evaluate(
          input_fn=lambda: input_fn('test', FLAGS.data_dir, _NUM_SAMPLES['test']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)

  elif FLAGS.mode == 'predict':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    
    predict_from_file(estimator, FLAGS.batch_size, FLAGS.predict_from_file, FLAGS.predict_to_file)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
