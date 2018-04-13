from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import scipy.stats
import tensorflow as tf
import six
import random
import json
import encoder

_NUM_SAMPLES = {
  'train' : 500,
  'test' : 17,
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

parser.add_argument('--num_layers', type=int, default=1,
                    help='The number of nodes in a cell.')

parser.add_argument('--hidden_size', type=int, default=32,
                    help='The number of nodes in a cell.')

parser.add_argument('--B', type=int, default=5,
                    help='The number of non-input-nodes in a cell.')

parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay.')

parser.add_argument('--max_gradient_norm', type=float, default=5.0,
                    help='Weight decay.')

parser.add_argument('--vocab_size', type=int, default=26,
                    help='Vocabulary size.')

parser.add_argument('--train_epochs', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--eval_frequency', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--lr_schedule', type=str, default='constant',
                    choices=['constant', 'decay'],
                    help='Learning rate schedule schema.')

parser.add_argument('--lr', type=float, default=1.0,
                    help='Learning rate when learning rate schedule is constant.')

# Below are arguments for predicting
parser.add_argument('--decode_from_file', type=str, default=None,
                    help='File to decode from.')

parser.add_argument('--decode_to_file', type=str, default=None,
                    help='File to store predictions.')

def input_fn(params, mode, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  def get_filenames(mode, data_dir):
    """Returns a list of filenames."""
    if mode == 'train':
      return [os.path.join(data_dir, 'train.input'), os.path.join(data_dir, 'train.target')]
    else:
      return [os.path.join(data_dir, 'test.input'), os.path.join(data_dir, 'test.target')]

  files = get_filenames(mode, data_dir)
  input_dataset = tf.data.TextLineDataset(files[0])
  target_dataset = tf.data.TextLineDataset(files[1])

  dataset = tf.data.Dataset.zip((input_dataset, target_dataset))
  
  is_training = mode == 'train'

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_SAMPLES['train'])

  def decode_record(src, tgt):
    src = tf.string_split([src]).values
    src = tf.string_to_number(src, out_type=tf.int32)
    tgt = tf.string_to_number(tgt, out_type=tf.float32)
    return (src, tgt)

  dataset = dataset.map(decode_record)

  #dataset = dataset.prefetch(2 * batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  batched_examples = iterator.get_next()

  inputs, targets = batched_examples

  assert inputs.shape.ndims == 2
  while inputs.shape.ndims < 3:
    inputs = tf.expand_dims(inputs, axis=-1)
  assert targets.shape.ndims == 1
  while targets.shape.ndims < 2:
    targets = tf.expand_dims(targets, axis=-1)

  inputs = {
    'inputs' : inputs,
    'targets' : targets,
  }
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

def _del_dict_nones(d):
  for k in list(d.keys()):
    if d[k] is None:
      del d[k]

def model_fn(features, labels, mode, params):
  inputs = features['inputs']
  targets = features.get('targets', None)

  res = encoder.encoder(inputs, params, mode == tf.estimator.ModeKeys.TRAIN)
  predict_value = res['predict_value']
  structure_emb = res['structure_emb']

  tf.identity(predict_value, name='predict_value')  

  predictions = {
    'inputs' : inputs,
    'targets' : targets,
    'predict_value': predict_value,
    'structure_emb':structure_emb,
  }

  _del_dict_nones(predictions)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=predict_value)
  
  tf.identity(mean_squared_error, name='squared_error')
  tf.summary.scalar('mean_squared_error', mean_squared_error)
  # Add weight decay to the loss.
  loss = mean_squared_error + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  if mode == tf.estimator.ModeKeys.EVAL:
    #TODO: pearson
    squared_error = tf.metrics.mean_squared_error(labels, predict_value)
    metrics = {
      'squared_error': squared_error,}

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
    #train_op = optimizer.minimize(loss, global_step)
    gradients, variables = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op)


def predict_from_file(estimator, batch_size, decode_from_file, decode_to_file=None):
  def infer_input_fn():
    dataset = tf.data.TextLineDataset(decode_from_file)
    dataset = dataset.map(lambda record: tf.string_to_number(tf.string_split([record]).values, out_type=tf.int32))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs = iterator.get_next()
    assert inputs.shape.ndims == 2
    while inputs.shape.ndims < 3:
      inputs = tf.expand_dims(inputs, axis=-1)

    return {
      'inputs' : inputs, 
    }, None
  
  scores, embs = [], []
  result_iter = estimator.predict(infer_input_fn)
  for result in result_iter:
    predict_value = result['predict_value'].flatten()
    emb = result['structure_emb'].flatten()
    predict_value = ' '.join(map(str, predict_value))
    emb = ' '.join(list(map(str, emb)))
    tf.logging.info('Inference results OUTPUT: {}'.format(predict_value))
    scores.append(predict_value)
    embs.append(emb)

  if decode_to_file:
    score_output_filename = '{}.score'.format(decode_to_file)
    emb_output_filename = '{}.emb'.format(decode_to_file)
  else:
    score_output_filename = '{}.score'.format(decode_from_file)
    emb_output_filename = '{}.emb'.format(decode_from_file)

  tf.logging.info('Writing results into {0} and {1}'.format(score_output_filename, emb_output_filename))
  with tf.gfile.Open(score_output_filename, 'w') as f:
    for score in scores:
      f.write('{}\n'.format(score))
  with open(emb_output_filename, 'w') as f:
    for emb in embs:
      f.write('{}\n'.format(emb))

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
          'mean_squared_error': 'squared_error',#'mean_squared_error'
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      estimator.train(
          input_fn=lambda: input_fn(
              params, 'train', params['data_dir'], params['batch_size'], params['eval_frequency']),
          hooks=[logging_hook])
      
      # Evaluate the model and print results
      """
      eval_results = estimator.evaluate(
          input_fn=lambda: input_fn('test', params['data_dir'], _NUM_SAMPLES['test']))
      tf.logging.info('Evaluation on test data set')
      print(eval_results)
      """
      result_iter = estimator.predict(lambda: input_fn(params, 'test', params['data_dir'], params['batch_size']))
      predictions_list, targets_list = [], []
      for i, result in enumerate(result_iter):
        predict_value = result['predict_value'].flatten()#[0]
        targets = result['targets'].flatten()#[0]
        predictions_list.extend(predict_value)
        targets_list.extend(targets)
      predictions_list = np.array(predictions_list)
      targets_list = np.array(targets_list)
      mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
      sorted_predictions_list = np.argsort(predictions_list)
      sorted_targets_list = np.argsort(targets_list)
      pearson_result = scipy.stats.spearmanr(sorted_predictions_list, sorted_targets_list)
      tf.logging.info('pearson correlation = {0}, pvalue = {1}'.format(pearson_result.correlation, pearson_result.pvalue))
      tf.logging.info('mean squared error = {0}'.format(mse))

  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
  
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    """eval_results = estimator.evaluate(
          input_fn=lambda: input_fn('test', FLAGS.data_dir, _NUM_SAMPLES['test']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)"""
    result_iter = estimator.estimator.predict(lambda: input_fn(params, 'test', FLAGS.data_dir, _NUM_SAMPLES['test']))
    predictions_list, targets_list = [], []
    for i, result in enumerate(result_iter):
      predict_value = result['predict_value'].flatten()
      targets = result['targets'].flatten()
      predictions_list.extend(predict_value)
      targets_list.extend(targets)
    predictions_list = np.array(predictions_list)
    targets_list = np.array(targets_list)
    mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
    sorted_predictions_list = np.argsort(predictions_list)
    sorted_targets_list = np.argsort(targets_list)
    pearson_result = scipy.stats.spearmanr(sorted_predictions_list, sorted_targets_list)
    tf.logging.info('pearson correlation = {0}, pvalue = {1}'.format(pearson_result.correlation, pearson_result.pvalue))
    tf.logging.info('mean squared error = {0}'.format(mse))

  elif FLAGS.mode == 'predict':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    
    predict_from_file(estimator, FLAGS.batch_size, FLAGS.decode_from_file, FLAGS.decode_to_file)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)