from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
import model
import six

_NUM_SAMPLES = {
  'train' : 500,
  'test' : 100,
}

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='Train, or test.')

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

parser.add_argument('--train_steps', type=int, default=300,
                    help='The number of epochs to train.')

parser.add_argument('--eval_frequency', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--lr_schedule', type=str, default='decay',
                    choices=['constant', 'decay'],
                    help='Learning rate schedule schema.')

parser.add_argument('--lr', type=float, default='0.1',
                    help='Learning rate when learning rate schedule is constant.')


def get_filenames(mode, data_dir):
  """Returns a list of filenames."""
  if mode == 'train':
    return [os.path.join(data_dir, 'train')]
  else:
    return [os.path.join(data_dir, 'test')]

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
      data_fileds = {
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

  dataset = dataset.map(decode_record, num_threads=4)
  dataset = dataset.map(cast_int64_to_int32, num_threads=4)

  dataset = dataset.prefetch(2 * batch_size)
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  batched_examples = iterator.get_next()

  inputs = batched_examples['inputs']
  targets = batched_examples['targets']
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
  inputs = features['inputs']

  predict_value = model.encoder(inputs, params, mode == tf.estimator.ModeKeys.TRAIN)

  predictions = {
      'predict_value': predict_value,
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  mse = tf.losses.mean_squared_error(labels=labels, predictions=predict_value)
  tf.identity(mse, name='mean_squared_error')
  tf.summary.scalar('mean_squared_error', mse)

  # Add weight decay to the loss.
  loss = mse + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

  _log_variable_sizes(tf.trainable_variables(), "Trainable Variables")

  if mode == tf.estimator.ModeKeys.EVAL:
    mse = tf.metrics.mean_squared_error(labels, predict_value)
    #TODO: pearson
    metrics = {'mean_squared_error': mse}

    return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      eval_metric_ops=metrics)

  assert mode == tf.estimator.ModeKeys.TRAIN:

  global_step = tf.train.get_or_create_global_step()

  if params['lr_schedule'] == 'decay':
    batches_per_epoch = num_images / params['batch_size']
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 200, 300]]
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
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op)
  
def get_params():
  params = vars(FLAGS)

  if FLAGS.restore:
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)

  return params 


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if FLAGS.mode == 'train':
    params = get_params(FLAGS.random_sample)

    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
      json.dump(params, f)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    estimator = tf.estimator.Estimator(
      model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params=params)
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
      tensors_to_log = {
          'learning_rate': 'learning_rate',
          'mean_squared_error': 'mean_squared_error'
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      estimator.train(
          input_fn=lambda: input_fn(
              'train', FLAGS.data_dir, params['batch_size'], params['epochs_per_eval']),
          hooks=[logging_hook])
      
      # Evaluate the model and print results
      eval_results = estimator.evaluate(
          input_fn=lambda: input_fn('test', FLAGS.data_dir, params['batch_size']))
      tf.logging.info('Evaluation on test data set')
      print(eval_results)

  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
  
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    eval_results = cifar_classifier.evaluate(
          input_fn=lambda: input_fn('test', FLAGS.data_dir, params['batch_size']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)

  elif FLAGS.mode == 'predict':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)

    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    eval_results = cifar_classifier.predict(
          input_fn=lambda: input_fn('test', FLAGS.data_dir, params['batch_size']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)
    


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
