from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')


def to_example(dictionary):
  """Helper: build tf.Example from (string -> int/float/str list) dictionary."""
  features = {}
  for (k, v) in six.iteritems(dictionary):
    if not v:
      raise ValueError("Empty generated field: %s", str((k, v)))
    if isinstance(v[0], six.integer_types):
      features[k] = tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    elif isinstance(v[0], float):
      features[k] = tf.train.Feature(float_list=tf.train.FloatList(value=v))
    elif isinstance(v[0], six.string_types):
      if not six.PY2:  # Convert in python 3.
        v = [bytes(x, "utf-8") for x in v]
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    elif isinstance(v[0], bytes):
      features[k] = tf.train.Feature(bytes_list=tf.train.BytesList(value=v))
    else:
      raise ValueError("Value for %s is not a recognized type; v: %s type: %s" %
                       (k, str(v[0]), str(type(v[0]))))
  return tf.train.Example(features=tf.train.Features(feature=features))


class DataReader():
  def __init__(self):
    pass

  def encode_source(self, sequence):
    sequence = sequence.strip().split(' ')
    sequence_ints = [int(i) for i in sequence]
    return sequence_ints

  def encode_target(self, sequence):
    sequence = sequence.split(' ')
    return float(sequence[0])

  def geretator(self, source_path, target_path, data_dir):
    tf.logging.info('Generating data')
    source_path = os.path.join(data_dir, source_path)
    target_path = os.path.join(data_dir, target_path)
    with tf.gfile.GFile(source_path, mode='r') as source_file:
      with tf.gfile.GFile(target_path, mode='r') as target_file:
        source = source_file.readline()
        target = target_file.readline()
        while source and target:
          soure_ints = self.encode_source(source)
          target_floats = self.encode_target(target)
          yield {"inputs": soure_ints, "targets": target_floats}
          source = source_file.readline()
          target = target_file.readline()

  def generate_files(self, generator, output_filename, data_dir):
    writer = tf.python_io.TFRecordWriter(os.path.join(data_dir, output_filename))
    counter, shard = 0, 0
    for case in generator:
      if counter > 0 and counter % 100 == 0:
        tf.logging.info("Generating case %d." % counter)
      counter += 1
      sequence_example = to_example(case)
      writer.write(sequence_example.SerializeToString())
    writer.close()

  def generate_data(self, data_dir):
    self.generate_files(self.geretator('train.input', 'train.target', data_dir), 'train.tfrecords', data_dir)
    self.generate_files(self.geretator('test.input', 'test.target', data_dir), 'test.tfrecords', data_dir)

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  data = DataReader()
  data.generate_data(FLAGS.data_dir)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)