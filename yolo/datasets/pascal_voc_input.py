"""Routine for decoding the pascal-voc2007 binary file format"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf 

IMAGE_SIZE = 448

NUM_CLASSES = 20

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 9000

def read_pascal_voc(filenames_queue):
  """Reads and parses examples from pascal voc data files

  Args:
    filename_queue: A queue of strings with the filenames to read from

  Returns:
    An object representing a sing example with the filenames to read from
      height: number of rows in the result (448)
      width : number of columns in the result (448)
      depth : number of color channels in the result (3)
      labels : A list of (class_label, xmin, ymin, xmax, ymax)
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class PASCALRECORD(object):
    pass 
  result = PASCALRECORD()

  label_bytes = 20 * 5 * 4
  result.height = 448
  result.width = 448
  result.depth = 3

  image_bytes = result.height * result.width * result.depth

  record_bytes = label_bytes + image_bytes

  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(filenames_queue)

  record_bytes1 = tf.decode_raw(value, tf.uint8)

  image = tf.slice(record_bytes1, [0], [image_bytes])
  image = tf.reshape(image, [result.height, result.width, result.depth])
  result.uint8image = image 

  record_bytes2 = tf.decode_raw(value, tf.int32)

  labels = tf.slice(record_bytes2, [image_bytes//4], [label_bytes//4])

  result.labels = labels

  return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain 
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size
    labels: Labels. 2D tensor of [batch_size, 20 * 5] size
  """
  
  num_preprocess_threads = 10
  
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size,
      min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity=min_queue_examples + 3 * batch_size)

  tf.image_summary('images', images)

  return images, label_batch 

def inputs(data_dir, batch_size):
  """Construct input for pascal-voc2007 evaluation using
  the Reader ops.

  Args:
    data_dir: Path to the pascal-voc2007 data directory
    batch_size: Number of images per batch

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
    labels: Labels. 2D tensor of [batch_size, 20 * 5]
  """
  filenames = [ os.path.join(data_dir, file) for file in os.listdir(data_dir)]

  num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

  # Create a queue that produces the filenames to read.
  filenames_queue = tf.train.string_input_producer(filenames)

  read_input = read_pascal_voc(filenames_queue)

  float_image = tf.cast(read_input.uint8image, tf.float32)

  height = IMAGE_SIZE
  width = IMAGE_SIZE

  min_fraction_of_examples_in_queue = 0.3
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.labels,
                                         min_queue_examples, batch_size,
                                         shuffle=True)


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for pascal-voc2007 evaluation using
  the Reader ops.

  Args:
    data_dir: Path to the pascal-voc2007 data directory
    batch_size: Number of images per batch

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size
    labels: Labels. 2D tensor of [batch_size, 20 * 5]
  UNDO
  """ 
  return inputs(data_dir, batch_size)

