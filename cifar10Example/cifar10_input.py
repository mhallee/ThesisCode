# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import pandas

# Process images of this size. Note that this differs from the original
# image size of 512x512. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 250

# Global constants describing the data set.
NUM_CLASSES = 2
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
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

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  print("Data_dir")
  print(data_dir)
  df = pandas.read_csv(os.path.join(data_dir, 
            'training.csv'),index_col=0)
  filenames = tf.constant(df.as_matrix(["File"]))
  labels = tf.constant(df.as_matrix(["Labels"]))


  with tf.name_scope('data_augmentation'):

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #read images
    dataset = dataset.map(_parse_function)



    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
# Function copied from https://www.tensorflow.org/guide/datasets
def _parse_function(filename, label):
  print("I'm PARSING")
  print(filename)
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  tf.summary.image
  #if re-implimenting, change the return variable, as well
  #image_resized = tf.image.resize_images(image_decoded, [28, 28])
  
  return image_decoded, label


def inputs(eval_data, data_dir, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  print(data_dir)
  if not eval_data:
    df = pandas.read_csv(os.path.join(data_dir, 
            'training.csv'),index_col=0)
    filenames = tf.constant(df.as_matrix(["File"]).flatten())
    labels = tf.constant(df.as_matrix(["Labels"]).flatten())
  else:
    df = pandas.read_csv(os.path.join(data_dir, 
            'development.csv'),index_col=0)
    filenames = tf.constant(df['File'])
    labels = tf.constant(df['Label'])

  
  with tf.name_scope('input'):
    #create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    #read images
    dataset = dataset.map(_parse_function)

    print("mapped!!!")
    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(resized_image)
    #float_image.set_shape([height, width, 3])

    # Set the shapes of tensors.
    #labels.set_shape([1])

  # Generate a batch of images and labels by building up a queue of examples.
  return dataset.batch(batch_size), labels
