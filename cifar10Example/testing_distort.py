from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

with tf.device('/cpu:0'):
      images, labels = cifar10.distorted_inputs()
      print("Control returned to test")
      print (type(images))
      print (type(labels))