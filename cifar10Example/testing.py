from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

with tf.device('/cpu:0'):
      images, labels = cifar10.inputs(eval_data=False)
      print("Control returned to test")
      print (type(images))
      print (type(labels))
      with tf.Session() as sess:
      	it = images.make_one_shot_iterator().get_next()
      	print(sess.run(it))
      	tf.summary.image('input', it[0], 3)
      	merged = tf.summary.merge_all()
      	writer = tf.summary.FileWriter('\\tb', sess.graph)
      	writer.add_summary(merged)

