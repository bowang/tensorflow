# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Tests for tensorflow.contrib.proto.decode_caffe_datum_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glog as log
import numpy
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib import proto
from tensorflow.python.platform import resource_loader


class DecodeCaffeDatumOpTest(tf.test.TestCase):

  def testDecodeCaffeDatumFromLmdb(self):
    reader = tf.LMDBRecordReader()
    path = os.path.join(resource_loader.get_data_files_path(),
                        'testdata', "tiny_imagenet_lmdb")
    filename_queue = tf.train.string_input_producer([path])
    key, records = reader.read(filename_queue)

    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)

      for i in range(9):
        values = sess.run([records])
        data, label = proto.decode_caffe_datum(values[0])
        data_eval = data.eval()
        img_data = numpy.array(bytearray(data_eval))\
            .reshape(data_eval.shape[2], data_eval.shape[0], data_eval.shape[1])
        # Cannot show images in the test mode
        # plt.imshow(img_data.transpose([1, 2, 0]))
        # plt.show()
        self.assertEqual(label.eval()[0][0], i + 1)

      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  tf.test.main()
