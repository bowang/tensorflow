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
"""Encoding and decoding protobuf datum."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.proto.ops import gen_decode_caffe_datum_op_py
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import errors
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging


ops.RegisterShape('DecodeCaffeDatum')(common_shapes.call_cpp_shape_fn)


def decode_caffe_datum(contents):
  """Create an op that decodes the contents of a Caffe datum.

  Args:
    contents: The binary contents of the Caffe datum to decode.

  Returns:
    The decoded data in the first tensor and the label in the second tensor.
  """
  return gen_decode_caffe_datum_op_py.decode_caffe_datum(contents)


ops.NotDifferentiable('DecodeCaffeDatum')


def _load_library(name, op_list=None):
  """Loads a .so file containing the specified operators.

  Args:
    name: The name of the .so file to load.
    op_list: A list of names of operators that the library should have. If None
        then the .so file's contents will not be verified.

  Raises:
    NameError if one of the required ops is missing.
  """
  try:
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
      for lib_op in library.OP_LIST.op:
        if lib_op.name == expected_op:
          break
      else:
        raise NameError('Could not find operator %s in dynamic library %s' %
                        (expected_op, name))
  except errors.NotFoundError:
    logging.warning('%s file could not be loaded.', name)


if os.name != 'nt':
  _load_library('proto.so', ['DecodeCaffeDatum'])
