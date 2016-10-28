/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CONTRIB_PROTO_DECODE_PROTO_OP_H_
#define TENSORFLOW_CONTRIB_PROTO_DECODE_PROTO_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace proto {

template <class Proto>
class DecodeProtoOp : public OpKernel {
 public:
  explicit DecodeProtoOp(OpKernelConstruction* context)
      : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(
        context, context->num_inputs() == 1,
        errors::InvalidArgument("DecodeProto requires exactly one input."));
    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(input.shape()),
        errors::InvalidArgument("input must be scalar but got shape ",
                                input.shape().DebugString()));

    const tensorflow::StringPiece proto_contents = input.scalar<string>()();
    Proto proto;
    proto.ParseFromString(proto_contents.ToString());

    Decode(context, proto);
  }

 protected:
  // Decode the proto into the output tensors.
  // Returns true if success, false otherwise.
  void Decode(OpKernelContext* context, Proto& proto);
};

}  // namespace proto
}  // namespace tensorflow

#endif
