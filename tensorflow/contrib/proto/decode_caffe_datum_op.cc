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

#include "tensorflow/contrib/proto/decode_proto_op.h"
#include "tensorflow/contrib/proto/proto/caffe_datum.pb.h"

namespace tensorflow {
namespace proto {

using shape_inference::InferenceContext;

template<>
void DecodeProtoOp<caffe::Datum>::Decode(OpKernelContext* context,
                                         caffe::Datum& datum) {
  Tensor* data_output = nullptr;
  Tensor* label_output = nullptr;

  OP_REQUIRES_OK(context, context->allocate_output(
      0, TensorShape({datum.height(), datum.width(), datum.channels()}),
      &data_output));

  OP_REQUIRES_OK(context, context->allocate_output(
      1, TensorShape({1, 1}), &label_output));

  if (datum.encoded()) {
    // TODO
    errors::Unimplemented("Encoded datum is not supported yet");
  }
  else {
    uint8* dst = data_output->flat<uint8>().data();
    size_t size_in_bytes = datum.channels() * datum.width() * datum.height();

    // Copy data into the data output tensor
    memcpy(dst, datum.data().c_str(), size_in_bytes);

    // Copy the label into the label output tensor
    *(label_output->flat<int32>().data()) = datum.label();
  }
}

template class DecodeProtoOp<caffe::Datum>;

REGISTER_KERNEL_BUILDER(Name("DecodeCaffeDatum").Device(DEVICE_CPU),
                        DecodeProtoOp<caffe::Datum>);

REGISTER_OP("DecodeCaffeDatum")
    .Input("contents: string")
    .Output("data: uint8")
    .Output("label: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape( {InferenceContext::kUnknownDim,
                InferenceContext::kUnknownDim, InferenceContext::kUnknownDim}));
      c->set_output(1, c->MakeShape( {1, 1}));
      return Status::OK();
     })
    .Doc(R"doc(
Decodes the contents of a Caffe datum into a tensor.

contents: The binary Caffe datum contents.
)doc");

}  // namespace proto
}  // namespace tensorflow
