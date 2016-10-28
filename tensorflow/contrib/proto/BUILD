# Ops that decode and encode protobuf data

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_copts",
    "tf_custom_op_library",
    "tf_py_test",
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
    "tf_proto_library_cc",
    "tf_proto_library_py",
)

filegroup(
    name = "test_data",
    srcs = glob(["testdata/**"]),
)

tf_proto_library(
    name = "protos",
    srcs = glob(
        ["proto/*.proto"],
    ),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    visibility = ["//visibility:private"],
)

cc_library(
    name = "decode_caffe_datum_op_cc",
    hdrs = ["decode_proto_op.h"],
    srcs = ["decode_caffe_datum_op.cc"],
    copts = tf_copts(),
    linkstatic = 1,
    visibility = ["//visibility:private"],
    deps = [
        ":protos_cc",
        "//tensorflow/core:framework_headers_lib",
        "//third_party/eigen3",
    ],
    alwayslink = 1,
)

tf_custom_op_library(
    name = "proto.so",
    deps = [
        ":decode_caffe_datum_op_cc",
    ],
)

cc_library(
    name = "proto_op_lib",
    deps = [
        ":decode_caffe_datum_op_cc",
    ],
)

tf_gen_op_wrapper_py(
    name = "decode_caffe_datum_op_py",
    require_shape_functions = True,
    visibility = ["//visibility:private"],
    deps = [
        ":decode_caffe_datum_op_cc",
    ],
)

tf_py_test(
    name = "decode_caffe_datum_op_test",
    srcs = ["decode_caffe_datum_op_test.py"],
    additional_deps = [
        ":proto_ops_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ],
    data = [
        ":test_data",
    ],
    tags = ["manual"],
)

py_library(
    name = "proto_ops_py",
    srcs = [
        "__init__.py",
        "proto_ops.py",
    ],
    data = [":proto.so"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":decode_caffe_datum_op_py",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
