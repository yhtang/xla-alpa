load("//xla:xla.bzl", "xla_cc_binary")
load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//xla/mlir:__subpackages__"],
    licenses = ["notice"],
)

build_test(
    name = "xla-gpu-opt_build_test",
    targets = [
        ":xla-gpu-opt",
    ],
)

xla_cc_binary(
    name = "xla-gpu-opt",
    srcs = ["xla-gpu-opt.cc"],
    deps = [
        "//xla/mlir/backends/gpu/transforms:passes",
        "//xla/mlir_hlo:lhlo",
        "//xla/mlir_hlo:lhlo_gpu",
        "//xla/stream_executor:stream_executor_impl",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
    ],
)
