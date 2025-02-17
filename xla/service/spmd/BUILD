# Description: SPMD partitioning pass.

load("@tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

cc_library(
    name = "spmd_partitioner",
    srcs = [
        "convolution_handler.cc",
        "custom_call_handler.cc",
        "dot_handler.cc",
        "fft_handler.cc",
        "gather_scatter_handler.cc",
        "spmd_partitioner.cc",
        "spmd_partitioner_util.cc",
    ],
    hdrs = [
        "convolution_handler.h",
        "custom_call_handler.h",
        "spmd_partitioner.h",
        "spmd_partitioner_util.h",
    ],
    deps = [
        "//xla:comparison_util",
        "//xla:literal_util",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:util",
        "//xla:window_util",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client/lib:comparators",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/ir:hlo_reachability",
        "//xla/hlo/utils:hlo_query",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:call_graph",
        "//xla/service:custom_call_sharding_helper",
        "//xla/service:dot_as_convolution_util",
        "//xla/service:flatten_call_graph",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_lexer",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:pattern_matcher",
        "//xla/service:shape_inference",
        "//xla/service:sharding_propagation",
        "//xla/service:tuple_simplifier",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@tsl//tsl/platform:numbers",
        "@tsl//tsl/platform:statusor",
    ],
)

xla_cc_test(
    name = "spmd_partitioner_test",
    srcs = ["spmd_partitioner_test.cc"],
    deps = [
        ":spmd_partitioner",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:sharding_propagation",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@com_google_absl//absl/algorithm:container",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

xla_cc_test(
    name = "canonicalize_all_gather_for_cse_test",
    srcs = ["canonicalize_all_gather_for_cse_test.cc"],
    deps = [
        ":canonicalize_all_gather_for_cse",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "canonicalize_all_gather_for_cse",
    srcs = ["canonicalize_all_gather_for_cse.cc"],
    hdrs = ["canonicalize_all_gather_for_cse.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_query",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/types:span",
    ],
)

xla_cc_test(
    name = "schedule_aware_collective_ops_cse_test",
    srcs = ["schedule_aware_collective_ops_cse_test.cc"],
    deps = [
        ":schedule_aware_collective_ops_cse",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "schedule_aware_collective_ops_cse",
    srcs = ["schedule_aware_collective_ops_cse.cc"],
    hdrs = ["schedule_aware_collective_ops_cse.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/container:flat_hash_map",
        "@tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "stateful_rng_spmd_partitioner",
    srcs = ["stateful_rng_spmd_partitioner.cc"],
    hdrs = ["stateful_rng_spmd_partitioner.h"],
    deps = [
        ":spmd_partitioner",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@com_google_absl//absl/memory",
        # Added by Alpa
        ":auto_sharding",
    ],
)

xla_cc_test(
    name = "stateful_rng_spmd_partitioner_test",
    srcs = ["stateful_rng_spmd_partitioner_test.cc"],
    deps = [
        ":stateful_rng_spmd_partitioner",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:rng_expander",
        "//xla/service:sharding_propagation",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

cc_library(
    name = "collective_permute_motion",
    srcs = ["collective_permute_motion.cc"],
    hdrs = ["collective_permute_motion.h"],
    deps = [
        "//xla:comparison_util",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:while_loop_analysis",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

xla_cc_test(
    name = "collective_permute_motion_test",
    srcs = ["collective_permute_motion_test.cc"],
    deps = [
        ":collective_permute_motion",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/utils:hlo_matchers",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
    ],
)

cc_library(
    name = "partition_assignment",
    srcs = [
        "partition_assignment.cc",
    ],
    hdrs = [
        "partition_assignment.h",
    ],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
    ],
)

xla_cc_test(
    name = "partition_assignment_test",
    srcs = ["partition_assignment_test.cc"],
    deps = [
        ":partition_assignment",
        "//xla:xla_proto_cc",
        "//xla/tests:hlo_test_base",
        "//xla/tests:xla_internal_test_main",
        "@tsl//tsl/lib/core:status_test_util",
        "@tsl//tsl/platform:test",
    ],
)

# Added by Alpa
cc_library(
    name = "auto_sharding",
    srcs = ["auto_sharding.cc", "auto_sharding_util.cc", "auto_sharding_dot_handler.cc"],
    hdrs = ["auto_sharding.h", "auto_sharding_strategy.h", "auto_sharding_util.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/utils:hlo_live_range",
        "//xla/hlo/utils:hlo_sharding_util",
        "//xla/service:dump",
        "//xla/service:hlo_creation_utils",
        "//xla/service:hlo_memory_scheduler",
        "//xla/service:hlo_pass",
        "//xla/service:pass_context",
        "@pybind11",
    ],
)

cc_library(
    name = "redundant_slice_eliminator",
    srcs = ["redundant_slice_eliminator.cc"],
    hdrs = ["redundant_slice_eliminator.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_pass",
    ],
)

cc_library(
    name = "grad_acc_rewrite",
    srcs = ["grad_acc_rewrite.cc"],
    hdrs = ["grad_acc_rewrite.h"],
    deps = [
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "//xla/service:pass_context",
        "//xla/service/spmd:spmd_partitioner",
    ],
)

cc_library(
    name = "slice_auto_sharded_stages",
    srcs = ["slice_auto_sharded_stages.cc"],
    hdrs = ["slice_auto_sharded_stages.h"],
    deps = [
        ":auto_sharding",
        "//xla:shape_util",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_pass",
        "@pybind11",
    ],
)

cc_library(
    name = "alpa_compiler",
    srcs = ["alpa_compiler.cc"],
    hdrs = ["alpa_compiler.h"],
    deps = [
        ":auto_sharding",
        ":redundant_slice_eliminator",
        ":grad_acc_rewrite",
        ":slice_auto_sharded_stages",
        ":stateful_rng_spmd_partitioner",
        # Copied from gpu_compiler
        "//xla/hlo/transforms:hlo_constant_splitter",
        "//xla/pjrt:pjrt_client",
        "//xla/service:algebraic_simplifier",
        "//xla/service:all_reduce_reassociate",
        "//xla/service:call_inliner",
        "//xla/service:conditional_canonicalizer",
        "//xla/service:conditional_simplifier",
        "//xla/service:dot_decomposer",
        "//xla/service:dot_merger",
        "//xla/service:dump",
        "//xla/service:gather_expander",
        "//xla/service:gather_simplifier",
        "//xla/service:hlo_constant_folding",
        "//xla/service:hlo_cse",
        "//xla/service:hlo_dce",
        "//xla/service:hlo_pass",
        "//xla/service:hlo_pass_pipeline",
        "//xla/service:hlo_verifier",
        "//xla/service:reshape_mover",
        "//xla/service:scatter_expander",
        "//xla/service:scatter_simplifier",
        "//xla/service:sharding_propagation",
        "//xla/service:sharding_remover",
        "//xla/service:sort_simplifier",
        "//xla/service:transpose_folding",
        "//xla/service:tuple_simplifier",
        "//xla/service:while_loop_constant_sinking",
        "//xla/service:while_loop_simplifier",
        "//xla/service:zero_sized_hlo_elimination",
        "//xla/service/gpu:gpu_conv_rewriter",
        "//xla/service/gpu:gpu_shape_verifier",
        "//xla/service/gpu:matmul_utils",
    ]
)
# End Added by Alpa
