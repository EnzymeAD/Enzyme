package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

# Execution-based numerical tests for scf.for reverse-mode checkpointing with
# mutable memory: unlike the FileCheck-only IR-shape tests in
# //test/MLIR/ReverseMode, these actually JIT-run the differentiated function
# via mlir-runner and check the printed numerical result.

load("@llvm-project//llvm:lit_test.bzl", "lit_test")

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        data = [
            "Inputs/exec_main_10xf64.mlir.inc",
            "//:enzymemlir-opt",
            "//test:lit.cfg.py",
            "//test:lit.site.cfg.py",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//mlir:libmlir_c_runner_utils.so",
            "@llvm-project//mlir:libmlir_runner_utils.so",
            "@llvm-project//mlir:mlir-opt",
            "@llvm-project//mlir:mlir-runner",
        ],
    )
    for src in glob(["*.mlir"])
]
