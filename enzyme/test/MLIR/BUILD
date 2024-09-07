# MLIR-specific tests for Enzyme.

load("@llvm-project//llvm:lit_test.bzl", "lit_test")

[
    lit_test(
        name = "%s.test" % src,
        driver = "//test:run_lit",
        srcs = [src],
        data = [
            "//:enzymemlir-opt",
            "//test:lit.cfg.py",
            "//test:lit.site.cfg.py",
            "@llvm-project//clang:builtin_headers_gen",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:lli",
            "@llvm-project//llvm:not",
        ] + glob(["**/*.h"]),
    )
    for src in glob(
        [
            "**/*.mlir",
        ],
    )
]
