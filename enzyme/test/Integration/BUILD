# Enzyme integration tests.

load("@llvm-project//llvm:lit_test.bzl", "lit_test")

[
    lit_test(
        name = "%s.test" % src,
        srcs = [src],
        args = ["\"--FLAGS=$(CC_FLAGS)\""],
        data = [
            "//:enzyme-clang",
            "//:enzyme-clang++",
            "//:enzyme-opt",
            "//test:lit.cfg.py",
            "//test:lit.site.cfg.py",
            "@llvm-project//clang:builtin_headers_gen",
            "@llvm-project//llvm:FileCheck",
            "@llvm-project//llvm:count",
            "@llvm-project//llvm:lli",
            "@llvm-project//llvm:not",
            "//tools/cpp:cc_flags",
            "//tools/cpp:current_cc_toolchain",
        ] + glob(["**/*.h"]),
        driver = "//test:run_lit",
        toolchains = [
            "//tools/cpp:cc_flags",
            "//tools/cpp:current_cc_toolchain",
        ],
    )
    for src in glob(
        [
            "**/*.c",
            "**/*.cpp",
        ],
        exclude = [
            "**/*omp*.c",
            "**/gh_issue_1904.cpp",
            "**/CppSugar/sret.cpp",
            "**/ReverseMode/fbuff.cpp",
            "**/*blas*.cpp",
            "**/*ForwardMode/rosenbrock.cpp",
            "**/*ReverseMode/map.cpp",
            "**/*ReverseMode/multivecmax.cpp",
            "**/*ReverseMode/vecmax.cpp",
            "**/*Sparse/*",
            "**/*eigen*.cpp",
            "**/*integrate*.cpp",
        ],
    )
]
