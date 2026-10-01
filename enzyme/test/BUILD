package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

load("@llvm-project//llvm:lit_test.bzl", "lit_expand_template")

# Generates lit config input file by applying path placeholder substitutions
# similar to the configure_lit_site_cfg CMake macro.
lit_expand_template(
    name = "lit_site_cfg_py",
    testonly = True,
    out = "lit.site.cfg.py",
    dirname_substitutions = {
        "@LLVM_BINARY_DIR@": "@llvm-project//llvm:FileCheck",
        "@LLVM_TOOLS_BINARY_DIR@": "@llvm-project//llvm:FileCheck",
        "@LLVM_LIBS_DIR@": "@llvm-project//llvm:FileCheck",
    },
    substitutions = {
        "@LLVM_VERSION_MAJOR@": "18",
        "@ENZYME_SOURCE_DIR@": "",
        "@ENZYME_BINARY_DIR@": "",
        "@TARGET_TRIPLE@": "",
        "@TARGETS_TO_BUILD@": "ALL",
        "@LLVM_SHLIBEXT@": ".so",
        "@Fortran_COMPILER@": "",
    },
    template = "lit.site.cfg.py.in",
    visibility = [":__subpackages__"],
)

exports_files(
    ["lit.cfg.py"],
    visibility = [":__subpackages__"],
)
