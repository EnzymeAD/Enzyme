
licenses(["notice"])

load("@llvm-project//llvm:tblgen.bzl", "gentbl")

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_binary(
    name="enzyme-tblgen",
    srcs=["tools/enzyme-tblgen/enzyme-tblgen.cpp"]
    hdrs = glob(["tools/enzyme-tblgen/*.h"]),
)
gentbl(
    name = "derivatives",
    tbl_outs = [(
        "-gen-derivatives",
        "InstructionDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/targets/InstructionDerivatives.td",
    deps = [
        "@llvm-project//llvm:TableGen",
    ]
)





cc_library(
    name = "Enzyme",
    srcs = glob(["Enzyme/*.cpp", "Enzyme/TypeAnalysis/*.cpp", "Enzyme/Clang/EnzymeClang.cpp"]),
    hdrs = glob(["Enzyme/*.h", "Enzyme/TypeAnalysis/*.h"]),
    data = ["@llvm-project//clang:builtin_headers_gen"],
    copts = ["-DENZYME_RUNPASS=1"]
    deps = [
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:IRReader",
        ":derivatives"
    ],
)