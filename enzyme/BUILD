load("@llvm-project//llvm:tblgen.bzl", "gentbl")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//:__subpackages__"],
)

cc_library(
    name = "enzyme-tblgen-hdrs",
    hdrs = glob(["tools/enzyme-tblgen/*.h"]),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

cc_binary(
    name = "enzyme-tblgen",
    srcs = glob(["tools/enzyme-tblgen/*.cpp"]),
    deps = [
        ":enzyme-tblgen-hdrs",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

gentbl(
    name = "call-derivatives",
    tbl_outs = [(
        "-gen-call-derivatives",
        "CallDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "inst-derivatives",
    tbl_outs = [(
        "-gen-inst-derivatives",
        "InstructionDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "intr-derivatives",
    tbl_outs = [(
        "-gen-intr-derivatives",
        "IntrinsicDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "binop-derivatives",
    tbl_outs = [(
        "-gen-binop-derivatives",
        "BinopDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "blas-derivatives",
    tbl_outs = [(
        "-gen-blas-derivatives",
        "BlasDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "blas-attributor",
    tbl_outs = [(
        "-update-blas-declarations",
        "BlasAttributor.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "blas-typeanalysis",
    tbl_outs = [(
        "-gen-blas-typeanalysis",
        "BlasTA.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "blas-diffuseanalysis",
    tbl_outs = [(
        "-gen-blas-diffuseanalysis",
        "BlasDiffUse.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    td_srcs = ["Enzyme/BlasDerivatives.td"],
    deps = [
        ":enzyme-tblgen",
    ],
)

cc_library(
    name = "EnzymeStatic",
    srcs = glob([
        "Enzyme/*.cpp",
        "Enzyme/TypeAnalysis/*.cpp",
        "Enzyme/Clang/EnzymeClang.cpp",
    ]),
    hdrs = glob([
        "Enzyme/*.h",
        "Enzyme/TypeAnalysis/*.h",
    ]),
    copts = [
        "-DENZYME_RUNPASS=1",
        "-DENZYME_VERSION_MAJOR=0",
        "-DENZYME_VERSION_MINOR=0",
        "-DENZYME_VERSION_PATCH=79",
    ],
    data = ["@llvm-project//clang:builtin_headers_gen"],
    visibility = ["//visibility:public"],
    deps = [
        ":binop-derivatives",
        ":blas-attributor",
        ":blas-derivatives",
        ":blas-diffuseanalysis",
        ":blas-typeanalysis",
        ":call-derivatives",
        ":inst-derivatives",
        ":intr-derivatives",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:sema",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:AggressiveInstCombine",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:config",
    ],
)

sh_test(
    name = "placeholder_test",
    size = "small",
    srcs = ["placeholdertest.sh"],
    deps = [
        ":EnzymeStatic",
        "//testing/shbase",
    ],
)
