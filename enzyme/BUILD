load("@llvm-project//llvm:tblgen.bzl", "gentbl")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
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
    srcs = glob(
        [
            "Enzyme/*.cpp",
            "Enzyme/TypeAnalysis/*.cpp",
            "Enzyme/Clang/EnzymeClang.cpp",
        ],
        exclude = ["Enzyme/eopt.cpp"],
    ),
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
    alwayslink = 1,
)

cc_binary(
    name = "enzyme-clang",
    srcs = [],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-error=frame-larger-than=",
    ],
    deps = [
        ":EnzymeStatic",
        "@llvm-project//clang:clang-driver",
    ],
)

genrule(
    name = "gen_enzyme-clang++",
    srcs = [":enzyme-clang"],
    outs = ["enzyme-clang++"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
)

cc_binary(
    name = "enzyme-opt",
    srcs = ["Enzyme/eopt.cpp"],
    deps = [
        ":EnzymeStatic",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:opt-driver",
    ],
)

td_library(
    name = "EnzymeDialectTdFiles",
    srcs = [
        "Enzyme/MLIR/Dialect/Dialect.td",
    ],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeOpsIncGen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "Enzyme/MLIR/Dialect/EnzymeOps.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "Enzyme/MLIR/Dialect/EnzymeOps.cpp.inc",
        ),
        (
            [
                "-gen-dialect-decls",
                "-dialect=enzyme",
            ],
            "Enzyme/MLIR/Dialect/EnzymeOpsDialect.h.inc",
        ),
        (
            [
                "-gen-dialect-defs",
                "-dialect=enzyme",
            ],
            "Enzyme/MLIR/Dialect/EnzymeOpsDialect.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [":EnzymeDialectTdFiles"],
)

td_library(
    name = "EnzymePassesTdFiles",
    srcs = [
    ],
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymePassesIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=enzyme",
            ],
            "Enzyme/MLIR/Passes/Passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Passes/Passes.td",
    deps = [":EnzymePassesTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeTypesIncGen",
    tbl_outs = [
        (
            ["-gen-typedef-decls"],
            "Enzyme/MLIR/Dialect/EnzymeOpsTypes.h.inc",
        ),
        (
            ["-gen-typedef-defs"],
            "Enzyme/MLIR/Dialect/EnzymeOpsTypes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeEnumsIncGen",
    tbl_outs = [
        (
            ["-gen-enum-decls"],
            "Enzyme/MLIR/Dialect/EnzymeEnums.h.inc",
        ),
        (
            ["-gen-enum-defs"],
            "Enzyme/MLIR/Dialect/EnzymeEnums.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeAttributesIncGen",
    tbl_outs = [
        (
            ["-gen-attrdef-decls"],
            "Enzyme/MLIR/Dialect/EnzymeAttributes.h.inc",
        ),
        (
            ["-gen-attrdef-defs"],
            "Enzyme/MLIR/Dialect/EnzymeAttributes.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeTypeInterfacesIncGen",
    tbl_outs = [
        (
            ["--gen-type-interface-decls"],
            "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.h.inc",
        ),
        (
            ["--gen-type-interface-defs"],
            "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Interfaces/AutoDiffTypeInterface.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeOpInterfacesIncGen",
    tbl_outs = [
        (
            ["--gen-op-interface-decls"],
            "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.h.inc",
        ),
        (
            ["--gen-op-interface-defs"],
            "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.cpp.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Interfaces/AutoDiffOpInterface.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl(
    name = "affine-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/AffineDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/AffineDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/AffineDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "arith-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/ArithDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/ArithDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/ArithDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "llvm-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/LLVMDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/LLVMDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/LLVMDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "nvvm-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/NVVMDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/NVVMDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/NVVMDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "scf-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/SCFDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/SCFDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/SCFDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "cf-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/CFDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/CFDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/CFDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "memref-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/MemRefDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/MemRefDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/MemRefDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

gentbl(
    name = "math-derivatives",
    tbl_outs = [(
        "-gen-mlir-derivatives",
        "Enzyme/MLIR/Implementations/MathDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/MathDerivatives.td",
    td_srcs = [
        "Enzyme/MLIR/Implementations/MathDerivatives.td",
        "Enzyme/MLIR/Implementations/Common.td",
    ],
    deps = [
        ":enzyme-tblgen",
    ],
)

cc_library(
    name = "EnzymeMLIR",
    srcs = glob([
        "Enzyme/MLIR/Dialect/*.cpp",
        "Enzyme/MLIR/Passes/*.cpp",
        "Enzyme/MLIR/Interfaces/*.cpp",
        "Enzyme/MLIR/Analysis/*.cpp",
        "Enzyme/MLIR/Implementations/*.cpp",
    ]),
    hdrs = glob([
        "Enzyme/MLIR/Dialect/*.h",
        "Enzyme/MLIR/Passes/*.h",
        "Enzyme/MLIR/Interfaces/*.h",
        "Enzyme/MLIR/Analysis/*.h",
        "Enzyme/MLIR/Implementations/*.h",
        "Enzyme/Utils.h",
        "Enzyme/TypeAnalysis/*.h",
    ]),
    includes = [
        "Enzyme",
        "Enzyme/MLIR",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":EnzymeAttributesIncGen",
        ":EnzymeEnumsIncGen",
        ":EnzymeOpInterfacesIncGen",
        ":EnzymeOpsIncGen",
        ":EnzymePassesIncGen",
        ":EnzymeTypeInterfacesIncGen",
        ":EnzymeTypesIncGen",
        ":affine-derivatives",
        ":arith-derivatives",
        ":cf-derivatives",
        ":llvm-derivatives",
        ":math-derivatives",
        ":memref-derivatives",
        ":nvvm-derivatives",
        ":scf-derivatives",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:config",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithUtils",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:FunctionInterfaces",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LLVMCommonConversion",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:LinalgStructuredOpsIncGen",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:ViewLikeInterface",
    ],
)

cc_binary(
    name = "enzymemlir-opt",
    srcs = ["Enzyme/MLIR/enzymemlir-opt.cpp"],
    includes = ["Enzyme/MLIR"],
    visibility = ["//visibility:public"],
    deps = [
        ":EnzymeMLIR",
        "@llvm-project//mlir:AffineDialect",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

exports_files(["run_lit.sh"])

