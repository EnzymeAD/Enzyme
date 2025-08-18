load("@bazel_skylib//rules:expand_template.bzl", "expand_template")

# generate compilation commands for enzymemlir-opt
load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")
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
    visibility = ["//visibility:public"],
    deps = [
        ":enzyme-tblgen-hdrs",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//llvm:config",
    ],
)

td_library(
    name = "BlasDerivativesTdFiles",
    srcs = ["Enzyme/BlasDerivatives.td"],
)

gentbl_cc_library(
    name = "call-derivatives",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-call-derivatives"],
        "Enzyme/CallDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "inst-derivatives",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-inst-derivatives"],
        "Enzyme/InstructionDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "intr-derivatives",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-intr-derivatives"],
        "Enzyme/IntrinsicDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "binop-derivatives",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-binop-derivatives"],
        "Enzyme/BinopDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "blas-derivatives",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-blas-derivatives"],
        "Enzyme/BlasDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "blas-attributor",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-update-blas-declarations"],
        "Enzyme/BlasAttributor.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "blas-typeanalysis",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-blas-typeanalysis"],
        "Enzyme/BlasTA.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
)

gentbl_cc_library(
    name = "blas-diffuseanalysis",
    strip_include_prefix = "Enzyme",
    tbl_outs = [(
        ["-gen-blas-diffuseanalysis"],
        "Enzyme/BlasDiffUse.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/InstructionDerivatives.td",
    deps = [":BlasDerivativesTdFiles"],
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
    ]) + ["Enzyme/Clang/bundled_includes.h"],
    copts = [
        "-DENZYME_RUNPASS=1",
        "-DENZYME_VERSION_MAJOR=0",
        "-DENZYME_VERSION_MINOR=0",
        "-DENZYME_VERSION_PATCH=79",
        "-Wno-unused-variable",
        "-Wno-return-type",
    ],
    data = ["@llvm-project//clang:builtin_headers_gen"],
    includes = [
        "Enzyme/Clang",  # for bundled_includes.h
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":binop-derivatives",
        ":blas-attributor",
        ":blas-derivatives",
        ":blas-diffuseanalysis",
        ":blas-typeanalysis",
        ":bundled-includes",
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
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:config",
    ],
    alwayslink = 1,
)

# Modified from llvm_driver_cc_binary since that can't be used
# outside of llvm tree
expand_template(
    name = "_gen_enzyme-clang",
    out = "enzyme-clang-driver.cpp",
    substitutions = {"@TOOL_NAME@": "clang"},
    template = "@llvm-project//llvm:cmake/modules/llvm-driver-template.cpp.in",
)

cc_binary(
    name = "enzyme-clang",
    srcs = ["enzyme-clang-driver.cpp"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-error=frame-larger-than=",
        "-Denzyme_clang_main=clang_main",
    ],
    deps = [
        ":EnzymeStatic",
        "@llvm-project//clang:clang-driver",
        "@llvm-project//llvm:Support",
    ],
)

genrule(
    name = "bundled-includes",
    srcs = glob(["include/**"]) + ["scripts/bundle-includes.sh"],
    outs = ["Enzyme/Clang/bundled_includes.h"],
    cmd = "$(location :scripts/bundle-includes.sh) $(location :include/enzyme/enzyme) $@",
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
    includes = ["."],
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FunctionInterfacesTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:MemorySlotInterfacesTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@llvm-project//mlir:ViewLikeInterfaceTdFiles",
    ],
)

gentbl_cc_library(
    name = "EnzymeOpsIncGen",
    strip_include_prefix = "Enzyme/MLIR",
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
    strip_include_prefix = "Enzyme/MLIR",
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
    strip_include_prefix = "Enzyme/MLIR",
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
    strip_include_prefix = "Enzyme/MLIR",
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
    name = "EnzymeAttributeInterfacesIncGen",
    tbl_outs = [
        (
            ["-gen-attr-interface-decls"],
            "Enzyme/MLIR/Dialect/EnzymeAttributeInterfaces.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "Enzyme/MLIR/Dialect/EnzymeOps.td",
    deps = [":EnzymeDialectTdFiles"],
)

gentbl_cc_library(
    name = "EnzymeAttributesIncGen",
    strip_include_prefix = "Enzyme/MLIR",
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
    strip_include_prefix = "Enzyme",
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
    strip_include_prefix = "Enzyme",
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

td_library(
    name = "ImplementationsCommonTdFiles",
    srcs = [
        "Enzyme/MLIR/Implementations/Common.td",
    ],
)

gentbl_cc_library(
    name = "affine-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/AffineDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/AffineDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "linalg-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/LinalgDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/LinalgDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "arith-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/ArithDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/ArithDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "complex-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/ComplexDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/ComplexDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "llvm-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/LLVMDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/LLVMDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "nvvm-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/NVVMDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/NVVMDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "scf-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/SCFDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/SCFDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "cf-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/CFDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/CFDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "memref-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/MemRefDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/MemRefDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "math-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/MathDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/MathDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "func-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/FuncDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/FuncDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
)

gentbl_cc_library(
    name = "enzyme-derivatives",
    strip_include_prefix = "Enzyme/MLIR",
    tbl_outs = [(
        ["-gen-mlir-derivatives"],
        "Enzyme/MLIR/Implementations/EnzymeDerivatives.inc",
    )],
    tblgen = ":enzyme-tblgen",
    td_file = "Enzyme/MLIR/Implementations/EnzymeDerivatives.td",
    deps = [":ImplementationsCommonTdFiles"],
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
        ":EnzymeAttributeInterfacesIncGen",
        ":EnzymeAttributesIncGen",
        ":EnzymeEnumsIncGen",
        ":EnzymeOpInterfacesIncGen",
        ":EnzymeOpsIncGen",
        ":EnzymePassesIncGen",
        ":EnzymeTypeInterfacesIncGen",
        ":EnzymeTypesIncGen",
        ":affine-derivatives",
        ":linalg-derivatives",
        ":arith-derivatives",
        ":cf-derivatives",
        ":complex-derivatives",
        ":enzyme-derivatives",
        ":func-derivatives",
        ":llvm-derivatives",
        ":math-derivatives",
        ":memref-derivatives",
        ":nvvm-derivatives",
        ":scf-derivatives",
        "@com_google_absl//absl/status:statusor",
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
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:CastInterfaces",
        "@llvm-project//mlir:ComplexDialect",
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
        "@llvm-project//mlir:LinalgInterfaces",
        "@llvm-project//mlir:LinalgStructuredOpsIncGen",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemorySlotInterfaces",
        "@llvm-project//mlir:NVVMDialect",
        "@llvm-project//mlir:OpenMPDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
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
        "@llvm-project//mlir:AffineTransforms",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsyncDialect",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:DLTIDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
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
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:Transforms",
    ],
)

exports_files(["run_lit.sh"])

refresh_compile_commands(
    name = "refresh_compile_commands",
    targets = [":enzymemlir-opt"],
)
