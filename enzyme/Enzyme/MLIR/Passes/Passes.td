//===- Passes.td - Enzyme pass tablegen macros  ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_PASSES
#define ENZYME_PASSES

include "mlir/Pass/PassBase.td"

def DifferentiatePass : Pass<"enzyme"> {
  let summary = "Differentiate Passes";
  let dependentDialects = [
    "arith::ArithDialect",
    "complex::ComplexDialect",
    "cf::ControlFlowDialect"
  ];
  let constructor = "mlir::enzyme::createDifferentiatePass()";
}

def DifferentiateWrapperPass : Pass<"enzyme-wrap"> {
  let summary = "Add wrapper function to be differentiated";
  let dependentDialects = [
    "arith::ArithDialect",
    "complex::ComplexDialect",
    "cf::ControlFlowDialect",
    "enzyme::EnzymeDialect"
  ];
  let constructor = "mlir::enzyme::createDifferentiateWrapperPass()";
  let options = [
    Option<
      /*C++ variable name=*/"infn",
      /*CLI argument=*/"infn",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"Input function to differentiate"
    >,
    Option<
      /*C++ variable name=*/"outfn",
      /*CLI argument=*/"outfn",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"Output function to differentiate"
    >,
    Option<
      /*C++ variable name=*/"mode",
      /*CLI argument=*/"mode",
      /*type=*/"DerivativeMode",
      /*default=*/"DerivativeMode::ForwardMode",
      /*description=*/"mode to differentiate",
[{::llvm::cl::values(
	     clEnumValN(DerivativeMode::ForwardMode, "ForwardMode", "ForwardMode (default)"),
	     clEnumValN(DerivativeMode::ReverseModeCombined, "ReverseModeCombined", "Combined ReverseMode"),
	     clEnumValN(DerivativeMode::ReverseModePrimal, "ReverseModePrimal", "Forward Pass of ReverseMode"),
	     clEnumValN(DerivativeMode::ReverseModeGradient, "ReverseModeGradient", "Backward Pass of ReverseMode")
	   )}]
    >,
    Option<
      /*C++ variable name=*/"retTys",
      /*CLI argument=*/"retTys",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"The activity of the returns"
    >,
    Option<
      /*C++ variable name=*/"argTys",
      /*CLI argument=*/"argTys",
      /*type=*/"std::string",
      /*default=*/"",
      /*description=*/"The activity of the arguments"
    >,
  ];
}

def PrintActivityAnalysisPass : Pass<"print-activity-analysis"> {
  let summary = "Print the results of activity analysis";
  let constructor = "mlir::enzyme::createPrintActivityAnalysisPass()";
  let options = [
    ListOption<
      /*C++ variable name=*/"funcsToAnalyze",
      /*CLI argument=*/"funcs",
      /*type=*/"std::string",
      /*description=*/"The names of the functions to perform activity analysis on"
    >,
    Option<
      /*C++ variable name=*/"verbose",
      /*CLI argument=*/"verbose",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Print detailed activity information"
    >,
    Option<
      /*C++ variable name=*/"annotate",
      /*CLI argument=*/"annotate",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Annotate every operation and value with its activity"
    >,
    Option<
      /*C++ variable name=*/"dataflow",
      /*CLI argument=*/"dataflow",
      /*type=*/"bool",
      /*default=*/"true",
      /*description=*/"Whether to use the new Dataflow activity analysis"
    >,
    Option<
      /*C++ variable name=*/"inactiveArgs",
      /*CLI argument=*/"inactive-args",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether all arguments are assumed to be inactive"
    >,
    Option<
      /*C++ variable name=*/"duplicatedRet",
      /*CLI argument=*/"duplicated-ret",
      /*type=*/"bool",
      /*default=*/"false",
      /*description=*/"Whether the return is duplicated"
    >,
  ];
}

def PrintAliasAnalysisPass : Pass<"test-print-alias-analysis"> {
  let summary = "Print the results of alias analysis";
  let constructor = "mlir::enzyme::createPrintAliasAnalysisPass()";
}

def EnzymeOpsToMemRefPass : Pass<"convert-enzyme-to-memref"> {
  let summary = "Lower custom Enzyme ops to the MemRef dialect";
  let dependentDialects = [
    "memref::MemRefDialect",
    "arith::ArithDialect",
    "scf::SCFDialect",
    "cf::ControlFlowDialect",
  ];
  let constructor = "mlir::enzyme::createEnzymeToMemRefPass()";
}

def MathematicSimplificationPass : Pass<"enzyme-simplify-math"> {
  let summary = "Simplify basic mathematical operations";
  let constructor = "mlir::enzyme::createMathematicSimplificationPass()";
}

def AddToOpToIndexAndLoadPass : Pass<"add-to-op-to-index-and-load"> {
  let summary = "Convert AddTo op inside linalg generic to index and memref load/store DEPRECATED";
  let constructor = "mlir::enzyme::createAddToOpToIndexAndLoadPass()";
}

def AddToOpToSplitPass : Pass<"add-to-op-to-split"> {
  let summary = "Duplicate linalg.generic once for each output to remove enzyme.addTo op";
  let constructor = "mlir::enzyme::createAddToOpToSplitPass()";
}

def RemoveUnusedEnzymeOpsPass : Pass<"remove-unnecessary-enzyme-ops"> {
  let summary = "Remove Unnecessary Enzyme Ops";
  let constructor = "mlir::enzyme::createRemoveUnusedEnzymeOpsPass()";
}

def SimplifyMemrefCachePass : Pass<"simplify-memref-cache"> {
  let summary = "Rewrite all cache<memref<cache<...>>> to cache<memref<...>> if possible";
  let constructor = "mlir::enzyme::createSimplifyMemrefCachePass()";
}

#endif // ENZYME_PASSES
