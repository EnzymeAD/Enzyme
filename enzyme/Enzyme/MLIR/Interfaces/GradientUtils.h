//===- GradientUtils.h - Utilities for gradient interfaces -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include "Interfaces/CloneFunction.h"
#include "Interfaces/EnzymeLogic.h"

#include "Analysis/ActivityAnalysis.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace enzyme {

class MGradientUtils {
public:
  // From CacheUtility
  FunctionOpInterface newFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  FunctionOpInterface oldFunc;
  IRMapping invertedPointers;
  IRMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  SmallPtrSet<Block *, 4> blocksNotForAnalysis;
  std::unique_ptr<enzyme::ActivityAnalyzer> activityAnalyzer;

  MTypeAnalysis &TA;
  MTypeResults TR;
  bool omp;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 MTypeResults TR_, IRMapping &invertedPointers_,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 DIFFE_TYPE ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                 IRMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeMode mode, unsigned width, bool omp);
  void erase(Operation *op) { op->erase(); }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantValue(mlir::Value v) const;
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &Builder2);
  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM);
  void forceAugmentedReturns();

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);

  LogicalResult visitChild(Operation *op);
};

class MDiffeGradientUtils : public MGradientUtils {
public:
  MDiffeGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, IRMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      IRMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {}

  // Technically diffe constructor
  static MDiffeGradientUtils *
  CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, mlir::Type additionalArg, bool omp) {
    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      prefix = "fwddiffe";
      break;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      prefix = "diffe";
      break;
    case DerivativeMode::ReverseModePrimal:
      llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
    }

    if (width > 1)
      prefix += std::to_string(width);

    IRMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    IRMapping invertedPointers;
    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);
    MTypeResults TR; // TODO
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

}; // namespace enzyme
}; // namespace mlir
