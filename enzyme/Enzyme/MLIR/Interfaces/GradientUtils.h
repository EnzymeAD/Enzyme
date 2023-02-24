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

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/FunctionInterfaces.h"


namespace mlir {
namespace enzyme {

class MGradientUtils {
public:
  // From CacheUtility
  FunctionOpInterface newFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeModeMLIR mode;
  FunctionOpInterface oldFunc;
  BlockAndValueMapping invertedPointers;
  BlockAndValueMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  MTypeAnalysis &TA;
  bool omp;

  unsigned width;
  ArrayRef<DIFFE_TYPE_MLIR> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 BlockAndValueMapping &invertedPointers_,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 DIFFE_TYPE_MLIR ReturnActivity, ArrayRef<DIFFE_TYPE_MLIR> ArgDiffeTypes_,
                 BlockAndValueMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeModeMLIR mode, unsigned width, bool omp);
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
                      BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE_MLIR ActiveReturn,
                      ArrayRef<DIFFE_TYPE_MLIR> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeModeMLIR mode, unsigned width, bool omp)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {}

  // Technically diffe constructor
  static MDiffeGradientUtils *
  CreateFromClone(MEnzymeLogic &Logic, DerivativeModeMLIR mode, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE_MLIR retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE_MLIR> constant_args,
                  ReturnTypeMLIR returnValue, mlir::Type additionalArg, bool omp) {
    std::string prefix;

    switch (mode) {
    case DerivativeModeMLIR::ForwardMode:
    case DerivativeModeMLIR::ForwardModeSplit:
      prefix = "fwddiffe";
      break;
    case DerivativeModeMLIR::ReverseModeCombined:
    case DerivativeModeMLIR::ReverseModeGradient:
      prefix = "diffe";
      break;
    case DerivativeModeMLIR::ReverseModePrimal:
      llvm_unreachable("invalid DerivativeModeMLIR: ReverseModePrimal\n");
    }

    if (width > 1)
      prefix += std::to_string(width);

    BlockAndValueMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    BlockAndValueMapping invertedPointers;
    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

}; // namespace enzyme
}; // namespace mlir
