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
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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
  llvm::StringRef postpasses;
  const llvm::ArrayRef<bool> returnPrimals;
  const llvm::ArrayRef<bool> returnShadows;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;
  ArrayRef<DIFFE_TYPE> RetDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 MTypeResults TR_, IRMapping &invertedPointers_,
                 const llvm::ArrayRef<bool> returnPrimals,
                 const llvm::ArrayRef<bool> returnShadows,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 ArrayRef<DIFFE_TYPE> ReturnActivities,
                 ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                 IRMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeMode mode, unsigned width, bool omp,
                 llvm::StringRef postpasses);
  void erase(Operation *op) { op->erase(); }
  void replaceOrigOpWith(Operation *op, ValueRange vals) {
    for (auto &&[res, rep] : llvm::zip(op->getResults(), vals)) {
      originalToNewFn.map(res, rep);
    }
    auto newOp = getNewFromOriginal(op);
    newOp->replaceAllUsesWith(vals);
    originalToNewFnOps.erase(op);
  }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantInstruction(mlir::Operation *v) const;
  bool isConstantValue(mlir::Value v) const;
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &Builder2);
  void forceAugmentedReturns();

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);

  LogicalResult visitChild(Operation *op);

  void setDiffe(mlir::Value origv, mlir::Value newv, mlir::OpBuilder &builder);

  mlir::Type getShadowType(mlir::Type T) {
    auto iface = cast<AutoDiffTypeInterface>(T);
    return iface.getShadowType(width);
  }
};

class MDiffeGradientUtils : public MGradientUtils {
protected:
  IRMapping differentials;

  Block *initializationBlock;

public:
  mlir::Value getDifferential(mlir::Value origv);

  void setDiffe(mlir::Value origv, mlir::Value newv, mlir::OpBuilder &builder);

  void zeroDiffe(mlir::Value origv, mlir::OpBuilder &builder);

  mlir::Value diffe(mlir::Value origv, mlir::OpBuilder &builder);

  MDiffeGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, IRMapping &invertedPointers_,
                      const llvm::ArrayRef<bool> returnPrimals,
                      const llvm::ArrayRef<bool> returnShadows,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &activevals_,
                      ArrayRef<DIFFE_TYPE> RetActivity,
                      ArrayRef<DIFFE_TYPE> ArgActivity, IRMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp,
                      llvm::StringRef postpasses)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       returnPrimals, returnShadows, constantvalues_,
                       activevals_, RetActivity, ArgActivity, origToNew_,
                       origToNewOps_, mode, width, omp, postpasses),
        initializationBlock(&*(newFunc.getFunctionBody().begin())) {}

  // Technically diffe constructor
  static MDiffeGradientUtils *CreateFromClone(
      MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
      FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo,
      const llvm::ArrayRef<bool> returnPrimals,
      const llvm::ArrayRef<bool> returnShadows,
      ArrayRef<DIFFE_TYPE> RetActivity, ArrayRef<DIFFE_TYPE> ArgActivity,
      mlir::Type additionalArg, bool omp, llvm::StringRef postpasses) {
    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardModeError:
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
        mode, width, todiff, invertedPointers, ArgActivity, constant_values,
        nonconstant_values, returnvals, returnPrimals, returnShadows,
        RetActivity, prefix + todiff.getName(), originalToNew, originalToNewOps,
        additionalArg);
    MTypeResults TR; // TODO
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, TR, invertedPointers, returnPrimals,
        returnShadows, constant_values, nonconstant_values, RetActivity,
        ArgActivity, originalToNew, originalToNewOps, mode, width, omp,
        postpasses);
  }
};

}; // namespace enzyme
}; // namespace mlir
