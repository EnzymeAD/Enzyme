//===- GradientUtilsReverse.h - Utilities for gradient interfaces -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/FunctionInterfaces.h"

#include "EnzymeLogic.h"
#include "CloneFunction.h"

namespace mlir {
namespace enzyme {


class MGradientUtilsReverse {
public:
  // From CacheUtility
  FunctionOpInterface newFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  FunctionOpInterface oldFunc;
  BlockAndValueMapping invertedPointers;
  DenseMap<Value, SmallVector<mlir::Value, 4>> invertedPointersReverse;
  BlockAndValueMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  MTypeAnalysis &TA;
  MTypeResults TR;
  bool omp;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtilsReverse(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 MTypeResults TR_, BlockAndValueMapping &invertedPointers_,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 DIFFE_TYPE ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                 BlockAndValueMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeMode mode, unsigned width, bool omp);
  void erase(Operation *op) { op->erase(); }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantValue(mlir::Value v) const;
  bool hasInvertPointer(mlir::Value v);
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &Builder2);
  mlir::Value invertPointerReverseM(Value v, Block * askingOp);
  Optional<mlir::Value> invertPointerReverseMOptional(Value v, Block * askingOp);
  void mapInvertPointer(mlir::Value v, mlir::Value invertValue);
  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM);
  void forceAugmentedReturns();
  void forceAugmentedReturnsReverse();

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);
  
  LogicalResult visitChildReverse(Operation *op, OpBuilder& builder);
  LogicalResult visitChild(Operation *op);
};

class MDiffeGradientUtilsReverse : public MGradientUtilsReverse {
public:
  MDiffeGradientUtilsReverse(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp)
      : MGradientUtilsReverse(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {
  }
  static MDiffeGradientUtilsReverse * CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
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
    MTypeResults TR; // TODO
    return new MDiffeGradientUtilsReverse(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

} // namespace enzyme
} // namespace mlir
