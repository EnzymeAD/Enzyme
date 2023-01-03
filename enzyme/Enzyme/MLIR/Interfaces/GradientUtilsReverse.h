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
  FunctionOpInterface oldFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  BlockAndValueMapping invertedPointers;
  BlockAndValueMapping invertedPointersGlobal;
  DenseMap<Value, SmallVector<mlir::Value, 4>> invertedPointersReverse;
  Block *initializationBlock;

  BlockAndValueMapping mapReverseModeBlocks;
  DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments;

  BlockAndValueMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  MTypeAnalysis &TA;
  MTypeResults TR;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtilsReverse(MEnzymeLogic &Logic, 
                FunctionOpInterface newFunc_,
                FunctionOpInterface oldFunc_, 
                MTypeAnalysis &TA_,
                MTypeResults TR_, 
                BlockAndValueMapping invertedPointers_,
                const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                const SmallPtrSetImpl<mlir::Value> &activevals_,
                DIFFE_TYPE ReturnActivity, 
                ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                BlockAndValueMapping &originalToNewFn_,
                std::map<Operation *, Operation *> &originalToNewFnOps_,
                DerivativeMode mode, 
                unsigned width, 
                BlockAndValueMapping mapReverseModeBlocks_, 
                DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments_);
  void erase(Operation *op) { op->erase(); }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantValue(mlir::Value v) const;
  bool hasInvertPointer(mlir::Value v);
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &builder);
  void mapInvertPointer(mlir::Value v, mlir::Value invertValue, OpBuilder &builder);
  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM);
  void forceAugmentedReturnsReverse();
  Value insertInitBackwardCache(Type t);
  Value insertInitGradient(mlir::Value v);
  Type getIndexCacheType();
  Type getIndexType();
  Type getCacheType(Type t);
  Type getGradientType(Value t);

  void initInitializationBlock(BlockAndValueMapping invertedPointers_);

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);
  
  LogicalResult visitChildReverse(Operation *op, OpBuilder& builder);
};

class MDiffeGradientUtilsReverse : public MGradientUtilsReverse {
public:
  MDiffeGradientUtilsReverse(MEnzymeLogic &Logic, 
                      FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, 
                      MTypeAnalysis &TA,
                      MTypeResults TR, 
                      BlockAndValueMapping invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, 
                      unsigned width, 
                      BlockAndValueMapping mapReverseModeBlocks_, 
                      DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments_);

  static MDiffeGradientUtilsReverse * CreateFromClone(MEnzymeLogic &Logic, 
                  DerivativeMode mode, 
                  unsigned width,
                  FunctionOpInterface todiff, 
                  MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, 
                  DIFFE_TYPE retType,
                  bool diffeReturnArg, 
                  ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, 
                  mlir::Type additionalArg);
};

} // namespace enzyme
} // namespace mlir
