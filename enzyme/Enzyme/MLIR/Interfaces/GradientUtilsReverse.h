//===- GradientUtilsReverse.h - Utilities for gradient interfaces -------* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include <functional>

namespace mlir {
namespace enzyme {

class MGradientUtilsReverse {
public:
  MGradientUtilsReverse(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                        FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                        IRMapping invertedPointers_,
                        const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                        const SmallPtrSetImpl<mlir::Value> &activevals_,
                        DIFFE_TYPE ReturnActivity,
                        ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                        IRMapping &originalToNewFn_,
                        std::map<Operation *, Operation *> &originalToNewFnOps_,
                        DerivativeMode mode_, unsigned width,
                        SymbolTableCollection &symbolTable_);

  // From CacheUtility
  FunctionOpInterface newFunc;
  FunctionOpInterface oldFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  IRMapping invertedPointersGlobal;
  IRMapping invertedPointersShadow;
  IRMapping shadowValues;
  Block *initializationBlock;

  IRMapping mapReverseModeBlocks;
  DenseMap<Block *, SmallVector<std::pair<Value, Value>>> mapBlockArguments;

  IRMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  MTypeAnalysis &TA;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  SymbolTableCollection &symbolTable;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  void erase(Operation *op) { op->erase(); }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantValue(mlir::Value v) const;
  bool hasInvertPointer(mlir::Value v);
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &builder);
  void mapInvertPointer(mlir::Value v, mlir::Value invertValue,
                        OpBuilder &builder);

  mlir::Value getShadowValue(mlir::Value v);
  void mapShadowValue(mlir::Value v, mlir::Value invertValue,
                      OpBuilder &builder);

  void clearValue(mlir::Value v, OpBuilder &builder);

  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM);
  Type getIndexType();
  Value insertInit(Type t);

  SmallVector<std::function<std::pair<Value, Value>(Type)>> cacheCreatorHook;
  void
  registerCacheCreatorHook(std::function<std::pair<Value, Value>(Type)> hook);
  void
  deregisterCacheCreatorHook(std::function<std::pair<Value, Value>(Type)> hook);
  std::pair<Value, Value> getNewCache(Type t);

  // Cache
  Type getCacheType(Type t);
  Type getIndexCacheType();
  Value initAndPushCache(Value v, OpBuilder &builder);

  // Gradient
  Type getGradientType(Value t);
  Value insertInitGradient(mlir::Value v, OpBuilder &builder);

  // ShadowedGradient
  Type getShadowedGradientType(Value t);
  Value insertInitShadowedGradient(mlir::Value v, OpBuilder &builder);

  bool requiresShadow(Type t);

  void initInitializationBlock(IRMapping invertedPointers_,
                               ArrayRef<DIFFE_TYPE> argDiffeTypes);

  bool onlyUsedInParentBlock(Value v);

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);

  Value popCache(Value cache, OpBuilder &builder);

  void createReverseModeBlocks(Region &oldFunc, Region &newFunc,
                               bool isParentRegion = false);

  static MGradientUtilsReverse *
  CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, mlir::Type additionalArg,
                  SymbolTableCollection &symbolTable_);
};

} // namespace enzyme
} // namespace mlir
