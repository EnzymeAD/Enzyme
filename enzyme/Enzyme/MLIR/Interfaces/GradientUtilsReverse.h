//===- GradientUtilsReverse.h - Utilities for gradient interfaces -------* C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_GRADIENT_UTILS_REVERSE_H
#define ENZYME_MLIR_INTERFACES_GRADIENT_UTILS_REVERSE_H

#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include <functional>

#include "GradientUtils.h"

namespace mlir {
namespace enzyme {

class MGradientUtilsReverse : public MDiffeGradientUtils {
public:
  MGradientUtilsReverse(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                        FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                        IRMapping invertedPointers_,
                        const llvm::ArrayRef<bool> returnPrimals,
                        const llvm::ArrayRef<bool> returnShadows,
                        const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                        const SmallPtrSetImpl<mlir::Value> &activevals_,
                        ArrayRef<DIFFE_TYPE> ReturnActivity,
                        ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                        IRMapping &originalToNewFn_,
                        std::map<Operation *, Operation *> &originalToNewFnOps_,
                        DerivativeMode mode_, unsigned width);

  IRMapping mapReverseModeBlocks;

  void addToDiffe(mlir::Value oldGradient, mlir::Value addedGradient,
                  OpBuilder &builder);

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

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);

  Value popCache(Value cache, OpBuilder &builder);

  void createReverseModeBlocks(Region &oldFunc, Region &newFunc);

  static MGradientUtilsReverse *CreateFromClone(
      MEnzymeLogic &Logic, DerivativeMode mode_, unsigned width,
      FunctionOpInterface todiff, MTypeAnalysis &TA, MFnTypeInfo &oldTypeInfo,
      const ArrayRef<bool> returnPrimals, const ArrayRef<bool> returnShadows,
      llvm::ArrayRef<DIFFE_TYPE> retType,
      llvm::ArrayRef<DIFFE_TYPE> constant_args, mlir::Type additionalArg);
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_GRADIENT_UTILS_REVERSE_H
