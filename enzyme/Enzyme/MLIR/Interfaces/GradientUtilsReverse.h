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

#include "Analysis/DataFlowAliasAnalysis.h"
#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include <functional>
#include <optional>

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
                        DerivativeMode mode_, unsigned width, bool omp,
                        llvm::StringRef postpasses, bool verifyPostPasses,
                        bool strongZero);

  IRMapping mapReverseModeBlocks;

  // Set when the derivative is built to be split into an augmented forward
  // function and a reverse function. The forward sweep's stack is then gone
  // by the time the reverse sweep runs, so what lives on it and is used again
  // in the reverse sweep has to live elsewhere.
  bool splitting = false;

  // Cache for the Deferred free so that every queries from free knows whether
  // it is guaranteedfrees or not
  struct GuaranteedFrees {
    DenseMap<Operation *, Operation *> byAllocation;
    DenseSet<Operation *> frees;
  };
  std::optional<GuaranteedFrees> guaranteedFrees;

  void addToDiffe(mlir::Value oldGradient, mlir::Value addedGradient,
                  OpBuilder &builder);

  Type getIndexType();
  Value insertInit(Type t);

  // The shadows whose primals the forward sweep frees, by calling `dealloc`,
  // in a way the reverse sweep cannot match to their allocation. The reverse
  // sweep may still use them, so they are freed with `dealloc` once the
  // outermost derivative is done, and not before: a split derivative is
  // handed its caller's list, as one of its tape's caches. `elemType` is the
  // pointer type the list holds.
  Value getDeferredFrees(Type elemType, StringRef dealloc);
  static constexpr StringLiteral deferredFreeAttrName = "enzyme.deferred_free";

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
      llvm::ArrayRef<DIFFE_TYPE> constant_args, mlir::Type additionalArg,
      bool omp, llvm::StringRef postpasses, bool verifyPostPasses,
      bool strongZero);
};

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_GRADIENT_UTILS_REVERSE_H
