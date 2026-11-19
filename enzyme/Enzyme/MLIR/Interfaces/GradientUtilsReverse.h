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
#include <vector>

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

  // With optimizeCaches, what initAndPushCache returns is a handle on how
  // the reverse sweep gets the value back, which popCache follows:
  //  - Available: it uses the value as it is. A function argument is
  //    available, and so is a value defined outside any loop in a block that
  //    dominates every return, which runs once before the reverse sweep.
  //  - Recompute: it computes the value again, as a pure op from what it
  //    gets back of the op's operands, when that needs at most one value
  //    cached, as a shadow address does: from the loop counter.
  //  - Shared: the value is pushed once, where it is defined, rather than by
  //    each op that needs it; popCache reads the top, and the reverse sweep
  //    pops it once it gets back to the definition.
  struct ReverseValue {
    enum Kind { Available, Recompute, Shared } kind;
    // Available, Shared: the original value, if there is one, or the one
    // whose shadow it is.
    Value orig;
    bool shadow = false;
    // Available without an original value: the value itself.
    Value fwd;
    // Recompute: the forward op, and the handles of its operands.
    Operation *def = nullptr;
    SmallVector<Value> operands;
  };
  DenseMap<Value, ReverseValue> reverseValues;
  // The Shared cache of each original value, and of each one's shadow.
  DenseMap<std::pair<Value, unsigned>, Value> sharedCaches;
  // Records where the reverse sweep, built up to here in `reverseBB`, gets
  // back to where `origValues` are defined: where their Shared caches are
  // popped. A value can be cached after that point is built, by an op the
  // reverse sweep reaches before it, so the pops wait for the end.
  void recordReverseDefinitions(ValueRange origValues, Block *reverseBB);
  // Pops each Shared cache where its value's definition was recorded, and
  // erases the handles nothing was popped from.
  void finishReverseValues();

  void createReverseModeBlocks(Region &oldFunc, Region &newFunc);

private:
  struct ReversePlan {
    ReverseValue::Kind kind;
    Value orig;
    bool shadow;
    Value fwd;
    Operation *def = nullptr;
    std::vector<ReversePlan> operands;
  };
  FailureOr<ReversePlan> planReverseValue(Value v, unsigned depth,
                                          unsigned &newCaches);
  FailureOr<ReversePlan> planOriginal(Value orig, bool shadow, unsigned depth,
                                      unsigned &newCaches, unsigned loopDepth);
  unsigned loopDepthOf(Block *origBlock);
  std::optional<DenseMap<Block *, unsigned>> loopDepths;
  Value commitReversePlan(const ReversePlan &plan, Type type);
  Value materializeReverseValue(Value handle, OpBuilder &builder);
  Value originalOf(Value v);
  Value shadowedOriginalOf(Value v);
  bool isOnceBeforeReverse(Block *origBlock);
  DenseMap<Value, Value> newToOriginal;
  // Where each original value's definition is in the reverse sweep: after
  // the op, or at the start of the block if there is none.
  DenseMap<Value, std::pair<Block *, Operation *>> reverseDefinitions;
  size_t newToOriginalSize = 0;
  std::optional<DenseSet<Block *>> onceBeforeReverse;

public:
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
