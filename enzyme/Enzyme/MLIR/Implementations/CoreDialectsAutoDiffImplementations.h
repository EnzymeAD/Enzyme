//===- CoreDialectsAutoDiffImplementation.h - Impl registrations -* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains context registration facilities for external model
// implementations of the automatic differentiation interface for upstream MLIR
// dialects.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEMLIR_CORE_IMPL_H_
#define ENZYMEMLIR_CORE_IMPL_H_

#include "Interfaces/AutoDiffOpInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"

namespace mlir {
class DialectRegistry;
class Operation;
class OpBuilder;
class RegionSuccessor;

namespace enzyme {
class MGradientUtils;
class MGradientUtilsReverse;

namespace detail {
// Non-template implementation of
// AutoDiffUsingControlFlow::createForwardModeTangent.

LogicalResult controlFlowForwardHandler(Operation *op, OpBuilder &builder,
                                        MGradientUtils *gutils);

LogicalResult controlFlowForwardHandler(
    Operation *op, OpBuilder &builder, MGradientUtils *gutils,
    const llvm::SmallDenseSet<unsigned> &operandPositionsToShadow,
    const llvm::SmallDenseSet<unsigned> &resultPositionsToShadow);

// Implements forward-mode differentiation of branching operations.
// Assumes that successive shadows are legal
void branchingForwardHandler(Operation *op, OpBuilder &builder,
                             MGradientUtils *gutils);

// Implements forward-mode differentiation of region-terminator operations.
// Assumes that successive shadows are legal
void regionTerminatorForwardHandler(Operation *op, OpBuilder &builder,
                                    MGradientUtils *gutils);

// Implements reverse-mode differentiation of return operations.
void returnReverseHandler(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils);

// Implements forward-mode differentiation of read-only (including read-none)
// operations which do not perform computation
LogicalResult memoryIdentityForwardHandler(Operation *op, OpBuilder &builder,
                                           MGradientUtils *gutils,
                                           ArrayRef<int> storedVals);

// Implements shadow initialization differentiation of allocation
LogicalResult allocationForwardHandler(Operation *op, OpBuilder &builder,
                                       MGradientUtils *gutils, bool zero);

// Implements the forward autodiff interface for operations whose derivatives
// are can be inferred by analyzing their control flow and differentiating the
// nested operations.
template <typename OpTy>
class AutoDiffUsingControlFlow
    : public AutoDiffOpInterface::ExternalModel<AutoDiffUsingControlFlow<OpTy>,
                                                OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    return controlFlowForwardHandler(op, builder, gutils);
  }
};

// Implements the forward autodiff interface for operations whose derivatives
// are can be inferred by analyzing their branching properties.
template <typename OpTy>
class AutoDiffUsingBranch
    : public AutoDiffOpInterface::ExternalModel<AutoDiffUsingBranch<OpTy>,
                                                OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    branchingForwardHandler(op, builder, gutils);
    return success();
  }
};

// Implements the forward autodiff interface for operations whose derivatives
// are can be inferred by analyzing their region terminator properties.
template <typename OpTy>
class AutoDiffUsingRegionTerminator
    : public AutoDiffOpInterface::ExternalModel<
          AutoDiffUsingRegionTerminator<OpTy>, OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    regionTerminatorForwardHandler(op, builder, gutils);
    return success();
  }
};

template <typename OpTy>
class NoopRevAutoDiffInterface
    : public ReverseAutoDiffOpInterface::ExternalModel<
          NoopRevAutoDiffInterface<OpTy>, OpTy> {
public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

template <typename OpTy>
class ReturnRevAutoDiffInterface
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ReturnRevAutoDiffInterface<OpTy>, OpTy> {
public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    returnReverseHandler(op, builder, gutils);
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// Implements the forward autodiff interface for operations which are
// read only and identity like (aka not computing sin of mem read).
template <typename OpTy, int... storedvals>
class AutoDiffUsingMemoryIdentity
    : public AutoDiffOpInterface::ExternalModel<
          AutoDiffUsingMemoryIdentity<OpTy, storedvals...>, OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {

    return memoryIdentityForwardHandler(
        op, builder, gutils, std::initializer_list<int>{storedvals...});
  }
};

// Implements the forward autodiff interface for operations which are
// allocation like
template <typename OpTy>
class AutoDiffUsingAllocationFwd : public AutoDiffOpInterface::ExternalModel<
                                       AutoDiffUsingAllocationFwd<OpTy>, OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {

    return allocationForwardHandler(op, builder, gutils, /*zero*/ false);
  }
};

// Implements the reverse autodiff interface for operations which are
// allocation like
template <typename OpTy>
class AutoDiffUsingAllocationRev
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AutoDiffUsingAllocationRev<OpTy>, OpTy> {
public:
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return SmallVector<Value>();
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    (void)allocationForwardHandler(op, builder, (MGradientUtils *)gutils,
                                   /*zero*/ true);
  }
};
} // namespace detail

// Registers AutoDiffUsingControlFlow for the given op.
template <typename OpTy>
void registerAutoDiffUsingControlFlowInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingControlFlow<OpTy>>(
      context);
}
// Registers AutoDiffUsingBranch for the given op.
template <typename OpTy>
void registerAutoDiffUsingBranchInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingBranch<OpTy>>(context);
  OpTy::template attachInterface<detail::NoopRevAutoDiffInterface<OpTy>>(
      context);
}
// Registers AutoDiffUsingRegionTerminator for the given op.
template <typename OpTy>
void registerAutoDiffUsingRegionTerminatorInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingRegionTerminator<OpTy>>(
      context);
  OpTy::template attachInterface<detail::NoopRevAutoDiffInterface<OpTy>>(
      context);
}
// Registers registerAutoDiffUsingReturnInterface for the given op.
template <typename OpTy>
void registerAutoDiffUsingReturnInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingRegionTerminator<OpTy>>(
      context);
  OpTy::template attachInterface<detail::ReturnRevAutoDiffInterface<OpTy>>(
      context);
}
// Registers AutoDiffUsingMemoryIdentity for the given op.
template <typename OpTy, int... storedvals>
void registerAutoDiffUsingMemoryIdentityInterface(MLIRContext &context) {
  OpTy::template attachInterface<
      detail::AutoDiffUsingMemoryIdentity<OpTy, storedvals...>>(context);
}
// Registers AutoDiffUsingAllocation for the given op.
template <typename OpTy>
void registerAutoDiffUsingAllocationInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingAllocationFwd<OpTy>>(
      context);
  OpTy::template attachInterface<detail::AutoDiffUsingAllocationRev<OpTy>>(
      context);
}

// Interface registration hooks for individual upstream dialects.
void registerAffineDialectAutoDiffInterface(DialectRegistry &registry);
void registerArithDialectAutoDiffInterface(DialectRegistry &registry);
void registerBuiltinDialectAutoDiffInterface(DialectRegistry &registry);
void registerLLVMDialectAutoDiffInterface(DialectRegistry &registry);
void registerLLVMExtDialectAutoDiffInterface(DialectRegistry &registry);
void registerNVVMDialectAutoDiffInterface(DialectRegistry &registry);
void registerMemRefDialectAutoDiffInterface(DialectRegistry &registry);
void registerComplexDialectAutoDiffInterface(DialectRegistry &registry);
void registerSCFDialectAutoDiffInterface(DialectRegistry &registry);
void registerCFDialectAutoDiffInterface(DialectRegistry &registry);
void registerLinalgDialectAutoDiffInterface(DialectRegistry &registry);
void registerMathDialectAutoDiffInterface(DialectRegistry &registry);
void registerFuncDialectAutoDiffInterface(DialectRegistry &registry);
void registerTensorDialectAutoDiffInterface(DialectRegistry &registry);
void registerEnzymeDialectAutoDiffInterface(DialectRegistry &registry);

void registerCoreDialectAutodiffInterfaces(DialectRegistry &registry);

mlir::TypedAttr getConstantAttr(mlir::Type type, llvm::StringRef value);
} // namespace enzyme
} // namespace mlir

#endif
