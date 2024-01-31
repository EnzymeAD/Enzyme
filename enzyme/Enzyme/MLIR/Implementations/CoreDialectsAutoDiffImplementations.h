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

#include "Interfaces/AutoDiffOpInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class DialectRegistry;
class Operation;
class OpBuilder;

namespace enzyme {
class MGradientUtils;

namespace detail {
// Non-template implementation of
// AutoDiffUsingControlFlow::createForwardModeTangent.
LogicalResult controlFlowForwardHandler(Operation *op, OpBuilder &builder,
                                        MGradientUtils *gutils);

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
} // namespace detail

// Registers AutoDiffUsingControlFlow for the given op.
template <typename OpTy>
void registerAutoDiffUsingControlFlowInterface(MLIRContext &context) {
  OpTy::template attachInterface<detail::AutoDiffUsingControlFlow<OpTy>>(
      context);
}

// Interface registration hooks for individual upstream dialects.
void registerAffineDialectAutoDiffInterface(DialectRegistry &registry);
void registerArithDialectAutoDiffInterface(DialectRegistry &registry);
void registerBuiltinDialectAutoDiffInterface(DialectRegistry &registry);
void registerLLVMDialectAutoDiffInterface(DialectRegistry &registry);
void registerNVVMDialectAutoDiffInterface(DialectRegistry &registry);
void registerMemRefDialectAutoDiffInterface(DialectRegistry &registry);
void registerSCFDialectAutoDiffInterface(DialectRegistry &registry);
void registerLinalgDialectAutoDiffInterface(DialectRegistry &registry);
} // namespace enzyme
} // namespace mlir
