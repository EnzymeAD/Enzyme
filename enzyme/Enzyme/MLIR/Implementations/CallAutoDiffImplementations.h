//===- CallAutoDiffImplementations.h - Call external models -----* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Differentiating a call is the same work whichever dialect spelled it: find
// the callee, ask Logic for its derivative, and call that. What differs is how
// the call names its callee and how a call to the derivative is written, and
// both of those are already asked through interfaces -- CallOpInterface for the
// first and AutoDiffFunctionInterface::createCall for the second. So the models
// here are written once over the call op type and attached from each dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEMLIR_CALL_IMPL_H_
#define ENZYMEMLIR_CALL_IMPL_H_

#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace enzyme {

namespace detail {

// The callee of `op`, or null where it is not a direct call to something this
// can see the body of.
FunctionOpInterface getDirectCallee(Operation *op);

// Write a call to `fn`, in whatever dialect `fn` is a function of.
Operation *createCallToFunction(FunctionOpInterface fn, OpBuilder &builder,
                                Location loc, ValueRange args);

LogicalResult callForwardHandler(Operation *orig, OpBuilder &builder,
                                 MGradientUtils *gutils);

LogicalResult callReverseHandler(Operation *orig, OpBuilder &builder,
                                 MGradientUtilsReverse *gutils,
                                 SmallVector<Value> caches);

SmallVector<Value> callCacheValues(Operation *orig,
                                   MGradientUtilsReverse *gutils);

} // namespace detail

template <typename OpTy>
class AutoDiffCallFwd
    : public AutoDiffOpInterface::ExternalModel<AutoDiffCallFwd<OpTy>, OpTy> {
public:
  LogicalResult createForwardModeTangent(Operation *orig, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    return detail::callForwardHandler(orig, builder, gutils);
  }
};

template <typename OpTy>
class AutoDiffCallRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffCallRev<OpTy>,
                                                       OpTy> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return detail::callReverseHandler(orig, builder, gutils, caches);
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    return detail::callCacheValues(orig, gutils);
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

} // namespace enzyme
} // namespace mlir

#endif
