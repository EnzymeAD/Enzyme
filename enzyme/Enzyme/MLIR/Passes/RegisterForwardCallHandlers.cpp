//===- RegisterForwardCallHandlers.cpp - Test forward call handlers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that registers forward-mode call handlers for
// tests. The handlers call functions of the module that implement the rule.
//
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/GradientUtils.h"
#include "Passes/PassDetails.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_REGISTERFORWARDCALLHANDLERSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
/// How the handler registered under `<key>` differentiates a call.
enum class TestForwardCallHandlerKind {
  /// Replaces the call with `@<key>_fwd_dup`, which takes each operand and its
  /// shadow and returns each result and its shadow.
  FwdDup,
  /// Keeps the call and gets the shadows from `@<key>_fwd`.
  Fwd,
  /// Replaces the call with `@<key>_primal` before calling `@<key>_fwd`.
  PrimalFirst,
};
} // namespace

/// Calls `@name` at `builder`, or reports an error at `orig` if there is no
/// such function.
static FailureOr<func::CallOp> callTestRule(OpBuilder &builder, Operation *orig,
                                            StringRef name, TypeRange types,
                                            ValueRange args) {
  if (!SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
          orig, StringAttr::get(orig->getContext(), name))) {
    orig->emitError() << "test forward call handler: no function @" << name;
    return failure();
  }
  return func::CallOp::create(builder, orig->getLoc(), name, types, args);
}

/// Differentiates the call `orig` as described for `kind`.
static LogicalResult testForwardCallHandler(OpBuilder &builder, Operation *orig,
                                            MGradientUtils *gutils,
                                            StringRef key,
                                            TestForwardCallHandlerKind kind) {
  SmallVector<Value> primalArgs, args;
  for (Value operand : cast<CallOpInterface>(orig).getArgOperands()) {
    Value primal = gutils->getNewFromOriginal(operand);
    primalArgs.push_back(primal);
    args.push_back(primal);
    args.push_back(gutils->invertPointerM(operand, builder));
  }
  SmallVector<Type> types, shadowTypes, fwdDupTypes;
  for (Value result : orig->getResults()) {
    types.push_back(result.getType());
    shadowTypes.push_back(gutils->getShadowType(result.getType()));
    fwdDupTypes.push_back(types.back());
    fwdDupTypes.push_back(shadowTypes.back());
  }

  SmallVector<Value> primals, shadows;
  switch (kind) {
  case TestForwardCallHandlerKind::FwdDup: {
    FailureOr<func::CallOp> call = callTestRule(
        builder, orig, (key + "_fwd_dup").str(), fwdDupTypes, args);
    if (failed(call))
      return failure();
    for (unsigned i = 0, e = orig->getNumResults(); i < e; i++) {
      primals.push_back(call->getResult(2 * i));
      shadows.push_back(call->getResult(2 * i + 1));
    }
    break;
  }
  case TestForwardCallHandlerKind::Fwd: {
    FailureOr<func::CallOp> call =
        callTestRule(builder, orig, (key + "_fwd").str(), shadowTypes, args);
    if (failed(call))
      return failure();
    llvm::append_range(shadows, call->getResults());
    break;
  }
  case TestForwardCallHandlerKind::PrimalFirst: {
    FailureOr<func::CallOp> primalCall =
        callTestRule(builder, orig, (key + "_primal").str(), types, primalArgs);
    if (failed(primalCall))
      return failure();
    builder.setInsertionPointAfter(gutils->getNewFromOriginal(orig));
    replaceForwardCall(orig, primalCall->getResults(), gutils);
    FailureOr<func::CallOp> call =
        callTestRule(builder, orig, (key + "_fwd").str(), shadowTypes, args);
    if (failed(call))
      return failure();
    llvm::append_range(shadows, call->getResults());
    break;
  }
  }

  for (auto &&[result, shadow] : llvm::zip_equal(orig->getResults(), shadows))
    if (!gutils->isConstantValue(result))
      gutils->setDiffe(result, shadow, builder);
  if (kind == TestForwardCallHandlerKind::FwdDup)
    replaceForwardCall(orig, primals, gutils);
  return success();
}

namespace {
struct RegisterForwardCallHandlersPass
    : public enzyme::impl::RegisterForwardCallHandlersPassBase<
          RegisterForwardCallHandlersPass> {
  using RegisterForwardCallHandlersPassBase::
      RegisterForwardCallHandlersPassBase;

  void runOnOperation() override {
    EnzymeDialect *dialect = getContext().getLoadedDialect<EnzymeDialect>();
    for (const std::string &handler : handlers) {
      StringRef key = handler;
      TestForwardCallHandlerKind kind = TestForwardCallHandlerKind::FwdDup;
      if (key.consume_back(":fwd"))
        kind = TestForwardCallHandlerKind::Fwd;
      else if (key.consume_back(":primal-first"))
        kind = TestForwardCallHandlerKind::PrimalFirst;
      std::string name = key.str();
      dialect->registerForwardCallHandler(
          key, [name, kind](RewriterBase &rewriter, Operation *orig,
                            MGradientUtils *gutils) {
            return testForwardCallHandler(rewriter, orig, gutils, name, kind);
          });
    }
    markAllAnalysesPreserved();
  }
};
} // namespace
