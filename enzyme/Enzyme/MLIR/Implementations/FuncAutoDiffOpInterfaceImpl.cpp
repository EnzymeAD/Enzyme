//===- FuncAutoDiffOpInterfaceImpl.cpp - Interface external model --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR arithmetic dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/Ops.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/FuncDerivatives.inc"
} // namespace

static std::optional<func::FuncOp> getContainingFunction(Operation *orig) {
  Operation *parent;
  while (parent = orig->getParentOp()) {
    if (auto func = dyn_cast<func::FuncOp>(parent)) {
      return std::optional(func);
    }
  }

  return std::nullopt;
}

class AutoDiffCallRev
    : public ReverseAutoDiffOpInterface::ExternalModel<AutoDiffCallRev,
                                                       func::CallOp> {
public:
  LogicalResult createReverseModeAdjoint(Operation *orig, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    DerivativeMode mode = DerivativeMode::ReverseModeGradient;

    SymbolTable symbolTable = SymbolTable::getNearestSymbolTable(orig);

    func::CallOp callOp = cast<func::CallOp>(orig);

    auto parent = getContainingFunction(orig);
    if (parent.has_value() &&
        callOp.getCallee() == parent.value().getSymName()) {
      // TODO: Recursion chains
      orig->emitError() << "could not emit adjoint of recursive call at: "
                        << *orig << "\n";
      return failure();
    }

    Operation *callee = symbolTable.lookup(callOp.getCallee());
    auto fn = cast<FunctionOpInterface>(callee);

    auto narg = orig->getNumOperands();
    auto nret = orig->getNumResults();

    std::vector<DIFFE_TYPE> RetActivity;
    for (auto res : callOp.getResults()) {
      RetActivity.push_back(gutils->isConstantValue(res)
                                ? DIFFE_TYPE::CONSTANT
                                : DIFFE_TYPE::OUT_DIFF);
    }

    std::vector<DIFFE_TYPE> ArgActivity;
    for (auto arg : callOp.getOperands()) {
      ArgActivity.push_back(
          gutils->isConstantValue(arg) ? DIFFE_TYPE::CONSTANT
          : arg.getType().cast<AutoDiffTypeInterface>().isMutable()
              ? DIFFE_TYPE::DUP_ARG
              : DIFFE_TYPE::OUT_DIFF);
    }

    std::vector<bool> volatile_args(narg, true);
    std::vector<bool> returnShadow(narg, false);
    std::vector<bool> returnPrimal(nret, false);

    auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

    bool freeMemory = true;
    size_t width = 1;

    auto revFn = gutils->Logic.CreateReverseDiff(
        fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, returnShadow,
        mode, freeMemory, width, /*addedType*/ nullptr, type_args,
        volatile_args, /*augmented*/ nullptr);

    SmallVector<Value> revArguments;

    for (auto cache : caches) {
      revArguments.push_back(gutils->popCache(cache, builder));
    }

    for (auto result : callOp.getResults()) {
      if (gutils->isConstantValue(result))
        continue;
      revArguments.push_back(gutils->diffe(result, builder));
    }

    auto revCallOp = builder.create<func::CallOp>(
        orig->getLoc(), cast<func::FuncOp>(revFn), revArguments);

    int revIndex = 0;
    for (auto arg : callOp.getOperands()) {
      if (gutils->isConstantValue(arg))
        continue;
      auto diffe = revCallOp.getResult(revIndex);
      gutils->addToDiffe(arg, diffe, builder);
      revIndex++;
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *orig,
                                 MGradientUtilsReverse *gutils) const {
    SmallVector<Value> cachedArguments;

    Operation *newOp = gutils->getNewFromOriginal(orig);
    OpBuilder cacheBuilder(newOp);

    for (auto arg : orig->getOperands()) {
      Value cache = gutils->initAndPushCache(gutils->getNewFromOriginal(arg),
                                             cacheBuilder);
      cachedArguments.push_back(cache);
    }

    return cachedArguments;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

void mlir::enzyme::registerFuncDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, func::FuncDialect *) {
    registerInterfaces(context);
    func::CallOp::attachInterface<AutoDiffCallRev>(*context);
  });
}
