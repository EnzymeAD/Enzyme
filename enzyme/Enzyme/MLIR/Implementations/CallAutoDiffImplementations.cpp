//===- CallAutoDiffImplementations.cpp - Call external models ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The dialect-independent half of differentiating a call. See the header.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CallAutoDiffImplementations.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffTypeInterface.h"

using namespace mlir;
using namespace mlir::enzyme;
namespace edetail = mlir::enzyme::detail;

FunctionOpInterface edetail::getDirectCallee(Operation *op) {
  auto callOp = dyn_cast<CallOpInterface>(op);
  if (!callOp)
    return nullptr;
  auto sym = dyn_cast<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(op, sym));
}

Operation *edetail::createCallToFunction(FunctionOpInterface fn,
                                         OpBuilder &builder, Location loc,
                                         ValueRange args) {
  return cast<AutoDiffFunctionInterface>(fn.getOperation())
      .createCall(builder, loc, args);
}

LogicalResult edetail::callForwardHandler(Operation *orig, OpBuilder &builder,
                                          MGradientUtils *gutils) {
  DerivativeMode mode = DerivativeMode::ForwardMode;

  auto fn = getDirectCallee(orig);
  if (!fn) {
    orig->emitError() << "could not find the callee of: " << *orig << "\n";
    return failure();
  }

  auto narg = orig->getNumOperands();
  auto nret = orig->getNumResults();

  std::vector<DIFFE_TYPE> RetActivity;
  RetActivity.reserve(nret);
  for (auto res : orig->getResults()) {
    RetActivity.push_back(gutils->isConstantValue(res) ? DIFFE_TYPE::CONSTANT
                                                       : DIFFE_TYPE::DUP_ARG);
  }

  std::vector<DIFFE_TYPE> ArgActivity;
  ArgActivity.reserve(narg);
  for (auto arg : orig->getOperands()) {
    ArgActivity.push_back(gutils->isConstantValue(arg) ? DIFFE_TYPE::CONSTANT
                                                       : DIFFE_TYPE::DUP_ARG);
  }

  std::vector<bool> returnPrimal(nret, true);
  std::vector<bool> returnShadow(nret, false);

  auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

  bool freeMemory = true;
  size_t width = gutils->width;

  std::vector<bool> volatile_args(narg, false);

  auto forwardFn = gutils->Logic.CreateForwardDiff(
      fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, mode, freeMemory,
      width,
      /* addedType */ nullptr, type_args, volatile_args,
      /* augmented */ nullptr, gutils->omp, gutils->postpasses,
      gutils->verifyPostPasses, gutils->strongZero);

  SmallVector<Value> fwdArguments;

  for (auto &&[arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    fwdArguments.push_back(gutils->getNewFromOriginal(arg));
    if (act == DIFFE_TYPE::DUP_ARG)
      fwdArguments.push_back(gutils->invertPointerM(arg, builder));
  }

  auto *fwdCallOp =
      createCallToFunction(forwardFn, builder, orig->getLoc(), fwdArguments);

  SmallVector<Value> primals;
  primals.reserve(nret);

  int fwdIndex = 0;
  for (auto &&[ret, act] : llvm::zip_equal(orig->getResults(), RetActivity)) {
    auto fwdRet = fwdCallOp->getResult(fwdIndex);
    primals.push_back(fwdRet);

    fwdIndex++;

    if (act == DIFFE_TYPE::DUP_ARG) {
      gutils->setDiffe(ret, fwdCallOp->getResult(fwdIndex), builder);
      fwdIndex++;
    }
  }

  auto newOp = gutils->getNewFromOriginal(orig);
  gutils->replaceOrigOpWith(orig, primals);
  gutils->erase(newOp);

  return success();
}

LogicalResult edetail::callReverseHandler(Operation *orig, OpBuilder &builder,
                                          MGradientUtilsReverse *gutils,
                                          SmallVector<Value> caches) {
  DerivativeMode mode = DerivativeMode::ReverseModeGradient;

  auto fn = getDirectCallee(orig);
  if (!fn) {
    orig->emitError() << "could not find the callee of: " << *orig << "\n";
    return failure();
  }

  auto narg = orig->getNumOperands();
  auto nret = orig->getNumResults();

  std::vector<DIFFE_TYPE> RetActivity;
  for (auto res : orig->getResults()) {
    RetActivity.push_back(
        gutils->isConstantValue(res) ? DIFFE_TYPE::CONSTANT
        : cast<AutoDiffTypeInterface>(res.getType()).isMutable()
            ? DIFFE_TYPE::DUP_ARG
            : DIFFE_TYPE::OUT_DIFF);
  }

  std::vector<DIFFE_TYPE> ArgActivity;
  for (auto arg : orig->getOperands()) {
    ArgActivity.push_back(
        gutils->isConstantValue(arg) ? DIFFE_TYPE::CONSTANT
        : cast<AutoDiffTypeInterface>(arg.getType()).isMutable()
            ? DIFFE_TYPE::DUP_ARG
            : DIFFE_TYPE::OUT_DIFF);
  }

  if (llvm::any_of(RetActivity,
                   [&](auto act) { return act == DIFFE_TYPE::DUP_ARG; })) {
    orig->emitError() << "could not emit adjoint with mutable return types in: "
                      << *orig << "\n";
    return failure();
  }

  std::vector<bool> volatile_args(narg, true);
  std::vector<bool> returnShadow(nret, false);
  std::vector<bool> returnPrimal(nret, false);

  auto type_args = gutils->TA.getAnalyzedTypeInfo(fn);

  bool freeMemory = true;
  size_t width = gutils->width;

  auto revFn = gutils->Logic.CreateReverseDiff(
      fn, RetActivity, ArgActivity, gutils->TA, returnPrimal, returnShadow,
      mode, freeMemory, gutils->AtomicAdd, width, /*addedType*/ nullptr,
      type_args, volatile_args, /*augmented*/ nullptr, gutils->omp,
      gutils->postpasses, gutils->verifyPostPasses, gutils->strongZero,
      /*markReadonly=*/false);

  SmallVector<Value> revArguments;

  for (auto [arg, act, cache] :
       llvm::zip_equal(orig->getOperands(), ArgActivity, caches)) {
    revArguments.push_back(gutils->popCache(cache, builder));
    if (act == DIFFE_TYPE::DUP_ARG)
      revArguments.push_back(gutils->invertPointerM(arg, builder));
  }

  for (auto result : orig->getResults()) {
    if (gutils->isConstantValue(result))
      continue;
    revArguments.push_back(gutils->diffe(result, builder));
  }

  auto *revCallOp =
      createCallToFunction(revFn, builder, orig->getLoc(), revArguments);

  int revIndex = 0, fwdIndex = 0;
  for (auto [arg, act] : llvm::zip_equal(orig->getOperands(), ArgActivity)) {
    fwdIndex++;

    if (gutils->isConstantValue(arg))
      continue;

    if (act == DIFFE_TYPE::DUP_ARG) {
      cast<ClonableTypeInterface>(arg.getType())
          .freeClonedValue(builder, revArguments[fwdIndex - 1]);
      fwdIndex++;
    } else {
      auto diffe = revCallOp->getResult(revIndex);
      gutils->addToDiffe(arg, diffe, builder);
      revIndex++;
    }
  }

  return success();
}

SmallVector<Value> edetail::callCacheValues(Operation *orig,
                                            MGradientUtilsReverse *gutils) {
  SmallVector<Value> cachedArguments;

  Operation *newOp = gutils->getNewFromOriginal(orig);
  OpBuilder cacheBuilder(newOp);

  for (auto arg : orig->getOperands()) {
    Value toCache = gutils->getNewFromOriginal(arg);
    if (auto iface = dyn_cast<ClonableTypeInterface>(arg.getType())) {
      toCache = iface.cloneValue(cacheBuilder, toCache);
    }
    Value cache = gutils->initAndPushCache(toCache, cacheBuilder);
    cachedArguments.push_back(cache);
  }

  return cachedArguments;
}
