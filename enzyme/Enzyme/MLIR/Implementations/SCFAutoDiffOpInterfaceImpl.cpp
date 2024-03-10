//===- SCFAutoDiffOpInterfaceImpl.cpp - Interface external model ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream MLIR SCF dialect.
//
//===----------------------------------------------------------------------===//

#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <functional>

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/SCFDerivatives.inc"

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto forOp = cast<scf::ForOp>(op);

    // Begin Perform d(yielded value[i]) += d(result[i]); d(result[i]) = 0
    SmallVector<Value, 1> resDiffes;
    for (OpResult v : forOp.getResults()) {
      if (!gutils->isConstantValue(v)) {
        auto autoDiffType = cast<AutoDiffTypeInterface>(v.getType());
        if (!autoDiffType.isMutable()) {
          auto prev = gutils->diffe(v, builder);
          gutils->zeroDiffe(v, builder);
          resDiffes.push_back(prev);
          continue;
        }
      }
      resDiffes.push_back(nullptr);
    }

    for (auto &reg : op->getRegions()) {
      auto termIface =
          cast<RegionBranchTerminatorOpInterface>(reg.begin()->getTerminator());

      SmallVector<RegionSuccessor> successors;
      termIface.getSuccessorRegions(
          SmallVector<Attribute>(termIface->getNumOperands(), Attribute()),
          successors);

      for (auto &successor : successors) {
        if (!successor.isParent())
          continue;
        OperandRange operandRange = termIface.getSuccessorOperands(successor);
        assert(operandRange.size() == resDiffes.size());

        // There is an assumption here that there is only regions that branch to
        // the successor. Specifically, otherwise we would need to
        // gutils->addToDiffe select (if came from that result)
        for (auto &&[prev, post] : llvm::zip(operandRange, resDiffes)) {
          if (!post)
            continue;
          if (!gutils->isConstantValue(prev))
            gutils->addToDiffe(prev, post, builder);
        }
      }
    }
    // End Perform d(yielded value[i]) += d(result[i]); d(result[i]) = 0

    auto start = gutils->popCache(caches[0], builder);
    auto end = gutils->popCache(caches[1], builder);
    auto step = gutils->popCache(caches[2], builder);

    auto repFor = builder.create<scf::ForOp>(forOp.getLoc(), start, end, step,
                                             ArrayRef<Value>());
    // erase scf yield
    repFor.getBody()->begin()->erase();

    for (auto &&[oldReg, newReg] :
         llvm::zip(op->getRegions(), repFor->getRegions())) {

      // This code assumes at most one terminating block for each region (lest
      // the append happen multiple times)
      auto buildFuncReturnOp = [&](OpBuilder &builder, Block *oBB) {
        auto loc = oBB->rbegin()->getLoc();

        auto idx = repFor.getInductionVar();

        auto lhs = builder.create<arith::AddIOp>(loc, idx, step);

        // This needs to know a condition describing which predecessor this will
        // return to, to select the right value Here we use the condition i +
        // step >= end   to determine the last iteration

        auto condition = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, lhs, end);

        for (auto [arg, init_arg] :
             llvm::zip(oBB->getArguments().slice(1), forOp.getInitArgs())) {
          if (!gutils->isConstantValue(arg) &&
              !cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
            auto diffe = gutils->diffe(arg, builder);
            gutils->zeroDiffe(arg, builder);

            auto zero = cast<AutoDiffTypeInterface>(diffe.getType())
                            .createNullValue(builder, loc);
            auto outside =
                builder.create<arith::SelectOp>(loc, condition, diffe, zero);
            auto inside =
                builder.create<arith::SelectOp>(loc, condition, zero, diffe);

            // For each predecessor, if we came from that predecessor += the
            // shadow of the arg [after zero'ing]
            if (!gutils->isConstantValue(init_arg)) {
              gutils->addToDiffe(init_arg, outside, builder);
            }

            if (!gutils->isConstantValue(arg)) {
              gutils->addToDiffe(arg, inside, builder);
            }
          }
        }
        builder.create<scf::YieldOp>(loc);
      };

      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        gutils->mapReverseModeBlocks.map(&oBB, &revBB);
      }
      for (auto &&[oBB, revBB] : llvm::zip(oldReg, newReg)) {
        auto sub = gutils->Logic.visitChildren(&oBB, &revBB, gutils);
        if (!sub.succeeded())
          return sub;
        Block *newBB = gutils->getNewFromOriginal(&oBB);
        gutils->Logic.handlePredecessors(&oBB, newBB, &revBB, gutils,
                                         buildFuncReturnOp);
      }
    }
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto forOp = cast<scf::ForOp>(op);

    Operation *newOp = gutils->getNewFromOriginal(op);
    OpBuilder cacheBuilder(newOp);
    SmallVector<Value> caches;

    Value cacheLB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getLowerBound()), cacheBuilder);
    caches.push_back(cacheLB);

    Value cacheUB = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getUpperBound()), cacheBuilder);
    caches.push_back(cacheUB);

    Value cacheStep = gutils->initAndPushCache(
        gutils->getNewFromOriginal(forOp.getStep()), cacheBuilder);
    caches.push_back(cacheStep);

    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto forOp = cast<scf::ForOp>(op);
  }
};

} // namespace

void mlir::enzyme::registerSCFDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, scf::SCFDialect *) {
    registerInterfaces(context);
    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
  });
}
