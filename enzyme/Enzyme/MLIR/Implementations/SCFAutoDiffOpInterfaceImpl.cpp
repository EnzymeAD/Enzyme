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

LogicalResult mlir::enzyme::controlFlowForwardHandler(Operation *op, OpBuilder &builder, MGradientUtils *gutils) {
    // For all active results, add shadow types.
    // For now, assuming all results are relevant.
    Operation *newOp = gutils->getNewFromOriginal(op);
    SmallVector<Type> newOpResultTypes;
    newOpResultTypes.reserve(op->getNumResults() * 2);
    for (Value result : op->getResults()) {
      // TODO only if used (can we DCE the primal after having done the
      // derivative).
      newOpResultTypes.push_back(result.getType());
      if (gutils->isConstantValue(result))
        continue;
      auto typeIface = dyn_cast<AutoDiffTypeInterface>(result.getType());
      if (!typeIface)
        return failure();
      newOpResultTypes.push_back(typeIface.getShadowType());
    }

    // For all operands that are forwarded to the body, if they are active, also
    // add the shadow as operand.
    auto regionBranchOp = dyn_cast<RegionBranchOpInterface>(op);
    if (!regionBranchOp)
      return failure();

    SmallVector<RegionSuccessor> successors;
    // TODO: consider getEntrySuccessorRegions variation that cares about
    // activity and stuff.
    // TODO: we may need to record, for every successor, which of its inputs
    // need a shadow to recreate the body correctly.
    // TODO: support region-to-region control flow as opposed to
    // entry-to-region-to-exit (e.g., scf.while).
    llvm::SmallDenseSet<unsigned> operandPositionsToShadow;
    regionBranchOp.getSuccessorRegions(RegionBranchPoint::parent(), successors);
    for (const RegionSuccessor &successor : successors) {
      if (!successor.isParent() && successor.getSuccessor()->empty())
        continue;

      OperandRange operandRange =
          regionBranchOp.getEntrySuccessorOperands(successor);

      // Need to know which of the arguments are being forwarded to from
      // operands.
      for (auto &&[i, regionValue, operand] :
           llvm::enumerate(successor.getSuccessorInputs(), operandRange)) {
        if (gutils->isConstantValue(regionValue))
          continue;
        operandPositionsToShadow.insert(operandRange.getBeginOperandIndex() +
                                        i);
      }
    }
    SmallVector<Value> newOperands;
    newOperands.reserve(op->getNumOperands() + operandPositionsToShadow.size());
    for (OpOperand &operand : op->getOpOperands()) {
      newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
      if (operandPositionsToShadow.contains(operand.getOperandNumber()))
        newOperands.push_back(gutils->invertPointerM(operand.get(), builder));
    }
    // We are assuming the op can forward additional operands, listed
    // immediately after the original operands, to the same regions.
    // ^^
    // Our interface guarantees this.
    // We also assume that the region-holding op returns all of the values
    // yielded by terminators, and only those values.

    auto iface = dyn_cast<ControlFlowAutoDiffOpInterface>(op);
    if (!iface)
      return failure();
    Operation *replacement =
        iface.createWithShadows(builder, gutils, op, newOperands);
    for (auto &&[region, replacementRegion] :
         llvm::zip(newOp->getRegions(), replacement->getRegions())) {
      replacementRegion.takeBody(region);
    }

    // Inject the mapping for the new results into GradientUtil's shadow
    // table
    SmallVector<Value> reps;
    size_t idx = 0;
    for (Value r : op->getResults()) {
      // TODO only if used
      reps.push_back(replacement->getResult(idx));
      idx++;
      if (!gutils->isConstantValue(r)) {
        auto inverted = gutils->invertedPointers.lookupOrNull(r);
        assert(inverted);
        gutils->invertedPointers.map(r, replacement->getResult(idx));
        inverted.replaceAllUsesWith(replacement->getResult(idx));
        gutils->erase(inverted.getDefiningOp());
        idx++;
      }
    }

    // differentiate body
    for (auto &origRegion : op->getRegions()) {

      for (auto &origBlock : origRegion) {
        auto origTerm = origBlock.getTerminator();
        if (isa<RegionBranchTerminatorOpInterface>(origTerm))
        for (Operation &o : origBlock.without_terminator()) {
      if (failed(gutils->visitChild(&o)))
        return failure();
        }
        else
            for (Operation &o : origBlock.without_terminator()) {
      if (failed(gutils->visitChild(&o)))
        return failure();
        }
      }
    }

    for (auto &&[origRegion, replRegion] :
         llvm::zip(op->getRegions(), replacement->getRegions())) {
      for (auto &&[origBlock, replBlock] : llvm::zip(origRegion, replRegion)) {
        Operation *origTerminator = origBlock.getTerminator();
        auto termIface =
            dyn_cast<RegionBranchTerminatorOpInterface>(origTerminator);
        if (!termIface)
          continue;

        SmallVector<RegionSuccessor> successors;
        termIface.getSuccessorRegions({}, successors);

        llvm::SmallDenseSet<unsigned> operandsToShadow;
        for (auto &successor : successors) {
          OperandRange operandRange = termIface.getSuccessorOperands(successor);
          ValueRange targetValues = successor.isParent()
                                        ? op->getResults()
                                        : successor.getSuccessorInputs();
          assert(operandRange.size() == targetValues.size());
          for (auto &&[i, target] : llvm::enumerate(targetValues)) {
            if (!gutils->isConstantValue(target))
              operandsToShadow.insert(operandRange.getBeginOperandIndex() + i);
          }
        }
        SmallVector<Value> newOperands;
        newOperands.reserve(termIface->getNumOperands() +
                            operandsToShadow.size());
        for (OpOperand &operand : termIface->getOpOperands()) {
          newOperands.push_back(gutils->getNewFromOriginal(operand.get()));
          if (operandsToShadow.contains(operand.getOperandNumber()))
            newOperands.push_back(
                gutils->invertPointerM(operand.get(), builder));
        }

        // Assuming shadows following the originals are fine.
        // TODO: consider extending to have a ShadowableTerminatorOpInterface
        Operation *replTerminator = replBlock.getTerminator();
        builder.setInsertionPointToEnd(&replBlock);
        Operation *newTerminator = builder.clone(*replTerminator);
        newTerminator->setOperands(newOperands);
        gutils->erase(replTerminator);
      }
    }
    
    // Replace all uses of original results
    newOp->replaceAllUsesWith(reps);
    gutils->erase(newOp);

    return success();
}

namespace {
struct ForOpInterfaceCF
    : public ControlFlowAutoDiffOpInterface::ExternalModel<ForOpInterfaceCF,
                                                           scf::ForOp> {
  Operation *createWithShadows(Operation *op, OpBuilder &builder,
                               MGradientUtils *gutils, Operation *original,
                               ValueRange remappedOperands) const {
    scf::ForOpAdaptor adaptor(remappedOperands);
    auto repFor = builder.create<scf::ForOp>(
        op->getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());
    return repFor;
  }
};

struct ForOpInterface
    : public AutoDiffOpInterface::ExternalModel<ForOpInterface, scf::ForOp> {
  LogicalResult createForwardModeTangent(Operation *op, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    return controlFlowForwardHandler(op, builder, gutils);
  }
};

struct ForOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<ForOpInterfaceReverse,
                                                       scf::ForOp> {
  void createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                MGradientUtilsReverse *gutils,
                                SmallVector<Value> caches) const {
    auto forOp = cast<scf::ForOp>(op);

    SmallVector<Value> nArgs;
    for (Value v : forOp.getResults()) {
      if (auto iface = dyn_cast<AutoDiffTypeInterface>(v.getType())) {
        if (gutils->hasInvertPointer(v)) {
          nArgs.push_back(gutils->invertPointerM(v, builder));
        } else {
          nArgs.push_back(iface.createNullValue(builder, v.getLoc()));
        }
      }
    }

    auto repFor = builder.create<scf::ForOp>(
        forOp.getLoc(), gutils->popCache(caches[0], builder),
        gutils->popCache(caches[1], builder),
        gutils->popCache(caches[2], builder), nArgs); // TODO
    repFor.getRegion().begin()->erase();

    auto buildFuncReturnOp = [](OpBuilder &builder, Location loc,
                                SmallVector<Value> retargs) {
      builder.create<scf::YieldOp>(loc, retargs);
      return;
    };

    gutils->Logic.differentiate(gutils, forOp.getRegion(), repFor.getRegion(),
                                /*parentRegion=*/false, buildFuncReturnOp,
                                nullptr);

    // Insert the index which is carried by the scf for op.
    Type indexType = IndexType::get(builder.getContext());
    repFor.getRegion().insertArgument((unsigned)0, indexType, forOp.getLoc());

    for (const auto &[iterOperand, adjResult] :
         llvm::zip(forOp.getInitArgs(), repFor.getResults())) {
      if (gutils->hasInvertPointer(iterOperand)) {
        auto autoDiffType = cast<AutoDiffTypeInterface>(iterOperand.getType());
        Value before = gutils->invertPointerM(iterOperand, builder);
        Value after = autoDiffType.createAddOp(builder, forOp.getLoc(), before,
                                               adjResult);
        gutils->mapInvertPointer(iterOperand, after, builder);
      }
    }
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
    scf::ForOp::attachInterface<ForOpInterface>(*context);

    scf::ForOp::attachInterface<ForOpInterfaceReverse>(*context);
    scf::ForOp::attachInterface<ForOpInterfaceCF>(*context);
  });
}
