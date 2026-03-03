#include "Analysis/SampleDependenceAnalysis.h"

#include "Interfaces/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::enzyme;

SampleDependenceAnalysis::SampleDependenceAnalysis(MCMCRegionOp regionOp)
    : regionOp(regionOp) {
  runAnalysis();
}

void SampleDependenceAnalysis::markSampleDependent(Value value) {
  sampleDependentValues.insert(value);
}

void SampleDependenceAnalysis::runAnalysis() {
  // TODO: Handle recursive regions.
  if (regionOp.getLogpdfFnAttr()) {
    Block &entry = regionOp.getBody().front();
    if (!entry.getArguments().empty()) {
      markSampleDependent(entry.getArgument(0));
    }
  }

  regionOp.getBody().walk([&](SampleRegionOp sampleOp) {
    sampleOps.push_back(sampleOp);
    for (Value result : sampleOp.getResults()) {
      markSampleDependent(result);
    }
  });

  bool changed = true;
  while (changed) {
    changed = false;
    regionOp.getBody().walk([&](Operation *op) {
      if (isa<SampleRegionOp>(op))
        return;

      bool hasDependent = false;
      for (Value operand : op->getOperands()) {
        if (isSampleDependent(operand)) {
          hasDependent = true;
          break;
        }
      }

      if (hasDependent) {
        for (Value result : op->getResults()) {
          if (!isSampleDependent(result)) {
            markSampleDependent(result);
            changed = true;
          }
        }
      }
    });
  }
}

bool SampleDependenceAnalysis::isSampleDependent(Value value) const {
  return sampleDependentValues.contains(value);
}

bool SampleDependenceAnalysis::isSampleDependent(Operation *op) const {
  for (Value result : op->getResults()) {
    if (isSampleDependent(result))
      return true;
  }
  return false;
}

bool SampleDependenceAnalysis::canHoist(Operation *op) const {
  if (isa<SampleRegionOp>(op))
    return false;

  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  if (isSampleDependent(op))
    return false;

  for (Value operand : op->getOperands()) {
    if (isSampleDependent(operand))
      return false;
  }

  return true;
}

static bool checkOperandDominance(IRMapping &regionToOuter, DominanceInfo &dom,
                                  MCMCRegionOp regionOp,
                                  SetVector<Operation *> &toHoist,
                                  ValueRange values) {
  for (Value value : values) {
    if (dom.properlyDominates(value, regionOp))
      continue;
    if (isa<BlockArgument>(value)) {
      if (regionToOuter.contains(value))
        continue;
      return false;
    }
    if (Operation *defOp = value.getDefiningOp()) {
      if (toHoist.contains(defOp))
        continue;
    }
    return false;
  }
  return true;
}

static bool
hasMemoryConflict(ArrayRef<MemoryEffects::EffectInstance> opEffects,
                  ArrayRef<MemoryEffects::EffectInstance> stationaryEffects) {
  for (const auto &stationaryEffect : stationaryEffects) {
    for (const auto &opEffect : opEffects) {
      bool isConflict =
          (isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Read>(opEffect.getEffect())) ||
          (isa<MemoryEffects::Read>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Write>(opEffect.getEffect())) ||
          (isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Write>(opEffect.getEffect()));
      if (isConflict) {
        auto stationaryEffectCopy = stationaryEffect;
        auto opEffectCopy = opEffect;
        if (oputils::mayAlias(opEffectCopy, stationaryEffectCopy))
          return true;
      }
    }
  }
  return false;
}

bool enzyme::hoistSampleInvariantOps(MCMCRegionOp regionOp) {
  DominanceInfo dom(regionOp);
  PostDominanceInfo pdom(regionOp);
  SampleDependenceAnalysis sampleAnalysis(regionOp);

  Region &region = regionOp.getBody();
  if (region.empty())
    return false;

  IRMapping regionToOuter;
  Block &entryBlock = region.front();
  auto inputs = regionOp.getInputs();

  bool isLogpdfMode = static_cast<bool>(regionOp.getLogpdfFnAttr());

  for (auto [idx, blockArg] : llvm::enumerate(entryBlock.getArguments())) {
    if (isLogpdfMode && idx == 0)
      continue;
    if (idx < inputs.size()) {
      regionToOuter.map(blockArg, inputs[idx]);
    }
  }

  SetVector<Operation *> toHoist;
  SmallVector<MemoryEffects::EffectInstance> stationaryEffects;

  for (Block &blk : region) {
    if (!pdom.postDominates(&blk, &region.front()))
      continue;

    for (Operation &op : blk.without_terminator()) {
      bool canHoist = true;

      if (!sampleAnalysis.canHoist(&op))
        canHoist = false;

      SmallVector<MemoryEffects::EffectInstance> opEffects;
      bool couldCollect = oputils::collectOpEffects(&op, opEffects);
      if (!couldCollect)
        canHoist = false;

      if (canHoist) {
        canHoist = checkOperandDominance(regionToOuter, dom, regionOp, toHoist,
                                         op.getOperands());
      }

      if (canHoist && op.getNumRegions() > 0) {
        SetVector<Value> nestedValues;
        getUsedValuesDefinedAbove(op.getRegions(), nestedValues);
        canHoist = checkOperandDominance(regionToOuter, dom, regionOp, toHoist,
                                         nestedValues.getArrayRef());
      }

      if (canHoist) {
        canHoist = !hasMemoryConflict(opEffects, stationaryEffects);
      }

      if (canHoist) {
        for (OpOperand &operand : op.getOpOperands()) {
          operand.assign(regionToOuter.lookupOrDefault(operand.get()));
        }
        for (Region &nestedRegion : op.getRegions()) {
          SetVector<Value> nestedValues;
          getUsedValuesDefinedAbove(nestedRegion, nestedValues);
          for (Value v : nestedValues) {
            if (regionToOuter.contains(v)) {
              replaceAllUsesInRegionWith(v, regionToOuter.lookup(v),
                                         nestedRegion);
            }
          }
        }
        toHoist.insert(&op);
      } else {
        stationaryEffects.append(opEffects.begin(), opEffects.end());
      }
    }
  }

  SmallVector<Operation *> sortedToHoist(toHoist.begin(), toHoist.end());
  llvm::sort(sortedToHoist, [&dom](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock())
      return a->isBeforeInBlock(b);
    return dom.dominates(a->getBlock(), b->getBlock());
  });

  for (Operation *op : sortedToHoist) {
    op->moveBefore(regionOp);
  }

  return !sortedToHoist.empty();
}
