#include "Analysis/SampleDependenceAnalysis.h"

#include "Interfaces/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;
using namespace mlir::impulse;

SampleDependenceAnalysis::SampleDependenceAnalysis(InferRegionOp regionOp)
    : SampleDependenceAnalysis(regionOp, AnalysisTarget::Sampler) {}

SampleDependenceAnalysis::SampleDependenceAnalysis(InferRegionOp regionOp,
                                                   AnalysisTarget target)
    : regionOp(regionOp), target(target),
      builtSelection(regionOp.getSelectionAttr()),
      builtLogpdfFn(regionOp.getLogpdfFnAttr()),
      builtNumPositionArgs(regionOp.getNumPositionArgs()) {
  if (target == AnalysisTarget::Logpdf)
    runLogpdfAnalysis();
  else
    runSamplerAnalysis();
}

Region &SampleDependenceAnalysis::getTargetRegion() {
  if (target == AnalysisTarget::Logpdf)
    return regionOp.getLogpdf();
  return regionOp.getSampler();
}

void SampleDependenceAnalysis::markSampleDependent(Value value) {
  sampleDependentValues.insert(value);
}

void SampleDependenceAnalysis::propagateDependence(Region &region) {
  bool changed = true;
  while (changed) {
    changed = false;
    region.walk([&](Operation *op) {
      if (isa<SampleRegionOp>(op))
        return;

      bool hasDependent = false;
      for (Value operand : op->getOperands()) {
        if (isSampleDependent(operand)) {
          hasDependent = true;
          break;
        }
      }
      // Ops with regions also depend on the values their regions use.
      if (!hasDependent && op->getNumRegions() != 0)
        hasDependent =
            op->walk([&](Operation *inner) {
                if (inner == op)
                  return WalkResult::advance();
                for (Value operand : inner->getOperands())
                  if (isSampleDependent(operand))
                    return WalkResult::interrupt();
                if (isa<SampleRegionOp>(inner) && isSampleDependent(inner))
                  return WalkResult::interrupt();
                return WalkResult::advance();
              }).wasInterrupted();

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

void SampleDependenceAnalysis::runSamplerAnalysis() {
  if (regionOp.getLogpdfFnAttr()) {
    Block &entry = regionOp.getSampler().front();
    if (!entry.getArguments().empty()) {
      markSampleDependent(entry.getArgument(0));
    }
  }

  DenseSet<Attribute> selectedAddresses;
  bool hasSelection = false;
  if (auto selection = regionOp.getSelectionAttr()) {
    hasSelection = true;
    for (auto addr : selection)
      selectedAddresses.insert(addr);
  }

  regionOp.getSampler().walk([&](SampleRegionOp sampleOp) {
    sampleOps.push_back(sampleOp);
    auto address = sampleOp.getAddressAttr();
    bool isSelected =
        !hasSelection || !address || selectedAddresses.contains(address);
    if (isSelected) {
      for (Value result : sampleOp.getResults()) {
        markSampleDependent(result);
      }
    }
  });

  propagateDependence(regionOp.getSampler());
}

void SampleDependenceAnalysis::runLogpdfAnalysis() {
  Region &logpdf = regionOp.getLogpdf();
  if (logpdf.empty())
    return;

  Block &entry = logpdf.front();
  int64_t numPositionArgs = regionOp.getNumPositionArgs();

  for (int64_t i = 0;
       i < numPositionArgs && i < (int64_t)entry.getNumArguments(); ++i)
    markSampleDependent(entry.getArgument(i));

  propagateDependence(logpdf);
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

bool SampleDependenceAnalysis::isInTargetRegion(Operation *op) {
  Region *targetRegion;
  if (target == AnalysisTarget::Logpdf)
    targetRegion = &regionOp.getLogpdf();
  else
    targetRegion = &regionOp.getSampler();
  return targetRegion->isAncestor(op->getParentRegion());
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

bool SampleDependenceAnalysis::isSelectedSite(SampleRegionOp site) {
  ArrayAttr selection = regionOp.getSelectionAttr();
  Attribute address = site.getAddressAttr();
  return !selection || !address || llvm::is_contained(selection, address);
}

void SampleDependenceAnalysis::raise(Value value) {
  if (!sampleDependentValues.insert(value).second)
    return;
  ++stats.raised;
  raisedWorklist.push_back(value);
}

/// Site results are dependent only if the site is a source (selected).
void SampleDependenceAnalysis::raiseUser(Operation *user) {
  if (!isa<SampleRegionOp>(user))
    for (Value result : user->getResults())
      raise(result);
  for (Operation *parent = user->getParentOp();
       parent && parent != regionOp.getOperation();
       parent = parent->getParentOp()) {
    ++stats.ancestorSteps;
    if (!isa<SampleRegionOp>(parent))
      for (Value result : parent->getResults())
        raise(result);
  }
}

void SampleDependenceAnalysis::updateOperation(Operation *op) {
  ++stats.opsVisited;
  bool dependentSite = false;
  if (auto site = dyn_cast<SampleRegionOp>(op)) {
    if (target == AnalysisTarget::Sampler) {
      if (!llvm::is_contained(sampleOps, site))
        sampleOps.push_back(site);
      if (isSelectedSite(site)) {
        for (Value result : op->getResults())
          raise(result);
        dependentSite = true;
      }
    }
  }
  if (dependentSite || llvm::any_of(op->getOperands(), [&](Value operand) {
        return isSampleDependent(operand);
      }))
    raiseUser(op);
}

void SampleDependenceAnalysis::propagateRaised() {
  while (!raisedWorklist.empty()) {
    Value value = raisedWorklist.pop_back_val();
    for (Operation *user : value.getUsers()) {
      ++stats.useVisits;
      if (isInTargetRegion(user))
        raiseUser(user);
    }
  }
}

void SampleDependenceAnalysis::notifyOperationInserted(Operation *op) {
  if (!isInTargetRegion(op))
    return;
  // Clones and moves bring nested IR: evaluate the whole subtree.
  op->walk([&](Operation *nested) { updateOperation(nested); });
  propagateRaised();
}

void SampleDependenceAnalysis::notifyOperationModified(Operation *op) {
  if (!isInTargetRegion(op))
    return;
  updateOperation(op);
  propagateRaised();
}

void SampleDependenceAnalysis::notifyOperationReplaced(Operation *op,
                                                       ValueRange replacement) {
  if (!isInTargetRegion(op))
    return;
  // Called before the uses change; users are only ever raised.
  for (auto [result, newValue] : llvm::zip(op->getResults(), replacement)) {
    if (!newValue || !isSampleDependent(newValue) || isSampleDependent(result))
      continue;
    for (Operation *user : result.getUsers()) {
      ++stats.useVisits;
      if (isInTargetRegion(user))
        raiseUser(user);
    }
  }
  propagateRaised();
}

void SampleDependenceAnalysis::notifyOperationErased(Operation *op) {
  for (Value result : op->getResults())
    sampleDependentValues.erase(result);
  if (auto site = dyn_cast<SampleRegionOp>(op))
    llvm::erase(sampleOps, site);
}

void SampleDependenceAnalysis::notifyBlockInserted(Block *block) {
  Region *parent = block->getParent();
  if (!parent || !getTargetRegion().isAncestor(parent))
    return;
  for (Operation &op : *block)
    op.walk([&](Operation *nested) { updateOperation(nested); });
  propagateRaised();
}

void SampleDependenceAnalysis::notifyBlockErased(Block *block) {
  for (BlockArgument arg : block->getArguments())
    sampleDependentValues.erase(arg);
}

bool SampleDependenceAnalysis::sourcesChanged() {
  return regionOp.getSelectionAttr() != builtSelection ||
         regionOp.getLogpdfFnAttr() != builtLogpdfFn ||
         regionOp.getNumPositionArgs() != builtNumPositionArgs;
}

static bool checkOperandDominance(IRMapping &regionToOuter, DominanceInfo &dom,
                                  InferRegionOp regionOp,
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
        if (enzyme::oputils::mayAlias(opEffectCopy, stationaryEffectCopy))
          return true;
      }
    }
  }
  return false;
}

static bool hoistFromRegion(InferRegionOp regionOp,
                            SampleDependenceAnalysis &sampleAnalysis,
                            Region &region) {
  if (region.empty())
    return false;

  DominanceInfo dom(regionOp);
  PostDominanceInfo pdom(regionOp);

  IRMapping regionToOuter;
  Block &entryBlock = region.front();

  if (sampleAnalysis.getTarget() == AnalysisTarget::Sampler) {
    auto inputs = regionOp.getInputs();
    bool isLogpdfMode = static_cast<bool>(regionOp.getLogpdfFnAttr());
    for (auto [idx, blockArg] : llvm::enumerate(entryBlock.getArguments())) {
      if (isLogpdfMode && idx == 0)
        continue;
      if (idx < inputs.size())
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
      bool couldCollect = enzyme::oputils::collectOpEffects(&op, opEffects);
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

bool impulse::hoistSampleInvariantOps(InferRegionOp regionOp) {
  return hoistSampleInvariantOps(regionOp, AnalysisTarget::Sampler);
}

bool impulse::hoistSampleInvariantOps(InferRegionOp regionOp,
                                      AnalysisTarget target) {
  SampleDependenceAnalysis analysis(regionOp, target);
  return hoistFromRegion(regionOp, analysis, analysis.getTargetRegion());
}
