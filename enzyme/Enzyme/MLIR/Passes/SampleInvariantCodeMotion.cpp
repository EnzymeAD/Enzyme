//===- SampleInvariantCodeMotion.cpp - Sample-invariant code motion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/SampleDependenceAnalysis.h"
#include "Dialect/Impulse/Impulse.h"
#include "Interfaces/ImpulseRegionUtils.h"
#include "Interfaces/SICMRuleSetInterface.h"
#include "Interfaces/Utils.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include <optional>

using namespace mlir;
using namespace mlir::impulse;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SICMPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

/// True if every value of `values` is available above `regionOp`: it is
/// defined there, or is a block argument that `regionToOuter` maps, or a
/// result of an op of `toHoist`.
static bool checkOperandDominance(const IRMapping &regionToOuter,
                                  DominanceInfo &domInfo,
                                  InferRegionOp regionOp,
                                  const SetVector<Operation *> &toHoist,
                                  ValueRange values) {
  for (Value value : values) {
    if (domInfo.properlyDominates(value, regionOp))
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

/// True if an effect of `opEffects` and one of `stationaryEffects`, at least
/// one of them a write, may access the same memory.
static bool
hasMemoryConflict(ArrayRef<MemoryEffects::EffectInstance> opEffects,
                  ArrayRef<MemoryEffects::EffectInstance> stationaryEffects) {
  for (const MemoryEffects::EffectInstance &stationaryEffect :
       stationaryEffects) {
    for (const MemoryEffects::EffectInstance &opEffect : opEffects) {
      bool isConflict =
          (isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Read>(opEffect.getEffect())) ||
          (isa<MemoryEffects::Read>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Write>(opEffect.getEffect())) ||
          (isa<MemoryEffects::Write>(stationaryEffect.getEffect()) &&
           isa<MemoryEffects::Write>(opEffect.getEffect()));
      if (isConflict) {
        MemoryEffects::EffectInstance stationaryEffectCopy = stationaryEffect;
        MemoryEffects::EffectInstance opEffectCopy = opEffect;
        if (enzyme::oputils::mayAlias(opEffectCopy, stationaryEffectCopy))
          return true;
      }
    }
  }
  return false;
}

/// True if `op` is not a sample site or a terminator and neither it nor its
/// operands are sample-dependent.
static bool isHoistCandidate(const SampleDependenceAnalysis &analysis,
                             Operation *op) {
  if (isa<SampleRegionOp>(op) || op->hasTrait<OpTrait::IsTerminator>() ||
      analysis.isSampleDependent(op))
    return false;
  return llvm::none_of(op->getOperands(), [&](Value operand) {
    return analysis.isSampleDependent(operand);
  });
}

/// Moves above the infer_region of `analysis` every op of the target region
/// that is a hoist candidate, uses only values available above the
/// infer_region, is in a block that runs whenever the region runs, and has
/// no memory conflict with an op that stays. Returns true if it moved any.
static bool hoistSampleInvariantOps(SampleDependenceAnalysis &analysis) {
  Region &region = analysis.getTargetRegion();
  if (region.empty())
    return false;

  InferRegionOp regionOp = analysis.getRegionOp();
  DominanceInfo domInfo(regionOp);
  PostDominanceInfo postDomInfo(regionOp);

  IRMapping regionToOuter;
  if (analysis.getTarget() == AnalysisTarget::Sampling)
    for (auto [blockArg, input] :
         llvm::zip(region.front().getArguments(), regionOp.getInputs()))
      regionToOuter.map(blockArg, input);

  SetVector<Operation *> toHoist;
  SmallVector<MemoryEffects::EffectInstance> stationaryEffects;

  for (Block &block : region) {
    if (!postDomInfo.postDominates(&block, &region.front()))
      continue;

    for (Operation &op : block.without_terminator()) {
      bool canHoist = isHoistCandidate(analysis, &op);

      SmallVector<MemoryEffects::EffectInstance> opEffects;
      if (!enzyme::oputils::collectOpEffects(&op, opEffects))
        canHoist = false;

      if (canHoist)
        canHoist = checkOperandDominance(regionToOuter, domInfo, regionOp,
                                         toHoist, op.getOperands());

      if (canHoist && op.getNumRegions() > 0) {
        SetVector<Value> nestedValues;
        getUsedValuesDefinedAbove(op.getRegions(), nestedValues);
        canHoist = checkOperandDominance(regionToOuter, domInfo, regionOp,
                                         toHoist, nestedValues.getArrayRef());
      }

      if (canHoist)
        canHoist = !hasMemoryConflict(opEffects, stationaryEffects);

      if (!canHoist) {
        stationaryEffects.append(opEffects.begin(), opEffects.end());
        continue;
      }

      for (OpOperand &operand : op.getOpOperands())
        operand.assign(regionToOuter.lookupOrDefault(operand.get()));
      for (Region &nestedRegion : op.getRegions()) {
        SetVector<Value> nestedValues;
        getUsedValuesDefinedAbove(nestedRegion, nestedValues);
        for (Value value : nestedValues)
          if (regionToOuter.contains(value))
            replaceAllUsesInRegionWith(value, regionToOuter.lookup(value),
                                       nestedRegion);
      }
      toHoist.insert(&op);
    }
  }

  SmallVector<Operation *> sortedToHoist(toHoist.begin(), toHoist.end());
  llvm::sort(sortedToHoist, [&domInfo](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock())
      return a->isBeforeInBlock(b);
    return domInfo.dominates(a->getBlock(), b->getBlock());
  });

  for (Operation *op : sortedToHoist)
    op->moveBefore(regionOp);

  return !sortedToHoist.empty();
}

/// Hoists the constants under `root` and merges equal ones, as a greedy
/// rewrite of `root` would.
static void uniqueConstants(Operation *root) {
  OperationFolder folder(root->getContext());
  root->walk([&](Operation *op) {
    Attribute value;
    if (matchPattern(op, m_Constant(&value)))
      folder.insertKnownConstant(op, value);
  });
}

namespace {

/// Applies every IR change it is notified of to the analyses of `cache`.
struct SampleDependenceCacheUpdater : public RewriterBase::Listener {
  explicit SampleDependenceCacheUpdater(SampleDependenceCache &cache)
      : cache(cache) {}

  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint) override {
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationInserted(op);
    });
  }
  void notifyBlockInserted(Block *block, Region *, Region::iterator) override {
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyBlockInserted(block);
    });
  }
  void notifyBlockErased(Block *block) override {
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyBlockErased(block);
    });
  }
  void notifyOperationModified(Operation *op) override {
    if (isa<InferRegionOp>(op))
      cache.drop(op, /*onlyIfSourcesChanged=*/true);
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationModified(op);
    });
  }
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationReplaced(op, replacement);
    });
  }
  void notifyOperationErased(Operation *op) override {
    if (isa<InferRegionOp>(op))
      cache.drop(op, /*onlyIfSourcesChanged=*/false);
    cache.forEach([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationErased(op);
    });
  }

  SampleDependenceCache &cache;
};

/// Checks the incremental updates of the sample-dependence analyses: after
/// each change, which it first forwards to the updater, every analysis of the
/// cache must mark every value that a fresh analysis of the IR marks
/// sample-dependent; extra marks only lose precision. It also updates and
/// checks one analysis per non-empty region of the infer_region, so that the
/// updates are checked when no rule asks for an analysis. Reports the first
/// missed value as an error.
class IncrementalAnalysisChecker : public RewriterBase::ForwardingListener {
public:
  IncrementalAnalysisChecker(SampleDependenceCacheUpdater &updater,
                             InferRegionOp regionOp)
      : ForwardingListener(&updater), cache(updater.cache) {
    if (!regionOp.getSampling().empty())
      regionAnalyses.push_back(std::make_unique<SampleDependenceAnalysis>(
          regionOp, AnalysisTarget::Sampling));
    if (!regionOp.getLogpdf().empty())
      regionAnalyses.push_back(std::make_unique<SampleDependenceAnalysis>(
          regionOp, AnalysisTarget::Logpdf));
  }

  bool foundMissedValue() const { return missedValue; }

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    ForwardingListener::notifyOperationInserted(op, previous);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyOperationInserted(op);
    check("insertion", op);
  }
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator previousIt) override {
    ForwardingListener::notifyBlockInserted(block, previous, previousIt);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyBlockInserted(block);
    if (Operation *parent = block->getParentOp())
      check("block insertion", parent);
  }
  void notifyBlockErased(Block *block) override {
    ForwardingListener::notifyBlockErased(block);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyBlockErased(block);
    if (Operation *parent = block->getParentOp())
      check("block erasure", parent, block->getArguments());
  }
  void notifyOperationModified(Operation *op) override {
    ForwardingListener::notifyOperationModified(op);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyOperationModified(op);
    check("modification", op);
  }
  void notifyOperationReplaced(Operation *op, Operation *newOp) override {
    notifyOperationReplaced(op, newOp->getResults());
  }
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    ForwardingListener::notifyOperationReplaced(op, replacement);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyOperationReplaced(op, replacement);
    check("replacement", op);
  }
  void notifyOperationErased(Operation *op) override {
    ForwardingListener::notifyOperationErased(op);
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      analysis->notifyOperationErased(op);
    check("erasure", op, op->getResults());
  }

private:
  /// `erased` are the values an erasure is about to remove: they are still
  /// in the IR but no longer in the incremental analyses.
  void check(StringRef kind, Operation *op, ValueRange erased = {}) {
    if (missedValue)
      return;
    auto missesValue = [&](SampleDependenceAnalysis &incremental) {
      SampleDependenceAnalysis fresh(incremental.getRegionOp(),
                                     incremental.getTarget());
      return llvm::any_of(fresh.getSampleDependentValues(), [&](Value value) {
        return !incremental.isSampleDependent(value) &&
               !llvm::is_contained(erased, value);
      });
    };
    cache.forEach([&](SampleDependenceAnalysis &incremental) {
      missedValue = missedValue || missesValue(incremental);
    });
    for (std::unique_ptr<SampleDependenceAnalysis> &analysis : regionAnalyses)
      missedValue = missedValue || missesValue(*analysis);
    if (missedValue)
      op->emitError("sicm: the incremental sample-dependence analysis missed "
                    "a sample-dependent value after the ")
          << kind << " of '" << op->getName() << "'";
  }

  SampleDependenceCache &cache;
  SmallVector<std::unique_ptr<SampleDependenceAnalysis>> regionAnalyses;
  bool missedValue = false;
};

struct SICMPass : public enzyme::impl::SICMPassBase<SICMPass> {
  using SICMPassBase::SICMPassBase;

  std::shared_ptr<OwningOpRef<ModuleOp>> scheduleModule;
  transform::NamedSequenceOp scheduleEntry;
  SmallVector<const SICMRuleSetInterface *> ruleSets;
  SICMRuleOptions options;
  /// The rules of every rewrite step of a run, built at the first step.
  std::optional<FrozenRewritePatternSet> rules;

  void getDependentDialects(DialectRegistry &registry) const override {
    SICMPassBase::getDependentDialects(registry);
    if (!schedule.empty())
      registry.insert<transform::TransformDialect, pdl::PDLDialect,
                      pdl_interp::PDLInterpDialect>();
  }

  LogicalResult initialize(MLIRContext *context) override {
    if (schedule.empty())
      return success();
    OwningOpRef<ModuleOp> script =
        parseSourceFile<ModuleOp>(schedule.getValue(), ParserConfig(context));
    if (!script)
      return failure();
    (*script)->walk([&](transform::NamedSequenceOp sequence) {
      if (sequence.getSymName() != scheduleEntryPoint.getValue())
        return WalkResult::advance();
      scheduleEntry = sequence;
      return WalkResult::interrupt();
    });
    if (!scheduleEntry)
      return emitError((*script)->getLoc())
             << "sicm: the schedule has no transform.named_sequence @"
             << scheduleEntryPoint.getValue();
    scheduleModule = std::make_shared<OwningOpRef<ModuleOp>>(std::move(script));
    return success();
  }

  LogicalResult populateRules(RewritePatternSet &patterns) {
    if (scheduleEntry) {
      scheduleEntry.walk([&](transform::ApplyPatternsOp applyOp) {
        if (applyOp.getPatterns().empty())
          return;
        for (Operation &descriptor : applyOp.getPatterns().front())
          cast<transform::PatternDescriptorOpInterface>(descriptor)
              .populatePatterns(patterns);
      });
    } else {
      for (const SICMRuleSetInterface *ruleSet : ruleSets)
        ruleSet->populateRules(patterns);
    }
    for (const SICMRuleSetInterface *ruleSet : ruleSets)
      if (failed(ruleSet->configureRules(patterns, options, getOperation())))
        return failure();
    return success();
  }

  void runSICMOnRegion(InferRegionOp regionOp, AnalysisTarget target) {
    for (int64_t iter = 0; iter < maxIterations; ++iter) {
      bool anyChanged = false;

      if (!rules) {
        RewritePatternSet patterns(getOperation()->getContext());
        if (failed(populateRules(patterns))) {
          signalPassFailure();
          return;
        }
        rules.emplace(std::move(patterns));
      }
      SampleDependenceCache analysisCache;
      SampleDependenceCacheUpdater updater(analysisCache);
      std::optional<IncrementalAnalysisChecker> checker;
      if (verifyIncrementalAnalysis)
        checker.emplace(updater, regionOp);
      ScopedSampleDependenceCache scopedCache(analysisCache);
      GreedyRewriteConfig config;
      if (checker)
        config.setListener(&*checker);
      else
        config.setListener(&updater);
      SmallVector<Operation *> regionOps;
      for (Region &region : regionOp->getRegions())
        region.walk([&](Operation *op) { regionOps.push_back(op); });
      bool patternChanged = false;
      if (!regionOps.empty())
        (void)applyOpPatternsGreedily(regionOps, *rules, config,
                                      &patternChanged);
      anyChanged |= patternChanged;
      if (checker && checker->foundMissedValue())
        signalPassFailure();
      uniqueConstants(getOperation());

      SampleDependenceAnalysis hoistAnalysis(regionOp, target);
      anyChanged |= hoistSampleInvariantOps(hoistAnalysis);

      if (!anyChanged)
        break;
    }
  }

  /// Fills `options` from `rule-options`. Fails after reporting every entry
  /// that is not `key=value` or that no loaded rule set knows.
  LogicalResult parseRuleOptions() {
    options.clear();
    bool invalid = false;
    for (const std::string &entry : ruleOptions) {
      auto [key, value] = StringRef(entry).split('=');
      if (key.empty() || value.empty()) {
        getOperation()->emitError("sicm: rule-options entry '")
            << entry << "' is not of the form key=value";
        invalid = true;
        continue;
      }
      if (llvm::none_of(ruleSets, [&](const SICMRuleSetInterface *ruleSet) {
            return ruleSet->hasOption(key);
          })) {
        getOperation()->emitError("sicm: no loaded rule set has an option '")
            << key << "'";
        invalid = true;
        continue;
      }
      options[key] = value.str();
    }
    return failure(invalid);
  }

  void runOnOperation() override {
    rules.reset();
    ruleSets.clear();
    for (Dialect *dialect : getContext().getLoadedDialects())
      if (auto *ruleSet =
              dialect->getRegisteredInterface<SICMRuleSetInterface>())
        ruleSets.push_back(ruleSet);
    if (failed(parseRuleOptions()))
      return signalPassFailure();

    SmallVector<InferRegionOp> regions;
    getOperation()->walk([&](InferRegionOp op) { regions.push_back(op); });

    for (InferRegionOp regionOp : regions) {
      // `sampling` goes first: the `logpdf` region built from it next
      // inherits what it hoists.
      if (!regionOp.getSampling().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Sampling);

      constructLogpdfRegion(regionOp, [&](Operation *op) {
        return llvm::any_of(ruleSets, [&](const SICMRuleSetInterface *ruleSet) {
          return ruleSet->isHoistableFromBranch(op);
        });
      });

      if (!regionOp.getLogpdf().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Logpdf);
    }
    for (const SICMRuleSetInterface *ruleSet : ruleSets)
      ruleSet->finalize(getOperation(), options);
  }
};

} // end anonymous namespace
