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
#include "Passes/Passes.h"

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <map>
#include <optional>

using namespace mlir;
using namespace mlir::impulse;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SICMPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

using SharedSampleDependenceAnalysis =
    std::shared_ptr<SampleDependenceAnalysis>;

/// Sound iff every value a fresh analysis marks sample-dependent is also
/// marked by the incremental one; extra marks only lose precision.
class IncrementalAnalysisVerifier {
public:
  IncrementalAnalysisVerifier(SampleDependenceCache &cache,
                              InferRegionOp regionOp)
      : cache(cache), regionOp(regionOp) {
    if (!regionOp.getSampling().empty())
      shadows.push_back(std::make_shared<SampleDependenceAnalysis>(
          regionOp, AnalysisTarget::Sampling));
    if (!regionOp.getLogpdf().empty())
      shadows.push_back(std::make_shared<SampleDependenceAnalysis>(
          regionOp, AnalysisTarget::Logpdf));
    for (auto &shadow : shadows)
      fresh[keyOf(*shadow)] = shadow;
  }

  template <typename Fn> void forEachTracked(Fn fn) {
    cache.forEach(fn);
    for (auto &shadow : shadows)
      fn(*shadow);
  }
  template <typename Fn> void forEachShadow(Fn fn) {
    for (auto &shadow : shadows)
      fn(*shadow);
  }

  void check(StringRef kind, Operation *op, ValueRange excluded = {}) {
    ++numChecks;
    fresh.clear();
    forEachTracked([&](SampleDependenceAnalysis &incremental) {
      SampleDependenceAnalysis &exact = getFresh(incremental);
      for (Value value : exact.getSampleDependentValues()) {
        if (incremental.isSampleDependent(value) ||
            llvm::is_contained(excluded, value))
          continue;
        ++numViolations;
        if (numViolations <= 20) {
          llvm::errs() << "sicm-verify-incremental: UNSOUND after " << kind
                       << " of '" << op->getName() << "'";
          if (currentPattern)
            llvm::errs() << " (rule " << patternName(*currentPattern) << ')';
          llvm::errs() << ": sample-dependent value not in the incremental "
                          "analysis: ";
          printValue(value);
          llvm::errs() << '\n';
        }
      }
      uint64_t stale = 0;
      for (Value value : incremental.getSampleDependentValues())
        if (!exact.isSampleDependent(value))
          ++stale;
      maxStale = std::max(maxStale, stale);
    });
    // Erasures are notified before the erase, so the fresh analyses go stale.
    if (!excluded.empty() || kind.ends_with("erasure"))
      fresh.clear();
  }

  void classifyReplacement(Operation *op, ValueRange replacement) {
    auto it = llvm::find_if(
        shadows, [&](auto &shadow) { return shadow->isInTargetRegion(op); });
    if (it == shadows.end())
      return;
    SampleDependenceAnalysis exact((*it)->getRegionOp(), (*it)->getTarget());
    for (auto [result, newValue] : llvm::zip(op->getResults(), replacement)) {
      if (!newValue)
        continue;
      bool before = exact.isSampleDependent(result);
      bool after = exact.isSampleDependent(newValue);
      if (before == after) {
        ++numSameLabel;
        continue;
      }
      ++(after ? numAddsDependence : numRemovesDependence);
      llvm::errs() << "sicm-verify-incremental: replacement "
                   << (after ? "adds" : "removes") << " a dependence: '"
                   << op->getName() << "' by "
                   << (currentPattern ? patternName(*currentPattern)
                                      : std::string("folding"))
                   << '\n';
    }
  }

  void patternBegin(const Pattern &pattern) {
    currentPattern = &pattern;
    windowStart.clear();
    for (auto &shadow : shadows)
      windowStart.push_back(getFresh(*shadow).getSampleDependentValues());
    created.clear();
    erased.clear();
  }

  void patternEnd(const Pattern &pattern, LogicalResult status) {
    currentPattern = nullptr;
    if (failed(status))
      return;
    ++numApplications;
    for (auto [shadow, before] : llvm::zip(shadows, windowStart)) {
      SampleDependenceAnalysis &exact = getFresh(*shadow);
      SmallVector<std::pair<Value, bool>> changed;
      for (Value value : before)
        if (!erased.contains(value) && !exact.isSampleDependent(value))
          changed.push_back({value, false});
      for (Value value : exact.getSampleDependentValues())
        if (!created.contains(value) && !before.contains(value))
          changed.push_back({value, true});
      if (changed.empty())
        continue;
      ++numLabelChangingApplications;
      llvm::errs() << "sicm-verify-incremental: rule " << patternName(pattern)
                   << " changes the label of " << changed.size()
                   << " surviving value(s)\n";
      for (auto [value, nowDependent] : changed) {
        llvm::errs() << "sicm-verify-incremental:   "
                     << (nowDependent ? "now dependent: " : "now invariant: ");
        printValue(value);
        llvm::errs() << '\n';
      }
    }
  }

  static void printValue(Value value) {
    if (Operation *def = value.getDefiningOp()) {
      llvm::errs() << "result " << cast<OpResult>(value).getResultNumber()
                   << " of ";
      def->print(llvm::errs(), OpPrintingFlags().skipRegions());
      return;
    }
    auto arg = cast<BlockArgument>(value);
    llvm::errs() << "block argument " << arg.getArgNumber() << " in '"
                 << arg.getOwner()->getParentOp()->getName() << "'";
  }

  void noteCreated(Operation *op) {
    op->walk([&](Operation *nested) {
      for (Value result : nested->getResults())
        created.insert(result);
      for (Region &region : nested->getRegions())
        for (Block &block : region)
          for (Value arg : block.getArguments())
            created.insert(arg);
    });
  }
  void noteCreated(Block *block) {
    for (Value arg : block->getArguments())
      created.insert(arg);
  }
  void noteErased(Operation *op) {
    for (Value result : op->getResults())
      erased.insert(result);
  }
  void noteErased(Block *block) {
    for (Value arg : block->getArguments())
      erased.insert(arg);
  }
  void noteDroppedEntries(unsigned n) { numDroppedEntries += n; }

  bool report(StringRef targetName, int64_t round) {
    uint64_t staleAtEnd = 0, values = 0;
    SampleDependenceAnalysis::UpdateStats work;
    fresh.clear();
    forEachShadow([&](SampleDependenceAnalysis &incremental) {
      SampleDependenceAnalysis &exact = getFresh(incremental);
      for (Value value : incremental.getSampleDependentValues())
        if (!exact.isSampleDependent(value))
          ++staleAtEnd;
      incremental.getTargetRegion().walk([&](Operation *op) {
        values += op->getNumResults();
        for (Region &region : op->getRegions())
          for (Block &block : region)
            values += block.getNumArguments();
      });
      for (Block &block : incremental.getTargetRegion())
        values += block.getNumArguments();
      const auto &s = incremental.getUpdateStats();
      work.raised += s.raised;
      work.useVisits += s.useVisits;
      work.ancestorSteps += s.ancestorSteps;
      work.opsVisited += s.opsVisited;
    });
    llvm::errs() << "sicm-verify-incremental: " << targetName << " round "
                 << round << ": " << numChecks << " checks, " << numViolations
                 << " unsound, stale " << staleAtEnd << " at end (max "
                 << maxStale << "); replacements " << numSameLabel
                 << " same label, " << numAddsDependence << " add, "
                 << numRemovesDependence << " remove; " << numApplications
                 << " rule applications, " << numLabelChangingApplications
                 << " change a label; " << numDroppedEntries
                 << " rebuilt; work " << work.raised << " raised, "
                 << work.useVisits << " use visits, " << work.ancestorSteps
                 << " ancestor steps, " << work.opsVisited << " transfers over "
                 << values << " values\n";
    return numViolations != 0;
  }

private:
  using Key = std::pair<Operation *, AnalysisTarget>;
  static Key keyOf(SampleDependenceAnalysis &analysis) {
    return {analysis.getRegionOp().getOperation(), analysis.getTarget()};
  }
  SampleDependenceAnalysis &getFresh(SampleDependenceAnalysis &incremental) {
    auto &entry = fresh[keyOf(incremental)];
    if (!entry)
      entry = std::make_shared<SampleDependenceAnalysis>(
          incremental.getRegionOp(), incremental.getTarget());
    return *entry;
  }
  static std::string patternName(const Pattern &pattern) {
    if (!pattern.getDebugName().empty())
      return pattern.getDebugName().str();
    if (std::optional<OperationName> root = pattern.getRootKind())
      return ("<pattern rooted at " + root->getStringRef() + ">").str();
    return "<pattern>";
  }

  SampleDependenceCache &cache;
  InferRegionOp regionOp;
  SmallVector<SharedSampleDependenceAnalysis> shadows;
  std::map<Key, SharedSampleDependenceAnalysis> fresh;

  const Pattern *currentPattern = nullptr;
  SmallVector<DenseSet<Value>> windowStart;
  DenseSet<Value> created, erased;

  uint64_t numChecks = 0, numViolations = 0, maxStale = 0;
  uint64_t numSameLabel = 0, numAddsDependence = 0, numRemovesDependence = 0;
  uint64_t numApplications = 0, numLabelChangingApplications = 0;
  uint64_t numDroppedEntries = 0;
};

struct SampleDependenceCacheUpdater : public RewriterBase::Listener {
  SampleDependenceCacheUpdater(SampleDependenceCache &cache,
                               IncrementalAnalysisVerifier *verifier)
      : cache(cache), verifier(verifier) {}

  template <typename Fn> void forEachAnalysis(Fn fn) {
    if (verifier)
      verifier->forEachTracked(fn);
    else
      cache.forEach(fn);
  }

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationInserted(op);
    });
    if (verifier) {
      if (!previous.isSet())
        verifier->noteCreated(op);
      verifier->check("insertion", op);
    }
  }
  void notifyBlockInserted(Block *block, Region *previous,
                           Region::iterator) override {
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyBlockInserted(block);
    });
    if (verifier) {
      if (!previous)
        verifier->noteCreated(block);
      if (Operation *parent = block->getParentOp())
        verifier->check("block insertion", parent);
    }
  }
  void notifyBlockErased(Block *block) override {
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyBlockErased(block);
    });
    if (verifier) {
      verifier->noteErased(block);
      if (Operation *parent = block->getParentOp())
        verifier->check("block erasure", parent, block->getArguments());
    }
  }
  void notifyOperationModified(Operation *op) override {
    if (isa<InferRegionOp>(op)) {
      unsigned dropped = cache.drop(op, /*onlyIfSourcesChanged=*/true);
      if (verifier)
        verifier->noteDroppedEntries(dropped);
    }
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationModified(op);
    });
    if (verifier)
      verifier->check("modification", op);
  }
  void notifyOperationReplaced(Operation *op, ValueRange replacement) override {
    if (verifier)
      verifier->classifyReplacement(op, replacement);
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationReplaced(op, replacement);
    });
    if (verifier)
      verifier->check("replacement", op);
  }
  void notifyOperationErased(Operation *op) override {
    if (isa<InferRegionOp>(op)) {
      unsigned dropped = cache.drop(op, /*onlyIfSourcesChanged=*/false);
      if (verifier)
        verifier->noteDroppedEntries(dropped);
    }
    forEachAnalysis([&](SampleDependenceAnalysis &analysis) {
      analysis.notifyOperationErased(op);
    });
    if (verifier) {
      verifier->noteErased(op);
      verifier->check("erasure", op, op->getResults());
    }
  }
  void notifyPatternBegin(const Pattern &pattern, Operation *) override {
    if (verifier)
      verifier->patternBegin(pattern);
  }
  void notifyPatternEnd(const Pattern &pattern, LogicalResult status) override {
    if (verifier)
      verifier->patternEnd(pattern, status);
  }

private:
  SampleDependenceCache &cache;
  IncrementalAnalysisVerifier *verifier;
};

struct SICMPass : public enzyme::impl::SICMPassBase<SICMPass> {
  using SICMPassBase::SICMPassBase;

  std::shared_ptr<OwningOpRef<ModuleOp>> scheduleModule;
  transform::NamedSequenceOp scheduleEntry;
  SmallVector<const SICMRuleSetInterface *> ruleSets;
  SICMRuleOptions options;

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

  bool runSICMOnRegion(InferRegionOp regionOp, AnalysisTarget target) {
    bool everChanged = false;
    for (int64_t iter = 0; iter < maxIterations; ++iter) {
      bool anyChanged = false;

      RewritePatternSet patterns(getOperation()->getContext());
      if (failed(populateRules(patterns))) {
        signalPassFailure();
        return everChanged;
      }
      SampleDependenceCache analysisCache;
      std::optional<IncrementalAnalysisVerifier> verifier;
      if (verifyIncrementalAnalysis)
        verifier.emplace(analysisCache, regionOp);
      SampleDependenceCacheUpdater updater(analysisCache,
                                           verifier ? &*verifier : nullptr);
      ScopedSampleDependenceCache scopedCache(analysisCache);
      GreedyRewriteConfig config;
      config.setListener(&updater);
      SmallVector<Operation *> regionOps;
      for (Region &region : regionOp->getRegions())
        region.walk([&](Operation *op) { regionOps.push_back(op); });
      bool patternChanged = false;
      if (!regionOps.empty())
        (void)applyOpPatternsGreedily(regionOps, std::move(patterns), config,
                                      &patternChanged);
      anyChanged |= patternChanged;
      if (verifier &&
          verifier->report(
              target == AnalysisTarget::Logpdf ? "logpdf" : "sampler", iter)) {
        regionOp->emitError("sicm: the incremental sample-dependence "
                            "analysis missed a sample-dependent value");
        signalPassFailure();
      }
      // CSE and hoist constants, as a module-wide greedy rewrite would.
      {
        OperationFolder folder(getOperation()->getContext());
        getOperation()->walk([&](Operation *op) {
          Attribute value;
          if (matchPattern(op, m_Constant(&value)))
            folder.insertKnownConstant(op, value);
        });
      }

      anyChanged |= hoistSampleInvariantOps(regionOp, target);

      everChanged |= anyChanged;
      if (!anyChanged)
        break;
    }
    return everChanged;
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
      // Sampler first: what it hoists is inherited by the logpdf region built
      // next. The analysis needs a sampler entry block (none in custom logpdf).
      if (!regionOp.getSampling().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Sampling);

      constructLogpdfRegion(regionOp);

      if (!regionOp.getLogpdf().empty())
        runSICMOnRegion(regionOp, AnalysisTarget::Logpdf);
    }
    for (const SICMRuleSetInterface *ruleSet : ruleSets)
      ruleSet->finalize(getOperation(), options);
  }
};

} // end anonymous namespace
