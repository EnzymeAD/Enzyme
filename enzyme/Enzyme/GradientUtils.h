//===- GradientUtils.h - Helper class and utilities for AD       ---------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file declares two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_GUTILS_H_
#define ENZYME_GUTILS_H_

#include <functional>
#include <map>
#include <string>

#include <llvm/Config/llvm-config.h>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include "ActivityAnalysis.h"
#include "CacheUtility.h"
#include "EnzymeLogic.h"
#include "Utils.h"

#include "llvm-c/Core.h"

class GradientUtils;
extern llvm::StringMap<std::function<llvm::Value *(
    llvm::IRBuilder<> &, llvm::CallInst *, llvm::ArrayRef<llvm::Value *>,
    GradientUtils *)>>
    shadowHandlers;

class DiffeGradientUtils;
extern llvm::StringMap<std::pair<
    std::function<bool(llvm::IRBuilder<> &, llvm::CallInst *, GradientUtils &,
                       llvm::Value *&, llvm::Value *&, llvm::Value *&)>,
    std::function<void(llvm::IRBuilder<> &, llvm::CallInst *,
                       DiffeGradientUtils &, llvm::Value *)>>>
    customCallHandlers;

extern llvm::StringMap<
    std::function<bool(llvm::IRBuilder<> &, llvm::CallInst *, GradientUtils &,
                       llvm::Value *&, llvm::Value *&)>>
    customFwdCallHandlers;

extern "C" {
extern llvm::cl::opt<bool> EnzymeRuntimeActivityCheck;
extern llvm::cl::opt<bool> EnzymeInactiveDynamic;
extern llvm::cl::opt<bool> EnzymeFreeInternalAllocations;
extern llvm::cl::opt<bool> EnzymeRematerialize;
}
extern llvm::SmallVector<unsigned int, 9> MD_ToCopy;

struct InvertedPointerConfig : llvm::ValueMapConfig<const llvm::Value *> {
  typedef GradientUtils *ExtraData;
  static void onDelete(ExtraData gutils, const llvm::Value *old);
};

class InvertedPointerVH final : public llvm::CallbackVH {
public:
  GradientUtils *gutils;
  InvertedPointerVH(GradientUtils *gutils) : gutils(gutils) {}
  InvertedPointerVH(GradientUtils *gutils, llvm::Value *V)
      : InvertedPointerVH(gutils) {
    setValPtr(V);
  }
  void deleted() override final;

  void allUsesReplacedWith(llvm::Value *new_value) override final {
    setValPtr(new_value);
  }
  virtual ~InvertedPointerVH() {}
};

enum class AugmentedStruct;
class GradientUtils : public CacheUtility {
public:
  EnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  llvm::Function *oldFunc;
  llvm::ValueMap<const llvm::Value *, InvertedPointerVH> invertedPointers;
  llvm::DominatorTree &OrigDT;
  llvm::PostDominatorTree &OrigPDT;
  llvm::LoopInfo &OrigLI;
  llvm::ScalarEvolution &OrigSE;

  /// (Original) Blocks which dominate all returns
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> BlocksDominatingAllReturns;

  llvm::SmallPtrSet<llvm::BasicBlock *, 4> notForAnalysis;
  std::shared_ptr<ActivityAnalyzer> ATA;
  llvm::SmallVector<llvm::BasicBlock *, 12> originalBlocks;

  /// Allocations which are known to always be freed before the
  /// reverse, to the list of frees that must apply to this allocation.
  llvm::ValueMap<const llvm::CallInst *,
                 llvm::SmallPtrSet<const llvm::CallInst *, 1>>
      allocationsWithGuaranteedFree;

  /// Frees which can always be eliminated as the post dominate
  /// an allocation (which will itself be freed).
  llvm::SmallPtrSet<const llvm::CallInst *, 1> postDominatingFrees;

  /// Deallocations that should be kept in the forward pass because
  /// they deallocation memory which isn't necessary for the reverse
  /// pass
  llvm::SmallPtrSet<const llvm::CallInst *, 1> forwardDeallocations;

  /// Map of primal block to corresponding block(s) in reverse
  std::map<llvm::BasicBlock *, llvm::SmallVector<llvm::BasicBlock *, 4>>
      reverseBlocks;
  /// Map of block in reverse to corresponding primal block
  std::map<llvm::BasicBlock *, llvm::BasicBlock *> reverseBlockToPrimal;

  /// A set of tape extractions to enforce a cache of
  /// rather than attempting to recompute.
  llvm::SmallPtrSet<llvm::Instruction *, 4> TapesToPreventRecomputation;

  llvm::ValueMap<llvm::PHINode *, llvm::WeakTrackingVH> fictiousPHIs;
  llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> originalToNewFn;
  llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> newToOriginalFn;
  llvm::SmallVector<llvm::CallInst *, 4> originalCalls;

  llvm::SmallPtrSet<llvm::Instruction *, 4> unnecessaryIntermediates;

  const std::map<llvm::Instruction *, bool> *can_modref_map;
  const std::map<llvm::CallInst *, const std::vector<bool>>
      *overwritten_args_map_ptr;
  const llvm::SmallPtrSetImpl<const llvm::Value *> *unnecessaryValuesP;

  llvm::SmallVector<llvm::OperandBundleDef, 2> getInvertedBundles(
      llvm::CallInst *orig, llvm::ArrayRef<ValueType> types,
      llvm::IRBuilder<> &Builder2, bool lookup,
      const llvm::ValueToValueMapTy &available = llvm::ValueToValueMapTy());

  llvm::Value *getNewIfOriginal(llvm::Value *originst) const;

  llvm::Value *tid;
  llvm::Value *ompThreadId();

  llvm::Value *numThreads;
  llvm::Value *ompNumThreads();

  llvm::Value *getOrInsertTotalMultiplicativeProduct(llvm::Value *val,
                                                     LoopContext &lc);

  llvm::Value *getOrInsertConditionalIndex(llvm::Value *val, LoopContext &lc,
                                           bool pickTrue);

  bool assumeDynamicLoopOfSizeOne(llvm::Loop *L) const override;

  llvm::DebugLoc getNewFromOriginal(const llvm::DebugLoc L) const;

  llvm::Value *getNewFromOriginal(const llvm::Value *originst) const;

  llvm::Instruction *getNewFromOriginal(const llvm::Instruction *newinst) const;

  llvm::BasicBlock *getNewFromOriginal(const llvm::BasicBlock *newinst) const;

  llvm::Value *hasUninverted(const llvm::Value *inverted) const;

  llvm::BasicBlock *getOriginalFromNew(const llvm::BasicBlock *newinst) const;

  llvm::Value *isOriginal(const llvm::Value *newinst) const;

  llvm::Instruction *isOriginal(const llvm::Instruction *newinst) const;

  llvm::BasicBlock *isOriginal(const llvm::BasicBlock *newinst) const;

  struct LoadLikeCall {
    llvm::CallInst *loadCall;
    llvm::Value *operand;
    LoadLikeCall() = default;
    LoadLikeCall(llvm::CallInst *a, llvm::Value *b) : loadCall(a), operand(b) {}
  };

  struct Rematerializer {
    // Loads which may need to be rematerialized.
    llvm::SmallVector<llvm::LoadInst *, 1> loads;

    // Loads-like calls which need the memory initialized for the reverse.
    llvm::SmallVector<LoadLikeCall, 1> loadLikeCalls;

    // Operations which must be rerun to rematerialize
    // the value.
    llvm::SmallPtrSet<llvm::Instruction *, 1> stores;

    // Operations which deallocate the value.
    llvm::SmallPtrSet<llvm::Instruction *, 1> frees;

    // Loop scope (null if not loop scoped).
    llvm::Loop *LI;

    Rematerializer() : loads(), stores(), frees(), LI(nullptr) {}
    Rematerializer(llvm::ArrayRef<llvm::LoadInst *> loads,
                   llvm::ArrayRef<LoadLikeCall> loadLikeCalls,
                   const llvm::SmallPtrSetImpl<llvm::Instruction *> &stores,
                   const llvm::SmallPtrSetImpl<llvm::Instruction *> &frees,
                   llvm::Loop *LI)
        : loads(loads.begin(), loads.end()),
          loadLikeCalls(loadLikeCalls.begin(), loadLikeCalls.end()),
          stores(stores.begin(), stores.end()),
          frees(frees.begin(), frees.end()), LI(LI) {}
  };

  struct ShadowRematerializer {
    /// Operations which must be rerun to rematerialize
    /// the original value.
    llvm::SmallPtrSet<llvm::Instruction *, 1> stores;

    /// Operations which deallocate the value.
    llvm::SmallPtrSet<llvm::Instruction *, 1> frees;

    /// Whether the shadow must be initialized in the primal.
    bool primalInitialize;

    /// Loop scope (null if not loop scoped).
    llvm::Loop *LI;

    ShadowRematerializer()
        : stores(), frees(), primalInitialize(), LI(nullptr) {}
    ShadowRematerializer(
        const llvm::SmallPtrSetImpl<llvm::Instruction *> &stores,
        const llvm::SmallPtrSetImpl<llvm::Instruction *> &frees,
        bool primalInitialize, llvm::Loop *LI)
        : stores(stores.begin(), stores.end()),
          frees(frees.begin(), frees.end()), primalInitialize(primalInitialize),
          LI(LI) {}
  };

  llvm::ValueMap<llvm::Value *, Rematerializer> rematerializableAllocations;

  /// Only loaded from and stored to (not captured), mapped to the stores (and
  /// memset). Boolean denotes whether the primal initializes the shadow as well
  /// (for use) as a structure which carries data.
  llvm::ValueMap<llvm::Value *, ShadowRematerializer> backwardsOnlyShadows;

  void computeForwardingProperties(llvm::Instruction *V);
  void computeGuaranteedFrees();

private:
  llvm::SmallVector<llvm::WeakTrackingVH, 4> addedTapeVals;
  unsigned tapeidx;
  llvm::Value *tape;

  std::map<llvm::BasicBlock *,
           llvm::ValueMap<llvm::Value *,
                          std::map<llvm::BasicBlock *, llvm::WeakTrackingVH>>>
      unwrap_cache;
  std::map<llvm::BasicBlock *,
           llvm::ValueMap<llvm::Value *, llvm::WeakTrackingVH>>
      lookup_cache;

public:
  void replaceAndRemoveUnwrapCacheFor(llvm::Value *A, llvm::Value *B);

  llvm::BasicBlock *addReverseBlock(llvm::BasicBlock *currentBlock,
                                    llvm::Twine const &name,
                                    bool forkCache = true, bool push = true);

  bool legalRecompute(const llvm::Value *val,
                      const llvm::ValueToValueMapTy &available,
                      llvm::IRBuilder<> *BuilderM, bool reverse = false,
                      bool legalRecomputeCache = true) const;

  std::map<const llvm::Value *, bool> knownRecomputeHeuristic;
  bool shouldRecompute(const llvm::Value *val,
                       const llvm::ValueToValueMapTy &available,
                       llvm::IRBuilder<> *BuilderM);

  llvm::ValueMap<const llvm::Instruction *, AssertingReplacingVH>
      unwrappedLoads;

  void replaceAWithB(llvm::Value *A, llvm::Value *B,
                     bool storeInCache = false) override;

  void erase(llvm::Instruction *I) override;

  void eraseWithPlaceholder(llvm::Instruction *I,
                            const llvm::Twine &suffix = "_replacementA",
                            bool erase = true);

  // TODO consider invariant group and/or valueInvariant group

  void setTape(llvm::Value *newtape);

  void dumpPointers();

  int getIndex(
      std::pair<llvm::Instruction *, CacheType> idx,
      const std::map<std::pair<llvm::Instruction *, CacheType>, int> &mapping);

  int getIndex(
      std::pair<llvm::Instruction *, CacheType> idx,
      std::map<std::pair<llvm::Instruction *, CacheType>, int> &mapping);

  llvm::Value *cacheForReverse(llvm::IRBuilder<> &BuilderQ, llvm::Value *malloc,
                               int idx, bool replace = true);

  llvm::ArrayRef<llvm::WeakTrackingVH> getTapeValues() const {
    return addedTapeVals;
  }

public:
  llvm::AAResults &OrigAA;
  TypeAnalysis &TA;
  TypeResults TR;
  bool omp;

private:
  unsigned width;

public:
  unsigned getWidth() { return width; }

  llvm::ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

public:
  GradientUtils(EnzymeLogic &Logic, llvm::Function *newFunc_,
                llvm::Function *oldFunc_, llvm::TargetLibraryInfo &TLI_,
                TypeAnalysis &TA_, TypeResults TR_,
                llvm::ValueToValueMapTy &invertedPointers_,
                const llvm::SmallPtrSetImpl<llvm::Value *> &constantvalues_,
                const llvm::SmallPtrSetImpl<llvm::Value *> &activevals_,
                DIFFE_TYPE ReturnActivity,
                llvm::ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                llvm::ValueMap<const llvm::Value *, AssertingReplacingVH>
                    &originalToNewFn_,
                DerivativeMode mode, unsigned width, bool omp);

public:
  DIFFE_TYPE getDiffeType(llvm::Value *v, bool foreignFunction) const;

  DIFFE_TYPE getReturnDiffeType(llvm::Value *orig, bool *primalReturnUsedP,
                                bool *shadowReturnUsedP) const;

  static GradientUtils *
  CreateFromClone(EnzymeLogic &Logic, unsigned width, llvm::Function *todiff,
                  llvm::TargetLibraryInfo &TLI, TypeAnalysis &TA,
                  FnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  llvm::ArrayRef<DIFFE_TYPE> constant_args, bool returnUsed,
                  bool shadowReturnUsed,
                  std::map<AugmentedStruct, int> &returnMapping, bool omp);

  llvm::ValueMap<const llvm::Value *, llvm::MDNode *>
      differentialAliasScopeDomains;
  llvm::ValueMap<const llvm::Value *, llvm::DenseMap<ssize_t, llvm::MDNode *>>
      differentialAliasScope;
  llvm::MDNode *getDerivativeAliasScope(const llvm::Value *origptr,
                                        ssize_t newptr);

#if LLVM_VERSION_MAJOR >= 10
  void setPtrDiffe(llvm::Instruction *orig, llvm::Value *ptr,
                   llvm::Value *newval, llvm::IRBuilder<> &BuilderM,
                   llvm::MaybeAlign align, bool isVolatile,
                   llvm::AtomicOrdering ordering, llvm::SyncScope::ID syncScope,
                   llvm::Value *mask, llvm::ArrayRef<llvm::Metadata *> noAlias,
                   llvm::ArrayRef<llvm::Metadata *> scopes);
#else
  void setPtrDiffe(llvm::Instruction *orig, llvm::Value *ptr,
                   llvm::Value *newval, llvm::IRBuilder<> &BuilderM,
                   unsigned align, bool isVolatile,
                   llvm::AtomicOrdering ordering, llvm::SyncScope::ID syncScope,
                   llvm::Value *mask, llvm::ArrayRef<llvm::Metadata *> noAlias,
                   llvm::ArrayRef<llvm::Metadata *> scopes);
#endif

private:
  llvm::BasicBlock *originalForReverseBlock(llvm::BasicBlock &BB2) const;

public:
  //! This cache stores blocks we may insert as part of getReverseOrLatchMerge
  //! to handle inverse iv iteration
  //  As we don't want to create redundant blocks, we use this convenient cache
  std::map<std::tuple<llvm::BasicBlock *, llvm::BasicBlock *>,
           llvm::BasicBlock *>
      newBlocksForLoop_cache;

  //! This cache stores a rematerialized forward pass in the loop
  //! specified
  std::map<llvm::Loop *, llvm::BasicBlock *> rematerializedLoops_cache;
  llvm::BasicBlock *getReverseOrLatchMerge(llvm::BasicBlock *BB,
                                           llvm::BasicBlock *branchingBlock);

  void forceContexts();

  void computeMinCache();

  bool isOriginalBlock(const llvm::BasicBlock &BB) const;

  llvm::SmallVector<llvm::Instruction *, 1>
      rematerializedPrimalOrShadowAllocations;

  void eraseFictiousPHIs();

  void forceActiveDetection();

  bool isConstantValue(llvm::Value *val) const;

  bool isConstantInstruction(const llvm::Instruction *inst) const;

  bool getContext(llvm::BasicBlock *BB, LoopContext &lc);

  void forceAugmentedReturns();

private:
  // For a given value, a list of basic blocks where an unwrap to has already
  // produced a warning.
  std::map<llvm::Instruction *, std::set<llvm::BasicBlock *>> UnwrappedWarnings;

public:
  /// if full unwrap, don't just unwrap this instruction, but also its operands,
  /// etc
  llvm::Value *unwrapM(llvm::Value *const val, llvm::IRBuilder<> &BuilderM,
                       const llvm::ValueToValueMapTy &available,
                       UnwrapMode unwrapMode, llvm::BasicBlock *scope = nullptr,
                       bool permitCache = true) override final;

  void ensureLookupCached(llvm::Instruction *inst, bool shouldFree = true,
                          llvm::BasicBlock *scope = nullptr,
                          llvm::MDNode *TBAA = nullptr);

  std::map<llvm::Instruction *,
           llvm::ValueMap<llvm::BasicBlock *, llvm::WeakTrackingVH>>
      lcssaFixes;
  std::map<llvm::PHINode *, llvm::WeakTrackingVH> lcssaPHIToOrig;
  llvm::Value *fixLCSSA(llvm::Instruction *inst, llvm::BasicBlock *forwardBlock,
                        bool legalInBlock = false);

  llvm::Value *lookupM(llvm::Value *val, llvm::IRBuilder<> &BuilderM,
                       const llvm::ValueToValueMapTy &incoming_availalble =
                           llvm::ValueToValueMapTy(),
                       bool tryLegalRecomputeCheck = true,
                       llvm::BasicBlock *scope = nullptr) override;

  llvm::Value *invertPointerM(llvm::Value *val, llvm::IRBuilder<> &BuilderM,
                              bool nullShadow = false);

  static llvm::Constant *GetOrCreateShadowConstant(
      EnzymeLogic &Logic, llvm::TargetLibraryInfo &TLI, TypeAnalysis &TA,
      llvm::Constant *F, DerivativeMode mode, unsigned width, bool AtomicAdd);

  static llvm::Constant *GetOrCreateShadowFunction(
      EnzymeLogic &Logic, llvm::TargetLibraryInfo &TLI, TypeAnalysis &TA,
      llvm::Function *F, DerivativeMode mode, unsigned width, bool AtomicAdd);

  void branchToCorrespondingTarget(
      llvm::BasicBlock *ctx, llvm::IRBuilder<> &BuilderM,
      const std::map<llvm::BasicBlock *,
                     std::vector<std::pair</*pred*/ llvm::BasicBlock *,
                                           /*successor*/ llvm::BasicBlock *>>>
          &targetToPreds,
      const std::map<llvm::BasicBlock *, llvm::PHINode *> *replacePHIs =
          nullptr);

  void getReverseBuilder(llvm::IRBuilder<> &Builder2, bool original = true);

  void getForwardBuilder(llvm::IRBuilder<> &Builder2);

  static llvm::Type *getShadowType(llvm::Type *ty, unsigned width);

  llvm::Type *getShadowType(llvm::Type *ty);

  static llvm::Value *extractMeta(llvm::IRBuilder<> &Builder, llvm::Value *Agg,
                                  unsigned off, const llvm::Twine &name = "");
  static llvm::Value *extractMeta(llvm::IRBuilder<> &Builder, llvm::Value *Agg,
                                  llvm::ArrayRef<unsigned> off,
                                  const llvm::Twine &name = "");

  static llvm::Value *recursiveFAdd(llvm::IRBuilder<> &B, llvm::Value *lhs,
                                    llvm::Value *rhs,
                                    llvm::ArrayRef<unsigned> lhs_off = {},
                                    llvm::ArrayRef<unsigned> rhs_off = {},
                                    llvm::Value *prev = nullptr,
                                    bool vectorLayer = false);

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element. Return values of f are collected and wrapped.
  template <typename Func, typename... Args>
  llvm::Value *applyChainRule(llvm::Type *diffType, llvm::IRBuilder<> &Builder,
                              Func rule, Args... args) {
    if (width > 1) {
      const int size = sizeof...(args);
      llvm::Value *vals[size] = {args...};

      for (size_t i = 0; i < size; ++i)
        if (vals[i])
          assert(llvm::cast<llvm::ArrayType>(vals[i]->getType())
                     ->getNumElements() == width);

      llvm::Type *wrappedType = llvm::ArrayType::get(diffType, width);
      llvm::Value *res = llvm::UndefValue::get(wrappedType);
      for (unsigned int i = 0; i < getWidth(); ++i) {
        auto tup = std::tuple<Args...>{
            (args ? extractMeta(Builder, args, i) : nullptr)...};
        auto diff = std::apply(rule, std::move(tup));
        res = Builder.CreateInsertValue(res, diff, {i});
      }
      return res;
    } else {
      return rule(args...);
    }
  }

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element. Return values of f are collected and wrapped.
  template <typename Func, typename... Args>
  void applyChainRule(llvm::IRBuilder<> &Builder, Func rule, Args... args) {
    if (width > 1) {
      const int size = sizeof...(args);
      llvm::Value *vals[size] = {args...};

      for (size_t i = 0; i < size; ++i)
        if (vals[i])
          assert(llvm::cast<llvm::ArrayType>(vals[i]->getType())
                     ->getNumElements() == width);

      for (unsigned int i = 0; i < getWidth(); ++i) {
        auto tup = std::tuple<Args...>{
            (args ? extractMeta(Builder, args, i) : nullptr)...};
        std::apply(rule, std::move(tup));
      }
    } else {
      rule(args...);
    }
  }

  /// Unwraps an collection of constant vector derivatives from their internal
  /// representations and applies a function f to each element.
  template <typename Func>
  llvm::Value *applyChainRule(llvm::Type *diffType,
                              llvm::ArrayRef<llvm::Constant *> diffs,
                              llvm::IRBuilder<> &Builder, Func rule) {
    if (width > 1) {
      for (auto diff : diffs) {
        assert(diff);
        assert(llvm::cast<llvm::ArrayType>(diff->getType())->getNumElements() ==
               width);
      }
      llvm::Type *wrappedType = llvm::ArrayType::get(diffType, width);
      llvm::Value *res = llvm::UndefValue::get(wrappedType);
      for (unsigned int i = 0; i < getWidth(); ++i) {
        llvm::SmallVector<llvm::Constant *, 3> extracted_diffs;
        for (auto diff : diffs) {
          extracted_diffs.push_back(
              llvm::cast<llvm::Constant>(extractMeta(Builder, diff, i)));
        }
        auto diff = rule(extracted_diffs);
        res = Builder.CreateInsertValue(res, diff, {i});
      }
      return res;
    } else {
      return rule(diffs);
    }
  }

  bool needsCacheWholeAllocation(const llvm::Value *V) const;
};

void SubTransferHelper(GradientUtils *gutils, DerivativeMode Mode,
                       llvm::Type *secretty, llvm::Intrinsic::ID intrinsic,
                       unsigned dstalign, unsigned srcalign, unsigned offset,
                       bool dstConstant, llvm::Value *shadow_dst,
                       bool srcConstant, llvm::Value *shadow_src,
                       llvm::Value *length, llvm::Value *isVolatile,
                       llvm::CallInst *MTI, bool allowForward = true,
                       bool shadowsLookedUp = false,
                       bool backwardsShadow = false);
#endif
