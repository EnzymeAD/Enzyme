//===- CacheUtility.h - Caching values in the forward pass for later use
//---===//
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
// This file declares a base helper class CacheUtility that manages the cache
// of values from the forward pass for later use.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_CACHE_UTILITY_H
#define ENZYME_CACHE_UTILITY_H

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"

#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "FunctionUtils.h"
#include "MustExitScalarEvolution.h"

/// Container for all loop information to synthesize gradients
struct LoopContext {
  /// Canonical induction variable of the loop
  llvm::PHINode *var;

  /// Increment of the induction
  llvm::Instruction *incvar;

  /// Allocation of induction variable of reverse pass
  llvm::AllocaInst *antivaralloc;

  /// Header of this loop
  llvm::BasicBlock *header;

  /// Preheader of this loop
  llvm::BasicBlock *preheader;

  /// Whether this loop has a statically analyzable number of iterations
  bool dynamic;

  /// limit is last value of a canonical induction variable
  /// iters is number of times loop is run (thus iters = limit + 1)
  llvm::Value *maxLimit;

  llvm::Value *trueLimit;

  /// All blocks this loop exits too
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> exitBlocks;

  /// Parent loop of this loop
  llvm::Loop *parent;
};
static inline bool operator==(const LoopContext &lhs, const LoopContext &rhs) {
  return lhs.parent == rhs.parent;
}

/// Pack 8 bools together in a single byte
extern llvm::cl::opt<bool> EfficientBoolCache;

/// Print additional debug info relevant to performance
extern llvm::cl::opt<bool> EnzymePrintPerf;

/// Modes of potential unwraps
enum class UnwrapMode {
  // It is already known that it is legal to fully unwrap
  // this instruction. This means unwrap this instruction,
  // its operands, etc
  LegalFullUnwrap,
  // Attempt to fully unwrap this, looking up whenever it
  // is not legal to unwrap
  AttemptFullUnwrapWithLookup,
  // Attempt to fully unwrap this
  AttemptFullUnwrap,
  // Unwrap the current instruction but not its operand
  AttemptSingleUnwrap,
};

class CacheUtility {
public:
  /// The function whose instructions we are caching
  llvm::Function *const newFunc;

  /// Various analysis results of newFunc
  llvm::TargetLibraryInfo &TLI;
  llvm::DominatorTree DT;

protected:
  llvm::LoopInfo LI;
  llvm::AssumptionCache AC;
  MustExitScalarEvolution SE;

public:
  // Helper basicblock where all new allocations will be added to
  // This includes allocations for cache variables
  llvm::BasicBlock *inversionAllocs;

protected:
  CacheUtility(llvm::TargetLibraryInfo &TLI, llvm::Function *newFunc)
      : newFunc(newFunc), TLI(TLI), DT(*newFunc), LI(DT), AC(*newFunc),
        SE(*newFunc, TLI, AC, DT, LI), ompOffset(nullptr),
        ompTrueLimit(nullptr) {
    inversionAllocs = llvm::BasicBlock::Create(newFunc->getContext(),
                                               "allocsForInversion", newFunc);
  }

public:
  virtual ~CacheUtility();

private:
  /// Map of Loop to requisite loop information needed for AD (forward/reverse
  /// induction/etc)
  std::map<llvm::Loop *, LoopContext> loopContexts;

public:
  /// Given a BasicBlock BB in newFunc, set loopContext to the relevant
  /// contained loop and return true. If BB is not in a loop, return false
  bool getContext(llvm::BasicBlock *BB, LoopContext &loopContext);
  /// Return whether the given instruction is used as necessary as part of a
  /// loop context This includes as the canonical induction variable or
  /// increment
  bool isInstructionUsedInLoopInduction(llvm::Instruction &I) {
    for (auto &context : loopContexts) {
      if (context.second.var == &I || context.second.incvar == &I ||
          context.second.maxLimit == &I || context.second.trueLimit == &I) {
        return true;
      }
    }
    return false;
  }

  /// Print out all currently cached values
  void dumpScope() {
    llvm::errs() << "scope:\n";
    for (auto a : scopeMap) {
      llvm::errs() << "   scopeMap[" << *a.first << "] = " << *a.second.first
                   << " ctx:" << a.second.second.Block->getName() << "\n";
    }
    llvm::errs() << "end scope\n";
  }

  /// Erase this instruction both from LLVM modules and any local
  /// data-structures
  virtual void erase(llvm::Instruction *I);
  /// Replace this instruction both in LLVM modules and any local
  /// data-structures
  virtual void replaceAWithB(llvm::Value *A, llvm::Value *B,
                             bool storeInCache = false);

  // Context information to request calculation of loop limit information
  struct LimitContext {
    // A block inside of the loop, defining the location
    llvm::BasicBlock *Block;
    // Instead of getting the actual limits, return a limit of one
    bool ForceSingleIteration;

    LimitContext(llvm::BasicBlock *Block, bool ForceSingleIteration = false)
        : Block(Block), ForceSingleIteration(ForceSingleIteration) {}
  };

  llvm::Value *ompOffset;
  llvm::Value *ompTrueLimit;

  /// Given a LimitContext ctx, representing a location inside a loop nest,
  /// break each of the loops up into chunks of loops where each chunk's number
  /// of iterations can be computed at the chunk preheader. Every dynamic loop
  /// defines the start of a chunk. SubLimitType is a vector of chunk objects.
  /// More specifically it is a vector of { # iters in a Chunk (sublimit), Chunk
  /// } Each chunk object is a vector of loops contained within the chunk. For
  /// every loop, this returns pair of the LoopContext and the limit of that
  /// loop Both the vector of Chunks and vector of Loops within a Chunk go from
  /// innermost loop to outermost loop.
  typedef std::vector<std::pair<
      /*sublimit*/ llvm::Value *,
      /*loop limits*/ std::vector<std::pair<LoopContext, llvm::Value *>>>>
      SubLimitType;
  SubLimitType getSubLimits(LimitContext ctx);

private:
  /// Internal data structure used by getSubLimit to avoid computing the same
  /// loop limit multiple times if possible. Map's a desired limitMinus1 (see
  /// getSubLimits) and the block the true limit requested to the value of the
  /// limit accessible at that block
  std::map<std::pair<llvm::Value *, llvm::BasicBlock *>, llvm::Value *>
      LimitCache;
  /// Internal data structure used by getSubLimit to avoid computing the
  /// cumulative loop limit multiple times if possible. Map's a desired pair of
  /// operands to be multiplied (see getSubLimits) and the block the cumulative
  /// limit requested to the value of the limit accessible at that block This
  /// cache is also shared with computeIndexOfChunk
  std::map<std::tuple<llvm::Value *, llvm::Value *, llvm::BasicBlock *>,
           llvm::Value *>
      SizeCache;

  /// Given a loop context, compute the corresponding index into said loop at
  /// the IRBuilder<>
  llvm::Value *computeIndexOfChunk(
      bool inForwardPass, llvm::IRBuilder<> &v,
      const std::vector<std::pair<LoopContext, llvm::Value *>> &containedloops,
      llvm::Value *outerOffset);

private:
  /// Given a cache allocation and an index denoting how many Chunks deep the
  /// allocation is being indexed into, return the invariant metadata describing
  /// used to describe loads/stores to the indexed pointer
  /// Note that the cache allocation should either be an allocainst (if in
  /// fwd/both) or an extraction from the tape
  std::map<std::pair<llvm::Value *, int>, llvm::MDNode *>
      CachePointerInvariantGroups;
  /// Given a value being cached, return the invariant metadata of any
  /// loads/stores to memory storing that value
  std::map<llvm::Value *, llvm::MDNode *> ValueInvariantGroups;

protected:
  /// A map of values being cached to their underlying allocation/limit context
  std::map<llvm::Value *, std::pair<llvm::AllocaInst *, LimitContext>> scopeMap;

  /// A map of allocations to a vector of instruction used to create by the
  /// allocation Keeping track of these values is useful for deallocation. This
  /// is stored as a vector explicitly to order theses instructions in such a
  /// way that they can be erased by iterating in reverse order.
  std::map<llvm::AllocaInst *, std::vector<llvm::Instruction *>>
      scopeInstructions;

  /// A map of allocations to a set of instructions which free memory as part of
  /// the cache.
  std::map<llvm::AllocaInst *, std::set<llvm::CallInst *>> scopeFrees;

  /// A map of allocations to a set of instructions which allocate memory as
  /// part of the cache
  std::map<llvm::AllocaInst *, std::vector<llvm::CallInst *>> scopeAllocs;

public:
  /// Create a cache of Type T at the given LimitContext. If allocateInternal is
  /// set this will allocate the requesite memory. If extraSize is set,
  /// allocations will be a factor of extraSize larger
  llvm::AllocaInst *createCacheForScope(LimitContext ctx, llvm::Type *T,
                                        llvm::StringRef name, bool shouldFree,
                                        bool allocateInternal = true,
                                        llvm::Value *extraSize = nullptr);

  /// High-level utility to "unwrap" an instruction at a new location specified
  /// by BuilderM. Depending on the mode, it will either just unwrap this
  /// instruction, all of its instructions operands, and optionally lookup
  /// values when it is not legal to unwrap. If a value cannot be unwrap'd at a
  /// given location, this will null. This high-level utility should be
  /// implemented based off the low-level caching infrastructure provided in
  /// this class.
  virtual llvm::Value *unwrapM(llvm::Value *const val,
                               llvm::IRBuilder<> &BuilderM,
                               const llvm::ValueToValueMapTy &available,
                               UnwrapMode mode) = 0;

  /// High-level utility to get the value an instruction at a new location
  /// specified by BuilderM. Unlike unwrap, this function can never fail --
  /// falling back to creating a cache if necessary. This function is
  /// prepopulated with a set of values that are already known to be available
  /// and may contain optimizations for getting the value in more efficient ways
  /// (e.g. unwrap'ing when legal, looking up equivalent values, etc). This
  /// high-level utility should be implemented based off the low-level caching
  /// infrastructure provided in this class.
  virtual llvm::Value *
  lookupM(llvm::Value *val, llvm::IRBuilder<> &BuilderM,
          const llvm::ValueToValueMapTy &incoming_availalble =
              llvm::ValueToValueMapTy(),
          bool tryLegalityCheck = true) = 0;

  /// If an allocation is requested to be freed, this subclass will be called to
  /// chose how and where to free it. It is by default not implemented, falling
  /// back to an error. Subclasses who want to free memory should implement this
  /// function.
  virtual void freeCache(llvm::BasicBlock *forwardPreheader,
                         const SubLimitType &antimap, int i,
                         llvm::AllocaInst *alloc,
                         llvm::ConstantInt *byteSizeOfType,
                         llvm::Value *storeInto, llvm::MDNode *InvariantMD) {
    assert(0 && "freeing cache not handled in this scenario");
    llvm_unreachable("freeing cache not handled in this scenario");
  }

  /// Given an allocation defined at a particular ctx, store the value val
  /// in the cache at the location defined in the given builder
  void storeInstructionInCache(LimitContext ctx, llvm::IRBuilder<> &BuilderM,
                               llvm::Value *val, llvm::AllocaInst *cache);

  /// Given an allocation defined at a particular ctx, store the instruction
  /// in the cache right after the instruction is executed
  void storeInstructionInCache(LimitContext ctx, llvm::Instruction *inst,
                               llvm::AllocaInst *cache);

  /// Given an allocation specified by the LimitContext ctx and cache, compute a
  /// pointer that can hold the underlying type being cached. This value should
  /// be computed at BuilderM. Optionally, instructions needed to generate this
  /// pointer can be stored in scopeInstructions
  llvm::Value *getCachePointer(bool inForwardPass, llvm::IRBuilder<> &BuilderM,
                               LimitContext ctx, llvm::Value *cache, bool isi1,
                               bool storeInInstructionsMap,
                               llvm::Value *extraSize = nullptr);

  /// Given an allocation specified by the LimitContext ctx and cache, lookup
  /// the underlying cached value.
  llvm::Value *lookupValueFromCache(bool inForwardPass,
                                    llvm::IRBuilder<> &BuilderM,
                                    LimitContext ctx, llvm::Value *cache,
                                    bool isi1, llvm::Value *extraSize = nullptr,
                                    llvm::Value *extraOffset = nullptr);

protected:
  // List of values loaded from the cache
  llvm::SmallPtrSet<llvm::LoadInst *, 10> CacheLookups;
};

// Create a new canonical induction variable of Type Ty for Loop L
// Return the variable and the increment instruction
std::pair<llvm::PHINode *, llvm::Instruction *>
InsertNewCanonicalIV(llvm::Loop *L, llvm::Type *Ty, std::string name = "iv");

// Attempt to rewrite all phinode's in the loop in terms of the
// induction variable
void RemoveRedundantIVs(llvm::BasicBlock *Header, llvm::PHINode *CanonicalIV,
                        MustExitScalarEvolution &SE,
                        std::function<void(llvm::Instruction *)> eraser);
#endif