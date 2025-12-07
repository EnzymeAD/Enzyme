//=- SimplifyGVN.cpp - GVN-like load forwarding optimization ============//
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
// This file contains a GVN-like optimization pass that forwards loads from
// noalias/nocapture arguments to their corresponding stores, with support
// for offsets and type conversions.
//
// This pass addresses the limitation of LLVM's built-in GVN pass which has
// a small limit on the number of instructions/memory offsets it analyzes
// via its use of the memdep analysis.
//
// Algorithm:
// 1. Identify function arguments with noalias and nocapture attributes
// 2. Verify all uses are exclusively loads, stores, or GEP instructions
// 3. For each load from such an argument:
//    a. Find all stores to the argument with constant offsets
//    b. Find a dominating store that covers the load's memory range
//    c. Check that no aliasing store exists between the store and load
//    d. If safe, replace the load with the stored value, performing
//       type conversion or extraction as needed
//
// Example transformation:
//   define i32 @foo(i32* noalias nocapture %ptr) {
//     store i32 42, i32* %ptr
//     %v = load i32, i32* %ptr
//     ret i32 %v
//   }
// becomes:
//   define i32 @foo(i32* noalias nocapture %ptr) {
//     store i32 42, i32* %ptr
//     ret i32 42
//   }
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/CFG.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/Local.h"

#include "SimplifyGVN.h"
#include "Utils.h"

using namespace llvm;

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "simplify-gvn"

namespace {

// Extract a value with potential type conversion
Value *extractValue(IRBuilder<> &Builder, Value *StoredVal, Type *LoadType,
                    const DataLayout &DL, APInt LoadOffset, APInt StoreOffset,
                    uint64_t LoadSize) {
  Type *StoreType = StoredVal->getType();
  uint64_t StoreSize = DL.getTypeStoreSize(StoreType);

  // Calculate relative offset
  int64_t RelativeOffset = (LoadOffset - StoreOffset).getSExtValue();

  // Check if the load is completely within the stored value
  if (RelativeOffset < 0 ||
      (uint64_t)RelativeOffset + LoadSize > StoreSize) {
    return nullptr;
  }

  // If types match and offsets are the same, return directly
  if (RelativeOffset == 0 && LoadType == StoreType) {
    return StoredVal;
  }

  // Handle extraction with offset or type mismatch
  // First, bitcast to an integer type if needed
  if (!StoreType->isIntegerTy()) {
    IntegerType *IntTy = Builder.getIntNTy(StoreSize * 8);
    StoredVal = Builder.CreateBitCast(StoredVal, IntTy);
  }

  // Extract the relevant bits if there's an offset
  if (RelativeOffset > 0) {
    uint64_t ShiftBits = RelativeOffset * 8;
    StoredVal = Builder.CreateLShr(StoredVal, ShiftBits);
  }

  // Truncate to the load size if needed
  IntegerType *LoadIntTy = Builder.getIntNTy(LoadSize * 8);
  if (StoredVal->getType() != LoadIntTy) {
    StoredVal = Builder.CreateTrunc(StoredVal, LoadIntTy);
  }

  // Bitcast to the final type if needed
  if (LoadIntTy != LoadType) {
    if (LoadType->isPointerTy()) {
      StoredVal = Builder.CreateIntToPtr(StoredVal, LoadType);
    } else {
      StoredVal = Builder.CreateBitCast(StoredVal, LoadType);
    }
  }

  return StoredVal;
}

// Check if there are any conflicting stores between SI and LI
// Uses BFS to check all paths from SI's block to LI's block
bool hasConflictingStoreBetween(
    StoreInst *SI, LoadInst *LI, const DataLayout &DL,
    const DenseMap<BasicBlock *, SmallVector<std::pair<StoreInst *, APInt>, 4>>
        &BlockToStores,
    APInt LoadOffset, uint64_t LoadSize, APInt StoreOffset) {
  
  BasicBlock *SIBlock = SI->getParent();
  BasicBlock *LIBlock = LI->getParent();
  
  // Lambda to check if a store conflicts with the load
  auto conflictsWith = [&](StoreInst *Store, const APInt &Offset) {
    if (Store == SI)
      return false;
    
    uint64_t StSize = DL.getTypeStoreSize(Store->getValueOperand()->getType());
    int64_t RelOffset = (LoadOffset - Offset).getSExtValue();
    
    // Check if store overlaps with load location
    return !(RelOffset + (int64_t)LoadSize <= 0 ||
             RelOffset >= (int64_t)StSize);
  };
  
  // If SI and LI are in the same block, check instructions between them
  if (SIBlock == LIBlock) {
    auto It = BlockToStores.find(SIBlock);
    if (It == BlockToStores.end())
      return false;
    
    bool afterSI = false;
    for (auto &[Store, Offset] : It->second) {
      if (Store == SI) {
        afterSI = true;
        continue;
      }
      // Check if this store comes before LI
      if (Store->comesBefore(LI)) {
        if (afterSI && SI->comesBefore(Store)) {
          if (conflictsWith(Store, Offset))
            return true;
        }
      }
    }
    return false;
  }
  
  // Check for conflicts after SI in its block
  auto SIIt = BlockToStores.find(SIBlock);
  if (SIIt != BlockToStores.end()) {
    for (auto &[Store, Offset] : SIIt->second) {
      if (Store == SI)
        continue;
      if (SI->comesBefore(Store)) {
        if (conflictsWith(Store, Offset))
          return true;
      }
    }
  }
  
  // BFS from SI's successors, but only check blocks that can reach LI
  // This is important to handle mutually exclusive branches correctly
  SmallPtrSet<BasicBlock *, 32> Visited;
  SmallVector<BasicBlock *, 16> Worklist;
  
  // Helper to check if a block can reach LI's block
  // We do a simple reachability check
  auto canReachLI = [&](BasicBlock *BB) -> bool {
    if (BB == LIBlock)
      return true;
    SmallPtrSet<BasicBlock *, 32> ReachVisited;
    SmallVector<BasicBlock *, 16> ReachWorklist;
    ReachWorklist.push_back(BB);
    ReachVisited.insert(BB);
    
    while (!ReachWorklist.empty()) {
      BasicBlock *Current = ReachWorklist.pop_back_val();
      for (BasicBlock *Succ : successors(Current)) {
        if (Succ == LIBlock)
          return true;
        if (ReachVisited.insert(Succ).second)
          ReachWorklist.push_back(Succ);
      }
    }
    return false;
  };
  
  // Add successors of SI's block that can reach LI
  for (BasicBlock *Succ : successors(SIBlock)) {
    if (canReachLI(Succ) && Visited.insert(Succ).second)
      Worklist.push_back(Succ);
  }
  
  while (!Worklist.empty()) {
    BasicBlock *BB = Worklist.pop_back_val();
    
    // Check if this block contains LI
    if (BB == LIBlock) {
      // Check stores before LI in this block
      auto It = BlockToStores.find(BB);
      if (It != BlockToStores.end()) {
        for (auto &[Store, Offset] : It->second) {
          if (Store->comesBefore(LI)) {
            if (conflictsWith(Store, Offset))
              return true;
          }
        }
      }
      continue; // Don't add successors of LI's block
    }
    
    // Check all stores in this block
    auto It = BlockToStores.find(BB);
    if (It != BlockToStores.end()) {
      for (auto &[Store, Offset] : It->second) {
        if (conflictsWith(Store, Offset))
          return true;
      }
    }
    
    // Add successors that can reach LI
    for (BasicBlock *Succ : successors(BB)) {
      if (canReachLI(Succ) && Visited.insert(Succ).second)
        Worklist.push_back(Succ);
    }
  }
  
  return false;
}

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, PostDominatorTree &PDT,
                 const DataLayout &DL) {
  bool Changed = false;

  // Find noalias arguments
  SmallVector<Argument *, 4> CandidateArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && Arg.hasNoAliasAttr()) {
      CandidateArgs.push_back(&Arg);
    }
  }

  if (CandidateArgs.empty())
    return false;

  // For each candidate argument, collect stores and loads with their offsets
  for (Argument *Arg : CandidateArgs) {
    // Collect all stores and loads to this argument with offsets
    SmallVector<std::pair<StoreInst *, APInt>, 8> Stores;
    SmallVector<std::pair<LoadInst *, APInt>, 8> Loads;
    
    // WorkList tracks (Value*, Offset from Arg)
    SmallVector<std::pair<Value *, APInt>, 16> ToProcess;
    SmallPtrSet<Value *, 16> Visited;
    
    APInt ZeroOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);
    ToProcess.push_back({Arg, ZeroOffset});

    while (!ToProcess.empty()) {
      auto [V, CurrentOffset] = ToProcess.pop_back_val();
      
      // Skip if already visited
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          Loads.push_back({LI, CurrentOffset});
        } else if (auto *SI = dyn_cast<StoreInst>(U)) {
          // Check if this is a store TO the pointer (not storing the pointer value)
          if (SI->getPointerOperand() == V) {
            Stores.push_back({SI, CurrentOffset});
          } else {
            // Pointer value is being stored somewhere - reject this argument
            goto next_argument;
          }
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // Compute the offset for this GEP
          APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
          if (!GEP->accumulateConstantOffset(DL, GEPOffset)) {
            // Cannot compute constant offset - reject this argument
            goto next_argument;
          }
          
          APInt NewOffset = CurrentOffset + GEPOffset;
          ToProcess.push_back({GEP, NewOffset});
        } else if (auto *CI = dyn_cast<CastInst>(U)) {
          // Casts don't change offset
          ToProcess.push_back({CI, CurrentOffset});
        } else {
          // Unknown use - reject this argument
          goto next_argument;
        }
      }
    }

    // Scope for BlockToStores to avoid goto issues
    {
      // Build a map from basic blocks to stores for efficient path checking
      DenseMap<BasicBlock *, SmallVector<std::pair<StoreInst *, APInt>, 4>>
          BlockToStores;
      for (auto &[SI, Offset] : Stores) {
        BlockToStores[SI->getParent()].push_back({SI, Offset});
      }

      // Try to forward stores to loads
      for (auto &[LI, LoadOffset] : Loads) {
        uint64_t LoadSize = DL.getTypeStoreSize(LI->getType());

        // Find a dominating store that can satisfy this load
        StoreInst *DominatingStore = nullptr;
        APInt StoreOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);

        for (auto &[SI, CurStoreOffset] : Stores) {
          if (!DT.dominates(SI, LI))
            continue;

          uint64_t StoreSize =
              DL.getTypeStoreSize(SI->getValueOperand()->getType());

          // Check if the store covers the load
          int64_t RelOffset = (LoadOffset - CurStoreOffset).getSExtValue();
          if (RelOffset < 0 || (uint64_t)RelOffset + LoadSize > StoreSize)
            continue;

          // Check if there's no conflicting store between SI and LI using BFS
          if (!hasConflictingStoreBetween(SI, LI, DL, BlockToStores, LoadOffset,
                                          LoadSize, CurStoreOffset)) {
            DominatingStore = SI;
            StoreOffset = CurStoreOffset;
            break;
          }
        }

        if (DominatingStore) {
          // Try to extract the value from the store
          IRBuilder<> Builder(LI);
          Value *StoredVal = DominatingStore->getValueOperand();

          Value *ExtractedVal =
              extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                           StoreOffset, LoadSize);

          if (ExtractedVal) {
            LLVM_DEBUG(dbgs() << "SimplifyGVN: Forwarding store to load\n"
                              << "  Store: " << *DominatingStore << "\n"
                              << "  Load:  " << *LI << "\n"
                              << "  Value: " << *ExtractedVal << "\n");

            LI->replaceAllUsesWith(ExtractedVal);
            Changed = true;
          }
        }
      }
    }
    
next_argument:
    continue;
  }

  return Changed;
}

class SimplifyGVN final : public FunctionPass {
public:
  static char ID;
  SimplifyGVN() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &PDT = getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
    const DataLayout &DL = F.getParent()->getDataLayout();
    return simplifyGVN(F, DT, PDT, DL);
  }
};

} // namespace

FunctionPass *createSimplifyGVNPass() { return new SimplifyGVN(); }

extern "C" void LLVMAddSimplifyGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSimplifyGVNPass());
}

char SimplifyGVN::ID = 0;

static RegisterPass<SimplifyGVN>
    X("simplify-gvn", "GVN-like load forwarding optimization");

SimplifyGVNNewPM::Result
SimplifyGVNNewPM::run(Function &F, FunctionAnalysisManager &FAM) {
  bool Changed = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  Changed = simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F),
                        FAM.getResult<PostDominatorTreeAnalysis>(F), DL);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimplifyGVNNewPM::Key;
