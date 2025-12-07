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

// Helper to check if a store dominates and completely covers a load
static bool dominatesAndCovers(StoreInst *SI, LoadInst *LI,
                                 const APInt &StoreOffset, const APInt &LoadOffset,
                                 uint64_t LoadSize, const DataLayout &DL,
                                 DominatorTree &DT) {
  if (!DT.dominates(SI, LI))
    return false;

  uint64_t StoreSize = DL.getTypeStoreSize(SI->getValueOperand()->getType());
  int64_t RelOffset = (LoadOffset - StoreOffset).getSExtValue();
  return RelOffset >= 0 && (uint64_t)RelOffset + LoadSize <= StoreSize;
}

// Helper to check if two memory ranges alias
// Range1: [Offset1, Offset1 + Size1)
// Range2: [Offset2, Offset2 + Size2)
static bool memoryRangesAlias(const APInt &Offset1, uint64_t Size1,
                                const APInt &Offset2, uint64_t Size2) {
  // Check if range2 ends before range1 begins
  if ((Offset2 + Size2).sle(Offset1))
    return false;
  
  // Check if range1 ends before range2 begins
  if ((Offset1 + Size1).sle(Offset2))
    return false;
  
  // Otherwise, they may alias
  return true;
}

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, const DataLayout &DL) {
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

    // Try to forward stores to loads using simplified algorithm
    {
      for (auto &[LI, LoadOffset] : Loads) {
        uint64_t LoadSize = DL.getTypeStoreSize(LI->getType());
        
        // Step 1: Find all stores that may alias with this load
        SmallVector<std::pair<StoreInst *, APInt>, 8> AliasingStores;
        for (auto &[SI, StoreOffset] : Stores) {
          uint64_t StoreSize = DL.getTypeStoreSize(SI->getValueOperand()->getType());
          if (memoryRangesAlias(LoadOffset, LoadSize, StoreOffset, StoreSize)) {
            AliasingStores.push_back({SI, StoreOffset});
          }
        }
        
        // Step 2: Filter to dominating + covering stores
        SmallVector<std::pair<StoreInst *, APInt>, 8> DominatingCoveringStores;
        for (auto &[SI, StoreOffset] : AliasingStores) {
          if (dominatesAndCovers(SI, LI, StoreOffset, LoadOffset, LoadSize, DL, DT)) {
            DominatingCoveringStores.push_back({SI, StoreOffset});
          }
        }
        
        // Step 3: If only one aliasing store and it's dominating+covering, forward
        if (AliasingStores.size() == 1 && DominatingCoveringStores.size() == 1) {
          StoreInst *SI = DominatingCoveringStores[0].first;
          APInt StoreOffset = DominatingCoveringStores[0].second;
          
          IRBuilder<> Builder(LI);
          Value *StoredVal = SI->getValueOperand();
          Value *ExtractedVal = extractValue(Builder, StoredVal, LI->getType(), DL,
                                              LoadOffset, StoreOffset, LoadSize);
          
          if (ExtractedVal) {
            LLVM_DEBUG(dbgs() << "SimplifyGVN: Forwarding (single alias)\n"
                              << "  Store: " << *SI << "\n"
                              << "  Load:  " << *LI << "\n");
            LI->replaceAllUsesWith(ExtractedVal);
            Changed = true;
          }
          continue;
        }
        
        // Step 4: If no dominating+covering stores, bail
        if (DominatingCoveringStores.empty())
          continue;
        
        // Step 5: Build map of last store in each block before LI
        DenseMap<BasicBlock *, std::pair<StoreInst *, APInt>> LastStoreInBlockBeforeLI;
        for (auto &[SI, StoreOffset] : AliasingStores) {
          BasicBlock *BB = SI->getParent();
          if (BB == LI->getParent()) {
            // Only consider stores before LI in the same block
            if (SI->comesBefore(LI)) {
              auto &Entry = LastStoreInBlockBeforeLI[BB];
              if (!Entry.first || Entry.first->comesBefore(SI)) {
                Entry = {SI, StoreOffset};
              }
            }
          } else {
            // For other blocks, take the last store in the block
            auto &Entry = LastStoreInBlockBeforeLI[BB];
            if (!Entry.first || Entry.first->comesBefore(SI)) {
              Entry = {SI, StoreOffset};
            }
          }
        }
        
        // Step 6: Check if LI's parent block has a dominating+covering store
        BasicBlock *LIBlock = LI->getParent();
        auto It = LastStoreInBlockBeforeLI.find(LIBlock);
        if (It != LastStoreInBlockBeforeLI.end()) {
          StoreInst *SI = It->second.first;
          APInt StoreOffset = It->second.second;
          
          if (dominatesAndCovers(SI, LI, StoreOffset, LoadOffset, LoadSize, DL, DT)) {
            IRBuilder<> Builder(LI);
            Value *StoredVal = SI->getValueOperand();
            Value *ExtractedVal = extractValue(Builder, StoredVal, LI->getType(), DL,
                                                LoadOffset, StoreOffset, LoadSize);
            
            if (ExtractedVal) {
              LLVM_DEBUG(dbgs() << "SimplifyGVN: Forwarding (same block)\n"
                                << "  Store: " << *SI << "\n"
                                << "  Load:  " << *LI << "\n");
              LI->replaceAllUsesWith(ExtractedVal);
              Changed = true;
            }
            continue;
          }
          
          // Not dominating+covering, bail
          continue;
        }
        
        // Step 7: BFS backwards from LI's parent block
        SmallPtrSet<BasicBlock *, 32> Visited;
        SmallVector<BasicBlock *, 16> Worklist;
        StoreInst *Candidate = nullptr;
        APInt CandidateOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);
        
        // Start with predecessors of LI's block
        for (BasicBlock *Pred : predecessors(LIBlock)) {
          if (Visited.insert(Pred).second)
            Worklist.push_back(Pred);
        }
        
        while (!Worklist.empty()) {
          BasicBlock *BB = Worklist.pop_back_val();
          
          auto It = LastStoreInBlockBeforeLI.find(BB);
          if (It != LastStoreInBlockBeforeLI.end()) {
            StoreInst *SI = It->second.first;
            APInt StoreOffset = It->second.second;
            
            if (!dominatesAndCovers(SI, LI, StoreOffset, LoadOffset, LoadSize, DL, DT)) {
              // Non-dominating+covering store on path, bail
              Candidate = nullptr;
              break;
            }
            
            // Found dominating+covering store
            if (!Candidate) {
              Candidate = SI;
              CandidateOffset = StoreOffset;
            } else if (Candidate != SI) {
              // Multiple different candidates, bail
              Candidate = nullptr;
              break;
            }
          }
          
          // Continue BFS
          for (BasicBlock *Pred : predecessors(BB)) {
            if (Visited.insert(Pred).second)
              Worklist.push_back(Pred);
          }
        }
        
        // Step 8: If unique candidate found, forward
        if (Candidate) {
          IRBuilder<> Builder(LI);
          Value *StoredVal = Candidate->getValueOperand();
          Value *ExtractedVal = extractValue(Builder, StoredVal, LI->getType(), DL,
                                              LoadOffset, CandidateOffset, LoadSize);
          
          if (ExtractedVal) {
            LLVM_DEBUG(dbgs() << "SimplifyGVN: Forwarding (BFS candidate)\n"
                              << "  Store: " << *Candidate << "\n"
                              << "  Load:  " << *LI << "\n");
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

  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    const DataLayout &DL = F.getParent()->getDataLayout();
    return simplifyGVN(F, DT, DL);
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
  Changed = simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F), DL);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimplifyGVNNewPM::Key;
