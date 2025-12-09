//=- SimpleGVN.cpp - GVN-like load forwarding optimization ============//
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

#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/IR/LegacyPassManager.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/Local.h"

#include "SimpleGVN.h"
#include "Utils.h"

using namespace llvm;

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "simple-gvn"

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
  if (RelativeOffset < 0 || (uint64_t)RelativeOffset + LoadSize > StoreSize) {
    return nullptr;
  }

  // If types match and offsets are the same, return directly
  if (RelativeOffset == 0 && LoadType == StoreType) {
    return StoredVal;
  }

  if (RelativeOffset == 0 && isa<PointerType>(LoadType) &&
      isa<PointerType>(StoreType)) {
    return Builder.CreatePointerCast(StoredVal, LoadType);
  }

  if (RelativeOffset == 0 && StoreSize >= LoadSize &&
      StoreType->isAggregateType()) {
    auto first = Builder.CreateExtractValue(StoredVal, 0);
    auto res = extractValue(Builder, first, LoadType, DL, LoadOffset,
                            StoreOffset, LoadSize);
    if (res) {
      return res;
    } else {
      if (auto I = dyn_cast<Instruction>(first))
        I->eraseFromParent();
    }
  }

  // Handle extraction with offset or type mismatch
  // First, bitcast to an integer type if needed
  if (!StoreType->isIntegerTy()) {
    IntegerType *IntTy = Builder.getIntNTy(StoreSize * 8);
    if (!CastInst::castIsValid(Instruction::BitCast, StoredVal->getType(),
                               IntTy)) {
      return nullptr;
    }
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
      if (!CastInst::castIsValid(Instruction::BitCast, StoredVal->getType(),
                                 LoadType)) {
        return nullptr;
      }
      StoredVal = Builder.CreateBitCast(StoredVal, LoadType);
    }
  }

  return StoredVal;
}

// Helper to check if a source instruction dominates and completely covers a
// target instruction's memory access
// For stores: checks if store covers a load
// For loads: checks if load covers another load
static bool dominatesAndCovers(Instruction *Source, Instruction *Target,
                               const APInt &SourceOffset,
                               const APInt &TargetOffset, uint64_t TargetSize,
                               const DataLayout &DL, DominatorTree &DT) {
  if (!DT.dominates(Source, Target))
    return false;

  // Get the size of the source memory access
  uint64_t SourceSize;
  if (auto *SI = dyn_cast<StoreInst>(Source)) {
    SourceSize = DL.getTypeStoreSize(SI->getValueOperand()->getType());
  } else if (auto *LI = dyn_cast<LoadInst>(Source)) {
    SourceSize = DL.getTypeStoreSize(LI->getType());
  } else {
    return false;
  }

  int64_t RelOffset = (TargetOffset - SourceOffset).getSExtValue();
  return RelOffset >= 0 && (uint64_t)RelOffset + TargetSize <= SourceSize;
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

// Collect memory operations (loads, stores) and calls for a given pointer value
// Returns false if the value has uses that prevent optimization
// Nocapture calls are only rejected (causing failure) if Calls is empty on
// entry If Calls is non-empty on entry, nocapture calls are collected
static bool
collectMemoryOps(Value *Arg, const DataLayout &DL,
                 SmallVectorImpl<std::pair<StoreInst *, APInt>> &Stores,
                 SmallVectorImpl<std::pair<LoadInst *, APInt>> &Loads,
                 SmallVectorImpl<std::pair<CallInst *, APInt>> &Calls) {
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

    for (Use &U : V->uses()) {
      User *Usr = U.getUser();
      if (auto *LI = dyn_cast<LoadInst>(Usr)) {
        Loads.push_back({LI, CurrentOffset});
      } else if (auto *SI = dyn_cast<StoreInst>(Usr)) {
        // Check if this is a store TO the pointer (not storing the pointer
        // value)
        if (SI->getPointerOperand() == V) {
          Stores.push_back({SI, CurrentOffset});
        } else {
          // Pointer value is being stored somewhere - reject this argument
          return false;
        }
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
        // Compute the offset for this GEP
        APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (!GEP->accumulateConstantOffset(DL, GEPOffset)) {
          // Cannot compute constant offset - reject this argument
          return false;
        }

        APInt NewOffset = CurrentOffset + GEPOffset;
        ToProcess.push_back({GEP, NewOffset});
      } else if (auto *CI = dyn_cast<CastInst>(Usr)) {
        // Casts don't change offset
        ToProcess.push_back({CI, CurrentOffset});
      } else if (auto *Call = dyn_cast<CallInst>(Usr)) {
        // Get the argument index from the Use
        unsigned ArgIdx = U.getOperandNo();
        if (isNoCapture(Call, ArgIdx)) {
          Calls.push_back({Call, CurrentOffset});
        } else {
          // Call that may capture - reject this argument
          return false;
        }
      } else {
        // Unknown use - reject this argument
        return false;
      }
    }
  }

  return true;
}

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, const DataLayout &DL) {
  bool Changed = false;

  // Find noalias arguments
  SmallVector<Value *, 4> CandidateArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && Arg.hasNoAliasAttr()) {
      CandidateArgs.push_back(&Arg);
    }
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<AllocaInst>(&I)) {
        CandidateArgs.push_back(&I);
      }
    }
  }

  if (CandidateArgs.empty())
    return false;

  // For each candidate argument, collect stores and loads with their offsets
  for (Value *Arg : CandidateArgs) {
    // Collect all stores and loads to this argument with offsets
    SmallVector<std::pair<StoreInst *, APInt>, 8> Stores;
    SmallVector<std::pair<LoadInst *, APInt>, 8> Loads;
    SmallVector<std::pair<CallInst *, APInt>, 8> Calls;

    // First pass: strict collection (no nocapture calls) for store-load
    // forwarding (pass empty Calls to reject nocapture calls)
    if (!collectMemoryOps(Arg, DL, Stores, Loads, Calls)) {
      // Argument has uses that prevent optimization
      continue;
    }

    APInt ZeroOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);

    // Try to forward {stores, previous loads} to loads using simplified
    // algorithm
    for (auto &[LI, LoadOffset] : Loads) {
      uint64_t LoadSize = DL.getTypeStoreSize(LI->getType());

      // Step 1: Find all stores that may alias with this load
      SmallVector<std::tuple<Instruction *, APInt, uint64_t>, 8> AliasingStores;
      for (auto &[SI, StoreOffset] : Stores) {
        uint64_t StoreSize =
            DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (memoryRangesAlias(LoadOffset, LoadSize, StoreOffset, StoreSize)) {
          AliasingStores.push_back({SI, StoreOffset, StoreSize});
        }
      }

      // Assume the call can touch any memory, so just set it to directly
      // overlap.
      for (auto &[CI, CallOffset] : Calls) {
        AliasingStores.push_back({CI, LoadOffset, LoadSize});
      }

      // Step 2: Filter to dominating + covering stores
      // Tuple of instruction storing, offset in the instruction, and the
      // equivalent value.
      SmallVector<std::tuple<Instruction *, APInt, Value *>, 8>
          DominatingCoveringStores;
      for (auto &[I, StoreOffset, StoreSize] : AliasingStores) {
        if (auto SI = dyn_cast<StoreInst>(I))
          if (dominatesAndCovers(SI, LI, StoreOffset, LoadOffset, LoadSize, DL,
                                 DT)) {
            DominatingCoveringStores.push_back(
                {SI, StoreOffset, SI->getValueOperand()});
          }
      }

      // Step 3: If only one aliasing store and it's dominating+covering,
      // forward
      if (AliasingStores.size() == 1 && DominatingCoveringStores.size() == 1) {
        Instruction *SI = std::get<0>(DominatingCoveringStores[0]);
        APInt StoreOffset = std::get<1>(DominatingCoveringStores[0]);

        IRBuilder<> Builder(LI);
        Value *StoredVal = std::get<2>(DominatingCoveringStores[0]);
        Value *ExtractedVal =
            extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                         StoreOffset, LoadSize);

        if (ExtractedVal) {
          LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (single alias)\n"
                            << "  Store: " << *SI << "\n"
                            << "  Load:  " << *LI << "\n");
          LI->replaceAllUsesWith(ExtractedVal);
          LI->eraseFromParent();
          LI = nullptr;
          Changed = true;
        }
        continue;
      }

      for (auto &[LI2, LoadOffset2] : Loads) {
        if (!LI2 || LI2 == LI)
          continue;
        if (dominatesAndCovers(LI2, LI, LoadOffset2, LoadOffset, LoadSize, DL,
                               DT)) {
          DominatingCoveringStores.emplace_back(LI2, LoadOffset2, LI2);
        }
      }

      // Step 4: If no dominating+covering stores, bail
      if (DominatingCoveringStores.empty()) {
        continue;
      }

      // Step 5: Build map of last store in each block before LI
      DenseMap<BasicBlock *, std::tuple<Instruction *, APInt, uint64_t>>
          LastStoreInBlockBeforeLI;
      for (auto &[SI, StoreOffset, Size] : AliasingStores) {
        BasicBlock *BB = SI->getParent();
        if (BB == LI->getParent()) {
          // Only consider stores before LI in the same block
          if (SI->comesBefore(LI)) {
            auto &Entry = LastStoreInBlockBeforeLI[BB];
            if (!std::get<0>(Entry) || std::get<0>(Entry)->comesBefore(SI)) {
              Entry = {SI, StoreOffset, Size};
            }
          }
        } else {
          // For other blocks, take the last store in the block
          auto &Entry = LastStoreInBlockBeforeLI[BB];
          if (!std::get<0>(Entry) || std::get<0>(Entry)->comesBefore(SI)) {
            Entry = {SI, StoreOffset, Size};
          }
        }
      }

      // Step 6: Check if LI's parent block has a dominating+covering store
      BasicBlock *LIBlock = LI->getParent();
      auto It = LastStoreInBlockBeforeLI.find(LIBlock);
      if (It != LastStoreInBlockBeforeLI.end()) {
        Instruction *SI = std::get<0>(It->second);

        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (SI == DCS ||
              (DCS->getParent() == LI->getParent() && SI->comesBefore(DCS))) {

            IRBuilder<> Builder(LI);
            Value *ExtractedVal =
                extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                             StoreOffset, LoadSize);

            if (ExtractedVal) {
              LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (same block)\n"
                                << "  Store: " << *DCS << "\n"
                                << "  Load:  " << *LI << "\n");
              LI->replaceAllUsesWith(ExtractedVal);
              LI->eraseFromParent();
              LI = nullptr;
              Changed = true;
              break;
            }
          }
        }
        if (LI == nullptr) {
          continue;
        }
      } else {
        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (DCS->getParent() == LI->getParent()) {

            IRBuilder<> Builder(LI);
            Value *ExtractedVal =
                extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                             StoreOffset, LoadSize);

            if (ExtractedVal) {
              LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (same block)\n"
                                << "  Store: " << *DCS << "\n"
                                << "  Load:  " << *LI << "\n");
              LI->replaceAllUsesWith(ExtractedVal);
              LI->eraseFromParent();
              LI = nullptr;
              Changed = true;
              break;
            }
          }
        }
        if (LI == nullptr) {
          continue;
        }
      }

      // Step 7: BFS backwards from LI's parent block
      SmallPtrSet<BasicBlock *, 32> Visited;
      SmallVector<BasicBlock *, 16> Worklist;
      StoreInst *Candidate = nullptr;
      APInt CandidateOffset = ZeroOffset;

      // Start with predecessors of LI's block
      for (BasicBlock *Pred : predecessors(LIBlock)) {
        if (Visited.insert(Pred).second)
          Worklist.push_back(Pred);
      }

      while (!Worklist.empty()) {
        BasicBlock *BB = Worklist.pop_back_val();

        auto It = LastStoreInBlockBeforeLI.find(BB);
        if (It != LastStoreInBlockBeforeLI.end()) {
          StoreInst *SI = dyn_cast<StoreInst>(std::get<0>(It->second));
          APInt StoreOffset = std::get<1>(It->second);

          if (!SI || !dominatesAndCovers(SI, LI, StoreOffset, LoadOffset,
                                         LoadSize, DL, DT)) {
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
        Value *ExtractedVal =
            extractValue(Builder, StoredVal, LI->getType(), DL, LoadOffset,
                         CandidateOffset, LoadSize);

        if (ExtractedVal) {
          LLVM_DEBUG(dbgs() << "SimpleGVN: Forwarding (BFS candidate)\n"
                            << "  Store: " << *Candidate << "\n"
                            << "  Load:  " << *LI << "\n");
          LI->replaceAllUsesWith(ExtractedVal);
          LI->eraseFromParent();
          LI = nullptr;
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

class SimpleGVN final : public FunctionPass {
public:
  static char ID;
  SimpleGVN() : FunctionPass(ID) {}

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

FunctionPass *createSimpleGVNPass() { return new SimpleGVN(); }

extern "C" void LLVMAddSimpleGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSimpleGVNPass());
}

char SimpleGVN::ID = 0;

static RegisterPass<SimpleGVN> X("simple-gvn",
                                 "GVN-like load forwarding optimization");

SimpleGVNNewPM::Result SimpleGVNNewPM::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  bool Changed = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  Changed = simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F), DL);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimpleGVNNewPM::Key;
