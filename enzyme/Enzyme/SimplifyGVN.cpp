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

// Check if an argument has noalias and nocapture attributes
bool isNoAliasNoCapture(const Argument *Arg) {
  return Arg->hasNoAliasAttr() && Arg->hasNoCaptureAttr();
}

// Compute the offset of a pointer relative to a base pointer
// Returns true if a constant offset can be determined
bool getConstantOffset(const DataLayout &DL, Value *Ptr, Value *Base,
                       APInt &Offset) {
  Offset = APInt(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);

  // Strip pointer casts
  Ptr = Ptr->stripPointerCasts();
  Base = Base->stripPointerCasts();

  if (Ptr == Base)
    return true;

  // Handle GEP instructions
  if (auto *GEP = dyn_cast<GEPOperator>(Ptr)) {
    APInt GEPOffset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
    if (!GEP->accumulateConstantOffset(DL, GEPOffset))
      return false;

    Value *GEPBase = GEP->getPointerOperand()->stripPointerCasts();
    if (GEPBase == Base) {
      Offset = GEPOffset;
      return true;
    }

    // Recursively check if GEP base has offset from Base
    APInt BaseOffset(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
    if (getConstantOffset(DL, GEPBase, Base, BaseOffset)) {
      Offset = BaseOffset + GEPOffset;
      return true;
    }
  }

  return false;
}

// Check if all uses of a value are loads, stores, or GEPs (recursively)
bool areUsesOnlyLoadStoreGEP(Value *V, SmallPtrSetImpl<Value *> &Visited) {
  if (!Visited.insert(V).second)
    return true;

  for (User *U : V->users()) {
    if (isa<LoadInst>(U))
      continue;

    // Check if this is a store TO the pointer (not storing the pointer value)
    if (auto *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getPointerOperand() == V)
        continue;
      // If the pointer value is being stored somewhere, reject
      return false;
    }

    if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
      if (!areUsesOnlyLoadStoreGEP(GEP, Visited))
        return false;
      continue;
    }

    // Allow any cast instruction (bitcast, addrspacecast, etc.)
    if (isa<CastInst>(U)) {
      if (!areUsesOnlyLoadStoreGEP(cast<Instruction>(U), Visited))
        return false;
      continue;
    }

    return false;
  }

  return true;
}

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

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, const DataLayout &DL,
                 AAResults &AA) {
  bool Changed = false;

  // Find noalias/nocapture arguments
  SmallVector<Argument *, 4> CandidateArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy() && isNoAliasNoCapture(&Arg)) {
      // Check if all uses are loads, stores, or GEPs
      SmallPtrSet<Value *, 16> Visited;
      if (areUsesOnlyLoadStoreGEP(&Arg, Visited)) {
        CandidateArgs.push_back(&Arg);
      }
    }
  }

  if (CandidateArgs.empty())
    return false;

  // For each candidate argument, collect stores and try to forward to loads
  for (Argument *Arg : CandidateArgs) {
    // Collect all stores to this argument
    SmallVector<StoreInst *, 8> Stores;
    SmallPtrSet<Value *, 16> WorkList;
    WorkList.insert(Arg);

    SmallVector<Value *, 16> ToProcess(WorkList.begin(), WorkList.end());
    while (!ToProcess.empty()) {
      Value *V = ToProcess.pop_back_val();

      for (User *U : V->users()) {
        if (auto *SI = dyn_cast<StoreInst>(U)) {
          if (SI->getPointerOperand() == V) {
            Stores.push_back(SI);
          }
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          if (WorkList.insert(GEP).second) {
            ToProcess.push_back(GEP);
          }
        } else if (isa<CastInst>(U)) {
          if (WorkList.insert(cast<Instruction>(U)).second) {
            ToProcess.push_back(cast<Instruction>(U));
          }
        }
      }
    }

    // Collect all loads from this argument
    SmallVector<LoadInst *, 8> Loads;
    WorkList.clear();
    WorkList.insert(Arg);
    ToProcess.assign(WorkList.begin(), WorkList.end());

    while (!ToProcess.empty()) {
      Value *V = ToProcess.pop_back_val();

      for (User *U : V->users()) {
        if (auto *LI = dyn_cast<LoadInst>(U)) {
          if (LI->getPointerOperand() == V) {
            Loads.push_back(LI);
          }
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          if (WorkList.insert(GEP).second) {
            ToProcess.push_back(GEP);
          }
        } else if (isa<CastInst>(U)) {
          if (WorkList.insert(cast<Instruction>(U)).second) {
            ToProcess.push_back(cast<Instruction>(U));
          }
        }
      }
    }

    // Try to forward stores to loads
    for (LoadInst *LI : Loads) {
      Value *LoadPtr = LI->getPointerOperand();
      APInt LoadOffset(DL.getIndexTypeSizeInBits(LoadPtr->getType()), 0);

      if (!getConstantOffset(DL, LoadPtr, Arg, LoadOffset))
        continue;

      uint64_t LoadSize = DL.getTypeStoreSize(LI->getType());

      // Find a dominating store that can satisfy this load
      StoreInst *DominatingStore = nullptr;
      APInt StoreOffset(DL.getIndexTypeSizeInBits(LoadPtr->getType()), 0);

      for (StoreInst *SI : Stores) {
        if (!DT.dominates(SI, LI))
          continue;

        Value *StorePtr = SI->getPointerOperand();
        APInt CurStoreOffset(DL.getIndexTypeSizeInBits(StorePtr->getType()),
                             0);

        if (!getConstantOffset(DL, StorePtr, Arg, CurStoreOffset))
          continue;

        uint64_t StoreSize =
            DL.getTypeStoreSize(SI->getValueOperand()->getType());

        // Check if the store covers the load
        int64_t RelOffset = (LoadOffset - CurStoreOffset).getSExtValue();
        if (RelOffset < 0 || (uint64_t)RelOffset + LoadSize > StoreSize)
          continue;

        // Check if there's no aliasing store in between
        bool HasIntermediateStore = false;
        for (StoreInst *OtherSI : Stores) {
          if (OtherSI == SI)
            continue;

          if (DT.dominates(SI, OtherSI) && DT.dominates(OtherSI, LI)) {
            // Check if this store may alias
            Value *OtherStorePtr = OtherSI->getPointerOperand();
            APInt OtherStoreOffset(
                DL.getIndexTypeSizeInBits(OtherStorePtr->getType()), 0);

            if (!getConstantOffset(DL, OtherStorePtr, Arg, OtherStoreOffset)) {
              // Can't determine offset, assume it may alias
              HasIntermediateStore = true;
              break;
            }

            uint64_t OtherStoreSize =
                DL.getTypeStoreSize(OtherSI->getValueOperand()->getType());

            // Check if the store overlaps with the load
            int64_t OtherRelOffset =
                (LoadOffset - OtherStoreOffset).getSExtValue();
            if (!(OtherRelOffset + (int64_t)LoadSize <= 0 ||
                  OtherRelOffset >= (int64_t)OtherStoreSize)) {
              HasIntermediateStore = true;
              break;
            }
          }
        }

        if (!HasIntermediateStore) {
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

  return Changed;
}

class SimplifyGVN final : public FunctionPass {
public:
  static char ID;
  SimplifyGVN() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    const DataLayout &DL = F.getParent()->getDataLayout();
    return simplifyGVN(F, DT, DL, AA);
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
  Changed = simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F), DL,
                        FAM.getResult<AAManager>(F));
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimplifyGVNNewPM::Key;
