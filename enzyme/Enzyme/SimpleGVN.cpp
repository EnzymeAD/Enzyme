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

static bool getFixedTypeStoreSize(const DataLayout &DL, Type *Ty,
                                  uint64_t &Size) {
  TypeSize TS = DL.getTypeStoreSize(Ty);
  if (TS.isScalable())
    return false;
  Size = TS.getKnownMinValue();
  return true;
}

static bool coversRange(const APInt &SourceOffset, uint64_t SourceSize,
                        const APInt &TargetOffset, uint64_t TargetSize,
                        uint64_t &RelativeOffsetOut) {
  unsigned W = std::max<unsigned>({SourceOffset.getBitWidth(),
                                   TargetOffset.getBitWidth(), 64}) +
               2;
  APInt Diff = TargetOffset.sext(W) - SourceOffset.sext(W);
  if (Diff.isNegative())
    return false;
  if ((Diff + APInt(W, TargetSize)).sgt(APInt(W, SourceSize)))
    return false;
  RelativeOffsetOut = Diff.getZExtValue();
  return true;
}

// Extract a value with potential type conversion
Value *extractValue(IRBuilder<> &Builder, Value *StoredVal, Type *LoadType,
                    const DataLayout &DL, APInt LoadOffset, APInt StoreOffset,
                    uint64_t LoadSize) {
  Type *StoreType = StoredVal->getType();
  uint64_t StoreSize = 0;
  if (!getFixedTypeStoreSize(DL, StoreType, StoreSize))
    return nullptr;

  // Check if the load is completely within the stored value
  uint64_t RelativeOffset = 0;
  if (!coversRange(StoreOffset, StoreSize, LoadOffset, LoadSize,
                   RelativeOffset)) {
    return nullptr;
  }

  // If types match and offsets are the same, return directly
  if (RelativeOffset == 0 && LoadType == StoreType) {
    return StoredVal;
  }

  if (RelativeOffset == 0 && isa<PointerType>(LoadType) &&
      isa<PointerType>(StoreType)) {
    auto FromAS = cast<PointerType>(StoreType)->getAddressSpace();
    auto ToAS = cast<PointerType>(LoadType)->getAddressSpace();
    if (FromAS == 10 && ToAS == 0 && StoreSize == LoadSize &&
        DL.getPointerSize(0) == LoadSize) {
      Module *mod = Builder.GetInsertBlock()->getModule();
      Function *F = mod->getFunction("julia.pointer_from_objref");
      if (!F) {
        Type *T_jlvalue = StructType::get(mod->getContext());
        Type *T_prjlvalue = getPointerType(T_jlvalue, 11); // Derived is 11
        Type *T_pjlvalue = getPointerType(T_jlvalue, 0);
        FunctionType *FTy = FunctionType::get(T_pjlvalue, {T_prjlvalue}, false);
        F = Function::Create(FTy, Function::ExternalLinkage,
                             "julia.pointer_from_objref", mod);
      }

      Type *T_jlvalue = StructType::get(mod->getContext());
      Value *TrackedVal =
          Builder.CreateAddrSpaceCast(StoredVal, getPointerType(T_jlvalue, 11));
      Value *RawPtr = Builder.CreateCall(F, {TrackedVal});
      return Builder.CreatePointerCast(RawPtr, LoadType);
    }
    if (FromAS == ToAS)
      return Builder.CreatePointerCast(StoredVal, LoadType);
    return nullptr;
  }

  if (RelativeOffset == 0 && StoreSize >= LoadSize &&
      StoreType->isAggregateType()) {
    bool HasFirst = false;
    if (auto *ST = dyn_cast<StructType>(StoreType))
      HasFirst = ST->getNumElements() > 0;
    else if (auto *AT = dyn_cast<ArrayType>(StoreType))
      HasFirst = AT->getNumElements() > 0;
    if (HasFirst) {
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
  }

  // Cannot synthesize zero-bit integer conversions
  if (LoadSize == 0 || StoreSize == 0) {
    return nullptr;
  }

  // Validate all conversions before emitting any IR so failed extractions
  // never leave dead instructions behind.
  IntegerType *StoreIntTy = StoreType->isIntegerTy()
                                ? cast<IntegerType>(StoreType)
                                : Builder.getIntNTy(StoreSize * 8);
  if (!StoreType->isIntegerTy() &&
      !CastInst::castIsValid(Instruction::BitCast, StoreType, StoreIntTy)) {
    return nullptr;
  }

  uint64_t ShiftBits = DL.isBigEndian()
                           ? (StoreSize - RelativeOffset - LoadSize) * 8
                           : RelativeOffset * 8;
  if (ShiftBits > 0 && ShiftBits >= StoreIntTy->getIntegerBitWidth()) {
    return nullptr;
  }

  IntegerType *LoadIntTy = Builder.getIntNTy(LoadSize * 8);
  if (LoadIntTy != LoadType) {
    if (LoadType->isPointerTy()) {
      if (cast<PointerType>(LoadType)->getAddressSpace() == 10) {
        if (DL.isNonIntegralAddressSpace(0) ||
            DL.getPointerSize(0) != LoadSize) {
          return nullptr;
        }
      } else if (DL.isNonIntegralPointerType(LoadType)) {
        return nullptr;
      }
    } else if (!CastInst::castIsValid(Instruction::BitCast, LoadIntTy,
                                      LoadType)) {
      return nullptr;
    }
  }

  // Handle extraction with offset or type mismatch
  // First, bitcast to an integer type if needed
  if (!StoreType->isIntegerTy()) {
    StoredVal = Builder.CreateBitCast(StoredVal, StoreIntTy);
  }

  // Extract the relevant bits if there's an offset (or big-endian high bytes)
  if (ShiftBits > 0) {
    StoredVal = Builder.CreateLShr(StoredVal, ShiftBits);
  }

  // Truncate or extend to the load size if needed
  if (StoredVal->getType() != LoadIntTy) {
    unsigned StoredWidth = StoredVal->getType()->getIntegerBitWidth();
    unsigned LoadWidth = LoadIntTy->getIntegerBitWidth();
    if (StoredWidth > LoadWidth) {
      StoredVal = Builder.CreateTrunc(StoredVal, LoadIntTy);
    } else if (StoredWidth < LoadWidth) {
      StoredVal = Builder.CreateZExt(StoredVal, LoadIntTy);
    }
  }

  // Bitcast to the final type if needed
  if (LoadIntTy != LoadType) {
    if (LoadType->isPointerTy()) {
      if (cast<PointerType>(LoadType)->getAddressSpace() == 10) {
        Type *EmptyStructTy = StructType::get(LoadType->getContext());
        Type *PtrAddrSpace0 = getPointerType(EmptyStructTy, 0);
        Value *Ptr0 = Builder.CreateIntToPtr(StoredVal, PtrAddrSpace0);
        StoredVal = Builder.CreateAddrSpaceCast(Ptr0, LoadType);
      } else {
        StoredVal = Builder.CreateIntToPtr(StoredVal, LoadType);
      }
    } else {
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
  uint64_t SourceSize = 0;
  if (auto *SI = dyn_cast<StoreInst>(Source)) {
    if (!getFixedTypeStoreSize(DL, SI->getValueOperand()->getType(),
                               SourceSize))
      return false;
  } else if (auto *LI = dyn_cast<LoadInst>(Source)) {
    if (!getFixedTypeStoreSize(DL, LI->getType(), SourceSize))
      return false;
  } else {
    return false;
  }

  uint64_t RelOffset = 0;
  return coversRange(SourceOffset, SourceSize, TargetOffset, TargetSize,
                     RelOffset);
}

// Helper to check if two memory ranges alias
// Range1: [Offset1, Offset1 + Size1)
// Range2: [Offset2, Offset2 + Size2)
static bool memoryRangesAlias(const APInt &Offset1, uint64_t Size1,
                              const APInt &Offset2, uint64_t Size2) {
  unsigned W =
      std::max<unsigned>({Offset1.getBitWidth(), Offset2.getBitWidth(), 64}) +
      2;
  APInt Start1 = Offset1.sext(W);
  APInt End1 = Start1 + APInt(W, Size1);
  APInt Start2 = Offset2.sext(W);
  APInt End2 = Start2 + APInt(W, Size2);

  // Check if range2 ends before range1 begins
  if (End2.sle(Start1))
    return false;

  // Check if range1 ends before range2 begins
  if (End1.sle(Start2))
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
        if (!LI->isSimple()) {
          // Volatile/atomic loads cannot be eliminated or reordered across.
          return false;
        }
        uint64_t LoadSize = 0;
        if (getFixedTypeStoreSize(DL, LI->getType(), LoadSize))
          Loads.push_back({LI, CurrentOffset});
      } else if (auto *SI = dyn_cast<StoreInst>(Usr)) {
        // Reject storing the pointer value itself (including self-stores
        // `store ptr %p, ptr %p`, which escape `%p` into memory).
        if (SI->getValueOperand() == V) {
          return false;
        }
        if (!SI->isSimple()) {
          // Volatile/atomic stores cannot be forwarded or reordered across.
          return false;
        }
        uint64_t StoreSize = 0;
        if (!getFixedTypeStoreSize(DL, SI->getValueOperand()->getType(),
                                   StoreSize)) {
          // Scalable store has unknown byte extent and could alias any offset.
          return false;
        }
        Stores.push_back({SI, CurrentOffset});
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(Usr)) {
        if (GEP->getPointerOperand() != V || !GEP->getType()->isPointerTy()) {
          return false;
        }
        // Compute the offset for this GEP
        APInt GEPOffset(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (!GEP->accumulateConstantOffset(DL, GEPOffset)) {
          // Cannot compute constant offset - reject this argument
          return false;
        }
        if (GEPOffset.getBitWidth() != CurrentOffset.getBitWidth()) {
          return false;
        }

        APInt NewOffset = CurrentOffset + GEPOffset;
        ToProcess.push_back({GEP, NewOffset});
      } else if (auto *CI = dyn_cast<CastInst>(Usr)) {
        // Only pointer-preserving casts maintain the same address/offset.
        // Reject ptrtoint/etc., which can truncate bits or escape the pointer.
        if (!isa<BitCastInst, AddrSpaceCastInst>(CI) ||
            !CI->getType()->isPointerTy() ||
            DL.getIndexTypeSizeInBits(CI->getType()) !=
                CurrentOffset.getBitWidth()) {
          return false;
        }
        ToProcess.push_back({CI, CurrentOffset});
      } else if (auto *Call = dyn_cast<CallInst>(Usr)) {
        // Only check nocapture on actual call arguments, not callee or bundles.
        if (!Call->isArgOperand(&U)) {
          return false;
        }
        unsigned ArgIdx = Call->getArgOperandNo(&U);
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

static bool hasInterveningMod(Instruction *Start, Instruction *End,
                              LoadInst *LI, AAResults &AA,
                              TargetLibraryInfo &TLI) {
  assert(Start->getParent() == End->getParent() &&
         "Start and End must be in the same BasicBlock");
  for (Instruction *I = Start->getNextNode(); I != End; I = I->getNextNode()) {
    if (auto *OtherLI = dyn_cast<LoadInst>(I)) {
      if (!OtherLI->isSimple()) {
        return true;
      }
    }
    if (I->mayWriteToMemory()) {
      if (writesToMemoryReadBy(nullptr, AA, TLI, LI, I)) {
        return true;
      }
    }
  }
  return false;
}

// Main optimization function
bool simplifyGVN(Function &F, DominatorTree &DT, const DataLayout &DL,
                 AAResults &AA, TargetLibraryInfo &TLI) {
  bool Changed = false;

  // Find candidates (noalias arguments, allocas, and call returns)
  SmallVector<std::pair<Value *, bool>, 4> CandidateArgs;
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      CandidateArgs.push_back({&Arg, Arg.hasNoAliasAttr()});
    }
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<AllocaInst>(&I)) {
        CandidateArgs.push_back({&I, true});
      } else if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->getType()->isPointerTy()) {
          CandidateArgs.push_back({CB, CB->hasRetAttr(Attribute::NoAlias)});
        }
      }
    }
  }

  if (CandidateArgs.empty())
    return false;

  // For each candidate argument, collect stores and loads with their offsets
  for (auto &pair : CandidateArgs) {
    Value *Arg = pair.first;
    bool isNoAlias = pair.second;

    // Collect all stores and loads to this argument with offsets
    SmallVector<std::pair<StoreInst *, APInt>, 8> Stores;
    SmallVector<std::pair<LoadInst *, APInt>, 8> Loads;
    SmallVector<std::pair<CallInst *, APInt>, 8> Calls;

    // Collect memory operations; reject candidate if pointer can escape.
    if (!collectMemoryOps(Arg, DL, Stores, Loads, Calls)) {
      // Argument has uses that prevent optimization
      continue;
    }

    APInt ZeroOffset(DL.getIndexTypeSizeInBits(Arg->getType()), 0);

    // Try to forward {stores, previous loads} to loads using simplified
    // algorithm
    for (auto &[LI, LoadOffset] : Loads) {
      if (!LI || !DT.isReachableFromEntry(LI->getParent()))
        continue;

      uint64_t LoadSize = 0;
      if (!getFixedTypeStoreSize(DL, LI->getType(), LoadSize))
        continue;

      // Step 1: Find all stores that may alias with this load
      SmallVector<std::tuple<Instruction *, APInt, uint64_t>, 8> AliasingStores;
      for (auto &[SI, StoreOffset] : Stores) {
        uint64_t StoreSize = 0;
        if (!getFixedTypeStoreSize(DL, SI->getValueOperand()->getType(),
                                   StoreSize))
          continue;
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

        bool Safe = true;
        if (!isNoAlias) {
          if (SI->getParent() != LI->getParent()) {
            Safe = false;
          } else {
            Safe = !hasInterveningMod(SI, LI, LI, AA, TLI);
          }
        }
        if (Safe) {
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
        }
        if (LI == nullptr)
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

      // Step 5: Track the last aliasing store/call before LI in LI's block (for
      // same-block forwarding) AND the last aliasing store/call in every block
      // including LI's own block (for backward BFS across loop backedges).
      BasicBlock *LIBlock = LI->getParent();
      Instruction *LastStoreBeforeLI = nullptr;
      DenseMap<BasicBlock *, std::tuple<Instruction *, APInt, uint64_t>>
          LastStoreInBlock;
      for (auto &[SI, StoreOffset, Size] : AliasingStores) {
        BasicBlock *BB = SI->getParent();
        if (BB == LIBlock && SI->comesBefore(LI)) {
          if (!LastStoreBeforeLI || LastStoreBeforeLI->comesBefore(SI)) {
            LastStoreBeforeLI = SI;
          }
        }
        auto &Entry = LastStoreInBlock[BB];
        if (!std::get<0>(Entry) || std::get<0>(Entry)->comesBefore(SI)) {
          Entry = {SI, StoreOffset, Size};
        }
      }

      // Step 6: Check if LI's parent block has a dominating+covering store/load
      if (LastStoreBeforeLI) {
        Instruction *SI = LastStoreBeforeLI;

        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (SI == DCS ||
              (DCS->getParent() == LI->getParent() && SI->comesBefore(DCS))) {
            bool Safe = true;
            if (!isNoAlias) {
              Safe = !hasInterveningMod(DCS, LI, LI, AA, TLI);
            }
            if (!Safe)
              continue;

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
        continue;
      } else {
        for (auto &&[DCS, StoreOffset, StoredVal] : DominatingCoveringStores) {
          if (DCS->getParent() == LI->getParent()) {
            bool Safe = true;
            if (!isNoAlias) {
              Safe = !hasInterveningMod(DCS, LI, LI, AA, TLI);
            }
            if (!Safe)
              continue;

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

      if (isNoAlias) {
        // Step 7: BFS backwards from LI's parent block. Using LastStoreInBlock
        // ensures any store/call occurring after LI in LIBlock is observed if
        // LIBlock is reachable along a loop backedge.
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

          auto It = LastStoreInBlock.find(BB);
          if (It != LastStoreInBlock.end()) {
            StoreInst *SI = dyn_cast<StoreInst>(std::get<0>(It->second));
            APInt StoreOffset = std::get<1>(It->second);

            if (!SI || !dominatesAndCovers(SI, LI, StoreOffset, LoadOffset,
                                           LoadSize, DL, DT)) {
              // Non-dominating+covering store or call on path, bail
              Candidate = nullptr;
              break;
            }

            // Found dominating+covering store; since it is the last aliasing
            // store in BB and covers LI, stop searching past BB.
            if (!Candidate) {
              Candidate = SI;
              CandidateOffset = StoreOffset;
            } else if (Candidate != SI) {
              // Multiple different candidates, bail
              Candidate = nullptr;
              break;
            }
            continue;
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
  }
  return Changed;
}

class SimpleGVN final : public FunctionPass {
public:
  static char ID;
  SimpleGVN() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    auto &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
    const DataLayout &DL = F.getParent()->getDataLayout();
    return simplifyGVN(F, DT, DL, AA, TLI);
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
  Changed = simplifyGVN(F, FAM.getResult<DominatorTreeAnalysis>(F), DL,
                        FAM.getResult<AAManager>(F),
                        FAM.getResult<TargetLibraryAnalysis>(F));
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

llvm::AnalysisKey SimpleGVNNewPM::Key;
