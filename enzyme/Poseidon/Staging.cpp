//=- Staging.cpp - shared-memory staging narrowing ------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Staging.h"
#include "Expansion.h"
#include "Flags.h"
#include "Optimize.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

// Narrow addrspace(3) staging buffers whose FP64 values are only ever consumed
// at FP32: stride K -> K/2, `load float` + fpext, fptrunc at the stores. Fires
// only when every consumer of a staged load rounds to FP32 (fptrunc or fcmp),
// which keeps the transform value-preserving.
bool narrowSharedStaging(llvm::Function &F) {
  using namespace llvm;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  for (GlobalVariable &G : M->globals()) {
    if (G.getAddressSpace() != 3)
      continue;

    SmallVector<Value *, 4> casts;
    for (User *U : G.users()) {
      if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::AddrSpaceCast)
          casts.push_back(CE);
      } else if (auto *AC = dyn_cast<AddrSpaceCastInst>(U)) {
        if (AC->getFunction() == &F)
          casts.push_back(AC);
      }
    }
    if (casts.empty())
      continue;

    SmallVector<GetElementPtrInst *, 8> elemGEPs;
    SmallVector<GetElementPtrInst *, 32> compGEPs;
    SmallVector<LoadInst *, 64> loads;
    SmallVector<StoreInst *, 16> stores;
    SmallVector<MemCpyInst *, 8> memcpyStores;
    int strideK = 0;
    bool bail = false;

    for (Value *C : casts) {
      for (User *U : C->users()) {
        auto *GEP = dyn_cast<GetElementPtrInst>(U);
        if (!GEP || GEP->getFunction() != &F) {
          if (auto *I = dyn_cast<Instruction>(U))
            if (I->getFunction() == &F)
              bail = true;
          continue;
        }
        auto *AT = dyn_cast<ArrayType>(GEP->getSourceElementType());
        if (!AT || !AT->getElementType()->isIntegerTy(8) ||
            (AT->getNumElements() % 2) != 0) {
          bail = true;
          continue;
        }
        if (strideK == 0)
          strideK = (int)AT->getNumElements();
        else if (strideK != (int)AT->getNumElements())
          bail = true;
        elemGEPs.push_back(GEP);
      }
    }
    if (flags::Print)
      llvm::errs() << "[narrow] G=@" << G.getName() << " casts=" << casts.size()
                   << " elemGEPs=" << elemGEPs.size() << " strideK=" << strideK
                   << " bail=" << bail << "\n";
    if (bail || elemGEPs.empty())
      continue;

    SmallVector<Value *, 16> work(elemGEPs.begin(), elemGEPs.end());
    SmallPtrSet<Value *, 32> seen(work.begin(), work.end());
    while (!work.empty() && !bail) {
      Value *P = work.pop_back_val();
      for (User *U : P->users()) {
        if (auto *L = dyn_cast<LoadInst>(U)) {
          if (L->getType() != DoubleTy || L->getPointerOperand() != P) {
            if (flags::Print)
              llvm::errs() << "[narrow] bail: load " << *L << "\n";
            bail = true;
            break;
          }
          // Value-preserving only if every consumer rounds to FP32: admit
          // fptrunc-to-float and fcmp guards; a genuine FP64 consumer means the
          // solver kept FP64 -> bail this buffer.
          for (User *LU : L->users()) {
            if (auto *FT = dyn_cast<FPTruncInst>(LU))
              if (FT->getType()->isFloatTy())
                continue;
            if (isa<FCmpInst>(LU))
              continue;
            if (flags::Print)
              llvm::errs() << "[narrow] bail: FP64 consumer of staged load: "
                           << *LU << "\n";
            bail = true;
            break;
          }
          if (bail)
            break;
          loads.push_back(L);
        } else if (auto *S = dyn_cast<StoreInst>(U)) {
          if (S->getValueOperand()->getType() != DoubleTy ||
              S->getPointerOperand() != P) {
            bail = true;
            break;
          }
          stores.push_back(S);
        } else if (auto *CG = dyn_cast<GetElementPtrInst>(U)) {
          if (!CG->getSourceElementType()->isIntegerTy(8) ||
              CG->getNumIndices() != 1 ||
              !isa<ConstantInt>(CG->getOperand(1))) {
            if (flags::Print)
              llvm::errs() << "[narrow] bail: comp GEP " << *CG << "\n";
            bail = true;
            break;
          }
          compGEPs.push_back(CG);
          if (seen.insert(CG).second)
            work.push_back(CG);
        } else if (auto *MC = dyn_cast<MemCpyInst>(U)) {
          // tile-load `shared[tid] = global[tid]` lowered to a struct memcpy.
          // Only handle: staging ptr is the DEST, constant length == strideK.
          auto *Len = dyn_cast<ConstantInt>(MC->getLength());
          if (MC->getArgOperand(0) != P || !Len ||
              Len->getZExtValue() != (uint64_t)strideK || (strideK % 8) != 0) {
            if (flags::Print)
              llvm::errs() << "[narrow] bail: memcpy " << *MC << "\n";
            bail = true;
            break;
          }
          memcpyStores.push_back(MC);
        } else {
          if (flags::Print)
            llvm::errs() << "[narrow] bail: unexpected user of staging ptr: "
                         << *U << "\n";
          bail = true;
          break;
        }
      }
    }
    if (flags::Print)
      llvm::errs() << "[narrow]   loads=" << loads.size()
                   << " stores=" << stores.size()
                   << " compGEPs=" << compGEPs.size() << " bail=" << bail
                   << "\n";
    if (bail || (loads.empty() && stores.empty() && memcpyStores.empty()))
      continue;

    // Rewrite. Order: elem GEPs, comp GEPs, loads, stores.
    for (auto *EG : elemGEPs) {
      IRBuilder<> B(EG);
      SmallVector<Value *, 2> idx(EG->idx_begin(), EG->idx_end());
      Value *ng = B.CreateGEP(ArrayType::get(I8Ty, strideK / 2),
                              EG->getPointerOperand(), idx, EG->getName(),
                              EG->isInBounds());
      EG->replaceAllUsesWith(ng);
    }
    for (auto *CG : compGEPs) {
      IRBuilder<> B(CG);
      int64_t off = cast<ConstantInt>(CG->getOperand(1))->getSExtValue();
      Value *no =
          B.CreateGEP(I8Ty, CG->getPointerOperand(), B.getInt64(off / 2),
                      CG->getName(), CG->isInBounds());
      CG->replaceAllUsesWith(no);
    }
    for (auto *L : loads) {
      IRBuilder<> B(L);
      Value *nl = B.CreateAlignedLoad(FloatTy, L->getPointerOperand(), Align(4),
                                      L->getName());
      Value *back =
          B.CreateFPExt(nl, DoubleTy); // O3 folds fptrunc(fpext(x))->x
      L->replaceAllUsesWith(back);
      L->eraseFromParent();
    }
    for (auto *S : stores) {
      IRBuilder<> B(S);
      Value *vf = B.CreateFPTrunc(S->getValueOperand(), FloatTy);
      B.CreateAlignedStore(vf, S->getPointerOperand(), Align(4));
      S->eraseFromParent();
    }
    for (auto *MC : memcpyStores) {
      IRBuilder<> B(MC);
      Value *dst = MC->getArgOperand(0);
      Value *src = MC->getArgOperand(1);
      int n = strideK / 8;
      for (int k = 0; k < n; ++k) {
        Value *sp = k ? B.CreateGEP(I8Ty, src, B.getInt64(8 * k)) : src;
        Value *d = B.CreateAlignedLoad(DoubleTy, sp, Align(8));
        Value *f = B.CreateFPTrunc(d, FloatTy);
        Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(4 * k)) : dst;
        B.CreateAlignedStore(f, dp, Align(4));
      }
      MC->eraseFromParent();
    }
    for (auto *CG : compGEPs)
      if (CG->use_empty())
        CG->eraseFromParent();
    for (auto *EG : elemGEPs)
      if (EG->use_empty())
        EG->eraseFromParent();
    anyChange = true;
    llvm::errs() << "[poseidon] narrowed shared staging buffer @" << G.getName()
                 << " (" << loads.size() << " loads, " << stores.size()
                 << " stores, stride " << strideK << "->" << strideK / 2
                 << ")\n";
  }
  return anyChange;
}

// df64 analog of narrowSharedStaging: when every staged load is consumed by an
// emitF64ToDS Dekker split, store the {hi,lo} pair once at the staging store
// (hi@+0, lo@+4 in the same 8-byte slot) and turn each split into a pair load.
bool narrowSharedStagingDS(llvm::Function &F) {
  using namespace llvm;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);

  auto matchSplit = [&](LoadInst *L, FPTruncInst *&hiT, FPTruncInst *&loT,
                        FPExtInst *&hib, BinaryOperator *&los) -> bool {
    hiT = loT = nullptr;
    hib = nullptr;
    los = nullptr;
    for (User *U : L->users()) {
      if (auto *FT = dyn_cast<FPTruncInst>(U)) {
        if (!FT->getType()->isFloatTy() || hiT)
          return false;
        hiT = FT;
      } else if (auto *BO = dyn_cast<BinaryOperator>(U)) {
        if (BO->getOpcode() != Instruction::FSub ||
            !BO->getType()->isDoubleTy() || los)
          return false;
        los = BO;
      } else if (!isa<FCmpInst>(U)) {
        return false;
      }
    }
    if (!hiT || !los || los->getOperand(0) != L)
      return false;
    hib = dyn_cast<FPExtInst>(los->getOperand(1));
    if (!hib || hib->getOperand(0) != hiT || !los->hasOneUse())
      return false;
    loT = dyn_cast<FPTruncInst>(*los->user_begin());
    return loT && loT->getType()->isFloatTy();
  };

  bool anyChange = false;
  for (GlobalVariable &G : M->globals()) {
    if (G.getAddressSpace() != 3)
      continue;
    SmallVector<Value *, 4> casts;
    for (User *U : G.users()) {
      if (auto *CE = dyn_cast<ConstantExpr>(U)) {
        if (CE->getOpcode() == Instruction::AddrSpaceCast)
          casts.push_back(CE);
      } else if (auto *AC = dyn_cast<AddrSpaceCastInst>(U)) {
        if (AC->getFunction() == &F)
          casts.push_back(AC);
      }
    }
    if (casts.empty())
      continue;

    SmallVector<GetElementPtrInst *, 8> elemGEPs;
    int strideK = 0;
    bool bail = false;
    for (Value *C : casts)
      for (User *U : C->users()) {
        auto *GEP = dyn_cast<GetElementPtrInst>(U);
        if (!GEP || GEP->getFunction() != &F) {
          if (auto *I = dyn_cast<Instruction>(U))
            if (I->getFunction() == &F)
              bail = true;
          continue;
        }
        auto *AT = dyn_cast<ArrayType>(GEP->getSourceElementType());
        if (!AT || !AT->getElementType()->isIntegerTy(8) ||
            (AT->getNumElements() % 8) != 0) {
          bail = true;
          continue;
        }
        if (strideK == 0)
          strideK = (int)AT->getNumElements();
        else if (strideK != (int)AT->getNumElements())
          bail = true;
        elemGEPs.push_back(GEP);
      }
    if (bail || elemGEPs.empty())
      continue;

    SmallVector<LoadInst *, 64> loads;
    SmallVector<StoreInst *, 16> stores;
    SmallVector<MemCpyInst *, 8> memcpys;
    SmallVector<Value *, 16> work(elemGEPs.begin(), elemGEPs.end());
    SmallPtrSet<Value *, 32> seen(work.begin(), work.end());
    while (!work.empty() && !bail) {
      Value *P = work.pop_back_val();
      for (User *U : P->users()) {
        if (auto *L = dyn_cast<LoadInst>(U)) {
          FPTruncInst *hiT, *loT;
          FPExtInst *hib;
          BinaryOperator *los;
          if (L->getType() != DoubleTy || L->getPointerOperand() != P ||
              !matchSplit(L, hiT, loT, hib, los)) {
            bail = true;
            break;
          }
          loads.push_back(L);
        } else if (auto *S = dyn_cast<StoreInst>(U)) {
          if (S->getValueOperand()->getType() != DoubleTy ||
              S->getPointerOperand() != P) {
            bail = true;
            break;
          }
          stores.push_back(S);
        } else if (auto *CG = dyn_cast<GetElementPtrInst>(U)) {
          if (!CG->getSourceElementType()->isIntegerTy(8) ||
              CG->getNumIndices() != 1 ||
              !isa<ConstantInt>(CG->getOperand(1))) {
            bail = true;
            break;
          }
          if (seen.insert(CG).second)
            work.push_back(CG);
        } else if (auto *MC = dyn_cast<MemCpyInst>(U)) {
          auto *Len = dyn_cast<ConstantInt>(MC->getLength());
          if (MC->getArgOperand(0) != P || !Len ||
              Len->getZExtValue() != (uint64_t)strideK || (strideK % 8) != 0) {
            bail = true;
            break;
          }
          memcpys.push_back(MC);
        } else {
          bail = true;
          break;
        }
      }
    }
    if (bail || (loads.empty() && stores.empty() && memcpys.empty()))
      continue;

    auto splitStore = [&](IRBuilder<> &B, Value *val, Value *p) {
      Value *hi = B.CreateFPTrunc(val, FloatTy);
      Value *lo = B.CreateFPTrunc(
          B.CreateFSub(val, B.CreateFPExt(hi, DoubleTy)), FloatTy);
      B.CreateAlignedStore(hi, p, Align(4));
      B.CreateAlignedStore(lo, B.CreateGEP(I8Ty, p, B.getInt64(4)), Align(4));
    };

    for (auto *L : loads) {
      IRBuilder<> B(L);
      Value *p = L->getPointerOperand();
      Value *nhi = B.CreateAlignedLoad(FloatTy, p, Align(4));
      Value *nlo = B.CreateAlignedLoad(
          FloatTy, B.CreateGEP(I8Ty, p, B.getInt64(4)), Align(4));
      FPTruncInst *hiT, *loT;
      FPExtInst *hib;
      BinaryOperator *los;
      matchSplit(L, hiT, loT, hib, los); // guaranteed by the gate above
      hiT->replaceAllUsesWith(nhi);
      loT->replaceAllUsesWith(nlo);
      loT->eraseFromParent();
      los->eraseFromParent();
      hib->eraseFromParent();
      hiT->eraseFromParent();
      // Tolerated fcmp guards still read the original double; give them
      // fpext(nhi), which O3 folds to a float compare. Sign-equivalent for
      // realistic values.
      if (!L->use_empty())
        L->replaceAllUsesWith(B.CreateFPExt(nhi, DoubleTy));
      L->eraseFromParent();
    }
    for (auto *S : stores) {
      IRBuilder<> B(S);
      splitStore(B, S->getValueOperand(), S->getPointerOperand());
      S->eraseFromParent();
    }
    for (auto *MC : memcpys) {
      IRBuilder<> B(MC);
      Value *dst = MC->getArgOperand(0), *src = MC->getArgOperand(1);
      for (int k = 0; k < strideK / 8; ++k) {
        Value *sp = k ? B.CreateGEP(I8Ty, src, B.getInt64(8 * k)) : src;
        Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(8 * k)) : dst;
        splitStore(B, B.CreateAlignedLoad(DoubleTy, sp, Align(8)), dp);
      }
      MC->eraseFromParent();
    }
    anyChange = true;
    if (flags::Print)
      llvm::errs() << "[poseidon] narrowed shared staging (df64) @"
                   << G.getName() << " (" << loads.size() << " loads, "
                   << stores.size() << " stores, " << memcpys.size()
                   << " memcpys)\n";
  }
  return anyChange;
}

//===----------------------------------------------------------------------===//
// Narrowing of addrspace(3) staging buffers that reach the rewritten function
// as POINTER PARAMETERS.
//
// Two transforms, matching the two global-rooted arms above:
//   FP32: the buffer is a flat array of doubles, so narrowing is exactly
//         "halve every byte offset from the base and load/store float".
//   DS:   the double slot already holds two floats, so the LAYOUT is unchanged
//         and only the accesses move: store {hi@+0, lo@+4} once at the staging
//         store and rewire each read's Dekker split to a direct pair load.
//===----------------------------------------------------------------------===//
namespace {

struct ParamStagedAccess {
  SmallVector<GetElementPtrInst *, 48> order; // rewrite order, parents first
  SmallPtrSet<Value *, 32> arrSet;            // [N x i8] GEPs
  SmallPtrSet<Value *, 32> byteSet;           // i8 GEPs, offset a multiple of 8
  SmallPtrSet<Value *, 32> dblSet;            // double-typed GEPs
  SmallVector<LoadInst *, 128> loads;
  SmallVector<StoreInst *, 32> stores;
  SmallVector<unsigned, 2> argIdx;
  unsigned foreignConsumers = 0; // readers the strict gate would have refused
};

// `load double %L` consumed exactly by emitF64ToDS:
//   hi = fptrunc(L); los = fsub double(L, fpext(hi)); lo = fptrunc(los)
static bool matchDekkerSplit(LoadInst *L, FPTruncInst *&hiT, FPTruncInst *&loT,
                             FPExtInst *&hib, BinaryOperator *&los) {
  hiT = loT = nullptr;
  hib = nullptr;
  los = nullptr;
  for (User *U : L->users()) {
    if (auto *FT = dyn_cast<FPTruncInst>(U)) {
      if (!FT->getType()->isFloatTy() || hiT)
        return false;
      hiT = FT;
    } else if (auto *BO = dyn_cast<BinaryOperator>(U)) {
      if (BO->getOpcode() != Instruction::FSub ||
          !BO->getType()->isDoubleTy() || los)
        return false;
      los = BO;
    } else if (!isa<FCmpInst>(U)) {
      return false;
    }
  }
  if (!hiT || !los || los->getOperand(0) != L)
    return false;
  hib = dyn_cast<FPExtInst>(los->getOperand(1));
  if (!hib || hib->getOperand(0) != hiT || !los->hasOneUse())
    return false;
  loT = dyn_cast<FPTruncInst>(*los->user_begin());
  return loT && loT->getType()->isFloatTy();
}

// Resolve G to parameters of F and walk the pointer graph. False if any access
// is outside the modelled shapes (the buffer is then left alone).
static bool collectParamStaged(Function &F, Function *Proxy, GlobalVariable &G,
                               bool dsMode, bool speculative,
                               ParamStagedAccess &out) {
  Module *M = F.getParent();
  const DataLayout &DL = M->getDataLayout();
  Type *DoubleTy = Type::getDoubleTy(M->getContext());

  SmallVector<Value *, 4> casts;
  for (User *U : G.users()) {
    if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        casts.push_back(CE);
        continue;
      }
    } else if (isa<AddrSpaceCastInst>(U)) {
      casts.push_back(U);
      continue;
    }
    // Any other direct user of the global (a GEP in the kernel, say) means
    // accesses exist that this arm would not rewrite: leave the buffer alone.
    return false;
  }
  if (casts.empty())
    return false;
  // Every use of every cast must be a call (the buffer is handed to a callee)
  // or an instruction inside F; a GEP in a THIRD function would be missed.
  for (Value *C : casts)
    for (User *CU : C->users()) {
      if (isa<CallBase>(CU))
        continue;
      auto *I = dyn_cast<Instruction>(CU);
      if (!I || I->getFunction() != &F)
        return false;
    }

  for (unsigned i = 0; i < F.arg_size(); ++i) {
    if (!F.getArg(i)->getType()->isPointerTy())
      continue;
    if (Value *SA = siteArg(Proxy, i))
      if (llvm::is_contained(casts, SA))
        out.argIdx.push_back(i);
  }
  if (out.argIdx.empty())
    for (Value *C : casts)
      for (User *CU : C->users()) {
        auto *CB = dyn_cast<CallBase>(CU);
        if (!CB || CB->getCalledFunction() != Proxy)
          continue;
        for (unsigned i = 0; i < CB->arg_size() && i < F.arg_size(); ++i)
          if (CB->getArgOperand(i) == C &&
              F.getArg(i)->getType()->isPointerTy() &&
              !llvm::is_contained(out.argIdx, i))
            out.argIdx.push_back(i);
      }
  if (out.argIdx.empty())
    return false;

  SmallPtrSet<Value *, 32> seen;
  SmallVector<Value *, 16> work;
  for (unsigned i : out.argIdx) {
    work.push_back(F.getArg(i));
    seen.insert(F.getArg(i));
  }
  while (!work.empty()) {
    Value *P = work.pop_back_val();
    for (User *U : P->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || I->getFunction() != &F)
        return false;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        if (GEP->getPointerOperand() != P)
          return false;
        Type *SET = GEP->getSourceElementType();
        if (auto *AT = dyn_cast<ArrayType>(SET)) {
          if (!AT->getElementType()->isIntegerTy(8) ||
              (AT->getNumElements() % 2) != 0 || GEP->getNumIndices() != 1) {
            if (flags::Print)
              llvm::errs() << "[narrow-param] bail: array GEP " << *GEP << "\n";
            return false;
          }
          out.arrSet.insert(GEP);
        } else if (SET->isIntegerTy(8)) {
          // Halving the byte offset is the whole FP32 transform, so it must be
          // a whole number of doubles: a constant multiple of 8, or a value
          // whose low three bits are provably zero (clang emits
          // `gep i8, p, idx<<3`).
          bool ok = GEP->getNumIndices() == 1;
          if (ok) {
            if (auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1)))
              ok = (C->getSExtValue() % 8) == 0;
            else
              ok = llvm::computeKnownBits(GEP->getOperand(1), DL)
                       .countMinTrailingZeros() >= 3;
          }
          if (!ok) {
            if (flags::Print)
              llvm::errs() << "[narrow-param] bail: byte GEP " << *GEP << "\n";
            return false;
          }
          out.byteSet.insert(GEP);
        } else if (SET->isDoubleTy()) {
          if (GEP->getNumIndices() != 1)
            return false;
          out.dblSet.insert(GEP);
        } else {
          if (flags::Print)
            llvm::errs() << "[narrow-param] bail: GEP " << *GEP << "\n";
          return false;
        }
        if (seen.insert(GEP).second) {
          out.order.push_back(GEP);
          work.push_back(GEP);
        }
      } else if (auto *L = dyn_cast<LoadInst>(U)) {
        if (L->getType() != DoubleTy || L->getPointerOperand() != P) {
          if (flags::Print)
            llvm::errs() << "[narrow-param] bail: load " << *L << "\n";
          return false;
        }
        FPTruncInst *hiT, *loT;
        FPExtInst *hib;
        BinaryOperator *los;
        bool isSplit = matchDekkerSplit(L, hiT, loT, hib, los);
        if (dsMode) {
          // DS wants Dekker splits; anything else is a foreign reader.
          if (!isSplit) {
            ++out.foreignConsumers;
            if (!speculative) {
              if (flags::Print)
                llvm::errs() << "[narrow-param] bail: non-split reader of "
                                "staged load: "
                             << *L << "\n";
              return false;
            }
          }
        } else {
          // FP32 must not touch a Dekker split: narrowing the slot to a single
          // float makes `lo` identically zero and the expansion candidate would
          // price as free. That buffer belongs to the DS arm.
          if (isSplit) {
            if (flags::Print)
              llvm::errs() << "[narrow-param] bail: Dekker-split reader (DS "
                              "arm's buffer): "
                           << *L << "\n";
            return false;
          }
          for (User *LU : L->users()) {
            if (auto *FT = dyn_cast<FPTruncInst>(LU))
              if (FT->getType()->isFloatTy())
                continue;
            if (isa<FCmpInst>(LU))
              continue;
            ++out.foreignConsumers;
            if (!speculative) {
              if (flags::Print)
                llvm::errs()
                    << "[narrow-param] bail: FP64 consumer of staged load: "
                    << *LU << "\n";
              return false;
            }
          }
        }
        out.loads.push_back(L);
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        if (S->getValueOperand()->getType() != DoubleTy ||
            S->getPointerOperand() != P) {
          if (flags::Print)
            llvm::errs() << "[narrow-param] bail: store " << *S << "\n";
          return false;
        }
        out.stores.push_back(S);
      } else {
        if (flags::Print)
          llvm::errs() << "[narrow-param] bail: unexpected user " << *U << "\n";
        return false;
      }
    }
  }
  return !(out.loads.empty() && out.stores.empty());
}

} // namespace

bool narrowSharedStagingParam(llvm::Function &F, llvm::Function *Proxy,
                              bool speculative) {
  if (!Proxy)
    Proxy = &F;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  for (GlobalVariable &G : M->globals()) {
    if (G.getAddressSpace() != 3)
      continue;
    ParamStagedAccess acc;
    if (!collectParamStaged(F, Proxy, G, /*dsMode=*/false, speculative, acc))
      continue;
    if (flags::Print) {
      llvm::errs() << "[narrow-param fp32] G=@" << G.getName() << " args=";
      for (unsigned i : acc.argIdx)
        llvm::errs() << i << " ";
      llvm::errs() << " geps=" << acc.order.size()
                   << " loads=" << acc.loads.size()
                   << " stores=" << acc.stores.size()
                   << " foreign=" << acc.foreignConsumers
                   << " spec=" << speculative << "\n";
    }

    SmallVector<Instruction *, 48> deadGEPs;
    for (GetElementPtrInst *GEP : acc.order) {
      IRBuilder<> B(GEP);
      SmallVector<Value *, 2> idx(GEP->idx_begin(), GEP->idx_end());
      Value *ng = nullptr;
      if (acc.arrSet.contains(GEP)) {
        auto *AT = cast<ArrayType>(GEP->getSourceElementType());
        ng = B.CreateGEP(ArrayType::get(I8Ty, AT->getNumElements() / 2),
                         GEP->getPointerOperand(), idx, GEP->getName(),
                         GEP->isInBounds());
      } else if (acc.byteSet.contains(GEP)) {
        Value *off = GEP->getOperand(1);
        Value *half =
            isa<ConstantInt>(off)
                ? (Value *)B.getInt64(cast<ConstantInt>(off)->getSExtValue() /
                                      2)
                : B.CreateLShr(off, ConstantInt::get(off->getType(), 1), "",
                               /*isExact=*/true);
        ng = B.CreateGEP(I8Ty, GEP->getPointerOperand(), half, GEP->getName(),
                         GEP->isInBounds());
      } else {
        ng = B.CreateGEP(FloatTy, GEP->getPointerOperand(), idx, GEP->getName(),
                         GEP->isInBounds());
      }
      GEP->replaceAllUsesWith(ng);
      deadGEPs.push_back(GEP);
    }
    for (auto *L : acc.loads) {
      IRBuilder<> B(L);
      Value *nl = B.CreateAlignedLoad(FloatTy, L->getPointerOperand(), Align(4),
                                      L->getName());
      Value *back = B.CreateFPExt(nl, DoubleTy); // O3 folds fptrunc(fpext(x))
      L->replaceAllUsesWith(back);
      L->eraseFromParent();
    }
    for (auto *S : acc.stores) {
      IRBuilder<> B(S);
      Value *vf = B.CreateFPTrunc(S->getValueOperand(), FloatTy);
      B.CreateAlignedStore(vf, S->getPointerOperand(), Align(4));
      S->eraseFromParent();
    }
    for (auto *I : llvm::reverse(deadGEPs))
      if (I->use_empty())
        I->eraseFromParent();

    anyChange = true;
    llvm::errs() << "[poseidon] narrowed param-staged shared buffer @"
                 << G.getName() << " to fp32 (" << acc.loads.size()
                 << " loads, " << acc.stores.size() << " stores"
                 << (speculative ? ", pricing-speculative" : "") << ")\n";
  }
  return anyChange;
}

bool narrowSharedStagingParamDS(llvm::Function &F, llvm::Function *Proxy,
                                bool speculative) {
  if (!Proxy)
    Proxy = &F;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  for (GlobalVariable &G : M->globals()) {
    if (G.getAddressSpace() != 3)
      continue;
    ParamStagedAccess acc;
    if (!collectParamStaged(F, Proxy, G, /*dsMode=*/true, speculative, acc))
      continue;
    if (flags::Print) {
      llvm::errs() << "[narrow-param df64] G=@" << G.getName() << " args=";
      for (unsigned i : acc.argIdx)
        llvm::errs() << i << " ";
      llvm::errs() << " geps=" << acc.order.size()
                   << " loads=" << acc.loads.size()
                   << " stores=" << acc.stores.size()
                   << " foreign=" << acc.foreignConsumers
                   << " spec=" << speculative << "\n";
    }

    // Layout size is unchanged (a double slot holds two floats), so the GEPs
    // are left exactly as they are; only the accesses move.
    auto splitStore = [&](IRBuilder<> &B, Value *val, Value *p) {
      Value *hi = B.CreateFPTrunc(val, FloatTy);
      Value *lo = B.CreateFPTrunc(
          B.CreateFSub(val, B.CreateFPExt(hi, DoubleTy)), FloatTy);
      B.CreateAlignedStore(hi, p, Align(4));
      B.CreateAlignedStore(lo, B.CreateGEP(I8Ty, p, B.getInt64(4)), Align(4));
    };

    for (auto *L : acc.loads) {
      IRBuilder<> B(L);
      Value *p = L->getPointerOperand();
      Value *nhi = B.CreateAlignedLoad(FloatTy, p, Align(4));
      Value *nlo = B.CreateAlignedLoad(
          FloatTy, B.CreateGEP(I8Ty, p, B.getInt64(4)), Align(4));
      FPTruncInst *hiT, *loT;
      FPExtInst *hib;
      BinaryOperator *los;
      if (matchDekkerSplit(L, hiT, loT, hib, los)) {
        hiT->replaceAllUsesWith(nhi);
        loT->replaceAllUsesWith(nlo);
        loT->eraseFromParent();
        los->eraseFromParent();
        hib->eraseFromParent();
        hiT->eraseFromParent();
      }
      // Any remaining reader (a tolerated fcmp guard, or -- in speculative
      // pricing -- another unit still reading this buffer at FP64) gets the
      // exact reconstruction hi+lo.
      if (!L->use_empty())
        L->replaceAllUsesWith(B.CreateFAdd(B.CreateFPExt(nhi, DoubleTy),
                                           B.CreateFPExt(nlo, DoubleTy)));
      L->eraseFromParent();
    }
    for (auto *S : acc.stores) {
      IRBuilder<> B(S);
      splitStore(B, S->getValueOperand(), S->getPointerOperand());
      S->eraseFromParent();
    }

    anyChange = true;
    llvm::errs() << "[poseidon] narrowed param-staged shared buffer @"
                 << G.getName() << " to df64 (" << acc.loads.size()
                 << " loads, " << acc.stores.size() << " stores"
                 << (speculative ? ", pricing-speculative" : "") << ")\n";
  }
  // Recorded on the proxy: the expansion materializer calls this early, so by
  // the time materializeFPSolution's post-pass calls it again the buffer is
  // already converted and the second call correctly reports "nothing to do".
  // Without the note that no-op reads as "narrowing never fired" and the
  // speculative-pricing soundness check warns on a solve where it did.
  if (anyChange && !speculative)
    Proxy->addFnAttr("poseidon-narrowed-param-staging-ds");
  return anyChange;
}

bool applyStagingNarrowing(Function &F, bool announce,
                           function_ref<void(Function &)> between,
                           ParamArm param, Function *paramProxy,
                           bool paramSpeculative, bool *narrowedParam) {
  bool changed = narrowSharedStaging(F);
  if (changed && announce && flags::Print)
    llvm::errs() << "[poseidon] applied shared-staging narrowing in "
                 << F.getName() << "\n";
  if (param != ParamArm::None) {
    Function *proxy = paramProxy ? paramProxy : &F;
    bool fired = false;
    if (param == ParamArm::Both || param == ParamArm::FP32)
      if (narrowSharedStagingParam(F, proxy, paramSpeculative)) {
        fired = true;
        if (announce && flags::Print)
          llvm::errs() << "[poseidon] applied param-staged shared narrowing in "
                       << F.getName() << "\n";
      }
    if (param == ParamArm::Both || param == ParamArm::DS)
      if (narrowSharedStagingParamDS(F, proxy, paramSpeculative)) {
        fired = true;
        if (announce && flags::Print)
          llvm::errs() << "[poseidon] applied param-staged df64 narrowing in "
                       << F.getName() << "\n";
      }
    // ... or it already fired during expansion materialization, which runs
    // before this post-pass and leaves it with nothing left to convert.
    if (F.hasFnAttribute("poseidon-narrowed-param-staging-ds"))
      fired = true;
    changed |= fired;
    if (narrowedParam)
      *narrowedParam = fired;
  }
  if (between)
    between(F);
  if (narrowSharedStagingDS(F)) {
    changed = true;
    if (announce && flags::Print)
      llvm::errs() << "[poseidon] applied shared-staging df64 narrowing in "
                   << F.getName() << "\n";
  }
  return changed;
}

} // namespace poseidon
