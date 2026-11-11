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
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "Targets/Target.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace poseidon {

// Narrow addrspace(3) staging buffers whose FP64 values are only ever consumed
// at FP32: stride K -> K/2, `load float` + fpext, fptrunc at the stores. Fires
// only when every consumer of a staged load rounds to FP32 (fptrunc or fcmp),
// which keeps the transform value-preserving.
bool narrowSharedStaging(const Options &O, llvm::Function &F) {
  using namespace llvm;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
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
    if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
              llvm::errs() << "[narrow] bail: memcpy " << *MC << "\n";
            bail = true;
            break;
          }
          memcpyStores.push_back(MC);
        } else {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow] bail: unexpected user of staging ptr: "
                         << *U << "\n";
          bail = true;
          break;
        }
      }
    }
    if (O.Diagnostics.Print)
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
bool narrowSharedStagingDS(const Options &O, llvm::Function &F) {
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
  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
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
    if (O.Diagnostics.Print)
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

struct DekkerSplit {
  FPTruncInst *hiT, *loT;
  FPExtInst *hib;
  BinaryOperator *los;
};

// A staged load the expansion split once per use it converted (a value read
// twice, as MFEM's stages read their shared inputs): every reader belongs to
// one of the splits, or is a tolerated fcmp.
static bool matchDekkerSplits(LoadInst *L, SmallVectorImpl<DekkerSplit> &out) {
  SmallVector<FPTruncInst *, 4> his;
  SmallVector<BinaryOperator *, 4> loss;
  for (User *U : L->users()) {
    auto *FT = dyn_cast<FPTruncInst>(U);
    auto *BO = dyn_cast<BinaryOperator>(U);
    if (FT && FT->getType()->isFloatTy())
      his.push_back(FT);
    else if (BO && BO->getOpcode() == Instruction::FSub &&
             BO->getOperand(0) == L && BO->getType()->isDoubleTy())
      loss.push_back(BO);
    else if (!isa<FCmpInst>(U))
      return false;
  }
  if (his.empty() || his.size() != loss.size())
    return false;
  SmallPtrSet<FPTruncInst *, 4> paired;
  for (BinaryOperator *los : loss) {
    auto *hib = dyn_cast<FPExtInst>(los->getOperand(1));
    if (!hib || !hib->hasOneUse() || !los->hasOneUse())
      return false;
    auto *hiT = dyn_cast<FPTruncInst>(hib->getOperand(0));
    auto *loT = dyn_cast<FPTruncInst>(*los->user_begin());
    if (!hiT || !llvm::is_contained(his, hiT) || !paired.insert(hiT).second ||
        !loT || !loT->getType()->isFloatTy())
      return false;
    out.push_back({hiT, loT, hib, los});
  }
  return true;
}

static bool walkStaged(const Options &O, Function &F, ArrayRef<Value *> roots,
                       bool dsMode, ParamStagedAccess &out);

// Resolve G to parameters of F and walk the pointer graph. False if any access
// is outside the modelled shapes (the buffer is then left alone).
static bool collectParamStaged(const Session &S, Function &F, Function *Proxy,
                               GlobalVariable &G, bool dsMode,
                               ParamStagedAccess &out) {
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
    if (Value *SA = siteArg(S, Proxy, i))
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

  SmallVector<Value *, 4> roots;
  for (unsigned i : out.argIdx)
    roots.push_back(F.getArg(i));
  return walkStaged(S.Opts, F, roots, dsMode, out);
}

// The buffer F accesses directly: each addrspacecast of G in F is a root once
// the constant expressions F reaches G through (offset views such as MFEM's
// `sm0 + 2`) are instructions. Shared memory lives for one launch, so only an
// access in a function F calls could meet F's; one of those, or an access of
// G that is not through a generic pointer, leaves the buffer alone.
static bool collectGlobalStaged(const Options &O, Function &F,
                                GlobalVariable &G, bool dsMode,
                                ParamStagedAccess &out) {
  convertUsersOfConstantsToInstructions({&G}, &F,
                                        /*RemoveDeadConstants=*/false);
  SetVector<Function *> callees;
  callees.insert(&F);
  for (size_t f = 0; f < callees.size(); ++f)
    for (Instruction &I : instructions(*callees[f]))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *callee = CB->getCalledFunction();
            callee && !callee->isDeclaration())
          callees.insert(callee);
  SmallVector<Value *, 8> roots;
  SmallVector<User *, 8> users(G.users());
  SmallPtrSet<User *, 16> seenUsers;
  while (!users.empty()) {
    User *U = users.pop_back_val();
    if (!seenUsers.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == &F) {
        if (!isa<AddrSpaceCastInst>(I))
          return false;
        roots.push_back(I);
      } else if (callees.contains(I->getFunction())) {
        return false;
      }
    } else if (isa<Constant>(U)) {
      users.append(U->user_begin(), U->user_end());
    }
  }
  return !roots.empty() && walkStaged(O, F, roots, dsMode, out);
}

static bool walkStaged(const Options &O, Function &F, ArrayRef<Value *> roots,
                       bool dsMode, ParamStagedAccess &out) {
  Module *M = F.getParent();
  const DataLayout &DL = M->getDataLayout();
  Type *DoubleTy = Type::getDoubleTy(M->getContext());
  SmallPtrSet<Value *, 32> seen;
  SmallVector<Value *, 16> work;
  for (Value *R : roots) {
    work.push_back(R);
    seen.insert(R);
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
            if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
              llvm::errs() << "[narrow-param] bail: byte GEP " << *GEP << "\n";
            return false;
          }
          out.byteSet.insert(GEP);
        } else if (SET->isDoubleTy()) {
          if (GEP->getNumIndices() != 1)
            return false;
          out.dblSet.insert(GEP);
        } else {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: GEP " << *GEP << "\n";
          return false;
        }
        if (seen.insert(GEP).second) {
          out.order.push_back(GEP);
          work.push_back(GEP);
        }
      } else if (auto *L = dyn_cast<LoadInst>(U)) {
        if (L->getType() != DoubleTy || L->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: load " << *L << "\n";
          return false;
        }
        FPTruncInst *hiT, *loT;
        FPExtInst *hib;
        BinaryOperator *los;
        bool isSplit = matchDekkerSplit(L, hiT, loT, hib, los);
        SmallVector<DekkerSplit, 2> splits;
        if (dsMode) {
          // DS wants Dekker splits; anything else is a foreign reader.
          if (!isSplit && !(O.Materialization.NarrowMultiviewStaging &&
                            matchDekkerSplits(L, splits))) {
            if (O.Diagnostics.Print)
              llvm::errs() << "[narrow-param] bail: non-split reader of "
                              "staged load: "
                           << *L << "\n";
            return false;
          }
        } else {
          // FP32 must not touch a Dekker split: narrowing the slot to a single
          // float makes `lo` identically zero and the expansion candidate would
          // price as free. That buffer belongs to the DS arm.
          if (isSplit) {
            if (O.Diagnostics.Print)
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
            if (O.Diagnostics.Print)
              llvm::errs()
                  << "[narrow-param] bail: FP64 consumer of staged load: "
                  << *LU << "\n";
            return false;
          }
        }
        out.loads.push_back(L);
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        if (S->getValueOperand()->getType() != DoubleTy ||
            S->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: store " << *S << "\n";
          return false;
        }
        out.stores.push_back(S);
      } else {
        if (O.Diagnostics.Print)
          llvm::errs() << "[narrow-param] bail: unexpected user " << *U << "\n";
        return false;
      }
    }
  }
  return !(out.loads.empty() && out.stores.empty());
}

} // namespace

bool narrowSharedStagingParam(Session &S, llvm::Function &F,
                              llvm::Function *Proxy) {
  const Options &O = S.Opts;
  if (!Proxy)
    Proxy = &F;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    ParamStagedAccess acc;
    if (!collectParamStaged(S, F, Proxy, G, /*dsMode=*/false, acc)) {
      acc = ParamStagedAccess();
      if (!O.Materialization.NarrowMultiviewStaging || !isSiteClone(S, Proxy) ||
          !collectGlobalStaged(O, F, G, /*dsMode=*/false, acc))
        continue;
    }
    if (O.Diagnostics.Print) {
      llvm::errs() << "[narrow-param fp32] G=@" << G.getName() << " args=";
      for (unsigned i : acc.argIdx)
        llvm::errs() << i << " ";
      llvm::errs() << " geps=" << acc.order.size()
                   << " loads=" << acc.loads.size()
                   << " stores=" << acc.stores.size() << "\n";
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
    llvm::errs() << "[poseidon] narrowed "
                 << (acc.argIdx.empty() ? "multiview" : "param-staged")
                 << " shared buffer @" << G.getName() << " to fp32 ("
                 << acc.loads.size() << " loads, " << acc.stores.size()
                 << " stores)\n";
  }
  return anyChange;
}

bool narrowSharedStagingParamDS(Session &S, llvm::Function &F,
                                llvm::Function *Proxy) {
  const Options &O = S.Opts;
  if (!Proxy)
    Proxy = &F;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  Type *DoubleTy = Type::getDoubleTy(Ctx);
  Type *FloatTy = Type::getFloatTy(Ctx);
  Type *I8Ty = Type::getInt8Ty(Ctx);
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    ParamStagedAccess acc;
    if (!collectParamStaged(S, F, Proxy, G, /*dsMode=*/true, acc)) {
      acc = ParamStagedAccess();
      if (!O.Materialization.NarrowMultiviewStaging || !isSiteClone(S, Proxy) ||
          !collectGlobalStaged(O, F, G, /*dsMode=*/true, acc))
        continue;
    }
    if (O.Diagnostics.Print) {
      llvm::errs() << "[narrow-param df64] G=@" << G.getName() << " args=";
      for (unsigned i : acc.argIdx)
        llvm::errs() << i << " ";
      llvm::errs() << " geps=" << acc.order.size()
                   << " loads=" << acc.loads.size()
                   << " stores=" << acc.stores.size() << "\n";
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
      SmallVector<DekkerSplit, 2> splits;
      if (matchDekkerSplit(L, hiT, loT, hib, los)) {
        hiT->replaceAllUsesWith(nhi);
        loT->replaceAllUsesWith(nlo);
        loT->eraseFromParent();
        los->eraseFromParent();
        hib->eraseFromParent();
        hiT->eraseFromParent();
      } else if (O.Materialization.NarrowMultiviewStaging &&
                 matchDekkerSplits(L, splits)) {
        for (DekkerSplit &d : splits) {
          d.hiT->replaceAllUsesWith(nhi);
          d.loT->replaceAllUsesWith(nlo);
          d.loT->eraseFromParent();
          d.los->eraseFromParent();
          d.hib->eraseFromParent();
          d.hiT->eraseFromParent();
        }
      }
      // A remaining reader (a tolerated fcmp guard) gets the exact
      // reconstruction hi+lo.
      if (!L->use_empty()) {
        Value *hi64 = B.CreateFPExt(nhi, DoubleTy);
        Value *lo64 = B.CreateFPExt(nlo, DoubleTy);
        L->replaceAllUsesWith(B.CreateFAdd(hi64, lo64));
      }
      L->eraseFromParent();
    }
    for (auto *S : acc.stores) {
      IRBuilder<> B(S);
      splitStore(B, S->getValueOperand(), S->getPointerOperand());
      S->eraseFromParent();
    }

    anyChange = true;
    llvm::errs() << "[poseidon] narrowed "
                 << (acc.argIdx.empty() ? "multiview" : "param-staged")
                 << " shared buffer @" << G.getName() << " to df64 ("
                 << acc.loads.size() << " loads, " << acc.stores.size()
                 << " stores)\n";
  }
  return anyChange;
}

bool applyStagingNarrowing(Session &S, Function &F, bool announce,
                           ParamArm param, Function *paramProxy) {
  const Options &O = S.Opts;
  bool changed = narrowSharedStaging(O, F);
  if (changed && announce && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] applied shared-staging narrowing in "
                 << F.getName() << "\n";
  Function *proxy = paramProxy ? paramProxy : &F;
  if (param == ParamArm::Both || param == ParamArm::FP32)
    if (narrowSharedStagingParam(S, F, proxy)) {
      changed = true;
      if (announce && O.Diagnostics.Print)
        llvm::errs() << "[poseidon] applied param-staged shared narrowing in "
                     << F.getName() << "\n";
    }
  if (param == ParamArm::Both || param == ParamArm::DS)
    if (narrowSharedStagingParamDS(S, F, proxy)) {
      changed = true;
      if (announce && O.Diagnostics.Print)
        llvm::errs() << "[poseidon] applied param-staged df64 narrowing in "
                     << F.getName() << "\n";
    }
  if (narrowSharedStagingDS(O, F)) {
    changed = true;
    if (announce && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] applied shared-staging df64 narrowing in "
                   << F.getName() << "\n";
  }
  return changed;
}

} // namespace poseidon
