//===- Staging.cpp - shared-memory staging narrowing ----------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Staging.h"
#include "Expansion.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
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

// The addrspacecasts of the shared buffer G that F can reach: the constant
// expressions and F's own cast instructions.
static SmallVector<Value *, 4> collectSharedCasts(GlobalVariable &G,
                                                  Function &F) {
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
  return casts;
}

// Stores `val` at `p` as the {hi, lo} float pair of the df64 layout: hi at
// +0, lo at +4.
static void emitSplitStore(IRBuilder<> &B, Value *val, Value *p) {
  Value *hi = B.CreateFPTrunc(val, B.getFloatTy());
  Value *lo = B.CreateFPTrunc(
      B.CreateFSub(val, B.CreateFPExt(hi, B.getDoubleTy())), B.getFloatTy());
  B.CreateAlignedStore(hi, p, Align(4));
  B.CreateAlignedStore(lo, B.CreateGEP(B.getInt8Ty(), p, B.getInt64(4)),
                       Align(4));
}

// The staged load L as a load of the float at its narrowed slot, extended back.
static void narrowLoadToF32(LoadInst *L) {
  IRBuilder<> B(L);
  Value *nl = B.CreateAlignedLoad(B.getFloatTy(), L->getPointerOperand(),
                                  Align(4), L->getName());
  Value *back = B.CreateFPExt(nl, B.getDoubleTy());
  L->replaceAllUsesWith(back);
  L->eraseFromParent();
}

// The staging store S as a store of the float its value rounds to.
static void narrowStoreToF32(StoreInst *S) {
  IRBuilder<> B(S);
  Value *vf = B.CreateFPTrunc(S->getValueOperand(), B.getFloatTy());
  B.CreateAlignedStore(vf, S->getPointerOperand(), Align(4));
  S->eraseFromParent();
}

// The staged load L, read by Dekker splits and fcmps, as the load of the pair
// its slot holds: each split reads the pair, an fcmp the exact hi + lo.
static void narrowLoadToPair(LoadInst *L, MutableArrayRef<DekkerSplit> splits) {
  IRBuilder<> B(L);
  Value *p = L->getPointerOperand();
  Value *nhi = B.CreateAlignedLoad(B.getFloatTy(), p, Align(4));
  Value *loPtr = B.CreateGEP(B.getInt8Ty(), p, B.getInt64(4));
  Value *nlo = B.CreateAlignedLoad(B.getFloatTy(), loPtr, Align(4));
  for (DekkerSplit &d : splits)
    d.replaceWith(nhi, nlo);
  if (!L->use_empty()) {
    Value *hi64 = B.CreateFPExt(nhi, B.getDoubleTy());
    Value *lo64 = B.CreateFPExt(nlo, B.getDoubleTy());
    L->replaceAllUsesWith(B.CreateFAdd(hi64, lo64));
  }
  L->eraseFromParent();
}

// The staging store S as the store of its value's {hi, lo} pair.
static void narrowStoreToPair(StoreInst *S) {
  IRBuilder<> B(S);
  emitSplitStore(B, S->getValueOperand(), S->getPointerOperand());
  S->eraseFromParent();
}

// Narrow addrspace(3) staging buffers F reaches through the global. To FP32
// (stride K -> K/2, `load float` + fpext, fptrunc at the stores) when every
// reader of a staged load rounds it to FP32 (fptrunc or fcmp), which keeps the
// transform value-preserving. To df64 (the layout kept, the {hi, lo} pair
// stored once at the staging store, hi at +0 and lo at +4 of the 8-byte slot,
// each Dekker split turned into a pair load) when every staged load is read by
// one Dekker split.
static bool narrowGlobalStaging(const Options &O, Function &F, bool ds,
                                bool announce) {
  Module *M = F.getParent();
  Type *DoubleTy = Type::getDoubleTy(M->getContext());
  Type *I8Ty = Type::getInt8Ty(M->getContext());
  // The FP32 narrowing runs first on the same buffers, so only it traces them.
  const bool trace = O.Diagnostics.Print && !ds;
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;

    SmallVector<Value *, 4> casts = collectSharedCasts(G, F);
    if (casts.empty())
      continue;

    SmallVector<GetElementPtrInst *, 8> elemGEPs;
    SmallVector<GetElementPtrInst *, 32> compGEPs;
    SmallVector<LoadInst *, 64> loads;
    SmallVector<StoreInst *, 16> stores;
    SmallVector<MemCpyInst *, 8> memcpys;
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
            (AT->getNumElements() % (ds ? 8 : 2)) != 0) {
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
    if (trace)
      llvm::errs() << "[narrow] G=@" << G.getName() << " casts=" << casts.size()
                   << " elemGEPs=" << elemGEPs.size() << " strideK=" << strideK
                   << " bail=" << bail << '\n';
    if (bail || elemGEPs.empty())
      continue;

    SmallVector<Value *, 16> work(elemGEPs.begin(), elemGEPs.end());
    SmallPtrSet<Value *, 32> seen(work.begin(), work.end());
    while (!work.empty() && !bail) {
      Value *P = work.pop_back_val();
      for (User *U : P->users()) {
        if (auto *L = dyn_cast<LoadInst>(U)) {
          if (L->getType() != DoubleTy || L->getPointerOperand() != P) {
            if (trace)
              llvm::errs() << "[narrow] bail: load " << *L << '\n';
            bail = true;
            break;
          }
          if (ds) {
            SmallVector<DekkerSplit, 1> splits;
            if (!matchDekkerSplits(L, /*allowFCmp=*/true, splits) ||
                splits.size() != 1) {
              bail = true;
              break;
            }
          } else {
            // Value-preserving only if every reader rounds to FP32; fcmp
            // guards are tolerated.
            for (User *LU : L->users()) {
              if (auto *FT = dyn_cast<FPTruncInst>(LU))
                if (FT->getType()->isFloatTy())
                  continue;
              if (isa<FCmpInst>(LU))
                continue;
              if (trace)
                llvm::errs()
                    << "[narrow] bail: FP64 consumer of staged load: " << *LU
                    << '\n';
              bail = true;
              break;
            }
            if (bail)
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
            if (trace)
              llvm::errs() << "[narrow] bail: comp GEP " << *CG << '\n';
            bail = true;
            break;
          }
          compGEPs.push_back(CG);
          if (seen.insert(CG).second)
            work.push_back(CG);
        } else if (auto *MC = dyn_cast<MemCpyInst>(U)) {
          // A tile load lowered to a struct memcpy: the staging pointer is its
          // destination and one slot its length.
          auto *Len = dyn_cast<ConstantInt>(MC->getLength());
          if (MC->getArgOperand(0) != P || !Len ||
              Len->getZExtValue() != (uint64_t)strideK || (strideK % 8) != 0) {
            if (trace)
              llvm::errs() << "[narrow] bail: memcpy " << *MC << '\n';
            bail = true;
            break;
          }
          memcpys.push_back(MC);
        } else {
          if (trace)
            llvm::errs() << "[narrow] bail: unexpected user of staging ptr: "
                         << *U << '\n';
          bail = true;
          break;
        }
      }
    }
    if (trace)
      llvm::errs() << "[narrow]   loads=" << loads.size()
                   << " stores=" << stores.size()
                   << " compGEPs=" << compGEPs.size() << " bail=" << bail
                   << '\n';
    if (bail || (loads.empty() && stores.empty() && memcpys.empty()))
      continue;

    if (!ds) {
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
    }
    for (auto *L : loads) {
      if (!ds) {
        narrowLoadToF32(L);
        continue;
      }
      SmallVector<DekkerSplit, 1> splits;
      matchDekkerSplits(L, /*allowFCmp=*/true, splits);
      narrowLoadToPair(L, splits);
    }
    for (auto *S : stores) {
      if (ds)
        narrowStoreToPair(S);
      else
        narrowStoreToF32(S);
    }
    // Each slot of the tile is loaded as a double and stored narrowed.
    for (auto *MC : memcpys) {
      IRBuilder<> B(MC);
      Value *dst = MC->getArgOperand(0), *src = MC->getArgOperand(1);
      for (int k = 0; k < strideK / 8; ++k) {
        Value *sp = k ? B.CreateGEP(I8Ty, src, B.getInt64(8 * k)) : src;
        if (ds) {
          Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(8 * k)) : dst;
          Value *d = B.CreateAlignedLoad(DoubleTy, sp, Align(8));
          emitSplitStore(B, d, dp);
          continue;
        }
        Value *d = B.CreateAlignedLoad(DoubleTy, sp, Align(8));
        Value *f = B.CreateFPTrunc(d, B.getFloatTy());
        Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(4 * k)) : dst;
        B.CreateAlignedStore(f, dp, Align(4));
      }
      MC->eraseFromParent();
    }
    if (!ds) {
      for (auto *CG : compGEPs)
        if (CG->use_empty())
          CG->eraseFromParent();
      for (auto *EG : elemGEPs)
        if (EG->use_empty())
          EG->eraseFromParent();
    }
    anyChange = true;
    if (!announce || !O.Diagnostics.Print)
      continue;
    if (ds)
      llvm::errs() << "[poseidon] narrowed shared staging (df64) @"
                   << G.getName() << " (" << loads.size() << " loads, "
                   << stores.size() << " stores, " << memcpys.size()
                   << " memcpys)\n";
    else
      llvm::errs() << "[poseidon] narrowed shared staging buffer @"
                   << G.getName() << " (" << loads.size() << " loads, "
                   << stores.size() << " stores, stride " << strideK << "->"
                   << strideK / 2 << ")\n";
  }
  return anyChange;
}

//===----------------------------------------------------------------------===//
// Narrowing of addrspace(3) staging buffers that reach the rewritten function
// as pointer parameters. The FP32 transform halves every byte offset from the
// base and loads and stores float. The DS transform keeps the layout, a double
// slot holding the {hi, lo} pair, and moves only the accesses.
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

} // namespace

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
  // or an instruction inside F; a GEP in a third function would be missed.
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
// the constant expressions F reaches G through (offset views into the buffer)
// are instructions. Shared memory lives for one launch, so only an access in a
// function F calls could meet F's; one of those, or an access of G that is not
// through a generic pointer, leaves the buffer alone.
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
              llvm::errs() << "[narrow-param] bail: array GEP " << *GEP << '\n';
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
              llvm::errs() << "[narrow-param] bail: byte GEP " << *GEP << '\n';
            return false;
          }
          out.byteSet.insert(GEP);
        } else if (SET->isDoubleTy()) {
          if (GEP->getNumIndices() != 1)
            return false;
          out.dblSet.insert(GEP);
        } else {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: GEP " << *GEP << '\n';
          return false;
        }
        if (seen.insert(GEP).second) {
          out.order.push_back(GEP);
          work.push_back(GEP);
        }
      } else if (auto *L = dyn_cast<LoadInst>(U)) {
        if (L->getType() != DoubleTy || L->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: load " << *L << '\n';
          return false;
        }
        // A load the expansion converted once per use it read the value at
        // has one split per use; only the multiview narrowing takes more
        // than one.
        SmallVector<DekkerSplit, 2> splits;
        const bool split = matchDekkerSplits(L, /*allowFCmp=*/true, splits);
        const bool isSplit = split && splits.size() == 1;
        if (dsMode) {
          // DS wants Dekker splits; anything else is a foreign reader.
          if (!isSplit &&
              !(O.Materialization.NarrowMultiviewStaging && split)) {
            if (O.Diagnostics.Print)
              llvm::errs() << "[narrow-param] bail: non-split reader of "
                              "staged load: "
                           << *L << '\n';
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
                           << *L << '\n';
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
                  << *LU << '\n';
            return false;
          }
        }
        out.loads.push_back(L);
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        if (S->getValueOperand()->getType() != DoubleTy ||
            S->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[narrow-param] bail: store " << *S << '\n';
          return false;
        }
        out.stores.push_back(S);
      } else {
        if (O.Diagnostics.Print)
          llvm::errs() << "[narrow-param] bail: unexpected user " << *U << '\n';
        return false;
      }
    }
  }
  return !(out.loads.empty() && out.stores.empty());
}

// The staged accesses of G that a parameter arm rewrites: through a parameter
// of F, or else, under -poseidon-narrow-multiview-staging, in the site clone
// directly.
static bool collectStaged(Session &S, Function &F, Function *Proxy,
                          GlobalVariable &G, bool dsMode,
                          ParamStagedAccess &acc) {
  const Options &O = S.Opts;
  if (!collectParamStaged(S, F, Proxy, G, dsMode, acc)) {
    acc = ParamStagedAccess();
    if (!O.Materialization.NarrowMultiviewStaging || !isSiteClone(S, Proxy) ||
        !collectGlobalStaged(O, F, G, dsMode, acc))
      return false;
  }
  if (O.Diagnostics.Print) {
    llvm::errs() << "[narrow-param " << (dsMode ? "df64" : "fp32") << "] G=@"
                 << G.getName() << " args=";
    for (unsigned i : acc.argIdx)
      llvm::errs() << i << ' ';
    llvm::errs() << " geps=" << acc.order.size()
                 << " loads=" << acc.loads.size()
                 << " stores=" << acc.stores.size() << '\n';
  }
  return true;
}

// The narrowing of buffers that reach F as a pointer parameter, to FP32 or to
// df64 pairs; see narrowSharedStagingParamDS.
static bool narrowParamStaging(Session &S, Function &F, Function &Proxy,
                               bool ds, bool announce) {
  const Options &O = S.Opts;
  Module *M = F.getParent();
  Type *FloatTy = Type::getFloatTy(M->getContext());
  Type *I8Ty = Type::getInt8Ty(M->getContext());
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    ParamStagedAccess acc;
    if (!collectStaged(S, F, &Proxy, G, ds, acc))
      continue;

    if (ds) {
      // A double slot holds the pair, so the GEPs stay as they are; only the
      // accesses move.
      for (auto *L : acc.loads) {
        SmallVector<DekkerSplit, 2> splits;
        matchDekkerSplits(L, /*allowFCmp=*/true, splits);
        narrowLoadToPair(L, splits);
      }
      for (auto *St : acc.stores)
        narrowStoreToPair(St);
    } else {
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
          ng = B.CreateGEP(FloatTy, GEP->getPointerOperand(), idx,
                           GEP->getName(), GEP->isInBounds());
        }
        GEP->replaceAllUsesWith(ng);
        deadGEPs.push_back(GEP);
      }
      for (auto *L : acc.loads)
        narrowLoadToF32(L);
      for (auto *St : acc.stores)
        narrowStoreToF32(St);
      for (auto *I : llvm::reverse(deadGEPs))
        if (I->use_empty())
          I->eraseFromParent();
    }

    anyChange = true;
    if (announce && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowed "
                   << (acc.argIdx.empty() ? "multiview" : "param-staged")
                   << " shared buffer @" << G.getName() << " to "
                   << (ds ? "df64" : "fp32") << " (" << acc.loads.size()
                   << " loads, " << acc.stores.size() << " stores)\n";
  }
  return anyChange;
}

bool narrowSharedStagingParamDS(Session &S, Function &F, Function &Proxy,
                                bool announce) {
  return narrowParamStaging(S, F, Proxy, /*ds=*/true, announce);
}

bool applyStagingNarrowing(Session &S, Function &F, bool announce,
                           ParamArm param, Function &paramProxy) {
  const Options &O = S.Opts;
  bool changed = narrowGlobalStaging(O, F, /*ds=*/false, announce);
  if (changed && announce && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] applied shared-staging narrowing in "
                 << F.getName() << '\n';
  if (O.Materialization.NarrowParamStaging) {
    if (param != ParamArm::DS &&
        narrowParamStaging(S, F, paramProxy, /*ds=*/false, announce)) {
      changed = true;
      if (announce && O.Diagnostics.Print)
        llvm::errs() << "[poseidon] applied param-staged shared narrowing in "
                     << F.getName() << '\n';
    }
    if (param != ParamArm::FP32 &&
        narrowParamStaging(S, F, paramProxy, /*ds=*/true, announce)) {
      changed = true;
      if (announce && O.Diagnostics.Print)
        llvm::errs() << "[poseidon] applied param-staged df64 narrowing in "
                     << F.getName() << '\n';
    }
  }
  if (narrowGlobalStaging(O, F, /*ds=*/true, announce)) {
    changed = true;
    if (announce && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] applied shared-staging df64 narrowing in "
                   << F.getName() << '\n';
  }
  return changed;
}

} // namespace poseidon
