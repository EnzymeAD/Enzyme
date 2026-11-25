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
#include "Sites/Markers.h"
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

#include <optional>

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

namespace {

// A site whose body Proxy is solved in F, Proxy itself or a pricing clone of
// it, and the functions that can run in the same launch: those that make the
// site call, those that call them, and everything any of them or F calls. The
// body the site was cut from and the site a clone copies never run with F.
class SiteLaunch {
public:
  SiteLaunch(const Session &S, const Function &F, const Function &Proxy);

  // Narrowing a shared buffer changes the layout every access to it in the
  // launch reads, and the narrowings rewrite F's accesses alone. So G may be
  // narrowed only if F reaches it through a generic pointer and no other
  // function of the launch reads or writes it, except by handing it to the
  // site.
  bool onlySiteAccesses(const GlobalVariable &G) const;

private:
  // Whether I hands what it is given to the site: a call of a site function,
  // or the marker call naming one.
  bool callsSite(const Instruction &I) const;

  const Function &F;
  SmallPtrSet<const Function *, 4> site;
  SmallPtrSet<const Function *, 16> mates;
};

} // namespace

SiteLaunch::SiteLaunch(const Session &S, const Function &F,
                       const Function &Proxy)
    : F(F) {
  site.insert(&F);
  site.insert(&Proxy);
  if (const Function *source = siteOriginOf(S, Proxy))
    site.insert(source);

  SmallVector<const Function *, 16> callers;
  auto addCallersOf = [&](const Function &Callee) {
    SmallVector<const User *, 8> users(Callee.users());
    SmallPtrSet<const User *, 16> seen;
    while (!users.empty()) {
      const User *U = users.pop_back_val();
      if (!seen.insert(U).second)
        continue;
      if (auto *I = dyn_cast<Instruction>(U)) {
        const Function *Fn = I->getFunction();
        if (!site.contains(Fn) && mates.insert(Fn).second)
          callers.push_back(Fn);
      } else if (isa<Constant>(U)) {
        users.append(U->user_begin(), U->user_end());
      }
    }
  };
  for (const Function *Fn : site)
    addCallersOf(*Fn);
  for (size_t i = 0; i < callers.size(); ++i)
    addCallersOf(*callers[i]);

  SmallVector<const Function *, 16> todo(callers.begin(), callers.end());
  todo.push_back(&F);
  while (!todo.empty()) {
    const Function *Fn = todo.pop_back_val();
    for (const Instruction &I : instructions(*Fn))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (const Function *Callee = CB->getCalledFunction();
            Callee && !Callee->isDeclaration() && !site.contains(Callee) &&
            mates.insert(Callee).second)
          todo.push_back(Callee);
  }
}

bool SiteLaunch::callsSite(const Instruction &I) const {
  auto *CB = dyn_cast<CallBase>(&I);
  const Function *Callee = CB ? CB->getCalledFunction() : nullptr;
  if (!Callee)
    return false;
  if (site.contains(Callee))
    return true;
  return isMarkerCall(Callee->getName()) &&
         any_of(CB->args(), [&](const Use &A) {
           return site.contains(dyn_cast<Function>(A->stripPointerCasts()));
         });
}

bool SiteLaunch::onlySiteAccesses(const GlobalVariable &G) const {
  // Each use, with whether it reaches G through an addrspacecast.
  SmallVector<std::pair<const User *, bool>, 16> uses;
  for (const User *U : G.users())
    uses.push_back({U, false});
  SmallPtrSet<const User *, 16> seen;
  while (!uses.empty()) {
    auto [U, generic] = uses.pop_back_val();
    if (!seen.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      const Function *Fn = I->getFunction();
      if (Fn == &F) {
        if (!generic && !isa<AddrSpaceCastInst>(I))
          return false;
      } else if (mates.contains(Fn) && !callsSite(*I)) {
        return false;
      }
      continue;
    }
    auto *CE = dyn_cast<ConstantExpr>(U);
    const bool viaCast =
        generic || (CE && CE->getOpcode() == Instruction::AddrSpaceCast);
    for (const User *CU : U->users())
      uses.push_back({CU, viaCast});
  }
  return true;
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
static bool narrowGlobalStaging(const Session &S, const SiteLaunch &launch,
                                Function &F, bool ds, bool announce) {
  const Options &O = S.Opts;
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
    if (!launch.onlySiteAccesses(G)) {
      if (trace)
        llvm::errs() << "[narrow] bail: @" << G.getName()
                     << " is read or written beside the site in its launch\n";
      continue;
    }

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
// as pointer parameters or through views of any shape. The FP32 transform
// halves every byte offset from the base and loads and stores float. The DS
// transform keeps the layout, a double slot holding the {hi, lo} pair, and
// moves only the accesses.
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

// The parameters of F the site binds to G, in `argIdx`.
static void stagedParams(const Session &S, Function &F, Function *Proxy,
                         GlobalVariable &G, SmallVectorImpl<unsigned> &argIdx) {
  SmallVector<Value *, 4> casts;
  for (User *U : G.users())
    if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::AddrSpaceCast)
        casts.push_back(CE);
    } else if (isa<AddrSpaceCastInst>(U)) {
      casts.push_back(U);
    }
  if (casts.empty())
    return;

  for (unsigned i = 0; i < F.arg_size(); ++i) {
    if (!F.getArg(i)->getType()->isPointerTy())
      continue;
    if (Value *SA = siteArg(S, Proxy, i))
      if (llvm::is_contained(casts, SA))
        argIdx.push_back(i);
  }
  if (argIdx.empty())
    for (Value *C : casts)
      for (User *CU : C->users()) {
        auto *CB = dyn_cast<CallBase>(CU);
        if (!CB || CB->getCalledFunction() != Proxy)
          continue;
        for (unsigned i = 0; i < CB->arg_size() && i < F.arg_size(); ++i)
          if (CB->getArgOperand(i) == C &&
              F.getArg(i)->getType()->isPointerTy() &&
              !llvm::is_contained(argIdx, i))
            argIdx.push_back(i);
      }
}

// The instructions of F that use G, through constant expressions or not.
static SmallVector<Instruction *, 8> usersIn(Function &F, GlobalVariable &G) {
  SmallVector<Instruction *, 8> found;
  SmallVector<User *, 8> users(G.users());
  SmallPtrSet<User *, 16> seen;
  while (!users.empty()) {
    User *U = users.pop_back_val();
    if (!seen.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (I->getFunction() == &F)
        found.push_back(I);
    } else if (isa<Constant>(U)) {
      users.append(U->user_begin(), U->user_end());
    }
  }
  return found;
}

// F's own casts of G, once the constant expressions F reaches G through
// (offset views into the buffer) are instructions; none when F does not reach
// G itself.
static void stagedCasts(Function &F, GlobalVariable &G,
                        SmallVectorImpl<Value *> &roots) {
  if (usersIn(F, G).empty())
    return;
  convertUsersOfConstantsToInstructions({&G}, &F,
                                        /*RemoveDeadConstants=*/false);
  for (Instruction *I : usersIn(F, G))
    roots.push_back(I);
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
        // has one split per use.
        SmallVector<DekkerSplit, 2> splits;
        const bool split = matchDekkerSplits(L, /*allowFCmp=*/true, splits);
        const bool isSplit = split && splits.size() == 1;
        if (dsMode) {
          // DS wants Dekker splits; anything else is a foreign reader.
          if (!split) {
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

// The staged accesses of G that a parameter arm rewrites: through the
// parameters of F the site binds to G and through F's own casts of G, in
// views of any shape. False if any access is outside the modelled shapes (the
// buffer is then left alone).
static bool collectStaged(Session &S, const SiteLaunch &launch, Function &F,
                          Function *Proxy, GlobalVariable &G, bool dsMode,
                          ParamStagedAccess &acc) {
  const Options &O = S.Opts;
  // The site's caller values are known for a registered site alone.
  if (!isSiteClone(S, Proxy))
    return false;
  if (!launch.onlySiteAccesses(G)) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[narrow-param] bail: @" << G.getName()
                   << " is read or written beside the site in its launch\n";
    return false;
  }
  stagedParams(S, F, Proxy, G, acc.argIdx);
  SmallVector<Value *, 8> roots;
  for (unsigned i : acc.argIdx)
    roots.push_back(F.getArg(i));
  stagedCasts(F, G, roots);
  if (roots.empty() || !walkStaged(O, F, roots, dsMode, acc))
    return false;
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
  std::optional<SiteLaunch> launch;
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    if (!launch)
      launch.emplace(S, F, Proxy);
    ParamStagedAccess acc;
    if (!collectStaged(S, *launch, F, &Proxy, G, ds, acc))
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
  const SiteLaunch launch(S, F, paramProxy);
  bool changed = narrowGlobalStaging(S, launch, F, /*ds=*/false, announce);
  if (changed && announce && O.Diagnostics.Print)
    llvm::errs() << "[poseidon] applied shared-staging narrowing in "
                 << F.getName() << '\n';
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
  if (narrowGlobalStaging(S, launch, F, /*ds=*/true, announce)) {
    changed = true;
    if (announce && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] applied shared-staging df64 narrowing in "
                   << F.getName() << '\n';
  }
  return changed;
}

} // namespace poseidon
