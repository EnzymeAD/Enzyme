//===- Narrowing.cpp - shared staging after a precision change ------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Narrowing.h"
#include "Core/FPNode.h"
#include "Core/Session.h"
#include "Expansion.h"
#include "Sites/Markers.h"
#include "Sites/SiteIdentity.h"
#include "Support/Options.h"
#include "Targets/Target.h"

#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
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

//===----------------------------------------------------------------------===//
// Which buffers may be narrowed
//===----------------------------------------------------------------------===//

// The addrspacecasts of the shared buffer G: its constant expressions and its
// cast instructions, those of F alone when F is given.
static SmallVector<Value *> sharedCasts(GlobalVariable &G,
                                        const Function *F = nullptr) {
  SmallVector<Value *> casts;
  for (User *U : G.users()) {
    if (auto *CE = dyn_cast<ConstantExpr>(U)) {
      if (CE->getOpcode() == Instruction::AddrSpaceCast)
        casts.push_back(CE);
    } else if (auto *AC = dyn_cast<AddrSpaceCastInst>(U)) {
      if (!F || AC->getFunction() == F)
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

// Calls `visit(I, viaCast)` on each instruction that uses `V`, directly or
// through constants, depth first, with whether its path passes an
// addrspacecast expression; stops at the first `visit` that returns false and
// returns whether none did.
template <typename ValueT, typename Visit>
static bool allInstructionUsers(ValueT &V, Visit visit) {
  using UserPtr = decltype(*V.user_begin());
  SmallVector<std::pair<UserPtr, bool>> users;
  for (UserPtr U : V.users())
    users.push_back({U, false});
  SmallPtrSet<UserPtr, 16> seen;
  while (!users.empty()) {
    auto [U, viaCast] = users.pop_back_val();
    if (!seen.insert(U).second)
      continue;
    if (auto *I = dyn_cast<Instruction>(U)) {
      if (!visit(*I, viaCast))
        return false;
      continue;
    }
    auto *CE = dyn_cast<ConstantExpr>(U);
    const bool cast =
        viaCast || (CE && CE->getOpcode() == Instruction::AddrSpaceCast);
    for (UserPtr CU : U->users())
      users.push_back({CU, cast});
  }
  return true;
}

SiteLaunch::SiteLaunch(const Session &S, const Function &F,
                       const Function &Proxy)
    : F(F) {
  site.insert(&F);
  site.insert(&Proxy);
  if (const Function *source = siteOriginOf(S.get<SiteRecords>(), Proxy))
    site.insert(source);

  SmallVector<const Function *> callers;
  auto addCallersOf = [&](const Function &Callee) {
    allInstructionUsers(Callee, [&](const Instruction &I, bool) {
      const Function *Caller = I.getFunction();
      if (!site.contains(Caller) && mates.insert(Caller).second)
        callers.push_back(Caller);
      return true;
    });
  };
  for (const Function *SiteFunction : site)
    addCallersOf(*SiteFunction);
  for (size_t i = 0; i < callers.size(); ++i)
    addCallersOf(*callers[i]);

  SmallVector<const Function *> todo(callers.begin(), callers.end());
  todo.push_back(&F);
  while (!todo.empty()) {
    const Function *Caller = todo.pop_back_val();
    for (const Instruction &I : instructions(*Caller))
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
  return allInstructionUsers(G, [&](const Instruction &I, bool generic) {
    const Function *Accessor = I.getFunction();
    if (Accessor == &F)
      return generic || isa<AddrSpaceCastInst>(I);
    return !mates.contains(Accessor) || callsSite(I);
  });
}

// The first reader of the staged load L that neither rounds it to FP32 nor
// compares it: narrowing L to FP32 preserves what the others read. Null when
// there is none.
static User *firstWiderReader(LoadInst &L) {
  for (User *U : L.users()) {
    if (auto *FT = dyn_cast<FPTruncInst>(U))
      if (FT->getType()->isFloatTy())
        continue;
    if (isa<FCmpInst>(U))
      continue;
    return U;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Rewriting one access
//===----------------------------------------------------------------------===//

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
static void narrowLoadToPair(LoadInst *L) {
  SmallVector<DekkerSplit> splits;
  matchDekkerSplits(L, /*allowFCmp=*/true, splits);
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

// GEP into a buffer of doubles as the same GEP into the buffer of floats: an
// [N x i8] element of N/2 bytes, an i8 offset halved, a double index over
// floats. The GEP is left for its users' rewrite to empty.
static void halveOffset(GetElementPtrInst *GEP) {
  IRBuilder<> B(GEP);
  Type *I8Ty = B.getInt8Ty();
  SmallVector<Value *, 2> idx(GEP->idx_begin(), GEP->idx_end());
  Type *Source = GEP->getSourceElementType();
  Value *halved = nullptr;
  if (auto *AT = dyn_cast<ArrayType>(Source)) {
    halved = B.CreateGEP(ArrayType::get(I8Ty, AT->getNumElements() / 2),
                         GEP->getPointerOperand(), idx, GEP->getName(),
                         GEP->isInBounds());
  } else if (Source->isIntegerTy(8)) {
    Value *off = GEP->getOperand(1);
    Value *half =
        isa<ConstantInt>(off)
            ? (Value *)B.getInt64(cast<ConstantInt>(off)->getSExtValue() / 2)
            : B.CreateLShr(off, ConstantInt::get(off->getType(), 1), "",
                           /*isExact=*/true);
    halved = B.CreateGEP(I8Ty, GEP->getPointerOperand(), half, GEP->getName(),
                         GEP->isInBounds());
  } else {
    assert(Source->isDoubleTy() && "a staged view is of bytes or doubles");
    halved = B.CreateGEP(B.getFloatTy(), GEP->getPointerOperand(), idx,
                         GEP->getName(), GEP->isInBounds());
  }
  GEP->replaceAllUsesWith(halved);
}

//===----------------------------------------------------------------------===//
// Buffers F reaches through the global, at one stride
//===----------------------------------------------------------------------===//

namespace {

// The accesses of a shared buffer through F's casts of the global: element
// GEPs of one stride, constant byte offsets into an element, and the loads,
// stores and tile copies through them.
struct GlobalStagedAccess {
  SmallVector<GetElementPtrInst *> elemGEPs;
  SmallVector<GetElementPtrInst *> compGEPs;
  SmallVector<LoadInst *> loads;
  SmallVector<StoreInst *> stores;
  SmallVector<MemCpyInst *> memcpys;
  // The bytes of an element.
  int strideK = 0;
};

} // namespace

// The element GEPs of `casts` in F, all of one stride whose element holds a
// whole number of slots of the layout. False if F also uses a cast otherwise.
static bool collectElementGEPs(Function &F, ArrayRef<Value *> casts, bool ds,
                               GlobalStagedAccess &acc) {
  bool fits = true;
  for (Value *C : casts) {
    for (User *U : C->users()) {
      auto *GEP = dyn_cast<GetElementPtrInst>(U);
      if (!GEP || GEP->getFunction() != &F) {
        if (auto *I = dyn_cast<Instruction>(U))
          if (I->getFunction() == &F)
            fits = false;
        continue;
      }
      auto *AT = dyn_cast<ArrayType>(GEP->getSourceElementType());
      const unsigned slotBytes = ds ? 8 : 2;
      if (!AT || !AT->getElementType()->isIntegerTy(8) ||
          (AT->getNumElements() % slotBytes) != 0) {
        fits = false;
        continue;
      }
      if (acc.strideK == 0)
        acc.strideK = (int)AT->getNumElements();
      else if (acc.strideK != (int)AT->getNumElements())
        fits = false;
      acc.elemGEPs.push_back(GEP);
    }
  }
  return fits;
}

// Records U, a user of the staged pointer P, in `acc`; false if the arm cannot
// rewrite it. `worklist` receives the byte offsets to walk next.
static bool addGlobalStagedUser(User *U, Value *P, bool ds, bool trace,
                                GlobalStagedAccess &acc,
                                SmallPtrSetImpl<Value *> &seen,
                                SmallVectorImpl<Value *> &worklist) {
  if (auto *L = dyn_cast<LoadInst>(U)) {
    if (!L->getType()->isDoubleTy() || L->getPointerOperand() != P) {
      if (trace)
        llvm::errs() << "[poseidon] narrowing: bail: load " << *L << '\n';
      return false;
    }
    if (ds) {
      SmallVector<DekkerSplit, 1> splits;
      if (!matchDekkerSplits(L, /*allowFCmp=*/true, splits) ||
          splits.size() != 1)
        return false;
    } else if (User *wider = firstWiderReader(*L)) {
      if (trace)
        llvm::errs()
            << "[poseidon] narrowing: bail: FP64 consumer of staged load: "
            << *wider << '\n';
      return false;
    }
    acc.loads.push_back(L);
    return true;
  }
  if (auto *S = dyn_cast<StoreInst>(U)) {
    if (!S->getValueOperand()->getType()->isDoubleTy() ||
        S->getPointerOperand() != P)
      return false;
    acc.stores.push_back(S);
    return true;
  }
  if (auto *CG = dyn_cast<GetElementPtrInst>(U)) {
    if (!CG->getSourceElementType()->isIntegerTy(8) ||
        CG->getNumIndices() != 1 || !isa<ConstantInt>(CG->getOperand(1))) {
      if (trace)
        llvm::errs() << "[poseidon] narrowing: bail: comp GEP " << *CG << '\n';
      return false;
    }
    acc.compGEPs.push_back(CG);
    if (seen.insert(CG).second)
      worklist.push_back(CG);
    return true;
  }
  if (auto *MC = dyn_cast<MemCpyInst>(U)) {
    // A tile load lowered to a struct memcpy: the staging pointer is its
    // destination and one slot its length.
    auto *Len = dyn_cast<ConstantInt>(MC->getLength());
    if (MC->getArgOperand(0) != P || !Len ||
        Len->getZExtValue() != (uint64_t)acc.strideK ||
        (acc.strideK % 8) != 0) {
      if (trace)
        llvm::errs() << "[poseidon] narrowing: bail: memcpy " << *MC << '\n';
      return false;
    }
    acc.memcpys.push_back(MC);
    return true;
  }
  if (trace)
    llvm::errs()
        << "[poseidon] narrowing: bail: unexpected user of staging ptr: " << *U
        << '\n';
  return false;
}

// The accesses of G a global arm rewrites in F, if every one fits it.
static std::optional<GlobalStagedAccess>
collectGlobalStaged(GlobalVariable &G, Function &F, bool ds, bool trace) {
  SmallVector<Value *> casts = sharedCasts(G, &F);
  if (casts.empty())
    return std::nullopt;

  GlobalStagedAccess acc;
  bool fits = collectElementGEPs(F, casts, ds, acc);
  if (trace)
    llvm::errs() << "[poseidon] narrowing: G=@" << G.getName()
                 << " casts=" << casts.size()
                 << " elemGEPs=" << acc.elemGEPs.size()
                 << " strideK=" << acc.strideK << " bail=" << !fits << '\n';
  if (!fits || acc.elemGEPs.empty())
    return std::nullopt;

  SmallVector<Value *> worklist(acc.elemGEPs.begin(), acc.elemGEPs.end());
  SmallPtrSet<Value *, 32> seen(worklist.begin(), worklist.end());
  while (!worklist.empty() && fits) {
    Value *P = worklist.pop_back_val();
    for (User *U : P->users())
      if (!addGlobalStagedUser(U, P, ds, trace, acc, seen, worklist)) {
        fits = false;
        break;
      }
  }
  if (trace)
    llvm::errs() << "[poseidon] narrowing:   loads=" << acc.loads.size()
                 << " stores=" << acc.stores.size()
                 << " compGEPs=" << acc.compGEPs.size() << " bail=" << !fits
                 << '\n';
  if (!fits || (acc.loads.empty() && acc.stores.empty() && acc.memcpys.empty()))
    return std::nullopt;
  return acc;
}

// The tile copy MC as a copy of each slot, loaded as a double and stored
// narrowed.
static void narrowTileCopy(MemCpyInst *MC, int strideK, bool ds) {
  IRBuilder<> B(MC);
  Type *I8Ty = B.getInt8Ty();
  Value *dst = MC->getArgOperand(0), *src = MC->getArgOperand(1);
  for (int k = 0; k < strideK / 8; ++k) {
    Value *sp = k ? B.CreateGEP(I8Ty, src, B.getInt64(8 * k)) : src;
    if (ds) {
      Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(8 * k)) : dst;
      Value *d = B.CreateAlignedLoad(B.getDoubleTy(), sp, Align(8));
      emitSplitStore(B, d, dp);
      continue;
    }
    Value *d = B.CreateAlignedLoad(B.getDoubleTy(), sp, Align(8));
    Value *f = B.CreateFPTrunc(d, B.getFloatTy());
    Value *dp = k ? B.CreateGEP(I8Ty, dst, B.getInt64(4 * k)) : dst;
    B.CreateAlignedStore(f, dp, Align(4));
  }
  MC->eraseFromParent();
}

// Rewrites `acc` to FP32 (element K -> K/2 bytes, `load float` + fpext,
// fptrunc at the stores) or to df64 (the layout kept, each slot holding the
// {hi, lo} pair, each Dekker split turned into a pair load).
static void rewriteGlobalStaged(GlobalStagedAccess &acc, bool ds) {
  if (!ds) {
    for (GetElementPtrInst *EG : acc.elemGEPs)
      halveOffset(EG);
    for (GetElementPtrInst *CG : acc.compGEPs)
      halveOffset(CG);
  }
  for (LoadInst *L : acc.loads) {
    if (ds)
      narrowLoadToPair(L);
    else
      narrowLoadToF32(L);
  }
  for (StoreInst *S : acc.stores) {
    if (ds)
      narrowStoreToPair(S);
    else
      narrowStoreToF32(S);
  }
  for (MemCpyInst *MC : acc.memcpys)
    narrowTileCopy(MC, acc.strideK, ds);
  if (!ds) {
    for (GetElementPtrInst *CG : acc.compGEPs)
      if (CG->use_empty())
        CG->eraseFromParent();
    for (GetElementPtrInst *EG : acc.elemGEPs)
      if (EG->use_empty())
        EG->eraseFromParent();
  }
}

// Narrows the shared buffers F reaches through the global at one stride: to
// FP32 when every reader of a staged load rounds it to FP32 (fptrunc or fcmp),
// to df64 when every staged load is read by one Dekker split.
static bool narrowGlobalStaging(const Session &S, const SiteLaunch &launch,
                                Function &F, bool ds, bool announce) {
  const Options &O = S.Opts;
  Module *M = F.getParent();
  // The FP32 narrowing runs first on the same buffers, so only it traces them.
  const bool trace = O.Diagnostics.Print && !ds;
  bool anyChange = false;

  const Target &target = Target::of(*M);
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    std::optional<GlobalStagedAccess> acc =
        collectGlobalStaged(G, F, ds, trace);
    if (!acc)
      continue;
    if (!launch.onlySiteAccesses(G)) {
      if (trace)
        llvm::errs() << "[poseidon] narrowing: bail: @" << G.getName()
                     << " is read or written beside the site in its launch\n";
      continue;
    }
    rewriteGlobalStaged(*acc, ds);
    anyChange = true;
    if (!announce || !O.Diagnostics.Print)
      continue;
    if (ds)
      llvm::errs() << "[poseidon] narrowed shared staging (df64) @"
                   << G.getName() << " (" << acc->loads.size() << " loads, "
                   << acc->stores.size() << " stores, " << acc->memcpys.size()
                   << " memcpys)\n";
    else
      llvm::errs() << "[poseidon] narrowed shared staging buffer @"
                   << G.getName() << " (" << acc->loads.size() << " loads, "
                   << acc->stores.size() << " stores, stride " << acc->strideK
                   << "->" << acc->strideK / 2 << ")\n";
  }
  return anyChange;
}

//===----------------------------------------------------------------------===//
// Buffers F reaches as pointer parameters or through views of any shape
//
// The FP32 transform halves every byte offset from the base and loads and
// stores float. The df64 transform keeps the layout, a double slot holding the
// {hi, lo} pair, and moves only the accesses.
//===----------------------------------------------------------------------===//

namespace {

struct StagedAccess {
  // The views of the buffer, parents first.
  SmallVector<GetElementPtrInst *> views;
  SmallVector<LoadInst *> loads;
  SmallVector<StoreInst *> stores;
  // The parameters of F the site binds to the buffer.
  SmallVector<unsigned, 2> argIdx;
};

} // namespace

// The parameters of F the site binds to G.
static SmallVector<unsigned, 2> stagedParams(const Session &S, Function &F,
                                             Function *Proxy,
                                             GlobalVariable &G) {
  SmallVector<unsigned, 2> argIdx;
  SmallVector<Value *> casts = sharedCasts(G);
  if (casts.empty())
    return argIdx;

  for (unsigned i = 0; i < F.arg_size(); ++i) {
    if (!F.getArg(i)->getType()->isPointerTy())
      continue;
    if (Value *SA = siteArg(S.get<SiteRecords>(), Proxy, i))
      if (is_contained(casts, SA))
        argIdx.push_back(i);
  }
  if (!argIdx.empty())
    return argIdx;
  for (Value *C : casts)
    for (User *CU : C->users()) {
      auto *CB = dyn_cast<CallBase>(CU);
      if (!CB || CB->getCalledFunction() != Proxy)
        continue;
      for (unsigned i = 0; i < CB->arg_size() && i < F.arg_size(); ++i)
        if (CB->getArgOperand(i) == C &&
            F.getArg(i)->getType()->isPointerTy() && !is_contained(argIdx, i))
          argIdx.push_back(i);
    }
  return argIdx;
}

// The instructions of F that use G, through constant expressions or not.
static SmallVector<Instruction *> usersIn(Function &F, GlobalVariable &G) {
  SmallVector<Instruction *> found;
  allInstructionUsers(G, [&](Instruction &I, bool) {
    if (I.getFunction() == &F)
      found.push_back(&I);
    return true;
  });
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
  append_range(roots, usersIn(F, G));
}

// Whether GEP, a view of the staged pointer, has an offset the FP32 transform
// halves exactly: [N x i8] elements of an even N, a whole number of doubles of
// bytes, or doubles.
static bool viewFits(const Options &O, const DataLayout &DL,
                     GetElementPtrInst *GEP) {
  Type *Source = GEP->getSourceElementType();
  if (auto *AT = dyn_cast<ArrayType>(Source)) {
    if (AT->getElementType()->isIntegerTy(8) &&
        (AT->getNumElements() % 2) == 0 && GEP->getNumIndices() == 1)
      return true;
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: array GEP "
                   << *GEP << '\n';
    return false;
  }
  if (Source->isIntegerTy(8)) {
    // A constant multiple of 8, or a value whose low three bits are provably
    // zero (clang emits `gep i8, p, idx<<3`).
    bool whole = GEP->getNumIndices() == 1;
    if (whole) {
      if (auto *C = dyn_cast<ConstantInt>(GEP->getOperand(1)))
        whole = (C->getSExtValue() % 8) == 0;
      else
        whole =
            computeKnownBits(GEP->getOperand(1), DL).countMinTrailingZeros() >=
            3;
    }
    if (!whole && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: byte GEP " << *GEP
                   << '\n';
    return whole;
  }
  if (Source->isDoubleTy())
    return GEP->getNumIndices() == 1;
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] narrowing parameters: bail: GEP " << *GEP
                 << '\n';
  return false;
}

// Whether the readers of the staged load L are what the arm keeps exact:
// Dekker splits for df64; for FP32, roundings to FP32 and comparisons, and no
// Dekker split, since narrowing its slot to one float makes `lo` identically
// zero and prices the expansion as free.
static bool loadFits(const Options &O, LoadInst *L, bool dsMode) {
  // A load the expansion converted once per use it read the value at has one
  // split per use.
  SmallVector<DekkerSplit, 2> splits;
  const bool split = matchDekkerSplits(L, /*allowFCmp=*/true, splits);
  if (dsMode) {
    if (!split && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: non-split reader "
                      "of staged load: "
                   << *L << '\n';
    return split;
  }
  if (split && splits.size() == 1) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: Dekker-split "
                      "reader (DS arm's "
                      "buffer): "
                   << *L << '\n';
    return false;
  }
  if (User *wider = firstWiderReader(*L)) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: FP64 consumer of "
                      "staged load: "
                   << *wider << '\n';
    return false;
  }
  return true;
}

// Collects in `out` the accesses through `roots` in F; false if one is outside
// the modelled shapes or there is none.
static bool walkStaged(const Options &O, Function &F, ArrayRef<Value *> roots,
                       bool dsMode, StagedAccess &out) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  SmallPtrSet<Value *, 32> seen(roots.begin(), roots.end());
  SmallVector<Value *> worklist(roots.begin(), roots.end());
  while (!worklist.empty()) {
    Value *P = worklist.pop_back_val();
    for (User *U : P->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || I->getFunction() != &F)
        return false;
      if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        if (GEP->getPointerOperand() != P || !viewFits(O, DL, GEP))
          return false;
        if (seen.insert(GEP).second) {
          out.views.push_back(GEP);
          worklist.push_back(GEP);
        }
      } else if (auto *L = dyn_cast<LoadInst>(U)) {
        if (!L->getType()->isDoubleTy() || L->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[poseidon] narrowing parameters: bail: load " << *L
                         << '\n';
          return false;
        }
        if (!loadFits(O, L, dsMode))
          return false;
        out.loads.push_back(L);
      } else if (auto *S = dyn_cast<StoreInst>(U)) {
        if (!S->getValueOperand()->getType()->isDoubleTy() ||
            S->getPointerOperand() != P) {
          if (O.Diagnostics.Print)
            llvm::errs() << "[poseidon] narrowing parameters: bail: store "
                         << *S << '\n';
          return false;
        }
        out.stores.push_back(S);
      } else {
        if (O.Diagnostics.Print)
          llvm::errs()
              << "[poseidon] narrowing parameters: bail: unexpected user " << *U
              << '\n';
        return false;
      }
    }
  }
  return !(out.loads.empty() && out.stores.empty());
}

// The staged accesses of G that a parameter arm rewrites: through the
// parameters of F the site binds to G and through F's own casts of G, in
// views of any shape. None if any access is outside the modelled shapes (the
// buffer is then left alone).
static std::optional<StagedAccess>
collectStaged(Session &S, const SiteLaunch &launch, Function &F,
              Function *Proxy, GlobalVariable &G, bool dsMode) {
  const Options &O = S.Opts;
  // The site's caller values are known for a registered site alone.
  if (!isSiteClone(S.get<SiteRecords>(), Proxy))
    return std::nullopt;
  if (!launch.onlySiteAccesses(G)) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowing parameters: bail: @" << G.getName()
                   << " is read or written beside the site in its launch\n";
    return std::nullopt;
  }
  StagedAccess acc;
  acc.argIdx = stagedParams(S, F, Proxy, G);
  SmallVector<Value *> roots;
  for (unsigned i : acc.argIdx)
    roots.push_back(F.getArg(i));
  stagedCasts(F, G, roots);
  if (roots.empty() || !walkStaged(O, F, roots, dsMode, acc))
    return std::nullopt;
  if (O.Diagnostics.Print) {
    llvm::errs() << "[poseidon] narrowing parameters to "
                 << (dsMode ? "df64" : "fp32") << ": G=@" << G.getName()
                 << " args=";
    for (unsigned i : acc.argIdx)
      llvm::errs() << i << ' ';
    llvm::errs() << " geps=" << acc.views.size()
                 << " loads=" << acc.loads.size()
                 << " stores=" << acc.stores.size() << '\n';
  }
  return acc;
}

// Rewrites `acc` to FP32 or to df64 pairs.
static void rewriteStaged(StagedAccess &acc, bool ds) {
  if (ds) {
    // A double slot holds the pair, so the views stay as they are; only the
    // accesses move.
    for (LoadInst *L : acc.loads)
      narrowLoadToPair(L);
    for (StoreInst *S : acc.stores)
      narrowStoreToPair(S);
    return;
  }
  for (GetElementPtrInst *GEP : acc.views)
    halveOffset(GEP);
  for (LoadInst *L : acc.loads)
    narrowLoadToF32(L);
  for (StoreInst *S : acc.stores)
    narrowStoreToF32(S);
  for (GetElementPtrInst *GEP : reverse(acc.views))
    if (GEP->use_empty())
      GEP->eraseFromParent();
}

// Narrows the shared buffers that reach F as a pointer parameter or through
// views of any shape, to FP32 or to df64 pairs.
static bool narrowParamStaging(Session &S, Function &F, Function &Proxy,
                               bool ds, bool announce) {
  const Options &O = S.Opts;
  Module *M = F.getParent();
  bool anyChange = false;

  const Target &target = Target::of(*M);
  std::optional<SiteLaunch> launch;
  for (GlobalVariable &G : M->globals()) {
    if (!target.isShared(G))
      continue;
    if (!launch)
      launch.emplace(S, F, Proxy);
    std::optional<StagedAccess> acc =
        collectStaged(S, *launch, F, &Proxy, G, ds);
    if (!acc)
      continue;
    rewriteStaged(*acc, ds);
    anyChange = true;
    if (announce && O.Diagnostics.Print)
      llvm::errs() << "[poseidon] narrowed "
                   << (acc->argIdx.empty() ? "multiview" : "param-staged")
                   << " shared buffer @" << G.getName() << " to "
                   << (ds ? "df64" : "fp32") << " (" << acc->loads.size()
                   << " loads, " << acc->stores.size() << " stores)\n";
  }
  return anyChange;
}

//===----------------------------------------------------------------------===//
// Entry points
//===----------------------------------------------------------------------===//

bool narrowToPairs(Session &S, Function &F, Function &Proxy, bool announce) {
  return narrowParamStaging(S, F, Proxy, /*ds=*/true, announce);
}

bool narrowSharedStaging(Session &S, Function &F, bool announce,
                         ParamNarrowing param, Function &paramProxy) {
  const Options &O = S.Opts;
  const bool say = announce && O.Diagnostics.Print;
  const SiteLaunch launch(S, F, paramProxy);
  bool changed = narrowGlobalStaging(S, launch, F, /*ds=*/false, announce);
  if (changed && say)
    llvm::errs() << "[poseidon] applied shared-staging narrowing in "
                 << F.getName() << '\n';
  if (param != ParamNarrowing::Pairs &&
      narrowParamStaging(S, F, paramProxy, /*ds=*/false, announce)) {
    changed = true;
    if (say)
      llvm::errs() << "[poseidon] applied param-staged shared narrowing in "
                   << F.getName() << '\n';
  }
  if (param != ParamNarrowing::FP32 &&
      narrowParamStaging(S, F, paramProxy, /*ds=*/true, announce)) {
    changed = true;
    if (say)
      llvm::errs() << "[poseidon] applied param-staged df64 narrowing in "
                   << F.getName() << '\n';
  }
  if (narrowGlobalStaging(S, launch, F, /*ds=*/true, announce)) {
    changed = true;
    if (say)
      llvm::errs() << "[poseidon] applied shared-staging df64 narrowing in "
                   << F.getName() << '\n';
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// Subgraphs that share a staging buffer
//===----------------------------------------------------------------------===//

// The memory object a value is loaded from / stored to when that object is a
// staging buffer (pointer argument or addrspace(3) global), else nullptr.
static const Value *stagingObjectOf(const Value *Ptr) {
  const Value *Obj = getUnderlyingObject(Ptr);
  if (!Obj)
    return nullptr;
  if (isa<Argument>(Obj) && Obj->getType()->isPointerTy())
    return Obj;
  if (auto *GV = dyn_cast<GlobalVariable>(Obj))
    if (Target::of(*GV->getParent()).isShared(*GV))
      return Obj;
  return nullptr;
}

// The objects `objectOf` gives for the pointers a subgraph reads through the
// loads feeding it and writes through the stores of its outputs.
static SmallPtrSet<const Value *, 4>
touchedObjects(const Subgraph &subgraph,
               function_ref<const Value *(const Value *)> objectOf) {
  SmallPtrSet<const Value *, 4> touched;
  auto add = [&](const Value *Ptr) {
    if (const Value *O = objectOf(Ptr))
      touched.insert(O);
  };
  for (Value *In : subgraph.inputs)
    if (auto *LI = dyn_cast<LoadInst>(In))
      add(LI->getPointerOperand());
  for (Instruction *Op : subgraph.operations)
    for (Value *V : Op->operands())
      if (auto *LI = dyn_cast<LoadInst>(V))
        add(LI->getPointerOperand());
  for (Instruction *Out : subgraph.outputs)
    for (User *U : Out->users())
      if (auto *SI = dyn_cast<StoreInst>(U))
        if (SI->getValueOperand() == Out)
          add(SI->getPointerOperand());
  return touched;
}

void mergeSharedStagingSubgraphs(const Options &O, Function &F,
                                 SmallVectorImpl<Subgraph> &subgraphs) {
  if (!O.Candidates.SubgraphMergeStaging || subgraphs.size() < 2)
    return;

  SmallVector<SmallPtrSet<const Value *, 4>, 8> touched;
  for (const Subgraph &subgraph : subgraphs)
    touched.push_back(touchedObjects(subgraph, stagingObjectOf));

  IntEqClasses classes(subgraphs.size());
  for (size_t i = 0; i < subgraphs.size(); ++i)
    for (size_t j = i + 1; j < subgraphs.size(); ++j)
      if (llvm::any_of(touched[i],
                       [&](const Value *O) { return touched[j].count(O); }))
        classes.join(i, j);
  classes.compress();
  if (classes.getNumClasses() == subgraphs.size())
    return; // nothing shares a buffer
  SmallVector<SmallVector<unsigned, 8>, 8> groups(classes.getNumClasses());
  for (unsigned i = 0; i < subgraphs.size(); ++i)
    groups[classes[i]].push_back(i);

  SmallVector<Subgraph, 8> merged;
  for (ArrayRef<unsigned> group : groups) {
    if (group.size() == 1) {
      merged.push_back(subgraphs[group.front()]);
      continue;
    }
    Subgraph M;
    for (unsigned idx : group) {
      for (Instruction *Op : subgraphs[idx].operations)
        M.operations.insert(Op);
      for (Instruction *Out : subgraphs[idx].outputs)
        M.outputs.insert(Out);
    }
    // An input of one member may be produced by another member; those edges
    // are internal to the merged unit and must not be re-declared as inputs.
    for (unsigned idx : group)
      for (Value *In : subgraphs[idx].inputs) {
        auto *I = dyn_cast<Instruction>(In);
        if (I && (M.operations.contains(I) || M.outputs.contains(I)))
          continue;
        M.inputs.insert(In);
      }
    if (O.Diagnostics.Print) {
      SmallPtrSet<const Value *, 4> objs;
      for (unsigned idx : group)
        for (const Value *O : touched[idx])
          objs.insert(O);
      llvm::errs() << "[poseidon] merged " << group.size()
                   << " FP subgraphs sharing " << objs.size()
                   << " staging buffer(s) into one optimization unit ("
                   << M.operations.size() << " operations, " << M.inputs.size()
                   << " inputs, " << M.outputs.size() << " outputs) in "
                   << F.getName() << '\n';
    }
    merged.push_back(std::move(M));
  }

  subgraphs = std::move(merged);
}

} // namespace poseidon
