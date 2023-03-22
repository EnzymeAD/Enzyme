//===- Utils.cpp - Definition of miscellaneous utilities ------------------===//
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
// This file defines miscellaneous utilities that are used as part of the
// AD process.
//
//===----------------------------------------------------------------------===//
#include "Utils.h"
#include "TypeAnalysis/TypeAnalysis.h"

#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include "TypeAnalysis/TBAA.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "llvm-c/Core.h"

#include "LibraryFuncs.h"

using namespace llvm;

EnzymeFailure::EnzymeFailure(Twine RemarkName, const DiagnosticLocation &Loc,
                             const Instruction *CodeRegion)
    : DiagnosticInfoUnsupported(*CodeRegion->getParent()->getParent(),
                                RemarkName, Loc) {}

/// Create function to computer nearest power of two
Value *nextPowerOfTwo(IRBuilder<> &B, Value *V) {
  assert(V->getType()->isIntegerTy());
  IntegerType *T = cast<IntegerType>(V->getType());
  V = B.CreateAdd(V, ConstantInt::get(T, -1));
  for (size_t i = 1; i < T->getBitWidth(); i *= 2) {
    V = B.CreateOr(V, B.CreateLShr(V, ConstantInt::get(T, i)));
  }
  V = B.CreateAdd(V, ConstantInt::get(T, 1));
  return V;
}

Value *getOrInsertOpFloatSum(Module &M, Type *OpPtr, ConcreteType CT,
                             Type *intType, IRBuilder<> &B2) {
  std::string name = "__enzyme_mpi_sum" + CT.str();
  assert(CT.isFloat());
  auto FlT = CT.isFloat();

  if (auto Glob = M.getGlobalVariable(name)) {
#if LLVM_VERSION_MAJOR > 7
    return B2.CreateLoad(Glob->getValueType(), Glob);
#else
    return B2.CreateLoad(Glob);
#endif
  }

  Type *types[] = {PointerType::getUnqual(FlT), PointerType::getUnqual(FlT),
                   PointerType::getUnqual(intType), OpPtr};
  FunctionType *FuT =
      FunctionType::get(Type::getVoidTy(M.getContext()), types, false);

#if LLVM_VERSION_MAJOR >= 9
  Function *F =
      cast<Function>(M.getOrInsertFunction(name + "_run", FuT).getCallee());
#else
  Function *F = cast<Function>(M.getOrInsertFunction(name + "_run", FuT));
#endif

  F->setLinkage(Function::LinkageTypes::InternalLinkage);
  F->addFnAttr(Attribute::ArgMemOnly);
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::AlwaysInline);
  F->addParamAttr(0, Attribute::NoCapture);
  F->addParamAttr(0, Attribute::ReadOnly);
  F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::NoCapture);
  F->addParamAttr(2, Attribute::ReadOnly);
  F->addParamAttr(3, Attribute::NoCapture);
  F->addParamAttr(3, Attribute::ReadNone);

  BasicBlock *entry = BasicBlock::Create(M.getContext(), "entry", F);
  BasicBlock *body = BasicBlock::Create(M.getContext(), "for.body", F);
  BasicBlock *end = BasicBlock::Create(M.getContext(), "for.end", F);

  auto src = F->arg_begin();
  src->setName("src");
  auto dst = src + 1;
  dst->setName("dst");
  auto lenp = dst + 1;
  lenp->setName("lenp");
  Value *len;
  // TODO consider using datatype arg and asserting same size as assumed
  // by type analysis

  {
    IRBuilder<> B(entry);
#if LLVM_VERSION_MAJOR > 7
    len = B.CreateLoad(lenp->getType()->getPointerElementType(), lenp);
#else
    len = B.CreateLoad(lenp);
#endif
    B.CreateCondBr(B.CreateICmpEQ(len, ConstantInt::get(len->getType(), 0)),
                   end, body);
  }

  {
    IRBuilder<> B(body);
    B.setFastMathFlags(getFast());
    PHINode *idx = B.CreatePHI(len->getType(), 2, "idx");
    idx->addIncoming(ConstantInt::get(len->getType(), 0), entry);

#if LLVM_VERSION_MAJOR > 7
    Value *dsti = B.CreateInBoundsGEP(dst->getType()->getPointerElementType(),
                                      dst, idx, "dst.i");
    LoadInst *dstl =
        B.CreateLoad(dsti->getType()->getPointerElementType(), dsti, "dst.i.l");

    Value *srci = B.CreateInBoundsGEP(src->getType()->getPointerElementType(),
                                      src, idx, "src.i");
    LoadInst *srcl =
        B.CreateLoad(srci->getType()->getPointerElementType(), srci, "src.i.l");
#else
    Value *dsti = B.CreateInBoundsGEP(dst, idx, "dst.i");
    LoadInst *dstl = B.CreateLoad(dsti, "dst.i.l");

    Value *srci = B.CreateInBoundsGEP(src, idx, "src.i");
    LoadInst *srcl = B.CreateLoad(srci, "src.i.l");
#endif

    B.CreateStore(B.CreateFAdd(srcl, dstl), dsti);

    Value *next =
        B.CreateNUWAdd(idx, ConstantInt::get(len->getType(), 1), "idx.next");
    idx->addIncoming(next, body);
    B.CreateCondBr(B.CreateICmpEQ(len, next), end, body);
  }

  {
    IRBuilder<> B(end);
    B.CreateRetVoid();
  }

  Type *rtypes[] = {Type::getInt8PtrTy(M.getContext()), intType, OpPtr};
  FunctionType *RFT = FunctionType::get(intType, rtypes, false);

  Constant *RF = M.getNamedValue("MPI_Op_create");
  if (!RF) {
#if LLVM_VERSION_MAJOR >= 9
    RF =
        cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT).getCallee());
#else
    RF = cast<Function>(M.getOrInsertFunction("MPI_Op_create", RFT));
#endif
  } else {
    RF = ConstantExpr::getBitCast(RF, PointerType::getUnqual(RFT));
  }

  GlobalVariable *GV = new GlobalVariable(
      M, OpPtr->getPointerElementType(), false, GlobalVariable::InternalLinkage,
      UndefValue::get(OpPtr->getPointerElementType()), name);

  Type *i1Ty = Type::getInt1Ty(M.getContext());
  GlobalVariable *initD = new GlobalVariable(
      M, i1Ty, false, GlobalVariable::InternalLinkage,
      ConstantInt::getFalse(M.getContext()), name + "_initd");

  // Finish initializing mpi sum
  // https://www.mpich.org/static/docs/v3.2/www3/MPI_Op_create.html
  FunctionType *IFT = FunctionType::get(Type::getVoidTy(M.getContext()),
                                        ArrayRef<Type *>(), false);

#if LLVM_VERSION_MAJOR >= 9
  Function *initializerFunction = cast<Function>(
      M.getOrInsertFunction(name + "initializer", IFT).getCallee());
#else
  Function *initializerFunction =
      cast<Function>(M.getOrInsertFunction(name + "initializer", IFT));
#endif

  initializerFunction->setLinkage(Function::LinkageTypes::InternalLinkage);
  initializerFunction->addFnAttr(Attribute::NoUnwind);

  {
    BasicBlock *entry =
        BasicBlock::Create(M.getContext(), "entry", initializerFunction);
    BasicBlock *run =
        BasicBlock::Create(M.getContext(), "run", initializerFunction);
    BasicBlock *end =
        BasicBlock::Create(M.getContext(), "end", initializerFunction);
    IRBuilder<> B(entry);
#if LLVM_VERSION_MAJOR > 7
    B.CreateCondBr(
        B.CreateLoad(initD->getType()->getPointerElementType(), initD), end,
        run);
#else
    B.CreateCondBr(B.CreateLoad(initD), end, run);
#endif

    B.SetInsertPoint(run);
    Value *args[] = {ConstantExpr::getPointerCast(F, rtypes[0]),
                     ConstantInt::get(rtypes[1], 1, false),
                     ConstantExpr::getPointerCast(GV, rtypes[2])};
    B.CreateCall(RFT, RF, args);
    B.CreateStore(ConstantInt::getTrue(M.getContext()), initD);
    B.CreateBr(end);
    B.SetInsertPoint(end);
    B.CreateRetVoid();
  }

  B2.CreateCall(M.getFunction(name + "initializer"));
#if LLVM_VERSION_MAJOR > 7
  return B2.CreateLoad(GV->getValueType(), GV);
#else
  return B2.CreateLoad(GV);
#endif
}

void mayExecuteAfter(SmallVectorImpl<Instruction *> &results, Instruction *inst,
                     const SmallPtrSetImpl<Instruction *> &stores,
                     const Loop *region) {
  using namespace llvm;
  std::map<BasicBlock *, SmallVector<Instruction *, 1>> maybeBlocks;
  BasicBlock *instBlk = inst->getParent();
  for (auto store : stores) {
    BasicBlock *storeBlk = store->getParent();
    if (instBlk == storeBlk) {
      // if store doesn't come before, exit.

      if (store != inst) {
        BasicBlock::const_iterator It = storeBlk->begin();
        for (; &*It != store && &*It != inst; ++It)
          /*empty*/;
        // if inst comes first (e.g. before store) in the
        // block, return true
        if (&*It == inst) {
          results.push_back(store);
        }
      }
      maybeBlocks[storeBlk].push_back(store);
    } else {
      maybeBlocks[storeBlk].push_back(store);
    }
  }

  if (maybeBlocks.size() == 0)
    return;

  SmallVector<BasicBlock *, 2> todo;
  for (auto B : successors(instBlk)) {
    if (region && region->getHeader() == B) {
      continue;
    }
    todo.push_back(B);
  }

  SmallPtrSet<BasicBlock *, 2> seen;
  while (todo.size()) {
    auto cur = todo.back();
    todo.pop_back();
    if (seen.count(cur))
      continue;
    seen.insert(cur);
    auto found = maybeBlocks.find(cur);
    if (found != maybeBlocks.end()) {
      for (auto store : found->second)
        results.push_back(store);
      maybeBlocks.erase(found);
    }
    for (auto B : successors(cur)) {
      if (region && region->getHeader() == B) {
        continue;
      }
      todo.push_back(B);
    }
  }
}

bool overwritesToMemoryReadByLoop(ScalarEvolution &SE, LoopInfo &LI,
                                  DominatorTree &DT, Instruction *maybeReader,
                                  const SCEV *LoadStart, const SCEV *LoadEnd,
                                  Instruction *maybeWriter,
                                  const SCEV *StoreStart, const SCEV *StoreEnd,
                                  Loop *scope) {
  // The store may either occur directly after the load in the current loop
  // nest, or prior to the load in a subsequent iteration of the loop nest
  // Generally:
  // L0 -> scope -> L1 -> L2 -> L3 -> load_L4 -> load_L5 ...  Load
  //                               \-> store_L4 -> store_L5 ... Store
  // We begin by finding the common ancestor of the two loops, which may
  // be none.
  Loop *anc = getAncestor(LI.getLoopFor(maybeReader->getParent()),
                          LI.getLoopFor(maybeWriter->getParent()));

  // The surrounding scope must contain the ancestor
  if (scope) {
    assert(anc);
    assert(scope == anc || scope->contains(anc));
  }

  // Consider the case where the load and store don't share any common loops.
  // That is to say, there's no loops in [scope, ancestor) we need to consider
  // having a store in a  later iteration overwrite the load of a previous
  // iteration.
  //
  // An example of this overwriting would be a "left shift"
  //   for (int j = 1; j<N; j++) {
  //      load A[j]
  //      store A[j-1]
  //    }
  //
  // Ignoring such ancestors, if we compare the two regions to have no direct
  // overlap we can return that it doesn't overwrite memory if the regions
  // don't overlap at any level of region expansion. That is to say, we can
  // expand the start or end, for any loop to be the worst case scenario
  // given the loop bounds.
  //
  // However, now let us consider the case where there are surrounding loops.
  // If the storing boundary is represented by an induction variable of one
  // of these common loops, we must conseratively expand it all the way to the
  // end. We will also mark the loops we may expand. If we encounter all
  // intervening loops in this fashion, and it is proven safe in these cases,
  // the region does not overlap. However, if we don't encounter all surrounding
  // loops in our induction expansion, we may simply be repeating the write
  // which we should also ensure we say the region may overlap (due to the
  // repetition).
  //
  // Since we also have a Loop scope, we can ignore any common loops at the
  // scope level or above

  /// We force all ranges for all loops in range ... [scope, anc], .... cur
  /// to expand the number of iterations

  SmallPtrSet<const Loop *, 1> visitedAncestors;
  auto skipLoop = [&](const Loop *L) {
    assert(L);
    if (scope && L->contains(scope))
      return false;

    if (anc && (anc == L || anc->contains(L))) {
      visitedAncestors.insert(L);
      return true;
    }
    return false;
  };

  // Check the boounds  of an [... endprev][startnext ...] for potential
  // overlaps. The boolean EndIsStore is true of the EndPev represents
  // the store and should have its loops expanded, or if that should
  // apply to StartNed.
  auto hasOverlap = [&](const SCEV *EndPrev, const SCEV *StartNext,
                        bool EndIsStore) {
    for (auto slim = StartNext; slim != SE.getCouldNotCompute();) {
      bool sskip = false;
      if (!EndIsStore)
        if (auto startL = dyn_cast<SCEVAddRecExpr>(slim))
          if (skipLoop(startL->getLoop()) &&
              SE.isKnownNonPositive(startL->getStepRecurrence(SE))) {
            sskip = true;
          }

      if (!sskip)
        for (auto elim = EndPrev; elim != SE.getCouldNotCompute();) {
          {

            bool eskip = false;
            if (EndIsStore)
              if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
                if (skipLoop(endL->getLoop()) &&
                    SE.isKnownNonNegative(endL->getStepRecurrence(SE))) {
                  eskip = true;
                }
              }

            // Moreover because otherwise SE cannot "groupScevByComplexity"
            // we need to ensure that if both slim/elim are AddRecv
            // they must be in the same loop, or one loop must dominate
            // the other.
            if (!eskip) {

              if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
                auto EH = endL->getLoop()->getHeader();
                if (auto startL = dyn_cast<SCEVAddRecExpr>(slim)) {
                  auto SH = startL->getLoop()->getHeader();
                  if (EH != SH && !DT.dominates(EH, SH) &&
                      !DT.dominates(SH, EH))
                    eskip = true;
                }
              }
            }
            if (!eskip) {
              auto sub = SE.getMinusSCEV(slim, elim);
              if (sub != SE.getCouldNotCompute() && SE.isKnownNonNegative(sub))
                return false;
            }
          }

          if (auto endL = dyn_cast<SCEVAddRecExpr>(elim)) {
            if (SE.isKnownNonPositive(endL->getStepRecurrence(SE))) {
              elim = endL->getStart();
              continue;
            } else if (SE.isKnownNonNegative(endL->getStepRecurrence(SE))) {
#if LLVM_VERSION_MAJOR >= 12
              auto ebd = SE.getSymbolicMaxBackedgeTakenCount(endL->getLoop());
#else
              auto ebd = SE.getBackedgeTakenCount(endL->getLoop());
#endif
              if (ebd == SE.getCouldNotCompute())
                break;
              elim = endL->evaluateAtIteration(ebd, SE);
              continue;
            }
          }
          break;
        }

      if (auto startL = dyn_cast<SCEVAddRecExpr>(slim)) {
        if (SE.isKnownNonNegative(startL->getStepRecurrence(SE))) {
          slim = startL->getStart();
          continue;
        } else if (SE.isKnownNonPositive(startL->getStepRecurrence(SE))) {
#if LLVM_VERSION_MAJOR >= 12
          auto sbd = SE.getSymbolicMaxBackedgeTakenCount(startL->getLoop());
#else
          auto sbd = SE.getBackedgeTakenCount(startL->getLoop());
#endif
          if (sbd == SE.getCouldNotCompute())
            break;
          slim = startL->evaluateAtIteration(sbd, SE);
          continue;
        }
      }
      break;
    }
    return true;
  };

  // There is no overwrite if either the stores all occur before the loads
  // [S, S+Size][start load, L+Size]
  visitedAncestors.clear();
  if (!hasOverlap(StoreEnd, LoadStart, /*EndIsStore*/ true)) {
    // We must have seen all common loops as induction variables
    // to be legal, lest we have a repetition of the store.
    bool legal = true;
    for (const Loop *L = anc; anc != scope; anc = anc->getParentLoop()) {
      if (!visitedAncestors.count(L))
        legal = false;
    }
    if (legal)
      return false;
  }

  // There is no overwrite if either the loads all occur before the stores
  // [start load, L+Size] [S, S+Size]
  visitedAncestors.clear();
  if (!hasOverlap(LoadEnd, StoreStart, /*EndIsStore*/ false)) {
    // We must have seen all common loops as induction variables
    // to be legal, lest we have a repetition of the store.
    bool legal = true;
    for (const Loop *L = anc; anc != scope; anc = anc->getParentLoop()) {
      if (!visitedAncestors.count(L))
        legal = false;
    }
    if (legal)
      return false;
  }
  return true;
}

bool overwritesToMemoryReadBy(AAResults &AA, TargetLibraryInfo &TLI,
                              ScalarEvolution &SE, LoopInfo &LI,
                              DominatorTree &DT, Instruction *maybeReader,
                              Instruction *maybeWriter, Loop *scope) {
  using namespace llvm;
  if (!writesToMemoryReadBy(AA, TLI, maybeReader, maybeWriter))
    return false;
  const SCEV *LoadBegin = SE.getCouldNotCompute();
  const SCEV *LoadEnd = SE.getCouldNotCompute();

  const SCEV *StoreBegin = SE.getCouldNotCompute();
  const SCEV *StoreEnd = SE.getCouldNotCompute();

  if (auto LI = dyn_cast<LoadInst>(maybeReader)) {
    LoadBegin = SE.getSCEV(LI->getPointerOperand());
    if (LoadBegin != SE.getCouldNotCompute()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
#if LLVM_VERSION_MAJOR >= 10
      auto TS = SE.getConstant(
          APInt(64, DL.getTypeStoreSize(LI->getType()).getFixedSize()));
#else
      auto TS = SE.getConstant(APInt(64, DL.getTypeStoreSize(LI->getType())));
#endif
      LoadEnd = SE.getAddExpr(LoadBegin, TS);
    }
  }
  if (auto SI = dyn_cast<StoreInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(SI->getPointerOperand());
    if (StoreBegin != SE.getCouldNotCompute()) {
      auto &DL = maybeWriter->getModule()->getDataLayout();
#if LLVM_VERSION_MAJOR >= 10
      auto TS = SE.getConstant(
          APInt(64, DL.getTypeStoreSize(SI->getValueOperand()->getType())
                        .getFixedSize()));
#else
      auto TS = SE.getConstant(
          APInt(64, DL.getTypeStoreSize(SI->getValueOperand()->getType())));
#endif
      StoreEnd = SE.getAddExpr(StoreBegin, TS);
    }
  }
  if (auto MS = dyn_cast<MemSetInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto TS = SE.getConstant(APInt(64, Len->getValue().getLimitedValue()));
        StoreEnd = SE.getAddExpr(StoreBegin, TS);
      }
    }
  }
  if (auto MS = dyn_cast<MemTransferInst>(maybeWriter)) {
    StoreBegin = SE.getSCEV(MS->getArgOperand(0));
    if (StoreBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto TS = SE.getConstant(APInt(64, Len->getValue().getLimitedValue()));
        StoreEnd = SE.getAddExpr(StoreBegin, TS);
      }
    }
  }
  if (auto MS = dyn_cast<MemTransferInst>(maybeReader)) {
    LoadBegin = SE.getSCEV(MS->getArgOperand(1));
    if (LoadBegin != SE.getCouldNotCompute()) {
      if (auto Len = dyn_cast<ConstantInt>(MS->getArgOperand(2))) {
        auto TS = SE.getConstant(APInt(64, Len->getValue().getLimitedValue()));
        LoadEnd = SE.getAddExpr(LoadBegin, TS);
      }
    }
  }

  if (!overwritesToMemoryReadByLoop(SE, LI, DT, maybeReader, LoadBegin, LoadEnd,
                                    maybeWriter, StoreBegin, StoreEnd, scope))
    return false;

  return true;
}

/// Return whether maybeReader can read from memory written to by maybeWriter
bool writesToMemoryReadBy(AAResults &AA, TargetLibraryInfo &TLI,
                          Instruction *maybeReader, Instruction *maybeWriter) {
  assert(maybeReader->getParent()->getParent() ==
         maybeWriter->getParent()->getParent());
  using namespace llvm;
  if (auto call = dyn_cast<CallInst>(maybeWriter)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isCertainPrint(funcName) || isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
    if (funcName == "jl_array_copy" || funcName == "ijl_array_copy")
      return false;

    // Isend only writes to inaccessible mem only
    if (funcName == "MPI_Send" || funcName == "PMPI_Send") {
      return false;
    }
    // Wait only overwrites memory in the status and request.
    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait" ||
        funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
#if LLVM_VERSION_MAJOR > 11
      auto loc = LocationSize::afterPointer();
#else
      auto loc = MemoryLocation::UnknownSize;
#endif
      size_t off = (funcName == "MPI_Wait" || funcName == "PMPI_Wait") ? 0 : 1;
      // No alias with status
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(off + 1),
                                     loc))) {
        // No alias with request
        if (!isRefSet(AA.getModRefInfo(maybeReader,
                                       call->getArgOperand(off + 0), loc)))
          return false;
        auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                             ->getParent()
                                             ->getParent()
                                             ->getDataLayout())[{-1}];
        // Could still conflict with the mpi_request unless a non pointer
        // type.
        if (R != BaseType::Unknown && R != BaseType::Anything &&
            R != BaseType::Pointer)
          return false;
      }
    }
    // Isend only writes to inaccessible mem and request.
    if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
      auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                           ->getParent()
                                           ->getParent()
                                           ->getDataLayout())[{-1}];
      // Could still conflict with the mpi_request, unless either
      // synchronous, or a non pointer type.
      if (R != BaseType::Unknown && R != BaseType::Anything &&
          R != BaseType::Pointer)
        return false;
#if LLVM_VERSION_MAJOR > 11
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                     LocationSize::afterPointer())))
        return false;
#else
      if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                     MemoryLocation::UnknownSize)))
        return false;
#endif
      return false;
    }
    if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv" ||
        funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
      ConcreteType type(BaseType::Unknown);
      if (Constant *C = dyn_cast<Constant>(call->getArgOperand(2))) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_double") {
            type = ConcreteType(Type::getDoubleTy(C->getContext()));
          } else if (GV->getName() == "ompi_mpi_float") {
            type = ConcreteType(Type::getFloatTy(C->getContext()));
          }
        }
      }
      if (type.isKnown()) {
        auto R = parseTBAA(*maybeReader, maybeReader->getParent()
                                             ->getParent()
                                             ->getParent()
                                             ->getDataLayout())[{-1}];
        if (R.isKnown() && type != R) {
          // Could still conflict with the mpi_request, unless either
          // synchronous, or a non pointer type.
          if (funcName == "MPI_Recv" || funcName == "PMPI_Recv" ||
              (R != BaseType::Anything && R != BaseType::Pointer))
            return false;
#if LLVM_VERSION_MAJOR > 11
          if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                         LocationSize::afterPointer())))
            return false;
#else
          if (!isRefSet(AA.getModRefInfo(maybeReader, call->getArgOperand(6),
                                         MemoryLocation::UnknownSize)))
            return false;
#endif
        }
      }
    }
    if (auto II = dyn_cast<IntrinsicInst>(call)) {
      if (II->getIntrinsicID() == Intrinsic::stacksave)
        return false;
      if (II->getIntrinsicID() == Intrinsic::stackrestore)
        return false;
      if (II->getIntrinsicID() == Intrinsic::trap)
        return false;
#if LLVM_VERSION_MAJOR >= 13
      if (II->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl)
        return false;
#endif
    }

#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand()))
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue()))
#endif
    {
      if (StringRef(iasm->getAsmString()).contains("exit"))
        return false;
    }
  }
  if (auto call = dyn_cast<CallInst>(maybeReader)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }

    if (auto II = dyn_cast<IntrinsicInst>(call)) {
      if (II->getIntrinsicID() == Intrinsic::stacksave)
        return false;
      if (II->getIntrinsicID() == Intrinsic::stackrestore)
        return false;
      if (II->getIntrinsicID() == Intrinsic::trap)
        return false;
#if LLVM_VERSION_MAJOR >= 13
      if (II->getIntrinsicID() == Intrinsic::experimental_noalias_scope_decl)
        return false;
#endif
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeWriter)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
    if (funcName == "jl_array_copy" || funcName == "ijl_array_copy")
      return false;

#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand()))
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue()))
#endif
    {
      if (StringRef(iasm->getAsmString()).contains("exit"))
        return false;
    }
  }
  if (auto call = dyn_cast<InvokeInst>(maybeReader)) {
    StringRef funcName = getFuncNameFromCall(call);

    if (isDebugFunction(call->getCalledFunction()))
      return false;

    if (isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return false;
    }

    if (isMemFreeLibMFunction(funcName)) {
      return false;
    }
  }
  assert(maybeWriter->mayWriteToMemory());
  assert(maybeReader->mayReadFromMemory());

  if (auto li = dyn_cast<LoadInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(li)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeReader)) {
    return isModSet(AA.getModRefInfo(maybeWriter, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemTransferInst>(maybeReader)) {
    return isModSet(
        AA.getModRefInfo(maybeWriter, MemoryLocation::getForSource(mti)));
  }

  if (auto si = dyn_cast<StoreInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(si)));
  }
  if (auto rmw = dyn_cast<AtomicRMWInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(rmw)));
  }
  if (auto xch = dyn_cast<AtomicCmpXchgInst>(maybeWriter)) {
    return isRefSet(AA.getModRefInfo(maybeReader, MemoryLocation::get(xch)));
  }
  if (auto mti = dyn_cast<MemIntrinsic>(maybeWriter)) {
    return isRefSet(
        AA.getModRefInfo(maybeReader, MemoryLocation::getForDest(mti)));
  }

  if (auto cb = dyn_cast<CallInst>(maybeReader)) {
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  if (auto cb = dyn_cast<InvokeInst>(maybeReader)) {
    return isModOrRefSet(AA.getModRefInfo(maybeWriter, cb));
  }
  errs() << " maybeReader: " << *maybeReader << " maybeWriter: " << *maybeWriter
         << "\n";
  llvm_unreachable("unknown inst2");
}

Function *GetFunctionFromValue(Value *fn) {
  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  if (!isa<Function>(fn)) {
    return nullptr;
  }

  return cast<Function>(fn);
}
