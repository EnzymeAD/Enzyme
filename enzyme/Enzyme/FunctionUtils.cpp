//===- FunctionUtils.cpp - Implementation of function utilities -----------===//
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
// This file defines utilities on LLVM Functions that are used as part of the AD
// process.
//
//===----------------------------------------------------------------------===//
#include "FunctionUtils.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#include "llvm/Analysis/TypeBasedAliasAnalysis.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Analysis/PhiValues.h"
#endif
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Utils.h"
#endif

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#endif

#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "CacheUtility.h"

#define DEBUG_TYPE "enzyme"
using namespace llvm;

static cl::opt<bool>
    EnzymePreopt("enzyme-preopt", cl::init(true), cl::Hidden,
                 cl::desc("Run enzyme preprocessing optimizations"));

static cl::opt<bool> EnzymeInline("enzyme-inline", cl::init(false), cl::Hidden,
                                  cl::desc("Force inlining of autodiff"));

static cl::opt<bool> EnzymeLowerGlobals("enzyme-lower-globals", cl::init(false), cl::Hidden,
                                  cl::desc("Lower globals to locals assuming the global values are not needed outside of this gradient"));

static cl::opt<int>
    EnzymeInlineCount("enzyme-inline-count", cl::init(10000), cl::Hidden,
                      cl::desc("Limit of number of functions to inline"));

// Locally run mem2reg on F, if ASsumptionCache AC is given it will
// be updated
static bool PromoteMemoryToRegister(Function &F, DominatorTree &DT,
                                    AssumptionCache *AC = nullptr) {
  std::vector<AllocaInst *> Allocas;
  BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
  bool Changed = false;

  while (true) {
    Allocas.clear();

    // Find allocas that are safe to promote, by looking at all instructions in
    // the entry node
    for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
        if (isAllocaPromotable(AI))
          Allocas.push_back(AI);

    if (Allocas.empty())
      break;

    PromoteMemToReg(Allocas, DT, AC);
    Changed = true;
  }
  return Changed;
}

/// Return whether this function eventually calls itself
static bool IsFunctionRecursive(Function *F) {
  enum RecurType {
    MaybeRecursive = 1,
    NotRecursive = 2,
    DefinitelyRecursive = 3,
  };

  static std::map<const Function *, RecurType> Results;

  // If we haven't seen this function before, look at all callers
  // and mark this as potentially recursive. If we see this function
  // still as marked as MaybeRecursive, we will definitionally have
  // found an eventual caller of the original function. If not,
  // the function does not eventually call itself (in a static way)
  if (Results.find(F) == Results.end()) {
    Results[F] = MaybeRecursive; // staging
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto call = dyn_cast<CallInst>(&I)) {
          if (call->getCalledFunction() == nullptr)
            continue;
          if (call->getCalledFunction()->empty())
            continue;
          IsFunctionRecursive(call->getCalledFunction());
        }
        if (auto call = dyn_cast<InvokeInst>(&I)) {
          if (call->getCalledFunction() == nullptr)
            continue;
          if (call->getCalledFunction()->empty())
            continue;
          IsFunctionRecursive(call->getCalledFunction());
        }
      }
    }
    if (Results[F] == MaybeRecursive) {
      Results[F] = NotRecursive; // not recursive
    }
  } else if (Results[F] == MaybeRecursive) {
    Results[F] = DefinitelyRecursive; // definitely recursive
  }
  assert(Results[F] != MaybeRecursive);
  return Results[F] == DefinitelyRecursive;
}

static inline bool OnlyUsedInOMP(AllocaInst *AI) {
  bool ompUse = false;
  for (auto U : AI->users()) {
    if (isa<StoreInst>(U))
      continue;
    if (auto CI = dyn_cast<CallInst>(U)) {
      if (auto F = CI->getCalledFunction()) {
        if (F->getName() == "__kmpc_for_static_init_4" ||
            F->getName() == "__kmpc_for_static_init_4u" ||
            F->getName() == "__kmpc_for_static_init_8" ||
            F->getName() == "__kmpc_for_static_init_8u") {
          ompUse = true;
        }
      }
    }
  }

  if (!ompUse)
    return false;
  return true;
}

/// Convert necessary stack allocations into mallocs for use in the reverse
/// pass. Specifically if we're not topLevel all allocations must be upgraded
/// Even if topLevel any allocations that aren't in the entry block (and
/// therefore may not be reachable in the reverse pass) must be upgraded.
static inline void UpgradeAllocasToMallocs(Function *NewF, bool topLevel) {
  std::vector<AllocaInst *> ToConvert;

  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto AI = dyn_cast<AllocaInst>(&I)) {
        bool UsableEverywhere = AI->getParent() == &NewF->getEntryBlock();
        // TODO use is_value_needed_in_reverse (requiring GradientUtils)
        if (OnlyUsedInOMP(AI))
          continue;
        if (!UsableEverywhere || !topLevel) {
          ToConvert.push_back(AI);
        }
      }
    }
  }

  for (auto AI : ToConvert) {
    std::string nam = AI->getName().str();
    AI->setName("");

    // Ensure we insert the malloc after the allocas
    Instruction *insertBefore = AI;
    while (isa<AllocaInst>(insertBefore->getNextNode())) {
      insertBefore = insertBefore->getNextNode();
      assert(insertBefore);
    }

    auto i64 = Type::getInt64Ty(NewF->getContext());
    auto rep = CallInst::CreateMalloc(
        insertBefore, i64, AI->getAllocatedType(),
        ConstantInt::get(
            i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
                     AI->getAllocatedType()) /
                     8),
        IRBuilder<>(insertBefore).CreateZExtOrTrunc(AI->getArraySize(), i64),
        nullptr, nam);
    assert(rep->getType() == AI->getType());
    AI->replaceAllUsesWith(rep);
    AI->eraseFromParent();
  }
}

// Create a stack variable containing the size of the allocation
// error if not possible (e.g. not local)
static inline AllocaInst *
OldAllocationSize(Value *Ptr, CallInst *Loc, Function *NewF, IntegerType *T,
                  const std::map<CallInst *, Value *> &reallocSizes) {
  IRBuilder<> B(&*NewF->getEntryBlock().begin());
  AllocaInst *AI = B.CreateAlloca(T);

  std::set<std::pair<Value *, Instruction *>> seen;
  std::deque<std::pair<Value *, Instruction *>> todo = {{Ptr, Loc}};

  while (todo.size()) {
    auto next = todo.front();
    todo.pop_front();
    if (seen.count(next))
      continue;
    seen.insert(next);

    if (auto CI = dyn_cast<CastInst>(next.first)) {
      todo.push_back({CI->getOperand(0), CI});
      continue;
    }

    // Assume zero size if realloc of undef pointer
    if (isa<UndefValue>(next.first)) {
      B.SetInsertPoint(next.second);
      B.CreateStore(ConstantInt::get(T, 0), AI);
      continue;
    }

    if (auto CE = dyn_cast<ConstantExpr>(next.first)) {
      if (CE->isCast()) {
        todo.push_back({CE->getOperand(0), next.second});
        continue;
      }
    }

    if (auto C = dyn_cast<Constant>(next.first)) {
      if (C->isNullValue()) {
        B.SetInsertPoint(next.second);
        B.CreateStore(ConstantInt::get(T, 0), AI);
        continue;
      }
    }
    if (auto CI = dyn_cast<ConstantInt>(next.first)) {
      // if negative or below 0xFFF this cannot possibly represent
      // a real pointer, so ignore this case by setting to 0
      if (CI->isNegative() || CI->getLimitedValue() <= 0xFFF) {
        B.SetInsertPoint(next.second);
        B.CreateStore(ConstantInt::get(T, 0), AI);
        continue;
      }
    }

    // Todo consider more general method for selects
    if (auto SI = dyn_cast<SelectInst>(next.first)) {
      if (auto C1 = dyn_cast<ConstantInt>(SI->getTrueValue())) {
        // if negative or below 0xFFF this cannot possibly represent
        // a real pointer, so ignore this case by setting to 0
        if (C1->isNegative() || C1->getLimitedValue() <= 0xFFF) {
          if (auto C2 = dyn_cast<ConstantInt>(SI->getFalseValue())) {
            if (C2->isNegative() || C2->getLimitedValue() <= 0xFFF) {
              B.SetInsertPoint(next.second);
              B.CreateStore(ConstantInt::get(T, 0), AI);
              continue;
            }
          }
        }
      }
    }

    if (auto PN = dyn_cast<PHINode>(next.first)) {
      for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
        todo.push_back({PN->getIncomingValue(i),
                        PN->getIncomingBlock(i)->getTerminator()});
      }
      continue;
    }

    if (auto CI = dyn_cast<CallInst>(next.first)) {
      if (auto F = CI->getCalledFunction()) {
        if (F->getName() == "malloc") {
          B.SetInsertPoint(next.second);
          B.CreateStore(CI->getArgOperand(0), AI);
          continue;
        }
        if (F->getName() == "calloc") {
          B.SetInsertPoint(next.second);
          B.CreateStore(B.CreateMul(CI->getArgOperand(0), CI->getArgOperand(1)),
                        AI);
          continue;
        }
        if (F->getName() == "realloc") {
          assert(reallocSizes.find(CI) != reallocSizes.end());
          B.SetInsertPoint(next.second);
          B.CreateStore(reallocSizes.find(CI)->second, AI);
          continue;
        }
      }
    }

    if (auto LI = dyn_cast<LoadInst>(next.first)) {
      bool success = false;
      for (Instruction *prev = LI->getPrevNode(); prev != nullptr;
           prev = prev->getPrevNode()) {
        if (auto CI = dyn_cast<CallInst>(prev)) {
          if (auto F = CI->getCalledFunction()) {
            if (F->getName() == "posix_memalign" &&
                CI->getArgOperand(0) == LI->getOperand(0)) {
              B.SetInsertPoint(next.second);
              B.CreateStore(CI->getArgOperand(2), AI);
              success = true;
              break;
            }
          }
        }
        if (prev->mayWriteToMemory()) {
          break;
        }
      }
      if (success)
        continue;
    }

    // llvm::errs() << *NewF->getParent() << "\n";
    // llvm::errs() << *NewF << "\n";
    EmitFailure("DynamicReallocSize", Loc->getDebugLoc(), Loc,
                "could not statically determine size of realloc ", *Loc,
                " - because of - ", *next.first);

    std::string allocName;
    switch (llvm::Triple(NewF->getParent()->getTargetTriple()).getOS()) {
    case llvm::Triple::Linux:
    case llvm::Triple::FreeBSD:
    case llvm::Triple::NetBSD:
    case llvm::Triple::OpenBSD:
    case llvm::Triple::Fuchsia:
      allocName = "malloc_usable_size";
      break;

    case llvm::Triple::Darwin:
    case llvm::Triple::IOS:
    case llvm::Triple::MacOSX:
    case llvm::Triple::WatchOS:
    case llvm::Triple::TvOS:
      allocName = "malloc_size";
      break;

    case llvm::Triple::Win32:
      allocName = "_msize";
      break;

    default:
      llvm_unreachable("unknown reallocation for OS");
    }

    AttributeList list;
    list = list.addAttribute(NewF->getContext(), AttributeList::FunctionIndex,
                             Attribute::ReadOnly);
    list = list.addParamAttribute(NewF->getContext(), 0, Attribute::ReadNone);
    list = list.addParamAttribute(NewF->getContext(), 0, Attribute::NoCapture);
    auto allocSize = NewF->getParent()->getOrInsertFunction(
        allocName,
        FunctionType::get(
            IntegerType::get(NewF->getContext(), 8 * sizeof(size_t)),
            {Type::getInt8PtrTy(NewF->getContext())}, /*isVarArg*/ false),
        list);

    B.SetInsertPoint(Loc);
    Value *sz = B.CreateZExtOrTrunc(B.CreateCall(allocSize, {Ptr}), T);
    B.CreateStore(sz, AI);
    return AI;

    llvm_unreachable("DynamicReallocSize");
  }
  return AI;
}

/// Calls to realloc with an appropriate implementation
void ReplaceReallocs(Function *NewF, bool mem2reg) {
  if (mem2reg) {
    DominatorTree DT(*NewF);
    PromoteMemoryToRegister(*NewF, DT);
  }

  std::vector<CallInst *> ToConvert;
  std::map<CallInst *, Value *> reallocSizes;
  IntegerType *T;

  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "realloc") {
            ToConvert.push_back(CI);
            IRBuilder<> B(CI->getNextNode());
            T = cast<IntegerType>(CI->getArgOperand(1)->getType());
            reallocSizes[CI] = B.CreatePHI(T, 0);
          }
        }
      }
    }
  }

  std::vector<AllocaInst *> memoryLocations;

  for (auto CI : ToConvert) {
    AllocaInst *AI =
        OldAllocationSize(CI->getArgOperand(0), CI, NewF, T, reallocSizes);

    BasicBlock *resize =
        BasicBlock::Create(CI->getContext(), "resize" + CI->getName(), NewF);
    assert(resize->getParent() == NewF);

    BasicBlock *splitParent = CI->getParent();
    BasicBlock *nextBlock = splitParent->splitBasicBlock(CI);

    splitParent->getTerminator()->eraseFromParent();
    IRBuilder<> B(splitParent);

    Value *p = CI->getArgOperand(0);
    Value *req = CI->getArgOperand(1);
    Value *old = B.CreateLoad(AI);

    Value *cmp = B.CreateICmpULE(req, old);
    // if (req < old)
    B.CreateCondBr(cmp, nextBlock, resize);

    B.SetInsertPoint(resize);
    //    size_t newsize = nextPowerOfTwo(req);
    //    void* next = malloc(newsize);
    //    memcpy(next, p, newsize);
    //    free(p);
    //    return { next, newsize };

    Value *newsize = nextPowerOfTwo(B, req);
    CallInst *next = cast<CallInst>(CallInst::CreateMalloc(
        resize, newsize->getType(), Type::getInt8Ty(CI->getContext()), newsize,
        nullptr, (Function *)nullptr, ""));
    resize->getInstList().push_back(next);
    B.SetInsertPoint(resize);

    auto volatile_arg = ConstantInt::getFalse(CI->getContext());

    Value *nargs[] = {next, p, old, volatile_arg};

    Type *tys[] = {next->getType(), p->getType(), old->getType()};

    auto memcpyF =
        Intrinsic::getDeclaration(NewF->getParent(), Intrinsic::memcpy, tys);

    auto mem = cast<CallInst>(B.CreateCall(memcpyF, nargs));
    mem->setCallingConv(memcpyF->getCallingConv());

    CallInst *freeCall = cast<CallInst>(CallInst::CreateFree(p, resize));
    resize->getInstList().push_back(freeCall);
    B.SetInsertPoint(resize);

    B.CreateBr(nextBlock);

    // else
    //   return { p, old }
    B.SetInsertPoint(&*nextBlock->begin());

    PHINode *retPtr = B.CreatePHI(CI->getType(), 2);
    retPtr->addIncoming(p, splitParent);
    retPtr->addIncoming(next, resize);
    CI->replaceAllUsesWith(retPtr);
    std::string nam = CI->getName().str();
    CI->setName("");
    retPtr->setName(nam);
    Value *nextSize = B.CreateSelect(cmp, old, req);
    reallocSizes[CI]->replaceAllUsesWith(nextSize);
    cast<PHINode>(reallocSizes[CI])->eraseFromParent();
    reallocSizes[CI] = nextSize;
  }

  for (auto CI : ToConvert) {
    CI->eraseFromParent();
  }

  DominatorTree DT(*NewF);
  PromoteMemToReg(memoryLocations, DT, /*AC*/ nullptr);
}

/// Perform recursive inlinining on NewF up to the given limit
static void ForceRecursiveInlining(Function *NewF, size_t Limit) {
  for (size_t count = 0; count < Limit; count++) {
    for (auto &BB : *NewF) {
      for (auto &I : BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          if (CI->getCalledFunction() == nullptr)
            continue;
          if (CI->getCalledFunction()->empty())
            continue;
          if (CI->getCalledFunction()->getName().startswith(
                  "_ZN3std2io5stdio6_print"))
            continue;
          if (CI->getCalledFunction()->getName().startswith("_ZN4core3fmt"))
            continue;
          if (CI->getCalledFunction()->hasFnAttribute(
                  Attribute::ReturnsTwice) ||
              CI->getCalledFunction()->hasFnAttribute(Attribute::NoInline))
            continue;
          if (IsFunctionRecursive(CI->getCalledFunction())) {
            LLVM_DEBUG(llvm::dbgs()
                       << "not inlining recursive "
                       << CI->getCalledFunction()->getName() << "\n");
            continue;
          }
          InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 11
          InlineFunction(*CI, IFI);
#else
          InlineFunction(CI, IFI);
#endif
          goto outermostContinue;
        }
      }
    }

    // No functions were inlined, break
    break;

  outermostContinue:;
  }
}

void CanonicalizeLoops(Function *F, TargetLibraryInfo &TLI) {

  DominatorTree DT(*F);
  LoopInfo LI(DT);
  AssumptionCache AC(*F);
  MustExitScalarEvolution SE(*F, TLI, AC, DT, LI);
  for (auto &L : LI) {
    auto pair =
        InsertNewCanonicalIV(L, Type::getInt64Ty(F->getContext()), "tiv");
    PHINode *CanonicalIV = pair.first;
    assert(CanonicalIV);
    RemoveRedundantIVs(L->getHeader(), CanonicalIV, SE,
                       [&](Instruction *I) { I->eraseFromParent(); });
  }
}

Function *preprocessForClone(Function *F, AAResults &AA, TargetLibraryInfo &TLI,
                             bool topLevel) {
  static std::map<std::pair<Function *, bool>, Function *> cache;

  // If we've already processed this, return the previous version
  // and derive aliasing information
  if (cache.find(std::make_pair(F, topLevel)) != cache.end()) {
    Function *NewF = cache[std::make_pair(F, topLevel)];
    AssumptionCache *AC = new AssumptionCache(*NewF);
    DominatorTree *DT = new DominatorTree(*NewF);
    LoopInfo *LI = new LoopInfo(*DT);
#if LLVM_VERSION_MAJOR > 6
    PhiValues *PV = new PhiValues(*NewF);
#endif
    auto BAA = new BasicAAResult(NewF->getParent()->getDataLayout(),
#if LLVM_VERSION_MAJOR > 6
                                 *NewF,
#endif
                                 TLI, *AC, DT, LI
#if LLVM_VERSION_MAJOR > 6
                                 ,
                                 PV
#endif
    );
    AA.addAAResult(*BAA);
    AA.addAAResult(*(new TypeBasedAAResult()));
    return NewF;
  }

  Function *NewF =
      Function::Create(F->getFunctionType(), F->getLinkage(),
                       "preprocess_" + F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    VMap[i] = j;
    j->setName(i->getName());
    ++i;
    ++j;
  }

  SmallVector<ReturnInst *, 4> Returns;
  // if (auto SP = F->getSubProgram()) {
  //  VMap[SP] = DISubprogram::get(SP);
  //}

  CloneFunctionInto(NewF, F, VMap,
                    /*ModuleLevelChanges*/ F->getSubprogram() != nullptr,
                    Returns, "", nullptr);
  NewF->setAttributes(F->getAttributes());

  if (EnzymePreopt) {
    if (EnzymeInline) {
      ForceRecursiveInlining(NewF, /*Limit*/ EnzymeInlineCount);
    }
  }

  if (EnzymeLowerGlobals) {
    std::vector<CallInst*> Calls;
    std::vector<ReturnInst*> Returns;
    for (BasicBlock &BB : *NewF) {
      for (Instruction &I : BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          Calls.push_back(CI);
        }
        if (auto RI = dyn_cast<ReturnInst>(&I)) {
          Returns.push_back(RI);
        }
      }
    }

    //AAResults AA2;
    //AA2.addAAResult(AA);
      // Alias analysis is necessary to ensure can query whether we can move a
      // forward pass function
      AssumptionCache AC(*NewF);
      DominatorTree DT(*NewF);
      LoopInfo LI(DT);
  #if LLVM_VERSION_MAJOR > 6
      PhiValues PV(*NewF);
  #endif
      BasicAAResult AA2(NewF->getParent()->getDataLayout(),
  #if LLVM_VERSION_MAJOR > 6
                                  *NewF,
  #endif
                                  TLI, AC, &DT, &LI
  #if LLVM_VERSION_MAJOR > 6
                                  ,
                                  &PV
  #endif
      );
      //AA2.addAAResult(BAA);
      //TypeBasedAAResult TBAAR();
      //AA2.addAAResult(TBAAR);

    for (auto &g : NewF->getParent()->globals()) {
      std::set<Constant*> seen;
      std::deque<Constant*> todo = { (Constant*)&g };
      bool inF = false;
      while(todo.size()) {
        auto GV = todo.front();
        todo.pop_front();
        if (!seen.insert(GV).second) continue;
        for (auto u : GV->users()) {
          if (auto C = dyn_cast<Constant>(u)) {
            todo.push_back(C);
          } else if (auto I = dyn_cast<Instruction>(u)) {
            if (I->getParent()->getParent() == NewF) {
              inF = true;
              goto doneF;
            }
          }
        }
      }
      doneF:;
      if (inF) {
        bool seen = false;    
        MemoryLocation
#if LLVM_VERSION_MAJOR >= 12
        Loc = MemoryLocation(&g, LocationSize::beforeOrAfterPointer());
#elif LLVM_VERSION_MAJOR >= 9
        Loc = MemoryLocation(&g, LocationSize::unknown());
#else
        Loc = MemoryLocation(&g, MemoryLocation::UnknownSize);
#endif

        for (CallInst* CI : Calls) {
          if (isa<IntrinsicInst>(CI)) continue;
          Function* F = CI->getCalledFunction();
          #if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
          #else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
          #endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                    F = fn;
                }
            }
          if (F && (isMemFreeLibMFunction(F->getName()) || F->getName() == "__fd_sincos_1")) {
            continue;
          }
          if (llvm::isModOrRefSet(AA2.getModRefInfo(CI, Loc))) {
            seen = true;
            goto endCheck;
          }
        }
        endCheck:;
        if (!seen) {
          IRBuilder<> bb(&NewF->getEntryBlock(), NewF->getEntryBlock().begin());
          AllocaInst *antialloca = bb.CreateAlloca(g.getValueType(),
            g.getType()->getPointerAddressSpace(), nullptr, g.getName() + "_local");

          if (g.getAlignment()) {
      #if LLVM_VERSION_MAJOR >= 10
            antialloca->setAlignment(Align(g.getAlignment()));
      #else
            antialloca->setAlignment(g.getAlignment());
      #endif
          }

          std::map<Constant*, Value*> remap;
          remap[&g] = antialloca;

          std::deque<Constant*> todo = { &g };
          while(todo.size()) {
            auto GV = todo.front();
            todo.pop_front();
            if (&g != GV && remap.find(GV) != remap.end()) continue;
            Value* replaced = nullptr;
            if (remap.find(GV) != remap.end()) {
              replaced = remap[GV];
            } else if (auto CE = dyn_cast<ConstantExpr>(GV)) {
              auto I = CE->getAsInstruction();
              bb.Insert(I);
              assert(isa<Constant>(I->getOperand(0)));
              assert(remap[cast<Constant>(I->getOperand(0))]);
              I->setOperand(0, remap[cast<Constant>(I->getOperand(0))]);
              replaced = remap[GV] = I;
            }
            assert(replaced && "unhandled constantexpr");

            std::vector<std::pair<Instruction*, size_t>> uses;
            for (Use &U : GV->uses()) {
              if (auto I = dyn_cast<Instruction>(U.getUser())) {
                if (I->getParent()->getParent() == NewF) {
                  uses.emplace_back(I, U.getOperandNo());
                }
              }
              if (auto C = dyn_cast<Constant>(U.getUser())) {
                assert(C != &g);
                todo.push_back(C);
              }
            }
            for(auto &U : uses) {
              U.first->setOperand(U.second, replaced);
            }
          }

          auto ld = bb.CreateLoad(&g);
          auto st = bb.CreateStore(ld, antialloca);
          if (g.getAlignment()) {
  #if LLVM_VERSION_MAJOR >= 10
            st->setAlignment(Align(g.getAlignment()));
            ld->setAlignment(Align(g.getAlignment()));
  #else
            st->setAlignment(g.getAlignment());
            ld->setAlignment(g.getAlignment());
  #endif
          }
          for (ReturnInst* RI : Returns) {
            IRBuilder<> IB(RI);
            auto ld = IB.CreateLoad(antialloca);
            auto st = IB.CreateStore(ld, &g);
            if (g.getAlignment()) {
    #if LLVM_VERSION_MAJOR >= 10
              st->setAlignment(Align(g.getAlignment()));
              ld->setAlignment(Align(g.getAlignment()));
    #else
              st->setAlignment(g.getAlignment());
              ld->setAlignment(g.getAlignment());
    #endif
            }
          }
        }
      }
    }
  }

  {
    std::vector<Instruction *> FreesToErase;
    for (auto &BB : *NewF) {
      for (auto &I : BB) {

        if (auto CI = dyn_cast<CallInst>(&I)) {

          Function *called = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
          {
            if (castinst->isCast()) {
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (isDeallocationFunction(*fn, TLI)) {
                  called = fn;
                }
              }
            }
          }

          if (called && isDeallocationFunction(*called, TLI)) {
            FreesToErase.push_back(CI);
          }
        }
      }
    }
    // TODO we should ensure these are kept to avoid accidentially creating
    // a memory leak
    for (auto Free : FreesToErase) {
      Free->eraseFromParent();
    }
  }

  if (EnzymePreopt) {
    {
      FunctionAnalysisManager AM;
      AM.registerPass([] { return TargetLibraryAnalysis(); });
      LowerInvokePass().run(*NewF, AM);
#if LLVM_VERSION_MAJOR >= 9
      llvm::EliminateUnreachableBlocks(*NewF);
#else
      removeUnreachableBlocks(*NewF);
#endif
    }

    {
      DominatorTree DT(*NewF);
      PromoteMemoryToRegister(*NewF, DT);
    }

    {
      FunctionAnalysisManager AM;
      AM.registerPass([] { return AAManager(); });
      AM.registerPass([] { return ScalarEvolutionAnalysis(); });
      AM.registerPass([] { return AssumptionAnalysis(); });
      AM.registerPass([] { return TargetLibraryAnalysis(); });
      AM.registerPass([] { return TargetIRAnalysis(); });
      AM.registerPass([] { return MemorySSAAnalysis(); });
      AM.registerPass([] { return DominatorTreeAnalysis(); });
      AM.registerPass([] { return MemoryDependenceAnalysis(); });
      AM.registerPass([] { return LoopAnalysis(); });
      AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
      AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
      AM.registerPass([] { return LazyValueAnalysis(); });
#if LLVM_VERSION_MAJOR >= 8
      AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR <= 7
      GVN().run(*NewF, AM);
#endif
      SROA().run(*NewF, AM);
    }

    ReplaceReallocs(NewF);

    {
      FunctionAnalysisManager AM;
      AM.registerPass([] { return AAManager(); });
      AM.registerPass([] { return ScalarEvolutionAnalysis(); });
      AM.registerPass([] { return AssumptionAnalysis(); });
      AM.registerPass([] { return TargetLibraryAnalysis(); });
      AM.registerPass([] { return TargetIRAnalysis(); });
      AM.registerPass([] { return MemorySSAAnalysis(); });
      AM.registerPass([] { return DominatorTreeAnalysis(); });
      AM.registerPass([] { return MemoryDependenceAnalysis(); });
      AM.registerPass([] { return LoopAnalysis(); });
      AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
      AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
      AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
      AM.registerPass([] { return LazyValueAnalysis(); });
      SROA().run(*NewF, AM);

#if LLVM_VERSION_MAJOR >= 12
      SimplifyCFGOptions scfgo;
#else
      SimplifyCFGOptions scfgo(
          /*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false,
          /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true,
          /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
#endif
      SimplifyCFGPass(scfgo).run(*NewF, AM);
    }
  }

  ReplaceReallocs(NewF);

  // Run LoopSimplifyPass to ensure preheaders exist on all loops
  {
    FunctionAnalysisManager AM;
    AM.registerPass([] { return LoopAnalysis(); });
    AM.registerPass([] { return DominatorTreeAnalysis(); });
    AM.registerPass([] { return ScalarEvolutionAnalysis(); });
    AM.registerPass([] { return AssumptionAnalysis(); });
#if LLVM_VERSION_MAJOR >= 8
    AM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif
#if LLVM_VERSION_MAJOR >= 8
    AM.registerPass([] { return MemorySSAAnalysis(); });
#endif
    LoopSimplifyPass().run(*NewF, AM);
  }

  // For subfunction calls upgrade stack allocations to mallocs
  // to ensure availability in the reverse pass
  // TODO we should ensure these are kept to avoid accidentially creating
  // a memory leak
  UpgradeAllocasToMallocs(NewF, topLevel);

  {
    // Alias analysis is necessary to ensure can query whether we can move a
    // forward pass function
    AssumptionCache *AC = new AssumptionCache(*NewF);
    DominatorTree *DT = new DominatorTree(*NewF);
    LoopInfo *LI = new LoopInfo(*DT);
#if LLVM_VERSION_MAJOR > 6
    PhiValues *PV = new PhiValues(*NewF);
#endif
    auto BAA = new BasicAAResult(NewF->getParent()->getDataLayout(),
#if LLVM_VERSION_MAJOR > 6
                                 *NewF,
#endif
                                 TLI, *AC, DT, LI
#if LLVM_VERSION_MAJOR > 6
                                 ,
                                 PV
#endif
    );
    AA.addAAResult(*BAA);
    AA.addAAResult(*(new TypeBasedAAResult()));
  }

  CanonicalizeLoops(NewF, TLI);

  if (EnzymePrint)
    llvm::errs() << "after simplification :\n" << *NewF << "\n";

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (1)");
  }
  cache[std::make_pair(F, topLevel)] = NewF;
  return NewF;
}

Function *CloneFunctionWithReturns(
    bool topLevel, Function *&F, AAResults &AA, TargetLibraryInfo &TLI,
    ValueToValueMapTy &ptrInputs, const std::vector<DIFFE_TYPE> &constant_args,
    SmallPtrSetImpl<Value *> &constants, SmallPtrSetImpl<Value *> &nonconstant,
    SmallPtrSetImpl<Value *> &returnvals, ReturnType returnValue, Twine name,
    ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type *additionalArg) {
  assert(!F->empty());
  F = preprocessForClone(F, AA, TLI, topLevel);
  std::vector<Type *> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::ArgsWithTwoReturns)
    RetTypes.push_back(F->getReturnType());
  if (returnValue == ReturnType::ArgsWithTwoReturns)
    RetTypes.push_back(F->getReturnType());
  std::vector<Type *> ArgTypes;

  ValueToValueMapTy VMap;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;
  for (const Argument &I : F->args()) {
    ArgTypes.push_back(I.getType());
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(I.getType());
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(I.getType());
    }
    ++argno;
  }

  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto ri = dyn_cast<ReturnInst>(&I)) {
        if (auto rv = ri->getReturnValue()) {
          returnvals.insert(rv);
        }
      }
    }
  }

  if (diffeReturnArg) {
    assert(!F->getReturnType()->isVoidTy());
    ArgTypes.push_back(F->getReturnType());
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }
  Type *RetType = StructType::get(F->getContext(), RetTypes);
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.clear();
    RetTypes.push_back(Type::getInt8PtrTy(F->getContext()));
    if (returnValue == ReturnType::TapeAndTwoReturns) {
      RetTypes.push_back(F->getReturnType());
      RetTypes.push_back(F->getReturnType());
    } else if (returnValue == ReturnType::TapeAndReturn) {
      RetTypes.push_back(F->getReturnType());
    }
    RetType = StructType::get(F->getContext(), RetTypes);
  }

  bool noReturn = RetTypes.size() == 0;
  if (noReturn)
    RetType = Type::getVoidTy(RetType->getContext());

  // Create a new function type...
  FunctionType *FTy =
      FunctionType::get(RetType, ArgTypes, F->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(FTy, F->getLinkage(), name, F->getParent());
  if (diffeReturnArg) {
    auto I = NewF->arg_end();
    I--;
    if (additionalArg)
      I--;
    I->setName("differeturn");
  }
  if (additionalArg) {
    auto I = NewF->arg_end();
    I--;
    I->setName("tapeArg");
  }

  {
    unsigned ii = 0;
    for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
      VMap[i] = j;
      ++j;
      ++i;
      if (constant_args[ii] == DIFFE_TYPE::DUP_ARG ||
          constant_args[ii] == DIFFE_TYPE::DUP_NONEED) {
        ++j;
      }
      ++ii;
    }
  }

  // Loop over the arguments, copying the names of the mapped arguments over...
  Function::arg_iterator DestI = NewF->arg_begin();

  for (const Argument &I : F->args())
    if (VMap.count(&I) == 0) {     // Is this argument preserved?
      DestI->setName(I.getName()); // Copy the name over...
      VMap[&I] = &*DestI++;        // Add mapping to VMap
    }
  SmallVector<ReturnInst *, 4> Returns;
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
  if (VMapO) {
    VMapO->insert(VMap.begin(), VMap.end());
    VMapO->getMDMap() = VMap.getMDMap();
  }

  bool hasPtrInput = false;
  unsigned ii = 0, jj = 0;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    if (constant_args[ii] == DIFFE_TYPE::CONSTANT) {
      constants.insert(i);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName()
                     << " constant arg " << *j << "\n";
    } else {
      nonconstant.insert(i);
      if (printconst)
        llvm::errs() << "in new function " << NewF->getName()
                     << " nonconstant arg " << *j << "\n";
    }

    if (constant_args[ii] == DIFFE_TYPE::DUP_ARG ||
        constant_args[ii] == DIFFE_TYPE::DUP_NONEED) {
      hasPtrInput = true;
      ptrInputs[i] = (j + 1);
      if (F->hasParamAttribute(ii, Attribute::NoCapture)) {
        NewF->addParamAttr(jj + 1, Attribute::NoCapture);
      }

      j->setName(i->getName());
      ++j;
      j->setName(i->getName() + "'");
      nonconstant.insert(j);
      ++j;
      jj += 2;

      ++i;

    } else {
      j->setName(i->getName());
      ++j;
      ++jj;
      ++i;
    }
    ++ii;
  }

  if (hasPtrInput) {
    if (NewF->hasFnAttribute(Attribute::ReadNone)) {
      NewF->removeFnAttr(Attribute::ReadNone);
    }
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
      NewF->removeFnAttr(Attribute::ReadOnly);
    }
  }
  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
  assert(NewF->hasLocalLinkage());

  return NewF;
}

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
void optimizeIntermediate(GradientUtils *gutils, bool topLevel, Function *F) {
  {
    DominatorTree DT(*F);
    PromoteMemoryToRegister(*F, DT);
  }

  FunctionAnalysisManager AM;
  AM.registerPass([] { return AAManager(); });
  AM.registerPass([] { return ScalarEvolutionAnalysis(); });
  AM.registerPass([] { return AssumptionAnalysis(); });
  AM.registerPass([] { return TargetLibraryAnalysis(); });
  AM.registerPass([] { return TargetIRAnalysis(); });
  AM.registerPass([] { return MemorySSAAnalysis(); });
  AM.registerPass([] { return DominatorTreeAnalysis(); });
  AM.registerPass([] { return MemoryDependenceAnalysis(); });
  AM.registerPass([] { return LoopAnalysis(); });
  AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
  AM.registerPass([] { return PhiValuesAnalysis(); });
#endif
  AM.registerPass([] { return LazyValueAnalysis(); });
  LoopAnalysisManager LAM;
  AM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });
  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(AM); });

#if LLVM_VERSION_MAJOR <= 7
  GVN().run(*F, AM);
  SROA().run(*F, AM);
  EarlyCSEPass(/*memoryssa*/ true).run(*F, AM);
#endif

  PassManagerBuilder Builder;
  Builder.OptLevel = 2;
  legacy::FunctionPassManager PM(F->getParent());
  Builder.populateFunctionPassManager(PM);
  PM.run(*F);

  // DCEPass().run(*F, AM);
}
