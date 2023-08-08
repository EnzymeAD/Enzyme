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

#include "DiffeGradientUtils.h"
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
#include "llvm/Passes/PassBuilder.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"

#if LLVM_VERSION_MAJOR < 16
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#endif
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"

#include "llvm/Analysis/PhiValues.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "llvm/Transforms/Utils.h"

#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/CorrelatedValuePropagation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/LoopIdiomRecognize.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LowerInvoke.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include "llvm/IR/LegacyPassManager.h"
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"

#include "CacheUtility.h"

#define DEBUG_TYPE "enzyme"
using namespace llvm;

extern "C" {
cl::opt<bool> EnzymePreopt("enzyme-preopt", cl::init(true), cl::Hidden,
                           cl::desc("Run enzyme preprocessing optimizations"));

cl::opt<bool> EnzymeInline("enzyme-inline", cl::init(false), cl::Hidden,
                           cl::desc("Force inlining of autodiff"));

cl::opt<bool> EnzymeNoAlias("enzyme-noalias", cl::init(false), cl::Hidden,
                            cl::desc("Force noalias of autodiff"));
#if LLVM_VERSION_MAJOR < 16
cl::opt<bool>
    EnzymeAggressiveAA("enzyme-aggressive-aa", cl::init(false), cl::Hidden,
                       cl::desc("Use more unstable but aggressive LLVM AA"));
#endif
cl::opt<bool> EnzymeLowerGlobals(
    "enzyme-lower-globals", cl::init(false), cl::Hidden,
    cl::desc("Lower globals to locals assuming the global values are not "
             "needed outside of this gradient"));

cl::opt<int>
    EnzymeInlineCount("enzyme-inline-count", cl::init(10000), cl::Hidden,
                      cl::desc("Limit of number of functions to inline"));

cl::opt<bool> EnzymeCoalese("enzyme-coalese", cl::init(false), cl::Hidden,
                            cl::desc("Whether to coalese memory allocations"));

static cl::opt<bool> EnzymePHIRestructure(
    "enzyme-phi-restructure", cl::init(false), cl::Hidden,
    cl::desc("Whether to restructure phi's to have better unwrap behavior"));

cl::opt<bool>
    EnzymeNameInstructions("enzyme-name-instructions", cl::init(false),
                           cl::Hidden,
                           cl::desc("Have enzyme name all instructions"));

cl::opt<bool> EnzymeSelectOpt("enzyme-select-opt", cl::init(true), cl::Hidden,
                              cl::desc("Run Enzyme select optimization"));

cl::opt<int> EnzymePostOptLevel(
    "enzyme-post-opt-level", cl::init(0), cl::Hidden,
    cl::desc("Post optimization level within Enzyme differentiated function"));
}

/// Is the use of value val as an argument of call CI potentially captured
bool couldFunctionArgumentCapture(llvm::CallInst *CI, llvm::Value *val) {
  Function *F = CI->getCalledFunction();

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

  if (F == nullptr)
    return true;

  if (F->getIntrinsicID() == Intrinsic::memset)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memcpy)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memmove)
    return false;

  auto arg = F->arg_begin();
#if LLVM_VERSION_MAJOR >= 14
  for (size_t i = 0, size = CI->arg_size(); i < size; i++)
#else
  for (size_t i = 0, size = CI->getNumArgOperands(); i < size; i++)
#endif
  {
    if (val == CI->getArgOperand(i)) {
      // This is a vararg, assume captured
      if (arg == F->arg_end()) {
        return true;
      } else {
        if (!arg->hasNoCaptureAttr()) {
          return true;
        }
      }
    }
    if (arg != F->arg_end())
      arg++;
  }
  // No argument captured
  return false;
}

enum RecurType {
  MaybeRecursive = 1,
  NotRecursive = 2,
  DefinitelyRecursive = 3,
};
/// Return whether this function eventually calls itself
static bool
IsFunctionRecursive(Function *F,
                    std::map<const Function *, RecurType> &Results) {

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
          IsFunctionRecursive(call->getCalledFunction(), Results);
        }
        if (auto call = dyn_cast<InvokeInst>(&I)) {
          if (call->getCalledFunction() == nullptr)
            continue;
          if (call->getCalledFunction()->empty())
            continue;
          IsFunctionRecursive(call->getCalledFunction(), Results);
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
    if (auto SI = dyn_cast<StoreInst>(U))
      if (SI->getPointerOperand() == AI)
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

void RecursivelyReplaceAddressSpace(Value *AI, Value *rep, bool legal) {
  SmallVector<std::tuple<Value *, Value *, Instruction *>, 1> Todo;
  for (auto U : AI->users()) {
    Todo.push_back(
        std::make_tuple((Value *)rep, (Value *)AI, cast<Instruction>(U)));
  }
  SmallVector<Instruction *, 1> toErase;
  if (auto I = dyn_cast<Instruction>(AI))
    toErase.push_back(I);
  SmallVector<StoreInst *, 1> toPostCache;
  while (Todo.size()) {
    auto cur = Todo.back();
    Todo.pop_back();
    Value *rep = std::get<0>(cur);
    Value *prev = std::get<1>(cur);
    Value *inst = std::get<2>(cur);
    if (auto ASC = dyn_cast<AddrSpaceCastInst>(inst)) {
      auto AS = cast<PointerType>(rep->getType())->getAddressSpace();
      if (AS == ASC->getDestAddressSpace()) {
        ASC->replaceAllUsesWith(rep);
        toErase.push_back(ASC);
        continue;
      }
      ASC->setOperand(0, rep);
      continue;
    }
    if (auto CI = dyn_cast<CastInst>(inst)) {
      if (!CI->getType()->isPointerTy()) {
        CI->setOperand(0, rep);
        continue;
      }
      IRBuilder<> B(CI);
      auto nCI = cast<CastInst>(B.CreateCast(
          CI->getOpcode(), rep,
#if LLVM_VERSION_MAJOR < 18
          PointerType::get(CI->getType()->getPointerElementType(),
                           cast<PointerType>(rep->getType())->getAddressSpace())
#else
          rep->getType()
#endif
              ));
      nCI->takeName(CI);
      for (auto U : CI->users()) {
        Todo.push_back(
            std::make_tuple((Value *)nCI, (Value *)CI, cast<Instruction>(U)));
      }
      toErase.push_back(CI);
      continue;
    }
    if (auto GEP = dyn_cast<GetElementPtrInst>(inst)) {
      IRBuilder<> B(GEP);
      SmallVector<Value *, 1> ind(GEP->indices());
      auto nGEP = cast<GetElementPtrInst>(
          B.CreateGEP(GEP->getSourceElementType(), rep, ind));
      nGEP->takeName(GEP);
      for (auto U : GEP->users()) {
        Todo.push_back(
            std::make_tuple((Value *)nGEP, (Value *)GEP, cast<Instruction>(U)));
      }
      toErase.push_back(GEP);
      continue;
    }
    if (auto II = dyn_cast<IntrinsicInst>(inst)) {
      if (isIntelSubscriptIntrinsic(*II)) {

        const std::array<size_t, 4> idxArgsIndices{{0, 1, 2, 4}};
        const size_t ptrArgIndex = 3;

        SmallVector<Value *, 5> args(5);
        for (auto i : idxArgsIndices) {
          Value *idx = II->getOperand(i);
          args[i] = idx;
        }
        args[ptrArgIndex] = rep;

        IRBuilder<> B(II);
        auto nII = cast<CallInst>(B.CreateCall(II->getCalledFunction(), args));
#if LLVM_VERSION_MAJOR >= 13
        // Must copy the elementtype attribute as it is needed by the intrinsic
        nII->addParamAttr(
            ptrArgIndex,
            II->getParamAttr(ptrArgIndex, Attribute::AttrKind::ElementType));
#endif
        nII->takeName(II);
        for (auto U : II->users()) {
          Todo.push_back(
              std::make_tuple((Value *)nII, (Value *)II, cast<Instruction>(U)));
        }
        toErase.push_back(II);
        continue;
      }
    }
    if (auto LI = dyn_cast<LoadInst>(inst)) {
      LI->setOperand(0, rep);
      continue;
    }
    if (auto SI = dyn_cast<StoreInst>(inst)) {
      if (SI->getPointerOperand() == prev) {
        SI->setOperand(1, rep);
        toPostCache.push_back(SI);
        continue;
      }
    }
    if (auto MS = dyn_cast<MemSetInst>(inst)) {
      IRBuilder<> B(MS);

      Value *nargs[] = {rep, MS->getArgOperand(1), MS->getArgOperand(2),
                        MS->getArgOperand(3)};

      Type *tys[] = {nargs[0]->getType(), nargs[2]->getType()};

      auto nMS = cast<CallInst>(B.CreateCall(
          Intrinsic::getDeclaration(MS->getParent()->getParent()->getParent(),
                                    Intrinsic::memset, tys),
          nargs));
      nMS->copyMetadata(*MS);
      nMS->setAttributes(MS->getAttributes());
      toErase.push_back(MS);
      continue;
    }
    if (auto MTI = dyn_cast<MemTransferInst>(inst)) {
      IRBuilder<> B(MTI);

      Value *nargs[4] = {MTI->getArgOperand(0), MTI->getArgOperand(1),
                         MTI->getArgOperand(2), MTI->getArgOperand(3)};

      if (nargs[0] == prev)
        nargs[0] = rep;

      if (nargs[1] == prev)
        nargs[1] = rep;

      Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                     nargs[2]->getType()};

      auto nMTI = cast<CallInst>(B.CreateCall(
          Intrinsic::getDeclaration(MTI->getParent()->getParent()->getParent(),
                                    MTI->getIntrinsicID(), tys),
          nargs));
      nMTI->copyMetadata(*MTI);
      nMTI->setAttributes(MTI->getAttributes());
      toErase.push_back(MTI);
      continue;
    }
    if (auto CI = dyn_cast<CallInst>(inst)) {
      if (auto F = CI->getCalledFunction()) {
        if (F->getName() == "julia.write_barrier" && legal) {
          toErase.push_back(CI);
          continue;
        }
      }
      IRBuilder<> B(CI);
      auto Addr = B.CreateAddrSpaceCast(rep, prev->getType());
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 0; i < CI->arg_size(); i++)
#else
      for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
      {
        if (CI->getArgOperand(i) == prev) {
          CI->setArgOperand(i, Addr);
        }
      }
      continue;
    }
    if (auto I = dyn_cast<Instruction>(inst))
      llvm::errs() << *I->getParent()->getParent() << "\n";
    llvm::errs() << " rep: " << *rep << " prev: " << *prev << " inst: " << *inst
                 << "\n";
    llvm_unreachable("Illegal address space propagation");
  }

  for (auto I : llvm::reverse(toErase)) {
    I->eraseFromParent();
  }
  for (auto SI : toPostCache) {
    IRBuilder<> B(SI->getNextNode());
    PostCacheStore(SI, B);
  }
}

/// Convert necessary stack allocations into mallocs for use in the reverse
/// pass. Specifically if we're not topLevel all allocations must be upgraded
/// Even if topLevel any allocations that aren't in the entry block (and
/// therefore may not be reachable in the reverse pass) must be upgraded.
static inline void
UpgradeAllocasToMallocs(Function *NewF, DerivativeMode mode,
                        SmallPtrSetImpl<llvm::BasicBlock *> &Unreachable) {
  SmallVector<AllocaInst *, 4> ToConvert;

  for (auto &BB : *NewF) {
    if (Unreachable.count(&BB))
      continue;
    for (auto &I : BB) {
      if (auto AI = dyn_cast<AllocaInst>(&I)) {
        bool UsableEverywhere = AI->getParent() == &NewF->getEntryBlock();
        // TODO use is_value_needed_in_reverse (requiring GradientUtils)
        if (OnlyUsedInOMP(AI))
          continue;
        if (!UsableEverywhere || mode != DerivativeMode::ReverseModeCombined) {
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
    IRBuilder<> B(insertBefore);
    CallInst *CI = nullptr;
    Instruction *ZeroInst = nullptr;
    auto rep = CreateAllocation(
        B, AI->getAllocatedType(), B.CreateZExtOrTrunc(AI->getArraySize(), i64),
        nam, &CI, /*ZeroMem*/ EnzymeZeroCache ? &ZeroInst : nullptr);
#if LLVM_VERSION_MAJOR > 10
    auto align = AI->getAlign().value();
#else
    auto align = AI->getAlignment();
#endif
    CI->setMetadata(
        "enzyme_fromstack",
        MDNode::get(CI->getContext(),
                    {ConstantAsMetadata::get(ConstantInt::get(
                        IntegerType::get(AI->getContext(), 64), align))}));

    if (rep != CI) {
      cast<Instruction>(rep)->setMetadata("enzyme_caststack",
                                          MDNode::get(CI->getContext(), {}));
    }
    if (ZeroInst) {
      ZeroInst->setMetadata("enzyme_zerostack",
                            MDNode::get(CI->getContext(), {}));
    }

    auto PT0 = cast<PointerType>(rep->getType());
    auto PT1 = cast<PointerType>(AI->getType());
    if (PT0->getAddressSpace() != PT1->getAddressSpace()) {
      RecursivelyReplaceAddressSpace(AI, rep, /*legal*/ false);
    } else {
      assert(rep->getType() == AI->getType());
      AI->replaceAllUsesWith(rep);
      AI->eraseFromParent();
    }
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
#if LLVM_VERSION_MAJOR >= 14
    list = list.addFnAttribute(NewF->getContext(), Attribute::ReadOnly);
#else
    list = list.addAttribute(NewF->getContext(), AttributeList::FunctionIndex,
                             Attribute::ReadOnly);
#endif
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

void PreProcessCache::AlwaysInline(Function *NewF) {

  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  FAM.invalidate(*NewF, PA);
  SmallVector<CallInst *, 2> ToInline;
  SmallVector<Instruction *, 2> ToErase;
  // TODO this logic should be combined with the dynamic loop emission
  // to minimize the number of branches if the realloc is used for multiple
  // values with the same bound.
  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (hasMetadata(&I, "enzyme_zerostack")) {
        if (isa<AllocaInst>(getBaseObject(I.getOperand(0)))) {
          ToErase.push_back(&I);
          continue;
        }
      }
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (!CI->getCalledFunction())
          continue;
        if (CI->getCalledFunction()->hasFnAttribute(Attribute::AlwaysInline))
          ToInline.push_back(CI);
      }
    }
  }
  for (auto I : ToErase) {
    I->eraseFromParent();
  }
  for (auto CI : ToInline) {
    InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 11
    InlineFunction(*CI, IFI);
#else
    InlineFunction(CI, IFI);
#endif
  }
}

void PreProcessCache::LowerAllocAddr(Function *NewF) {
  SmallVector<Instruction *, 1> Todo;
  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (hasMetadata(&I, "enzyme_backstack")) {
        Todo.push_back(&I);
        // TODO
        // I.eraseMetadata("enzyme_backstack");
      }
    }
  }
  for (auto T : Todo) {
    auto T0 = T->getOperand(0);
    if (auto CI = dyn_cast<BitCastInst>(T0))
      T0 = CI->getOperand(0);
    auto AI = cast<AllocaInst>(T0);
    llvm::Value *AIV = AI;
#if LLVM_VERSION_MAJOR < 18
    if (AIV->getType()->getPointerElementType() !=
        T->getType()->getPointerElementType()) {
      IRBuilder<> B(AI->getNextNode());
      AIV = B.CreateBitCast(
          AIV, PointerType::get(
                   T->getType()->getPointerElementType(),
                   cast<PointerType>(AI->getType())->getAddressSpace()));
    }
#endif
    RecursivelyReplaceAddressSpace(T, AIV, /*legal*/ true);
  }
}

/// Calls to realloc with an appropriate implementation
void PreProcessCache::ReplaceReallocs(Function *NewF, bool mem2reg) {
  if (mem2reg) {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  SmallVector<CallInst *, 4> ToConvert;
  std::map<CallInst *, Value *> reallocSizes;
  IntegerType *T = nullptr;

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

  SmallVector<AllocaInst *, 4> memoryLocations;

  for (auto CI : ToConvert) {
    assert(T);
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
    Value *old = B.CreateLoad(AI->getAllocatedType(), AI);
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

    Module *M = NewF->getParent();
    Type *BPTy = Type::getInt8PtrTy(NewF->getContext());
    auto MallocFunc =
        M->getOrInsertFunction("malloc", BPTy, newsize->getType());
    auto next = B.CreateCall(MallocFunc, newsize);
    B.SetInsertPoint(resize);

    auto volatile_arg = ConstantInt::getFalse(CI->getContext());

    Value *nargs[] = {next, p, old, volatile_arg};

    Type *tys[] = {next->getType(), p->getType(), old->getType()};

    auto memcpyF =
        Intrinsic::getDeclaration(NewF->getParent(), Intrinsic::memcpy, tys);

    auto mem = cast<CallInst>(B.CreateCall(memcpyF, nargs));
    mem->setCallingConv(memcpyF->getCallingConv());

    Type *VoidTy = Type::getVoidTy(M->getContext());
    auto FreeFunc = M->getOrInsertFunction("free", VoidTy, BPTy);
    B.CreateCall(FreeFunc, p);
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

  PreservedAnalyses PA;
  FAM.invalidate(*NewF, PA);

  PA = PromotePass().run(*NewF, FAM);
  FAM.invalidate(*NewF, PA);
}

Function *CreateMPIWrapper(Function *F) {
  std::string name = ("enzyme_wrapmpi$$" + F->getName() + "#").str();
  if (auto W = F->getParent()->getFunction(name))
    return W;
  Type *types = {F->getFunctionType()->getParamType(0)};
  auto FT = FunctionType::get(F->getReturnType(), types, false);
  Function *W = Function::Create(FT, GlobalVariable::InternalLinkage, name,
                                 F->getParent());
  llvm::Attribute::AttrKind attrs[] = {
    Attribute::WillReturn,
#if LLVM_VERSION_MAJOR >= 12
    Attribute::MustProgress,
#endif
#if LLVM_VERSION_MAJOR < 16
    Attribute::ReadOnly,
#endif
    Attribute::Speculatable,
    Attribute::NoUnwind,
    Attribute::AlwaysInline,
#if LLVM_VERSION_MAJOR >= 10
    Attribute::NoFree,
    Attribute::NoSync,
#endif
#if LLVM_VERSION_MAJOR < 16
    Attribute::InaccessibleMemOnly
#endif
  };
  for (auto attr : attrs) {
#if LLVM_VERSION_MAJOR >= 14
    W->addFnAttr(attr);
#else
    W->addAttribute(AttributeList::FunctionIndex, attr);
#endif
  }
#if LLVM_VERSION_MAJOR >= 16
  W->setOnlyAccessesInaccessibleMemory();
  W->setOnlyReadsMemory();
#endif
#if LLVM_VERSION_MAJOR >= 14
  W->addFnAttr(Attribute::get(F->getContext(), "enzyme_inactive"));
#else
  W->addAttribute(AttributeList::FunctionIndex,
                  Attribute::get(F->getContext(), "enzyme_inactive"));
#endif
  BasicBlock *entry = BasicBlock::Create(W->getContext(), "entry", W);
  IRBuilder<> B(entry);
  auto alloc = B.CreateAlloca(F->getReturnType());
  Value *args[] = {W->arg_begin(), alloc};

  auto T = F->getFunctionType()->getParamType(1);
  if (!isa<PointerType>(T)) {
    assert(isa<IntegerType>(T));
    args[1] = B.CreatePtrToInt(args[1], T);
  }
  B.CreateCall(F, args);
  B.CreateRet(B.CreateLoad(F->getReturnType(), alloc));
  return W;
}
template <typename T>
static void SimplifyMPIQueries(Function &NewF, FunctionAnalysisManager &FAM) {
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(NewF);
  SmallVector<T *, 4> Todo;
  SmallVector<T *, 4> OMPBounds;
  for (auto &BB : NewF) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<T>(&I)) {
        Function *Fn = CI->getCalledFunction();
        if (Fn == nullptr)
          continue;
        if (Fn->getName() == "MPI_Comm_rank" ||
            Fn->getName() == "PMPI_Comm_rank" ||
            Fn->getName() == "MPI_Comm_size" ||
            Fn->getName() == "PMPI_Comm_size") {
          Todo.push_back(CI);
        }
        if (Fn->getName() == "__kmpc_for_static_init_4" ||
            Fn->getName() == "__kmpc_for_static_init_4u" ||
            Fn->getName() == "__kmpc_for_static_init_8" ||
            Fn->getName() == "__kmpc_for_static_init_8u") {
          OMPBounds.push_back(CI);
        }
      }
    }
  }
  if (Todo.size() == 0 && OMPBounds.size() == 0)
    return;
  for (auto CI : Todo) {
    IRBuilder<> B(CI);
    Value *arg[] = {CI->getArgOperand(0)};
    SmallVector<OperandBundleDef, 2> Defs;
    CI->getOperandBundlesAsDefs(Defs);
    auto res =
        B.CreateCall(CreateMPIWrapper(CI->getCalledFunction()), arg, Defs);
    Value *storePointer = CI->getArgOperand(1);

    // Comm_rank and Comm_size return Err, assume 0 is success
    CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0));
    CI->eraseFromParent();

    while (auto Cast = dyn_cast<CastInst>(storePointer)) {
      storePointer = Cast->getOperand(0);
      if (Cast->use_empty())
        Cast->eraseFromParent();
    }

    B.SetInsertPoint(res);

    if (auto PT = dyn_cast<PointerType>(storePointer->getType())) {
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
      if (PT->getContext().supportsTypedPointers()) {
#endif
        if (PT->getPointerElementType() != res->getType())
          storePointer = B.CreateBitCast(
              storePointer,
              PointerType::get(res->getType(), PT->getAddressSpace()));
#if LLVM_VERSION_MAJOR >= 15
      }
#endif
#endif
    } else {
      assert(isa<IntegerType>(storePointer->getType()));
      storePointer = B.CreateIntToPtr(storePointer,
                                      PointerType::getUnqual(res->getType()));
    }
    if (isa<AllocaInst>(storePointer)) {
      // If this is only loaded from, immedaitely replace
      // Immediately replace all dominated stores.
      SmallVector<LoadInst *, 2> LI;
      bool nonload = false;
      for (auto &U : storePointer->uses()) {
        if (auto L = dyn_cast<LoadInst>(U.getUser())) {
          LI.push_back(L);
        } else
          nonload = true;
      }
      if (!nonload) {
        for (auto L : LI) {
          if (DT.dominates(res, L)) {
            L->replaceAllUsesWith(res);
            L->eraseFromParent();
          }
        }
      }
    }
    if (auto II = dyn_cast<InvokeInst>(res)) {
      B.SetInsertPoint(II->getNormalDest()->getFirstNonPHI());
    } else {
      B.SetInsertPoint(res->getNextNode());
    }
    B.CreateStore(res, storePointer);
  }
  for (auto Bound : OMPBounds) {
    for (int i = 4; i <= 6; i++) {
      auto AI = cast<AllocaInst>(Bound->getArgOperand(i));
      IRBuilder<> B(AI);
      auto AI2 = B.CreateAlloca(AI->getAllocatedType(), nullptr,
                                AI->getName() + "_smpl");
      B.SetInsertPoint(Bound);
      B.CreateStore(B.CreateLoad(AI->getAllocatedType(), AI), AI2);
      Bound->setArgOperand(i, AI2);
      if (auto II = dyn_cast<InvokeInst>(Bound)) {
        B.SetInsertPoint(II->getNormalDest()->getFirstNonPHI());
      } else {
        B.SetInsertPoint(Bound->getNextNode());
      }
      B.CreateStore(B.CreateLoad(AI2->getAllocatedType(), AI2), AI);
      Bound->addParamAttr(i, Attribute::NoCapture);
    }
  }
  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  FAM.invalidate(NewF, PA);
}

/// Perform recursive inlinining on NewF up to the given limit
static void ForceRecursiveInlining(Function *NewF, size_t Limit) {
  std::map<const Function *, RecurType> RecurResults;
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
          if (CI->getCalledFunction()->getName().startswith("enzyme_wrapmpi$$"))
            continue;
          if (CI->getCalledFunction()->hasFnAttribute(
                  Attribute::ReturnsTwice) ||
              CI->getCalledFunction()->hasFnAttribute(Attribute::NoInline))
            continue;
          if (IsFunctionRecursive(CI->getCalledFunction(), RecurResults)) {
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

void CanonicalizeLoops(Function *F, FunctionAnalysisManager &FAM) {
  LoopSimplifyPass().run(*F, FAM);
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(*F);
  TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*F);
  MustExitScalarEvolution SE(*F, TLI, AC, DT, LI);
  for (Loop *L : LI.getLoopsInPreorder()) {
    auto pair =
        InsertNewCanonicalIV(L, Type::getInt64Ty(F->getContext()), "iv");
    PHINode *CanonicalIV = pair.first;
    assert(CanonicalIV);
    RemoveRedundantIVs(
        L->getHeader(), CanonicalIV, pair.second, SE,
        [&](Instruction *I, Value *V) { I->replaceAllUsesWith(V); },
        [&](Instruction *I) { I->eraseFromParent(); });
  }
  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserve<PostDominatorTreeAnalysis>();
  PA.preserve<TypeBasedAA>();
  PA.preserve<BasicAA>();
  PA.preserve<ScopedNoAliasAA>();
  FAM.invalidate(*F, PA);
}

void RemoveRedundantPHI(Function *F, FunctionAnalysisManager &FAM) {
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  for (BasicBlock &BB : *F) {
    for (BasicBlock::iterator II = BB.begin(); isa<PHINode>(II);) {
      PHINode *PN = cast<PHINode>(II);
      ++II;
      SmallPtrSet<Value *, 2> vals;
      SmallPtrSet<PHINode *, 2> done;
      SmallVector<PHINode *, 2> todo = {PN};
      while (todo.size() > 0) {
        PHINode *N = todo.back();
        todo.pop_back();
        if (done.count(N))
          continue;
        done.insert(N);
        if (vals.size() == 0 && todo.size() == 0 && PN != N &&
            DT.dominates(N, PN)) {
          vals.insert(N);
          break;
        }
        for (auto &v : N->incoming_values()) {
          if (isa<UndefValue>(v))
            continue;
          if (auto NN = dyn_cast<PHINode>(v)) {
            todo.push_back(NN);
            continue;
          }
          vals.insert(v);
          if (vals.size() > 1)
            break;
        }
        if (vals.size() > 1)
          break;
      }
      if (vals.size() == 1) {
        auto V = *vals.begin();
        if (!isa<Instruction>(V) || DT.dominates(cast<Instruction>(V), PN)) {
          PN->replaceAllUsesWith(V);
          PN->eraseFromParent();
        }
      }
    }
  }
}

PreProcessCache::PreProcessCache() {
  // Explicitly chose AA passes that are stateless
  // and will not be invalidated
  FAM.registerPass([] { return TypeBasedAA(); });
  FAM.registerPass([] { return BasicAA(); });
  MAM.registerPass([] { return GlobalsAA(); });
  // CallGraphAnalysis required for GlobalsAA
  MAM.registerPass([] { return CallGraphAnalysis(); });

  FAM.registerPass([] { return ScopedNoAliasAA(); });

  // SCEVAA causes some breakage/segfaults
  // disable for now, consider enabling in future
  // FAM.registerPass([] { return SCEVAA(); });

#if LLVM_VERSION_MAJOR < 16
  if (EnzymeAggressiveAA)
    FAM.registerPass([] { return CFLSteensAA(); });
#endif

  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  LAM.registerPass([&] { return FunctionAnalysisManagerLoopProxy(FAM); });
  FAM.registerPass([&] { return LoopAnalysisManagerFunctionProxy(LAM); });

  FAM.registerPass([] {
    auto AM = AAManager();
    AM.registerFunctionAnalysis<BasicAA>();
    AM.registerFunctionAnalysis<TypeBasedAA>();
    AM.registerModuleAnalysis<GlobalsAA>();
    AM.registerFunctionAnalysis<ScopedNoAliasAA>();

    // broken for different reasons
    // AM.registerFunctionAnalysis<SCEVAA>();

#if LLVM_VERSION_MAJOR < 16
    if (EnzymeAggressiveAA)
      AM.registerFunctionAnalysis<CFLSteensAA>();
#endif

    return AM;
  });

  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
}

llvm::AAResults &
PreProcessCache::getAAResultsFromFunction(llvm::Function *NewF) {
  return FAM.getResult<AAManager>(*NewF);
}

void setFullWillReturn(Function *NewF) {
  for (auto &BB : *NewF) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
#if LLVM_VERSION_MAJOR >= 14
        CI->addFnAttr(Attribute::WillReturn);
        CI->addFnAttr(Attribute::MustProgress);
#elif LLVM_VERSION_MAJOR >= 12
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::WillReturn);
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::MustProgress);
#else
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::WillReturn);
#endif
      }
      if (auto CI = dyn_cast<InvokeInst>(&I)) {
#if LLVM_VERSION_MAJOR >= 14
        CI->addFnAttr(Attribute::WillReturn);
        CI->addFnAttr(Attribute::MustProgress);
#elif LLVM_VERSION_MAJOR >= 12
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::WillReturn);
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::MustProgress);
#else
        CI->addAttribute(AttributeList::FunctionIndex, Attribute::WillReturn);
#endif
      }
    }
  }
}

Function *PreProcessCache::preprocessForClone(Function *F,
                                              DerivativeMode mode) {

  if (mode == DerivativeMode::ReverseModeGradient)
    mode = DerivativeMode::ReverseModePrimal;
  if (mode == DerivativeMode::ForwardModeSplit)
    mode = DerivativeMode::ReverseModePrimal;

  // If we've already processed this, return the previous version
  // and derive aliasing information
  if (cache.find(std::make_pair(F, mode)) != cache.end()) {
    Function *NewF = cache[std::make_pair(F, mode)];
    return NewF;
  }

  Function *NewF =
      Function::Create(F->getFunctionType(), F->getLinkage(),
                       "preprocess_" + F->getName(), F->getParent());

  ValueToValueMapTy VMap;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    VMap[i] = j;
    j->setName(i->getName());
    if (EnzymeNoAlias && j->getType()->isPointerTy()) {
      j->addAttr(Attribute::NoAlias);
    }
    ++i;
    ++j;
  }

  SmallVector<ReturnInst *, 4> Returns;

#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(
      NewF, F, VMap,
      /*ModuleLevelChanges*/ CloneFunctionChangeType::LocalChangesOnly, Returns,
      "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap,
                    /*ModuleLevelChanges*/ F->getSubprogram() != nullptr,
                    Returns, "", nullptr);
#endif
  CloneOrigin[NewF] = F;
  NewF->setAttributes(F->getAttributes());
  if (EnzymeNoAlias)
    for (auto j = NewF->arg_begin(); j != NewF->arg_end(); j++) {
      if (j->getType()->isPointerTy()) {
        j->addAttr(Attribute::NoAlias);
      }
    }
#if LLVM_VERSION_MAJOR >= 14
  NewF->addFnAttr(Attribute::WillReturn);
  NewF->addFnAttr(Attribute::MustProgress);
#else
  NewF->addAttribute(AttributeList::FunctionIndex, Attribute::WillReturn);
#if LLVM_VERSION_MAJOR >= 12
  NewF->addAttribute(AttributeList::FunctionIndex, Attribute::MustProgress);
#endif
#endif
  setFullWillReturn(NewF);

  if (EnzymePreopt) {
    if (EnzymeInline) {
      ForceRecursiveInlining(NewF, /*Limit*/ EnzymeInlineCount);
      setFullWillReturn(NewF);
      PreservedAnalyses PA;
      FAM.invalidate(*NewF, PA);
    }
  }

  {
    SmallVector<CallInst *, 4> ItersToErase;
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
              if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
                called = fn;
            }
          }

          if (called && called->getName() == "__enzyme_iter") {
            ItersToErase.push_back(CI);
          }
        }
      }
    }
    for (auto Call : ItersToErase) {
      IRBuilder<> B(Call);
      Call->setArgOperand(
          0, B.CreateAdd(Call->getArgOperand(0), Call->getArgOperand(1)));
    }
  }

  // Assume allocations do not return null
  {
    TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*F);
    SmallVector<Instruction *, 4> CmpsToErase;
    SmallVector<BasicBlock *, 4> BranchesToErase;
    for (auto &BB : *NewF) {
      for (auto &I : BB) {
        if (auto IC = dyn_cast<ICmpInst>(&I)) {
          if (!IC->isEquality())
            continue;
          for (int i = 0; i < 2; i++) {
            if (isa<ConstantPointerNull>(IC->getOperand(1 - i)))
              if (isAllocationCall(IC->getOperand(i), TLI)) {
                for (auto U : IC->users()) {
                  if (auto BI = dyn_cast<BranchInst>(U))
                    BranchesToErase.push_back(BI->getParent());
                }
                IC->replaceAllUsesWith(
                    IC->getPredicate() == ICmpInst::ICMP_NE
                        ? ConstantInt::getTrue(I.getContext())
                        : ConstantInt::getFalse(I.getContext()));
                CmpsToErase.push_back(&I);
                break;
              }
          }
        }
      }
    }
    for (auto I : CmpsToErase)
      I->eraseFromParent();
    for (auto BE : BranchesToErase)
      ConstantFoldTerminator(BE);
  }

  SimplifyMPIQueries<CallInst>(*NewF, FAM);
  SimplifyMPIQueries<InvokeInst>(*NewF, FAM);
  {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  if (EnzymeLowerGlobals) {
    SmallVector<CallInst *, 4> Calls;
    SmallVector<ReturnInst *, 4> Returns;
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

    // TODO consider using TBAA and globals as well
    // instead of just BasicAA
    AAResults AA2(FAM.getResult<TargetLibraryAnalysis>(*NewF));
    AA2.addAAResult(FAM.getResult<BasicAA>(*NewF));
    AA2.addAAResult(FAM.getResult<TypeBasedAA>(*NewF));
    AA2.addAAResult(FAM.getResult<ScopedNoAliasAA>(*NewF));

    for (auto &g : NewF->getParent()->globals()) {
      bool inF = false;
      {
        std::set<Constant *> seen;
        std::deque<Constant *> todo = {(Constant *)&g};
        while (todo.size()) {
          auto GV = todo.front();
          todo.pop_front();
          if (!seen.insert(GV).second)
            continue;
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
      }
    doneF:;
      if (inF) {
        bool activeCall = false;
        bool hasWrite = false;
        MemoryLocation
#if LLVM_VERSION_MAJOR >= 12
            Loc = MemoryLocation(&g, LocationSize::beforeOrAfterPointer());
#else
            Loc = MemoryLocation(&g, LocationSize::unknown());
#endif

        for (CallInst *CI : Calls) {
          if (isa<IntrinsicInst>(CI))
            continue;
          Function *F = CI->getCalledFunction();
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
          if (F && isMemFreeLibMFunction(F->getName())) {
            continue;
          }
          if (F && F->getName().contains("__enzyme_integer")) {
            continue;
          }
          if (F && F->getName().contains("__enzyme_pointer")) {
            continue;
          }
          if (F && F->getName().contains("__enzyme_float")) {
            continue;
          }
          if (F && F->getName().contains("__enzyme_double")) {
            continue;
          }
          if (F && (F->getName().startswith("f90io") ||
                    F->getName() == "ftnio_fmt_write64" ||
                    F->getName() == "__mth_i_ipowi" ||
                    F->getName() == "f90_pausea")) {
            continue;
          }
          if (llvm::isModOrRefSet(AA2.getModRefInfo(CI, Loc))) {
            llvm::errs() << " failed to inline global: " << g << " due to "
                         << *CI << "\n";
            activeCall = true;
            break;
          }
        }

        if (!activeCall) {
          std::set<Value *> seen;
          std::deque<Value *> todo = {(Value *)&g};
          while (todo.size()) {
            auto GV = todo.front();
            todo.pop_front();
            if (!seen.insert(GV).second)
              continue;
            for (auto u : GV->users()) {
              if (isa<Constant>(u) || isa<GetElementPtrInst>(u) ||
                  isa<CastInst>(u) || isa<LoadInst>(u)) {
                todo.push_back(u);
                continue;
              }

              if (auto II = dyn_cast<IntrinsicInst>(u)) {
                if (isIntelSubscriptIntrinsic(*II)) {
                  todo.push_back(u);
                  continue;
                }
              }

              if (auto CI = dyn_cast<CallInst>(u)) {
                Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
                if (auto castinst =
                        dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
                if (auto castinst =
                        dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
                {
                  if (castinst->isCast())
                    if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                      F = fn;
                    }
                }
                if (F && isMemFreeLibMFunction(F->getName())) {
                  continue;
                }
                if (F && F->getName().contains("__enzyme_integer")) {
                  continue;
                }
                if (F && F->getName().contains("__enzyme_pointer")) {
                  continue;
                }
                if (F && F->getName().contains("__enzyme_float")) {
                  continue;
                }
                if (F && F->getName().contains("__enzyme_double")) {
                  continue;
                }
                if (F && (F->getName().startswith("f90io") ||
                          F->getName() == "ftnio_fmt_write64" ||
                          F->getName() == "__mth_i_ipowi" ||
                          F->getName() == "f90_pausea")) {
                  continue;
                }

                if (couldFunctionArgumentCapture(CI, GV)) {
                  hasWrite = true;
                  goto endCheck;
                }

                if (llvm::isModSet(AA2.getModRefInfo(CI, Loc))) {
                  hasWrite = true;
                  goto endCheck;
                }
              }

              else if (auto I = dyn_cast<Instruction>(u)) {
                if (llvm::isModSet(AA2.getModRefInfo(I, Loc))) {
                  hasWrite = true;
                  goto endCheck;
                }
              }
            }
          }
        }

      endCheck:;
        if (!activeCall && hasWrite) {
          IRBuilder<> bb(&NewF->getEntryBlock(), NewF->getEntryBlock().begin());
          AllocaInst *antialloca = bb.CreateAlloca(
              g.getValueType(), g.getType()->getPointerAddressSpace(), nullptr,
              g.getName() + "_local");

          if (g.getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
            antialloca->setAlignment(Align(g.getAlignment()));
#else
            antialloca->setAlignment(g.getAlignment());
#endif
          }

          std::map<Constant *, Value *> remap;
          remap[&g] = antialloca;

          std::deque<Constant *> todo = {&g};
          while (todo.size()) {
            auto GV = todo.front();
            todo.pop_front();
            if (&g != GV && remap.find(GV) != remap.end())
              continue;
            Value *replaced = nullptr;
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

            SmallVector<std::pair<Instruction *, size_t>, 4> uses;
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
            for (auto &U : uses) {
              U.first->setOperand(U.second, replaced);
            }
          }

          Value *args[] = {
              bb.CreateBitCast(antialloca, Type::getInt8PtrTy(g.getContext())),
              bb.CreateBitCast(&g, Type::getInt8PtrTy(g.getContext())),
              ConstantInt::get(
                  Type::getInt64Ty(g.getContext()),
                  g.getParent()->getDataLayout().getTypeAllocSizeInBits(
                      g.getValueType()) /
                      8),
              ConstantInt::getFalse(g.getContext())};

          Type *tys[] = {args[0]->getType(), args[1]->getType(),
                         args[2]->getType()};
          auto intr =
              Intrinsic::getDeclaration(g.getParent(), Intrinsic::memcpy, tys);
          {

            auto cal = bb.CreateCall(intr, args);
            if (g.getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
              cal->addParamAttr(
                  0, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
              cal->addParamAttr(
                  1, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
#else
              cal->addParamAttr(0, Attribute::getWithAlignment(
                                       g.getContext(), g.getAlignment()));
              cal->addParamAttr(1, Attribute::getWithAlignment(
                                       g.getContext(), g.getAlignment()));
#endif
            }
          }

          std::swap(args[0], args[1]);

          for (ReturnInst *RI : Returns) {
            IRBuilder<> IB(RI);
            auto cal = IB.CreateCall(intr, args);
            if (g.getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
              cal->addParamAttr(
                  0, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
              cal->addParamAttr(
                  1, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
#else
              cal->addParamAttr(0, Attribute::getWithAlignment(
                                       g.getContext(), g.getAlignment()));
              cal->addParamAttr(1, Attribute::getWithAlignment(
                                       g.getContext(), g.getAlignment()));
#endif
            }
          }
        }
      }
    }

#if LLVM_VERSION_MAJOR < 14
    using OptimizationLevel = llvm::PassBuilder::OptimizationLevel;
#endif

    auto Level = OptimizationLevel::O2;

    PassBuilder PB;
#if LLVM_VERSION_MAJOR >= 12
    FunctionPassManager FPM =
        PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
#else
    FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(
        Level, PassBuilder::ThinLTOPhase::None);
#endif
    auto PA = FPM.run(*F, FAM);
    FAM.invalidate(*F, PA);
  }

  if (EnzymePreopt) {
    {
      auto PA = LowerInvokePass().run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }
    {
      auto PA = UnreachableBlockElimPass().run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }

    {
      auto PA = PromotePass().run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }

    {
#if LLVM_VERSION_MAJOR >= 16 && !defined(FLANG)
      auto PA = SROAPass(llvm::SROAOptions::ModifyCFG).run(*NewF, FAM);
#elif LLVM_VERSION_MAJOR >= 14 && !defined(FLANG)
      auto PA = SROAPass().run(*NewF, FAM);
#else
      auto PA = SROA().run(*NewF, FAM);
#endif
      FAM.invalidate(*NewF, PA);
    }

    if (mode != DerivativeMode::ForwardMode)
      ReplaceReallocs(NewF);

    {
#if LLVM_VERSION_MAJOR >= 16 && !defined(FLANG)
      auto PA = SROAPass(llvm::SROAOptions::PreserveCFG).run(*NewF, FAM);
#elif LLVM_VERSION_MAJOR >= 14 && !defined(FLANG)
      auto PA = SROAPass().run(*NewF, FAM);
#else
      auto PA = SROA().run(*NewF, FAM);
#endif
      FAM.invalidate(*NewF, PA);
    }

#if LLVM_VERSION_MAJOR >= 12
    SimplifyCFGOptions scfgo;
#else
    SimplifyCFGOptions scfgo(
        /*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false,
        /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true,
        /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
#endif
    {
      auto PA = SimplifyCFGPass(scfgo).run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }
  }

  if (mode != DerivativeMode::ForwardMode)
    ReplaceReallocs(NewF);

  if (mode == DerivativeMode::ReverseModePrimal ||
      mode == DerivativeMode::ReverseModeGradient ||
      mode == DerivativeMode::ReverseModeCombined) {
    // For subfunction calls upgrade stack allocations to mallocs
    // to ensure availability in the reverse pass
    auto unreachable = getGuaranteedUnreachable(NewF);
    UpgradeAllocasToMallocs(NewF, mode, unreachable);
  }

  CanonicalizeLoops(NewF, FAM);
  RemoveRedundantPHI(NewF, FAM);

  // Run LoopSimplifyPass to ensure preheaders exist on all loops
  {
    auto PA = LoopSimplifyPass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
  }

  {
    SmallVector<Instruction *, 4> ToErase;
    for (auto &BB : *NewF) {
      for (auto &I : BB) {
        if (auto MTI = dyn_cast<MemTransferInst>(&I)) {

          if (auto CI = dyn_cast<ConstantInt>(MTI->getOperand(2))) {
            if (CI->getValue() == 0) {
              ToErase.push_back(MTI);
            }
          }
        }
      }
    }
    for (auto E : ToErase) {
      E->eraseFromParent();
    }
    PreservedAnalyses PA;
    PA.preserve<AssumptionAnalysis>();
    PA.preserve<TargetLibraryAnalysis>();
    PA.preserve<LoopAnalysis>();
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<PostDominatorTreeAnalysis>();
    PA.preserve<TypeBasedAA>();
    PA.preserve<BasicAA>();
    PA.preserve<ScopedNoAliasAA>();
    PA.preserve<ScalarEvolutionAnalysis>();
    PA.preserve<PhiValuesAnalysis>();

    FAM.invalidate(*NewF, PA);

    if (EnzymeNameInstructions) {
      for (auto &Arg : NewF->args()) {
        if (!Arg.hasName())
          Arg.setName("arg");
      }
      for (BasicBlock &BB : *NewF) {
        if (!BB.hasName())
          BB.setName("bb");

        for (Instruction &I : BB) {
          if (!I.hasName() && !I.getType()->isVoidTy())
            I.setName("i");
        }
      }
    }
  }

  if (EnzymePHIRestructure) {
    if (false) {
    reset:;
      PreservedAnalyses PA;
      FAM.invalidate(*NewF, PA);
    }

    SmallVector<BasicBlock *, 4> MultiBlocks;
    for (auto &B : *NewF) {
      if (B.hasNPredecessorsOrMore(3))
        MultiBlocks.push_back(&B);
    }

    LoopInfo &LI = FAM.getResult<LoopAnalysis>(*NewF);
    for (BasicBlock *B : MultiBlocks) {

      // Map of function edges to list of values possible
      std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
               std::set<BasicBlock *>>
          done;
      {
        std::deque<std::tuple<
            std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
            BasicBlock *>>
            Q; // newblock, target

        for (auto P : predecessors(B)) {
          Q.emplace_back(std::make_pair(P, B), P);
        }

        for (std::tuple<
                 std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
                 BasicBlock *>
                 trace;
             Q.size() > 0;) {
          trace = Q.front();
          Q.pop_front();
          auto edge = std::get<0>(trace);
          auto block = edge.first;
          auto target = std::get<1>(trace);

          if (done[edge].count(target))
            continue;
          done[edge].insert(target);

          Loop *blockLoop = LI.getLoopFor(block);

          for (BasicBlock *Pred : predecessors(block)) {
            // Don't go up the backedge as we can use the last value if desired
            // via lcssa
            if (blockLoop && blockLoop->getHeader() == block &&
                blockLoop == LI.getLoopFor(Pred))
              continue;

            Q.push_back(
                std::tuple<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>(
                    std::make_pair(Pred, block), target));
          }
        }
      }

      SmallPtrSet<BasicBlock *, 4> Preds;
      for (auto &pair : done) {
        Preds.insert(pair.first.first);
      }

      for (auto BB : Preds) {
        bool illegal = false;
        SmallPtrSet<BasicBlock *, 2> UnionSet;
        size_t numSuc = 0;
        for (BasicBlock *sucI : successors(BB)) {
          numSuc++;
          const auto &SI = done[std::make_pair(BB, sucI)];
          if (SI.size() == 0) {
            // sucI->getName();
            illegal = true;
            break;
          }
          for (auto si : SI) {
            UnionSet.insert(si);

            for (BasicBlock *sucJ : successors(BB)) {
              if (sucI == sucJ)
                continue;
              if (done[std::make_pair(BB, sucJ)].count(si)) {
                illegal = true;
                goto endIllegal;
              }
            }
          }
        }
      endIllegal:;

        if (!illegal && numSuc > 1 && !B->hasNPredecessors(UnionSet.size())) {
          BasicBlock *Ins =
              BasicBlock::Create(BB->getContext(), "tmpblk", BB->getParent());
          IRBuilder<> Builder(Ins);
          for (auto &phi : B->phis()) {
            auto nphi = Builder.CreatePHI(phi.getType(), 2);
            SmallVector<BasicBlock *, 4> Blocks;

            for (auto blk : UnionSet) {
              nphi->addIncoming(phi.getIncomingValueForBlock(blk), blk);
              phi.removeIncomingValue(blk, /*deleteifempty*/ false);
            }

            phi.addIncoming(nphi, Ins);
          }
          Builder.CreateBr(B);
          for (auto blk : UnionSet) {
            auto term = blk->getTerminator();
            for (unsigned Idx = 0, NumSuccessors = term->getNumSuccessors();
                 Idx != NumSuccessors; ++Idx)
              if (term->getSuccessor(Idx) == B)
                term->setSuccessor(Idx, Ins);
          }
          goto reset;
        }
      }
    }
  }

  if (EnzymePrint)
    llvm::errs() << "after simplification :\n" << *NewF << "\n";

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (1)");
  }
  cache[std::make_pair(F, mode)] = NewF;
  return NewF;
}

FunctionType *getFunctionTypeForClone(
    llvm::FunctionType *FTy, DerivativeMode mode, unsigned width,
    llvm::Type *additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE returnType) {
  SmallVector<Type *, 4> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::Return) {
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(
          GradientUtils::getShadowType(FTy->getReturnType(), width));
    } else {
      RetTypes.push_back(FTy->getReturnType());
    }
  } else if (returnValue == ReturnType::ArgsWithTwoReturns ||
             returnValue == ReturnType::TwoReturns) {
    RetTypes.push_back(FTy->getReturnType());
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(
          GradientUtils::getShadowType(FTy->getReturnType(), width));
    } else {
      RetTypes.push_back(FTy->getReturnType());
    }
  }
  SmallVector<Type *, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto &I : FTy->params()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(GradientUtils::getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(GradientUtils::getShadowType(I, width));
    }
    ++argno;
  }

  if (diffeReturnArg) {
    assert(!FTy->getReturnType()->isVoidTy());
    ArgTypes.push_back(
        GradientUtils::getShadowType(FTy->getReturnType(), width));
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }
  Type *RetType = StructType::get(FTy->getContext(), RetTypes);
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.clear();
    RetTypes.push_back(getDefaultAnonymousTapeType(FTy->getContext()));
    if (returnValue == ReturnType::TapeAndTwoReturns) {
      RetTypes.push_back(FTy->getReturnType());
      RetTypes.push_back(
          GradientUtils::getShadowType(FTy->getReturnType(), width));
    } else if (returnValue == ReturnType::TapeAndReturn) {
      if (returnType != DIFFE_TYPE::CONSTANT &&
          returnType != DIFFE_TYPE::OUT_DIFF)
        RetTypes.push_back(
            GradientUtils::getShadowType(FTy->getReturnType(), width));
      else
        RetTypes.push_back(FTy->getReturnType());
    }
    RetType = StructType::get(FTy->getContext(), RetTypes);
  } else if (returnValue == ReturnType::Return) {
    assert(RetTypes.size() == 1);
    RetType = RetTypes[0];
  } else if (returnValue == ReturnType::TwoReturns) {
    assert(RetTypes.size() == 2);
  }

  bool noReturn = RetTypes.size() == 0;
  if (noReturn)
    RetType = Type::getVoidTy(RetType->getContext());

  // Create a new function type...
  return FunctionType::get(RetType, ArgTypes, FTy->isVarArg());
}

Function *PreProcessCache::CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, Function *&F,
    ValueToValueMapTy &ptrInputs, ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<Value *> &constants, SmallPtrSetImpl<Value *> &nonconstant,
    SmallPtrSetImpl<Value *> &returnvals, ReturnType returnValue,
    DIFFE_TYPE returnType, const Twine &name,
    llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> *VMapO,
    bool diffeReturnArg, llvm::Type *additionalArg) {
  assert(!F->empty());
  F = preprocessForClone(F, mode);
  llvm::ValueToValueMapTy VMap;
  llvm::FunctionType *FTy = getFunctionTypeForClone(
      F->getFunctionType(), mode, width, additionalArg, constant_args,
      diffeReturnArg, returnValue, returnType);

  for (BasicBlock &BB : *F) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (auto rv = ri->getReturnValue()) {
        returnvals.insert(rv);
      }
    }
  }

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
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif
  CloneOrigin[NewF] = F;
  if (VMapO) {
    for (const auto &data : VMap)
      VMapO->insert(std::pair<const llvm::Value *, AssertingReplacingVH>(
          data.first, (llvm::Value *)data.second));
    VMapO->getMDMap() = VMap.getMDMap();
  }

  bool hasPtrInput = false;
  unsigned ii = 0, jj = 0;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    if (F->hasParamAttribute(ii, Attribute::StructRet)) {
      NewF->addParamAttr(jj, Attribute::get(F->getContext(), "enzyme_sret"));
#if LLVM_VERSION_MAJOR >= 13
      // TODO
      // NewF->addParamAttr(
      //    jj,
      //    Attribute::get(
      //        F->getContext(), Attribute::AttrKind::ElementType,
      //        F->getParamAttribute(ii,
      //        Attribute::StructRet).getValueAsType()));
#endif
    }
    if (F->getAttributes().hasParamAttr(ii, "enzymejl_returnRoots")) {
      NewF->addParamAttr(
          jj, F->getAttributes().getParamAttr(ii, "enzymejl_returnRoots"));
#if LLVM_VERSION_MAJOR >= 13
      // TODO
      // NewF->addParamAttr(jj, F->getParamAttribute(ii,
      // Attribute::ElementType));
#endif
    }
    for (auto ty : PrimalParamAttrsToPreserve)
      if (F->getAttributes().hasParamAttr(ii, ty)) {
        auto attr = F->getAttributes().getParamAttr(ii, ty);
        NewF->addParamAttr(jj, attr);
      }
    if (constant_args[ii] == DIFFE_TYPE::CONSTANT) {
      if (!i->hasByValAttr())
        constants.insert(i);
      if (EnzymePrintActivity)
        llvm::errs() << "in new function " << NewF->getName()
                     << " constant arg " << *j << "\n";
    } else {
      nonconstant.insert(i);
      if (EnzymePrintActivity)
        llvm::errs() << "in new function " << NewF->getName()
                     << " nonconstant arg " << *j << "\n";
    }

    // Always remove nonnull/noundef since the caller may choose to pass undef
    // as an arg if provably it will not be used in the reverse pass
    if (constant_args[ii] == DIFFE_TYPE::DUP_NONEED ||
        mode == DerivativeMode::ReverseModeGradient) {
      if (F->hasParamAttribute(ii, Attribute::NonNull)) {
        NewF->removeParamAttr(jj, Attribute::NonNull);
      }
#if LLVM_VERSION_MAJOR >= 11
      if (F->hasParamAttribute(ii, Attribute::NoUndef)) {
        NewF->removeParamAttr(jj, Attribute::NoUndef);
      }
#endif
    }

    if (constant_args[ii] == DIFFE_TYPE::DUP_ARG ||
        constant_args[ii] == DIFFE_TYPE::DUP_NONEED) {
      hasPtrInput = true;
      ptrInputs[i] = (j + 1);
      // TODO: find a way to keep the attributes in vector mode.
      if (width == 1)
        for (auto ty : ShadowParamAttrsToPreserve)
          if (F->getAttributes().hasParamAttr(ii, ty)) {
            auto attr = F->getAttributes().getParamAttr(ii, ty);
            NewF->addParamAttr(jj + 1, attr);
          }

      if (F->getAttributes().hasParamAttr(ii, "enzymejl_returnRoots")) {
        if (width == 1) {
          NewF->addParamAttr(jj + 1, F->getAttributes().getParamAttr(
                                         ii, "enzymejl_returnRoots"));
        } else {
          NewF->addParamAttr(jj + 1, Attribute::get(F->getContext(),
                                                    "enzymejl_returnRoots_v"));
        }
#if LLVM_VERSION_MAJOR >= 13
        // TODO
        // NewF->addParamAttr(jj + 1,
        //                   F->getParamAttribute(ii, Attribute::ElementType));
#endif
      }

      if (F->hasParamAttribute(ii, Attribute::StructRet)) {
        if (width == 1) {
#if LLVM_VERSION_MAJOR >= 12
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret"));
#else
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret"));
#endif
#if LLVM_VERSION_MAJOR >= 13
          // TODO
          // NewF->addParamAttr(
          //     jj + 1,
          //     Attribute::get(F->getContext(),
          //     Attribute::AttrKind::ElementType,
          //                    F->getParamAttribute(ii, Attribute::StructRet)
          //                        .getValueAsType()));
#endif
        } else {
#if LLVM_VERSION_MAJOR >= 12
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret_v"));
#else
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret_v"));
#endif
#if LLVM_VERSION_MAJOR >= 13
          // TODO
          // NewF->addParamAttr(
          //     jj + 1,
          //     Attribute::get(F->getContext(),
          //     Attribute::AttrKind::ElementType,
          //                    F->getParamAttribute(ii, Attribute::StructRet)
          //                        .getValueAsType()));
#endif
        }
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

  if (hasPtrInput && (mode == DerivativeMode::ReverseModeCombined ||
                      mode == DerivativeMode::ReverseModeGradient)) {
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
      NewF->removeFnAttr(Attribute::ReadOnly);
    }
#if LLVM_VERSION_MAJOR >= 16
    auto eff = NewF->getMemoryEffects();
    for (auto loc : MemoryEffects::locations()) {
      if (loc == MemoryEffects::Location::InaccessibleMem)
        continue;
      auto mr = eff.getModRef(loc);
      if (isModSet(mr))
        eff |= MemoryEffects(loc, ModRefInfo::Ref);
      if (isRefSet(mr))
        eff |= MemoryEffects(loc, ModRefInfo::Mod);
    }
    NewF->setMemoryEffects(eff);
#endif
  }
  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
  assert(NewF->hasLocalLinkage());

  return NewF;
}

void CoaleseTrivialMallocs(Function &F, DominatorTree &DT) {
  std::map<BasicBlock *, std::vector<std::pair<CallInst *, CallInst *>>>
      LegalMallocs;

  std::map<Metadata *, std::vector<CallInst *>> frees;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (auto F2 = CI->getCalledFunction()) {
          if (F2->getName() == "free") {
            if (auto MD = hasMetadata(CI, "enzyme_cache_free")) {
              Metadata *op = MD->getOperand(0);
              frees[op].push_back(CI);
            }
          }
        }
      }
    }
  }

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "malloc") {
            CallInst *freeCall = nullptr;
            for (auto U : CI->users()) {
              if (auto CI2 = dyn_cast<CallInst>(U)) {
                if (auto F2 = CI2->getCalledFunction()) {
                  if (F2->getName() == "free") {
                    if (DT.dominates(CI, CI2)) {
                      freeCall = CI2;
                      break;
                    }
                  }
                }
              }
            }
            if (!freeCall) {
              if (auto MD = hasMetadata(CI, "enzyme_cache_alloc")) {
                Metadata *op = MD->getOperand(0);
                if (frees[op].size() == 1)
                  freeCall = frees[op][0];
              }
            }
            if (freeCall)
              LegalMallocs[&BB].emplace_back(CI, freeCall);
          }
        }
      }
    }
  }
  for (auto &pair : LegalMallocs) {
    if (pair.second.size() < 2)
      continue;
    CallInst *First = pair.second[0].first;
    for (auto &z : pair.second) {
      if (!DT.dominates(First, z.first))
        First = z.first;
    }
    bool legal = true;
    for (auto &z : pair.second) {
      if (auto inst = dyn_cast<Instruction>(z.first->getArgOperand(0)))
        if (!DT.dominates(inst, First))
          legal = true;
    }
    if (!legal)
      continue;
    IRBuilder<> B(First);
    Value *Size = First->getArgOperand(0);
    for (auto &z : pair.second) {
      if (z.first == First)
        continue;
      Size = B.CreateAdd(
          B.CreateOr(B.CreateSub(Size, ConstantInt::get(Size->getType(), 1)),
                     ConstantInt::get(Size->getType(), 15)),
          ConstantInt::get(Size->getType(), 1));
      z.second->eraseFromParent();
      IRBuilder<> B2(z.first);
      Value *gepPtr = B2.CreateInBoundsGEP(Type::getInt8Ty(First->getContext()),
                                           First, Size);
      z.first->replaceAllUsesWith(gepPtr);
      Size = B.CreateAdd(Size, z.first->getArgOperand(0));
      z.first->eraseFromParent();
    }
    auto NewMalloc =
        cast<CallInst>(B.CreateCall(First->getCalledFunction(), Size));
    NewMalloc->copyIRFlags(First);
    NewMalloc->setMetadata("enzyme_cache_alloc",
                           hasMetadata(First, "enzyme_cache_alloc"));
    First->replaceAllUsesWith(NewMalloc);
    First->eraseFromParent();
  }
}

void SelectOptimization(Function *F) {
  DominatorTree DT(*F);
  for (auto &BB : *F) {
    if (auto BI = dyn_cast<BranchInst>(BB.getTerminator())) {
      if (BI->isConditional()) {
        for (auto &I : BB) {
          if (auto SI = dyn_cast<SelectInst>(&I)) {
            if (SI->getCondition() == BI->getCondition()) {
              for (Value::use_iterator UI = SI->use_begin(), E = SI->use_end();
                   UI != E;) {
                Use &U = *UI;
                ++UI;
                if (DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(0)), U))
                  U.set(SI->getTrueValue());
                else if (DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(1)),
                                      U))
                  U.set(SI->getFalseValue());
              }
            }
          }
        }
      }
    }
  }
}

void ReplaceFunctionImplementation(Module &M) {
  for (Function &Impl : M) {
    for (auto attr : {"implements", "implements2"}) {
      if (!Impl.hasFnAttribute(attr))
        continue;
      const Attribute &A = Impl.getFnAttribute(attr);

      const StringRef SpecificationName = A.getValueAsString();
      Function *Specification = M.getFunction(SpecificationName);
      if (!Specification) {
        LLVM_DEBUG(dbgs() << "Found implementation '" << Impl.getName()
                          << "' but no matching specification with name '"
                          << SpecificationName
                          << "', potentially inlined and/or eliminated.\n");
        continue;
      }
      LLVM_DEBUG(dbgs() << "Replace specification '" << Specification->getName()
                        << "' with implementation '" << Impl.getName()
                        << "'\n");

      for (auto I = Specification->use_begin(), UE = Specification->use_end();
           I != UE;) {
        auto &use = *I;
        ++I;
        auto cext = ConstantExpr::getBitCast(&Impl, Specification->getType());
        if (cast<Instruction>(use.getUser())->getParent()->getParent() == &Impl)
          continue;
        use.set(cext);
        if (auto CI = dyn_cast<CallInst>(use.getUser())) {
#if LLVM_VERSION_MAJOR >= 11
          if (CI->getCalledOperand() == cext ||
              CI->getCalledFunction() == &Impl)
#else
          if (CI->getCalledValue() == cext || CI->getCalledFunction() == &Impl)
#endif
          {
            CI->setCallingConv(Impl.getCallingConv());
          }
        }
      }
    }
  }
}

void PreProcessCache::optimizeIntermediate(Function *F) {
  PromotePass().run(*F, FAM);
#if LLVM_VERSION_MAJOR >= 14 && !defined(FLANG)
  GVNPass().run(*F, FAM);
#else
  GVN().run(*F, FAM);
#endif
#if LLVM_VERSION_MAJOR >= 16 && !defined(FLANG)
  SROAPass(llvm::SROAOptions::PreserveCFG).run(*F, FAM);
#elif LLVM_VERSION_MAJOR >= 14 && !defined(FLANG)
  SROAPass().run(*F, FAM);
#else
  SROA().run(*F, FAM);
#endif

  if (EnzymeSelectOpt) {
#if LLVM_VERSION_MAJOR >= 12
    SimplifyCFGOptions scfgo;
#else
    SimplifyCFGOptions scfgo(
        /*unsigned BonusThreshold=*/1, /*bool ForwardSwitchCond=*/false,
        /*bool SwitchToLookup=*/false, /*bool CanonicalLoops=*/true,
        /*bool SinkCommon=*/true, /*AssumptionCache *AssumpCache=*/nullptr);
#endif
    SimplifyCFGPass(scfgo).run(*F, FAM);
    CorrelatedValuePropagationPass().run(*F, FAM);
    SelectOptimization(F);
  }
  // EarlyCSEPass(/*memoryssa*/ true).run(*F, FAM);

  if (EnzymeCoalese)
    CoaleseTrivialMallocs(*F, FAM.getResult<DominatorTreeAnalysis>(*F));

  ReplaceFunctionImplementation(*F->getParent());

  PreservedAnalyses PA;
  FAM.invalidate(*F, PA);

#if LLVM_VERSION_MAJOR < 14
  using OptimizationLevel = llvm::PassBuilder::OptimizationLevel;
#endif
  OptimizationLevel Level = OptimizationLevel::O0;

  switch (EnzymePostOptLevel) {
  default:
  case 0:
    Level = OptimizationLevel::O0;
    break;
  case 1:
    Level = OptimizationLevel::O1;
    break;
  case 2:
    Level = OptimizationLevel::O2;
    break;
  case 3:
    Level = OptimizationLevel::O3;
    break;
  }
  if (Level != OptimizationLevel::O0) {
    PassBuilder PB;
#if LLVM_VERSION_MAJOR >= 12
    FunctionPassManager FPM =
        PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
#else
    FunctionPassManager FPM = PB.buildFunctionSimplificationPipeline(
        Level, PassBuilder::ThinLTOPhase::None);
#endif
    PA = FPM.run(*F, FAM);
    FAM.invalidate(*F, PA);
  }

  // TODO actually run post optimizations.
}

void PreProcessCache::clear() {
  LAM.clear();
  FAM.clear();
  MAM.clear();
  cache.clear();
}

bool LowerSparsification(llvm::Function *F, bool replaceAll) {
  auto &DL = F->getParent()->getDataLayout();
  bool changed = false;
  SmallVector<CallInst *, 1> todo;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (getFuncNameFromCall(CI).contains("__enzyme_todense")) {
          todo.push_back(CI);
        }
      }
    }
  }
  for (auto CI : todo) {
    changed = true;
    auto load_fn = cast<Function>(getBaseObject(CI->getArgOperand(0)));
    auto store_fn = cast<Function>(getBaseObject(CI->getArgOperand(1)));
    size_t argstart = 2;
#if LLVM_VERSION_MAJOR >= 14
    size_t num_args = CI->arg_size();
#else
    size_t num_args = CI->getNumArgOperands();
#endif
    SmallVector<std::pair<Instruction *, Value *>, 1> users;

    for (auto U : CI->users()) {
      users.push_back(std::make_pair(cast<Instruction>(U), CI));
    }
    IntegerType *intTy = IntegerType::get(CI->getContext(), 64);
    auto toInt = [&](IRBuilder<> &B, llvm::Value *V) {
      if (auto PT = dyn_cast<PointerType>(V->getType())) {
        if (PT->getAddressSpace() != 0) {
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
          if (CI->getContext().supportsTypedPointers()) {
#endif
            V = B.CreateAddrSpaceCast(
                V, PointerType::getUnqual(PT->getPointerElementType()));
#if LLVM_VERSION_MAJOR >= 15
          } else {
            V = B.CreateAddrSpaceCast(V,
                                      PointerType::getUnqual(PT->getContext()));
          }
#endif
#else
          V = B.CreateAddrSpaceCast(V,
                                    PointerType::getUnqual(PT->getContext()));
#endif
        }
        return B.CreatePtrToInt(V, intTy);
      }
      auto IT = cast<IntegerType>(V->getType());
      if (IT == intTy)
        return V;
      return B.CreateZExtOrTrunc(V, intTy);
    };
    SmallVector<Instruction *, 1> toErase;

    ValueToValueMapTy replacements;
    replacements[CI] = Constant::getNullValue(CI->getType());
    Instruction *remaining = nullptr;
    while (users.size()) {
      auto pair = users.back();
      users.pop_back();
      auto U = pair.first;
      auto val = pair.second;
      if (replacements.count(U))
        continue;

      IRBuilder B(U);
      if (auto CI = dyn_cast<CastInst>(U)) {
        for (auto U : CI->users()) {
          users.push_back(std::make_pair(cast<Instruction>(U), CI));
        }
        auto rep =
            B.CreateCast(CI->getOpcode(), replacements[val], CI->getDestTy());
        if (auto I = dyn_cast<Instruction>(rep))
          I->setDebugLoc(CI->getDebugLoc());
        replacements[CI] = rep;
        continue;
      }
      /*
      if (auto CI = dyn_cast<PHINode>(U)) {
        for (auto U : CI->users()) {
          users.push_back(std::make_pair(cast<Instruction>(U), CI));
        }
        continue;
      }
      */
      if (auto CI = dyn_cast<CallInst>(U)) {
        auto funcName = getFuncNameFromCall(CI);
        if (funcName == "julia.pointer_from_objref") {
          for (auto U : CI->users()) {
            users.push_back(std::make_pair(cast<Instruction>(U), CI));
          }
#if LLVM_VERSION_MAJOR >= 11
          auto *F = CI->getCalledOperand();
#else
          auto *F = CI->getCalledValue();
#endif

          SmallVector<Value *, 1> args;
#if LLVM_VERSION_MAJOR >= 14
          for (auto &arg : CI->args())
#else
          for (auto &arg : CI->arg_operands())
#endif
            args.push_back(replacements[arg]);

          auto FT = CI->getFunctionType();

          auto cal = cast<CallInst>(B.CreateCall(FT, F, args));
          cal->setCallingConv(CI->getCallingConv());
          cal->setDebugLoc(CI->getDebugLoc());
          replacements[CI] = cal;
          continue;
        }
      }
      if (auto CI = dyn_cast<GetElementPtrInst>(U)) {
        for (auto U : CI->users()) {
          users.push_back(std::make_pair(cast<Instruction>(U), CI));
        }
        SmallVector<Value *, 1> inds;
        bool allconst = true;
        for (auto &ind : CI->indices()) {
          if (!isa<ConstantInt>(ind)) {
            allconst = false;
          }
          inds.push_back(ind);
        }
        Value *gep;

        if (inds.size() == 1) {
          gep = ConstantInt::get(
              intTy,
              (DL.getTypeSizeInBits(CI->getSourceElementType()) + 7) / 8);
          gep = B.CreateMul(intTy == inds[0]->getType()
                                ? inds[0]
                                : B.CreateZExtOrTrunc(inds[0], intTy),
                            gep, "", true, true);
          gep = B.CreateAdd(B.CreatePtrToInt(replacements[val], intTy), gep);
          gep = B.CreateIntToPtr(gep, CI->getType());
        } else if (!allconst) {
          gep =
              B.CreateGEP(CI->getSourceElementType(), replacements[val], inds);
          if (auto ge = cast<GetElementPtrInst>(gep))
            ge->setIsInBounds(CI->isInBounds());
        } else {
          APInt ai(64, 0);
          CI->accumulateConstantOffset(DL, ai);
          gep = B.CreateIntToPtr(ConstantInt::get(intTy, ai), CI->getType());
        }
        if (auto I = dyn_cast<Instruction>(gep))
          I->setDebugLoc(CI->getDebugLoc());
        replacements[CI] = gep;
        continue;
      }
      if (auto LI = dyn_cast<LoadInst>(U)) {
        auto diff = toInt(B, replacements[LI->getPointerOperand()]);
        SmallVector<Value *, 2> args;
        args.push_back(diff);
        for (size_t i = argstart; i < num_args; i++)
          args.push_back(CI->getArgOperand(i));
        if (load_fn->getFunctionType()->getNumParams() != args.size()) {
          llvm::errs() << *load_fn << "\n";
        } else {
          for (size_t i = 0; i < args.size(); i++) {
            if (load_fn->getFunctionType()->getParamType(i) !=
                args[i]->getType()) {
              llvm::errs() << "i: " << i << " lfn: " << *load_fn
                           << " arg[i]: " << *args[i] << "\n";
              break;
            }
          }
        }
        CallInst *call = B.CreateCall(load_fn, args);
        call->setDebugLoc(LI->getDebugLoc());
        Value *tmp = call;
        if (tmp->getType() != LI->getType())
          tmp = B.CreateBitCast(tmp, LI->getType());
        LI->replaceAllUsesWith(tmp);

        if (load_fn->hasFnAttribute(Attribute::AlwaysInline)) {
          InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 11
          InlineFunction(*call, IFI);
#else
          InlineFunction(call, IFI);
#endif
        }
        toErase.push_back(LI);
        continue;
      }
      if (auto SI = dyn_cast<StoreInst>(U)) {
        assert(SI->getValueOperand() != val);
        auto diff = toInt(B, replacements[SI->getPointerOperand()]);
        SmallVector<Value *, 2> args;
        args.push_back(SI->getValueOperand());
        if (args[0]->getType() != store_fn->getFunctionType()->getParamType(0))
          args[0] = B.CreateBitCast(
              args[0], store_fn->getFunctionType()->getParamType(0));
        args.push_back(diff);
        for (size_t i = argstart; i < num_args; i++)
          args.push_back(CI->getArgOperand(i));
        auto call = B.CreateCall(store_fn, args);
        call->setDebugLoc(SI->getDebugLoc());
        if (load_fn->hasFnAttribute(Attribute::AlwaysInline)) {
          InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 11
          InlineFunction(*call, IFI);
#else
          InlineFunction(call, IFI);
#endif
        }
        toErase.push_back(SI);
        continue;
      }
      remaining = U;
    }
    for (auto U : toErase)
      U->eraseFromParent();

    if (!remaining) {
      CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
      CI->eraseFromParent();
    } else if (replaceAll) {
      llvm::errs() << " U: " << *remaining << " of " << *CI << "\n";
      llvm_unreachable("Illegal replacement use of enzyme sparisification");
    }
  }
  return changed;
}
