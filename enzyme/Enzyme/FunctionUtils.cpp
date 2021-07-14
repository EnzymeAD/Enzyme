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

#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/CodeGen/UnreachableBlockElim.h"

#include "llvm/Analysis/CFLSteensAliasAnalysis.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Analysis/PhiValues.h"
#endif
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#if LLVM_VERSION_MAJOR > 6
#include "llvm/Transforms/Utils.h"
#endif

#include "llvm/Transforms/Utils/Cloning.h"

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

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

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

cl::opt<bool>
    EnzymeAggressiveAA("enzyme-aggressive-aa", cl::init(false), cl::Hidden,
                       cl::desc("Use more unstable but aggressive LLVM AA"));

cl::opt<bool> EnzymeLowerGlobals(
    "enzyme-lower-globals", cl::init(false), cl::Hidden,
    cl::desc("Lower globals to locals assuming the global values are not "
             "needed outside of this gradient"));

cl::opt<int>
    EnzymeInlineCount("enzyme-inline-count", cl::init(10000), cl::Hidden,
                      cl::desc("Limit of number of functions to inline"));

cl::opt<bool> EnzymeCoalese("enzyme-coalese", cl::init(false), cl::Hidden,
                            cl::desc("Whether to coalese memory allocations"));

#if LLVM_VERSION_MAJOR >= 8
static cl::opt<bool> EnzymePHIRestructure(
    "enzyme-phi-restructure", cl::init(false), cl::Hidden,
    cl::desc("Whether to restructure phi's to have better unwrap behavior"));
#endif

cl::opt<bool>
    EnzymeNameInstructions("enzyme-name-instructions", cl::init(false),
                           cl::Hidden,
                           cl::desc("Have enzyme name all instructions"));

cl::opt<bool> EnzymeSelectOpt("enzyme-select-opt", cl::init(true), cl::Hidden,
                              cl::desc("Run Enzyme select optimization"));
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

  if (F->empty())
    return false;

  auto arg = F->arg_begin();
  for (size_t i = 0, size = CI->getNumArgOperands(); i < size; i++) {
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
static inline void UpgradeAllocasToMallocs(Function *NewF,
                                           DerivativeMode mode) {
  std::vector<AllocaInst *> ToConvert;

  for (auto &BB : *NewF) {
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
    auto rep = CallInst::CreateMalloc(
        insertBefore, i64, AI->getAllocatedType(),
        ConstantInt::get(
            i64, NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
                     AI->getAllocatedType()) /
                     8),
        IRBuilder<>(insertBefore).CreateZExtOrTrunc(AI->getArraySize(), i64),
        nullptr, nam);
    CallInst *CI = dyn_cast<CallInst>(rep);
    if (auto C = dyn_cast<CastInst>(rep))
      CI = cast<CallInst>(C->getOperand(0));
    CI->setMetadata("enzyme_fromstack", MDNode::get(CI->getContext(), {}));
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
void PreProcessCache::ReplaceReallocs(Function *NewF, bool mem2reg) {
  if (mem2reg) {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
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

  PreservedAnalyses PA;
  FAM.invalidate(*NewF, PA);

  PA = PromotePass().run(*NewF, FAM);
  FAM.invalidate(*NewF, PA);
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

  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(*F);
  AssumptionCache &AC = FAM.getResult<AssumptionAnalysis>(*F);
  TargetLibraryInfo &TLI = FAM.getResult<TargetLibraryAnalysis>(*F);
  MustExitScalarEvolution SE(*F, TLI, AC, DT, LI);
  for (Loop *L : LI) {
    auto pair =
        InsertNewCanonicalIV(L, Type::getInt64Ty(F->getContext()), "tiv");
    PHINode *CanonicalIV = pair.first;
    assert(CanonicalIV);
    RemoveRedundantIVs(
        L->getHeader(), CanonicalIV, SE,
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
  FAM.invalidate(*F, PA);
}

PreProcessCache::PreProcessCache() {
  MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
  FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });
  FAM.registerPass([] { return AssumptionAnalysis(); });
  FAM.registerPass([] { return TargetLibraryAnalysis(); });
  FAM.registerPass([] { return LoopAnalysis(); });
  FAM.registerPass([] { return DominatorTreeAnalysis(); });
#if LLVM_VERSION_MAJOR > 6
  FAM.registerPass([] { return PhiValuesAnalysis(); });
#endif

  FAM.registerPass([] { return DependenceAnalysis(); });

  // Explicitly chose AA passes that are stateless
  // and will not be invalidated
  FAM.registerPass([] { return TypeBasedAA(); });
  FAM.registerPass([] { return BasicAA(); });
  MAM.registerPass([] { return GlobalsAA(); });

  // SCEVAA causes some breakage/segfaults
  // disable for now, consider enabling in future
  // FAM.registerPass([] { return SCEVAA(); });

  if (EnzymeAggressiveAA)
    FAM.registerPass([] { return CFLSteensAA(); });

  FAM.registerPass([] {
    auto AM = AAManager();
    AM.registerFunctionAnalysis<BasicAA>();
    AM.registerFunctionAnalysis<TypeBasedAA>();
    AM.registerModuleAnalysis<GlobalsAA>();

    // broken for different reasons
    // AM.registerFunctionAnalysis<SCEVAA>();
    if (EnzymeAggressiveAA)
      AM.registerFunctionAnalysis<CFLSteensAA>();
    return AM;
  });

  // used in optimizeintermediate
  FAM.registerPass([] { return ScalarEvolutionAnalysis(); });

  FAM.registerPass([] { return TargetIRAnalysis(); });
  FAM.registerPass([] { return MemorySSAAnalysis(); });
  FAM.registerPass([] { return MemoryDependenceAnalysis(); });
  FAM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
  FAM.registerPass([] { return LazyValueAnalysis(); });
#if LLVM_VERSION_MAJOR >= 8
  FAM.registerPass([] { return PassInstrumentationAnalysis(); });
#endif

  // Used by GradientUtils
  FAM.registerPass([] { return PostDominatorTreeAnalysis(); });
}

llvm::AAResults &
PreProcessCache::getAAResultsFromFunction(llvm::Function *NewF) {
  return FAM.getResult<AAManager>(*NewF);
}

Function *PreProcessCache::preprocessForClone(Function *F,
                                              DerivativeMode mode) {

  // If we've already processed this, return the previous version
  // and derive aliasing information
  if (cache.find(std::make_pair(
          F, mode == DerivativeMode::ReverseModeCombined)) != cache.end()) {
    Function *NewF =
        cache[std::make_pair(F, mode == DerivativeMode::ReverseModeCombined)];
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

  if (EnzymePreopt) {
    if (EnzymeInline) {
      ForceRecursiveInlining(NewF, /*Limit*/ EnzymeInlineCount);
      PreservedAnalyses PA;
      FAM.invalidate(*NewF, PA);
    }
  }

  {
    std::vector<CallInst *> ItersToErase;
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

  if (EnzymeLowerGlobals) {
    std::vector<CallInst *> Calls;
    std::vector<ReturnInst *> Returns;
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
#elif LLVM_VERSION_MAJOR >= 9
            Loc = MemoryLocation(&g, LocationSize::unknown());
#else
            Loc = MemoryLocation(&g, MemoryLocation::UnknownSize);
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
          if (F && (isMemFreeLibMFunction(F->getName()) ||
                    F->getName() == "__fd_sincos_1")) {
            continue;
          }
          if (F && F->getName() == "__enzyme_integer") {
            continue;
          }
          if (F && F->getName() == "__enzyme_pointer") {
            continue;
          }
          if (F && F->getName() == "__enzyme_float") {
            continue;
          }
          if (F && F->getName() == "__enzyme_double") {
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
                if (F && (isMemFreeLibMFunction(F->getName()) ||
                          F->getName() == "__fd_sincos_1")) {
                  continue;
                }
                if (F && F->getName() == "__enzyme_integer") {
                  continue;
                }
                if (F && F->getName() == "__enzyme_pointer") {
                  continue;
                }
                if (F && F->getName() == "__enzyme_float") {
                  continue;
                }
                if (F && F->getName() == "__enzyme_double") {
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

            std::vector<std::pair<Instruction *, size_t>> uses;
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

          SmallVector<Value *, 4> args;
          args.push_back(
              bb.CreateBitCast(antialloca, Type::getInt8PtrTy(g.getContext())));
          args.push_back(
              bb.CreateBitCast(&g, Type::getInt8PtrTy(g.getContext())));
          args.push_back(ConstantInt::get(
              Type::getInt64Ty(g.getContext()),
              g.getParent()->getDataLayout().getTypeAllocSizeInBits(
                  g.getValueType()) /
                  8));
          args.push_back(ConstantInt::getFalse(g.getContext()));

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

    PassManagerBuilder Builder;
    Builder.OptLevel = 2;
    legacy::FunctionPassManager PM(NewF->getParent());
    Builder.populateFunctionPassManager(PM);
    PM.run(*NewF);
    {
      PreservedAnalyses PA;
      FAM.invalidate(*NewF, PA);
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
                if (isDeallocationFunction(
                        *fn, FAM.getResult<TargetLibraryAnalysis>(*NewF))) {
                  called = fn;
                }
              }
            }
          }

          if (called &&
              isDeallocationFunction(
                  *called, FAM.getResult<TargetLibraryAnalysis>(*NewF))) {
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
    PreservedAnalyses PA;
    PA.preserve<AssumptionAnalysis>();
    PA.preserve<TargetLibraryAnalysis>();
    PA.preserve<LoopAnalysis>();
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<PostDominatorTreeAnalysis>();
    PA.preserve<TypeBasedAA>();
    PA.preserve<BasicAA>();
    PA.preserve<ScalarEvolutionAnalysis>();
#if LLVM_VERSION_MAJOR > 6
    PA.preserve<PhiValuesAnalysis>();
#endif
    FAM.invalidate(*NewF, PA);
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
#if LLVM_VERSION_MAJOR <= 7
      auto PA = GVN().run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
#endif
    }

    {
      auto PA = SROA().run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }

    ReplaceReallocs(NewF);

    {
      auto PA = SROA().run(*NewF, FAM);
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

  ReplaceReallocs(NewF);

  // Run LoopSimplifyPass to ensure preheaders exist on all loops
  auto PA = LoopSimplifyPass().run(*NewF, FAM);
  FAM.invalidate(*NewF, PA);

  // For subfunction calls upgrade stack allocations to mallocs
  // to ensure availability in the reverse pass
  UpgradeAllocasToMallocs(NewF, mode);

  CanonicalizeLoops(NewF, FAM);

  {
    std::vector<Instruction *> ToErase;
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
    PA.preserve<ScalarEvolutionAnalysis>();
#if LLVM_VERSION_MAJOR > 6
    PA.preserve<PhiValuesAnalysis>();
#endif
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

#if LLVM_VERSION_MAJOR >= 8
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
#endif

  if (EnzymePrint)
    llvm::errs() << "after simplification :\n" << *NewF << "\n";

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (1)");
  }
  cache[std::make_pair(F, mode == DerivativeMode::ReverseModeCombined)] = NewF;
  return NewF;
}

Function *PreProcessCache::CloneFunctionWithReturns(
    DerivativeMode mode, Function *&F, ValueToValueMapTy &ptrInputs,
    const std::vector<DIFFE_TYPE> &constant_args,
    SmallPtrSetImpl<Value *> &constants, SmallPtrSetImpl<Value *> &nonconstant,
    SmallPtrSetImpl<Value *> &returnvals, ReturnType returnValue, Twine name,
    ValueToValueMapTy *VMapO, bool diffeReturnArg, llvm::Type *additionalArg) {
  assert(!F->empty());
  F = preprocessForClone(F, mode);
  std::vector<Type *> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::ArgsWithTwoReturns ||
      returnValue == ReturnType::Return ||
      returnValue == ReturnType::TwoReturns)
    RetTypes.push_back(F->getReturnType());
  if (returnValue == ReturnType::ArgsWithTwoReturns ||
      returnValue == ReturnType::TwoReturns)
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
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif
  CloneOrigin[NewF] = F;
  if (VMapO) {
    VMapO->insert(VMap.begin(), VMap.end());
    VMapO->getMDMap() = VMap.getMDMap();
  }

  bool hasPtrInput = false;
  unsigned ii = 0, jj = 0;
  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
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

void CoaleseTrivialMallocs(Function &F, DominatorTree &DT) {
  std::map<BasicBlock *, std::vector<std::pair<CallInst *, CallInst *>>>
      LegalMallocs;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "malloc") {
            for (auto U : CI->users()) {
              if (auto CI2 = dyn_cast<CallInst>(U)) {
                if (auto F2 = CI2->getCalledFunction()) {
                  if (F2->getName() == "free") {
                    if (DT.dominates(CI, CI2)) {
                      LegalMallocs[&BB].emplace_back(CI, CI2);
                    }
                  }
                }
              }
            }
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
      z.first->replaceAllUsesWith(B2.CreateInBoundsGEP(First, Size));
      Size = B.CreateAdd(Size, z.first->getArgOperand(0));
      z.first->eraseFromParent();
    }
    auto NewMalloc =
        cast<CallInst>(B.CreateCall(First->getCalledFunction(), Size));
    NewMalloc->copyIRFlags(First);
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
void PreProcessCache::optimizeIntermediate(Function *F) {
  PromotePass().run(*F, FAM);
  GVN().run(*F, FAM);
  SROA().run(*F, FAM);

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

  for (Function &Impl : *F->getParent()) {
    if (!Impl.hasFnAttribute("implements"))
      continue;
    const Attribute &A = Impl.getFnAttribute("implements");

    const StringRef SpecificationName = A.getValueAsString();
    Function *Specification = F->getParent()->getFunction(SpecificationName);
    if (!Specification) {
      LLVM_DEBUG(dbgs() << "Found implementation '" << Impl.getName()
                        << "' but no matching specification with name '"
                        << SpecificationName
                        << "', potentially inlined and/or eliminated.\n");
      continue;
    }
    LLVM_DEBUG(dbgs() << "Replace specification '" << Specification->getName()
                      << "' with implementation '" << Impl.getName() << "'\n");

    for (auto I = Specification->use_begin(), UE = Specification->use_end();
         I != UE;) {
      auto &use = *I;
      ++I;
      auto cext = ConstantExpr::getBitCast(&Impl, Specification->getType());
      use.set(cext);
      if (auto CI = dyn_cast<CallInst>(use.getUser())) {
#if LLVM_VERSION_MAJOR >= 11
        if (CI->getCalledOperand() == cext || CI->getCalledFunction() == &Impl)
#else
        if (CI->getCalledValue() == cext || CI->getCalledFunction() == &Impl)
#endif
        {
          CI->setCallingConv(Impl.getCallingConv());
        }
      }
    }
  }

  PassManagerBuilder Builder;
  Builder.OptLevel = 2;
  legacy::FunctionPassManager PM(F->getParent());
  Builder.populateFunctionPassManager(PM);
  PM.run(*F);
  {
    PreservedAnalyses PA;
    FAM.invalidate(*F, PA);
  }
  if (EnzymeCoalese)
    CoaleseTrivialMallocs(*F, FAM.getResult<DominatorTreeAnalysis>(*F));
  // DCEPass().run(*F, AM);
}

void PreProcessCache::clear() {
  FAM.clear();
  cache.clear();
}
