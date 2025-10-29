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

#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
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
#include <set>

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

#include "llvm/Support/TimeProfiler.h"

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

#include "llvm/Transforms/Utils/CodeExtractor.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

#include "llvm/IR/LegacyPassManager.h"
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#endif
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"

#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include <optional>

#include "CacheUtility.h"
#include "Utils.h"

#define addAttribute addAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define hasAttribute hasAttributeAtIndex

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

cl::opt<bool> EnzymeAutoSparsity("enzyme-auto-sparsity", cl::init(false),
                                 cl::Hidden,
                                 cl::desc("Run Enzyme auto sparsity"));

cl::opt<int> EnzymePostOptLevel(
    "enzyme-post-opt-level", cl::init(0), cl::Hidden,
    cl::desc("Post optimization level within Enzyme differentiated function"));

cl::opt<bool> EnzymeAlwaysInlineDiff(
    "enzyme-always-inline", cl::init(false), cl::Hidden,
    cl::desc("Mark generated functions as always-inline"));
}

/// Is the use of value val as an argument of call CI potentially captured
bool couldFunctionArgumentCapture(llvm::CallInst *CI, llvm::Value *val) {
  Function *F = CI->getCalledFunction();

  if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
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
  for (size_t i = 0, size = CI->arg_size(); i < size; i++) {
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

bool anyJuliaObjects(Type *T) {
  if (isSpecialPtr(T))
    return true;
  if (auto ST = dyn_cast<StructType>(T)) {
    for (auto elem : ST->elements()) {
      if (anyJuliaObjects(elem))
        return true;
    }
    return false;
  }
  if (auto AT = dyn_cast<ArrayType>(T)) {
    return anyJuliaObjects(AT->getElementType());
  }
  if (auto VT = dyn_cast<VectorType>(T)) {
    return anyJuliaObjects(VT->getElementType());
  }
  return false;
}

SmallVector<Value *, 1> getJuliaObjects(Value *v, IRBuilder<> &B) {
  SmallVector<Value *, 1> todo = {v};
  SmallVector<Value *, 1> done;
  while (todo.size()) {
    auto cur = todo.pop_back_val();
    auto T = cur->getType();
    if (!anyJuliaObjects(T)) {
      continue;
    }
    if (isSpecialPtr(T)) {
      done.push_back(cur);
      continue;
    }
    if (auto ST = dyn_cast<StructType>(T)) {
      for (auto en : llvm::enumerate(ST->elements())) {
        if (anyJuliaObjects(en.value())) {
          auto V2 = B.CreateExtractValue(cur, en.index());
          todo.push_back(V2);
        }
      }
      continue;
    }
    if (auto AT = dyn_cast<ArrayType>(T)) {
      for (size_t i = 0; i < AT->getNumElements(); i++) {
        todo.push_back(B.CreateExtractValue(cur, i));
      }
      continue;
    }
    if (auto VT = dyn_cast<VectorType>(T)) {
      assert(!VT->getElementCount().isScalable());
      size_t numElems = VT->getElementCount().getKnownMinValue();
      for (size_t i = 0; i < numElems; i++) {
        todo.push_back(B.CreateExtractElement(cur, i));
      }
      continue;
    }
    llvm_unreachable("unknown source of julia type");
  }
  return done;
}

void RecursivelyReplaceAddressSpace(
    SmallVector<std::tuple<Value *, Value *, Instruction *>, 1> &Todo,
    SmallVector<Instruction *, 1> &toErase, bool legal) {
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
      if (EnzymeJuliaAddrLoad &&
          cast<PointerType>(rep->getType())->getAddressSpace() == 0 &&
          ASC->getDestAddressSpace() == 11) {
        for (auto U : ASC->users()) {
          Todo.push_back(
              std::make_tuple(rep, (Value *)ASC, cast<Instruction>(U)));
        }
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
      auto nCI0 = B.CreateCast(
          CI->getOpcode(), rep,
#if LLVM_VERSION_MAJOR < 17
          PointerType::get(CI->getType()->getPointerElementType(),
                           cast<PointerType>(rep->getType())->getAddressSpace())
#else
          rep->getType()
#endif
      );
      if (auto nCI = dyn_cast<CastInst>(nCI0))
        nCI->takeName(CI);
      for (auto U : CI->users()) {
        Todo.push_back(
            std::make_tuple((Value *)nCI0, (Value *)CI, cast<Instruction>(U)));
      }
      toErase.push_back(CI);
      continue;
    }
    if (auto GEP = dyn_cast<GetElementPtrInst>(inst)) {
      IRBuilder<> B(GEP);
      if (EnzymeJuliaAddrLoad &&
          cast<PointerType>(rep->getType())->getAddressSpace() == 10) {
        rep = B.CreateAddrSpaceCast(
            rep,
#if LLVM_VERSION_MAJOR < 17
            PointerType::get(rep->getType()->getPointerElementType(), 11)
#else
            PointerType::get(rep->getContext(), 11)
#endif
        );
      }
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
    if (auto P = dyn_cast<PHINode>(inst)) {
      auto NumOperands = P->getNumIncomingValues();
      SmallVector<Value *, 1> replacedOperands(NumOperands, nullptr);
      for (size_t i = 0; i < NumOperands; i++)
        if (P->getOperand(i) == prev)
          replacedOperands[i] = rep;

      for (auto tval : Todo) {
        if (std::get<2>(tval) != P)
          continue;
        for (size_t i = 0; i < NumOperands; i++)
          if (P->getOperand(i) == std::get<1>(tval)) {
            replacedOperands[i] = std::get<0>(tval);
          }
      }
      bool allReplaced = true;
      for (size_t i = 0; i < NumOperands; i++) {
        if (!replacedOperands[i]) {
          allReplaced = false;
        }
      }
      if (!allReplaced) {
        bool remainingArePHIs = true;
        for (auto v : Todo) {
          if (isa<PHINode>(std::get<2>(v))) {
          } else {
            remainingArePHIs = false;
          }
        }
        if (!remainingArePHIs) {
          Todo.insert(Todo.begin(), cur);
          continue;
        }
      } else {
        IRBuilder<> B(&(*P->getParent()->getFirstNonPHIOrDbgOrLifetime()));
        auto nP = B.CreatePHI(rep->getType(), P->getNumOperands());
        for (size_t i = 0; i < NumOperands; i++) {
          nP->addIncoming(replacedOperands[i], P->getIncomingBlock(i));
        }
        nP->takeName(P);
        for (auto U : P->users()) {
          Todo.push_back(
              std::make_tuple((Value *)nP, (Value *)P, cast<Instruction>(U)));
        }
        toErase.push_back(P);
        for (int i = Todo.size() - 1; i >= 0; i--) {
          if (std::get<2>(Todo[i]) != P)
            continue;
          Todo.erase(Todo.begin() + i);
        }
        continue;
      }
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
        // Must copy the elementtype attribute as it is needed by the intrinsic
        nII->addParamAttr(
            ptrArgIndex,
            II->getParamAttr(ptrArgIndex, Attribute::AttrKind::ElementType));
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
      if (EnzymeJuliaAddrLoad &&
          cast<PointerType>(rep->getType())->getAddressSpace() == 10) {
        IRBuilder<> B(LI);
        rep = B.CreateAddrSpaceCast(
            rep,
#if LLVM_VERSION_MAJOR < 17
            PointerType::get(rep->getType()->getPointerElementType(), 11)
#else
            PointerType::get(rep->getContext(), 11)
#endif
        );
      }
      LI->setOperand(0, rep);
      continue;
    }
    if (auto SI = dyn_cast<StoreInst>(inst)) {
      if (SI->getPointerOperand() == prev) {
        if (EnzymeJuliaAddrLoad &&
            cast<PointerType>(rep->getType())->getAddressSpace() == 10) {
          IRBuilder<> B(SI);
          rep = B.CreateAddrSpaceCast(
              rep,
#if LLVM_VERSION_MAJOR < 17
              PointerType::get(rep->getType()->getPointerElementType(), 11)
#else
              PointerType::get(rep->getContext(), 11)
#endif
          );
        }
        SI->setOperand(1, rep);
        if (EnzymeJuliaAddrLoad &&
            cast<PointerType>(rep->getType())->getAddressSpace() == 11 &&
            cast<PointerType>(SI->getPointerOperand()->getType())
                    ->getAddressSpace() == 0) {
          IRBuilder<> B(SI);
          auto subvals = getJuliaObjects(SI->getValueOperand(), B);
          if (subvals.size()) {
            auto JLT =
                PointerType::get(StructType::get(SI->getContext(), {}), 10);
            auto FT = FunctionType::get(JLT, {}, true);
            auto wb = B.GetInsertBlock()
                          ->getParent()
                          ->getParent()
                          ->getOrInsertFunction("julia.write_barrier", FT);
            auto obj = getBaseObject(rep);
            assert(obj->getType() == JLT);
            subvals.insert(subvals.begin(), obj);
            B.CreateCall(wb, subvals);
          }
        }
        toPostCache.push_back(SI);
        continue;
      }
    }
    if (auto MS = dyn_cast<MemSetInst>(inst)) {
      IRBuilder<> B(MS);

      Value *nargs[] = {MS->getArgOperand(0), MS->getArgOperand(1),
                        MS->getArgOperand(2), MS->getArgOperand(3)};

      if (nargs[0] == prev)
        nargs[0] = rep;

      if (nargs[1] == prev)
        nargs[1] = rep;

      Type *tys[] = {nargs[0]->getType(), nargs[2]->getType()};
      auto nMS = cast<CallInst>(B.CreateCall(
          getIntrinsicDeclaration(MS->getParent()->getParent()->getParent(),
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
          getIntrinsicDeclaration(MTI->getParent()->getParent()->getParent(),
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
        if (F->getName() == "julia.write_barrier_binding" && legal) {
          toErase.push_back(CI);
          continue;
        }
      }
      IRBuilder<> B(CI);
      auto Addr = B.CreateAddrSpaceCast(rep, prev->getType());
      for (size_t i = 0; i < CI->arg_size(); i++) {
        if (CI->getArgOperand(i) == prev) {
          CI->setArgOperand(i, Addr);
        }
      }
      continue;
    }
    if (auto IVI = dyn_cast<InsertValueInst>(inst)) {
      if (IVI->getInsertedValueOperand() == prev && EnzymeJuliaAddrLoad &&
          cast<PointerType>(rep->getType())->getAddressSpace() == 0 &&
          cast<PointerType>(IVI->getInsertedValueOperand()->getType())
                  ->getAddressSpace() == 11) {
        IRBuilder<> B(IVI);
        auto Addr = B.CreateAddrSpaceCast(rep, prev->getType());
        IVI->setOperand(1, Addr);
        continue;
      }
    }
    if (auto I = dyn_cast<Instruction>(inst))
      llvm::errs() << *I->getParent()->getParent() << "\n";
    llvm::errs() << " inst: " << *inst << "\n";
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

void RecursivelyReplaceAddressSpace(Value *AI, Value *rep, bool legal) {
  SmallVector<std::tuple<Value *, Value *, Instruction *>, 1> Todo;
  for (auto U : AI->users()) {
    Todo.push_back(
        std::make_tuple((Value *)rep, (Value *)AI, cast<Instruction>(U)));
  }
  SmallVector<Instruction *, 1> toErase;
  if (auto I = dyn_cast<Instruction>(AI))
    toErase.push_back(I);
  RecursivelyReplaceAddressSpace(Todo, toErase, legal);
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

#if LLVM_VERSION_MAJOR >= 22
  Function *start_lifetime = nullptr;
  Function *end_lifetime = nullptr;
#endif

  SmallVector<std::tuple<Value *, Value *, Instruction *>, 1> Todo;
  SmallVector<Instruction *, 1> toErase;
  for (auto AI : ToConvert) {
    std::string nam = AI->getName().str();
    AI->setName("");

#if LLVM_VERSION_MAJOR >= 22
    for (auto U : llvm::make_early_inc_range(AI->users())) {
      if (auto II = dyn_cast<IntrinsicInst>(U)) {
        if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
          if (!start_lifetime) {
            start_lifetime = cast<Function>(
                NewF->getParent()
                    ->getOrInsertFunction(
                        "llvm.enzyme.lifetime_start",
                        FunctionType::get(Type::getVoidTy(NewF->getContext()),
                                          {}, true))
                    .getCallee());
          }
          IRBuilder<> B(II);
          SmallVector<Value *, 2> args(II->arg_size());
          for (unsigned i = 0; i < II->arg_size(); ++i) {
            args[i] = II->getArgOperand(i);
          }
          auto newI = B.CreateCall(start_lifetime, args);
          newI->takeName(II);
          newI->setDebugLoc(II->getDebugLoc());
          II->eraseFromParent();
          continue;
        }
        if (II->getIntrinsicID() == Intrinsic::lifetime_end) {
          if (!end_lifetime) {
            end_lifetime = cast<Function>(
                NewF->getParent()
                    ->getOrInsertFunction(
                        "llvm.enzyme.lifetime_end",
                        FunctionType::get(Type::getVoidTy(NewF->getContext()),
                                          {}, true))
                    .getCallee());
          }
          IRBuilder<> B(II);
          SmallVector<Value *, 2> args(II->arg_size());
          for (unsigned i = 0; i < II->arg_size(); ++i) {
            args[i] = II->getArgOperand(i);
          }
          auto newI = B.CreateCall(end_lifetime, args);
          newI->takeName(II);
          newI->setDebugLoc(II->getDebugLoc());
          II->eraseFromParent();
          continue;
        }
      }
    }
#endif

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
    auto align = AI->getAlign().value();
    CI->setMetadata(
        "enzyme_fromstack",
        MDNode::get(CI->getContext(),
                    {ConstantAsMetadata::get(ConstantInt::get(
                        IntegerType::get(AI->getContext(), 64), align))}));

    for (auto MD : {"enzyme_active", "enzyme_inactive", "enzyme_type",
                    "enzymejl_allocart", "enzymejl_allocart_name"})
      if (auto M = AI->getMetadata(MD))
        CI->setMetadata(MD, M);

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
      for (auto U : AI->users()) {
        Todo.push_back(
            std::make_tuple((Value *)rep, (Value *)AI, cast<Instruction>(U)));
      }
      if (auto I = dyn_cast<Instruction>(AI))
        toErase.push_back(I);
    } else {
      assert(rep->getType() == AI->getType());
      AI->replaceAllUsesWith(rep);
      AI->eraseFromParent();
    }
  }
  RecursivelyReplaceAddressSpace(Todo, toErase, /*legal*/ false);
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

      auto v2 = simplifyLoad(LI);
      if (v2) {
        todo.push_back({v2, next.second});
        continue;
      }
    }

    EmitFailure("DynamicReallocSize", Loc->getDebugLoc(), Loc,
                "could not statically determine size of realloc ", *Loc,
                " - because of - ", *next.first);
    return AI;

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
    list = list.addFnAttribute(NewF->getContext(), Attribute::ReadOnly);
    list = list.addParamAttribute(NewF->getContext(), 0, Attribute::ReadNone);
    list = addFunctionNoCapture(NewF->getContext(), list, 0);
    auto allocSize = NewF->getParent()->getOrInsertFunction(
        allocName,
        FunctionType::get(
            IntegerType::get(NewF->getContext(), 8 * sizeof(size_t)),
            {getInt8PtrTy(NewF->getContext())}, /*isVarArg*/ false),
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
  // TODO this logic should be combined with the dynamic loop emission
  // to minimize the number of branches if the realloc is used for multiple
  // values with the same bound.
  for (auto &BB : *NewF) {
    for (auto &I : make_early_inc_range(BB)) {
      if (hasMetadata(&I, "enzyme_zerostack")) {
        if (isa<AllocaInst>(getBaseObject(I.getOperand(0)))) {
          I.eraseFromParent();
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

  for (auto CI : ToInline) {
    InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR >= 18 && LLVM_VERSION_MAJOR < 21
    auto F = CI->getCalledFunction();
    if (CI->getParent()->IsNewDbgInfoFormat != F->IsNewDbgInfoFormat) {
      if (CI->getParent()->IsNewDbgInfoFormat) {
        F->convertToNewDbgValues();
      } else {
        F->convertFromNewDbgValues();
      }
    }
#endif
    InlineFunction(*CI, IFI);
  }
}

// Simplify all extractions to use inserted values, if possible.
void simplifyExtractions(Function *NewF) {
  // First rewrite/remove any extractions
  for (auto &BB : *NewF) {
    IRBuilder<> B(&BB);
    auto first = BB.begin();
    auto last = BB.empty() ? BB.end() : std::prev(BB.end());
    for (auto it = first; it != last;) {
      auto inst = &*it;
      // We iterate first here, since we may delete the instruction
      // in the body
      ++it;
      if (auto E = dyn_cast<ExtractValueInst>(inst)) {
        auto rep = GradientUtils::extractMeta(B, E->getAggregateOperand(),
                                              E->getIndices(), E->getName(),
                                              /*fallback*/ false);
        if (rep) {
          E->replaceAllUsesWith(rep);
          E->eraseFromParent();
        }
      }
    }
  }
  // Now that there may be unused insertions, delete them. We keep a list of
  // todo's since deleting an insertvalue may cause a different insertvalue to
  // have no uses
  SmallVector<InsertValueInst *, 1> todo;
  for (auto &BB : *NewF) {
    for (auto &inst : BB)
      if (auto I = dyn_cast<InsertValueInst>(&inst)) {
        if (I->getNumUses() == 0)
          todo.push_back(I);
      }
  }
  while (todo.size()) {
    auto I = todo.pop_back_val();
    auto op = I->getAggregateOperand();
    I->eraseFromParent();
    if (auto I2 = dyn_cast<InsertValueInst>(op))
      if (I2->getNumUses() == 0)
        todo.push_back(I2);
  }
}

void PreProcessCache::LowerAllocAddr(Function *NewF) {
  simplifyExtractions(NewF);
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
#if LLVM_VERSION_MAJOR < 17
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

#if LLVM_VERSION_MAJOR >= 22
  {
    auto start_lifetime =
        NewF->getParent()->getFunction("llvm.enzyme.lifetime_start");
    auto end_lifetime =
        NewF->getParent()->getFunction("llvm.enzyme.lifetime_end");

    SmallVector<CallInst *, 1> Todo;
    for (auto &BB : *NewF) {
      for (auto &I : BB) {
        if (auto CB = dyn_cast<CallInst>(&I)) {
          if (!CB->getCalledFunction())
            continue;
          if (CB->getCalledFunction() == start_lifetime ||
              CB->getCalledFunction() == end_lifetime) {
            Todo.push_back(CB);
          }
        }
      }
    }

    for (auto CB : Todo) {
      if (!isa<AllocaInst>(CB->getArgOperand(1))) {
        CB->eraseFromParent();
        continue;
      }
      IRBuilder<> B(CB);
      if (CB->getCalledFunction() == start_lifetime) {
        B.CreateLifetimeStart(cast<ConstantInt>(CB->getArgOperand(0)));
      } else {
        B.CreateLifetimeEnd(cast<ConstantInt>(CB->getArgOperand(0)));
      }
      CB->eraseFromParent();
    }
  }
#endif
}

/// Calls to realloc with an appropriate implementation
void PreProcessCache::ReplaceReallocs(Function *NewF, bool mem2reg) {
  if (mem2reg) {
    auto PA = PromotePass().run(*NewF, FAM);
    FAM.invalidate(*NewF, PA);
#if !defined(FLANG)
    PA = GVNPass().run(*NewF, FAM);
#else
    PA = GVN().run(*NewF, FAM);
#endif
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
    Type *BPTy = getInt8PtrTy(NewF->getContext());
    auto MallocFunc =
        M->getOrInsertFunction("malloc", BPTy, newsize->getType());
    auto next = B.CreateCall(MallocFunc, newsize);
    B.SetInsertPoint(resize);

    auto volatile_arg = ConstantInt::getFalse(CI->getContext());

    Value *nargs[] = {next, p, old, volatile_arg};

    Type *tys[] = {next->getType(), p->getType(), old->getType()};

    auto memcpyF =
        getIntrinsicDeclaration(NewF->getParent(), Intrinsic::memcpy, tys);

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
    Attribute::MustProgress,
#if LLVM_VERSION_MAJOR < 16
    Attribute::ReadOnly,
#endif
    Attribute::Speculatable,
    Attribute::NoUnwind,
    Attribute::AlwaysInline,
    Attribute::NoFree,
    Attribute::NoSync,
#if LLVM_VERSION_MAJOR < 16
    Attribute::InaccessibleMemOnly
#endif
  };
  for (auto attr : attrs) {
    W->addFnAttr(attr);
  }
#if LLVM_VERSION_MAJOR >= 16
  W->setOnlyAccessesInaccessibleMemory();
  W->setOnlyReadsMemory();
#endif
  W->addFnAttr(Attribute::get(F->getContext(), "enzyme_inactive"));
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

static void SimplifyMPIQueries(Function &NewF, FunctionAnalysisManager &FAM) {
  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(NewF);
  SmallVector<CallBase *, 4> Todo;
  SmallVector<CallBase *, 4> OMPBounds;
  for (auto &BB : NewF) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallBase>(&I)) {
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
    CallBase *res = nullptr;
    if (auto II = dyn_cast<InvokeInst>(CI))
      res = B.CreateInvoke(CreateMPIWrapper(CI->getCalledFunction()),
                           II->getNormalDest(), II->getUnwindDest(), arg, Defs);
    else
      res = B.CreateCall(CreateMPIWrapper(CI->getCalledFunction()), arg, Defs);
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
      (void)PT;
#if LLVM_VERSION_MAJOR < 17
      if (PT->getContext().supportsTypedPointers()) {
        if (PT->getPointerElementType() != res->getType())
          storePointer = B.CreateBitCast(
              storePointer,
              PointerType::get(res->getType(), PT->getAddressSpace()));
      }
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
      addCallSiteNoCapture(Bound, i);
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
          if (startsWith(CI->getCalledFunction()->getName(),
                         "_ZN3std2io5stdio6_print"))
            continue;
          if (startsWith(CI->getCalledFunction()->getName(), "_ZN4core3fmt"))
            continue;
          if (startsWith(CI->getCalledFunction()->getName(),
                         "enzyme_wrapmpi$$"))
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
          InlineFunction(*CI, IFI);
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
        CI->addFnAttr(Attribute::WillReturn);
        CI->addFnAttr(Attribute::MustProgress);
      }
      if (auto CI = dyn_cast<InvokeInst>(&I)) {
        CI->addFnAttr(Attribute::WillReturn);
        CI->addFnAttr(Attribute::MustProgress);
      }
    }
  }
}

void SplitPHIs(llvm::Function &F) {
  SetVector<Instruction *> todo;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isa<PHINode>(&I)) {
        todo.insert(&I);
      } else if (isa<SelectInst>(&I)) {
        todo.insert(&I);
      }
    }
  }
  while (todo.size()) {
    auto cur = todo.pop_back_val();
    IRBuilder<> B(cur);
    auto ST = dyn_cast<StructType>(cur->getType());
    if (!ST)
      continue;
    bool justExtract = true;
    for (auto U : cur->users()) {
      if (!isa<ExtractValueInst>(U)) {
        justExtract = false;
        break;
      }
      if (cast<ExtractValueInst>(U)->getIndices().size() == 0) {
        justExtract = false;
        break;
      }
    }
    if (!justExtract)
      continue;

    SmallVector<Value *, 1> replacements;
    for (size_t i = 0, e = ST->getNumElements(); i < e; i++) {
      if (auto cur2 = dyn_cast<PHINode>(cur)) {
        auto nPhi =
            B.CreatePHI(ST->getElementType(i), cur2->getNumIncomingValues(),
                        cur->getName() + ".extract." + std::to_string(i));
        for (auto &&[blk, val] :
             llvm::zip(cur2->blocks(), cur2->incoming_values())) {
          IRBuilder B2(blk->getTerminator());
          nPhi->addIncoming(GradientUtils::extractMeta(B2, val, i), blk);
        }
        replacements.push_back(nPhi);
        todo.insert(nPhi);
      } else {
        auto cur3 = cast<SelectInst>(cur);
        auto rep = B.CreateSelect(
            cur3->getCondition(),
            GradientUtils::extractMeta(B, cur3->getTrueValue(), i),
            GradientUtils::extractMeta(B, cur3->getFalseValue(), i),
            cur->getName() + ".extract." + std::to_string(i));
        replacements.push_back(rep);
        if (auto sel = dyn_cast<SelectInst>(rep))
          todo.insert(sel);
      }
    }
    for (auto &U : make_early_inc_range(cur->uses())) {
      auto user = cast<ExtractValueInst>(U.getUser());
      Value *rep = replacements[user->getIndices()[0]];
      IRBuilder<> B(user);
      if (user->getIndices().size() > 1)
        rep = B.CreateExtractValue(rep, user->getIndices().slice(1));
      assert(rep->getType() == user->getType());
      user->replaceAllUsesWith(rep);
      user->eraseFromParent();
    }
    cur->eraseFromParent();
  }
}

// returns if newly legal, subject to the pending calls
bool DetectReadonlyOrThrowFn(llvm::Function &F,
                             SmallPtrSetImpl<Function *> &calls_todo,
                             llvm::TargetLibraryInfo &TLI, bool &local) {
  if (isReadOnlyOrThrow(&F))
    return false;
  if (F.empty())
    return false;
  const auto unreachable = getGuaranteedUnreachable(&F);
  for (auto &BB : F) {
    if (unreachable.find(&BB) != unreachable.end()) {
      continue;
    }
    for (auto &I : BB) {
      if (!I.mayWriteToMemory())
        continue;
      if (hasMetadata(&I, "enzyme_ReadOnlyOrThrow"))
        continue;
      if (hasMetadata(&I, "enzyme_LocalReadOnlyOrThrow"))
        continue;

      if (auto MTI = dyn_cast<MemTransferInst>(&I)) {
        auto Obj = getBaseObject(MTI->getOperand(0));
        // Storing into local memory is fine since it definitionally will not be
        // seen outside the function. Note, even if one stored into x =
        // malloc(..), and stored x into a global/arg pointer, that second store
        // would trigger not readonly.
        if (isa<AllocaInst>(Obj))
          continue;
        if (isAllocationCall(Obj, TLI)) {
          if (local)
            continue;
          if (notCaptured(Obj))
            continue;
          local = true;
          continue;
        }
        if (auto arg = dyn_cast<Argument>(Obj)) {
          if (arg->hasStructRetAttr() ||
              arg->getParent()
                  ->getAttribute(arg->getArgNo() + AttributeList::FirstArgIndex,
                                 "enzymejl_returnRoots")
                  .isValid()) {
            local = true;
            continue;
          }
        }
      }
      if (auto MSI = dyn_cast<MemSetInst>(&I)) {
        auto Obj = getBaseObject(MSI->getOperand(0));
        // Storing into local memory is fine since it definitionally will not be
        // seen outside the function. Note, even if one stored into x =
        // malloc(..), and stored x into a global/arg pointer, that second store
        // would trigger not readonly.
        if (isa<AllocaInst>(Obj))
          continue;
        if (isAllocationCall(Obj, TLI)) {
          if (local)
            continue;
          if (notCaptured(Obj))
            continue;
          local = true;
          continue;
        }
        if (auto arg = dyn_cast<Argument>(Obj)) {
          if (arg->hasStructRetAttr() ||
              arg->getParent()
                  ->getAttribute(arg->getArgNo() + AttributeList::FirstArgIndex,
                                 "enzymejl_returnRoots")
                  .isValid()) {
            local = true;
            continue;
          }
        }
      }

      if (auto CI = dyn_cast<CallBase>(&I)) {
        if (isLocalReadOnlyOrThrow(CI)) {
          continue;
        }
        if (isAllocationCall(CI, TLI)) {
          continue;
        }
        if (getFuncNameFromCall(CI) == "zeroType") {
          auto Obj = getBaseObject(CI->getArgOperand(0));
          // Storing into local memory is fine since it definitionally will not
          // be seen outside the function. Note, even if one stored into x =
          // malloc(..), and stored x into a global/arg pointer, that second
          // store would trigger not readonly.
          if (isa<AllocaInst>(Obj))
            continue;
          if (isAllocationCall(Obj, TLI)) {
            if (local)
              continue;
            if (notCaptured(Obj))
              continue;
            local = true;
            continue;
          }
          if (auto arg = dyn_cast<Argument>(Obj)) {
            if (arg->hasStructRetAttr() ||
                arg->getParent()
                    ->getAttribute(arg->getArgNo() +
                                       AttributeList::FirstArgIndex,
                                   "enzymejl_returnRoots")
                    .isValid()) {
              local = true;
              continue;
            }
          }
        }
        if (auto F2 = CI->getCalledFunction()) {
          if (isDebugFunction(F2))
            continue;
          if (F2->getName() == "julia.write_barrier")
            continue;
          if (F2->getCallingConv() == CI->getCallingConv()) {
            if (F2 == &F)
              continue;
            if (isReadOnlyOrThrow(F2))
              continue;
            if (!F2->empty()) {
              if (EnzymePrintPerf) {
                EmitWarning(
                    "WritingInstruction", I, "Instruction could write forcing ",
                    F.getName(),
                    " to not be marked readonly_or_throw per sub-call of ",
                    F2->getName());
              }
              calls_todo.insert(F2);
              continue;
            }
          }
        }
      }
      if (auto SI = dyn_cast<StoreInst>(&I)) {
        auto Obj = getBaseObject(SI->getPointerOperand());
        // Storing into local memory is fine since it definitionally will not be
        // seen outside the function. Note, even if one stored into x =
        // malloc(..), and stored x into a global/arg pointer, that second store
        // would trigger not readonly.
        if (isa<AllocaInst>(Obj))
          continue;
        if (isAllocationCall(Obj, TLI)) {
          if (local)
            continue;
          if (notCaptured(Obj))
            continue;
          local = true;
          continue;
        }
        if (auto arg = dyn_cast<Argument>(Obj)) {
          if (arg->hasStructRetAttr() ||
              arg->getParent()
                  ->getAttribute(arg->getArgNo() + AttributeList::FirstArgIndex,
                                 "enzymejl_returnRoots")
                  .isValid()) {
            local = true;
            continue;
          }
        }
      }
      // ignore atomic load impacts
      if (isa<LoadInst>(&I))
        continue;
      if (EnzymeJuliaAddrLoad && isa<FenceInst>(&I)) {
        if (auto prev = dyn_cast_or_null<CallBase>(I.getPrevNode())) {
          if (auto F = prev->getCalledFunction())
            if (F->getName() == "julia.safepoint")
              continue;
        }
        if (auto prev = dyn_cast_or_null<CallBase>(I.getNextNode())) {
          if (auto F = prev->getCalledFunction())
            if (F->getName() == "julia.safepoint")
              continue;
        }
      }

      if (EnzymePrintPerf) {
        EmitWarning("WritingInstruction", I, "Instruction could write forcing ",
                    F.getName(), " to not be marked readonly_or_throw", I);
      }
      return false;
    }
  }

  if (calls_todo.size() == 0) {
    if (local)
      F.addFnAttr("enzyme_LocalReadOnlyOrThrow");
    else
      F.addFnAttr("enzyme_ReadOnlyOrThrow");
  }
  return true;
}

bool DetectReadonlyOrThrow(Module &M) {

  bool changed = false;

  PassBuilder PB;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Set of functions newly deduced readonlyorthrow by this pass
  SmallVector<llvm::Function *> todo;

  // Map of functions which could be readonly if all functions in the set are
  // marked readonly
  DenseMap<llvm::Function *, SmallPtrSet<Function *, 1>> todo_map;

  // Map from a function `f` to all the functions that have `f` as a
  // prerequisite for being readonly. Inverse of `todo_map`
  DenseMap<llvm::Function *, SmallPtrSet<Function *, 1>> inverse_todo_map;

  SmallPtrSet<Function *, 1> LocalReadOnlyFunctions;

  for (Function &F : M) {
    SmallPtrSet<Function *, 1> calls_todo;
    auto &TLI = FAM.getResult<TargetLibraryAnalysis>(F);
    bool local = false;
    if (DetectReadonlyOrThrowFn(F, calls_todo, TLI, local)) {
      if (local)
        LocalReadOnlyFunctions.insert(&F);
      if (calls_todo.size() == 0) {
        changed = true;
        todo.push_back(&F);
      } else {
        for (auto F2 : calls_todo) {
          inverse_todo_map[F2].insert(&F);
        }
      }
      todo_map[&F] = std::move(calls_todo);
    }
  }

  while (todo.size()) {
    auto cur = todo.pop_back_val();
    auto found = inverse_todo_map.find(cur);

    // Nobody needs cur as a prerequisite
    if (found == inverse_todo_map.end()) {
      continue;
    }
    for (auto F2 : found->second) {
      auto found2 = todo_map.find(F2);
      assert(found2 != todo_map.end());
      auto &fwd_set = found2->second;
      fwd_set.erase(cur);
      if (fwd_set.size() == 0) {
        if (LocalReadOnlyFunctions.contains(F2))
          F2->addFnAttr("enzyme_LocalReadOnlyOrThrow");
        else
          F2->addFnAttr("enzyme_ReadOnlyOrThrow");
        todo.push_back(F2);
        todo_map.erase(F2);
      }
    }

    inverse_todo_map.erase(found);
  }
  return changed;
}

Function *PreProcessCache::preprocessForClone(Function *F,
                                              DerivativeMode mode) {

  TimeTraceScope timeScope("preprocessForClone", F->getName());

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

  if (!F->empty()) {
    CloneFunctionInto(
        NewF, F, VMap,
        /*ModuleLevelChanges*/ CloneFunctionChangeType::LocalChangesOnly,
        Returns, "", nullptr);
  }
  CloneOrigin[NewF] = F;
  NewF->setAttributes(F->getAttributes());
  if (EnzymeNoAlias)
    for (auto j = NewF->arg_begin(); j != NewF->arg_end(); j++) {
      if (j->getType()->isPointerTy()) {
        j->addAttr(Attribute::NoAlias);
      }
    }
  NewF->addFnAttr(Attribute::WillReturn);
  NewF->addFnAttr(Attribute::MustProgress);
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
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
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

  SimplifyMPIQueries(*NewF, FAM);
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
        MemoryLocation Loc =
            MemoryLocation(&g, LocationSize::beforeOrAfterPointer());

        for (CallInst *CI : Calls) {
          if (isa<IntrinsicInst>(CI))
            continue;
          Function *F = CI->getCalledFunction();
          if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
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
          if (F && (startsWith(F->getName(), "f90io") ||
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
                if (auto castinst =
                        dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
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
                if (F && (startsWith(F->getName(), "f90io") ||
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
            antialloca->setAlignment(Align(g.getAlignment()));
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
              bb.CreateBitCast(antialloca, getInt8PtrTy(g.getContext())),
              bb.CreateBitCast(&g, getInt8PtrTy(g.getContext())),
              ConstantInt::get(
                  Type::getInt64Ty(g.getContext()),
                  g.getParent()->getDataLayout().getTypeAllocSizeInBits(
                      g.getValueType()) /
                      8),
              ConstantInt::getFalse(g.getContext())};

          Type *tys[] = {args[0]->getType(), args[1]->getType(),
                         args[2]->getType()};
          auto intr =
              getIntrinsicDeclaration(g.getParent(), Intrinsic::memcpy, tys);
          {

            auto cal = bb.CreateCall(intr, args);
            if (g.getAlignment()) {
              cal->addParamAttr(
                  0, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
              cal->addParamAttr(
                  1, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
            }
          }

          std::swap(args[0], args[1]);

          for (ReturnInst *RI : Returns) {
            IRBuilder<> IB(RI);
            auto cal = IB.CreateCall(intr, args);
            if (g.getAlignment()) {
              cal->addParamAttr(
                  0, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
              cal->addParamAttr(
                  1, Attribute::getWithAlignment(g.getContext(),
                                                 Align(g.getAlignment())));
            }
          }
        }
      }
    }

    auto Level = OptimizationLevel::O2;

    PassBuilder PB;
    FunctionPassManager FPM =
        PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
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
#elif !defined(FLANG)
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
#elif !defined(FLANG)
      auto PA = SROAPass().run(*NewF, FAM);
#else
      auto PA = SROA().run(*NewF, FAM);
#endif
      FAM.invalidate(*NewF, PA);
    }

    SimplifyCFGOptions scfgo;
    {
      auto PA = SimplifyCFGPass(scfgo).run(*NewF, FAM);
      FAM.invalidate(*NewF, PA);
    }
  }

  {
    SplitPHIs(*NewF);
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
    for (auto &BB : *NewF) {
      for (auto &I : make_early_inc_range(BB)) {
        if (auto MTI = dyn_cast<MemTransferInst>(&I)) {

          if (auto CI = dyn_cast<ConstantInt>(MTI->getOperand(2))) {
            if (CI->getValue() == 0) {
              MTI->eraseFromParent();
            }
          }
        }
      }
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

  {
    SmallPtrSet<Function *, 1> calls_todo;
    bool local = false;
    DetectReadonlyOrThrowFn(*NewF, calls_todo,
                            FAM.getResult<TargetLibraryAnalysis>(*NewF), local);
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

FunctionType *getFunctionTypeForClone(llvm::FunctionType *FTy,
                                      DerivativeMode mode, unsigned width,
                                      llvm::Type *additionalArg,
                                      llvm::ArrayRef<DIFFE_TYPE> constant_args,
                                      bool diffeReturnArg, bool returnTape,
                                      bool returnPrimal, bool returnShadow) {
  SmallVector<Type *, 4> RetTypes;
  if (returnPrimal)
    RetTypes.push_back(FTy->getReturnType());
  if (returnShadow)
    RetTypes.push_back(
        GradientUtils::getShadowType(FTy->getReturnType(), width));
  SmallVector<Type *, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto &I : FTy->params()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(GradientUtils::getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF && !returnTape) {
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
  if (returnTape) {
    RetTypes.insert(RetTypes.begin(),
                    getDefaultAnonymousTapeType(FTy->getContext()));
  }

  if (RetTypes.size() == 0)
    RetType = Type::getVoidTy(RetType->getContext());
  else if (RetTypes.size() == 1 && (returnPrimal || returnShadow) &&
           mode != DerivativeMode::ReverseModeCombined)
    RetType = RetTypes[0];
  else
    RetType = StructType::get(FTy->getContext(), RetTypes);

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
    SmallPtrSetImpl<Value *> &returnvals, bool returnTape, bool returnPrimal,
    bool returnShadow, const Twine &name,
    llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> *VMapO,
    bool diffeReturnArg, llvm::Type *additionalArg) {
  if (!F->empty())
    F = preprocessForClone(F, mode);
  llvm::ValueToValueMapTy VMap;
  llvm::FunctionType *FTy = getFunctionTypeForClone(
      F->getFunctionType(), mode, width, additionalArg, constant_args,
      diffeReturnArg, returnTape, returnPrimal, returnShadow);

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
  if (!F->empty()) {
    CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                      Returns, "", nullptr);
  }
  if (NewF->empty()) {
    auto entry = BasicBlock::Create(NewF->getContext(), "entry", NewF);
    IRBuilder<> B(entry);
    B.CreateUnreachable();
  }
  CloneOrigin[NewF] = F;
  if (VMapO) {
    for (const auto &data : VMap)
      VMapO->insert(std::pair<const llvm::Value *, AssertingReplacingVH>(
          data.first, (llvm::Value *)data.second));
    VMapO->getMDMap() = VMap.getMDMap();
  }

  for (auto attr : {"enzyme_ta_norecur", "frame-pointer"})
    if (F->getAttributes().hasAttribute(AttributeList::FunctionIndex, attr)) {
      NewF->addAttribute(
          AttributeList::FunctionIndex,
          F->getAttributes().getAttribute(AttributeList::FunctionIndex, attr));
    }

  for (auto attr :
       {"enzyme_type", "enzymejl_parmtype", "enzymejl_parmtype_ref"})
    if (F->getAttributes().hasAttribute(AttributeList::ReturnIndex, attr)) {
      NewF->addAttribute(
          AttributeList::ReturnIndex,
          F->getAttributes().getAttribute(AttributeList::ReturnIndex, attr));
    }

  bool hasPtrInput = false;
  unsigned ii = 0, jj = 0;

  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    if (F->hasParamAttribute(ii, Attribute::StructRet)) {
      NewF->addParamAttr(jj, Attribute::get(F->getContext(), "enzyme_sret"));
      // TODO
      // NewF->addParamAttr(
      //    jj,
      //    Attribute::get(
      //        F->getContext(), Attribute::AttrKind::ElementType,
      //        F->getParamAttribute(ii,
      //        Attribute::StructRet).getValueAsType()));
    }
    if (F->getAttributes().hasParamAttr(ii, "enzymejl_returnRoots")) {
      NewF->addParamAttr(
          jj, F->getAttributes().getParamAttr(ii, "enzymejl_returnRoots"));
      // TODO
      // NewF->addParamAttr(jj, F->getParamAttribute(ii,
      // Attribute::ElementType));
    }
    for (auto attr :
         {"enzymejl_parmtype", "enzymejl_parmtype_ref", "enzyme_type"})
      if (F->getAttributes().hasParamAttr(ii, attr)) {
        NewF->addParamAttr(jj, F->getAttributes().getParamAttr(ii, attr));
        for (auto ty : PrimalParamAttrsToPreserve)
          if (F->getAttributes().hasParamAttr(ii, ty)) {
            auto attr = F->getAttributes().getParamAttr(ii, ty);
            NewF->addParamAttr(jj, attr);
          }
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

    // Always remove nonnull/noundef since the caller may choose to pass
    // undef as an arg if provably it will not be used in the reverse pass
    if (constant_args[ii] == DIFFE_TYPE::DUP_NONEED ||
        mode == DerivativeMode::ReverseModeGradient) {
      if (F->hasParamAttribute(ii, Attribute::NonNull)) {
        NewF->removeParamAttr(jj, Attribute::NonNull);
      }
      if (F->hasParamAttribute(ii, Attribute::NoUndef)) {
        NewF->removeParamAttr(jj, Attribute::NoUndef);
      }
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

      for (auto attr :
           {"enzymejl_parmtype", "enzymejl_parmtype_ref", "enzyme_type"})
        if (F->getAttributes().hasParamAttr(ii, attr)) {
          if (width == 1)
            NewF->addParamAttr(jj + 1,
                               F->getAttributes().getParamAttr(ii, attr));
        }

      if (F->getAttributes().hasParamAttr(ii, "enzymejl_returnRoots")) {
        if (width == 1) {
          NewF->addParamAttr(jj + 1, F->getAttributes().getParamAttr(
                                         ii, "enzymejl_returnRoots"));
        } else {
          NewF->addParamAttr(jj + 1, Attribute::get(F->getContext(),
                                                    "enzymejl_returnRoots_v"));
        }
        // TODO
        // NewF->addParamAttr(jj + 1,
        //                   F->getParamAttribute(ii,
        //                   Attribute::ElementType));
      }

      if (F->hasParamAttribute(ii, Attribute::StructRet)) {
        if (width == 1) {
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret"));
          // TODO
          // NewF->addParamAttr(
          //     jj + 1,
          //     Attribute::get(F->getContext(),
          //     Attribute::AttrKind::ElementType,
          //                    F->getParamAttribute(ii,
          //                    Attribute::StructRet)
          //                        .getValueAsType()));
        } else {
          NewF->addParamAttr(jj + 1,
                             Attribute::get(F->getContext(), "enzyme_sret_v"));
          // TODO
          // NewF->addParamAttr(
          //     jj + 1,
          //     Attribute::get(F->getContext(),
          //     Attribute::AttrKind::ElementType,
          //                    F->getParamAttribute(ii,
          //                    Attribute::StructRet)
          //                        .getValueAsType()));
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
  if (EnzymeAlwaysInlineDiff)
    NewF->addFnAttr(Attribute::AlwaysInline);
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
          legal = false;
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
          if (CI->getCalledOperand() == cext ||
              CI->getCalledFunction() == &Impl) {
            CI->setCallingConv(Impl.getCallingConv());
          }
        }
      }
    }
  }
}

void PreProcessCache::optimizeIntermediate(Function *F) {
  PreservedAnalyses PA;
  PA = PromotePass().run(*F, FAM);
  FAM.invalidate(*F, PA);
#if !defined(FLANG)
  PA = GVNPass().run(*F, FAM);
#else
  PA = GVN().run(*F, FAM);
#endif
  FAM.invalidate(*F, PA);
#if LLVM_VERSION_MAJOR >= 16 && !defined(FLANG)
  PA = SROAPass(llvm::SROAOptions::PreserveCFG).run(*F, FAM);
#elif !defined(FLANG)
  PA = SROAPass().run(*F, FAM);
#else
  PA = SROA().run(*F, FAM);
#endif
  FAM.invalidate(*F, PA);

  if (EnzymeSelectOpt) {
    SimplifyCFGOptions scfgo;
    PA = SimplifyCFGPass(scfgo).run(*F, FAM);
    FAM.invalidate(*F, PA);
    PA = CorrelatedValuePropagationPass().run(*F, FAM);
    FAM.invalidate(*F, PA);
    SelectOptimization(F);
  }
  // EarlyCSEPass(/*memoryssa*/ true).run(*F, FAM);

  if (EnzymeCoalese)
    CoaleseTrivialMallocs(*F, FAM.getResult<DominatorTreeAnalysis>(*F));

  ReplaceFunctionImplementation(*F->getParent());

  {
    PreservedAnalyses PA;
    FAM.invalidate(*F, PA);
  }

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
    FunctionPassManager FPM =
        PB.buildFunctionSimplificationPipeline(Level, ThinOrFullLTOPhase::None);
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

// Returns if a is guaranteed to be equivalent to not b
static bool isNot(Value *a, Value *b) {
  // cmp pred, a, b    and cmp inverse(pred), a, b
  if (auto I1 = dyn_cast<CmpInst>(a))
    if (auto I2 = dyn_cast<CmpInst>(b))
      if (I1->getOperand(0) == I2->getOperand(0) &&
          I1->getOperand(1) == I2->getOperand(1) &&
          I1->getPredicate() == I2->getInversePredicate())
        return true;
  // a := xor true, b
  if (auto I = dyn_cast<Instruction>(a))
    if (I->getOpcode() == Instruction::Xor)
      for (int i = 0; i < 2; i++) {
        if (I->getOperand(i) == b)
          if (auto CI = dyn_cast<ConstantInt>(I->getOperand(1 - i)))
#if LLVM_VERSION_MAJOR > 16
            if (CI->getValue().isAllOnes())
#else
            if (CI->getValue().isAllOnesValue())
#endif
              return true;
      }
  // b := xor true, a
  if (auto I = dyn_cast<Instruction>(b))
    if (I->getOpcode() == Instruction::Xor)
      for (int i = 0; i < 2; i++) {
        if (I->getOperand(i) == a)
          if (auto CI = dyn_cast<ConstantInt>(I->getOperand(1 - i)))
#if LLVM_VERSION_MAJOR > 16
            if (CI->getValue().isAllOnes())
#else
            if (CI->getValue().isAllOnesValue())
#endif
              return true;
      }
  return false;
}

struct compare_insts {
public:
  DominatorTree &DT;
  LoopInfo &LI;
  compare_insts(DominatorTree &DT, LoopInfo &LI) : DT(DT), LI(LI) {}

  // return true if A appears later than B.
  bool operator()(Instruction *A, Instruction *B) const {
    if (A == B) {
      return false;
    }
    if (A->getParent() == B->getParent()) {
      return !A->comesBefore(B);
    }
    auto AB = A->getParent();
    auto BB = B->getParent();
    assert(AB->getParent() == BB->getParent());

    for (auto prev = BB->getPrevNode(); prev; prev = prev->getPrevNode()) {
      if (prev == AB)
        return false;
    }
    return true;
  }
};

class DominatorOrderSet : public std::set<Instruction *, compare_insts> {
public:
  DominatorOrderSet(DominatorTree &DT, LoopInfo &LI)
      : std::set<Instruction *, compare_insts>(compare_insts(DT, LI)) {}
  bool contains(Instruction *I) const {
    auto __i = find(I);
    return __i != end();
  }
  void remove(Instruction *I) {
    auto __i = find(I);
    assert(__i != end());
    erase(__i);
  }
  Instruction *pop_back_val() {
    auto back = end();
    back--;
    auto v = *back;
    erase(back);
    return v;
  }
};

bool directlySparse(Value *z) {
  if (isa<UIToFPInst>(z))
    return true;
  if (isa<SIToFPInst>(z))
    return true;
  if (isa<ZExtInst>(z))
    return true;
  if (isa<SExtInst>(z))
    return true;
  if (auto SI = dyn_cast<SelectInst>(z)) {
    if (auto CI = dyn_cast<ConstantInt>(SI->getTrueValue()))
      if (CI->isZero())
        return true;
    if (auto CI = dyn_cast<ConstantInt>(SI->getFalseValue()))
      if (CI->isZero())
        return true;
  }
  return false;
}

typedef DominatorOrderSet QueueType;

Function *getProductIntrinsic(llvm::Module &M, llvm::Type *T) {
  std::string name = "__enzyme_product.";
  if (T->isFloatTy())
    name += "f32";
  else if (T->isDoubleTy())
    name += "f64";
  else if (T->isIntegerTy())
    name += "i" + std::to_string(cast<IntegerType>(T)->getBitWidth());
  else
    assert(0);
  auto FT = llvm::FunctionType::get(T, {}, true);
  AttributeList AL;
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::ReadNone);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoUnwind);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoFree);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoSync);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::WillReturn);
  return cast<Function>(M.getOrInsertFunction(name, FT, AL).getCallee());
}

Function *getSumIntrinsic(llvm::Module &M, llvm::Type *T) {
  std::string name = "__enzyme_sum.";
  if (T->isFloatTy())
    name += "f32";
  else if (T->isDoubleTy())
    name += "f64";
  else if (T->isIntegerTy())
    name += "i" + std::to_string(cast<IntegerType>(T)->getBitWidth());
  else
    assert(0);
  auto FT = llvm::FunctionType::get(T, {}, true);
  AttributeList AL;
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::ReadNone);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoUnwind);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoFree);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::NoSync);
  AL = AL.addAttribute(T->getContext(), AttributeList::FunctionIndex,
                       Attribute::WillReturn);
  return cast<Function>(M.getOrInsertFunction(name, FT, AL).getCallee());
}

CallInst *isProduct(llvm::Value *v) {
  if (auto prod = dyn_cast<CallInst>(v))
    if (auto F = getFunctionFromCall(prod))
      if (startsWith(F->getName(), "__enzyme_product"))
        return prod;
  return nullptr;
}

CallInst *isSum(llvm::Value *v) {
  if (auto prod = dyn_cast<CallInst>(v))
    if (auto F = getFunctionFromCall(prod))
      if (startsWith(F->getName(), "__enzyme_sum"))
        return prod;
  return nullptr;
}

SmallVector<Value *, 1> callOperands(llvm::CallBase *CB) {
  return SmallVector<Value *, 1>(CB->args().begin(), CB->args().end());
}

bool guaranteedDataDependent(Value *z) {
  if (isa<LoadInst>(z))
    return true;
  if (isa<Constant>(z))
    return true;
  if (auto BO = dyn_cast<BinaryOperator>(z))
    return guaranteedDataDependent(BO->getOperand(0)) &&
           guaranteedDataDependent(BO->getOperand(1));
  if (auto C = dyn_cast<CastInst>(z))
    return guaranteedDataDependent(C->getOperand(0));
  if (auto S = isSum(z)) {
    for (auto op : callOperands(S))
      if (guaranteedDataDependent(op))
        return true;
    return false;
  }
  if (auto S = isProduct(z)) {
    for (auto op : callOperands(S))
      if (!guaranteedDataDependent(op))
        return false;
    return true;
  }
  if (auto II = dyn_cast<IntrinsicInst>(z)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::sqrt:
    case Intrinsic::sin:
    case Intrinsic::cos:
#if LLVM_VERSION_MAJOR >= 19
    case Intrinsic::sinh:
    case Intrinsic::cosh:
    case Intrinsic::tanh:
#endif
      return guaranteedDataDependent(II->getArgOperand(0));
    default:
      break;
    }
  }
  return false;
}

std::optional<std::string> fixSparse_inner(Instruction *cur, llvm::Function &F,
                                           QueueType &Q, DominatorTree &DT,
                                           ScalarEvolution &SE, LoopInfo &LI,
                                           const DataLayout &DL) {
  auto push = [&](llvm::Value *V) {
    if (V == cur)
      return V;
    assert(V);
    if (auto I = dyn_cast<Instruction>(V)) {
      Q.insert(I);
      for (auto U : I->users()) {
        if (auto I2 = dyn_cast<Instruction>(U)) {
          if (I2 == cur)
            continue;
          Q.insert(I2);
        }
      }
    }
    return V;
  };
  auto pushcse = [&](llvm::Value *V) -> llvm::Value * {
    if (auto I = dyn_cast<Instruction>(V)) {
      for (size_t i = 0; i < I->getNumOperands(); i++) {
        if (auto I2 = dyn_cast<Instruction>(I->getOperand(i))) {
          Instruction *candidate = nullptr;
          for (auto U : I2->users()) {
            candidate = dyn_cast<Instruction>(U);
            if (!candidate)
              continue;
            if (candidate == I && candidate->getType() != I->getType()) {
              candidate = nullptr;
              continue;
            }
            bool isSame = candidate->isIdenticalTo(I);
            if (!isSame) {
              if (auto P1 = isProduct(I))
                if (auto P2 = isProduct(I2)) {
                  std::multiset<llvm::Value *> s1;
                  std::multiset<llvm::Value *> s2;
                  for (auto &v : callOperands(P1))
                    s1.insert(v);
                  for (auto &v : callOperands(P2))
                    s2.insert(v);
                  isSame = s1 == s2;
                }
              if (auto P1 = isSum(I))
                if (auto P2 = isSum(I2)) {
                  std::multiset<llvm::Value *> s1;
                  std::multiset<llvm::Value *> s2;
                  for (auto &v : callOperands(P1))
                    s1.insert(v);
                  for (auto &v : callOperands(P2))
                    s2.insert(v);
                  isSame = s1 == s2;
                }
            }
            if (!isSame) {
              candidate = nullptr;
              continue;
            }

            if (DT.dominates(candidate, I)) {
              break;
            }
            candidate = nullptr;
          }
          if (candidate) {
            I->eraseFromParent();
            return candidate;
          }
          break;
        }
      }
      return push(I);
    }
    return V;
  };
  auto replaceAndErase = [&](llvm::Instruction *I, llvm::Value *candidate) {
    for (auto U : I->users())
      push(U);
    I->replaceAllUsesWith(candidate);
    push(candidate);

    SetVector<Instruction *> operands;
    for (size_t i = 0; i < I->getNumOperands(); i++) {
      if (auto I2 = dyn_cast<Instruction>(I->getOperand(i))) {
        if ((!I2->mayWriteToMemory() ||
             (isa<CallInst>(I2) && isReadOnly(cast<CallInst>(I2)))))
          operands.insert(I2);
      }
    }
    if (Q.contains(I)) {
      Q.remove(I);
    }
    assert(!Q.contains(I));
    I->eraseFromParent();
    for (auto op : operands)
      if (op->getNumUses() == 0) {
        if (Q.contains(op))
          Q.remove(op);
        op->eraseFromParent();
      }
  };
  if (!cur->getType()->isVoidTy() &&
      (!cur->mayWriteToMemory() ||
       (isa<CallInst>(cur) && isReadOnly(cast<CallInst>(cur))))) {
    // DCE
    if (cur->getNumUses() == 0) {
      for (size_t i = 0; i < cur->getNumOperands(); i++)
        push(cur->getOperand(i));
      assert(!Q.contains(cur));
      cur->eraseFromParent();
      return "DCE";
    }
    // CSE
    {
      for (size_t i = 0; i < cur->getNumOperands(); i++) {
        if (auto I = dyn_cast<Instruction>(cur->getOperand(i))) {
          Instruction *candidate = nullptr;
          bool reverse = false;
          for (auto U : I->users()) {
            candidate = dyn_cast<Instruction>(U);
            if (!candidate)
              continue;
            if (candidate == cur && candidate->getType() != cur->getType()) {
              candidate = nullptr;
              continue;
            }
            bool isSame = candidate->isIdenticalTo(cur);
            if (!isSame) {
              if (auto P1 = isProduct(candidate))
                if (auto P2 = isProduct(cur)) {
                  std::multiset<llvm::Value *> s1;
                  std::multiset<llvm::Value *> s2;
                  for (auto &v : callOperands(P1))
                    s1.insert(v);
                  for (auto &v : callOperands(P2))
                    s2.insert(v);
                  isSame = s1 == s2;
                }
              if (auto P1 = isSum(candidate))
                if (auto P2 = isSum(cur)) {
                  std::multiset<llvm::Value *> s1;
                  std::multiset<llvm::Value *> s2;
                  for (auto &v : callOperands(P1))
                    s1.insert(v);
                  for (auto &v : callOperands(P2))
                    s2.insert(v);
                  isSame = s1 == s2;
                }
            }

            if (!isSame) {
              candidate = nullptr;
              continue;
            }

            if (DT.dominates(candidate, cur)) {
              break;
            } else if (DT.dominates(cur, candidate)) {
              reverse = true;
              break;
            }
            candidate = nullptr;
          }
          if (candidate) {
            if (reverse) {
              if (Q.contains(candidate))
                Q.remove(candidate);
              auto tmp = candidate;
              candidate = cur;
              cur = tmp;
            }
            replaceAndErase(cur, candidate);
            return "CSE";
          }
          break;
        }
      }
    }
  }

  if (auto SI = dyn_cast<SelectInst>(cur))
    if (auto CI = dyn_cast<ConstantInt>(SI->getCondition())) {
      if (CI->isOne()) {
        replaceAndErase(cur, SI->getTrueValue());
        return "SelectToTrue";
      } else {
        replaceAndErase(cur, SI->getFalseValue());
        return "SelectToFalse";
      }
    }
  if (cur->getOpcode() == Instruction::Or) {
    for (int i = 0; i < 2; i++) {
      if (auto C = dyn_cast<ConstantInt>(cur->getOperand(i))) {
        // or a, 0 -> a
        if (C->isZero()) {
          replaceAndErase(cur, cur->getOperand(1 - i));
          return "OrZero";
        }
        // or a, 1 -> 1
        if (C->isOne() && cur->getType()->isIntegerTy(1)) {
          replaceAndErase(cur, C);
          return "OrOne";
        }
      }
    }
  }
  if (cur->getOpcode() == Instruction::And) {
    for (int i = 0; i < 2; i++) {
      if (auto C = dyn_cast<ConstantInt>(cur->getOperand(i))) {
        // and a, 1 -> a
        if (C->isOne() && cur->getType()->isIntegerTy(1)) {
          replaceAndErase(cur, cur->getOperand(1 - i));
          return "AndOne";
        }
        // and a, 0 -> 0
        if (C->isZero()) {
          replaceAndErase(cur, C);
          return "AndZero";
        }
      }
    }
  }

  IRBuilder<> B(cur);
  if (auto CI = dyn_cast<CastInst>(cur))
    if (auto C = dyn_cast<Constant>(CI->getOperand(0))) {
      replaceAndErase(
          cur, cast<Constant>(B.CreateCast(CI->getOpcode(), C, CI->getType())));
      return "CastConstProp";
    }
  std::function<Value *(Value *, Value *, Value *)> replace = [&](Value *val,
                                                                  Value *orig,
                                                                  Value *with) {
    if (val == orig) {
      return with;
    }
    if (isNot(val, orig)) {
      return pushcse(B.CreateNot(with));
    }
    if (isa<PHINode>(val))
      return val;

    if (auto I = dyn_cast<Instruction>(val)) {
      if (I->mayWriteToMemory() &&
          !(isa<CallInst>(I) && isReadOnly(cast<CallInst>(I))))
        return val;

      if (I->getOpcode() == Instruction::Add) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateAdd(lhs, rhs, "sel." + I->getName(),
                                   I->hasNoUnsignedWrap(),
                                   I->hasNoSignedWrap()));
      }

      if (I->getOpcode() == Instruction::Sub) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateSub(lhs, rhs, "sel." + I->getName(),
                                   I->hasNoUnsignedWrap(),
                                   I->hasNoSignedWrap()));
      }

      if (I->getOpcode() == Instruction::Mul) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateMul(lhs, rhs, "sel." + I->getName(),
                                   I->hasNoUnsignedWrap(),
                                   I->hasNoSignedWrap()));
      }

      if (I->getOpcode() == Instruction::And) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateAnd(lhs, rhs, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::Or) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateOr(lhs, rhs, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::Xor) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateXor(lhs, rhs, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::FAdd) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateFAddFMF(lhs, rhs, I, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::FSub) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateFSubFMF(lhs, rhs, I, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::FMul) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(B.CreateFMulFMF(lhs, rhs, I, "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::ZExt) {
        Value *op = replace(I->getOperand(0), orig, with);
        if (op == I->getOperand(0))
          return val;
        push(I);
        return pushcse(B.CreateZExt(op, I->getType(), "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::SExt) {
        Value *op = replace(I->getOperand(0), orig, with);
        if (op == I->getOperand(0))
          return val;
        push(I);
        return pushcse(B.CreateSExt(op, I->getType(), "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::UIToFP) {
        Value *op = replace(I->getOperand(0), orig, with);
        if (op == I->getOperand(0))
          return val;
        push(I);
        return pushcse(B.CreateUIToFP(op, I->getType(), "sel." + I->getName()));
      }

      if (I->getOpcode() == Instruction::SIToFP) {
        Value *op = replace(I->getOperand(0), orig, with);
        if (op == I->getOperand(0))
          return val;
        push(I);
        return pushcse(B.CreateSIToFP(op, I->getType(), "sel." + I->getName()));
      }

      if (auto CI = dyn_cast<CmpInst>(I)) {
        Value *lhs = replace(I->getOperand(0), orig, with);
        Value *rhs = replace(I->getOperand(1), orig, with);
        if (lhs == I->getOperand(0) && rhs == I->getOperand(1))
          return val;
        push(I);
        return pushcse(
            B.CreateCmp(CI->getPredicate(), lhs, rhs, "sel." + I->getName()));
      }

      if (auto SI = dyn_cast<SelectInst>(I)) {
        Value *cond = replace(SI->getCondition(), orig, with);
        Value *tval = replace(SI->getTrueValue(), orig, with);
        Value *fval = replace(SI->getFalseValue(), orig, with);
        if (cond == SI->getCondition() && tval == SI->getTrueValue() &&
            fval == SI->getFalseValue())
          return val;
        push(I);
        if (auto CI = dyn_cast<ConstantInt>(cond)) {
          if (CI->isOne())
            return tval;
          else
            return fval;
        }
        return pushcse(B.CreateSelect(cond, tval, fval, "sel." + I->getName()));
      }

      if (isProduct(I) || isSum(I)) {
        auto C = cast<CallBase>(I);
        auto ops = callOperands(C);
        bool changed = false;
        for (auto &op : ops) {
          auto next = replace(op, orig, with);
          if (next != op) {
            changed = true;
            op = next;
          }
        }
        if (!changed)
          return (Value *)I;
        push(I);
        pushcse(
            B.CreateCall(getFunctionFromCall(C), ops, "sel." + I->getName()));
      }
    }
    return val;
  };

  if (auto II = dyn_cast<IntrinsicInst>(cur))
    if (II->getIntrinsicID() == Intrinsic::fmuladd ||
        II->getIntrinsicID() == Intrinsic::fma) {
      B.setFastMathFlags(getFast());
      auto mul = pushcse(B.CreateFMul(II->getOperand(0), II->getOperand(1)));
      auto add = pushcse(B.CreateFAdd(mul, II->getOperand(2)));
      replaceAndErase(cur, add);
      return "FMulAddExpand";
    }

  if (auto BO = dyn_cast<BinaryOperator>(cur)) {
    if (BO->getOpcode() == Instruction::FMul && BO->isFast()) {
      Value *args[2] = {BO->getOperand(0), BO->getOperand(1)};
      auto mul = pushcse(
          B.CreateCall(getProductIntrinsic(*F.getParent(), BO->getType()), args,
                       cur->getName()));
      replaceAndErase(cur, mul);
      return "FMulToProduct";
    }
    if (BO->getOpcode() == Instruction::FDiv && BO->isFast()) {
      auto c0 = dyn_cast<ConstantFP>(BO->getOperand(0));
      if (!c0 || !c0->isExactlyValue(1.0)) {
        B.setFastMathFlags(getFast());
        auto div = pushcse(B.CreateFDivFMF(ConstantFP::get(BO->getType(), 1.0),
                                           BO->getOperand(1), BO));
        auto mul = pushcse(
            B.CreateFMulFMF(BO->getOperand(0), div, BO, cur->getName()));
        replaceAndErase(cur, mul);
        return "FDivToFMul";
      }
    }
    if (BO->getOpcode() == Instruction::FAdd && BO->isFast()) {
      Value *args[2] = {BO->getOperand(0), BO->getOperand(1)};
      auto mul = pushcse(
          B.CreateCall(getSumIntrinsic(*F.getParent(), BO->getType()), args));
      replaceAndErase(cur, mul);
      return "FAddToSum";
    }
    if (BO->getOpcode() == Instruction::FSub && BO->isFast()) {
      B.setFastMathFlags(getFast());
      Value *args[2] = {BO->getOperand(0),
                        pushcse(B.CreateFNeg(BO->getOperand(1)))};
      auto mul =
          pushcse(B.CreateCall(getSumIntrinsic(*F.getParent(), BO->getType()),
                               args, cur->getName()));
      replaceAndErase(cur, mul);
      return "FAddToSum";
    }
  }
  if (cur->getOpcode() == Instruction::FNeg) {
    B.setFastMathFlags(getFast());
    auto mul =
        pushcse(B.CreateFMulFMF(ConstantFP::get(cur->getType(), -1.0),
                                cur->getOperand(0), cur, cur->getName()));
    replaceAndErase(cur, mul);
    return "FNegToMul";
  }

  if (auto SI = dyn_cast<SelectInst>(cur)) {
    if (auto tc = dyn_cast<ConstantFP>(SI->getTrueValue()))
      if (auto fc = dyn_cast<ConstantFP>(SI->getFalseValue()))
        if (fc->isZero()) {
          if (tc->isExactlyValue(1.0)) {
            auto res =
                pushcse(B.CreateUIToFP(SI->getCondition(), tc->getType()));
            replaceAndErase(cur, res);
            return "SelToUIFP";
          }
          if (tc->isExactlyValue(-1.0)) {
            auto res =
                pushcse(B.CreateSIToFP(SI->getCondition(), tc->getType()));
            replaceAndErase(cur, res);
            return "SelToSIFP";
          }
        }
  }

  if (auto P = isProduct(cur)) {
    SmallVector<Value *, 1> operands;
    std::optional<APFloat> constval;
    bool changed = false;
    for (auto &v : callOperands(P))

    {
      if (auto P2 = isProduct(v)) {
        for (auto &v2 : callOperands(P2)) {
          push(v2);
          operands.push_back(v2);
        }
        push(P2);
        changed = true;
        continue;
      }
      if (auto C = dyn_cast<ConstantFP>(v)) {
        if (C->isExactlyValue(1.0)) {
          changed = true;
          continue;
        }
        if (C->isZero()) {
          replaceAndErase(cur, C);
          return "ZeroProduct";
        }
        if (!constval) {
          constval = C->getValue();
          continue;
        }
        constval = (*constval) * C->getValue();
        changed = true;
        continue;
      }
      if (auto op = dyn_cast<SelectInst>(v)) {
        if (auto tc = dyn_cast<ConstantFP>(op->getTrueValue()))
          if (tc->isZero()) {
            operands.push_back(pushcse(B.CreateUIToFP(
                pushcse(B.CreateNot(op->getCondition())), op->getType())));
            operands.push_back(op->getFalseValue());
            changed = true;
            continue;
          }
        if (auto tc = dyn_cast<ConstantFP>(op->getFalseValue()))
          if (tc->isZero()) {
            operands.push_back(
                pushcse(B.CreateUIToFP(op->getCondition(), op->getType())));
            operands.push_back(op->getTrueValue());
            changed = true;
            continue;
          }
      }
      operands.push_back(v);
    }
    if (constval)
      operands.push_back(ConstantFP::get(cur->getType(), *constval));

    if (operands.size() == 0) {
      replaceAndErase(cur, ConstantFP::get(cur->getType(), 1.0));
      return "EmptyProduct";
    }
    if (operands.size() == 1) {
      replaceAndErase(cur, operands[0]);
      return "SingleProduct";
    }
    if (changed) {
      auto mul = pushcse(
          B.CreateCall(getProductIntrinsic(*F.getParent(), cur->getType()),
                       operands, cur->getName()));
      replaceAndErase(cur, mul);
      return "ProductSimplification";
    }
  }

  if (auto P = isSum(cur)) {
    // map from operand, to number of counts
    std::map<Value *, unsigned> operands;
    std::optional<APFloat> constval;
    bool changed = false;
    for (auto &v : callOperands(P)) {
      if (auto P2 = isSum(v)) {
        for (auto &v2 : callOperands(P2)) {
          push(v2);
          operands[v2]++;
        }
        push(P2);
        changed = true;
        continue;
      }
      if (auto C = dyn_cast<ConstantFP>(v)) {
        if (C->isExactlyValue(0.0)) {
          changed = true;
          continue;
        }
        if (!constval) {
          constval = C->getValue();
          continue;
        }
        constval = (*constval) + C->getValue();
        changed = true;
        continue;
      }
      operands[v]++;
    }
    if (constval)
      operands[ConstantFP::get(cur->getType(), *constval)]++;

    if (operands.size() == 0) {
      replaceAndErase(cur, ConstantFP::get(cur->getType(), 0.0));
      return "EmptySum";
    }
    SmallVector<Value *, 1> args;
    for (auto &pair : operands) {
      if (pair.second == 1) {
        args.push_back(pair.first);
        continue;
      }
      changed = true;
      Value *sargs[] = {pair.first,
                        ConstantFP::get(cur->getType(), (double)pair.second)};
      args.push_back(pushcse(B.CreateCall(
          getProductIntrinsic(*F.getParent(), cur->getType()), sargs)));
    }
    if (args.size() == 1) {
      replaceAndErase(cur, args[0]);
      return "SingleSum";
    }
    if (changed) {
      auto sum =
          pushcse(B.CreateCall(getSumIntrinsic(*F.getParent(), cur->getType()),
                               args, cur->getName()));
      replaceAndErase(cur, sum);
      return "SumSimplification";
    }
  }

  if (auto P = isProduct(cur)) {
    SmallVector<Value *, 1> operands;
    SmallVector<Value *, 1> conditions;
    for (auto &v : callOperands(P)) {
      // z = uitofp i1 c to float -> select c, (prod withot z), 0
      if (auto op = dyn_cast<UIToFPInst>(v)) {
        if (op->getOperand(0)->getType()->isIntegerTy(1)) {
          conditions.push_back(op->getOperand(0));
          continue;
        }
      }
      // z = sitofp i1 c to float -> select c, (-prod withot z), 0
      if (auto op = dyn_cast<SIToFPInst>(v)) {
        if (op->getOperand(0)->getType()->isIntegerTy(1)) {
          conditions.push_back(op->getOperand(0));
          operands.push_back(ConstantFP::get(cur->getType(), -1.0));
          continue;
        }
      }
      if (auto op = dyn_cast<SelectInst>(v)) {
        if (auto tc = dyn_cast<ConstantFP>(op->getTrueValue()))
          if (tc->isZero()) {
            conditions.push_back(pushcse(B.CreateNot(op->getCondition())));
            operands.push_back(op->getFalseValue());
            continue;
          }
        if (auto tc = dyn_cast<ConstantFP>(op->getFalseValue()))
          if (tc->isZero()) {
            conditions.push_back(op->getCondition());
            operands.push_back(op->getTrueValue());
            continue;
          }
      }
      operands.push_back(v);
    }

    if (conditions.size()) {
      auto mul = pushcse(B.CreateCall(
          getProductIntrinsic(*F.getParent(), cur->getType()), operands));
      Value *condition = nullptr;
      for (auto v : conditions) {
        assert(v->getType()->isIntegerTy(1));
        if (condition == nullptr) {
          condition = v;
          continue;
        }
        condition = pushcse(B.CreateAnd(condition, v));
      }
      auto zero = ConstantFP::get(cur->getType(), 0.0);
      auto sel = pushcse(B.CreateSelect(condition, mul, zero, cur->getName()));
      replaceAndErase(cur, sel);
      return "ProductSelect";
    }
  }

  // TODO
  if (auto P = isSum(cur)) {
    // whether negated
    SmallVector<std::pair<Value *, bool>, 1> conditions;
    bool legal = true;
    for (auto &v : callOperands(P)) {
      // z = uitofp i1 c to float -> select c, (prod withot z), 0
      if (auto op = dyn_cast<UIToFPInst>(v)) {
        if (op->getOperand(0)->getType()->isIntegerTy(1)) {
          conditions.emplace_back(op->getOperand(0), false);
          continue;
        }
      }
      // z = sitofp i1 c to float -> select c, (-prod withot z), 0
      if (auto op = dyn_cast<SIToFPInst>(v)) {
        if (op->getOperand(0)->getType()->isIntegerTy(1)) {
          conditions.emplace_back(op->getOperand(0), false);
          continue;
        }
      }
      if (auto op = dyn_cast<SelectInst>(v)) {
        if (auto tc = dyn_cast<ConstantFP>(op->getTrueValue()))
          if (tc->isZero()) {
            conditions.emplace_back(op->getCondition(), true);
            continue;
          }
        if (auto tc = dyn_cast<ConstantFP>(op->getFalseValue()))
          if (tc->isZero()) {
            conditions.emplace_back(op->getCondition(), false);
            continue;
          }
      }
      legal = false;
      break;
    }
    Value *condition = nullptr;
    if (legal)
      for (size_t i = 0; i < conditions.size(); i++) {
        size_t count = 0;
        for (size_t j = 0; j < conditions.size(); j++) {
          if (((conditions[i].first == conditions[j].first) &&
               (conditions[i].second == conditions[i].second)) ||
              ((isNot(conditions[i].first, conditions[j].first) &&
                (conditions[i].second != conditions[i].second))))
            count++;
        }
        if (count == conditions.size() && count > 1) {
          condition = conditions[i].first;
          if (conditions[i].second)
            condition = pushcse(B.CreateNot(condition, "sumpnot"));
          break;
        }
      }

    if (condition) {

      SmallVector<Value *, 1> operands;
      for (auto &v : callOperands(P)) {
        // z = uitofp i1 c to float -> select c, (prod withot z), 0
        if (auto op = dyn_cast<UIToFPInst>(v)) {
          if (op->getOperand(0)->getType()->isIntegerTy(1)) {
            operands.push_back(ConstantFP::get(cur->getType(), 1.0));
            continue;
          }
        }
        // z = sitofp i1 c to float -> select c, (-prod withot z), 0
        if (auto op = dyn_cast<SIToFPInst>(v)) {
          if (op->getOperand(0)->getType()->isIntegerTy(1)) {
            operands.push_back(ConstantFP::get(cur->getType(), -1.0));
            continue;
          }
        }
        if (auto op = dyn_cast<SelectInst>(v)) {
          if (auto tc = dyn_cast<ConstantFP>(op->getTrueValue()))
            if (tc->isZero()) {
              operands.push_back(op->getFalseValue());
              continue;
            }
          if (auto tc = dyn_cast<ConstantFP>(op->getFalseValue()))
            if (tc->isZero()) {
              operands.push_back(op->getTrueValue());
              continue;
            }
        }
        llvm::errs() << " unhandled call op sumselect: " << *v << "\n";
        assert(0);
      }

      if (conditions.size()) {
        auto sum = pushcse(B.CreateCall(
            getSumIntrinsic(*F.getParent(), cur->getType()), operands));
        auto zero = ConstantFP::get(cur->getType(), 0.0);
        auto sel =
            pushcse(B.CreateSelect(condition, sum, zero, cur->getName()));
        replaceAndErase(cur, sel);
        return "SumSelect";
      }
    }
  }
  // (a1*b1) + (a1*c1) + (a1*d1 ) + ... -> a1 * (b1 + c1 + d1 + ...)
  if (auto S = isSum(cur)) {
    SmallVector<Value *, 1> allOps;
    auto combine = [](const SmallVector<Value *, 1> &lhs,
                      SmallVector<Value *, 1> rhs) {
      SmallVector<Value *, 1> out;
      for (auto v : lhs) {
        bool seen = false;
        for (auto &v2 : rhs) {
          if (v == v2) {
            v2 = nullptr;
            seen = true;
            break;
          }
        }
        if (seen) {
          out.push_back(v);
        }
      }
      return out;
    };
    auto subtract = [](SmallVector<Value *, 1> lhs,
                       const SmallVector<Value *, 1> &rhs) {
      for (auto v : rhs) {
        auto found = find(lhs, v);
        assert(found != lhs.end());
        lhs.erase(found);
      }
      return lhs;
    };
    bool seen = false;
    bool legal = true;
    for (auto op : callOperands(S)) {
      auto P = isProduct(op);
      if (!P) {
        legal = false;
        break;
      }
      if (!seen) {
        allOps = callOperands(P);
        seen = true;
        continue;
      }
      allOps = combine(allOps, callOperands(P));
    }

    if (legal && allOps.size() > 0) {
      SmallVector<Value *, 1> operands;
      for (auto op : callOperands(S)) {
        auto P = isProduct(op);
        push(op);
        auto sub = subtract(callOperands(P), allOps);
        auto newprod = pushcse(B.CreateCall(
            getProductIntrinsic(*F.getParent(), S->getType()), sub));
        operands.push_back(newprod);
      }
      auto newsum = pushcse(B.CreateCall(
          getSumIntrinsic(*F.getParent(), S->getType()), operands));
      allOps.push_back(newsum);
      auto fprod = pushcse(B.CreateCall(
          getProductIntrinsic(*F.getParent(), S->getType()), allOps));
      replaceAndErase(cur, fprod);
      return "SumFactor";
    }
  }

  /*
  // add (ext (x  == expr )), ( ext (x == expr + 1)) ->  -expr == c2 ) and c1
  != c2  -> false if (cur->getOpcode() == Instruction::Add) for (int j=0; j<2;
  j++) if (auto c0 = dyn_cast<ZExtInst>(cur->getOperand(j))) if (auto cmp0 =
  dyn_cast<ICmpInst>(c0->getOperand(0))) if (auto c1 =
  dyn_cast<CastInst>(cur->getOperand(1-j))) if (auto cmp1 =
  dyn_cast<ICmpInst>(c0->getOperand(0))) if (cmp0->getPredicate() ==
  ICmpInst::ICMP_EQ && cmp1->getPredicate() == ICmpInst::ICMP_EQ)
            {
          for (size_t i0 = 0; i0 < 2; i0++)
            for (size_t i1 = 0; i1 < 2; i1++)
              if (cmp0->getOperand(1 - i0) == cmp1->getOperand(1 - i1))
                auto e0 = SE.getSCEV(cmp0->getOperand(i0));
                auto e1 = SE.getSCEV(cmp1->getOperand(i1));
                auto m = SE.getMinusSCEV(e0, e1, SCEV::NoWrapMask);
                if (auto C = dyn_cast<SCEVConstant>(m)) {
                  // if c1 == c2 don't need the and they are equivalent
                  if (C->getValue()->isZero()) {
                  } else {
                      auto sel0 = pushcse(B.CreateSelect(cmp0,
  ConstantInt::get(cur->getType(), isa<ZExtInst>(cmp0) ? 1 : -1),
  ConstantInt::get(cur->getType(), 0));
                    // if non one constant they must be distinct.
                    replaceAndErase(cur,
                                    ConstantInt::getFalse(cur->getContext()));
                    return "AndNEExpr";
                  }
                }
              }
  }
  */

  if (auto fcmp = dyn_cast<FCmpInst>(cur)) {
    auto predicate = fcmp->getPredicate();
    if (predicate == FCmpInst::FCMP_OEQ || predicate == FCmpInst::FCMP_UEQ ||
        predicate == FCmpInst::FCMP_UNE || predicate == FCmpInst::FCMP_ONE) {
      for (int i = 0; i < 2; i++)
        if (auto C = dyn_cast<ConstantFP>(fcmp->getOperand(i))) {
          if (C->isZero()) {
            // (a1*a2*...an) == 0 -> (a1 == 0) || (a2 == 0) || ... (a2 == 0)
            // (a1*a2*...an) != 0 -> ![ (a1 == 0) || (a2 == 0) || ... (a2 ==
            // 0)
            // ]
            if (auto P = isProduct(fcmp->getOperand(1 - i))) {
              Value *res = nullptr;

              auto eq_predicate = predicate;
              if (predicate == FCmpInst::FCMP_UNE ||
                  predicate == FCmpInst::FCMP_ONE)
                eq_predicate = fcmp->getInversePredicate();

              for (auto &v : callOperands(P)) {
                auto ncmp1 = pushcse(B.CreateFCmp(eq_predicate, v, C));
                if (!res)
                  res = ncmp1;
                else
                  res = pushcse(B.CreateOr(res, ncmp1));
              }

              if (predicate == FCmpInst::FCMP_UNE ||
                  predicate == FCmpInst::FCMP_ONE) {
                res = pushcse(B.CreateNot(res));
              }

              replaceAndErase(cur, res);
              return "CmpProductSplit";
            }

            // (a1*b1) + (a1*c1) + (a1*d1 ) + ... ?= 0 -> a1 * (b1 + c1 + d1 +
            // ...) ?= 0
            if (auto S = isSum(fcmp->getOperand(1 - i))) {
              SmallVector<Value *, 1> allOps;
              auto combine = [](const SmallVector<Value *, 1> &lhs,
                                SmallVector<Value *, 1> rhs) {
                SmallVector<Value *, 1> out;
                for (auto v : lhs) {
                  bool seen = false;
                  for (auto &v2 : rhs) {
                    if (v == v2) {
                      v2 = nullptr;
                      seen = true;
                      break;
                    }
                  }
                  if (seen) {
                    out.push_back(v);
                  }
                }
                return out;
              };
              auto subtract = [](SmallVector<Value *, 1> lhs,
                                 const SmallVector<Value *, 1> &rhs) {
                for (auto v : rhs) {
                  auto found = find(lhs, v);
                  assert(found != lhs.end());
                  lhs.erase(found);
                }
                return lhs;
              };
              bool seen = false;
              bool legal = true;
              for (auto op : callOperands(S)) {
                auto P = isProduct(op);
                if (!P) {
                  legal = false;
                  break;
                }
                if (!seen) {
                  allOps = callOperands(P);
                  seen = true;
                  continue;
                }
                allOps = combine(allOps, callOperands(P));
              }

              if (legal && allOps.size() > 0) {
                SmallVector<Value *, 1> operands;
                for (auto op : callOperands(S)) {
                  auto P = isProduct(op);
                  push(op);
                  auto sub = subtract(callOperands(P), allOps);
                  auto newprod = pushcse(B.CreateCall(
                      getProductIntrinsic(*F.getParent(), C->getType()), sub));
                  operands.push_back(newprod);
                }
                auto newsum = pushcse(B.CreateCall(
                    getSumIntrinsic(*F.getParent(), C->getType()), operands));
                allOps.push_back(newsum);
                auto fprod = pushcse(B.CreateCall(
                    getProductIntrinsic(*F.getParent(), C->getType()), allOps));
                auto fcmp = pushcse(B.CreateCmp(predicate, fprod, C));
                replaceAndErase(cur, fcmp);
                return "CmpSumFactor";
              }
            }
          }
        }
    }
  }

  if (auto fcmp = dyn_cast<FCmpInst>(cur)) {
    auto predicate = fcmp->getPredicate();
    if (predicate == FCmpInst::FCMP_OEQ || predicate == FCmpInst::FCMP_UEQ ||
        predicate == FCmpInst::FCMP_UNE || predicate == FCmpInst::FCMP_ONE) {
      for (int i = 0; i < 2; i++)
        if (auto C = dyn_cast<ConstantFP>(fcmp->getOperand(i))) {
          if (C->isZero()) {
            // a + b == 0 -> ( (a == 0 & b == 0) || a == -b)
            if (auto S = isSum(fcmp->getOperand(1 - i))) {
              auto allOps = callOperands(S);
              if (!llvm::any_of(allOps, guaranteedDataDependent)) {
                auto eq_predicate = predicate;
                if (predicate == FCmpInst::FCMP_UNE ||
                    predicate == FCmpInst::FCMP_ONE)
                  eq_predicate = fcmp->getInversePredicate();

                Value *op_checks = nullptr;
                for (auto a : allOps) {
                  auto a_e0 = pushcse(B.CreateFCmp(eq_predicate, a, C));
                  if (op_checks == nullptr)
                    op_checks = a_e0;
                  else
                    op_checks = pushcse(B.CreateAnd(op_checks, a_e0));
                }
                SmallVector<Value *, 1> slice;
                for (size_t i = 1; i < allOps.size(); i++)
                  slice.push_back(allOps[i]);
                auto ane = pushcse(B.CreateFCmp(
                    eq_predicate, pushcse(B.CreateFNeg(allOps[0])),
                    pushcse(B.CreateCall(getFunctionFromCall(S), slice))));
                auto ori = pushcse(B.CreateOr(op_checks, ane));
                if (predicate == FCmpInst::FCMP_UNE ||
                    predicate == FCmpInst::FCMP_ONE) {
                  ori = pushcse(B.CreateNot(ori));
                }
                replaceAndErase(cur, ori);
                return "Sum2ZeroSplit";
              }
            }
          }
        }
    }
  }

  //  (zext a) + (zext b) ?= 0 -> zext a ?= - zext b
  if (auto icmp = dyn_cast<CmpInst>(cur)) {
    if (icmp->getPredicate() == CmpInst::ICMP_EQ ||
        icmp->getPredicate() == CmpInst::ICMP_NE) {
      for (int i = 0; i < 2; i++)
        if (auto C = dyn_cast<ConstantInt>(icmp->getOperand(i)))
          if (C->isZero())
            if (auto add = dyn_cast<BinaryOperator>(icmp->getOperand(1 - i)))
              if (add->getOpcode() == Instruction::Add)
                if (auto a0 = dyn_cast<CastInst>(add->getOperand(0)))
                  if (auto a1 = dyn_cast<CastInst>(add->getOperand(1)))
                    if (a0->getOperand(0)->getType() ==
                            a1->getOperand(0)->getType() &&
                        (isa<ZExtInst>(a0) || isa<SExtInst>(a0))) {
                      auto cmp2 = pushcse(B.CreateCmp(
                          icmp->getPredicate(), a0, pushcse(B.CreateNeg(a1))));
                      replaceAndErase(cur, cmp2);
                      return "CmpExt0Shuffle";
                    }
    }
  }

  // sub 0, (zext i1 to N) -> sext i1 to N
  // sub 0, (sext i1 to N) -> zext i1 to N
  if (auto sub = dyn_cast<BinaryOperator>(cur))
    if (sub->getOpcode() == Instruction::Sub)
      if (auto C = dyn_cast<ConstantInt>(sub->getOperand(0)))
        if (C->isZero())
          if (auto a0 = dyn_cast<CastInst>(sub->getOperand(1)))
            if (a0->getOperand(0)->getType()->isIntegerTy(1)) {

              Value *tmp = nullptr;
              if (isa<ZExtInst>(a0))
                tmp = pushcse(B.CreateSExt(a0->getOperand(0), a0->getType()));
              else if (isa<SExtInst>(a0))
                tmp = pushcse(B.CreateZExt(a0->getOperand(0), a0->getType()));
              else
                assert(0);
              replaceAndErase(cur, tmp);
              return "NegSZExtI1";
            }

  if ((cur->getOpcode() == Instruction::LShr ||
       cur->getOpcode() == Instruction::SDiv ||
       cur->getOpcode() == Instruction::UDiv) &&
      cur->isExact())
    if (auto C2 = dyn_cast<ConstantInt>(cur->getOperand(1)))
      if (auto mul = dyn_cast<BinaryOperator>(cur->getOperand(0))) {
        //  (lshr exact (mul a, C1), C2), C -> mul a, (lhsr exact C1, C2) if
        //  C2 divides C1
        if (mul->getOpcode() == Instruction::Mul)
          for (int i0 = 0; i0 < 2; i0++)
            if (auto C1 = dyn_cast<ConstantInt>(mul->getOperand(i0))) {
              auto lhs = C1->getValue();
              APInt rhs = C2->getValue();
              if (cur->getOpcode() == Instruction::LShr) {
                rhs = APInt(rhs.getBitWidth(), 1) << rhs;
              }

              APInt div, rem;
              if (cur->getOpcode() == Instruction::LShr ||
                  cur->getOpcode() == Instruction::UDiv)
                APInt::udivrem(lhs, rhs, div, rem);
              else
                APInt::sdivrem(lhs, rhs, div, rem);
              if (rem == 0) {
                auto res = pushcse(B.CreateMul(
                    mul->getOperand(1 - i0),
                    ConstantInt::get(cur->getType(), div),
                    "mdiv." + cur->getName(), mul->hasNoUnsignedWrap(),
                    mul->hasNoSignedWrap()));
                push(mul);
                replaceAndErase(cur, res);
                return "IMulDivConst";
              }
            }
        //  (lshr exact (add a, C1), C2), C -> add a, (lhsr exact C1, C2) if
        //  C2
        if (mul->getOpcode() == Instruction::Add)
          for (int i0 = 0; i0 < 2; i0++)
            if (auto C1 = dyn_cast<ConstantInt>(mul->getOperand(i0))) {
              auto lhs = C1->getValue();
              APInt rhs = C2->getValue();
              if (cur->getOpcode() == Instruction::LShr) {
                rhs = APInt(rhs.getBitWidth(), 1) << rhs;
              }

              APInt div, rem;
              if (cur->getOpcode() == Instruction::LShr ||
                  cur->getOpcode() == Instruction::UDiv)
                APInt::udivrem(lhs, rhs, div, rem);
              else
                APInt::sdivrem(lhs, rhs, div, rem);
              if (rem == 0 && ((mul->hasNoUnsignedWrap() &&
                                (cur->getOpcode() == Instruction::LShr ||
                                 cur->getOpcode() == Instruction::UDiv)) ||
                               (mul->hasNoSignedWrap() &&
                                (cur->getOpcode() == Instruction::AShr ||
                                 cur->getOpcode() == Instruction::SDiv)))) {
                auto res = pushcse(B.CreateAdd(
                    mul->getOperand(1 - i0),
                    ConstantInt::get(cur->getType(), div),
                    "madd." + cur->getName(), mul->hasNoUnsignedWrap(),
                    mul->hasNoSignedWrap()));
                push(mul);
                replaceAndErase(cur, res);
                return "IAddDivConst";
              }
            }
      }

  // mul (mul a, const1), (mul b, const2) -> mul (mul a, b), (const1, const2)
  if (cur->getOpcode() == Instruction::FMul)
    if (cur->isFast())
      if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(0)))
        if (mul1->getOpcode() == Instruction::FMul && mul1->isFast())
          if (auto mul2 = dyn_cast<Instruction>(cur->getOperand(1)))
            if (mul2->getOpcode() == Instruction::FMul && mul2->isFast()) {
              for (auto i1 = 0; i1 < 2; i1++)
                for (auto i2 = 0; i2 < 2; i2++)
                  if (isa<Constant>(mul1->getOperand(i1)))
                    if (isa<Constant>(mul2->getOperand(i2))) {

                      auto n0 = pushcse(
                          B.CreateFMulFMF(mul1->getOperand(1 - i1),
                                          mul2->getOperand(1 - i2), cur));
                      auto n1 = pushcse(B.CreateFMulFMF(
                          mul1->getOperand(i1), mul2->getOperand(i2), cur));
                      auto n2 = pushcse(B.CreateFMulFMF(n0, n1, cur));
                      push(mul1);
                      push(mul2);
                      replaceAndErase(cur, n2);
                      return "MulMulConstConst";
                    }
            }

  // mul (mul a, const1), const2 -> mul a, (mul const1, const2)
  if ((cur->getOpcode() == Instruction::FMul && cur->isFast()) ||
      cur->getOpcode() == Instruction::Mul)
    for (auto i1 = 0; i1 < 2; i1++)
      if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(i1)))
        if (((mul1->getOpcode() == Instruction::FMul && mul1->isFast())) ||
            mul1->getOpcode() == Instruction::FMul)
          if (auto const2 = dyn_cast<Constant>(cur->getOperand(1 - i1)))
            for (auto i2 = 0; i2 < 2; i2++)
              if (auto const1 = dyn_cast<Constant>(mul1->getOperand(i2))) {
                Value *res = nullptr;
                if (cur->getOpcode() == Instruction::FMul) {
                  auto const3 = pushcse(B.CreateFMulFMF(const1, const2, mul1));
                  res = pushcse(
                      B.CreateFMulFMF(mul1->getOperand(1 - i2), const3, cur));
                } else {
                  auto const3 = pushcse(B.CreateMul(const1, const2));
                  res = pushcse(B.CreateMul(mul1->getOperand(1 - i2), const3));
                }
                push(mul1);
                replaceAndErase(cur, res);
                return "MulConstConst";
              }

  if (auto fcmp = dyn_cast<FCmpInst>(cur)) {
    if (fcmp->getPredicate() == FCmpInst::FCMP_OEQ) {
      for (int i = 0; i < 2; i++)
        if (auto C = dyn_cast<ConstantFP>(fcmp->getOperand(i))) {
          if (C->isZero()) {
            if (auto fmul = dyn_cast<BinaryOperator>(fcmp->getOperand(1 - i))) {
              // (a*b) == 0 -> (a == 0) || (b == 0)
              if (fmul->getOpcode() == Instruction::FMul) {
                auto ncmp1 = pushcse(
                    B.CreateFCmp(fcmp->getPredicate(), fmul->getOperand(0), C));
                auto ncmp2 = pushcse(
                    B.CreateFCmp(fcmp->getPredicate(), fmul->getOperand(1), C));
                auto ori = pushcse(B.CreateOr(ncmp1, ncmp2));
                replaceAndErase(cur, ori);
                return "CmpFMulSplit";
              }
              // (a/b) == 0 -> (a == 0)
              if (fmul->getOpcode() == Instruction::FDiv) {
                auto ncmp1 = pushcse(
                    B.CreateFCmp(fcmp->getPredicate(), fmul->getOperand(0), C));
                replaceAndErase(cur, ncmp1);
                return "CmpFDivSplit";
              }
              // (a - b) ?= 0 -> a ?= b
              if (fmul->getOpcode() == Instruction::FSub) {
                auto ncmp1 = pushcse(B.CreateFCmp(fcmp->getPredicate(),
                                                  fmul->getOperand(0),
                                                  fmul->getOperand(1)));
                replaceAndErase(cur, ncmp1);
                return "CmpFSubSplit";
              }
            }
            if (auto cast = dyn_cast<SIToFPInst>(fcmp->getOperand(1 - i))) {
              auto ncmp1 = pushcse(B.CreateICmp(
                  ICmpInst::ICMP_EQ, cast->getOperand(0),
                  ConstantInt::get(cast->getOperand(0)->getType(), 0)));
              replaceAndErase(cur, ncmp1);
              return "SFCmpToICmp";
            }
            if (auto cast = dyn_cast<UIToFPInst>(fcmp->getOperand(1 - i))) {
              auto ncmp1 = pushcse(B.CreateICmp(
                  ICmpInst::ICMP_EQ, cast->getOperand(0),
                  ConstantInt::get(cast->getOperand(0)->getType(), 0)));
              replaceAndErase(cur, ncmp1);
              return "UFCmpToICmp";
            }
            if (auto SI = dyn_cast<SelectInst>(fcmp->getOperand(1 - i))) {
              auto res = pushcse(
                  B.CreateSelect(SI->getCondition(),
                                 pushcse(B.CreateCmp(fcmp->getPredicate(), C,
                                                     SI->getTrueValue())),
                                 pushcse(B.CreateCmp(fcmp->getPredicate(), C,
                                                     SI->getFalseValue()))));
              replaceAndErase(cur, res);
              return "FCmpSelect";
            }
          }
        }
    }
  }
  if (auto fcmp = dyn_cast<CmpInst>(cur)) {
    if (fcmp->getPredicate() == CmpInst::ICMP_EQ ||
        fcmp->getPredicate() == CmpInst::ICMP_NE ||
        fcmp->getPredicate() == CmpInst::FCMP_OEQ ||
        fcmp->getPredicate() == CmpInst::FCMP_ONE) {

      // a + c ?= a  ->  c ?= 0  , if fast
      for (int i = 0; i < 2; i++)
        if (auto inst = dyn_cast<Instruction>(fcmp->getOperand(i)))
          if (inst->getOpcode() == Instruction::FAdd && inst->isFast())
            for (int i2 = 0; i2 < 2; i2++)
              if (inst->getOperand(i2) == fcmp->getOperand(1 - i)) {
                auto res = pushcse(
                    B.CreateCmp(fcmp->getPredicate(), inst->getOperand(1 - i2),
                                ConstantFP::get(inst->getType(), 0)));
                replaceAndErase(cur, res);
                return "CmpFAddSame";
              }

      // a == b -> a & b | !a & !b
      // a != b -> a & !b | !a & b
      if (fcmp->getOperand(0)->getType()->isIntegerTy(1)) {
        auto a = fcmp->getOperand(0);
        auto b = fcmp->getOperand(1);
        if (fcmp->getPredicate() == CmpInst::ICMP_EQ) {
          auto res = pushcse(
              B.CreateOr(pushcse(B.CreateAnd(a, b)),
                         pushcse(B.CreateAnd(pushcse(B.CreateNot(a)),
                                             pushcse(B.CreateNot(b))))));
          replaceAndErase(cur, res);
          return "CmpI1EQ";
        }
        if (fcmp->getPredicate() == CmpInst::ICMP_NE) {
          auto res = pushcse(
              B.CreateOr(pushcse(B.CreateAnd(pushcse(B.CreateNot(a)), b)),
                         pushcse(B.CreateAnd(a, pushcse(B.CreateNot(b))))));
          replaceAndErase(cur, res);
          return "CmpI1NE";
        }
      }

      for (int i = 0; i < 2; i++)
        if (auto CI = dyn_cast<ConstantInt>(fcmp->getOperand(i)))
          if (CI->isZero()) {
            // a + a ?= 0 -> a ?= 0
            if (auto addI = dyn_cast<Instruction>(fcmp->getOperand(1 - i))) {
              if (addI->getOpcode() == Instruction::Add &&
                  addI->getOperand(0) == addI->getOperand(1)) {
                Value *res = pushcse(
                    B.CreateCmp(fcmp->getPredicate(), addI->getOperand(0), CI));
                replaceAndErase(cur, res);
                return "CmpAddAdd";
              }
              // (a-b) ?= 0 -> a ?= b
              if (addI->getOpcode() == Instruction::Sub) {
                auto ncmp1 = pushcse(B.CreateICmp(fcmp->getPredicate(),
                                                  addI->getOperand(0),
                                                  addI->getOperand(1)));
                replaceAndErase(cur, ncmp1);
                return "CmpISubSplit";
              }
            }
          }

      // (a * b) == (c * b) -> (a == c) ||  b == 0
      // (a * b) != (c * b) -> (a != c) && b != 0
      // auto S1 = SE.getSCEV(cur->getOperand(0));
      // auto S2 = SE.getSCEV(cur->getOperand(1));
      // llvm::errs() <<" attempting push: " << *cur << " S1: " << *S1 << "
      // S2: " << *S2 << " and " << *cur->getOperand(0) << " " <<
      // *cur->getOperand(1) << "\n";
      if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(0)))
        if (auto mul2 = dyn_cast<Instruction>(cur->getOperand(1))) {
          if (mul1->getOpcode() == Instruction::Mul &&
              mul2->getOpcode() == Instruction::Mul &&
              mul1->hasNoUnsignedWrap() && mul1->hasNoSignedWrap() &&
              mul2->hasNoUnsignedWrap() && mul2->hasNoSignedWrap()) {
            for (int i = 0; i < 2; i++) {
              if (mul1->getOperand(i) == mul2->getOperand(i)) {
                Value *res = pushcse(B.CreateICmp(fcmp->getPredicate(),
                                                  mul1->getOperand(1 - i),
                                                  mul2->getOperand(1 - i)));
                auto b = mul1->getOperand(i);
                if (fcmp->getPredicate() == CmpInst::ICMP_EQ) {
                  Value *bZero = pushcse(B.CreateICmp(
                      CmpInst::ICMP_EQ, b, ConstantInt::get(b->getType(), 0)));
                  res = pushcse(B.CreateOr(res, bZero));
                } else {
                  Value *bZero = pushcse(B.CreateICmp(
                      ICmpInst::ICMP_NE, b, ConstantInt::get(b->getType(), 0)));
                  res = pushcse(B.CreateAnd(res, bZero));
                }
                replaceAndErase(cur, res);
                return "CmpMulCommon";
              }
            }
          }
          // same as above but now with floats
          if (mul1->getOpcode() == Instruction::FMul &&
              mul2->getOpcode() == Instruction::FMul && mul1->isFast() &&
              mul2->isFast()) {
            for (int i = 0; i < 2; i++) {
              if (mul1->getOperand(i) == mul2->getOperand(i)) {
                Value *res = pushcse(B.CreateFCmp(fcmp->getPredicate(),
                                                  mul1->getOperand(1 - i),
                                                  mul2->getOperand(1 - i)));
                auto b = mul1->getOperand(i);
                if (fcmp->getPredicate() == CmpInst::FCMP_OEQ) {
                  Value *bZero = pushcse(B.CreateCmp(
                      CmpInst::FCMP_OEQ, b, ConstantFP::get(b->getType(), 0)));
                  res = pushcse(B.CreateOr(res, bZero));
                } else {
                  Value *bZero = pushcse(B.CreateCmp(
                      CmpInst::FCMP_ONE, b, ConstantFP::get(b->getType(), 0)));
                  res = pushcse(B.CreateAnd(res, bZero));
                }
                replaceAndErase(cur, res);
                return "CmpMulfCommon";
              }
            }
          }

          // (uitofp a ) ?= (uitofp b) -> a ?= b
          for (auto cond : {Instruction::UIToFP, Instruction::SIToFP})
            if (mul1->getOpcode() == cond && mul2->getOpcode() == cond &&
                mul1->getOperand(0)->getType() ==
                    mul2->getOperand(0)->getType()) {
              Value *res = pushcse(B.CreateICmp(
                  fcmp->getPredicate() == CmpInst::FCMP_OEQ ? CmpInst::ICMP_EQ
                                                            : CmpInst::ICMP_NE,
                  mul1->getOperand(0), mul2->getOperand(0)));
              replaceAndErase(cur, res);
              return "CmpUIToFP";
            }

          // (zext a ) ?= (zext b) -> a ?= b
          if (mul1->getOpcode() == Instruction::ZExt &&
              mul2->getOpcode() == Instruction::ZExt &&
              mul1->getOperand(0)->getType() ==
                  mul2->getOperand(0)->getType()) {
            Value *res =
                pushcse(B.CreateICmp(fcmp->getPredicate(), mul1->getOperand(0),
                                     mul2->getOperand(0)));
            replaceAndErase(cur, res);
            return "CmpZExt";
          }

          // (zext i1 a ) == (sext i1 b) -> (!a & !b)
          // (zext i1 a ) != (sext i1 b) -> (a | b)
          if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(0)))
            if (auto mul2 = dyn_cast<Instruction>(cur->getOperand(1)))
              if (((mul1->getOpcode() == Instruction::ZExt &&
                    mul2->getOpcode() == Instruction::SExt) ||
                   (mul1->getOpcode() == Instruction::SExt &&
                    mul2->getOpcode() == Instruction::ZExt)) &&
                  mul1->getOperand(0)->getType() ==
                      mul2->getOperand(0)->getType() &&
                  mul1->getOperand(0)->getType()->isIntegerTy(1)) {

                Value *na = mul1->getOperand(0);
                Value *nb = mul2->getOperand(0);

                if (fcmp->getPredicate() == ICmpInst::ICMP_EQ) {
                  na = pushcse(B.CreateNot(na));
                  nb = pushcse(B.CreateNot(nb));
                }

                Value *res = nullptr;
                if (fcmp->getPredicate() == ICmpInst::ICMP_EQ)
                  res = pushcse(B.CreateAnd(na, nb));
                else
                  res = pushcse(B.CreateOr(na, nb));

                replaceAndErase(cur, res);
                return "CmpZExtSExt";
              }
        }
    }
    if (fcmp->getPredicate() == ICmpInst::ICMP_EQ) {
      for (int i = 0; i < 2; i++) {
        if (auto C = dyn_cast<ConstantInt>(fcmp->getOperand(i))) {
          if (C->isZero()) {
            if (auto fmul = dyn_cast<BinaryOperator>(fcmp->getOperand(1 - i))) {
              // (a*b) == 0 -> (a == 0) || (b == 0)
              if (fmul->getOpcode() == Instruction::Mul) {
                auto ncmp1 = pushcse(
                    B.CreateICmp(fcmp->getPredicate(), fmul->getOperand(0), C));
                auto ncmp2 = pushcse(
                    B.CreateICmp(fcmp->getPredicate(), fmul->getOperand(1), C));
                auto ori = pushcse(B.CreateOr(ncmp1, ncmp2));
                replaceAndErase(cur, ori);
                return "CmpIMulSplit";
              }
            }
          }
        }
      }
    }
  }

  if (cur->getOpcode() == Instruction::FAdd) {
    // add x, x -> mul 2.0
    if (cur->getOperand(0) == cur->getOperand(1) && cur->isFast()) {
      auto res = pushcse(B.CreateFMulFMF(
          cur->getOperand(0), ConstantFP::get(cur->getType(), 2.0), cur));
      replaceAndErase(cur, res);
      return "AddToMul2";
    }
  }

  if (cur->getOpcode() == Instruction::Add) {
    // add x, (y * -1) -> sub x, y
    for (int i = 0; i < 2; i++) {
      if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(i)))
        if (mul1->getOpcode() == Instruction::Mul) {
          for (int j = 0; j < 2; j++) {
            if (auto C = dyn_cast<ConstantInt>(mul1->getOperand(j))) {
              if (C->isMinusOne()) {
                auto res = pushcse(B.CreateSub(cur->getOperand(1 - i),
                                               mul1->getOperand(1 - j)));
                push(mul1);

                replaceAndErase(cur, res);
                return "AddToSub";
              }
            }
          }
        }
    }
  }

  if (auto SI = dyn_cast<SelectInst>(cur)) {
    auto shouldMove = [](Value *v) { return isa<Constant>(v); };

    /*
    // select c, 0, x -> fmul (uitofp (!c)), x
    if (auto C1 = dyn_cast<ConstantFP>(SI->getTrueValue())) {
      if (C1->isZero()) {
        auto n = pushcse(B.CreateNot(SI->getCondition()));
        auto val = pushcse(B.CreateUIToFP(n, SI->getType()));
        auto res = pushcse(B.CreateFMul(val, SI->getFalseValue()));
        if (auto I = dyn_cast<Instruction>(res))
          I->setFast(true);
        replaceAndErase(cur, res);
        return true;
      }
    }
    // select c, x, 0 -> fmul (uitofp c), x
    if (auto C1 = dyn_cast<ConstantFP>(SI->getFalseValue())) {
      if (C1->isZero()) {
        auto val = pushcse(B.CreateUIToFP(SI->getCondition(), SI->getType()));
        auto res = pushcse(B.CreateFMul(val, SI->getTrueValue()));
        if (auto I = dyn_cast<Instruction>(res))
          I->setFast(true);
        replaceAndErase(cur, res);
        return true;
      }
    }
    */

    // select c, (mul x y), 0 -> mul x, (select c, y, 0)
    for (int i = 0; i < 2; i++)
      if (auto inst = dyn_cast<Instruction>(SI->getOperand(1 + i)))
        if (inst->getOpcode() == Instruction::Mul)
          // inst->getOpcode() == Instruction::FMul)
          if (auto C = dyn_cast<Constant>(SI->getOperand(1 + (1 - i))))
            if ((isa<ConstantInt>(C) && cast<ConstantInt>(C)->isZero()) ||
                (isa<ConstantFP>(C) && cast<ConstantFP>(C)->isZero()))
              for (int j = 0; j < 2; j++)
                if (shouldMove(inst->getOperand(j))) {
                  auto x = inst->getOperand(j);
                  auto y = inst->getOperand(1 - j);
                  auto isel = pushcse(B.CreateSelect(
                      SI->getCondition(), (i == 0) ? y : C, (i == 0) ? C : y,
                      "smulmove." + SI->getName()));
                  Value *imul;
                  if (cur->getType()->isIntegerTy())
                    imul = pushcse(B.CreateMul(isel, x, "",
                                               inst->hasNoUnsignedWrap(),
                                               inst->hasNoSignedWrap()));
                  else
                    imul = pushcse(B.CreateFMulFMF(isel, x, inst, ""));

                  replaceAndErase(cur, imul);
                  return "SelMulMove";
                }

    // select c, (sitofp x), (sitofp y) ->  sitofp (select c, x, y)
    // select c, c5, (sitofp y) ->  sitofp (select c, c5, y)
    {
      Value *ops[2] = {nullptr, nullptr};
      bool legal = true;
      for (int i = 0; i < 2; i++) {
        if (isa<ConstantFP>(SI->getOperand(1 + i))) {
          ops[i] = nullptr;
          continue;
        }
        if (auto CI = dyn_cast<CastInst>(SI->getOperand(1 + i))) {
          if (CI->getOpcode() == Instruction::SIToFP) {
            ops[i] = CI->getOperand(0);
            continue;
          }
        }
        legal = false;
        break;
      }
      for (int i = 0; i < 2; i++) {
        if (!ops[i] && ops[1 - i])
          ops[i] = ConstantInt::get(ops[1 - i]->getType(), 0);
      }
      for (int i = 0; i < 2; i++) {
        if (ops[i] == nullptr || ops[i]->getType() != ops[0]->getType()) {
          legal = false;
          break;
        }
      }
      if (legal) {
        auto isel = pushcse(B.CreateSelect(SI->getCondition(), ops[0], ops[1],
                                           "seltofp." + SI->getName()));
        auto res = pushcse(B.CreateSIToFP(isel, SI->getType()));

        replaceAndErase(cur, res);
        return "SelSIMerge";
      }
    }
  }

  if (cur->getOpcode() == Instruction::Mul) {
    for (int i = 0; i < 2; i++) {
      // mul (x, 1) -> x
      if (auto C = dyn_cast<ConstantInt>(cur->getOperand(i)))
        if (C->isOne()) {
          replaceAndErase(cur, cur->getOperand(1 - i));
          return "MulIdent";
        }

      // mul (zext i1 x), y -> mul (zext i1 x) y[x->1]
      if (auto Z = dyn_cast<ZExtInst>(cur->getOperand(i)))
        if (Z->getOperand(0)->getType()->isIntegerTy(1)) {
          auto prev = cur->getOperand(1 - i);
          auto next = replace(prev, Z->getOperand(0),
                              ConstantInt::getTrue(cur->getContext()));
          if (next != prev) {
            auto res = pushcse(B.CreateMul(Z, next, "postmul." + cur->getName(),
                                           cur->hasNoUnsignedWrap(),
                                           cur->hasNoSignedWrap()));
            replaceAndErase(cur, res);
            return "MulReplaceZExt";
          }
        }
    }

    /*
  // mul x, (select c, 0, y) -> select c (mul x 0), (mul x y)
  for (int i=0; i<2; i++)
  if (auto SI = dyn_cast<SelectInst>(cur->getOperand(i)))
  for (int j=0; j<2; j++)
  if (auto CI = dyn_cast<ConstantInt>(SI->getOperand(1+j)))
    if (CI->isZero()) {
            auto tval = (j == 0) ? CI :
  pushcse(B.CreateMul(SI->getTrueValue(), cur->getOperand(1-i), "tval." +
  cur->getName(), cur->hasNoUnsignedWrap(), cur->hasNoSignedWrap())); auto
  fval = (j == 1) ? CI : pushcse(B.CreateMul(SI->getFalseValue(),
  cur->getOperand(1-i), "fval." + cur->getName(), cur->hasNoUnsignedWrap(),
                               cur->hasNoSignedWrap()));

          auto res = pushcse(B.CreateSelect(SI->getCondition(), tval, fval));

          replaceAndErase(cur, res);
          return true;
        }
        */

    // mul (sub x, y), -c   -> mul (sub, y, x), c
    for (int i = 0; i < 2; i++)
      if (auto inst = dyn_cast<Instruction>(cur->getOperand(i)))
        if (inst->getOpcode() == Instruction::Sub)
          if (auto CI = dyn_cast<ConstantInt>(cur->getOperand(1 - i)))
            if (CI->isNegative()) {
              auto sub2 = pushcse(B.CreateSub(
                  inst->getOperand(1), inst->getOperand(0), "",
                  inst->hasNoUnsignedWrap(), inst->hasNoSignedWrap()));
              auto mul2 = pushcse(B.CreateMul(
                  sub2, ConstantInt::get(CI->getType(), -CI->getValue()), "",
                  cur->hasNoUnsignedWrap(), cur->hasNoSignedWrap()));

              replaceAndErase(cur, mul2);
              return "MulSubNegConst";
            }
  }

  if (cur->getOpcode() == Instruction::Sub)
    if (auto CI = dyn_cast<ConstantInt>(cur->getOperand(0)))
      if (CI->isZero())
        if (auto zext = dyn_cast<Instruction>(cur->getOperand(1))) {
          // sub 0, (zext i1 x) -> sext x
          if (zext->getOpcode() == Instruction::ZExt &&
              zext->getOperand(0)->getType()->isIntegerTy(1)) {
            auto res =
                pushcse(B.CreateSExt(zext->getOperand(0), cur->getType()));
            replaceAndErase(cur, res);
            return "SubZExt";
          }
          // sub 0, (mul nsw nuw constant, x) -> mul nsw nuw -constant, x
          if (zext->getOpcode() == Instruction::Mul &&
              zext->hasNoUnsignedWrap() && zext->hasNoSignedWrap()) {
            for (int i = 0; i < 2; i++)
              if (auto CI = dyn_cast<ConstantInt>(zext->getOperand(i))) {
                auto res = pushcse(B.CreateMul(
                    zext->getOperand(1 - i),
                    ConstantInt::get(CI->getType(), -CI->getValue()),
                    "neg." + zext->getName(), true, true));
                replaceAndErase(cur, res);
                return "SubMulConstant";
              }
          }
        }

  // add (zext (and c1, x) ), (zext (and c1, y)) -> select c1, (add (zext x),
  // (zext y)), 0
  /*
  if (cur->getOpcode() == Instruction::Add ||
      cur->getOpcode() == Instruction::Sub ||
      cur->getOpcode() == Instruction::Mul)
    if (auto inst1 = dyn_cast<Instruction>(cur->getOperand(0)))
      if (auto inst2 = dyn_cast<Instruction>(cur->getOperand(1)))
        if (inst1->getOpcode() == Instruction::ZExt && inst2->getOpcode() ==
  Instruction::ZExt) if (auto and1 =
  dyn_cast<Instruction>(inst1->getOperand(0))) if (auto and2 =
  dyn_cast<Instruction>(inst2->getOperand(0))) if
  (and1->getType()->isIntegerTy(1) && and2->getType()->isIntegerTy(1) &&
  and1->getOpcode() == Instruction::And && and2->getOpcode() ==
  Instruction::And) { bool done = false; for (int i1=0; i1<2; i1++) for (int
  i2=0; i2<2; i2++) if (and1->getOperand(i1) == and2->getOperand(i2)) { auto
  c1 = and1->getOperand(i1); auto x = and1->getOperand(1-i1); x =
  pushcse(B.CreateZExt(x, inst1->getType()));  auto y =
  and2->getOperand(1-i2);

                y = pushcse(B.CreateZExt(y, inst2->getType()));

              Value *res = nullptr;
              switch (cur->getOpcode()) {
              case Instruction::Add:
                res = pushcse(B.CreateAdd(x, y, "", cur->hasNoUnsignedWrap(),
  cur->hasNoSignedWrap())); break; case Instruction::Sub: res = B.CreateSub(x,
  y,
  "", cur->hasNoUnsignedWrap(), cur->hasNoSignedWrap()); break; case
  Instruction::Mul: res = B.CreateMul(x, y, "", cur->hasNoUnsignedWrap(),
  cur->hasNoSignedWrap()); break; default: llvm_unreachable("Illegal opcode");
              }
              res = pushcse(B.CreateSelect(c1, res,
  Constant::getNullValue(cur->getType())));

                replaceAndErase(cur, res);
                return;
            }
            }
  */

  // add (select %c   c0, x), (select %c, c1, y) -> select %c, (add c0, c1),
  // (add x, y) and for sub/mul/cmp
  if (cur->getOpcode() == Instruction::Add ||
      cur->getOpcode() == Instruction::Sub ||
      cur->getOpcode() == Instruction::Mul ||
      cur->getOpcode() == Instruction::FAdd ||
      cur->getOpcode() == Instruction::FSub ||
      cur->getOpcode() == Instruction::FMul ||
      // cur->getOpcode() == Instruction::SIToFP ||
      // cur->getOpcode() == Instruction::UIToFP ||
      cur->getOpcode() == Instruction::ICmp ||
      cur->getOpcode() == Instruction::FCmp) {

    Value *SI1cond = nullptr;
    Value *SI1tval = nullptr;
    Value *SI1fval = nullptr;
    if (auto SI1 = dyn_cast<SelectInst>(cur->getOperand(0))) {
      SI1cond = SI1->getCondition();
      SI1tval = SI1->getTrueValue();
      SI1fval = SI1->getFalseValue();
    }
    if (auto SI1 = dyn_cast<ZExtInst>(cur->getOperand(0)))
      if (SI1->getOperand(0)->getType()->isIntegerTy(1)) {
        SI1cond = SI1->getOperand(0);
        SI1tval = SI1;
        SI1fval = ConstantInt::get(SI1->getType(), 0);
      }
    if (auto SI1 = dyn_cast<SExtInst>(cur->getOperand(0)))
      if (SI1->getOperand(0)->getType()->isIntegerTy(1)) {
        SI1cond = SI1->getOperand(0);
        SI1tval = SI1;
        SI1fval = ConstantInt::get(SI1->getType(), 0);
      }
    Value *SI2cond = nullptr;
    Value *SI2tval = nullptr;
    Value *SI2fval = nullptr;

    auto op2 = cur->getOperand((cur->getOpcode() == Instruction::SIToFP ||
                                cur->getOpcode() == Instruction::UIToFP)
                                   ? 0
                                   : 1);
    if (auto SI2 = dyn_cast<SelectInst>(op2)) {
      SI2cond = SI2->getCondition();
      SI2tval = SI2->getTrueValue();
      SI2fval = SI2->getFalseValue();
    }
    if (auto SI2 = dyn_cast<ZExtInst>(op2))
      if (SI2->getOperand(0)->getType()->isIntegerTy(1)) {
        SI2cond = SI2->getOperand(0);
        SI2tval = SI2;
        SI2fval = ConstantInt::get(SI2->getType(), 0);
      }
    if (auto SI2 = dyn_cast<SExtInst>(op2))
      if (SI2->getOperand(0)->getType()->isIntegerTy(1)) {
        SI2cond = SI2->getOperand(0);
        SI2tval = SI2;
        SI2fval = ConstantInt::get(SI2->getType(), 0);
      }

    if (SI1cond && SI2cond && (SI1cond == SI2cond || isNot(SI1cond, SI2cond)))
      if ((SI1cond == SI2cond &&
           ((isa<Constant>(SI1tval) && isa<Constant>(SI2tval)) ||
            (isa<Constant>(SI1fval) && isa<Constant>(SI2fval)))) ||
          (SI1cond != SI2cond &&
           ((isa<Constant>(SI1tval) && isa<Constant>(SI2fval)) ||
            (isa<Constant>(SI1fval) && isa<Constant>(SI2tval))))

      ) {
        Value *tval = nullptr;
        Value *fval = nullptr;
        bool inverted = SI1cond != SI2cond;
        switch (cur->getOpcode()) {
        case Instruction::SIToFP:
          tval =
              B.CreateSIToFP(SI1tval, cur->getType(), "tval." + cur->getName());
          fval =
              B.CreateSIToFP(SI1fval, cur->getType(), "fval." + cur->getName());
          break;
        case Instruction::UIToFP:
          tval =
              B.CreateUIToFP(SI1tval, cur->getType(), "tval." + cur->getName());
          fval =
              B.CreateUIToFP(SI1fval, cur->getType(), "fval." + cur->getName());
          break;
        case Instruction::FAdd:
          tval = B.CreateFAddFMF(SI1tval, inverted ? SI2fval : SI2tval, cur,
                                 "tval." + cur->getName());
          fval = B.CreateFAddFMF(SI1fval, inverted ? SI2tval : SI2fval, cur,
                                 "fval." + cur->getName());
          break;
        case Instruction::FSub:
          tval = B.CreateFSubFMF(SI1tval, inverted ? SI2fval : SI2tval, cur,
                                 "tval." + cur->getName());
          fval = B.CreateFSubFMF(SI1fval, inverted ? SI2tval : SI2fval, cur,
                                 "fval." + cur->getName());
          break;
        case Instruction::FMul:
          tval = B.CreateFMulFMF(SI1tval, inverted ? SI2fval : SI2tval, cur,
                                 "tval." + cur->getName());
          fval = B.CreateFMulFMF(SI1fval, inverted ? SI2tval : SI2fval, cur,
                                 "fval." + cur->getName());
          break;
        case Instruction::Add:
          tval = B.CreateAdd(SI1tval, inverted ? SI2fval : SI2tval,
                             "tval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          fval = B.CreateAdd(SI1fval, inverted ? SI2tval : SI2fval,
                             "fval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          break;
        case Instruction::Sub:
          tval = B.CreateSub(SI1tval, inverted ? SI2fval : SI2tval,
                             "tval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          fval = B.CreateSub(SI1fval, inverted ? SI2tval : SI2fval,
                             "fval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          break;
        case Instruction::Mul:
          tval = B.CreateMul(SI1tval, inverted ? SI2fval : SI2tval,
                             "tval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          fval = B.CreateMul(SI1fval, inverted ? SI2tval : SI2fval,
                             "fval." + cur->getName(), cur->hasNoUnsignedWrap(),
                             cur->hasNoSignedWrap());
          break;
        case Instruction::ICmp:
        case Instruction::FCmp:
          tval = B.CreateCmp(cast<CmpInst>(cur)->getPredicate(), SI1tval,
                             inverted ? SI2fval : SI2tval,
                             "tval." + cur->getName());
          fval = B.CreateCmp(cast<CmpInst>(cur)->getPredicate(), SI1fval,
                             inverted ? SI2tval : SI2fval,
                             "fval." + cur->getName());
          break;
        default:
          llvm_unreachable("illegal opcode");
        }
        tval = pushcse(tval);
        fval = pushcse(fval);

        auto res = pushcse(
            B.CreateSelect(SI1cond, tval, fval, "selmerge." + cur->getName()));

        push(cur->getOperand(0));
        push(cur->getOperand(1));
        replaceAndErase(cur, res);
        return "BinopSelFuse";
      }
  }

  /*
  // and (i == c), (i != d) -> and (i == c) && (c != d)
  if (cur->getOpcode() == Instruction::And) {
    auto lhs = replace(cur->getOperand(0), cur->getOperand(1),
                       ConstantInt::getTrue(cur->getContext()));
    auto rhs = replace(cur->getOperand(1), cur->getOperand(0),
                       ConstantInt::getTrue(cur->getContext()));
    if (lhs != cur->getOperand(0) || rhs != cur->getOperand(1)) {
      auto res = pushcse(B.CreateAnd(lhs, rhs, "postand." + cur->getName()));
      replaceAndErase(cur, res);
      return "AndReplace2";
    }
  }
  */

  // and a, (or q, (not a)) -> and a q
  if (cur->getOpcode() == Instruction::And) {
    for (size_t i1 = 0; i1 < 2; i1++)
      if (auto inst2 = dyn_cast<Instruction>(cur->getOperand(1 - i1)))
        if (inst2->getOpcode() == Instruction::Or)
          for (size_t i2 = 0; i2 < 2; i2++)
            if (isNot(cur->getOperand(i1), inst2->getOperand(i2))) {
              auto q = inst2->getOperand(1 - i2);
              cur->setOperand(1 - i1, q);
              push(cur);
              push(q);
              push(inst2);
              push(cur->getOperand(i1));
              push(inst2->getOperand(i2));
              Q.insert(cur);
              for (auto U : cur->users())
                push(U);
              return "AndOrProp";
            }
  }

  // and (and a, b), a) -> and a, b
  if (cur->getOpcode() == Instruction::And) {
    for (size_t i1 = 0; i1 < 2; i1++)
      if (auto inst2 = dyn_cast<Instruction>(cur->getOperand(i1)))
        if (inst2->getOpcode() == Instruction::And)
          for (size_t i2 = 0; i2 < 2; i2++)
            if (inst2->getOperand(i2) == cur->getOperand(1 - i1)) {
              replaceAndErase(cur, inst2);
              return "AndAndProp";
            }
  }

  // or a, (and q, (not a)) -> and a q
  if (cur->getOpcode() == Instruction::And) {
    for (size_t i1 = 0; i1 < 2; i1++)
      if (auto inst2 = dyn_cast<Instruction>(cur->getOperand(1 - i1)))
        if (inst2->getOpcode() == Instruction::Or)
          for (size_t i2 = 0; i2 < 2; i2++)
            if (isNot(cur->getOperand(i1), inst2->getOperand(i2))) {
              auto q = inst2->getOperand(1 - i2);
              cur->setOperand(1 - i1, q);
              push(cur);
              push(q);
              push(inst2);
              push(cur->getOperand(i1));
              push(inst2->getOperand(i2));
              Q.insert(cur);
              for (auto U : cur->users())
                push(U);
              return "OrAndProp";
            }
  }

  // and ( (a +/- b) != c ), ( (d +/- b) != c )  -> and ( a != (c -/+ b) ), (
  // d != (c -/+ b) )
  //   also with or
  if (cur->getOpcode() == Instruction::And ||
      cur->getOpcode() == Instruction::Or) {
    for (auto cmpOp : {ICmpInst::ICMP_EQ, ICmpInst::ICMP_NE})
      for (auto interOp : {Instruction::Add, Instruction::Sub})
        if (auto cmp1 = dyn_cast<ICmpInst>(cur->getOperand(0)))
          if (auto cmp2 = dyn_cast<ICmpInst>(cur->getOperand(1)))
            for (size_t i1 = 0; i1 < 2; i1++)
              for (size_t i2 = 0; i2 < 2; i2++)
                if (cmp1->getOperand(1 - i1) == cmp2->getOperand(1 - i2) &&
                    cmp1->getPredicate() == cmpOp &&
                    cmp2->getPredicate() == cmpOp)
                  if (auto add1 = dyn_cast<Instruction>(cmp1->getOperand(i1)))
                    if (auto add2 = dyn_cast<Instruction>(cmp2->getOperand(i2)))
                      if (add1->getOpcode() == interOp &&
                          add2->getOpcode() == interOp)
                        for (size_t ia = 0; ia < 2; ia++)
                          if (add1->getOperand(ia) == add2->getOperand(ia)) {

                            auto b = add1->getOperand(ia);
                            auto c = cmp1->getOperand(1 - i1);
                            auto a = add1->getOperand(1 - ia);
                            auto d = add2->getOperand(1 - ia);

                            Value *res = nullptr;
                            if (interOp == Instruction::Add)
                              res = pushcse(B.CreateSub(ia == 0 ? b : c,
                                                        ia == 0 ? c : b));
                            else
                              res = pushcse(B.CreateAdd(ia == 0 ? b : c,
                                                        ia == 0 ? c : b));

                            auto lhs = pushcse(B.CreateCmp(cmpOp, a, res));
                            auto rhs = pushcse(B.CreateCmp(cmpOp, d, res));

                            Value *fres = nullptr;
                            if (cur->getOpcode() == Instruction::And)
                              fres = pushcse(B.CreateAnd(lhs, rhs));
                            else
                              fres = pushcse(B.CreateOr(lhs, rhs));

                            replaceAndErase(cur, fres);
                            return "AndLinearShift";
                          }
  }

  // and ( expr == c1 ), ( expr == c2 ) and c1 != c2  -> false
  if (cur->getOpcode() == Instruction::And) {
    for (auto cmpOp : {ICmpInst::ICMP_EQ})
      if (auto cmp1 = dyn_cast<ICmpInst>(cur->getOperand(0)))
        if (auto cmp2 = dyn_cast<ICmpInst>(cur->getOperand(1)))
          for (size_t i1 = 0; i1 < 2; i1++)
            for (size_t i2 = 0; i2 < 2; i2++)
              if (cmp1->getOperand(1 - i1) == cmp2->getOperand(1 - i2) &&
                  cmp1->getPredicate() == cmpOp &&
                  cmp2->getPredicate() == cmpOp) {
                auto c1 = SE.getSCEV(cmp1->getOperand(i1));
                auto c2 = SE.getSCEV(cmp2->getOperand(i2));
                auto m = SE.getMinusSCEV(c1, c2, SCEV::NoWrapMask);
                if (auto C = dyn_cast<SCEVConstant>(m)) {
                  // if c1 == c2 don't need the and they are equivalent
                  if (C->getValue()->isZero()) {
                    push(cmp1);
                    push(cmp2);
                    replaceAndErase(cur, cmp1);
                    return "AndEQExpr";
                  } else {
                    // if non one constant they must be distinct.
                    replaceAndErase(cur,
                                    ConstantInt::getFalse(cur->getContext()));
                    return "AndNEExpr";
                  }
                }
              }
  }

  // (a | b) == 0 -> a == 0 & b == 0
  if (auto icmp = dyn_cast<ICmpInst>(cur))
    if (icmp->getPredicate() == ICmpInst::ICMP_EQ &&
        cur->getType()->isIntegerTy(1))
      for (int i = 0; i < 2; i++)
        if (auto C = dyn_cast<ConstantInt>(icmp->getOperand(i)))
          if (C->isZero())
            if (auto z = dyn_cast<BinaryOperator>(icmp->getOperand(1 - i)))
              if (z->getOpcode() == BinaryOperator::Or) {
                auto a0 = pushcse(B.CreateICmpEQ(z->getOperand(0), C));
                auto b0 = pushcse(B.CreateICmpEQ(z->getOperand(1), C));
                auto res = pushcse(B.CreateAnd(a0, b0));
                push(z);
                push(icmp);
                replaceAndErase(cur, res);
                return "OrEQZero";
              }

  //  add (mul a b), (mul c, b) -> mul (add a, c), b
  if (cur->getOpcode() == Instruction::Sub ||
      cur->getOpcode() == Instruction::Add) {
    if (auto mul1 = dyn_cast<Instruction>(cur->getOperand(0)))
      if (auto mul2 = dyn_cast<Instruction>(cur->getOperand(1)))
        if ((mul1->getOpcode() == Instruction::Mul &&
             mul2->getOpcode() == Instruction::Mul) ||
            (mul1->getOpcode() == Instruction::FMul &&
             mul2->getOpcode() == Instruction::FMul && mul1->isFast() &&
             mul2->isFast() && cur->isFast())) {
          for (int i1 = 0; i1 < 2; i1++)
            for (int i2 = 0; i2 < 2; i2++) {
              if (mul1->getOperand(i1) == mul2->getOperand(i2)) {
                Value *res = nullptr;
                switch (cur->getOpcode()) {
                case Instruction::Add:
                  res = B.CreateAdd(mul1->getOperand(1 - i1),
                                    mul2->getOperand(1 - i2));
                  break;
                case Instruction::Sub:
                  res = B.CreateSub(mul1->getOperand(1 - i1),
                                    mul2->getOperand(1 - i2));
                  break;
                case Instruction::FAdd:
                  res = B.CreateFAddFMF(mul1->getOperand(1 - i1),
                                        mul2->getOperand(1 - i2), cur);
                  break;
                case Instruction::FSub:
                  res = B.CreateFSubFMF(mul1->getOperand(1 - i1),
                                        mul2->getOperand(1 - i2), cur);
                  break;
                default:
                  llvm_unreachable("Illegal opcode");
                }
                res = pushcse(res);
                Value *res2 = nullptr;
                if (cur->getType()->isIntegerTy())
                  res2 = B.CreateMul(
                      res, mul1->getOperand(i1), "",
                      mul1->hasNoUnsignedWrap() && mul1->hasNoUnsignedWrap(),
                      mul2->hasNoSignedWrap() && mul2->hasNoSignedWrap());
                else
                  res2 = B.CreateFMulFMF(res, mul1->getOperand(i1), cur);

                res2 = pushcse(res2);

                replaceAndErase(cur, res2);
                return "InvDistributive";
              }
            }
        }
  }

  // fadd (ext a), (ext b) -> ext (a + b)
  // fsub (ext a), (ext b) -> ext (a - b)
  // fmul (ext a), (ext b) -> ext (a * b)
  if (cur->getOpcode() == Instruction::FSub ||
      cur->getOpcode() == Instruction::FAdd ||
      cur->getOpcode() == Instruction::FMul ||
      cur->getOpcode() == Instruction::FNeg ||
      (isSum(cur) && callOperands(cast<CallBase>(cur)).size() == 2)) {
    auto opcode = cur->getOpcode();
    if (isSum(cur))
      opcode = Instruction::FAdd;
    auto Ty = B.getInt64Ty();
    SmallPtrSet<Instruction *, 1> temporaries;
    SmallVector<Instruction *, 1> precasts;
    Value *lhs = nullptr;

    Value *prelhs = (cur->getOpcode() == Instruction::FNeg)
                        ? ConstantFP::get(cur->getType(), 0.0)
                        : cur->getOperand(0);
    Value *prerhs = (cur->getOpcode() == Instruction::FNeg)
                        ? cur->getOperand(0)
                        : cur->getOperand(1);

    APInt minval(64, 0);
    APInt maxval(64, 0);
    if (auto C = dyn_cast<ConstantFP>(prelhs)) {
      APSInt Tmp(64);
      bool isExact = false;
      C->getValue().convertToInteger(Tmp, llvm::RoundingMode::TowardZero,
                                     &isExact);
      if (isExact || C->isZero()) {
        minval = maxval = Tmp;
        lhs = ConstantInt::get(Ty, Tmp);
      }
    }
    if (auto ext = dyn_cast<CastInst>(prelhs)) {
      if (ext->getOpcode() == Instruction::UIToFP ||
          ext->getOpcode() == Instruction::SIToFP) {
        precasts.push_back(ext);
        auto ity = cast<IntegerType>(ext->getOperand(0)->getType());
        bool md = false;
        if (auto I = dyn_cast<Instruction>(ext->getOperand(0)))
          if (auto MD = hasMetadata(I, LLVMContext::MD_range)) {
            md = true;
            minval =
                cast<ConstantInt>(
                    cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                    ->getValue()
                    .zextOrTrunc(64);
            maxval =
                cast<ConstantInt>(
                    cast<ConstantAsMetadata>(MD->getOperand(1))->getValue())
                    ->getValue()
                    .zextOrTrunc(64);
          }
        if (!md) {
          if (ext->getOpcode() == Instruction::UIToFP)
            maxval = APInt::getMaxValue(ity->getBitWidth()).zextOrTrunc(64);
          else {
            maxval =
                APInt::getSignedMaxValue(ity->getBitWidth()).zextOrTrunc(64);
            minval =
                APInt::getSignedMinValue(ity->getBitWidth()).zextOrTrunc(64);
          }
        }
        if (ext->getOperand(0)->getType() == Ty)
          lhs = ext->getOperand(0);
        else if (ity->getBitWidth() < Ty->getBitWidth()) {
          if (ext->getOpcode() == Instruction::UIToFP)
            lhs = B.CreateZExt(ext->getOperand(0), Ty);
          else
            lhs = B.CreateSExt(ext->getOperand(0), Ty);
          if (auto I = dyn_cast<Instruction>(lhs))
            if (I != ext->getOperand(0))
              temporaries.insert(I);
        }
      }
    }

    Value *rhs = nullptr;

    if (auto C = dyn_cast<ConstantFP>(prerhs)) {
      APSInt Tmp(64);
      bool isExact = false;
      C->getValue().convertToInteger(Tmp, llvm::RoundingMode::TowardZero,
                                     &isExact);
      if (isExact || C->isZero()) {
        rhs = ConstantInt::get(Ty, Tmp);
        switch (opcode) {
        case Instruction::FAdd:
          minval += Tmp;
          maxval += Tmp;
          break;
        case Instruction::FSub:
        case Instruction::FNeg:
          minval -= Tmp;
          maxval -= Tmp;
          break;
        case Instruction::FMul:
          minval *= Tmp;
          maxval *= Tmp;
          break;
        default:
          llvm_unreachable("Illegal opcode");
        }
      }
    }
    if (auto ext = dyn_cast<CastInst>(prerhs)) {
      if (ext->getOpcode() == Instruction::UIToFP ||
          ext->getOpcode() == Instruction::SIToFP) {
        precasts.push_back(ext);
        auto ity = cast<IntegerType>(ext->getOperand(0)->getType());
        bool md = false;
        APInt rhsMin(64, 0);
        APInt rhsMax(64, 0);
        if (auto I = dyn_cast<Instruction>(ext->getOperand(0)))
          if (auto MD = hasMetadata(I, LLVMContext::MD_range)) {
            md = true;
            rhsMin =
                cast<ConstantInt>(
                    cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                    ->getValue()
                    .zextOrTrunc(64);
            rhsMax =
                cast<ConstantInt>(
                    cast<ConstantAsMetadata>(MD->getOperand(1))->getValue())
                    ->getValue()
                    .zextOrTrunc(64);
          }
        if (!md) {
          if (ext->getOpcode() == Instruction::UIToFP) {
            rhsMax = APInt::getMaxValue(ity->getBitWidth()).zextOrTrunc(64);
            rhsMin = APInt(64, 0);
          } else {
            rhsMax =
                APInt::getSignedMaxValue(ity->getBitWidth()).zextOrTrunc(64);
            rhsMin =
                APInt::getSignedMinValue(ity->getBitWidth()).zextOrTrunc(64);
          }
        }
        switch (opcode) {
        case Instruction::FAdd:
          minval += rhsMin;
          maxval += rhsMax;
          break;
        case Instruction::FSub:
        case Instruction::FNeg:
          minval -= rhsMax;
          maxval -= rhsMin;
          break;
        case Instruction::FMul: {
          auto minf = [&](APInt a, APInt b) { return a.sle(b) ? a : b; };
          auto maxf = [&](APInt a, APInt b) { return a.sle(b) ? b : b; };
          minval = minf(
              minval * rhsMin,
              minf(minval * rhsMax, minf(maxval * rhsMin, maxval * rhsMax)));
          maxval = maxf(
              minval * rhsMin,
              maxf(minval * rhsMax, maxf(maxval * rhsMin, maxval * rhsMax)));
          break;
        }
        default:
          llvm_unreachable("Illegal opcode");
        }
        if (ext->getOperand(0)->getType() == Ty)
          rhs = ext->getOperand(0);
        else if (ity->getBitWidth() < Ty->getBitWidth()) {
          if (ext->getOpcode() == Instruction::UIToFP)
            rhs = B.CreateZExt(ext->getOperand(0), Ty);
          else
            rhs = B.CreateSExt(ext->getOperand(0), Ty);
          if (auto I = dyn_cast<Instruction>(rhs))
            if (I != ext->getOperand(0))
              temporaries.insert(I);
        }
      }
    }

    if (lhs && rhs) {
      Value *res = nullptr;
      if (temporaries.count(dyn_cast<Instruction>(lhs)))
        lhs = pushcse(lhs);
      if (temporaries.count(dyn_cast<Instruction>(rhs)))
        rhs = pushcse(rhs);
      switch (opcode) {
      case Instruction::FAdd:
        res = B.CreateAdd(lhs, rhs, "", false, true);
        break;
      case Instruction::FSub:
      case Instruction::FNeg:
        res = B.CreateSub(lhs, rhs, "", false, true);
        break;
      case Instruction::FMul:
        res = B.CreateMul(lhs, rhs, "", false, true);
        break;
      default:
        llvm_unreachable("Illegal opcode");
      }
      res = pushcse(res);
      for (auto I : precasts)
        push(I);
      /*
      if (auto I = dyn_cast<Instruction>(res)) {
        Q.insert(I);
        Metadata *vals[] = {(Metadata *)ConstantAsMetadata::get(
                                ConstantInt::get(Ty, minval)),
                            (Metadata *)ConstantAsMetadata::get(
                                ConstantInt::get(Ty, maxval))};
        I->setMetadata(LLVMContext::MD_range,
                       MDNode::get(I->getContext(), vals));
      }
      */
      auto ext = pushcse(B.CreateSIToFP(res, cur->getType()));
      replaceAndErase(cur, ext);
      return "BinopExtToExtBinop";

    } else {
      for (auto I : temporaries)
        I->eraseFromParent();
    }
  }

  // select(cond, const1, b) ?= const2 -> select(cond, const1 ?= const2, b ?=
  // const2)
  if (auto fcmp = dyn_cast<FCmpInst>(cur))
    for (int i = 0; i < 2; i++)
      if (auto const2 = dyn_cast<Constant>(fcmp->getOperand(i)))
        if (auto sel = dyn_cast<SelectInst>(fcmp->getOperand(1 - i)))
          if (isa<Constant>(sel->getTrueValue()) ||
              isa<Constant>(sel->getFalseValue())) {
            auto tval = pushcse(B.CreateFCmp(fcmp->getPredicate(),
                                             sel->getTrueValue(), const2));
            auto fval = pushcse(B.CreateFCmp(fcmp->getPredicate(),
                                             sel->getFalseValue(), const2));
            auto res = pushcse(B.CreateSelect(sel->getCondition(), tval, fval));
            replaceAndErase(cur, res);
            return "FCmpSelectConst";
          }

  // mul (mul a, const), b:not_sparse_or_const -> mul (mul a, b), const
  //   note we avoid the case where b = (mul a, const) since otherwise
  //   we create an infinite recursion
  //   and also we make sure b isn't sparse, since sparse is the first
  //   precedence for pushing, then constant, then others
  if (cur->getOpcode() == Instruction::FMul)
    if (cur->isFast() && cur->getOperand(0) != cur->getOperand(1))
      for (auto ic = 0; ic < 2; ic++)
        if (auto mul = dyn_cast<Instruction>(cur->getOperand(ic)))
          if (mul->getOpcode() == Instruction::FMul && mul->isFast()) {
            auto b = cur->getOperand(1 - ic);
            if (!isa<Constant>(b) && !directlySparse(b)) {

              for (int i = 0; i < 2; i++)
                if (auto C = dyn_cast<Constant>(mul->getOperand(i))) {
                  auto n0 =
                      pushcse(B.CreateFMulFMF(mul->getOperand(1 - i), b, mul));
                  auto n1 = pushcse(B.CreateFMulFMF(n0, C, cur));
                  push(mul);

                  replaceAndErase(cur, n1);
                  return "MulMulConst";
                }
            }
          }

  // (mul c, a) +/- (mul c, b) -> mul c, (a +/- b)
  if (cur->getOpcode() == Instruction::FAdd ||
      cur->getOpcode() == Instruction::FSub) {
    if (auto mul1 = dyn_cast<BinaryOperator>(cur->getOperand(0))) {
      if (mul1->getOpcode() == Instruction::FMul && mul1->isFast()) {
        if (auto mul2 = dyn_cast<BinaryOperator>(cur->getOperand(1))) {
          if (mul2->getOpcode() == Instruction::FMul && mul2->isFast()) {
            for (int i = 0; i < 2; i++) {
              for (int j = 0; j < 2; j++) {
                if (mul1->getOperand(i) == mul2->getOperand(j)) {
                  auto c = mul1->getOperand(i);
                  auto a = mul1->getOperand(1 - i);
                  auto b = mul2->getOperand(1 - j);
                  Value *intermediate = nullptr;

                  if (cur->getOpcode() == Instruction::FAdd)
                    intermediate = pushcse(B.CreateFAddFMF(a, b, cur));
                  else
                    intermediate = pushcse(B.CreateFSubFMF(a, b, cur));

                  auto res = pushcse(B.CreateFMulFMF(c, intermediate, cur));
                  push(mul1);
                  push(mul2);
                  replaceAndErase(cur, res);
                  return "FAddMulConstMulConst";
                }
              }
            }
          }
        }
      }
    }
  }

  // fmul a, (sitofp (imul c:const, b)) -> fmul (fmul (a, (sitofp c))),
  // (sitofp b)

  if (cur->getOpcode() == Instruction::FMul && cur->isFast()) {
    for (int i = 0; i < 2; i++)
      if (auto z = dyn_cast<Instruction>(cur->getOperand(i)))
        if (isa<SIToFPInst>(z) || isa<UIToFPInst>(z))
          if (auto imul = dyn_cast<BinaryOperator>(z->getOperand(0)))
            if (imul->getOpcode() == Instruction::Mul)
              for (int j = 0; j < 2; j++)
                if (auto c = dyn_cast<Constant>(imul->getOperand(j))) {
                  auto b = imul->getOperand(1 - j);
                  auto a = cur->getOperand(1 - i);

                  auto c_fp = pushcse(B.CreateSIToFP(c, cur->getType()));
                  auto b_fp = pushcse(B.CreateSIToFP(b, cur->getType()));
                  auto n_mul = pushcse(B.CreateFMulFMF(a, c_fp, cur));
                  auto res = pushcse(
                      B.CreateFMulFMF(n_mul, b_fp, cur, cur->getName()));
                  push(imul);
                  push(z);
                  replaceAndErase(cur, res);
                  return "FMulIMulConstRotate";
                }
  }

  if (cur->getOpcode() == Instruction::FDiv) {
    Value *prelhs = cur->getOperand(0);
    Value *b = cur->getOperand(1);

    // fdiv (sitofp a), b -> select (a == 0), 0 [ (fdiv 1 / b) * sitofp a]
    if (auto ext = dyn_cast<CastInst>(prelhs)) {
      if (ext->getOpcode() == Instruction::UIToFP ||
          ext->getOpcode() == Instruction::SIToFP) {
        push(ext);

        Value *condition = pushcse(
            B.CreateICmpEQ(ext->getOperand(0),
                           ConstantInt::get(ext->getOperand(0)->getType(), 0),
                           "sdivcmp." + cur->getName()));

        Value *fdiv = pushcse(
            B.CreateFMulFMF(pushcse(B.CreateFDivFMF(
                                ConstantFP::get(cur->getType(), 1.0), b, cur)),
                            ext, cur));

        Value *sel = pushcse(
            B.CreateSelect(condition, ConstantFP::get(cur->getType(), 0.0),
                           fdiv, "sfdiv." + cur->getName()));

        replaceAndErase(cur, sel);
        return "FDivSIToFPProp";
      }
    }
    // fdiv (select c, 0, a), b -> select c, 0 (fdiv a, b)
    if (auto SI = dyn_cast<SelectInst>(prelhs)) {
      auto tvalC = dyn_cast<ConstantFP>(SI->getTrueValue());
      auto fvalC = dyn_cast<ConstantFP>(SI->getFalseValue());
      if ((tvalC && tvalC->isZero()) || (fvalC && fvalC->isZero())) {
        push(SI);
        auto ntval =
            (tvalC && tvalC->isZero())
                ? tvalC
                : pushcse(B.CreateFDivFMF(SI->getTrueValue(), b, cur,
                                          "sfdiv2_t." + cur->getName()));
        auto nfval =
            (fvalC && fvalC->isZero())
                ? fvalC
                : pushcse(B.CreateFDivFMF(SI->getFalseValue(), b, cur,
                                          "sfdiv2_f." + cur->getName()));

        // Work around bad fdivfmf, fixed in LLVM 16+
        // https://github.com/llvm/llvm-project/commit/4f3b1c6dd6ef6c7b5bb79f058e3b7ba4bcdf4566
#if LLVM_VERSION_MAJOR < 16
        for (auto v : {ntval, nfval})
          if (auto I = dyn_cast<Instruction>(v))
            I->setFastMathFlags(cur->getFastMathFlags());
#endif

        auto res = pushcse(B.CreateSelect(SI->getCondition(), ntval, nfval,
                                          "sfdiv2." + cur->getName()));

        replaceAndErase(cur, res);
        return "FDivSelectProp";
      }
    }
  }

  // div (mul a:not_sparse, b:is_sparse), c -> mul (div, a, c), b:is_sparse
  if (cur->getOpcode() == Instruction::FDiv) {
    auto c = cur->getOperand(1);
    if (auto z = dyn_cast<BinaryOperator>(cur->getOperand(0))) {
      if (z->getOpcode() == Instruction::FMul) {
        for (int i = 0; i < 2; i++) {

          Value *a = z->getOperand(i);
          Value *b = z->getOperand(1 - i);
          if (directlySparse(a))
            continue;
          if (!directlySparse(b))
            continue;

          Value *inner_fdiv = pushcse(B.CreateFDivFMF(a, c, cur));
          Value *outer_fmul = pushcse(B.CreateFMulFMF(inner_fdiv, b, z));
          push(z);
          replaceAndErase(cur, outer_fmul);
          return "FDivFMulSparseProp";
        }
      }
    }
  }

  if (cur->getOpcode() == Instruction::FMul)
    for (int i = 0; i < 2; i++) {

      Value *prelhs = cur->getOperand(i);
      Value *b = cur->getOperand(1 - i);

      // fmul (fmul x:constant, y):z, b:constant .
      if (isa<Constant>(b))
        if (auto z = dyn_cast<BinaryOperator>(prelhs)) {
          if (z->getOpcode() == Instruction::FMul) {
            for (int j = 0; j < 2; j++) {
              auto x = z->getOperand(i);
              if (!isa<Constant>(x))
                continue;
              auto y = z->getOperand(1 - i);
              Value *inner_fmul = pushcse(B.CreateFMulFMF(x, b, cur));
              Value *outer_fmul = pushcse(B.CreateFMulFMF(inner_fmul, y, z));
              push(z);
              replaceAndErase(cur, outer_fmul);
              return "FMulFMulConstantReorder";
            }
          }
        }

      auto integralFloat = [](Value *z) {
        if (auto C = dyn_cast<ConstantFP>(z)) {
          APSInt Tmp(64);
          bool isExact = false;
          C->getValue().convertToInteger(Tmp, llvm::RoundingMode::TowardZero,
                                         &isExact);
          if (isExact || C->isZero()) {
            return true;
          }
        }
        return false;
      };

      // fmul (fmul x:sparse, y):z, b
      //   1) If x and y are both sparse, do nothing and let the inner fmul be
      //      simplified into a single sparse instruction. Thus, we may assume
      //      y is not sparse.
      //   2) if b is sparse, swap it to be fmul (fmul x, b), y  so the inner
      //      sparsity can be simplified.
      //   3) otherwise b is not sparse and we should push the sparsity to
      //      be the outermost value
      if (auto z = dyn_cast<BinaryOperator>(prelhs)) {
        if (z->getOpcode() == Instruction::FMul) {
          for (int j = 0; j < 2; j++) {
            auto x = z->getOperand(j);
            if (!directlySparse(x))
              continue;
            auto y = z->getOperand(1 - j);
            if (directlySparse(y))
              continue;

            if (directlySparse(b) || integralFloat(b)) {
              push(z);
              Value *inner_fmul = pushcse(
                  B.CreateFMulFMF(x, b, cur, "mulisr." + cur->getName()));
              Value *outer_fmul = pushcse(
                  B.CreateFMulFMF(inner_fmul, y, z, "mulisr." + z->getName()));
              replaceAndErase(cur, outer_fmul);
              return "FMulFMulSparseReorder";
            } else {
              push(z);
              Value *inner_fmul = pushcse(
                  B.CreateFMulFMF(y, b, cur, "mulisp." + cur->getName()));
              Value *outer_fmul = pushcse(
                  B.CreateFMulFMF(inner_fmul, x, z, "mulisp." + z->getName()));
              replaceAndErase(cur, outer_fmul);
              return "FMulFMulSparsePush";
            }
          }
        }
      }

      /*
      auto contains = [](MDNode *MD, Value *V) {
        if (!MD)
          return false;
        for (auto &op : MD->operands()) {
          auto V2 = cast<ValueAsMetadata>(op)->getValue();
          if (V == V2)
            return true;
        }
        return false;
      };

      // fmul (sitofp a), b -> select (a == 0), 0 [noprop fmul ( sitofp a), b]
      if (true || !contains(hasMetadata(cur, "enzyme_fmulnoprop"), prelhs))
        if (auto ext = dyn_cast<CastInst>(prelhs)) {
          if (ext->getOpcode() == Instruction::UIToFP ||
              ext->getOpcode() == Instruction::SIToFP) {
            push(ext);

            Value *condition = pushcse(B.CreateICmpEQ(
                ext->getOperand(0),
                ConstantInt::get(ext->getOperand(0)->getType(), 0),
                "mulcsicmp." + cur->getName()));

            Value *fmul = pushcse(B.CreateFMulFMF(ext, b, cur));
            if (auto I = dyn_cast<Instruction>(fmul)) {
              SmallVector<Metadata *, 1> nodes;
              if (auto MD = hasMetadata(cur, "enzyme_fmulnoprop")) {
                for (auto &M : MD->operands()) {
                  nodes.push_back(M.get());
                }
              }
              nodes.push_back(ValueAsMetadata::get(ext));
              I->setMetadata("enzyme_fmulnoprop",
                             MDNode::get(I->getContext(), nodes));
            }

            Value *sel = pushcse(
                B.CreateSelect(condition, ConstantFP::get(cur->getType(),
      0.0), fmul, "mulcsi." + cur->getName()));

            replaceAndErase(cur, sel);
            return "FMulSIToFPProp";
          }
        }
      */

      // fmul (select c, 0, a), b -> select c, 0 (fmul a, b)
      if (auto SI = dyn_cast<SelectInst>(prelhs)) {
        auto tvalC = dyn_cast<ConstantFP>(SI->getTrueValue());
        auto fvalC = dyn_cast<ConstantFP>(SI->getFalseValue());
        if ((tvalC && tvalC->isZero()) || (fvalC && fvalC->isZero())) {
          push(SI);
          auto ntval =
              (tvalC && tvalC->isZero())
                  ? tvalC
                  : pushcse(B.CreateFMulFMF(SI->getTrueValue(), b, cur));
          auto nfval =
              (fvalC && fvalC->isZero())
                  ? fvalC
                  : pushcse(B.CreateFMulFMF(SI->getFalseValue(), b, cur));
          auto res = pushcse(B.CreateSelect(SI->getCondition(), ntval, nfval,
                                            "mulsi." + cur->getName()));

          replaceAndErase(cur, res);
          return "FMulSelectProp";
        }
      }
    }

  if (auto icmp = dyn_cast<BinaryOperator>(cur)) {
    if (icmp->getOpcode() == Instruction::Xor) {
      for (int i = 0; i < 2; i++) {
        if (auto C = dyn_cast<ConstantInt>(icmp->getOperand(i))) {
          // !(cmp a, b) -> inverse(cmp), a, b
          if (C->isOne()) {
            if (auto scmp = dyn_cast<CmpInst>(icmp->getOperand(1 - i))) {
              auto next = pushcse(
                  B.CreateCmp(scmp->getInversePredicate(), scmp->getOperand(0),
                              scmp->getOperand(1), "not." + scmp->getName()));
              replaceAndErase(cur, next);
              return "NotCmp";
            }
          }
        }
      }
    }
  }

  // select cmp, (ext tval), (ext fval) ->  (cmp & tval) | (!cmp & fval)
  if (auto SI = dyn_cast<SelectInst>(cur)) {

    Value *trueVal = nullptr;
    if (auto C = dyn_cast<ConstantFP>(SI->getTrueValue())) {
      if (C->isZero()) {
        trueVal = ConstantInt::getFalse(SI->getContext());
      }
      if (C->isExactlyValue(1.0)) {
        trueVal = ConstantInt::getTrue(SI->getContext());
      }
    }
    if (auto ext = dyn_cast<CastInst>(SI->getTrueValue())) {
      if (ext->getOperand(0)->getType()->isIntegerTy(1))
        trueVal = ext->getOperand(0);
    }
    Value *falseVal = nullptr;
    if (auto C = dyn_cast<ConstantFP>(SI->getFalseValue())) {
      if (C->isZero()) {
        falseVal = ConstantInt::getFalse(SI->getContext());
      }
      if (C->isExactlyValue(1.0)) {
        falseVal = ConstantInt::getTrue(SI->getContext());
      }
    }
    if (auto ext = dyn_cast<CastInst>(SI->getFalseValue())) {
      if (ext->getOperand(0)->getType()->isIntegerTy(1))
        falseVal = ext->getOperand(0);
    }
    if (trueVal && falseVal) {
      auto ncmp1 = pushcse(B.CreateAnd(SI->getCondition(), trueVal));
      auto notV = pushcse(B.CreateNot(SI->getCondition()));
      auto ncmp2 = pushcse(B.CreateAnd(notV, falseVal));
      auto ori = pushcse(B.CreateOr(ncmp1, ncmp2));
      auto ext = pushcse(B.CreateUIToFP(ori, SI->getType()));
      replaceAndErase(cur, ext);
      return "SelectI1Ext";
    }
  }
  // select cmp, (i1 tval), (i1 fval) ->  (cmp & tval) | (!cmp & fval)
  if (cur->getType()->isIntegerTy(1))
    if (auto SI = dyn_cast<SelectInst>(cur)) {
      auto ncmp1 = pushcse(B.CreateAnd(SI->getCondition(), SI->getTrueValue()));
      auto notV = pushcse(B.CreateNot(SI->getCondition()));
      auto ncmp2 = pushcse(B.CreateAnd(notV, SI->getFalseValue()));
      auto ori = pushcse(B.CreateOr(ncmp1, ncmp2));
      replaceAndErase(cur, ori);
      return "SelectI1";
    }

  if (auto PN = dyn_cast<PHINode>(cur)) {
    B.SetInsertPoint(PN->getParent()->getFirstNonPHI());
    if (SE.isSCEVable(PN->getType())) {
      auto S = SE.getSCEV(PN);

      bool legal = false;
      if (auto SV = dyn_cast<SCEVUnknown>(S)) {
        auto val = SV->getValue();
        legal |= isa<Constant>(val) || isa<Argument>(val);
        if (auto I = dyn_cast<Instruction>(val)) {
          auto L = LI.getLoopFor(I->getParent());
          if ((!L || L->getCanonicalInductionVariable() != I) && I != PN)
            legal = true;
        }
      }
      if (isa<SCEVAddRecExpr>(S)) {
        auto L = LI.getLoopFor(PN->getParent());
        assert(L);
        if (L->getCanonicalInductionVariable() != PN)
          legal = true;
      }

      if (legal) {
        for (auto U : cur->users()) {
          push(U);
        }
        auto point = PN->getParent()->getFirstNonPHI();
        auto tmp = cast<PHINode>(pushcse(B.CreatePHI(cur->getType(), 1)));
        cur->replaceAllUsesWith(tmp);
        cur->eraseFromParent();

        Value *newIV = nullptr;
        {
          SCEVExpander Exp(SE, DL, "sparseenzyme");
          // We place that at first non phi as it may produce a non-phi
          // instruction and must thus be expanded after all phi's
          newIV = Exp.expandCodeFor(S, tmp->getType(), point);
          // sadly this doesn't exist on 11
          for (auto I : Exp.getAllInsertedInstructions())
            Q.insert(I);
        }

        tmp->replaceAllUsesWith(newIV);
        tmp->eraseFromParent();
        return "InductVarSCEV";
      }
    }
    // phi a, a -> a
    {
      bool legal = true;
      for (size_t i = 1; i < PN->getNumIncomingValues(); i++) {
        auto v = PN->getIncomingValue(i);
        if (v != PN->getIncomingValue(0)) {
          legal = false;
          break;
        }
      }
      if (legal) {
        auto val = PN->getIncomingValue(0);
        replaceAndErase(cur, val);
        return "PhiMerge";
      }
    }
    // phi (idx=0) ? b, a, a -> select (idx == 0), b, a
    if (auto L = LI.getLoopFor(PN->getParent()))
      if (L->getHeader() == PN->getParent())
        if (auto idx = L->getCanonicalInductionVariable())
          if (auto PH = L->getLoopPreheader()) {
            bool legal = idx != PN;
            auto ph_idx = PN->getBasicBlockIndex(PH);
            assert(ph_idx >= 0);
            for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
              if ((int)i == ph_idx)
                continue;
              auto v = PN->getIncomingValue(i);
              if (v != PN->getIncomingValue(1 - ph_idx)) {
                legal = false;
                break;
              }
              // The given var must dominate the loop
              if (isa<Constant>(v))
                continue;
              if (isa<Argument>(v))
                continue;
              // exception for the induction itself, which we handle specially
              if (v == idx)
                continue;
              auto I = cast<Instruction>(v);
              if (!DT.dominates(I, PN)) {
                legal = false;
                break;
              }
            }
            if (legal) {
              auto val = PN->getIncomingValue(1 - ph_idx);
              push(val);
              if (val == idx) {
                val = pushcse(
                    B.CreateSub(idx, ConstantInt::get(idx->getType(), 1)));
              }

              auto val2 = PN->getIncomingValue(ph_idx);
              push(val2);

              auto c0 = ConstantInt::get(idx->getType(), 0);
              // if (val2 == c0 && PN->getIncomingValue(1 - ph_idx) == idx) {
              //  val = B.CreateBinaryIntrinsic(Intrinsic::umax, c0, val);
              //} else {
              auto eq = pushcse(B.CreateICmpEQ(idx, c0));
              val = pushcse(
                  B.CreateSelect(eq, val2, val, "phisel." + cur->getName()));
              //}

              replaceAndErase(cur, val);
              return "PhiLoop0Sel";
            }
          }
    // phi (sitofp a), (sitofp b) -> sitofp (phi a, b)
    {
      SmallVector<Value *, 1> negOps;
      SmallVector<Instruction *, 1> prevNegOps;
      bool legal = true;
      for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
        auto v = PN->getIncomingValue(i);
        if (auto C = dyn_cast<ConstantFP>(v)) {
          APSInt Tmp(64);
          bool isExact = false;
          C->getValue().convertToInteger(Tmp, llvm::RoundingMode::TowardZero,
                                         &isExact);
          if (isExact || C->isZero()) {
            negOps.push_back(ConstantInt::get(B.getInt64Ty(), Tmp));
            continue;
          }
        }
        if (auto fneg = dyn_cast<Instruction>(v)) {
          if (fneg->getOpcode() == Instruction::SIToFP &&
              cast<IntegerType>(fneg->getOperand(0)->getType())
                      ->getBitWidth() == 64) {
            negOps.push_back(fneg->getOperand(0));
            prevNegOps.push_back(fneg);
            continue;
          }
        }
        legal = false;
      }
      if (legal) {
        auto PN2 = cast<PHINode>(
            pushcse(B.CreatePHI(B.getInt64Ty(), PN->getNumIncomingValues())));
        PN2->takeName(PN);
        for (auto val : llvm::enumerate(negOps))
          PN2->addIncoming(val.value(), PN->getIncomingBlock(val.index()));

        push(PN2);

        auto fneg = pushcse(B.CreateSIToFP(PN2, PN->getType()));

        for (auto I : prevNegOps)
          push(I);
        replaceAndErase(cur, fneg);
        return "PhiSIToFP";
      }
    }
    // phi (fneg a), (fneg b) -> fneg (phi a, b)
    {
      SmallVector<Value *, 1> negOps;
      SmallVector<Instruction *, 1> prevNegOps;
      bool legal = true;
      bool hasNeg = false;
      for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
        auto v = PN->getIncomingValue(i);
        if (auto C = dyn_cast<ConstantFP>(v)) {
          negOps.push_back(C->isZero() ? C : pushcse(B.CreateFNeg(C)));
          continue;
        }
        if (auto fneg = dyn_cast<Instruction>(v)) {
          if (fneg->getOpcode() == Instruction::FNeg) {
            negOps.push_back(fneg->getOperand(0));
            prevNegOps.push_back(fneg);
            continue;
          }
        }
        legal = false;
      }
      if (legal && hasNeg) {
        for (auto val : llvm::enumerate(negOps))
          PN->setIncomingValue(val.index(), val.value());

        push(PN);

        auto fneg = pushcse(B.CreateFNeg(PN));

        for (auto &U : cur->uses()) {
          if (U.getUser() == fneg)
            continue;
          push(U.getUser());
          U.set(fneg);
        }
        for (auto I : prevNegOps)
          push(I);
        return "PhiFNeg";
      }
    }
    // phi (neg a), (neg b) -> neg (phi a, b)
    {
      SmallVector<Value *, 1> negOps;
      SmallVector<Instruction *, 1> prevNegOps;
      bool legal = true;
      bool hasNeg = false;
      for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
        auto v = PN->getIncomingValue(i);
        if (auto C = dyn_cast<ConstantInt>(v)) {
          negOps.push_back(pushcse(B.CreateNeg(C)));
          continue;
        }
        if (auto fneg = dyn_cast<BinaryOperator>(v)) {
          if (auto CI = dyn_cast<ConstantInt>(fneg->getOperand(0)))
            if (fneg->getOpcode() == Instruction::Sub && CI->isZero()) {
              negOps.push_back(fneg->getOperand(1));
              prevNegOps.push_back(fneg);
              hasNeg = true;
              continue;
            }
        }
        legal = false;
      }
      if (legal && hasNeg) {
        for (auto val : llvm::enumerate(negOps))
          PN->setIncomingValue(val.index(), val.value());

        push(PN);

        auto fneg = pushcse(B.CreateNeg(PN));

        for (auto &U : cur->uses()) {
          if (U.getUser() == fneg)
            continue;
          push(U.getUser());
          U.set(fneg);
        }
        for (auto I : prevNegOps)
          push(I);
        return "PHINeg";
      }
    }
    // p = phi (mul a, c), (mul b, d) -> mul (phi a, b), (phi c, d)    if
    // a,b,c != p
    {
      for (auto code :
           {(unsigned)Instruction::Mul, (unsigned)Instruction::Sub,
            (unsigned)Instruction::Add, (unsigned)Instruction::ZExt,
            (unsigned)Instruction::UIToFP, (unsigned)Instruction::ICmp,
            (unsigned)Instruction::FMul, (unsigned)Instruction::Or,
            (unsigned)Instruction::And}) {
        SmallVector<Value *, 1> lhsOps;
        SmallVector<Value *, 1> rhsOps;
        SmallVector<Instruction *, 1> prevOps;
        bool legal = true;
        bool fast = false;
        bool NUW = false;
        bool NSW = false;
        size_t numOps = 0;
        std::optional<llvm::CmpInst::Predicate> cmpPredicate;
        switch (code) {
        case Instruction::FMul:
        case Instruction::FSub:
        case Instruction::FAdd:
          fast = true;
          numOps = 2;
          break;
        case Instruction::Mul:
        case Instruction::Add:
          NUW = NSW = true;
          numOps = 2;
          break;
        case Instruction::Sub:
          NSW = true;
          numOps = 2;
          break;
        case Instruction::ICmp:
        case Instruction::FCmp:
        case Instruction::Or:
        case Instruction::And:
          numOps = 2;
          break;
        case Instruction::ZExt:
        case Instruction::UIToFP:
          numOps = 1;
          break;
        default:;
          llvm_unreachable("unknown opcode");
        }
        bool changed = false;
        for (size_t i = 0; i < PN->getNumIncomingValues(); i++) {
          auto v = PN->getIncomingValue(i);
          if (auto C = dyn_cast<ConstantInt>(v)) {
            if (code == Instruction::ZExt) {
              lhsOps.push_back(ConstantInt::getFalse(C->getContext()));
              continue;
            } else if (C->isZero()) {
              rhsOps.push_back(C);
              lhsOps.push_back(C);
              continue;
            }
          }
          if (auto C = dyn_cast<ConstantFP>(v)) {
            if (code == Instruction::UIToFP) {
              if (C->isZero()) {
                lhsOps.push_back(ConstantInt::getFalse(C->getContext()));
              }
            } else if (code == Instruction::FMul || code == Instruction::FSub ||
                       code == Instruction::FAdd) {
              if (C->isZero()) {
                rhsOps.push_back(C);
                lhsOps.push_back(C);
                continue;
              }
            }
          }
          if (auto fneg = dyn_cast<Instruction>(v)) {
            if (fneg->getOpcode() == code) {
              switch (code) {
              case Instruction::FMul:
              case Instruction::FSub:
              case Instruction::FAdd:
                fast &= fneg->isFast();
                if (fneg->getOperand(0) == PN)
                  legal = false;
                if (fneg->getOperand(1) == PN)
                  legal = false;
                lhsOps.push_back(fneg->getOperand(0));
                rhsOps.push_back(fneg->getOperand(1));
                break;
              case Instruction::Mul:
              case Instruction::Sub:
              case Instruction::Add:
                NUW &= fneg->hasNoUnsignedWrap();
                NSW &= fneg->hasNoSignedWrap();
                if (fneg->getOperand(0) == PN)
                  legal = false;
                if (fneg->getOperand(1) == PN)
                  legal = false;
                lhsOps.push_back(fneg->getOperand(0));
                rhsOps.push_back(fneg->getOperand(1));
                break;
              case Instruction::Or:
              case Instruction::And:
                if (fneg->getOperand(0) == PN)
                  legal = false;
                if (fneg->getOperand(1) == PN)
                  legal = false;
                lhsOps.push_back(fneg->getOperand(0));
                rhsOps.push_back(fneg->getOperand(1));
                break;
              case Instruction::ICmp:
              case Instruction::FCmp:
                if (fneg->getOperand(0) == PN)
                  legal = false;
                if (fneg->getOperand(1) == PN)
                  legal = false;
                if (cmpPredicate) {
                  if (*cmpPredicate != cast<CmpInst>(fneg)->getPredicate())
                    legal = false;
                } else {
                  cmpPredicate = cast<CmpInst>(fneg)->getPredicate();
                }
                lhsOps.push_back(fneg->getOperand(0));
                rhsOps.push_back(fneg->getOperand(1));
                break;
              case Instruction::ZExt:
              case Instruction::UIToFP:
                if (cast<IntegerType>(fneg->getOperand(0)->getType())
                        ->getBitWidth() != 1)
                  legal = false;
                lhsOps.push_back(fneg->getOperand(0));
                break;
              default:
                llvm_unreachable("unhandled opcode");
              }
              prevOps.push_back(fneg);
              changed = true;
              continue;
            }
          }
          legal = false;
        }

        int preheader_fix = -1;

        if (code == Instruction::ICmp || code == Instruction::FCmp) {
          if (!cmpPredicate)
            legal = false;
          auto L = LI.getLoopFor(PN->getParent());
          if (legal && L && L->getLoopPreheader() &&
              L->getCanonicalInductionVariable() &&
              L->getHeader() == PN->getParent()) {
            auto ph_idx = PN->getBasicBlockIndex(L->getLoopPreheader());
            if (isa<ConstantInt>(PN->getIncomingValue(ph_idx))) {
              lhsOps[ph_idx] =
                  Constant::getNullValue(lhsOps[1 - ph_idx]->getType());
              rhsOps[ph_idx] =
                  Constant::getNullValue(rhsOps[1 - ph_idx]->getType());
              preheader_fix = ph_idx;
            }
          }
          for (auto v : lhsOps)
            if (v->getType() != lhsOps[0]->getType())
              legal = false;
          for (auto v : rhsOps)
            if (v->getType() != rhsOps[0]->getType())
              legal = false;
        }

        if (legal && changed) {
          auto lhsPN = cast<PHINode>(pushcse(
              B.CreatePHI(lhsOps[0]->getType(), PN->getNumIncomingValues())));
          PHINode *rhsPN = nullptr;
          if (numOps == 2)
            rhsPN = cast<PHINode>(pushcse(
                B.CreatePHI(rhsOps[0]->getType(), PN->getNumIncomingValues())));

          for (auto val : llvm::enumerate(lhsOps))
            lhsPN->addIncoming(val.value(), PN->getIncomingBlock(val.index()));

          if (numOps == 2) {
            for (auto val : llvm::enumerate(rhsOps))
              rhsPN->addIncoming(val.value(),
                                 PN->getIncomingBlock(val.index()));
          }

          Value *fneg = nullptr;
          switch (code) {
          case Instruction::FMul:
            fneg = B.CreateFMul(lhsPN, rhsPN);
            if (auto I = dyn_cast<Instruction>(fneg))
              I->setFast(fast);
            break;
          case Instruction::FAdd:
            fneg = B.CreateFAdd(lhsPN, rhsPN);
            if (auto I = dyn_cast<Instruction>(fneg))
              I->setFast(fast);
            break;
          case Instruction::FSub:
            fneg = B.CreateFSub(lhsPN, rhsPN);
            if (auto I = dyn_cast<Instruction>(fneg))
              I->setFast(fast);
            break;
          case Instruction::Mul:
            fneg = B.CreateMul(lhsPN, rhsPN, "", NUW, NSW);
            break;
          case Instruction::Add:
            fneg = B.CreateAdd(lhsPN, rhsPN, "", NUW, NSW);
            break;
          case Instruction::Sub:
            fneg = B.CreateSub(lhsPN, rhsPN, "", NUW, NSW);
            break;
          case Instruction::ZExt:
            fneg = B.CreateZExt(lhsPN, PN->getType());
            break;
          case Instruction::FCmp:
          case Instruction::ICmp:
            fneg = B.CreateCmp(*cmpPredicate, lhsPN, rhsPN);
            break;
          case Instruction::UIToFP:
            fneg = B.CreateUIToFP(lhsPN, PN->getType());
            break;
          case Instruction::Or:
            fneg = B.CreateOr(lhsPN, rhsPN);
            break;
          case Instruction::And:
            fneg = B.CreateAnd(lhsPN, rhsPN);
            break;
          default:
            llvm_unreachable("unhandled opcode");
          }

          push(fneg);

          if (preheader_fix != -1) {
            auto L = LI.getLoopFor(PN->getParent());
            auto idx = L->getCanonicalInductionVariable();
            auto eq = pushcse(
                B.CreateICmpEQ(idx, ConstantInt::get(idx->getType(), 0)));
            fneg =
                pushcse(B.CreateSelect(eq, PN->getIncomingValue(preheader_fix),
                                       fneg, "phphisel." + cur->getName()));
          }

          replaceAndErase(cur, fneg);
          return "PHIBinop";
        }
      }
    }
    // phi  -> select
    if (PN->getNumIncomingValues() == 2) {
      for (int i = 0; i < 2; i++) {
        auto prev = PN->getIncomingBlock(i);
        if (!DT.dominates(prev, PN->getParent())) {
          continue;
        }
        auto br = dyn_cast<BranchInst>(prev->getTerminator());
        if (!br) {
          continue;
        }
        if (!br->isConditional()) {
          continue;
        }
        if (br->getSuccessor(0) != PN->getParent()) {
          continue;
        }
        if (br->getSuccessor(1) != PN->getIncomingBlock(1 - i)) {
          continue;
        }

        Value *specVal = PN->getIncomingValue(1 - i);
        SetVector<Value *, std::deque<Value *>> todo;
        todo.insert(specVal);
        SetVector<Instruction *> toMove;
        bool legal = true;
        while (!todo.empty()) {
          auto cur = *todo.begin();
          todo.erase(todo.begin());
          auto I = dyn_cast<Instruction>(cur);
          if (!I)
            continue;
          if (I->mayReadOrWriteMemory()) {
            legal = false;
            break;
          }
          if (DT.dominates(I, PN))
            continue;
          for (size_t i = 0; i < I->getNumOperands(); i++)
            todo.insert(I->getOperand(i));
          toMove.insert(I);
        }
        if (!legal)
          continue;
        for (auto iter = toMove.rbegin(), end = toMove.rend(); iter != end;
             iter++) {
          (*iter)->moveBefore(br);
        }
        auto sel = pushcse(B.CreateSelect(
            br->getCondition(), PN->getIncomingValueForBlock(prev),
            PN->getIncomingValueForBlock(br->getSuccessor(1)),
            "tphisel." + cur->getName()));

        replaceAndErase(cur, sel);
        return "TPhiSel";
      }
    }
  }

  if (auto SI = dyn_cast<SelectInst>(cur)) {
    auto tval = replace(SI->getTrueValue(), SI->getCondition(),
                        ConstantInt::getTrue(SI->getContext()));
    auto fval = replace(SI->getFalseValue(), SI->getCondition(),
                        ConstantInt::getFalse(SI->getContext()));
    if (tval != SI->getTrueValue() || fval != SI->getFalseValue()) {
      auto res = pushcse(B.CreateSelect(SI->getCondition(), tval, fval,
                                        "postsel." + SI->getName()));
      replaceAndErase(cur, res);
      return "SelectReplace";
    }
  }

  // and a, b -> and a b[with a true]
  if (cur->getOpcode() == Instruction::And) {
    auto lhs = replace(cur->getOperand(0), cur->getOperand(1),
                       ConstantInt::getTrue(cur->getContext()));
    if (lhs != cur->getOperand(0)) {
      auto res = pushcse(
          B.CreateAnd(lhs, cur->getOperand(1), "postand." + cur->getName()));
      replaceAndErase(cur, res);
      return "AndReplaceLHS";
    }
    auto rhs = replace(cur->getOperand(1), cur->getOperand(0),
                       ConstantInt::getTrue(cur->getContext()));
    if (rhs != cur->getOperand(1)) {
      auto res = pushcse(
          B.CreateAnd(cur->getOperand(0), rhs, "postand." + cur->getName()));
      replaceAndErase(cur, res);
      return "AndReplaceRHS";
    }
  }

  // or a, b -> or a b[with a false]
  if (cur->getOpcode() == Instruction::Or) {
    auto lhs = replace(cur->getOperand(0), cur->getOperand(1),
                       ConstantInt::getFalse(cur->getContext()));
    if (lhs != cur->getOperand(0)) {
      auto res = pushcse(
          B.CreateOr(lhs, cur->getOperand(1), "postor." + cur->getName()));
      replaceAndErase(cur, res);
      return "OrReplaceLHS";
    }
    auto rhs = replace(cur->getOperand(1), cur->getOperand(0),
                       ConstantInt::getFalse(cur->getContext()));
    if (rhs != cur->getOperand(1)) {
      auto res = pushcse(
          B.CreateOr(cur->getOperand(0), rhs, "postor." + cur->getName()));
      replaceAndErase(cur, res);
      return "OrReplaceRHS";
    }
  }
  return {};
}

class Constraints;
raw_ostream &operator<<(raw_ostream &os, const Constraints &c);

struct ConstraintComparator {
  bool operator()(std::shared_ptr<const Constraints> lhs,
                  std::shared_ptr<const Constraints> rhs) const;
};

struct ConstraintContext {
  ScalarEvolution &SE;
  const Loop *loopToSolve;
  const SmallVectorImpl<Instruction *> &Assumptions;
  DominatorTree &DT;
  using InnerTy = std::shared_ptr<const Constraints>;
  using SetTy = std::set<InnerTy, ConstraintComparator>;
  SetTy seen;
  ConstraintContext(ScalarEvolution &SE, const Loop *loopToSolve,
                    const SmallVectorImpl<Instruction *> &Assumptions,
                    DominatorTree &DT)
      : SE(SE), loopToSolve(loopToSolve), Assumptions(Assumptions), DT(DT) {
    assert(loopToSolve);
  }
  ConstraintContext(const ConstraintContext &) = delete;
  ConstraintContext(const ConstraintContext &ctx, InnerTy lhs)
      : SE(ctx.SE), loopToSolve(ctx.loopToSolve), Assumptions(ctx.Assumptions),
        DT(ctx.DT), seen(ctx.seen) {
    seen.insert(lhs);
  }
  ConstraintContext(const ConstraintContext &ctx, InnerTy lhs, InnerTy rhs)
      : SE(ctx.SE), loopToSolve(ctx.loopToSolve), Assumptions(ctx.Assumptions),
        DT(ctx.DT), seen(ctx.seen) {
    seen.insert(lhs);
    seen.insert(rhs);
  }
  bool contains(InnerTy x) const { return seen.count(x) != 0; }
};

bool cannotDependOnLoopIV(const SCEV *S, const Loop *L) {
  assert(L);
  if (isa<SCEVConstant>(S))
    return true;
  if (auto M = dyn_cast<SCEVAddExpr>(S)) {
    for (auto o : M->operands())
      if (!cannotDependOnLoopIV(o, L))
        return false;
    return true;
  }
  if (auto M = dyn_cast<SCEVMulExpr>(S)) {
    for (auto o : M->operands())
      if (!cannotDependOnLoopIV(o, L))
        return false;
    return true;
  }
  if (auto M = dyn_cast<SCEVUDivExpr>(S)) {
    for (auto o : {M->getLHS(), M->getRHS()})
      if (!cannotDependOnLoopIV(o, L))
        return false;
    return true;
  }
  if (auto UV = dyn_cast<SCEVUnknown>(S)) {
    auto U = UV->getValue();
    if (isa<Argument>(U))
      return true;
    if (isa<Constant>(U))
      return true;
    auto I = cast<Instruction>(U);
    return !L->contains(I->getParent());
  }
  if (auto addrec = dyn_cast<SCEVAddRecExpr>(S)) {
    if (addrec->getLoop() == L)
      return false;
    for (auto o : addrec->operands())
      if (!cannotDependOnLoopIV(o, L))
        return false;
    return true;
  }
  if (auto expr = dyn_cast<SCEVSignExtendExpr>(S)) {
    return cannotDependOnLoopIV(expr->getOperand(), L);
  }
  llvm::errs() << " cannot tell if depends on loop iv: " << *S << "\n";
  return false;
}

const SCEV *evaluateAtLoopIter(const SCEV *V, ScalarEvolution &SE,
                               const Loop *find, const SCEV *replace) {
  assert(find);
  if (cannotDependOnLoopIV(V, find))
    return V;
  if (auto addrec = dyn_cast<SCEVAddRecExpr>(V)) {
    if (addrec->getLoop() == find) {
      auto V2 = addrec->evaluateAtIteration(replace, SE);
      return evaluateAtLoopIter(V2, SE, find, replace);
    }
  }
  if (auto div = dyn_cast<SCEVUDivExpr>(V)) {
    auto lhs = evaluateAtLoopIter(div->getLHS(), SE, find, replace);
    if (!lhs)
      return nullptr;
    auto rhs = evaluateAtLoopIter(div->getRHS(), SE, find, replace);
    if (!rhs)
      return nullptr;
    return SE.getUDivExpr(lhs, rhs);
  }
  return nullptr;
}

class Constraints : public std::enable_shared_from_this<Constraints> {
public:
  const enum class Type {
    Union = 0,
    Intersect = 1,
    Compare = 2,
    All = 3,
    None = 4
  } ty;

  using InnerTy = std::shared_ptr<const Constraints>;

  using SetTy = std::set<InnerTy, ConstraintComparator>;

  const SetTy values;

  const SCEV *const node;
  // whether equal to the node, or not equal to the node
  bool isEqual;
  // the loop of the iv comparing against.
  const llvm::Loop *const Loop;
  // using SetTy = SmallVector<InnerTy, 0>;
  // using SetTy = SetVector<InnerTy, SmallVector<InnerTy, 0>,
  // std::set<InnerTy>>;

  Constraints()
      : ty(Type::Union), values(), node(nullptr), isEqual(false),
        Loop(nullptr) {}

private:
  Constraints(const SCEV *v, bool isEqual, const llvm::Loop *Loop, bool)
      : ty(Type::Compare), values(), node(v), isEqual(isEqual), Loop(Loop) {}

public:
  static InnerTy make_compare(const SCEV *v, bool isEqual,
                              const llvm::Loop *Loop,
                              const ConstraintContext &ctx);

  Constraints(Type t)
      : ty(t), values(), node(nullptr), isEqual(false), Loop(nullptr) {
    assert(t == Type::All || t == Type::None);
  }
  Constraints(Type t, const SetTy &c, bool check = true)
      : ty(t), values(c), node(nullptr), isEqual(false), Loop(nullptr) {
    assert(t != Type::All);
    assert(t != Type::None);
    assert(c.size() != 0);
    assert(c.size() != 1);
#ifndef NDEBUG
    SmallVector<InnerTy, 1> tmp(c.begin(), c.end());
    for (unsigned i = 0; i < tmp.size(); i++)
      for (unsigned j = 0; j < i; j++)
        assert(*tmp[i] != *tmp[j]);
    if (t == Type::Intersect) {
      for (auto &v : c) {
        assert(v->ty != Type::Intersect);
      }
    }
    if (t == Type::Union) {
      for (auto &v : c) {
        assert(v->ty != Type::Union);
      }
    }
    if (t == Type::Intersect && check) {
      for (unsigned i = 0; i < tmp.size(); i++)
        if (tmp[i]->ty == Type::Compare && tmp[i]->isEqual && tmp[i]->Loop)
          for (unsigned j = 0; j < tmp.size(); j++)
            if (tmp[j]->ty == Type::Compare)
              if (auto s = dyn_cast<SCEVAddRecExpr>(tmp[j]->node))
                assert(s->getLoop() != tmp[i]->Loop);
    }
#endif
  }

  bool operator==(const Constraints &rhs) const {
    if (ty != rhs.ty) {
      return false;
    }
    if (node != rhs.node) {
      return false;
    }
    if (isEqual != rhs.isEqual) {
      return false;
    }
    if (Loop != rhs.Loop) {
      return false;
    }
    if (values.size() != rhs.values.size()) {
      return false;
    }
    for (auto pair : llvm::zip(values, rhs.values)) {
      if (*std::get<0>(pair) != *std::get<1>(pair))
        return false;
    }
    return true;
    //) && !(rhs.values < values)
    /*
for (size_t i=0; i<values.size(); i++)
if (*values[i] != *rhs.values[i]) return false;
return true;
    */
  }
  bool operator>(const Constraints &rhs) const { return rhs < *this; }
  bool operator<(const Constraints &rhs) const {
    if (ty < rhs.ty) {
      return true;
    }
    if (ty > rhs.ty) {
      return false;
    }
    if (node < rhs.node) {
      return true;
    }
    if (node > rhs.node) {
      return false;
    }
    if (isEqual < rhs.isEqual) {
      return true;
    }
    if (isEqual > rhs.isEqual) {
      return false;
    }
    if (Loop < rhs.Loop) {
      return true;
    }
    if (Loop > rhs.Loop) {
      return false;
    }
    if (values.size() < rhs.values.size()) {
      return true;
    }
    if (values.size() > rhs.values.size()) {
      return false;
    }
    for (auto pair : llvm::zip(values, rhs.values)) {
      if (*std::get<0>(pair) < *std::get<1>(pair))
        return true;
      if (*std::get<0>(pair) > *std::get<1>(pair))
        return false;
    }
    return false;
  }
  unsigned hash() const {
    unsigned res = 5 * (unsigned)ty +
                   DenseMapInfo<const SCEV *>::getHashValue(node) + isEqual;
    res = llvm::detail::combineHashValue(res, (unsigned)(size_t)Loop);
    for (auto v : values)
      res = llvm::detail::combineHashValue(res, v->hash());
    return res;
  }
  bool operator!=(const Constraints &rhs) const { return !(*this == rhs); }
  static InnerTy all() {
    static auto allv = std::make_shared<Constraints>(Type::All);
    return allv;
  }
  static InnerTy none() {
    static auto nonev = std::make_shared<Constraints>(Type::None);
    return nonev;
  }
  bool isNone() const { return ty == Type::None; }
  bool isAll() const { return ty == Type::All; }
  static void insert(SetTy &set, InnerTy ty) {
    set.insert(ty);
    int mcount = 0;
    for (auto &v : set)
      if (*v == *ty)
        mcount++;
    assert(mcount == 1);
    /*
                    for (auto &v : set)
                            if (*v == *ty)
                                    return;
                    set.push_back(ty);
    */
  }
  static SetTy intersect(const SetTy &lhs, const SetTy &rhs) {
    SetTy res;
    for (auto &v : lhs)
      if (rhs.count(v))
        res.insert(v);
    return res;
  }
  static void set_subtract(SetTy &set, const SetTy &rhs) {
    for (auto &v : rhs)
      if (set.count(v))
        set.erase(v);
    /*
    for (const auto &val : rhs)
    for (auto I = set.begin(); I != set.end(); I++) {
            if (**I == *val) {
                    set.erase(I);
                    break;
            }
    }
*/
  }
  __attribute__((noinline)) void dump() const { llvm::errs() << *this << "\n"; }
  InnerTy notB(const ConstraintContext &ctx) const {
    switch (ty) {
    case Type::None:
      return Constraints::all();
    case Type::All:
      return Constraints::none();
    case Type::Compare:
      return make_compare(node, !isEqual, Loop, ctx);
    case Type::Union: {
      // not of or's is and of not's
      SetTy next;
      for (const auto &v : values)
        insert(next, v->notB(ctx));
      if (next.size() == 1)
        llvm::errs() << " uold : " << *this << "\n";
      return std::make_shared<Constraints>(Type::Intersect, next);
    }
    case Type::Intersect: {
      // not of and's is or of not's
      SetTy next;
      for (const auto &v : values)
        insert(next, v->notB(ctx));
      if (next.size() == 1)
        llvm::errs() << " old : " << *this << "\n";
      return std::make_shared<Constraints>(Type::Union, next);
    }
    }
    return Constraints::none();
  }
  InnerTy orB(InnerTy rhs, const ConstraintContext &ctx) const {
    auto notLHS = notB(ctx);
    if (!notLHS)
      return nullptr;
    auto notRHS = rhs->notB(ctx);
    if (!notRHS)
      return nullptr;
    auto andV = notLHS->andB(notRHS, ctx);
    if (!andV)
      return nullptr;
    auto res = andV->notB(ctx);
    return res;
  }
  InnerTy andB(const InnerTy rhs, const ConstraintContext &ctx) const {
    assert(rhs);
    if (*rhs == *this)
      return shared_from_this();
    if (rhs->isNone())
      return rhs;
    if (rhs->isAll())
      return shared_from_this();
    if (isNone())
      return shared_from_this();
    if (isAll())
      return rhs;

    // llvm::errs() << " anding: " << *this << " with " << *rhs << "\n";
    if (ctx.contains(shared_from_this()) || ctx.contains(rhs)) {
      // llvm::errs() << " %%% stopping recursion\n";
      return nullptr;
    }
    if (ty == Type::Compare && rhs->ty == Type::Compare) {
      auto sub = ctx.SE.getMinusSCEV(node, rhs->node);
      if (Loop == rhs->Loop) {
        // llvm::errs() << " + sameloop, sub=" << *sub << "\n";
        if (auto cst = dyn_cast<SCEVConstant>(sub)) {
          // the two solves are equivalent to each other
          if (cst->getValue()->isZero()) {
            // iv = a and iv = a
            //   also iv != a and iv != a
            if (isEqual == rhs->isEqual)
              return shared_from_this();
            else {
              // iv = a and iv != a
              return Constraints::none();
            }
          } else {
            // the two solves are guaranteed to be distinct
            // iv == 0 and iv == 1
            if (isEqual && rhs->isEqual) {
              return Constraints::none();

            } else if (!isEqual && !rhs->isEqual) {
              // iv != 0 and iv != 1
              SetTy vals;
              insert(vals, shared_from_this());
              insert(vals, rhs);
              return std::make_shared<Constraints>(Type::Intersect, vals);
            } else if (!isEqual) {
              assert(rhs->isEqual);
              // iv != 0 and iv == 1
              return rhs;
              ;
            } else {
              // iv == 0 and iv != 1
              assert(isEqual);
              assert(!rhs->isEqual);
              return shared_from_this();
            }
          }
        } else if (isEqual || rhs->isEqual) {
          // llvm::errs() << " + botheq\n";
          // eq(i, a) & i ?= b -> eq(i, a) & (a ?= b)
          if (auto addrec = dyn_cast<SCEVAddRecExpr>(sub)) {
            // we want a ?= b, but we can only represent loopvar ?= something
            // so suppose a-b is of the form X + Y * lv  then a-b ?= 0 is
            //   X + Y * lv ?= 0 -> lv ?= - X / Y
            if (addrec->isAffine()) {
              auto X = addrec->getStart();
              auto Y = addrec->getStepRecurrence(ctx.SE);
              auto MinusX = X;

              if (isa<SCEVConstant>(Y) &&
                  cast<SCEVConstant>(Y)->getAPInt().isNegative())
                Y = ctx.SE.getNegativeSCEV(Y);
              else
                MinusX = ctx.SE.getNegativeSCEV(X);

              auto div = ctx.SE.getUDivExpr(MinusX, Y);
              auto div_e = ctx.SE.getUDivExactExpr(MinusX, Y);
              // in case of inexact division, check that these exactly equal
              // for replacement

              if (div == div_e) {
                if (isEqual) {
                  auto res = make_compare(div, /*isEqual*/ rhs->isEqual,
                                          addrec->getLoop(), ctx);
                  // llvm::errs() << " simplified rhs to: " << *res << "\n";
                  return andB(res, ctx);
                } else {
                  assert(rhs->isEqual);
                  auto res = make_compare(div, /*isEqual*/ isEqual,
                                          addrec->getLoop(), ctx);
                  // llvm::errs() << " simplified lhs to: " << *res << "\n";
                  return rhs->andB(res, ctx);
                }
              }
            }
          }
          if (isEqual && rhs->Loop &&
              cannotDependOnLoopIV(sub, ctx.loopToSolve)) {
            auto res = make_compare(sub, /*isEqual*/ rhs->isEqual,
                                    /*loop*/ nullptr, ctx);
            // llvm::errs() << " simplified(noloop) rhs from " << *rhs
            //             << " to: " << *res << "\n";
            return andB(res, ctx);
          }
          if (rhs->isEqual && Loop &&
              cannotDependOnLoopIV(sub, ctx.loopToSolve)) {
            auto res =
                make_compare(sub, /*isEqual*/ isEqual, /*loop*/ nullptr, ctx);
            // llvm::errs() << " simplified(noloop) lhs from " << *rhs
            //             << " to: " << *res << "\n";
            return rhs->andB(res, ctx);
          }

          llvm::errs() << " warning: potential but unhandled simplification of "
                          "equalities: "
                       << *this << " and " << *rhs << " sub: " << *sub << "\n";
        }
      }

      if (isEqual) {
        if (Loop)
          if (auto rep = evaluateAtLoopIter(rhs->node, ctx.SE, Loop, node))
            if (rep != rhs->node) {
              auto newrhs = make_compare(rep, rhs->isEqual, rhs->Loop, ctx);
              return andB(newrhs, ctx);
            }

        // not loop -> node == 0
        if (!Loop) {
          for (auto sub1 : {ctx.SE.getMinusSCEV(node, rhs->node),
                            ctx.SE.getMinusSCEV(rhs->node, node)}) {
            // llvm::errs() << " maybe replace lhs: " << *this << " rhs: " <<
            // *rhs
            //             << " sub1: " << *sub1 << "\n";
            auto newrhs = make_compare(sub1, rhs->isEqual, rhs->Loop, ctx);
            if (*newrhs == *this)
              return shared_from_this();
            if (!isa<SCEVConstant>(rhs->node) && isa<SCEVConstant>(sub1)) {
              return andB(newrhs, ctx);
            }
          }
        }
      }

      if (rhs->isEqual) {
        if (rhs->Loop)
          if (auto rep = evaluateAtLoopIter(node, ctx.SE, rhs->Loop, rhs->node))
            if (rep != node) {
              auto newlhs = make_compare(rep, isEqual, Loop, ctx);
              return newlhs->andB(rhs, ctx);
            }

        // not loop -> node == 0
        if (!rhs->Loop) {
          for (auto sub1 : {ctx.SE.getMinusSCEV(node, rhs->node),
                            ctx.SE.getMinusSCEV(rhs->node, node)}) {
            // llvm::errs() << " maybe replace lhs2: " << *this << " rhs: " <<
            // *rhs
            //             << " sub1: " << *sub1 << "\n";
            auto newlhs = make_compare(sub1, isEqual, Loop, ctx);
            if (*newlhs == *this)
              return shared_from_this();
            if (!isa<SCEVConstant>(node) && isa<SCEVConstant>(sub1)) {
              return newlhs->andB(rhs, ctx);
            }
          }
        }
      }

      if (!Loop && !rhs->Loop && isEqual == rhs->isEqual) {
        if (node == ctx.SE.getNegativeSCEV(rhs->node))
          return shared_from_this();
      }

      SetTy vals;
      insert(vals, shared_from_this());
      insert(vals, rhs);
      if (vals.size() == 1) {
        llvm::errs() << "this: " << *this << " rhs: " << *rhs << "\n";
      }
      auto res = std::make_shared<Constraints>(Type::Intersect, vals);
      // llvm::errs() << " naiive comp merge: " << *res << "\n";
      return res;
    }
    if (ty == Type::Intersect && rhs->ty == Type::Intersect) {
      auto tmp = shared_from_this();
      for (const auto &v : rhs->values) {
        auto tmp2 = tmp->andB(v, ctx);
        if (!tmp2)
          return nullptr;
        tmp = std::move(tmp2);
      }
      return tmp;
    }
    if (ty == Type::Intersect && rhs->ty == Type::Compare) {
      SetTy vals;
      // Force internal merging to do individual compares
      bool foldedIn = false;
      for (auto en : llvm::enumerate(values)) {
        auto i = en.index();
        auto v = en.value();
        assert(v->ty != Type::Intersect);
        assert(v->ty != Type::All);
        assert(v->ty != Type::None);
        assert(v->ty == Type::Compare || v->ty == Type::Union);
        if (foldedIn) {
          insert(vals, v);
          continue;
        }
        // this is either a compare or a union
        auto tmp = rhs->andB(v, ctx);
        if (!tmp)
          return nullptr;
        switch (tmp->ty) {
        case Type::Union:
        case Type::All:
          llvm_unreachable("Impossible");
        case Type::None:
          return Constraints::none();
        case Type::Compare:
          insert(vals, tmp);
          foldedIn = true;
          break;
        // if intersected, these two were not foldable, try folding into later
        case Type::Intersect: {
          SetTy fuse;
          insert(fuse, rhs);
          insert(fuse, v);

          Constraints trivialFuse(Type::Intersect, fuse, false);

          // If this is not just making an intersect of the two operands,
          // remerge.
          if (trivialFuse != *tmp) {
            InnerTy newlhs = Constraints::all();
            bool legal = true;
            for (auto en2 : llvm::enumerate(values)) {
              auto i2 = en2.index();
              auto v2 = en2.value();
              if (i2 == i)
                continue;
              auto newlhs2 = newlhs->andB(v2, ctx);
              if (!newlhs2) {
                legal = false;
                break;
              }
              newlhs = std::move(newlhs2);
            }
            if (legal) {
              return newlhs->andB(tmp, ctx);
            }
          }
          insert(vals, v);
        }
        }
      }
      if (!foldedIn) {
        insert(vals, rhs);
        return std::make_shared<Constraints>(Type::Intersect, vals);
      } else {
        auto cur = Constraints::all();
        for (auto &iv : vals) {
          auto cur2 = cur->andB(iv, ctx);
          if (!cur2)
            return nullptr;
          cur = std::move(cur2);
        }
        return cur;
      }
    }
    if ((ty == Type::Intersect || ty == Type::Compare) &&
        rhs->ty == Type::Union) {
      SetTy unionVals = rhs->values;
      bool changed = false;
      SetTy ivVals;
      if (ty == Type::Intersect)
        ivVals = values;
      else
        insert(ivVals, shared_from_this());

      ConstraintContext ctxd(ctx, shared_from_this(), rhs);

      for (const auto &iv : ivVals) {
        SetTy nextunionVals;
        bool midchanged = false;
        for (auto &uv : unionVals) {
          auto tmp = iv->andB(uv, ctxd);
          if (!tmp) {
            midchanged = false;
            nextunionVals = unionVals;
            break;
          }
          switch (tmp->ty) {
          case Type::None:
          case Type::Compare:
          case Type::Union:
            insert(nextunionVals, tmp);
            changed |= tmp != uv;
            break;
          case Type::Intersect: {
            SetTy fuse;
            if (uv->ty == Type::Intersect)
              fuse = uv->values;
            else {
              assert(uv->ty == Type::Compare);
              insert(fuse, uv);
            }
            insert(fuse, iv);

            Constraints trivialFuse(Type::Intersect, fuse, false);
            if (trivialFuse != *tmp) {
              insert(nextunionVals, tmp);
              midchanged = true;
              break;
            }

            insert(nextunionVals, uv);
            break;
          }
          case Type::All:
            llvm_unreachable("Impossible");
          }
        }
        if (midchanged) {
          unionVals = nextunionVals;
          changed = true;
        }
      }

      if (changed) {
        auto cur = Constraints::none();
        for (auto uv : unionVals) {
          cur = cur->orB(uv, ctxd);
          if (!cur)
            break;
        }

        if (*cur != *rhs)
          return andB(cur, ctx);
      }

      SetTy vals = ivVals;
      insert(vals, rhs);
      return std::make_shared<Constraints>(Type::Intersect, vals);
    }
    // Handled above via symmetry
    if (rhs->ty == Type::Intersect || rhs->ty == Type::Compare) {
      return rhs->andB(shared_from_this(), ctx);
    }
    // (m or a or b or d) and (m or a or c or e ...) -> m or a or ( (b or d)
    // and (c or e))
    if (ty == Type::Union && rhs->ty == Type::Union) {
      if (*this == *rhs->notB(ctx)) {
        return Constraints::none();
      }
      SetTy intersection = intersect(values, rhs->values);
      if (intersection.size() != 0) {
        InnerTy other_lhs = remove(intersection);
        InnerTy other_rhs = rhs->remove(intersection);
        InnerTy remainder;
        if (intersection.size() == 1)
          remainder = *intersection.begin();
        else {
          remainder = std::make_shared<Constraints>(Type::Union, intersection);
        }
        return remainder->orB(other_lhs->andB(other_rhs, ctx), ctx);
      }

      bool changed = false;
      SetTy lhsVals = values;
      SetTy rhsVals = rhs->values;

      ConstraintContext ctxd(ctx, shared_from_this(), rhs);

      SetTy distributedVals;
      for (const auto &l1 : lhsVals) {
        bool subchanged = false;
        SetTy subDistributedVals;
        for (auto &r1 : rhsVals) {
          auto tmp = l1->andB(r1, ctxd);
          if (!tmp) {
            subchanged = false;
            break;
          }

          if (l1->ty == Type::Intersect || r1->ty == Type::Intersect) {
            subchanged = true;
            insert(subDistributedVals, tmp);
          } else {

            SetTy fuse;
            insert(fuse, l1);
            insert(fuse, r1);
            assert(fuse.size() == 2);
            Constraints trivialFuse(Type::Intersect, fuse);
            if ((trivialFuse != *tmp) || distributedVals.count(tmp)) {
              subchanged = true;
            }
            insert(subDistributedVals, tmp);
          }
        }
        if (subchanged) {
          for (auto sub : subDistributedVals)
            insert(distributedVals, sub);
          changed = true;
        } else {
          auto midand = l1->andB(rhs, ctxd);
          if (!midand) {
            changed = false;
            break;
          }
          insert(distributedVals, midand);
        }
      }

      if (changed) {
        auto cur = Constraints::none();
        bool legal = true;
        for (auto &uv : distributedVals) {
          auto cur2 = cur->orB(uv, ctxd);
          if (!cur2) {
            legal = false;
            break;
          }
          cur = std::move(cur2);
        }
        if (legal) {
          return cur;
        }
      }

      SetTy vals;
      insert(vals, shared_from_this());
      insert(vals, rhs);
      auto res = std::make_shared<Constraints>(Type::Intersect, vals);
      return res;
    }
    llvm::errs() << " andB this: " << *this << " rhs: " << *rhs << "\n";
    llvm_unreachable("Illegal predicate state");
  }
  // what this would be like when removing the following list of constraints
  InnerTy remove(const SetTy &sub) const {
    assert(ty == Type::Union || ty == Type::Intersect);
    SetTy res = values;
    set_subtract(res, sub);
    // res.set_subtract(sub);
    if (res.size() == 0) {
      if (ty == Type::Union)
        return Constraints::none();
      else
        return Constraints::all();
    } else if (res.size() == 1) {
      return *res.begin();
    } else {
      return std::make_shared<Constraints>(ty, res);
    }
  }
  SmallVector<std::pair<Value *, Value *>, 1>
  allSolutions(SCEVExpander &Exp, llvm::Type *T, Instruction *IP,
               const ConstraintContext &ctx, IRBuilder<> &B) const;
};

void dump(const Constraints &c) { c.dump(); }
void dump(std::shared_ptr<const Constraints> c) { c->dump(); }

bool ConstraintComparator::operator()(
    std::shared_ptr<const Constraints> lhs,
    std::shared_ptr<const Constraints> rhs) const {
  return *lhs < *rhs;
}

raw_ostream &operator<<(raw_ostream &os, const Constraints &c) {
  switch (c.ty) {
  case Constraints::Type::All:
    return os << "All";
  case Constraints::Type::None:
    return os << "None";
  case Constraints::Type::Union: {
    os << "(Union ";
    for (auto v : c.values)
      os << *v << ", ";
    os << ")";
    return os;
  }
  case Constraints::Type::Intersect: {
    os << "(Intersect ";
    for (auto v : c.values)
      os << *v << ", ";
    os << ")";
    return os;
  }
  case Constraints::Type::Compare: {
    if (c.isEqual)
      os << "(eq ";
    else
      os << "(ne ";
    os << *c.node << ", L=";
    if (c.Loop)
      os << c.Loop->getHeader()->getName();
    else
      os << "nullptr";
    return os << ")";
  }
  }
  return os;
}

SmallVector<std::pair<Value *, Value *>, 1>
Constraints::allSolutions(SCEVExpander &Exp, llvm::Type *T, Instruction *IP,
                          const ConstraintContext &ctx, IRBuilder<> &B) const {
  switch (ty) {
  case Type::None:
    return {};
  case Type::All:
    llvm::errs() << *this << "\n";
    llvm_unreachable("All not handled");
  case Type::Compare: {
    Value *cond = ConstantInt::getTrue(T->getContext());
    if (ctx.loopToSolve != Loop) {
      assert(ctx.loopToSolve);
      Value *ivVal = Exp.expandCodeFor(node, T, IP);
      Value *iv = nullptr;
      if (Loop) {
        iv = Loop->getCanonicalInductionVariable();
        assert(iv);
      } else {
        iv = ConstantInt::getNullValue(ivVal->getType());
      }
      if (isEqual)
        cond = B.CreateICmpEQ(ivVal, iv);
      else
        cond = B.CreateICmpNE(ivVal, iv);
      return {std::make_pair((Value *)nullptr, cond)};
    }
    if (isEqual) {
      return {std::make_pair(Exp.expandCodeFor(node, T, IP), cond)};
    }
    EmitFailure("NoSparsification", IP->getDebugLoc(), IP,
                "Negated solution not handled: ", *this);
    assert(0);
    return {};
  }
  case Type::Union: {
    SmallVector<std::pair<Value *, Value *>, 1> vals;
    for (auto v : values)
      for (auto sol : v->allSolutions(Exp, T, IP, ctx, B))
        vals.push_back(sol);
    return vals;
  }
  case Type::Intersect: {
    {
      SmallVector<InnerTy, 1> vals(values.begin(), values.end());
      ssize_t unionidx = -1;
      for (unsigned i = 0; i < vals.size(); i++) {
        if (vals[i]->ty == Type::Union) {
          unionidx = i;
          bool allne = true;
          for (auto &v : vals[i]->values) {
            if (v->ty != Type::Compare || v->isEqual) {
              allne = false;
              break;
            }
          }
          if (allne)
            break;
        }
      }
      if (unionidx != -1) {
        auto others = Constraints::all();
        for (unsigned j = 0; j < vals.size(); j++)
          if (unionidx != j)
            others = others->andB(vals[j], ctx);
        SmallVector<std::pair<Value *, Value *>, 1> resvals;
        for (auto &v : vals[unionidx]->values) {
          auto tmp = v->andB(others, ctx);
          for (const auto &sol : tmp->allSolutions(Exp, T, IP, ctx, B))
            resvals.push_back(sol);
        }
        return resvals;
      }
    }
    Value *solVal = nullptr;
    Value *cond = ConstantInt::getTrue(T->getContext());
    for (auto v : values) {
      auto sols = v->allSolutions(Exp, T, IP, ctx, B);
      if (sols.size() != 1) {
        llvm::errs() << *this << "\n";
        for (auto s : sols)
          if (s.first)
            llvm::errs() << " + sol: " << *s.first << " " << *s.second << "\n";
          else
            llvm::errs() << " + sol: " << s.first << " " << *s.second << "\n";
        llvm::errs() << " v: " << *v << " this: " << *this << "\n";
        llvm_unreachable("Intersect not handled (solsize>1)");
      }
      auto sol = sols[0];
      if (sol.first) {
        if (solVal != nullptr) {
          llvm::errs() << *this << "\n";
          llvm::errs() << " prevsolVal: " << *solVal << "\n";
          llvm_unreachable("Intersect not handled (prevsolval)");
        }
        assert(solVal == nullptr);
        solVal = sol.first;
      }
      cond = B.CreateAnd(cond, sol.second);
    }
    return {std::make_pair(solVal, cond)};
  }
  }
  return {};
}

constexpr bool SparseDebug = false;
std::shared_ptr<const Constraints>
getSparseConditions(bool &legal, Value *val,
                    std::shared_ptr<const Constraints> defaultFloat,
                    Instruction *scope, const ConstraintContext &ctx) {
  if (auto I = dyn_cast<Instruction>(val)) {
    // Binary `and` is a bit-wise `umin`.
    if (I->getOpcode() == Instruction::And) {
      auto lhs = getSparseConditions(legal, I->getOperand(0),
                                     Constraints::all(), I, ctx);
      auto rhs = getSparseConditions(legal, I->getOperand(1),
                                     Constraints::all(), I, ctx);
      auto res = lhs->andB(rhs, ctx);
      assert(res);
      assert(ctx.seen.size() == 0);
      if (SparseDebug) {
        llvm::errs() << " getSparse(and, " << *I << "), lhs("
                     << *I->getOperand(0) << ") = " << *lhs << "\n";
        llvm::errs() << " getSparse(and, " << *I << "), rhs("
                     << *I->getOperand(1) << ") = " << *rhs << "\n";
        llvm::errs() << " getSparse(and, " << *I << ") = " << *res << "\n";
      }
      return res;
    }

    // Binary `or` is a bit-wise `umax`.
    if (I->getOpcode() == Instruction::Or) {
      auto lhs = getSparseConditions(legal, I->getOperand(0),
                                     Constraints::none(), I, ctx);
      auto rhs = getSparseConditions(legal, I->getOperand(1),
                                     Constraints::none(), I, ctx);
      auto res = lhs->orB(rhs, ctx);
      if (SparseDebug) {
        llvm::errs() << " getSparse(or, " << *I << "), lhs("
                     << *I->getOperand(0) << ") = " << *lhs << "\n";
        llvm::errs() << " getSparse(or, " << *I << "), rhs("
                     << *I->getOperand(1) << ") = " << *rhs << "\n";
        llvm::errs() << " getSparse(or, " << *I << ") = " << *res << "\n";
      }
      return res;
    }

    if (I->getOpcode() == Instruction::Xor) {
      for (int i = 0; i < 2; i++) {
        if (auto C = dyn_cast<ConstantInt>(I->getOperand(i)))
          if (C->isOne()) {
            auto pres =
                getSparseConditions(legal, I->getOperand(1 - i),
                                    defaultFloat->notB(ctx), scope, ctx);
            auto res = pres->notB(ctx);
            if (SparseDebug) {
              llvm::errs() << " getSparse(not, " << *I << "), prev ("
                           << *I->getOperand(0) << ") = " << *pres << "\n";
              llvm::errs() << " getSparse(not, " << *I << ") = " << *res
                           << "\n";
            }
            return res;
          }
      }
    }

    if (auto icmp = dyn_cast<ICmpInst>(I)) {
      auto L = ctx.loopToSolve;
      auto lhs = ctx.SE.getSCEVAtScope(icmp->getOperand(0), L);
      auto rhs = ctx.SE.getSCEVAtScope(icmp->getOperand(1), L);
      if (SparseDebug) {
        llvm::errs() << " lhs: " << *lhs << "\n";
        llvm::errs() << " rhs: " << *rhs << "\n";
      }

      auto sub1 = ctx.SE.getMinusSCEV(lhs, rhs);

      if (icmp->getPredicate() == ICmpInst::ICMP_EQ ||
          icmp->getPredicate() == ICmpInst::ICMP_NE) {
        if (auto add = dyn_cast<SCEVAddRecExpr>(sub1)) {
          if (add->isAffine()) {
            // 0 === A + B * inc -> -A / B = inc
            auto A = add->getStart();
            if (auto B =
                    dyn_cast<SCEVConstant>(add->getStepRecurrence(ctx.SE))) {

              auto MA = A;
              if (B->getAPInt().isNegative())
                B = cast<SCEVConstant>(ctx.SE.getNegativeSCEV(B));
              else
                MA = ctx.SE.getNegativeSCEV(A);
              auto div = ctx.SE.getUDivExpr(MA, B);
              auto div_e = ctx.SE.getUDivExactExpr(MA, B);
              if (div == div_e) {
                auto res = Constraints::make_compare(
                    div, icmp->getPredicate() == ICmpInst::ICMP_EQ,
                    add->getLoop(), ctx);
                if (SparseDebug) {
                  llvm::errs()
                      << " getSparse(icmp, " << *I << ") = " << *res << "\n";
                }
                return res;
              }
            }
          }
        }
        if (cannotDependOnLoopIV(sub1, ctx.loopToSolve)) {
          auto res = Constraints::make_compare(
              sub1, icmp->getPredicate() == ICmpInst::ICMP_EQ, nullptr, ctx);
          llvm::errs() << " getSparse(icmp_noloop, " << *I << ") = " << *res
                       << "\n";
          return res;
        }
      }
      if (scope)
        EmitWarning("NoSparsification", *I,
                    " No sparsification: not sparse solvable(icmp): ", *I,
                    " via ", *sub1);
      if (SparseDebug) {
        llvm::errs() << " getSparse(icmp_dflt, " << *I
                     << ") = " << *defaultFloat << "\n";
      }
      return defaultFloat;
    }

    // cmp x, 1.0 ->   false/true
    if (auto fcmp = dyn_cast<FCmpInst>(I)) {
      auto res = defaultFloat;
      if (SparseDebug) {
        llvm::errs() << " getSparse(fcmp, " << *I << ") = " << *res << "\n";
      }
      return res;

      if (fcmp->getPredicate() == CmpInst::FCMP_OEQ ||
          fcmp->getPredicate() == CmpInst::FCMP_UEQ) {
        return Constraints::all();
      } else if (fcmp->getPredicate() == CmpInst::FCMP_ONE ||
                 fcmp->getPredicate() == CmpInst::FCMP_UNE) {
        return Constraints::none();
      }
    }
  }

  if (scope) {
    EmitFailure("NoSparsification", scope->getDebugLoc(), scope,
                " No sparsification: not sparse solvable: ", *val);
  }
  legal = false;
  return defaultFloat;
}

Constraints::InnerTy Constraints::make_compare(const SCEV *v, bool isEqual,
                                               const llvm::Loop *Loop,
                                               const ConstraintContext &ctx) {
  if (!Loop) {
    assert(!isa<SCEVAddRecExpr>(v));
    SmallVector<Instruction *, 1> noassumption;
    ConstraintContext ctx2(ctx.SE, ctx.loopToSolve, noassumption, ctx.DT);
    for (auto I : ctx.Assumptions) {
      bool legal = true;
      if (I->getParent()->getParent() !=
          ctx.loopToSolve->getHeader()->getParent())
        continue;
      auto parsedCond = getSparseConditions(legal, I->getOperand(0),
                                            Constraints::none(), nullptr, ctx2);
      bool dominates = ctx.DT.dominates(I, ctx.loopToSolve->getHeader());
      if (legal && dominates) {
        if (parsedCond->ty == Type::Compare && !parsedCond->Loop) {
          if (parsedCond->node == v ||
              parsedCond->node == ctx.SE.getNegativeSCEV(v)) {
            InnerTy res;
            if (parsedCond->isEqual == isEqual)
              res = Constraints::all();
            else
              res = Constraints::none();
            return res;
          }
        }
      }
    }
  }
  // cannot have negative loop canonical induction var
  if (Loop)
    if (auto C = dyn_cast<SCEVConstant>(v))
      if (C->getAPInt().isNegative()) {
        if (isEqual)
          return Constraints::none();
        else
          return Constraints::all();
      }
  return InnerTy(new Constraints(v, isEqual, Loop, false));
}

void fixSparseIndices(llvm::Function &F, llvm::FunctionAnalysisManager &FAM,
                      SetVector<BasicBlock *> &toDenseBlocks) {

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);
  auto &DL = F.getParent()->getDataLayout();

  QueueType Q(DT, LI);
  {
    llvm::SetVector<BasicBlock *> todoBlocks;
    for (auto b : toDenseBlocks) {
      auto L = LI.getLoopFor(b);
      if (L) {
        for (auto B : L->getBlocks())
          todoBlocks.insert(B);
      }
    }
    for (auto BB : todoBlocks)
      for (auto &I : *BB)
        if (!I.getType()->isVoidTy()) {
          Q.insert(&I);
          assert(Q.contains(&I));
        }
  }

  // llvm::errs() << " pre fix inner: " << F << "\n";

  // Full simplification
  while (!Q.empty()) {
    auto cur = Q.pop_back_val();
    /*
    std::set<Instruction *> prev;
    for (auto v : Q)
      prev.insert(v);
    // llvm::errs() << "\n\n\n\n" << F << "\n";
    llvm::errs() << "cur: " << *cur << "\n";
    */
    auto changed = fixSparse_inner(cur, F, Q, DT, SE, LI, DL);
    (void)changed;
    /*
    if (changed) {
      llvm::errs() << "changed: " << *changed << "\n";

      for (auto I : Q)
        if (!prev.count(I))
          llvm::errs() << " + " << *I << "\n";
      // llvm::errs() << F << "\n\n";
    }
    */
  }

  // llvm::errs() << " post fix inner " << F << "\n";

  SmallVector<std::pair<BasicBlock *, BranchInst *>, 1> sparseBlocks;
  bool legalToSparse = true;
  for (auto &B : F)
    if (auto br = dyn_cast<BranchInst>(B.getTerminator()))
      if (br->isConditional())
        for (int bidx = 0; bidx < 2; bidx++)
          if (auto uncond_br =
                  dyn_cast<BranchInst>(br->getSuccessor(bidx)->getTerminator()))
            if (!uncond_br->isConditional())
              if (uncond_br->getSuccessor(0) == br->getSuccessor(1 - bidx)) {
                auto blk = br->getSuccessor(bidx);
                int countSparse = 0;
                for (auto &I : *blk) {
                  if (auto CI = dyn_cast<CallInst>(&I)) {
                    if (auto F = CI->getCalledFunction()) {
                      if (F->hasFnAttribute("enzyme_sparse_accumulate")) {
                        countSparse++;
                      }
                    }
                  }
                }
                if (countSparse == 0)
                  continue;
                if (countSparse > 1) {
                  legalToSparse = false;
                  EmitFailure(
                      "NoSparsification", br->getDebugLoc(), br, "F: ", F,
                      "\nMultiple distinct sparse stores in same block: ",
                      *blk);
                  break;
                }

                for (auto &I : *blk) {
                  if (auto CI = dyn_cast<CallInst>(&I)) {
                    if (auto F = CI->getCalledFunction()) {
                      if (F->hasFnAttribute("enzyme_sparse_accumulate")) {
                        continue;
                      }
                    }
                    if (isReadOnly(CI))
                      continue;
                  }
                  if (!I.mayWriteToMemory())
                    continue;

                  legalToSparse = false;
                  EmitFailure(
                      "NoSparsification", br->getDebugLoc(), br, "F: ", F,
                      "\nIllegal writing instruction in sparse block: ", I);
                  break;
                }

                if (!legalToSparse) {
                  break;
                }

                auto L = LI.getLoopFor(blk);
                if (!L) {
                  legalToSparse = false;
                  EmitFailure("NoSparsification", br->getDebugLoc(), br,
                              "F: ", F, "\nCould not find loop for: ", *blk);
                  break;
                }
                auto idx = L->getCanonicalInductionVariable();
                if (!idx) {
                  legalToSparse = false;
                  EmitFailure("NoSparsification", br->getDebugLoc(), br,
                              "F: ", F, "\nL:", *L,
                              "\nCould not find loop index: ", *L->getHeader());
                  break;
                }
                assert(idx);
                auto preheader = L->getLoopPreheader();
                if (!preheader) {
                  legalToSparse = false;
                  EmitFailure("NoSparsification", br->getDebugLoc(), br,
                              "F: ", F, "\nL:", *L,
                              "\nCould not find loop preheader");
                  break;
                }
                sparseBlocks.emplace_back(blk, br);
              }

  if (!legalToSparse) {
    return;
  }

  // block, bound, scev for indexset
  std::map<Loop *,
           std::pair<std::pair<PHINode *, PHINode *>,
                     SmallVector<std::pair<BasicBlock *,
                                           std::shared_ptr<const Constraints>>,
                                 1>>>
      forSparsification;

  SmallVector<Instruction *, 1> Assumptions;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto II = dyn_cast<IntrinsicInst>(&I))
        if (II->getIntrinsicID() == Intrinsic::assume)
          Assumptions.push_back(II);

  bool sawError = false;

  for (auto [blk, br] : sparseBlocks) {
    auto L = LI.getLoopFor(blk);
    assert(L);
    auto idx = L->getCanonicalInductionVariable();
    assert(idx);
    auto preheader = L->getLoopPreheader();
    assert(preheader);

    // default is condition avoids sparse, negated is condition goes
    // to sparse
    auto cond = br->getCondition();
    bool negated = br->getSuccessor(0) == blk;

    bool legal = true;
    // Whether the i1 value does not contain any icmp's
    std::function<bool(Value *)> onlyDataDependentValues = [&](Value *val) {
      auto I = cast<Instruction>(val);
      if (I->getOpcode() == Instruction::Or) {
        return onlyDataDependentValues(I->getOperand(0)) &&
               onlyDataDependentValues(I->getOperand(1));
      }
      if (I->getOpcode() == Instruction::And) {
        return onlyDataDependentValues(I->getOperand(0)) &&
               onlyDataDependentValues(I->getOperand(1));
      }
      if (isa<FCmpInst>(I))
        return true;
      if (isa<ICmpInst>(I))
        return false;
      EmitFailure("NoSparsification", I->getDebugLoc(), I,
                  " No sparsification: bad datadepedent values check: ", *I);
      legal = false;
      return true;
    };

    // Simplify variable val which is known to branch away from the
    // actual store (if not negated) or to the store (if negated)
    // if! negated the result may become more false if negated the
    // result may become more true

    //

    // default is condition avoids sparse, negated is condition goes
    // to sparse
    Instruction *context =
        isa<Instruction>(cond) ? cast<Instruction>(cond) : idx;
    ConstraintContext cctx(SE, L, Assumptions, DT);
    auto solutions = getSparseConditions(
        legal, cond, negated ? Constraints::all() : Constraints::none(),
        context, cctx);
    // llvm::errs() << " solutions pre negate: " << *solutions << "\n";
    if (!negated) {
      solutions = solutions->notB(cctx);
    }
    // llvm::errs() << " solutions post negate: " << *solutions << "\n";
    if (!legal) {
      sawError = true;
      continue;
    }

    if (solutions == Constraints::none() || solutions == Constraints::all()) {
      EmitFailure(
          "NoSparsification", context->getDebugLoc(), context, "F: ", F,
          "\nL: ", *L, "\ncond: ", *cond, " negated:", negated,
          "\n No sparsification: not sparse solvable(nosoltn): solutions:",
          *solutions);
      sawError = true;
    }
    // llvm::errs() << " found solvable solutions " << *solutions << "\n";

    if (forSparsification.count(L) == 0) {
      {
        IRBuilder<> PB(preheader->getTerminator());
        forSparsification[L].first =
            std::make_pair(PB.CreatePHI(idx->getType(), 0, "ph.idx"),
                           PB.CreatePHI(idx->getType(), 0, "loop.idx"));
      }

      Value *LoopCount = nullptr;

      IRBuilder<> B(L->getHeader()->getFirstNonPHI());
      {
        SCEVExpander Exp(SE, DL, "sparseenzyme");
        auto LoopCountS = SE.getBackedgeTakenCount(L);
        LoopCount = B.CreateAdd(
            ConstantInt::get(idx->getType(), 1),
            Exp.expandCodeFor(LoopCountS, idx->getType(), &blk->front()));
      }
      Value *inbounds = B.CreateAnd(
          B.CreateICmpSLT(idx, LoopCount),
          B.CreateICmpSGE(idx, ConstantInt::get(idx->getType(), 0)));
      Value *args[] = {inbounds, forSparsification[L].first.second};
      B.CreateCall(F.getParent()->getOrInsertFunction(
                       "enzyme.sparse.inbounds", B.getVoidTy(),
                       inbounds->getType(), idx->getType()),
                   args);
    }

    IRBuilder<> B(br);
    B.SetInsertPoint(br);
    auto nidx = B.CreateICmpEQ(
        forSparsification[L].first.first,
        ConstantInt::get(idx->getType(), forSparsification[L].second.size()));
    // TODO check direction
    if (!negated)
      nidx = B.CreateNot(nidx);

    br->setCondition(nidx);
    forSparsification[L].second.emplace_back(blk, solutions);
  }

  if (sawError) {
    for (auto &pair : forSparsification) {
      for (auto PN : {pair.second.first.first, pair.second.first.second}) {
        PN->replaceAllUsesWith(UndefValue::get(PN->getType()));
        PN->eraseFromParent();
      }
    }
    if (llvm::verifyFunction(F, &llvm::errs())) {
      llvm::errs() << F << "\n";
      report_fatal_error("function failed verification (6)");
    }
    return;
  }

  if (forSparsification.size() == 0) {
    auto context = &F.getEntryBlock().front();
    EmitFailure("NoSparsification", context->getDebugLoc(), context, "F: ", F,
                "\n Found no stores for sparsification");
    return;
  }

  for (const auto &pair : forSparsification) {
    auto L = pair.first;
    auto [PN, inductPN] = pair.second.first;

    auto ph = L->getLoopPreheader();
#if LLVM_VERSION_MAJOR >= 20
    CodeExtractor ext(L->getBlocks(), &DT);
#else
    CodeExtractor ext(DT, *L);
#endif
    CodeExtractorAnalysisCache cache(F);
    SetVector<Value *> Inputs, Outputs;
    auto F2 = ext.extractCodeRegion(cache, Inputs, Outputs);
    assert(F2);
    F2->addFnAttr(Attribute::AlwaysInline);

    for (auto U : F2->users())
      cast<Instruction>(U)->eraseFromParent();

    ssize_t induct_idx = -1;
    ssize_t off_idx = -1;
    for (auto en : llvm::enumerate(Inputs)) {
      if (en.value() == inductPN)
        induct_idx = en.index();
      if (en.value() == PN)
        off_idx = en.index();
    }
    assert(induct_idx != -1);
    assert(off_idx != -1);

    auto L2 = LI.getLoopFor(F2->getEntryBlock().getSingleSuccessor());
    auto new_idx = F2->getArg(induct_idx);
    auto L2Header = L2->getHeader();
    auto new_lidx = L2->getCanonicalInductionVariable();

    auto idxty = new_idx->getType();

    auto new_pn = F2->getArg(off_idx);
    // Find all sparse accumulates we weren't meant to handle
    {
      SmallVector<CallInst *, 1> toErase;
      // First delete any accumulates in sub loops
      for (auto SL : L2->getSubLoops())
        for (auto B : SL->getBlocks())
          for (auto &I : *B)
            if (auto CI = dyn_cast<CallInst>(&I))
              if (auto F = CI->getCalledFunction()) {
                if (F->hasFnAttribute("enzyme_sparse_accumulate")) {
                  toErase.push_back(CI);
                  continue;
                }
              }
      for (auto C : toErase)
        C->eraseFromParent();
      toErase.clear();
      // Next delete any accumulates not in latchany loops
      for (auto B : L2->getBlocks()) {
        bool guarded = false;
        if (auto P = B->getSinglePredecessor())
          if (auto S = B->getSingleSuccessor())
            if (auto BI = dyn_cast<BranchInst>(P->getTerminator()))
              if (BI->isConditional())
                for (size_t i = 0; i < 2; i++)
                  if (BI->getSuccessor(i) == B &&
                      BI->getSuccessor(1 - i) == S) {
                    auto val = BI->getCondition();
                    if (auto xori = dyn_cast<Instruction>(val))
                      if (xori->getOpcode() == Instruction::Xor)
                        val = xori->getOperand(0);
                    if (auto cmp = dyn_cast<ICmpInst>(val))
                      if (cmp->getOperand(0) == new_pn ||
                          cmp->getOperand(1) == new_pn)
                        guarded = true;
                  }
        if (guarded)
          continue;
        for (auto &I : *B)
          if (auto CI = dyn_cast<CallInst>(&I))
            if (auto F = CI->getCalledFunction()) {
              if (F->hasFnAttribute("enzyme_sparse_accumulate")) {
                toErase.push_back(CI);
                continue;
              }
            }
      }
      for (auto C : toErase)
        C->eraseFromParent();
      toErase.clear();
    }

    auto guard = L2->getLoopLatch()->getTerminator();
    assert(guard);
    IRBuilder<> G(guard);
    G.CreateRetVoid();
    guard->eraseFromParent();
    new_lidx->replaceAllUsesWith(new_idx);
    new_lidx->eraseFromParent();

    auto phterm = ph->getTerminator();
    IRBuilder<> B(phterm);

    // We extracted code, reset analyses.
    /*
    DT.reset();
    SE.forgetAllLoops();
    */

    for (auto en : llvm::enumerate(pair.second.second)) {
      auto off = en.index();
      auto &solutions = en.value().second;
      ConstraintContext ctx(SE, L, Assumptions, DT);
      SCEVExpander Exp(SE, DL, "sparseenzyme", /*preservelcssa*/ false);
      auto sols = solutions->allSolutions(Exp, idxty, phterm, ctx, B);
      SmallVector<Value *, 1> prevSols;
      for (auto [sol, condition] : sols) {
        SmallVector<Value *, 1> args(Inputs.begin(), Inputs.end());
        args[off_idx] = ConstantInt::get(idxty, off);
        args[induct_idx] = sol;
        for (auto sol2 : prevSols)
          condition = B.CreateAnd(condition, B.CreateICmpNE(sol, sol2));
        prevSols.push_back(sol);
        auto BB = B.GetInsertBlock();
        auto B2 = BB->splitBasicBlock(B.GetInsertPoint(), "poststore");
        B2->moveAfter(BB);
        BB->getTerminator()->eraseFromParent();
        B.SetInsertPoint(BB);
        auto callB = BasicBlock::Create(BB->getContext(), "tostore",
                                        BB->getParent(), B2);
        B.CreateCondBr(condition, callB, B2);
        B.SetInsertPoint(callB);
        B.CreateCall(F2, args);
        B.CreateBr(B2);
        B.SetInsertPoint(B2->getTerminator());
      }
      auto blk = en.value().first;
      auto term = blk->getTerminator();
      IRBuilder<> B2(blk);
      B2.CreateRetVoid();
      term->eraseFromParent();
    }

    PN->eraseFromParent();

    for (auto &I : *L2Header) {
      auto boundsCheck = dyn_cast<CallInst>(&I);
      if (!boundsCheck)
        continue;
      auto BF = boundsCheck->getCalledFunction();
      if (!BF)
        continue;
      if (BF->getName() != "enzyme.sparse.inbounds")
        continue;

      auto boundsCond = boundsCheck->getArgOperand(0);

      auto next = L2Header->splitBasicBlock(boundsCheck);

      auto exit = BasicBlock::Create(F2->getContext(), "bounds.exit", F2,
                                     L2Header->getNextNode());
      {
        IRBuilder B(exit);
        B.CreateRetVoid();
      }
      L2Header->getTerminator()->eraseFromParent();

      {
        IRBuilder B(L2Header);
        B.CreateCondBr(boundsCond, next, exit);
      }
      boundsCheck->eraseFromParent();
      inductPN->eraseFromParent();

      break;
    }
  }

  for (auto &F2 : F.getParent()->functions()) {
    if (startsWith(F2.getName(), "__enzyme_product")) {
      SmallVector<Instruction *, 1> toErase;
      for (llvm::User *I : F2.users()) {
        auto CB = cast<CallBase>(I);
        IRBuilder<> B(CB);
        B.setFastMathFlags(getFast());
        Value *res = nullptr;
        for (auto v : callOperands(CB)) {
          if (res == nullptr)
            res = v;
          else {
            res = B.CreateFMul(res, v);
          }
        }
        CB->replaceAllUsesWith(res);
        toErase.push_back(CB);
      }
      for (auto CB : toErase)
        CB->eraseFromParent();
    } else if (startsWith(F2.getName(), "__enzyme_sum")) {
      SmallVector<Instruction *, 1> toErase;
      for (llvm::User *I : F2.users()) {
        auto CB = cast<CallBase>(I);
        IRBuilder<> B(CB);
        B.setFastMathFlags(getFast());
        Value *res = nullptr;
        for (auto v : callOperands(CB)) {
          if (res == nullptr)
            res = v;
          else {
            res = B.CreateFAdd(res, v);
          }
        }
        CB->replaceAllUsesWith(res);
        toErase.push_back(CB);
      }
      for (auto CB : toErase)
        CB->eraseFromParent();
    }
  }
}

void replaceToDense(llvm::CallBase *CI, bool replaceAll, llvm::Function *F,
                    const llvm::DataLayout &DL) {
  auto load_fn = cast<Function>(getBaseObject(CI->getArgOperand(0)));
  auto store_fn = cast<Function>(getBaseObject(CI->getArgOperand(1)));
  size_t argstart = 2;
  size_t num_args = CI->arg_size();
  SmallVector<std::pair<Instruction *, Value *>, 1> users;

  for (auto U : CI->users()) {
    users.push_back(std::make_pair(cast<Instruction>(U), CI));
  }
  IntegerType *intTy = IntegerType::get(CI->getContext(), 64);
  auto toInt = [&](IRBuilder<> &B, llvm::Value *V) {
    if (auto PT = dyn_cast<PointerType>(V->getType())) {
      if (PT->getAddressSpace() != 0) {
#if LLVM_VERSION_MAJOR < 17
        if (CI->getContext().supportsTypedPointers()) {
          V = B.CreateAddrSpaceCast(
              V, PointerType::getUnqual(PT->getPointerElementType()));
        } else {
          V = B.CreateAddrSpaceCast(V,
                                    PointerType::getUnqual(PT->getContext()));
        }
#else
        V = B.CreateAddrSpaceCast(V, PointerType::getUnqual(PT->getContext()));
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
    if (auto SI = dyn_cast<SelectInst>(U)) {
      for (auto U : SI->users()) {
        users.push_back(std::make_pair(cast<Instruction>(U), SI));
      }
      auto tval = SI->getTrueValue();
      auto fval = SI->getFalseValue();
      auto rep = B.CreateSelect(
          SI->getCondition(),
          replacements.count(tval) ? (Value *)replacements[tval] : tval,
          replacements.count(fval) ? (Value *)replacements[fval] : fval);
      if (auto I = dyn_cast<Instruction>(rep))
        I->setDebugLoc(SI->getDebugLoc());
      replacements[SI] = rep;
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
        auto *F = CI->getCalledOperand();

        SmallVector<Value *, 1> args;
        for (auto &arg : CI->args())
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
            intTy, (DL.getTypeSizeInBits(CI->getSourceElementType()) + 7) / 8);
        gep = B.CreateMul(intTy == inds[0]->getType()
                              ? inds[0]
                              : B.CreateZExtOrTrunc(inds[0], intTy),
                          gep, "", true, true);
        gep = B.CreateAdd(B.CreatePtrToInt(replacements[val], intTy), gep);
        gep = B.CreateIntToPtr(gep, CI->getType());
      } else if (!allconst) {
        gep = B.CreateGEP(CI->getSourceElementType(), replacements[val], inds);
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
        auto fnName = load_fn->getName();
        auto found_numargs = load_fn->getFunctionType()->getNumParams();
        auto expected_numargs = args.size();
        EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                    " incorrect number of arguments to loader function ",
                    fnName, " expected ", expected_numargs, " found ",
                    found_numargs, " - ", *load_fn->getFunctionType());
        continue;
      } else {
        bool tocontinue = false;
        for (size_t i = 0; i < args.size(); i++) {
          if (load_fn->getFunctionType()->getParamType(i) !=
              args[i]->getType()) {
            auto fnName = load_fn->getName();
            EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                        " incorrect type of argument ", i,
                        " to loader function ", fnName, " expected ",
                        *args[i]->getType(), " found ",
                        load_fn->getFunctionType()->params()[i]);
            tocontinue = true;
            args[i] = UndefValue::get(args[i]->getType());
          }
        }
        if (tocontinue)
          continue;
      }
      CallInst *call = B.CreateCall(load_fn, args);
      call->setDebugLoc(LI->getDebugLoc());
      Value *tmp = call;
      if (tmp->getType() != LI->getType()) {
        if (CastInst::castIsValid(Instruction::BitCast, tmp, LI->getType()))
          tmp = B.CreateBitCast(tmp, LI->getType());
        else {
          auto fnName = load_fn->getName();
          EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                      " incorrect return type of loader function ", fnName,
                      " expected ", *LI->getType(), " found ",
                      *call->getType());
          tmp = UndefValue::get(LI->getType());
        }
      }
      LI->replaceAllUsesWith(tmp);

      if (load_fn->hasFnAttribute(Attribute::AlwaysInline)) {
        InlineFunctionInfo IFI;
        InlineFunction(*call, IFI);
      }
      toErase.push_back(LI);
      continue;
    }
    if (auto SI = dyn_cast<StoreInst>(U)) {
      assert(SI->getValueOperand() != val);
      auto diff = toInt(B, replacements[SI->getPointerOperand()]);
      SmallVector<Value *, 2> args;
      args.push_back(SI->getValueOperand());
      auto sty = store_fn->getFunctionType()->getParamType(0);
      if (args[0]->getType() != store_fn->getFunctionType()->getParamType(0)) {
        if (CastInst::castIsValid(Instruction::BitCast, args[0], sty))
          args[0] = B.CreateBitCast(args[0], sty);
        else {
          auto args0ty = args[0]->getType();
          EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                      " first argument of store function must be the type of "
                      "the store found fn arg type ",
                      *sty, " expected ", *args0ty);
          args[0] = UndefValue::get(sty);
        }
      }
      args.push_back(diff);
      for (size_t i = argstart; i < num_args; i++)
        args.push_back(CI->getArgOperand(i));

      if (store_fn->getFunctionType()->getNumParams() != args.size()) {
        auto fnName = store_fn->getName();
        auto found_numargs = store_fn->getFunctionType()->getNumParams();
        auto expected_numargs = args.size();
        EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                    " incorrect number of arguments to store function ", fnName,
                    " expected ", expected_numargs, " found ", found_numargs,
                    " - ", *store_fn->getFunctionType());
        continue;
      } else {
        bool tocontinue = false;
        for (size_t i = 0; i < args.size(); i++) {
          if (store_fn->getFunctionType()->getParamType(i) !=
              args[i]->getType()) {
            auto fnName = store_fn->getName();
            EmitFailure("IllegalSparse", CI->getDebugLoc(), CI,
                        " incorrect type of argument ", i,
                        " to storeer function ", fnName, " expected ",
                        *args[i]->getType(), " found ",
                        store_fn->getFunctionType()->params()[i]);
            tocontinue = true;
            args[i] = UndefValue::get(args[i]->getType());
          }
        }
        if (tocontinue)
          continue;
      }
      auto call = B.CreateCall(store_fn, args);
      call->setDebugLoc(SI->getDebugLoc());
      if (store_fn->hasFnAttribute(Attribute::AlwaysInline)) {
        InlineFunctionInfo IFI;
        InlineFunction(*call, IFI);
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
    EmitFailure("IllegalSparse", remaining->getDebugLoc(), remaining,
                " Illegal remaining use (", *remaining, ") of todense (", *CI,
                ") in function ", *F);
  }
}

bool LowerSparsification(llvm::Function *F, bool replaceAll) {
  auto &DL = F->getParent()->getDataLayout();
  bool changed = false;
  SmallVector<CallBase *, 1> todo;
  SetVector<BasicBlock *> toDenseBlocks;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (getFuncNameFromCall(CI).contains("__enzyme_todense")) {
          todo.push_back(CI);
          toDenseBlocks.insert(&BB);
        }
      }
    }
  }
  for (auto CI : todo) {
    changed = true;
    replaceToDense(CI, replaceAll, F, DL);
  }
  todo.clear();

  if (changed && EnzymeAutoSparsity) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    SimplifyCFGPass(SimplifyCFGOptions()).run(*F, FAM);
    InstCombinePass().run(*F, FAM);
    // required to make preheaders
    LoopSimplifyPass().run(*F, FAM);
    fixSparseIndices(*F, FAM, toDenseBlocks);
  }

  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (getFuncNameFromCall(CI).contains("__enzyme_post_sparse_todense")) {
          todo.push_back(CI);
        }
      }
    }
  }
  for (auto CI : todo) {
    changed = true;
    replaceToDense(CI, replaceAll, F, DL);
  }
  return changed;
}
