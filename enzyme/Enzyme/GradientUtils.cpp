//===- GradientUtils.cpp - Helper class and utilities for AD     ---------===//
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
// This file define two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <functional>
#include <map>
#include <string>

#include "GradientUtils.h"
#include "MustExitScalarEvolution.h"
#include "Utils.h"

#include "DifferentialUseAnalysis.h"
#include "LibraryFuncs.h"
#include "TypeAnalysis/TBAA.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#define hasAttribute hasAttributeAtIndex
#endif

using namespace llvm;

StringMap<std::function<Value *(IRBuilder<> &, CallInst *, ArrayRef<Value *>,
                                GradientUtils *)>>
    shadowHandlers;
StringMap<std::function<CallInst *(IRBuilder<> &, Value *)>> shadowErasers;

StringMap<
    std::pair<std::function<bool(IRBuilder<> &, CallInst *, GradientUtils &,
                                 Value *&, Value *&, Value *&)>,
              std::function<void(IRBuilder<> &, CallInst *,
                                 DiffeGradientUtils &, Value *)>>>
    customCallHandlers;

StringMap<std::function<bool(IRBuilder<> &, CallInst *, GradientUtils &,
                             Value *&, Value *&)>>
    customFwdCallHandlers;

extern "C" {
llvm::cl::opt<bool>
    EnzymeNewCache("enzyme-new-cache", cl::init(true), cl::Hidden,
                   cl::desc("Use new cache decision algorithm"));

llvm::cl::opt<bool> EnzymeMinCutCache("enzyme-mincut-cache", cl::init(true),
                                      cl::Hidden,
                                      cl::desc("Use Enzyme Mincut algorithm"));

llvm::cl::opt<bool> EnzymeLoopInvariantCache(
    "enzyme-loop-invariant-cache", cl::init(true), cl::Hidden,
    cl::desc("Attempt to hoist cache outside of loop"));

llvm::cl::opt<bool> EnzymeInactiveDynamic(
    "enzyme-inactive-dynamic", cl::init(true), cl::Hidden,
    cl::desc("Force wholy inactive dynamic loops to have 0 iter reverse pass"));

llvm::cl::opt<bool>
    EnzymeRuntimeActivityCheck("enzyme-runtime-activity", cl::init(false),
                               cl::Hidden,
                               cl::desc("Perform runtime activity checks"));

llvm::cl::opt<bool>
    EnzymeSharedForward("enzyme-shared-forward", cl::init(false), cl::Hidden,
                        cl::desc("Forward Shared Memory from definitions"));

llvm::cl::opt<bool>
    EnzymeRegisterReduce("enzyme-register-reduce", cl::init(false), cl::Hidden,
                         cl::desc("Reduce the amount of register reduce"));
llvm::cl::opt<bool>
    EnzymeSpeculatePHIs("enzyme-speculate-phis", cl::init(false), cl::Hidden,
                        cl::desc("Speculatively execute phi computations"));
llvm::cl::opt<bool> EnzymeFreeInternalAllocations(
    "enzyme-free-internal-allocations", cl::init(true), cl::Hidden,
    cl::desc("Always free internal allocations (disable if allocation needs "
             "access outside)"));

llvm::cl::opt<bool>
    EnzymeRematerialize("enzyme-rematerialize", cl::init(true), cl::Hidden,
                        cl::desc("Rematerialize allocations/shadows in the "
                                 "reverse rather than caching"));

llvm::cl::opt<bool>
    EnzymeVectorSplitPhi("enzyme-vector-split-phi", cl::init(true), cl::Hidden,
                         cl::desc("Split phis according to vector size"));

llvm::cl::opt<bool>
    EnzymePrintDiffUse("enzyme-print-diffuse", cl::init(false), cl::Hidden,
                       cl::desc("Print differential use analysis"));
}

SmallVector<unsigned int, 9> MD_ToCopy = {
    LLVMContext::MD_dbg,
    LLVMContext::MD_tbaa,
    LLVMContext::MD_tbaa_struct,
    LLVMContext::MD_range,
    LLVMContext::MD_nonnull,
    LLVMContext::MD_dereferenceable,
    LLVMContext::MD_dereferenceable_or_null};

static bool isPotentialLastLoopValue(llvm::Value *val,
                                     const llvm::BasicBlock *loc,
                                     const llvm::LoopInfo &LI) {
  if (llvm::Instruction *inst = llvm::dyn_cast<llvm::Instruction>(val)) {
    const llvm::Loop *InstLoop = LI.getLoopFor(inst->getParent());
    if (InstLoop == nullptr) {
      return false;
    }
    for (const llvm::Loop *L = LI.getLoopFor(loc); L; L = L->getParentLoop()) {
      if (L == InstLoop)
        return false;
    }
    return true;
  }
  return false;
}

GradientUtils::GradientUtils(
    EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
    TargetLibraryInfo &TLI_, TypeAnalysis &TA_, TypeResults TR_,
    ValueToValueMapTy &invertedPointers_,
    const SmallPtrSetImpl<Value *> &constantvalues_,
    const SmallPtrSetImpl<Value *> &activevals_, DIFFE_TYPE ReturnActivity,
    ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
    llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> &originalToNewFn_,
    DerivativeMode mode, unsigned width, bool omp)
    : CacheUtility(TLI_, newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_),
      invertedPointers(),
      OrigDT(Logic.PPC.FAM.getResult<llvm::DominatorTreeAnalysis>(*oldFunc_)),
      OrigPDT(
          Logic.PPC.FAM.getResult<llvm::PostDominatorTreeAnalysis>(*oldFunc_)),
      OrigLI(Logic.PPC.FAM.getResult<llvm::LoopAnalysis>(*oldFunc_)),
      OrigSE(Logic.PPC.FAM.getResult<llvm::ScalarEvolutionAnalysis>(*oldFunc_)),
      notForAnalysis(getGuaranteedUnreachable(oldFunc_)),
      ATA(new ActivityAnalyzer(
          Logic.PPC, Logic.PPC.getAAResultsFromFunction(oldFunc_),
          notForAnalysis, TLI_, constantvalues_, activevals_, ReturnActivity)),
      tid(nullptr), numThreads(nullptr),
      OrigAA(Logic.PPC.getAAResultsFromFunction(oldFunc_)), TA(TA_), TR(TR_),
      omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
      overwritten_args_map_ptr(nullptr) {
  if (oldFunc_->getSubprogram()) {
    assert(originalToNewFn_.hasMD());
  }

  for (BasicBlock &BB : *oldFunc) {
    for (Instruction &I : BB) {
      if (auto CI = dyn_cast<CallInst>(&I)) {
        originalCalls.push_back(CI);
      }
    }
  }

  originalToNewFn.getMDMap() = originalToNewFn_.getMDMap();

  if (oldFunc_->getSubprogram()) {
    assert(originalToNewFn.hasMD());
  }
  for (auto pair : invertedPointers_) {
    invertedPointers.insert(std::make_pair(
        (const Value *)pair.first, InvertedPointerVH(this, pair.second)));
  }
  originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
  for (BasicBlock &oBB : *oldFunc) {
    for (Instruction &oI : oBB) {
      newToOriginalFn[originalToNewFn[&oI]] = &oI;
    }
    newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
  }
  for (Argument &oArg : oldFunc->args()) {
    newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
  }
  for (BasicBlock &BB : *newFunc) {
    originalBlocks.emplace_back(&BB);
  }
  tape = nullptr;
  tapeidx = 0;
  assert(originalBlocks.size() > 0);

  SmallVector<BasicBlock *, 4> ReturningBlocks;
  for (BasicBlock &BB : *oldFunc) {
    if (isa<ReturnInst>(BB.getTerminator()))
      ReturningBlocks.push_back(&BB);
  }
  for (BasicBlock &BB : *oldFunc) {
    bool legal = true;
    for (auto BRet : ReturningBlocks) {
      if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
        legal = false;
        break;
      }
    }
    if (legal)
      BlocksDominatingAllReturns.insert(&BB);
  }
}

SmallVector<OperandBundleDef, 2>
GradientUtils::getInvertedBundles(CallInst *orig, ArrayRef<ValueType> types,
                                  IRBuilder<> &Builder2, bool lookup,
                                  const ValueToValueMapTy &available) {
  assert(!(lookup && mode == DerivativeMode::ForwardMode));

  SmallVector<OperandBundleDef, 2> OrigDefs;
  orig->getOperandBundlesAsDefs(OrigDefs);
  SmallVector<OperandBundleDef, 2> Defs;
  bool anyPrimal = false;
  bool anyShadow = false;
  for (auto ty : types) {
    if (ty == ValueType::Primal || ty == ValueType::Both)
      anyPrimal = true;
    if (ty == ValueType::Shadow || ty == ValueType::Both)
      anyShadow = true;
  }
  for (auto bund : OrigDefs) {
    // Only handle jl_roots tag (for now).
    if (bund.getTag() != "jl_roots") {
      errs() << "unsupported tag " << bund.getTag() << " for " << *orig << "\n";
      llvm_unreachable("unsupported tag");
    }
    SmallVector<Value *, 2> bunds;
    // In the future we can reduce the number of roots
    // we preserve by identifying which operands they
    // correspond to. For now, fall back and preserve all
    // primals and shadows
    // assert(bund.inputs().size() == types.size());
    for (auto inp : bund.inputs()) {
      if (anyPrimal) {
        Value *newv = getNewFromOriginal(inp);
        if (lookup)
          newv = lookupM(newv, Builder2, available);
        bunds.push_back(newv);
      }
      if (anyShadow && !isConstantValue(inp)) {
        Value *shadow = invertPointerM(inp, Builder2);
        if (lookup)
          shadow = lookupM(shadow, Builder2);
        bunds.push_back(shadow);
      }
    }
    Defs.push_back(OperandBundleDef(bund.getTag().str(), bunds));
  }
  return Defs;
}

Value *GradientUtils::getNewIfOriginal(Value *originst) const {
  assert(originst);
  auto f = originalToNewFn.find(originst);
  if (f == originalToNewFn.end()) {
    return originst;
  }
  assert(f != originalToNewFn.end());
  if (f->second == nullptr) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *originst << "\n";
  }
  assert(f->second);
  return f->second;
}

Value *GradientUtils::ompThreadId() {
  if (tid)
    return tid;
  IRBuilder<> B(inversionAllocs);

  auto FT = FunctionType::get(Type::getInt64Ty(B.getContext()),
                              ArrayRef<Type *>(), false);
  auto FN = newFunc->getParent()->getOrInsertFunction("omp_get_thread_num", FT);
  auto CI = B.CreateCall(FN);
  if (auto F = getFunctionFromCall(CI)) {
#if LLVM_VERSION_MAJOR >= 16
    F->setOnlyAccessesInaccessibleMemory();
    F->setOnlyReadsMemory();
#else
    F->addFnAttr(Attribute::InaccessibleMemOnly);
    F->addFnAttr(Attribute::ReadOnly);
#endif
  }
#if LLVM_VERSION_MAJOR >= 16
  CI->setOnlyAccessesInaccessibleMemory();
  CI->setOnlyReadsMemory();
#else
  CI->addAttribute(AttributeList::FunctionIndex,
                   Attribute::InaccessibleMemOnly);
  CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
#endif
  return tid = CI;
}

Value *GradientUtils::ompNumThreads() {
  if (numThreads)
    return numThreads;
  IRBuilder<> B(inversionAllocs);

  auto FT = FunctionType::get(Type::getInt64Ty(B.getContext()),
                              ArrayRef<Type *>(), false);
  auto FN =
      newFunc->getParent()->getOrInsertFunction("omp_get_max_threads", FT);
  auto CI = B.CreateCall(FN);
  if (auto F = getFunctionFromCall(CI)) {
#if LLVM_VERSION_MAJOR >= 16
    F->setOnlyAccessesInaccessibleMemory();
    F->setOnlyReadsMemory();
#else
    F->addFnAttr(Attribute::InaccessibleMemOnly);
    F->addFnAttr(Attribute::ReadOnly);
#endif
  }
#if LLVM_VERSION_MAJOR >= 16
  CI->setOnlyAccessesInaccessibleMemory();
  CI->setOnlyReadsMemory();
#else
  CI->addAttribute(AttributeList::FunctionIndex,
                   Attribute::InaccessibleMemOnly);
  CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
#endif
  return numThreads = CI;
}

Value *GradientUtils::getOrInsertTotalMultiplicativeProduct(Value *val,
                                                            LoopContext &lc) {
  // TODO optimize if val is invariant to loopContext
  assert(val->getType()->isFPOrFPVectorTy());
  for (auto &I : *lc.header) {
    if (auto PN = dyn_cast<PHINode>(&I)) {
      if (PN->getType() != val->getType())
        continue;
      Value *ival = PN->getIncomingValueForBlock(lc.preheader);
      if (auto CDV = dyn_cast<ConstantDataVector>(ival)) {
        if (CDV->isSplat())
          ival = CDV->getSplatValue();
      }
      if (auto C = dyn_cast<ConstantFP>(ival)) {
        if (!C->isExactlyValue(APFloat(C->getType()->getFltSemantics(), "1"))) {
          continue;
        }
      } else
        continue;
      for (auto IB : PN->blocks()) {
        if (IB == lc.preheader)
          continue;

        if (auto BO =
                dyn_cast<BinaryOperator>(PN->getIncomingValueForBlock(IB))) {
          if (BO->getOpcode() != BinaryOperator::FMul)
            goto continueOutermost;
          if (BO->getOperand(0) == PN && BO->getOperand(1) == val)
            return BO;
          if (BO->getOperand(1) == PN && BO->getOperand(0) == val)
            return BO;
        } else
          goto continueOutermost;
      }
    } else
      break;
  continueOutermost:;
  }

  IRBuilder<> lbuilder(lc.header, lc.header->begin());
  auto PN = lbuilder.CreatePHI(val->getType(), 2);
  Constant *One = ConstantFP::get(val->getType()->getScalarType(), "1");
  if (VectorType *VTy = dyn_cast<VectorType>(val->getType())) {
#if LLVM_VERSION_MAJOR >= 11
    One = ConstantVector::getSplat(VTy->getElementCount(), One);
#else
    One = ConstantVector::getSplat(VTy->getNumElements(), One);
#endif
  }
  PN->addIncoming(One, lc.preheader);
  lbuilder.SetInsertPoint(lc.header->getFirstNonPHI());
  if (auto inst = dyn_cast<Instruction>(val)) {
    if (DT.dominates(PN, inst))
      lbuilder.SetInsertPoint(inst->getNextNode());
  }
  Value *red = lbuilder.CreateFMul(PN, val);
  for (auto pred : predecessors(lc.header)) {
    if (pred == lc.preheader)
      continue;
    PN->addIncoming(red, pred);
  }
  return red;
}

Value *GradientUtils::getOrInsertConditionalIndex(Value *val, LoopContext &lc,
                                                  bool pickTrue) {
  assert(val->getType()->isIntOrIntVectorTy(1));
  // TODO optimize if val is invariant to loopContext
  for (auto &I : *lc.header) {
    if (auto PN = dyn_cast<PHINode>(&I)) {
      if (PN->getNumIncomingValues() == 0)
        continue;
      if (PN->getType() != lc.incvar->getType())
        continue;
      Value *ival = PN->getIncomingValueForBlock(lc.preheader);
      if (auto C = dyn_cast<Constant>(ival)) {
        if (!C->isNullValue()) {
          continue;
        }
      } else
        continue;
      for (auto IB : PN->blocks()) {
        if (IB == lc.preheader)
          continue;

        if (auto SI = dyn_cast<SelectInst>(PN->getIncomingValueForBlock(IB))) {
          if (SI->getCondition() != val)
            goto continueOutermost;
          if (pickTrue && SI->getFalseValue() == PN) {
            // TODO handle vector of
            if (SI->getTrueValue() == lc.incvar)
              return SI;
          }
          if (!pickTrue && SI->getTrueValue() == PN) {
            // TODO handle vector of
            if (SI->getFalseValue() == lc.incvar)
              return SI;
          }
        } else
          goto continueOutermost;
      }
    } else
      break;
  continueOutermost:;
  }

  IRBuilder<> lbuilder(lc.header, lc.header->begin());
  auto PN = lbuilder.CreatePHI(lc.incvar->getType(), 2);
  Constant *Zero =
      Constant::getNullValue(lc.incvar->getType()->getScalarType());
  PN->addIncoming(Zero, lc.preheader);
  lbuilder.SetInsertPoint(lc.incvar->getNextNode());
  Value *red = lc.incvar;
  if (VectorType *VTy = dyn_cast<VectorType>(val->getType())) {
#if LLVM_VERSION_MAJOR >= 12
    red = lbuilder.CreateVectorSplat(VTy->getElementCount(), red);
#else
    red = lbuilder.CreateVectorSplat(VTy->getNumElements(), red);
#endif
  }
  if (auto inst = dyn_cast<Instruction>(val)) {
    if (DT.dominates(PN, inst))
      lbuilder.SetInsertPoint(inst->getNextNode());
  }
  assert(red->getType() == PN->getType());
  red = lbuilder.CreateSelect(val, pickTrue ? red : PN, pickTrue ? PN : red);
  for (auto pred : predecessors(lc.header)) {
    if (pred == lc.preheader)
      continue;
    PN->addIncoming(red, pred);
  }
  return red;
}

bool GradientUtils::assumeDynamicLoopOfSizeOne(Loop *L) const {
  if (!EnzymeInactiveDynamic)
    return false;
  auto OL = OrigLI.getLoopFor(isOriginal(L->getHeader()));
  assert(OL);
  for (auto OB : OL->getBlocks()) {
    for (auto &OI : *OB) {
      if (!isConstantInstruction(&OI))
        return false;
    }
  }
  return true;
}

DebugLoc GradientUtils::getNewFromOriginal(const DebugLoc L) const {
  if (L.get() == nullptr)
    return nullptr;
  if (!oldFunc->getSubprogram())
    return L;
  assert(originalToNewFn.hasMD());
  auto opt = originalToNewFn.getMappedMD(L.getAsMDNode());
#if LLVM_VERSION_MAJOR >= 16
  if (!opt.has_value())
    return L;
  assert(opt.has_value());
  return DebugLoc(cast<MDNode>(opt.value()));
#else
  if (!opt.hasValue())
    return L;
  assert(opt.hasValue());
  return DebugLoc(cast<MDNode>(*opt.getPointer()));
#endif
}

Value *GradientUtils::getNewFromOriginal(const Value *originst) const {
  assert(originst);
  if (isa<ConstantData>(originst))
    return const_cast<Value *>(originst);
  auto f = originalToNewFn.find(originst);
  if (f == originalToNewFn.end()) {
    errs() << *oldFunc << "\n";
    errs() << *newFunc << "\n";
    dumpMap(originalToNewFn, [&](const Value *const &v) -> bool {
      if (isa<Instruction>(originst))
        return isa<Instruction>(v);
      if (isa<BasicBlock>(originst))
        return isa<BasicBlock>(v);
      if (isa<Function>(originst))
        return isa<Function>(v);
      if (isa<Argument>(originst))
        return isa<Argument>(v);
      if (isa<Constant>(originst))
        return isa<Constant>(v);
      return true;
    });
    llvm::errs() << *originst << "\n";
  }
  assert(f != originalToNewFn.end());
  if (f->second == nullptr) {
    errs() << *oldFunc << "\n";
    errs() << *newFunc << "\n";
    errs() << *originst << "\n";
  }
  assert(f->second);
  return f->second;
}

Instruction *
GradientUtils::getNewFromOriginal(const Instruction *newinst) const {
  auto ninst = getNewFromOriginal((Value *)newinst);
  if (!isa<Instruction>(ninst)) {
    errs() << *oldFunc << "\n";
    errs() << *newFunc << "\n";
    errs() << *ninst << " - " << *newinst << "\n";
  }
  return cast<Instruction>(ninst);
}

BasicBlock *GradientUtils::getNewFromOriginal(const BasicBlock *newinst) const {
  return cast<BasicBlock>(getNewFromOriginal((Value *)newinst));
}

Value *GradientUtils::hasUninverted(const Value *inverted) const {
  for (auto v : invertedPointers) {
    if (v.second == inverted)
      return const_cast<Value *>(v.first);
  }
  return nullptr;
}

BasicBlock *GradientUtils::getOriginalFromNew(const BasicBlock *newinst) const {
  assert(newinst->getParent() == newFunc);
  auto found = newToOriginalFn.find(newinst);
  assert(found != newToOriginalFn.end());
  Value *res = found->second;
  return cast<BasicBlock>(res);
}

Value *GradientUtils::isOriginal(const Value *newinst) const {
  if (isa<Constant>(newinst) || isa<UndefValue>(newinst))
    return const_cast<Value *>(newinst);
  if (auto arg = dyn_cast<Argument>(newinst)) {
    assert(arg->getParent() == newFunc);
  }
  if (auto inst = dyn_cast<Instruction>(newinst)) {
    assert(inst->getParent()->getParent() == newFunc);
  }
  auto found = newToOriginalFn.find(newinst);
  if (found == newToOriginalFn.end())
    return nullptr;
  return found->second;
}

Instruction *GradientUtils::isOriginal(const Instruction *newinst) const {
  return cast_or_null<Instruction>(isOriginal((const Value *)newinst));
}

BasicBlock *GradientUtils::isOriginal(const BasicBlock *newinst) const {
  return cast_or_null<BasicBlock>(isOriginal((const Value *)newinst));
}

Value *GradientUtils::unwrapM(Value *const val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &available,
                              UnwrapMode unwrapMode, BasicBlock *scope,
                              bool permitCache) {
  assert(val);
  assert(val->getName() != "<badref>");
  assert(val->getType());

  for (auto pair : available) {
    assert(pair.first);
    assert(pair.first->getType());
    if (pair.second) {
      assert(pair.second->getType());
      assert(pair.first->getType() == pair.second->getType());
    }
  }

  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_mustcache")) {
    return val;
  }

  if (available.count(val)) {
    auto avail = available.lookup(val);
    assert(avail->getType());
    if (avail->getType() != val->getType()) {
      llvm::errs() << "val: " << *val << "\n";
      llvm::errs() << "available[val]: " << *available.lookup(val) << "\n";
    }
    assert(available.lookup(val)->getType() == val->getType());
    return available.lookup(val);
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (inversionAllocs && inst->getParent() == inversionAllocs) {
      return val;
    }
    // if (inst->getParent() == &newFunc->getEntryBlock()) {
    //  return inst;
    //}
    if (inst->getParent()->getParent() == newFunc &&
        isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from block domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
    assert(!TapesToPreventRecomputation.count(inst));
  }

  std::pair<Value *, BasicBlock *> idx = std::make_pair(val, scope);
  // assert(!val->getName().startswith("$tapeload"));
  if (permitCache) {
    auto found0 = unwrap_cache.find(BuilderM.GetInsertBlock());
    if (found0 != unwrap_cache.end()) {
      auto found1 = found0->second.find(idx.first);
      if (found1 != found0->second.end()) {
        auto found2 = found1->second.find(idx.second);
        if (found2 != found1->second.end()) {

          auto cachedValue = found2->second;
          if (cachedValue == nullptr) {
            found1->second.erase(idx.second);
            if (found1->second.size() == 0) {
              found0->second.erase(idx.first);
            }
          } else {
            if (cachedValue->getType() != val->getType()) {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "val: " << *val << "\n";
              llvm::errs() << "unwrap_cache[cidx]: " << *cachedValue << "\n";
            }
            assert(cachedValue->getType() == val->getType());
            return cachedValue;
          }
        }
      }
    }
  }

  if (this->mode == DerivativeMode::ReverseModeGradient ||
      this->mode == DerivativeMode::ForwardModeSplit ||
      this->mode == DerivativeMode::ReverseModeCombined)
    if (auto inst = dyn_cast<Instruction>(val)) {
      if (inst->getParent()->getParent() == newFunc) {
        if (unwrapMode == UnwrapMode::LegalFullUnwrap &&
            this->mode != DerivativeMode::ReverseModeCombined) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead
          Instruction *orig = isOriginal(inst);
          // If a given value has been chosen to be cached, do not compute the
          // operands to unwrap it, instead simply emit a placeholder to be
          // replaced by the cache load later. This placeholder should only be
          // returned when the original value would be recomputed (e.g. this
          // function would not return null). Since this case assumes everything
          // can be recomputed, simply return the placeholder.
          if (orig && knownRecomputeHeuristic.find(orig) !=
                          knownRecomputeHeuristic.end()) {
            if (!knownRecomputeHeuristic[orig]) {
              assert(inst->getParent()->getParent() == newFunc);
              auto placeholder = BuilderM.CreatePHI(
                  val->getType(), 0, val->getName() + "_krcLFUreplacement");
              unwrappedLoads[placeholder] = inst;
              SmallVector<Metadata *, 1> avail;
              for (auto pair : available)
                if (pair.second)
                  avail.push_back(MDNode::get(
                      placeholder->getContext(),
                      {ValueAsMetadata::get(const_cast<Value *>(pair.first)),
                       ValueAsMetadata::get(pair.second)}));
              placeholder->setMetadata(
                  "enzyme_available",
                  MDNode::get(placeholder->getContext(), avail));
              if (!permitCache)
                return placeholder;
              return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                                 [idx.second] = placeholder;
            }
          }
        } else if (unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead
          Instruction *orig = isOriginal(inst);
          // If a given value has been chosen to be cached, do not compute the
          // operands to unwrap it, instead simply emit a placeholder to be
          // replaced by the cache load later. This placeholder should only be
          // returned when the original value would be recomputed (e.g. this
          // function would not return null). See note below about the condition
          // as applied to this case.
          if (orig && knownRecomputeHeuristic.find(orig) !=
                          knownRecomputeHeuristic.end()) {
            if (!knownRecomputeHeuristic[orig]) {
              if (mode == DerivativeMode::ReverseModeCombined) {
                // Don't unnecessarily cache a value if the caching
                // heuristic says we should preserve this precise (and not
                // an lcssa wrapped) value
                if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
                  Value *nval = inst;
                  if (scope)
                    nval = fixLCSSA(inst, scope);
                  if (nval == inst)
                    goto endCheck;
                }
              } else {
                // Note that this logic (original load must dominate or
                // alternatively be in the reverse block) is only valid iff when
                // applicable (here if in split mode), an overwritten load
                // cannot be hoisted outside of a loop to be used as a loop
                // limit. This optimization is currently done in the combined
                // mode (e.g. if a load isn't modified between a prior insertion
                // point and the actual load, it is legal to recompute).
                if (!isOriginalBlock(*BuilderM.GetInsertBlock()) ||
                    DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                  assert(inst->getParent()->getParent() == newFunc);
                  auto placeholder = BuilderM.CreatePHI(
                      val->getType(), 0,
                      val->getName() + "_krcAFUWLreplacement");
                  unwrappedLoads[placeholder] = inst;
                  SmallVector<Metadata *, 1> avail;
                  for (auto pair : available)
                    if (pair.second)
                      avail.push_back(
                          MDNode::get(placeholder->getContext(),
                                      {ValueAsMetadata::get(
                                           const_cast<Value *>(pair.first)),
                                       ValueAsMetadata::get(pair.second)}));
                  placeholder->setMetadata(
                      "enzyme_available",
                      MDNode::get(placeholder->getContext(), avail));
                  if (!permitCache)
                    return placeholder;
                  return unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                                     [idx.second] = placeholder;
                }
              }
            }
          }
        } else if (unwrapMode != UnwrapMode::LegalFullUnwrapNoTapeReplace &&
                   mode != DerivativeMode::ReverseModeCombined) {
          // TODO this isOriginal is a bottleneck, the new mapping of
          // knownRecompute should be precomputed and maintained to lookup
          // instead

          // If a given value has been chosen to be cached, do not compute the
          // operands to unwrap it if it is not legal to do so. This prevents
          // the creation of unused versions of the instruction's operand, which
          // may be assumed to never be used and thus cause an error when they
          // are inadvertantly cached.
          Value *orig = isOriginal(val);
          if (orig && knownRecomputeHeuristic.find(orig) !=
                          knownRecomputeHeuristic.end()) {
            if (!knownRecomputeHeuristic[orig]) {
              if (!legalRecompute(orig, available, &BuilderM))
                return nullptr;

              assert(isa<LoadInst>(orig) == isa<LoadInst>(val));
            }
          }
        }
      }
    }

#define getOpFullest(Builder, vtmp, frominst, lookupInst, check)               \
  ({                                                                           \
    Value *v = vtmp;                                                           \
    BasicBlock *origParent = frominst;                                         \
    Value *___res;                                                             \
    if (unwrapMode == UnwrapMode::LegalFullUnwrap ||                           \
        unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||              \
        unwrapMode == UnwrapMode::AttemptFullUnwrap ||                         \
        unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {               \
      if (v == val)                                                            \
        ___res = nullptr;                                                      \
      else                                                                     \
        ___res = unwrapM(v, Builder, available, unwrapMode, origParent,        \
                         permitCache);                                         \
      if (!___res && unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {  \
        bool noLookup = false;                                                 \
        auto found = available.find(v);                                        \
        if (found != available.end() && !found->second)                        \
          noLookup = true;                                                     \
        if (auto opinst = dyn_cast<Instruction>(v))                            \
          if (isOriginalBlock(*Builder.GetInsertBlock())) {                    \
            if (!DT.dominates(opinst, &*Builder.GetInsertPoint()))             \
              noLookup = true;                                                 \
          }                                                                    \
        origParent = lookupInst;                                               \
        if (!noLookup)                                                         \
          ___res = lookupM(v, Builder, available, v != val, origParent);       \
      }                                                                        \
      if (___res)                                                              \
        assert(___res->getType() == v->getType() && "uw");                     \
    } else {                                                                   \
      origParent = lookupInst;                                                 \
      assert(unwrapMode == UnwrapMode::AttemptSingleUnwrap);                   \
      auto found = available.find(v);                                          \
      assert(found == available.end() || found->second);                       \
      ___res = lookupM(v, Builder, available, v != val, origParent);           \
      if (___res && ___res->getType() != v->getType()) {                       \
        llvm::errs() << *newFunc << "\n";                                      \
        llvm::errs() << " v = " << *v << " res = " << *___res << "\n";         \
      }                                                                        \
      if (___res)                                                              \
        assert(___res->getType() == v->getType() && "lu");                     \
    }                                                                          \
    ___res;                                                                    \
  })
#define getOpFull(Builder, vtmp, frominst)                                     \
  ({                                                                           \
    BasicBlock *parent = scope;                                                \
    if (parent == nullptr)                                                     \
      if (auto originst = dyn_cast<Instruction>(val))                          \
        parent = originst->getParent();                                        \
    getOpFullest(Builder, vtmp, frominst, parent, true);                       \
  })
#define getOpUnchecked(vtmp)                                                   \
  ({                                                                           \
    BasicBlock *parent = scope;                                                \
    getOpFullest(BuilderM, vtmp, parent, parent, false);                       \
  })
#define getOp(vtmp)                                                            \
  ({                                                                           \
    BasicBlock *parent = scope;                                                \
    if (parent == nullptr)                                                     \
      if (auto originst = dyn_cast<Instruction>(val))                          \
        parent = originst->getParent();                                        \
    getOpFullest(BuilderM, vtmp, parent, parent, true);                        \
  })

  if (isa<Argument>(val) || isa<Constant>(val)) {
    return val;
#if LLVM_VERSION_MAJOR >= 10
  } else if (auto op = dyn_cast<FreezeInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFreeze(op0, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
#endif
  } else if (auto op = dyn_cast<CastInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(),
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
    auto op0 = getOp(op->getAggregateOperand());
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(),
                                                op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<InsertValueInst>(val)) {
    // Unwrapped Aggregate, Indices, parent
    SmallVector<std::tuple<Value *, ArrayRef<unsigned>, InsertValueInst *>, 1>
        insertElements;

    Value *agg = op;
    while (auto op1 = dyn_cast<InsertValueInst>(agg)) {
      if (Value *orig = isOriginal(op1)) {
        if (knownRecomputeHeuristic.count(orig)) {
          if (!knownRecomputeHeuristic[orig]) {
            break;
          }
        }
      }
      Value *valOp = op1->getInsertedValueOperand();
      valOp = getOp(valOp);
      if (valOp == nullptr)
        goto endCheck;
      insertElements.push_back({valOp, op1->getIndices(), op1});
      agg = op1->getAggregateOperand();
    }

    Value *toreturn = getOp(agg);
    if (toreturn == nullptr)
      goto endCheck;
    for (auto &&[valOp, idcs, parent] : reverse(insertElements)) {
      toreturn = BuilderM.CreateInsertValue(toreturn, valOp, idcs,
                                            parent->getName() + "_unwrap");

      if (permitCache)
        unwrap_cache[BuilderM.GetInsertBlock()][parent][idx.second] = toreturn;
      if (auto newi = dyn_cast<Instruction>(toreturn)) {
        newi->copyIRFlags(parent);
        unwrappedLoads[newi] = val;
        if (newi->getParent()->getParent() != parent->getParent()->getParent())
          newi->setDebugLoc(nullptr);
      }
    }

    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ExtractElementInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateExtractElement(op0, op1, op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<InsertElementInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto op2 = getOp(op->getOperand(2));
    if (op2 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateInsertElement(op0, op1, op2, op->getName() + "_unwrap");
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ShuffleVectorInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
#if LLVM_VERSION_MAJOR >= 11
    auto toreturn = BuilderM.CreateShuffleVector(
        op0, op1, op->getShuffleMaskForBitcode(), op->getName() + "'_unwrap");
#else
    auto toreturn = BuilderM.CreateShuffleVector(op0, op1, op->getOperand(2),
                                                 op->getName() + "'_unwrap");
#endif
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<BinaryOperator>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    if (op0->getType() != op1->getType()) {
      llvm::errs() << " op: " << *op << " op0: " << *op0 << " op1: " << *op1
                   << " p0: " << *op->getOperand(0)
                   << "  p1: " << *op->getOperand(1) << "\n";
    }
    assert(op0->getType() == op1->getType());
    auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1,
                                         op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ICmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<FCmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (isa<FPMathOperator>(val) &&
             cast<FPMathOperator>(val)->getOpcode() == Instruction::FNeg) {
    auto op = cast<FPMathOperator>(val);
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFNeg(op0, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() !=
          cast<Instruction>(val)->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<SelectInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto op2 = getOp(op->getOperand(2));
    if (op2 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateSelect(op0, op1, op2, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(op);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != op->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
    auto ptr = getOp(inst->getPointerOperand());
    if (ptr == nullptr)
      goto endCheck;
    SmallVector<Value *, 4> ind;
    // llvm::errs() << "inst: " << *inst << "\n";
    for (unsigned i = 0; i < inst->getNumIndices(); ++i) {
      Value *a = inst->getOperand(1 + i);
      auto op = getOp(a);
      if (op == nullptr)
        goto endCheck;
      ind.push_back(op);
    }
    auto toreturn = BuilderM.CreateGEP(inst->getSourceElementType(), ptr, ind,
                                       inst->getName() + "_unwrap");
    if (isa<GetElementPtrInst>(toreturn))
      cast<GetElementPtrInst>(toreturn)->setIsInBounds(inst->isInBounds());
    if (auto newi = dyn_cast<Instruction>(toreturn)) {
      newi->copyIRFlags(inst);
      unwrappedLoads[newi] = val;
      if (newi->getParent()->getParent() != inst->getParent()->getParent())
        newi->setDebugLoc(nullptr);
    }
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto load = dyn_cast<LoadInst>(val)) {
    if (load->getMetadata("enzyme_noneedunwrap"))
      return load;

    bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                     unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
    if (!legalMove) {
      BasicBlock *parent = nullptr;
      if (isOriginalBlock(*BuilderM.GetInsertBlock()))
        parent = BuilderM.GetInsertBlock();
      if (!parent ||
          LI.getLoopFor(parent) == LI.getLoopFor(load->getParent()) ||
          DT.dominates(load, parent)) {
        legalMove = legalRecompute(load, available, &BuilderM);
      } else {
        legalMove =
            legalRecompute(load, available, &BuilderM, /*reverse*/ false,
                           /*legalRecomputeCache*/ false);
      }
    }
    if (!legalMove) {
      auto &warnMap = UnwrappedWarnings[load];
      if (!warnMap.count(BuilderM.GetInsertBlock())) {
        EmitWarning("UncacheableUnwrap", *load, "Load cannot be unwrapped ",
                    *load, " in ", BuilderM.GetInsertBlock()->getName(), " - ",
                    BuilderM.GetInsertBlock()->getParent()->getName(), " mode ",
                    unwrapMode);
        warnMap.insert(BuilderM.GetInsertBlock());
      }
      goto endCheck;
    }

    Value *pidx = getOp(load->getOperand(0));

    if (pidx == nullptr) {
      goto endCheck;
    }

    if (pidx->getType() != load->getOperand(0)->getType()) {
      llvm::errs() << "load: " << *load << "\n";
      llvm::errs() << "load->getOperand(0): " << *load->getOperand(0) << "\n";
      llvm::errs() << "idx: " << *pidx << " unwrapping: " << *val
                   << " mode=" << unwrapMode << "\n";
    }
    assert(pidx->getType() == load->getOperand(0)->getType());

    auto toreturn =
        BuilderM.CreateLoad(load->getType(), pidx, load->getName() + "_unwrap");
    llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
    ToCopy2.push_back(LLVMContext::MD_noalias);
    ToCopy2.push_back(LLVMContext::MD_alias_scope);
    toreturn->copyMetadata(*load, ToCopy2);
    toreturn->copyIRFlags(load);
    if (load->getParent()->getParent() == newFunc)
      if (auto orig = isOriginal(load)) {
        SmallVector<Metadata *, 1> scopeMD = {
            getDerivativeAliasScope(orig->getOperand(0), -1)};
        if (auto prev = orig->getMetadata(LLVMContext::MD_alias_scope)) {
          for (auto &M : cast<MDNode>(prev)->operands()) {
            scopeMD.push_back(M);
          }
        }
        auto scope = MDNode::get(orig->getContext(), scopeMD);
        toreturn->setMetadata(LLVMContext::MD_alias_scope, scope);

        SmallVector<Metadata *, 1> MDs;
        for (size_t j = 0; j < getWidth(); j++) {
          MDs.push_back(getDerivativeAliasScope(orig->getOperand(0), j));
        }
        if (auto prev = orig->getMetadata(LLVMContext::MD_noalias)) {
          for (auto &M : cast<MDNode>(prev)->operands()) {
            MDs.push_back(M);
          }
        }
        auto noscope = MDNode::get(orig->getContext(), MDs);
        toreturn->setMetadata(LLVMContext::MD_noalias, noscope);
      }
    unwrappedLoads[toreturn] = load;
    if (toreturn->getParent()->getParent() != load->getParent()->getParent())
      toreturn->setDebugLoc(nullptr);
    else
      toreturn->setDebugLoc(getNewFromOriginal(load->getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 10
    toreturn->setAlignment(load->getAlign());
#else
    toreturn->setAlignment(load->getAlignment());
#endif
    toreturn->setVolatile(load->isVolatile());
    toreturn->setOrdering(load->getOrdering());
    toreturn->setSyncScopeID(load->getSyncScopeID());
    if (toreturn->getParent()->getParent() != load->getParent()->getParent())
      toreturn->setDebugLoc(nullptr);
    else
      toreturn->setDebugLoc(getNewFromOriginal(load->getDebugLoc()));
    toreturn->setMetadata(LLVMContext::MD_tbaa,
                          load->getMetadata(LLVMContext::MD_tbaa));
    auto invar_group = load->getMetadata(LLVMContext::MD_invariant_group);
    if (!invar_group) {
      bool legal = true;
      if (load->getParent()->getParent() != newFunc)
        legal = false;
      else if (auto norig = isOriginal(load))
        for (const auto &pair : rematerializableAllocations) {
          for (auto V : pair.second.loads)
            if (V == norig) {
              legal = false;
              break;
            }
          if (!legal)
            break;
        }
      if (legal) {
        invar_group = MDNode::getDistinct(load->getContext(), {});
        load->setMetadata(LLVMContext::MD_invariant_group, invar_group);
      }
    }
    toreturn->setMetadata(LLVMContext::MD_invariant_group, invar_group);
    // TODO adding to cache only legal if no alias of any future writes
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<CallInst>(val)) {

    bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                     unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
    if (!legalMove) {
      legalMove = legalRecompute(op, available, &BuilderM);
    }
    if (!legalMove)
      goto endCheck;

    SmallVector<Value *, 4> args;
#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < op->arg_size(); ++i)
#else
    for (unsigned i = 0; i < op->getNumArgOperands(); ++i)
#endif
    {
      args.emplace_back(getOp(op->getArgOperand(i)));
      if (args[i] == nullptr)
        goto endCheck;
    }

#if LLVM_VERSION_MAJOR >= 11
    Value *fn = getOp(op->getCalledOperand());
#else
    Value *fn = getOp(op->getCalledValue());
#endif
    if (fn == nullptr)
      goto endCheck;

    auto toreturn =
        cast<CallInst>(BuilderM.CreateCall(op->getFunctionType(), fn, args));
    toreturn->copyIRFlags(op);
    toreturn->setAttributes(op->getAttributes());
    toreturn->setCallingConv(op->getCallingConv());
    toreturn->setTailCallKind(op->getTailCallKind());
    if (toreturn->getParent()->getParent() == op->getParent()->getParent())
      toreturn->setDebugLoc(getNewFromOriginal(op->getDebugLoc()));
    else
      toreturn->setDebugLoc(nullptr);
    if (permitCache)
      unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = toreturn;
    unwrappedLoads[toreturn] = val;
    return toreturn;
  } else if (auto phi = dyn_cast<PHINode>(val)) {
    if (phi->getNumIncomingValues() == 0) {
      // This is a placeholder shadow for a load, rather than falling
      // back to the uncached variant, use the proper procedure for
      // an inverted load
      if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(phi))) {
        // Almost identical code to unwrap load (replacing use of shadow
        // where appropriate)
        if (dli->getMetadata("enzyme_noneedunwrap"))
          return dli;

        bool legalMove = unwrapMode == UnwrapMode::LegalFullUnwrap ||
                         unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace;
        if (!legalMove) {
          // TODO actually consider whether this is legal to move to the new
          // location, rather than recomputable anywhere
          legalMove = legalRecompute(dli, available, &BuilderM);
        }
        if (!legalMove) {
          auto &warnMap = UnwrappedWarnings[phi];
          if (!warnMap.count(BuilderM.GetInsertBlock())) {
            EmitWarning("UncacheableUnwrap", *dli,
                        "Differential Load cannot be unwrapped ", *dli, " in ",
                        BuilderM.GetInsertBlock()->getName(), " mode ",
                        unwrapMode);
            warnMap.insert(BuilderM.GetInsertBlock());
          }
          return nullptr;
        }

        Value *pidx = nullptr;

        if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
          pidx = invertPointerM(dli->getOperand(0), BuilderM);
        } else {
          pidx = lookupM(invertPointerM(dli->getOperand(0), BuilderM), BuilderM,
                         available);
        }

        if (pidx == nullptr)
          goto endCheck;

        if (pidx->getType() != getShadowType(dli->getOperand(0)->getType())) {
          llvm::errs() << "dli: " << *dli << "\n";
          llvm::errs() << "dli->getOperand(0): " << *dli->getOperand(0) << "\n";
          llvm::errs() << "pidx: " << *pidx << "\n";
        }
        assert(pidx->getType() == getShadowType(dli->getOperand(0)->getType()));

        size_t s_idx = 0;
        Value *toreturn = applyChainRule(
            dli->getType(), BuilderM,
            [&](Value *pidx) {
              auto toreturn = BuilderM.CreateLoad(dli->getType(), pidx,
                                                  phi->getName() + "_unwrap");
              if (auto newi = dyn_cast<Instruction>(toreturn)) {
                newi->copyIRFlags(dli);
                unwrappedLoads[toreturn] = dli;
              }
#if LLVM_VERSION_MAJOR >= 10
              toreturn->setAlignment(dli->getAlign());
#else
              toreturn->setAlignment(dli->getAlignment());
#endif
              toreturn->setVolatile(dli->isVolatile());
              toreturn->setOrdering(dli->getOrdering());
              toreturn->setSyncScopeID(dli->getSyncScopeID());
              llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
              toreturn->copyMetadata(*dli, ToCopy2);
              SmallVector<Metadata *, 1> scopeMD = {
                  getDerivativeAliasScope(dli->getOperand(0), s_idx)};
              if (auto prev = dli->getMetadata(LLVMContext::MD_alias_scope)) {
                for (auto &M : cast<MDNode>(prev)->operands()) {
                  scopeMD.push_back(M);
                }
              }
              auto scope = MDNode::get(dli->getContext(), scopeMD);
              toreturn->setMetadata(LLVMContext::MD_alias_scope, scope);

              SmallVector<Metadata *, 1> MDs;
              for (ssize_t j = -1; j < getWidth(); j++) {
                if (j != (ssize_t)s_idx)
                  MDs.push_back(getDerivativeAliasScope(dli->getOperand(0), j));
              }
              if (auto prev = dli->getMetadata(LLVMContext::MD_noalias)) {
                for (auto &M : cast<MDNode>(prev)->operands()) {
                  MDs.push_back(M);
                }
              }
              if (MDs.size()) {
                auto noscope = MDNode::get(dli->getContext(), MDs);
                toreturn->setMetadata(LLVMContext::MD_noalias, noscope);
              }
              toreturn->setDebugLoc(getNewFromOriginal(dli->getDebugLoc()));
              s_idx++;
              return toreturn;
            },
            pidx);

        // TODO adding to cache only legal if no alias of any future writes
        if (permitCache)
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
              toreturn;
        assert(val->getType() == toreturn->getType());
        return toreturn;
      }
      goto endCheck;
    }
    assert(phi->getNumIncomingValues() != 0);

    // If requesting loop bound and are requesting the total size.
    // Rather than generating a new lcssa variable, use the existing loop exact
    // bound var
    BasicBlock *ivctx = scope;
    if (!ivctx)
      ivctx = BuilderM.GetInsertBlock();
    if (newFunc == ivctx->getParent() && !isOriginalBlock(*ivctx)) {
      ivctx = originalForReverseBlock(*ivctx);
    }
    if ((ivctx == phi->getParent() || DT.dominates(phi, ivctx)) &&
        (!isOriginalBlock(*BuilderM.GetInsertBlock()) ||
         DT.dominates(phi, &*BuilderM.GetInsertPoint()))) {
      LoopContext lc;
      bool loopVar = false;
      if (getContext(phi->getParent(), lc) && lc.var == phi) {
        loopVar = true;
      } else {
        Value *V = nullptr;
        bool legal = true;
        for (auto &val : phi->incoming_values()) {
          if (isa<UndefValue>(val))
            continue;
          if (V == nullptr)
            V = val;
          else if (V != val) {
            legal = false;
            break;
          }
        }
        if (legal) {
          if (auto I = dyn_cast_or_null<PHINode>(V)) {
            if (getContext(I->getParent(), lc) && lc.var == I) {
              loopVar = true;
            }
          }
        }
      }
      if (loopVar) {
        if (!lc.dynamic) {
          Value *lim = getOp(lc.trueLimit);
          if (lim) {
            unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
                lim;
            return lim;
          }
        } else if (unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup &&
                   reverseBlocks.size() > 0) {
          // Must be in a reverse pass fashion for a lookup to index bound to be
          // legal
          assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
          LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                            lc.preheader);
          Value *lim = lookupValueFromCache(
              lc.var->getType(),
              /*forwardPass*/ false, BuilderM, lctx,
              getDynamicLoopLimit(LI.getLoopFor(lc.header)),
              /*isi1*/ false, available);
          unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] = lim;
          return lim;
        }
      }
    }

    auto parent = phi->getParent();

    // Don't attempt to unroll a loop induction variable in other
    // circumstances
    auto &LLI = Logic.PPC.FAM.getResult<LoopAnalysis>(*parent->getParent());
    std::set<BasicBlock *> prevIteration;
    if (LLI.isLoopHeader(parent)) {
      if (phi->getNumIncomingValues() != 2) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      auto L = LLI.getLoopFor(parent);
      for (auto PH : predecessors(parent)) {
        if (L->contains(PH))
          prevIteration.insert(PH);
      }
      if (prevIteration.size() && !legalRecompute(phi, available, &BuilderM)) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
    }
    for (auto &val : phi->incoming_values()) {
      if (isPotentialLastLoopValue(val, parent, LLI)) {
        if (unwrapMode == UnwrapMode::LegalFullUnwrap) {
          llvm::errs() << " module: " << *newFunc->getParent() << "\n";
          llvm::errs() << " newFunc: " << *newFunc << "\n";
          llvm::errs() << " parent: " << *parent << "\n";
          llvm::errs() << " val: " << *val << "\n";
        }
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
    }

    if (phi->getNumIncomingValues() == 1) {
      assert(phi->getIncomingValue(0) != phi);
      auto toreturn = getOpUnchecked(phi->getIncomingValue(0));
      if (toreturn == nullptr || toreturn == phi) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      assert(val->getType() == toreturn->getType());
      return toreturn;
    }

    std::set<BasicBlock *> targetToPreds;
    // Map of function edges to list of values possible
    std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
             std::set<BasicBlock *>>
        done;
    {
      std::deque<std::tuple<
          std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
          BasicBlock *>>
          Q; // newblock, target

      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Q.push_back(
            std::make_pair(std::make_pair(phi->getIncomingBlock(i), parent),
                           phi->getIncomingBlock(i)));
        targetToPreds.insert(phi->getIncomingBlock(i));
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

        if (DT.dominates(block, phi->getParent()))
          continue;

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

    std::set<BasicBlock *> blocks;
    for (auto pair : done) {
      const auto &edge = pair.first;
      blocks.insert(edge.first);
    }

    BasicBlock *oldB = BuilderM.GetInsertBlock();
    if (BuilderM.GetInsertPoint() != oldB->end()) {
      assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
      goto endCheck;
    }

    BasicBlock *fwd = oldB;
    bool inReverseBlocks = false;
    if (!isOriginalBlock(*fwd)) {
      auto found = reverseBlockToPrimal.find(oldB);
      if (found == reverseBlockToPrimal.end()) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }
      fwd = found->second;
      inReverseBlocks =
          std::find(reverseBlocks[fwd].begin(), reverseBlocks[fwd].end(),
                    oldB) != reverseBlocks[fwd].end();
    }

    auto eraseBlocks = [&](ArrayRef<BasicBlock *> blocks, BasicBlock *bret) {
      SmallVector<BasicBlock *, 2> revtopo;
      {
        SmallPtrSet<BasicBlock *, 2> seen;
        std::function<void(BasicBlock *)> dfs = [&](BasicBlock *B) {
          if (seen.count(B))
            return;
          seen.insert(B);
          if (B->getTerminator())
            for (auto S : successors(B))
              if (!seen.count(S))
                dfs(S);
          revtopo.push_back(B);
        };
        for (auto B : blocks)
          dfs(B);
        if (!seen.count(bret))
          revtopo.insert(revtopo.begin(), bret);
      }

      SmallVector<Instruction *, 4> toErase;
      for (auto B : revtopo) {
        if (B == bret)
          continue;
        for (auto &I : llvm::reverse(*B)) {
          toErase.push_back(&I);
        }
        unwrap_cache.erase(B);
        lookup_cache.erase(B);
        if (reverseBlocks.size() > 0) {
          auto tfwd = reverseBlockToPrimal[B];
          assert(tfwd);
          auto rfound = reverseBlocks.find(tfwd);
          assert(rfound != reverseBlocks.end());
          auto &tlst = rfound->second;
          auto found = std::find(tlst.begin(), tlst.end(), B);
          if (found != tlst.end())
            tlst.erase(found);
          reverseBlockToPrimal.erase(B);
        }
      }
      for (auto I : toErase) {
        erase(I);
      }
      for (auto B : revtopo)
        B->eraseFromParent();
    };

    if (targetToPreds.size() == 3) {
      for (auto block : blocks) {
        if (!DT.dominates(block, phi->getParent()))
          continue;
        std::set<BasicBlock *> foundtargets;
        std::set<BasicBlock *> uniqueTargets;
        for (BasicBlock *succ : successors(block)) {
          auto edge = std::make_pair(block, succ);
          for (BasicBlock *target : done[edge]) {
            if (foundtargets.find(target) != foundtargets.end()) {
              goto rnextpair;
            }
            foundtargets.insert(target);
            if (done[edge].size() == 1)
              uniqueTargets.insert(target);
          }
        }
        if (foundtargets.size() != 3)
          goto rnextpair;
        if (uniqueTargets.size() != 1)
          goto rnextpair;

        {
          BasicBlock *subblock = nullptr;
          for (auto block2 : blocks) {
            {
              // The second split block must not have a parent with an edge
              // to a block other than to itself, which can reach any of its
              // two targets.
              // TODO verify this
              for (auto P : predecessors(block2)) {
                for (auto S : successors(P)) {
                  if (S == block2)
                    continue;
                  auto edge = std::make_pair(P, S);
                  if (done.find(edge) != done.end()) {
                    for (auto target : done[edge]) {
                      if (foundtargets.find(target) != foundtargets.end() &&
                          uniqueTargets.find(target) == uniqueTargets.end())
                        goto nextblock;
                    }
                  }
                }
              }
              std::set<BasicBlock *> seen2;
              for (BasicBlock *succ : successors(block2)) {
                auto edge = std::make_pair(block2, succ);
                if (done[edge].size() != 1) {
                  // llvm::errs() << " -- failed from noonesize\n";
                  goto nextblock;
                }
                for (BasicBlock *target : done[edge]) {
                  if (seen2.find(target) != seen2.end()) {
                    // llvm::errs() << " -- failed from not uniqueTargets\n";
                    goto nextblock;
                  }
                  seen2.insert(target);
                  if (foundtargets.find(target) == foundtargets.end()) {
                    // llvm::errs() << " -- failed from not unknown target\n";
                    goto nextblock;
                  }
                  if (uniqueTargets.find(target) != uniqueTargets.end()) {
                    // llvm::errs() << " -- failed from not same target\n";
                    goto nextblock;
                  }
                }
              }
              if (seen2.size() != 2) {
                // llvm::errs() << " -- failed from not 2 seen\n";
                goto nextblock;
              }
              subblock = block2;
              break;
            }
          nextblock:;
          }

          if (subblock == nullptr)
            goto rnextpair;

          {
            auto bi1 = cast<BranchInst>(block->getTerminator());

            auto cond1 = getOp(bi1->getCondition());
            if (cond1 == nullptr) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }
            auto bi2 = cast<BranchInst>(subblock->getTerminator());
            auto cond2 = getOp(bi2->getCondition());
            if (cond2 == nullptr) {
              assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
              goto endCheck;
            }

            SmallVector<BasicBlock *, 3> predBlocks = {bi2->getSuccessor(0),
                                                       bi2->getSuccessor(1)};
            for (int i = 0; i < 2; i++) {
              auto edge = std::make_pair(block, bi1->getSuccessor(i));
              if (done[edge].size() == 1) {
                predBlocks.push_back(bi1->getSuccessor(i));
              }
            }

            SmallVector<Value *, 2> vals;

            SmallVector<BasicBlock *, 2> blocks;
            SmallVector<BasicBlock *, 2> endingBlocks;

            BasicBlock *last = oldB;

            BasicBlock *bret = BasicBlock::Create(
                val->getContext(), oldB->getName() + "_phimerge", newFunc);

            for (size_t i = 0; i < predBlocks.size(); i++) {
              BasicBlock *valparent = (i < 2) ? subblock : block;
              assert(done.find(std::make_pair(valparent, predBlocks[i])) !=
                     done.end());
              assert(done[std::make_pair(valparent, predBlocks[i])].size() ==
                     1);
              blocks.push_back(BasicBlock::Create(
                  val->getContext(), oldB->getName() + "_phirc", newFunc));
              blocks[i]->moveAfter(last);
              last = blocks[i];
              if (inReverseBlocks)
                reverseBlocks[fwd].push_back(blocks[i]);
              reverseBlockToPrimal[blocks[i]] = fwd;
              IRBuilder<> B(blocks[i]);

              for (auto pair : unwrap_cache[oldB])
                unwrap_cache[blocks[i]].insert(pair);
              for (auto pair : lookup_cache[oldB])
                lookup_cache[blocks[i]].insert(pair);
              auto PB = *done[std::make_pair(valparent, predBlocks[i])].begin();

              if (auto inst = dyn_cast<Instruction>(
                      phi->getIncomingValueForBlock(PB))) {
                // Recompute the phi computation with the conditional if:
                // 1) the instruction may read from memory AND does not
                //    dominate the current insertion point (thereby
                //    potentially making such recomputation without the
                //    condition illegal)
                // 2) the value is a call or load and option is set to not
                //    speculatively recompute values within a phi
                //            OR
                // 3) the value comes from a previous iteration.
                BasicBlock *nextScope = PB;
                // if (inst->getParent() == nextScope) nextScope =
                // phi->getParent();
                if (prevIteration.count(PB)) {
                  assert(0 && "tri block prev iteration unhandled");
                } else if (!DT.dominates(inst->getParent(), phi->getParent()) ||
                           (!EnzymeSpeculatePHIs &&
                            (isa<CallInst>(inst) || isa<LoadInst>(inst))))
                  vals.push_back(getOpFull(B, inst, nextScope));
                else
                  vals.push_back(getOpFull(BuilderM, inst, nextScope));
              } else
                vals.push_back(
                    getOpFull(BuilderM, phi->getIncomingValueForBlock(PB), PB));

              if (!vals[i]) {
                eraseBlocks(blocks, bret);
                assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
                goto endCheck;
              }
              assert(val->getType() == vals[i]->getType());
              B.CreateBr(bret);
              endingBlocks.push_back(B.GetInsertBlock());
            }

            bret->moveAfter(last);

            BasicBlock *bsplit = BasicBlock::Create(
                val->getContext(), oldB->getName() + "_phisplt", newFunc);
            bsplit->moveAfter(oldB);
            if (inReverseBlocks)
              reverseBlocks[fwd].push_back(bsplit);
            reverseBlockToPrimal[bsplit] = fwd;
            BuilderM.CreateCondBr(
                cond1,
                (done[std::make_pair(block, bi1->getSuccessor(0))].size() == 1)
                    ? blocks[2]
                    : bsplit,
                (done[std::make_pair(block, bi1->getSuccessor(1))].size() == 1)
                    ? blocks[2]
                    : bsplit);

            BuilderM.SetInsertPoint(bsplit);
            BuilderM.CreateCondBr(cond2, blocks[0], blocks[1]);

            BuilderM.SetInsertPoint(bret);
            if (inReverseBlocks)
              reverseBlocks[fwd].push_back(bret);
            reverseBlockToPrimal[bret] = fwd;
            auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
            for (size_t i = 0; i < vals.size(); i++)
              toret->addIncoming(vals[i], endingBlocks[i]);
            assert(val->getType() == toret->getType());
            if (permitCache) {
              unwrap_cache[bret][idx.first][idx.second] = toret;
            }
            unwrappedLoads[toret] = val;
            for (auto pair : unwrap_cache[oldB])
              unwrap_cache[bret].insert(pair);
            for (auto pair : lookup_cache[oldB])
              lookup_cache[bret].insert(pair);
            return toret;
          }
        }
      rnextpair:;
      }
    }

    Instruction *equivalentTerminator = nullptr;

    if (prevIteration.size() == 1) {
      if (phi->getNumIncomingValues() == 2) {

        ValueToValueMapTy prevAvailable;
        for (const auto &pair : available)
          prevAvailable.insert(pair);
        LoopContext ctx;
        getContext(parent, ctx);
        Value *prevIdx;
        if (prevAvailable.count(ctx.var))
          prevIdx = prevAvailable[ctx.var];
        else {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
            // If we are using the phi in the reverse pass of a block inside the
            // loop itself the previous index variable (aka the previous inc) is
            // equivalent to the current load of antivaralloc
            if (LI.getLoopFor(ctx.header)->contains(fwd)) {
              prevIdx =
                  BuilderM.CreateLoad(ctx.var->getType(), ctx.antivaralloc);
            } else {
              // However, if we are using the phi of the reverse pass of a block
              // outside the loop we must be in the reverse pass of a block
              // after the loop. In which case, the previous index variable (aka
              // previous inc) is the total loop iteration count-1, aka the
              // trueLimit.
              Value *lim = nullptr;
              if (ctx.dynamic) {
                // Must be in a reverse pass fashion for a lookup to index bound
                // to be legal
                assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
                LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                  ctx.preheader);
                lim = lookupValueFromCache(
                    ctx.var->getType(),
                    /*forwardPass*/ false, BuilderM, lctx,
                    getDynamicLoopLimit(LI.getLoopFor(ctx.header)),
                    /*isi1*/ false, /*available*/ prevAvailable);
              } else {
                lim = lookupM(ctx.trueLimit, BuilderM, prevAvailable);
              }
              prevIdx = lim;
            }
          } else {
            prevIdx = ctx.var;
          }
        }
        // Prevent recursive unroll.
        prevAvailable[phi] = nullptr;
        SmallVector<Value *, 2> vals;

        SmallVector<BasicBlock *, 2> blocks;
        SmallVector<BasicBlock *, 2> endingBlocks;
        BasicBlock *last = oldB;

        BasicBlock *bret = BasicBlock::Create(
            val->getContext(), oldB->getName() + "_phimerge", newFunc);

        SmallVector<BasicBlock *, 2> preds(predecessors(phi->getParent()));

        for (auto tup : llvm::enumerate(preds)) {
          auto i = tup.index();
          BasicBlock *PB = tup.value();
          blocks.push_back(BasicBlock::Create(
              val->getContext(), oldB->getName() + "_phirc", newFunc));
          blocks[i]->moveAfter(last);
          last = blocks[i];
          if (reverseBlocks.size() > 0) {
            if (inReverseBlocks)
              reverseBlocks[fwd].push_back(blocks[i]);
            reverseBlockToPrimal[blocks[i]] = fwd;
          }
          IRBuilder<> B(blocks[i]);

          if (!prevIteration.count(PB)) {
            for (auto pair : unwrap_cache[oldB])
              unwrap_cache[blocks[i]].insert(pair);
            for (auto pair : lookup_cache[oldB])
              lookup_cache[blocks[i]].insert(pair);
          }

          if (auto inst =
                  dyn_cast<Instruction>(phi->getIncomingValueForBlock(PB))) {
            // Recompute the phi computation with the conditional if:
            // 1) the instruction may read from memory AND does not dominate
            //    the current insertion point (thereby potentially making such
            //    recomputation without the condition illegal)
            // 2) the value is a call or load and option is set to not
            //    speculatively recompute values within a phi
            //                OR
            // 3) the value comes from a previous iteration.
            BasicBlock *nextScope = PB;
            // if (inst->getParent() == nextScope) nextScope = phi->getParent();
            if (prevIteration.count(PB)) {
              prevAvailable[ctx.incvar] = prevIdx;
              prevAvailable[ctx.var] =
                  B.CreateSub(prevIdx, ConstantInt::get(prevIdx->getType(), 1),
                              "", /*NUW*/ true, /*NSW*/ false);
              Value *___res;
              if (unwrapMode == UnwrapMode::LegalFullUnwrap ||
                  unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||
                  unwrapMode == UnwrapMode::AttemptFullUnwrap ||
                  unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
                ___res = unwrapM(inst, B, prevAvailable, unwrapMode, nextScope,
                                 /*permitCache*/ false);
                if (!___res &&
                    unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
                  bool noLookup = false;
                  if (isOriginalBlock(*B.GetInsertBlock())) {
                    if (!DT.dominates(inst, &*B.GetInsertPoint()))
                      noLookup = true;
                  }
                  if (!noLookup) {
                    BasicBlock *nS2 = nextScope;
                    Value *v = inst;
                    ___res = lookupM(v, B, prevAvailable, v != val, nS2);
                  }
                }
                if (___res)
                  assert(___res->getType() == inst->getType() && "uw");
              } else {
                BasicBlock *nS2 = nextScope;
                Value *v = inst;
                ___res = lookupM(v, B, prevAvailable, v != val, nS2);
                if (___res && ___res->getType() != v->getType()) {
                  llvm::errs() << *newFunc << "\n";
                  llvm::errs() << " v = " << *v << " res = " << *___res << "\n";
                }
                if (___res)
                  assert(___res->getType() == inst->getType() && "lu");
              }
              vals.push_back(___res);
            } else if (!DT.dominates(inst->getParent(), phi->getParent()) ||
                       (!EnzymeSpeculatePHIs &&
                        (isa<CallInst>(inst) || isa<LoadInst>(inst))))
              vals.push_back(getOpFull(B, inst, nextScope));
            else
              vals.push_back(getOpFull(BuilderM, inst, nextScope));
          } else
            vals.push_back(phi->getIncomingValueForBlock(PB));

          if (!vals[i]) {
            eraseBlocks(blocks, bret);
            assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
            goto endCheck;
          }
          assert(val->getType() == vals[i]->getType());
          B.CreateBr(bret);
          endingBlocks.push_back(B.GetInsertBlock());
        }

        // Coming from a previous iteration is equivalent to the current
        // iteration at zero.
        Value *cond;
        if (prevIteration.count(preds[0]))
          cond = BuilderM.CreateICmpNE(prevIdx,
                                       ConstantInt::get(prevIdx->getType(), 0));
        else
          cond = BuilderM.CreateICmpEQ(prevIdx,
                                       ConstantInt::get(prevIdx->getType(), 0));

        if (blocks[0]->size() == 1 && blocks[1]->size() == 1) {
          if (auto B1 = dyn_cast<BranchInst>(blocks[0]->getTerminator()))
            if (auto B2 = dyn_cast<BranchInst>(blocks[1]->getTerminator()))
              if (B1->isUnconditional() && B2->isUnconditional() &&
                  B1->getSuccessor(0) == bret && B2->getSuccessor(0) == bret) {
                eraseBlocks(blocks, bret);
                Value *toret = BuilderM.CreateSelect(
                    cond, vals[0], vals[1], phi->getName() + "_unwrap");
                if (permitCache) {
                  unwrap_cache[BuilderM.GetInsertBlock()][idx.first]
                              [idx.second] = toret;
                }
                if (auto instRet = dyn_cast<Instruction>(toret)) {
                  unwrappedLoads[instRet] = val;
                }
                return toret;
              }
        }

        bret->moveAfter(last);
        BuilderM.CreateCondBr(cond, blocks[0], blocks[1]);

        BuilderM.SetInsertPoint(bret);
        if (inReverseBlocks)
          reverseBlocks[fwd].push_back(bret);
        reverseBlockToPrimal[bret] = fwd;
        auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
        for (size_t i = 0; i < vals.size(); i++)
          toret->addIncoming(vals[i], endingBlocks[i]);
        assert(val->getType() == toret->getType());
        if (permitCache) {
          unwrap_cache[bret][idx.first][idx.second] = toret;
        }
        for (auto pair : unwrap_cache[oldB])
          unwrap_cache[bret].insert(pair);
        for (auto pair : lookup_cache[oldB])
          lookup_cache[bret].insert(pair);
        unwrappedLoads[toret] = val;
        return toret;
      }
    }
    if (prevIteration.size() != 0) {
      llvm::errs() << "prev iteration: " << *phi << "\n";
      assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
      goto endCheck;
    }

    for (auto block : blocks) {
      if (!DT.dominates(block, phi->getParent()))
        continue;
      std::set<BasicBlock *> foundtargets;
      for (BasicBlock *succ : successors(block)) {
        auto edge = std::make_pair(block, succ);
        if (done[edge].size() != 1) {
          goto nextpair;
        }
        BasicBlock *target = *done[edge].begin();
        if (foundtargets.find(target) != foundtargets.end()) {
          goto nextpair;
        }
        foundtargets.insert(target);
      }
      if (foundtargets.size() != targetToPreds.size()) {
        goto nextpair;
      }

      if (DT.dominates(block, parent)) {
        equivalentTerminator = block->getTerminator();
        goto fast;
      }
    nextpair:;
    }
    goto endCheck;

  fast:;
    assert(equivalentTerminator);

    if (isa<BranchInst>(equivalentTerminator) ||
        isa<SwitchInst>(equivalentTerminator)) {
      BasicBlock *oldB = BuilderM.GetInsertBlock();

      SmallVector<BasicBlock *, 2> predBlocks;
      Value *cond = nullptr;
      if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
        cond = branch->getCondition();
        predBlocks.push_back(branch->getSuccessor(0));
        predBlocks.push_back(branch->getSuccessor(1));
      } else {
        auto SI = cast<SwitchInst>(equivalentTerminator);
        cond = SI->getCondition();
        predBlocks.push_back(SI->getDefaultDest());
        for (auto scase : SI->cases()) {
          predBlocks.push_back(scase.getCaseSuccessor());
        }
      }
      cond = getOp(cond);
      if (!cond) {
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }

      SmallVector<Value *, 2> vals;

      SmallVector<BasicBlock *, 2> blocks;
      SmallVector<BasicBlock *, 2> endingBlocks;

      BasicBlock *last = oldB;

      assert(prevIteration.size() == 0);

      BasicBlock *bret = BasicBlock::Create(
          val->getContext(), oldB->getName() + "_phimerge", newFunc);

      for (size_t i = 0; i < predBlocks.size(); i++) {
        assert(done.find(std::make_pair(equivalentTerminator->getParent(),
                                        predBlocks[i])) != done.end());
        assert(done[std::make_pair(equivalentTerminator->getParent(),
                                   predBlocks[i])]
                   .size() == 1);
        BasicBlock *PB = *done[std::make_pair(equivalentTerminator->getParent(),
                                              predBlocks[i])]
                              .begin();
        blocks.push_back(BasicBlock::Create(
            val->getContext(), oldB->getName() + "_phirc", newFunc));
        blocks[i]->moveAfter(last);
        last = blocks[i];
        if (reverseBlocks.size() > 0) {
          if (inReverseBlocks)
            reverseBlocks[fwd].push_back(blocks[i]);
          reverseBlockToPrimal[blocks[i]] = fwd;
        }
        IRBuilder<> B(blocks[i]);

        for (auto pair : unwrap_cache[oldB])
          unwrap_cache[blocks[i]].insert(pair);
        for (auto pair : lookup_cache[oldB])
          lookup_cache[blocks[i]].insert(pair);

        if (auto inst =
                dyn_cast<Instruction>(phi->getIncomingValueForBlock(PB))) {
          // Recompute the phi computation with the conditional if:
          // 1) the instruction may reat from memory AND does not dominate
          //    the current insertion point (thereby potentially making such
          //    recomputation without the condition illegal)
          // 2) the value is a call or load and option is set to not
          //    speculatively recompute values within a phi
          //                OR
          // 3) the value comes from a previous iteration.
          BasicBlock *nextScope = PB;
          // if (inst->getParent() == nextScope) nextScope = phi->getParent();
          if (!DT.dominates(inst->getParent(), phi->getParent()) ||
              (!EnzymeSpeculatePHIs &&
               (isa<CallInst>(inst) || isa<LoadInst>(inst))))
            vals.push_back(getOpFull(B, inst, nextScope));
          else
            vals.push_back(getOpFull(BuilderM, inst, nextScope));
        } else
          vals.push_back(phi->getIncomingValueForBlock(PB));

        if (!vals[i]) {
          eraseBlocks(blocks, bret);
          assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
          goto endCheck;
        }
        assert(val->getType() == vals[i]->getType());
        B.CreateBr(bret);
        endingBlocks.push_back(B.GetInsertBlock());
      }

      // Fast path to not make a split block if no additional instructions
      // were made in the two blocks
      if (isa<BranchInst>(equivalentTerminator) && blocks[0]->size() == 1 &&
          blocks[1]->size() == 1) {
        if (auto B1 = dyn_cast<BranchInst>(blocks[0]->getTerminator()))
          if (auto B2 = dyn_cast<BranchInst>(blocks[1]->getTerminator()))
            if (B1->isUnconditional() && B2->isUnconditional() &&
                B1->getSuccessor(0) == bret && B2->getSuccessor(0) == bret) {
              eraseBlocks(blocks, bret);
              Value *toret = BuilderM.CreateSelect(cond, vals[0], vals[1],
                                                   phi->getName() + "_unwrap");
              if (permitCache) {
                unwrap_cache[BuilderM.GetInsertBlock()][idx.first][idx.second] =
                    toret;
              }
              if (auto instRet = dyn_cast<Instruction>(toret)) {
                unwrappedLoads[instRet] = val;
              }
              return toret;
            }
      }

      if (BuilderM.GetInsertPoint() != oldB->end()) {
        eraseBlocks(blocks, bret);
        assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
        goto endCheck;
      }

      bret->moveAfter(last);
      if (isa<BranchInst>(equivalentTerminator)) {
        BuilderM.CreateCondBr(cond, blocks[0], blocks[1]);
      } else {
        auto SI = cast<SwitchInst>(equivalentTerminator);
        auto NSI = BuilderM.CreateSwitch(cond, blocks[0], SI->getNumCases());
        size_t idx = 1;
        for (auto scase : SI->cases()) {
          NSI->addCase(scase.getCaseValue(), blocks[idx]);
          idx++;
        }
      }
      BuilderM.SetInsertPoint(bret);
      if (inReverseBlocks)
        reverseBlocks[fwd].push_back(bret);
      reverseBlockToPrimal[bret] = fwd;
      auto toret = BuilderM.CreatePHI(val->getType(), vals.size());
      for (size_t i = 0; i < vals.size(); i++)
        toret->addIncoming(vals[i], endingBlocks[i]);
      assert(val->getType() == toret->getType());
      if (permitCache) {
        unwrap_cache[bret][idx.first][idx.second] = toret;
      }
      for (auto pair : unwrap_cache[oldB])
        unwrap_cache[bret].insert(pair);
      for (auto pair : lookup_cache[oldB])
        lookup_cache[bret].insert(pair);
      unwrappedLoads[toret] = val;
      return toret;
    }
    assert(unwrapMode != UnwrapMode::LegalFullUnwrap);
    goto endCheck;
  }

endCheck:
  assert(val);
  if (unwrapMode == UnwrapMode::LegalFullUnwrap ||
      unwrapMode == UnwrapMode::LegalFullUnwrapNoTapeReplace ||
      unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup) {
    assert(val->getName() != "<badref>");
    Value *nval = val;
    if (auto opinst = dyn_cast<Instruction>(nval))
      if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
        if (!DT.dominates(opinst, &*BuilderM.GetInsertPoint())) {
          if (unwrapMode != UnwrapMode::AttemptFullUnwrapWithLookup) {
            llvm::errs() << " oldF: " << *oldFunc << "\n";
            llvm::errs() << " opParen: " << *opinst->getParent()->getParent()
                         << "\n";
            llvm::errs() << " newF: " << *newFunc << "\n";
            llvm::errs() << " - blk: " << *BuilderM.GetInsertBlock();
            llvm::errs() << " opInst: " << *opinst << " mode=" << unwrapMode
                         << "\n";
          }
          assert(unwrapMode == UnwrapMode::AttemptFullUnwrapWithLookup);
          return nullptr;
        }
      }
    auto toreturn = lookupM(nval, BuilderM, available,
                            /*tryLegalRecomputeCheck*/ false, scope);
    assert(val->getType() == toreturn->getType());
    return toreturn;
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
    assert(val->getName() != "<badref>");
    auto &warnMap = UnwrappedWarnings[inst];
    if (!warnMap.count(BuilderM.GetInsertBlock())) {
      EmitWarning("NoUnwrap", *inst, "Cannot unwrap ", *val, " in ",
                  BuilderM.GetInsertBlock()->getName());
      warnMap.insert(BuilderM.GetInsertBlock());
    }
  }
  return nullptr;
}

void GradientUtils::ensureLookupCached(Instruction *inst, bool shouldFree,
                                       BasicBlock *scope, MDNode *TBAA) {
  assert(inst);
  if (scopeMap.find(inst) != scopeMap.end())
    return;
  if (shouldFree)
    assert(reverseBlocks.size());

  if (scope == nullptr)
    scope = inst->getParent();

  LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0, scope);

  AllocaInst *cache =
      createCacheForScope(lctx, inst->getType(), inst->getName(), shouldFree);
  assert(cache);
  Value *Val = inst;
  insert_or_assign(
      scopeMap, Val,
      std::pair<AssertingVH<AllocaInst>, LimitContext>(cache, lctx));
  storeInstructionInCache(lctx, inst, cache, TBAA);
}

Value *GradientUtils::fixLCSSA(Instruction *inst, BasicBlock *forwardBlock,
                               bool legalInBlock) {
  assert(inst->getName() != "<badref>");

  if (auto lcssaPHI = dyn_cast<PHINode>(inst)) {
    auto found = lcssaPHIToOrig.find(lcssaPHI);
    if (found != lcssaPHIToOrig.end())
      inst = cast<Instruction>(found->second);
  }

  if (inst->getParent() == inversionAllocs)
    return inst;

  if (!isOriginalBlock(*forwardBlock)) {
    forwardBlock = originalForReverseBlock(*forwardBlock);
  }

  bool containsLastLoopValue = isPotentialLastLoopValue(inst, forwardBlock, LI);

  // If the instruction cannot represent a loop value, return the original
  // instruction if it either is guaranteed to be available within the block,
  // or it is not needed to guaranteed availability.
  if (!containsLastLoopValue) {
    if (!legalInBlock)
      return inst;
    if (forwardBlock == inst->getParent() || DT.dominates(inst, forwardBlock))
      return inst;
  }

  // llvm::errs() << " inst: " << *inst << "\n";
  // llvm::errs() << " seen: " << *inst->getParent() << "\n";
  assert(inst->getParent() != inversionAllocs);
  assert(isOriginalBlock(*inst->getParent()));

  if (lcssaFixes.find(inst) == lcssaFixes.end()) {
    lcssaFixes[inst][inst->getParent()] = inst;
    SmallPtrSet<BasicBlock *, 4> seen;
    std::deque<BasicBlock *> todo = {inst->getParent()};
    while (todo.size()) {
      BasicBlock *cur = todo.front();
      todo.pop_front();
      if (seen.count(cur))
        continue;
      seen.insert(cur);
      for (auto Succ : successors(cur)) {
        todo.push_back(Succ);
      }
    }
    for (auto &BB : *inst->getParent()->getParent()) {
      if (!seen.count(&BB) ||
          (inst->getParent() != &BB && DT.dominates(&BB, inst->getParent()))) {
        // OrigPDT.dominates(isOriginal(inst->getParent()),
        //                  isOriginal(&BB)))) {
        lcssaFixes[inst][&BB] = UndefValue::get(inst->getType());
      }
    }
  }

  if (lcssaFixes[inst].find(forwardBlock) != lcssaFixes[inst].end()) {
    return lcssaFixes[inst][forwardBlock];
  }

  // TODO replace forwardBlock with the first block dominated by inst,
  // that dominates (or is) forwardBlock to ensuring maximum reuse
  IRBuilder<> lcssa(&forwardBlock->front());
  auto lcssaPHI =
      lcssa.CreatePHI(inst->getType(), 1, inst->getName() + "!manual_lcssa");
  lcssaFixes[inst][forwardBlock] = lcssaPHI;
  lcssaPHIToOrig[lcssaPHI] = inst;
  for (auto pred : predecessors(forwardBlock)) {
    Value *val = nullptr;
    if (inst->getParent() == pred || DT.dominates(inst, pred)) {
      val = inst;
    }
    if (val == nullptr) {
      val = fixLCSSA(inst, pred, /*legalInBlock*/ true);
      assert(val->getType() == inst->getType());
    }
    assert(val->getType() == inst->getType());
    lcssaPHI->addIncoming(val, pred);
  }

  SmallPtrSet<Value *, 2> vals;
  SmallVector<Value *, 2> todo(lcssaPHI->incoming_values().begin(),
                               lcssaPHI->incoming_values().end());
  while (todo.size()) {
    Value *v = todo.back();
    todo.pop_back();
    if (v == lcssaPHI)
      continue;
    vals.insert(v);
  }
  assert(vals.size() > 0);

  if (vals.size() > 1) {
    todo.append(vals.begin(), vals.end());
    vals.clear();
    while (todo.size()) {
      Value *v = todo.back();
      todo.pop_back();

      if (auto PN = dyn_cast<PHINode>(v))
        if (lcssaPHIToOrig.find(PN) != lcssaPHIToOrig.end()) {
          v = lcssaPHIToOrig[PN];
        }
      vals.insert(v);
    }
  }
  assert(vals.size() > 0);
  Value *val = nullptr;
  if (vals.size() == 1)
    val = *vals.begin();

  if (val && (!legalInBlock || !isa<Instruction>(val) ||
              DT.dominates(cast<Instruction>(val), lcssaPHI))) {

    if (!isPotentialLastLoopValue(val, forwardBlock, LI)) {
      bool nonSelfUse = false;
      for (auto u : lcssaPHI->users()) {
        if (u != lcssaPHI) {
          nonSelfUse = true;
          break;
        }
      }
      if (!nonSelfUse) {
        lcssaFixes[inst].erase(forwardBlock);
        while (lcssaPHI->getNumOperands())
          lcssaPHI->removeIncomingValue(lcssaPHI->getNumOperands() - 1, false);
        lcssaPHIToOrig.erase(lcssaPHI);
        lcssaPHI->eraseFromParent();
      }
      return val;
    }
  }
  return lcssaPHI;
}

Value *GradientUtils::cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc,
                                      int idx, bool replace) {
  assert(malloc);
  assert(BuilderQ.GetInsertBlock()->getParent() == newFunc);
  assert(isOriginalBlock(*BuilderQ.GetInsertBlock()));
  if (mode == DerivativeMode::ReverseModeCombined) {
    assert(!tape);
    return malloc;
  }

  if (auto CI = dyn_cast<CallInst>(malloc)) {
    if (auto F = CI->getCalledFunction()) {
      assert(F->getName() != "omp_get_thread_num");
    }
  }

  if (malloc->getType()->isTokenTy()) {
    llvm::errs() << " oldFunc: " << *oldFunc << "\n";
    llvm::errs() << " newFunc: " << *newFunc << "\n";
    llvm::errs() << " malloc: " << *malloc << "\n";
  }
  assert(!malloc->getType()->isTokenTy());
  {
    CountTrackedPointers T(malloc->getType());
    if (T.derived) {
      llvm::errs() << " oldFunc: " << *oldFunc << "\n";
      llvm::errs() << " newFunc: " << *newFunc << "\n";
      llvm::errs() << " malloc: " << *malloc << "\n";
    }
    assert(!T.derived);
  }

  if (tape) {
    if (idx >= 0 && !tape->getType()->isStructTy()) {
      llvm::errs() << "cacheForReverse incorrect tape type: " << *tape
                   << " idx: " << idx << "\n";
    }
    assert(idx < 0 || tape->getType()->isStructTy());
    if (idx >= 0 &&
        (unsigned)idx >= cast<StructType>(tape->getType())->getNumElements()) {
      llvm::errs() << "oldFunc: " << *oldFunc << "\n";
      llvm::errs() << "newFunc: " << *newFunc << "\n";
      if (malloc)
        llvm::errs() << "malloc: " << *malloc << "\n";
      llvm::errs() << "tape: " << *tape << "\n";
      llvm::errs() << "idx: " << idx << "\n";
    }
    assert(idx < 0 ||
           (unsigned)idx < cast<StructType>(tape->getType())->getNumElements());
    Value *ret =
        (idx < 0) ? tape : BuilderQ.CreateExtractValue(tape, {(unsigned)idx});

    if (ret->getType()->isEmptyTy()) {
      if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
        if (inst->getType() != ret->getType()) {
          llvm::errs() << "oldFunc: " << *oldFunc << "\n";
          llvm::errs() << "newFunc: " << *newFunc << "\n";
          llvm::errs() << "inst==malloc: " << *inst << "\n";
          llvm::errs() << "ret: " << *ret << "\n";
        }
        assert(inst->getType() == ret->getType());
        if (replace) {
          inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
          erase(inst);
        }
      }
      Type *retType = ret->getType();
      if (replace)
        if (auto ri = dyn_cast<Instruction>(ret))
          erase(ri);
      return UndefValue::get(retType);
    }

    LimitContext ctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                     BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(/*ReverseLimit*/ reverseBlocks.size() > 0,
                         inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }
    assert(isOriginalBlock(*ctx.Block));

    bool inLoop;
    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      ret->setName(malloc->getName() + "_fromtape");
      if (omp) {
        Value *tid = ompThreadId();
        Value *tPtr = BuilderQ.CreateInBoundsGEP(malloc->getType(), ret,
                                                 ArrayRef<Value *>(tid));
        ret = BuilderQ.CreateLoad(malloc->getType(), tPtr);
      }
    } else {
      if (idx >= 0)
        erase(cast<Instruction>(ret));
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      ret = (idx < 0) ? tape
                      : entryBuilder.CreateExtractValue(tape, {(unsigned)idx});

      assert(malloc);

      Type *innerType = nullptr;

#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
      if (ret->getContext().supportsTypedPointers()) {
#endif
        innerType = ret->getType();
        for (size_t i = 0,
                    limit = getSubLimits(
                                /*inForwardPass*/ true, nullptr,
                                LimitContext(
                                    /*ReverseLimit*/ reverseBlocks.size() > 0,
                                    BuilderQ.GetInsertBlock()))
                                .size();
             i < limit; ++i) {
          if (!isa<PointerType>(innerType)) {
            llvm::errs() << "mod: "
                         << *BuilderQ.GetInsertBlock()->getParent()->getParent()
                         << "\n";
            llvm::errs() << "fn: " << *BuilderQ.GetInsertBlock()->getParent()
                         << "\n";
            llvm::errs() << "bq insertblock: " << *BuilderQ.GetInsertBlock()
                         << "\n";
            llvm::errs() << "ret: " << *ret << " type: " << *ret->getType()
                         << "\n";
            llvm::errs() << "innerType: " << *innerType << "\n";
            if (malloc)
              llvm::errs() << " malloc: " << *malloc << " i=" << i
                           << " / lim = " << limit << "\n";
          }
          assert(isa<PointerType>(innerType));
          innerType = innerType->getPointerElementType();
        }
#if LLVM_VERSION_MAJOR >= 15
      } else {
        if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
            cast<IntegerType>(malloc->getType())->getBitWidth() == 1)
          innerType = Type::getInt8Ty(malloc->getContext());
        else
          innerType = malloc->getType();
      }
#endif
#else
      if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
          cast<IntegerType>(malloc->getType())->getBitWidth() == 1)
        innerType = Type::getInt8Ty(malloc->getContext());
      else
        innerType = malloc->getType();
#endif

      if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
          cast<IntegerType>(malloc->getType())->getBitWidth() == 1 &&
          innerType != ret->getType()) {
        assert(innerType == Type::getInt8Ty(malloc->getContext()));
      } else {
        if (innerType != malloc->getType()) {
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          llvm::errs() << "malloc->getType(): " << *malloc->getType() << "\n";
          llvm::errs() << "ret: " << *ret << " - " << *ret->getType() << "\n";
          llvm::errs() << "malloc: " << *malloc << "\n";
          assert(0 && "illegal loop cache type");
          llvm_unreachable("illegal loop cache type");
        }
      }

      LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                        BuilderQ.GetInsertBlock());
      AllocaInst *cache =
          createCacheForScope(lctx, innerType, "mdyncache_fromtape",
                              ((DiffeGradientUtils *)this)->FreeMemory, false);
      assert(malloc);
      bool isi1 = malloc->getType()->isIntegerTy() &&
                  cast<IntegerType>(malloc->getType())->getBitWidth() == 1;
      assert(isa<PointerType>(cache->getType()));
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
      if (cache->getContext().supportsTypedPointers()) {
#endif
        assert(cache->getType()->getPointerElementType() == ret->getType());
#if LLVM_VERSION_MAJOR >= 15
      }
#endif
#endif
      entryBuilder.CreateStore(ret, cache);

      auto v =
          lookupValueFromCache(innerType, /*forwardPass*/ true, BuilderQ, lctx,
                               cache, isi1, /*available*/ ValueToValueMapTy());
      if (malloc) {
        assert(v->getType() == malloc->getType());
      }
      insert_or_assign(scopeMap, v,
                       std::make_pair(AssertingVH<AllocaInst>(cache), ctx));
      ret = cast<Instruction>(v);
    }

    if (malloc && !isa<UndefValue>(malloc)) {
      if (malloc->getType() != ret->getType()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *malloc << "\n";
        llvm::errs() << *ret << "\n";
      }
      assert(malloc->getType() == ret->getType());

      if (replace) {
        auto found = newToOriginalFn.find(malloc);
        if (found != newToOriginalFn.end()) {
          Value *orig = found->second;
          originalToNewFn[orig] = ret;
          newToOriginalFn.erase(malloc);
          newToOriginalFn[ret] = orig;
        }
      }

      if (auto found = findInMap(scopeMap, malloc)) {
        // There already exists an alloaction for this, we should fully remove
        // it
        if (!inLoop) {

          // Remove stores into
          SmallVector<Instruction *, 3> stores(
              scopeInstructions[found->first].begin(),
              scopeInstructions[found->first].end());
          scopeInstructions.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          SmallVector<User *, 4> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              IRBuilder<> lb(li);
              if (replace) {

                Value *replacewith =
                    (idx < 0) ? tape
                              : lb.CreateExtractValue(tape, {(unsigned)idx});
                if (!inLoop && omp) {
                  Value *tid = ompThreadId();
                  Value *tPtr = lb.CreateInBoundsGEP(li->getType(), replacewith,
                                                     ArrayRef<Value *>(tid));
                  replacewith = lb.CreateLoad(li->getType(), tPtr);
                }
                if (li->getType() != replacewith->getType()) {
                  llvm::errs() << " oldFunc: " << *oldFunc << "\n";
                  llvm::errs() << " newFunc: " << *newFunc << "\n";
                  llvm::errs() << " malloc: " << *malloc << "\n";
                  llvm::errs() << " li: " << *li << "\n";
                  llvm::errs() << " u: " << *u << "\n";
                  llvm::errs() << " replacewith: " << *replacewith
                               << " idx=" << idx << " - tape=" << *tape << "\n";
                }
                assert(li->getType() == replacewith->getType());
                li->replaceAllUsesWith(replacewith);
              } else {
                auto phi =
                    lb.CreatePHI(li->getType(), 0, li->getName() + "_cfrphi");
                unwrappedLoads[phi] = malloc;
                li->replaceAllUsesWith(phi);
              }
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap1");
            }
          }

          {
            AllocaInst *preerase = found->first;
            scopeMap.erase(malloc);
            erase(preerase);
          }
        } else {
          // Remove allocations for scopealloc since it is already allocated
          // by the augmented forward pass
          // Remove stores into
          SmallVector<Instruction *, 3> stores(
              scopeInstructions[found->first].begin(),
              scopeInstructions[found->first].end());
          scopeInstructions.erase(found->first);
          scopeAllocs.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          // Remove frees
          SmallVector<CallInst *, 3> tofree(scopeFrees[found->first].begin(),
                                            scopeFrees[found->first].end());
          scopeFrees.erase(found->first);
          for (auto freeinst : tofree) {
            // This deque contains a list of operations
            // we can erasing upon erasing the free (and so on).
            // Since multiple operations can have the same operand,
            // this deque can contain the same value multiple times.
            // To remedy this we use a tracking value handle which will
            // be set to null when erased.
            std::deque<WeakTrackingVH> ops = {freeinst->getArgOperand(0)};
            erase(freeinst);

            while (ops.size()) {
              auto z = dyn_cast_or_null<Instruction>(ops[0]);
              ops.pop_front();
              if (z && z->getNumUses() == 0 && !z->isUsedByMetadata()) {
                for (unsigned i = 0; i < z->getNumOperands(); ++i) {
                  ops.push_back(z->getOperand(i));
                }
                erase(z);
              }
            }
          }

          // uses of the alloc
          SmallVector<User *, 4> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              // even with replace off, this can be replaced
              // as since we're in a loop this load is a load of cache
              // not of the final value (thereby overwriting the new
              // inst
              IRBuilder<> lb(li);
              auto replacewith =
                  (idx < 0) ? tape
                            : lb.CreateExtractValue(tape, {(unsigned)idx});
              li->replaceAllUsesWith(replacewith);
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap2");
            }
          }

          AllocaInst *preerase = found->first;
          scopeMap.erase(malloc);
          if (replace)
            erase(preerase);
        }
      }
      if (replace)
        cast<Instruction>(malloc)->replaceAllUsesWith(ret);
      ret->takeName(malloc);
      if (replace) {
        auto malloci = cast<Instruction>(malloc);
        if (malloci == &*BuilderQ.GetInsertPoint()) {
          BuilderQ.SetInsertPoint(malloci->getNextNode());
        }
        erase(malloci);
      }
    }
    return ret;
  } else {
    assert(malloc);

    assert(idx >= 0 && (unsigned)idx == addedTapeVals.size());

    if (isa<UndefValue>(malloc)) {
      addedTapeVals.push_back(malloc);
      return malloc;
    }

    LimitContext ctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                     BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(/*ReverseLimit*/ reverseBlocks.size() > 0,
                         inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }

    bool inLoop;

    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      Value *toStoreInTape = malloc;
      if (omp) {
        Value *numThreads = ompNumThreads();
        Value *tid = ompThreadId();
        IRBuilder<> entryBuilder(inversionAllocs);

        auto firstallocation =
            CreateAllocation(entryBuilder, malloc->getType(), numThreads,
                             malloc->getName() + "_malloccache");
        Value *tPtr = entryBuilder.CreateInBoundsGEP(
            malloc->getType(), firstallocation, ArrayRef<Value *>(tid));
        if (auto inst = dyn_cast<Instruction>(malloc)) {
          entryBuilder.SetInsertPoint(inst->getNextNode());
        }
        entryBuilder.CreateStore(malloc, tPtr);
        toStoreInTape = firstallocation;
      }
      addedTapeVals.push_back(toStoreInTape);
      return malloc;
    }

    ensureLookupCached(
        cast<Instruction>(malloc),
        /*shouldFree=*/reverseBlocks.size() > 0,
        /*scope*/ nullptr,
        cast<Instruction>(malloc)->getMetadata(LLVMContext::MD_tbaa));
    auto found2 = scopeMap.find(malloc);
    assert(found2 != scopeMap.end());
    assert(found2->second.first);

    Value *toadd;
    toadd = scopeAllocs[found2->second.first][0];
    for (auto u : toadd->users()) {
      if (auto ci = dyn_cast<CastInst>(u)) {
        toadd = ci;
        break;
      }
    }

    // llvm::errs() << " malloc: " << *malloc << "\n";
    // llvm::errs() << " toadd: " << *toadd << "\n";
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
    if (toadd->getContext().supportsTypedPointers()) {
#endif
      Type *innerType = toadd->getType();
      for (size_t i = 0,
                  limit = getSubLimits(
                              /*inForwardPass*/ true, nullptr,
                              LimitContext(
                                  /*ReverseLimit*/ reverseBlocks.size() > 0,
                                  BuilderQ.GetInsertBlock()))
                              .size();
           i < limit; ++i) {
        innerType = innerType->getPointerElementType();
      }
      if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
          toadd->getType() != innerType &&
          cast<IntegerType>(malloc->getType())->getBitWidth() == 1) {
        assert(innerType == Type::getInt8Ty(toadd->getContext()));
      } else {
        if (innerType != malloc->getType()) {
          llvm::errs() << "oldFunc:" << *oldFunc << "\n";
          llvm::errs() << "newFunc: " << *newFunc << "\n";
          llvm::errs() << " toadd: " << *toadd << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          llvm::errs() << "malloc: " << *malloc << "\n";
        }
        assert(innerType == malloc->getType());
      }
#if LLVM_VERSION_MAJOR >= 15
    }
#endif
#endif
    addedTapeVals.push_back(toadd);
    return malloc;
  }
  llvm::errs()
      << "Fell through on cacheForReverse. This should never happen.\n";
  assert(false);
}

/// Given an edge from BB to branchingBlock get the corresponding block to
/// branch to in the reverse pass
BasicBlock *GradientUtils::getReverseOrLatchMerge(BasicBlock *BB,
                                                  BasicBlock *branchingBlock) {
  assert(BB);
  // BB should be a forward pass block, assert that
  if (reverseBlocks.find(BB) == reverseBlocks.end()) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << "BB: " << *BB << "\n";
    llvm::errs() << "branchingBlock: " << *branchingBlock << "\n";
  }
  assert(reverseBlocks.find(BB) != reverseBlocks.end());
  assert(reverseBlocks.find(branchingBlock) != reverseBlocks.end());
  LoopContext lc;
  bool inLoop = getContext(BB, lc);

  LoopContext branchingContext;
  bool inLoopContext = getContext(branchingBlock, branchingContext);

  if (!inLoop)
    return reverseBlocks[BB].front();

  auto tup = std::make_tuple(BB, branchingBlock);
  if (newBlocksForLoop_cache.find(tup) != newBlocksForLoop_cache.end())
    return newBlocksForLoop_cache[tup];

  if (inLoop) {
    // If we're reversing a latch edge.
    bool incEntering = inLoopContext && branchingBlock == lc.header &&
                       lc.header == branchingContext.header;

    auto L = LI.getLoopFor(BB);
    auto latches = getLatches(L, lc.exitBlocks);
    // If we're reverseing a loop exit.
    bool exitEntering =
        std::find(latches.begin(), latches.end(), BB) != latches.end() &&
        std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) !=
            lc.exitBlocks.end();

    // If we're re-entering a loop, prepare a loop-level forward pass to
    // rematerialize any loop-scope rematerialization.
    if (incEntering || exitEntering) {
      SmallPtrSet<Instruction *, 1> loopRematerializations;
      SmallPtrSet<Instruction *, 1> loopReallocations;
      SmallPtrSet<Instruction *, 1> loopShadowReallocations;
      SmallPtrSet<Instruction *, 1> loopShadowZeroInits;
      SmallPtrSet<Instruction *, 1> loopShadowRematerializations;
      Loop *origLI = nullptr;
      for (auto pair : rematerializableAllocations) {
        if (pair.second.LI &&
            getNewFromOriginal(pair.second.LI->getHeader()) == L->getHeader()) {
          bool rematerialized = false;
          std::map<UsageKey, bool> Seen;
          for (auto pair : knownRecomputeHeuristic)
            if (!pair.second)
              Seen[UsageKey(pair.first, ValueType::Primal)] = false;

          if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                  ValueType::Primal>(this, pair.first, mode, Seen,
                                     notForAnalysis)) {
            rematerialized = true;
          }
          if (rematerialized) {
            if (auto inst = dyn_cast<Instruction>(pair.first))
              if (pair.second.LI->contains(inst->getParent())) {
                loopReallocations.insert(inst);
              }
            for (auto I : pair.second.stores)
              loopRematerializations.insert(I);
            origLI = pair.second.LI;
          }
        }
      }
      for (auto pair : backwardsOnlyShadows) {
        if (pair.second.LI &&
            getNewFromOriginal(pair.second.LI->getHeader()) == L->getHeader()) {
          if (auto inst = dyn_cast<Instruction>(pair.first)) {
            bool restoreStores = false;
            if (pair.second.LI->contains(inst->getParent())) {
              // TODO later make it so primalInitialize can be restored
              // rather than cached from primal
              if (!pair.second.primalInitialize) {
                loopShadowReallocations.insert(inst);
                restoreStores = true;
              }
            } else {
              // if (pair.second.primalInitialize) {
              //  loopShadowZeroInits.insert(inst);
              //}
              restoreStores = true;
            }
            if (restoreStores) {
              for (auto I : pair.second.stores) {
                loopShadowRematerializations.insert(I);
              }
            }
            origLI = pair.second.LI;
          }
        }
      }
      BasicBlock *resumeblock = reverseBlocks[BB].front();
      if (loopRematerializations.size() != 0 || loopReallocations.size() != 0 ||
          loopShadowRematerializations.size() != 0 ||
          loopShadowReallocations.size() != 0 ||
          loopShadowZeroInits.size() != 0) {
        auto found = rematerializedLoops_cache.find(L);
        if (found != rematerializedLoops_cache.end()) {
          resumeblock = found->second;
        } else {
          BasicBlock *enterB = BasicBlock::Create(
              BB->getContext(), "remat_enter", BB->getParent());
          rematerializedLoops_cache[L] = enterB;
          std::map<BasicBlock *, BasicBlock *> origToNewForward;
          for (auto B : origLI->getBlocks()) {
            BasicBlock *newB = BasicBlock::Create(
                B->getContext(),
                "remat_" + lc.header->getName() + "_" + B->getName(),
                BB->getParent());
            origToNewForward[B] = newB;
            reverseBlockToPrimal[newB] = getNewFromOriginal(B);
            if (B == origLI->getHeader()) {
              IRBuilder<> NB(newB);
              for (auto inst : loopShadowZeroInits) {
                auto anti = lookupM(invertPointerM(inst, NB), NB);
                StringRef funcName;
                SmallVector<Value *, 8> args;
                if (auto orig = dyn_cast<CallInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 14
                  for (auto &arg : orig->args())
#else
                  for (auto &arg : orig->arg_operands())
#endif
                  {
                    args.push_back(lookupM(getNewFromOriginal(arg), NB));
                  }
                  funcName = getFuncNameFromCall(orig);
                } else if (auto AI = dyn_cast<AllocaInst>(inst)) {
                  funcName = "malloc";
                  Value *sz =
                      lookupM(getNewFromOriginal(AI->getArraySize()), NB);

                  auto ci = ConstantInt::get(
                      sz->getType(),
                      B->getParent()
                              ->getParent()
                              ->getDataLayout()
                              .getTypeAllocSizeInBits(AI->getAllocatedType()) /
                          8);
                  sz = NB.CreateMul(sz, ci);
                  args.push_back(sz);
                }
                assert(funcName.size());

                applyChainRule(
                    NB,
                    [&](Value *anti) {
                      zeroKnownAllocation(NB, anti, args, funcName, TLI,
                                          dyn_cast<CallInst>(inst));
                    },
                    anti);
              }
            }
          }

          ValueToValueMapTy available;

          {
            IRBuilder<> NB(enterB);
            NB.CreateBr(origToNewForward[origLI->getHeader()]);
          }

          std::function<void(Loop *, bool)> handleLoop = [&](Loop *OL,
                                                             bool subLoop) {
            if (subLoop) {
              auto Header = OL->getHeader();
              IRBuilder<> NB(origToNewForward[Header]);
              LoopContext flc;
              getContext(getNewFromOriginal(Header), flc);

              auto iv = NB.CreatePHI(flc.var->getType(), 2, "fiv");
              auto inc = NB.CreateAdd(iv, ConstantInt::get(iv->getType(), 1));

              for (auto PH : predecessors(Header)) {
                if (notForAnalysis.count(PH))
                  continue;

                if (OL->contains(PH))
                  iv->addIncoming(inc, origToNewForward[PH]);
                else
                  iv->addIncoming(ConstantInt::get(iv->getType(), 0),
                                  origToNewForward[PH]);
              }
              available[flc.var] = iv;
              available[flc.incvar] = inc;
            }
            for (auto SL : OL->getSubLoops())
              handleLoop(SL, /*subLoop*/ true);
          };
          handleLoop(origLI, /*subLoop*/ false);

          for (auto B : origLI->getBlocks()) {
            auto newB = origToNewForward[B];
            IRBuilder<> NB(newB);

            // TODO fill available with relevant IV's surrounding and
            // IV's of inner loop phi's

            for (auto &I : *B) {
              // Only handle store, memset, and julia.write_barrier
              if (loopRematerializations.count(&I)) {
                if (auto SI = dyn_cast<StoreInst>(&I)) {
                  auto ts = NB.CreateStore(
                      lookupM(getNewFromOriginal(SI->getValueOperand()), NB,
                              available),
                      lookupM(getNewFromOriginal(SI->getPointerOperand()), NB,
                              available));
                  llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
                  ToCopy2.push_back(LLVMContext::MD_noalias);
                  ToCopy2.push_back(LLVMContext::MD_alias_scope);
                  ts->copyMetadata(*SI, ToCopy2);
#if LLVM_VERSION_MAJOR >= 10
                  ts->setAlignment(SI->getAlign());
#else
                  ts->setAlignment(SI->getAlignment());
#endif
                  ts->setVolatile(SI->isVolatile());
                  ts->setOrdering(SI->getOrdering());
                  ts->setSyncScopeID(SI->getSyncScopeID());
                  ts->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  StringRef funcName = getFuncNameFromCall(CI);
                  if (funcName == "enzyme_zerotype")
                    continue;
                  if (funcName == "julia.write_barrier" ||
                      isa<MemSetInst>(&I) || isa<MemTransferInst>(&I)) {

                    // TODO
                    SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                    for (auto &arg : CI->args())
#else
                    for (auto &arg : CI->arg_operands())
#endif
                      args.push_back(
                          lookupM(getNewFromOriginal(arg), NB, available));

                    SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                          ValueType::Primal);

                    auto Defs = getInvertedBundles(CI, BundleTypes, NB,
                                                   /*lookup*/ true, available);
#if LLVM_VERSION_MAJOR >= 11
                    auto cal =
                        NB.CreateCall(CI->getFunctionType(),
                                      CI->getCalledOperand(), args, Defs);
#else
                    auto cal = NB.CreateCall(CI->getCalledValue(), args, Defs);
#endif
                    cal->setAttributes(CI->getAttributes());
                    cal->setCallingConv(CI->getCallingConv());
                    cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  } else {
                    assert(isDeallocationFunction(funcName, TLI));
                    continue;
                  }
                } else {
                  assert(0 && "unhandlable loop rematerialization instruction");
                }
              } else if (loopReallocations.count(&I)) {
                LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                  &newFunc->getEntryBlock());

                auto inst = getNewFromOriginal((Value *)&I);

                auto found = scopeMap.find(inst);
                if (found == scopeMap.end()) {
                  AllocaInst *cache =
                      createCacheForScope(lctx, inst->getType(),
                                          inst->getName(), /*shouldFree*/ true);
                  assert(cache);
                  found = insert_or_assign(
                      scopeMap, inst,
                      std::pair<AssertingVH<AllocaInst>, LimitContext>(cache,
                                                                       lctx));
                }
                auto cache = found->second.first;
                if (auto MD = hasMetadata(&I, "enzyme_fromstack")) {
                  auto replacement = NB.CreateAlloca(
                      Type::getInt8Ty(I.getContext()),
                      lookupM(getNewFromOriginal(I.getOperand(0)), NB,
                              available));
                  auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                         MD->getOperand(0))
                                                         ->getValue())
                                       ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                  replacement->setAlignment(Align(Alignment));
#else
                  replacement->setAlignment(Alignment);
#endif
                  replacement->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  storeInstructionInCache(lctx, NB, replacement, cache);
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                  for (auto &arg : CI->args())
#else
                  for (auto &arg : CI->arg_operands())
#endif
                    args.push_back(
                        lookupM(getNewFromOriginal(arg), NB, available));

                  SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                        ValueType::Primal);

                  auto Defs = getInvertedBundles(CI, BundleTypes, NB,
                                                 /*lookup*/ true, available);
                  auto cal = NB.CreateCall(CI->getCalledFunction(), args, Defs);
                  llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
                  ToCopy2.push_back(LLVMContext::MD_noalias);
                  ToCopy2.push_back(LLVMContext::MD_alias_scope);
                  cal->copyMetadata(*CI, ToCopy2);
                  cal->setName("remat_" + CI->getName());
                  cal->setAttributes(CI->getAttributes());
                  cal->setCallingConv(CI->getCallingConv());
                  cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  storeInstructionInCache(lctx, NB, cal, cache);
                } else {
                  llvm::errs() << " realloc: " << I << "\n";
                  llvm_unreachable("Unknown loop reallocation");
                }
              }
              if (loopShadowRematerializations.count(&I)) {
                if (auto SI = dyn_cast<StoreInst>(&I)) {
                  Value *orig_ptr = SI->getPointerOperand();
                  Value *orig_val = SI->getValueOperand();
                  Type *valType = orig_val->getType();
                  assert(!isConstantValue(orig_ptr));

                  auto &DL = newFunc->getParent()->getDataLayout();

                  bool constantval = isConstantValue(orig_val) ||
                                     parseTBAA(I, DL).Inner0().isIntegral();

                  // TODO allow recognition of other types that could contain
                  // pointers [e.g. {void*, void*} or <2 x i64> ]
                  auto storeSize = DL.getTypeSizeInBits(valType) / 8;

                  //! Storing a floating point value
                  Type *FT = nullptr;
                  if (valType->isFPOrFPVectorTy()) {
                    FT = valType->getScalarType();
                  } else if (!valType->isPointerTy()) {
                    if (looseTypeAnalysis) {
                      auto fp = TR.firstPointer(storeSize, orig_ptr, &I,
                                                /*errifnotfound*/ false,
                                                /*pointerIntSame*/ true);
                      if (fp.isKnown()) {
                        FT = fp.isFloat();
                      } else if (isa<ConstantInt>(orig_val) ||
                                 valType->isIntOrIntVectorTy()) {
                        llvm::errs()
                            << "assuming type as integral for store: " << I
                            << "\n";
                        FT = nullptr;
                      } else {
                        TR.firstPointer(storeSize, orig_ptr, &I,
                                        /*errifnotfound*/ true,
                                        /*pointerIntSame*/ true);
                        llvm::errs()
                            << "cannot deduce type of store " << I << "\n";
                        assert(0 && "cannot deduce");
                      }
                    } else {
                      FT = TR.firstPointer(storeSize, orig_ptr, &I,
                                           /*errifnotfound*/ true,
                                           /*pointerIntSame*/ true)
                               .isFloat();
                    }
                  }
                  if (!FT) {
                    Value *valueop = nullptr;
                    if (constantval) {
                      Value *val =
                          lookupM(getNewFromOriginal(orig_val), NB, available);
                      valueop = val;
                      if (getWidth() > 1) {
                        Value *array =
                            UndefValue::get(getShadowType(val->getType()));
                        for (unsigned i = 0; i < getWidth(); ++i) {
                          array = NB.CreateInsertValue(array, val, {i});
                        }
                        valueop = array;
                      }
                    } else {
                      valueop =
                          lookupM(invertPointerM(orig_val, NB), NB, available);
                    }
                    SmallVector<Metadata *, 1> prevScopes;
                    if (auto prev =
                            SI->getMetadata(LLVMContext::MD_alias_scope)) {
                      for (auto &M : cast<MDNode>(prev)->operands()) {
                        prevScopes.push_back(M);
                      }
                    }
                    SmallVector<Metadata *, 1> prevNoAlias;
                    if (auto prev = SI->getMetadata(LLVMContext::MD_noalias)) {
                      for (auto &M : cast<MDNode>(prev)->operands()) {
                        prevNoAlias.push_back(M);
                      }
                    }
#if LLVM_VERSION_MAJOR >= 10
                    auto align = SI->getAlign();
#else
                    auto align = SI->getAlignment();
#endif
                    setPtrDiffe(SI, orig_ptr, valueop, NB, align,
                                SI->isVolatile(), SI->getOrdering(),
                                SI->getSyncScopeID(),
                                /*mask*/ nullptr, prevNoAlias, prevScopes);
                  }
                  // TODO shadow memtransfer
                } else if (auto MS = dyn_cast<MemSetInst>(&I)) {
                  if (!isConstantValue(MS->getArgOperand(0))) {
                    Value *args[4] = {
                        lookupM(invertPointerM(MS->getArgOperand(0), NB), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(1)), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(2)), NB,
                                available),
                        lookupM(getNewFromOriginal(MS->getArgOperand(3)), NB,
                                available)};

                    ValueType BundleTypes[4] = {
                        ValueType::Shadow, ValueType::Primal, ValueType::Primal,
                        ValueType::Primal};
                    auto Defs = getInvertedBundles(MS, BundleTypes, NB,
                                                   /*lookup*/ true, available);
                    auto cal =
                        NB.CreateCall(MS->getCalledFunction(), args, Defs);
                    llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
                    ToCopy2.push_back(LLVMContext::MD_noalias);
                    ToCopy2.push_back(LLVMContext::MD_alias_scope);
                    cal->copyMetadata(*MS, ToCopy2);
                    cal->setAttributes(MS->getAttributes());
                    cal->setCallingConv(MS->getCallingConv());
                    cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                  }
                } else if (auto CI = dyn_cast<CallInst>(&I)) {
                  StringRef funcName = getFuncNameFromCall(CI);
                  if (funcName == "julia.write_barrier") {

                    // TODO
                    SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
                    for (auto &arg : CI->args())
#else
                    for (auto &arg : CI->arg_operands())
#endif
                      if (!isConstantValue(arg))
                        args.push_back(
                            lookupM(invertPointerM(arg, NB), NB, available));

                    if (args.size()) {
                      SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                            ValueType::Primal);

                      auto Defs =
                          getInvertedBundles(CI, BundleTypes, NB,
                                             /*lookup*/ true, available);
#if LLVM_VERSION_MAJOR >= 11
                      auto cal =
                          NB.CreateCall(CI->getFunctionType(),
                                        CI->getCalledOperand(), args, Defs);
#else
                      auto cal =
                          NB.CreateCall(CI->getCalledValue(), args, Defs);
#endif
                      cal->setAttributes(CI->getAttributes());
                      cal->setCallingConv(CI->getCallingConv());
                      cal->setDebugLoc(getNewFromOriginal(I.getDebugLoc()));
                    }
                  } else {
                    assert(isDeallocationFunction(funcName, TLI));
                    continue;
                  }
                } else {
                  assert(
                      0 &&
                      "unhandlable loop shadow rematerialization instruction");
                }
              } else if (loopShadowReallocations.count(&I)) {

                LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                  &newFunc->getEntryBlock());
                auto ipfound = invertedPointers.find(&I);
                PHINode *placeholder = cast<PHINode>(&*ipfound->second);

                auto found = scopeMap.find(placeholder);
                if (found == scopeMap.end()) {
                  AllocaInst *cache = createCacheForScope(
                      lctx, placeholder->getType(), placeholder->getName(),
                      /*shouldFree*/ true);
                  assert(cache);
                  found = insert_or_assign(
                      scopeMap, (Value *&)placeholder,
                      std::pair<AssertingVH<AllocaInst>, LimitContext>(cache,
                                                                       lctx));
                }
                auto cache = found->second.first;
                Value *anti = nullptr;

                if (auto orig = dyn_cast<CallInst>(&I)) {
                  StringRef funcName = getFuncNameFromCall(orig);
                  assert(funcName.size());

                  auto dbgLoc = getNewFromOriginal(orig)->getDebugLoc();

                  SmallVector<Value *, 8> args;
#if LLVM_VERSION_MAJOR >= 14
                  for (auto &arg : orig->args())
#else
                  for (auto &arg : orig->arg_operands())
#endif
                  {
                    args.push_back(lookupM(getNewFromOriginal(arg), NB));
                  }

                  placeholder->setName("");
                  if (shadowHandlers.find(funcName) != shadowHandlers.end()) {

                    anti = shadowHandlers[funcName](NB, orig, args, this);
                  } else {
                    auto rule = [&]() {
#if LLVM_VERSION_MAJOR >= 11
                      Value *anti = NB.CreateCall(
                          orig->getFunctionType(), orig->getCalledOperand(),
                          args, orig->getName() + "'mi");
#else
                      Value *anti = NB.CreateCall(orig->getCalledValue(), args,
                                                  orig->getName() + "'mi");
#endif
                      cast<CallInst>(anti)->setAttributes(
                          orig->getAttributes());
                      cast<CallInst>(anti)->setCallingConv(
                          orig->getCallingConv());
                      cast<CallInst>(anti)->setDebugLoc(
                          getNewFromOriginal(I.getDebugLoc()));

                      cast<CallInst>(anti)->addAttribute(
                          AttributeList::ReturnIndex, Attribute::NoAlias);
                      cast<CallInst>(anti)->addAttribute(
                          AttributeList::ReturnIndex, Attribute::NonNull);
                      return anti;
                    };

                    anti = applyChainRule(orig->getType(), NB, rule);

                    if (auto MD = hasMetadata(orig, "enzyme_fromstack")) {
                      auto rule = [&](Value *anti) {
                        AllocaInst *replacement = NB.CreateAlloca(
                            Type::getInt8Ty(orig->getContext()), args[0]);
                        replacement->takeName(anti);
                        auto Alignment =
                            cast<ConstantInt>(
                                cast<ConstantAsMetadata>(MD->getOperand(0))
                                    ->getValue())
                                ->getLimitedValue();
#if LLVM_VERSION_MAJOR >= 10
                        replacement->setAlignment(Align(Alignment));
#else
                        replacement->setAlignment(Alignment);
#endif
                        replacement->setDebugLoc(
                            getNewFromOriginal(I.getDebugLoc()));
                        return replacement;
                      };

                      Value *replacement = applyChainRule(
                          Type::getInt8Ty(orig->getContext()), NB, rule, anti);

                      replaceAWithB(cast<Instruction>(anti), replacement);
                      erase(cast<Instruction>(anti));
                      anti = replacement;
                    }

                    applyChainRule(
                        NB,
                        [&](Value *anti) {
                          zeroKnownAllocation(NB, anti, args, funcName, TLI,
                                              orig);
                        },
                        anti);
                  }
                } else {
                  llvm_unreachable("Unknown shadow rematerialization value");
                }
                assert(anti);
                storeInstructionInCache(lctx, NB, anti, cache);
              }
            }

            llvm::SmallPtrSet<llvm::BasicBlock *, 8> origExitBlocks;
            getExitBlocks(origLI, origExitBlocks);
            // Remap a branch to the header to enter the incremented
            // reverse of that block.
            auto remap = [&](BasicBlock *rB) {
              // Remap of an exit branch is to go to the reverse
              // exiting block.
              if (origExitBlocks.count(rB)) {
                return reverseBlocks[getNewFromOriginal(B)].front();
              }
              // Reverse of an incrementing branch is go to the
              // reverse of the branching block.
              if (rB == origLI->getHeader())
                return reverseBlocks[getNewFromOriginal(B)].front();
              auto found = origToNewForward.find(rB);
              if (found == origToNewForward.end()) {
                llvm::errs() << *newFunc << "\n";
                llvm::errs() << *origLI << "\n";
                llvm::errs() << *rB << "\n";
              }
              assert(found != origToNewForward.end());
              return found->second;
            };

            // TODO clone terminator
            auto TI = B->getTerminator();
            assert(TI);
            if (notForAnalysis.count(B)) {
              NB.CreateUnreachable();
            } else if (auto BI = dyn_cast<BranchInst>(TI)) {
              if (BI->isUnconditional()) {
                if (notForAnalysis.count(BI->getSuccessor(0)))
                  NB.CreateUnreachable();
                else
                  NB.CreateBr(remap(BI->getSuccessor(0)));
              } else {
                if (notForAnalysis.count(BI->getSuccessor(0))) {
                  if (notForAnalysis.count(BI->getSuccessor(1))) {
                    NB.CreateUnreachable();
                  } else {
                    NB.CreateBr(remap(BI->getSuccessor(1)));
                  }
                } else if (notForAnalysis.count(BI->getSuccessor(1))) {
                  NB.CreateBr(remap(BI->getSuccessor(0)));
                } else {
                  NB.CreateCondBr(
                      lookupM(getNewFromOriginal(BI->getCondition()), NB,
                              available),
                      remap(BI->getSuccessor(0)), remap(BI->getSuccessor(1)));
                }
              }
            } else if (auto SI = dyn_cast<SwitchInst>(TI)) {
              auto NSI = NB.CreateSwitch(
                  lookupM(getNewFromOriginal(SI->getCondition()), NB,
                          available),
                  remap(SI->getDefaultDest()));
              for (auto cas : SI->cases()) {
                if (!notForAnalysis.count(cas.getCaseSuccessor()))
                  NSI->addCase(cas.getCaseValue(),
                               remap(cas.getCaseSuccessor()));
              }
            } else {
              assert(isa<UnreachableInst>(TI));
              NB.CreateUnreachable();
            }
            // Fixup phi nodes that may have their predecessors now changed by
            // the phi unwrapping
            if (!notForAnalysis.count(B) &&
                NB.GetInsertBlock() != origToNewForward[B]) {
              for (auto S0 : successors(B)) {
                if (!origToNewForward.count(S0))
                  continue;
                auto S = origToNewForward[S0];
                assert(S);
                for (auto I = S->begin(), E = S->end(); I != E; ++I) {
                  PHINode *orig = dyn_cast<PHINode>(&*I);
                  if (orig == nullptr)
                    break;
                  for (unsigned Op = 0, NumOps = orig->getNumOperands();
                       Op != NumOps; ++Op)
                    if (orig->getIncomingBlock(Op) == origToNewForward[B])
                      orig->setIncomingBlock(Op, NB.GetInsertBlock());
                }
              }
            }
          }
          resumeblock = enterB;
        }
      }

      if (incEntering) {
        BasicBlock *incB = BasicBlock::Create(
            BB->getContext(),
            "inc" + reverseBlocks[lc.header].front()->getName(),
            BB->getParent());
        incB->moveAfter(reverseBlocks[lc.header].back());

        IRBuilder<> tbuild(incB);

        Value *av = tbuild.CreateLoad(lc.var->getType(), lc.antivaralloc);
        Value *sub =
            tbuild.CreateAdd(av, ConstantInt::get(av->getType(), -1), "",
                             /*NUW*/ false, /*NSW*/ true);
        tbuild.CreateStore(sub, lc.antivaralloc);
        tbuild.CreateBr(resumeblock);
        return newBlocksForLoop_cache[tup] = incB;
      } else {
        assert(exitEntering);
        BasicBlock *incB = BasicBlock::Create(
            BB->getContext(),
            "merge" + reverseBlocks[lc.header].front()->getName() + "_" +
                branchingBlock->getName(),
            BB->getParent());
        incB->moveAfter(reverseBlocks[branchingBlock].back());

        IRBuilder<> tbuild(reverseBlocks[branchingBlock].back());

        Value *lim = nullptr;
        if (lc.dynamic && assumeDynamicLoopOfSizeOne(L)) {
          lim = ConstantInt::get(lc.var->getType(), 0);
        } else if (lc.dynamic) {
          // Must be in a reverse pass fashion for a lookup to index bound to be
          // legal
          assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
          LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                            lc.preheader);
          lim = lookupValueFromCache(
              lc.var->getType(),
              /*forwardPass*/ false, tbuild, lctx,
              getDynamicLoopLimit(LI.getLoopFor(lc.header)),
              /*isi1*/ false, /*available*/ ValueToValueMapTy());
        } else {
          lim = lookupM(lc.trueLimit, tbuild);
        }

        tbuild.SetInsertPoint(incB);
        tbuild.CreateStore(lim, lc.antivaralloc);
        tbuild.CreateBr(resumeblock);

        return newBlocksForLoop_cache[tup] = incB;
      }
    }
  }

  return newBlocksForLoop_cache[tup] = reverseBlocks[BB].front();
}

void GradientUtils::forceContexts() {
  for (auto BB : originalBlocks) {
    LoopContext lc;
    getContext(BB, lc);
  }
}

bool GradientUtils::legalRecompute(const Value *val,
                                   const ValueToValueMapTy &available,
                                   IRBuilder<> *BuilderM, bool reverse,
                                   bool legalRecomputeCache) const {
  {
    auto found = available.find(val);
    if (found != available.end()) {
      if (found->second)
        return true;
      else {
        return false;
      }
    }
  }

  if (auto phi = dyn_cast<PHINode>(val)) {
    if (auto uiv = hasUninverted(val)) {
      if (auto dli = dyn_cast_or_null<LoadInst>(uiv)) {
        return legalRecompute(
            dli, available, BuilderM,
            reverse); // TODO ADD && !TR.intType(getOriginal(dli),
                      // /*mustfind*/false).isPossibleFloat();
      }
      if (phi->getNumIncomingValues() == 0) {
        return false;
      }
    }

    if (phi->getNumIncomingValues() == 0) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *phi << "\n";
    }
    assert(phi->getNumIncomingValues() != 0);
    auto parent = phi->getParent();
    struct {
      Function *func;
      const LoopInfo &FLI;
    } options[2] = {{newFunc, LI}, {oldFunc, OrigLI}};
    for (const auto &tup : options) {
      if (parent->getParent() == tup.func) {
        for (auto &val : phi->incoming_values()) {
          if (isPotentialLastLoopValue(val, parent, tup.FLI)) {
            return false;
          }
        }
        if (tup.FLI.isLoopHeader(parent)) {
          // Currently can only recompute header
          // with two incoming values
          if (phi->getNumIncomingValues() != 2)
            return false;
          auto L = tup.FLI.getLoopFor(parent);

          // Only recomputable if non recursive.
          SmallPtrSet<Instruction *, 2> seen;
          SmallVector<Instruction *, 1> todo;
          for (auto PH : predecessors(parent)) {
            // Prior iterations must be recomputable without
            // this value.
            if (L->contains(PH)) {
              if (auto I =
                      dyn_cast<Instruction>(phi->getIncomingValueForBlock(PH)))
                if (L->contains(I->getParent()))
                  todo.push_back(I);
            }
          }

          while (todo.size()) {
            auto cur = todo.back();
            todo.pop_back();
            if (seen.count(cur))
              continue;
            seen.insert(cur);
            if (cur == phi)
              return false;
            for (auto &op : cur->operands()) {
              if (auto I = dyn_cast<Instruction>(op)) {
                if (L->contains(I->getParent()))
                  todo.push_back(I);
              }
            }
          }
        }
        return true;
      }
    }
    return false;
  }

  if (isa<Instruction>(val) &&
      cast<Instruction>(val)->getMetadata("enzyme_mustcache")) {
    return false;
  }

  // If this is a load from cache already, dont force a cache of this
  if (legalRecomputeCache && isa<LoadInst>(val) &&
      CacheLookups.count(cast<LoadInst>(val))) {
    return true;
  }

  // TODO consider callinst here

  if (auto li = dyn_cast<Instruction>(val)) {

    const IntrinsicInst *II;
    if (isa<LoadInst>(li) ||
        ((II = dyn_cast<IntrinsicInst>(li)) &&
         (II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_f ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_f ||
          II->getIntrinsicID() == Intrinsic::masked_load))) {
      // If this is an already unwrapped value, legal to recompute again.
      if (unwrappedLoads.find(li) != unwrappedLoads.end())
        return legalRecompute(unwrappedLoads.find(li)->second, available,
                              BuilderM, reverse);

      const Instruction *orig = nullptr;
      if (li->getParent()->getParent() == oldFunc) {
        orig = li;
      } else if (li->getParent()->getParent() == newFunc) {
        orig = isOriginal(li);
        // todo consider when we pass non original queries
        if (orig && !isa<LoadInst>(orig)) {
          return legalRecompute(orig, available, BuilderM, reverse,
                                legalRecomputeCache);
        }
      } else {
        llvm::errs() << " newFunc: " << *newFunc << "\n";
        llvm::errs() << " parent: " << *li->getParent()->getParent() << "\n";
        llvm::errs() << " li: " << *li << "\n";
        assert(0 && "illegal load legalRecopmute query");
      }

      if (orig) {
        assert(can_modref_map);
        auto found = can_modref_map->find(const_cast<Instruction *>(orig));
        if (found == can_modref_map->end()) {
          llvm::errs() << *newFunc << "\n";
          llvm::errs() << *oldFunc << "\n";
          llvm::errs() << "can_modref_map:\n";
          for (auto &pair : *can_modref_map) {
            llvm::errs() << " + " << *pair.first << ": " << pair.second
                         << " of func "
                         << pair.first->getParent()->getParent()->getName()
                         << "\n";
          }
          llvm::errs() << "couldn't find in can_modref_map: " << *li << " - "
                       << *orig << " in fn: "
                       << orig->getParent()->getParent()->getName();
        }
        assert(found != can_modref_map->end());
        if (!found->second)
          return true;
        // if insertion block of this function:
        BasicBlock *fwdBlockIfReverse = nullptr;
        if (BuilderM) {
          fwdBlockIfReverse = BuilderM->GetInsertBlock();
          if (!reverse) {
            auto found = reverseBlockToPrimal.find(BuilderM->GetInsertBlock());
            if (found != reverseBlockToPrimal.end()) {
              fwdBlockIfReverse = found->second;
              reverse = true;
            }
          }
          if (fwdBlockIfReverse->getParent() != oldFunc)
            fwdBlockIfReverse =
                cast_or_null<BasicBlock>(isOriginal(fwdBlockIfReverse));
        }
        if (mode == DerivativeMode::ReverseModeCombined && fwdBlockIfReverse) {
          if (reverse) {
            bool failed = false;
            allFollowersOf(
                const_cast<Instruction *>(orig), [&](Instruction *I) -> bool {
                  if (I->mayWriteToMemory() &&
                      writesToMemoryReadBy(
                          OrigAA, TLI,
                          /*maybeReader*/ const_cast<Instruction *>(orig),
                          /*maybeWriter*/ I)) {
                    failed = true;
                    EmitWarning(
                        "UncacheableLoad", *orig, "Load must be recomputed ",
                        *orig, " in reverse_",
                        BuilderM->GetInsertBlock()->getName(), " due to ", *I);
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });
            if (!failed)
              return true;
          } else {
            Instruction *origStart = &*BuilderM->GetInsertPoint();
            do {
              if (Instruction *og = isOriginal(origStart)) {
                origStart = og;
                break;
              }
              origStart = origStart->getNextNode();
            } while (true);
            if (OrigDT.dominates(origStart, const_cast<Instruction *>(orig))) {
              bool failed = false;

              allInstructionsBetween(
                  const_cast<GradientUtils *>(this)->LI, origStart,
                  const_cast<Instruction *>(orig), [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(
                            OrigAA, TLI,
                            /*maybeReader*/ const_cast<Instruction *>(orig),
                            /*maybeWriter*/ I)) {
                      failed = true;
                      EmitWarning("UncacheableLoad", *orig,
                                  "Load must be recomputed ", *orig, " in ",
                                  BuilderM->GetInsertBlock()->getName(),
                                  " due to ", *I);
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (!failed)
                return true;
            }
          }
        }
        return false;
      } else {
        if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(li))) {
          return legalRecompute(dli, available, BuilderM, reverse);
        }

        // TODO mark all the explicitly legal nodes (caches, etc)
        return true;
        llvm::errs() << *li << " orig: " << orig
                     << " parent: " << li->getParent()->getParent()->getName()
                     << "\n";
        llvm_unreachable("unknown load to redo!");
      }
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    auto n = getFuncNameFromCall(const_cast<CallInst *>(ci));
    auto called = ci->getCalledFunction();
    Intrinsic::ID ID = Intrinsic::not_intrinsic;

    if (ci->hasFnAttr("enzyme_shouldrecompute") ||
        (called && called->hasFnAttribute("enzyme_shouldrecompute")) ||
        isMemFreeLibMFunction(n, &ID) || n == "lgamma_r" || n == "lgammaf_r" ||
        n == "lgammal_r" || n == "__lgamma_r_finite" ||
        n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" || n == "tanh" ||
        n == "tanhf" || n == "__pow_finite" ||
        n == "julia.pointer_from_objref" || n.startswith("enzyme_wrapmpi$$") ||
        n == "omp_get_thread_num" || n == "omp_get_max_threads") {
      return true;
    }
#if LLVM_VERSION_MAJOR >= 14
    if (ci->doesNotAccessMemory())
#else
    if (ci->hasFnAttr(Attribute::ReadNone) ||
        (called && called->hasFnAttribute(Attribute::ReadNone)))
#endif
      return true;
    if (isPointerArithmeticInst(ci))
      return true;
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (inst->mayReadOrWriteMemory()) {
      return false;
    }
  }

  return true;
}

//! Given the option to recompute a value or re-use an old one, return true if
//! it is faster to recompute this value from scratch
bool GradientUtils::shouldRecompute(const Value *val,
                                    const ValueToValueMapTy &available,
                                    IRBuilder<> *BuilderM) {
  if (available.count(val))
    return true;
  // TODO: remake such that this returns whether a load to a cache is more
  // expensive than redoing the computation.

  // If this is a load from cache already, just reload this
  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_fromcache"))
    return true;

  if (!isa<Instruction>(val))
    return true;

  const Instruction *inst = cast<Instruction>(val);

  if (TapesToPreventRecomputation.count(inst))
    return false;

  if (knownRecomputeHeuristic.find(inst) != knownRecomputeHeuristic.end()) {
    return knownRecomputeHeuristic[inst];
  }
  if (auto OrigInst = isOriginal(inst)) {
    if (knownRecomputeHeuristic.find(OrigInst) !=
        knownRecomputeHeuristic.end()) {
      return knownRecomputeHeuristic[OrigInst];
    }
  }

  if (isa<CastInst>(val) || isa<GetElementPtrInst>(val))
    return true;

  if (EnzymeNewCache && !EnzymeMinCutCache) {
    // if this has operands that need to be loaded and haven't already been
    // loaded
    // TODO, just cache this
    for (auto &op : inst->operands()) {
      if (!legalRecompute(op, available, BuilderM)) {

        // If this is a load from cache already, dont force a cache of this
        if (isa<LoadInst>(op) && CacheLookups.count(cast<LoadInst>(op)))
          continue;

        // If a previously cached this operand, don't let it trigger the
        // heuristic for caching this value instead.
        if (scopeMap.find(op) != scopeMap.end())
          continue;

        // If the actually overwritten operand is in a different loop scope
        // don't cache this value instead as it may require more memory
        LoopContext lc1;
        LoopContext lc2;
        bool inLoop1 =
            getContext(const_cast<Instruction *>(inst)->getParent(), lc1);
        bool inLoop2 = getContext(cast<Instruction>(op)->getParent(), lc2);
        if (inLoop1 != inLoop2 || (inLoop1 && (lc1.header != lc2.header))) {
          continue;
        }

        // If a placeholder phi for inversion (and we know from above not
        // recomputable)
        if (!isa<PHINode>(op) &&
            dyn_cast_or_null<LoadInst>(hasUninverted(op))) {
          goto forceCache;
        }

        // Even if cannot recompute (say a phi node), don't force a reload if it
        // is possible to just use this instruction from forward pass without
        // issue
        if (auto i2 = dyn_cast<Instruction>(op)) {
          if (!i2->mayReadOrWriteMemory()) {
            LoopContext lc;
            bool inLoop = const_cast<GradientUtils *>(this)->getContext(
                i2->getParent(), lc);
            if (!inLoop) {
              // TODO upgrade this to be all returns that this could enter from
              BasicBlock *orig = isOriginal(i2->getParent());
              assert(orig);
              bool legal = BlocksDominatingAllReturns.count(orig);
              if (legal) {
                continue;
              }
            }
          }
        }
      forceCache:;
        EmitWarning("ChosenCache", *inst, "Choosing to cache use ", *inst,
                    " due to ", *op);
        return false;
      }
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
    if (!op->mayReadOrWriteMemory() || isReadNone(op))
      return true;
    switch (op->getIntrinsicID()) {
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::exp:
    case Intrinsic::log:
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f:
      return true;
    default:
      return false;
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    auto called = ci->getCalledFunction();
    auto n = getFuncNameFromCall(const_cast<CallInst *>(ci));
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    if ((called && called->hasFnAttribute("enzyme_shouldrecompute")) ||
        isMemFreeLibMFunction(n, &ID) || n == "lgamma_r" || n == "lgammaf_r" ||
        n == "lgammal_r" || n == "__lgamma_r_finite" ||
        n == "__lgammaf_r_finite" || n == "__lgammal_r_finite" || n == "tanh" ||
        n == "tanhf" || n == "__pow_finite" ||
        n == "julia.pointer_from_objref" || n.startswith("enzyme_wrapmpi$$") ||
        n == "omp_get_thread_num" || n == "omp_get_max_threads") {
      return true;
    }
    if (isPointerArithmeticInst(ci))
      return true;
  }

  // cache a call, assuming its longer to run that
  if (isa<CallInst>(val)) {
    llvm::errs() << " caching call: " << *val << "\n";
    // cast<CallInst>(val)->getCalledFunction()->dump();
    return false;
  }

  return true;
}

MDNode *GradientUtils::getDerivativeAliasScope(const Value *origptr,
                                               ssize_t newptr) {
  origptr = getBaseObject(origptr);

  auto found = differentialAliasScopeDomains.find(origptr);
  if (found == differentialAliasScopeDomains.end()) {
    MDBuilder MDB(oldFunc->getContext());
    MDNode *scope = MDB.createAnonymousAliasScopeDomain(
        (" diff: %" + origptr->getName()).str());
    // vec.first = scope;
    // found = differentialAliasScope.find(origptr);
    found = differentialAliasScopeDomains.insert(std::make_pair(origptr, scope))
                .first;
  }
  auto &mp = differentialAliasScope[origptr];
  auto found2 = mp.find(newptr);
  if (found2 == mp.end()) {
    MDBuilder MDB(oldFunc->getContext());
    std::string name;
    if (newptr == -1)
      name = "primal";
    else
      name = "shadow_" + std::to_string(newptr);
    found2 = mp.insert(std::make_pair(newptr, MDB.createAnonymousAliasScope(
                                                  found->second, name)))
                 .first;
  }
  return found2->second;
}

GradientUtils *GradientUtils::CreateFromClone(
    EnzymeLogic &Logic, unsigned width, Function *todiff,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, FnTypeInfo &oldTypeInfo,
    DIFFE_TYPE retType, ArrayRef<DIFFE_TYPE> constant_args, bool returnUsed,
    bool shadowReturnUsed, std::map<AugmentedStruct, int> &returnMapping,
    bool omp) {
  assert(!todiff->empty());
  Function *oldFunc = todiff;

  // Since this is forward pass this should always return the tape (at index 0)
  returnMapping[AugmentedStruct::Tape] = 0;

  int returnCount = 0;

  if (returnUsed) {
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    returnMapping[AugmentedStruct::Return] = returnCount + 1;
    ++returnCount;
  }

  // We don't need to differentially return something that we know is not a
  // pointer (or somehow needed for shadow analysis)
  if (shadowReturnUsed) {
    assert(retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED);
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    returnMapping[AugmentedStruct::DifferentialReturn] = returnCount + 1;
    ++returnCount;
  }

  ReturnType returnValue;
  if (returnCount == 0)
    returnValue = ReturnType::Tape;
  else if (returnCount == 1)
    returnValue = ReturnType::TapeAndReturn;
  else if (returnCount == 2)
    returnValue = ReturnType::TapeAndTwoReturns;
  else
    llvm_unreachable("illegal number of elements in augmented return struct");

  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  std::string prefix = "fakeaugmented";
  if (width > 1)
    prefix += std::to_string(width);
  prefix += "_";
  prefix += todiff->getName().str();

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      DerivativeMode::ReverseModePrimal, /* width */ width, oldFunc,
      invertedPointers, constant_args, constant_values, nonconstant_values,
      returnvals,
      /*returnValue*/ returnValue, retType, prefix, &originalToNew,
      /*diffeReturnArg*/ false, /*additionalArg*/ nullptr);

  // Convert overwritten args from the input function to the preprocessed
  // function

  FnTypeInfo typeInfo(oldFunc);
  {
    auto toarg = todiff->arg_begin();
    auto olarg = oldFunc->arg_begin();
    for (; toarg != todiff->arg_end(); ++toarg, ++olarg) {

      {
        auto fd = oldTypeInfo.Arguments.find(toarg);
        assert(fd != oldTypeInfo.Arguments.end());
        typeInfo.Arguments.insert(
            std::pair<Argument *, TypeTree>(olarg, fd->second));
      }

      {
        auto cfd = oldTypeInfo.KnownValues.find(toarg);
        assert(cfd != oldTypeInfo.KnownValues.end());
        typeInfo.KnownValues.insert(
            std::pair<Argument *, std::set<int64_t>>(olarg, cfd->second));
      }
    }
    typeInfo.Return = oldTypeInfo.Return;
  }

  TypeResults TR = TA.analyzeFunction(typeInfo);
  assert(TR.getFunction() == oldFunc);

  auto res = new GradientUtils(
      Logic, newFunc, oldFunc, TLI, TA, TR, invertedPointers, constant_values,
      nonconstant_values, retType, constant_args, originalToNew,
      DerivativeMode::ReverseModePrimal, width, omp);
  return res;
}

DIFFE_TYPE GradientUtils::getReturnDiffeType(llvm::Value *orig,
                                             bool *primalReturnUsedP,
                                             bool *shadowReturnUsedP) const {
  bool shadowReturnUsed = false;

  DIFFE_TYPE subretType;
  if (isConstantValue(orig)) {
    subretType = DIFFE_TYPE::CONSTANT;
  } else {
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      subretType = DIFFE_TYPE::DUP_ARG;
      shadowReturnUsed = true;
    } else {
      if (!orig->getType()->isFPOrFPVectorTy() &&
          TR.query(orig).Inner0().isPossiblePointer()) {
        if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                ValueType::Shadow>(this, orig,
                                   DerivativeMode::ReverseModePrimal,
                                   notForAnalysis)) {
          subretType = DIFFE_TYPE::DUP_ARG;
          shadowReturnUsed = true;
        } else
          subretType = DIFFE_TYPE::CONSTANT;
      } else {
        subretType = DIFFE_TYPE::OUT_DIFF;
      }
    }
  }

  if (primalReturnUsedP) {
    bool subretused =
        unnecessaryValuesP->find(orig) == unnecessaryValuesP->end();
    auto found = knownRecomputeHeuristic.find(orig);
    if (found != knownRecomputeHeuristic.end()) {
      if (!found->second) {
        subretused = true;
      }
    }
    *primalReturnUsedP = subretused;
  }

  if (shadowReturnUsedP)
    *shadowReturnUsedP = shadowReturnUsed;
  return subretType;
}

DIFFE_TYPE GradientUtils::getDiffeType(Value *v, bool foreignFunction) const {
  if (isConstantValue(v) && !foreignFunction) {
    return DIFFE_TYPE::CONSTANT;
  }

  auto argType = v->getType();

  if (!argType->isFPOrFPVectorTy() &&
      (TR.query(v).Inner0().isPossiblePointer() || foreignFunction)) {
    if (argType->isPointerTy()) {
      auto at = getBaseObject(v);
      if (auto arg = dyn_cast<Argument>(at)) {
        if (ArgDiffeTypes[arg->getArgNo()] == DIFFE_TYPE::DUP_NONEED) {
          return DIFFE_TYPE::DUP_NONEED;
        }
      } else if (isa<AllocaInst>(at) || isAllocationCall(at, TLI)) {
        assert(unnecessaryValuesP);
        if (unnecessaryValuesP->count(at))
          return DIFFE_TYPE::DUP_NONEED;
      }
    }
    return DIFFE_TYPE::DUP_ARG;
  } else {
    if (foreignFunction)
      assert(!argType->isIntOrIntVectorTy());
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit)
      return DIFFE_TYPE::DUP_ARG;
    else
      return DIFFE_TYPE::OUT_DIFF;
  }
}

Constant *GradientUtils::GetOrCreateShadowConstant(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA,
    Constant *oval, DerivativeMode mode, unsigned width, bool AtomicAdd) {
  if (isa<ConstantPointerNull>(oval)) {
    return oval;
  } else if (isa<UndefValue>(oval)) {
    return oval;
  } else if (isa<ConstantInt>(oval)) {
    return oval;
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getElementAsConstant(i), mode, width, AtomicAdd));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantArray::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantStruct::get(CD->getType(), Vals);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(GetOrCreateShadowConstant(
          Logic, TLI, TA, CD->getOperand(i), mode, width, AtomicAdd));
    }
    return ConstantVector::get(Vals);
  } else if (auto F = dyn_cast<Function>(oval)) {
    return GetOrCreateShadowFunction(Logic, TLI, TA, F, mode, width, AtomicAdd);
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    auto C = GetOrCreateShadowConstant(Logic, TLI, TA, arg->getOperand(0), mode,
                                       width, AtomicAdd);
    if (arg->isCast() || arg->getOpcode() == Instruction::GetElementPtr ||
        arg->getOpcode() == Instruction::Add) {
      SmallVector<Constant *, 8> NewOps;
      for (unsigned i = 0, e = arg->getNumOperands(); i != e; ++i)
        NewOps.push_back(i == 0 ? C : arg->getOperand(i));
      return arg->getWithOperands(NewOps);
    }
  } else if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (arg->getName() == "_ZTVN10__cxxabiv120__si_class_type_infoE" ||
        arg->getName() == "_ZTVN10__cxxabiv117__class_type_infoE" ||
        arg->getName() == "_ZTVN10__cxxabiv121__vmi_class_type_infoE")
      return arg;

    if (hasMetadata(arg, "enzyme_shadow")) {
      auto md = arg->getMetadata("enzyme_shadow");
      if (!isa<MDTuple>(md)) {
        llvm::errs() << *arg << "\n";
        llvm::errs() << *md << "\n";
        assert(0 && "cannot compute with global variable that doesn't have "
                    "marked shadow global");
        report_fatal_error(
            "cannot compute with global variable that doesn't "
            "have marked shadow global (metadata incorrect type)");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      return gvemd->getValue();
    }

    auto Arch = llvm::Triple(arg->getParent()->getTargetTriple()).getArch();
    int SharedAddrSpace = Arch == Triple::amdgcn
                              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                              : 3;
    int AddrSpace = cast<PointerType>(arg->getType())->getAddressSpace();
    if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
         Arch == Triple::amdgcn) &&
        AddrSpace == SharedAddrSpace) {
      assert(0 && "shared memory not handled in meta global");
    }

    // Create global variable locally if not externally visible
    if (arg->isConstant() || arg->hasInternalLinkage() ||
        arg->hasPrivateLinkage() ||
        (arg->hasExternalLinkage() && arg->hasInitializer())) {
      Type *type = arg->getValueType();
      auto shadow = new GlobalVariable(
          *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
          Constant::getNullValue(type), arg->getName() + "_shadow", arg,
          arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
          arg->isExternallyInitialized());
      arg->setMetadata("enzyme_shadow",
                       MDTuple::get(shadow->getContext(),
                                    {ConstantAsMetadata::get(shadow)}));
#if LLVM_VERSION_MAJOR >= 11
      shadow->setAlignment(arg->getAlign());
#else
      shadow->setAlignment(arg->getAlignment());
#endif
      shadow->setUnnamedAddr(arg->getUnnamedAddr());
      if (arg->hasInitializer())
        shadow->setInitializer(GetOrCreateShadowConstant(
            Logic, TLI, TA, cast<Constant>(arg->getOperand(0)), mode, width,
            AtomicAdd));
      return shadow;
    }
  }
  llvm::errs() << " unknown constant to create shadow of: " << *oval << "\n";
  llvm_unreachable("unknown constant to create shadow of");
}

Constant *GradientUtils::GetOrCreateShadowFunction(
    EnzymeLogic &Logic, TargetLibraryInfo &TLI, TypeAnalysis &TA, Function *fn,
    DerivativeMode mode, unsigned width, bool AtomicAdd) {
  //! Todo allow tape propagation
  //  Note that specifically this should _not_ be called with topLevel=true
  //  (since it may not be valid to always assume we can recompute the
  //  augmented primal) However, in the absence of a way to pass tape data
  //  from an indirect augmented (and also since we dont presently allow
  //  indirect augmented calls), topLevel MUST be true otherwise subcalls will
  //  not be able to lookup the augmenteddata/subdata (triggering an assertion
  //  failure, among much worse)
  bool isRealloc = false;
  if (fn->empty()) {
    if (hasMetadata(fn, "enzyme_callwrapper")) {
      auto md = fn->getMetadata("enzyme_callwrapper");
      if (!isa<MDTuple>(md)) {
        llvm::errs() << *fn << "\n";
        llvm::errs() << *md << "\n";
        assert(0 && "callwrapper of incorrect type");
        report_fatal_error("callwrapper of incorrect type");
      }
      auto md2 = cast<MDTuple>(md);
      assert(md2->getNumOperands() == 1);
      auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
      fn = cast<Function>(gvemd->getValue());
    } else {
      auto oldfn = fn;
      fn = Function::Create(oldfn->getFunctionType(), Function::InternalLinkage,
                            "callwrap_" + oldfn->getName(), oldfn->getParent());
      BasicBlock *entry = BasicBlock::Create(fn->getContext(), "entry", fn);
      IRBuilder<> B(entry);
      SmallVector<Value *, 4> args;
      for (auto &a : fn->args())
        args.push_back(&a);
      auto res = B.CreateCall(oldfn, args);
      if (fn->getReturnType()->isVoidTy())
        B.CreateRetVoid();
      else
        B.CreateRet(res);
      oldfn->setMetadata(
          "enzyme_callwrapper",
          MDTuple::get(oldfn->getContext(), {ConstantAsMetadata::get(fn)}));
      if (oldfn->getName() == "realloc")
        isRealloc = true;
    }
  }
  std::vector<bool> overwritten_args;
  FnTypeInfo type_args(fn);
  if (isRealloc) {
    llvm::errs() << "warning: assuming realloc only creates pointers\n";
    type_args.Return.insert({-1, -1}, BaseType::Pointer);
  }

  // conservatively assume that we can only cache existing floating types
  // (i.e. that all args are overwritten)
  std::vector<DIFFE_TYPE> types;
  for (auto &a : fn->args()) {
    overwritten_args.push_back(!a.getType()->isFPOrFPVectorTy());
    TypeTree TT;
    if (a.getType()->isFPOrFPVectorTy())
      TT.insert({-1}, ConcreteType(a.getType()->getScalarType()));
    type_args.Arguments.insert(std::pair<Argument *, TypeTree>(&a, TT));
    type_args.KnownValues.insert(
        std::pair<Argument *, std::set<int64_t>>(&a, {}));
    DIFFE_TYPE typ;
    if (a.getType()->isFPOrFPVectorTy()) {
      typ = mode == DerivativeMode::ForwardMode ? DIFFE_TYPE::DUP_ARG
                                                : DIFFE_TYPE::OUT_DIFF;
    } else if (a.getType()->isIntegerTy() &&
               cast<IntegerType>(a.getType())->getBitWidth() < 16) {
      typ = DIFFE_TYPE::CONSTANT;
    } else if (a.getType()->isVoidTy() || a.getType()->isEmptyTy()) {
      typ = DIFFE_TYPE::CONSTANT;
    } else {
      typ = DIFFE_TYPE::DUP_ARG;
    }
    types.push_back(typ);
  }

  DIFFE_TYPE retType = fn->getReturnType()->isFPOrFPVectorTy() &&
                               mode != DerivativeMode::ForwardMode
                           ? DIFFE_TYPE::OUT_DIFF
                           : DIFFE_TYPE::DUP_ARG;

  if (fn->getReturnType()->isVoidTy() || fn->getReturnType()->isEmptyTy() ||
      (fn->getReturnType()->isIntegerTy() &&
       cast<IntegerType>(fn->getReturnType())->getBitWidth() < 16))
    retType = DIFFE_TYPE::CONSTANT;

  if (mode != DerivativeMode::ForwardMode && retType == DIFFE_TYPE::DUP_ARG) {
    if (auto ST = dyn_cast<StructType>(fn->getReturnType())) {
      size_t numflt = 0;

      for (unsigned i = 0; i < ST->getNumElements(); ++i) {
        auto midTy = ST->getElementType(i);
        if (midTy->isFPOrFPVectorTy())
          numflt++;
      }
      if (numflt == ST->getNumElements())
        retType = DIFFE_TYPE::OUT_DIFF;
    }
  }

  switch (mode) {
  case DerivativeMode::ForwardMode: {
    Constant *newf = Logic.CreateForwardDiff(
        fn, retType, types, TA, false, mode, /*freeMemory*/ true, width,
        nullptr, type_args, overwritten_args, /*augmented*/ nullptr);

    assert(newf);

    std::string prefix = "_enzyme_forward";

    if (width > 1) {
      prefix += std::to_string(width);
    }

    std::string globalname = (prefix + "_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), newf->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, newf,
                              globalname);
    }

    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  case DerivativeMode::ForwardModeSplit: {
    auto &augdata = Logic.CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TA,
        /*returnUsed*/ !fn->getReturnType()->isEmptyTy() &&
            !fn->getReturnType()->isVoidTy(),
        /*shadowReturnUsed*/ false, type_args, overwritten_args,
        /*forceAnonymousTape*/ true, width, AtomicAdd);
    Constant *newf = Logic.CreateForwardDiff(
        fn, retType, types, TA, false, mode, /*freeMemory*/ true, width,
        nullptr, type_args, overwritten_args, /*augmented*/ &augdata);

    assert(newf);

    std::string prefix = "_enzyme_forwardsplit";

    if (width > 1) {
      prefix += std::to_string(width);
    }

    auto cdata = ConstantStruct::get(
        StructType::get(newf->getContext(),
                        {augdata.fn->getType(), newf->getType()}),
        {augdata.fn, newf});

    std::string globalname = (prefix + "_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), cdata->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, cdata,
                              globalname);
    }

    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
  case DerivativeMode::ReverseModePrimal: {
    // TODO re atomic add consider forcing it to be atomic always as fallback if
    // used in a parallel context
    bool returnUsed =
        !fn->getReturnType()->isEmptyTy() && !fn->getReturnType()->isVoidTy();
    bool shadowReturnUsed = returnUsed && (retType == DIFFE_TYPE::DUP_ARG ||
                                           retType == DIFFE_TYPE::DUP_NONEED);
    auto &augdata = Logic.CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TA, returnUsed, shadowReturnUsed,
        type_args, overwritten_args, /*forceAnonymousTape*/ true, width,
        AtomicAdd);
    Constant *newf = Logic.CreatePrimalAndGradient(
        (ReverseCacheKey){.todiff = fn,
                          .retType = retType,
                          .constant_args = types,
                          .overwritten_args = overwritten_args,
                          .returnUsed = false,
                          .shadowReturnUsed = false,
                          .mode = DerivativeMode::ReverseModeGradient,
                          .width = width,
                          .freeMemory = true,
                          .AtomicAdd = AtomicAdd,
                          .additionalType =
                              Type::getInt8PtrTy(fn->getContext()),
                          .forceAnonymousTape = true,
                          .typeInfo = type_args},
        TA,
        /*map*/ &augdata);
    assert(newf);
    auto cdata = ConstantStruct::get(
        StructType::get(newf->getContext(),
                        {augdata.fn->getType(), newf->getType()}),
        {augdata.fn, newf});
    std::string globalname = ("_enzyme_reverse_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), cdata->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, cdata,
                              globalname);
    }
    return ConstantExpr::getPointerCast(GV, fn->getType());
  }
  }
  llvm_unreachable("Illegal state: unknown mode for GetOrCreateShadowFunction");
}

void GradientUtils::getReverseBuilder(IRBuilder<> &Builder2, bool original) {
  assert(reverseBlocks.size());
  BasicBlock *BB = Builder2.GetInsertBlock();
  if (original)
    BB = getNewFromOriginal(BB);
  assert(reverseBlocks.find(BB) != reverseBlocks.end());
  BasicBlock *BB2 = reverseBlocks[BB].back();
  if (!BB2) {
    llvm::errs() << "oldFunc: " << oldFunc << "\n";
    llvm::errs() << "newFunc: " << newFunc << "\n";
    llvm::errs() << "could not invert " << *BB;
  }
  assert(BB2);

  if (BB2->getTerminator())
    Builder2.SetInsertPoint(BB2->getTerminator());
  else
    Builder2.SetInsertPoint(BB2);
  Builder2.SetCurrentDebugLocation(
      getNewFromOriginal(Builder2.getCurrentDebugLocation()));
  Builder2.setFastMathFlags(getFast());
}

void GradientUtils::getForwardBuilder(IRBuilder<> &Builder2) {
  Instruction *insert = &*Builder2.GetInsertPoint();
  Instruction *nInsert = getNewFromOriginal(insert);

  assert(nInsert);

  Builder2.SetInsertPoint(getNextNonDebugInstruction(nInsert));
  Builder2.SetCurrentDebugLocation(
      getNewFromOriginal(Builder2.getCurrentDebugLocation()));
  Builder2.setFastMathFlags(getFast());
}

#if LLVM_VERSION_MAJOR >= 10
void GradientUtils::setPtrDiffe(Instruction *orig, Value *ptr, Value *newval,
                                IRBuilder<> &BuilderM, MaybeAlign align,
                                bool isVolatile, AtomicOrdering ordering,
                                SyncScope::ID syncScope, Value *mask,
                                ArrayRef<Metadata *> noAlias,
                                ArrayRef<Metadata *> scopes)
#else
void GradientUtils::setPtrDiffe(Instruction *orig, Value *ptr, Value *newval,
                                IRBuilder<> &BuilderM, unsigned align,
                                bool isVolatile, AtomicOrdering ordering,
                                SyncScope::ID syncScope, Value *mask,
                                ArrayRef<Metadata *> noAlias,
                                ArrayRef<Metadata *> scopes)
#endif
{
  if (auto inst = dyn_cast<Instruction>(ptr)) {
    assert(inst->getParent()->getParent() == oldFunc);
  }
  if (auto arg = dyn_cast<Argument>(ptr)) {
    assert(arg->getParent() == oldFunc);
  }

  Value *origptr = ptr;

  ptr = invertPointerM(ptr, BuilderM);
  if (!isOriginalBlock(*BuilderM.GetInsertBlock()) &&
      mode != DerivativeMode::ForwardMode)
    ptr = lookupM(ptr, BuilderM);

  if (mask && !isOriginalBlock(*BuilderM.GetInsertBlock()) &&
      mode != DerivativeMode::ForwardMode)
    mask = lookupM(mask, BuilderM);

  size_t idx = 0;

  auto rule = [&](Value *ptr, Value *newval) {
    if (!mask) {
      auto ts = BuilderM.CreateStore(newval, ptr);
      if (align)
#if LLVM_VERSION_MAJOR >= 10
        ts->setAlignment(*align);
#else
        ts->setAlignment(align);
#endif
      ts->setVolatile(isVolatile);
      ts->setOrdering(ordering);
      ts->setSyncScopeID(syncScope);
      SmallVector<Metadata *, 1> scopeMD = {
          getDerivativeAliasScope(origptr, idx)};
      for (auto M : scopes)
        scopeMD.push_back(M);
      auto scope = MDNode::get(ts->getContext(), scopeMD);
      ts->setMetadata(LLVMContext::MD_alias_scope, scope);

      ts->setMetadata(LLVMContext::MD_tbaa,
                      orig->getMetadata(LLVMContext::MD_tbaa));
      ts->setMetadata(LLVMContext::MD_tbaa_struct,
                      orig->getMetadata(LLVMContext::MD_tbaa_struct));
      ts->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));

      SmallVector<Metadata *, 1> MDs;
      for (ssize_t j = -1; j < getWidth(); j++) {
        if (j != (ssize_t)idx)
          MDs.push_back(getDerivativeAliasScope(origptr, j));
      }
      for (auto M : noAlias)
        MDs.push_back(M);
      if (MDs.size()) {
        auto noscope = MDNode::get(ptr->getContext(), MDs);
        ts->setMetadata(LLVMContext::MD_noalias, noscope);
      }
    } else {
      Type *tys[] = {newval->getType(), ptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_store, tys);
      assert(align);
#if LLVM_VERSION_MAJOR >= 10
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(ptr->getContext()), align->value());
#else
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(ptr->getContext()), align);
#endif
      Value *args[] = {newval, ptr, alignv, mask};
      auto ts = BuilderM.CreateCall(F, args);
      ts->setCallingConv(F->getCallingConv());
      ts->setMetadata(LLVMContext::MD_tbaa,
                      orig->getMetadata(LLVMContext::MD_tbaa));
      ts->setMetadata(LLVMContext::MD_tbaa_struct,
                      orig->getMetadata(LLVMContext::MD_tbaa_struct));
      ts->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));
    }
    idx++;
  };

  applyChainRule(BuilderM, rule, ptr, newval);
}

Type *GradientUtils::getShadowType(Type *ty, unsigned width) {
  if (width > 1) {
    if (ty->isVoidTy())
      return ty;
    return ArrayType::get(ty, width);
  } else {
    return ty;
  }
}

Type *GradientUtils::getShadowType(Type *ty) {
  return getShadowType(ty, width);
}

Value *GradientUtils::extractMeta(IRBuilder<> &Builder, Value *Agg,
                                  unsigned off, const Twine &name) {
  return extractMeta(Builder, Agg, ArrayRef<unsigned>({off}), name);
}

Value *GradientUtils::extractMeta(IRBuilder<> &Builder, Value *Agg,
                                  ArrayRef<unsigned> off_init,
                                  const Twine &name) {
  std::vector<unsigned> off(off_init.begin(), off_init.end());
  while (off.size() != 0) {
    if (auto Ins = dyn_cast<InsertValueInst>(Agg)) {
      size_t until = Ins->getNumIndices();
      if (off.size() < until)
        until = off.size();
      bool subset = true;
      for (size_t i = 0; i < until; i++) {
        if (Ins->getIndices()[i] != off[i]) {
          subset = false;
          break;
        }
      }
      if (!subset) {
        Agg = Ins->getAggregateOperand();
        continue;
      } else if (until < Ins->getNumIndices()) {
        break;
      } else {
        off.erase(off.begin(), off.begin() + until);
        Agg = Ins->getInsertedValueOperand();
        continue;
      }
    }
    if (auto ext = dyn_cast<ExtractValueInst>(Agg)) {
      off.insert(off.begin(), ext->getIndices().begin(),
                 ext->getIndices().end());
      Agg = ext->getAggregateOperand();
      continue;
    }
    if (auto CA = dyn_cast<ConstantAggregateZero>(Agg)) {
      Agg = CA->getElementValue(off[0]);
      off.erase(off.begin(), off.begin() + 1);
    }
    break;
  }
  if (off.size() == 0)
    return Agg;
  if (Agg->getType()->isVectorTy() && off.size() == 1)
    return Builder.CreateExtractElement(Agg, off[0], name);

  return Builder.CreateExtractValue(Agg, off, name);
}

llvm::Value *GradientUtils::recursiveFAdd(llvm::IRBuilder<> &B,
                                          llvm::Value *lhs, llvm::Value *rhs,
                                          llvm::ArrayRef<unsigned> lhs_off,
                                          llvm::ArrayRef<unsigned> rhs_off,
                                          llvm::Value *prev, bool vectorLayer) {
  llvm::Type *lhs_ty = lhs->getType();
  if (!vectorLayer) {
    for (auto idx : lhs_off)
      lhs_ty = getSubType(lhs_ty, idx);
    llvm::Type *rhs_ty = rhs->getType();
    for (auto idx : rhs_off)
      rhs_ty = getSubType(rhs_ty, idx);
    assert(lhs_ty == rhs_ty);
  }
  if (lhs_ty->isFPOrFPVectorTy()) {
    if (lhs_off.size())
      lhs = extractMeta(B, lhs, lhs_off);
    if (rhs_off.size())
      rhs = extractMeta(B, rhs, rhs_off);
    llvm::Value *res = nullptr;
    if (auto fp = llvm::dyn_cast<llvm::ConstantFP>(lhs)) {
      if (fp->isZero())
        res = rhs;
    }
    if (auto fp = llvm::dyn_cast<llvm::ConstantFP>(rhs)) {
      if (fp->isZero())
        res = lhs;
    }
    if (!res) {
#if LLVM_VERSION_MAJOR >= 10
      if (auto *FPMO = dyn_cast<FPMathOperator>(rhs))
        if (FPMO->getOpcode() == Instruction::FNeg) {
          res = B.CreateFSub(lhs, FPMO->getOperand(0));
        }
#endif
    }
    if (!res) {
      if (auto *S = dyn_cast<BinaryOperator>(rhs)) {
        if (S->getOpcode() == Instruction::FSub) {
          if (auto C = dyn_cast<ConstantFP>(S->getOperand(0)))
            if (C->isZero())
              res = B.CreateFSub(lhs, S->getOperand(1));
        }
      }
    }
    if (!res) {
      res = B.CreateFAdd(lhs, rhs);
    }
    if (lhs_off.size()) {
      assert(prev);
      res = B.CreateInsertValue(prev, res, lhs_off);
    }
    return res;
  } else if (isa<ArrayType>(lhs_ty) || isa<StructType>(lhs_ty)) {
    if (prev == nullptr)
      prev = llvm::UndefValue::get(lhs_ty);

    size_t size;
    if (auto AT = dyn_cast<ArrayType>(lhs_ty))
      size = AT->getNumElements();
    else
      size = cast<StructType>(lhs_ty)->getNumElements();

    for (size_t i = 0; i < size; ++i) {
      llvm::SmallVector<unsigned, 1> nlhs_off(lhs_off.begin(), lhs_off.end());
      if (vectorLayer)
        nlhs_off.insert(nlhs_off.begin(), i);
      else
        nlhs_off.push_back(i);
      llvm::SmallVector<unsigned, 1> nrhs_off(rhs_off.begin(), rhs_off.end());
      if (vectorLayer)
        nrhs_off.insert(nrhs_off.begin(), i);
      else
        nrhs_off.push_back(i);
      prev = recursiveFAdd(B, lhs, rhs, nlhs_off, nrhs_off, prev);
    }
    return prev;
  }
  llvm_unreachable("Unknown type to recursively accumulate");
}

Value *GradientUtils::invertPointerM(Value *const oval, IRBuilder<> &BuilderM,
                                     bool nullShadow) {
  assert(oval);
  if (auto inst = dyn_cast<Instruction>(oval)) {
    assert(inst->getParent()->getParent() == oldFunc);
  }
  if (auto arg = dyn_cast<Argument>(oval)) {
    assert(arg->getParent() == oldFunc);
  }

  if (isa<ConstantPointerNull>(oval)) {
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (isa<UndefValue>(oval)) {
    if (nullShadow)
      return Constant::getNullValue(getShadowType(oval->getType()));
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (isa<ConstantInt>(oval)) {
    if (nullShadow)
      return Constant::getNullValue(getShadowType(oval->getType()));
    return applyChainRule(oval->getType(), BuilderM, [&]() { return oval; });
  } else if (auto CD = dyn_cast<ConstantDataArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumElements(); i < len; i++) {
      Value *val =
          invertPointerM(CD->getElementAsConstant(i), BuilderM, nullShadow);
      Vals.push_back(cast<Constant>(val));
    }
    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantArray::get(CD->getType(), Vals);
    };
    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantArray>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Value *val = invertPointerM(CD->getOperand(i), BuilderM, nullShadow);
      Vals.push_back(cast<Constant>(val));
    }

    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantArray::get(CD->getType(), Vals);
    };

    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantStruct>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(cast<Constant>(
          invertPointerM(CD->getOperand(i), BuilderM, nullShadow)));
    }

    auto rule = [&CD](ArrayRef<Constant *> Vals) {
      return ConstantStruct::get(CD->getType(), Vals);
    };
    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (auto CD = dyn_cast<ConstantVector>(oval)) {
    SmallVector<Constant *, 1> Vals;
    for (size_t i = 0, len = CD->getNumOperands(); i < len; i++) {
      Vals.push_back(cast<Constant>(
          invertPointerM(CD->getOperand(i), BuilderM, nullShadow)));
    }

    auto rule = [](ArrayRef<Constant *> Vals) {
      return ConstantVector::get(Vals);
    };

    return applyChainRule(CD->getType(), Vals, BuilderM, rule);
  } else if (isa<ConstantData>(oval) && nullShadow) {
    auto rule = [&oval]() { return Constant::getNullValue(oval->getType()); };

    return applyChainRule(oval->getType(), BuilderM, rule);
  }

  if (isConstantValue(oval) && !isa<InsertValueInst>(oval) &&
      !isa<ExtractValueInst>(oval) && !isa<InsertElementInst>(oval) &&
      !isa<ExtractElementInst>(oval)) {
    // NOTE, this is legal and the correct resolution, however, our activity
    // analysis honeypot no longer exists

    // Nulling the shadow for a constant is only necessary if any of the data
    // could contain a float (e.g. should not be applied to pointers).
    if (nullShadow) {
      auto ty = TR.query(oval);
      auto &dl = newFunc->getParent()->getDataLayout();
      size_t size = (dl.getTypeSizeInBits(oval->getType()) + 7) / 8;
      auto CT = ty[{-1}];
      bool couldContainFloat = CT.isFloat();
      bool allFloat = CT.isFloat();
      if (!CT.isKnown()) {
        size_t i = 0;
        for (; i < size;) {
          auto CT2 = ty[{(int)i}];
          if (CT2.isFloat() || !CT2.isKnown()) {
            couldContainFloat = true;
            break;
          }
          if (CT2 == BaseType::Pointer) {
            i += dl.getPointerSizeInBits() / 8;
            continue;
          }
          i++;
        }
      }
      if (couldContainFloat) {
        if (allFloat)
          return Constant::getNullValue(getShadowType(oval->getType()));
        else {
          IRBuilder<> bb(inversionAllocs);
          if (auto arg = dyn_cast<Instruction>(oval)) {
            arg = getNewFromOriginal(arg);
            // Go one after since otherwise we won't be able
            // to use in the store.
            arg = arg->getNextNode();
            while (auto PN = dyn_cast<PHINode>(arg)) {
              if (PN->getNumIncomingValues() == 0)
                break;
              arg = PN->getNextNode();
            }
            bb.SetInsertPoint(arg);
          }
          auto alloc = bb.CreateAlloca(oval->getType());
          auto AT = ArrayType::get(bb.getInt8Ty(), size);
          bb.CreateStore(getNewFromOriginal(oval), alloc);
          Value *cur = bb.CreatePointerCast(alloc, PointerType::getUnqual(AT));
          size_t i = 0;
          assert(size > 0);
          for (; i < size;) {
            auto CT2 = ty[{(int)i}];
            if (CT2 == BaseType::Pointer) {
              i += dl.getPointerSizeInBits() / 8;
              continue;
            } else if (auto flt = CT2.isFloat()) {
              auto ptr = bb.CreateConstInBoundsGEP2_32(AT, cur, 0, i);
              ptr = bb.CreatePointerCast(ptr, PointerType::getUnqual(flt));
              bb.CreateStore(Constant::getNullValue(flt), ptr);
              size_t chunk = 0;
              if (flt->isFloatTy()) {
                chunk = 4;
              } else if (flt->isDoubleTy()) {
                chunk = 8;
              } else if (flt->isHalfTy()) {
                chunk = 2;
              } else {
                llvm::errs() << *flt << "\n";
                assert(0 && "unhandled float type");
              }
              i += chunk;
            } else if (CT2 != BaseType::Integer) {
              auto ptr = bb.CreateConstInBoundsGEP2_32(AT, cur, 0, i);
              bb.CreateStore(Constant::getNullValue(bb.getInt8Ty()), ptr);
              i++;
            } else {
              i++;
            }
          }
          auto res = bb.CreateLoad(oval->getType(), alloc);
          auto rule = [&res]() { return res; };
          auto res2 = applyChainRule(oval->getType(), BuilderM, rule);
          invertedPointers.insert(std::make_pair(
              (const Value *)oval, InvertedPointerVH(this, res2)));
          return res2;
        }
      }
    }

    if (isa<ConstantExpr>(oval)) {
      auto rule = [&oval]() { return oval; };
      return applyChainRule(oval->getType(), BuilderM, rule);
    }

    Value *newval = getNewFromOriginal(oval);

    auto rule = [&]() { return newval; };

    return applyChainRule(oval->getType(), BuilderM, rule);
  }

  auto M = oldFunc->getParent();
  assert(oval);

  {
    auto ifound = invertedPointers.find(oval);
    if (ifound != invertedPointers.end()) {
      return &*ifound->second;
    }
  }

  if (mode != DerivativeMode::ForwardMode &&
      mode != DerivativeMode::ForwardModeSplit && nullShadow) {
    auto CT = TR.query(oval)[{-1}];
    if (CT.isFloat()) {
      return Constant::getNullValue(getShadowType(oval->getType()));
    }
  }

  if (isa<Argument>(oval) && cast<Argument>(oval)->hasByValAttr()) {
    IRBuilder<> bb(inversionAllocs);

    Type *subType = nullptr;
    auto attr = cast<Argument>(oval)->getAttribute(Attribute::ByVal);
    subType = attr.getValueAsType();

    auto rule1 = [&]() {
      AllocaInst *antialloca = bb.CreateAlloca(
          subType, cast<PointerType>(oval->getType())->getPointerAddressSpace(),
          nullptr, oval->getName() + "'ipa");

      auto dst_arg =
          bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
      auto len_arg = ConstantInt::get(
          Type::getInt64Ty(oval->getContext()),
          M->getDataLayout().getTypeAllocSizeInBits(subType) / 8);
      auto volatile_arg = ConstantInt::getFalse(oval->getContext());

      Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
      Type *tys[] = {dst_arg->getType(), len_arg->getType()};
      bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);

      return antialloca;
    };

    Value *antialloca = applyChainRule(oval->getType(), bb, rule1);

    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));

    return antialloca;
  } else if (auto arg = dyn_cast<GlobalAlias>(oval)) {
    Value *aliasTarget = arg->getAliasee();
    return invertPointerM(aliasTarget, BuilderM, nullShadow);
  } else if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (!hasMetadata(arg, "enzyme_shadow")) {

      if ((mode == DerivativeMode::ReverseModeCombined ||
           mode == DerivativeMode::ForwardMode) &&
          arg->getType()->getPointerAddressSpace() == 0) {
        auto CT = TR.query(arg)[{-1, -1}];
        // Can only localy replace a global variable if it is
        // known not to contain a pointer, which may be initialized
        // outside of this function to contain other memory which
        // will not have a shadow within the current function.
        if (CT.isKnown() && CT != BaseType::Pointer) {
          bool seen = false;
          MemoryLocation
#if LLVM_VERSION_MAJOR >= 12
              Loc = MemoryLocation(oval, LocationSize::beforeOrAfterPointer());
#else
              Loc = MemoryLocation(oval, LocationSize::unknown());
#endif
          for (CallInst *CI : originalCalls) {
            if (isa<IntrinsicInst>(CI))
              continue;
            if (!isConstantInstruction(CI)) {
              auto F = getFunctionFromCall(CI);
              if (F && isMemFreeLibMFunction(F->getName())) {
                continue;
              }
              if (llvm::isModOrRefSet(OrigAA.getModRefInfo(CI, Loc))) {
                seen = true;
                llvm::errs() << " cannot shadow-inline global " << *oval
                             << " due to " << *CI << "\n";
                goto endCheck;
              }
            }
          }
        endCheck:;
          if (!seen) {
            IRBuilder<> bb(inversionAllocs);
            Type *allocaTy = arg->getValueType();

            auto rule1 = [&]() {
              AllocaInst *antialloca = bb.CreateAlloca(
                  allocaTy, arg->getType()->getPointerAddressSpace(), nullptr,
                  arg->getName() + "'ipa");
              if (arg->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
                antialloca->setAlignment(Align(arg->getAlignment()));
#else
                antialloca->setAlignment(arg->getAlignment());
#endif
              }
              return antialloca;
            };

            Value *antialloca = applyChainRule(arg->getType(), bb, rule1);

            invertedPointers.insert(std::make_pair(
                (const Value *)oval, InvertedPointerVH(this, antialloca)));

            auto rule2 = [&](Value *antialloca) {
              auto dst_arg = bb.CreateBitCast(
                  antialloca, Type::getInt8PtrTy(arg->getContext()));
              auto val_arg =
                  ConstantInt::get(Type::getInt8Ty(arg->getContext()), 0);
              auto len_arg =
                  ConstantInt::get(Type::getInt64Ty(arg->getContext()),
                                   M->getDataLayout().getTypeAllocSizeInBits(
                                       arg->getValueType()) /
                                       8);
              auto volatile_arg = ConstantInt::getFalse(oval->getContext());

              Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
              Type *tys[] = {dst_arg->getType(), len_arg->getType()};
              auto memset = cast<CallInst>(bb.CreateCall(
                  Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
#if LLVM_VERSION_MAJOR >= 10
              if (arg->getAlignment()) {
                memset->addParamAttr(
                    0, Attribute::getWithAlignment(arg->getContext(),
                                                   Align(arg->getAlignment())));
              }
#else
              if (arg->getAlignment() != 0) {
                memset->addParamAttr(
                    0, Attribute::getWithAlignment(arg->getContext(),
                                                   arg->getAlignment()));
              }
#endif
              memset->addParamAttr(0, Attribute::NonNull);
              assert((width > 1 && antialloca->getType() ==
                                       ArrayType::get(arg->getType(), width)) ||
                     antialloca->getType() == arg->getType());
              return antialloca;
            };

            return applyChainRule(arg->getType(), bb, rule2, antialloca);
          }
        }
      }

      auto Arch =
          llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
      int SharedAddrSpace =
          Arch == Triple::amdgcn
              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
              : 3;
      int AddrSpace = cast<PointerType>(arg->getType())->getAddressSpace();
      if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
           Arch == Triple::amdgcn) &&
          AddrSpace == SharedAddrSpace) {
        llvm::errs() << "warning found shared memory\n";
        // #if LLVM_VERSION_MAJOR >= 11
        Type *type = arg->getValueType();
        // TODO this needs initialization by entry
        auto shadow = new GlobalVariable(
            *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
            UndefValue::get(type), arg->getName() + "_shadow", arg,
            arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
            arg->isExternallyInitialized());
        arg->setMetadata("enzyme_shadow",
                         MDTuple::get(shadow->getContext(),
                                      {ConstantAsMetadata::get(shadow)}));
        shadow->setMetadata("enzyme_internalshadowglobal",
                            MDTuple::get(shadow->getContext(), {}));
#if LLVM_VERSION_MAJOR >= 11
        shadow->setAlignment(arg->getAlign());
#else
        shadow->setAlignment(arg->getAlignment());
#endif
        shadow->setUnnamedAddr(arg->getUnnamedAddr());
        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }

      // Create global variable locally if not externally visible
      //  If a variable is constant, for forward mode it will also
      //  only be read, so invert initializing is fine.
      //  For reverse mode, any floats will be +='d into, but never
      //  read, and any pointers will be used as expected. The never
      //  read means even if two globals for floats, that's fine.
      //  As long as the pointers point to equivalent places (which
      //  they should from the same initialization), it is also ok.
      if (arg->hasInternalLinkage() || arg->hasPrivateLinkage() ||
          (arg->hasExternalLinkage() && arg->hasInitializer()) ||
          arg->isConstant()) {
        Type *elemTy = arg->getValueType();
        IRBuilder<> B(inversionAllocs);

        auto rule = [&]() {
          auto shadow = new GlobalVariable(
              *arg->getParent(), elemTy, arg->isConstant(), arg->getLinkage(),
              Constant::getNullValue(elemTy), arg->getName() + "_shadow", arg,
              arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
              arg->isExternallyInitialized());
          arg->setMetadata("enzyme_shadow",
                           MDTuple::get(shadow->getContext(),
                                        {ConstantAsMetadata::get(shadow)}));
#if LLVM_VERSION_MAJOR >= 11
          shadow->setAlignment(arg->getAlign());
#else
          shadow->setAlignment(arg->getAlignment());
#endif
          shadow->setUnnamedAddr(arg->getUnnamedAddr());

          return shadow;
        };

        Value *shadow = applyChainRule(oval->getType(), BuilderM, rule);

        if (arg->hasInitializer()) {
          applyChainRule(
              BuilderM,
              [&](Value *shadow, Value *ip) {
                cast<GlobalVariable>(shadow)->setInitializer(
                    cast<Constant>(ip));
              },
              shadow,
              invertPointerM(arg->getInitializer(), B, /*nullShadow*/ true));
        }

        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }

      llvm::errs() << *oldFunc->getParent() << "\n";
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *arg << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global");
    }
    auto md = arg->getMetadata("enzyme_shadow");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *arg << "\n";
      llvm::errs() << *md << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global (metadata incorrect type)");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto cs = cast<Constant>(gvemd->getValue());

    if (width > 1) {
      SmallVector<Constant *, 2> Vals;
      for (unsigned i = 0; i < width; ++i) {

        Constant *idxs[] = {
            ConstantInt::get(Type::getInt32Ty(cs->getContext()), 0),
            ConstantInt::get(Type::getInt32Ty(cs->getContext()), i)};
        Constant *elem = ConstantExpr::getInBoundsGetElementPtr(
            getShadowType(arg->getValueType()), cs, idxs);
        Vals.push_back(elem);
      }

      auto agg = ConstantArray::get(
          cast<ArrayType>(getShadowType(arg->getType())), Vals);

      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, agg)));
      return agg;
    } else {
      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, cs)));
      return cs;
    }
  } else if (auto fn = dyn_cast<Function>(oval)) {
    Constant *shadow =
        GetOrCreateShadowFunction(Logic, TLI, TA, fn, mode, width, AtomicAdd);
    if (width > 1) {
      SmallVector<Constant *, 3> arr;
      for (unsigned i = 0; i < width; ++i) {
        arr.push_back(shadow);
      }
      ArrayType *arrTy = ArrayType::get(shadow->getType(), width);
      shadow = ConstantArray::get(arrTy, arr);
    }
    return shadow;
  } else if (auto arg = dyn_cast<CastInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *invertOp = invertPointerM(arg->getOperand(0), bb, nullShadow);
    Type *shadowTy = arg->getDestTy();

    auto rule = [&](Value *invertOp) {
      return bb.CreateCast(arg->getOpcode(), invertOp, shadowTy,
                           arg->getName() + "'ipc");
    };

    Value *shadow = applyChainRule(shadowTy, bb, rule, invertOp);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    IRBuilder<> bb(inversionAllocs);
    auto ip = invertPointerM(arg->getOperand(0), bb, nullShadow);

    if (arg->isCast()) {
#if LLVM_VERSION_MAJOR < 18
      if (auto PT = dyn_cast<PointerType>(arg->getType())) {
        if (isConstantValue(arg->getOperand(0)) &&
            PT->getPointerElementType()->isFunctionTy()) {
          goto end;
        }
      }
#endif
      if (isa<Constant>(ip)) {
        auto rule = [&arg](Value *ip) {
          return ConstantExpr::getCast(arg->getOpcode(), cast<Constant>(ip),
                                       arg->getType());
        };

        return applyChainRule(arg->getType(), bb, rule, ip);

      } else {
        auto rule = [&](Value *ip) {
          return bb.CreateCast((Instruction::CastOps)arg->getOpcode(), ip,
                               arg->getType(), arg->getName() + "'ipc");
        };

        Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));

        return shadow;
      }
    } else if (arg->getOpcode() == Instruction::GetElementPtr) {
      if (auto C = dyn_cast<Constant>(ip)) {
        auto rule = [&arg, &C]() {
          SmallVector<Constant *, 8> NewOps;
          for (unsigned i = 0, e = arg->getNumOperands(); i != e; ++i)
            NewOps.push_back(i == 0 ? C : arg->getOperand(i));
          return cast<Value>(arg->getWithOperands(NewOps));
        };

        return applyChainRule(arg->getType(), bb, rule);
      } else {
        SmallVector<Value *, 4> invertargs;
        for (unsigned i = 0; i < arg->getNumOperands() - 1; ++i) {
          Value *b = getNewFromOriginal(arg->getOperand(1 + i));
          invertargs.push_back(b);
        }

        auto rule = [&bb, &arg, &invertargs](Value *ip) {
          // TODO mark this the same inbounds as the original
          return bb.CreateGEP(cast<GEPOperator>(ip)->getSourceElementType(), ip,
                              invertargs, arg->getName() + "'ipg");
        };

        Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

        invertedPointers.insert(std::make_pair(
            (const Value *)oval, InvertedPointerVH(this, shadow)));
        return shadow;
      }
    } else {
      llvm::errs() << *arg << "\n";
      assert(0 && "unhandled");
    }
    goto end;
  } else if (auto arg = dyn_cast<ExtractValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto ip = invertPointerM(arg->getOperand(0), bb, nullShadow);

    auto rule = [&bb, &arg, this](Value *ip) -> llvm::Value * {
      if (ip == getNewFromOriginal(arg->getOperand(0)))
        return getNewFromOriginal(arg);
      return bb.CreateExtractValue(ip, arg->getIndices(),
                                   arg->getName() + "'ipev");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto ip0 = invertPointerM(arg->getOperand(0), bb, nullShadow);
    auto ip1 = invertPointerM(arg->getOperand(1), bb, nullShadow);

    auto rule = [&bb, &arg](Value *ip0, Value *ip1) {
      return bb.CreateInsertValue(ip0, ip1, arg->getIndices(),
                                  arg->getName() + "'ipiv");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ExtractElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto ip = invertPointerM(arg->getVectorOperand(), bb, nullShadow);

    auto rule = [&](Value *ip) {
      return bb.CreateExtractElement(ip,
                                     getNewFromOriginal(arg->getIndexOperand()),
                                     arg->getName() + "'ipee");
      ;
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<InsertElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    Value *op2 = arg->getOperand(2);
    auto ip0 = invertPointerM(op0, bb, nullShadow);
    auto ip1 = invertPointerM(op1, bb, nullShadow);

    auto rule = [&](Value *ip0, Value *ip1) {
      return bb.CreateInsertElement(ip0, ip1, getNewFromOriginal(op2),
                                    arg->getName() + "'ipie");
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<ShuffleVectorInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    auto ip0 = invertPointerM(op0, bb, nullShadow);
    auto ip1 = invertPointerM(op1, bb, nullShadow);

    auto rule = [&bb, &arg](Value *ip0, Value *ip1) {
#if LLVM_VERSION_MAJOR >= 11
      return bb.CreateShuffleVector(ip0, ip1, arg->getShuffleMaskForBitcode(),
                                    arg->getName() + "'ipsv");
#else
      return bb.CreateShuffleVector(ip0, ip1, arg->getOperand(2),
                                    arg->getName() + "'ipsv");
#endif
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip0, ip1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<SelectInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    bb.setFastMathFlags(getFast());

    Value *itval = nullptr;
    {
      auto tval = arg->getTrueValue();
      if (!EnzymeRuntimeActivityCheck && CustomErrorHandler &&
          TR.query(arg)[{-1}].isPossiblePointer() && !isa<UndefValue>(tval) &&
          !isa<ConstantPointerNull>(tval) && isConstantValue(tval)) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Mismatched activity for: " << *arg << " const val: " << *tval;
        itval = unwrap(CustomErrorHandler(str.c_str(), wrap(arg),
                                          ErrorType::MixedActivityError, this,
                                          wrap(tval), wrap(&bb)));
      }
      if (!itval) {
        itval = invertPointerM(tval, bb, nullShadow);
      }
    }
    Value *ifval = nullptr;
    {
      auto fval = arg->getFalseValue();
      if (!EnzymeRuntimeActivityCheck && CustomErrorHandler &&
          TR.query(arg)[{-1}].isPossiblePointer() && !isa<UndefValue>(fval) &&
          !isa<ConstantPointerNull>(fval) && isConstantValue(fval)) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Mismatched activity for: " << *arg << " const val: " << *fval;
        ifval = unwrap(CustomErrorHandler(str.c_str(), wrap(arg),
                                          ErrorType::MixedActivityError, this,
                                          wrap(fval), wrap(&bb)));
      }
      if (!ifval) {
        ifval = invertPointerM(fval, bb, nullShadow);
      }
    }

    Value *shadow = applyChainRule(
        arg->getType(), bb,
        [&](Value *tv, Value *fv) {
          return bb.CreateSelect(getNewFromOriginal(arg->getCondition()), tv,
                                 fv, arg->getName() + "'ipse");
        },
        itval, ifval);
    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto arg = dyn_cast<LoadInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *ip = invertPointerM(op0, bb);

    SmallVector<Metadata *, 1> prevScopes;
    if (auto prev = arg->getMetadata(LLVMContext::MD_alias_scope)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        prevScopes.push_back(M);
      }
    }
    SmallVector<Metadata *, 1> prevNoAlias;
    if (auto prev = arg->getMetadata(LLVMContext::MD_noalias)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        prevNoAlias.push_back(M);
      }
    }
    size_t idx = 0;
    auto rule = [&](Value *ip) {
      auto li = bb.CreateLoad(arg->getType(), ip, arg->getName() + "'ipl");
      llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
      li->copyMetadata(*arg, ToCopy2);
      li->copyIRFlags(arg);

      SmallVector<Metadata *, 1> scopeMD = {getDerivativeAliasScope(op0, idx)};
      for (auto M : prevScopes)
        scopeMD.push_back(M);
      auto scope = MDNode::get(li->getContext(), scopeMD);
      li->setMetadata(LLVMContext::MD_alias_scope, scope);

      SmallVector<Metadata *, 1> MDs;
      for (ssize_t j = -1; j < getWidth(); j++) {
        if (j != (ssize_t)idx)
          MDs.push_back(getDerivativeAliasScope(op0, j));
      }
      for (auto M : prevNoAlias)
        MDs.push_back(M);
      if (MDs.size()) {
        auto noscope = MDNode::get(li->getContext(), MDs);
        li->setMetadata(LLVMContext::MD_noalias, noscope);
      }

#if LLVM_VERSION_MAJOR >= 10
      li->setAlignment(arg->getAlign());
#else
      li->setAlignment(arg->getAlignment());
#endif
      li->setDebugLoc(getNewFromOriginal(arg->getDebugLoc()));
      li->setVolatile(arg->isVolatile());
      li->setOrdering(arg->getOrdering());
      li->setSyncScopeID(arg->getSyncScopeID());
      idx++;
      return li;
    };

    Value *li = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, li)));
    return li;

  } else if (auto arg = dyn_cast<BinaryOperator>(oval)) {
    if (arg->getOpcode() == Instruction::FAdd)
      return getNewFromOriginal(arg);

    if (!arg->getType()->isIntOrIntVectorTy()) {
      llvm::errs() << *oval << "\n";
    }
    assert(arg->getType()->isIntOrIntVectorTy());
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *val0 = nullptr;
    Value *val1 = nullptr;

    val0 = invertPointerM(arg->getOperand(0), bb);
    val1 = invertPointerM(arg->getOperand(1), bb);
    assert(val0->getType() == val1->getType());

    auto rule = [&bb, &arg](Value *val0, Value *val1) {
      auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
      if (auto BI = dyn_cast<BinaryOperator>(li))
        BI->copyIRFlags(arg);
      return li;
    };

    Value *li = applyChainRule(arg->getType(), bb, rule, val0, val1);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, li)));
    return li;
  } else if (auto arg = dyn_cast<GetElementPtrInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    SmallVector<Value *, 4> invertargs;
    for (unsigned i = 0; i < arg->getNumIndices(); ++i) {
      Value *b = getNewFromOriginal(arg->getOperand(1 + i));
      invertargs.push_back(b);
    }
    Value *ip = invertPointerM(arg->getPointerOperand(), bb);

    auto rule = [&](Value *ip) {
      auto shadow = bb.CreateGEP(arg->getSourceElementType(), ip, invertargs,
                                 arg->getName() + "'ipg");

      if (auto gep = dyn_cast<GetElementPtrInst>(shadow))
        gep->setIsInBounds(arg->isInBounds());

      return shadow;
    };

    Value *shadow = applyChainRule(arg->getType(), bb, rule, ip);

    invertedPointers.insert(
        std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
    return shadow;
  } else if (auto inst = dyn_cast<AllocaInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(inst));
    Value *asize = getNewFromOriginal(inst->getArraySize());

    auto rule1 = [&]() {
      AllocaInst *antialloca = bb.CreateAlloca(
          inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(),
          asize, inst->getName() + "'ipa");
#if LLVM_VERSION_MAJOR >= 11
      antialloca->setAlignment(inst->getAlign());
#elif LLVM_VERSION_MAJOR == 10
      if (inst->getAlignment()) {
        antialloca->setAlignment(Align(inst->getAlignment()));
      }
#else
      if (inst->getAlignment()) {
        antialloca->setAlignment(inst->getAlignment());
      }
#endif
      return antialloca;
    };

    Value *antialloca = applyChainRule(oval->getType(), bb, rule1);

    invertedPointers.insert(std::make_pair(
        (const Value *)oval, InvertedPointerVH(this, antialloca)));

    if (auto ci = dyn_cast<ConstantInt>(asize)) {
      if (ci->isOne()) {

        auto rule = [&](Value *antialloca) {
          StoreInst *st = bb.CreateStore(
              Constant::getNullValue(inst->getAllocatedType()), antialloca);
#if LLVM_VERSION_MAJOR >= 11
          cast<StoreInst>(st)->setAlignment(inst->getAlign());
#elif LLVM_VERSION_MAJOR == 10
          if (inst->getAlignment()) {
            cast<StoreInst>(st)->setAlignment(Align(inst->getAlignment()));
          }
#else
          if (inst->getAlignment()) {
            cast<StoreInst>(st)->setAlignment(inst->getAlignment());
          }
#endif
        };

        applyChainRule(bb, rule, antialloca);

        return antialloca;
      } else {
        // TODO handle alloca of size > 1
      }
    }

    auto rule2 = [&](Value *antialloca) {
      auto dst_arg =
          bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
      auto len_arg = bb.CreateMul(
          bb.CreateZExtOrTrunc(asize, Type::getInt64Ty(oval->getContext())),
          ConstantInt::get(Type::getInt64Ty(oval->getContext()),
                           M->getDataLayout().getTypeAllocSizeInBits(
                               inst->getAllocatedType()) /
                               8),
          "", true, true);
      auto volatile_arg = ConstantInt::getFalse(oval->getContext());

      Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
      Type *tys[] = {dst_arg->getType(), len_arg->getType()};
      auto memset = cast<CallInst>(bb.CreateCall(
          Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
#if LLVM_VERSION_MAJOR >= 11
      memset->addParamAttr(
          0, Attribute::getWithAlignment(inst->getContext(), inst->getAlign()));
#elif LLVM_VERSION_MAJOR == 10
      if (inst->getAlignment() != 0) {
        memset->addParamAttr(
            0, Attribute::getWithAlignment(inst->getContext(),
                                           Align(inst->getAlignment())));
      }
#else
      if (inst->getAlignment() != 0) {
        memset->addParamAttr(0, Attribute::getWithAlignment(
                                    inst->getContext(), inst->getAlignment()));
      }
#endif
      memset->addParamAttr(0, Attribute::NonNull);
    };

    applyChainRule(bb, rule2, antialloca);

    return antialloca;
  } else if (auto II = dyn_cast<IntrinsicInst>(oval)) {
    if (isIntelSubscriptIntrinsic(*II)) {
      IRBuilder<> bb(getNewFromOriginal(II));

      const std::array<size_t, 4> idxArgsIndices{{0, 1, 2, 4}};
      const size_t ptrArgIndex = 3;

      SmallVector<Value *, 5> invertArgs(5);
      for (auto i : idxArgsIndices) {
        Value *idx = getNewFromOriginal(II->getOperand(i));
        invertArgs[i] = idx;
      }
      Value *invertPtrArg = invertPointerM(II->getOperand(ptrArgIndex), bb);
      invertArgs[ptrArgIndex] = invertPtrArg;

      auto rule = [&](Value *ip) {
        auto shadow = bb.CreateCall(II->getCalledFunction(), invertArgs);
        assert(isa<CallInst>(shadow));
#if LLVM_VERSION_MAJOR >= 13
        auto CI = cast<CallInst>(shadow);
        // Must copy the elementtype attribute as it is needed by the intrinsic
        CI->addParamAttr(
            ptrArgIndex,
            II->getParamAttr(ptrArgIndex, Attribute::AttrKind::ElementType));
#endif
        return shadow;
      };

      Value *shadow = applyChainRule(II->getType(), bb, rule, invertPtrArg);

      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, shadow)));
      return shadow;
    }

    IRBuilder<> bb(getNewFromOriginal(II));
    bb.setFastMathFlags(getFast());
    switch (II->getIntrinsicID()) {
    default:
      goto end;
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f: {
      return applyChainRule(
          II->getType(), bb,
          [&](Value *ptr) {
            Value *args[] = {ptr};
            auto li = bb.CreateCall(II->getCalledFunction(), args);
            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            li->copyMetadata(*II, ToCopy2);
            li->setDebugLoc(getNewFromOriginal(II->getDebugLoc()));
            return li;
          },
          invertPointerM(II->getArgOperand(0), bb));
    case Intrinsic::masked_load:
      return applyChainRule(
          II->getType(), bb,
          [&](Value *ptr, Value *defaultV) {
            Value *args[] = {ptr, getNewFromOriginal(II->getArgOperand(1)),
                             getNewFromOriginal(II->getArgOperand(2)),
                             defaultV};
            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            auto li = bb.CreateCall(II->getCalledFunction(), args);
            li->copyMetadata(*II, ToCopy2);
            li->setDebugLoc(getNewFromOriginal(II->getDebugLoc()));
            return li;
          },
          invertPointerM(II->getArgOperand(0), bb),
          invertPointerM(II->getArgOperand(3), bb, nullShadow));
    }
    }
  } else if (auto phi = dyn_cast<PHINode>(oval)) {

    if (phi->getNumIncomingValues() == 0) {
      dumpMap(invertedPointers);
      assert(0 && "illegal iv of phi");
    }
    std::map<Value *, std::set<BasicBlock *>> mapped;
    for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
      mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
    }

    if (false && mapped.size() == 1) {
      return invertPointerM(phi->getIncomingValue(0), BuilderM, nullShadow);
    }
#if 0
     else if (false && mapped.size() == 2) {
         IRBuilder <> bb(phi);
         auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
         //TODO this is not recursive

         int cnt = 0;
         Value* vals[2];
         for(auto v : mapped) {
            assert( cnt <= 1 );
            vals[cnt] = v.first;
            for (auto b : v.second) {
                which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
            }
            ++cnt;
         }
         auto result = BuilderM.CreateSelect(which, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
         return result;
     }
#endif

    else {
      auto NewV = getNewFromOriginal(phi);
      IRBuilder<> bb(NewV);
      bb.setFastMathFlags(getFast());
      // Note if the original phi node get's scev'd in NewF, it may
      // no longer be a phi and we need a new place to insert this phi
      // Note that if scev'd this can still be a phi with 0 incoming indicating
      // an unnecessary value to be replaced
      // TODO consider allowing the inverted pointer to become a scev
      if (!isa<PHINode>(NewV) ||
          cast<PHINode>(NewV)->getNumIncomingValues() == 0) {
        bb.SetInsertPoint(bb.GetInsertBlock(), bb.GetInsertBlock()->begin());
      }

      if (EnzymeVectorSplitPhi && width > 1) {
        IRBuilder<> postPhi(NewV->getParent()->getFirstNonPHI());
        Type *shadowTy = getShadowType(phi->getType());
        PHINode *tmp = bb.CreatePHI(shadowTy, phi->getNumIncomingValues());

        invertedPointers.insert(
            std::make_pair((const Value *)oval, InvertedPointerVH(this, tmp)));

        Type *wrappedType = ArrayType::get(phi->getType(), width);
        Value *res = UndefValue::get(wrappedType);

        SmallVector<Value *, 1> invertedVals;
        for (unsigned int j = 0; j < phi->getNumIncomingValues(); ++j) {
          IRBuilder<> pre(
              cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(j)))
                  ->getTerminator());
          Value *preval = phi->getIncomingValue(j);

          Value *val = nullptr;
          if (!EnzymeRuntimeActivityCheck && CustomErrorHandler &&
              TR.query(phi)[{-1}].isPossiblePointer() &&
              !isa<UndefValue>(preval) && !isa<ConstantPointerNull>(preval) &&
              isConstantValue(preval)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << *phi
               << " const val: " << *preval;
            val = unwrap(CustomErrorHandler(str.c_str(), wrap(phi),
                                            ErrorType::MixedActivityError, this,
                                            wrap(preval), wrap(&pre)));
          }
          if (!val) {
            val = invertPointerM(preval, pre, nullShadow);
          }
          invertedVals.push_back(val);
        }

        for (unsigned int i = 0; i < getWidth(); ++i) {
          PHINode *which =
              bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
          which->setDebugLoc(getNewFromOriginal(phi->getDebugLoc()));

          for (unsigned int j = 0; j < phi->getNumIncomingValues(); ++j) {
            IRBuilder<> pre(
                cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(j)))
                    ->getTerminator());
            Value *val = invertedVals[j];
            auto extracted_diff = extractMeta(pre, val, i);
            which->addIncoming(
                extracted_diff,
                cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(j))));
          }

          res = postPhi.CreateInsertValue(res, which, {i});
        }
        invertedPointers.erase((const Value *)oval);
        replaceAWithB(tmp, res);
        erase(tmp);

        invertedPointers.insert(
            std::make_pair((const Value *)oval, InvertedPointerVH(this, res)));

        return res;
      } else {
        Type *shadowTy = getShadowType(phi->getType());
        PHINode *which = bb.CreatePHI(shadowTy, phi->getNumIncomingValues());
        which->setDebugLoc(getNewFromOriginal(phi->getDebugLoc()));

        invertedPointers.insert(std::make_pair((const Value *)oval,
                                               InvertedPointerVH(this, which)));

        for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
          IRBuilder<> pre(
              cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i)))
                  ->getTerminator());

          Value *preval = phi->getIncomingValue(i);

          Value *val = nullptr;
          if (!EnzymeRuntimeActivityCheck && CustomErrorHandler &&
              TR.query(phi)[{-1}].isPossiblePointer() &&
              !isa<UndefValue>(preval) && !isa<ConstantPointerNull>(preval) &&
              isConstantValue(preval)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << *phi
               << " const val: " << *preval;
            val = unwrap(CustomErrorHandler(str.c_str(), wrap(phi),
                                            ErrorType::MixedActivityError, this,
                                            wrap(preval), wrap(&pre)));
          }
          if (!val) {
            val = invertPointerM(preval, pre, nullShadow);
          }

          which->addIncoming(val, cast<BasicBlock>(getNewFromOriginal(
                                      phi->getIncomingBlock(i))));
        }
        return which;
      }
    }
  }

end:;
  assert(BuilderM.GetInsertBlock());
  assert(BuilderM.GetInsertBlock()->getParent());
  assert(oval);

  if (CustomErrorHandler) {
    std::string str;
    raw_string_ostream ss(str);
    ss << "cannot find shadow for " << *oval;
    auto iv =
        unwrap(CustomErrorHandler(str.c_str(), wrap(oval), ErrorType::NoShadow,
                                  this, nullptr, wrap(&BuilderM)));
    if (iv) {
      invertedPointers.insert(
          std::make_pair((const Value *)oval, InvertedPointerVH(this, iv)));
      return iv;
    }
  }

  llvm::errs() << *newFunc->getParent() << "\n";
  llvm::errs() << "fn:" << *newFunc << "\noval=" << *oval
               << " icv=" << isConstantValue(oval) << "\n";
  for (auto z : invertedPointers) {
    llvm::errs() << "available inversion for " << *z.first << " of "
                 << *z.second << "\n";
  }
  assert(0 && "cannot find deal with ptr that isnt arg");
  report_fatal_error("cannot find deal with ptr that isnt arg");
}

Value *GradientUtils::lookupM(Value *val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &incoming_available,
                              bool tryLegalRecomputeCheck, BasicBlock *scope) {

  assert(mode == DerivativeMode::ReverseModePrimal ||
         mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined);

  assert(val->getName() != "<badref>");
  if (isa<Constant>(val)) {
    return val;
  }
  if (isa<BasicBlock>(val)) {
    return val;
  }
  if (isa<Function>(val)) {
    return val;
  }
  if (isa<UndefValue>(val)) {
    return val;
  }
  if (isa<Argument>(val)) {
    return val;
  }
  if (isa<MetadataAsValue>(val)) {
    return val;
  }
  if (isa<InlineAsm>(val)) {
    return val;
  }

  if (!isa<Instruction>(val)) {
    llvm::errs() << *val << "\n";
  }

  auto inst = cast<Instruction>(val);
  assert(inst->getName() != "<badref>");
  if (inversionAllocs && inst->getParent() == inversionAllocs) {
    return val;
  }
  assert(inst->getParent()->getParent() == newFunc);
  assert(BuilderM.GetInsertBlock()->getParent() == newFunc);
  if (scope == nullptr)
    scope = BuilderM.GetInsertBlock();
  assert(scope->getParent() == newFunc);

  bool reduceRegister = false;

  if (EnzymeRegisterReduce) {
    if (auto II = dyn_cast<IntrinsicInst>(inst)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::nvvm_ldu_global_i:
      case Intrinsic::nvvm_ldu_global_p:
      case Intrinsic::nvvm_ldu_global_f:
      case Intrinsic::nvvm_ldg_global_i:
      case Intrinsic::nvvm_ldg_global_p:
      case Intrinsic::nvvm_ldg_global_f:
        reduceRegister = true;
        break;
      default:
        break;
      }
    }
    if (auto LI = dyn_cast<LoadInst>(inst)) {
      auto Arch =
          llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
      unsigned int SharedAddrSpace =
          Arch == Triple::amdgcn
              ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
              : 3;
      if (cast<PointerType>(LI->getPointerOperand()->getType())
              ->getAddressSpace() == SharedAddrSpace) {
        reduceRegister |= tryLegalRecomputeCheck &&
                          legalRecompute(LI, incoming_available, &BuilderM) &&
                          shouldRecompute(LI, incoming_available, &BuilderM);
      }
    }
    if (!inst->mayReadOrWriteMemory()) {
      reduceRegister |= tryLegalRecomputeCheck &&
                        legalRecompute(inst, incoming_available, &BuilderM) &&
                        shouldRecompute(inst, incoming_available, &BuilderM);
    }
    if (this->isOriginalBlock(*BuilderM.GetInsertBlock()))
      reduceRegister = false;
  }

  if (!reduceRegister) {
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        Instruction *use = &*BuilderM.GetInsertPoint();
        while (isa<PHINode>(use))
          use = use->getNextNode();
        if (DT.dominates(inst, use)) {
          return inst;
        } else {
          llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
          llvm::errs() << "didn't dominate inst: " << *inst
                       << "  point: " << *BuilderM.GetInsertPoint()
                       << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
        }
      } else {
        if (inst->getParent() == BuilderM.GetInsertBlock() ||
            DT.dominates(inst, BuilderM.GetInsertBlock())) {
          // allowed from block domination
          return inst;
        } else {
          llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
          llvm::errs() << "didn't dominate inst: " << *inst
                       << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
        }
      }
      // This is a reverse block
    } else if (BuilderM.GetInsertBlock() != inversionAllocs) {
      // Something in the entry (or anything that dominates all returns, doesn't
      // need caching)
      BasicBlock *orig = isOriginal(inst->getParent());
      if (!orig) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "insertBlock: " << *BuilderM.GetInsertBlock() << "\n";
        llvm::errs() << "instP: " << *inst->getParent() << "\n";
        llvm::errs() << "inst: " << *inst << "\n";
      }
      assert(orig);

      // TODO upgrade this to be all returns that this could enter from
      bool legal = BlocksDominatingAllReturns.count(orig);
      if (legal) {

        BasicBlock *forwardBlock =
            isOriginal(originalForReverseBlock(*BuilderM.GetInsertBlock()));
        assert(forwardBlock);

        // Don't allow this if we're not definitely using the last iteration of
        // this value
        //   + either because the value isn't in a loop
        //   + or because the forward of the block usage location isn't in a
        //   loop (thus last iteration)
        //   + or because the loop nests share no ancestry

        bool loopLegal = true;
        for (Loop *idx = OrigLI.getLoopFor(orig); idx != nullptr;
             idx = idx->getParentLoop()) {
          for (Loop *fdx = OrigLI.getLoopFor(forwardBlock); fdx != nullptr;
               fdx = fdx->getParentLoop()) {
            if (idx == fdx) {
              loopLegal = false;
              break;
            }
          }
        }

        if (loopLegal) {
          return inst;
        }
      }
    }
  }

  if (lookup_cache[BuilderM.GetInsertBlock()].find(val) !=
      lookup_cache[BuilderM.GetInsertBlock()].end()) {
    auto result = lookup_cache[BuilderM.GetInsertBlock()][val];
    if (result == nullptr) {
      lookup_cache[BuilderM.GetInsertBlock()].erase(val);
    } else {
      assert(result);
      assert(result->getType());
      result = BuilderM.CreateBitCast(result, val->getType());
      assert(result->getType() == inst->getType());
      return result;
    }
  }

  ValueToValueMapTy available;
  for (auto pair : incoming_available) {
    if (pair.second)
      assert(pair.first->getType() == pair.second->getType());
    available[pair.first] = pair.second;
  }

  {
    BasicBlock *forwardPass = BuilderM.GetInsertBlock();
    if (forwardPass != inversionAllocs && !isOriginalBlock(*forwardPass)) {
      forwardPass = originalForReverseBlock(*forwardPass);
    }
    LoopContext lc;
    bool inLoop = getContext(forwardPass, lc);

    if (inLoop) {
      bool first = true;
      for (LoopContext idx = lc;; getContext(idx.parent->getHeader(), idx)) {
        if (available.count(idx.var) == 0) {
          if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
            available[idx.var] =
                BuilderM.CreateLoad(idx.var->getType(), idx.antivaralloc);
          } else {
            available[idx.var] = idx.var;
          }
        }
        if (!first && idx.var == inst)
          return available[idx.var];
        if (first) {
          first = false;
        }
        if (idx.parent == nullptr)
          break;
      }
    }
  }

  if (available.count(inst)) {
    assert(available[inst]->getType() == inst->getType());
    return available[inst];
  }

  // If requesting loop bound and not available from index per above
  // we must be requesting the total size. Rather than generating
  // a new lcssa variable, use the existing loop exact bound var
  {
    LoopContext lc;
    bool loopVar = false;
    if (getContext(inst->getParent(), lc) && lc.var == inst) {
      loopVar = true;
    } else if (auto phi = dyn_cast<PHINode>(inst)) {
      Value *V = nullptr;
      bool legal = true;
      for (auto &val : phi->incoming_values()) {
        if (isa<UndefValue>(val))
          continue;
        if (V == nullptr)
          V = val;
        else if (V != val) {
          legal = false;
          break;
        }
      }
      if (legal) {
        if (auto I = dyn_cast_or_null<PHINode>(V)) {
          if (getContext(I->getParent(), lc) && lc.var == I) {
            loopVar = true;
          }
        }
      }
    }
    if (loopVar) {
      Value *lim = nullptr;
      if (lc.dynamic) {
        // Must be in a reverse pass fashion for a lookup to index bound to be
        // legal
        assert(/*ReverseLimit*/ reverseBlocks.size() > 0);
        LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                          lc.preheader);
        lim = lookupValueFromCache(
            lc.var->getType(), /*forwardPass*/ false, BuilderM, lctx,
            getDynamicLoopLimit(LI.getLoopFor(lc.header)),
            /*isi1*/ false, available);
      } else {
        lim = lookupM(lc.trueLimit, BuilderM);
      }
      lookup_cache[BuilderM.GetInsertBlock()][val] = lim;
      return lim;
    }
  }

  Instruction *prelcssaInst = inst;

  assert(inst->getName() != "<badref>");
  val = fixLCSSA(inst, scope);
  if (isa<UndefValue>(val)) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
    llvm::errs() << *scope << "\n";
    llvm::errs() << *val << " inst " << *inst << "\n";
    assert(0 && "undef value upon lcssa");
  }
  inst = cast<Instruction>(val);
  assert(prelcssaInst->getType() == inst->getType());
  assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

  // Update index and caching per lcssa
  if (lookup_cache[BuilderM.GetInsertBlock()].find(val) !=
      lookup_cache[BuilderM.GetInsertBlock()].end()) {
    auto result = lookup_cache[BuilderM.GetInsertBlock()][val];
    if (result == nullptr) {
      lookup_cache[BuilderM.GetInsertBlock()].erase(val);
    } else {
      assert(result);
      assert(result->getType());
      result = BuilderM.CreateBitCast(result, val->getType());
      assert(result->getType() == inst->getType());
      return result;
    }
  }

  // TODO consider call as part of
  bool lrc = false, src = false;
  if (tryLegalRecomputeCheck &&
      (lrc = legalRecompute(prelcssaInst, available, &BuilderM))) {
    if ((src = shouldRecompute(prelcssaInst, available, &BuilderM))) {
      auto op = unwrapM(prelcssaInst, BuilderM, available,
                        UnwrapMode::AttemptSingleUnwrap, scope);
      if (op) {
        assert(op);
        assert(op->getType());
        if (op->getType() != inst->getType()) {
          llvm::errs() << " op: " << *op << " inst: " << *inst << "\n";
        }
        assert(op->getType() == inst->getType());
        if (!reduceRegister)
          lookup_cache[BuilderM.GetInsertBlock()][val] = op;
        return op;
      }
    } else {
      if (isa<LoadInst>(prelcssaInst)) {
      }
    }
  }

  if (auto li = dyn_cast<LoadInst>(inst))
    if (auto origInst = dyn_cast_or_null<LoadInst>(isOriginal(inst))) {
      auto liobj = getBaseObject(li->getPointerOperand());

      auto orig_liobj = getBaseObject(origInst->getPointerOperand());

      if (scopeMap.find(inst) == scopeMap.end()) {
        for (auto pair : scopeMap) {
          if (auto li2 = dyn_cast<LoadInst>(const_cast<Value *>(pair.first))) {

            auto li2obj = getBaseObject(li2->getPointerOperand());

            if (liobj == li2obj && DT.dominates(li2, li)) {
              auto orig2 = isOriginal(li2);
              if (!orig2)
                continue;

              bool failed = false;

              // llvm::errs() << "found potential candidate loads: oli:"
              //             << *origInst << " oli2: " << *orig2 << "\n";

              auto scev1 = SE.getSCEV(li->getPointerOperand());
              auto scev2 = SE.getSCEV(li2->getPointerOperand());
              // llvm::errs() << " scev1: " << *scev1 << " scev2: " << *scev2
              //             << "\n";

              allInstructionsBetween(
                  OrigLI, orig2, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, TLI,
                                             /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      // llvm::errs() << "FAILED: " << *I << "\n";
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                continue;

              if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
                if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                  if (ar1->getStart() != SE.getCouldNotCompute() &&
                      ar1->getStart() == ar2->getStart() &&
                      ar1->getStepRecurrence(SE) != SE.getCouldNotCompute() &&
                      ar1->getStepRecurrence(SE) ==
                          ar2->getStepRecurrence(SE)) {

                    LoopContext l1;
                    getContext(ar1->getLoop()->getHeader(), l1);
                    LoopContext l2;
                    getContext(ar2->getLoop()->getHeader(), l2);
                    if (l1.dynamic || l2.dynamic)
                      continue;

                    // TODO IF len(ar2) >= len(ar1) then we can replace li with
                    // li2
                    if (SE.getSCEV(l1.trueLimit) != SE.getCouldNotCompute() &&
                        SE.getSCEV(l1.trueLimit) == SE.getSCEV(l2.trueLimit)) {
                      // llvm::errs()
                      //    << " step1: " << *ar1->getStepRecurrence(SE)
                      //    << " step2: " << *ar2->getStepRecurrence(SE) <<
                      //    "\n";

                      inst = li2;
                      break;
                    }
                  }
                }
              }
            }
          }
        }

        auto scev1 = OrigSE.getSCEV(origInst->getPointerOperand());

        auto Arch =
            llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();
        unsigned int SharedAddrSpace =
            Arch == Triple::amdgcn
                ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                : 3;
        if (EnzymeSharedForward && scev1 != OrigSE.getCouldNotCompute() &&
            cast<PointerType>(orig_liobj->getType())->getAddressSpace() ==
                SharedAddrSpace) {
          Value *resultValue = nullptr;
          ValueToValueMapTy newavail;
          for (const auto &pair : available) {
            assert(pair.first->getType() == pair.second->getType());
            newavail[pair.first] = pair.second;
          }
          allDomPredecessorsOf(origInst, OrigDT, [&](Instruction *pred) {
            if (auto SI = dyn_cast<StoreInst>(pred)) {
              // auto NewSI = cast<StoreInst>(getNewFromOriginal(SI));
              auto si2obj = getBaseObject(SI->getPointerOperand());

              if (si2obj != orig_liobj)
                return false;

              bool lastStore = true;
              bool interveningSync = false;
              allInstructionsBetween(
                  OrigLI, SI, origInst, [&](Instruction *potentialAlias) {
                    if (!potentialAlias->mayWriteToMemory())
                      return false;
                    if (!writesToMemoryReadBy(OrigAA, TLI, origInst,
                                              potentialAlias))
                      return false;

                    if (auto II = dyn_cast<IntrinsicInst>(potentialAlias)) {
                      if (II->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
                          II->getIntrinsicID() == Intrinsic::amdgcn_s_barrier) {
                        interveningSync =
                            DT.dominates(SI, II) && DT.dominates(II, origInst);
                        allUnsyncdPredecessorsOf(
                            II,
                            [&](Instruction *mid) {
                              if (!mid->mayWriteToMemory())
                                return false;

                              if (mid == SI)
                                return false;

                              if (!writesToMemoryReadBy(OrigAA, TLI, origInst,
                                                        mid)) {
                                return false;
                              }
                              lastStore = false;
                              return true;
                            },
                            [&]() {
                              // if gone past entry
                              if (mode != DerivativeMode::ReverseModeCombined) {
                                lastStore = false;
                              }
                            });
                        if (!lastStore)
                          return true;
                        else
                          return false;
                      }
                    }

                    lastStore = false;
                    return true;
                  });

              if (!lastStore)
                return false;

              auto scev2 = OrigSE.getSCEV(SI->getPointerOperand());
              bool legal = scev1 == scev2;
              if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
                  if (ar2->getStart() != OrigSE.getCouldNotCompute() &&
                      ar1->getStart() == ar2->getStart() &&
                      ar2->getStepRecurrence(OrigSE) !=
                          OrigSE.getCouldNotCompute() &&
                      ar1->getStepRecurrence(OrigSE) ==
                          ar2->getStepRecurrence(OrigSE)) {

                    LoopContext l1;
                    getContext(getNewFromOriginal(ar1->getLoop()->getHeader()),
                               l1);
                    LoopContext l2;
                    getContext(getNewFromOriginal(ar2->getLoop()->getHeader()),
                               l2);
                    if (!l1.dynamic && !l2.dynamic) {
                      // TODO IF len(ar2) >= len(ar1) then we can replace li
                      // with li2
                      if (l1.trueLimit == l2.trueLimit) {
                        const Loop *L1 = ar1->getLoop();
                        while (L1) {
                          if (L1 == ar2->getLoop())
                            return false;
                          L1 = L1->getParentLoop();
                        }
                        newavail[l2.var] = available[l1.var];
                        legal = true;
                      }
                    }
                  }
                }
              }
              if (!legal) {
                Value *sval = SI->getPointerOperand();
                Value *lval = origInst->getPointerOperand();
                while (auto CI = dyn_cast<CastInst>(sval))
                  sval = CI->getOperand(0);
                while (auto CI = dyn_cast<CastInst>(lval))
                  lval = CI->getOperand(0);
                if (auto sgep = dyn_cast<GetElementPtrInst>(sval)) {
                  if (auto lgep = dyn_cast<GetElementPtrInst>(lval)) {
                    if (sgep->getPointerOperand() ==
                        lgep->getPointerOperand()) {
                      SmallVector<Value *, 3> svals;
                      for (auto &v : sgep->indices()) {
                        Value *q = v;
                        while (auto CI = dyn_cast<CastInst>(q))
                          q = CI->getOperand(0);
                        svals.push_back(q);
                      }
                      SmallVector<Value *, 3> lvals;
                      for (auto &v : lgep->indices()) {
                        Value *q = v;
                        while (auto CI = dyn_cast<CastInst>(q))
                          q = CI->getOperand(0);
                        lvals.push_back(q);
                      }
                      ValueToValueMapTy ThreadLookup;
                      bool legal = true;
                      for (size_t i = 0; i < svals.size(); i++) {
                        auto ss = OrigSE.getSCEV(svals[i]);
                        auto ls = OrigSE.getSCEV(lvals[i]);
                        if (cast<IntegerType>(ss->getType())->getBitWidth() >
                            cast<IntegerType>(ls->getType())->getBitWidth()) {
                          ls = OrigSE.getZeroExtendExpr(ls, ss->getType());
                        }
                        if (cast<IntegerType>(ss->getType())->getBitWidth() <
                            cast<IntegerType>(ls->getType())->getBitWidth()) {
                          ls = OrigSE.getTruncateExpr(ls, ss->getType());
                        }
                        if (ls != ss) {
                          if (auto II = dyn_cast<IntrinsicInst>(svals[i])) {
                            switch (II->getIntrinsicID()) {
                            case Intrinsic::nvvm_read_ptx_sreg_tid_x:
                            case Intrinsic::nvvm_read_ptx_sreg_tid_y:
                            case Intrinsic::nvvm_read_ptx_sreg_tid_z:
                            case Intrinsic::amdgcn_workitem_id_x:
                            case Intrinsic::amdgcn_workitem_id_y:
                            case Intrinsic::amdgcn_workitem_id_z:
                              ThreadLookup[getNewFromOriginal(II)] =
                                  BuilderM.CreateZExtOrTrunc(
                                      lookupM(getNewFromOriginal(lvals[i]),
                                              BuilderM, available),
                                      II->getType());
                              break;
                            default:
                              legal = false;
                              break;
                            }
                          } else {
                            legal = false;
                            break;
                          }
                        }
                      }
                      if (legal) {
                        for (auto pair : newavail) {
                          assert(pair.first->getType() ==
                                 pair.second->getType());
                          ThreadLookup[pair.first] = pair.second;
                        }
                        Value *recomp = unwrapM(
                            getNewFromOriginal(SI->getValueOperand()), BuilderM,
                            ThreadLookup, UnwrapMode::AttemptFullUnwrap, scope,
                            /*permitCache*/ false);
                        if (recomp) {
                          resultValue = recomp;
                          return true;
                          ;
                        }
                      }
                    }
                  }
                }
              }
              if (!legal)
                return false;
              return true;
            }
            return false;
          });

          if (resultValue) {
            if (resultValue->getType() != val->getType())
              resultValue = BuilderM.CreateBitCast(resultValue, val->getType());
            return resultValue;
          }
        }
      }

      auto loadSize = (li->getParent()
                           ->getParent()
                           ->getParent()
                           ->getDataLayout()
                           .getTypeAllocSizeInBits(li->getType()) +
                       7) /
                      8;

      // this is guarded because havent told cacheForReverse how to move
      if (mode == DerivativeMode::ReverseModeCombined)
        if (!li->isVolatile() && EnzymeLoopInvariantCache) {
          if (auto AI = dyn_cast<AllocaInst>(liobj)) {
            assert(isa<AllocaInst>(orig_liobj));
            if (auto AT = dyn_cast<ArrayType>(AI->getAllocatedType()))
              if (auto GEP =
                      dyn_cast<GetElementPtrInst>(li->getPointerOperand())) {
                if (GEP->getPointerOperand() == AI) {
                  LoopContext l1;
                  if (!getContext(li->getParent(), l1))
                    goto noSpeedCache;

                  BasicBlock *ctx = l1.preheader;

                  auto origPH = cast_or_null<BasicBlock>(isOriginal(ctx));
                  assert(origPH);
                  if (OrigPDT.dominates(origPH, origInst->getParent())) {
                    goto noSpeedCache;
                  }

                  Instruction *origTerm = origPH->getTerminator();
                  if (!origTerm)
                    llvm::errs() << *origTerm << "\n";
                  assert(origTerm);
                  IRBuilder<> OB(origTerm);
                  LoadInst *tmpload = OB.CreateLoad(AT, orig_liobj, "'tmpload");

                  bool failed = false;
                  allInstructionsBetween(
                      OrigLI, &*origTerm, origInst,
                      [&](Instruction *I) -> bool {
                        if (I->mayWriteToMemory() &&
                            writesToMemoryReadBy(OrigAA, TLI,
                                                 /*maybeReader*/ tmpload,
                                                 /*maybeWriter*/ I)) {
                          failed = true;
                          return /*earlyBreak*/ true;
                        }
                        return /*earlyBreak*/ false;
                      });
                  if (failed) {
                    tmpload->eraseFromParent();
                    goto noSpeedCache;
                  }
                  while (Loop *L = LI.getLoopFor(ctx)) {
                    BasicBlock *nctx = L->getLoopPreheader();
                    assert(nctx);
                    bool failed = false;
                    auto origPH = cast_or_null<BasicBlock>(isOriginal(nctx));
                    assert(origPH);
                    if (OrigPDT.dominates(origPH, origInst->getParent())) {
                      break;
                    }
                    Instruction *origTerm = origPH->getTerminator();
                    allInstructionsBetween(
                        OrigLI, &*origTerm, origInst,
                        [&](Instruction *I) -> bool {
                          if (I->mayWriteToMemory() &&
                              writesToMemoryReadBy(OrigAA, TLI,
                                                   /*maybeReader*/ tmpload,
                                                   /*maybeWriter*/ I)) {
                            failed = true;
                            return /*earlyBreak*/ true;
                          }
                          return /*earlyBreak*/ false;
                        });
                    if (failed)
                      break;
                    ctx = nctx;
                  }

                  tmpload->eraseFromParent();

                  IRBuilder<> v(ctx->getTerminator());

                  AllocaInst *cache = nullptr;

                  LoopContext tmp;
                  bool forceSingleIter = false;
                  if (!getContext(ctx, tmp)) {
                    forceSingleIter = true;
                  }
                  LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                    ctx, forceSingleIter);

                  if (auto found = findInMap(scopeMap, (Value *)liobj)) {
                    cache = found->first;
                  } else {
                    // if freeing reverseblocks must exist
                    assert(reverseBlocks.size());
                    cache = createCacheForScope(lctx, AT, li->getName(),
                                                /*shouldFree*/ true,
                                                /*allocate*/ true);
                    assert(cache);
                    scopeMap.insert(
                        std::make_pair(AI, std::make_pair(cache, lctx)));

                    v.setFastMathFlags(getFast());
                    assert(isOriginalBlock(*v.GetInsertBlock()));
                    Value *outer =
                        getCachePointer(AT,
                                        /*inForwardPass*/ true, v, lctx, cache,
                                        /*storeinstorecache*/ true,
                                        /*available*/ ValueToValueMapTy(),
                                        /*extraSize*/ nullptr);

                    auto ld = v.CreateLoad(AT, AI);
#if LLVM_VERSION_MAJOR >= 11
                    ld->setAlignment(AI->getAlign());
#elif LLVM_VERSION_MAJOR == 10
                    if (AI->getAlignment()) {
                      ld->setAlignment(Align(AI->getAlignment()));
                    }
#else
                    if (AI->getAlignment()) {
                      ld->setAlignment(AI->getAlignment());
                    }
#endif
                    scopeInstructions[cache].push_back(ld);
                    auto st = v.CreateStore(ld, outer);
                    auto bsize = newFunc->getParent()
                                     ->getDataLayout()
                                     .getTypeAllocSizeInBits(AT) /
                                 8;
                    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
                      st->setAlignment(Align(bsize));
#else
                      st->setAlignment(bsize);
#endif
                    }
                    scopeInstructions[cache].push_back(st);
                    for (auto post : PostCacheStore(st, v)) {
                      scopeInstructions[cache].push_back(post);
                    }
                  }

                  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
                  Value *outer = getCachePointer(
                      AT,
                      /*inForwardPass*/ false, BuilderM, lctx, cache,
                      /*storeinstorecache*/ true, available,
                      /*extraSize*/ nullptr);
                  SmallVector<Value *, 2> idxs;
                  for (auto &idx : GEP->indices()) {
                    idxs.push_back(lookupM(idx, BuilderM, available,
                                           tryLegalRecomputeCheck));
                  }

                  auto cptr = BuilderM.CreateGEP(GEP->getSourceElementType(),
                                                 outer, idxs);
                  cast<GetElementPtrInst>(cptr)->setIsInBounds(true);

                  // Retrieve the actual result
                  auto result = loadFromCachePointer(val->getType(), BuilderM,
                                                     cptr, cache);

                  assert(result->getType() == inst->getType());
                  lookup_cache[BuilderM.GetInsertBlock()][val] = result;
                  return result;
                }
              }
          }

          auto scev1 = SE.getSCEV(li->getPointerOperand());
          // Store in memcpy opt
          Value *lim = nullptr;
          BasicBlock *ctx = nullptr;
          Value *start = nullptr;
          Value *offset = nullptr;
          if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
            if (auto step =
                    dyn_cast<SCEVConstant>(ar1->getStepRecurrence(SE))) {
              if (step->getAPInt() != loadSize)
                goto noSpeedCache;

              LoopContext l1;
              getContext(ar1->getLoop()->getHeader(), l1);

              if (l1.dynamic)
                goto noSpeedCache;

              offset = available[l1.var];
              ctx = l1.preheader;

              IRBuilder<> v(ctx->getTerminator());

              auto origPH = cast_or_null<BasicBlock>(isOriginal(ctx));
              assert(origPH);
              if (OrigPDT.dominates(origPH, origInst->getParent())) {
                goto noSpeedCache;
              }

              lim = unwrapM(l1.trueLimit, v,
                            /*available*/ ValueToValueMapTy(),
                            UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!lim) {
                goto noSpeedCache;
              }
              lim = v.CreateAdd(lim, ConstantInt::get(lim->getType(), 1), "",
                                true, true);

              SmallVector<Instruction *, 4> toErase;
              {
#if LLVM_VERSION_MAJOR >= 12
                SCEVExpander Exp(SE,
                                 ctx->getParent()->getParent()->getDataLayout(),
                                 "enzyme");
#else
                fake::SCEVExpander Exp(
                    SE, ctx->getParent()->getParent()->getDataLayout(),
                    "enzyme");
#endif
                Exp.setInsertPoint(l1.header->getTerminator());
                Value *start0 = Exp.expandCodeFor(
                    ar1->getStart(), li->getPointerOperand()->getType());
                start = unwrapM(start0, v,
                                /*available*/ ValueToValueMapTy(),
                                UnwrapMode::AttemptFullUnwrapWithLookup);
                std::set<Value *> todo = {start0};
                while (todo.size()) {
                  Value *now = *todo.begin();
                  todo.erase(now);
                  if (Instruction *inst = dyn_cast<Instruction>(now)) {
                    if (inst != start && inst->getNumUses() == 0 &&
                        Exp.isInsertedInstruction(inst)) {
                      for (auto &op : inst->operands()) {
                        todo.insert(op);
                      }
                      toErase.push_back(inst);
                    }
                  }
                }
              }
              for (auto a : toErase)
                erase(a);

              if (!start)
                goto noSpeedCache;

              Instruction *origTerm = origPH->getTerminator();

              bool failed = false;
              allInstructionsBetween(
                  OrigLI, &*origTerm, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, TLI,
                                             /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                goto noSpeedCache;
            }
          }

          if (ctx && lim && start && offset) {
            Value *firstLim = lim;
            Value *firstStart = start;
            while (Loop *L = LI.getLoopFor(ctx)) {
              BasicBlock *nctx = L->getLoopPreheader();
              assert(nctx);
              bool failed = false;
              auto origPH = cast_or_null<BasicBlock>(isOriginal(nctx));
              assert(origPH);
              if (OrigPDT.dominates(origPH, origInst->getParent())) {
                break;
              }
              Instruction *origTerm = origPH->getTerminator();
              allInstructionsBetween(
                  OrigLI, &*origTerm, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(OrigAA, TLI,
                                             /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                break;
              IRBuilder<> nv(nctx->getTerminator());
              Value *nlim = unwrapM(firstLim, nv,
                                    /*available*/ ValueToValueMapTy(),
                                    UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!nlim)
                break;
              Value *nstart = unwrapM(firstStart, nv,
                                      /*available*/ ValueToValueMapTy(),
                                      UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!nstart)
                break;
              lim = nlim;
              start = nstart;
              ctx = nctx;
            }
            IRBuilder<> v(ctx->getTerminator());
            bool isi1 = val->getType()->isIntegerTy() &&
                        cast<IntegerType>(li->getType())->getBitWidth() == 1;

            AllocaInst *cache = nullptr;

            LoopContext tmp;
            bool forceSingleIter = false;
            if (!getContext(ctx, tmp)) {
              forceSingleIter = true;
            } else if (auto inst = dyn_cast<Instruction>(lim)) {
              if (inst->getParent() == ctx ||
                  !DT.dominates(inst->getParent(), ctx)) {
                forceSingleIter = true;
              }
            }
            LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0, ctx,
                              forceSingleIter);

            if (auto found = findInMap(scopeMap, (Value *)inst)) {
              cache = found->first;
            } else {
              // if freeing reverseblocks must exist
              assert(reverseBlocks.size());
              cache = createCacheForScope(lctx, li->getType(), li->getName(),
                                          /*shouldFree*/ true,
                                          /*allocate*/ true, /*extraSize*/ lim);
              assert(cache);
              scopeMap.insert(
                  std::make_pair(inst, std::make_pair(cache, lctx)));

              v.setFastMathFlags(getFast());
              assert(isOriginalBlock(*v.GetInsertBlock()));
              Value *outer =
                  getCachePointer(li->getType(),
                                  /*inForwardPass*/ true, v, lctx, cache,
                                  /*storeinstorecache*/ true,
                                  /*available*/ ValueToValueMapTy(),
                                  /*extraSize*/ lim);

              auto dst_arg = v.CreateBitCast(
                  outer,
                  Type::getInt8PtrTy(
                      inst->getContext(),
                      cast<PointerType>(outer->getType())->getAddressSpace()));
              scopeInstructions[cache].push_back(cast<Instruction>(dst_arg));
              auto src_arg = v.CreateBitCast(
                  start,
                  Type::getInt8PtrTy(
                      inst->getContext(),
                      cast<PointerType>(start->getType())->getAddressSpace()));
              auto len_arg =
                  v.CreateMul(ConstantInt::get(lim->getType(), loadSize), lim,
                              "", true, true);
              if (Instruction *I = dyn_cast<Instruction>(len_arg))
                scopeInstructions[cache].push_back(I);
              auto volatile_arg = ConstantInt::getFalse(inst->getContext());

              Value *nargs[] = {dst_arg, src_arg, len_arg, volatile_arg};

              Type *tys[] = {dst_arg->getType(), src_arg->getType(),
                             len_arg->getType()};

              auto memcpyF = Intrinsic::getDeclaration(newFunc->getParent(),
                                                       Intrinsic::memcpy, tys);
              auto mem = cast<CallInst>(v.CreateCall(memcpyF, nargs));

              mem->addParamAttr(0, Attribute::NonNull);
              mem->addParamAttr(1, Attribute::NonNull);

              auto bsize =
                  newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
                      li->getType()) /
                  8;
              if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
                mem->addParamAttr(0, Attribute::getWithAlignment(
                                         memcpyF->getContext(), Align(bsize)));
#else
                mem->addParamAttr(0, Attribute::getWithAlignment(
                                         memcpyF->getContext(), bsize));
#endif
              }

#if LLVM_VERSION_MAJOR >= 11
              mem->addParamAttr(1, Attribute::getWithAlignment(
                                       memcpyF->getContext(), li->getAlign()));
#elif LLVM_VERSION_MAJOR >= 10
              if (li->getAlign())
                mem->addParamAttr(
                    1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                   li->getAlign().getValue()));
#else
              if (li->getAlignment())
                mem->addParamAttr(
                    1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                   li->getAlignment()));
#endif

              scopeInstructions[cache].push_back(mem);
            }

            assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
            Value *result = lookupValueFromCache(
                inst->getType(),
                /*isForwardPass*/ false, BuilderM, lctx, cache, isi1, available,
                /*extraSize*/ lim, offset);
            assert(result->getType() == inst->getType());
            lookup_cache[BuilderM.GetInsertBlock()][val] = result;

            EmitWarning("Uncacheable", *inst, "Caching instruction ", *inst,
                        " legalRecompute: ", lrc, " shouldRecompute: ", src,
                        " tryLegalRecomputeCheck: ", tryLegalRecomputeCheck);
            return result;
          }
        }
    noSpeedCache:;
    }

  if (scopeMap.find(inst) == scopeMap.end()) {
    EmitWarning("Uncacheable", *inst, "Caching instruction ", *inst,
                " legalRecompute: ", lrc, " shouldRecompute: ", src,
                " tryLegalRecomputeCheck: ", tryLegalRecomputeCheck);
  }

  BasicBlock *scopeI = inst->getParent();
  if (auto origInst = isOriginal(inst)) {
    auto found = rematerializableAllocations.find(origInst);
    if (found != rematerializableAllocations.end())
      if (found->second.LI && found->second.LI->contains(origInst)) {
        // If not caching whole allocation and rematerializing the allocation
        // within the loop, force an entry-level scope so there is no need
        // to cache.
        if (!needsCacheWholeAllocation(origInst))
          scopeI = &newFunc->getEntryBlock();
      }
  } else {
    for (auto pair : backwardsOnlyShadows) {
      if (auto pinst = dyn_cast<Instruction>(pair.first))
        if (!pair.second.primalInitialize && pair.second.LI &&
            pair.second.LI->contains(pinst->getParent())) {
          auto found = invertedPointers.find(pair.first);
          if (found != invertedPointers.end() && found->second == inst) {
            scopeI = &newFunc->getEntryBlock();

            // Prevent the phi node from being stored into the cache by creating
            // it before the ensureLookupCached.
            if (scopeMap.find(inst) == scopeMap.end()) {
              LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                                scopeI);

              AllocaInst *cache = createCacheForScope(
                  lctx, inst->getType(), inst->getName(), /*shouldFree*/ true);
              assert(cache);
              insert_or_assign(scopeMap, (Value *&)inst,
                               std::pair<AssertingVH<AllocaInst>, LimitContext>(
                                   cache, lctx));
            }
            break;
          }
        }
    }
  }

  ensureLookupCached(inst, /*shouldFree*/ true, scopeI,
                     inst->getMetadata(LLVMContext::MD_tbaa));
  bool isi1 = inst->getType()->isIntegerTy() &&
              cast<IntegerType>(inst->getType())->getBitWidth() == 1;
  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
  auto found = findInMap(scopeMap, (Value *)inst);
  Value *result =
      lookupValueFromCache(inst->getType(), /*isForwardPass*/ false, BuilderM,
                           found->second, found->first, isi1, available);
  if (auto LI2 = dyn_cast<LoadInst>(result))
    if (auto LI1 = dyn_cast<LoadInst>(inst)) {
      llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
      ToCopy2.push_back(LLVMContext::MD_noalias);
      ToCopy2.push_back(LLVMContext::MD_alias_scope);
      LI2->copyMetadata(*LI1, ToCopy2);
    }
  if (result->getType() != inst->getType()) {
    llvm::errs() << "newFunc: " << *newFunc << "\n";
    llvm::errs() << "result: " << *result << "\n";
    llvm::errs() << "inst: " << *inst << "\n";
    llvm::errs() << "val: " << *val << "\n";
  }
  assert(result->getType() == inst->getType());
  lookup_cache[BuilderM.GetInsertBlock()][val] = result;
  assert(result);
  if (result->getType() != val->getType()) {
    result = BuilderM.CreateBitCast(result, val->getType());
  }
  assert(result->getType() == val->getType());
  assert(result->getType());
  return result;
}

BasicBlock *GradientUtils::originalForReverseBlock(BasicBlock &BB2) const {
  auto found = reverseBlockToPrimal.find(&BB2);
  if (found == reverseBlockToPrimal.end()) {
    errs() << "newFunc: " << *newFunc << "\n";
    errs() << BB2 << "\n";
  }
  assert(found != reverseBlockToPrimal.end());
  return found->second;
}

//! Given a map of edges we could have taken to desired target, compute a value
//! that determines which target should be branched to
//  This function attempts to determine an equivalent condition from earlier in
//  the code and use that if possible, falling back to creating a phi node of
//  which edge was taken if necessary This function can be used in two ways:
//   * If replacePHIs is null (usual case), this function does the branch
//   * If replacePHIs isn't null, do not perform the branch and instead replace
//   the PHI's with the derived condition as to whether we should branch to a
//   particular target
void GradientUtils::branchToCorrespondingTarget(
    BasicBlock *ctx, IRBuilder<> &BuilderM,
    const std::map<BasicBlock *,
                   std::vector<std::pair</*pred*/ BasicBlock *,
                                         /*successor*/ BasicBlock *>>>
        &targetToPreds,
    const std::map<BasicBlock *, PHINode *> *replacePHIs) {
  assert(targetToPreds.size() > 0);
  if (replacePHIs) {
    if (replacePHIs->size() == 0)
      return;

    for (auto x : *replacePHIs) {
      assert(targetToPreds.find(x.first) != targetToPreds.end());
    }
  }

  if (targetToPreds.size() == 1) {
    if (replacePHIs == nullptr) {
      if (!(BuilderM.GetInsertBlock()->size() == 0 ||
            !isa<BranchInst>(BuilderM.GetInsertBlock()->back()))) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *BuilderM.GetInsertBlock() << "\n";
      }
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateBr(targetToPreds.begin()->first);
    } else {
      for (auto pair : *replacePHIs) {
        pair.second->replaceAllUsesWith(
            ConstantInt::getTrue(pair.second->getContext()));
        pair.second->eraseFromParent();
      }
    }
    return;
  }

  // Map of function edges to list of targets this can branch to we have
  std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
           std::set<BasicBlock *>>
      done;
  {
    std::deque<
        std::tuple<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
                   BasicBlock *>>
        Q; // newblock, target

    for (auto pair : targetToPreds) {
      for (auto pred_edge : pair.second) {
        Q.push_back(std::make_pair(pred_edge, pair.first));
      }
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

      // If this block dominates the context, don't go back up as any
      // predecessors won't contain the conditions.
      if (DT.dominates(block, ctx))
        continue;

      Loop *blockLoop = LI.getLoopFor(block);

      for (BasicBlock *Pred : predecessors(block)) {
        // Don't go up the backedge as we can use the last value if desired via
        // lcssa
        if (blockLoop && blockLoop->getHeader() == block &&
            blockLoop == LI.getLoopFor(Pred))
          continue;

        Q.push_back(
            std::tuple<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>(
                std::make_pair(Pred, block), target));
      }
    }
  }

  IntegerType *T;
  if (targetToPreds.size() == 2)
    T = Type::getInt1Ty(BuilderM.getContext());
  else if (targetToPreds.size() < 256)
    T = Type::getInt8Ty(BuilderM.getContext());
  else
    T = Type::getInt32Ty(BuilderM.getContext());

  Instruction *equivalentTerminator = nullptr;

  std::set<BasicBlock *> blocks;

  // llvm::errs() << "\n\n<DONE = " << ctx->getName() << ">\n";
  for (auto pair : done) {
    const auto &edge = pair.first;
    blocks.insert(edge.first);
    // llvm::errs() << " edge  (" << edge.first->getName() << ", "
    //             << edge.second->getName() << ") : [";
    // for (auto s : pair.second)
    //  llvm::errs() << s->getName() << ",";
    // llvm::errs() << "]\n";
  }
  // llvm::errs() << "</DONE>\n";

  if (targetToPreds.size() == 3) {
    // Try `block` as a potential first split point.
    for (auto block : blocks) {
      {
        // The original split block must not have a parent with an edge
        // to a block other than to itself, which can reach any targets.
        if (!DT.dominates(block, ctx))
          continue;

        // For all successors and thus edges (block, succ):
        // 1) Ensure that no successors have overlapping potential
        // destinations (a list of destinations previously seen is in
        // foundtargets).
        // 2) The block branches to all 3 destinations (foundTargets==3)
        std::set<BasicBlock *> foundtargets;
        // 3) The unique target split off from the others is stored in
        //   uniqueTarget.
        std::set<BasicBlock *> uniqueTargets;
        for (BasicBlock *succ : successors(block)) {
          auto edge = std::make_pair(block, succ);
          for (BasicBlock *target : done[edge]) {
            if (foundtargets.find(target) != foundtargets.end()) {
              goto rnextpair;
            }
            foundtargets.insert(target);
            if (done[edge].size() == 1)
              uniqueTargets.insert(target);
          }
        }
        if (foundtargets.size() != 3)
          goto rnextpair;
        if (uniqueTargets.size() != 1)
          goto rnextpair;

        // Only handle cases where the split was due to a conditional
        // branch. This branch, `bi`, splits off uniqueTargets[0] from
        // the remainder of foundTargets.
        auto bi1 = dyn_cast<BranchInst>(block->getTerminator());
        if (!bi1)
          goto rnextpair;

        {
          // Find a second block `subblock` which splits the two merged
          // targets from each other.
          BasicBlock *subblock = nullptr;
          for (auto block2 : blocks) {
            {
              // The second split block must not have a parent with an edge
              // to a block other than to itself, which can reach any of its two
              // targets.
              // TODO verify this
              for (auto P : predecessors(block2)) {
                for (auto S : successors(P)) {
                  if (S == block2)
                    continue;
                  auto edge = std::make_pair(P, S);
                  if (done.find(edge) != done.end()) {
                    for (auto target : done[edge]) {
                      if (foundtargets.find(target) != foundtargets.end() &&
                          uniqueTargets.find(target) == uniqueTargets.end()) {
                        goto nextblock;
                      }
                    }
                  }
                }
              }

              // Again, a successful split must have unique targets.
              std::set<BasicBlock *> seen2;
              for (BasicBlock *succ : successors(block2)) {
                auto edge = std::make_pair(block2, succ);
                // Since there are only two targets, a successful split
                // condition has only 1 target per successor of block2.
                if (done[edge].size() != 1) {
                  goto nextblock;
                }
                for (BasicBlock *target : done[edge]) {
                  // block2 has non-unique targets.
                  if (seen2.find(target) != seen2.end()) {
                    goto nextblock;
                  }
                  seen2.insert(target);
                  // block2 has a target which is not part of the two needing
                  // to be split. The two needing to be split is equal to
                  //    foundtargets-uniqueTargets.
                  if (foundtargets.find(target) == foundtargets.end()) {
                    goto nextblock;
                  }
                  if (uniqueTargets.find(target) != uniqueTargets.end()) {
                    goto nextblock;
                  }
                }
              }
              // If we didn't find two valid successors, continue.
              if (seen2.size() != 2) {
                // llvm::errs() << " -- failed from not 2 seen\n";
                goto nextblock;
              }
              subblock = block2;
              break;
            }
          nextblock:;
          }

          // If no split block was found, try again.
          if (subblock == nullptr)
            goto rnextpair;

          // This branch, `bi2`, splits off the two blocks in
          // (foundTargets-uniqueTargets) from each other.
          auto bi2 = dyn_cast<BranchInst>(subblock->getTerminator());
          if (!bi2)
            goto rnextpair;

          // Condition cond1 splits off uniqueTargets[0] from
          // the remainder of foundTargets.
          auto cond1 = lookupM(bi1->getCondition(), BuilderM);

          // Condition cond2 splits off the two blocks in
          // (foundTargets-uniqueTargets) from each other.
          auto cond2 = lookupM(bi2->getCondition(), BuilderM);

          if (replacePHIs == nullptr) {
            BasicBlock *staging =
                BasicBlock::Create(oldFunc->getContext(), "staging", newFunc);
            auto stagingIfNeeded = [&](BasicBlock *B) {
              auto edge = std::make_pair(block, B);
              if (done[edge].size() == 1) {
                return *done[edge].begin();
              } else {
                assert(done[edge].size() == 2);
                return staging;
              }
            };
            BuilderM.CreateCondBr(cond1, stagingIfNeeded(bi1->getSuccessor(0)),
                                  stagingIfNeeded(bi1->getSuccessor(1)));
            BuilderM.SetInsertPoint(staging);
            BuilderM.CreateCondBr(
                cond2,
                *done[std::make_pair(subblock, bi2->getSuccessor(0))].begin(),
                *done[std::make_pair(subblock, bi2->getSuccessor(1))].begin());
          } else {
            Value *otherBranch = nullptr;
            for (unsigned i = 0; i < 2; ++i) {
              Value *val = cond1;
              if (i == 1)
                val = BuilderM.CreateNot(val, "anot1_");
              auto edge = std::make_pair(block, bi1->getSuccessor(i));
              if (done[edge].size() == 1) {
                auto found = replacePHIs->find(*done[edge].begin());
                if (found == replacePHIs->end())
                  continue;
                if (&*BuilderM.GetInsertPoint() == found->second) {
                  if (found->second->getNextNode())
                    BuilderM.SetInsertPoint(found->second->getNextNode());
                  else
                    BuilderM.SetInsertPoint(found->second->getParent());
                }
                found->second->replaceAllUsesWith(val);
                found->second->eraseFromParent();
              } else {
                otherBranch = val;
              }
            }

            for (unsigned i = 0; i < 2; ++i) {
              auto edge = std::make_pair(subblock, bi2->getSuccessor(i));
              auto found = replacePHIs->find(*done[edge].begin());
              if (found == replacePHIs->end())
                continue;

              Value *val = cond2;
              if (i == 1)
                val = BuilderM.CreateNot(val, "bnot1_");
              val = BuilderM.CreateAnd(val, otherBranch, "andVal" + Twine(i));
              if (&*BuilderM.GetInsertPoint() == found->second) {
                if (found->second->getNextNode())
                  BuilderM.SetInsertPoint(found->second->getNextNode());
                else
                  BuilderM.SetInsertPoint(found->second->getParent());
              }
              found->second->replaceAllUsesWith(val);
              found->second->eraseFromParent();
            }
          }

          return;
        }
      }
    rnextpair:;
    }
  }

  BasicBlock *forwardBlock = BuilderM.GetInsertBlock();

  if (!isOriginalBlock(*forwardBlock)) {
    forwardBlock = originalForReverseBlock(*forwardBlock);
  }

  for (auto block : blocks) {
    {
      // The original split block must not have a parent with an edge
      // to a block other than to itself, which can reach any targets.
      if (!DT.dominates(block, ctx))
        for (auto P : predecessors(block)) {
          for (auto S : successors(P)) {
            if (S == block)
              continue;
            auto edge = std::make_pair(P, S);
            if (done.find(edge) != done.end() && done[edge].size())
              goto nextpair;
          }
        }

      std::set<BasicBlock *> foundtargets;
      for (BasicBlock *succ : successors(block)) {
        auto edge = std::make_pair(block, succ);
        if (done[edge].size() != 1) {
          goto nextpair;
        }
        BasicBlock *target = *done[edge].begin();
        if (foundtargets.find(target) != foundtargets.end()) {
          goto nextpair;
        }
        foundtargets.insert(target);
      }
      if (foundtargets.size() != targetToPreds.size()) {
        goto nextpair;
      }

      if (forwardBlock == block || DT.dominates(block, forwardBlock)) {
        equivalentTerminator = block->getTerminator();
        goto fast;
      }
    }
  nextpair:;
  }
  goto nofast;

fast:;
  assert(equivalentTerminator);

  if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();
    assert(branch->getCondition());

    assert(branch->getCondition()->getType() == T);

    if (replacePHIs == nullptr) {
      if (!(BuilderM.GetInsertBlock()->size() == 0 ||
            !isa<BranchInst>(BuilderM.GetInsertBlock()->back()))) {
        llvm::errs() << "newFunc : " << *newFunc << "\n";
        llvm::errs() << "blk : " << *BuilderM.GetInsertBlock() << "\n";
      }
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(
          lookupM(branch->getCondition(), BuilderM),
          *done[std::make_pair(block, branch->getSuccessor(0))].begin(),
          *done[std::make_pair(block, branch->getSuccessor(1))].begin());
    } else {
      for (auto pair : *replacePHIs) {
        Value *phi = lookupM(branch->getCondition(), BuilderM);
        Value *val = nullptr;
        if (pair.first ==
            *done[std::make_pair(block, branch->getSuccessor(0))].begin()) {
          val = phi;
        } else if (pair.first ==
                   *done[std::make_pair(block, branch->getSuccessor(1))]
                        .begin()) {
          val = BuilderM.CreateNot(phi);
        } else {
          llvm::errs() << *pair.first->getParent() << "\n";
          llvm::errs() << *pair.first << "\n";
          llvm::errs() << *branch << "\n";
          llvm_unreachable("unknown successor for replacephi");
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else if (auto si = dyn_cast<SwitchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();

    IRBuilder<> pbuilder(equivalentTerminator);
    pbuilder.setFastMathFlags(getFast());

    if (replacePHIs == nullptr) {
      SwitchInst *swtch = BuilderM.CreateSwitch(
          lookupM(si->getCondition(), BuilderM),
          *done[std::make_pair(block, si->getDefaultDest())].begin());
      for (auto switchcase : si->cases()) {
        swtch->addCase(
            switchcase.getCaseValue(),
            *done[std::make_pair(block, switchcase.getCaseSuccessor())]
                 .begin());
      }
    } else {
      for (auto pair : *replacePHIs) {
        Value *cas = nullptr;
        for (auto c : si->cases()) {
          if (pair.first ==
              *done[std::make_pair(block, c.getCaseSuccessor())].begin()) {
            cas = c.getCaseValue();
            break;
          }
        }
        if (cas == nullptr) {
          assert(pair.first ==
                 *done[std::make_pair(block, si->getDefaultDest())].begin());
        }
        Value *val = nullptr;
        Value *phi = lookupM(si->getCondition(), BuilderM);

        if (cas) {
          val = BuilderM.CreateICmpEQ(cas, phi);
        } else {
          // default case
          val = ConstantInt::getFalse(pair.second->getContext());
          for (auto switchcase : si->cases()) {
            val = BuilderM.CreateOr(
                val, BuilderM.CreateICmpEQ(switchcase.getCaseValue(), phi));
          }
          val = BuilderM.CreateNot(val);
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else {
    llvm::errs() << "unknown equivalent terminator\n";
    llvm::errs() << *equivalentTerminator << "\n";
    llvm_unreachable("unknown equivalent terminator");
  }
  return;

nofast:;

  // if freeing reverseblocks must exist
  assert(reverseBlocks.size());
  LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0, ctx);
  AllocaInst *cache = createCacheForScope(lctx, T, "", /*shouldFree*/ true);
  SmallVector<BasicBlock *, 4> targets;
  {
    size_t idx = 0;
    std::map<BasicBlock * /*storingblock*/,
             std::map<ConstantInt * /*target*/,
                      std::vector<BasicBlock *> /*predecessors*/>>
        storing;
    for (const auto &pair : targetToPreds) {
      for (auto pred : pair.second) {
        storing[pred.first][ConstantInt::get(T, idx)].push_back(pred.second);
      }
      targets.push_back(pair.first);
      ++idx;
    }
    assert(targets.size() > 0);

    for (const auto &pair : storing) {
      IRBuilder<> pbuilder(pair.first);

      if (pair.first->getTerminator())
        pbuilder.SetInsertPoint(pair.first->getTerminator());

      pbuilder.setFastMathFlags(getFast());

      Value *tostore = ConstantInt::get(T, 0);

      if (pair.second.size() == 1) {
        tostore = pair.second.begin()->first;
      } else {
        assert(0 && "multi exit edges not supported");
        exit(1);
        // for(auto targpair : pair.second) {
        //     tostore = pbuilder.CreateOr(tostore, pred);
        //}
      }
      storeInstructionInCache(lctx, pbuilder, tostore, cache);
    }
  }

  bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
  Value *which = lookupValueFromCache(
      T,
      /*forwardPass*/ isOriginalBlock(*BuilderM.GetInsertBlock()), BuilderM,
      LimitContext(/*reversePass*/ reverseBlocks.size() > 0, ctx), cache, isi1,
      /*available*/ ValueToValueMapTy());
  assert(which);
  assert(which->getType() == T);

  if (replacePHIs == nullptr) {
    if (targetToPreds.size() == 2) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(which, /*true*/ targets[1], /*false*/ targets[0]);
    } else {
      assert(targets.size() > 0);
      auto swit =
          BuilderM.CreateSwitch(which, targets.back(), targets.size() - 1);
      for (unsigned i = 0; i < targets.size() - 1; ++i) {
        swit->addCase(ConstantInt::get(T, i), targets[i]);
      }
    }
  } else {
    for (unsigned i = 0; i < targets.size(); ++i) {
      auto found = replacePHIs->find(targets[i]);
      if (found == replacePHIs->end())
        continue;

      Value *val = nullptr;
      if (targets.size() == 2 && i == 0) {
        val = BuilderM.CreateNot(which);
      } else if (targets.size() == 2 && i == 1) {
        val = which;
      } else {
        val = BuilderM.CreateICmpEQ(ConstantInt::get(T, i), which);
      }
      if (&*BuilderM.GetInsertPoint() == found->second) {
        if (found->second->getNextNode())
          BuilderM.SetInsertPoint(found->second->getNextNode());
        else
          BuilderM.SetInsertPoint(found->second->getParent());
      }
      found->second->replaceAllUsesWith(val);
      found->second->eraseFromParent();
    }
  }
  return;
}

void GradientUtils::computeMinCache() {
  if (EnzymeMinCutCache) {
    SmallPtrSet<Value *, 4> Recomputes;

    std::map<UsageKey, bool> FullSeen;
    std::map<UsageKey, bool> OneLevelSeen;

    ValueToValueMapTy Available;

    std::map<Loop *, std::set<Instruction *>> LoopAvail;

    for (BasicBlock &BB : *oldFunc) {
      if (notForAnalysis.count(&BB))
        continue;
      auto L = OrigLI.getLoopFor(&BB);

      auto invariant = [&](Value *V) {
        if (isa<Constant>(V))
          return true;
        if (isa<Argument>(V))
          return true;
        if (auto I = dyn_cast<Instruction>(V)) {
          if (!L->contains(OrigLI.getLoopFor(I->getParent())))
            return true;
        }
        return false;
      };
      for (Instruction &I : BB) {
        if (auto PN = dyn_cast<PHINode>(&I)) {
          if (!OrigLI.isLoopHeader(&BB))
            continue;
          if (PN->getType()->isIntegerTy()) {
            bool legal = true;
            SmallPtrSet<Instruction *, 4> Increment;
            for (auto B : PN->blocks()) {
              if (OrigLI.getLoopFor(B) == L) {
                if (auto BO = dyn_cast<BinaryOperator>(
                        PN->getIncomingValueForBlock(B))) {
                  if (BO->getOpcode() == BinaryOperator::Add) {
                    if ((BO->getOperand(0) == PN &&
                         invariant(BO->getOperand(1))) ||
                        (BO->getOperand(1) == PN &&
                         invariant(BO->getOperand(0)))) {
                      Increment.insert(BO);
                    } else {
                      legal = false;
                    }
                  } else if (BO->getOpcode() == BinaryOperator::Sub) {
                    if (BO->getOperand(0) == PN &&
                        invariant(BO->getOperand(1))) {
                      Increment.insert(BO);
                    } else {
                      legal = false;
                    }
                  } else {
                    legal = false;
                  }
                } else {
                  legal = false;
                }
              }
            }
            if (legal) {
              LoopAvail[L].insert(PN);
              for (auto I : Increment)
                LoopAvail[L].insert(I);
            }
          }
        } else if (auto CI = dyn_cast<CallInst>(&I)) {
          StringRef funcName = getFuncNameFromCall(CI);
          if (isAllocationFunction(funcName, TLI))
            Available[CI] = CI;
        }
      }
    }

    SmallPtrSet<Instruction *, 3> NewLoopBoundReq;
    {
      std::deque<Instruction *> LoopBoundRequirements;

      for (auto &context : loopContexts) {
        for (auto val : {context.second.maxLimit, context.second.trueLimit}) {
          if (val)
            if (auto inst = dyn_cast<Instruction>(&*val)) {
              LoopBoundRequirements.push_back(inst);
            }
        }
      }
      SmallPtrSet<Instruction *, 3> Seen;
      while (LoopBoundRequirements.size()) {
        Instruction *val = LoopBoundRequirements.front();
        LoopBoundRequirements.pop_front();
        if (NewLoopBoundReq.count(val))
          continue;
        if (Seen.count(val))
          continue;
        Seen.insert(val);
        if (auto orig = isOriginal(val)) {
          NewLoopBoundReq.insert(orig);
        } else {
          for (auto &op : val->operands()) {
            if (auto inst = dyn_cast<Instruction>(op)) {
              LoopBoundRequirements.push_back(inst);
            }
          }
        }
      }
      for (auto inst : NewLoopBoundReq) {
        OneLevelSeen[UsageKey(inst, ValueType::Primal)] = true;
        FullSeen[UsageKey(inst, ValueType::Primal)] = true;
      }
    }

    auto minCutMode = (mode == DerivativeMode::ReverseModePrimal)
                          ? DerivativeMode::ReverseModeGradient
                          : mode;

    for (BasicBlock &BB : *oldFunc) {
      if (notForAnalysis.count(&BB))
        continue;
      ValueToValueMapTy Available2;
      for (auto a : Available)
        Available2[a.first] = a.second;
      for (Loop *L = OrigLI.getLoopFor(&BB); L != nullptr;
           L = L->getParentLoop()) {
        for (auto v : LoopAvail[L]) {
          Available2[v] = v;
        }
      }
      for (Instruction &I : BB) {
        if (!legalRecompute(&I, Available2, nullptr)) {
          if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                  ValueType::Primal>(this, &I, minCutMode, FullSeen,
                                     notForAnalysis)) {
            bool oneneed = DifferentialUseAnalysis::is_value_needed_in_reverse<
                ValueType::Primal,
                /*OneLevel*/ true>(this, &I, minCutMode, OneLevelSeen,
                                   notForAnalysis);
            if (oneneed) {
              knownRecomputeHeuristic[&I] = false;
            } else
              Recomputes.insert(&I);
          }
        }
      }
    }

    SmallPtrSet<Value *, 4> Intermediates;
    SmallPtrSet<Value *, 4> Required;

    Intermediates.clear();
    Required.clear();
    std::deque<Value *> todo(Recomputes.begin(), Recomputes.end());

    while (todo.size()) {
      Value *V = todo.front();
      todo.pop_front();
      if (Intermediates.count(V))
        continue;
      if (!DifferentialUseAnalysis::is_value_needed_in_reverse<
              ValueType::Primal>(this, V, minCutMode, FullSeen,
                                 notForAnalysis)) {
        continue;
      }
      if (!Recomputes.count(V)) {
        ValueToValueMapTy Available2;
        for (auto a : Available)
          Available2[a.first] = a.second;
        for (Loop *L = OrigLI.getLoopFor(cast<Instruction>(V)->getParent());
             L != nullptr; L = L->getParentLoop()) {
          for (auto v : LoopAvail[L]) {
            Available2[v] = v;
          }
        }
        if (!legalRecompute(V, Available2, nullptr)) {
          // if not legal to recompute, we would've already explicitly marked
          // this for caching if it was needed in reverse pass
          continue;
        }
      }
      Intermediates.insert(V);
      if (DifferentialUseAnalysis::is_value_needed_in_reverse<
              ValueType::Primal, /*OneLevel*/ true>(
              this, V, minCutMode, OneLevelSeen, notForAnalysis)) {
        Required.insert(V);
      } else {
        for (auto V2 : V->users()) {
          if (auto Inst = dyn_cast<Instruction>(V2))
            for (auto pair : rematerializableAllocations) {
              if (pair.second.stores.count(Inst)) {
                todo.push_back(pair.first);
              }
            }
          todo.push_back(V2);
        }
      }
    }

    SmallPtrSet<Value *, 5> MinReq;
    DifferentialUseAnalysis::minCut(oldFunc->getParent()->getDataLayout(),
                                    OrigLI, Recomputes, Intermediates, Required,
                                    MinReq, rematerializableAllocations, TLI);
    SmallPtrSet<Value *, 5> NeedGraph;
    for (Value *V : MinReq)
      NeedGraph.insert(V);
    for (Value *V : Required)
      todo.push_back(V);
    while (todo.size()) {
      Value *V = todo.front();
      todo.pop_front();
      if (NeedGraph.count(V))
        continue;
      NeedGraph.insert(V);
      if (auto I = dyn_cast<Instruction>(V))
        for (auto &V2 : I->operands()) {
          if (Intermediates.count(V2))
            todo.push_back(V2);
        }
    }

    for (auto V : Intermediates) {
      knownRecomputeHeuristic[V] = !MinReq.count(V);
      if (!NeedGraph.count(V)) {
        unnecessaryIntermediates.insert(cast<Instruction>(V));
      }
    }
  }
}

bool GradientUtils::isOriginalBlock(const BasicBlock &BB) const {
  for (auto A : originalBlocks) {
    if (A == &BB)
      return true;
  }
  return false;
}

void GradientUtils::eraseFictiousPHIs() {
  {
    for (auto P : rematerializedPrimalOrShadowAllocations) {
      Value *replacement = getUndefinedValueForType(P->getType());
      P->replaceAllUsesWith(replacement);
      erase(P);
    }
  }
  SmallVector<std::pair<PHINode *, Value *>, 4> phis;
  for (auto pair : fictiousPHIs)
    phis.emplace_back(pair.first, pair.second);
  fictiousPHIs.clear();

  for (auto pair : phis) {
    auto pp = pair.first;
    if (pp->getNumUses() != 0) {
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Illegal replace ficticious phi for: " << *pp << " of "
           << *pair.second;
        CustomErrorHandler(str.c_str(), wrap(pair.second),
                           ErrorType::IllegalReplaceFicticiousPHIs, this,
                           wrap(pp), nullptr);
      } else {
        llvm::errs() << "mod:" << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc:" << *newFunc << "\n";
        llvm::errs() << " pp: " << *pp << " of " << *pair.second << "\n";
        assert(pp->getNumUses() == 0);
      }
    }
    pp->replaceAllUsesWith(UndefValue::get(pp->getType()));
    erase(pp);
  }
}

void GradientUtils::forceActiveDetection() {
  for (auto &Arg : oldFunc->args()) {
    ATA->isConstantValue(TR, &Arg);
  }

  for (BasicBlock &BB : *oldFunc) {
    for (Instruction &I : BB) {
      bool const_inst = ATA->isConstantInstruction(TR, &I);
      bool const_value = ATA->isConstantValue(TR, &I);

      if (EnzymePrintActivity)
        llvm::errs() << I << " cv=" << const_value << " ci=" << const_inst
                     << "\n";
    }
  }
}

bool GradientUtils::isConstantValue(Value *val) const {
  if (auto inst = dyn_cast<Instruction>(val)) {
    assert(inst->getParent()->getParent() == oldFunc);
    return ATA->isConstantValue(TR, val);
  }

  if (auto arg = dyn_cast<Argument>(val)) {
    assert(arg->getParent() == oldFunc);
    return ATA->isConstantValue(TR, val);
  }

  //! Functions must be false so we can replace function with augmentation,
  //! fallback to analysis
  if (isa<Function>(val) || isa<InlineAsm>(val) || isa<Constant>(val) ||
      isa<UndefValue>(val) || isa<MetadataAsValue>(val)) {
    // llvm::errs() << "calling icv on: " << *val << "\n";
    return ATA->isConstantValue(TR, val);
  }

  if (auto gv = dyn_cast<GlobalVariable>(val)) {
    if (hasMetadata(gv, "enzyme_shadow"))
      return false;
    if (auto md = gv->getMetadata("enzyme_activity_value")) {
      auto res = cast<MDString>(md->getOperand(0))->getString();
      if (res == "const")
        return true;
      if (res == "active")
        return false;
    }
    if (EnzymeNonmarkedGlobalsInactive)
      return true;
    goto err;
  }
  if (isa<GlobalValue>(val)) {
    if (EnzymeNonmarkedGlobalsInactive)
      return true;
    goto err;
  }

err:;
  llvm::errs() << *oldFunc << "\n";
  llvm::errs() << *newFunc << "\n";
  llvm::errs() << *val << "\n";
  llvm::errs() << "  unknown did status attribute\n";
  assert(0 && "bad");
  exit(1);
}

bool GradientUtils::isConstantInstruction(const Instruction *inst) const {
  assert(inst->getParent()->getParent() == oldFunc);
  return ATA->isConstantInstruction(TR, const_cast<Instruction *>(inst));
}

bool GradientUtils::getContext(llvm::BasicBlock *BB, LoopContext &lc) {
  return CacheUtility::getContext(BB, lc,
                                  /*ReverseLimit*/ reverseBlocks.size() > 0);
}

void GradientUtils::forceAugmentedReturns() {
  assert(TR.getFunction() == oldFunc);

  for (BasicBlock &oBB : *oldFunc) {
    // Don't create derivatives for code that results in termination
    if (notForAnalysis.find(&oBB) != notForAnalysis.end())
      continue;

    LoopContext loopContext;
    getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

    for (Instruction &I : oBB) {
      Instruction *inst = &I;

      if (inst->getType()->isEmptyTy() || inst->getType()->isVoidTy())
        continue;

      if (mode == DerivativeMode::ForwardMode ||
          mode == DerivativeMode::ForwardModeSplit) {
        if (!isConstantValue(inst)) {
          IRBuilder<> BuilderZ(inst);
          getForwardBuilder(BuilderZ);
          Type *antiTy = getShadowType(inst->getType());
          PHINode *anti =
              BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'dual_phi");
          invertedPointers.insert(std::make_pair(
              (const Value *)inst, InvertedPointerVH(this, anti)));
        }
        continue;
      }

      if (inst->getType()->isFPOrFPVectorTy())
        continue; //! op->getType()->isPointerTy() &&
                  //! !op->getType()->isIntegerTy()) {

      if (!TR.query(inst)[{-1}].isPossiblePointer())
        continue;

      if (isa<LoadInst>(inst)) {
        IRBuilder<> BuilderZ(inst);
        getForwardBuilder(BuilderZ);
        Type *antiTy = getShadowType(inst->getType());
        PHINode *anti =
            BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'il_phi");
        invertedPointers.insert(
            std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));
        continue;
      }

      if (!isa<CallInst>(inst)) {
        continue;
      }

      CallInst *op = cast<CallInst>(inst);
      Function *called = op->getCalledFunction();

      if ((mode == DerivativeMode::ReverseModeGradient ||
           mode == DerivativeMode::ReverseModeCombined) &&
          called && called->getName() == "llvm.julia.gc_preserve_begin") {
        IRBuilder<> BuilderZ(inst);
        getForwardBuilder(BuilderZ);
        auto anti = BuilderZ.CreateCall(called, ArrayRef<Value *>(),
                                        op->getName() + "'ip");
        anti->setDebugLoc(getNewFromOriginal(op->getDebugLoc()));
        invertedPointers.insert(
            std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));
        continue;
      }

      if (isa<IntrinsicInst>(inst)) {
        continue;
      }

      if (isConstantValue(inst)) {
        continue;
      }

      IRBuilder<> BuilderZ(inst);
      getForwardBuilder(BuilderZ);

      // Shadow allocations must strictly preceede the primal, lest Julia have
      // GC issues. Consider the following: %r = gc_alloc() init %r
      // ...
      // if the shadow did not preceed
      // %r = gc_alloc()
      // %dr = gc_alloc()
      // zero %dr
      // init %r, %dr
      // ...
      // After %r, before %dr the %r memory would be uninit, so the allocator
      // inside %dr would hit garbage and segfault. However, by having the %dr
      // first, then it will be zero'd before the %r allocation, preventing the
      // issue.
      if (isAllocationCall(inst, TLI))
        BuilderZ.SetInsertPoint(getNewFromOriginal(inst));
      Type *antiTy = getShadowType(inst->getType());

      PHINode *anti = BuilderZ.CreatePHI(antiTy, 1, op->getName() + "'ip_phi");
      anti->setDebugLoc(getNewFromOriginal(op->getDebugLoc()));
      invertedPointers.insert(
          std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

      if (isAllocationCall(inst, TLI)) {
        anti->setName(op->getName() + "'mi");
      }
    }
  }
}

void InvertedPointerVH::deleted() {
  llvm::errs() << *gutils->oldFunc << "\n";
  llvm::errs() << *gutils->newFunc << "\n";
  gutils->dumpPointers();
  llvm::errs() << **this << "\n";
  assert(0 && "erasing something in invertedPointers map");
}

void SubTransferHelper(GradientUtils *gutils, DerivativeMode mode,
                       Type *secretty, Intrinsic::ID intrinsic,
                       unsigned dstalign, unsigned srcalign, unsigned offset,
                       bool dstConstant, Value *shadow_dst, bool srcConstant,
                       Value *shadow_src, Value *length, Value *isVolatile,
                       llvm::CallInst *MTI, bool allowForward,
                       bool shadowsLookedUp, bool backwardsShadow) {
  // TODO offset
  if (secretty) {
    // no change to forward pass if represents floats
    if (mode == DerivativeMode::ReverseModeGradient ||
        mode == DerivativeMode::ReverseModeCombined ||
        mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(MTI);
      if (mode == DerivativeMode::ForwardModeSplit)
        gutils->getForwardBuilder(Builder2);
      else
        gutils->getReverseBuilder(Builder2);

      // If the src is constant simply zero d_dst and don't propagate to d_src
      // (which thus == src and may be illegal)
      if (srcConstant) {
        // Don't zero in forward mode.
        if (mode != DerivativeMode::ForwardModeSplit) {

          Value *args[] = {
              shadowsLookedUp ? shadow_dst
                              : gutils->lookupM(shadow_dst, Builder2),
              ConstantInt::get(Type::getInt8Ty(MTI->getContext()), 0),
              gutils->lookupM(length, Builder2),
              ConstantInt::getFalse(MTI->getContext())};

          if (args[0]->getType()->isIntegerTy())
            args[0] = Builder2.CreateIntToPtr(
                args[0], Type::getInt8PtrTy(MTI->getContext()));

          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memsetIntr = Intrinsic::getDeclaration(
              MTI->getParent()->getParent()->getParent(), Intrinsic::memset,
              tys);
          auto cal = Builder2.CreateCall(memsetIntr, args);
          cal->setCallingConv(memsetIntr->getCallingConv());
          if (dstalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
            cal->addParamAttr(0, Attribute::getWithAlignment(MTI->getContext(),
                                                             Align(dstalign)));
#else
            cal->addParamAttr(
                0, Attribute::getWithAlignment(MTI->getContext(), dstalign));
#endif
          }
        }

      } else {
        auto dsto =
            (shadowsLookedUp || mode == DerivativeMode::ForwardModeSplit)
                ? shadow_dst
                : gutils->lookupM(shadow_dst, Builder2);
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        if (offset != 0) {
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(dsto->getContext()), dsto, offset);
        }
        auto srco =
            (shadowsLookedUp || mode == DerivativeMode::ForwardModeSplit)
                ? shadow_src
                : gutils->lookupM(shadow_src, Builder2);
        if (mode != DerivativeMode::ForwardModeSplit)
          dsto = Builder2.CreatePointerCast(
              dsto, PointerType::get(secretty, dstaddr));
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(srco->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        if (offset != 0) {
          srco = Builder2.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(srco->getContext()), srco, offset);
        }
        if (mode != DerivativeMode::ForwardModeSplit)
          srco = Builder2.CreatePointerCast(
              srco, PointerType::get(secretty, srcaddr));

        if (mode == DerivativeMode::ForwardModeSplit) {
#if LLVM_VERSION_MAJOR >= 11
          MaybeAlign dalign;
          if (dstalign)
            dalign = MaybeAlign(dstalign);
          MaybeAlign salign;
          if (srcalign)
            salign = MaybeAlign(srcalign);
#else
          auto dalign = dstalign;
          auto salign = srcalign;
#endif

          if (intrinsic == Intrinsic::memmove) {
            Builder2.CreateMemMove(dsto, dalign, srco, salign, length);
          } else {
            Builder2.CreateMemCpy(dsto, dalign, srco, salign, length);
          }
        } else {
          Value *args[]{
              Builder2.CreatePointerCast(dsto,
                                         PointerType::get(secretty, dstaddr)),
              Builder2.CreatePointerCast(srco,
                                         PointerType::get(secretty, srcaddr)),
              Builder2.CreateUDiv(
                  gutils->lookupM(length, Builder2),
                  ConstantInt::get(length->getType(),
                                   Builder2.GetInsertBlock()
                                           ->getParent()
                                           ->getParent()
                                           ->getDataLayout()
                                           .getTypeAllocSizeInBits(secretty) /
                                       8))};

          auto dmemcpy = ((intrinsic == Intrinsic::memcpy)
                              ? getOrInsertDifferentialFloatMemcpy
                              : getOrInsertDifferentialFloatMemmove)(
              *MTI->getParent()->getParent()->getParent(), secretty, dstalign,
              srcalign, dstaddr, srcaddr,
              cast<IntegerType>(length->getType())->getBitWidth());
          Builder2.CreateCall(dmemcpy, args);
        }
      }
    }
  } else {

    // if represents pointer or integer type then only need to modify forward
    // pass with the copy
    if ((allowForward && (mode == DerivativeMode::ReverseModePrimal ||
                          mode == DerivativeMode::ReverseModeCombined)) ||
        (backwardsShadow && (mode == DerivativeMode::ReverseModeGradient ||
                             mode == DerivativeMode::ForwardModeSplit))) {
      assert(!shadowsLookedUp);

      // It is questionable how the following case would even occur, but if
      // the dst is constant, we shouldn't do anything extra
      if (dstConstant) {
        return;
      }

      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(MTI));

      // If src is inactive, then we should copy from the regular pointer
      // (i.e. suppose we are copying constant memory representing dimensions
      // into a tensor)
      //  to ensure that the differential tensor is well formed for use
      //  OUTSIDE the derivative generation (as enzyme doesn't need this), we
      //  should also perform the copy onto the differential. Future
      //  Optimization (not implemented): If dst can never escape Enzyme code,
      //  we may omit this copy.
      // no need to update pointers, even if dst is active
      auto dsto = shadow_dst;
      if (dsto->getType()->isIntegerTy())
        dsto = BuilderZ.CreateIntToPtr(dsto,
                                       Type::getInt8PtrTy(MTI->getContext()));
      if (offset != 0) {
        dsto = BuilderZ.CreateConstInBoundsGEP1_64(
            Type::getInt8Ty(dsto->getContext()), dsto, offset);
      }
      auto srco = shadow_src;
      if (srco->getType()->isIntegerTy())
        srco = BuilderZ.CreateIntToPtr(srco,
                                       Type::getInt8PtrTy(MTI->getContext()));
      if (offset != 0) {
        srco = BuilderZ.CreateConstInBoundsGEP1_64(
            Type::getInt8Ty(srco->getContext()), srco, offset);
      }
      Value *args[] = {dsto, srco, length, isVolatile};

      Type *tys[] = {args[0]->getType(), args[1]->getType(),
                     args[2]->getType()};

      auto memtransIntr = Intrinsic::getDeclaration(
          gutils->newFunc->getParent(), intrinsic, tys);
      auto cal = BuilderZ.CreateCall(memtransIntr, args);
      cal->setAttributes(MTI->getAttributes());
      cal->setCallingConv(memtransIntr->getCallingConv());
      cal->setTailCallKind(MTI->getTailCallKind());

      if (dstalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
        cal->addParamAttr(
            0, Attribute::getWithAlignment(MTI->getContext(), Align(dstalign)));
#else
        cal->addParamAttr(
            0, Attribute::getWithAlignment(MTI->getContext(), dstalign));
#endif
      }
      if (srcalign != 0) {
#if LLVM_VERSION_MAJOR >= 10
        cal->addParamAttr(
            1, Attribute::getWithAlignment(MTI->getContext(), Align(srcalign)));
#else
        cal->addParamAttr(
            1, Attribute::getWithAlignment(MTI->getContext(), srcalign));
#endif
      }
    }
  }
}

void GradientUtils::computeForwardingProperties(Instruction *V) {
  if (!EnzymeRematerialize)
    return;

  // For the piece of memory V allocated within this scope, it will be
  // initialized in some way by the (augmented) forward pass. Loads and other
  // load-like operations will either require the allocation V itself to be
  // preserved for the reverse pass, or alternatively the tape for those
  // operations.
  //
  // Instead, we ask here whether or not we can restore the memory state of V in
  // the reverse pass by recreating all of the stores and store-like operations
  // into the V prior to their load-like uses.
  //
  // Notably, we only need to preserve the ability to reload any values actually
  // used in the reverse pass.

  std::map<UsageKey, bool> Seen;
  bool primalNeededInReverse =
      DifferentialUseAnalysis::is_value_needed_in_reverse<ValueType::Primal>(
          this, V, DerivativeMode::ReverseModeGradient, Seen, notForAnalysis);

  SmallVector<LoadInst *, 1> loads;
  SmallVector<LoadLikeCall, 1> loadLikeCalls;
  SmallPtrSet<Instruction *, 1> stores;
  SmallPtrSet<Instruction *, 1> storingOps;
  SmallPtrSet<Instruction *, 1> frees;
  SmallPtrSet<IntrinsicInst *, 1> LifetimeStarts;
  bool promotable = true;
  bool shadowpromotable = true;

  SmallVector<Instruction *, 1> shadowPointerLoads;

  std::set<std::pair<Instruction *, Value *>> seen;
  SmallVector<std::pair<Instruction *, Value *>, 1> todo;
  for (auto U : V->users())
    if (auto I = dyn_cast<Instruction>(U))
      todo.push_back(std::make_pair(I, V));
  while (todo.size()) {
    auto tup = todo.back();
    Instruction *cur = tup.first;
    Value *prev = tup.second;
    todo.pop_back();
    if (seen.count(tup))
      continue;
    seen.insert(tup);
    if (notForAnalysis.count(cur->getParent()))
      continue;
    if (isPointerArithmeticInst(cur)) {
      for (auto u : cur->users()) {
        if (auto I = dyn_cast<Instruction>(u))
          todo.push_back(std::make_pair(I, (Value *)cur));
      }
    } else if (auto load = dyn_cast<LoadInst>(cur)) {

      // If loaded value is an int or pointer, may need
      // to preserve initialization within the primal.
      auto TT = TR.query(load)[{-1}];
      if (!TT.isFloat()) {
        shadowPointerLoads.push_back(cur);
      }
      loads.push_back(load);
    } else if (auto store = dyn_cast<StoreInst>(cur)) {
      // TODO only add store to shadow iff non float type
      if (store->getValueOperand() == prev) {
        EmitWarning("NotPromotable", *cur, " Could not promote allocation ", *V,
                    " due to capturing store ", *cur);
        promotable = false;
        shadowpromotable = false;
        break;
      } else {
        stores.insert(store);
        storingOps.insert(store);
      }
    } else if (auto II = dyn_cast<IntrinsicInst>(cur)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::lifetime_start:
        LifetimeStarts.insert(II);
        break;
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
      case Intrinsic::dbg_label:
#if LLVM_VERSION_MAJOR <= 16
      case llvm::Intrinsic::dbg_addr:
#endif
      case Intrinsic::lifetime_end:
        break;
      case Intrinsic::memset: {
        stores.insert(II);
        storingOps.insert(II);
        break;
      }
      // TODO memtransfer(cpy/move)
      case Intrinsic::memcpy:
      case Intrinsic::memmove:
      default:
        promotable = false;
        shadowpromotable = false;
        EmitWarning("NotPromotable", *cur, " Could not promote allocation ", *V,
                    " due to unknown intrinsic ", *cur);
        break;
      }
    } else if (auto CI = dyn_cast<CallInst>(cur)) {
      StringRef funcName = getFuncNameFromCall(CI);
      if (isDeallocationFunction(funcName, TLI)) {
        frees.insert(CI);
        continue;
      }
      if (funcName == "julia.write_barrier") {
        stores.insert(CI);
        continue;
      }
      if (funcName == "enzyme_zerotype") {
        stores.insert(CI);
        continue;
      }

      size_t idx = 0;
      bool seenLoadLikeCall = false;
#if LLVM_VERSION_MAJOR >= 14
      for (auto &arg : CI->args())
#else
      for (auto &arg : CI->arg_operands())
#endif
      {
        if (arg != prev) {
          idx++;
          continue;
        }
        auto TT = TR.query(prev)[{-1, -1}];

        bool NoCapture = isNoCapture(CI, idx);

        bool ReadOnly = isReadOnly(CI, idx);

        bool WriteOnly = isWriteOnly(CI, idx);

        // If the pointer is captured, conservatively assume it is used in
        // nontrivial ways that make both the primal and shadow not promotable.
        if (!NoCapture) {
          shadowpromotable = false;
          promotable = false;
          EmitWarning("NotPromotable", *cur, " Could not promote allocation ",
                      *V, " due to unknown capturing call ", *cur);
          idx++;
          continue;
        }

        // From here on out we can assume the pointer is not captured, and only
        // written to or read from.

        // If we may read from the memory, consider this a load-like call
        // that must have all writes done in preparation for any reverse-pass
        // users.
        if (!WriteOnly) {
          if (!seenLoadLikeCall) {
            loadLikeCalls.push_back(LoadLikeCall(CI, prev));
            seenLoadLikeCall = true;
          }
        }

        // If we may write to memory, we cannot promote if any values
        // need the allocation or any descendants for the reverse pass.
        if (!ReadOnly) {
          if (primalNeededInReverse) {
            promotable = false;
            EmitWarning("NotPromotable", *cur, " Could not promote allocation ",
                        *V, " due to unknown writing call ", *cur);
          }
          storingOps.insert(cur);
        }

        // Consider shadow memory now.
        //
        // If the memory is all floats, there's no issue, since besides zero
        // initialization nothing should occur for them in the forward pass
        if (TT.isFloat()) {
        } else if (WriteOnly) {
          // Don't need in the case of int/pointer stores, (should be done by
          // fwd pass), and as isFloat above described does not prevent the
          // shadow
        } else {
          shadowPointerLoads.push_back(cur);
        }

        idx++;
      }

    } else {
      promotable = false;
      shadowpromotable = false;
      EmitWarning("NotPromotable", *cur, " Could not promote allocation ", *V,
                  " due to unknown instruction ", *cur);
    }
  }

  // Find the outermost loop of all stores, and the allocation/lifetime
  Loop *outer = OrigLI.getLoopFor(V->getParent());
  if (LifetimeStarts.size() == 1) {
    outer = OrigLI.getLoopFor((*LifetimeStarts.begin())->getParent());
  }

  for (auto S : stores) {
    outer = getAncestor(outer, OrigLI.getLoopFor(S->getParent()));
  }

  // May now read pointers for storing into other pointers. Therefore we
  // need to pre initialize the shadow.
  bool primalInitializationOfShadow = shadowPointerLoads.size() > 0;

  if (shadowpromotable && !isConstantValue(V)) {
    for (auto LI : shadowPointerLoads) {
      // Is there a store which could occur after the load.
      // This subsequent store would invalidate any loads being re-performed.
      SmallVector<Instruction *, 2> results;
      mayExecuteAfter(results, LI, storingOps, outer);
      for (auto res : results) {
        if (overwritesToMemoryReadBy(OrigAA, TLI, SE, OrigLI, OrigDT, LI, res,
                                     outer)) {
          EmitWarning("NotPromotable", *LI,
                      " Could not promote shadow allocation ", *V,
                      " due to pointer load ", *LI,
                      " which does not postdominates store ", *res);
          shadowpromotable = false;
          goto exitL;
        }
      }
    }
    // If there is a store not reproduced in the reverse pass (e.g. as part
    // of a write in a call), and this store is necessary to a pointer load of
    // the shadow, this is not materializable since the load will not return
    // the same value.
    {
      SmallVector<Instruction *, 2> nonReproducedStores;
      for (auto S : storingOps)
        if (!stores.count(S)) {
          SmallVector<Instruction *, 2> results;
          SmallPtrSet<Instruction *, 2> shadowPtrLoadSet(
              shadowPointerLoads.begin(), shadowPointerLoads.end());
          mayExecuteAfter(results, S, shadowPtrLoadSet, outer);
          if (results.size()) {
            EmitWarning("NotPromotable", *results[0],
                        " Could not promote shadow allocation ", *V,
                        " due to non-reproduced store ", *S,
                        " which may impact pointer load ", *results[0]);
            shadowpromotable = false;
            goto exitL;
          }
        }
    }
  exitL:;
    if (shadowpromotable) {
      backwardsOnlyShadows[V] = ShadowRematerializer(
          stores, frees, primalInitializationOfShadow, outer);
    }
  }

  if (!promotable)
    return;

  SmallPtrSet<LoadInst *, 1> rematerializable;

  // We currently require a rematerializable allocation to have
  // all of its loads be able to be performed again. Thus if
  // there is an overwriting store after a load in context,
  // it may no longer be rematerializable.
  for (auto LI : loads) {
    // Is there a store which could occur after the load.
    // In other words
    SmallVector<Instruction *, 2> results;
    mayExecuteAfter(results, LI, storingOps, outer);
    for (auto res : results) {
      if (overwritesToMemoryReadBy(OrigAA, TLI, SE, OrigLI, OrigDT, LI, res,
                                   outer)) {
        EmitWarning("NotPromotable", *LI, " Could not promote allocation ", *V,
                    " due to load ", *LI,
                    " which does not postdominates store ", *res);
        return;
      }
    }
    rematerializable.insert(LI);
  }
  for (auto LI : loadLikeCalls) {
    // Is there a store which could occur after the load.
    // In other words
    SmallVector<Instruction *, 2> results;
    mayExecuteAfter(results, LI.loadCall, storingOps, outer);
    for (auto res : results) {
      if (overwritesToMemoryReadBy(OrigAA, TLI, SE, OrigLI, OrigDT, LI.loadCall,
                                   res, outer)) {
        EmitWarning("NotPromotable", *LI.loadCall,
                    " Could not promote allocation ", *V,
                    " due to load-like call ", *LI.loadCall,
                    " which does not postdominates store ", *res);
        return;
      }
    }
  }
  rematerializableAllocations[V] =
      Rematerializer(loads, loadLikeCalls, stores, frees, outer);
}

BasicBlock *GradientUtils::addReverseBlock(BasicBlock *currentBlock,
                                           Twine const &name, bool forkCache,
                                           bool push) {
  assert(reverseBlocks.size());
  auto found = reverseBlockToPrimal.find(currentBlock);
  assert(found != reverseBlockToPrimal.end());

  SmallVector<BasicBlock *, 4> &vec = reverseBlocks[found->second];
  assert(vec.size());
  assert(vec.back() == currentBlock);

  BasicBlock *rev =
      BasicBlock::Create(currentBlock->getContext(), name, newFunc);
  rev->moveAfter(currentBlock);
  if (push)
    vec.push_back(rev);
  reverseBlockToPrimal[rev] = found->second;
  if (forkCache) {
    for (auto pair : unwrap_cache[currentBlock])
      unwrap_cache[rev].insert(pair);
    for (auto pair : lookup_cache[currentBlock])
      lookup_cache[rev].insert(pair);
  }
  return rev;
}

void GradientUtils::replaceAWithB(Value *A, Value *B, bool storeInCache) {
  if (A == B)
    return;
  assert(A->getType() == B->getType());

  if (auto iA = dyn_cast<Instruction>(A)) {
    if (unwrappedLoads.find(iA) != unwrappedLoads.end()) {
      auto iB = cast<Instruction>(B);
      unwrappedLoads[iB] = unwrappedLoads[iA];
      unwrappedLoads.erase(iA);
    }
  }

  // Check that the replacement doesn't already exist in the mapping
  // thereby resulting in a conflict.
  {
    auto found = newToOriginalFn.find(A);
    if (found != newToOriginalFn.end()) {
      auto foundB = newToOriginalFn.find(B);
      assert(foundB == newToOriginalFn.end());
    }
  }

  CacheUtility::replaceAWithB(A, B, storeInCache);
}

void GradientUtils::erase(Instruction *I) {
  assert(I);
  if (I->getParent()->getParent() != newFunc) {
    llvm::errs() << "newFunc: " << *newFunc << "\n";
    llvm::errs() << "paren: " << *I->getParent()->getParent() << "\n";
    llvm::errs() << "I: " << *I << "\n";
  }
  assert(I->getParent()->getParent() == newFunc);

  // not original, should not contain
  assert(!invertedPointers.count(I));
  // not original, should not contain
  assert(!originalToNewFn.count(I));

  originalToNewFn.erase(I);
  {
    auto found = newToOriginalFn.find(I);
    if (found != newToOriginalFn.end()) {
      Value *orig = found->second;
      newToOriginalFn.erase(found);
      originalToNewFn.erase(orig);
    }
  }
  {
    auto found = UnwrappedWarnings.find(I);
    if (found != UnwrappedWarnings.end()) {
      UnwrappedWarnings.erase(found);
    }
  }
  unwrappedLoads.erase(I);

  for (auto &pair : unwrap_cache) {
    if (pair.second.find(I) != pair.second.end())
      pair.second.erase(I);
  }

  for (auto &pair : lookup_cache) {
    if (pair.second.find(I) != pair.second.end())
      pair.second.erase(I);
  }
  CacheUtility::erase(I);
}

void GradientUtils::eraseWithPlaceholder(Instruction *I, const Twine &suffix,
                                         bool erase) {
  PHINode *pn = nullptr;
  if (!I->getType()->isVoidTy() && !I->getType()->isTokenTy()) {
    IRBuilder<> BuilderZ(I);
    auto pn = BuilderZ.CreatePHI(I->getType(), 1, I->getName() + suffix);
    fictiousPHIs[pn] = I;
    replaceAWithB(I, pn);
  }

  if (erase) {
    this->erase(I);
  }
}

void GradientUtils::setTape(Value *newtape) {
  assert(tape == nullptr);
  assert(newtape != nullptr);
  assert(tapeidx == 0);
  assert(addedTapeVals.size() == 0);
  tape = newtape;
}

void GradientUtils::dumpPointers() {
  errs() << "invertedPointers:\n";
  for (auto a : invertedPointers) {
    errs() << "   invertedPointers[" << *a.first << "] = " << *a.second << "\n";
  }
  errs() << "end invertedPointers\n";
}

int GradientUtils::getIndex(
    std::pair<Instruction *, CacheType> idx,
    const std::map<std::pair<Instruction *, CacheType>, int> &mapping) {
  assert(tape);
  auto found = mapping.find(idx);
  if (found == mapping.end()) {
    errs() << "oldFunc: " << *oldFunc << "\n";
    errs() << "newFunc: " << *newFunc << "\n";
    errs() << " <mapping>\n";
    for (auto &p : mapping) {
      errs() << "   idx: " << *p.first.first << ", " << p.first.second
             << " pos=" << p.second << "\n";
    }
    errs() << " </mapping>\n";

    errs() << "idx: " << *idx.first << ", " << idx.second << "\n";
    assert(0 && "could not find index in mapping");
  }
  return found->second;
}

int GradientUtils::getIndex(
    std::pair<Instruction *, CacheType> idx,
    std::map<std::pair<Instruction *, CacheType>, int> &mapping) {
  if (tape) {
    return getIndex(
        idx,
        (const std::map<std::pair<Instruction *, CacheType>, int> &)mapping);
  } else {
    if (mapping.find(idx) != mapping.end()) {
      return mapping[idx];
    }
    mapping[idx] = tapeidx;
    ++tapeidx;
    return mapping[idx];
  }
}

void GradientUtils::computeGuaranteedFrees() {
  SmallPtrSet<CallInst *, 2> allocsToPromote;
  for (auto &BB : *oldFunc) {
    if (notForAnalysis.count(&BB))
      continue;
    for (auto &I : BB) {
      if (auto AI = dyn_cast<AllocaInst>(&I))
        computeForwardingProperties(AI);

      auto CI = dyn_cast<CallInst>(&I);
      if (!CI)
        continue;

      StringRef funcName = getFuncNameFromCall(CI);

      if (isDeallocationFunction(funcName, TLI)) {
        llvm::Value *val = getBaseObject(CI->getArgOperand(0));

        if (auto dc = dyn_cast<CallInst>(val)) {
          StringRef sfuncName = getFuncNameFromCall(dc);
          if (isAllocationFunction(sfuncName, TLI)) {

            bool hasPDFree = false;
            if (dc->getParent() == CI->getParent() ||
                OrigPDT.dominates(CI->getParent(), dc->getParent())) {
              hasPDFree = true;
            }

            if (hasPDFree) {
              allocationsWithGuaranteedFree[dc].insert(CI);
            }
          }
        }
      }
      if (isAllocationFunction(funcName, TLI)) {
        allocsToPromote.insert(CI);
        if (hasMetadata(CI, "enzyme_fromstack")) {
          allocationsWithGuaranteedFree[CI].insert(CI);
        }
        if (funcName == "jl_alloc_array_1d" ||
            funcName == "jl_alloc_array_2d" ||
            funcName == "jl_alloc_array_3d" || funcName == "jl_array_copy" ||
            funcName == "ijl_alloc_array_1d" ||
            funcName == "ijl_alloc_array_2d" ||
            funcName == "ijl_alloc_array_3d" || funcName == "ijl_array_copy" ||
            funcName == "julia.gc_alloc_obj" ||
            funcName == "jl_gc_alloc_typed" ||
            funcName == "ijl_gc_alloc_typed") {
        }
      }
    }
  }
  for (CallInst *V : allocsToPromote) {
    // TODO compute if an only load/store (non capture)
    // allocaion by traversing its users. If so, mark
    // all of its load/stores, as now the loads can
    // potentially be rematerialized without a cache
    // of the allocation, but the operands of all stores.
    // This info needs to be provided to minCutCache
    // the derivative of store needs to redo the store,
    // isValueNeededInReverse needs to know to preserve the
    // store operands in this case, etc
    computeForwardingProperties(V);
  }
}

/// Perform the corresponding deallocation of tofree, given it was allocated by
/// allocationfn
// For updating below one should read MemoryBuiltins.cpp, TargetLibraryInfo.cpp
llvm::CallInst *freeKnownAllocation(llvm::IRBuilder<> &builder,
                                    llvm::Value *tofree,
                                    const llvm::StringRef allocationfn,
                                    const llvm::DebugLoc &debuglocation,
                                    const llvm::TargetLibraryInfo &TLI,
                                    llvm::CallInst *orig,
                                    GradientUtils *gutils) {
  assert(isAllocationFunction(allocationfn, TLI));

  if (allocationfn == "__rust_alloc" || allocationfn == "__rust_alloc_zeroed") {
    llvm_unreachable("todo - hook in rust allocation fns");
  }
  if (allocationfn == "julia.gc_alloc_obj" ||
      allocationfn == "jl_gc_alloc_typed" ||
      allocationfn == "ijl_gc_alloc_typed")
    return nullptr;

  if (allocationfn == "enzyme_allocator") {
    auto inds = getDeallocationIndicesFromCall(orig);
    SmallVector<Value *, 2> vals;
    for (auto ind : inds) {
      if (ind == -1)
        vals.push_back(tofree);
      else
        vals.push_back(gutils->lookupM(
            gutils->getNewFromOriginal(orig->getArgOperand(ind)), builder));
    }
    auto tocall = getDeallocatorFnFromCall(orig);
    auto freecall = builder.CreateCall(tocall, vals);
    freecall->setDebugLoc(debuglocation);
    return freecall;
  }

  if (allocationfn == "swift_allocObject") {
    Type *VoidTy = Type::getVoidTy(tofree->getContext());
    Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

    auto FT = FunctionType::get(VoidTy, ArrayRef<Type *>(IntPtrTy), false);
    Value *freevalue = builder.GetInsertBlock()
                           ->getParent()
                           ->getParent()
                           ->getOrInsertFunction("swift_release", FT)
                           .getCallee();
    CallInst *freecall = cast<CallInst>(CallInst::Create(
        FT, freevalue,
        ArrayRef<Value *>(builder.CreatePointerCast(tofree, IntPtrTy)), "",
        builder.GetInsertBlock()));
    freecall->setDebugLoc(debuglocation);
    if (isa<CallInst>(tofree) &&
        cast<CallInst>(tofree)->getAttributes().hasAttribute(
            AttributeList::ReturnIndex, Attribute::NonNull)) {
      freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
    }
    if (Function *F = dyn_cast<Function>(freevalue))
      freecall->setCallingConv(F->getCallingConv());
    if (freecall->getParent() == nullptr)
      builder.Insert(freecall);
    return freecall;
  }

  if (shadowErasers.find(allocationfn) != shadowErasers.end()) {
    return shadowErasers[allocationfn](builder, tofree);
  }

  if (tofree->getType()->isIntegerTy())
    tofree = builder.CreateIntToPtr(tofree,
                                    Type::getInt8PtrTy(tofree->getContext()));

  llvm::LibFunc libfunc;
  if (allocationfn == "calloc" || allocationfn == "malloc") {
    libfunc = LibFunc_malloc;
  } else {
    bool res = TLI.getLibFunc(allocationfn, libfunc);
    assert(res && "ought find known allocation fn");
  }

  llvm::LibFunc freefunc;

  switch (libfunc) {
  case LibFunc_malloc: // malloc(unsigned int);
  case LibFunc_valloc: // valloc(unsigned int);
    freefunc = LibFunc_free;
    break;

  case LibFunc_Znwj:                // new(unsigned int);
  case LibFunc_ZnwjRKSt9nothrow_t:  // new(unsigned int, nothrow);
  case LibFunc_ZnwjSt11align_val_t: // new(unsigned int, align_val_t)
  case LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t: // new(unsigned int,
                                                  // align_val_t, nothrow)

  case LibFunc_Znwm:                // new(unsigned long);
  case LibFunc_ZnwmRKSt9nothrow_t:  // new(unsigned long, nothrow);
  case LibFunc_ZnwmSt11align_val_t: // new(unsigned long, align_val_t)
  case LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t: // new(unsigned long,
                                                  // align_val_t, nothrow)
    freefunc = LibFunc_ZdlPv;
    break;

  case LibFunc_Znaj:                // new[](unsigned int);
  case LibFunc_ZnajRKSt9nothrow_t:  // new[](unsigned int, nothrow);
  case LibFunc_ZnajSt11align_val_t: // new[](unsigned int, align_val_t)
  case LibFunc_ZnajSt11align_val_tRKSt9nothrow_t: // new[](unsigned int,
                                                  // align_val_t, nothrow

  case LibFunc_Znam:                // new[](unsigned long);
  case LibFunc_ZnamRKSt9nothrow_t:  // new[](unsigned long, nothrow);
  case LibFunc_ZnamSt11align_val_t: // new[](unsigned long, align_val_t)
  case LibFunc_ZnamSt11align_val_tRKSt9nothrow_t: // new[](unsigned long,
                                                  // align_val_t, nothrow)
    freefunc = LibFunc_ZdaPv;
    break;

  case LibFunc_msvc_new_int:               // new(unsigned int);
  case LibFunc_msvc_new_int_nothrow:       // new(unsigned int, nothrow);
  case LibFunc_msvc_new_longlong:          // new(unsigned long long);
  case LibFunc_msvc_new_longlong_nothrow:  // new(unsigned long long, nothrow);
  case LibFunc_msvc_new_array_int:         // new[](unsigned int);
  case LibFunc_msvc_new_array_int_nothrow: // new[](unsigned int, nothrow);
  case LibFunc_msvc_new_array_longlong:    // new[](unsigned long long);
  case LibFunc_msvc_new_array_longlong_nothrow: // new[](unsigned long long,
                                                // nothrow);
    llvm_unreachable("msvc deletion not handled");

  default:
    llvm_unreachable("unknown allocation function");
  }
  llvm::StringRef freename = TLI.getName(freefunc);
  if (freefunc == LibFunc_free) {
    freename = "free";
    assert(freename == "free");
    if (freename != "free")
      llvm_unreachable("illegal free");
  }

  Type *VoidTy = Type::getVoidTy(tofree->getContext());
  Type *IntPtrTy = Type::getInt8PtrTy(tofree->getContext());

  auto FT = FunctionType::get(VoidTy, {IntPtrTy}, false);
  Value *freevalue = builder.GetInsertBlock()
                         ->getParent()
                         ->getParent()
                         ->getOrInsertFunction(freename, FT)
                         .getCallee();
  CallInst *freecall = cast<CallInst>(CallInst::Create(
      FT, freevalue, {builder.CreatePointerCast(tofree, IntPtrTy)}, "",
      builder.GetInsertBlock()));
  freecall->setDebugLoc(debuglocation);
  if (isa<CallInst>(tofree) &&
      cast<CallInst>(tofree)->getAttributes().hasAttribute(
          AttributeList::ReturnIndex, Attribute::NonNull)) {
    freecall->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
  }
  if (Function *F = dyn_cast<Function>(freevalue))
    freecall->setCallingConv(F->getCallingConv());
  if (freecall->getParent() == nullptr)
    builder.Insert(freecall);
  return freecall;
}

bool GradientUtils::needsCacheWholeAllocation(
    const llvm::Value *origInst) const {
  auto found = knownRecomputeHeuristic.find(origInst);
  if (found == knownRecomputeHeuristic.end())
    return false;
  if (!found->second)
    return true;
  SmallVector<std::pair<const Instruction *, size_t>, 1> todo;
  for (auto &use : origInst->uses())
    todo.push_back(
        std::make_pair(cast<Instruction>(use.getUser()), use.getOperandNo()));
  SmallSet<std::pair<const Instruction *, size_t>, 1> seen;
  while (todo.size()) {
    auto pair = todo.back();
    auto [cur, idx] = pair;
    todo.pop_back();
    if (seen.count(pair))
      continue;
    seen.insert(pair);
    // Loads are always fine
    if (isa<LoadInst>(cur))
      continue;

    if (auto II = dyn_cast<IntrinsicInst>(cur))
      if (II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_f ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_i ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_p ||
          II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_f ||
          II->getIntrinsicID() == Intrinsic::masked_load)
        continue;

    if (auto CI = dyn_cast<CallInst>(cur)) {
#if LLVM_VERSION_MAJOR >= 14
      if (idx < CI->arg_size())
#else
      if (idx < CI->getNumArgOperands())
#endif
      {
        if (isNoCapture(CI, idx))
          continue;
      }
    }

    found = knownRecomputeHeuristic.find(cur);
    if (found == knownRecomputeHeuristic.end())
      continue;

    // If caching this user, it cannot be a gep/cast of original
    if (!found->second) {
      assert(false && "caching potentially capturing/offset of allocation");
    } else {
      // if not caching this user, it is legal to recompute, consider its users
      for (auto &use : cur->uses()) {
        todo.push_back(std::make_pair(cast<Instruction>(use.getUser()),
                                      use.getOperandNo()));
      }
    }
  }
  return false;
}

void GradientUtils::replaceAndRemoveUnwrapCacheFor(llvm::Value *A,
                                                   llvm::Value *B) {
  SmallVector<Instruction *, 1> toErase;
  for (auto &pair : unwrap_cache) {
    auto found = pair.second.find(A);
    if (found != pair.second.end()) {
      for (auto &p : found->second) {
        Value *pre = p.second;
        replaceAWithB(pre, B);
        if (auto I = dyn_cast<Instruction>(pre)) {
          toErase.push_back(I);
        }
      }
      pair.second.erase(A);
    }
  }
  for (auto I : toErase) {
    erase(I);
  }
}
