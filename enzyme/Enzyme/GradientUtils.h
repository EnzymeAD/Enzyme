//===- GradientUtils.h - Helper class and utilities for AD       ---------===//
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
// This file declares two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_GUTILS_H_
#define ENZYME_GUTILS_H_

#include <algorithm>
#include <deque>
#include <map>

#include <llvm/Config/llvm-config.h>

#include "ActivityAnalysis.h"
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#include "Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "llvm/IR/Dominators.h"

#include "MustExitScalarEvolution.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Support/Casting.h"

#include "llvm/Transforms/Utils/ValueMapper.h"

#include "ActivityAnalysis.h"
#include "CacheUtility.h"
#include "EnzymeLogic.h"
#include "LibraryFuncs.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

#include "llvm-c/Core.h"

extern std::map<std::string, std::function<llvm::Value *(
                                 IRBuilder<> &, CallInst *, ArrayRef<Value *>)>>
    shadowHandlers;

class GradientUtils;
class DiffeGradientUtils;
extern std::map<
    std::string,
    std::pair<std::function<void(llvm::IRBuilder<> &, llvm::CallInst *,
                                 GradientUtils &, llvm::Value *&,
                                 llvm::Value *&, llvm::Value *&)>,
              std::function<void(llvm::IRBuilder<> &, llvm::CallInst *,
                                 DiffeGradientUtils &, llvm::Value *)>>>
    customCallHandlers;

extern "C" {
extern llvm::cl::opt<bool> EnzymeInactiveDynamic;
}

enum class AugmentedStruct;
class GradientUtils : public CacheUtility {
public:
  EnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  llvm::Function *oldFunc;
  ValueToValueMapTy invertedPointers;
  DominatorTree &OrigDT;
  PostDominatorTree &OrigPDT;
  LoopInfo &OrigLI;
  ScalarEvolution &OrigSE;
  std::shared_ptr<ActivityAnalyzer> ATA;
  SmallVector<BasicBlock *, 12> originalBlocks;
  std::map<BasicBlock *, std::vector<BasicBlock *>> reverseBlocks;
  SmallPtrSet<PHINode *, 4> fictiousPHIs;
  ValueToValueMapTy originalToNewFn;
  std::vector<CallInst *> originalCalls;

  SmallPtrSet<Instruction *, 4> unnecessaryIntermediates;

  const std::map<Instruction *, bool> *can_modref_map;

  Value *getNewIfOriginal(Value *originst) const {
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

  Value *getOrInsertTotalMultiplicativeProduct(Value *val, LoopContext &lc) {
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
          if (!C->isExactlyValue(
                  APFloat(C->getType()->getFltSemantics(), "1"))) {
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

  Value *getOrInsertConditionalIndex(Value *val, LoopContext &lc,
                                     bool pickTrue) {
    assert(val->getType()->isIntOrIntVectorTy(1));
    // TODO optimize if val is invariant to loopContext
    for (auto &I : *lc.header) {
      if (auto PN = dyn_cast<PHINode>(&I)) {
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

          if (auto SI =
                  dyn_cast<SelectInst>(PN->getIncomingValueForBlock(IB))) {
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

  bool assumeDynamicLoopOfSizeOne(llvm::Loop *L) const override {
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

  void setupOMPFor() {
    for (auto &BB : *oldFunc) {
      for (auto &I : BB) {
        if (CallInst *call = dyn_cast<CallInst>(&I)) {
          if (Function *F = call->getCalledFunction()) {
            if (F->getName() == "__kmpc_for_static_init_4" ||
                F->getName() == "__kmpc_for_static_init_4u" ||
                F->getName() == "__kmpc_for_static_init_8" ||
                F->getName() == "__kmpc_for_static_init_8u") {
              // todo what if bounds change between fwd/reverse
              IRBuilder<> pre(getNewFromOriginal(call));
              IntegerType *i64 = IntegerType::getInt64Ty(oldFunc->getContext());
              Value *lb = nullptr;
              for (auto u : call->getArgOperand(4)->users()) {
                if (auto si = dyn_cast<StoreInst>(u)) {
                  if (OrigDT.dominates(si, call)) {
                    lb = pre.CreateSExtOrTrunc(
                        getNewFromOriginal(si->getValueOperand()), i64);
                    break;
                  }
                }
              }
              assert(lb);
              Value *ub = nullptr;
              for (auto u : call->getArgOperand(5)->users()) {
                if (auto si = dyn_cast<StoreInst>(u)) {
                  if (OrigDT.dominates(si, call)) {
                    ub = pre.CreateSExtOrTrunc(
                        getNewFromOriginal(si->getValueOperand()), i64);
                    break;
                  }
                }
              }
              assert(ub);
              IRBuilder<> post(getNewFromOriginal(call)->getNextNode());
              auto lb_post = post.CreateSExtOrTrunc(
                  post.CreateLoad(getNewFromOriginal(call->getArgOperand(4))),
                  i64);
              ompOffset = post.CreateSub(lb_post, lb, "", true, true);
              ompTrueLimit = pre.CreateSub(ub, lb);
              return;
            }
          }
        }
      }
    }
    llvm::errs() << *oldFunc << "\n";
    assert(0 && "could not find openmp init");
    // ompOffset;
    // ompTrueLimit;
  }

  llvm::DebugLoc getNewFromOriginal(const llvm::DebugLoc L) const {
    if (L.get() == nullptr)
      return nullptr;
    if (!oldFunc->getSubprogram())
      return L;
    assert(originalToNewFn.hasMD());
    auto opt = originalToNewFn.getMappedMD(L.getAsMDNode());
    if (!opt.hasValue())
      return L;
    assert(opt.hasValue());
    return llvm::DebugLoc(cast<MDNode>(*opt.getPointer()));
  }

  Value *getNewFromOriginal(const Value *originst) const {
    assert(originst);
    auto f = originalToNewFn.find(originst);
    if (f == originalToNewFn.end()) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
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
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *originst << "\n";
    }
    assert(f->second);
    return f->second;
  }
  Instruction *getNewFromOriginal(const Instruction *newinst) const {
    auto ninst = getNewFromOriginal((Value *)newinst);
    if (!isa<Instruction>(ninst)) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *ninst << " - " << *newinst << "\n";
    }
    return cast<Instruction>(ninst);
  }
  BasicBlock *getNewFromOriginal(const BasicBlock *newinst) const {
    return cast<BasicBlock>(getNewFromOriginal((Value *)newinst));
  }

  Value *hasUninverted(const Value *inverted) const {
    for (auto v : invertedPointers) {
      if (v.second == inverted)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }
  BasicBlock *getOriginalFromNew(const BasicBlock *newinst) const {
    assert(newinst->getParent() == newFunc);
    for (auto &B : *oldFunc) {

      auto f = originalToNewFn.find(&B);
      assert(f != originalToNewFn.end());
      if (f->second == newinst)
        return &B;
    }
    llvm_unreachable("could not find original block");
  }

  Value *isOriginal(const Value *newinst) const {
    if (isa<Constant>(newinst) || isa<UndefValue>(newinst))
      return const_cast<Value *>(newinst);
    if (auto arg = dyn_cast<Argument>(newinst)) {
      assert(arg->getParent() == newFunc);
    }
    if (auto inst = dyn_cast<Instruction>(newinst)) {
      assert(inst->getParent()->getParent() == newFunc);
    }
    for (auto v : originalToNewFn) {
      if (v.second == newinst)
        return const_cast<Value *>(v.first);
    }
    return nullptr;
  }

  Instruction *isOriginal(const Instruction *newinst) const {
    return cast_or_null<Instruction>(isOriginal((const Value *)newinst));
  }
  BasicBlock *isOriginal(const BasicBlock *newinst) const {
    return cast_or_null<BasicBlock>(isOriginal((const Value *)newinst));
  }

private:
  SmallVector<Value *, 4> addedTapeVals;
  unsigned tapeidx;
  Value *tape;

  std::map<BasicBlock *, std::map<std::pair<Value *, BasicBlock *>, Value *>>
      unwrap_cache;
  std::map<BasicBlock *, std::map<Value *, Value *>> lookup_cache;

public:
  BasicBlock *addReverseBlock(BasicBlock *currentBlock, Twine name,
                              bool forkCache = true) {
    assert(reverseBlocks.size());

    // todo speed this up
    for (auto &pair : reverseBlocks) {
      std::vector<BasicBlock *> &vec = pair.second;
      if (vec.back() == currentBlock) {

        BasicBlock *rev =
            BasicBlock::Create(currentBlock->getContext(), name, newFunc);
        rev->moveAfter(currentBlock);
        vec.push_back(rev);
        if (forkCache) {
          unwrap_cache[rev] = unwrap_cache[currentBlock];
          lookup_cache[rev] = lookup_cache[currentBlock];
        }
        return rev;
      }
    }
    assert(0 && "cannot find reverse location to add into");
    llvm_unreachable("cannot find reverse location to add into");
  }

public:
  bool legalRecompute(const Value *val, const ValueToValueMapTy &available,
                      IRBuilder<> *BuilderM, bool reverse = false,
                      bool legalRecomputeCache = true) const;
  std::map<const Value *, bool> knownRecomputeHeuristic;
  bool shouldRecompute(const Value *val, const ValueToValueMapTy &available,
                       IRBuilder<> *BuilderM);

  ValueToValueMapTy unwrappedLoads;
  void replaceAWithB(Value *A, Value *B, bool storeInCache = false) override {
    if (A == B)
      return;
    assert(A->getType() == B->getType());
    for (unsigned i = 0; i < addedTapeVals.size(); ++i) {
      if (addedTapeVals[i] == A) {
        addedTapeVals[i] = B;
      }
    }
    for (auto pair : unwrappedLoads) {
      if (pair->second == A)
        pair->second = B;
    }
    if (unwrappedLoads.find(A) != unwrappedLoads.end()) {
      unwrappedLoads[B] = unwrappedLoads[A];
      unwrappedLoads.erase(A);
    }

    if (invertedPointers.find(A) != invertedPointers.end()) {
      invertedPointers[B] = invertedPointers[A];
      invertedPointers.erase(A);
    }
    if (auto orig = isOriginal(A)) {
      originalToNewFn[orig] = B;
    }

    CacheUtility::replaceAWithB(A, B, storeInCache);
  }

  void erase(Instruction *I) override {
    assert(I);
    invertedPointers.erase(I);
    originalToNewFn.erase(I);
    unwrappedLoads.erase(I);
  eraser:
    for (auto v : originalToNewFn) {
      if (v.second == I) {
        originalToNewFn.erase(v.first);
        goto eraser;
      }
    }
    for (auto v : invertedPointers) {
      if (v.second == I) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        dumpPointers();
        llvm::errs() << *v.first << "\n";
        llvm::errs() << *I << "\n";
        assert(0 && "erasing something in invertedPointers map");
      }
    }
    for (auto v : unwrappedLoads) {
      if (v.second == I) {
        assert(0 && "erasing something in unwrappedLoads map");
      }
    }

    for (auto &pair : unwrap_cache) {
      std::vector<std::pair<Value *, BasicBlock *>> cache_pairs;
      for (auto &a : pair.second) {
        if (a.second == I) {
          cache_pairs.push_back(a.first);
        }
        if (a.first.first == I) {
          cache_pairs.push_back(a.first);
        }
      }
      for (auto a : cache_pairs) {
        pair.second.erase(a);
      }
    }

    for (auto &pair : lookup_cache) {
      std::vector<Value *> cache_pairs;
      for (auto &a : pair.second) {
        if (a.second == I) {
          cache_pairs.push_back(a.first);
        }
        if (a.first == I) {
          cache_pairs.push_back(a.first);
        }
      }
      for (auto a : cache_pairs) {
        pair.second.erase(a);
      }
    }
    CacheUtility::erase(I);
  }
  // TODO consider invariant group and/or valueInvariant group

  void setTape(Value *newtape) {
    assert(tape == nullptr);
    assert(newtape != nullptr);
    assert(tapeidx == 0);
    assert(addedTapeVals.size() == 0);
    tape = newtape;
  }

  void dumpPointers() {
    llvm::errs() << "invertedPointers:\n";
    for (auto a : invertedPointers) {
      llvm::errs() << "   invertedPointers[" << *a.first << "] = " << *a.second
                   << "\n";
    }
    llvm::errs() << "end invertedPointers\n";
  }

  Value *createAntiMalloc(CallInst *orig, unsigned idx) {
    assert(orig->getParent()->getParent() == oldFunc);
    PHINode *placeholder = cast<PHINode>(invertedPointers[orig]);

    assert(placeholder->getParent()->getParent() == newFunc);
    placeholder->setName("");
    IRBuilder<> bb(placeholder);

    SmallVector<Value *, 8> args;
    for (unsigned i = 0; i < orig->getNumArgOperands(); ++i) {
      args.push_back(getNewFromOriginal(orig->getArgOperand(i)));
    }

    if (shadowHandlers.find(orig->getCalledFunction()->getName().str()) !=
        shadowHandlers.end()) {
      bb.SetInsertPoint(placeholder);
      Value *anti = shadowHandlers[orig->getCalledFunction()->getName().str()](
          bb, orig, args);
      invertedPointers[orig] = anti;
      // assert(placeholder != anti);

      bb.SetInsertPoint(placeholder);

      replaceAWithB(placeholder, anti);
      erase(placeholder);

      if (auto inst = dyn_cast<Instruction>(anti))
        bb.SetInsertPoint(inst);

      anti = cacheForReverse(bb, anti, idx);
      invertedPointers[orig] = anti;
      return anti;
    }

    Value *anti =
        bb.CreateCall(orig->getCalledFunction(), args, orig->getName() + "'mi");
    cast<CallInst>(anti)->setAttributes(orig->getAttributes());
    cast<CallInst>(anti)->setCallingConv(orig->getCallingConv());
    cast<CallInst>(anti)->setTailCallKind(orig->getTailCallKind());
    cast<CallInst>(anti)->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));

    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NoAlias);
    cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                       Attribute::NonNull);

    unsigned derefBytes = 0;
    if (orig->getCalledFunction()->getName() == "malloc" ||
        orig->getCalledFunction()->getName() == "_Znwm") {
      if (auto ci = dyn_cast<ConstantInt>(args[0])) {
        derefBytes = ci->getLimitedValue();
        cast<CallInst>(anti)->addDereferenceableAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        cast<CallInst>(anti)->addDereferenceableOrNullAttr(
            llvm::AttributeList::ReturnIndex, ci->getLimitedValue());
        CallInst *cal = cast<CallInst>(getNewFromOriginal(orig));
        cal->addDereferenceableAttr(llvm::AttributeList::ReturnIndex,
                                    ci->getLimitedValue());
        cal->addDereferenceableOrNullAttr(llvm::AttributeList::ReturnIndex,
                                          ci->getLimitedValue());
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NoAlias);
        cal->addAttribute(AttributeList::ReturnIndex, Attribute::NonNull);
      }
    }

    invertedPointers[orig] = anti;
    // assert(placeholder != anti);
    bb.SetInsertPoint(placeholder->getNextNode());
    replaceAWithB(placeholder, anti);
    erase(placeholder);

    anti = cacheForReverse(bb, anti, idx);
    invertedPointers[orig] = anti;

    if (tape == nullptr) {
      if (orig->getCalledFunction()->getName() == "julia.gc_alloc_obj") {
        Type *tys[] = {
            PointerType::get(StructType::get(orig->getContext()), 10)};
        FunctionType *FT =
            FunctionType::get(Type::getVoidTy(orig->getContext()), tys, true);
        bb.CreateCall(oldFunc->getParent()->getOrInsertFunction(
                          "julia.write_barrier", FT),
                      anti);
        if (mode != DerivativeMode::ReverseModeCombined) {
          EmitFailure("SplitGCAllocation", orig->getDebugLoc(), orig,
                      "Not handling Julia shadow GC allocation in split mode ",
                      *orig);
          return anti;
        }
      }

      if (orig->getCalledFunction()->getName() == "swift_allocObject") {
        EmitFailure(
            "SwiftShadowAllocation", orig->getDebugLoc(), orig,
            "Haven't implemented shadow allocator for `swift_allocObject`",
            *orig);
        return anti;
      }

      Value *dst_arg = anti;

      dst_arg = bb.CreateBitCast(
          dst_arg,
          Type::getInt8PtrTy(orig->getContext(),
                             anti->getType()->getPointerAddressSpace()));

      auto val_arg = ConstantInt::get(Type::getInt8Ty(orig->getContext()), 0);
      Value *size;
      // todo check if this memset is legal and if a write barrier is needed
      if (orig->getCalledFunction()->getName() == "julia.gc_alloc_obj") {
        size = args[1];
      } else {
        size = args[0];
      }
      auto len_arg =
          bb.CreateZExtOrTrunc(size, Type::getInt64Ty(orig->getContext()));
      auto volatile_arg = ConstantInt::getFalse(orig->getContext());

#if LLVM_VERSION_MAJOR == 6
      auto align_arg =
          ConstantInt::get(Type::getInt32Ty(orig->getContext()), 1);
      Value *nargs[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
      Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif

      Type *tys[] = {dst_arg->getType(), len_arg->getType()};

      auto memset = cast<CallInst>(
          bb.CreateCall(Intrinsic::getDeclaration(newFunc->getParent(),
                                                  Intrinsic::memset, tys),
                        nargs));
      // memset->addParamAttr(0, Attribute::getWithAlignment(Context,
      // inst->getAlignment()));
      memset->addParamAttr(0, Attribute::NonNull);
      if (derefBytes) {
        memset->addDereferenceableAttr(llvm::AttributeList::FirstArgIndex,
                                       derefBytes);
        memset->addDereferenceableOrNullAttr(llvm::AttributeList::FirstArgIndex,
                                             derefBytes);
      }
    }

    return anti;
  }

  int getIndex(std::pair<Instruction *, CacheType> idx,
               std::map<std::pair<Instruction *, CacheType>, int> &mapping) {
    if (tape) {
      if (mapping.find(idx) == mapping.end()) {
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << " <mapping>\n";
        for (auto &p : mapping) {
          llvm::errs() << "   idx: " << *p.first.first << ", " << p.first.second
                       << " pos=" << p.second << "\n";
        }
        llvm::errs() << " </mapping>\n";

        if (mapping.find(idx) == mapping.end()) {
          llvm::errs() << "idx: " << *idx.first << ", " << idx.second << "\n";
          assert(0 && "could not find index in mapping");
        }
      }
      return mapping[idx];
    } else {
      if (mapping.find(idx) != mapping.end()) {
        return mapping[idx];
      }
      mapping[idx] = tapeidx;
      ++tapeidx;
      return mapping[idx];
    }
  }

  Value *cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc, int idx,
                         bool ignoreType = false);

  const SmallVectorImpl<Value *> &getTapeValues() const {
    return addedTapeVals;
  }

public:
  AAResults &OrigAA;
  TypeAnalysis &TA;
  GradientUtils(EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
                TargetLibraryInfo &TLI_, TypeAnalysis &TA_,
                ValueToValueMapTy &invertedPointers_,
                const SmallPtrSetImpl<Value *> &constantvalues_,
                const SmallPtrSetImpl<Value *> &activevals_, bool ActiveReturn,
                ValueToValueMapTy &originalToNewFn_, DerivativeMode mode)
      : CacheUtility(TLI_, newFunc_), Logic(Logic), mode(mode),
        oldFunc(oldFunc_), invertedPointers(),
        OrigDT(Logic.PPC.FAM.getResult<llvm::DominatorTreeAnalysis>(*oldFunc_)),
        OrigPDT(Logic.PPC.FAM.getResult<llvm::PostDominatorTreeAnalysis>(
            *oldFunc_)),
        OrigLI(Logic.PPC.FAM.getResult<llvm::LoopAnalysis>(*oldFunc_)),
        OrigSE(
            Logic.PPC.FAM.getResult<llvm::ScalarEvolutionAnalysis>(*oldFunc_)),
        ATA(new ActivityAnalyzer(
            Logic.PPC, Logic.PPC.getAAResultsFromFunction(oldFunc_), TLI_,
            constantvalues_, activevals_, ActiveReturn)),
        OrigAA(Logic.PPC.getAAResultsFromFunction(oldFunc_)), TA(TA_) {
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
#if LLVM_VERSION_MAJOR <= 6
    OrigPDT.recalculate(*oldFunc_);
#endif
    invertedPointers.insert(invertedPointers_.begin(), invertedPointers_.end());
    originalToNewFn.insert(originalToNewFn_.begin(), originalToNewFn_.end());
    for (BasicBlock &BB : *newFunc) {
      originalBlocks.emplace_back(&BB);
    }
    tape = nullptr;
    tapeidx = 0;
    assert(originalBlocks.size() > 0);
  }

public:
  static GradientUtils *
  CreateFromClone(EnzymeLogic &Logic, Function *todiff, TargetLibraryInfo &TLI,
                  TypeAnalysis &TA, DIFFE_TYPE retType,
                  const std::vector<DIFFE_TYPE> &constant_args, bool returnUsed,
                  std::map<AugmentedStruct, int> &returnMapping);

  StoreInst *setPtrDiffe(Value *ptr, Value *newval, IRBuilder<> &BuilderM) {
    if (auto inst = dyn_cast<Instruction>(ptr)) {
      assert(inst->getParent()->getParent() == oldFunc);
    }
    if (auto arg = dyn_cast<Argument>(ptr)) {
      assert(arg->getParent() == oldFunc);
    }
    ptr = invertPointerM(ptr, BuilderM);
    return BuilderM.CreateStore(newval, ptr);
  }

private:
  BasicBlock *originalForReverseBlock(BasicBlock &BB2) const {
    assert(reverseBlocks.size() != 0);
    for (auto BB : originalBlocks) {
      auto it = reverseBlocks.find(BB);
      assert(it != reverseBlocks.end());
      if (std::find(it->second.begin(), it->second.end(), &BB2) !=
          it->second.end()) {
        return BB;
      }
    }
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << BB2 << "\n";
    assert(0 && "could not find original block for given reverse block");
    report_fatal_error("could not find original block for given reverse block");
  }

public:
  //! This cache stores blocks we may insert as part of getReverseOrLatchMerge
  //! to handle inverse iv iteration
  //  As we don't want to create redundant blocks, we use this convenient cache
  std::map<std::tuple<BasicBlock *, BasicBlock *>, BasicBlock *>
      newBlocksForLoop_cache;
  BasicBlock *getReverseOrLatchMerge(BasicBlock *BB,
                                     BasicBlock *branchingBlock);

  void forceContexts();

  void
  computeMinCache(TypeResults &TR,
                  const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable);

  bool isOriginalBlock(const BasicBlock &BB) const {
    for (auto A : originalBlocks) {
      if (A == &BB)
        return true;
    }
    return false;
  }

  void eraseFictiousPHIs() {
    for (auto pp : fictiousPHIs) {
      if (pp->getNumUses() != 0) {
        llvm::errs() << "mod:" << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc:" << *newFunc << "\n";
        llvm::errs() << " pp: " << *pp << "\n";
      }
      assert(pp->getNumUses() == 0);
      pp->replaceAllUsesWith(UndefValue::get(pp->getType()));
      erase(pp);
    }
    fictiousPHIs.clear();
  }

  TypeResults *my_TR;
  void forceActiveDetection(TypeResults &TR) {
    my_TR = &TR;
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

  bool isConstantValue(Value *val) const {
    if (auto inst = dyn_cast<Instruction>(val)) {
      assert(inst->getParent()->getParent() == oldFunc);
      return ATA->isConstantValue(*my_TR, val);
    }

    if (auto arg = dyn_cast<Argument>(val)) {
      assert(arg->getParent() == oldFunc);
      return ATA->isConstantValue(*my_TR, val);
    }

    //! Functions must be false so we can replace function with augmentation,
    //! fallback to analysis
    if (isa<Function>(val) || isa<InlineAsm>(val) || isa<Constant>(val) ||
        isa<UndefValue>(val) || isa<MetadataAsValue>(val)) {
      // llvm::errs() << "calling icv on: " << *val << "\n";
      return ATA->isConstantValue(*my_TR, val);
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

  bool isConstantInstruction(const Instruction *inst) const {
    assert(inst->getParent()->getParent() == oldFunc);
    return ATA->isConstantInstruction(*my_TR, const_cast<Instruction *>(inst));
  }

  bool getContext(llvm::BasicBlock *BB, LoopContext &lc) {
    return CacheUtility::getContext(BB, lc,
                                    /*ReverseLimit*/ reverseBlocks.size() > 0);
  }

  void forceAugmentedReturns(
      TypeResults &TR,
      const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
    assert(TR.info.Function == oldFunc);

    for (BasicBlock &oBB : *oldFunc) {
      // Don't create derivatives for code that results in termination
      if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end())
        continue;

      LoopContext loopContext;
      getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

      for (Instruction &I : oBB) {
        Instruction *inst = &I;

        if (inst->getType()->isEmptyTy())
          continue;

        if (inst->getType()->isFPOrFPVectorTy())
          continue; //! op->getType()->isPointerTy() &&
                    //! !op->getType()->isIntegerTy()) {

        if (!TR.query(inst).Inner0().isPossiblePointer())
          continue;

        if (isa<LoadInst>(inst)) {
          IRBuilder<> BuilderZ(inst);
          getForwardBuilder(BuilderZ);

          PHINode *anti = BuilderZ.CreatePHI(inst->getType(), 1,
                                             inst->getName() + "'il_phi");
          invertedPointers[inst] = anti;
          continue;
        }

        if (!isa<CallInst>(inst)) {
          continue;
        }

        if (isa<IntrinsicInst>(inst)) {
          continue;
        }

        if (isConstantValue(inst)) {
          continue;
        }

        CallInst *op = cast<CallInst>(inst);
        Function *called = op->getCalledFunction();

        if (called && isCertainPrintOrFree(called)) {
          continue;
        }

        IRBuilder<> BuilderZ(inst);
        getForwardBuilder(BuilderZ);

        PHINode *anti =
            BuilderZ.CreatePHI(op->getType(), 1, op->getName() + "'ip_phi");
        invertedPointers[inst] = anti;

        if (called && isAllocationFunction(*called, TLI)) {
          invertedPointers[inst]->setName(op->getName() + "'mi");
        }
      }
    }
  }

  /// if full unwrap, don't just unwrap this instruction, but also its operands,
  /// etc
  Value *unwrapM(Value *const val, IRBuilder<> &BuilderM,
                 const ValueToValueMapTy &available, UnwrapMode mode,
                 BasicBlock *scope = nullptr,
                 bool permitCache = true) override final;

  void ensureLookupCached(Instruction *inst, bool shouldFree = true) {
    assert(inst);
    if (scopeMap.find(inst) != scopeMap.end())
      return;
    if (shouldFree)
      assert(reverseBlocks.size());
    LimitContext lctx(/*ReverseLimit*/ reverseBlocks.size() > 0,
                      inst->getParent());

    AllocaInst *cache =
        createCacheForScope(lctx, inst->getType(), inst->getName(), shouldFree);
    assert(cache);
    Value *Val = inst;
    insert_or_assign(scopeMap, Val,
                     std::pair<AllocaInst *, LimitContext>(cache, lctx));
    storeInstructionInCache(lctx, inst, cache);
  }

  std::map<Instruction *, ValueMap<BasicBlock *, WeakTrackingVH>> lcssaFixes;
  std::map<PHINode *, WeakTrackingVH> lcssaPHIToOrig;
  Value *fixLCSSA(Instruction *inst, BasicBlock *forwardBlock,
                  bool mergeIfTrue = false, bool guaranteeVisible = true) {
    assert(inst->getName() != "<badref>");
    LoopContext lc;

    if (inst->getParent() == inversionAllocs)
      return inst;

    if (!isOriginalBlock(*forwardBlock)) {
      forwardBlock = originalForReverseBlock(*forwardBlock);
    }

    bool inLoop = getContext(inst->getParent(), lc);
    bool isChildLoop = false;

    if (inLoop) {
      auto builderLoop = LI.getLoopFor(forwardBlock);
      while (builderLoop) {
        if (builderLoop->getHeader() == lc.header) {
          isChildLoop = true;
          break;
        }
        builderLoop = builderLoop->getParentLoop();
      }
    }

    if ((!guaranteeVisible || forwardBlock == inst->getParent() ||
         DT.dominates(inst, forwardBlock)) &&
        (!inLoop || isChildLoop)) {
      return inst;
    }

    if (!inLoop || isChildLoop)
      mergeIfTrue = true;

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
        if (!seen.count(&BB) || (inst->getParent() != &BB &&
                                 DT.dominates(&BB, inst->getParent()))) {
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
        // Todo, this optimization can't be done unless the block is also proven
        // to never reach inst->getParent() as a successor
        /*
        for (const auto &pair : lcssaFixes[inst]) {
          if (!isa<UndefValue>(pair.second) &&
              (pred == pair.first || DT.dominates(pair.first, pred))) {
            val = pair.second;
            assert(pair.second->getType() == inst->getType());
            break;
          }
        }
        */
      }
      if (val == nullptr) {
        val = fixLCSSA(inst, pred, /*mergeIfPossible*/ true);
        assert(val->getType() == inst->getType());
      }
      assert(val->getType() == inst->getType());
      lcssaPHI->addIncoming(val, pred);
    }

    if (mergeIfTrue) {
      Value *val = lcssaPHI;
      for (Value *v : lcssaPHI->incoming_values()) {
        if (auto PN = dyn_cast<PHINode>(v))
          if (lcssaPHIToOrig.find(PN) != lcssaPHIToOrig.end()) {
            v = lcssaPHIToOrig[PN];
          }
        if (v == lcssaPHI)
          continue;
        if (val == lcssaPHI)
          val = v;
        if (v != val) {
          val = nullptr;
          break;
        }
      }
      if (val && val != lcssaPHI &&
          (!guaranteeVisible || !isa<Instruction>(val) ||
           DT.dominates(cast<Instruction>(val), lcssaPHI))) {
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
            lcssaPHI->removeIncomingValue(lcssaPHI->getNumOperands() - 1,
                                          false);
          lcssaPHIToOrig.erase(lcssaPHI);
          lcssaPHI->eraseFromParent();
        }
        return val;
      }
    }
    return lcssaPHI;
  }

  Value *
  lookupM(Value *val, IRBuilder<> &BuilderM,
          const ValueToValueMapTy &incoming_availalble = ValueToValueMapTy(),
          bool tryLegalRecomputeCheck = true) override;

  Value *invertPointerM(Value *val, IRBuilder<> &BuilderM);

  void branchToCorrespondingTarget(
      BasicBlock *ctx, IRBuilder<> &BuilderM,
      const std::map<BasicBlock *,
                     std::vector<std::pair</*pred*/ BasicBlock *,
                                           /*successor*/ BasicBlock *>>>
          &targetToPreds,
      const std::map<BasicBlock *, PHINode *> *replacePHIs = nullptr);

  void getReverseBuilder(IRBuilder<> &Builder2, bool original = true) {
    BasicBlock *BB = Builder2.GetInsertBlock();
    if (original)
      BB = getNewFromOriginal(BB);
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

  void getForwardBuilder(IRBuilder<> &Builder2) {
    Instruction *insert = &*Builder2.GetInsertPoint();
    Instruction *nInsert = getNewFromOriginal(insert);

    assert(nInsert);

    Builder2.SetInsertPoint(getNextNonDebugInstruction(nInsert));
    Builder2.SetCurrentDebugLocation(
        getNewFromOriginal(Builder2.getCurrentDebugLocation()));
    Builder2.setFastMathFlags(getFast());
  }
};

class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
                     TargetLibraryInfo &TLI, TypeAnalysis &TA,
                     ValueToValueMapTy &invertedPointers_,
                     const SmallPtrSetImpl<Value *> &constantvalues_,
                     const SmallPtrSetImpl<Value *> &returnvals_,
                     bool ActiveReturn, ValueToValueMapTy &origToNew_,
                     DerivativeMode mode)
      : GradientUtils(Logic, newFunc_, oldFunc_, TLI, TA, invertedPointers_,
                      constantvalues_, returnvals_, ActiveReturn, origToNew_,
                      mode) {
    assert(reverseBlocks.size() == 0);
    if (mode == DerivativeMode::ForwardMode) {
      return;
    }
    for (BasicBlock *BB : originalBlocks) {
      if (BB == inversionAllocs)
        continue;
      reverseBlocks[BB].push_back(BasicBlock::Create(
          BB->getContext(), "invert" + BB->getName(), newFunc));
    }
    assert(reverseBlocks.size() != 0);
  }

public:
  ValueToValueMapTy differentials;
  static DiffeGradientUtils *
  CreateFromClone(EnzymeLogic &Logic, DerivativeMode mode, Function *todiff,
                  TargetLibraryInfo &TLI, TypeAnalysis &TA, DIFFE_TYPE retType,
                  bool diffeReturnArg,
                  const std::vector<DIFFE_TYPE> &constant_args,
                  ReturnType returnValue, Type *additionalArg);

private:
  Value *getDifferential(Value *val) {
    assert(val);
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    assert(inversionAllocs);
    if (differentials.find(val) == differentials.end()) {
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr,
                                                     val->getName() + "'de");
      entryBuilder.CreateStore(Constant::getNullValue(val->getType()),
                               differentials[val]);
    }
    assert(cast<PointerType>(differentials[val]->getType())->getElementType() ==
           val->getType());
    return differentials[val];
  }

public:
  Value *diffe(Value *val, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!val->getType()->isVoidTy());
    return BuilderM.CreateLoad(getDifferential(val));
  }

  // Returns created select instructions, if any
  std::vector<SelectInst *> addToDiffe(Value *val, Value *dif,
                                       IRBuilder<> &BuilderM, Type *addingType,
                                       ArrayRef<Value *> idxs = {}) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);

    std::vector<SelectInst *> addedSelects;

    auto faddForNeg = [&](Value *old, Value *inc) {
      if (auto bi = dyn_cast<BinaryOperator>(inc)) {
        if (auto ci = dyn_cast<ConstantFP>(bi->getOperand(0))) {
          if (bi->getOpcode() == BinaryOperator::FSub && ci->isZero()) {
            return BuilderM.CreateFSub(old, bi->getOperand(1));
          }
        }
      }
      return BuilderM.CreateFAdd(old, inc);
    };

    auto faddForSelect = [&](Value *old, Value *dif) -> Value * {
      //! optimize fadd of select to select of fadd
      if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
        if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), old,
                faddForNeg(old, select->getFalseValue())));
            addedSelects.emplace_back(res);
            return res;
          }
        }
        if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(), faddForNeg(old, select->getTrueValue()),
                old));
            addedSelects.emplace_back(res);
            return res;
          }
        }
      }

      //! optimize fadd of bitcast select to select of bitcast fadd
      if (BitCastInst *bc = dyn_cast<BitCastInst>(dif)) {
        if (SelectInst *select = dyn_cast<SelectInst>(bc->getOperand(0))) {
          if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(), old,
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getFalseValue(),
                                                      bc->getDestTy()))));
              addedSelects.emplace_back(res);
              return res;
            }
          }
          if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
            if (ci->isZeroValue()) {
              SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                  select->getCondition(),
                  faddForNeg(old, BuilderM.CreateCast(bc->getOpcode(),
                                                      select->getTrueValue(),
                                                      bc->getDestTy())),
                  old));
              addedSelects.emplace_back(res);
              return res;
            }
          }
        }
      }

      // fallback
      return faddForNeg(old, dif);
    };

    if (val->getType()->isPointerTy()) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!val->getType()->isPointerTy());
    assert(!isConstantValue(val));

    Value *ptr = getDifferential(val);

    if (idxs.size() != 0) {
      SmallVector<Value *, 4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for (auto i : idxs)
        sv.push_back(i);
      ptr = BuilderM.CreateGEP(ptr, sv);
      cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
    }
    Value *old = BuilderM.CreateLoad(ptr);

    assert(dif->getType() == old->getType());
    Value *res = nullptr;
    if (old->getType()->isIntOrIntVectorTy()) {
      if (!addingType) {
        llvm::errs() << "module: " << *oldFunc->getParent() << "\n";
        llvm::errs() << "oldFunc: " << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << "val: " << *val << " old: " << old << "\n";
      }
      assert(addingType);
      assert(addingType->isFPOrFPVectorTy());

      auto oldBitSize = oldFunc->getParent()->getDataLayout().getTypeSizeInBits(
          old->getType());
      auto newBitSize =
          oldFunc->getParent()->getDataLayout().getTypeSizeInBits(addingType);

      if (oldBitSize > newBitSize && oldBitSize % newBitSize == 0 &&
          !addingType->isVectorTy()) {
#if LLVM_VERSION_MAJOR >= 11
        addingType =
            VectorType::get(addingType, oldBitSize / newBitSize, false);
#else
        addingType = VectorType::get(addingType, oldBitSize / newBitSize);
#endif
      }

      Value *bcold = BuilderM.CreateBitCast(old, addingType);
      Value *bcdif = BuilderM.CreateBitCast(dif, addingType);

      res = faddForSelect(bcold, bcdif);
      if (SelectInst *select = dyn_cast<SelectInst>(res)) {
        assert(addedSelects.back() == select);
        addedSelects.erase(addedSelects.end() - 1);
        res = BuilderM.CreateSelect(
            select->getCondition(),
            BuilderM.CreateBitCast(select->getTrueValue(), old->getType()),
            BuilderM.CreateBitCast(select->getFalseValue(), old->getType()));
        assert(select->getNumUses() == 0);
      } else {
        res = BuilderM.CreateBitCast(res, old->getType());
      }
      BuilderM.CreateStore(res, ptr);
      // store->setAlignment(align);
      return addedSelects;
    } else if (old->getType()->isFPOrFPVectorTy()) {
      // TODO consider adding type
      res = faddForSelect(old, dif);

      BuilderM.CreateStore(res, ptr);
      // store->setAlignment(align);
      return addedSelects;
    } else if (auto st = dyn_cast<StructType>(old->getType())) {
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        // TODO pass in full type tree here and recurse into tree.
        if (st->getElementType(i)->isPointerTy())
          continue;
        Value *v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
        SmallVector<Value *, 2> idx2(idxs.begin(), idxs.end());
        idx2.push_back(v);
        auto selects = addToDiffe(val, BuilderM.CreateExtractValue(dif, {i}),
                                  BuilderM, nullptr, idx2);
        for (auto select : selects) {
          addedSelects.push_back(select);
        }
      }
      return addedSelects;
    } else {
      llvm_unreachable("unknown type to add to diffe");
      exit(1);
    }
  }

  void setDiffe(Value *val, Value *toset, IRBuilder<> &BuilderM) {
    if (auto arg = dyn_cast<Argument>(val))
      assert(arg->getParent() == oldFunc);
    if (auto inst = dyn_cast<Instruction>(val))
      assert(inst->getParent()->getParent() == oldFunc);
    if (isConstantValue(val)) {
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *val << "\n";
    }
    assert(!isConstantValue(val));
    Value *tostore = getDifferential(val);
    if (toset->getType() !=
        cast<PointerType>(tostore->getType())->getElementType()) {
      llvm::errs() << "toset:" << *toset << "\n";
      llvm::errs() << "tostore:" << *tostore << "\n";
    }
    assert(toset->getType() ==
           cast<PointerType>(tostore->getType())->getElementType());
    BuilderM.CreateStore(toset, tostore);
  }

  void freeCache(llvm::BasicBlock *forwardPreheader,
                 const SubLimitType &sublimits, int i, llvm::AllocaInst *alloc,
                 llvm::ConstantInt *byteSizeOfType, llvm::Value *storeInto,
                 llvm::MDNode *InvariantMD) override {
    assert(reverseBlocks.find(forwardPreheader) != reverseBlocks.end());
    assert(reverseBlocks[forwardPreheader].size());
    IRBuilder<> tbuild(reverseBlocks[forwardPreheader].back());
    tbuild.setFastMathFlags(getFast());

    // ensure we are before the terminator if it exists
    if (tbuild.GetInsertBlock()->size() &&
        tbuild.GetInsertBlock()->getTerminator()) {
      tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getTerminator());
    }

    ValueToValueMapTy antimap;
    for (int j = sublimits.size() - 1; j >= i; j--) {
      auto &innercontainedloops = sublimits[j].second;
      for (auto riter = innercontainedloops.rbegin(),
                rend = innercontainedloops.rend();
           riter != rend; ++riter) {
        const auto &idx = riter->first;
        if (idx.var)
          antimap[idx.var] = tbuild.CreateLoad(idx.antivaralloc);
      }
    }

    auto forfree = cast<LoadInst>(tbuild.CreateLoad(
        unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap)));
    forfree->setMetadata(LLVMContext::MD_invariant_group, InvariantMD);
    forfree->setMetadata(
        LLVMContext::MD_dereferenceable,
        MDNode::get(forfree->getContext(),
                    {ConstantAsMetadata::get(byteSizeOfType)}));
    forfree->setName("forfree");
    unsigned bsize = (unsigned)byteSizeOfType->getZExtValue();
    if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
      forfree->setAlignment(Align(bsize));
#else
      forfree->setAlignment(bsize);
#endif
    }
    auto ci = cast<CallInst>(CallInst::CreateFree(
        tbuild.CreatePointerCast(forfree,
                                 Type::getInt8PtrTy(newFunc->getContext())),
        tbuild.GetInsertBlock()));
    if (newFunc->getSubprogram())
      ci->setDebugLoc(DILocation::get(newFunc->getContext(), 0, 0,
                                      newFunc->getSubprogram(), 0));
    ci->addAttribute(AttributeList::FirstArgIndex, Attribute::NonNull);
    if (ci->getParent() == nullptr) {
      tbuild.Insert(ci);
    }
    scopeFrees[alloc].insert(ci);
  }

//! align is the alignment that should be specified for load/store to pointer
#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(Value *origptr, Value *dif, IRBuilder<> &BuilderM,
                             MaybeAlign align, Value *OrigOffset = nullptr)
#else
  void addToInvertedPtrDiffe(Value *origptr, Value *dif, IRBuilder<> &BuilderM,
                             unsigned align, Value *OrigOffset = nullptr)
#endif
  {
    if (!(origptr->getType()->isPointerTy()) ||
        !(cast<PointerType>(origptr->getType())->getElementType() ==
          dif->getType())) {
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << "Origptr: " << *origptr << "\n";
      llvm::errs() << "Diff: " << *dif << "\n";
    }
    assert(origptr->getType()->isPointerTy());
    assert(cast<PointerType>(origptr->getType())->getElementType() ==
           dif->getType());

    assert(origptr->getType()->isPointerTy());
    assert(cast<PointerType>(origptr->getType())->getElementType() ==
           dif->getType());

    // const SCEV *S = SE.getSCEV(PN);
    // if (SE.getCouldNotCompute() == S)
    //  continue;

    Value *ptr = invertPointerM(origptr, BuilderM);
    assert(ptr);
    if (OrigOffset) {
      ptr = BuilderM.CreateGEP(
          ptr, lookupM(getNewFromOriginal(OrigOffset), BuilderM));
    }

    auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(origptr, 100);
#else
        GetUnderlyingObject(origptr, oldFunc->getParent()->getDataLayout(),
                            100);
#endif

    // atomics
    bool Atomic = AtomicAdd;
    auto Arch = llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch();

    // No need to do atomic on local memory for CUDA since it can't be raced
    // upon
    if (isa<AllocaInst>(TmpOrig) &&
        (Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
         Arch == Triple::amdgcn)) {
      Atomic = false;
    }

    if (Atomic) {
      // For amdgcn constant AS is 4 and if the primal is in it we need to cast
      // the derivative value to AS 1
      auto AS = cast<PointerType>(ptr->getType())->getAddressSpace();
      if (Arch == Triple::amdgcn && AS == 4) {
        ptr = BuilderM.CreateAddrSpaceCast(
            ptr, PointerType::get(
                     cast<PointerType>(ptr->getType())->getElementType(), 1));
      }

      /*
      while (auto ASC = dyn_cast<AddrSpaceCastInst>(ptr)) {
        ptr = ASC->getOperand(0);
      }
      while (auto ASC = dyn_cast<ConstantExpr>(ptr)) {
        if (!ASC->isCast()) break;
        if (ASC->getOpcode() != Instruction::AddrSpaceCast) break;
        ptr = ASC->getOperand(0);
      }
      */
      if (dif->getType()->isIntOrIntVectorTy()) {

        ptr = BuilderM.CreateBitCast(
            ptr, PointerType::get(
                     IntToFloatTy(dif->getType()),
                     cast<PointerType>(ptr->getType())->getAddressSpace()));
        dif = BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType()));
      }
#if LLVM_VERSION_MAJOR >= 9
      AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
      if (auto vt = dyn_cast<VectorType>(dif->getType())) {
#if LLVM_VERSION_MAJOR >= 12
        assert(!vt->getElementCount().isScalable());
        size_t numElems = vt->getElementCount().getKnownMinValue();
#else
        size_t numElems = vt->getNumElements();
#endif
        for (size_t i = 0; i < numElems; ++i) {
          auto vdif = BuilderM.CreateExtractElement(dif, i);
          Value *Idxs[] = {
              ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
          auto vptr = BuilderM.CreateGEP(ptr, Idxs);
#if LLVM_VERSION_MAJOR >= 13
          BuilderM.CreateAtomicRMW(op, vptr, vdif, align,
                                   AtomicOrdering::Monotonic,
                                   SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
          AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
              op, vptr, vdif, AtomicOrdering::Monotonic, SyncScope::System);
          if (align)
            rmw->setAlignment(align.getValue());
#else
          BuilderM.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                   SyncScope::System);
#endif
        }
      } else {
#if LLVM_VERSION_MAJOR >= 13
        BuilderM.CreateAtomicRMW(op, ptr, dif, align, AtomicOrdering::Monotonic,
                                 SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
        AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
            op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
        if (align)
          rmw->setAlignment(align.getValue());
#else
        BuilderM.CreateAtomicRMW(op, ptr, dif, AtomicOrdering::Monotonic,
                                 SyncScope::System);
#endif
      }
#else
      llvm::errs() << "unhandled atomic fadd on llvm version " << *ptr << " "
                   << *dif << "\n";
      llvm_unreachable("unhandled atomic fadd");
#endif
      return;
    }

    Value *res;
    LoadInst *old = BuilderM.CreateLoad(ptr);
#if LLVM_VERSION_MAJOR >= 10
    if (align)
      old->setAlignment(align.getValue());
#else
    old->setAlignment(align);
#endif

    if (old->getType()->isIntOrIntVectorTy()) {
      res = BuilderM.CreateFAdd(
          BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())),
          BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
      res = BuilderM.CreateBitCast(res, old->getType());
    } else if (old->getType()->isFPOrFPVectorTy()) {
      res = BuilderM.CreateFAdd(old, dif);
    } else {
      assert(old);
      assert(dif);
      llvm::errs() << *newFunc << "\n"
                   << "cannot handle type " << *old << "\n"
                   << *dif;
      assert(0 && "cannot handle type");
      report_fatal_error("cannot handle type");
    }
    StoreInst *st = BuilderM.CreateStore(res, ptr);
#if LLVM_VERSION_MAJOR >= 10
    if (align)
      st->setAlignment(align.getValue());
#else
    st->setAlignment(align);
#endif
  }
};
#endif
