//===- DiffeGradientUtils.cpp - Helper class and utilities for AD ---------===//
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

#include <string>

#include "DiffeGradientUtils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

DiffeGradientUtils::DiffeGradientUtils(
    EnzymeLogic &Logic, Function *newFunc_, Function *oldFunc_,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, TypeResults TR,
    ValueToValueMapTy &invertedPointers_,
    const SmallPtrSetImpl<Value *> &constantvalues_,
    const SmallPtrSetImpl<Value *> &returnvals_, DIFFE_TYPE ActiveReturn,
    ArrayRef<DIFFE_TYPE> constant_values,
    llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> &origToNew_,
    DerivativeMode mode, unsigned width, bool omp)
    : GradientUtils(Logic, newFunc_, oldFunc_, TLI, TA, TR, invertedPointers_,
                    constantvalues_, returnvals_, ActiveReturn, constant_values,
                    origToNew_, mode, width, omp) {
  assert(reverseBlocks.size() == 0);
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit) {
    return;
  }
  for (BasicBlock *BB : originalBlocks) {
    if (BB == inversionAllocs)
      continue;
    BasicBlock *RBB =
        BasicBlock::Create(BB->getContext(), "invert" + BB->getName(), newFunc);
    reverseBlocks[BB].push_back(RBB);
    reverseBlockToPrimal[RBB] = BB;
  }
  assert(reverseBlocks.size() != 0);
}

DiffeGradientUtils *DiffeGradientUtils::CreateFromClone(
    EnzymeLogic &Logic, DerivativeMode mode, unsigned width, Function *todiff,
    TargetLibraryInfo &TLI, TypeAnalysis &TA, FnTypeInfo &oldTypeInfo,
    DIFFE_TYPE retType, bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
    ReturnType returnValue, Type *additionalArg, bool omp) {
  assert(!todiff->empty());
  Function *oldFunc = todiff;
  assert(mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined ||
         mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit);
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  std::string prefix;

  switch (mode) {
  case DerivativeMode::ForwardMode:
  case DerivativeMode::ForwardModeSplit:
    prefix = "fwddiffe";
    break;
  case DerivativeMode::ReverseModeCombined:
  case DerivativeMode::ReverseModeGradient:
    prefix = "diffe";
    break;
  case DerivativeMode::ReverseModePrimal:
    llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
  }

  if (width > 1)
    prefix += std::to_string(width);

  auto newFunc = Logic.PPC.CloneFunctionWithReturns(
      mode, width, oldFunc, invertedPointers, constant_args, constant_values,
      nonconstant_values, returnvals, returnValue, retType,
      prefix + oldFunc->getName(), &originalToNew,
      /*diffeReturnArg*/ diffeReturnArg, additionalArg);

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

  auto res = new DiffeGradientUtils(Logic, newFunc, oldFunc, TLI, TA, TR,
                                    invertedPointers, constant_values,
                                    nonconstant_values, retType, constant_args,
                                    originalToNew, mode, width, omp);

  return res;
}

AllocaInst *DiffeGradientUtils::getDifferential(Value *val) {
  assert(val);
  if (auto arg = dyn_cast<Argument>(val))
    assert(arg->getParent() == oldFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == oldFunc);
  assert(inversionAllocs);

  Type *type = getShadowType(val->getType());
  if (differentials.find(val) == differentials.end()) {
    IRBuilder<> entryBuilder(inversionAllocs);
    entryBuilder.setFastMathFlags(getFast());
    differentials[val] =
        entryBuilder.CreateAlloca(type, nullptr, val->getName() + "'de");
    auto Alignment =
        oldFunc->getParent()->getDataLayout().getPrefTypeAlignment(type);
#if LLVM_VERSION_MAJOR >= 10
    differentials[val]->setAlignment(Align(Alignment));
#else
    differentials[val]->setAlignment(Alignment);
#endif
    ZeroMemory(entryBuilder, type, differentials[val],
               /*isTape*/ false);
  }
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
  if (val->getContext().supportsTypedPointers()) {
#endif
    assert(differentials[val]->getType()->getPointerElementType() == type);
#if LLVM_VERSION_MAJOR >= 15
  }
#endif
#endif
  return differentials[val];
}

Value *DiffeGradientUtils::diffe(Value *val, IRBuilder<> &BuilderM) {
  if (auto arg = dyn_cast<Argument>(val))
    assert(arg->getParent() == oldFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == oldFunc);

  if (isConstantValue(val)) {
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
    assert(0 && "getting diffe of constant value");
  }
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit)
    return invertPointerM(val, BuilderM);
  if (val->getType()->isPointerTy()) {
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
  }
  assert(!val->getType()->isPointerTy());
  assert(!val->getType()->isVoidTy());
  Type *ty = getShadowType(val->getType());
  return BuilderM.CreateLoad(ty, getDifferential(val));
}

SmallVector<SelectInst *, 4>
DiffeGradientUtils::addToDiffe(Value *val, Value *dif, IRBuilder<> &BuilderM,
                               Type *addingType, ArrayRef<Value *> idxs,
                               Value *mask) {
  assert(mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ReverseModeCombined);

  if (auto arg = dyn_cast<Argument>(val))
    assert(arg->getParent() == oldFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == oldFunc);

  SmallVector<SelectInst *, 4> addedSelects;

  auto faddForNeg = [&](Value *old, Value *inc, bool san) {
    if (auto bi = dyn_cast<BinaryOperator>(inc)) {
      if (auto ci = dyn_cast<ConstantFP>(bi->getOperand(0))) {
        if (bi->getOpcode() == BinaryOperator::FSub && ci->isZero()) {
          Value *res = BuilderM.CreateFSub(old, bi->getOperand(1));
          if (san)
            res = SanitizeDerivatives(val, res, BuilderM, mask);
          return res;
        }
      }
    }
    Value *res = BuilderM.CreateFAdd(old, inc);
    if (san)
      res = SanitizeDerivatives(val, res, BuilderM, mask);
    return res;
  };

  auto faddForSelect = [&](Value *old, Value *dif) -> Value * {
    //! optimize fadd of select to select of fadd
    if (SelectInst *select = dyn_cast<SelectInst>(dif)) {
      if (Constant *ci = dyn_cast<Constant>(select->getTrueValue())) {
        if (ci->isZeroValue()) {
          SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
              select->getCondition(), old,
              faddForNeg(old, select->getFalseValue(), false)));
          addedSelects.emplace_back(res);
          return SanitizeDerivatives(val, res, BuilderM, mask);
        }
      }
      if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
        if (ci->isZeroValue()) {
          SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
              select->getCondition(),
              faddForNeg(old, select->getTrueValue(), false), old));
          addedSelects.emplace_back(res);
          return SanitizeDerivatives(val, res, BuilderM, mask);
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
                faddForNeg(old,
                           BuilderM.CreateCast(bc->getOpcode(),
                                               select->getFalseValue(),
                                               bc->getDestTy()),
                           false)));
            addedSelects.emplace_back(res);
            return SanitizeDerivatives(val, res, BuilderM, mask);
          }
        }
        if (Constant *ci = dyn_cast<Constant>(select->getFalseValue())) {
          if (ci->isZeroValue()) {
            SelectInst *res = cast<SelectInst>(BuilderM.CreateSelect(
                select->getCondition(),
                faddForNeg(old,
                           BuilderM.CreateCast(bc->getOpcode(),
                                               select->getTrueValue(),
                                               bc->getDestTy()),
                           false),
                old));
            addedSelects.emplace_back(res);
            return SanitizeDerivatives(val, res, BuilderM, mask);
          }
        }
      }
    }

    // fallback
    return faddForNeg(old, dif, true);
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
    SmallVector<Value *, 4> sv = {
        ConstantInt::get(Type::getInt32Ty(val->getContext()), 0)};
    for (auto i : idxs)
      sv.push_back(i);
    ptr = BuilderM.CreateGEP(getShadowType(val->getType()), ptr, sv);
    cast<GetElementPtrInst>(ptr)->setIsInBounds(true);
  }
  Value *old = BuilderM.CreateLoad(dif->getType(), ptr);

  assert(dif->getType() == old->getType());
  Value *res = nullptr;
  if (old->getType()->isIntOrIntVectorTy()) {
    if (!addingType) {
      if (looseTypeAnalysis) {
        if (old->getType()->isIntegerTy(64))
          addingType = Type::getDoubleTy(old->getContext());
        else if (old->getType()->isIntegerTy(32))
          addingType = Type::getFloatTy(old->getContext());
      }
    }
    if (!addingType) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "oldFunc: " << *oldFunc << "\n";
      ss << "Cannot deduce adding type of: " << *old << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(old), ErrorType::NoType,
                           &TR.analyzer, nullptr, wrap(&BuilderM));
        return addedSelects;
      } else {
        TR.dump();
        DebugLoc loc;
        if (auto inst = dyn_cast<Instruction>(old))
          EmitFailure("CannotDeduceType", inst->getDebugLoc(), inst, ss.str());
        else {
          llvm::errs() << ss.str() << "\n";
          llvm_unreachable("Cannot deduce adding type");
        }
        return addedSelects;
      }
    }
    assert(addingType);
    assert(addingType->isFPOrFPVectorTy());

    auto oldBitSize =
        oldFunc->getParent()->getDataLayout().getTypeSizeInBits(old->getType());
    auto newBitSize =
        oldFunc->getParent()->getDataLayout().getTypeSizeInBits(addingType);

    if (oldBitSize > newBitSize && oldBitSize % newBitSize == 0 &&
        !addingType->isVectorTy()) {
#if LLVM_VERSION_MAJOR >= 11
      addingType = VectorType::get(addingType, oldBitSize / newBitSize, false);
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
    if (!mask) {
      BuilderM.CreateStore(res, ptr);
      // store->setAlignment(align);
    } else {
      Type *tys[] = {res->getType(), ptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_store, tys);
#if LLVM_VERSION_MAJOR > 10
      auto align = cast<AllocaInst>(ptr)->getAlign().value();
#else
      auto align = cast<AllocaInst>(ptr)->getAlignment();
#endif
      assert(align);
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
      Value *args[] = {res, ptr, alignv, mask};
      BuilderM.CreateCall(F, args);
    }
    return addedSelects;
  } else if (old->getType()->isFPOrFPVectorTy()) {
    // TODO consider adding type
    res = faddForSelect(old, dif);

    if (!mask) {
      BuilderM.CreateStore(res, ptr);
      // store->setAlignment(align);
    } else {
      Type *tys[] = {res->getType(), ptr->getType()};
      auto F = Intrinsic::getDeclaration(oldFunc->getParent(),
                                         Intrinsic::masked_store, tys);
#if LLVM_VERSION_MAJOR > 10
      auto align = cast<AllocaInst>(ptr)->getAlign().value();
#else
      auto align = cast<AllocaInst>(ptr)->getAlignment();
#endif
      assert(align);
      Value *alignv =
          ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
      Value *args[] = {res, ptr, alignv, mask};
      BuilderM.CreateCall(F, args);
    }
    return addedSelects;
  } else if (auto st = dyn_cast<StructType>(old->getType())) {
    assert(!mask);
    if (mask)
      llvm_unreachable("cannot handle recursive addToDiffe with mask");
    for (unsigned i = 0; i < st->getNumElements(); ++i) {
      // TODO pass in full type tree here and recurse into tree.
      if (st->getElementType(i)->isPointerTy())
        continue;
      Value *v = ConstantInt::get(Type::getInt32Ty(st->getContext()), i);
      SmallVector<Value *, 2> idx2(idxs.begin(), idxs.end());
      idx2.push_back(v);
      // FIXME: reconsider if passing a nullptr is correct here.
      auto selects = addToDiffe(val, extractMeta(BuilderM, dif, i), BuilderM,
                                nullptr, idx2);
      for (auto select : selects) {
        addedSelects.push_back(select);
      }
    }
    return addedSelects;
  } else if (auto at = dyn_cast<ArrayType>(old->getType())) {
    assert(!mask);
    if (mask)
      llvm_unreachable("cannot handle recursive addToDiffe with mask");
    if (at->getElementType()->isPointerTy())
      return addedSelects;
    for (unsigned i = 0; i < at->getNumElements(); ++i) {
      // TODO pass in full type tree here and recurse into tree.
      Value *v = ConstantInt::get(Type::getInt32Ty(at->getContext()), i);
      SmallVector<Value *, 2> idx2(idxs.begin(), idxs.end());
      idx2.push_back(v);
      auto selects = addToDiffe(val, extractMeta(BuilderM, dif, i), BuilderM,
                                addingType, idx2);
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

void DiffeGradientUtils::setDiffe(Value *val, Value *toset,
                                  IRBuilder<> &BuilderM) {
  if (auto arg = dyn_cast<Argument>(val))
    assert(arg->getParent() == oldFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == oldFunc);
  if (isConstantValue(val)) {
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << *val << "\n";
  }
  assert(!isConstantValue(val));
  toset = SanitizeDerivatives(val, toset, BuilderM);
  if (mode == DerivativeMode::ForwardMode ||
      mode == DerivativeMode::ForwardModeSplit) {
    assert(getShadowType(val->getType()) == toset->getType());
    auto found = invertedPointers.find(val);
    assert(found != invertedPointers.end());
    auto placeholder0 = &*found->second;
    auto placeholder = cast<PHINode>(placeholder0);
    invertedPointers.erase(found);
    replaceAWithB(placeholder, toset);
    placeholder->replaceAllUsesWith(toset);
    erase(placeholder);
    invertedPointers.insert(
        std::make_pair((const Value *)val, InvertedPointerVH(this, toset)));
    return;
  }
  Value *tostore = getDifferential(val);
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
  if (toset->getContext().supportsTypedPointers()) {
#endif
    if (toset->getType() != tostore->getType()->getPointerElementType()) {
      llvm::errs() << "toset:" << *toset << "\n";
      llvm::errs() << "tostore:" << *tostore << "\n";
    }
    assert(toset->getType() == tostore->getType()->getPointerElementType());
#if LLVM_VERSION_MAJOR >= 15
  }
#endif
#endif
  BuilderM.CreateStore(toset, tostore);
}

CallInst *DiffeGradientUtils::freeCache(BasicBlock *forwardPreheader,
                                        const SubLimitType &sublimits, int i,
                                        AllocaInst *alloc,
                                        ConstantInt *byteSizeOfType,
                                        Value *storeInto, MDNode *InvariantMD) {
  if (!FreeMemory)
    return nullptr;
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
      if (idx.var) {
        antimap[idx.var] =
            tbuild.CreateLoad(idx.var->getType(), idx.antivaralloc);
      }
    }
  }

  Value *metaforfree =
      unwrapM(storeInto, tbuild, antimap, UnwrapMode::LegalFullUnwrap);
  Type *T;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
  if (metaforfree->getContext().supportsTypedPointers()) {
#endif
    T = metaforfree->getType()->getPointerElementType();
#if LLVM_VERSION_MAJOR >= 15
  } else {
    T = PointerType::getUnqual(metaforfree->getContext());
  }
#endif
#else
  T = PointerType::getUnqual(metaforfree->getContext());
#endif
  LoadInst *forfree = cast<LoadInst>(tbuild.CreateLoad(T, metaforfree));
  forfree->setMetadata(LLVMContext::MD_invariant_group, InvariantMD);
  forfree->setMetadata(LLVMContext::MD_dereferenceable,
                       MDNode::get(forfree->getContext(),
                                   ArrayRef<Metadata *>(ConstantAsMetadata::get(
                                       byteSizeOfType))));
  forfree->setName("forfree");
  unsigned align = getCacheAlignment(
      (unsigned)newFunc->getParent()->getDataLayout().getPointerSize());
#if LLVM_VERSION_MAJOR >= 10
  forfree->setAlignment(Align(align));
#else
  forfree->setAlignment(align);
#endif

  CallInst *ci = CreateDealloc(tbuild, forfree);
  if (ci) {
    if (newFunc->getSubprogram())
      ci->setDebugLoc(DILocation::get(newFunc->getContext(), 0, 0,
                                      newFunc->getSubprogram(), 0));
    scopeFrees[alloc].insert(ci);
  }
  return ci;
}

#if LLVM_VERSION_MAJOR >= 10
void DiffeGradientUtils::addToInvertedPtrDiffe(Instruction *orig,
                                               Value *origVal, Type *addingType,
                                               unsigned start, unsigned size,
                                               Value *origptr, Value *dif,
                                               IRBuilder<> &BuilderM,
                                               MaybeAlign align, Value *mask)
#else
void DiffeGradientUtils::addToInvertedPtrDiffe(Instruction *orig,
                                               Value *origVal, Type *addingType,
                                               unsigned start, unsigned size,
                                               Value *origptr, Value *dif,
                                               IRBuilder<> &BuilderM,
                                               unsigned align, Value *mask)
#endif
{
  auto &DL = oldFunc->getParent()->getDataLayout();

  auto addingSize = (DL.getTypeSizeInBits(addingType) + 1) / 8;
  if (addingSize != size) {
    assert(size > addingSize);
#if LLVM_VERSION_MAJOR >= 12
    addingType =
        VectorType::get(addingType, size / addingSize, /*isScalable*/ false);
#else
    addingType = VectorType::get(addingType, size / addingSize);
#endif
    size = (size / addingSize) * addingSize;
  }

  Value *ptr;

  switch (mode) {
  case DerivativeMode::ForwardModeSplit:
  case DerivativeMode::ForwardMode:
    ptr = invertPointerM(origptr, BuilderM);
    break;
  case DerivativeMode::ReverseModePrimal:
    assert(false && "Invalid derivative mode (ReverseModePrimal)");
    break;
  case DerivativeMode::ReverseModeGradient:
  case DerivativeMode::ReverseModeCombined:
    ptr = lookupM(invertPointerM(origptr, BuilderM), BuilderM);
    break;
  }

  bool needsCast = false;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
  if (origptr->getContext().supportsTypedPointers()) {
#endif
    needsCast = origptr->getType()->getPointerElementType() != addingType;
#if LLVM_VERSION_MAJOR >= 15
  }
#endif
#endif

  assert(ptr);
  if (start != 0 || needsCast) {
    auto rule = [&](Value *ptr) {
      if (start != 0) {
        auto i8 = Type::getInt8Ty(ptr->getContext());
        ptr = BuilderM.CreatePointerCast(
            ptr, PointerType::get(
                     i8, cast<PointerType>(ptr->getType())->getAddressSpace()));
        auto off = ConstantInt::get(Type::getInt64Ty(ptr->getContext()), start);
        ptr = BuilderM.CreateInBoundsGEP(i8, ptr, off);
      }
      if (needsCast) {
        ptr = BuilderM.CreatePointerCast(
            ptr, PointerType::get(
                     addingType,
                     cast<PointerType>(ptr->getType())->getAddressSpace()));
      }
      return ptr;
    };
    ptr = applyChainRule(
        PointerType::get(
            addingType,
            cast<PointerType>(origptr->getType())->getAddressSpace()),
        BuilderM, rule, ptr);
  }

  if (getWidth() == 1)
    needsCast = dif->getType() != addingType;
  else if (auto AT = cast<ArrayType>(dif->getType()))
    needsCast = AT->getElementType() != addingType;
  else
    needsCast =
        cast<VectorType>(dif->getType())->getElementType() != addingType;

  if (start != 0 || needsCast) {
    auto rule = [&](Value *dif) {
      if (start != 0) {
        IRBuilder<> A(inversionAllocs);
        auto i8 = Type::getInt8Ty(ptr->getContext());
        auto prevSize = (DL.getTypeSizeInBits(dif->getType()) + 1) / 8;
        Type *tys[] = {ArrayType::get(i8, start), addingType,
                       ArrayType::get(i8, prevSize - start - size)};
        auto ST = StructType::get(i8->getContext(), tys, /*isPacked*/ true);
        auto Al = A.CreateAlloca(ST);
        BuilderM.CreateStore(dif,
                             BuilderM.CreatePointerCast(
                                 Al, PointerType::getUnqual(dif->getType())));
        Value *idxs[] = {
            ConstantInt::get(Type::getInt64Ty(ptr->getContext()), 0),
            ConstantInt::get(Type::getInt32Ty(ptr->getContext()), 1)};

        auto difp = BuilderM.CreateInBoundsGEP(ST, Al, idxs);
        dif = BuilderM.CreateLoad(addingType, difp);
      }
      if (dif->getType() != addingType) {
        auto difSize = (DL.getTypeSizeInBits(dif->getType()) + 1) / 8;
        if (difSize < size) {
          llvm::errs() << " ds: " << difSize << " as: " << size << "\n";
          llvm::errs() << " dif: " << *dif << " adding: " << *addingType
                       << "\n";
        }
        assert(difSize >= size);
        if (CastInst::castIsValid(Instruction::CastOps::BitCast, dif,
                                  addingType))
          dif = BuilderM.CreateBitCast(dif, addingType);
        else {
          IRBuilder<> A(inversionAllocs);
          auto Al = A.CreateAlloca(addingType);
          BuilderM.CreateStore(dif,
                               BuilderM.CreatePointerCast(
                                   Al, PointerType::getUnqual(dif->getType())));
          dif = BuilderM.CreateLoad(addingType, Al);
        }
      }
      return dif;
    };
    dif = applyChainRule(addingType, BuilderM, rule, dif);
  }

  auto TmpOrig = getBaseObject(origptr);

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
  // Moreover no need to do atomic on local shadows regardless since they are
  // not captured/escaping and created in this function. This assumes that
  // all additional parallelism in this function is outlined.
  if (backwardsOnlyShadows.find(TmpOrig) != backwardsOnlyShadows.end())
    Atomic = false;

  if (Atomic) {
    // For amdgcn constant AS is 4 and if the primal is in it we need to cast
    // the derivative value to AS 1
    if (Arch == Triple::amdgcn &&
        cast<PointerType>(origptr->getType())->getAddressSpace() == 4) {
      auto rule = [&](Value *ptr) {
        return BuilderM.CreateAddrSpaceCast(ptr,
                                            PointerType::get(addingType, 1));
      };
      ptr =
          applyChainRule(PointerType::get(addingType, 1), BuilderM, rule, ptr);
    }

    assert(!mask);
    if (mask) {
      llvm::errs() << "unhandled masked atomic fadd on llvm version " << *ptr
                   << " " << *dif << " mask: " << *mask << "\n";
      llvm_unreachable("unhandled masked atomic fadd");
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
    AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
    if (auto vt = dyn_cast<VectorType>(addingType)) {
#if LLVM_VERSION_MAJOR >= 12
      assert(!vt->getElementCount().isScalable());
      size_t numElems = vt->getElementCount().getKnownMinValue();
#else
      size_t numElems = vt->getNumElements();
#endif
      auto rule = [&](Value *dif, Value *ptr) {
        for (size_t i = 0; i < numElems; ++i) {
          auto vdif = BuilderM.CreateExtractElement(dif, i);
          vdif = SanitizeDerivatives(orig, vdif, BuilderM);
          Value *Idxs[] = {
              ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
          auto vptr = BuilderM.CreateGEP(addingType, ptr, Idxs);
#if LLVM_VERSION_MAJOR >= 13
          MaybeAlign alignv = align;
          if (alignv) {
            if (start != 0) {
              // todo make better alignment calculation
#if LLVM_VERSION_MAJOR >= 16
              assert(alignv.value().value() != 0);
              if (start % alignv.value().value() != 0)
#else
              assert(alignv.getValue().value() != 0);
              if (start % alignv.getValue().value() != 0)
#endif
              {
                alignv = Align(1);
              }
            }
          }
          BuilderM.CreateAtomicRMW(op, vptr, vdif, alignv,
                                   AtomicOrdering::Monotonic,
                                   SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
          AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
              op, vptr, vdif, AtomicOrdering::Monotonic, SyncScope::System);
          if (align) {
            auto alignv = align.getValue().value();
            if (start != 0) {
              assert(alignv != 0);
              // todo make better alignment calculation
              if (start % alignv != 0) {
                alignv = 1;
              }
            }
            rmw->setAlignment(Align(alignv));
          }
#else
          BuilderM.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                   SyncScope::System);
#endif
        }
      };
      applyChainRule(BuilderM, rule, dif, ptr);
    } else {
      auto rule = [&](Value *dif, Value *ptr) {
        dif = SanitizeDerivatives(orig, dif, BuilderM);
#if LLVM_VERSION_MAJOR >= 13
        MaybeAlign alignv = align;
        if (alignv) {
          if (start != 0) {
            // todo make better alignment calculation
#if LLVM_VERSION_MAJOR >= 16
            assert(alignv.value().value() != 0);
            if (start % alignv.value().value() != 0) {
#else
            assert(alignv.getValue().value() != 0);
            if (start % alignv.getValue().value() != 0) {
#endif
              alignv = Align(1);
            }
          }
        }
        BuilderM.CreateAtomicRMW(op, ptr, dif, alignv,
                                 AtomicOrdering::Monotonic, SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
        AtomicRMWInst *rmw = BuilderM.CreateAtomicRMW(
            op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
        if (align) {
          auto alignv = align.getValue().value();
          if (start != 0) {
            assert(alignv != 0);
            // todo make better alignment calculation
            if (start % alignv != 0) {
              alignv = 1;
            }
          }
          rmw->setAlignment(Align(alignv));
        }
#else
        BuilderM.CreateAtomicRMW(op, ptr, dif, AtomicOrdering::Monotonic,
                                 SyncScope::System);
#endif
      };
      applyChainRule(BuilderM, rule, dif, ptr);
    }
    return;
  }

  if (!mask) {

    size_t idx = 0;
    auto rule = [&](Value *ptr, Value *dif) {
      auto LI = BuilderM.CreateLoad(addingType, ptr);

      Value *res = BuilderM.CreateFAdd(LI, dif);
      res = SanitizeDerivatives(orig, res, BuilderM);
      StoreInst *st = BuilderM.CreateStore(res, ptr);

      SmallVector<Metadata *, 1> scopeMD = {
          getDerivativeAliasScope(origptr, idx)};
      if (auto origValI = dyn_cast_or_null<Instruction>(origVal))
        if (auto MD = origValI->getMetadata(LLVMContext::MD_alias_scope)) {
          auto MDN = cast<MDNode>(MD);
          for (auto &o : MDN->operands())
            scopeMD.push_back(o);
        }
      auto scope = MDNode::get(LI->getContext(), scopeMD);
      LI->setMetadata(LLVMContext::MD_alias_scope, scope);
      st->setMetadata(LLVMContext::MD_alias_scope, scope);

      SmallVector<Metadata *, 1> MDs;
      for (ssize_t j = -1; j < getWidth(); j++) {
        if (j != (ssize_t)idx)
          MDs.push_back(getDerivativeAliasScope(origptr, j));
      }
      if (auto origValI = dyn_cast_or_null<Instruction>(origVal))
        if (auto MD = origValI->getMetadata(LLVMContext::MD_noalias)) {
          auto MDN = cast<MDNode>(MD);
          for (auto &o : MDN->operands())
            MDs.push_back(o);
        }
      idx++;
      auto noscope = MDNode::get(ptr->getContext(), MDs);
      LI->setMetadata(LLVMContext::MD_noalias, noscope);
      st->setMetadata(LLVMContext::MD_noalias, noscope);

      if (origVal && isa<Instruction>(origVal) && start == 0 &&
          size == (DL.getTypeSizeInBits(origVal->getType()) + 7) / 8) {
        auto origValI = cast<Instruction>(origVal);
        LI->copyMetadata(*origValI, MD_ToCopy);
        unsigned int StoreData[] = {LLVMContext::MD_tbaa,
                                    LLVMContext::MD_tbaa_struct};
        for (auto MD : StoreData)
          st->setMetadata(MD, origValI->getMetadata(MD));
      }

      LI->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));
      st->setDebugLoc(getNewFromOriginal(orig->getDebugLoc()));

      if (align) {
#if LLVM_VERSION_MAJOR >= 16
        auto alignv = align ? align.value().value() : 0;
#elif LLVM_VERSION_MAJOR >= 10
        auto alignv = align ? align.getValue().value() : 0;
#else
        auto alignv = align;
#endif
        if (alignv != 0) {
          if (start != 0) {
            // todo make better alignment calculation
            if (start % alignv != 0) {
              alignv = 1;
            }
          }
#if LLVM_VERSION_MAJOR >= 10
          LI->setAlignment(Align(alignv));
          st->setAlignment(Align(alignv));
#else
          LI->setAlignment(alignv);
          st->setAlignment(alignv);
#endif
        }
      }
    };
    applyChainRule(BuilderM, rule, ptr, dif);
  } else {
    Type *tys[] = {addingType, origptr->getType()};
    auto LF = Intrinsic::getDeclaration(oldFunc->getParent(),
                                        Intrinsic::masked_load, tys);
    auto SF = Intrinsic::getDeclaration(oldFunc->getParent(),
                                        Intrinsic::masked_store, tys);
#if LLVM_VERSION_MAJOR >= 10
    unsigned aligni = align ? align->value() : 0;
#else
    unsigned aligni = align;
#endif
    if (aligni != 0)
      if (start != 0) {
        // todo make better alignment calculation
        if (start % aligni != 0) {
          aligni = 1;
        }
      }
    Value *alignv =
        ConstantInt::get(Type::getInt32Ty(mask->getContext()), aligni);
    auto rule = [&](Value *ptr, Value *dif) {
      Value *largs[] = {ptr, alignv, mask,
                        Constant::getNullValue(dif->getType())};
      Value *LI = BuilderM.CreateCall(LF, largs);
      Value *res = BuilderM.CreateFAdd(LI, dif);
      res = SanitizeDerivatives(orig, res, BuilderM, mask);
      Value *sargs[] = {res, ptr, alignv, mask};
      BuilderM.CreateCall(SF, sargs);
    };
    applyChainRule(BuilderM, rule, ptr, dif);
  }
}

#if LLVM_VERSION_MAJOR >= 10
void DiffeGradientUtils::addToInvertedPtrDiffe(
    llvm::Instruction *orig, llvm::Value *origVal, TypeTree vd,
    unsigned LoadSize, llvm::Value *origptr, llvm::Value *prediff,
    llvm::IRBuilder<> &Builder2, MaybeAlign alignment, llvm::Value *premask)
#else
void DiffeGradientUtils::addToInvertedPtrDiffe(
    llvm::Instruction *orig, llvm::Value *origVal, TypeTree vd,
    unsigned LoadSize, llvm::Value *origptr, llvm::Value *prediff,
    llvm::IRBuilder<> &Builder2, unsigned alignment, llvm::Value *premask)
#endif
{

  unsigned start = 0;
  unsigned size = LoadSize;

  assert(prediff);

  BasicBlock *merge = nullptr;

  while (1) {
    unsigned nextStart = size;

    auto dt = vd[{-1}];
    for (size_t i = start; i < size; ++i) {
      bool Legal = true;
      dt.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
      if (!Legal) {
        nextStart = i;
        break;
      }
    }
    if (!dt.isKnown()) {
      TR.dump();
      llvm::errs() << " vd:" << vd.str() << " start:" << start
                   << " size: " << size << " dt:" << dt.str() << "\n";
    }
    assert(dt.isKnown());

    if (Type *isfloat = dt.isFloat()) {

      if (origVal) {
        if (start == 0 && nextStart == LoadSize) {
          setDiffe(origVal,
                   Constant::getNullValue(getShadowType(origVal->getType())),
                   Builder2);
        } else {
          Value *tostore = getDifferential(origVal);

          auto i8 = Type::getInt8Ty(tostore->getContext());
          if (start != 0) {
            tostore = Builder2.CreatePointerCast(
                tostore,
                PointerType::get(
                    i8,
                    cast<PointerType>(tostore->getType())->getAddressSpace()));
            auto off = ConstantInt::get(Type::getInt64Ty(tostore->getContext()),
                                        start);
            tostore = Builder2.CreateInBoundsGEP(i8, tostore, off);
          }
          auto AT = ArrayType::get(i8, nextStart - start);
          tostore = Builder2.CreatePointerCast(
              tostore,
              PointerType::get(
                  AT,
                  cast<PointerType>(tostore->getType())->getAddressSpace()));
          Builder2.CreateStore(Constant::getNullValue(AT), tostore);
        }
      }

      if (!isConstantValue(origptr)) {
        if (EnzymeRuntimeActivityCheck && !merge) {
          Value *shadow = Builder2.CreateICmpNE(
              lookupM(getNewFromOriginal(origptr), Builder2),
              lookupM(invertPointerM(origptr, Builder2), Builder2));

          BasicBlock *current = Builder2.GetInsertBlock();
          BasicBlock *conditional =
              addReverseBlock(current, current->getName() + "_active");
          merge = addReverseBlock(conditional, current->getName() + "_amerge");
          Builder2.CreateCondBr(shadow, conditional, merge);
          Builder2.SetInsertPoint(conditional);
        }
        // Masked partial type is unhanled.
        if (premask)
          assert(start == 0 && nextStart == LoadSize);
        addToInvertedPtrDiffe(orig, origVal, isfloat, start, nextStart - start,
                              origptr, prediff, Builder2, alignment, premask);
      }
    }

    if (nextStart == size)
      break;
    start = nextStart;
  }
  if (merge) {
    Builder2.CreateBr(merge);
    Builder2.SetInsertPoint(merge);
  }
}
