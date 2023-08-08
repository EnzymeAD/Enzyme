//===- AdjointGenerator.h - Implementation of Adjoint's of instructions --===//
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
// This file contains an instruction visitor AdjointGenerator that generates
// the corresponding augmented forward pass code, and adjoints for all
// LLVM instructions.
//
//===----------------------------------------------------------------------===//
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "DiffeGradientUtils.h"
#include "DifferentialUseAnalysis.h"
#include "EnzymeLogic.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"
#include "TraceUtils.h"
#include "TypeAnalysis/TBAA.h"

#define DEBUG_TYPE "enzyme"

// Helper instruction visitor that generates adjoints
template <class AugmentedReturnType = AugmentedReturn *>
class AdjointGenerator
    : public llvm::InstVisitor<AdjointGenerator<AugmentedReturnType>> {
private:
  // Type of code being generated (forward, reverse, or both)
  const DerivativeMode Mode;

  GradientUtils *const gutils;
  llvm::ArrayRef<DIFFE_TYPE> constant_args;
  DIFFE_TYPE retType;
  TypeResults &TR = gutils->TR;
  std::function<unsigned(llvm::Instruction *, CacheType)> getIndex;
  const std::map<llvm::CallInst *, const std::vector<bool>>
      overwritten_args_map;
  const llvm::SmallPtrSetImpl<llvm::Instruction *> *returnuses;
  AugmentedReturnType augmentedReturn;
  const std::map<llvm::ReturnInst *, llvm::StoreInst *> *replacedReturns;

  const llvm::SmallPtrSetImpl<const llvm::Value *> &unnecessaryValues;
  const llvm::SmallPtrSetImpl<const llvm::Instruction *>
      &unnecessaryInstructions;
  const llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryStores;
  const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable;
  llvm::AllocaInst *dretAlloca;

public:
  AdjointGenerator(
      DerivativeMode Mode, GradientUtils *gutils,
      llvm::ArrayRef<DIFFE_TYPE> constant_args, DIFFE_TYPE retType,
      std::function<unsigned(llvm::Instruction *, CacheType)> getIndex,
      const std::map<llvm::CallInst *, const std::vector<bool>>
          overwritten_args_map,
      const llvm::SmallPtrSetImpl<llvm::Instruction *> *returnuses,
      AugmentedReturnType augmentedReturn,
      const std::map<llvm::ReturnInst *, llvm::StoreInst *> *replacedReturns,
      const llvm::SmallPtrSetImpl<const llvm::Value *> &unnecessaryValues,
      const llvm::SmallPtrSetImpl<const llvm::Instruction *>
          &unnecessaryInstructions,
      const llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryStores,
      const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable,
      llvm::AllocaInst *dretAlloca)
      : Mode(Mode), gutils(gutils), constant_args(constant_args),
        retType(retType), getIndex(getIndex),
        overwritten_args_map(overwritten_args_map), returnuses(returnuses),
        augmentedReturn(augmentedReturn), replacedReturns(replacedReturns),
        unnecessaryValues(unnecessaryValues),
        unnecessaryInstructions(unnecessaryInstructions),
        unnecessaryStores(unnecessaryStores), oldUnreachable(oldUnreachable),
        dretAlloca(dretAlloca) {
    using namespace llvm;

    assert(TR.getFunction() == gutils->oldFunc);
    for (auto &pair : TR.analyzer.analysis) {
      if (auto in = dyn_cast<Instruction>(pair.first)) {
        if (in->getParent()->getParent() != gutils->oldFunc) {
          llvm::errs() << "inf: " << *in->getParent()->getParent() << "\n";
          llvm::errs() << "gutils->oldFunc: " << *gutils->oldFunc << "\n";
          llvm::errs() << "in: " << *in << "\n";
        }
        assert(in->getParent()->getParent() == gutils->oldFunc);
      }
    }
  }

  void eraseIfUnused(llvm::Instruction &I, bool erase = true,
                     bool check = true) {
    using namespace llvm;

    bool used =
        unnecessaryInstructions.find(&I) == unnecessaryInstructions.end();
    if (!used) {
      // if decided to cache a value, preserve it here for later
      // replacement in EnzymeLogic
      auto found = gutils->knownRecomputeHeuristic.find(&I);
      if (found != gutils->knownRecomputeHeuristic.end() && !found->second)
        used = true;
    }
    auto iload = gutils->getNewFromOriginal((llvm::Value *)&I);
    if (used && check)
      return;

    if (auto newi = dyn_cast<Instruction>(iload))
      gutils->eraseWithPlaceholder(newi, "_replacementA", erase);
  }

  llvm::Value *MPI_TYPE_SIZE(llvm::Value *DT, llvm::IRBuilder<> &B,
                             llvm::Type *intType) {
    using namespace llvm;

    if (DT->getType()->isIntegerTy())
      DT = B.CreateIntToPtr(DT, Type::getInt8PtrTy(DT->getContext()));

    if (Constant *C = dyn_cast<Constant>(DT)) {
      while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
        C = CE->getOperand(0);
      }
      if (auto GV = dyn_cast<GlobalVariable>(C)) {
        if (GV->getName() == "ompi_mpi_double") {
          return ConstantInt::get(intType, 8, false);
        } else if (GV->getName() == "ompi_mpi_float") {
          return ConstantInt::get(intType, 4, false);
        }
      }
    }
    Type *pargs[] = {Type::getInt8PtrTy(DT->getContext()),
                     PointerType::getUnqual(intType)};
    auto FT = FunctionType::get(intType, pargs, false);
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(intType);
    llvm::Value *args[] = {DT, alloc};
    if (DT->getType() != pargs[0])
      args[0] = B.CreateBitCast(args[0], pargs[0]);
    AttributeList AL;
    AL = AL.addParamAttribute(DT->getContext(), 0,
                              Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(DT->getContext(), 0,
                              Attribute::AttrKind::NoCapture);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(DT->getContext(), 1,
                              Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(DT->getContext(), 1,
                              Attribute::AttrKind::NoCapture);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#else
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
#if LLVM_VERSION_MAJOR < 14
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::ArgMemOnly);
    AL = AL.addAttribute(DT->getContext(), AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
    auto CI = B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Type_size", FT, AL),
        args);
#if LLVM_VERSION_MAJOR >= 16
    CI->setOnlyAccessesArgMemory();
#else
#if LLVM_VERSION_MAJOR >= 14
    CI->addAttributeAtIndex(AttributeList::FunctionIndex,
                            Attribute::ArgMemOnly);
#else
    CI->addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
#endif
#endif
    return B.CreateLoad(intType, alloc);
  }

  // To be double-checked against the functionality needed and the respective
  // implementation in Adjoint-MPI
  llvm::Value *MPI_COMM_RANK(llvm::Value *comm, llvm::IRBuilder<> &B,
                             llvm::Type *rankTy) {
    using namespace llvm;

    Type *pargs[] = {comm->getType(), PointerType::getUnqual(rankTy)};
    auto FT = FunctionType::get(rankTy, pargs, false);
    auto &context = comm->getContext();
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(rankTy);
    AttributeList AL;
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
    llvm::Value *args[] = {comm, alloc};
    B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Comm_rank", FT, AL),
        args);
    return B.CreateLoad(rankTy, alloc);
  }

  llvm::Value *MPI_COMM_SIZE(llvm::Value *comm, llvm::IRBuilder<> &B,
                             llvm::Type *rankTy) {
    using namespace llvm;

    Type *pargs[] = {comm->getType(), PointerType::getUnqual(rankTy)};
    auto FT = FunctionType::get(rankTy, pargs, false);
    auto &context = comm->getContext();
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(rankTy);
    AttributeList AL;
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::ReadOnly);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoCapture);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoUnwind);
#endif
#if LLVM_VERSION_MAJOR >= 14
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
#else
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoFree);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::NoSync);
    AL = AL.addAttribute(context, AttributeList::FunctionIndex,
                         Attribute::AttrKind::WillReturn);
#endif
    llvm::Value *args[] = {comm, alloc};
    B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Comm_size", FT, AL),
        args);
    return B.CreateLoad(rankTy, alloc);
  }

  void visitInstruction(llvm::Instruction &inst) {
    using namespace llvm;

    // TODO explicitly handle all instructions rather than using the catch all
    // below

    switch (inst.getOpcode()) {
#include "InstructionDerivatives.inc"
    default:
      break;
    }

    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << *gutils->oldFunc << "\n";
    ss << *gutils->newFunc << "\n";
    ss << "in Mode: " << to_string(Mode) << "\n";
    ss << "cannot handle unknown instruction\n" << inst;
    if (CustomErrorHandler) {
      IRBuilder<> Builder2(&inst);
      getForwardBuilder(Builder2);
      CustomErrorHandler(ss.str().c_str(), wrap(&inst), ErrorType::NoDerivative,
                         gutils, nullptr, wrap(&Builder2));
      return;
    } else {
      EmitFailure("NoDerivative", inst.getDebugLoc(), &inst, ss.str());
      return;
    }
  }

  // Common function for falling back to the implementation
  // of dual propagation, as available in invertPointerM.
  void forwardModeInvertedPointerFallback(llvm::Instruction &I) {
    using namespace llvm;

    if (gutils->isConstantValue(&I))
      return;
    auto found = gutils->invertedPointers.find(&I);
    assert(found != gutils->invertedPointers.end());
    auto placeholder = cast<PHINode>(&*found->second);
    gutils->invertedPointers.erase(found);

    if (!DifferentialUseAnalysis::is_value_needed_in_reverse<ValueType::Shadow>(
            gutils, &I, Mode, oldUnreachable)) {
      gutils->erase(placeholder);
      return;
    }

    IRBuilder<> Builder2(&I);
    getForwardBuilder(Builder2);

    auto toset = gutils->invertPointerM(&I, Builder2, /*nullShadow*/ true);

    gutils->replaceAWithB(placeholder, toset);
    placeholder->replaceAllUsesWith(toset);
    gutils->erase(placeholder);
    gutils->invertedPointers.insert(
        std::make_pair((const Value *)&I, InvertedPointerVH(gutils, toset)));
    return;
  }

  void visitAllocaInst(llvm::AllocaInst &I) {
    eraseIfUnused(I);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(I);
      return;
    }
    default:
      return;
    }
  }

  void visitICmpInst(llvm::ICmpInst &I) { eraseIfUnused(I); }

  void visitFCmpInst(llvm::FCmpInst &I) { eraseIfUnused(I); }

#if LLVM_VERSION_MAJOR >= 10
  void visitLoadLike(llvm::Instruction &I, llvm::MaybeAlign alignment,
                     bool constantval,
#else
  void visitLoadLike(llvm::Instruction &I, unsigned alignment, bool constantval,
#endif
                     llvm::Value *mask = nullptr,
                     llvm::Value *orig_maskInit = nullptr) {
    using namespace llvm;

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 1) / 8;

    assert(Mode == DerivativeMode::ForwardMode || gutils->can_modref_map);
    assert(Mode == DerivativeMode::ForwardMode ||
           gutils->can_modref_map->find(&I) != gutils->can_modref_map->end());
    bool can_modref = Mode == DerivativeMode::ForwardMode
                          ? false
                          : gutils->can_modref_map->find(&I)->second;

    constantval |= gutils->isConstantValue(&I);

    BasicBlock *parent = I.getParent();
    Type *type = gutils->getShadowType(I.getType());

    auto *newi = dyn_cast<Instruction>(gutils->getNewFromOriginal(&I));

    SmallVector<Metadata *, 1> scopeMD = {
        gutils->getDerivativeAliasScope(I.getOperand(0), -1)};
    if (auto prev = I.getMetadata(LLVMContext::MD_alias_scope)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        scopeMD.push_back(M);
      }
    }
    auto scope = MDNode::get(I.getContext(), scopeMD);
    newi->setMetadata(LLVMContext::MD_alias_scope, scope);

    SmallVector<Metadata *, 1> MDs;
    for (size_t j = 0; j < gutils->getWidth(); j++) {
      MDs.push_back(gutils->getDerivativeAliasScope(I.getOperand(0), j));
    }
    if (auto prev = I.getMetadata(LLVMContext::MD_noalias)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        MDs.push_back(M);
      }
    }
    auto noscope = MDNode::get(I.getContext(), MDs);
    newi->setMetadata(LLVMContext::MD_noalias, noscope);

    auto vd = TR.query(&I);

    IRBuilder<> BuilderZ(newi);
    if (!vd.isKnown()) {
      auto ET = I.getType();
      if (looseTypeAnalysis || true) {
        vd = defaultTypeTreeForLLVM(ET, &I);
        EmitWarning("CannotDeduceType", I, "failed to deduce type of load ", I);
        goto known;
      }
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of load " << I;
        CustomErrorHandler(str.c_str(), wrap(&I), ErrorType::NoType,
                           &TR.analyzer, nullptr, wrap(&BuilderZ));
      } else {
        EmitFailure("CannotDeduceType", I.getDebugLoc(), &I,
                    "failed to deduce type of load ", I);

        TR.intType(LoadSize, &I, /*errifnotfound*/ true,
                   /*pointerIntSame*/ true);
        llvm_unreachable("bad mti");
      }
    known:;
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      if (!constantval) {
        auto found = gutils->invertedPointers.find(&I);
        assert(found != gutils->invertedPointers.end());
        Instruction *placeholder = cast<Instruction>(&*found->second);
        assert(placeholder->getType() == type);
        gutils->invertedPointers.erase(found);

        // only make shadow where caching needed
        if (!DifferentialUseAnalysis::is_value_needed_in_reverse<
                ValueType::Shadow>(gutils, &I, Mode, oldUnreachable)) {
          gutils->erase(placeholder);
          return;
        }

        if (can_modref) {
          if (vd[{-1}].isPossiblePointer()) {
            Value *newip = gutils->cacheForReverse(
                BuilderZ, placeholder, getIndex(&I, CacheType::Shadow));
            assert(newip->getType() == type);
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)&I, InvertedPointerVH(gutils, newip)));
          } else {
            gutils->erase(placeholder);
          }
        } else {
          Value *newip = gutils->invertPointerM(&I, BuilderZ);
          if (EnzymeRuntimeActivityCheck && vd[{-1}].isFloat()) {
            // TODO handle mask
            assert(!mask);

            auto rule = [&](Value *inop, Value *newip) -> Value * {
              Value *shadow = BuilderZ.CreateICmpNE(
                  gutils->getNewFromOriginal(I.getOperand(0)), inop);
              newip = CreateSelect(BuilderZ, shadow, newip,
                                   Constant::getNullValue(newip->getType()));
              return newip;
            };
            newip = applyChainRule(
                I.getType(), BuilderZ, rule,
                gutils->invertPointerM(I.getOperand(0), BuilderZ), newip);
          }
          assert(newip->getType() == type);
          placeholder->replaceAllUsesWith(newip);
          gutils->erase(placeholder);
          gutils->invertedPointers.erase(&I);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&I, InvertedPointerVH(gutils, newip)));
        }
      }
      return;
    }

    //! Store inverted pointer loads that need to be cached for use in reverse
    //! pass
    if (vd[{-1}].isPossiblePointer()) {
      auto found = gutils->invertedPointers.find(&I);
      if (found != gutils->invertedPointers.end()) {
        Instruction *placeholder = cast<Instruction>(&*found->second);
        assert(placeholder->getType() == type);
        gutils->invertedPointers.erase(found);

        if (!constantval) {
          Value *newip = nullptr;

          // TODO: In the case of fwd mode this should be true if the loaded
          // value itself is used as a pointer.
          bool needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
              ValueType::Shadow>(gutils, &I, Mode, oldUnreachable);

          switch (Mode) {

          case DerivativeMode::ReverseModePrimal:
          case DerivativeMode::ReverseModeCombined: {
            if (!needShadow) {
              gutils->erase(placeholder);
            } else {
              newip = gutils->invertPointerM(&I, BuilderZ);
              assert(newip->getType() == type);
              if (Mode == DerivativeMode::ReverseModePrimal && can_modref &&
                  DifferentialUseAnalysis::is_value_needed_in_reverse<
                      ValueType::Shadow>(gutils, &I,
                                         DerivativeMode::ReverseModeGradient,
                                         oldUnreachable)) {
                gutils->cacheForReverse(BuilderZ, newip,
                                        getIndex(&I, CacheType::Shadow));
              }
              placeholder->replaceAllUsesWith(newip);
              gutils->erase(placeholder);
              gutils->invertedPointers.insert(std::make_pair(
                  (const Value *)&I, InvertedPointerVH(gutils, newip)));
            }
            break;
          }
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode: {
            assert(0 && "impossible branch");
            return;
          }
          case DerivativeMode::ReverseModeGradient: {
            if (!needShadow) {
              gutils->erase(placeholder);
            } else {
              // only make shadow where caching needed
              if (can_modref) {
                newip = gutils->cacheForReverse(
                    BuilderZ, placeholder, getIndex(&I, CacheType::Shadow));
                assert(newip->getType() == type);
                gutils->invertedPointers.insert(std::make_pair(
                    (const Value *)&I, InvertedPointerVH(gutils, newip)));
              } else {
                newip = gutils->invertPointerM(&I, BuilderZ);
                assert(newip->getType() == type);
                placeholder->replaceAllUsesWith(newip);
                gutils->erase(placeholder);
                gutils->invertedPointers.insert(std::make_pair(
                    (const Value *)&I, InvertedPointerVH(gutils, newip)));
              }
            }
            break;
          }
          }

        } else {
          gutils->erase(placeholder);
        }
      }
    }

    Value *inst = newi;

    //! Store loads that need to be cached for use in reverse pass

    // Only cache value here if caching decision isn't precomputed.
    // Otherwise caching will be done inside EnzymeLogic.cpp at
    // the end of the function jointly.
    if (Mode != DerivativeMode::ForwardMode &&
        !gutils->knownRecomputeHeuristic.count(&I) && can_modref &&
        !gutils->unnecessaryIntermediates.count(&I)) {
      // we can pre initialize all the knownRecomputeHeuristic values to false
      // (not needing) as we may assume that minCutCache already preserves
      // everything it requires.
      std::map<UsageKey, bool> Seen;
      bool primalNeededInReverse = false;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second) {
          Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          if (pair.first == &I)
            primalNeededInReverse = true;
        }
      primalNeededInReverse |=
          DifferentialUseAnalysis::is_value_needed_in_reverse<
              ValueType::Primal>(gutils, &I, Mode, Seen, oldUnreachable);
      if (primalNeededInReverse) {
        inst = gutils->cacheForReverse(BuilderZ, newi,
                                       getIndex(&I, CacheType::Self));
        assert(inst->getType() == type);

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          assert(inst != newi);
        } else {
          assert(inst == newi);
        }
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal)
      return;

    if (constantval)
      return;

    if (nonmarkedglobals_inactiveloads) {
      // Assume that non enzyme_shadow globals are inactive
      //  If we ever store to a global variable, we will error if it doesn't
      //  have a shadow This allows functions who only read global memory to
      //  have their derivative computed Note that this is too aggressive for
      //  general programs as if the global aliases with an argument something
      //  that is written to, then we will have a logical error
      if (auto arg = dyn_cast<GlobalVariable>(I.getOperand(0))) {
        if (!hasMetadata(arg, "enzyme_shadow")) {
          return;
        }
      }
    }

    // Only propagate if instruction is active. The value can be active and not
    // the instruction if the value is a potential pointer. This may not be
    // caught by type analysis is the result does not have a known type.
    if (!gutils->isConstantInstruction(&I)) {
      switch (Mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        assert(0 && "impossible branch");
        return;
      }
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined: {

        IRBuilder<> Builder2(&I);
        getReverseBuilder(Builder2);

        Value *prediff = nullptr;

        for (ssize_t i = -1; i < (ssize_t)LoadSize; ++i) {
          if (vd[{(int)i}].isFloat()) {
            prediff = diffe(&I, Builder2);
            break;
          }
        }

        Value *premask = nullptr;

        if (prediff && mask &&
            (!gutils->isConstantValue(I.getOperand(0)) ||
             !gutils->isConstantValue(orig_maskInit))) {
          premask = lookup(mask, Builder2);
        }

        if (prediff)
          ((DiffeGradientUtils *)gutils)
              ->addToInvertedPtrDiffe(&I, &I, vd, LoadSize, I.getOperand(0),
                                      prediff, Builder2, alignment, premask);

        unsigned start = 0;
        unsigned size = LoadSize;

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
            if (premask && !gutils->isConstantValue(orig_maskInit)) {
              // Masked partial type is unhanled.
              if (premask)
                assert(start == 0 && nextStart == LoadSize);
              addToDiffe(orig_maskInit, prediff, Builder2, isfloat,
                         Builder2.CreateNot(premask));
            }
          }

          if (nextStart == size)
            break;
          start = nextStart;
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:
        break;
      }
    }
  }

  void visitLoadInst(llvm::LoadInst &LI) {
    using namespace llvm;

    // If a load of an omp init argument, don't cache for reverse
    // and don't do any adjoint propagation (assumed integral)
    for (auto U : LI.getPointerOperand()->users()) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            eraseIfUnused(LI);
            return;
          }
        }
      }
    }

#if LLVM_VERSION_MAJOR >= 10
    auto alignment = LI.getAlign();
#else
    auto alignment = LI.getAlignment();
#endif

    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    bool constantval = parseTBAA(LI, DL).Inner0().isIntegral();
    visitLoadLike(LI, alignment, constantval);
    eraseIfUnused(LI);
  }

  void visitAtomicRMWInst(llvm::AtomicRMWInst &I) {
    using namespace llvm;

    if (gutils->isConstantInstruction(&I) && gutils->isConstantValue(&I)) {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(I, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(I);
      }
      return;
    }

    IRBuilder<> BuilderZ(&I);
    getForwardBuilder(BuilderZ);

    switch (I.getOperation()) {
    case AtomicRMWInst::FAdd:
    case AtomicRMWInst::FSub: {

      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeSplit) {
        auto rule = [&](Value *ptr, Value *dif) -> Value * {
          if (dif == nullptr)
            dif = Constant::getNullValue(I.getType());
          if (!gutils->isConstantInstruction(&I)) {
            assert(ptr);
            AtomicRMWInst *rmw = nullptr;
#if LLVM_VERSION_MAJOR >= 13
            rmw = BuilderZ.CreateAtomicRMW(I.getOperation(), ptr, dif,
                                           I.getAlign(), I.getOrdering(),
                                           I.getSyncScopeID());
#elif LLVM_VERSION_MAJOR >= 11
            rmw = BuilderZ.CreateAtomicRMW(I.getOperation(), ptr, dif,
                                           I.getOrdering(), I.getSyncScopeID());
            rmw->setAlignment(I.getAlign());
#else
                               rmw = BuilderZ.CreateAtomicRMW(
                                   I.getOperation(), ptr, dif, I.getOrdering(),
                                   I.getSyncScopeID());
#endif
            rmw->setVolatile(I.isVolatile());
            if (gutils->isConstantValue(&I))
              return Constant::getNullValue(dif->getType());
            else
              return rmw;
          } else {
            assert(gutils->isConstantValue(&I));
            return Constant::getNullValue(dif->getType());
          }
        };

        Value *diff = applyChainRule(
            I.getType(), BuilderZ, rule,
            gutils->isConstantValue(I.getPointerOperand())
                ? nullptr
                : gutils->invertPointerM(I.getPointerOperand(), BuilderZ),
            gutils->isConstantValue(I.getValOperand())
                ? nullptr
                : gutils->invertPointerM(I.getValOperand(), BuilderZ));
        if (!gutils->isConstantValue(&I))
          setDiffe(&I, diff, BuilderZ);
        return;
      }
      if (Mode == DerivativeMode::ReverseModePrimal) {
        eraseIfUnused(I);
        return;
      }
      if ((Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient) &&
          gutils->isConstantValue(&I)) {
        if (!gutils->isConstantValue(I.getValOperand())) {
          assert(!gutils->isConstantValue(I.getPointerOperand()));
          IRBuilder<> Builder2(&I);
          getReverseBuilder(Builder2);
          Value *ip = gutils->invertPointerM(I.getPointerOperand(), Builder2);
          ip = lookup(ip, Builder2);
          auto order = I.getOrdering();
          if (order == AtomicOrdering::Release)
            order = AtomicOrdering::Monotonic;
          else if (order == AtomicOrdering::AcquireRelease)
            order = AtomicOrdering::Acquire;

          auto rule = [&](Value *ip) -> Value * {
            LoadInst *dif1 =
                Builder2.CreateLoad(I.getType(), ip, I.isVolatile());

#if LLVM_VERSION_MAJOR >= 11
            dif1->setAlignment(I.getAlign());
#else
            const DataLayout &DL = I.getModule()->getDataLayout();
            auto tmpAlign = DL.getTypeStoreSize(I.getValOperand()->getType());
#if LLVM_VERSION_MAJOR >= 10
            dif1->setAlignment(MaybeAlign(tmpAlign.getFixedSize()));
#else
            dif1->setAlignment(tmpAlign);
#endif
#endif
            dif1->setOrdering(order);
            dif1->setSyncScopeID(I.getSyncScopeID());
            return dif1;
          };
          Value *diff = applyChainRule(I.getType(), Builder2, rule, ip);

          addToDiffe(I.getValOperand(), diff, Builder2,
                     I.getValOperand()->getType()->getScalarType());
        }
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(I, /*erase*/ true, /*check*/ false);
        } else
          eraseIfUnused(I);
        return;
      }
      break;
    }
    default:
      break;
    }

    if (looseTypeAnalysis) {
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      auto valType = I.getValOperand()->getType();
      auto storeSize = DL.getTypeSizeInBits(valType) / 8;
      auto fp = TR.firstPointer(storeSize, I.getPointerOperand(), &I,
                                /*errifnotfound*/ false,
                                /*pointerIntSame*/ true);
      if (!fp.isKnown() && valType->isIntOrIntVectorTy()) {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(I, /*erase*/ true, /*check*/ false);
        } else
          eraseIfUnused(I);
        return;
      }
    }
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << *I.getParent()->getParent() << "\n" << I << "\n";
    ss << " Active atomic inst not yet handled";
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(&I), ErrorType::NoDerivative,
                         gutils, nullptr, wrap(&BuilderZ));
      return;
    } else {
      EmitFailure("NoDerivative", I.getDebugLoc(), &I, ss.str());
      return;
    }
  }

  void visitStoreInst(llvm::StoreInst &SI) {
    using namespace llvm;

    // If a store of an omp init argument, don't delete in reverse
    // and don't do any adjoint propagation (assumed integral)
    for (auto U : SI.getPointerOperand()->users()) {
      if (auto CI = dyn_cast<CallInst>(U)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            return;
          }
        }
      }
    }
#if LLVM_VERSION_MAJOR >= 10
    auto align = SI.getAlign();
#else
    auto align = SI.getAlignment();
#endif

    visitCommonStore(SI, SI.getPointerOperand(), SI.getValueOperand(), align,
                     SI.isVolatile(), SI.getOrdering(), SI.getSyncScopeID(),
                     /*mask=*/nullptr);

    bool forceErase = false;
    if (Mode == DerivativeMode::ReverseModeGradient) {
      for (const auto &pair : gutils->rematerializableAllocations) {
        if (pair.second.stores.count(&SI) && pair.second.LI) {
          forceErase = true;
        }
      }
    }
    if (forceErase)
      eraseIfUnused(SI, /*erase*/ true, /*check*/ false);
    else
      eraseIfUnused(SI);
  }

#if LLVM_VERSION_MAJOR >= 10
  void visitCommonStore(llvm::Instruction &I, llvm::Value *orig_ptr,
                        llvm::Value *orig_val, llvm::MaybeAlign align,
                        bool isVolatile, llvm::AtomicOrdering ordering,
                        llvm::SyncScope::ID syncScope, llvm::Value *mask)
#else
  void visitCommonStore(llvm::Instruction &I, llvm::Value *orig_ptr,
                        llvm::Value *orig_val, unsigned align, bool isVolatile,
                        llvm::AtomicOrdering ordering,
                        llvm::SyncScope::ID syncScope, llvm::Value *mask)
#endif
  {
    using namespace llvm;

    Value *val = gutils->getNewFromOriginal(orig_val);
    Type *valType = orig_val->getType();

    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    if (unnecessaryStores.count(&I)) {
      return;
    }

    if (gutils->isConstantValue(orig_ptr)) {
      return;
    }

    SmallVector<Metadata *, 1> scopeMD = {
        gutils->getDerivativeAliasScope(orig_ptr, -1)};
    SmallVector<Metadata *, 1> prevScopes;
    if (auto prev = I.getMetadata(LLVMContext::MD_alias_scope)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        scopeMD.push_back(M);
        prevScopes.push_back(M);
      }
    }
    auto scope = MDNode::get(I.getContext(), scopeMD);
    auto NewI = gutils->getNewFromOriginal(&I);
    NewI->setMetadata(LLVMContext::MD_alias_scope, scope);

    SmallVector<Metadata *, 1> MDs;
    SmallVector<Metadata *, 1> prevNoAlias;
    for (size_t j = 0; j < gutils->getWidth(); j++) {
      MDs.push_back(gutils->getDerivativeAliasScope(orig_ptr, j));
    }
    if (auto prev = I.getMetadata(LLVMContext::MD_noalias)) {
      for (auto &M : cast<MDNode>(prev)->operands()) {
        MDs.push_back(M);
        prevNoAlias.push_back(M);
      }
    }
    auto noscope = MDNode::get(I.getContext(), MDs);
    NewI->setMetadata(LLVMContext::MD_noalias, noscope);

    bool constantval = gutils->isConstantValue(orig_val) ||
                       parseTBAA(I, DL).Inner0().isIntegral();

    IRBuilder<> BuilderZ(NewI);
    BuilderZ.setFastMathFlags(getFast());

    // TODO allow recognition of other types that could contain pointers [e.g.
    // {void*, void*} or <2 x i64> ]
    auto storeSize = DL.getTypeSizeInBits(valType) / 8;

    auto vd = TR.query(orig_ptr).Lookup(storeSize, DL);

    if (!vd.isKnown()) {
      if (looseTypeAnalysis || true) {
        vd = defaultTypeTreeForLLVM(valType, &I);
        EmitWarning("CannotDeduceType", I, "failed to deduce type of xtore ",
                    I);
        goto known;
      }
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of store " << I;
        CustomErrorHandler(str.c_str(), wrap(&I), ErrorType::NoType,
                           &TR.analyzer, nullptr, wrap(&BuilderZ));
        return;
      } else {
        EmitFailure("CannotDeduceType", I.getDebugLoc(), &I,
                    "failed to deduce type of store ", I);

        TR.intType(storeSize, orig_ptr, /*errifnotfound*/ true,
                   /*pointerIntSame*/ true);
        llvm_unreachable("bad mti");
      }
    known:;
    }

    auto dt = vd[{-1}];
    for (size_t i = 0; i < storeSize; ++i) {
      bool Legal = true;
      dt.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
      if (!Legal) {
        if (CustomErrorHandler) {
          std::string str;
          raw_string_ostream ss(str);
          ss << "Cannot deduce single type of store " << I;
          CustomErrorHandler(str.c_str(), wrap(&I), ErrorType::NoType,
                             &TR.analyzer, nullptr, wrap(&BuilderZ));
          return;
        } else {
          EmitFailure("CannotDeduceType", I.getDebugLoc(), &I,
                      "failed to deduce single type of store ", I);
        }
      }
    }

    if (Mode == DerivativeMode::ForwardMode) {

      Value *diff = nullptr;
      if (!EnzymeRuntimeActivityCheck && CustomErrorHandler && constantval) {
        if (dt.isPossiblePointer() && vd[{-1, -1}] != BaseType::Integer) {
          if (!isa<UndefValue>(orig_val) &&
              !isa<ConstantPointerNull>(orig_val)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << I
               << " const val: " << *orig_val;
            diff = unwrap(CustomErrorHandler(
                str.c_str(), wrap(&I), ErrorType::MixedActivityError, gutils,
                wrap(orig_val), wrap(&BuilderZ)));
          }
        }
      }

      // TODO type analyze
      if (!diff) {
        if (!constantval)
          diff =
              gutils->invertPointerM(orig_val, BuilderZ, /*nullShadow*/ true);
        else if (orig_val->getType()->isPointerTy() ||
                 dt == BaseType::Pointer || dt == BaseType::Integer)
          diff =
              gutils->invertPointerM(orig_val, BuilderZ, /*nullShadow*/ false);
        else
          diff =
              gutils->invertPointerM(orig_val, BuilderZ, /*nullShadow*/ true);
      }

      gutils->setPtrDiffe(&I, orig_ptr, diff, BuilderZ, align, isVolatile,
                          ordering, syncScope, mask, prevNoAlias, prevScopes);

      return;
    }

    //! Storing a floating point value
    if (Type *FT = dt.isFloat()) {
      //! Only need to update the reverse function
      switch (Mode) {
      case DerivativeMode::ReverseModePrimal:
        break;
      case DerivativeMode::ReverseModeGradient:
      case DerivativeMode::ReverseModeCombined: {
        IRBuilder<> Builder2(&I);
        getReverseBuilder(Builder2);

        if (constantval) {
          gutils->setPtrDiffe(
              &I, orig_ptr,
              Constant::getNullValue(gutils->getShadowType(valType)), Builder2,
              align, isVolatile, ordering, syncScope, mask, prevNoAlias,
              prevScopes);
        } else {
          Value *diff;
          if (!mask) {
            Value *dif1Ptr =
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

            size_t idx = 0;
            auto rule = [&](Value *dif1Ptr) {
              LoadInst *dif1 =
                  Builder2.CreateLoad(valType, dif1Ptr, isVolatile);
              if (align)
#if LLVM_VERSION_MAJOR >= 11
                dif1->setAlignment(*align);
#else
                dif1->setAlignment(align);
#endif
              dif1->setOrdering(ordering);
              dif1->setSyncScopeID(syncScope);

              SmallVector<Metadata *, 1> scopeMD = {
                  gutils->getDerivativeAliasScope(orig_ptr, idx)};
              for (auto M : prevScopes)
                scopeMD.push_back(M);

              SmallVector<Metadata *, 1> MDs;
              for (ssize_t j = -1; j < gutils->getWidth(); j++) {
                if (j != (ssize_t)idx)
                  MDs.push_back(gutils->getDerivativeAliasScope(orig_ptr, j));
              }
              for (auto M : prevNoAlias)
                MDs.push_back(M);

              dif1->setMetadata(LLVMContext::MD_alias_scope,
                                MDNode::get(I.getContext(), scopeMD));
              dif1->setMetadata(LLVMContext::MD_noalias,
                                MDNode::get(I.getContext(), MDs));
              dif1->setMetadata(LLVMContext::MD_tbaa,
                                I.getMetadata(LLVMContext::MD_tbaa));
              dif1->setMetadata(LLVMContext::MD_tbaa_struct,
                                I.getMetadata(LLVMContext::MD_tbaa_struct));
              idx++;
              return dif1;
            };

            diff = applyChainRule(valType, Builder2, rule, dif1Ptr);
          } else {
            mask = lookup(mask, Builder2);
            Type *tys[] = {valType, orig_ptr->getType()};
            auto F = Intrinsic::getDeclaration(gutils->oldFunc->getParent(),
                                               Intrinsic::masked_load, tys);
#if LLVM_VERSION_MAJOR >= 10
            Value *alignv =
                ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                 align ? align->value() : 0);
#else
            Value *alignv =
                ConstantInt::get(Type::getInt32Ty(mask->getContext()), align);
#endif
            Value *ip =
                lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

            auto rule = [&](Value *ip) {
              Value *args[] = {ip, alignv, mask,
                               Constant::getNullValue(valType)};
              diff = Builder2.CreateCall(F, args);
              return diff;
            };

            diff = applyChainRule(valType, Builder2, rule, ip);
          }

          gutils->setPtrDiffe(
              &I, orig_ptr,
              Constant::getNullValue(gutils->getShadowType(valType)), Builder2,
              align, isVolatile, ordering, syncScope, mask, prevNoAlias,
              prevScopes);
          addToDiffe(orig_val, diff, Builder2, FT, mask);
        }
        break;
      }
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        IRBuilder<> Builder2(&I);
        getForwardBuilder(Builder2);

        Type *diffeTy = gutils->getShadowType(valType);

        Value *diff = constantval ? Constant::getNullValue(diffeTy)
                                  : diffe(orig_val, Builder2);
        gutils->setPtrDiffe(&I, orig_ptr, diff, Builder2, align, isVolatile,
                            ordering, syncScope, mask, prevNoAlias, prevScopes);

        break;
      }
      }

      //! Storing an integer or pointer
    } else {
      //! Only need to update the forward function

      // Don't reproduce mpi null requests
      if (constantval)
        if (Constant *C = dyn_cast<Constant>(orig_val)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_request_null") {
              return;
            }
          }
        }

      bool backwardsShadow = false;
      bool forwardsShadow = true;
      for (auto pair : gutils->backwardsOnlyShadows) {
        if (pair.second.stores.count(&I)) {
          backwardsShadow = true;
          forwardsShadow = pair.second.primalInitialize;
          if (auto inst = dyn_cast<Instruction>(pair.first))
            if (!forwardsShadow && pair.second.LI &&
                pair.second.LI->contains(inst->getParent()))
              backwardsShadow = false;
        }
      }

      if ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined &&
           (forwardsShadow || backwardsShadow)) ||
          Mode == DerivativeMode::ForwardMode) {

        Value *valueop = nullptr;

        if (constantval) {
          if (!EnzymeRuntimeActivityCheck && CustomErrorHandler) {
            if (dt.isPossiblePointer() && vd[{-1, -1}] != BaseType::Integer) {
              if (!isa<UndefValue>(orig_val) &&
                  !isa<ConstantPointerNull>(orig_val)) {
                std::string str;
                raw_string_ostream ss(str);
                ss << "Mismatched activity for: " << I
                   << " const val: " << *orig_val;
                valueop = unwrap(CustomErrorHandler(
                    str.c_str(), wrap(&I), ErrorType::MixedActivityError,
                    gutils, wrap(orig_val), wrap(&BuilderZ)));
              }
            }
          }
          if (!valueop) {
            valueop = val;
            if (gutils->getWidth() > 1) {
              Value *array =
                  UndefValue::get(gutils->getShadowType(val->getType()));
              for (unsigned i = 0; i < gutils->getWidth(); ++i) {
                array = BuilderZ.CreateInsertValue(array, val, {i});
              }
              valueop = array;
            }
          }
        } else {
          valueop = gutils->invertPointerM(orig_val, BuilderZ);
        }
        gutils->setPtrDiffe(&I, orig_ptr, valueop, BuilderZ, align, isVolatile,
                            ordering, syncScope, mask, prevNoAlias, prevScopes);
      }
    }
  }

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep) {
    eraseIfUnused(gep);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(gep);
      return;
    }
    default:
      return;
    }
  }

  void visitPHINode(llvm::PHINode &phi) {
    eraseIfUnused(phi);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(phi);
      return;
    }
    }
  }

  void visitCastInst(llvm::CastInst &I) {
    using namespace llvm;

    eraseIfUnused(I);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&I))
        return;

      if (I.getType()->isPointerTy() ||
          I.getOpcode() == CastInst::CastOps::PtrToInt)
        return;

      Value *orig_op0 = I.getOperand(0);
      Value *op0 = gutils->getNewFromOriginal(orig_op0);

      IRBuilder<> Builder2(&I);
      getReverseBuilder(Builder2);

      if (!gutils->isConstantValue(orig_op0)) {
        size_t size = 1;
        if (orig_op0->getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   orig_op0->getType()) +
               7) /
              8;
        Type *FT = TR.addingType(size, orig_op0);
        if (!FT) {
          std::string str;
          raw_string_ostream ss(str);
          ss << "Cannot deduce adding type of " << I;
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&I), ErrorType::NoType,
                               &TR.analyzer, nullptr, wrap(&Builder2));
            return;
          } else {
            TR.dump();
            EmitFailure("CannotDeduceType", I.getDebugLoc(), &I, ss.str());
            return;
          }
        }
        assert(FT);

        auto rule = [&](Value *dif) {
          if (I.getOpcode() == CastInst::CastOps::FPTrunc ||
              I.getOpcode() == CastInst::CastOps::FPExt) {
            return Builder2.CreateFPCast(dif, op0->getType());
          } else if (I.getOpcode() == CastInst::CastOps::BitCast) {
            return Builder2.CreateBitCast(dif, op0->getType());
          } else if (I.getOpcode() == CastInst::CastOps::Trunc) {
            // TODO CHECK THIS
            return Builder2.CreateZExt(dif, op0->getType());
          } else {
            std::string s;
            llvm::raw_string_ostream ss(s);
            ss << *I.getParent()->getParent() << "\n";
            ss << "cannot handle above cast " << I << "\n";
            if (CustomErrorHandler) {
              CustomErrorHandler(ss.str().c_str(), wrap(&I),
                                 ErrorType::NoDerivative, gutils, nullptr,
                                 wrap(&Builder2));
              return (llvm::Value *)UndefValue::get(op0->getType());
            } else {
              TR.dump();
              llvm::errs() << ss.str() << "\n";
              report_fatal_error("unknown instruction");
            }
          }
        };

        Value *dif = diffe(&I, Builder2);
        Value *diff = applyChainRule(op0->getType(), Builder2, rule, dif);

        addToDiffe(orig_op0, diff, Builder2, FT);
      }

      Type *diffTy = gutils->getShadowType(I.getType());
      setDiffe(&I, Constant::getNullValue(diffTy), Builder2);

      break;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(I);
      return;
    }
    }
  }

  void visitSelectInst(llvm::SelectInst &SI) {
    eraseIfUnused(SI);

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal:
      return;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      if (gutils->isConstantInstruction(&SI))
        return;
      if (SI.getType()->isPointerTy())
        return;
      createSelectInstAdjoint(SI);
      return;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(SI);
      return;
    }
    }
  }

  void createSelectInstAdjoint(llvm::SelectInst &SI) {
    using namespace llvm;

    Value *op0 = gutils->getNewFromOriginal(SI.getOperand(0));
    Value *orig_op1 = SI.getOperand(1);
    Value *op1 = gutils->getNewFromOriginal(orig_op1);
    Value *orig_op2 = SI.getOperand(2);
    Value *op2 = gutils->getNewFromOriginal(orig_op2);

    // TODO fix all the reverse builders
    IRBuilder<> Builder2(&SI);
    getReverseBuilder(Builder2);

    Value *dif1 = nullptr;
    Value *dif2 = nullptr;

    size_t size = 1;
    if (orig_op1->getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  orig_op1->getType()) +
              7) /
             8;
    // Required loopy phi = [in, BO, BO, ..., BO]
    //  1) phi is only used in this B0
    //  2) BO dominates all latches
    //  3) phi == B0 whenever not coming from preheader [implies 2]
    //  4) [optional but done for ease] one exit to make it easier to
    //  calculation the product at that point
    for (int i = 0; i < 2; i++)
      if (auto P0 = dyn_cast<PHINode>(SI.getOperand(i + 1))) {
        LoopContext lc;
        SmallVector<Instruction *, 4> activeUses;
        for (auto u : P0->users()) {
          if (!gutils->isConstantInstruction(cast<Instruction>(u))) {
            activeUses.push_back(cast<Instruction>(u));
          } else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
            activeUses.push_back(cast<Instruction>(u));
        }
        if (activeUses.size() == 1 && activeUses[0] == &SI &&
            gutils->getContext(gutils->getNewFromOriginal(P0->getParent()),
                               lc) &&
            gutils->getNewFromOriginal(P0->getParent()) == lc.header) {
          SmallVector<BasicBlock *, 1> Latches;
          gutils->OrigLI.getLoopFor(P0->getParent())->getLoopLatches(Latches);
          bool allIncoming = true;
          for (auto Latch : Latches) {
            if (&SI != P0->getIncomingValueForBlock(Latch)) {
              allIncoming = false;
              break;
            }
          }
          if (allIncoming && lc.exitBlocks.size() == 1) {
            if (!gutils->isConstantValue(SI.getOperand(2 - i))) {
              auto addingType = TR.addingType(size, SI.getOperand(2 - i));
              if (addingType || !looseTypeAnalysis) {
                auto index = gutils->getOrInsertConditionalIndex(
                    gutils->getNewFromOriginal(SI.getOperand(0)), lc, i == 1);
                IRBuilder<> EB(*lc.exitBlocks.begin());
                getReverseBuilder(EB, /*original=*/false);
                Value *inc = lookup(lc.incvar, Builder2);
                if (VectorType *VTy =
                        dyn_cast<VectorType>(SI.getOperand(0)->getType())) {
#if LLVM_VERSION_MAJOR >= 12
                  inc = Builder2.CreateVectorSplat(VTy->getElementCount(), inc);
#else
                  inc = Builder2.CreateVectorSplat(VTy->getNumElements(), inc);
#endif
                }
                Value *dif = CreateSelect(
                    Builder2,
                    Builder2.CreateICmpEQ(gutils->lookupM(index, EB), inc),
                    diffe(&SI, Builder2),
                    Constant::getNullValue(
                        gutils->getShadowType(op1->getType())));
                addToDiffe(SI.getOperand(2 - i), dif, Builder2, addingType);
              }
            }
            return;
          }
        }
      }

    if (!gutils->isConstantValue(orig_op1))
      dif1 = CreateSelect(
          Builder2, lookup(op0, Builder2), diffe(&SI, Builder2),
          Constant::getNullValue(gutils->getShadowType(op1->getType())),
          "diffe" + op1->getName());
    if (!gutils->isConstantValue(orig_op2))
      dif2 = CreateSelect(
          Builder2, lookup(op0, Builder2),
          Constant::getNullValue(gutils->getShadowType(op2->getType())),
          diffe(&SI, Builder2), "diffe" + op2->getName());

    setDiffe(&SI, Constant::getNullValue(gutils->getShadowType(SI.getType())),
             Builder2);
    if (dif1) {
      Type *addingType = TR.addingType(size, orig_op1);
      if (addingType || !looseTypeAnalysis)
        addToDiffe(orig_op1, dif1, Builder2, addingType);
      else
        llvm::errs() << " warning: assuming integral for " << SI << "\n";
    }
    if (dif2) {
      Type *addingType = TR.addingType(size, orig_op2);
      if (addingType || !looseTypeAnalysis)
        addToDiffe(orig_op2, dif2, Builder2, addingType);
      else
        llvm::errs() << " warning: assuming integral for " << SI << "\n";
    }
  }

  void visitExtractElementInst(llvm::ExtractElementInst &EEI) {
    using namespace llvm;

    eraseIfUnused(EEI);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(EEI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&EEI))
        return;
      IRBuilder<> Builder2(&EEI);
      getReverseBuilder(Builder2);

      Value *orig_vec = EEI.getVectorOperand();

      if (!gutils->isConstantValue(orig_vec)) {
        Value *sv[] = {gutils->getNewFromOriginal(EEI.getIndexOperand())};

        size_t size = 1;
        if (EEI.getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EEI.getType()) +
               7) /
              8;
        ((DiffeGradientUtils *)gutils)
            ->addToDiffe(orig_vec, diffe(&EEI, Builder2), Builder2,
                         TR.addingType(size, &EEI), sv);
      }
      setDiffe(&EEI,
               Constant::getNullValue(gutils->getShadowType(EEI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitInsertElementInst(llvm::InsertElementInst &IEI) {
    using namespace llvm;

    eraseIfUnused(IEI);

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(IEI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&IEI))
        return;
      IRBuilder<> Builder2(&IEI);
      getReverseBuilder(Builder2);

      Value *dif1 = diffe(&IEI, Builder2);

      Value *orig_op0 = IEI.getOperand(0);
      Value *orig_op1 = IEI.getOperand(1);
      Value *op1 = gutils->getNewFromOriginal(orig_op1);
      Value *op2 = gutils->getNewFromOriginal(IEI.getOperand(2));

      size_t size0 = 1;
      if (orig_op0->getType()->isSized())
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_op0->getType()) +
             7) /
            8;
      size_t size1 = 1;
      if (orig_op1->getType()->isSized())
        size1 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_op1->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_op0))
        addToDiffe(
            orig_op0,
            Builder2.CreateInsertElement(
                dif1,
                Constant::getNullValue(gutils->getShadowType(op1->getType())),
                lookup(op2, Builder2)),
            Builder2, TR.addingType(size0, orig_op0));

      if (!gutils->isConstantValue(orig_op1))
        addToDiffe(orig_op1,
                   Builder2.CreateExtractElement(dif1, lookup(op2, Builder2)),
                   Builder2, TR.addingType(size1, orig_op1));

      setDiffe(&IEI,
               Constant::getNullValue(gutils->getShadowType(IEI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitShuffleVectorInst(llvm::ShuffleVectorInst &SVI) {
    using namespace llvm;

    eraseIfUnused(SVI);

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(SVI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&SVI))
        return;
      IRBuilder<> Builder2(&SVI);
      getReverseBuilder(Builder2);

      auto loaded = diffe(&SVI, Builder2);
#if LLVM_VERSION_MAJOR >= 12
      auto count =
          cast<VectorType>(SVI.getOperand(0)->getType())->getElementCount();
      assert(!count.isScalable());
      size_t l1 = count.getKnownMinValue();
#else
      size_t l1 =
          cast<VectorType>(SVI.getOperand(0)->getType())->getNumElements();
#endif
      uint64_t instidx = 0;

      for (size_t idx : SVI.getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx - l1);
        Value *sv[] = {
            ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx)};

        if (!gutils->isConstantValue(SVI.getOperand(opnum))) {
          size_t size = 1;
          if (SVI.getOperand(opnum)->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(SVI.getOperand(opnum)->getType()) +
                    7) /
                   8;
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(SVI.getOperand(opnum),
                           Builder2.CreateExtractElement(loaded, instidx),
                           Builder2, TR.addingType(size, SVI.getOperand(opnum)),
                           sv);
        }
        ++instidx;
      }
      setDiffe(&SVI,
               Constant::getNullValue(gutils->getShadowType(SVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitExtractValueInst(llvm::ExtractValueInst &EVI) {
    using namespace llvm;

    eraseIfUnused(EVI);

    if (!gutils->isConstantValue(&EVI) && gutils->isConstantValue(&EVI)) {
      llvm::errs() << *gutils->oldFunc->getParent() << "\n";
      llvm::errs() << EVI << "\n";
      llvm_unreachable("Illegal activity for extractvalue");
    }

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {
      forwardModeInvertedPointerFallback(EVI);
      return;
    }
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      if (gutils->isConstantInstruction(&EVI))
        return;
      if (EVI.getType()->isPointerTy())
        return;
      IRBuilder<> Builder2(&EVI);
      getReverseBuilder(Builder2);

      Value *orig_op0 = EVI.getOperand(0);

      auto prediff = diffe(&EVI, Builder2);

      // todo const
      if (!gutils->isConstantValue(orig_op0)) {
        SmallVector<Value *, 4> sv;
        for (auto i : EVI.getIndices())
          sv.push_back(ConstantInt::get(Type::getInt32Ty(EVI.getContext()), i));
        size_t size = 1;
        if (EVI.getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EVI.getType()) +
               7) /
              8;
        for (size_t i = 0; i < gutils->getWidth(); ++i) {
          Value *tdiff = (gutils->getWidth() == 1)
                             ? prediff
                             : gutils->extractMeta(Builder2, prediff, i);
          SmallVector<Value *, 4> sv2 = sv;
          if (gutils->getWidth() != 1)
            sv2.insert(sv2.begin(),
                       ConstantInt::get(Type::getInt32Ty(EVI.getContext()), i));
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(orig_op0, tdiff, Builder2, TR.addingType(size, &EVI),
                           sv2);
        }
      }

      setDiffe(&EVI,
               Constant::getNullValue(gutils->getShadowType(EVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void visitInsertValueInst(llvm::InsertValueInst &IVI) {
    using namespace llvm;

    eraseIfUnused(IVI);
    if (gutils->isConstantValue(&IVI))
      return;

    if (Mode == DerivativeMode::ReverseModePrimal)
      return;

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      forwardModeInvertedPointerFallback(IVI);
      return;
    }

    bool hasNonPointer = false;
    if (auto st = dyn_cast<StructType>(IVI.getType())) {
      for (unsigned i = 0; i < st->getNumElements(); ++i) {
        if (!st->getElementType(i)->isPointerTy()) {
          hasNonPointer = true;
        }
      }
    } else if (auto at = dyn_cast<ArrayType>(IVI.getType())) {
      if (!at->getElementType()->isPointerTy()) {
        hasNonPointer = true;
      }
    }
    if (!hasNonPointer)
      return;

    bool floatingInsertion = false;
    for (InsertValueInst *iv = &IVI;;) {
      size_t size0 = 1;
      if (iv->getInsertedValueOperand()->getType()->isSized() &&
          (iv->getInsertedValueOperand()->getType()->isIntOrIntVectorTy() ||
           iv->getInsertedValueOperand()->getType()->isFPOrFPVectorTy()))
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 iv->getInsertedValueOperand()->getType()) +
             7) /
            8;
      auto it = TR.intType(size0, iv->getInsertedValueOperand(), false);
      if (it.isFloat() || !it.isKnown()) {
        floatingInsertion = true;
        break;
      }
      Value *val = iv->getAggregateOperand();
      if (gutils->isConstantValue(val))
        break;
      if (auto dc = dyn_cast<InsertValueInst>(val)) {
        iv = dc;
      } else {
        // unsure where this came from, conservatively assume contains float
        floatingInsertion = true;
        break;
      }
    }

    if (!floatingInsertion)
      return;

    // TODO handle pointers
    // TODO type analysis handle structs

    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode:
      assert(0 && "should be handled above");
      return;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {
      IRBuilder<> Builder2(&IVI);
      getReverseBuilder(Builder2);

      Value *orig_inserted = IVI.getInsertedValueOperand();
      Value *orig_agg = IVI.getAggregateOperand();

      size_t size0 = 1;
      if (orig_inserted->getType()->isSized())
        size0 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_inserted->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_inserted)) {
        auto it = TR.intType(size0, orig_inserted, /*errIfFalse*/ false);
        Type *flt = it.isFloat();
        if (!it.isKnown()) {
          bool found = false;

          if (looseTypeAnalysis) {
            if (orig_inserted->getType()->isFPOrFPVectorTy()) {
              flt = orig_inserted->getType()->getScalarType();
              found = true;
            } else if (orig_inserted->getType()->isIntOrIntVectorTy() ||
                       orig_inserted->getType()->isPointerTy()) {
              flt = nullptr;
              found = true;
            }
          }
          if (!found) {
            if (CustomErrorHandler) {
              std::string str;
              raw_string_ostream ss(str);
              ss << "Cannot deduce type of insertvalue " << IVI;
              CustomErrorHandler(str.c_str(), wrap(&IVI), ErrorType::NoType,
                                 &TR.analyzer, nullptr, wrap(&Builder2));
            } else {
              EmitFailure("CannotDeduceType", IVI.getDebugLoc(), &IVI,
                          "failed to deduce type of insertvalue ", IVI);
            }
          }
        }
        if (flt) {
          auto rule = [&](Value *prediff) {
            return Builder2.CreateExtractValue(prediff, IVI.getIndices());
          };
          auto prediff = diffe(&IVI, Builder2);
          auto dindex =
              applyChainRule(orig_inserted->getType(), Builder2, rule, prediff);
          addToDiffe(orig_inserted, dindex, Builder2, flt);
        }
      }

      size_t size1 = 1;
      if (orig_agg->getType()->isSized() &&
          (orig_agg->getType()->isIntOrIntVectorTy() ||
           orig_agg->getType()->isFPOrFPVectorTy()))
        size1 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_agg->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_agg)) {
        auto rule = [&](Value *prediff) {
          return Builder2.CreateInsertValue(
              prediff, Constant::getNullValue(orig_inserted->getType()),
              IVI.getIndices());
        };
        auto prediff = diffe(&IVI, Builder2);
        auto dindex =
            applyChainRule(orig_agg->getType(), Builder2, rule, prediff);
        addToDiffe(orig_agg, dindex, Builder2, TR.addingType(size1, orig_agg));
      }

      setDiffe(&IVI,
               Constant::getNullValue(gutils->getShadowType(IVI.getType())),
               Builder2);
      return;
    }
    case DerivativeMode::ReverseModePrimal: {
      return;
    }
    }
  }

  void getReverseBuilder(llvm::IRBuilder<> &Builder2, bool original = true) {
    ((GradientUtils *)gutils)->getReverseBuilder(Builder2, original);
  }

  void getForwardBuilder(llvm::IRBuilder<> &Builder2) {
    ((GradientUtils *)gutils)->getForwardBuilder(Builder2);
  }

  llvm::Value *diffe(llvm::Value *val, llvm::IRBuilder<> &Builder) {
    assert(Mode != DerivativeMode::ReverseModePrimal);
    return ((DiffeGradientUtils *)gutils)->diffe(val, Builder);
  }

  void setDiffe(llvm::Value *val, llvm::Value *dif,
                llvm::IRBuilder<> &Builder) {
    assert(Mode != DerivativeMode::ReverseModePrimal);
    ((DiffeGradientUtils *)gutils)->setDiffe(val, dif, Builder);
  }

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element. Return values of f are collected and wrapped.
  template <typename Func, typename... Args>
  llvm::Value *applyChainRule(llvm::Type *diffType, llvm::IRBuilder<> &Builder,
                              Func rule, Args... args) {
    return ((GradientUtils *)gutils)
        ->applyChainRule(diffType, Builder, rule, args...);
  }

  /// Unwraps a vector derivative from its internal representation and applies a
  /// function f to each element.
  template <typename Func, typename... Args>
  void applyChainRule(llvm::IRBuilder<> &Builder, Func rule, Args... args) {
    ((GradientUtils *)gutils)->applyChainRule(Builder, rule, args...);
  }

  /// Unwraps an collection of constant vector derivatives from their internal
  /// representations and applies a function f to each element.
  template <typename Func>
  void applyChainRule(llvm::ArrayRef<llvm::Value *> diffs,
                      llvm::IRBuilder<> &Builder, Func rule) {
    ((GradientUtils *)gutils)->applyChainRule(diffs, Builder, rule);
  }

  bool shouldFree() {
    assert(Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit);
    return ((DiffeGradientUtils *)gutils)->FreeMemory;
  }

  llvm::SmallVector<llvm::SelectInst *, 4>
  addToDiffe(llvm::Value *val, llvm::Value *dif, llvm::IRBuilder<> &Builder,
             llvm::Type *T, llvm::Value *mask = nullptr) {
    return ((DiffeGradientUtils *)gutils)
        ->addToDiffe(val, dif, Builder, T, /*idxs*/ {}, mask);
  }

  llvm::Value *lookup(llvm::Value *val, llvm::IRBuilder<> &Builder) {
    return gutils->lookupM(val, Builder);
  }

  void visitBinaryOperator(llvm::BinaryOperator &BO) {
    eraseIfUnused(BO);

    size_t size = 1;
    if (BO.getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  BO.getType()) +
              7) /
             8;

    if (BO.getType()->isIntOrIntVectorTy() &&
        TR.intType(size, &BO, /*errifnotfound*/ false) == BaseType::Pointer) {
      return;
    }

    if (BO.getOpcode() == llvm::Instruction::FDiv &&
        (Mode == DerivativeMode::ReverseModeGradient ||
         Mode == DerivativeMode::ReverseModeCombined) &&
        !gutils->isConstantValue(&BO)) {
      using namespace llvm;
      // Required loopy phi = [in, BO, BO, ..., BO]
      //  1) phi is only used in this B0
      //  2) BO dominates all latches
      //  3) phi == B0 whenever not coming from preheader [implies 2]
      //  4) [optional but done for ease] one exit to make it easier to
      //  calculation the product at that point
      Value *orig_op0 = BO.getOperand(0);
      if (auto P0 = dyn_cast<PHINode>(orig_op0)) {
        LoopContext lc;
        SmallVector<Instruction *, 4> activeUses;
        for (auto u : P0->users()) {
          if (!gutils->isConstantInstruction(cast<Instruction>(u))) {
            activeUses.push_back(cast<Instruction>(u));
          } else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
            activeUses.push_back(cast<Instruction>(u));
        }
        if (activeUses.size() == 1 && activeUses[0] == &BO &&
            gutils->getContext(gutils->getNewFromOriginal(P0->getParent()),
                               lc) &&
            gutils->getNewFromOriginal(P0->getParent()) == lc.header) {
          SmallVector<BasicBlock *, 1> Latches;
          gutils->OrigLI.getLoopFor(P0->getParent())->getLoopLatches(Latches);
          bool allIncoming = true;
          for (auto Latch : Latches) {
            if (&BO != P0->getIncomingValueForBlock(Latch)) {
              allIncoming = false;
              break;
            }
          }
          if (allIncoming && lc.exitBlocks.size() == 1) {

            IRBuilder<> Builder2(&BO);
            getReverseBuilder(Builder2);

            Value *orig_op1 = BO.getOperand(1);
            bool constantval0 = gutils->isConstantValue(orig_op0);
            bool constantval1 = gutils->isConstantValue(orig_op1);

            Value *dif0 = nullptr;
            Value *dif1 = nullptr;
            Value *idiff = diffe(&BO, Builder2);

            Type *addingType = BO.getType();

            if (!constantval1) {
              IRBuilder<> EB(*lc.exitBlocks.begin());
              getReverseBuilder(EB, /*original=*/false);
              Value *Pstart = P0->getIncomingValueForBlock(
                  gutils->getOriginalFromNew(lc.preheader));
              if (gutils->isConstantValue(Pstart)) {
                Value *lop0 = lookup(gutils->getNewFromOriginal(&BO), EB);
                Value *lop1 =
                    lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
                auto rule = [&](Value *idiff) {
                  auto res = Builder2.CreateFDiv(
                      Builder2.CreateFNeg(Builder2.CreateFMul(idiff, lop0)),
                      lop1);
                  if (EnzymeStrongZero) {
                    res = CreateSelect(
                        Builder2,
                        Builder2.CreateFCmpOEQ(
                            idiff, Constant::getNullValue(idiff->getType())),
                        idiff, res);
                  }
                  return res;
                };
                dif1 =
                    applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
              } else {
                auto product = gutils->getOrInsertTotalMultiplicativeProduct(
                    gutils->getNewFromOriginal(orig_op1), lc);
                IRBuilder<> EB(*lc.exitBlocks.begin());
                getReverseBuilder(EB, /*original=*/false);
                Value *s = lookup(gutils->getNewFromOriginal(Pstart), Builder2);
                Value *lop0 = lookup(product, EB);
                Value *lop1 =
                    lookup(gutils->getNewFromOriginal(orig_op1), Builder2);
                auto rule = [&](Value *idiff) {
                  auto res = Builder2.CreateFDiv(
                      Builder2.CreateFNeg(Builder2.CreateFMul(
                          s, Builder2.CreateFDiv(idiff, lop0))),
                      lop1);
                  if (EnzymeStrongZero) {
                    res = CreateSelect(
                        Builder2,
                        Builder2.CreateFCmpOEQ(
                            idiff, Constant::getNullValue(idiff->getType())),
                        idiff, res);
                  }
                  return res;
                };
                dif1 =
                    applyChainRule(orig_op1->getType(), Builder2, rule, idiff);
              }
              addToDiffe(orig_op1, dif1, Builder2, addingType);
            }
            return;
          }
        }
      }
    }

    {
      using namespace llvm;
      switch (BO.getOpcode()) {
#include "BinopDerivatives.inc"
      default:
        break;
      }
    }

    switch (Mode) {
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined:
      if (gutils->isConstantInstruction(&BO))
        return;
      createBinaryOperatorAdjoint(BO);
      break;
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      createBinaryOperatorDual(BO);
      break;
    case DerivativeMode::ReverseModePrimal:
      return;
    }
  }

  void createBinaryOperatorAdjoint(llvm::BinaryOperator &BO) {
    using namespace llvm;

    IRBuilder<> Builder2(&BO);
    getReverseBuilder(Builder2);

    Value *orig_op0 = BO.getOperand(0);
    Value *orig_op1 = BO.getOperand(1);
    bool constantval0 = gutils->isConstantValue(orig_op0);
    bool constantval1 = gutils->isConstantValue(orig_op1);

    Value *dif0 = nullptr;
    Value *dif1 = nullptr;
    Value *idiff = diffe(&BO, Builder2);

    Type *addingType = BO.getType();

    switch (BO.getOpcode()) {
    case Instruction::LShr: {
      if (!constantval0) {
        if (auto ci = dyn_cast<ConstantInt>(orig_op1)) {
          size_t size = 1;
          if (orig_op0->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(orig_op0->getType()) +
                    7) /
                   8;

          if (Type *flt = TR.addingType(size, orig_op0)) {
            auto bits = gutils->newFunc->getParent()
                            ->getDataLayout()
                            .getTypeAllocSizeInBits(flt);
            if (ci->getSExtValue() >= (int64_t)bits &&
                ci->getSExtValue() % bits == 0) {
              auto rule = [&](Value *idiff) {
                return Builder2.CreateShl(idiff, ci);
              };
              dif0 = applyChainRule(orig_op0->getType(), Builder2, rule, idiff);
              addingType = flt;
              goto done;
            }
          }
        }
      }
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer and is constant
        return;
      }
      goto def;
    }
    case Instruction::And: {
      // If & against 0b10000000000 and a float the result is 0
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);
              // Derivative is zero (equivalent to rounding as just chopping off
              // bits of mantissa), no update
              return;
            }
          }
        }
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer and is constant
        return;
      }
      goto def;
    }
    case Instruction::Xor: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If ^ against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          if (containsOnlyAtMostTopBit(BO.getOperand(i), eFT, dl, &FT)) {
            setDiffe(
                &BO,
                Constant::getNullValue(gutils->getShadowType(BO.getType())),
                Builder2);
            auto isZero = Builder2.CreateICmpEQ(
                lookup(gutils->getNewFromOriginal(BO.getOperand(i)), Builder2),
                Constant::getNullValue(BO.getType()));
            auto rule = [&](Value *idiff) {
              auto ext = Builder2.CreateBitCast(idiff, FT);
              auto neg = Builder2.CreateFNeg(ext);
              neg = CreateSelect(Builder2, isZero, ext, neg);
              neg = Builder2.CreateBitCast(neg, BO.getType());
              return neg;
            };
            auto bc = applyChainRule(BO.getOperand(1 - i)->getType(), Builder2,
                                     rule, idiff);
            addToDiffe(BO.getOperand(1 - i), bc, Builder2, FT);
            return;
          }
        }
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer and is constant
        return;
      }
      goto def;
    }
    case Instruction::Or: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            auto AP = CI->getValue();
            bool validXor = false;
#if LLVM_VERSION_MAJOR > 16
            if (AP.isZero())
#else
            if (AP.isNullValue())
#endif
            {
              validXor = true;
            } else if (
                !AP.isNegative() &&
                ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                  && (AP & ~0b01111111100000000000000000000000ULL).isZero()
#else
                  && (AP & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                      ) ||
                 (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                  &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isZero()
#else
                  &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()
#endif
                      ))) {
              validXor = true;
            }
            if (validXor) {
              setDiffe(
                  &BO,
                  Constant::getNullValue(gutils->getShadowType(BO.getType())),
                  Builder2);

              auto arg = lookup(
                  gutils->getNewFromOriginal(BO.getOperand(1 - i)), Builder2);

              auto rule = [&](Value *idiff) {
                auto prev = Builder2.CreateOr(arg, BO.getOperand(i));
                prev = Builder2.CreateSub(prev, arg, "", /*NUW*/ true,
                                          /*NSW*/ false);
                uint64_t num = 0;
                if (FT->isFloatTy()) {
                  num = 127ULL << 23;
                } else {
                  assert(FT->isDoubleTy());
                  num = 1023ULL << 52;
                }
                prev = Builder2.CreateAdd(
                    prev, ConstantInt::get(prev->getType(), num, false), "",
                    /*NUW*/ true, /*NSW*/ true);
                prev = Builder2.CreateBitCast(
                    checkedMul(Builder2, Builder2.CreateBitCast(idiff, FT),
                               Builder2.CreateBitCast(prev, FT)),
                    prev->getType());
                return prev;
              };

              Value *prev = applyChainRule(BO.getOperand(1 - i)->getType(),
                                           Builder2, rule, idiff);
              addToDiffe(BO.getOperand(1 - i), prev, Builder2, FT);
              return;
            }
          }
        }
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer or is constant
        return;
      }
      goto def;
    }
    case Instruction::SDiv:
    case Instruction::Shl:
    case Instruction::Mul:
    case Instruction::Sub:
    case Instruction::Add: {
      if (looseTypeAnalysis) {
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer add is constant
        return;
      }
      goto def;
    }
    default:
    def:;
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        ss << " constantarg[" << arg << "] = " << gutils->isConstantValue(&arg)
           << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          ss << v << ",";
        ss << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          ss << " constantinst[" << I
             << "] = " << gutils->isConstantInstruction(&I)
             << " val:" << gutils->isConstantValue(&I)
             << " type: " << TR.query(&I).str() << "\n";
        }
      ss << "cannot handle unknown binary operator: " << BO << "\n";
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(&BO), ErrorType::NoDerivative,
                           gutils, nullptr, wrap(&Builder2));
      } else {
        llvm::errs() << ss.str() << "\n";
        report_fatal_error("unknown binary operator");
      }
    }

  done:;
    if (dif0 || dif1)
      setDiffe(&BO, Constant::getNullValue(gutils->getShadowType(BO.getType())),
               Builder2);
    if (dif0)
      addToDiffe(orig_op0, dif0, Builder2, addingType);
    if (dif1)
      addToDiffe(orig_op1, dif1, Builder2, addingType);
  }

  void createBinaryOperatorDual(llvm::BinaryOperator &BO) {
    using namespace llvm;

    if (gutils->isConstantInstruction(&BO)) {
      forwardModeInvertedPointerFallback(BO);
      return;
    }

    IRBuilder<> Builder2(&BO);
    getForwardBuilder(Builder2);

    Value *orig_op0 = BO.getOperand(0);
    Value *orig_op1 = BO.getOperand(1);

    bool constantval0 = gutils->isConstantValue(orig_op0);
    bool constantval1 = gutils->isConstantValue(orig_op1);

    Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                     constantval1 ? nullptr : diffe(orig_op1, Builder2)};

    switch (BO.getOpcode()) {
    case Instruction::And: {
      // If & against 0b10000000000 and a float the result is 0
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;
      Type *diffTy = gutils->getShadowType(BO.getType());

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            if (eFT->isDoubleTy() && CI->getValue() == -134217728) {
              setDiffe(&BO, Constant::getNullValue(diffTy), Builder2);
              // Derivative is zero (equivalent to rounding as just chopping off
              // bits of mantissa), no update
              return;
            }
          }
        }
      if (looseTypeAnalysis) {
        forwardModeInvertedPointerFallback(BO);
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer and is constant
        return;
      }
      goto def;
    }
    case Instruction::Xor: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;

      Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                       constantval1 ? nullptr : diffe(orig_op1, Builder2)};

      for (int i = 0; i < 2; ++i) {
        if (containsOnlyAtMostTopBit(BO.getOperand(i), eFT, dl, &FT) &&
            dif[1 - i] && !dif[i]) {
          auto isZero = Builder2.CreateICmpEQ(
              gutils->getNewFromOriginal(BO.getOperand(i)),
              Constant::getNullValue(BO.getType()));
          auto rule = [&](Value *idiff) {
            auto ext = Builder2.CreateBitCast(idiff, FT);
            auto neg = Builder2.CreateFNeg(ext);
            neg = CreateSelect(Builder2, isZero, ext, neg);
            neg = Builder2.CreateBitCast(neg, BO.getType());
            return neg;
          };
          auto bc = applyChainRule(BO.getOperand(1 - i)->getType(), Builder2,
                                   rule, dif[1 - i]);
          setDiffe(&BO, bc, Builder2);
          return;
        }
      }
      if (looseTypeAnalysis) {
        forwardModeInvertedPointerFallback(BO);
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer and is constant
        return;
      }
      goto def;
    }
    case Instruction::Or: {
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;

      Value *dif[2] = {constantval0 ? nullptr : diffe(orig_op0, Builder2),
                       constantval1 ? nullptr : diffe(orig_op1, Builder2)};

      auto FT = TR.query(&BO).IsAllFloat(size);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
#if LLVM_VERSION_MAJOR >= 12
            FT = VectorType::get(FT, CV->getType()->getElementCount());
#else
            FT = VectorType::get(FT, CV->getType()->getNumElements());
#endif
          }
          if (CI && dl.getTypeSizeInBits(eFT) ==
                        dl.getTypeSizeInBits(CI->getType())) {
            auto AP = CI->getValue();
            bool validXor = false;
#if LLVM_VERSION_MAJOR > 16
            if (AP.isZero())
#else
            if (AP.isNullValue())
#endif
            {
              validXor = true;
            } else if (
                !AP.isNegative() &&
                ((FT->isFloatTy()
#if LLVM_VERSION_MAJOR > 16
                  && (AP & ~0b01111111100000000000000000000000ULL).isZero()
#else
                  && (AP & ~0b01111111100000000000000000000000ULL).isNullValue()
#endif
                      ) ||
                 (FT->isDoubleTy()
#if LLVM_VERSION_MAJOR > 16
                  &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isZero()
#else
                  &&
                  (AP &
                   ~0b0111111111110000000000000000000000000000000000000000000000000000ULL)
                      .isNullValue()
#endif
                      ))) {
              validXor = true;
            }
            if (validXor) {
              auto rule = [&](Value *difi) {
                auto arg = gutils->getNewFromOriginal(BO.getOperand(1 - i));
                auto prev = Builder2.CreateOr(arg, BO.getOperand(i));
                prev = Builder2.CreateSub(prev, arg, "", /*NUW*/ true,
                                          /*NSW*/ false);
                uint64_t num = 0;
                if (FT->isFloatTy()) {
                  num = 127ULL << 23;
                } else {
                  assert(FT->isDoubleTy());
                  num = 1023ULL << 52;
                }
                prev = Builder2.CreateAdd(
                    prev, ConstantInt::get(prev->getType(), num, false), "",
                    /*NUW*/ true, /*NSW*/ true);
                prev = Builder2.CreateBitCast(
                    checkedMul(Builder2, Builder2.CreateBitCast(difi, FT),
                               Builder2.CreateBitCast(prev, FT)),
                    prev->getType());

                return prev;
              };

              auto diffe =
                  applyChainRule(BO.getType(), Builder2, rule, dif[1 - i]);
              setDiffe(&BO, diffe, Builder2);
              return;
            }
          }
        }
      if (looseTypeAnalysis) {
        forwardModeInvertedPointerFallback(BO);
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer or is constant
        return;
      }
      goto def;
    }
    case Instruction::Shl:
    case Instruction::Mul:
    case Instruction::Sub:
    case Instruction::Add: {
      if (looseTypeAnalysis) {
        forwardModeInvertedPointerFallback(BO);
        llvm::errs() << "warning: binary operator is integer and constant: "
                     << BO << "\n";
        // if loose type analysis, assume this integer add is constant
        return;
      }
      goto def;
    }
    default:
    def:;
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << *gutils->oldFunc << "\n";
      for (auto &arg : gutils->oldFunc->args()) {
        ss << " constantarg[" << arg << "] = " << gutils->isConstantValue(&arg)
           << " type: " << TR.query(&arg).str() << " - vals: {";
        for (auto v : TR.knownIntegralValues(&arg))
          ss << v << ",";
        ss << "}\n";
      }
      for (auto &BB : *gutils->oldFunc)
        for (auto &I : BB) {
          ss << " constantinst[" << I
             << "] = " << gutils->isConstantInstruction(&I)
             << " val:" << gutils->isConstantValue(&I)
             << " type: " << TR.query(&I).str() << "\n";
        }
      ss << "cannot handle unknown binary operator: " << BO << "\n";
      if (CustomErrorHandler) {
        auto rval = unwrap(CustomErrorHandler(ss.str().c_str(), wrap(&BO),
                                              ErrorType::NoDerivative, gutils,
                                              nullptr, wrap(&Builder2)));
        if (!rval)
          rval = Constant::getNullValue(gutils->getShadowType(BO.getType()));
        if (!gutils->isConstantValue(&BO))
          setDiffe(&BO, rval, Builder2);
      } else {
        EmitFailure("NoDerivative", BO.getDebugLoc(), &BO, ss.str());
        return;
      }
      break;
    }
  }

  void visitMemSetInst(llvm::MemSetInst &MS) { visitMemSetCommon(MS); }

  void visitMemSetCommon(llvm::CallInst &MS) {
    using namespace llvm;

    IRBuilder<> BuilderZ(&MS);
    getForwardBuilder(BuilderZ);

    IRBuilder<> Builder2(&MS);
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined)
      getReverseBuilder(Builder2);

    bool forceErase = false;
    if (Mode == DerivativeMode::ReverseModeGradient) {
      for (const auto &pair : gutils->rematerializableAllocations) {
        if (pair.second.stores.count(&MS) && pair.second.LI) {
          forceErase = true;
        }
      }
    }
    if (forceErase)
      eraseIfUnused(MS, /*erase*/ true, /*check*/ false);
    else
      eraseIfUnused(MS);

    Value *orig_op0 = MS.getArgOperand(0);
    Value *orig_op1 = MS.getArgOperand(1);

    // If constant destination then no operation needs doing
    if (gutils->isConstantValue(orig_op0)) {
      return;
    }

    if (!gutils->isConstantValue(orig_op1)) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "couldn't handle non constant inst in memset to "
            "propagate differential to\n"
         << MS;
      if (CustomErrorHandler) {
        CustomErrorHandler(ss.str().c_str(), wrap(&MS), ErrorType::NoDerivative,
                           gutils, nullptr, wrap(&BuilderZ));
      } else {
        llvm::errs() << ss.str() << "\n";
        report_fatal_error("non constant in memset");
      }
    }

    if (Mode == DerivativeMode::ForwardMode) {
      Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
      Value *op1 = gutils->getNewFromOriginal(MS.getArgOperand(1));
      Value *op2 = gutils->getNewFromOriginal(MS.getArgOperand(2));
      Value *op3 = nullptr;
#if LLVM_VERSION_MAJOR >= 14
      if (3 < MS.arg_size())
#else
      if (3 < MS.getNumArgOperands())
#endif
      {
        op3 = gutils->getNewFromOriginal(MS.getOperand(3));
      }

      auto Defs =
          gutils->getInvertedBundles(&MS,
                                     {ValueType::Shadow, ValueType::Primal,
                                      ValueType::Primal, ValueType::Primal},
                                     BuilderZ, /*lookup*/ false);

      applyChainRule(
          BuilderZ,
          [&](Value *op0) {
            SmallVector<Value *, 4> args = {op0, op1, op2};
            if (op3)
              args.push_back(op3);
            auto cal = BuilderZ.CreateCall(MS.getCalledFunction(), args, Defs);
            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            cal->copyMetadata(MS, ToCopy2);
            if (auto m = hasMetadata(&MS, "enzyme_zerostack"))
              cal->setMetadata("enzyme_zerostack", m);
            cal->setAttributes(MS.getAttributes());
            cal->setCallingConv(MS.getCallingConv());
            cal->setTailCallKind(MS.getTailCallKind());
            cal->setDebugLoc(gutils->getNewFromOriginal(MS.getDebugLoc()));
          },
          op0);
      return;
    }

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&MS)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(MS.getOperand(2))) {
      size = ci->getLimitedValue();
    }

    // TODO note that we only handle memset of ONE type (aka memset of {int,
    // double} not allowed)

    if (size == 0) {
      llvm::errs() << MS << "\n";
    }
    assert(size != 0);

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto vd = TR.query(MS.getOperand(0)).Data0().ShiftIndices(DL, 0, size, 0);

    if (!vd.isKnownPastPointer()) {
      // If unknown type results, consider the intersection of all incoming.
      if (isa<PHINode>(MS.getOperand(0)) || isa<SelectInst>(MS.getOperand(0))) {
        SmallVector<Value *, 2> todo = {MS.getOperand(0)};
        bool set = false;
        SmallSet<Value *, 2> seen;
        TypeTree vd2;
        while (todo.size()) {
          Value *cur = todo.back();
          todo.pop_back();
          if (seen.count(cur))
            continue;
          seen.insert(cur);
          if (auto PN = dyn_cast<PHINode>(cur)) {
            for (size_t i = 0, end = PN->getNumIncomingValues(); i < end; i++) {
              todo.push_back(PN->getIncomingValue(i));
            }
            continue;
          }
          if (auto S = dyn_cast<SelectInst>(cur)) {
            todo.push_back(S->getTrueValue());
            todo.push_back(S->getFalseValue());
            continue;
          }
          if (auto CE = dyn_cast<ConstantExpr>(cur)) {
            if (CE->isCast()) {
              todo.push_back(CE->getOperand(0));
              continue;
            }
          }
          if (auto CI = dyn_cast<CastInst>(cur)) {
            todo.push_back(CI->getOperand(0));
            continue;
          }
          if (isa<ConstantPointerNull>(cur))
            continue;
          if (auto CI = dyn_cast<ConstantInt>(cur))
            if (CI->isZero())
              continue;
          auto curTT = TR.query(cur).Data0().ShiftIndices(DL, 0, size, 0);
          if (!set)
            vd2 = curTT;
          else
            vd2 &= curTT;
          set = true;
        }
        vd = vd2;
      }
    }
    if (!vd.isKnownPastPointer()) {
      if (looseTypeAnalysis) {
        if (auto CI = dyn_cast<CastInst>(MS.getOperand(0))) {
#if LLVM_VERSION_MAJOR < 18
          if (auto PT = dyn_cast<PointerType>(CI->getSrcTy())) {
            auto ET = PT->getPointerElementType();
            while (1) {
              if (auto ST = dyn_cast<StructType>(ET)) {
                if (ST->getNumElements()) {
                  ET = ST->getElementType(0);
                  continue;
                }
              }
              if (auto AT = dyn_cast<ArrayType>(ET)) {
                ET = AT->getElementType();
                continue;
              }
              break;
            }
            if (ET->isFPOrFPVectorTy()) {
              vd = TypeTree(ConcreteType(ET->getScalarType())).Only(0, &MS);
              goto known;
            }
            if (ET->isPointerTy()) {
              vd = TypeTree(BaseType::Pointer).Only(0, &MS);
              goto known;
            }
            if (ET->isIntOrIntVectorTy()) {
              vd = TypeTree(BaseType::Integer).Only(0, &MS);
              goto known;
            }
          }
#endif
        }
        if (auto gep = dyn_cast<GetElementPtrInst>(MS.getOperand(0))) {
          if (auto AT = dyn_cast<ArrayType>(gep->getSourceElementType())) {
            if (AT->getElementType()->isIntegerTy()) {
              vd = TypeTree(BaseType::Integer).Only(0, &MS);
              goto known;
            }
          }
        }
        EmitWarning("CannotDeduceType", MS, "failed to deduce type of memset ",
                    MS);
        vd = TypeTree(BaseType::Pointer).Only(0, &MS);
        goto known;
      }
      if (CustomErrorHandler) {
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of memset " << MS;
        CustomErrorHandler(str.c_str(), wrap(&MS), ErrorType::NoType,
                           &TR.analyzer, nullptr, wrap(&BuilderZ));
      } else {
        EmitFailure("CannotDeduceType", MS.getDebugLoc(), &MS,
                    "failed to deduce type of memset ", MS);

        TR.firstPointer(size, MS.getOperand(0), &MS, /*errifnotfound*/ true,
                        /*pointerIntSame*/ true);
        llvm_unreachable("bad msi");
      }
    }
  known:;

#if 0
#if LLVM_VERSION_MAJOR >= 10
    unsigned dstalign = dstAlign.valueOrOne().value();
    unsigned srcalign = srcAlign.valueOrOne().value();
#else
    unsigned dstalign = dstAlign;
    unsigned srcalign = srcAlign;
#endif
#endif

    unsigned start = 0;

    Value *op1 = gutils->getNewFromOriginal(MS.getArgOperand(1));
    Value *new_size = gutils->getNewFromOriginal(MS.getArgOperand(2));
    Value *op3 = nullptr;
#if LLVM_VERSION_MAJOR >= 14
    if (3 < MS.arg_size())
#else
    if (3 < MS.getNumArgOperands())
#endif
    {
      op3 = gutils->getNewFromOriginal(MS.getOperand(3));
    }

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

      Value *length = new_size;
      if (nextStart != size) {
        length = ConstantInt::get(new_size->getType(), nextStart);
      }
      if (start != 0)
        length = BuilderZ.CreateSub(
            length, ConstantInt::get(new_size->getType(), start));

#if 0
      unsigned subdstalign = dstalign;
      // todo make better alignment calculation
      if (dstalign != 0) {
        if (start % dstalign != 0) {
          dstalign = 1;
        }
      }
      unsigned subsrcalign = srcalign;
      // todo make better alignment calculation
      if (srcalign != 0) {
        if (start % srcalign != 0) {
          srcalign = 1;
        }
      }
#endif

      Value *shadow_dst = gutils->invertPointerM(MS.getOperand(0), BuilderZ);

      // TODO ponder forward split mode
      Type *secretty = dt.isFloat();
      if (!secretty &&
          ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
           (Mode == DerivativeMode::ReverseModeCombined && forwardsShadow) ||
           (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
           (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow))) {
        auto Defs =
            gutils->getInvertedBundles(&MS,
                                       {ValueType::Shadow, ValueType::Primal,
                                        ValueType::Primal, ValueType::Primal},
                                       BuilderZ, /*lookup*/ false);
        auto rule = [&](Value *op0) {
          if (start != 0) {
            Value *idxs[] = {
                ConstantInt::get(Type::getInt32Ty(op0->getContext()), start)};
            op0 =
                BuilderZ.CreateInBoundsGEP(BuilderZ.getInt8PtrTy(), op0, idxs);
          }
          SmallVector<Value *, 4> args = {op0, op1, length};
          if (op3)
            args.push_back(op3);
          auto cal = BuilderZ.CreateCall(MS.getCalledFunction(), args, Defs);
          llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
          ToCopy2.push_back(LLVMContext::MD_noalias);
          if (auto m = hasMetadata(&MS, "enzyme_zerostack"))
            cal->setMetadata("enzyme_zerostack", m);
          cal->copyMetadata(MS, ToCopy2);
          cal->setAttributes(MS.getAttributes());
          cal->setCallingConv(MS.getCallingConv());
          cal->setTailCallKind(MS.getTailCallKind());
          cal->setDebugLoc(gutils->getNewFromOriginal(MS.getDebugLoc()));
        };

        applyChainRule(BuilderZ, rule, shadow_dst);
      }
      if (secretty && (Mode == DerivativeMode::ReverseModeGradient ||
                       Mode == DerivativeMode::ReverseModeCombined)) {

        auto Defs =
            gutils->getInvertedBundles(&MS,
                                       {ValueType::Shadow, ValueType::Primal,
                                        ValueType::Primal, ValueType::Primal},
                                       BuilderZ, /*lookup*/ true);
        Value *op1l = gutils->lookupM(op1, Builder2);
        Value *op3l = op3;
        if (op3l)
          op3l = gutils->lookupM(op3l, BuilderZ);
        length = gutils->lookupM(length, Builder2);
        auto rule = [&](Value *op0) {
          if (start != 0) {
            Value *idxs[] = {
                ConstantInt::get(Type::getInt32Ty(op0->getContext()), start)};
            op0 =
                Builder2.CreateInBoundsGEP(Builder2.getInt8PtrTy(), op0, idxs);
          }
          SmallVector<Value *, 4> args = {op0, op1l, length};
          if (op3l)
            args.push_back(op3l);
          auto cal = Builder2.CreateCall(MS.getCalledFunction(), args, Defs);
          llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
          ToCopy2.push_back(LLVMContext::MD_noalias);
          cal->copyMetadata(MS, ToCopy2);
          if (auto m = hasMetadata(&MS, "enzyme_zerostack"))
            cal->setMetadata("enzyme_zerostack", m);
          cal->setAttributes(MS.getAttributes());
          cal->setCallingConv(MS.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(MS.getDebugLoc()));
        };

        applyChainRule(Builder2, rule, gutils->lookupM(shadow_dst, Builder2));
      }

      if (nextStart == size)
        break;
      start = nextStart;
    }
  }

  void visitMemTransferInst(llvm::MemTransferInst &MTI) {
    using namespace llvm;
    Value *isVolatile = gutils->getNewFromOriginal(MTI.getOperand(3));
#if LLVM_VERSION_MAJOR >= 10
    auto srcAlign = MTI.getSourceAlign();
    auto dstAlign = MTI.getDestAlign();
#else
    auto srcAlign = MTI.getSourceAlignment();
    auto dstAlign = MTI.getDestAlignment();
#endif
    visitMemTransferCommon(MTI.getIntrinsicID(), srcAlign, dstAlign, MTI,
                           MTI.getOperand(0), MTI.getOperand(1),
                           gutils->getNewFromOriginal(MTI.getOperand(2)),
                           isVolatile);
  }

#if LLVM_VERSION_MAJOR >= 10
  void visitMemTransferCommon(llvm::Intrinsic::ID ID, llvm::MaybeAlign srcAlign,
                              llvm::MaybeAlign dstAlign, llvm::CallInst &MTI,
                              llvm::Value *orig_dst, llvm::Value *orig_src,
                              llvm::Value *new_size, llvm::Value *isVolatile)
#else
  void visitMemTransferCommon(llvm::Intrinsic::ID ID, unsigned srcAlign,
                              unsigned dstAlign, llvm::CallInst &MTI,
                              llvm::Value *orig_dst, llvm::Value *orig_src,
                              llvm::Value *new_size, llvm::Value *isVolatile)
#endif
  {
    using namespace llvm;

    if (gutils->isConstantValue(MTI.getOperand(0))) {
      eraseIfUnused(MTI);
      return;
    }

    if (unnecessaryStores.count(&MTI)) {
      eraseIfUnused(MTI);
      return;
    }

    // copying into nullptr is invalid (not sure why it exists here), but we
    // shouldn't do it in reverse pass or shadow
    if (isa<ConstantPointerNull>(orig_dst) ||
        TR.query(orig_dst).Inner0() == BaseType::Anything) {
      eraseIfUnused(MTI);
      return;
    }

    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(new_size)) {
      size = ci->getLimitedValue();
    }

    // TODO note that we only handle memcpy/etc of ONE type (aka memcpy of {int,
    // double} not allowed)
    if (size == 0) {
      eraseIfUnused(MTI);
      return;
    }

    if (Mode == DerivativeMode::ForwardMode &&
        gutils->isConstantValue(orig_dst)) {
      eraseIfUnused(MTI);
      return;
    }

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto vd = TR.query(orig_dst).Data0().ShiftIndices(DL, 0, size, 0);
    vd |= TR.query(orig_src).Data0().ShiftIndices(DL, 0, size, 0);

    bool errorIfNoType = true;
    if (Mode == DerivativeMode::ForwardMode &&
        (!gutils->isConstantValue(orig_src) && !EnzymeRuntimeActivityCheck)) {
      errorIfNoType = false;
    }

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MTI));

    if (!vd.isKnownPastPointer()) {
      if (looseTypeAnalysis) {
        for (auto val : {orig_dst, orig_src}) {
          if (auto CI = dyn_cast<CastInst>(val)) {
#if LLVM_VERSION_MAJOR < 18
            if (auto PT = dyn_cast<PointerType>(CI->getSrcTy())) {
              auto ET = PT->getPointerElementType();
              while (1) {
                if (auto ST = dyn_cast<StructType>(ET)) {
                  if (ST->getNumElements()) {
                    ET = ST->getElementType(0);
                    continue;
                  }
                }
                if (auto AT = dyn_cast<ArrayType>(ET)) {
                  ET = AT->getElementType();
                  continue;
                }
                break;
              }
              if (ET->isFPOrFPVectorTy()) {
                vd = TypeTree(ConcreteType(ET->getScalarType())).Only(0, &MTI);
                goto known;
              }
              if (ET->isPointerTy()) {
                vd = TypeTree(BaseType::Pointer).Only(0, &MTI);
                goto known;
              }
              if (ET->isIntOrIntVectorTy()) {
                vd = TypeTree(BaseType::Integer).Only(0, &MTI);
                goto known;
              }
            }
#endif
          }
          if (auto gep = dyn_cast<GetElementPtrInst>(val)) {
            if (auto AT = dyn_cast<ArrayType>(gep->getSourceElementType())) {
              if (AT->getElementType()->isIntegerTy()) {
                vd = TypeTree(BaseType::Integer).Only(0, &MTI);
                goto known;
              }
            }
          }
        }
        if (errorIfNoType)
          EmitWarning("CannotDeduceType", MTI, "failed to deduce type of copy ",
                      MTI);
        vd = TypeTree(BaseType::Pointer).Only(0, &MTI);
        goto known;
      }
      if (errorIfNoType) {
        if (CustomErrorHandler) {
          std::string str;
          raw_string_ostream ss(str);
          ss << "Cannot deduce type of copy " << MTI;
          CustomErrorHandler(str.c_str(), wrap(&MTI), ErrorType::NoType,
                             &TR.analyzer, nullptr, wrap(&BuilderZ));
        } else {
          EmitFailure("CannotDeduceType", MTI.getDebugLoc(), &MTI,
                      "failed to deduce type of copy ", MTI);

          TR.firstPointer(size, orig_dst, &MTI, /*errifnotfound*/ true,
                          /*pointerIntSame*/ true);
          llvm_unreachable("bad mti");
        }
      } else {
        vd = TypeTree(BaseType::Pointer).Only(0, &MTI);
      }
    }
  known:;

  // llvm::errs() << "MIT: " << MTI << "|size: " << size << " vd: " <<
  // vd.str() << "\n";

#if LLVM_VERSION_MAJOR >= 10
    unsigned dstalign = dstAlign.valueOrOne().value();
    unsigned srcalign = srcAlign.valueOrOne().value();
#else
    unsigned dstalign = dstAlign;
    unsigned srcalign = srcAlign;
#endif

    unsigned start = 0;

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&MTI)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    while (1) {
      unsigned nextStart = size;

      auto dt = vd[{-1}];
      for (size_t i = start; i < size; ++i) {
        bool Legal = true;
        auto tmp = dt;
        tmp.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
        if (!Legal) {
          if (Mode == DerivativeMode::ForwardMode) {
            // if both are floats (of any type), forward mode is the same.
            //   + [potentially zero if const, otherwise copy]
            // if both are int/pointer (of any type), also the same
            //   + copy
            // if known non-constant, also the same
            //   + copy
            if ((dt.isFloat() == nullptr) ==
                (vd[{(int)i}].isFloat() == nullptr)) {
              Legal = true;
            }
            if (!gutils->isConstantValue(orig_src) &&
                !EnzymeRuntimeActivityCheck) {
              Legal = true;
            }
          }
          if (!Legal) {
            nextStart = i;
            break;
          }
        } else
          dt = tmp;
      }
      if (!dt.isKnown()) {
        TR.dump();
        llvm::errs() << " vd:" << vd.str() << " start:" << start
                     << " size: " << size << " dt:" << dt.str() << "\n";
      }
      assert(dt.isKnown());

      Value *length = new_size;
      if (nextStart != size) {
        length = ConstantInt::get(new_size->getType(), nextStart);
      }
      if (start != 0)
        length = BuilderZ.CreateSub(
            length, ConstantInt::get(new_size->getType(), start));

      unsigned subdstalign = dstalign;
      // todo make better alignment calculation
      if (dstalign != 0) {
        if (start % dstalign != 0) {
          dstalign = 1;
        }
      }
      unsigned subsrcalign = srcalign;
      // todo make better alignment calculation
      if (srcalign != 0) {
        if (start % srcalign != 0) {
          srcalign = 1;
        }
      }
      IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MTI));
      Value *shadow_dst = gutils->isConstantValue(orig_dst)
                              ? nullptr
                              : gutils->invertPointerM(orig_dst, BuilderZ);
      Value *shadow_src = gutils->isConstantValue(orig_src)
                              ? nullptr
                              : gutils->invertPointerM(orig_src, BuilderZ);

      auto rev_rule = [&](Value *shadow_dst, Value *shadow_src) {
        if (shadow_dst == nullptr)
          shadow_dst = gutils->getNewFromOriginal(orig_dst);
        if (shadow_src == nullptr)
          shadow_src = gutils->getNewFromOriginal(orig_src);
        SubTransferHelper(
            gutils, Mode, dt.isFloat(), ID, subdstalign, subsrcalign,
            /*offset*/ start, gutils->isConstantValue(orig_dst), shadow_dst,
            gutils->isConstantValue(orig_src), shadow_src,
            /*length*/ length, /*volatile*/ isVolatile, &MTI,
            /*allowForward*/ forwardsShadow, /*shadowsLookedup*/ false,
            /*backwardsShadow*/ backwardsShadow);
      };

      auto fwd_rule = [&](Value *ddst, Value *dsrc) {
        if (ddst == nullptr)
          ddst = gutils->getNewFromOriginal(orig_dst);
        if (dsrc == nullptr)
          dsrc = gutils->getNewFromOriginal(orig_src);
#if LLVM_VERSION_MAJOR >= 10
        MaybeAlign dalign;
        if (subdstalign)
          dalign = MaybeAlign(subdstalign);
        MaybeAlign salign;
        if (subsrcalign)
          salign = MaybeAlign(subsrcalign);
#else
        auto dalign = dstalign;
        auto salign = srcalign;
#endif
        if (ddst->getType()->isIntegerTy())
          ddst = BuilderZ.CreateIntToPtr(
              ddst, Type::getInt8PtrTy(ddst->getContext()));
        if (start != 0) {
          ddst = BuilderZ.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(ddst->getContext()), ddst, start);
        }
        CallInst *call;
        // TODO add EnzymeRuntimeActivity (correctness)
        if (dt.isFloat() && gutils->isConstantValue(orig_src)) {
          call = BuilderZ.CreateMemSet(
              ddst, ConstantInt::get(Type::getInt8Ty(ddst->getContext()), 0),
              length, salign, isVolatile);
        } else {
          if (dsrc->getType()->isIntegerTy())
            dsrc = BuilderZ.CreateIntToPtr(
                dsrc, Type::getInt8PtrTy(dsrc->getContext()));
          if (start != 0) {
            dsrc = BuilderZ.CreateConstInBoundsGEP1_64(
                Type::getInt8Ty(ddst->getContext()), dsrc, start);
          }
          if (ID == Intrinsic::memmove) {
            call = BuilderZ.CreateMemMove(ddst, dalign, dsrc, salign, length);
          } else {
            call = BuilderZ.CreateMemCpy(ddst, dalign, dsrc, salign, length);
          }
          call->setAttributes(MTI.getAttributes());
        }
        // TODO shadow scope/noalias (performance)
        call->setMetadata(LLVMContext::MD_alias_scope,
                          MTI.getMetadata(LLVMContext::MD_alias_scope));
        call->setMetadata(LLVMContext::MD_noalias,
                          MTI.getMetadata(LLVMContext::MD_noalias));
        call->setMetadata(LLVMContext::MD_tbaa,
                          MTI.getMetadata(LLVMContext::MD_tbaa));
        call->setMetadata(LLVMContext::MD_tbaa_struct,
                          MTI.getMetadata(LLVMContext::MD_tbaa_struct));
        call->setMetadata(LLVMContext::MD_invariant_group,
                          MTI.getMetadata(LLVMContext::MD_invariant_group));
        call->setTailCallKind(MTI.getTailCallKind());
      };

      if (Mode == DerivativeMode::ForwardMode)
        applyChainRule(BuilderZ, fwd_rule, shadow_dst, shadow_src);
      else
        applyChainRule(BuilderZ, rev_rule, shadow_dst, shadow_src);

      if (nextStart == size)
        break;
      start = nextStart;
    }

    eraseIfUnused(MTI);
  }

  void visitFenceInst(llvm::FenceInst &FI) {
    using namespace llvm;

    switch (Mode) {
    default:
      break;
    case DerivativeMode::ReverseModeGradient:
    case DerivativeMode::ReverseModeCombined: {
      IRBuilder<> Builder2(&FI);
      getReverseBuilder(Builder2);
      auto order = FI.getOrdering();
      switch (order) {
      case AtomicOrdering::Acquire:
        order = AtomicOrdering::Release;
        break;
      case AtomicOrdering::Release:
        order = AtomicOrdering::Acquire;
        break;
      default:
        break;
      }
      Builder2.CreateFence(order, FI.getSyncScopeID());
    }
    }
    eraseIfUnused(FI);
  }

  void visitIntrinsicInst(llvm::IntrinsicInst &II) {
    using namespace llvm;

    if (II.getIntrinsicID() == Intrinsic::stacksave) {
      eraseIfUnused(II, /*erase*/ true, /*check*/ false);
      return;
    }
    if (II.getIntrinsicID() == Intrinsic::stackrestore ||
        II.getIntrinsicID() == Intrinsic::lifetime_end) {
      eraseIfUnused(II, /*erase*/ true, /*check*/ false);
      return;
    }

    // When compiling Enzyme against standard LLVM, and not Intel's
    // modified version of LLVM, the intrinsic `llvm.intel.subscript` is
    // not fully understood by LLVM. One of the results of this is that the ID
    // of the intrinsic is set to Intrinsic::not_intrinsic - hence we are
    // handling the intrinsic here.
    if (isIntelSubscriptIntrinsic(II)) {
      if (Mode == DerivativeMode::ForwardModeSplit ||
          Mode == DerivativeMode::ForwardMode) {
        forwardModeInvertedPointerFallback(II);
      }
    } else {
      SmallVector<Value *, 2> orig_ops(II.getNumOperands());

      for (unsigned i = 0; i < II.getNumOperands(); ++i) {
        orig_ops[i] = II.getOperand(i);
      }
      if (handleAdjointForIntrinsic(II.getIntrinsicID(), II, orig_ops))
        return;
    }
    if (gutils->knownRecomputeHeuristic.find(&II) !=
        gutils->knownRecomputeHeuristic.end()) {
      if (!gutils->knownRecomputeHeuristic[&II]) {
        CallInst *const newCall =
            cast<CallInst>(gutils->getNewFromOriginal(&II));
        IRBuilder<> BuilderZ(newCall);
        BuilderZ.setFastMathFlags(getFast());

        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&II, CacheType::Self));
      }
    }
    eraseIfUnused(II);
  }

  bool
  handleAdjointForIntrinsic(llvm::Intrinsic::ID ID, llvm::Instruction &I,
                            llvm::SmallVectorImpl<llvm::Value *> &orig_ops) {
    using namespace llvm;

    Module *M = I.getParent()->getParent()->getParent();

    switch (ID) {
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f: {
      auto CI = cast<ConstantInt>(I.getOperand(1));
#if LLVM_VERSION_MAJOR >= 10
      visitLoadLike(I, /*Align*/ MaybeAlign(CI->getZExtValue()),
                    /*constantval*/ false);
#else
      visitLoadLike(I, /*Align*/ CI->getZExtValue(), /*constantval*/ false);
#endif
      return false;
    }
    default:
      break;
    }

    if (ID == Intrinsic::masked_store) {
      auto align0 = cast<ConstantInt>(I.getOperand(2))->getZExtValue();
#if LLVM_VERSION_MAJOR >= 10
      auto align = MaybeAlign(align0);
#else
      auto align = align0;
#endif
      visitCommonStore(I, /*orig_ptr*/ I.getOperand(1),
                       /*orig_val*/ I.getOperand(0), align,
                       /*isVolatile*/ false, llvm::AtomicOrdering::NotAtomic,
                       SyncScope::SingleThread,
                       /*mask*/ gutils->getNewFromOriginal(I.getOperand(3)));
      return false;
    }
    if (ID == Intrinsic::masked_load) {
      auto align0 = cast<ConstantInt>(I.getOperand(1))->getZExtValue();
#if LLVM_VERSION_MAJOR >= 10
      auto align = MaybeAlign(align0);
#else
      auto align = align0;
#endif
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      bool constantval = parseTBAA(I, DL).Inner0().isIntegral();
      visitLoadLike(I, align, constantval,
                    /*mask*/ gutils->getNewFromOriginal(I.getOperand(2)),
                    /*orig_maskInit*/ I.getOperand(3));
      return false;
    }

    auto mod = I.getParent()->getParent()->getParent();
    auto called = cast<CallInst>(&I)->getCalledFunction();
    switch (ID) {
#include "IntrinsicDerivatives.inc"
    default:
      break;
    }

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      switch (ID) {
      case Intrinsic::nvvm_barrier0:
      case Intrinsic::nvvm_barrier0_popc:
      case Intrinsic::nvvm_barrier0_and:
      case Intrinsic::nvvm_barrier0_or:
      case Intrinsic::nvvm_membar_cta:
      case Intrinsic::nvvm_membar_gl:
      case Intrinsic::nvvm_membar_sys:
      case Intrinsic::amdgcn_s_barrier:
        return false;
      default:
        if (gutils->isConstantInstruction(&I))
          return false;
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        ss << "cannot handle (augmented) unknown intrinsic\n" << I;
        if (CustomErrorHandler) {
          IRBuilder<> BuilderZ(&I);
          getForwardBuilder(BuilderZ);
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, gutils, nullptr,
                             wrap(&BuilderZ));
          return false;
        } else {
          EmitFailure("NoDerivative", I.getDebugLoc(), &I, ss.str());
          return false;
        }
      }
      return false;
    }

    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient: {

      IRBuilder<> Builder2(&I);
      getReverseBuilder(Builder2);

      Value *vdiff = nullptr;
      if (!gutils->isConstantValue(&I)) {
        vdiff = diffe(&I, Builder2);
        setDiffe(&I, Constant::getNullValue(gutils->getShadowType(I.getType())),
                 Builder2);
      }

      switch (ID) {

      case Intrinsic::nvvm_barrier0_popc:
      case Intrinsic::nvvm_barrier0_and:
      case Intrinsic::nvvm_barrier0_or: {
        SmallVector<Value *, 1> args = {};
        auto cal = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0), args));
        cal->setCallingConv(
            Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0)
                ->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
        return false;
      }

      case Intrinsic::nvvm_barrier0:
      case Intrinsic::amdgcn_s_barrier:
      case Intrinsic::nvvm_membar_cta:
      case Intrinsic::nvvm_membar_gl:
      case Intrinsic::nvvm_membar_sys: {
        SmallVector<Value *, 1> args = {};
        auto cal = cast<CallInst>(
            Builder2.CreateCall(Intrinsic::getDeclaration(M, ID), args));
        cal->setCallingConv(Intrinsic::getDeclaration(M, ID)->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
        return false;
      }

      case Intrinsic::lifetime_start: {
        if (gutils->isConstantInstruction(&I))
          return false;
        SmallVector<Value *, 2> args = {
            lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2),
            lookup(gutils->getNewFromOriginal(orig_ops[1]), Builder2)};
        Type *tys[] = {args[1]->getType()};
        auto cal = Builder2.CreateCall(
            Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
        cal->setCallingConv(
            Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys)
                ->getCallingConv());
        return false;
      }

#if LLVM_VERSION_MAJOR >= 12
      case Intrinsic::vector_reduce_fmax: {
        if (vdiff && !gutils->isConstantValue(orig_ops[0])) {
          auto prev = lookup(gutils->getNewFromOriginal(orig_ops[0]), Builder2);
          auto VT = cast<VectorType>(orig_ops[0]->getType());

          assert(!VT->getElementCount().isScalable());
          size_t numElems = VT->getElementCount().getKnownMinValue();
          SmallVector<Value *> elems;
          SmallVector<Value *> cmps;

          for (size_t i = 0; i < numElems; ++i)
            elems.push_back(Builder2.CreateExtractElement(prev, (uint64_t)i));

          Value *curmax = elems[0];
          for (size_t i = 0; i < numElems - 1; ++i) {
            cmps.push_back(Builder2.CreateFCmpOLT(curmax, elems[i + 1]));
            if (i + 2 != numElems)
              curmax = CreateSelect(Builder2, cmps[i], elems[i + 1], curmax);
          }

          auto rule = [&](Value *vdiff) {
            auto nv = Constant::getNullValue(orig_ops[0]->getType());
            Value *res = Builder2.CreateInsertElement(nv, vdiff, (uint64_t)0);

            for (size_t i = 0; i < numElems - 1; ++i) {
              auto rhs_v = Builder2.CreateInsertElement(nv, vdiff, i + 1);
              res = CreateSelect(Builder2, cmps[i], rhs_v, res);
            }
            return res;
          };
          Value *dif0 =
              applyChainRule(orig_ops[0]->getType(), Builder2, rule, vdiff);
          addToDiffe(orig_ops[0], dif0, Builder2, I.getType());
        }
        return false;
      }
#endif
      default:
        if (gutils->isConstantInstruction(&I))
          return false;

        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
#else
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
             << I;
#endif
        else
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        if (CustomErrorHandler) {
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, gutils, nullptr,
                             wrap(&Builder2));
          return false;
        } else {
          EmitFailure("NoDerivative", I.getDebugLoc(), &I, ss.str());
          return false;
        }
      }
      return false;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode: {

      IRBuilder<> Builder2(&I);
      getForwardBuilder(Builder2);

      switch (ID) {

#if LLVM_VERSION_MAJOR >= 12
      case Intrinsic::vector_reduce_fmax: {
        if (gutils->isConstantInstruction(&I))
          return false;
        auto prev = gutils->getNewFromOriginal(orig_ops[0]);
        auto VT = cast<VectorType>(orig_ops[0]->getType());

        assert(!VT->getElementCount().isScalable());
        size_t numElems = VT->getElementCount().getKnownMinValue();
        SmallVector<Value *> elems;
        SmallVector<Value *> cmps;

        for (size_t i = 0; i < numElems; ++i)
          elems.push_back(Builder2.CreateExtractElement(prev, (uint64_t)i));

        Value *curmax = elems[0];
        for (size_t i = 0; i < numElems - 1; ++i) {
          cmps.push_back(Builder2.CreateFCmpOLT(curmax, elems[i + 1]));
          if (i + 2 != numElems)
            curmax = CreateSelect(Builder2, cmps[i], elems[i + 1], curmax);
        }

        auto rule = [&](Value *vdiff) {
          Value *res = Builder2.CreateExtractElement(vdiff, (uint64_t)0);

          for (size_t i = 0; i < numElems - 1; ++i) {
            auto rhs_v = Builder2.CreateExtractElement(vdiff, i + 1);
            res = CreateSelect(Builder2, cmps[i], rhs_v, res);
          }
          return res;
        };
        auto vdiff = diffe(orig_ops[0], Builder2);

        Value *dif = applyChainRule(I.getType(), Builder2, rule, vdiff);
        setDiffe(&I, dif, Builder2);
        return false;
      }
#endif
      default:
        if (gutils->isConstantInstruction(&I))
          return false;
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
#if LLVM_VERSION_MAJOR >= 13
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
#else
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>()) << "\n"
             << I;
#endif
        else
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        if (CustomErrorHandler) {
          CustomErrorHandler(ss.str().c_str(), wrap(&I),
                             ErrorType::NoDerivative, gutils, nullptr,
                             wrap(&Builder2));
          setDiffe(&I,
                   Constant::getNullValue(gutils->getShadowType(I.getType())),
                   Builder2);
          return false;
        } else {
          EmitFailure("NoDerivative", I.getDebugLoc(), &I, ss.str());
          return false;
        }
      }
      return false;
    }
    }
  }

// first one allows adding attributes to blas functions declared in the second
#include "BlasAttributor.inc"
#include "BlasDerivatives.inc"

  void visitOMPCall(llvm::CallInst &call) {
    using namespace llvm;

    Function *kmpc = call.getCalledFunction();

    if (overwritten_args_map.find(&call) == overwritten_args_map.end()) {
      llvm::errs() << " call: " << call << "\n";
      for (auto &pair : overwritten_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(overwritten_args_map.find(&call) != overwritten_args_map.end());
    const std::vector<bool> &overwritten_args =
        overwritten_args_map.find(&call)->second;

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    Function *task = dyn_cast<Function>(call.getArgOperand(2));
    if (task == nullptr && isa<ConstantExpr>(call.getArgOperand(2))) {
      task = dyn_cast<Function>(
          cast<ConstantExpr>(call.getArgOperand(2))->getOperand(0));
    }
    if (task == nullptr) {
      llvm::errs() << "could not derive underlying task from omp call: " << call
                   << "\n";
      llvm_unreachable("could not derive underlying task from omp call");
    }
    if (task->empty()) {
      llvm::errs()
          << "could not derive underlying task contents from omp call: " << call
          << "\n";
      llvm_unreachable(
          "could not derive underlying task contents from omp call");
    }

    auto called = task;
    // bool modifyPrimal = true;

    bool foreignFunction = called == nullptr;

    SmallVector<Value *, 8> args = {0, 0, 0};
    SmallVector<Value *, 8> pre_args = {0, 0, 0};
    std::vector<DIFFE_TYPE> argsInverted = {DIFFE_TYPE::CONSTANT,
                                            DIFFE_TYPE::CONSTANT};
    SmallVector<Instruction *, 4> postCreate;
    SmallVector<Instruction *, 4> userReplace;

    SmallVector<Value *, 4> OutTypes;
    SmallVector<Type *, 4> OutFPTypes;

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 3; i < call.arg_size(); ++i)
#else
    for (unsigned i = 3; i < call.getNumArgOperands(); ++i)
#endif
    {

      auto argi = gutils->getNewFromOriginal(call.getArgOperand(i));

      pre_args.push_back(argi);

      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        args.push_back(lookup(argi, Builder2));
      }

      auto argTy = gutils->getDiffeType(call.getArgOperand(i), foreignFunction);
      argsInverted.push_back(argTy);

      if (argTy == DIFFE_TYPE::CONSTANT) {
        continue;
      }

      auto argType = argi->getType();

      if (argTy == DIFFE_TYPE::DUP_ARG || argTy == DIFFE_TYPE::DUP_NONEED) {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          args.push_back(
              lookup(gutils->invertPointerM(call.getArgOperand(i), Builder2),
                     Builder2));
        }
        pre_args.push_back(
            gutils->invertPointerM(call.getArgOperand(i), BuilderZ));

        // Note sometimes whattype mistakenly says something should be constant
        // [because composed of integer pointers alone]
        assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      } else {
        assert(TR.query(call.getArgOperand(i)).Inner0().isFloat());
        OutTypes.push_back(call.getArgOperand(i));
        OutFPTypes.push_back(argType);
        assert(whatType(argType, Mode) == DIFFE_TYPE::OUT_DIFF ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      }
    }

    DIFFE_TYPE subretType = DIFFE_TYPE::CONSTANT;

    Value *tape = nullptr;
    CallInst *augmentcall = nullptr;
    // Value *cachereplace = nullptr;

    // TODO consider reduction of int 0 args
    FnTypeInfo nextTypeInfo(called);

    if (called) {
      std::map<Value *, std::set<int64_t>> intseen;

      TypeTree IntPtr;
      IntPtr.insert({-1, -1}, BaseType::Integer);
      IntPtr.insert({-1}, BaseType::Pointer);

      int argnum = 0;
      for (auto &arg : called->args()) {
        if (argnum <= 1) {
          nextTypeInfo.Arguments.insert(
              std::pair<Argument *, TypeTree>(&arg, IntPtr));
          nextTypeInfo.KnownValues.insert(
              std::pair<Argument *, std::set<int64_t>>(&arg, {0}));
        } else {
          nextTypeInfo.Arguments.insert(std::pair<Argument *, TypeTree>(
              &arg, TR.query(call.getArgOperand(argnum - 2 + 3))));
          nextTypeInfo.KnownValues.insert(
              std::pair<Argument *, std::set<int64_t>>(
                  &arg,
                  TR.knownIntegralValues(call.getArgOperand(argnum - 2 + 3))));
        }

        ++argnum;
      }
      nextTypeInfo.Return = TR.query(&call);
    }

    // std::optional<std::map<std::pair<Instruction*, std::string>, unsigned>>
    // sub_index_map;
    // Optional<int> tapeIdx;
    // Optional<int> returnIdx;
    // Optional<int> differetIdx;

    const AugmentedReturn *subdata = nullptr;
    if (Mode == DerivativeMode::ReverseModeGradient) {
      assert(augmentedReturn);
      if (augmentedReturn) {
        auto fd = augmentedReturn->subaugmentations.find(&call);
        if (fd != augmentedReturn->subaugmentations.end()) {
          subdata = fd->second;
        }
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      if (called) {
        subdata = &gutils->Logic.CreateAugmentedPrimal(
            cast<Function>(called), subretType, argsInverted,
            TR.analyzer.interprocedural, /*return is used*/ false,
            /*shadowReturnUsed*/ false, nextTypeInfo, overwritten_args, false,
            gutils->getWidth(),
            /*AtomicAdd*/ true,
            /*OpenMP*/ true);
        if (Mode == DerivativeMode::ReverseModePrimal) {
          assert(augmentedReturn);
          auto subaugmentations =
              (std::map<const llvm::CallInst *, AugmentedReturn *>
                   *)&augmentedReturn->subaugmentations;
          insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
              *subaugmentations, &call, (AugmentedReturn *)subdata);
        }

        assert(subdata);
        auto newcalled = subdata->fn;

        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          ValueToValueMapTy VMap;
          newcalled = CloneFunction(newcalled, VMap);
          auto tapeArg = newcalled->arg_end();
          tapeArg--;
          Type *tapeElemType = subdata->tapeType;
          SmallVector<std::pair<ssize_t, Value *>, 4> geps;
          SmallPtrSet<Instruction *, 4> gepsToErase;
          for (auto a : tapeArg->users()) {
            if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
              auto idx = gep->idx_begin();
              idx++;
              auto cidx = cast<ConstantInt>(idx->get());
              assert(gep->getNumIndices() == 2);
              SmallPtrSet<StoreInst *, 1> storesToErase;
              for (auto st : gep->users()) {
                auto SI = cast<StoreInst>(st);
                Value *op = SI->getValueOperand();
                storesToErase.insert(SI);
                geps.emplace_back(cidx->getLimitedValue(), op);
              }
              for (auto SI : storesToErase)
                SI->eraseFromParent();
              gepsToErase.insert(gep);
            } else if (auto SI = dyn_cast<StoreInst>(a)) {
              Value *op = SI->getValueOperand();
              gepsToErase.insert(SI);
              geps.emplace_back(-1, op);
            } else {
              llvm::errs() << "unknown tape user: " << a << "\n";
              assert(0 && "unknown tape user");
              llvm_unreachable("unknown tape user");
            }
          }
          for (auto gep : gepsToErase)
            gep->eraseFromParent();
          IRBuilder<> ph(&*newcalled->getEntryBlock().begin());
          tape = UndefValue::get(tapeElemType);
          ValueToValueMapTy available;
          auto subarg = newcalled->arg_begin();
          subarg++;
          subarg++;
          for (size_t i = 3; i < pre_args.size(); ++i) {
            available[&*subarg] = pre_args[i];
            subarg++;
          }
          for (auto pair : geps) {
            Value *op = pair.second;
            Value *alloc = op;
            Value *replacement = gutils->unwrapM(op, BuilderZ, available,
                                                 UnwrapMode::LegalFullUnwrap);
            tape =
                pair.first == -1
                    ? replacement
                    : BuilderZ.CreateInsertValue(tape, replacement, pair.first);
            if (auto ci = dyn_cast<CastInst>(alloc)) {
              alloc = ci->getOperand(0);
            }
            if (auto uload = dyn_cast<Instruction>(replacement)) {
              gutils->unwrappedLoads.erase(uload);
              if (auto ci = dyn_cast<CastInst>(replacement)) {
                if (auto ucast = dyn_cast<Instruction>(ci->getOperand(0)))
                  gutils->unwrappedLoads.erase(ucast);
              }
            }
            if (auto ci = dyn_cast<CallInst>(alloc)) {
              if (auto F = ci->getCalledFunction()) {
                // Store cached values
                if (F->getName() == "malloc") {
                  const_cast<AugmentedReturn *>(subdata)
                      ->tapeIndiciesToFree.emplace(pair.first);
                  Value *Idxs[] = {
                      ConstantInt::get(Type::getInt64Ty(tapeArg->getContext()),
                                       0),
                      ConstantInt::get(Type::getInt32Ty(tapeArg->getContext()),
                                       pair.first)};
                  op->replaceAllUsesWith(ph.CreateLoad(
                      op->getType(),
                      pair.first == -1
                          ? tapeArg
                          : ph.CreateInBoundsGEP(tapeElemType, tapeArg, Idxs)));
                  cast<Instruction>(op)->eraseFromParent();
                  if (op != alloc)
                    ci->eraseFromParent();
                  continue;
                }
              }
            }
            Value *Idxs[] = {
                ConstantInt::get(Type::getInt64Ty(tapeArg->getContext()), 0),
                ConstantInt::get(Type::getInt32Ty(tapeArg->getContext()),
                                 pair.first)};
            op->replaceAllUsesWith(ph.CreateLoad(
                op->getType(),
                pair.first == -1
                    ? tapeArg
                    : ph.CreateInBoundsGEP(tapeElemType, tapeArg, Idxs)));
            cast<Instruction>(op)->eraseFromParent();
          }
          assert(tape);
          auto alloc =
              IRBuilder<>(gutils->inversionAllocs).CreateAlloca(tapeElemType);
          BuilderZ.CreateStore(tape, alloc);
          pre_args.push_back(alloc);
          assert(tape);
          gutils->cacheForReverse(BuilderZ, tape,
                                  getIndex(&call, CacheType::Tape));
        }

        auto numargs = ConstantInt::get(Type::getInt32Ty(call.getContext()),
                                        pre_args.size() - 3);
        pre_args[0] = gutils->getNewFromOriginal(call.getArgOperand(0));
        pre_args[1] = numargs;
        pre_args[2] = BuilderZ.CreatePointerCast(
            newcalled, kmpc->getFunctionType()->getParamType(2));
        augmentcall =
            BuilderZ.CreateCall(kmpc->getFunctionType(), kmpc, pre_args);
        augmentcall->setCallingConv(call.getCallingConv());
        augmentcall->setDebugLoc(
            gutils->getNewFromOriginal(call.getDebugLoc()));
        BuilderZ.SetInsertPoint(
            gutils->getNewFromOriginal(&call)->getNextNode());
        gutils->erase(gutils->getNewFromOriginal(&call));
      } else {
        assert(0 && "unhandled unknown outline");
      }
    }

    if (!subdata) {
      llvm::errs() << *gutils->oldFunc->getParent() << "\n";
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      llvm::errs() << *called << "\n";
      llvm_unreachable("no subdata");
    }

    auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
    assert(found == subdata->returns.end());

    found = subdata->returns.find(AugmentedStruct::Return);
    assert(found == subdata->returns.end());

    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      if (Mode == DerivativeMode::ReverseModeGradient) {
        BuilderZ.SetInsertPoint(
            gutils->getNewFromOriginal(&call)->getNextNode());
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }

      Function *newcalled = nullptr;
      if (called) {
        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          if (Mode == DerivativeMode::ReverseModeGradient) {
            if (tape == nullptr)
              tape = BuilderZ.CreatePHI(subdata->tapeType, 0, "tapeArg");
            tape = gutils->cacheForReverse(BuilderZ, tape,
                                           getIndex(&call, CacheType::Tape));
          }
          tape = lookup(tape, Builder2);
          auto alloc = IRBuilder<>(gutils->inversionAllocs)
                           .CreateAlloca(tape->getType());
          Builder2.CreateStore(tape, alloc);
          args.push_back(alloc);
        }

        if (Mode == DerivativeMode::ReverseModeGradient && subdata) {
          for (size_t i = 0; i < argsInverted.size(); i++) {
            if (subdata->constant_args[i] == argsInverted[i])
              continue;
            assert(subdata->constant_args[i] == DIFFE_TYPE::DUP_ARG);
            assert(argsInverted[i] == DIFFE_TYPE::DUP_NONEED);
            argsInverted[i] = DIFFE_TYPE::DUP_ARG;
          }
        }

        newcalled = gutils->Logic.CreatePrimalAndGradient(
            (ReverseCacheKey){.todiff = cast<Function>(called),
                              .retType = subretType,
                              .constant_args = argsInverted,
                              .overwritten_args = overwritten_args,
                              .returnUsed = false,
                              .shadowReturnUsed = false,
                              .mode = DerivativeMode::ReverseModeGradient,
                              .width = gutils->getWidth(),
                              .freeMemory = true,
                              .AtomicAdd = true,
                              .additionalType =
                                  tape ? PointerType::getUnqual(tape->getType())
                                       : nullptr,
                              .forceAnonymousTape = false,
                              .typeInfo = nextTypeInfo},
            TR.analyzer.interprocedural, subdata,
            /*omp*/ true);

        if (subdata->returns.find(AugmentedStruct::Tape) !=
            subdata->returns.end()) {
          auto tapeArg = newcalled->arg_end();
          tapeArg--;
          LoadInst *tape = nullptr;
          for (auto u : tapeArg->users()) {
            assert(!tape);
            if (!isa<LoadInst>(u)) {
              llvm::errs() << " newcalled: " << *newcalled << "\n";
              llvm::errs() << " u: " << *u << "\n";
            }
            tape = cast<LoadInst>(u);
          }
          assert(tape);
          SmallVector<Value *, 4> extracts;
          if (subdata->tapeIndices.size() == 1) {
            assert(subdata->tapeIndices.begin()->second == -1);
            extracts.push_back(tape);
          } else {
            for (auto a : tape->users()) {
              extracts.push_back(a);
            }
          }
          SmallVector<LoadInst *, 4> geps;
          for (auto E : extracts) {
            AllocaInst *AI = nullptr;
            for (auto U : E->users()) {
              if (auto SI = dyn_cast<StoreInst>(U)) {
                assert(SI->getValueOperand() == E);
                AI = cast<AllocaInst>(SI->getPointerOperand());
              }
            }
            if (AI) {
              for (auto U : AI->users()) {
                if (auto LI = dyn_cast<LoadInst>(U)) {
                  geps.push_back(LI);
                }
              }
            }
          }
          size_t freeCount = 0;
          for (auto LI : geps) {
            CallInst *freeCall = nullptr;
            for (auto LU : LI->users()) {
              if (auto CI = dyn_cast<CallInst>(LU)) {
                if (auto F = CI->getCalledFunction()) {
                  if (F->getName() == "free") {
                    freeCall = CI;
                    break;
                  }
                }
              } else if (auto BC = dyn_cast<CastInst>(LU)) {
                for (auto CU : BC->users()) {
                  if (auto CI = dyn_cast<CallInst>(CU)) {
                    if (auto F = CI->getCalledFunction()) {
                      if (F->getName() == "free") {
                        freeCall = CI;
                        break;
                      }
                    }
                  }
                }
                if (freeCall)
                  break;
              }
            }
            if (freeCall) {
              freeCall->eraseFromParent();
              freeCount++;
            }
          }
        }

        Value *OutAlloc = nullptr;
        auto ST = StructType::get(newcalled->getContext(), OutFPTypes);
        if (OutTypes.size()) {
          OutAlloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(ST);
          args.push_back(OutAlloc);

          SmallVector<Type *, 3> MetaTypes;
          for (auto P :
               cast<Function>(newcalled)->getFunctionType()->params()) {
            MetaTypes.push_back(P);
          }
          MetaTypes.push_back(PointerType::getUnqual(ST));
          auto FT = FunctionType::get(Type::getVoidTy(newcalled->getContext()),
                                      MetaTypes, false);
#if LLVM_VERSION_MAJOR >= 10
          Function *F =
              Function::Create(FT, GlobalVariable::InternalLinkage,
                               cast<Function>(newcalled)->getName() + "#out",
                               *task->getParent());
#else
          Function *F = Function::Create(
              FT, GlobalVariable::InternalLinkage,
              cast<Function>(newcalled)->getName() + "#out", task->getParent());
#endif
          BasicBlock *entry =
              BasicBlock::Create(newcalled->getContext(), "entry", F);
          IRBuilder<> B(entry);
          SmallVector<Value *, 2> SubArgs;
          for (auto &arg : F->args())
            SubArgs.push_back(&arg);
          Value *cacheArg = SubArgs.back();
          SubArgs.pop_back();
          Value *outdiff = B.CreateCall(newcalled, SubArgs);
          for (size_t ee = 0; ee < OutTypes.size(); ee++) {
            Value *dif = B.CreateExtractValue(outdiff, ee);
            Value *Idxs[] = {
                ConstantInt::get(Type::getInt64Ty(ST->getContext()), 0),
                ConstantInt::get(Type::getInt32Ty(ST->getContext()), ee)};
            Value *ptr = B.CreateInBoundsGEP(ST, cacheArg, Idxs);

            if (dif->getType()->isIntOrIntVectorTy()) {

              ptr = B.CreateBitCast(
                  ptr,
                  PointerType::get(
                      IntToFloatTy(dif->getType()),
                      cast<PointerType>(ptr->getType())->getAddressSpace()));
              dif = B.CreateBitCast(dif, IntToFloatTy(dif->getType()));
            }

#if LLVM_VERSION_MAJOR >= 10
            MaybeAlign align;
#else
            unsigned align = 0;
#endif

            AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
            if (auto vt = dyn_cast<VectorType>(dif->getType())) {
#if LLVM_VERSION_MAJOR >= 12
              assert(!vt->getElementCount().isScalable());
              size_t numElems = vt->getElementCount().getKnownMinValue();
#else
              size_t numElems = vt->getNumElements();
#endif
              for (size_t i = 0; i < numElems; ++i) {
                auto vdif = B.CreateExtractElement(dif, i);
                Value *Idxs[] = {
                    ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
                    ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
                auto vptr = B.CreateInBoundsGEP(vt, ptr, Idxs);
#if LLVM_VERSION_MAJOR >= 13
                B.CreateAtomicRMW(op, vptr, vdif, align,
                                  AtomicOrdering::Monotonic, SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
                AtomicRMWInst *rmw =
                    B.CreateAtomicRMW(op, vptr, vdif, AtomicOrdering::Monotonic,
                                      SyncScope::System);
                if (align)
                  rmw->setAlignment(align.getValue());
#else
                                   B.CreateAtomicRMW(op, vptr, vdif,
                                                     AtomicOrdering::Monotonic,
                                                     SyncScope::System);
#endif
              }
            } else {
#if LLVM_VERSION_MAJOR >= 13
              B.CreateAtomicRMW(op, ptr, dif, align, AtomicOrdering::Monotonic,
                                SyncScope::System);
#elif LLVM_VERSION_MAJOR >= 11
              AtomicRMWInst *rmw = B.CreateAtomicRMW(
                  op, ptr, dif, AtomicOrdering::Monotonic, SyncScope::System);
              if (align)
                rmw->setAlignment(align.getValue());
#else
                                 B.CreateAtomicRMW(op, ptr, dif,
                                                   AtomicOrdering::Monotonic,
                                                   SyncScope::System);
#endif
            }
          }
          B.CreateRetVoid();
          newcalled = F;
        }

        auto numargs = ConstantInt::get(Type::getInt32Ty(call.getContext()),
                                        args.size() - 3);
        args[0] =
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2);
        args[1] = numargs;
        args[2] = Builder2.CreatePointerCast(
            newcalled, kmpc->getFunctionType()->getParamType(2));

        CallInst *diffes =
            Builder2.CreateCall(kmpc->getFunctionType(), kmpc, args);
        diffes->setCallingConv(call.getCallingConv());
        diffes->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

        for (size_t i = 0; i < OutTypes.size(); i++) {

          size_t size = 1;
          if (OutTypes[i]->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(OutTypes[i]->getType()) +
                    7) /
                   8;
          Value *Idxs[] = {
              ConstantInt::get(Type::getInt64Ty(call.getContext()), 0),
              ConstantInt::get(Type::getInt32Ty(call.getContext()), i)};
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(OutTypes[i],
                           Builder2.CreateLoad(
                               OutFPTypes[i],
                               Builder2.CreateInBoundsGEP(ST, OutAlloc, Idxs)),
                           Builder2, TR.addingType(size, OutTypes[i]));
        }

        if (tape && shouldFree()) {
          for (auto idx : subdata->tapeIndiciesToFree) {
            CreateDealloc(Builder2,
                          idx == -1 ? tape
                                    : Builder2.CreateExtractValue(tape, idx));
          }
        }
      } else {
        assert(0 && "openmp indirect unhandled");
      }
    }
  }

  void DifferentiableMemCopyFloats(
      llvm::CallInst &call, llvm::Value *origArg, llvm::Value *dsto,
      llvm::Value *srco, llvm::Value *len_arg, llvm::IRBuilder<> &Builder2,
      llvm::ArrayRef<llvm::OperandBundleDef> ReverseDefs) {
    using namespace llvm;

    size_t size = 1;
    if (auto ci = dyn_cast<ConstantInt>(len_arg)) {
      size = ci->getLimitedValue();
    }
    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto vd = TR.query(origArg).Data0().ShiftIndices(DL, 0, size, 0);
    if (!vd.isKnownPastPointer()) {
#if LLVM_VERSION_MAJOR < 18
      if (looseTypeAnalysis) {
        if (isa<CastInst>(origArg) &&
            cast<CastInst>(origArg)->getSrcTy()->isPointerTy() &&
            cast<CastInst>(origArg)
                ->getSrcTy()
                ->getPointerElementType()
                ->isFPOrFPVectorTy()) {
          vd = TypeTree(ConcreteType(cast<CastInst>(origArg)
                                         ->getSrcTy()
                                         ->getPointerElementType()
                                         ->getScalarType()))
                   .Only(0, &call);
          goto knownF;
        }
      }
#endif
      EmitFailure("CannotDeduceType", call.getDebugLoc(), &call,
                  "failed to deduce type of copy ", call);

      TR.firstPointer(size, origArg, &call, /*errifnotfound*/ true,
                      /*pointerIntSame*/ true);
      llvm_unreachable("bad mti");
    }
  knownF:;
    unsigned start = 0;
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

      Value *length = len_arg;
      if (nextStart != size) {
        length = ConstantInt::get(len_arg->getType(), nextStart);
      }
      if (start != 0)
        length = Builder2.CreateSub(
            length, ConstantInt::get(len_arg->getType(), start));

      if (auto secretty = dt.isFloat()) {
        auto offset = start;
        if (dsto->getType()->isIntegerTy())
          dsto = Builder2.CreateIntToPtr(
              dsto, Type::getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        auto secretpt = PointerType::get(secretty, dstaddr);
        if (offset != 0) {
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(dsto->getContext()), dsto, offset);
        }
        if (srco->getType()->isIntegerTy())
          srco = Builder2.CreateIntToPtr(
              srco, Type::getInt8PtrTy(dsto->getContext()));
        unsigned srcaddr =
            cast<PointerType>(srco->getType())->getAddressSpace();
        secretpt = PointerType::get(secretty, srcaddr);

        if (offset != 0) {
          srco = Builder2.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(srco->getContext()), srco, offset);
        }
        Value *args[3] = {
            Builder2.CreatePointerCast(dsto, secretpt),
            Builder2.CreatePointerCast(srco, secretpt),
            Builder2.CreateUDiv(
                length,

                ConstantInt::get(length->getType(),
                                 Builder2.GetInsertBlock()
                                         ->getParent()
                                         ->getParent()
                                         ->getDataLayout()
                                         .getTypeAllocSizeInBits(secretty) /
                                     8))};

        auto dmemcpy = getOrInsertDifferentialFloatMemcpy(
            *Builder2.GetInsertBlock()->getParent()->getParent(), secretty,
            /*dstalign*/ 1, /*srcalign*/ 1, dstaddr, srcaddr,
            cast<IntegerType>(length->getType())->getBitWidth());

        Builder2.CreateCall(dmemcpy, args, ReverseDefs);
      }

      if (nextStart == size)
        break;
      start = nextStart;
    }
  }

  void handleMPI(llvm::CallInst &call, llvm::Function *called,
                 llvm::StringRef funcName) {
    using namespace llvm;

    assert(called);
    assert(gutils->getWidth() == 1);

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    // MPI send / recv can only send float/integers
    if (funcName == "PMPI_Isend" || funcName == "MPI_Isend" ||
        funcName == "PMPI_Irecv" || funcName == "MPI_Irecv") {
      if (!gutils->isConstantInstruction(&call)) {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          assert(!gutils->isConstantValue(call.getOperand(0)));
          assert(!gutils->isConstantValue(call.getOperand(6)));
          Value *d_req = gutils->invertPointerM(call.getOperand(6), BuilderZ);
          if (d_req->getType()->isIntegerTy()) {
            d_req = BuilderZ.CreateIntToPtr(
                d_req,
                PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
          }

          auto i64 = Type::getInt64Ty(call.getContext());
          auto impi = getMPIHelper(call.getContext());

          Value *impialloc =
              CreateAllocation(BuilderZ, impi, ConstantInt::get(i64, 1));
          BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

          d_req = BuilderZ.CreateBitCast(
              d_req, PointerType::getUnqual(impialloc->getType()));
          Value *d_req_prev = BuilderZ.CreateLoad(impialloc->getType(), d_req);
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(d_req_prev,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::Old>(BuilderZ, impialloc, impi));
          BuilderZ.CreateStore(impialloc, d_req);

          if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            Value *tysize =
                MPI_TYPE_SIZE(gutils->getNewFromOriginal(call.getOperand(2)),
                              BuilderZ, call.getType());

            auto len_arg = BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(1)),
                Type::getInt64Ty(call.getContext()));
            len_arg = BuilderZ.CreateMul(
                len_arg,
                BuilderZ.CreateZExtOrTrunc(tysize,
                                           Type::getInt64Ty(call.getContext())),
                "", true, true);

            Value *firstallocation =
                CreateAllocation(BuilderZ, Type::getInt8Ty(call.getContext()),
                                 len_arg, "mpirecv_malloccache");
            BuilderZ.CreateStore(
                firstallocation,
                getMPIMemberPtr<MPI_Elem::Buf>(BuilderZ, impialloc, impi));
            BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));
          } else {
            Value *ibuf = gutils->invertPointerM(call.getOperand(0), BuilderZ);
            if (ibuf->getType()->isIntegerTy())
              ibuf = BuilderZ.CreateIntToPtr(
                  ibuf, Type::getInt8PtrTy(call.getContext()));
            BuilderZ.CreateStore(ibuf, getMPIMemberPtr<MPI_Elem::Buf>(
                                           BuilderZ, impialloc, impi));
          }

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(1)), i64),
              getMPIMemberPtr<MPI_Elem::Count>(BuilderZ, impialloc, impi));

          Value *dataType = gutils->getNewFromOriginal(call.getOperand(2));
          if (dataType->getType()->isIntegerTy())
            dataType = BuilderZ.CreateIntToPtr(
                dataType, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(dataType,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::DataType>(BuilderZ, impialloc, impi));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(3)), i64),
              getMPIMemberPtr<MPI_Elem::Src>(BuilderZ, impialloc, impi));

          BuilderZ.CreateStore(
              BuilderZ.CreateZExtOrTrunc(
                  gutils->getNewFromOriginal(call.getOperand(4)), i64),
              getMPIMemberPtr<MPI_Elem::Tag>(BuilderZ, impialloc, impi));

          Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
          if (comm->getType()->isIntegerTy())
            comm = BuilderZ.CreateIntToPtr(
                comm, Type::getInt8PtrTy(dataType->getContext()));
          BuilderZ.CreateStore(
              BuilderZ.CreatePointerCast(comm,
                                         Type::getInt8PtrTy(call.getContext())),
              getMPIMemberPtr<MPI_Elem::Comm>(BuilderZ, impialloc, impi));

          BuilderZ.CreateStore(
              ConstantInt::get(
                  Type::getInt8Ty(impialloc->getContext()),
                  (funcName == "MPI_Isend" || funcName == "PMPI_Isend")
                      ? (int)MPI_CallType::ISEND
                      : (int)MPI_CallType::IRECV),
              getMPIMemberPtr<MPI_Elem::Call>(BuilderZ, impialloc, impi));
          // TODO old
        }
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);

          Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 18
          if (Function *recvfn =
                  called->getParent()->getFunction("PMPI_Wait")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          }
          if (Function *recvfn = called->getParent()->getFunction("MPI_Wait")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          }
#endif
          if (statusType == nullptr) {
            statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
            llvm::errs() << " warning could not automatically determine mpi "
                            "status type, assuming [24 x i8]\n";
          }
          Value *req =
              lookup(gutils->getNewFromOriginal(call.getOperand(6)), Builder2);
          Value *d_req = lookup(
              gutils->invertPointerM(call.getOperand(6), Builder2), Builder2);
          if (d_req->getType()->isIntegerTy()) {
            d_req = Builder2.CreateIntToPtr(
                d_req, Type::getInt8PtrTy(call.getContext()));
          }
          auto impi = getMPIHelper(call.getContext());
          Type *helperTy = llvm::PointerType::getUnqual(impi);
          Value *helper = Builder2.CreatePointerCast(
              d_req, PointerType::getUnqual(helperTy));
          helper = Builder2.CreateLoad(helperTy, helper);

          auto i64 = Type::getInt64Ty(call.getContext());

          Value *firstallocation;
          firstallocation = Builder2.CreateLoad(
              Type::getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::Buf>(Builder2, helper, impi));
          Value *len_arg = nullptr;
          if (auto C = dyn_cast<Constant>(
                  gutils->getNewFromOriginal(call.getOperand(1)))) {
            len_arg = Builder2.CreateZExtOrTrunc(C, i64);
          } else {
            len_arg = Builder2.CreateLoad(
                i64, getMPIMemberPtr<MPI_Elem::Count>(Builder2, helper, impi));
          }
          Value *tysize = nullptr;
          if (auto C = dyn_cast<Constant>(
                  gutils->getNewFromOriginal(call.getOperand(2)))) {
            tysize = C;
          } else {
            tysize = Builder2.CreateLoad(
                Type::getInt8PtrTy(call.getContext()),
                getMPIMemberPtr<MPI_Elem::DataType>(Builder2, helper, impi));
          }

          Value *prev;
          prev = Builder2.CreateLoad(
              Type::getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::Old>(Builder2, helper, impi));

          Builder2.CreateStore(
              prev, Builder2.CreatePointerCast(
                        d_req, PointerType::getUnqual(prev->getType())));

          assert(shouldFree());

          assert(tysize);
          tysize = MPI_TYPE_SIZE(tysize, Builder2, call.getType());

          Value *args[] = {/*req*/ req,
                           /*status*/ IRBuilder<>(gutils->inversionAllocs)
                               .CreateAlloca(statusType)};
          FunctionCallee waitFunc = nullptr;
          for (auto name : {"PMPI_Wait", "MPI_Wait"})
            if (Function *recvfn = called->getParent()->getFunction(name)) {
              auto statusArg = recvfn->arg_end();
              statusArg--;
              if (statusArg->getType()->isIntegerTy())
                args[1] =
                    Builder2.CreatePtrToInt(args[1], statusArg->getType());
              else
                args[1] = Builder2.CreateBitCast(args[1], statusArg->getType());
              waitFunc = recvfn;
              break;
            }
          if (!waitFunc) {
            Type *types[sizeof(args) / sizeof(*args)];
            for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
              types[i] = args[i]->getType();
            FunctionType *FT = FunctionType::get(call.getType(), types, false);
            waitFunc = called->getParent()->getOrInsertFunction("MPI_Wait", FT);
          }
          assert(waitFunc);

          // Need to preserve the shadow Request (operand 6 in isend/irecv),
          // which becomes operand 0 for iwait.
          auto ReqDefs = gutils->getInvertedBundles(
              &call,
              {ValueType::None, ValueType::None, ValueType::None,
               ValueType::None, ValueType::None, ValueType::None,
               ValueType::Shadow},
              Builder2, /*lookup*/ true);

          auto BufferDefs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::None, ValueType::None,
               ValueType::None, ValueType::None, ValueType::None,
               ValueType::None},
              Builder2, /*lookup*/ true);

          auto fcall = Builder2.CreateCall(waitFunc, args, ReqDefs);
          fcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
          if (auto F = dyn_cast<Function>(waitFunc.getCallee()))
            fcall->setCallingConv(F->getCallingConv());
          len_arg = Builder2.CreateMul(
              len_arg,
              Builder2.CreateZExtOrTrunc(
                  tysize, Type::getInt64Ty(Builder2.getContext())),
              "", true, true);
          if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
            auto val_arg =
                ConstantInt::get(Type::getInt8Ty(Builder2.getContext()), 0);
            auto volatile_arg = ConstantInt::getFalse(Builder2.getContext());
            assert(!gutils->isConstantValue(call.getOperand(0)));
            auto dbuf = firstallocation;
            Value *nargs[] = {dbuf, val_arg, len_arg, volatile_arg};
            Type *tys[] = {dbuf->getType(), len_arg->getType()};

            auto memset = cast<CallInst>(Builder2.CreateCall(
                Intrinsic::getDeclaration(called->getParent(),
                                          Intrinsic::memset, tys),
                nargs, BufferDefs));
            memset->addParamAttr(0, Attribute::NonNull);
          } else if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
            assert(!gutils->isConstantValue(call.getOperand(0)));
            Value *shadow = lookup(
                gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);

            // TODO add operand bundle (unless force inlined?)
            DifferentiableMemCopyFloats(call, call.getOperand(0),
                                        firstallocation, shadow, len_arg,
                                        Builder2, BufferDefs);

            if (shouldFree()) {
              CreateDealloc(Builder2, firstallocation);
            }
          } else
            assert(0 && "illegal mpi");

          CreateDealloc(Builder2, helper);
        }
        if (Mode == DerivativeMode::ForwardMode) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          assert(!gutils->isConstantValue(call.getOperand(0)));
          assert(!gutils->isConstantValue(call.getOperand(6)));

          Value *buf = gutils->invertPointerM(call.getOperand(0), Builder2);
          Value *count = gutils->getNewFromOriginal(call.getOperand(1));
          Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
          Value *source = gutils->getNewFromOriginal(call.getOperand(3));
          Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
          Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
          Value *request = gutils->invertPointerM(call.getOperand(6), Builder2);

          Value *args[] = {
              /*buf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*source*/ source,
              /*tag*/ tag,
              /*comm*/ comm,
              /*request*/ request,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
      Value *d_reqp = nullptr;
      auto impi = getMPIHelper(call.getContext());
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        Value *req = gutils->getNewFromOriginal(call.getOperand(0));
        Value *d_req = gutils->invertPointerM(call.getOperand(0), BuilderZ);

        if (req->getType()->isIntegerTy()) {
          req = BuilderZ.CreateIntToPtr(
              req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *isNull = nullptr;
        if (auto GV = gutils->newFunc->getParent()->getNamedValue(
                "ompi_request_null")) {
          Value *reql = BuilderZ.CreatePointerCast(
              req, PointerType::getUnqual(GV->getType()));
          reql = BuilderZ.CreateLoad(GV->getType(), reql);
          isNull = BuilderZ.CreateICmpEQ(reql, GV);
        }

        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        d_reqp = BuilderZ.CreateLoad(
            PointerType::getUnqual(impi),
            BuilderZ.CreatePointerCast(
                d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));
        if (isNull)
          d_reqp =
              CreateSelect(BuilderZ, isNull,
                           Constant::getNullValue(d_reqp->getType()), d_reqp);
        if (auto I = dyn_cast<Instruction>(d_reqp))
          gutils->TapesToPreventRecomputation.insert(I);
        d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                         getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));
        Value *req =
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);

        if (Mode != DerivativeMode::ReverseModeCombined) {
          d_reqp = BuilderZ.CreatePHI(PointerType::getUnqual(impi), 0);
          d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                           getIndex(&call, CacheType::Tape));
        } else
          assert(d_reqp);
        d_reqp = lookup(d_reqp, Builder2);

        Value *isNull = Builder2.CreateICmpEQ(
            d_reqp, Constant::getNullValue(d_reqp->getType()));

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_nonnull");
        BasicBlock *endBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_end",
            /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(isNull, endBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

        Value *cache = Builder2.CreateLoad(impi, d_reqp);

        Value *args[] = {
            getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
            req};
        Type *types[sizeof(args) / sizeof(*args) - 1];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
          types[i] = args[i]->getType();
        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, call.getOperand(0)->getType());

        // Need to preserve the shadow Request (operand 0 in wait).
        // However, this doesn't end up preserving
        // the underlying buffers for the adjoint. To rememdy, force inline.
        auto cal =
            Builder2.CreateCall(dwait, args,
                                gutils->getInvertedBundles(
                                    &call, {ValueType::Shadow, ValueType::None},
                                    Builder2, /*lookup*/ true));
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 14
        cal->addFnAttr(Attribute::AlwaysInline);
#else
        cal->addAttribute(AttributeList::FunctionIndex,
                          Attribute::AlwaysInline);
#endif
        Builder2.CreateBr(endBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }
        Builder2.SetInsertPoint(endBlock);
      } else if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));

        Value *request =
            gutils->invertPointerM(call.getArgOperand(0), Builder2);
        Value *status = gutils->invertPointerM(call.getArgOperand(1), Builder2);

        if (request->getType()->isIntegerTy()) {
          request = Builder2.CreateIntToPtr(
              request,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *args[] = {/*request*/ request,
                         /*status*/ status};

        auto Defs = gutils->getInvertedBundles(
            &call, {ValueType::Shadow, ValueType::Shadow}, Builder2,
            /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif

        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
      Value *d_reqp = nullptr;
      auto impi = getMPIHelper(call.getContext());
      PointerType *reqType = PointerType::getUnqual(impi);
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        Value *count = gutils->getNewFromOriginal(call.getOperand(0));
        Value *req = gutils->getNewFromOriginal(call.getOperand(1));
        Value *d_req = gutils->invertPointerM(call.getOperand(1), BuilderZ);

        if (req->getType()->isIntegerTy()) {
          req = BuilderZ.CreateIntToPtr(
              req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Function *dsave = getOrInsertDifferentialWaitallSave(
            *gutils->oldFunc->getParent(),
            {count->getType(), req->getType(), d_req->getType()}, reqType);

        d_reqp = BuilderZ.CreateCall(dsave, {count, req, d_req});
        cast<CallInst>(d_reqp)->setCallingConv(dsave->getCallingConv());
        cast<CallInst>(d_reqp)->setDebugLoc(
            gutils->getNewFromOriginal(call.getDebugLoc()));
        d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                         getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(1)));
        Value *count =
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);
        Value *req_orig =
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2);

        if (Mode != DerivativeMode::ReverseModeCombined) {
          d_reqp = BuilderZ.CreatePHI(PointerType::getUnqual(reqType), 0);
          d_reqp = gutils->cacheForReverse(BuilderZ, d_reqp,
                                           getIndex(&call, CacheType::Tape));
        }

        d_reqp = lookup(d_reqp, Builder2);

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *loopBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_loop");
        BasicBlock *nonnullBlock = gutils->addReverseBlock(
            loopBlock, currentBlock->getName() + "_nonnull");
        BasicBlock *eloopBlock = gutils->addReverseBlock(
            nonnullBlock, currentBlock->getName() + "_eloop");
        BasicBlock *endBlock = gutils->addReverseBlock(
            eloopBlock, currentBlock->getName() + "_end",
            /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(
            Builder2.CreateICmpNE(count,
                                  ConstantInt::get(count->getType(), 0, false)),
            loopBlock, endBlock);

        Builder2.SetInsertPoint(loopBlock);
        auto idx = Builder2.CreatePHI(count->getType(), 2);
        idx->addIncoming(ConstantInt::get(count->getType(), 0, false),
                         currentBlock);
        Value *inc = Builder2.CreateAdd(
            idx, ConstantInt::get(count->getType(), 1, false), "", true, true);
        idx->addIncoming(inc, eloopBlock);

        Value *idxs[] = {idx};
        Value *req = Builder2.CreateInBoundsGEP(reqType, req_orig, idxs);
        Value *d_req = Builder2.CreateInBoundsGEP(reqType, d_reqp, idxs);

        d_req = Builder2.CreateLoad(
            PointerType::getUnqual(impi),
            Builder2.CreatePointerCast(
                d_req, PointerType::getUnqual(PointerType::getUnqual(impi))));

        Value *isNull = Builder2.CreateICmpEQ(
            d_req, Constant::getNullValue(d_req->getType()));

        Builder2.CreateCondBr(isNull, eloopBlock, nonnullBlock);
        Builder2.SetInsertPoint(nonnullBlock);

        Value *cache = Builder2.CreateLoad(impi, d_req);

        Value *args[] = {
            getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
            getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
            req};
        Type *types[sizeof(args) / sizeof(*args) - 1];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
          types[i] = args[i]->getType();
        Function *dwait = getOrInsertDifferentialMPI_Wait(
            *called->getParent(), types, req->getType());
        // Need to preserve the shadow Request (operand 6 in isend/irecv), which
        // becomes operand 0 for iwait. However, this doesn't end up preserving
        // the underlying buffers for the adjoint. To remedy, force inline the
        // function.
        auto cal = Builder2.CreateCall(
            dwait, args,
            gutils->getInvertedBundles(&call,
                                       {ValueType::None, ValueType::None,
                                        ValueType::None, ValueType::None,
                                        ValueType::None, ValueType::None,
                                        ValueType::Shadow},
                                       Builder2, /*lookup*/ true));
        cal->setCallingConv(dwait->getCallingConv());
        cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
#if LLVM_VERSION_MAJOR >= 14
        cal->addFnAttr(Attribute::AlwaysInline);
#else
        cal->addAttribute(AttributeList::FunctionIndex,
                          Attribute::AlwaysInline);
#endif
        Builder2.CreateBr(eloopBlock);

        Builder2.SetInsertPoint(eloopBlock);
        Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, count), endBlock,
                              loopBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }
        Builder2.SetInsertPoint(endBlock);
        if (shouldFree()) {
          CreateDealloc(Builder2, d_reqp);
        }
      } else if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> Builder2(&call);

        assert(!gutils->isConstantValue(call.getOperand(1)));

        Value *count = gutils->getNewFromOriginal(call.getOperand(0));
        Value *array_of_requests = gutils->invertPointerM(
            gutils->getNewFromOriginal(call.getOperand(1)), Builder2);
        if (array_of_requests->getType()->isIntegerTy()) {
          array_of_requests = Builder2.CreateIntToPtr(
              array_of_requests,
              PointerType::getUnqual(Type::getInt8PtrTy(call.getContext())));
        }

        Value *args[] = {
            /*count*/ count,
            /*array_of_requests*/ array_of_requests,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::None, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::Shadow},
            Builder2, /*lookup*/ true);

#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif

        Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
        return;
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Send" || funcName == "MPI_Ssend" ||
        funcName == "PMPI_Send" || funcName == "PMPI_Ssend") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
        if (called->getContext().supportsTypedPointers()) {
#endif
          if (Function *recvfn = called->getParent()->getFunction("MPI_Recv")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          } else if (Function *recvfn =
                         called->getParent()->getFunction("PMPI_Recv")) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
              statusType = PT->getPointerElementType();
          }
#if LLVM_VERSION_MAJOR >= 15
        }
#endif
#endif
        if (statusType == nullptr) {
          statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
          llvm::errs() << " warning could not automatically determine mpi "
                          "status type, assuming [24 x i8]\n";
        }

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *src = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          src = lookup(src, Builder2);

        Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          tag = lookup(tag, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*buf*/ shadow,
              /*count*/ count,
              /*datatype*/ datatype,
              /*dest*/ src,
              /*tag*/ tag,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }

        Value *args[] = {
            /*buf*/ NULL,
            /*count*/ count,
            /*datatype*/ datatype,
            /*src*/ src,
            /*tag*/ tag,
            /*comm*/ comm,
            /*status*/
            IRBuilder<>(gutils->inversionAllocs).CreateAlloca(statusType)};

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        auto len_arg = Builder2.CreateZExtOrTrunc(
            args[1], Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        Value *firstallocation =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             len_arg, "mpirecv_malloccache");
        args[0] = firstallocation;

        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();
        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        Builder2.SetInsertPoint(Builder2.GetInsertBlock());

        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);

        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Recv", FT), args);
        fcall->setCallingConv(call.getCallingConv());

        DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                    shadow, len_arg, Builder2, BufferDefs);

        if (shouldFree()) {
          CreateDealloc(Builder2, firstallocation);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *source = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          source = lookup(source, Builder2);

        Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          tag = lookup(tag, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *args[] = {
            shadow, count, datatype, source, tag, comm,
        };

        auto Defs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::None},
            Builder2, /*lookup*/ !forwardMode);

        if (forwardMode) {
#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif

          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }

        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();
        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        auto fcall = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Send", FT), args,
            Defs);
        fcall->setCallingConv(call.getCallingConv());

        auto dst_arg = Builder2.CreateBitCast(
            args[0], Type::getInt8PtrTy(call.getContext()));
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto len_arg = Builder2.CreateZExtOrTrunc(
            args[1], Type::getInt64Ty(call.getContext()));
        auto tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());

        Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
        Type *tys[] = {dst_arg->getType(), len_arg->getType()};

        auto MemsetDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            nargs));
        memset->addParamAttr(0, Attribute::NonNull);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
    //           MPI_Comm comm )
    // 1. if root, malloc intermediate buffer
    // 2. reduce sum diff(buffer) into intermediate
    // 3. if root, set shadow(buffer) = intermediate [memcpy] then free
    // 3-e. else, set shadow(buffer) = 0 [memset]
    if (funcName == "MPI_Bcast") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
        if (!forwardMode)
          shadow = lookup(shadow, Builder2);
        if (shadow->getType()->isIntegerTy())
          shadow = Builder2.CreateIntToPtr(
              shadow, Type::getInt8PtrTy(call.getContext()));

        ConcreteType CT = TR.firstPointer(1, call.getOperand(0), &call);
        auto MPI_OP_type = Type::getInt8PtrTy(call.getContext());
        Type *MPI_OP_Ptr_type = PointerType::getUnqual(MPI_OP_type);

        Value *count = gutils->getNewFromOriginal(call.getOperand(1));
        if (!forwardMode)
          count = lookup(count, Builder2);
        Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);
        Value *root = gutils->getNewFromOriginal(call.getOperand(3));
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(call.getOperand(4));
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*buffer*/ shadow,
              /*count*/ count,
              /*datatype*/ datatype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. if root, malloc intermediate buffer, else undef
        PHINode *buf;

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          Value *rootbuf =
              CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                               len_arg, "mpireduce_malloccache");
          Builder2.CreateBr(mergeBlock);

          Builder2.SetInsertPoint(mergeBlock);

          buf = Builder2.CreatePHI(rootbuf->getType(), 2);
          buf->addIncoming(rootbuf, rootBlock);
          buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);
        }

        // Need to preserve the shadow buffer.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 2. reduce sum diff(buffer) into intermediate
        {
          // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
          // MPI_Datatype datatype,
          //     MPI_Op op, int root, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow,
              /*recvbuf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op (MPI_SUM)*/
              getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
                                    MPI_OP_Ptr_type, MPI_OP_type, CT,
                                    root->getType(), Builder2),
              /*int root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);

          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Reduce", FT), args,
              BufferDefs);
        }

        // 3. if root, set shadow(buffer) = intermediate [memcpy]
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *nonrootBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_nonroot", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            nonrootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              nonrootBlock);

        Builder2.SetInsertPoint(rootBlock);

        {
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *nargs[] = {shadow, buf, len_arg, volatile_arg};

          Type *tys[] = {shadow->getType(), buf->getType(), len_arg->getType()};

          auto memcpyF = Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                                   Intrinsic::memcpy, tys);

          auto mem =
              cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
          mem->setCallingConv(memcpyF->getCallingConv());

          // Free up the memory of the buffer
          if (shouldFree()) {
            CreateDealloc(Builder2, buf);
          }
        }

        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(nonrootBlock);

        // 3-e. else, set shadow(buffer) = 0 [memset]
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 1.5 if root, set intermediate = diff(recvbuffer)
    // 2. MPI_Bcast intermediate to all
    // 3. if root, Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
    // MPI_Datatype datatype,
    //                      MPI_Op op, int root, MPI_Comm comm)

    if (funcName == "MPI_Reduce" || funcName == "PMPI_Reduce") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        // TODO insert a check for sum

        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        // Get the operations from MPI_Receive
        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_recvbuf = call.getOperand(1);
        Value *orig_count = call.getOperand(2);
        Value *orig_datatype = call.getOperand(3);
        Value *orig_op = call.getOperand(4);
        Value *orig_root = call.getOperand(5);
        Value *orig_comm = call.getOperand(6);

        bool isSum = false;
        if (Constant *C = dyn_cast<Constant>(orig_op)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_mpi_op_sum") {
              isSum = true;
            }
          }
          // MPICH
          if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
            if (CI->getValue() == 1476395011) {
              isSum = true;
            }
          }
        }
        if (!isSum) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << *gutils->oldFunc << "\n";
          ss << *gutils->newFunc << "\n";
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&call),
                               ErrorType::NoDerivative, gutils, nullptr,
                               wrap(&BuilderZ));
          } else {
            EmitFailure("NoDerivative", call.getDebugLoc(), &call, ss.str());
            return;
          }
        }

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ true);

        Value *count = gutils->getNewFromOriginal(orig_count);
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(orig_datatype);
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *op = lookup(gutils->getNewFromOriginal(orig_op), Builder2);
        if (!forwardMode)
          op = lookup(op, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*recvbuf*/ shadow_recvbuf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
               ValueType::Primal, ValueType::Primal, ValueType::Primal,
               ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. Alloc intermediate buffer
        Value *buf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             len_arg, "mpireduce_malloccache");

        // 1.5 if root, set intermediate = diff(recvbuffer)
        {

          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          {
            auto volatile_arg = ConstantInt::getFalse(call.getContext());
            Value *nargs[] = {buf, shadow_recvbuf, len_arg, volatile_arg};

            Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                           len_arg->getType()};

            auto memcpyF = Intrinsic::getDeclaration(
                gutils->newFunc->getParent(), Intrinsic::memcpy, tys);

            auto mem =
                cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
            mem->setCallingConv(memcpyF->getCallingConv());
          }

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 2. MPI_Bcast intermediate to all
        {
          // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int
          // root,
          //     MPI_Comm comm )
          Value *args[] = {
              /*buf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*int root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Bcast", FT), args,
              BufferDefs);
        }

        // 3. if root, Zero diff(recvbuffer) [memset to 0]
        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffers
    // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
    //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

    if (funcName == "MPI_Allreduce") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        // TODO insert a check for sum

        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        // Get the operations from MPI_Receive
        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_recvbuf = call.getOperand(1);
        Value *orig_count = call.getOperand(2);
        Value *orig_datatype = call.getOperand(3);
        Value *orig_op = call.getOperand(4);
        Value *orig_comm = call.getOperand(5);

        bool isSum = false;
        if (Constant *C = dyn_cast<Constant>(orig_op)) {
          while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
            C = CE->getOperand(0);
          }
          if (auto GV = dyn_cast<GlobalVariable>(C)) {
            if (GV->getName() == "ompi_mpi_op_sum") {
              isSum = true;
            }
          }
          // MPICH
          if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
            if (CI->getValue() == 1476395011) {
              isSum = true;
            }
          }
        }
        if (!isSum) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << *gutils->oldFunc << "\n";
          ss << *gutils->newFunc << "\n";
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&call),
                               ErrorType::NoDerivative, gutils, nullptr,
                               wrap(&BuilderZ));
          } else {
            EmitFailure("NoDerivative", call.getDebugLoc(), &call, ss.str());
            return;
          }
        }

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Shadow, ValueType::Primal,
             ValueType::Primal, ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ !forwardMode);

        Value *count = gutils->getNewFromOriginal(orig_count);
        if (!forwardMode)
          count = lookup(count, Builder2);

        Value *datatype = gutils->getNewFromOriginal(orig_datatype);
        if (!forwardMode)
          datatype = lookup(datatype, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *op = gutils->getNewFromOriginal(orig_op);
        if (!forwardMode)
          op = lookup(op, Builder2);

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*recvbuf*/ shadow_recvbuf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*comm*/ comm,
          };

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args,
                              BufferDefs);

          return;
        }

        Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType());

        // Get the length for the allocation of the intermediate buffer
        auto len_arg = Builder2.CreateZExtOrTrunc(
            count, Type::getInt64Ty(call.getContext()));
        len_arg =
            Builder2.CreateMul(len_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // 1. Alloc intermediate buffer
        Value *buf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             len_arg, "mpireduce_malloccache");

        // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
        {
          // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
          //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*recvbuf*/ buf,
              /*count*/ count,
              /*datatype*/ datatype,
              /*op*/ op,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Allreduce", FT),
              args, BufferDefs);
        }

        // 3. Zero diff(recvbuffer) [memset to 0]
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                      Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    len_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 2. Scatter diff(recvbuffer) to intermediate buffer
    // 3. if root, Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //           int root, MPI_Comm comm)

    if (funcName == "MPI_Gather") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!sendcount)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }

        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. Alloc intermediate buffer
        Value *buf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             sendlen_arg, "mpireduce_malloccache");

        // 2. Scatter diff(recvbuffer) to intermediate buffer
        {
          // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
          // sendtype,
          //     void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
          //     MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*sendcount*/ recvcount,
              /*sendtype*/ recvtype,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*op*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Scatter", FT), args,
              BufferDefs);
        }

        // 3. if root, Zero diff(recvbuffer) [memset to 0]
        {

          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);
          auto recvlen_arg = Builder2.CreateZExtOrTrunc(
              recvcount, Type::getInt64Ty(call.getContext()));
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, root->getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);

          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. if root, malloc intermediate buffer, else undef
    // 2. Gather diff(recvbuffer) to intermediate buffer
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. if root, free intermediate buffer

    // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
    // sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
    //           MPI_Comm comm)
    if (funcName == "MPI_Scatter") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_root = call.getOperand(6);
        Value *orig_comm = call.getOperand(7);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!forwardMode)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *root = gutils->getNewFromOriginal(orig_root);
        if (!forwardMode)
          root = lookup(root, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType());
        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*root*/ root,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal, ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }
        // Get the length for the allocation of the intermediate buffer
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcount, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal, ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. if root, malloc intermediate buffer, else undef
        PHINode *buf;
        PHINode *sendlen_phi;

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          auto sendlen_arg = Builder2.CreateZExtOrTrunc(
              sendcount, Type::getInt64Ty(call.getContext()));
          sendlen_arg = Builder2.CreateMul(
              sendlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          sendlen_arg = Builder2.CreateMul(
              sendlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, root->getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);

          Value *rootbuf =
              CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                               sendlen_arg, "mpireduce_malloccache");

          Builder2.CreateBr(mergeBlock);

          Builder2.SetInsertPoint(mergeBlock);

          buf = Builder2.CreatePHI(rootbuf->getType(), 2);
          buf->addIncoming(rootbuf, rootBlock);
          buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);

          sendlen_phi = Builder2.CreatePHI(sendlen_arg->getType(), 2);
          sendlen_phi->addIncoming(sendlen_arg, rootBlock);
          sendlen_phi->addIncoming(UndefValue::get(sendlen_arg->getType()),
                                   currentBlock);
        }

        // 2. Gather diff(recvbuffer) to intermediate buffer
        {
          // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype
          // sendtype,
          //     void *recvbuf, int recvcount, MPI_Datatype recvtype,
          //     int root, MPI_Comm comm)
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*sendcount*/ recvcount,
              /*sendtype*/ recvtype,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*root*/ root,
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(
              called->getParent()->getOrInsertFunction("MPI_Gather", FT), args,
              BufferDefs);
        }

        // 3. Zero diff(recvbuffer) [memset to 0]
        {
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        }

        // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
        // 5. if root, free intermediate buffer

        {
          BasicBlock *currentBlock = Builder2.GetInsertBlock();
          BasicBlock *rootBlock = gutils->addReverseBlock(
              currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
          BasicBlock *mergeBlock = gutils->addReverseBlock(
              rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

          Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                                mergeBlock);

          Builder2.SetInsertPoint(rootBlock);

          // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
          DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                      sendlen_phi, Builder2, BufferDefs);

          // Free up intermediate buffer
          if (shouldFree()) {
            CreateDealloc(Builder2, buf);
          }

          Builder2.CreateBr(mergeBlock);
          Builder2.SetInsertPoint(mergeBlock);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Approximate algo (for sum):  -> if statement yet to be
    // 1. malloc intermediate buffer
    // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
    // intermediate buffer
    // 3. Zero diff(recvbuffer) [memset to 0]
    // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
    // 5. free intermediate buffer

    // int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype
    // sendtype,
    //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
    //           MPI_Comm comm)

    if (funcName == "MPI_Allgather") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode) {
        bool forwardMode = Mode == DerivativeMode::ForwardMode;

        IRBuilder<> Builder2 =
            forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
        if (forwardMode) {
          getForwardBuilder(Builder2);
        } else {
          getReverseBuilder(Builder2);
        }

        Value *orig_sendbuf = call.getOperand(0);
        Value *orig_sendcount = call.getOperand(1);
        Value *orig_sendtype = call.getOperand(2);
        Value *orig_recvbuf = call.getOperand(3);
        Value *orig_recvcount = call.getOperand(4);
        Value *orig_recvtype = call.getOperand(5);
        Value *orig_comm = call.getOperand(6);

        Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
        if (!forwardMode)
          shadow_recvbuf = lookup(shadow_recvbuf, Builder2);

        if (shadow_recvbuf->getType()->isIntegerTy())
          shadow_recvbuf = Builder2.CreateIntToPtr(
              shadow_recvbuf, Type::getInt8PtrTy(call.getContext()));

        Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
        if (!forwardMode)
          shadow_sendbuf = lookup(shadow_sendbuf, Builder2);

        if (shadow_sendbuf->getType()->isIntegerTy())
          shadow_sendbuf = Builder2.CreateIntToPtr(
              shadow_sendbuf, Type::getInt8PtrTy(call.getContext()));

        Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
        if (!forwardMode)
          recvcount = lookup(recvcount, Builder2);

        Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
        if (!forwardMode)
          recvtype = lookup(recvtype, Builder2);

        Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
        if (!forwardMode)
          sendcount = lookup(sendcount, Builder2);

        Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
        if (!forwardMode)
          sendtype = lookup(sendtype, Builder2);

        Value *comm = gutils->getNewFromOriginal(orig_comm);
        if (!forwardMode)
          comm = lookup(comm, Builder2);

        Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType());

        if (forwardMode) {
          Value *args[] = {
              /*sendbuf*/ shadow_sendbuf,
              /*sendcount*/ sendcount,
              /*sendtype*/ sendtype,
              /*recvbuf*/ shadow_recvbuf,
              /*recvcount*/ recvcount,
              /*recvtype*/ recvtype,
              /*comm*/ comm,
          };

          auto Defs = gutils->getInvertedBundles(
              &call,
              {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Shadow, ValueType::Primal, ValueType::Primal,
               ValueType::Primal},
              Builder2, /*lookup*/ false);

#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Builder2.CreateCall(call.getFunctionType(), callval, args, Defs);
          return;
        }
        // Get the length for the allocation of the intermediate buffer
        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcount, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);

        // Need to preserve the shadow send/recv buffers.
        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Shadow, ValueType::Primal, ValueType::Primal,
             ValueType::Primal},
            Builder2, /*lookup*/ true);

        // 1. Alloc intermediate buffer
        Value *buf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             sendlen_arg, "mpireduce_malloccache");

        ConcreteType CT = TR.firstPointer(1, orig_sendbuf, &call);
        auto MPI_OP_type = Type::getInt8PtrTy(call.getContext());
        Type *MPI_OP_Ptr_type = PointerType::getUnqual(MPI_OP_type);

        // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
        // intermediate buffer
        {
          // int MPI_Reduce_scatter_block(const void* send_buffer,
          //                    void* receive_buffer,
          //                    int count,
          //                    MPI_Datatype datatype,
          //                    MPI_Op operation,
          //                    MPI_Comm communicator);
          Value *args[] = {
              /*sendbuf*/ shadow_recvbuf,
              /*recvbuf*/ buf,
              /*recvcount*/ sendcount,
              /*recvtype*/ sendtype,
              /*op (MPI_SUM)*/
              getOrInsertOpFloatSum(*gutils->newFunc->getParent(),
                                    MPI_OP_Ptr_type, MPI_OP_type, CT,
                                    call.getType(), Builder2),
              /*comm*/ comm,
          };
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();

          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          Builder2.CreateCall(called->getParent()->getOrInsertFunction(
                                  "MPI_Reduce_scatter_block", FT),
                              args, BufferDefs);
        }

        // 3. zero diff(recvbuffer) [memset to 0]
        {
          auto recvlen_arg = Builder2.CreateZExtOrTrunc(
              recvcount, Type::getInt64Ty(call.getContext()));
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);
          recvlen_arg = Builder2.CreateMul(
              recvlen_arg,
              Builder2.CreateZExtOrTrunc(
                  MPI_COMM_SIZE(comm, Builder2, call.getType()),
                  Type::getInt64Ty(call.getContext())),
              "", true, true);
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
          Type *tys[] = {args[0]->getType(), args[2]->getType()};
          auto memset = cast<CallInst>(Builder2.CreateCall(
              Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                        Intrinsic::memset, tys),
              args, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        }

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_arg, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Adjoint of barrier is to place a barrier at the corresponding
    // location in the reverse.
    if (funcName == "MPI_Barrier") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
        Builder2.CreateCall(call.getFunctionType(), callval, args);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Remove free's in forward pass so the comm can be used in the reverse
    // pass
    if (funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect") {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Adjoint of MPI_Comm_split / MPI_Graph_create (which allocates a comm in a
    // pointer) is to free the created comm at the corresponding place in the
    // reverse pass
    auto commFound = MPIInactiveCommAllocators.find(funcName);
    if (commFound != MPIInactiveCommAllocators.end()) {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        Value *args[] = {lookup(call.getOperand(commFound->second), Builder2)};
        Type *types[] = {args[0]->getType()};

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction("MPI_Comm_free", FT),
            args);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    llvm::errs() << *gutils->oldFunc->getParent() << "\n";
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << call << "\n";
    llvm::errs() << called << "\n";
    llvm_unreachable("Unhandled MPI FUNCTION");
  }

  void recursivelyHandleSubfunction(llvm::CallInst &call,
                                    llvm::Function *called,
                                    const std::vector<bool> &overwritten_args,
                                    bool shadowReturnUsed,
                                    DIFFE_TYPE subretType, bool subretused) {
    using namespace llvm;

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    CallInst *newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));

    bool foreignFunction = called == nullptr;

    FnTypeInfo nextTypeInfo(called);

    if (called) {
      nextTypeInfo = TR.getCallInfo(call, *called);
    }

    const AugmentedReturn *subdata = nullptr;
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ForwardModeSplit) {
      assert(augmentedReturn);
      if (augmentedReturn) {
        auto fd = augmentedReturn->subaugmentations.find(&call);
        if (fd != augmentedReturn->subaugmentations.end()) {
          subdata = fd->second;
        }
      }
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      SmallVector<Value *, 8> args;
      std::vector<DIFFE_TYPE> argsInverted;
      std::map<int, Type *> gradByVal;
      std::map<int, std::vector<Attribute>> structAttrs;

#if LLVM_VERSION_MAJOR >= 14
      for (unsigned i = 0; i < call.arg_size(); ++i)
#else
      for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
      {

        if (call.paramHasAttr(i, Attribute::StructRet)) {
          structAttrs[args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
              Attribute::get(call.getContext(), "enzyme_sret")
#else
              Attribute::get(call.getContext(), "enzyme_sret")
#endif
          );
#if LLVM_VERSION_MAJOR >= 13
          // TODO
          // structAttrs[args.size()].push_back(Attribute::get(
          //     call.getContext(), Attribute::AttrKind::ElementType,
          //     call.getParamAttr(i, Attribute::StructRet).getValueAsType()));
#endif
        }
        if (call.getAttributes().hasParamAttr(i, "enzymejl_returnRoots")) {
          structAttrs[args.size()].push_back(
              call.getParamAttr(i, "enzymejl_returnRoots"));
        }
        for (auto ty : PrimalParamAttrsToPreserve)
          if (call.getAttributes().hasParamAttr(i, ty)) {
            auto attr = call.getAttributes().getParamAttr(i, ty);
            structAttrs[args.size()].push_back(attr);
          }

        auto argi = gutils->getNewFromOriginal(call.getArgOperand(i));

        if (call.isByValArgument(i)) {
          gradByVal[args.size()] = call.getParamByValType(i);
        }

        bool writeOnlyNoCapture = true;
        bool readOnly = true;
        if (!isNoCapture(&call, i)) {
          writeOnlyNoCapture = false;
        }
        if (!isWriteOnly(&call, i)) {
          writeOnlyNoCapture = false;
        }
        if (!isReadOnly(&call, i)) {
          readOnly = false;
        }

        if (shouldDisableNoWrite(&call))
          writeOnlyNoCapture = false;

        auto argTy =
            gutils->getDiffeType(call.getArgOperand(i), foreignFunction);

        bool replace =
            (argTy == DIFFE_TYPE::DUP_NONEED &&
             (writeOnlyNoCapture ||
              !isa<Argument>(getBaseObject(call.getArgOperand(i))))) ||
            (writeOnlyNoCapture && Mode == DerivativeMode::ForwardModeSplit) ||
            (writeOnlyNoCapture && readOnly);

        if (replace) {
          argi = getUndefinedValueForType(argi->getType());
        }
        argsInverted.push_back(argTy);
        args.push_back(argi);

        if (argTy == DIFFE_TYPE::CONSTANT) {
          continue;
        }

        if (gutils->getWidth() == 1)
          for (auto ty : ShadowParamAttrsToPreserve)
            if (call.getAttributes().hasParamAttr(i, ty)) {
              auto attr = call.getAttributes().getParamAttr(i, ty);
              structAttrs[args.size()].push_back(attr);
            }

        if (call.getAttributes().hasParamAttr(i, "enzymejl_returnRoots")) {
          if (gutils->getWidth() == 1) {
            structAttrs[args.size()].push_back(
                call.getParamAttr(i, "enzymejl_returnRoots"));
          } else {
            structAttrs[args.size()].push_back(
                Attribute::get(call.getContext(), "enzyme_sret_v"));
          }
        }
        if (call.paramHasAttr(i, Attribute::StructRet)) {
          if (gutils->getWidth() == 1) {
            structAttrs[args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
                Attribute::get(call.getContext(), "enzyme_sret")
            // orig->getParamAttr(i, Attribute::StructRet).getValueAsType());
#else
                Attribute::get(call.getContext(), "enzyme_sret")
#endif
            );
#if LLVM_VERSION_MAJOR >= 13
            // TODO
            // structAttrs[args.size()].push_back(Attribute::get(
            //     call.getContext(), Attribute::AttrKind::ElementType,
            //     call.getParamAttr(i,
            //     Attribute::StructRet).getValueAsType()));
#endif
          } else {
            structAttrs[args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
                Attribute::get(call.getContext(), "enzyme_sret")
#else
                Attribute::get(call.getContext(), "enzyme_sret_v")
#endif
            );
#if LLVM_VERSION_MAJOR >= 13
            // TODO
            // structAttrs[args.size()].push_back(Attribute::get(
            //     call.getContext(), Attribute::AttrKind::ElementType,
            //     call.getParamAttr(i,
            //     Attribute::StructRet).getValueAsType()));
#endif
          }
        }

        assert(argTy == DIFFE_TYPE::DUP_ARG || argTy == DIFFE_TYPE::DUP_NONEED);

        args.push_back(gutils->invertPointerM(call.getArgOperand(i), Builder2));
      }
#if LLVM_VERSION_MAJOR >= 16
      std::optional<int> tapeIdx;
#else
      Optional<int> tapeIdx;
#endif
      if (subdata) {
        auto found = subdata->returns.find(AugmentedStruct::Tape);
        if (found != subdata->returns.end()) {
          tapeIdx = found->second;
        }
      }
      Value *tape = nullptr;
#if LLVM_VERSION_MAJOR >= 16
      if (tapeIdx.has_value())
#else
      if (tapeIdx.hasValue())
#endif
      {

#if LLVM_VERSION_MAJOR >= 16
        auto idx = tapeIdx.value();
#else
        auto idx = tapeIdx.getValue();
#endif
        FunctionType *FT = subdata->fn->getFunctionType();

        tape = BuilderZ.CreatePHI(
            (tapeIdx == -1)
                ? FT->getReturnType()
                : cast<StructType>(FT->getReturnType())->getElementType(idx),
            1, "tapeArg");

        assert(!tape->getType()->isEmptyTy());
        gutils->TapesToPreventRecomputation.insert(cast<Instruction>(tape));
        tape = gutils->cacheForReverse(BuilderZ, tape,
                                       getIndex(&call, CacheType::Tape));
        args.push_back(tape);
      }

      Value *newcalled = nullptr;
      FunctionType *FT = nullptr;

      if (called) {
        newcalled = gutils->Logic.CreateForwardDiff(
            cast<Function>(called), subretType, argsInverted,
            TR.analyzer.interprocedural, /*returnValue*/ subretused, Mode,
            ((DiffeGradientUtils *)gutils)->FreeMemory, gutils->getWidth(),
            tape ? tape->getType() : nullptr, nextTypeInfo, overwritten_args,
            /*augmented*/ subdata);
        FT = cast<Function>(newcalled)->getFunctionType();
      } else {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        newcalled = gutils->invertPointerM(callval, BuilderZ);

        if (gutils->getWidth() > 1) {
          newcalled = BuilderZ.CreateExtractValue(newcalled, {0});
        }

        ErrorIfRuntimeInactive(
            BuilderZ, gutils->getNewFromOriginal(callval), newcalled,
            "Attempting to call an indirect active function "
            "whose runtime value is inactive",
            gutils->getNewFromOriginal(call.getDebugLoc()), &call);

        auto ft = call.getFunctionType();
        bool retActive = subretType != DIFFE_TYPE::CONSTANT;

        ReturnType subretVal =
            subretused
                ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                : (retActive ? ReturnType::Return : ReturnType::Void);

        FT = getFunctionTypeForClone(
            ft, Mode, gutils->getWidth(), tape ? tape->getType() : nullptr,
            argsInverted, false, subretVal, subretType);
        PointerType *fptype = PointerType::getUnqual(FT);
        newcalled = BuilderZ.CreatePointerCast(newcalled,
                                               PointerType::getUnqual(fptype));
        newcalled = BuilderZ.CreateLoad(fptype, newcalled);
      }

      assert(newcalled);
      assert(FT);

      SmallVector<ValueType, 2> BundleTypes;
      for (auto A : argsInverted)
        if (A == DIFFE_TYPE::CONSTANT)
          BundleTypes.push_back(ValueType::Primal);
        else
          BundleTypes.push_back(ValueType::Both);

      auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                             /*lookup*/ false);

      CallInst *diffes = Builder2.CreateCall(FT, newcalled, args, Defs);
      diffes->setCallingConv(call.getCallingConv());
      diffes->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

      for (auto pair : gradByVal) {
        diffes->addParamAttr(
            pair.first,
            Attribute::getWithByValType(diffes->getContext(), pair.second));
      }

      for (auto &pair : structAttrs) {
        for (auto val : pair.second)
          diffes->addParamAttr(pair.first, val);
      }

      auto newcall = gutils->getNewFromOriginal(&call);
      auto ifound = gutils->invertedPointers.find(&call);
      Value *primal = nullptr;
      Value *diffe = nullptr;

      if (subretused && subretType != DIFFE_TYPE::CONSTANT) {
        primal = Builder2.CreateExtractValue(diffes, 0);
        diffe = Builder2.CreateExtractValue(diffes, 1);
      } else if (subretType != DIFFE_TYPE::CONSTANT) {
        diffe = diffes;
      } else if (!FT->getReturnType()->isVoidTy()) {
        primal = diffes;
      }

      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        if (primal) {
          gutils->replaceAWithB(newcall, primal);
          gutils->erase(newcall);
        } else {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
        if (diffe) {
          gutils->replaceAWithB(placeholder, diffe);
        } else {
          gutils->invertedPointers.erase(ifound);
        }
        gutils->erase(placeholder);
      } else {
        if (primal && diffe) {
          gutils->replaceAWithB(newcall, primal);
          if (!gutils->isConstantValue(&call)) {
            setDiffe(&call, diffe, Builder2);
          }
          gutils->erase(newcall);
        } else if (diffe) {
          setDiffe(&call, diffe, Builder2);
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        } else if (primal) {
          gutils->replaceAWithB(newcall, primal);
          gutils->erase(newcall);
        } else {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
      }

      return;
    }

    bool modifyPrimal = shouldAugmentCall(&call, gutils);

    SmallVector<Value *, 8> args;
    SmallVector<Value *, 8> pre_args;
    std::vector<DIFFE_TYPE> argsInverted;
    SmallVector<Instruction *, 4> postCreate;
    SmallVector<Instruction *, 4> userReplace;
    std::map<int, Type *> preByVal;
    std::map<int, Type *> gradByVal;
    std::map<int, std::vector<Attribute>> structAttrs;

    bool replaceFunction = false;

    if (Mode == DerivativeMode::ReverseModeCombined && !foreignFunction) {
      replaceFunction = legalCombinedForwardReverse(
          &call, *replacedReturns, postCreate, userReplace, gutils,
          unnecessaryInstructions, oldUnreachable, subretused);
      if (replaceFunction) {
        modifyPrimal = false;
      }
    }

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < call.arg_size(); ++i)
#else
    for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
    {

      auto argi = gutils->getNewFromOriginal(call.getArgOperand(i));

      if (call.isByValArgument(i)) {
        preByVal[pre_args.size()] = call.getParamByValType(i);
      }
      if (call.getAttributes().hasParamAttr(i, "enzymejl_returnRoots")) {
        structAttrs[pre_args.size()].push_back(
            call.getParamAttr(i, "enzymejl_returnRoots"));
      }
      if (call.paramHasAttr(i, Attribute::StructRet)) {
        structAttrs[pre_args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
            // TODO persist types
            Attribute::get(call.getContext(), "enzyme_sret")
        // Attribute::get(orig->getContext(), "enzyme_sret",
        // orig->getParamAttr(ii, Attribute::StructRet).getValueAsType());
#else
            // TODO persist types
            Attribute::get(call.getContext(), "enzyme_sret")
        // Attribute::get(orig->getContext(), "enzyme_sret");
#endif
        );
      }
      for (auto ty : PrimalParamAttrsToPreserve)
        if (call.getAttributes().hasParamAttr(i, ty)) {
          auto attr = call.getAttributes().getParamAttr(i, ty);
          structAttrs[pre_args.size()].push_back(attr);
        }

      auto argTy = gutils->getDiffeType(call.getArgOperand(i), foreignFunction);

      bool writeOnlyNoCapture = true;
      bool readNoneNoCapture = false;
      if (!isNoCapture(&call, i)) {
        writeOnlyNoCapture = false;
        readNoneNoCapture = false;
      }
      if (!isWriteOnly(&call, i)) {
        writeOnlyNoCapture = false;
      }
      if (!(isReadOnly(&call, i) && isWriteOnly(&call, i))) {
        readNoneNoCapture = false;
      }

      if (shouldDisableNoWrite(&call)) {
        writeOnlyNoCapture = false;
        readNoneNoCapture = false;
      }

      Value *prearg = argi;
      // Keep the existing passed value if coming from outside.
      if (readNoneNoCapture ||
          (argTy == DIFFE_TYPE::DUP_NONEED &&
           (writeOnlyNoCapture ||
            !isa<Argument>(getBaseObject(call.getArgOperand(i)))))) {
        prearg = getUndefinedValueForType(argi->getType());
      }
      pre_args.push_back(prearg);

      if (Mode != DerivativeMode::ReverseModePrimal) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        if (call.isByValArgument(i)) {
          gradByVal[args.size()] = call.getParamByValType(i);
        }

        if ((writeOnlyNoCapture && !replaceFunction) ||
            (readNoneNoCapture ||
             (argTy == DIFFE_TYPE::DUP_NONEED &&
              (writeOnlyNoCapture ||
               !isa<Argument>(getBaseObject(call.getOperand(i))))))) {
          argi = getUndefinedValueForType(argi->getType());
        }
        args.push_back(lookup(argi, Builder2));
      }

      argsInverted.push_back(argTy);

      if (argTy == DIFFE_TYPE::CONSTANT) {
        continue;
      }

      auto argType = argi->getType();

      if (argTy == DIFFE_TYPE::DUP_ARG || argTy == DIFFE_TYPE::DUP_NONEED) {
        if (gutils->getWidth() == 1)
          for (auto ty : ShadowParamAttrsToPreserve)
            if (call.getAttributes().hasParamAttr(i, ty)) {
              auto attr = call.getAttributes().getParamAttr(i, ty);
              structAttrs[pre_args.size()].push_back(attr);
            }

        if (call.getAttributes().hasParamAttr(i, "enzymejl_returnRoots")) {
          if (gutils->getWidth() == 1) {
            structAttrs[pre_args.size()].push_back(
                call.getParamAttr(i, "enzymejl_returnRoots"));
          } else {
            structAttrs[pre_args.size()].push_back(
                Attribute::get(call.getContext(), "enzymejl_returnRoots_v"));
          }
        }
        if (call.paramHasAttr(i, Attribute::StructRet)) {
          if (gutils->getWidth() == 1) {
            structAttrs[pre_args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
                // TODO persist types
                Attribute::get(call.getContext(), "enzyme_sret")
            // Attribute::get(orig->getContext(), "enzyme_sret",
            // orig->getParamAttr(ii, Attribute::StructRet).getValueAsType());
#else
                Attribute::get(call.getContext(), "enzyme_sret")
#endif
            );
          } else {
            structAttrs[pre_args.size()].push_back(
#if LLVM_VERSION_MAJOR >= 12
                // TODO persist types
                Attribute::get(call.getContext(), "enzyme_sret_v")
            // Attribute::get(orig->getContext(), "enzyme_sret_v",
            // gutils->getShadowType(orig->getParamAttr(ii,
            // Attribute::StructRet).getValueAsType()));
#else
                Attribute::get(call.getContext(), "enzyme_sret_v")
#endif
            );
          }
        }
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);

          Value *darg = nullptr;

          if (writeOnlyNoCapture && !replaceFunction &&
              TR.query(call.getArgOperand(i))[{-1, -1}] == BaseType::Pointer) {
            darg = getUndefinedValueForType(argi->getType());
          } else {
            darg = gutils->invertPointerM(call.getArgOperand(i), Builder2);
          }
          args.push_back(lookup(darg, Builder2));
        }
        pre_args.push_back(
            gutils->invertPointerM(call.getArgOperand(i), BuilderZ));

        // Note sometimes whattype mistakenly says something should be
        // constant [because composed of integer pointers alone]
        assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      } else {
        if (foreignFunction)
          assert(!argType->isIntOrIntVectorTy());
        assert(whatType(argType, Mode) == DIFFE_TYPE::OUT_DIFF ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      }
    }
    SmallVector<ValueType, 2> BundleTypes;
    for (auto A : argsInverted)
      if (A == DIFFE_TYPE::CONSTANT)
        BundleTypes.push_back(ValueType::Primal);
      else
        BundleTypes.push_back(ValueType::Both);
    if (called) {
#if LLVM_VERSION_MAJOR >= 14
      if (call.arg_size() !=
          cast<Function>(called)->getFunctionType()->getNumParams())
#else
      if (call.getNumArgOperands() !=
          cast<Function>(called)->getFunctionType()->getNumParams())
#endif
      {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << call << "\n";
        assert(0 && "number of arg operands != function parameters");
      }
      assert(argsInverted.size() ==
             cast<Function>(called)->getFunctionType()->getNumParams());
    }

    Value *tape = nullptr;
    CallInst *augmentcall = nullptr;
    Value *cachereplace = nullptr;

    // std::optional<std::map<std::pair<Instruction*, std::string>,
    // unsigned>> sub_index_map;
#if LLVM_VERSION_MAJOR >= 16
    std::optional<int> tapeIdx;
    std::optional<int> returnIdx;
    std::optional<int> differetIdx;
#else
    Optional<int> tapeIdx;
    Optional<int> returnIdx;
    Optional<int> differetIdx;
#endif
    if (modifyPrimal) {

      Value *newcalled = nullptr;
      FunctionType *FT = nullptr;
      const AugmentedReturn *fnandtapetype = nullptr;

      if (!called) {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        Value *uncast = callval;
        while (auto CE = dyn_cast<ConstantExpr>(uncast)) {
          if (CE->isCast()) {
            uncast = CE->getOperand(0);
            continue;
          }
          break;
        }
        if (isa<ConstantInt>(uncast)) {
          std::string str;
          raw_string_ostream ss(str);
          ss << "cannot find shadow for " << *callval
             << " for use as function in " << call;
          if (CustomErrorHandler) {
            CustomErrorHandler(ss.str().c_str(), wrap(&call),
                               ErrorType::NoDerivative, gutils, nullptr,
                               wrap(&BuilderZ));
          } else {
            EmitFailure("NoDerivative", call.getDebugLoc(), &call, ss.str());
            return;
          }
        }
        newcalled = gutils->invertPointerM(callval, BuilderZ);

        if (Mode != DerivativeMode::ReverseModeGradient)
          ErrorIfRuntimeInactive(
              BuilderZ, gutils->getNewFromOriginal(callval), newcalled,
              "Attempting to call an indirect active function "
              "whose runtime value is inactive",
              gutils->getNewFromOriginal(call.getDebugLoc()), &call);

        FunctionType *ft = call.getFunctionType();

        std::set<llvm::Type *> seen;
        DIFFE_TYPE subretType = whatType(call.getType(), Mode,
                                         /*intAreConstant*/ false, seen);
        auto res = getDefaultFunctionTypeForAugmentation(
            ft, /*returnUsed*/ true, /*subretType*/ subretType);
        FT = FunctionType::get(
            StructType::get(newcalled->getContext(), res.second), res.first,
            ft->isVarArg());
        auto fptype = PointerType::getUnqual(FT);
        newcalled = BuilderZ.CreatePointerCast(newcalled,
                                               PointerType::getUnqual(fptype));
        newcalled = BuilderZ.CreateLoad(fptype, newcalled);
        tapeIdx = 0;

        if (!call.getType()->isVoidTy()) {
          returnIdx = 1;
          if (subretType == DIFFE_TYPE::DUP_ARG ||
              subretType == DIFFE_TYPE::DUP_NONEED) {
            differetIdx = 2;
          }
        }
      } else {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          subdata = &gutils->Logic.CreateAugmentedPrimal(
              cast<Function>(called), subretType, argsInverted,
              TR.analyzer.interprocedural, /*return is used*/ subretused,
              shadowReturnUsed, nextTypeInfo, overwritten_args, false,
              gutils->getWidth(), gutils->AtomicAdd);
          if (Mode == DerivativeMode::ReverseModePrimal) {
            assert(augmentedReturn);
            auto subaugmentations =
                (std::map<const llvm::CallInst *, AugmentedReturn *>
                     *)&augmentedReturn->subaugmentations;
            insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
                *subaugmentations, &call, (AugmentedReturn *)subdata);
          }
        }
        if (!subdata) {
          llvm::errs() << *gutils->oldFunc->getParent() << "\n";
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << *called << "\n";
        }
        assert(subdata);
        fnandtapetype = subdata;
        newcalled = subdata->fn;
        FT = cast<Function>(newcalled)->getFunctionType();

        auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
        if (found != subdata->returns.end()) {
          differetIdx = found->second;
        } else {
          assert(!shadowReturnUsed);
        }

        found = subdata->returns.find(AugmentedStruct::Return);
        if (found != subdata->returns.end()) {
          returnIdx = found->second;
        } else {
          assert(!subretused);
        }

        found = subdata->returns.find(AugmentedStruct::Tape);
        if (found != subdata->returns.end()) {
          tapeIdx = found->second;
        }
      }
      // sub_index_map = fnandtapetype.tapeIndices;

      assert(newcalled);
      assert(FT);

      // llvm::errs() << "seeing sub_index_map of " << sub_index_map->size()
      // << " in ap " << cast<Function>(called)->getName() << "\n";
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModePrimal) {

        if (false) {
        badaugmentedfn:;
          auto NC = dyn_cast<Function>(newcalled);
          llvm::errs() << *gutils->oldFunc << "\n";
          llvm::errs() << *gutils->newFunc << "\n";
          if (NC)
            llvm::errs() << " trying to call " << NC->getName() << " " << *FT
                         << "\n";
          else
            llvm::errs() << " trying to call " << *newcalled << " " << *FT
                         << "\n";

          for (unsigned i = 0; i < pre_args.size(); ++i) {
            assert(pre_args[i]);
            assert(pre_args[i]->getType());
            llvm::errs() << "args[" << i << "] = " << *pre_args[i]
                         << " FT:" << *FT->getParamType(i) << "\n";
          }
          assert(0 && "calling with wrong number of arguments");
          exit(1);
        }

        if (pre_args.size() != FT->getNumParams())
          goto badaugmentedfn;

        for (unsigned i = 0; i < pre_args.size(); ++i) {
          if (pre_args[i]->getType() == FT->getParamType(i))
            continue;
          else if (!call.getCalledFunction())
            pre_args[i] =
                BuilderZ.CreateBitCast(pre_args[i], FT->getParamType(i));
          else
            goto badaugmentedfn;
        }

        augmentcall = BuilderZ.CreateCall(
            FT, newcalled, pre_args,
            gutils->getInvertedBundles(&call, BundleTypes, BuilderZ,
                                       /*lookup*/ false));
        augmentcall->setCallingConv(call.getCallingConv());
        augmentcall->setDebugLoc(
            gutils->getNewFromOriginal(call.getDebugLoc()));

        for (auto pair : preByVal) {
          augmentcall->addParamAttr(
              pair.first, Attribute::getWithByValType(augmentcall->getContext(),
                                                      pair.second));
        }

        for (auto &pair : structAttrs) {
          for (auto val : pair.second)
            augmentcall->addParamAttr(pair.first, val);
        }

        if (!augmentcall->getType()->isVoidTy())
          augmentcall->setName(call.getName() + "_augmented");

#if LLVM_VERSION_MAJOR >= 16
        if (tapeIdx.has_value())
#else
        if (tapeIdx.hasValue())
#endif
        {
#if LLVM_VERSION_MAJOR >= 16
          auto tval = tapeIdx.value();
#else
          auto tval = tapeIdx.getValue();
#endif
          tape = (tval == -1) ? augmentcall
                              : BuilderZ.CreateExtractValue(
                                    augmentcall, {(unsigned)tval}, "subcache");
          if (tape->getType()->isEmptyTy()) {
            auto tt = tape->getType();
            gutils->erase(cast<Instruction>(tape));
            tape = UndefValue::get(tt);
          } else {
            gutils->TapesToPreventRecomputation.insert(cast<Instruction>(tape));
          }
          tape = gutils->cacheForReverse(BuilderZ, tape,
                                         getIndex(&call, CacheType::Tape));
        }

        if (subretused) {
          Value *dcall = nullptr;
          assert(returnIdx);
          assert(augmentcall);
#if LLVM_VERSION_MAJOR >= 16
          auto rval = returnIdx.value();
#else
          auto rval = returnIdx.getValue();
#endif
          dcall = (rval < 0) ? augmentcall
                             : BuilderZ.CreateExtractValue(augmentcall,
                                                           {(unsigned)rval});
          gutils->originalToNewFn[&call] = dcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[dcall] = &call;

          assert(dcall->getType() == call.getType());
          assert(dcall);

          if (!gutils->isConstantValue(&call)) {
            if (!call.getType()->isFPOrFPVectorTy() &&
                TR.query(&call).Inner0().isPossiblePointer()) {
            } else if (Mode != DerivativeMode::ReverseModePrimal) {
              ((DiffeGradientUtils *)gutils)->differentials[dcall] =
                  ((DiffeGradientUtils *)gutils)->differentials[newCall];
              ((DiffeGradientUtils *)gutils)->differentials.erase(newCall);
            }
          }
          assert(dcall->getType() == call.getType());
          gutils->replaceAWithB(newCall, dcall);

          if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
            cast<Instruction>(dcall)->takeName(newCall);
          }

          if (Mode == DerivativeMode::ReverseModePrimal &&
              !gutils->unnecessaryIntermediates.count(&call)) {

            std::map<UsageKey, bool> Seen;
            bool primalNeededInReverse = false;
            for (auto pair : gutils->knownRecomputeHeuristic)
              if (!pair.second) {
                if (pair.first == &call) {
                  primalNeededInReverse = true;
                  break;
                } else {
                  Seen[UsageKey(pair.first, ValueType::Primal)] = false;
                }
              }
            if (!primalNeededInReverse) {

              auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                    ? DerivativeMode::ReverseModeGradient
                                    : Mode;
              primalNeededInReverse =
                  DifferentialUseAnalysis::is_value_needed_in_reverse<
                      ValueType::Primal>(gutils, &call, minCutMode, Seen,
                                         oldUnreachable);
            }
            if (primalNeededInReverse)
              gutils->cacheForReverse(BuilderZ, dcall,
                                      getIndex(&call, CacheType::Self));
          }
          BuilderZ.SetInsertPoint(newCall->getNextNode());
          gutils->erase(newCall);
        } else {
          BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          gutils->originalToNewFn[&call] = augmentcall;
          gutils->newToOriginalFn[augmentcall] = &call;
        }

      } else {
        if (subdata && subdata->returns.find(AugmentedStruct::Tape) ==
                           subdata->returns.end()) {
        } else {
          // assert(!tape);
          // assert(subdata);
          if (!tape) {
#if LLVM_VERSION_MAJOR >= 16
            assert(tapeIdx.has_value());
            auto tval = tapeIdx.value();
#else
            assert(tapeIdx.hasValue());
            auto tval = tapeIdx.getValue();
#endif
            tape = BuilderZ.CreatePHI(
                (tapeIdx == -1) ? FT->getReturnType()
                                : cast<StructType>(FT->getReturnType())
                                      ->getElementType(tval),
                1, "tapeArg");
          }
          tape = gutils->cacheForReverse(BuilderZ, tape,
                                         getIndex(&call, CacheType::Tape));
        }

        if (subretused) {
          if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                  ValueType::Primal>(gutils, &call, Mode, oldUnreachable) &&
              !gutils->unnecessaryIntermediates.count(&call)) {
            cachereplace = BuilderZ.CreatePHI(call.getType(), 1,
                                              call.getName() + "_tmpcacheB");
            cachereplace = gutils->cacheForReverse(
                BuilderZ, cachereplace, getIndex(&call, CacheType::Self));
          } else {
            auto pn = BuilderZ.CreatePHI(
                call.getType(), 1, (call.getName() + "_replacementE").str());
            gutils->fictiousPHIs[pn] = &call;
            cachereplace = pn;
          }
        } else {
          // TODO move right after newCall for the insertion point of BuilderZ

          BuilderZ.SetInsertPoint(BuilderZ.GetInsertPoint()->getNextNode());
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
      }

      auto ifound = gutils->invertedPointers.find(&call);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);

        bool subcheck = (subretType == DIFFE_TYPE::DUP_ARG ||
                         subretType == DIFFE_TYPE::DUP_NONEED);

        //! We only need the shadow pointer for non-forward Mode if it is used
        //! in a non return setting
        bool hasNonReturnUse = false;
        for (auto use : call.users()) {
          if (Mode == DerivativeMode::ReverseModePrimal ||
              !isa<ReturnInst>(
                  use)) { // || returnuses.find(cast<Instruction>(use)) ==
                          // returnuses.end()) {
            hasNonReturnUse = true;
          }
        }

        if (subcheck && hasNonReturnUse) {

          Value *newip = nullptr;
          if (Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ReverseModePrimal) {

#if LLVM_VERSION_MAJOR >= 16
            auto drval = differetIdx.value();
#else
            auto drval = differetIdx.getValue();
#endif
            newip = (drval < 0)
                        ? augmentcall
                        : BuilderZ.CreateExtractValue(augmentcall,
                                                      {(unsigned)drval},
                                                      call.getName() + "'ac");
            assert(newip->getType() == call.getType());
            placeholder->replaceAllUsesWith(newip);
            if (placeholder == &*BuilderZ.GetInsertPoint()) {
              BuilderZ.SetInsertPoint(placeholder->getNextNode());
            }
            gutils->erase(placeholder);
          } else {
            newip = placeholder;
          }

          newip = gutils->cacheForReverse(BuilderZ, newip,
                                          getIndex(&call, CacheType::Shadow));

          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&call, InvertedPointerVH(gutils, newip)));
        } else {
          gutils->invertedPointers.erase(ifound);
          if (placeholder == &*BuilderZ.GetInsertPoint()) {
            BuilderZ.SetInsertPoint(placeholder->getNextNode());
          }
          gutils->erase(placeholder);
        }
      }

      if (fnandtapetype && fnandtapetype->tapeType &&
          (Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit) &&
          shouldFree()) {
        assert(tape);
        auto tapep = BuilderZ.CreatePointerCast(
            tape, PointerType::get(
                      fnandtapetype->tapeType,
                      cast<PointerType>(tape->getType())->getAddressSpace()));
        auto truetape =
            BuilderZ.CreateLoad(fnandtapetype->tapeType, tapep, "tapeld");
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        CreateDealloc(BuilderZ, tape);
        tape = truetape;
      }
    } else {
      auto ifound = gutils->invertedPointers.find(&call);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
      }
      if (/*!topLevel*/ Mode != DerivativeMode::ReverseModeCombined &&
          subretused && !call.doesNotAccessMemory()) {
        if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                ValueType::Primal>(gutils, &call, Mode, oldUnreachable) &&
            !gutils->unnecessaryIntermediates.count(&call)) {
          assert(!replaceFunction);
          cachereplace = BuilderZ.CreatePHI(call.getType(), 1,
                                            call.getName() + "_cachereplace2");
          cachereplace = gutils->cacheForReverse(
              BuilderZ, cachereplace, getIndex(&call, CacheType::Self));
        } else {
          auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                       call.getName() + "_replacementC");
          gutils->fictiousPHIs[pn] = &call;
          cachereplace = pn;
        }
      }

      if (!subretused && !replaceFunction)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    }

    // Note here down only contains the reverse bits
    if (Mode == DerivativeMode::ReverseModePrimal) {
      return;
    }

    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);

    Value *newcalled = nullptr;
    FunctionType *FT = nullptr;

    DerivativeMode subMode = (replaceFunction || !modifyPrimal)
                                 ? DerivativeMode::ReverseModeCombined
                                 : DerivativeMode::ReverseModeGradient;
    if (called) {
      if (Mode == DerivativeMode::ReverseModeGradient && subdata) {
        for (size_t i = 0; i < argsInverted.size(); i++) {
          if (subdata->constant_args[i] == argsInverted[i])
            continue;
          assert(subdata->constant_args[i] == DIFFE_TYPE::DUP_ARG);
          assert(argsInverted[i] == DIFFE_TYPE::DUP_NONEED);
          argsInverted[i] = DIFFE_TYPE::DUP_ARG;
        }
      }

      newcalled = gutils->Logic.CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = cast<Function>(called),
                            .retType = subretType,
                            .constant_args = argsInverted,
                            .overwritten_args = overwritten_args,
                            .returnUsed = replaceFunction && subretused,
                            .shadowReturnUsed =
                                shadowReturnUsed && replaceFunction,
                            .mode = subMode,
                            .width = gutils->getWidth(),
                            .freeMemory = true,
                            .AtomicAdd = gutils->AtomicAdd,
                            .additionalType = tape ? tape->getType() : nullptr,
                            .forceAnonymousTape = false,
                            .typeInfo = nextTypeInfo},
          TR.analyzer.interprocedural, subdata);
      if (!newcalled)
        return;
      FT = cast<Function>(newcalled)->getFunctionType();
    } else {

      assert(subMode != DerivativeMode::ReverseModeCombined);

#if LLVM_VERSION_MAJOR >= 11
      auto callval = call.getCalledOperand();
#else
      auto callval = call.getCalledValue();
#endif

      if (gutils->isConstantValue(callval)) {
        llvm::errs() << *gutils->newFunc->getParent() << "\n";
        llvm::errs() << " orig: " << call << " callval: " << *callval << "\n";
      }
      assert(!gutils->isConstantValue(callval));
      newcalled = lookup(gutils->invertPointerM(callval, Builder2), Builder2);

      auto ft = call.getFunctionType();

      auto res =
          getDefaultFunctionTypeForGradient(ft, /*subretType*/ subretType);
      // TODO Note there is empty tape added here, replace with generic
      res.first.push_back(Type::getInt8PtrTy(newcalled->getContext()));
      FT = FunctionType::get(
          StructType::get(newcalled->getContext(), res.second), res.first,
          ft->isVarArg());
      auto fptype = PointerType::getUnqual(FT);
      newcalled =
          Builder2.CreatePointerCast(newcalled, PointerType::getUnqual(fptype));
      newcalled = Builder2.CreateLoad(
          fptype, Builder2.CreateConstGEP1_64(fptype, newcalled, 1));
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF) {
      args.push_back(diffe(&call, Builder2));
    }

    if (tape) {
      auto ntape = gutils->lookupM(tape, Builder2);
      assert(ntape);
      assert(ntape->getType());
      args.push_back(ntape);
    }

    assert(newcalled);
    assert(FT);

    if (false) {
    badfn:;
      auto NC = dyn_cast<Function>(newcalled);
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      if (NC)
        llvm::errs() << " trying to call " << NC->getName() << " " << *FT
                     << "\n";
      else
        llvm::errs() << " trying to call " << *newcalled << " " << *FT << "\n";

      for (unsigned i = 0; i < args.size(); ++i) {
        assert(args[i]);
        assert(args[i]->getType());
        llvm::errs() << "args[" << i << "] = " << *args[i]
                     << " FT:" << *FT->getParamType(i) << "\n";
      }
      assert(0 && "calling with wrong number of arguments");
      exit(1);
    }

    if (args.size() != FT->getNumParams())
      goto badfn;

    for (unsigned i = 0; i < args.size(); ++i) {
      if (args[i]->getType() == FT->getParamType(i))
        continue;
      else if (!call.getCalledFunction())
        args[i] = Builder2.CreateBitCast(args[i], FT->getParamType(i));
      else
        goto badfn;
    }

    CallInst *diffes =
        Builder2.CreateCall(FT, newcalled, args,
                            gutils->getInvertedBundles(
                                &call, BundleTypes, Builder2, /*lookup*/ true));
    diffes->setCallingConv(call.getCallingConv());
    diffes->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

    for (auto pair : gradByVal) {
      diffes->addParamAttr(pair.first, Attribute::getWithByValType(
                                           diffes->getContext(), pair.second));
    }

    for (auto &pair : structAttrs) {
      for (auto val : pair.second)
        diffes->addParamAttr(pair.first, val);
    }

    unsigned structidx = 0;
    if (replaceFunction) {
      if (subretused)
        structidx++;
      if (shadowReturnUsed)
        structidx++;
    }

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < call.arg_size(); ++i)
#else
    for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
    {
      if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
        Value *diffeadd = Builder2.CreateExtractValue(diffes, {structidx});
        ++structidx;

        if (!gutils->isConstantValue(call.getArgOperand(i))) {
          size_t size = 1;
          if (call.getArgOperand(i)->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(call.getArgOperand(i)->getType()) +
                    7) /
                   8;

          addToDiffe(call.getArgOperand(i), diffeadd, Builder2,
                     TR.addingType(size, call.getArgOperand(i)));
        }
      }
    }

    if (diffes->getType()->isVoidTy()) {
      if (structidx != 0) {
        llvm::errs() << *gutils->oldFunc->getParent() << "\n";
        llvm::errs() << "diffes: " << *diffes << " structidx=" << structidx
                     << " subretused=" << subretused
                     << " shadowReturnUsed=" << shadowReturnUsed << "\n";
      }
      assert(structidx == 0);
    } else {
      assert(cast<StructType>(diffes->getType())->getNumElements() ==
             structidx);
    }

    if (subretType == DIFFE_TYPE::OUT_DIFF)
      setDiffe(&call,
               Constant::getNullValue(gutils->getShadowType(call.getType())),
               Builder2);

    if (replaceFunction) {

      // if a function is replaced for joint forward/reverse, handle inverted
      // pointers
      auto ifound = gutils->invertedPointers.find(&call);
      if (ifound != gutils->invertedPointers.end()) {
        auto placeholder = cast<PHINode>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        if (shadowReturnUsed) {
          dumpMap(gutils->invertedPointers);
          auto dretval = cast<Instruction>(
              Builder2.CreateExtractValue(diffes, {subretused ? 1U : 0U}));
          /* todo handle this case later */
          assert(!subretused);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&call, InvertedPointerVH(gutils, dretval)));
        }
        gutils->erase(placeholder);
      }

      Instruction *retval = nullptr;

      if (subretused) {
        retval = cast<Instruction>(Builder2.CreateExtractValue(diffes, {0}));
        if (retval) {
          gutils->replaceAndRemoveUnwrapCacheFor(newCall, retval);
        }
        gutils->replaceAWithB(newCall, retval, /*storeInCache*/ true);
      } else {
        eraseIfUnused(call, /*erase*/ false, /*check*/ false);
      }

      for (auto a : postCreate) {
        a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
      }

      gutils->originalToNewFn[&call] = retval ? retval : diffes;
      gutils->newToOriginalFn.erase(newCall);
      gutils->newToOriginalFn[retval ? retval : diffes] = &call;

      gutils->erase(newCall);

      return;
    }

    if (cachereplace) {
      if (subretused) {
        Value *dcall = nullptr;
        assert(cachereplace->getType() == call.getType());
        assert(dcall == nullptr);
        dcall = cachereplace;
        assert(dcall);

        if (!gutils->isConstantValue(&call)) {
          gutils->originalToNewFn[&call] = dcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[dcall] = &call;
          if (!call.getType()->isFPOrFPVectorTy() &&
              TR.query(&call).Inner0().isPossiblePointer()) {
          } else {
            ((DiffeGradientUtils *)gutils)->differentials[dcall] =
                ((DiffeGradientUtils *)gutils)->differentials[newCall];
            ((DiffeGradientUtils *)gutils)->differentials.erase(newCall);
          }
        }
        assert(dcall->getType() == call.getType());
        newCall->replaceAllUsesWith(dcall);
        if (isa<Instruction>(dcall) && !isa<PHINode>(dcall)) {
          cast<Instruction>(dcall)->takeName(&call);
        }
        gutils->erase(newCall);
      } else {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        if (augmentcall) {
          gutils->originalToNewFn[&call] = augmentcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[augmentcall] = &call;
        }
      }
    }
    return;
  }

  // Return
  void visitCallInst(llvm::CallInst &call) {
    using namespace llvm;

    // When compiling Enzyme against standard LLVM, and not Intel's
    // modified version of LLVM, the intrinsic `llvm.intel.subscript` is
    // not fully understood by LLVM. One of the results of this is that the
    // visitor dispatches to visitCallInst, rather than visitIntrinsicInst, when
    // presented with the intrinsic - hence why we are handling it here.
    if (getFuncNameFromCall(&call).startswith("llvm.intel.subscript")) {
      assert(isa<IntrinsicInst>(call));
      visitIntrinsicInst(cast<IntrinsicInst>(call));
      return;
    }

    CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    IRBuilder<> BuilderZ(newCall);
    BuilderZ.setFastMathFlags(getFast());

    if (overwritten_args_map.find(&call) == overwritten_args_map.end() &&
        Mode != DerivativeMode::ForwardMode) {
      llvm::errs() << " call: " << call << "\n";
      for (auto &pair : overwritten_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(overwritten_args_map.find(&call) != overwritten_args_map.end() ||
           Mode == DerivativeMode::ForwardMode);
    const std::vector<bool> &overwritten_args =
        Mode == DerivativeMode::ForwardMode
            ? std::vector<bool>()
            : overwritten_args_map.find(&call)->second;

    auto called = getFunctionFromCall(&call);
    StringRef funcName = getFuncNameFromCall(&call);

    bool subretused = false;
    bool shadowReturnUsed = false;
    DIFFE_TYPE subretType =
        gutils->getReturnDiffeType(&call, &subretused, &shadowReturnUsed);

    if (Mode == DerivativeMode::ForwardMode) {
      auto found = customFwdCallHandlers.find(funcName);
      if (found != customFwdCallHandlers.end()) {
        Value *invertedReturn = nullptr;
        auto ifound = gutils->invertedPointers.find(&call);
        if (ifound != gutils->invertedPointers.end()) {
          invertedReturn = cast<PHINode>(&*ifound->second);
        }

        Value *normalReturn = subretused ? newCall : nullptr;

        bool noMod = found->second(BuilderZ, &call, *gutils, normalReturn,
                                   invertedReturn);
        if (noMod) {
          if (subretused)
            assert(normalReturn == newCall);
          eraseIfUnused(call);
        }

        if (ifound != gutils->invertedPointers.end()) {
          auto placeholder = cast<PHINode>(&*ifound->second);
          if (invertedReturn && invertedReturn != placeholder) {
            if (invertedReturn->getType() !=
                gutils->getShadowType(call.getType())) {
              llvm::errs() << " o: " << call << "\n";
              llvm::errs() << " ot: " << *call.getType() << "\n";
              llvm::errs() << " ir: " << *invertedReturn << "\n";
              llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
              llvm::errs() << " p: " << *placeholder << "\n";
              llvm::errs() << " PT: " << *placeholder->getType() << "\n";
              llvm::errs() << " newCall: " << *newCall << "\n";
              llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
            }
            assert(invertedReturn->getType() ==
                   gutils->getShadowType(call.getType()));
            placeholder->replaceAllUsesWith(invertedReturn);
            gutils->erase(placeholder);
            gutils->invertedPointers.insert(
                std::make_pair((const Value *)&call,
                               InvertedPointerVH(gutils, invertedReturn)));
          } else {
            gutils->invertedPointers.erase(&call);
            gutils->erase(placeholder);
          }
        }

        if (normalReturn && normalReturn != newCall) {
          assert(normalReturn->getType() == newCall->getType());
          gutils->replaceAWithB(newCall, normalReturn);
          gutils->erase(newCall);
        }
        return;
      }
    }

    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ReverseModeGradient) {
      auto found = customCallHandlers.find(funcName);
      if (found != customCallHandlers.end()) {
        IRBuilder<> Builder2(&call);
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined)
          getReverseBuilder(Builder2);

        Value *invertedReturn = nullptr;
        auto ifound = gutils->invertedPointers.find(&call);
        PHINode *placeholder = nullptr;
        if (ifound != gutils->invertedPointers.end()) {
          placeholder = cast<PHINode>(&*ifound->second);
          if (shadowReturnUsed)
            invertedReturn = placeholder;
        }

        Value *normalReturn = subretused ? newCall : nullptr;

        Value *tape = nullptr;

        Type *tapeType = nullptr;

        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined) {
          bool noMod = found->second.first(BuilderZ, &call, *gutils,
                                           normalReturn, invertedReturn, tape);
          if (noMod) {
            if (subretused)
              assert(normalReturn == newCall);
            eraseIfUnused(call);
          }
          if (tape) {
            tapeType = tape->getType();
            gutils->cacheForReverse(BuilderZ, tape,
                                    getIndex(&call, CacheType::Tape));
          }
          if (Mode == DerivativeMode::ReverseModePrimal) {
            assert(augmentedReturn);
            auto subaugmentations =
                (std::map<const llvm::CallInst *, AugmentedReturn *>
                     *)&augmentedReturn->subaugmentations;
            insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
                *subaugmentations, &call, (AugmentedReturn *)tapeType);
          }
        }

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          if (Mode == DerivativeMode::ReverseModeGradient &&
              augmentedReturn->tapeIndices.find(
                  std::make_pair(&call, CacheType::Tape)) !=
                  augmentedReturn->tapeIndices.end()) {
            assert(augmentedReturn);
            auto subaugmentations =
                (std::map<const llvm::CallInst *, AugmentedReturn *>
                     *)&augmentedReturn->subaugmentations;
            auto fd = subaugmentations->find(&call);
            assert(fd != subaugmentations->end());
            // Note we are using the storage space here to persist
            // the LLVM type, as storing a new augmentedReturn has issues
            // regarding persisting the data structure, and when it will
            // be freed, since it will no longer live in the map in
            // EnzymeLogic.
            tapeType = (llvm::Type *)fd->second;

            tape = BuilderZ.CreatePHI(tapeType, 0);
            tape = gutils->cacheForReverse(BuilderZ, tape,
                                           getIndex(&call, CacheType::Tape),
                                           /*ignoreType*/ true);
          }
          if (tape)
            tape = gutils->lookupM(tape, Builder2);
          found->second.second(Builder2, &call, *(DiffeGradientUtils *)gutils,
                               tape);
        }

        if (placeholder) {
          if (!shadowReturnUsed) {
            gutils->invertedPointers.erase(&call);
            gutils->erase(placeholder);
          } else {
            if (invertedReturn && invertedReturn != placeholder) {
              if (invertedReturn->getType() !=
                  gutils->getShadowType(call.getType())) {
                llvm::errs() << " o: " << call << "\n";
                llvm::errs() << " ot: " << *call.getType() << "\n";
                llvm::errs() << " ir: " << *invertedReturn << "\n";
                llvm::errs() << " irt: " << *invertedReturn->getType() << "\n";
                llvm::errs() << " p: " << *placeholder << "\n";
                llvm::errs() << " PT: " << *placeholder->getType() << "\n";
                llvm::errs() << " newCall: " << *newCall << "\n";
                llvm::errs() << " newCallT: " << *newCall->getType() << "\n";
              }
              assert(invertedReturn->getType() ==
                     gutils->getShadowType(call.getType()));
              placeholder->replaceAllUsesWith(invertedReturn);
              gutils->erase(placeholder);
            } else
              invertedReturn = placeholder;

            invertedReturn = gutils->cacheForReverse(
                BuilderZ, invertedReturn, getIndex(&call, CacheType::Shadow));

            gutils->invertedPointers.insert(
                std::make_pair((const Value *)&call,
                               InvertedPointerVH(gutils, invertedReturn)));
          }
        }

        bool primalNeededInReverse;

        if (gutils->knownRecomputeHeuristic.count(&call)) {
          primalNeededInReverse = !gutils->knownRecomputeHeuristic[&call];
        } else {
          std::map<UsageKey, bool> Seen;
          for (auto pair : gutils->knownRecomputeHeuristic)
            if (!pair.second)
              Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  ValueType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
        }
        if (subretused && primalNeededInReverse) {
          if (normalReturn != newCall) {
            assert(normalReturn->getType() == newCall->getType());
            gutils->replaceAWithB(newCall, normalReturn);
            BuilderZ.SetInsertPoint(newCall->getNextNode());
            gutils->erase(newCall);
          }
          normalReturn = gutils->cacheForReverse(
              BuilderZ, normalReturn, getIndex(&call, CacheType::Self));
        } else {
          if (normalReturn && normalReturn != newCall) {
            assert(normalReturn->getType() == newCall->getType());
            assert(Mode != DerivativeMode::ReverseModeGradient);
            gutils->replaceAWithB(newCall, normalReturn);
            BuilderZ.SetInsertPoint(newCall->getNextNode());
            gutils->erase(newCall);
          } else if (Mode == DerivativeMode::ReverseModeGradient)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
        return;
      }
    }

    if (Mode != DerivativeMode::ReverseModePrimal && called) {
      if (funcName == "__kmpc_for_static_init_4" ||
          funcName == "__kmpc_for_static_init_4u" ||
          funcName == "__kmpc_for_static_init_8" ||
          funcName == "__kmpc_for_static_init_8u") {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
        assert(fini);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
        fcall->setCallingConv(fini->getCallingConv());
        return;
      }
    }

    if ((funcName.startswith("MPI_") || funcName.startswith("PMPI_")) &&
        (!gutils->isConstantInstruction(&call) || funcName == "MPI_Barrier" ||
         funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect" ||
         MPIInactiveCommAllocators.find(funcName) !=
             MPIInactiveCommAllocators.end())) {
      handleMPI(call, called, funcName);
      return;
    }

    if (!called || called->empty()) {
      if (auto blas = extractBLAS(funcName)) {
#if LLVM_VERSION_MAJOR >= 16
        if (handleBLAS(call, called, blas.value(), overwritten_args))
#else
        if (handleBLAS(call, called, blas.getValue(), overwritten_args))
#endif
          return;
      }
    }

    if (funcName == "printf" || funcName == "puts" ||
        funcName.startswith("_ZN3std2io5stdio6_print") ||
        funcName.startswith("_ZN4core3fmt")) {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }
      return;
    }
    if (called && (called->getName().contains("__enzyme_float") ||
                   called->getName().contains("__enzyme_double") ||
                   called->getName().contains("__enzyme_integer") ||
                   called->getName().contains("__enzyme_pointer"))) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    // Handle lgamma, safe to recompute so no store/change to forward
    if (called) {
      if (funcName == "__kmpc_fork_call") {
        visitOMPCall(call);
        return;
      }

      if (funcName == "__kmpc_for_static_init_4" ||
          funcName == "__kmpc_for_static_init_4u" ||
          funcName == "__kmpc_for_static_init_8" ||
          funcName == "__kmpc_for_static_init_8u") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          auto fini =
              called->getParent()->getFunction("__kmpc_for_static_fini");
          assert(fini);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2)};
          auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
          fcall->setCallingConv(fini->getCallingConv());
        }
        return;
      }
      if (funcName == "__kmpc_for_static_fini") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        }
        return;
      }
      // TODO check
      // Adjoint of barrier is to place a barrier at the corresponding
      // location in the reverse.
      if (funcName == "__kmpc_barrier") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
#if LLVM_VERSION_MAJOR >= 11
          auto callval = call.getCalledOperand();
#else
          auto callval = call.getCalledValue();
#endif
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2),
              lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2)};
          Builder2.CreateCall(call.getFunctionType(), callval, args);
        }
        return;
      }
      if (funcName == "__kmpc_critical") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          auto crit2 = called->getParent()->getFunction("__kmpc_end_critical");
          assert(crit2);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                     Builder2)};
          auto fcall =
              Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
          fcall->setCallingConv(crit2->getCallingConv());
        }
        return;
      }
      if (funcName == "__kmpc_end_critical") {
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          auto crit2 = called->getParent()->getFunction("__kmpc_critical");
          assert(crit2);
          Value *args[] = {
              lookup(gutils->getNewFromOriginal(call.getArgOperand(0)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                     Builder2),
              lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                     Builder2)};
          auto fcall =
              Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
          fcall->setCallingConv(crit2->getCallingConv());
        }
        return;
      }

      if (funcName.startswith("__kmpc") &&
          funcName != "__kmpc_global_thread_num") {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << call << "\n";
        assert(0 && "unhandled openmp function");
        llvm_unreachable("unhandled openmp function");
      }

      auto mod = call.getParent()->getParent()->getParent();
#include "CallDerivatives.inc"

      if (funcName == "llvm.julia.gc_preserve_end") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {

          auto begin_call = cast<CallInst>(call.getOperand(0));

          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          SmallVector<Value *, 1> args;
#if LLVM_VERSION_MAJOR >= 14
          for (auto &arg : begin_call->args())
#else
          for (auto &arg : begin_call->arg_operands())
#endif
          {
            bool primalUsed = false;
            bool shadowUsed = false;
            gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

            if (primalUsed)
              args.push_back(
                  gutils->lookupM(gutils->getNewFromOriginal(arg), Builder2));

            if (!gutils->isConstantValue(arg) && shadowUsed) {
              Value *ptrshadow = gutils->lookupM(
                  gutils->invertPointerM(arg, BuilderZ), Builder2);
              if (gutils->getWidth() == 1)
                args.push_back(ptrshadow);
              else
                for (size_t i = 0; i < gutils->getWidth(); ++i)
                  args.push_back(gutils->extractMeta(Builder2, ptrshadow, i));
            }
          }

          auto newp = Builder2.CreateCall(
              called->getParent()->getOrInsertFunction(
                  "llvm.julia.gc_preserve_begin",
                  FunctionType::get(Type::getTokenTy(call.getContext()),
                                    ArrayRef<Type *>(), true)),
              args);
          auto ifound = gutils->invertedPointers.find(begin_call);
          assert(ifound != gutils->invertedPointers.end());
          auto placeholder = cast<CallInst>(&*ifound->second);
          gutils->invertedPointers.erase(ifound);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)begin_call, InvertedPointerVH(gutils, newp)));

          gutils->replaceAWithB(placeholder, newp);
          gutils->erase(placeholder);
        }
        return;
      }
      if (funcName == "llvm.julia.gc_preserve_begin") {
        SmallVector<Value *, 1> args;
#if LLVM_VERSION_MAJOR >= 14
        for (auto &arg : call.args())
#else
        for (auto &arg : call.arg_operands())
#endif
        {
          bool primalUsed = false;
          bool shadowUsed = false;
          gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

          if (primalUsed)
            args.push_back(gutils->getNewFromOriginal(arg));

          if (!gutils->isConstantValue(arg) && shadowUsed) {
            Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
            if (gutils->getWidth() == 1)
              args.push_back(ptrshadow);
            else
              for (size_t i = 0; i < gutils->getWidth(); ++i)
                args.push_back(gutils->extractMeta(BuilderZ, ptrshadow, i));
          }
        }

        auto newp = BuilderZ.CreateCall(called, args);
        auto oldp = gutils->getNewFromOriginal(&call);
        gutils->replaceAWithB(oldp, newp);
        gutils->erase(oldp);

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModeCombined) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);

          auto ifound = gutils->invertedPointers.find(&call);
          assert(ifound != gutils->invertedPointers.end());
          auto placeholder = cast<CallInst>(&*ifound->second);
          Builder2.CreateCall(called->getParent()->getOrInsertFunction(
                                  "llvm.julia.gc_preserve_end",
                                  FunctionType::get(Builder2.getVoidTy(),
                                                    call.getType(), false)),
                              placeholder);
        }
        return;
      }

      // Functions that only modify pointers and don't allocate memory,
      // needs to be run on shadow in primal
      if (funcName == "_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_"
                      "node_baseS0_RS_") {
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(call.getArgOperand(3)))
          return;
        SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
        for (auto &arg : call.args())
#else
        for (auto &arg : call.arg_operands())
#endif
        {
          if (gutils->isConstantValue(arg))
            args.push_back(gutils->getNewFromOriginal(arg));
          else
            args.push_back(gutils->invertPointerM(arg, BuilderZ));
        }
        BuilderZ.CreateCall(called, args);
        return;
      }

      // Functions that initialize a shadow data structure (with no
      // other arguments) needs to be run on shadow in primal.
      if (funcName == "_ZNSt8ios_baseC2Ev" ||
          funcName == "_ZNSt8ios_baseD2Ev" || funcName == "_ZNSt6localeC1Ev" ||
          funcName == "_ZNSt6localeD1Ev" ||
          funcName == "_ZNKSt5ctypeIcE13_M_widen_initEv") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(call.getArgOperand(0)))
          return;
        Value *args[] = {
            gutils->invertPointerM(call.getArgOperand(0), BuilderZ)};
        BuilderZ.CreateCall(called, args);
        return;
      }

      if (funcName == "_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_"
                      "streambufIcS1_E") {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return;
        }
        if (gutils->isConstantValue(call.getArgOperand(0)))
          return;
        Value *args[] = {
            gutils->invertPointerM(call.getArgOperand(0), BuilderZ),
            gutils->invertPointerM(call.getArgOperand(1), BuilderZ)};
        BuilderZ.CreateCall(called, args);
        return;
      }

      // if constant instruction and readonly (thus must be pointer return)
      // and shadow return recomputable from shadow arguments.
      if (funcName == "__dynamic_cast" ||
          funcName == "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base" ||
          funcName == "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base" ||
          funcName == "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base" ||
          funcName == "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base" ||
          funcName == "jl_ptr_to_array" || funcName == "jl_ptr_to_array_1d") {
        bool shouldCache = false;
        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            shouldCache = true;
          }
        }
        ValueToValueMapTy empty;
        bool lrc = gutils->legalRecompute(&call, empty, nullptr);

        if (!gutils->isConstantValue(&call)) {
          auto ifound = gutils->invertedPointers.find(&call);
          assert(ifound != gutils->invertedPointers.end());
          auto placeholder = cast<PHINode>(&*ifound->second);

          if (subretType == DIFFE_TYPE::DUP_ARG) {
            Value *shadow = placeholder;
            if (lrc || Mode == DerivativeMode::ReverseModePrimal ||
                Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ForwardMode) {
              if (gutils->isConstantValue(call.getArgOperand(0)))
                shadow = gutils->getNewFromOriginal(&call);
              else {
                SmallVector<Value *, 2> args;
                size_t i = 0;
#if LLVM_VERSION_MAJOR >= 14
                for (auto &arg : call.args())
#else
                for (auto &arg : call.arg_operands())
#endif
                {
                  if (gutils->isConstantValue(arg) ||
                      (funcName == "__dynamic_cast" && i > 0) ||
                      (funcName == "jl_ptr_to_array_1d" && i != 1) ||
                      (funcName == "jl_ptr_to_array" && i != 1))
                    args.push_back(gutils->getNewFromOriginal(arg));
                  else
                    args.push_back(gutils->invertPointerM(arg, BuilderZ));
                  i++;
                }
                shadow = BuilderZ.CreateCall(called, args);
              }
            }

            bool needsReplacement = true;
            if (!lrc && (Mode == DerivativeMode::ReverseModePrimal ||
                         Mode == DerivativeMode::ReverseModeGradient)) {
              shadow = gutils->cacheForReverse(
                  BuilderZ, shadow, getIndex(&call, CacheType::Shadow));
              if (Mode == DerivativeMode::ReverseModeGradient)
                needsReplacement = false;
            }
            gutils->invertedPointers.erase((const Value *)&call);
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)&call, InvertedPointerVH(gutils, shadow)));
            if (needsReplacement) {
              assert(shadow != placeholder);
              gutils->replaceAWithB(placeholder, shadow);
              gutils->erase(placeholder);
            }
          } else {
            gutils->invertedPointers.erase((const Value *)&call);
            gutils->erase(placeholder);
          }
        }

        if (Mode == DerivativeMode::ForwardMode) {
          eraseIfUnused(call);
          assert(gutils->isConstantInstruction(&call));
          return;
        }

        if (!shouldCache && !lrc) {
          std::map<UsageKey, bool> Seen;
          for (auto pair : gutils->knownRecomputeHeuristic)
            Seen[UsageKey(pair.first, ValueType::Primal)] = false;
          bool primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  ValueType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
          shouldCache = primalNeededInReverse;
        }

        if (shouldCache) {
          BuilderZ.SetInsertPoint(newCall->getNextNode());
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self));
        }
        eraseIfUnused(call);
        assert(gutils->isConstantInstruction(&call));
        return;
      }

      if (called) {
        if (funcName == "julia.write_barrier") {
          bool backwardsShadow = false;
          bool forwardsShadow = true;
          for (auto pair : gutils->backwardsOnlyShadows) {
            if (pair.second.stores.count(&call)) {
              backwardsShadow = true;
              forwardsShadow = pair.second.primalInitialize;
              if (auto inst = dyn_cast<Instruction>(pair.first))
                if (!forwardsShadow && pair.second.LI &&
                    pair.second.LI->contains(inst->getParent()))
                  backwardsShadow = false;
              break;
            }
          }

          if (Mode == DerivativeMode::ForwardMode ||
              (Mode == DerivativeMode::ReverseModeCombined &&
               (forwardsShadow || backwardsShadow)) ||
              (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
              (Mode == DerivativeMode::ReverseModeGradient &&
               backwardsShadow)) {
            SmallVector<Value *, 1> iargs;
            IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
#if LLVM_VERSION_MAJOR >= 14
            for (auto &arg : call.args())
#else
            for (auto &arg : call.arg_operands())
#endif
            {
              if (!gutils->isConstantValue(arg)) {
                Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
                applyChainRule(
                    BuilderZ,
                    [&](Value *ptrshadow) { iargs.push_back(ptrshadow); },
                    ptrshadow);
              }
            }
            if (iargs.size()) {
              BuilderZ.CreateCall(called, iargs);
            }
          }

          bool forceErase = false;
          if (Mode == DerivativeMode::ReverseModeGradient) {
            for (const auto &pair : gutils->rematerializableAllocations) {
              if (pair.second.stores.count(&call) && pair.second.LI) {
                forceErase = true;
              }
            }
          }
          if (forceErase)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          else
            eraseIfUnused(call);

          return;
        }
        Intrinsic::ID ID = Intrinsic::not_intrinsic;
        if (isMemFreeLibMFunction(funcName, &ID)) {
          if (Mode == DerivativeMode::ReverseModePrimal ||
              gutils->isConstantInstruction(&call)) {

            if (gutils->knownRecomputeHeuristic.find(&call) !=
                gutils->knownRecomputeHeuristic.end()) {
              if (!gutils->knownRecomputeHeuristic[&call]) {
                gutils->cacheForReverse(BuilderZ, newCall,
                                        getIndex(&call, CacheType::Self));
              }
            }
            eraseIfUnused(call);
            return;
          }

          if (ID != Intrinsic::not_intrinsic) {
            SmallVector<Value *, 2> orig_ops(call.getNumOperands());
            for (unsigned i = 0; i < call.getNumOperands(); ++i) {
              orig_ops[i] = call.getOperand(i);
            }
            bool cached = handleAdjointForIntrinsic(ID, call, orig_ops);
            if (!cached) {
              if (gutils->knownRecomputeHeuristic.find(&call) !=
                  gutils->knownRecomputeHeuristic.end()) {
                if (!gutils->knownRecomputeHeuristic[&call]) {
                  gutils->cacheForReverse(BuilderZ, newCall,
                                          getIndex(&call, CacheType::Self));
                }
              }
            }
            eraseIfUnused(call);
            return;
          }
        }
      }
    }
#if LLVM_VERSION_MAJOR >= 11
    if (auto assembly = dyn_cast<InlineAsm>(call.getCalledOperand()))
#else
    if (auto assembly = dyn_cast<InlineAsm>(call.getCalledValue()))
#endif
    {
      if (assembly->getAsmString() == "maxpd $1, $0") {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            gutils->isConstantInstruction(&call)) {

          if (gutils->knownRecomputeHeuristic.find(&call) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[&call]) {
              gutils->cacheForReverse(BuilderZ, newCall,
                                      getIndex(&call, CacheType::Self));
            }
          }
          eraseIfUnused(call);
          return;
        }

        SmallVector<Value *, 2> orig_ops(call.getNumOperands());
        for (unsigned i = 0; i < call.getNumOperands(); ++i) {
          orig_ops[i] = call.getOperand(i);
        }
        handleAdjointForIntrinsic(Intrinsic::maxnum, call, orig_ops);
        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self));
          }
        }
        eraseIfUnused(call);
        return;
      }
    }

    if (funcName == "realloc") {
      if (Mode == DerivativeMode::ForwardMode) {
        if (!gutils->isConstantValue(&call)) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
          for (unsigned i = 0; i < call.arg_size(); ++i)
#else
          for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
          {
            auto arg = call.getArgOperand(i);
            if (i == 0) {
              assert(!gutils->isConstantValue(arg));
              arg = gutils->invertPointerM(arg, Builder2);
            } else {
              arg = gutils->getNewFromOriginal(arg);
            }
            args.push_back(arg);
          }
          auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();

          auto rule = [&]() {
            SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                  ValueType::Primal);

            auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                   /*lookup*/ false);

            CallInst *CI = Builder2.CreateCall(
                call.getFunctionType(), call.getCalledFunction(), args, Defs);
            CI->setAttributes(call.getAttributes());
            CI->setCallingConv(call.getCallingConv());
            CI->setTailCallKind(call.getTailCallKind());
            CI->setDebugLoc(dbgLoc);
            return CI;
          };

          Value *CI = applyChainRule(call.getType(), Builder2, rule);

          auto found = gutils->invertedPointers.find(&call);
          PHINode *placeholder = cast<PHINode>(&*found->second);

          gutils->invertedPointers.erase(found);
          gutils->replaceAWithB(placeholder, CI);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(
              std::make_pair(&call, InvertedPointerVH(gutils, CI)));
        }
        eraseIfUnused(call);
        return;
      }
    }

    if (isAllocationFunction(funcName, gutils->TLI)) {

      bool constval = gutils->isConstantValue(&call);

      if (!constval) {
        auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();
        auto found = gutils->invertedPointers.find(&call);
        PHINode *placeholder = cast<PHINode>(&*found->second);
        IRBuilder<> bb(placeholder);

        SmallVector<Value *, 8> args;
#if LLVM_VERSION_MAJOR >= 14
        for (auto &arg : call.args())
#else
        for (auto &arg : call.arg_operands())
#endif
        {
          args.push_back(gutils->getNewFromOriginal(arg));
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ForwardModeSplit) {

          Value *anti = placeholder;
          // If rematerializable allocations and split mode, we can
          // simply elect to build the entire piece in the reverse
          // since it should be possible to perform any shadow stores
          // of pointers (from rematerializable property) and it does
          // not escape the function scope (lest it not be
          // rematerializable) so all input derivatives remain zero.
          bool backwardsShadow = false;
          bool forwardsShadow = true;
          bool inLoop = false;
          bool isAlloca = isa<AllocaInst>(&call);
          {
            auto found = gutils->backwardsOnlyShadows.find(&call);
            if (found != gutils->backwardsOnlyShadows.end()) {
              backwardsShadow = true;
              forwardsShadow = found->second.primalInitialize;
              // If in a loop context, maintain the same free behavior.
              if (found->second.LI &&
                  found->second.LI->contains(call.getParent()))
                inLoop = true;
            }
          }
          {

            if (!forwardsShadow) {
              if (Mode == DerivativeMode::ReverseModePrimal) {
                // Needs a stronger replacement check/assertion.
                Value *replacement =
                    getUndefinedValueForType(placeholder->getType());
                gutils->replaceAWithB(placeholder, replacement);
                gutils->invertedPointers.erase(found);
                gutils->invertedPointers.insert(std::make_pair(
                    &call, InvertedPointerVH(gutils, replacement)));
                gutils->erase(placeholder);
                anti = nullptr;
                goto endAnti;
              } else if (inLoop) {
                gutils->rematerializedPrimalOrShadowAllocations.push_back(
                    placeholder);
                goto endAnti;
              }
            }
            placeholder->setName("");
            if (shadowHandlers.find(funcName) != shadowHandlers.end()) {
              bb.SetInsertPoint(placeholder);

              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow)) {
                anti = applyChainRule(call.getType(), bb, [&]() {
                  return shadowHandlers[funcName](bb, &call, args, gutils);
                });
                if (anti->getType() != placeholder->getType()) {
                  llvm::errs() << "orig: " << call << "\n";
                  llvm::errs() << "placeholder: " << *placeholder << "\n";
                  llvm::errs() << "anti: " << *anti << "\n";
                }
                gutils->invertedPointers.erase(found);
                bb.SetInsertPoint(placeholder);

                gutils->replaceAWithB(placeholder, anti);
                gutils->erase(placeholder);
              }

              if (auto inst = dyn_cast<Instruction>(anti))
                bb.SetInsertPoint(inst);

              if (!backwardsShadow)
                anti = gutils->cacheForReverse(
                    bb, anti, getIndex(&call, CacheType::Shadow));
            } else {
              bool zeroed = false;
              auto rule = [&]() {
#if LLVM_VERSION_MAJOR >= 11
                Value *anti = bb.CreateCall(call.getFunctionType(),
                                            call.getCalledOperand(), args,
                                            call.getName() + "'mi");
#else
                anti = bb.CreateCall(call.getCalledValue(), args,
                                     call.getName() + "'mi");
#endif
                cast<CallInst>(anti)->setAttributes(call.getAttributes());
                cast<CallInst>(anti)->setCallingConv(call.getCallingConv());
                cast<CallInst>(anti)->setTailCallKind(call.getTailCallKind());
                cast<CallInst>(anti)->setDebugLoc(dbgLoc);

                if (anti->getType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 14
                  cast<CallInst>(anti)->addAttributeAtIndex(
                      AttributeList::ReturnIndex, Attribute::NoAlias);
                  cast<CallInst>(anti)->addAttributeAtIndex(
                      AttributeList::ReturnIndex, Attribute::NonNull);
#else
                  cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                                     Attribute::NoAlias);
                  cast<CallInst>(anti)->addAttribute(AttributeList::ReturnIndex,
                                                     Attribute::NonNull);
#endif

                  if (funcName == "malloc" || funcName == "_Znwm") {
                    if (auto ci = dyn_cast<ConstantInt>(args[0])) {
                      unsigned derefBytes = ci->getLimitedValue();
                      CallInst *cal =
                          cast<CallInst>(gutils->getNewFromOriginal(&call));
#if LLVM_VERSION_MAJOR >= 14
                      cast<CallInst>(anti)->addDereferenceableRetAttr(
                          derefBytes);
                      cal->addDereferenceableRetAttr(derefBytes);
#if !defined(FLANG) && !defined(ROCM)
                      AttrBuilder B(ci->getContext());
#else
                      AttrBuilder B;
#endif
                      B.addDereferenceableOrNullAttr(derefBytes);
                      cast<CallInst>(anti)->setAttributes(
                          cast<CallInst>(anti)
                              ->getAttributes()
                              .addRetAttributes(call.getContext(), B));
                      cal->setAttributes(cal->getAttributes().addRetAttributes(
                          call.getContext(), B));
                      cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                               Attribute::NoAlias);
                      cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                               Attribute::NonNull);
#else
                      cast<CallInst>(anti)->addDereferenceableAttr(
                          llvm::AttributeList::ReturnIndex, derefBytes);
                      cal->addDereferenceableAttr(
                          llvm::AttributeList::ReturnIndex, derefBytes);
                      cast<CallInst>(anti)->addDereferenceableOrNullAttr(
                          llvm::AttributeList::ReturnIndex, derefBytes);
                      cal->addDereferenceableOrNullAttr(
                          llvm::AttributeList::ReturnIndex, derefBytes);
                      cal->addAttribute(AttributeList::ReturnIndex,
                                        Attribute::NoAlias);
                      cal->addAttribute(AttributeList::ReturnIndex,
                                        Attribute::NonNull);
#endif
                    }
                  }
                  if (Mode == DerivativeMode::ReverseModeCombined ||
                      (Mode == DerivativeMode::ReverseModePrimal &&
                       forwardsShadow) ||
                      (Mode == DerivativeMode::ReverseModeGradient &&
                       backwardsShadow) ||
                      (Mode == DerivativeMode::ForwardModeSplit &&
                       backwardsShadow)) {
                    if (!inLoop) {
                      zeroKnownAllocation(bb, anti, args, funcName, gutils->TLI,
                                          &call);
                      zeroed = true;
                    }
                  }
                }
                return anti;
              };

              anti = applyChainRule(call.getType(), bb, rule);

              gutils->invertedPointers.erase(found);
              if (&*bb.GetInsertPoint() == placeholder)
                bb.SetInsertPoint(placeholder->getNextNode());
              gutils->replaceAWithB(placeholder, anti);
              gutils->erase(placeholder);

              if (!backwardsShadow)
                anti = gutils->cacheForReverse(
                    bb, anti, getIndex(&call, CacheType::Shadow));
              else {
                if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
                  isAlloca = true;
                  bb.SetInsertPoint(cast<Instruction>(anti));
                  Value *Size;
                  if (funcName == "malloc")
                    Size = args[0];
                  else if (funcName == "julia.gc_alloc_obj" ||
                           funcName == "jl_gc_alloc_typed" ||
                           funcName == "ijl_gc_alloc_typed")
                    Size = args[1];
                  else
                    llvm_unreachable("Unknown allocation to upgrade");

                  Type *elTy = Type::getInt8Ty(call.getContext());
                  std::string name = "";
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
                  if (call.getContext().supportsTypedPointers()) {
#endif
                    for (auto U : call.users()) {
                      if (hasMetadata(cast<Instruction>(U),
                                      "enzyme_caststack")) {
                        elTy = U->getType()->getPointerElementType();
                        Value *tsize = ConstantInt::get(
                            Size->getType(),
                            (gutils->newFunc->getParent()
                                 ->getDataLayout()
                                 .getTypeAllocSizeInBits(elTy) +
                             7) /
                                8);
                        Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
                        name = (U->getName() + "'ai").str();
                        break;
                      }
                    }
#if LLVM_VERSION_MAJOR >= 15
                  }
#endif
#endif
                  auto rule = [&](Value *anti) {
                    Value *replacement = bb.CreateAlloca(elTy, Size, name);
                    if (name.size() == 0)
                      replacement->takeName(anti);
                    else
                      anti->setName("");
                    auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                           MD->getOperand(0))
                                                           ->getValue())
                                         ->getLimitedValue();
                    if (Alignment) {
#if LLVM_VERSION_MAJOR >= 10
                      cast<AllocaInst>(replacement)
                          ->setAlignment(Align(Alignment));
#else
                      cast<AllocaInst>(replacement)->setAlignment(Alignment);
#endif
                    }
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
                    if (call.getContext().supportsTypedPointers()) {
#endif
                      if (anti->getType()->getPointerElementType() != elTy)
                        replacement = bb.CreatePointerCast(
                            replacement,
                            PointerType::getUnqual(
                                anti->getType()->getPointerElementType()));
#if LLVM_VERSION_MAJOR >= 15
                    }
#endif
#endif
                    if (int AS = cast<PointerType>(anti->getType())
                                     ->getAddressSpace()) {
                      llvm::PointerType *PT;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
                      if (call.getContext().supportsTypedPointers()) {
#endif
                        PT = PointerType::get(
                            anti->getType()->getPointerElementType(), AS);
#if LLVM_VERSION_MAJOR >= 15
                      } else {
                        PT = PointerType::get(anti->getContext(), AS);
                      }
#endif
#endif
                      replacement = bb.CreateAddrSpaceCast(replacement, PT);
                      cast<Instruction>(replacement)
                          ->setMetadata(
                              "enzyme_backstack",
                              MDNode::get(replacement->getContext(), {}));
                    }
                    return replacement;
                  };

                  auto replacement =
                      applyChainRule(call.getType(), bb, rule, anti);
                  gutils->replaceAWithB(cast<Instruction>(anti), replacement);
                  gutils->erase(cast<Instruction>(anti));
                  anti = replacement;
                }
              }

              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow) ||
                  (Mode == DerivativeMode::ForwardModeSplit &&
                   backwardsShadow)) {
                if (!inLoop) {
                  assert(zeroed);
                }
              }
            }
            gutils->invertedPointers.insert(
                std::make_pair(&call, InvertedPointerVH(gutils, anti)));
          }
        endAnti:;
          if (((Mode == DerivativeMode::ReverseModeCombined && shouldFree()) ||
               (Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
               (Mode == DerivativeMode::ForwardModeSplit && shouldFree())) &&
              !isAlloca) {
            IRBuilder<> Builder2(&call);
            getReverseBuilder(Builder2);
            assert(anti);
            Value *tofree = lookup(anti, Builder2);
            assert(tofree);
            assert(tofree->getType());
            auto rule = [&](Value *tofree) {
              auto CI = freeKnownAllocation(Builder2, tofree, funcName, dbgLoc,
                                            gutils->TLI, &call, gutils);
              if (CI)
#if LLVM_VERSION_MAJOR >= 14
                CI->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                        Attribute::NonNull);
#else
                CI->addAttribute(AttributeList::FirstArgIndex,
                                 Attribute::NonNull);
#endif
            };
            applyChainRule(Builder2, rule, tofree);
          }
        } else if (Mode == DerivativeMode::ForwardMode) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          SmallVector<Value *, 2> args;
#if LLVM_VERSION_MAJOR >= 14
          for (unsigned i = 0; i < call.arg_size(); ++i)
#else
          for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
          {
            auto arg = call.getArgOperand(i);
            args.push_back(gutils->getNewFromOriginal(arg));
          }

          auto rule = [&]() {
            SmallVector<ValueType, 2> BundleTypes(args.size(),
                                                  ValueType::Primal);

            auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                   /*lookup*/ false);

            CallInst *CI = Builder2.CreateCall(
                call.getFunctionType(), call.getCalledFunction(), args, Defs);
            CI->setAttributes(call.getAttributes());
            CI->setCallingConv(call.getCallingConv());
            CI->setTailCallKind(call.getTailCallKind());
            CI->setDebugLoc(dbgLoc);
            return CI;
          };

          Value *CI = applyChainRule(call.getType(), Builder2, rule);

          auto found = gutils->invertedPointers.find(&call);
          PHINode *placeholder = cast<PHINode>(&*found->second);

          gutils->invertedPointers.erase(found);
          gutils->replaceAWithB(placeholder, CI);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(
              std::make_pair(&call, InvertedPointerVH(gutils, CI)));
        }
      }

      // Cache and rematerialization irrelevant for forward mode.
      if (Mode == DerivativeMode::ForwardMode) {
        eraseIfUnused(call);
        return;
      }

      std::map<UsageKey, bool> Seen;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second)
          Seen[UsageKey(pair.first, ValueType::Primal)] = false;
      bool primalNeededInReverse =
          Mode == DerivativeMode::ForwardMode
              ? false
              : DifferentialUseAnalysis::is_value_needed_in_reverse<
                    ValueType::Primal>(gutils, &call, Mode, Seen,
                                       oldUnreachable);

      bool cacheWholeAllocation = gutils->needsCacheWholeAllocation(&call);
      if (cacheWholeAllocation) {
        primalNeededInReverse = true;
      }

      auto restoreFromStack = [&](MDNode *MD) {
        IRBuilder<> B(newCall);
        Value *Size;
        if (funcName == "malloc")
          Size = call.getArgOperand(0);
        else if (funcName == "julia.gc_alloc_obj" ||
                 funcName == "jl_gc_alloc_typed" ||
                 funcName == "ijl_gc_alloc_typed")
          Size = call.getArgOperand(1);
        else
          llvm_unreachable("Unknown allocation to upgrade");
        Size = gutils->getNewFromOriginal(Size);

        if (auto CI = dyn_cast<ConstantInt>(Size)) {
          B.SetInsertPoint(gutils->inversionAllocs);
        }
        Type *elTy = Type::getInt8Ty(call.getContext());
        Instruction *I = nullptr;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
        if (call.getContext().supportsTypedPointers()) {
#endif
          for (auto U : call.users()) {
            if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
              elTy = U->getType()->getPointerElementType();
              Value *tsize = ConstantInt::get(
                  Size->getType(), (gutils->newFunc->getParent()
                                        ->getDataLayout()
                                        .getTypeAllocSizeInBits(elTy) +
                                    7) /
                                       8);
              Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
              I = gutils->getNewFromOriginal(cast<Instruction>(U));
              break;
            }
          }
#if LLVM_VERSION_MAJOR >= 15
        }
#endif
#endif
        Value *replacement = B.CreateAlloca(elTy, Size);
        if (I)
          replacement->takeName(I);
        else
          replacement->takeName(newCall);
        auto Alignment =
            cast<ConstantInt>(
                cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
                ->getLimitedValue();
        // Don't set zero alignment
        if (Alignment) {
#if LLVM_VERSION_MAJOR >= 10
          cast<AllocaInst>(replacement)->setAlignment(Align(Alignment));
#else
          cast<AllocaInst>(replacement)->setAlignment(Alignment);
#endif
        }
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
        if (call.getContext().supportsTypedPointers()) {
#endif
          if (call.getType()->getPointerElementType() != elTy)
            replacement = B.CreatePointerCast(
                replacement, PointerType::getUnqual(
                                 call.getType()->getPointerElementType()));

#if LLVM_VERSION_MAJOR >= 15
        }
#endif
#endif
        if (int AS = cast<PointerType>(call.getType())->getAddressSpace()) {

          llvm::PointerType *PT;
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
          if (call.getContext().supportsTypedPointers()) {
#endif
            PT = PointerType::get(call.getType()->getPointerElementType(), AS);
#if LLVM_VERSION_MAJOR >= 15
          } else {
            PT = PointerType::get(call.getContext(), AS);
          }
#endif
#endif
          replacement = B.CreateAddrSpaceCast(replacement, PT);
          cast<Instruction>(replacement)
              ->setMetadata("enzyme_backstack",
                            MDNode::get(replacement->getContext(), {}));
        }
        gutils->replaceAWithB(newCall, replacement);
        gutils->erase(newCall);
      };

      // Don't erase any store that needs to be preserved for a
      // rematerialization
      {
        auto found = gutils->rematerializableAllocations.find(&call);
        if (found != gutils->rematerializableAllocations.end()) {
          // If rematerializing (e.g. needed in reverse, but not needing
          //  the whole allocation):
          if (primalNeededInReverse && !cacheWholeAllocation) {
            assert(!unnecessaryValues.count(&call));
            // if rematerialize, don't ever cache and downgrade to stack
            // allocation where possible.
            if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
              if (Mode == DerivativeMode::ReverseModeGradient &&
                  found->second.LI) {
                gutils->rematerializedPrimalOrShadowAllocations.push_back(
                    newCall);
              } else {
                restoreFromStack(MD);
              }
              return;
            }

            // No need to free GC.
            if (funcName == "ijl_alloc_array_1d" ||
                funcName == "ijl_alloc_array_2d" ||
                funcName == "ijl_alloc_array_3d" ||
                funcName == "ijl_array_copy" ||
                funcName == "jl_alloc_array_1d" ||
                funcName == "jl_alloc_array_2d" ||
                funcName == "jl_alloc_array_3d" ||
                funcName == "jl_array_copy" ||
                funcName == "julia.gc_alloc_obj" ||
                funcName == "jl_gc_alloc_typed" ||
                funcName == "ijl_gc_alloc_typed") {
              if (Mode == DerivativeMode::ReverseModeGradient &&
                  found->second.LI)
                gutils->rematerializedPrimalOrShadowAllocations.push_back(
                    newCall);
              return;
            }

            // Otherwise if in reverse pass, free the newly created allocation.
            if (Mode == DerivativeMode::ReverseModeGradient ||
                Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ForwardModeSplit) {
              IRBuilder<> Builder2(&call);
              getReverseBuilder(Builder2);
              auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
              freeKnownAllocation(Builder2, lookup(newCall, Builder2), funcName,
                                  dbgLoc, gutils->TLI, &call, gutils);
              if (Mode == DerivativeMode::ReverseModeGradient &&
                  found->second.LI && found->second.LI->contains(&call))
                gutils->rematerializedPrimalOrShadowAllocations.push_back(
                    newCall);
              return;
            }
            // If in primal, do nothing (keeping the original caching behavior)
            if (Mode == DerivativeMode::ReverseModePrimal)
              return;
          } else if (!cacheWholeAllocation) {
            if (unnecessaryValues.count(&call)) {
              eraseIfUnused(call, /*erase*/ true, /*check*/ false);
              return;
            }
            // If not caching allocation and not needed in the reverse, we can
            // use the original freeing behavior for the function. If in the
            // reverse pass we should not recreate this allocation.
            if (Mode == DerivativeMode::ReverseModeGradient)
              eraseIfUnused(call, /*erase*/ true, /*check*/ false);
            else if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
              restoreFromStack(MD);
            }
            return;
          }
        }
      }

      // If an allocation is not needed in the reverse, maintain the original
      // free behavior and do not rematerialize this for the reverse. However,
      // this is only safe to perform for allocations with a guaranteed free
      // as can we can only guarantee that we don't erase those frees.
      bool hasPDFree = gutils->allocationsWithGuaranteedFree.count(&call);
      if (!primalNeededInReverse && hasPDFree) {
        if (unnecessaryValues.count(&call)) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return;
        }
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        } else {
          if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
            restoreFromStack(MD);
          }
        }
        return;
      }

      // If an object is managed by the GC do not preserve it for later free,
      // Thus it only needs caching if there is a need for it in the reverse.
      if (funcName == "jl_alloc_array_1d" || funcName == "jl_alloc_array_2d" ||
          funcName == "jl_alloc_array_3d" || funcName == "jl_array_copy" ||
          funcName == "ijl_alloc_array_1d" ||
          funcName == "ijl_alloc_array_2d" ||
          funcName == "ijl_alloc_array_3d" || funcName == "ijl_array_copy" ||
          funcName == "julia.gc_alloc_obj" || funcName == "jl_gc_alloc_typed" ||
          funcName == "ijl_gc_alloc_typed") {
        if (!subretused) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return;
        }
        if (!primalNeededInReverse) {
          if (Mode == DerivativeMode::ReverseModeGradient ||
              Mode == DerivativeMode::ForwardModeSplit) {
            auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                         call.getName() + "_replacementJ");
            gutils->fictiousPHIs[pn] = &call;
            gutils->replaceAWithB(newCall, pn);
            gutils->erase(newCall);
          }
        } else if (Mode != DerivativeMode::ReverseModeCombined) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self));
        }
        return;
      }

      if (EnzymeFreeInternalAllocations)
        hasPDFree = true;

      // TODO enable this if we need to free the memory
      // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
      // TO FREE'ing
      if ((primalNeededInReverse &&
           !gutils->unnecessaryIntermediates.count(&call)) ||
          hasPDFree) {
        Value *nop = gutils->cacheForReverse(BuilderZ, newCall,
                                             getIndex(&call, CacheType::Self));
        if (hasPDFree &&
            ((Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
             Mode == DerivativeMode::ReverseModeCombined ||
             (Mode == DerivativeMode::ForwardModeSplit && shouldFree()))) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
          freeKnownAllocation(Builder2, lookup(nop, Builder2), funcName, dbgLoc,
                              gutils->TLI, &call, gutils);
        }
      } else if (Mode == DerivativeMode::ReverseModeGradient ||
                 Mode == DerivativeMode::ReverseModeCombined ||
                 Mode == DerivativeMode::ForwardModeSplit) {
        // Note that here we cannot simply replace with null as users who
        // try to find the shadow pointer will use the shadow of null rather
        // than the true shadow of this
        auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                     call.getName() + "_replacementB");
        gutils->fictiousPHIs[pn] = &call;
        gutils->replaceAWithB(newCall, pn);
        gutils->erase(newCall);
      }

      return;
    }

    if (funcName == "julia.pointer_from_objref") {
      if (gutils->isConstantValue(&call)) {
        eraseIfUnused(call);
        return;
      }

      auto ifound = gutils->invertedPointers.find(&call);
      assert(ifound != gutils->invertedPointers.end());

      auto placeholder = cast<PHINode>(&*ifound->second);

      bool needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
          ValueType::Shadow>(gutils, &call, Mode, oldUnreachable);
      if (!needShadow) {
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
        eraseIfUnused(call);
        return;
      }

      Value *ptrshadow =
          gutils->invertPointerM(call.getArgOperand(0), BuilderZ);

      Value *val = applyChainRule(
          call.getType(), BuilderZ,
          [&](Value *v) -> Value * { return BuilderZ.CreateCall(called, {v}); },
          ptrshadow);

      gutils->replaceAWithB(placeholder, val);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return;
    }
    if (funcName.contains("__enzyme_todense")) {
      if (gutils->isConstantValue(&call)) {
        eraseIfUnused(call);
        return;
      }

      auto ifound = gutils->invertedPointers.find(&call);
      assert(ifound != gutils->invertedPointers.end());

      auto placeholder = cast<PHINode>(&*ifound->second);

      bool needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
          ValueType::Shadow>(gutils, &call, Mode, oldUnreachable);
      if (!needShadow) {
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
        eraseIfUnused(call);
        return;
      }

      SmallVector<Value *, 3> args;
      for (size_t i = 0; i < 2; i++)
        args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 2; i < call.arg_size(); ++i)
#else
      for (size_t i = 2; i < call.getNumArgOperands(); ++i)
#endif
        args.push_back(gutils->invertPointerM(call.getArgOperand(0), BuilderZ));

      Value *res = UndefValue::get(gutils->getShadowType(call.getType()));
      if (gutils->getWidth() == 1) {
        res = BuilderZ.CreateCall(called, args);
      } else {
        for (size_t w = 0; w < gutils->getWidth(); ++w) {
          SmallVector<Value *, 3> targs = {args[0], args[1]};
#if LLVM_VERSION_MAJOR >= 14
          for (size_t i = 2; i < call.arg_size(); ++i)
#else
          for (size_t i = 2; i < call.getNumArgOperands(); ++i)
#endif
            targs.push_back(GradientUtils::extractMeta(BuilderZ, args[i], w));

          auto tres = BuilderZ.CreateCall(called, targs);
          res = BuilderZ.CreateInsertValue(res, tres, w);
        }
      }

      gutils->replaceAWithB(placeholder, res);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return;
    }

    if (funcName == "memcpy" || funcName == "memmove") {
      auto ID = (funcName == "memcpy") ? Intrinsic::memcpy : Intrinsic::memmove;
#if LLVM_VERSION_MAJOR >= 10
      visitMemTransferCommon(ID, /*srcAlign*/ MaybeAlign(1),
                             /*dstAlign*/ MaybeAlign(1), call,
                             call.getArgOperand(0), call.getArgOperand(1),
                             gutils->getNewFromOriginal(call.getArgOperand(2)),
                             ConstantInt::getFalse(call.getContext()));
#else
      visitMemTransferCommon(ID, /*srcAlign*/ 1,
                             /*dstAlign*/ 1, call, call.getArgOperand(0),
                             call.getArgOperand(1),
                             gutils->getNewFromOriginal(call.getArgOperand(2)),
                             ConstantInt::getFalse(call.getContext()));
#endif
      return;
    }
    if (funcName == "memset" || funcName == "memset_pattern16") {
      visitMemSetCommon(call);
      return;
    }
    if (funcName == "enzyme_zerotype") {
      IRBuilder<> BuilderZ(&call);
      getForwardBuilder(BuilderZ);

      bool forceErase = Mode == DerivativeMode::ReverseModeGradient ||
                        Mode == DerivativeMode::ForwardModeSplit;

      if (forceErase)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      else
        eraseIfUnused(call);

      Value *orig_op0 = call.getArgOperand(0);

      // If constant destination then no operation needs doing
      if (gutils->isConstantValue(orig_op0)) {
        return;
      }

      if (!forceErase) {
        Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
        Value *op1 = gutils->getNewFromOriginal(call.getArgOperand(1));
        Value *op2 = gutils->getNewFromOriginal(call.getArgOperand(2));
        auto Defs = gutils->getInvertedBundles(
            &call, {ValueType::Shadow, ValueType::Primal, ValueType::Primal},
            BuilderZ, /*lookup*/ false);

        applyChainRule(
            BuilderZ,
            [&](Value *op0) {
              SmallVector<Value *, 4> args = {op0, op1, op2};
              auto cal =
                  BuilderZ.CreateCall(call.getCalledFunction(), args, Defs);
              llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
              ToCopy2.push_back(LLVMContext::MD_noalias);
              cal->copyMetadata(call, ToCopy2);
              cal->setAttributes(call.getAttributes());
              if (auto m = hasMetadata(&call, "enzyme_zerostack"))
                cal->setMetadata("enzyme_zerostack", m);
              cal->setCallingConv(call.getCallingConv());
              cal->setTailCallKind(call.getTailCallKind());
              cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
            },
            op0);
      }
      return;
    }
    if (funcName == "cuStreamCreate") {
      Value *val = nullptr;
      llvm::Type *PT = Type::getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
      if (call.getContext().supportsTypedPointers()) {
#endif
        if (isa<PointerType>(call.getArgOperand(0)->getType()))
          PT = call.getArgOperand(0)->getType()->getPointerElementType();
#if LLVM_VERSION_MAJOR >= 15
      }
#endif
#endif
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        val = gutils->getNewFromOriginal(call.getOperand(0));
        if (!isa<PointerType>(val->getType()))
          val = BuilderZ.CreateIntToPtr(val, PointerType::getUnqual(PT));
        val = BuilderZ.CreateLoad(PT, val);
        val = gutils->cacheForReverse(BuilderZ, val,
                                      getIndex(&call, CacheType::Tape));

      } else if (Mode == DerivativeMode::ReverseModeGradient) {
        PHINode *toReplace =
            BuilderZ.CreatePHI(PT, 1, call.getName() + "_psxtmp");
        val = gutils->cacheForReverse(BuilderZ, toReplace,
                                      getIndex(&call, CacheType::Tape));
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        val = gutils->lookupM(val, Builder2);
        auto FreeFunc = gutils->newFunc->getParent()->getOrInsertFunction(
            "cuStreamDestroy", call.getType(), PT);
        Value *nargs[] = {val};
        Builder2.CreateCall(FreeFunc, nargs);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }
    if (funcName == "cuStreamDestroy") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }
    if (funcName == "cuStreamSynchronize") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        Value *nargs[] = {gutils->lookupM(
            gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        Builder2.CreateCall(call.getFunctionType(), callval, nargs);
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }
    if (funcName == "posix_memalign" || funcName == "cuMemAllocAsync" ||
        funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2" ||
        funcName == "cudaMalloc" || funcName == "cudaMallocAsync" ||
        funcName == "cudaMallocHost" || funcName == "cudaMallocFromPoolAsync") {
      bool constval = gutils->isConstantInstruction(&call);

      Value *val;
      llvm::Type *PT = Type::getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 18
#if LLVM_VERSION_MAJOR >= 15
      if (call.getContext().supportsTypedPointers()) {
#endif
        if (isa<PointerType>(call.getArgOperand(0)->getType()))
          PT = call.getArgOperand(0)->getType()->getPointerElementType();
#if LLVM_VERSION_MAJOR >= 15
      }
#endif
#endif
      if (!constval) {
        Value *stream = nullptr;
        if (funcName == "cuMemAllocAsync")
          stream = gutils->getNewFromOriginal(call.getArgOperand(2));
        else if (funcName == "cudaMallocAsync")
          stream = gutils->getNewFromOriginal(call.getArgOperand(2));
        else if (funcName == "cudaMallocFromPoolAsync")
          stream = gutils->getNewFromOriginal(call.getArgOperand(3));

        auto M = gutils->newFunc->getParent();

        if (Mode == DerivativeMode::ReverseModePrimal ||
            Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ForwardMode) {
          Value *ptrshadow =
              gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
          SmallVector<Value *, 1> args;
          SmallVector<ValueType, 1> valtys;
          args.push_back(ptrshadow);
          valtys.push_back(ValueType::Shadow);
#if LLVM_VERSION_MAJOR >= 14
          for (size_t i = 1; i < call.arg_size(); ++i)
#else
          for (size_t i = 1; i < call.getNumArgOperands(); ++i)
#endif
          {
            args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
            valtys.push_back(ValueType::Primal);
          }

          auto Defs = gutils->getInvertedBundles(&call, valtys, BuilderZ,
                                                 /*lookup*/ false);

          val = applyChainRule(
              PT, BuilderZ,
              [&](Value *ptrshadow) {
                args[0] = ptrshadow;

                BuilderZ.CreateCall(called, args, Defs);
                if (!isa<PointerType>(ptrshadow->getType()))
                  ptrshadow = BuilderZ.CreateIntToPtr(
                      ptrshadow, PointerType::getUnqual(PT));
                Value *val = BuilderZ.CreateLoad(PT, ptrshadow);

                auto dst_arg = BuilderZ.CreateBitCast(
                    val, Type::getInt8PtrTy(call.getContext()));

                auto val_arg =
                    ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
                auto len_arg = gutils->getNewFromOriginal(
                    call.getArgOperand((funcName == "posix_memalign") ? 2 : 1));

                if (funcName == "posix_memalign" ||
                    funcName == "cudaMallocHost") {
                  auto volatile_arg = ConstantInt::getFalse(call.getContext());

                  Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};

                  Type *tys[] = {dst_arg->getType(), len_arg->getType()};

                  auto memset = cast<CallInst>(BuilderZ.CreateCall(
                      Intrinsic::getDeclaration(gutils->newFunc->getParent(),
                                                Intrinsic::memset, tys),
                      nargs));
                  // memset->addParamAttr(0,
                  // Attribute::getWithAlignment(Context,
                  // inst->getAlignment()));
                  memset->addParamAttr(0, Attribute::NonNull);
                } else if (funcName == "cudaMalloc") {
                  Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                  auto F = M->getOrInsertFunction(
                      "cudaMemset",
                      FunctionType::get(call.getType(), tys, false));
                  Value *nargs[] = {dst_arg, val_arg, len_arg};
                  auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                  memset->addParamAttr(0, Attribute::NonNull);
                } else if (funcName == "cudaMallocAsync" ||
                           funcName == "cudaMallocFromPoolAsync") {
                  Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                                 stream->getType()};
                  auto F = M->getOrInsertFunction(
                      "cudaMemsetAsync",
                      FunctionType::get(call.getType(), tys, false));
                  Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                  auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                  memset->addParamAttr(0, Attribute::NonNull);
                } else if (funcName == "cuMemAllocAsync") {
                  Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                                 stream->getType()};
                  auto F = M->getOrInsertFunction(
                      "cuMemsetD8Async",
                      FunctionType::get(call.getType(), tys, false));
                  Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                  auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                  memset->addParamAttr(0, Attribute::NonNull);
                } else if (funcName == "cuMemAlloc" ||
                           funcName == "cuMemAlloc_v2") {
                  Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                  auto F = M->getOrInsertFunction(
                      "cuMemsetD8",
                      FunctionType::get(call.getType(), tys, false));
                  Value *nargs[] = {dst_arg, val_arg, len_arg};
                  auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                  memset->addParamAttr(0, Attribute::NonNull);
                } else {
                  llvm_unreachable("unhandled allocation");
                }
                return val;
              },
              ptrshadow);

          if (Mode != DerivativeMode::ForwardMode)
            val = gutils->cacheForReverse(BuilderZ, val,
                                          getIndex(&call, CacheType::Tape));
        } else if (Mode == DerivativeMode::ReverseModeGradient) {
          PHINode *toReplace = BuilderZ.CreatePHI(gutils->getShadowType(PT), 1,
                                                  call.getName() + "_psxtmp");
          val = gutils->cacheForReverse(BuilderZ, toReplace,
                                        getIndex(&call, CacheType::Tape));
        }

        if (Mode == DerivativeMode::ReverseModeCombined ||
            Mode == DerivativeMode::ReverseModeGradient) {
          if (shouldFree()) {
            IRBuilder<> Builder2(&call);
            getReverseBuilder(Builder2);
            Value *tofree = gutils->lookupM(val, Builder2, ValueToValueMapTy(),
                                            /*tryLegalRecompute*/ false);

            Type *VoidTy = Type::getVoidTy(M->getContext());
            Type *IntPtrTy = Type::getInt8PtrTy(M->getContext());

            Value *streamL = nullptr;
            if (stream)
              streamL = gutils->lookupM(stream, Builder2);

            applyChainRule(
                BuilderZ,
                [&](Value *tofree) {
                  if (funcName == "posix_memalign") {
                    auto FreeFunc =
                        M->getOrInsertFunction("free", VoidTy, IntPtrTy);
                    Builder2.CreateCall(FreeFunc, tofree);
                  } else if (funcName == "cuMemAllocAsync") {
                    auto FreeFunc = M->getOrInsertFunction(
                        "cuMemFreeAsync", VoidTy, IntPtrTy, streamL->getType());
                    Value *nargs[] = {tofree, streamL};
                    Builder2.CreateCall(FreeFunc, nargs);
                  } else if (funcName == "cuMemAlloc" ||
                             funcName == "cuMemAlloc_v2") {
                    auto FreeFunc =
                        M->getOrInsertFunction("cuMemFree", VoidTy, IntPtrTy);
                    Value *nargs[] = {tofree};
                    Builder2.CreateCall(FreeFunc, nargs);
                  } else if (funcName == "cudaMalloc") {
                    auto FreeFunc =
                        M->getOrInsertFunction("cudaFree", VoidTy, IntPtrTy);
                    Value *nargs[] = {tofree};
                    Builder2.CreateCall(FreeFunc, nargs);
                  } else if (funcName == "cudaMallocAsync" ||
                             funcName == "cudaMallocFromPoolAsync") {
                    auto FreeFunc = M->getOrInsertFunction(
                        "cudaFreeAsync", VoidTy, IntPtrTy, streamL->getType());
                    Value *nargs[] = {tofree, streamL};
                    Builder2.CreateCall(FreeFunc, nargs);
                  } else if (funcName == "cudaMallocHost") {
                    auto FreeFunc = M->getOrInsertFunction("cudaFreeHost",
                                                           VoidTy, IntPtrTy);
                    Value *nargs[] = {tofree};
                    Builder2.CreateCall(FreeFunc, nargs);
                  } else
                    llvm_unreachable("unknown function to free");
                },
                tofree);
          }
        }
      }

      // TODO enable this if we need to free the memory
      // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
      // TO FREE'ing
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else if (Mode == DerivativeMode::ReverseModePrimal) {
        // if (is_value_needed_in_reverse<Primal>(
        //        TR, gutils, orig, /*topLevel*/ Mode ==
        //        DerivativeMode::Both))
        //        {

        //  gutils->cacheForReverse(BuilderZ, newCall,
        //                          getIndex(orig, CacheType::Self));
        //} else if (Mode != DerivativeMode::Forward) {
        // Note that here we cannot simply replace with null as users who try
        // to find the shadow pointer will use the shadow of null rather than
        // the true shadow of this
        //}
      } else if (Mode == DerivativeMode::ReverseModeCombined && shouldFree()) {
        IRBuilder<> Builder2(newCall->getNextNode());
        auto ptrv = gutils->getNewFromOriginal(call.getOperand(0));
        if (!isa<PointerType>(ptrv->getType()))
          ptrv = BuilderZ.CreateIntToPtr(ptrv, PointerType::getUnqual(PT));
        auto load = Builder2.CreateLoad(PT, ptrv, "posix_preread");
        Builder2.SetInsertPoint(&call);
        getReverseBuilder(Builder2);
        auto tofree = gutils->lookupM(load, Builder2, ValueToValueMapTy(),
                                      /*tryLegal*/ false);
        Value *streamL = nullptr;
        if (funcName == "cuMemAllocAsync")
          streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
        else if (funcName == "cudaMallocAsync")
          streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
        else if (funcName == "cudaMallocFromPoolAsync")
          streamL = gutils->getNewFromOriginal(call.getArgOperand(3));
        if (streamL)
          streamL = gutils->lookupM(streamL, Builder2);

        auto M = gutils->newFunc->getParent();
        Type *VoidTy = Type::getVoidTy(M->getContext());
        Type *IntPtrTy = Type::getInt8PtrTy(M->getContext());

        if (funcName == "posix_memalign") {
          auto FreeFunc = M->getOrInsertFunction("free", VoidTy, IntPtrTy);
          Builder2.CreateCall(FreeFunc, tofree);
        } else if (funcName == "cuMemAllocAsync") {
          auto FreeFunc = M->getOrInsertFunction("cuMemFreeAsync", VoidTy,
                                                 IntPtrTy, streamL->getType());
          Value *nargs[] = {tofree, streamL};
          Builder2.CreateCall(FreeFunc, nargs);
        } else if (funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2") {
          auto FreeFunc = M->getOrInsertFunction("cuMemFree", VoidTy, IntPtrTy);
          Value *nargs[] = {tofree};
          Builder2.CreateCall(FreeFunc, nargs);
        } else if (funcName == "cudaMalloc") {
          auto FreeFunc = M->getOrInsertFunction("cudaFree", VoidTy, IntPtrTy);
          Value *nargs[] = {tofree};
          Builder2.CreateCall(FreeFunc, nargs);
        } else if (funcName == "cudaMallocAsync" ||
                   funcName == "cudaMallocFromPoolAsync") {
          auto FreeFunc = M->getOrInsertFunction("cudaFreeAsync", VoidTy,
                                                 IntPtrTy, streamL->getType());
          Value *nargs[] = {tofree, streamL};
          Builder2.CreateCall(FreeFunc, nargs);
        } else if (funcName == "cudaMallocHost") {
          auto FreeFunc =
              M->getOrInsertFunction("cudaFreeHost", VoidTy, IntPtrTy);
          Value *nargs[] = {tofree};
          Builder2.CreateCall(FreeFunc, nargs);
        } else
          llvm_unreachable("unknown function to free");
      }

      return;
    }

    // Remove free's in forward pass so the memory can be used in the reverse
    // pass
    if (isDeallocationFunction(funcName, gutils->TLI)) {
      assert(gutils->invertedPointers.find(&call) ==
             gutils->invertedPointers.end());

      if (Mode == DerivativeMode::ForwardMode) {
        if (!gutils->isConstantValue(call.getArgOperand(0))) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto origfree = call.getArgOperand(0);
          auto newfree = gutils->getNewFromOriginal(call.getArgOperand(0));
          auto tofree = gutils->invertPointerM(origfree, Builder2);

          Function *free = getOrInsertCheckedFree(
              *call.getModule(), &call, newfree->getType(), gutils->getWidth());

          SmallVector<Value *, 3> args;
          args.push_back(newfree);

          auto rule = [&args](Value *tofree) { args.push_back(tofree); };
          applyChainRule(Builder2, rule, tofree);

          auto frees = Builder2.CreateCall(free->getFunctionType(), free, args);
          frees->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

          return;
        }
        eraseIfUnused(call);
      }
#if LLVM_VERSION_MAJOR >= 11
      auto callval = call.getCalledOperand();
#else
      auto callval = call.getCalledValue();
#endif

      for (auto rmat : gutils->backwardsOnlyShadows) {
        if (rmat.second.frees.count(&call)) {
          bool shouldFree = false;
          if (rmat.second.primalInitialize) {
            if (Mode == DerivativeMode::ReverseModePrimal)
              shouldFree = true;
          }

          if (shouldFree) {
            IRBuilder<> Builder2(&call);
            getForwardBuilder(Builder2);
            auto origfree = call.getArgOperand(0);
            auto tofree = gutils->invertPointerM(origfree, Builder2);
            if (tofree != origfree) {
              SmallVector<Value *, 2> args = {tofree};
              CallInst *CI =
                  Builder2.CreateCall(call.getFunctionType(), callval, args);
              CI->setAttributes(call.getAttributes());
            }
          }
          break;
        }
      }

      // If a rematerializable allocation.
      for (auto rmat : gutils->rematerializableAllocations) {
        if (rmat.second.frees.count(&call)) {
          // Leave the original free behavior since this won't be used
          // in the reverse pass in split mode
          if (Mode == DerivativeMode::ReverseModePrimal) {
            eraseIfUnused(call);
            return;
          } else if (Mode == DerivativeMode::ReverseModeGradient) {
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
            return;
          } else {
            assert(Mode == DerivativeMode::ReverseModeCombined);
            std::map<UsageKey, bool> Seen;
            for (auto pair : gutils->knownRecomputeHeuristic)
              if (!pair.second)
                Seen[UsageKey(pair.first, ValueType::Primal)] = false;
            bool primalNeededInReverse =
                DifferentialUseAnalysis::is_value_needed_in_reverse<
                    ValueType::Primal>(gutils, rmat.first, Mode, Seen,
                                       oldUnreachable);
            bool cacheWholeAllocation =
                gutils->needsCacheWholeAllocation(rmat.first);
            if (cacheWholeAllocation) {
              primalNeededInReverse = true;
            }
            // If in a loop context, maintain the same free behavior, unless
            // caching whole allocation.
            if (!cacheWholeAllocation) {
              eraseIfUnused(call);
              return;
            }
            assert(!unnecessaryValues.count(rmat.first));
            assert(primalNeededInReverse);
          }
        }
      }

      if (gutils->forwardDeallocations.count(&call)) {
        if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        } else
          eraseIfUnused(call);
        return;
      }

      if (gutils->postDominatingFrees.count(&call)) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return;
      }

      llvm::Value *val = getBaseObject(call.getArgOperand(0));
      if (isa<ConstantPointerNull>(val)) {
        llvm::errs() << "removing free of null pointer\n";
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return;
      }

      // TODO HANDLE FREE
      llvm::errs() << "freeing without malloc " << *val << "\n";
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return;
    }

    if (call.hasFnAttr("enzyme_sample")) {
      if (Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ReverseModeGradient)
        return;

      bool constval = gutils->isConstantInstruction(&call);

      if (constval)
        return;

      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

#if LLVM_VERSION_MAJOR >= 14
      auto trace = call.getArgOperand(call.arg_size() - 1);
#else
      auto trace = call.getArgOperand(call.getNumArgOperands() - 1);
#endif
      auto address = call.getArgOperand(0);

      auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
      auto daddress = lookup(gutils->getNewFromOriginal(address), Builder2);

      Value *dchoice;
      if (TR.query(&call)[{-1}].isPossiblePointer()) {
        dchoice = gutils->invertPointerM(&call, Builder2);
      } else {
        dchoice = diffe(&call, Builder2);
      }

#if LLVM_VERSION_MAJOR >= 10
      if (call.hasMetadata("enzyme_gradient_setter")) {
#else
      if (call.getMetadata("enzyme_gradient_setter")) {
#endif
        auto gradient_setter = cast<Function>(
            cast<ValueAsMetadata>(
                call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
                ->getValue());

        TraceUtils::InsertChoiceGradient(
            Builder2, gradient_setter->getFunctionType(), gradient_setter,
            daddress, dchoice, dtrace);
      }

      return;
    }

    if (call.hasFnAttr("enzyme_insert_argument")) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      auto name = call.getArgOperand(0);
      auto arg = call.getArgOperand(1);
      auto trace = call.getArgOperand(2);

      auto gradient_setter = cast<Function>(
          cast<ValueAsMetadata>(
              call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
              ->getValue());

      auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
      auto dname = lookup(gutils->getNewFromOriginal(name), Builder2);
      Value *darg;

      if (TR.query(arg)[{-1}].isPossiblePointer()) {
        darg = gutils->invertPointerM(arg, Builder2);
      } else {
        darg = diffe(arg, Builder2);
      }

      TraceUtils::InsertArgumentGradient(Builder2,
                                         gradient_setter->getFunctionType(),
                                         gradient_setter, dname, darg, dtrace);
      return;
    }

    if (gutils->isConstantInstruction(&call) &&
        gutils->isConstantValue(&call)) {
      bool noFree = Mode == DerivativeMode::ForwardMode;
      noFree |= call.hasFnAttr(Attribute::NoFree);
      if (!noFree && called) {
        noFree |= called->hasFnAttribute(Attribute::NoFree);
      }
      if (!noFree && !EnzymeGlobalActivity) {
        bool mayActiveFree = false;
#if LLVM_VERSION_MAJOR >= 14
        for (unsigned i = 0; i < call.arg_size(); ++i)
#else
        for (unsigned i = 0; i < call.getNumArgOperands(); ++i)
#endif
        {
          Value *a = call.getOperand(i);
          if (gutils->isConstantValue(a))
            continue;
          if (!TR.query(a)[{-1}].isPossiblePointer())
            continue;
          mayActiveFree = true;
          break;
        }
        if (!mayActiveFree)
          noFree = true;
      }
      if (!noFree) {
#if LLVM_VERSION_MAJOR >= 11
        auto callval = call.getCalledOperand();
#else
        auto callval = call.getCalledValue();
#endif
        newCall->setCalledOperand(gutils->Logic.CreateNoFree(callval));
      }
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self));
          eraseIfUnused(call);
          return;
        }
      }

      // If we need this value and it is illegal to recompute it (it writes or
      // may load overwritten data)
      //    Store and reload it
      if (Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode && subretused &&
          (call.mayWriteToMemory() ||
           !gutils->legalRecompute(&call, ValueToValueMapTy(), nullptr))) {
        if (!gutils->unnecessaryIntermediates.count(&call)) {

          std::map<UsageKey, bool> Seen;
          bool primalNeededInReverse = false;
          for (auto pair : gutils->knownRecomputeHeuristic)
            if (!pair.second) {
              if (pair.first == &call) {
                primalNeededInReverse = true;
                break;
              } else {
                Seen[UsageKey(pair.first, ValueType::Primal)] = false;
              }
            }
          if (!primalNeededInReverse) {

            auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                  ? DerivativeMode::ReverseModeGradient
                                  : Mode;
            primalNeededInReverse =
                DifferentialUseAnalysis::is_value_needed_in_reverse<
                    ValueType::Primal>(gutils, &call, minCutMode, Seen,
                                       oldUnreachable);
          }
          if (primalNeededInReverse)
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self));
        }
        eraseIfUnused(call);
        return;
      }

      // If this call may write to memory and is a copy (in the just reverse
      // pass), erase it
      //  Any uses of it should be handled by the case above so it is safe to
      //  RAUW
      if (call.mayWriteToMemory() &&
          (Mode == DerivativeMode::ReverseModeGradient ||
           Mode == DerivativeMode::ForwardModeSplit)) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return;
      }

      // if call does not write memory and isn't used, we can erase it
      if (!call.mayWriteToMemory() && !subretused) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return;
      }

      return;
    }

    return recursivelyHandleSubfunction(call, called, overwritten_args,
                                        shadowReturnUsed, subretType,
                                        subretused);
  }
};
