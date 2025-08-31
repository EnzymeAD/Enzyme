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

#ifndef ENZYME_ADJOINT_GENERATOR_H
#define ENZYME_ADJOINT_GENERATOR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicsX86.h"
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
class AdjointGenerator : public llvm::InstVisitor<AdjointGenerator> {
private:
  // Type of code being generated (forward, reverse, or both)
  const DerivativeMode Mode;

  GradientUtils *const gutils;
  llvm::ArrayRef<DIFFE_TYPE> constant_args;
  DIFFE_TYPE retType;
  TypeResults &TR = gutils->TR;
  std::function<unsigned(llvm::Instruction *, CacheType, llvm::IRBuilder<> &)>
      getIndex;
  const std::map<llvm::CallInst *, std::pair<bool, const std::vector<bool>>>
      overwritten_args_map;
  const AugmentedReturn *augmentedReturn;
  const std::map<llvm::ReturnInst *, llvm::StoreInst *> *replacedReturns;

  const llvm::SmallPtrSetImpl<const llvm::Value *> &unnecessaryValues;
  const llvm::SmallPtrSetImpl<const llvm::Instruction *>
      &unnecessaryInstructions;
  const llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryStores;
  const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable;

public:
  AdjointGenerator(
      DerivativeMode Mode, GradientUtils *gutils,
      llvm::ArrayRef<DIFFE_TYPE> constant_args, DIFFE_TYPE retType,
      std::function<unsigned(llvm::Instruction *, CacheType,
                             llvm::IRBuilder<> &)>
          getIndex,
      const std::map<llvm::CallInst *, std::pair<bool, const std::vector<bool>>>
          overwritten_args_map,
      const AugmentedReturn *augmentedReturn,
      const std::map<llvm::ReturnInst *, llvm::StoreInst *> *replacedReturns,
      const llvm::SmallPtrSetImpl<const llvm::Value *> &unnecessaryValues,
      const llvm::SmallPtrSetImpl<const llvm::Instruction *>
          &unnecessaryInstructions,
      const llvm::SmallPtrSetImpl<const llvm::Instruction *> &unnecessaryStores,
      const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable)
      : Mode(Mode), gutils(gutils), constant_args(constant_args),
        retType(retType), getIndex(getIndex),
        overwritten_args_map(overwritten_args_map),
        augmentedReturn(augmentedReturn), replacedReturns(replacedReturns),
        unnecessaryValues(unnecessaryValues),
        unnecessaryInstructions(unnecessaryInstructions),
        unnecessaryStores(unnecessaryStores), oldUnreachable(oldUnreachable) {
    using namespace llvm;

    assert(TR.getFunction() == gutils->oldFunc);
    for (auto &pair : TR.analyzer->analysis) {
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
      gutils->eraseWithPlaceholder(newi, &I, "_replacementA", erase);
  }

  llvm::Value *MPI_TYPE_SIZE(llvm::Value *DT, llvm::IRBuilder<> &B,
                             llvm::Type *intType) {
    using namespace llvm;

    if (DT->getType()->isIntegerTy())
      DT = B.CreateIntToPtr(DT, getInt8PtrTy(DT->getContext()));

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
    Type *pargs[] = {getInt8PtrTy(DT->getContext()),
                     PointerType::getUnqual(intType)};
    auto FT = FunctionType::get(intType, pargs, false);
    auto alloc = IRBuilder<>(gutils->inversionAllocs).CreateAlloca(intType);
    llvm::Value *args[] = {DT, alloc};
    if (DT->getType() != pargs[0])
      args[0] = B.CreateBitCast(args[0], pargs[0]);
    AttributeList AL;
    AL = AL.addParamAttribute(DT->getContext(), 0,
                              Attribute::AttrKind::ReadOnly);
    AL = addFunctionNoCapture(DT->getContext(), AL, 0);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(DT->getContext(), 1,
                              Attribute::AttrKind::WriteOnly);
    AL = addFunctionNoCapture(DT->getContext(), AL, 1);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NoAlias);
    AL =
        AL.addParamAttribute(DT->getContext(), 1, Attribute::AttrKind::NonNull);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(DT->getContext(), AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
    auto CI = B.CreateCall(
        B.GetInsertBlock()->getParent()->getParent()->getOrInsertFunction(
            "MPI_Type_size", FT, AL),
        args);
#if LLVM_VERSION_MAJOR >= 16
    CI->setOnlyAccessesArgMemory();
#else
    CI->addAttributeAtIndex(AttributeList::FunctionIndex,
                            Attribute::ArgMemOnly);
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
    AL = addFunctionNoCapture(context, AL, 0);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = addFunctionNoCapture(context, AL, 1);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
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
    AL = addFunctionNoCapture(context, AL, 0);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 0, Attribute::AttrKind::NonNull);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::WriteOnly);
    AL = addFunctionNoCapture(context, AL, 1);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NoAlias);
    AL = AL.addParamAttribute(context, 1, Attribute::AttrKind::NonNull);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoUnwind);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoFree);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::NoSync);
    AL = AL.addAttributeAtIndex(context, AttributeList::FunctionIndex,
                                Attribute::AttrKind::WillReturn);
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
    ss << "in Mode: " << to_string(Mode) << "\n";
    ss << "cannot handle unknown instruction\n" << inst;
    IRBuilder<> Builder2(&inst);
    getForwardBuilder(Builder2);
    EmitNoDerivativeError(ss.str(), inst, gutils, Builder2);
    if (!gutils->isConstantValue(&inst)) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ForwardModeSplit)
        setDiffe(&inst,
                 Constant::getNullValue(gutils->getShadowType(inst.getType())),
                 Builder2);
    }
    if (!inst.getType()->isVoidTy()) {
      for (auto &U :
           make_early_inc_range(gutils->getNewFromOriginal(&inst)->uses())) {
        U.set(UndefValue::get(inst.getType()));
      }
    }
    eraseIfUnused(inst, /*erase*/ true, /*check*/ false);
    return;
  }

  // Common function for falling back to the implementation
  // of dual propagation, as available in invertPointerM.
  void forwardModeInvertedPointerFallback(llvm::Instruction &I) {
    using namespace llvm;

    auto found = gutils->invertedPointers.find(&I);
    if (gutils->isConstantValue(&I)) {
      assert(found == gutils->invertedPointers.end());
      return;
    }

    assert(found != gutils->invertedPointers.end());
    auto placeholder = cast<PHINode>(&*found->second);
    gutils->invertedPointers.erase(found);

    if (!DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, &I, Mode, oldUnreachable)) {
      gutils->erase(placeholder);
      return;
    }

    IRBuilder<> Builder2(&I);
    getForwardBuilder(Builder2);

    auto toset = gutils->invertPointerM(&I, Builder2, /*nullShadow*/ true);

    assert(toset != placeholder);

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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
      forwardModeInvertedPointerFallback(I);
      return;
    }
    default:
      return;
    }
  }

  void visitICmpInst(llvm::ICmpInst &I) { eraseIfUnused(I); }

  void visitFCmpInst(llvm::FCmpInst &I) { eraseIfUnused(I); }

  void visitLoadLike(llvm::Instruction &I, llvm::MaybeAlign alignment,
                     bool constantval, llvm::Value *mask = nullptr,
                     llvm::Value *orig_maskInit = nullptr) {
    using namespace llvm;

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto LoadSize = (DL.getTypeSizeInBits(I.getType()) + 1) / 8;

    assert(Mode == DerivativeMode::ForwardMode ||
           Mode == DerivativeMode::ForwardModeError || gutils->can_modref_map);
    assert(Mode == DerivativeMode::ForwardMode ||
           Mode == DerivativeMode::ForwardModeError ||
           gutils->can_modref_map->find(&I) != gutils->can_modref_map->end());
    bool can_modref = (Mode == DerivativeMode::ForwardMode ||
                       Mode == DerivativeMode::ForwardModeError)
                          ? false
                          : gutils->can_modref_map->find(&I)->second;

    constantval |= gutils->isConstantValue(&I);

    Type *type = gutils->getShadowType(I.getType());
    (void)type;

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
      std::string str;
      raw_string_ostream ss(str);
      ss << "Cannot deduce type of load " << I;
      auto ET = I.getType();
      if (looseTypeAnalysis || true) {
        vd = defaultTypeTreeForLLVM(ET, &I);
        ss << ", assumed " << vd.str() << "\n";
        EmitWarning("CannotDeduceType", I, ss.str());
        goto known;
      }
      EmitNoTypeError(str, I, gutils, BuilderZ);
    known:;
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      if (!constantval) {
        auto found = gutils->invertedPointers.find(&I);
        assert(found != gutils->invertedPointers.end());
        Instruction *placeholder = cast<Instruction>(&*found->second);
        assert(placeholder->getType() == type);
        gutils->invertedPointers.erase(found);

        // only make shadow where caching needed
        if (!DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Shadow>(gutils, &I, Mode, oldUnreachable)) {
          gutils->erase(placeholder);
          return;
        }

        if (can_modref) {
          if (vd[{-1}].isPossiblePointer()) {
            Value *newip = gutils->cacheForReverse(
                BuilderZ, placeholder,
                getIndex(&I, CacheType::Shadow, BuilderZ));
            assert(newip->getType() == type);
            gutils->invertedPointers.insert(std::make_pair(
                (const Value *)&I, InvertedPointerVH(gutils, newip)));
          } else {
            gutils->erase(placeholder);
          }
        } else {
          Value *newip = gutils->invertPointerM(&I, BuilderZ);
          if (gutils->runtimeActivity && vd[{-1}].isFloat()) {
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
              QueryType::Shadow>(gutils, &I, Mode, oldUnreachable);

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
                      QueryType::Shadow>(gutils, &I,
                                         DerivativeMode::ReverseModeGradient,
                                         oldUnreachable)) {
                gutils->cacheForReverse(
                    BuilderZ, newip, getIndex(&I, CacheType::Shadow, BuilderZ));
              }
              placeholder->replaceAllUsesWith(newip);
              gutils->erase(placeholder);
              gutils->invertedPointers.insert(std::make_pair(
                  (const Value *)&I, InvertedPointerVH(gutils, newip)));
            }
            break;
          }
          case DerivativeMode::ForwardModeSplit:
          case DerivativeMode::ForwardMode:
          case DerivativeMode::ForwardModeError: {
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
                    BuilderZ, placeholder,
                    getIndex(&I, CacheType::Shadow, BuilderZ));
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
        Mode != DerivativeMode::ForwardModeError &&
        !gutils->knownRecomputeHeuristic.count(&I) && can_modref &&
        !gutils->unnecessaryIntermediates.count(&I)) {
      // we can pre initialize all the knownRecomputeHeuristic values to false
      // (not needing) as we may assume that minCutCache already preserves
      // everything it requires.
      std::map<UsageKey, bool> Seen;
      bool primalNeededInReverse = false;
      for (auto pair : gutils->knownRecomputeHeuristic)
        if (!pair.second) {
          Seen[UsageKey(pair.first, QueryType::Primal)] = false;
          if (pair.first == &I)
            primalNeededInReverse = true;
        }
      auto cacheMode = (Mode == DerivativeMode::ReverseModePrimal)
                           ? DerivativeMode::ReverseModeGradient
                           : Mode;
      primalNeededInReverse |=
          DifferentialUseAnalysis::is_value_needed_in_reverse<
              QueryType::Primal>(gutils, &I, cacheMode, Seen, oldUnreachable);
      if (primalNeededInReverse) {
        inst = gutils->cacheForReverse(BuilderZ, newi,
                                       getIndex(&I, CacheType::Self, BuilderZ));
        (void)inst;
        assert(inst->getType() == type);

        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit ||
            Mode == DerivativeMode::ForwardModeError) {
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
      case DerivativeMode::ForwardMode:
      case DerivativeMode::ForwardModeError: {
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

        if (prediff && mask) {
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
            std::string str;
            raw_string_ostream ss(str);
            ss << "Cannot deduce type of load " << I;
            ss << " vd:" << vd.str() << " start:" << start << " size: " << size
               << " dt:" << dt.str() << "\n";
            EmitNoTypeError(str, I, gutils, BuilderZ);
            continue;
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

    auto alignment = LI.getAlign();
    auto &DL = gutils->newFunc->getParent()->getDataLayout();

    bool constantval = parseTBAA(LI, DL, nullptr)[{-1}].isIntegral();
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
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ForwardModeSplit) {
        auto rule = [&](Value *ptr, Value *dif) -> Value * {
          if (dif == nullptr)
            dif = Constant::getNullValue(I.getType());
          if (!gutils->isConstantInstruction(&I)) {
            if (!ptr) {
              if (!gutils->runtimeActivity) {
                std::string str;
                raw_string_ostream ss(str);
                ss << "Mismatched activity for: " << I
                   << " const val: " << *I.getPointerOperand();
                Value *diff = Constant::getNullValue(dif->getType());
                if (CustomErrorHandler)
                  diff = unwrap(CustomErrorHandler(
                      str.c_str(), wrap(&I), ErrorType::MixedActivityError,
                      gutils, wrap(orig_val), wrap(&BuilderZ)));
                else
                  EmitWarning("MixedActivityError", I, ss.str());
                return diff;
              }
            }
            AtomicRMWInst *rmw = nullptr;
            rmw = BuilderZ.CreateAtomicRMW(I.getOperation(), ptr, dif,
                                           I.getAlign(), I.getOrdering(),
                                           I.getSyncScopeID());
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

            dif1->setAlignment(I.getAlign());
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
    EmitNoDerivativeError(ss.str(), I, gutils, BuilderZ);
    if (!gutils->isConstantValue(&I)) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ForwardModeSplit)
        setDiffe(&I, Constant::getNullValue(gutils->getShadowType(I.getType())),
                 BuilderZ);
    }
    if (!I.getType()->isVoidTy()) {
      for (auto &U :
           make_early_inc_range(gutils->getNewFromOriginal(&I)->uses())) {
        U.set(UndefValue::get(I.getType()));
      }
    }
    eraseIfUnused(I, /*erase*/ true, /*check*/ false);
    return;
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
    auto align = SI.getAlign();

    visitCommonStore(SI, SI.getPointerOperand(), SI.getValueOperand(), align,
                     SI.isVolatile(), SI.getOrdering(), SI.getSyncScopeID(),
                     /*mask=*/nullptr);

    bool forceErase = false;
    if (Mode == DerivativeMode::ReverseModeGradient) {
      // Since we won't redo the store in the reverse pass, do not
      // force the write barrier.
      forceErase = true;
      for (const auto &pair : gutils->rematerializableAllocations) {
        // However, if we are rematerailizing the allocationa and not
        // inside the loop level rematerialization, we do still need the
        // reverse passes ``fake primal'' store and therefore write barrier
        if (pair.second.stores.count(&SI) &&
            (!pair.second.LI || !pair.second.LI->contains(&SI))) {
          forceErase = false;
        }
      }
    }
    if (forceErase)
      eraseIfUnused(SI, /*erase*/ true, /*check*/ false);
    else
      eraseIfUnused(SI);
  }

  void visitCommonStore(llvm::Instruction &I, llvm::Value *orig_ptr,
                        llvm::Value *orig_val, llvm::MaybeAlign prevalign,
                        bool isVolatile, llvm::AtomicOrdering ordering,
                        llvm::SyncScope::ID syncScope, llvm::Value *mask) {
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
                       parseTBAA(I, DL, nullptr)[{-1}].isIntegral();

    IRBuilder<> BuilderZ(NewI);
    BuilderZ.setFastMathFlags(getFast());

    // TODO allow recognition of other types that could contain pointers [e.g.
    // {void*, void*} or <2 x i64> ]
    auto storeSize = (DL.getTypeSizeInBits(valType) + 7) / 8;

    auto vd = TR.query(orig_ptr).Lookup(storeSize, DL);

    if (!vd.isKnown()) {
      std::string str;
      raw_string_ostream ss(str);
      ss << "Cannot deduce type of store " << I;
      if (looseTypeAnalysis || true) {
        vd = defaultTypeTreeForLLVM(valType, &I);
        ss << ", assumed " << vd.str() << "\n";
        EmitWarning("CannotDeduceType", I, ss.str());
        goto known;
      }
      EmitNoTypeError(str, I, gutils, BuilderZ);
      return;
    known:;
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {

      auto dt = vd[{-1}];
      // Only need the full type in forward mode, if storing a constant
      // and therefore may need to zero some floats.
      if (constantval)
        for (size_t i = 0; i < storeSize; ++i) {
          bool Legal = true;
          dt.checkedOrIn(vd[{(int)i}], /*PointerIntSame*/ true, Legal);
          if (!Legal) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Cannot deduce single type of store " << I << vd.str()
               << " size: " << storeSize;
            EmitNoTypeError(str, I, gutils, BuilderZ);
            return;
          }
        }

      Value *diff = nullptr;
      if (!gutils->runtimeActivity && constantval) {
        if (dt.isPossiblePointer() && vd[{-1, -1}] != BaseType::Integer) {
          if (!isa<UndefValue>(orig_val) &&
              !isa<ConstantPointerNull>(orig_val)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << I
               << " const val: " << *orig_val;
            if (CustomErrorHandler)
              diff = unwrap(CustomErrorHandler(
                  str.c_str(), wrap(&I), ErrorType::MixedActivityError, gutils,
                  wrap(orig_val), wrap(&BuilderZ)));
            else
              EmitWarning("MixedActivityError", I, ss.str());
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

      gutils->setPtrDiffe(&I, orig_ptr, diff, BuilderZ, prevalign, 0, storeSize,
                          isVolatile, ordering, syncScope, mask, prevNoAlias,
                          prevScopes);

      return;
    }

    unsigned start = 0;

    while (1) {
      unsigned nextStart = storeSize;

      auto dt = vd[{-1}];
      for (size_t i = start; i < storeSize; ++i) {
        auto nex = vd[{(int)i}];
        if ((nex == BaseType::Anything && dt.isFloat()) ||
            (dt == BaseType::Anything && nex.isFloat())) {
          nextStart = i;
          break;
        }
        bool Legal = true;
        dt.checkedOrIn(nex, /*PointerIntSame*/ true, Legal);
        if (!Legal) {
          nextStart = i;
          break;
        }
      }
      unsigned size = nextStart - start;
      if (!dt.isKnown()) {

        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of store " << I << vd.str()
           << " start: " << start << " size: " << size
           << " storeSize: " << storeSize;
        EmitNoTypeError(str, I, gutils, BuilderZ);
        break;
      }

      MaybeAlign align;
      if (prevalign) {
        if (start % prevalign->value() == 0)
          align = prevalign;
        else
          align = Align(1);
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
                Constant::getNullValue(gutils->getShadowType(valType)),
                Builder2, align, start, size, isVolatile, ordering, syncScope,
                mask, prevNoAlias, prevScopes);
          } else {
            Value *diff;
            Value *maskL = mask;
            if (!mask) {
              Value *dif1Ptr =
                  lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

              size_t idx = 0;
              auto rule = [&](Value *dif1Ptr) {
                LoadInst *dif1 =
                    Builder2.CreateLoad(valType, dif1Ptr, isVolatile);
                if (align)
                  dif1->setAlignment(*align);
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
              maskL = lookup(mask, Builder2);
              Type *tys[] = {valType, orig_ptr->getType()};
              auto F = getIntrinsicDeclaration(gutils->oldFunc->getParent(),
                                               Intrinsic::masked_load, tys);
              Value *alignv =
                  ConstantInt::get(Type::getInt32Ty(mask->getContext()),
                                   align ? align->value() : 0);
              Value *ip =
                  lookup(gutils->invertPointerM(orig_ptr, Builder2), Builder2);

              auto rule = [&](Value *ip) {
                Value *args[] = {ip, alignv, maskL,
                                 Constant::getNullValue(valType)};
                diff = Builder2.CreateCall(F, args);
                return diff;
              };

              diff = applyChainRule(valType, Builder2, rule, ip);
            }

            gutils->setPtrDiffe(
                &I, orig_ptr,
                Constant::getNullValue(gutils->getShadowType(valType)),
                Builder2, align, start, size, isVolatile, ordering, syncScope,
                mask, prevNoAlias, prevScopes);
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_val, diff, Builder2, FT, start, size, {},
                             maskL);
          }
          break;
        }
        case DerivativeMode::ForwardModeSplit:
        case DerivativeMode::ForwardModeError:
        case DerivativeMode::ForwardMode: {
          IRBuilder<> Builder2(&I);
          getForwardBuilder(Builder2);

          Type *diffeTy = gutils->getShadowType(valType);

          Value *diff = constantval
                            ? Constant::getNullValue(diffeTy)
                            : gutils->invertPointerM(orig_val, Builder2,
                                                     /*nullShadow*/ true);
          gutils->setPtrDiffe(&I, orig_ptr, diff, Builder2, align, start, size,
                              isVolatile, ordering, syncScope, mask,
                              prevNoAlias, prevScopes);

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
                continue;
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
            Mode == DerivativeMode::ForwardMode ||
            Mode == DerivativeMode::ForwardModeError) {

          Value *valueop = nullptr;

          if (constantval) {
            if (!gutils->runtimeActivity) {
              if (dt.isPossiblePointer() && vd[{-1, -1}] != BaseType::Integer) {
                if (!isa<UndefValue>(orig_val) &&
                    !isa<ConstantPointerNull>(orig_val)) {
                  std::string str;
                  raw_string_ostream ss(str);
                  ss << "Mismatched activity for: " << I
                     << " const val: " << *orig_val;
                  if (CustomErrorHandler)
                    valueop = unwrap(CustomErrorHandler(
                        str.c_str(), wrap(&I), ErrorType::MixedActivityError,
                        gutils, wrap(orig_val), wrap(&BuilderZ)));
                  else
                    EmitWarning("MixedActivityError", I, ss.str());
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
          gutils->setPtrDiffe(&I, orig_ptr, valueop, BuilderZ, align, start,
                              size, isVolatile, ordering, syncScope, mask,
                              prevNoAlias, prevScopes);
        }
      }

      if (nextStart == storeSize)
        break;
      start = nextStart;
    }
  }

  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep) {
    eraseIfUnused(gep);
    switch (Mode) {
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
        if (!FT && looseTypeAnalysis) {
          if (auto ET = I.getSrcTy()->getScalarType())
            if (ET->isFPOrFPVectorTy()) {
              FT = ET;
              EmitWarning("CannotDeduceType", I,
                          "failed to deduce adding type of cast ", I,
                          " assumed ", FT, " from src");
            }
        }
        if (!FT && looseTypeAnalysis) {
          if (auto ET = I.getDestTy()->getScalarType())
            if (ET->isFPOrFPVectorTy()) {
              FT = ET;
              EmitWarning("CannotDeduceType", I,
                          "failed to deduce adding type of cast ", I,
                          " assumed ", FT, " from dst");
            }
        }
        if (!FT) {
          if (TR.query(orig_op0)[{-1}] == BaseType::Integer &&
              TR.query(&I)[{-1}] == BaseType::Integer)
            return;
          if (looseTypeAnalysis) {
            if (auto ET = I.getSrcTy()->getScalarType())
              if (ET->isIntOrIntVectorTy()) {
                EmitWarning("CannotDeduceType", I,
                            "failed to deduce adding type of cast ", I,
                            " assumed integral from src");
                return;
              }
          }
          std::string str;
          raw_string_ostream ss(str);
          ss << "Cannot deduce adding type (cast) of " << I;
          EmitNoTypeError(str, I, gutils, Builder2);
        }

        if (FT) {

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
              EmitNoDerivativeError(ss.str(), I, gutils, Builder2);
              return (llvm::Value *)UndefValue::get(op0->getType());
            }
          };

          Value *dif = diffe(&I, Builder2);
          Value *diff = applyChainRule(op0->getType(), Builder2, rule, dif);

          addToDiffe(orig_op0, diff, Builder2, FT);
        }
      }

      Type *diffTy = gutils->getShadowType(I.getType());
      setDiffe(&I, Constant::getNullValue(diffTy), Builder2);

      break;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
          gutils->OrigLI->getLoopFor(P0->getParent())->getLoopLatches(Latches);
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
                  inc = Builder2.CreateVectorSplat(VTy->getElementCount(), inc);
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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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

        size_t size = 1;
        if (EEI.getType()->isSized())
          size =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EEI.getType()) +
               7) /
              8;
        auto diff = diffe(&EEI, Builder2);
        if (gutils->getWidth() == 1) {
          Value *sv[] = {gutils->getNewFromOriginal(EEI.getIndexOperand())};
          ((DiffeGradientUtils *)gutils)
              ->addToDiffe(orig_vec, diff, Builder2, TR.addingType(size, &EEI),
                           sv);
        } else {
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            Value *sv[] = {nullptr,
                           gutils->getNewFromOriginal(EEI.getIndexOperand())};
            sv[0] = ConstantInt::get(sv[1]->getType(), i);
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_vec, gutils->extractMeta(Builder2, diff, i),
                             Builder2, TR.addingType(size, &EEI), sv);
          }
        }
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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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

      if (!gutils->isConstantValue(orig_op0)) {
        if (gutils->getWidth() == 1) {
          addToDiffe(
              orig_op0,
              Builder2.CreateInsertElement(
                  dif1,
                  Constant::getNullValue(gutils->getShadowType(op1->getType())),
                  lookup(op2, Builder2)),
              Builder2, TR.addingType(size0, orig_op0));
        } else {
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            Value *sv[] = {ConstantInt::get(op2->getType(), i)};
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_op0,
                             Builder2.CreateInsertElement(
                                 gutils->extractMeta(Builder2, dif1, i),
                                 Constant::getNullValue(op1->getType()),
                                 lookup(op2, Builder2)),
                             Builder2, TR.addingType(size0, orig_op0), sv);
          }
        }
      }

      if (!gutils->isConstantValue(orig_op1)) {
        if (gutils->getWidth() == 1) {
          addToDiffe(orig_op1,
                     Builder2.CreateExtractElement(dif1, lookup(op2, Builder2)),
                     Builder2, TR.addingType(size1, orig_op1));
        } else {
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            Value *sv[] = {ConstantInt::get(op2->getType(), i)};
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_op1,
                             Builder2.CreateExtractElement(
                                 gutils->extractMeta(Builder2, dif1, i),
                                 lookup(op2, Builder2)),
                             Builder2, TR.addingType(size1, orig_op1), sv);
          }
        }
      }

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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
      auto count =
          cast<VectorType>(SVI.getOperand(0)->getType())->getElementCount();
      assert(!count.isScalable());
      size_t l1 = count.getKnownMinValue();
      uint64_t instidx = 0;

      for (size_t idx : SVI.getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx - l1);

        if (!gutils->isConstantValue(SVI.getOperand(opnum))) {
          size_t size = 1;
          if (SVI.getOperand(opnum)->getType()->isSized())
            size = (gutils->newFunc->getParent()
                        ->getDataLayout()
                        .getTypeSizeInBits(SVI.getOperand(opnum)->getType()) +
                    7) /
                   8;
          if (gutils->getWidth() == 1) {
            Value *sv[] = {
                ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx)};
            Value *toadd = Builder2.CreateExtractElement(loaded, instidx);
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(SVI.getOperand(opnum), toadd, Builder2,
                             TR.addingType(size, SVI.getOperand(opnum)), sv);
          } else {
            for (size_t i = 0; i < gutils->getWidth(); i++) {
              Value *sv[] = {
                  ConstantInt::get(Type::getInt32Ty(SVI.getContext()), i),
                  ConstantInt::get(Type::getInt32Ty(SVI.getContext()), opidx)};
              Value *toadd = Builder2.CreateExtractElement(
                  GradientUtils::extractMeta(Builder2, loaded, i), instidx);
              ((DiffeGradientUtils *)gutils)
                  ->addToDiffe(SVI.getOperand(opnum), toadd, Builder2,
                               TR.addingType(size, SVI.getOperand(opnum)), sv);
            }
          }
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
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeError: {
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
        size_t storeSize = 1;
        if (EVI.getType()->isSized())
          storeSize =
              (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                   EVI.getType()) +
               7) /
              8;

        unsigned start = 0;
        auto vd = TR.query(&EVI);

        while (1) {
          unsigned nextStart = storeSize;

          auto dt = vd[{-1}];
          for (size_t i = start; i < storeSize; ++i) {
            auto nex = vd[{(int)i}];
            if ((nex == BaseType::Anything && dt.isFloat()) ||
                (dt == BaseType::Anything && nex.isFloat())) {
              nextStart = i;
              break;
            }
            bool Legal = true;
            dt.checkedOrIn(nex, /*PointerIntSame*/ true, Legal);
            if (!Legal) {
              nextStart = i;
              break;
            }
          }
          unsigned size = nextStart - start;
          if (!dt.isKnown()) {
            bool found = false;
            if (looseTypeAnalysis) {
              if (EVI.getType()->isFPOrFPVectorTy()) {
                dt = ConcreteType(EVI.getType()->getScalarType());
                found = true;
              } else if (EVI.getType()->isIntOrIntVectorTy() ||
                         EVI.getType()->isPointerTy()) {
                dt = BaseType::Integer;
                found = true;
              }
            }
            if (!found) {
              std::string str;
              raw_string_ostream ss(str);
              ss << "Cannot deduce type of extract " << EVI << vd.str()
                 << " start: " << start << " size: " << size
                 << " extractSize: " << storeSize;
              EmitNoTypeError(str, EVI, gutils, Builder2);
            }
          }
          if (auto FT = dt.isFloat())
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_op0, prediff, Builder2, FT, start, size, sv,
                             nullptr, /*ignoreFirstSlicesToDiff*/ sv.size());

          if (nextStart == storeSize)
            break;
          start = nextStart;
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
        Mode == DerivativeMode::ForwardModeSplit ||
        Mode == DerivativeMode::ForwardModeError) {
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
    case DerivativeMode::ForwardModeError:
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
        auto TT = TR.query(orig_inserted);

        unsigned start = 0;
        Value *dindex = nullptr;

        while (1) {
          unsigned nextStart = size0;

          auto dt = TT[{-1}];
          for (size_t i = start; i < size0; ++i) {
            auto nex = TT[{(int)i}];
            if ((nex == BaseType::Anything && dt.isFloat()) ||
                (dt == BaseType::Anything && nex.isFloat())) {
              nextStart = i;
              break;
            }
            bool Legal = true;
            dt.checkedOrIn(nex, /*PointerIntSame*/ true, Legal);
            if (!Legal) {
              nextStart = i;
              break;
            }
          }
          Type *flt = dt.isFloat();
          if (!dt.isKnown()) {
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
              std::string str;
              raw_string_ostream ss(str);
              ss << "Cannot deduce type of insertvalue ins " << IVI
                 << " size: " << size0 << " TT: " << TT.str();
              EmitNoTypeError(str, IVI, gutils, Builder2);
            }
          }

          if (flt) {
            if (!dindex) {
              auto rule = [&](Value *prediff) {
                return Builder2.CreateExtractValue(prediff, IVI.getIndices());
              };
              auto prediff = diffe(&IVI, Builder2);
              dindex = applyChainRule(orig_inserted->getType(), Builder2, rule,
                                      prediff);
            }

            auto TT = TR.query(orig_inserted);

            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_inserted, dindex, Builder2, flt, start,
                             nextStart - start);
          }
          if (nextStart == size0)
            break;
          start = nextStart;
        }
      }

      size_t size1 = 1;
      if (orig_agg->getType()->isSized())
        size1 =
            (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                 orig_agg->getType()) +
             7) /
            8;

      if (!gutils->isConstantValue(orig_agg)) {

        auto TT = TR.query(orig_agg);

        unsigned start = 0;

        Value *dindex = nullptr;

        while (1) {
          unsigned nextStart = size1;

          auto dt = TT[{-1}];
          for (size_t i = start; i < size1; ++i) {
            auto nex = TT[{(int)i}];
            if ((nex == BaseType::Anything && dt.isFloat()) ||
                (dt == BaseType::Anything && nex.isFloat())) {
              nextStart = i;
              break;
            }
            bool Legal = true;
            dt.checkedOrIn(nex, /*PointerIntSame*/ true, Legal);
            if (!Legal) {
              nextStart = i;
              break;
            }
          }
          Type *flt = dt.isFloat();
          if (!dt.isKnown()) {
            bool found = false;
            if (looseTypeAnalysis) {
              if (orig_agg->getType()->isFPOrFPVectorTy()) {
                flt = orig_agg->getType()->getScalarType();
                found = true;
              } else if (orig_agg->getType()->isIntOrIntVectorTy() ||
                         orig_agg->getType()->isPointerTy()) {
                flt = nullptr;
                found = true;
              }
            }
            if (!found) {
              std::string str;
              raw_string_ostream ss(str);
              ss << "Cannot deduce type of insertvalue agg " << IVI
                 << " start: " << start << " size: " << size1
                 << " TT: " << TT.str();
              EmitNoTypeError(str, IVI, gutils, Builder2);
            }
          }

          if (flt) {
            if (!dindex) {
              auto rule = [&](Value *prediff) {
                return Builder2.CreateInsertValue(
                    prediff, Constant::getNullValue(orig_inserted->getType()),
                    IVI.getIndices());
              };
              auto prediff = diffe(&IVI, Builder2);
              dindex =
                  applyChainRule(orig_agg->getType(), Builder2, rule, prediff);
            }
            ((DiffeGradientUtils *)gutils)
                ->addToDiffe(orig_agg, dindex, Builder2, flt, start,
                             nextStart - start);
          }
          if (nextStart == size1)
            break;
          start = nextStart;
        }
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
          } else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u)) {
            activeUses.push_back(cast<Instruction>(u));
          }
        }
        if (activeUses.size() == 1 && activeUses[0] == &BO &&
            gutils->getContext(gutils->getNewFromOriginal(P0->getParent()),
                               lc) &&
            gutils->getNewFromOriginal(P0->getParent()) == lc.header) {
          SmallVector<BasicBlock *, 1> Latches;
          gutils->OrigLI->getLoopFor(P0->getParent())->getLoopLatches(Latches);
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

            Value *dif1 = nullptr;
            Value *idiff = diffe(&BO, Builder2);

            Type *addingType = BO.getType();

            if (!gutils->isConstantValue(orig_op1)) {
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
                  if (gutils->strongZero) {
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
                  if (gutils->strongZero) {
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
    case DerivativeMode::ForwardModeError:
    case DerivativeMode::ForwardModeSplit:
      createBinaryOperatorDual(BO);
      break;
    case DerivativeMode::ReverseModePrimal:
      return;
    }
  }

  void createBinaryOperatorAdjoint(llvm::BinaryOperator &BO) {
    if (gutils->isConstantInstruction(&BO)) {
      return;
    }
    using namespace llvm;

    IRBuilder<> Builder2(&BO);
    getReverseBuilder(Builder2);

    Value *orig_op0 = BO.getOperand(0);
    Value *orig_op1 = BO.getOperand(1);

    Value *dif0 = nullptr;
    Value *dif1 = nullptr;
    Value *idiff = diffe(&BO, Builder2);

    Type *addingType = BO.getType();

    switch (BO.getOpcode()) {
    case Instruction::LShr: {
      if (!gutils->isConstantValue(orig_op0)) {
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

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
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

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
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

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
            FT = VectorType::get(FT, CV->getType()->getElementCount());
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
            FT = VectorType::get(FT, CV->getType()->getElementCount());
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
                    checkedMul(gutils->strongZero, Builder2,
                               Builder2.CreateBitCast(idiff, FT),
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
        llvm::errs()
            << "warning: binary operator is integer and assumed constant: "
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
      EmitNoDerivativeError(ss.str(), BO, gutils, Builder2);
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

    switch (BO.getOpcode()) {
    case Instruction::And: {
      // If & against 0b10000000000 and a float the result is 0
      auto &dl = gutils->oldFunc->getParent()->getDataLayout();
      auto size = dl.getTypeSizeInBits(BO.getType()) / 8;
      Type *diffTy = gutils->getShadowType(BO.getType());

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
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

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
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

      auto FT = TR.query(&BO).IsAllFloat(size, dl);
      auto eFT = FT;
      // If & against 0b10000000000 and a float the result is a float
      if (FT)
        for (int i = 0; i < 2; ++i) {
          auto CI = dyn_cast<ConstantInt>(BO.getOperand(i));
          if (auto CV = dyn_cast<ConstantVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
            FT = VectorType::get(FT, CV->getType()->getElementCount());
          }
          if (auto CV = dyn_cast<ConstantDataVector>(BO.getOperand(i))) {
            CI = dyn_cast_or_null<ConstantInt>(CV->getSplatValue());
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
                    checkedMul(gutils->strongZero, Builder2,
                               Builder2.CreateBitCast(difi, FT),
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
      auto rval = EmitNoDerivativeError(ss.str(), BO, gutils, Builder2);
      if (!rval)
        rval = Constant::getNullValue(gutils->getShadowType(BO.getType()));
      auto ifound = gutils->invertedPointers.find(&BO);
      if (!gutils->isConstantValue(&BO)) {
        if (ifound != gutils->invertedPointers.end()) {
          auto placeholder = cast<PHINode>(&*ifound->second);
          gutils->invertedPointers.erase(ifound);
          gutils->replaceAWithB(placeholder, rval);
          gutils->erase(placeholder);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&BO, InvertedPointerVH(gutils, rval)));
        }
      } else {
        assert(ifound == gutils->invertedPointers.end());
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

    bool activeValToSet = !gutils->isConstantValue(orig_op1);
    if (activeValToSet)
      if (auto CI = dyn_cast<ConstantInt>(orig_op1))
        if (CI->isZero())
          activeValToSet = false;
    if (activeValToSet) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "couldn't handle non constant inst in memset to "
            "propagate differential to\n"
         << MS;
      EmitNoDerivativeError(ss.str(), MS, gutils, BuilderZ);
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
      Value *op1 = gutils->getNewFromOriginal(MS.getArgOperand(1));
      Value *op2 = gutils->getNewFromOriginal(MS.getArgOperand(2));
      Value *op3 = nullptr;
      if (3 < MS.arg_size()) {
        op3 = gutils->getNewFromOriginal(MS.getOperand(3));
      }

      auto Defs =
          gutils->getInvertedBundles(&MS,
                                     {ValueType::Shadow, ValueType::Primal,
                                      ValueType::Primal, ValueType::Primal},
                                     BuilderZ, /*lookup*/ false);

      auto funcName = getFuncNameFromCall(&MS);
      applyChainRule(
          BuilderZ,
          [&](Value *op0) {
            SmallVector<Value *, 4> args = {op0, op1, op2};
            if (op3)
              args.push_back(op3);

            CallInst *cal;
            if (startsWith(funcName, "memset_pattern") ||
                startsWith(funcName, "llvm.experimental.memset"))
              cal = Builder2.CreateMemSet(
                  op0, ConstantInt::get(Builder2.getInt8Ty(), 0), op2, {});
            else
              cal = BuilderZ.CreateCall(MS.getCalledFunction(), args, Defs);

            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            cal->copyMetadata(MS, ToCopy2);
            if (auto m = hasMetadata(&MS, "enzyme_zerostack"))
              cal->setMetadata("enzyme_zerostack", m);

            if (startsWith(funcName, "memset_pattern") ||
                startsWith(funcName, "llvm.experimental.memset")) {
              AttributeList NewAttrs;
              for (auto idx :
                   {AttributeList::ReturnIndex, AttributeList::FunctionIndex,
                    AttributeList::FirstArgIndex})
                for (auto attr : MS.getAttributes().getAttributes(idx))
                  NewAttrs =
                      NewAttrs.addAttributeAtIndex(MS.getContext(), idx, attr);
              cal->setAttributes(NewAttrs);
            } else
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

    // Offsets of the form Optional<floating type>, segment start, segment size
    std::vector<std::tuple<Type *, size_t, size_t>> toIterate;

    // Special handling mechanism to bypass TA limitations by supporting
    // arbitrary sized types.
    if (auto MD = hasMetadata(&MS, "enzyme_truetype")) {
      toIterate = parseTrueType(MD, Mode, false);
    } else {
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      auto vd = TR.query(MS.getOperand(0)).Data0().ShiftIndices(DL, 0, size, 0);

      if (!vd.isKnownPastPointer()) {
        // If unknown type results, and zeroing known undef allocation, consider
        // integers
        if (auto CI = dyn_cast<ConstantInt>(MS.getOperand(1)))
          if (CI->isZero()) {
            auto root = getBaseObject(MS.getOperand(0));
            bool writtenTo = false;
            bool undefMemory =
                isa<AllocaInst>(root) || isAllocationCall(root, gutils->TLI);
            if (auto arg = dyn_cast<Argument>(root))
              if (arg->hasStructRetAttr())
                undefMemory = true;
            if (undefMemory) {
              Instruction *cur = MS.getPrevNode();
              while (cur) {
                if (cur == root)
                  break;
                if (auto MCI = dyn_cast<ConstantInt>(MS.getOperand(2))) {
                  if (auto II = dyn_cast<IntrinsicInst>(cur)) {
                    if (II->getCalledFunction()->getName() ==
                        "llvm.enzyme.lifetime_start") {
                      if (getBaseObject(II->getOperand(1)) == root) {
                        if (auto CI2 =
                                dyn_cast<ConstantInt>(II->getOperand(0))) {
                          if (MCI->getValue().ule(CI2->getValue()))
                            break;
                        }
                      }
                      cur = cur->getPrevNode();
                      continue;
                    }
                    // If the start of the lifetime for more memory than being
                    // memset, its valid.
                    if (II->getIntrinsicID() == Intrinsic::lifetime_start) {
                      if (getBaseObject(II->getOperand(1)) == root) {
                        if (auto CI2 =
                                dyn_cast<ConstantInt>(II->getOperand(0))) {
                          if (MCI->getValue().ule(CI2->getValue()))
                            break;
                        }
                      }
                      cur = cur->getPrevNode();
                      continue;
                    }
                  }
                }
                if (cur->mayWriteToMemory()) {
                  writtenTo = true;
                  break;
                }
                cur = cur->getPrevNode();
              }

              if (!writtenTo) {
                vd = TypeTree(BaseType::Pointer);
                vd.insert({-1}, BaseType::Integer);
              }
            }
          }
      }

      if (!vd.isKnownPastPointer()) {
        // If unknown type results, consider the intersection of all incoming.
        if (isa<PHINode>(MS.getOperand(0)) ||
            isa<SelectInst>(MS.getOperand(0))) {
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
              for (size_t i = 0, end = PN->getNumIncomingValues(); i < end;
                   i++) {
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
#if LLVM_VERSION_MAJOR < 17
          if (auto CI = dyn_cast<CastInst>(MS.getOperand(0))) {
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
          }
#endif
          if (auto gep = dyn_cast<GetElementPtrInst>(MS.getOperand(0))) {
            if (auto AT = dyn_cast<ArrayType>(gep->getSourceElementType())) {
              if (AT->getElementType()->isIntegerTy()) {
                vd = TypeTree(BaseType::Integer).Only(0, &MS);
                goto known;
              }
            }
          }
          EmitWarning("CannotDeduceType", MS,
                      "failed to deduce type of memset ", MS);
          vd = TypeTree(BaseType::Pointer).Only(0, &MS);
          goto known;
        }
        std::string str;
        raw_string_ostream ss(str);
        ss << "Cannot deduce type of memset " << MS;
        EmitNoTypeError(str, MS, gutils, BuilderZ);
        return;
      }
    known:;
      {
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
          toIterate.emplace_back(dt.isFloat(), start, nextStart - start);

          if (nextStart == size)
            break;
          start = nextStart;
        }
      }
    }

#if 0
    unsigned dstalign = dstAlign.valueOrOne().value();
    unsigned srcalign = srcAlign.valueOrOne().value();
#endif

    Value *op1 = gutils->getNewFromOriginal(MS.getArgOperand(1));
    Value *new_size = gutils->getNewFromOriginal(MS.getArgOperand(2));
    Value *op3 = nullptr;
    if (3 < MS.arg_size()) {
      op3 = gutils->getNewFromOriginal(MS.getOperand(3));
    }

    for (auto &&[secretty_ref, seg_start_ref, seg_size_ref] : toIterate) {
      auto secretty = secretty_ref;
      auto seg_start = seg_start_ref;
      auto seg_size = seg_size_ref;

      Value *length = new_size;
      if (seg_start != std::get<1>(toIterate.back())) {
        length = ConstantInt::get(new_size->getType(), seg_start + seg_size);
      }
      if (seg_start != 0)
        length = BuilderZ.CreateSub(
            length, ConstantInt::get(new_size->getType(), seg_start));

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
          if (seg_start != 0) {
            Value *idxs[] = {ConstantInt::get(
                Type::getInt32Ty(op0->getContext()), seg_start)};
            op0 = BuilderZ.CreateInBoundsGEP(Type::getInt8Ty(op0->getContext()),
                                             op0, idxs);
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
          if (seg_start != 0) {
            Value *idxs[] = {ConstantInt::get(
                Type::getInt32Ty(op0->getContext()), seg_start)};
            op0 = Builder2.CreateInBoundsGEP(Type::getInt8Ty(op0->getContext()),
                                             op0, idxs);
          }
          SmallVector<Value *, 4> args = {op0, op1l, length};
          if (op3l)
            args.push_back(op3l);
          CallInst *cal;
          auto funcName = getFuncNameFromCall(&MS);
          if (startsWith(funcName, "memset_pattern") ||
              startsWith(funcName, "llvm.experimental.memset"))
            cal = Builder2.CreateMemSet(
                op0, ConstantInt::get(Builder2.getInt8Ty(), 0), length, {});
          else
            cal = Builder2.CreateCall(MS.getCalledFunction(), args, Defs);
          llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
          ToCopy2.push_back(LLVMContext::MD_noalias);
          cal->copyMetadata(MS, ToCopy2);
          if (auto m = hasMetadata(&MS, "enzyme_zerostack"))
            cal->setMetadata("enzyme_zerostack", m);

          if (startsWith(funcName, "memset_pattern") ||
              startsWith(funcName, "llvm.experimental.memset")) {
            AttributeList NewAttrs;
            for (auto idx :
                 {AttributeList::ReturnIndex, AttributeList::FunctionIndex,
                  AttributeList::FirstArgIndex})
              for (auto attr : MS.getAttributes().getAttributes(idx))
                NewAttrs =
                    NewAttrs.addAttributeAtIndex(MS.getContext(), idx, attr);
            cal->setAttributes(NewAttrs);
          } else
            cal->setAttributes(MS.getAttributes());
          cal->setCallingConv(MS.getCallingConv());
          cal->setDebugLoc(gutils->getNewFromOriginal(MS.getDebugLoc()));
        };

        applyChainRule(Builder2, rule, gutils->lookupM(shadow_dst, Builder2));
      }
    }
  }

  void visitMemTransferInst(llvm::MemTransferInst &MTI) {
    using namespace llvm;
    Value *isVolatile = gutils->getNewFromOriginal(MTI.getOperand(3));
    auto srcAlign = MTI.getSourceAlign();
    auto dstAlign = MTI.getDestAlign();
    visitMemTransferCommon(MTI.getIntrinsicID(), srcAlign, dstAlign, MTI,
                           MTI.getOperand(0), MTI.getOperand(1),
                           gutils->getNewFromOriginal(MTI.getOperand(2)),
                           isVolatile);
  }

  void visitMemTransferCommon(llvm::Intrinsic::ID ID, llvm::MaybeAlign srcAlign,
                              llvm::MaybeAlign dstAlign, llvm::CallInst &MTI,
                              llvm::Value *orig_dst, llvm::Value *orig_src,
                              llvm::Value *new_size, llvm::Value *isVolatile) {
    using namespace llvm;

    if (gutils->isConstantValue(MTI.getOperand(0))) {
      eraseIfUnused(MTI);
      return;
    }

    if (unnecessaryStores.count(&MTI)) {
      eraseIfUnused(MTI);
      return;
    }

    // memcpy of size 1 cannot move differentiable data [single byte copy]
    if (auto ci = dyn_cast<ConstantInt>(new_size)) {
      if (ci->getValue() == 1) {
        eraseIfUnused(MTI);
        return;
      }
    }

    // copying into nullptr is invalid (not sure why it exists here), but we
    // shouldn't do it in reverse pass or shadow
    if (isa<ConstantPointerNull>(orig_dst) ||
        TR.query(orig_dst)[{-1}] == BaseType::Anything) {
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

    if ((Mode == DerivativeMode::ForwardMode ||
         Mode == DerivativeMode::ForwardModeError) &&
        gutils->isConstantValue(orig_dst)) {
      eraseIfUnused(MTI);
      return;
    }

    // Offsets of the form Optional<floating type>, segment start, segment size
    std::vector<std::tuple<Type *, size_t, size_t>> toIterate;
    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&MTI));

    // Special handling mechanism to bypass TA limitations by supporting
    // arbitrary sized types.
    if (auto MD = hasMetadata(&MTI, "enzyme_truetype")) {
      toIterate = parseTrueType(MD, Mode,
                                !gutils->isConstantValue(orig_src) &&
                                    !gutils->runtimeActivity);
    } else {
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      auto vd = TR.query(orig_dst).Data0().ShiftIndices(DL, 0, size, 0);
      vd |= TR.query(orig_src).Data0().PurgeAnything().ShiftIndices(DL, 0, size,
                                                                    0);
      for (size_t i = 0; i < MTI.getNumOperands(); i++)
        if (MTI.getOperand(i) == orig_dst)
          if (MTI.getAttributes().hasParamAttr(i, "enzyme_type")) {
            auto attr = MTI.getAttributes().getParamAttr(i, "enzyme_type");
            auto TT =
                TypeTree::parse(attr.getValueAsString(), MTI.getContext());
            vd |= TT.Data0().ShiftIndices(DL, 0, size, 0);
            break;
          }

      bool errorIfNoType = true;
      if ((Mode == DerivativeMode::ForwardMode ||
           Mode == DerivativeMode::ForwardModeError) &&
          (!gutils->isConstantValue(orig_src) && !gutils->runtimeActivity)) {
        errorIfNoType = false;
      }

      if (!vd.isKnownPastPointer()) {
        if (looseTypeAnalysis) {
          for (auto val : {orig_dst, orig_src}) {
#if LLVM_VERSION_MAJOR < 17
            if (auto CI = dyn_cast<CastInst>(val)) {
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
                  vd =
                      TypeTree(ConcreteType(ET->getScalarType())).Only(0, &MTI);
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
            }
#endif
            if (auto gep = dyn_cast<GetElementPtrInst>(val)) {
              if (auto AT = dyn_cast<ArrayType>(gep->getSourceElementType())) {
                if (AT->getElementType()->isIntegerTy()) {
                  vd = TypeTree(BaseType::Integer).Only(0, &MTI);
                  goto known;
                }
              }
            }
          }
          // If the type is known, but outside of the known range
          // (but the memcpy size is a variable), attempt to use
          // the first type out of range as the memcpy type.
          if (size == 1 && !isa<ConstantInt>(new_size)) {
            for (auto ptr : {orig_dst, orig_src}) {
              vd = TR.query(ptr).Data0().ShiftIndices(DL, 0, -1, 0);
              if (vd.isKnownPastPointer()) {
                ConcreteType mv(BaseType::Unknown);
                size_t minInt = 0xFFFFFFFF;
                for (const auto &pair : vd.getMapping()) {
                  if (pair.first.size() != 1)
                    continue;
                  if (minInt < (size_t)pair.first[0])
                    continue;
                  minInt = pair.first[0];
                  mv = pair.second;
                }
                assert(mv != BaseType::Unknown);
                vd.insert({0}, mv);
                goto known;
              }
            }
          }
          if (errorIfNoType)
            EmitWarning("CannotDeduceType", MTI,
                        "failed to deduce type of copy ", MTI);
          vd = TypeTree(BaseType::Pointer).Only(0, &MTI);
          goto known;
        }
        if (errorIfNoType) {
          std::string str;
          raw_string_ostream ss(str);
          ss << "Cannot deduce type of copy " << MTI;
          EmitNoTypeError(str, MTI, gutils, BuilderZ);
          vd = TypeTree(BaseType::Integer).Only(0, &MTI);
        } else {
          vd = TypeTree(BaseType::Pointer).Only(0, &MTI);
        }
      }

    known:;
      {

        unsigned start = 0;
        while (1) {
          unsigned nextStart = size;

          auto dt = vd[{-1}];
          for (size_t i = start; i < size; ++i) {
            bool Legal = true;
            auto tmp = dt;
            auto next = vd[{(int)i}];
            tmp.checkedOrIn(next, /*PointerIntSame*/ true, Legal);
            // Prevent fusion of {Anything, Float} since anything is an int rule
            // but float requires zeroing.
            if ((dt == BaseType::Anything &&
                 (next != BaseType::Anything && next.isKnown())) ||
                (next == BaseType::Anything &&
                 (dt != BaseType::Anything && dt.isKnown())))
              Legal = false;
            if (!Legal) {
              if (Mode == DerivativeMode::ForwardMode ||
                  Mode == DerivativeMode::ForwardModeError) {
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
                    !gutils->runtimeActivity) {
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
          toIterate.emplace_back(dt.isFloat(), start, nextStart - start);

          if (nextStart == size)
            break;
          start = nextStart;
        }
      }
    }

    // llvm::errs() << "MIT: " << MTI << "|size: " << size << " vd: " <<
    // vd.str() << "\n";

    unsigned dstalign = dstAlign.valueOrOne().value();
    unsigned srcalign = srcAlign.valueOrOne().value();

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

    for (auto &&[floatTy_ref, seg_start_ref, seg_size_ref] : toIterate) {
      auto floatTy = floatTy_ref;
      auto seg_start = seg_start_ref;
      auto seg_size = seg_size_ref;

      Value *length = new_size;
      if (seg_start != std::get<1>(toIterate.back())) {
        length = ConstantInt::get(new_size->getType(), seg_start + seg_size);
      }
      if (seg_start != 0)
        length = BuilderZ.CreateSub(
            length, ConstantInt::get(new_size->getType(), seg_start));

      unsigned subdstalign = dstalign;
      // todo make better alignment calculation
      if (dstalign != 0) {
        if (seg_start % dstalign != 0) {
          dstalign = 1;
        }
      }
      unsigned subsrcalign = srcalign;
      // todo make better alignment calculation
      if (srcalign != 0) {
        if (seg_start % srcalign != 0) {
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
            gutils, Mode, floatTy, ID, subdstalign, subsrcalign,
            /*offset*/ seg_start, gutils->isConstantValue(orig_dst), shadow_dst,
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
        MaybeAlign dalign;
        if (subdstalign)
          dalign = MaybeAlign(subdstalign);
        MaybeAlign salign;
        if (subsrcalign)
          salign = MaybeAlign(subsrcalign);
        if (ddst->getType()->isIntegerTy())
          ddst =
              BuilderZ.CreateIntToPtr(ddst, getInt8PtrTy(ddst->getContext()));
        if (seg_start != 0) {
          ddst = BuilderZ.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(ddst->getContext()), ddst, seg_start);
        }
        CallInst *call;
        // TODO add gutils->runtimeActivity (correctness)
        if (floatTy && gutils->isConstantValue(orig_src)) {
          call = BuilderZ.CreateMemSet(
              ddst, ConstantInt::get(Type::getInt8Ty(ddst->getContext()), 0),
              length, salign, isVolatile);
        } else {
          if (dsrc->getType()->isIntegerTy())
            dsrc =
                BuilderZ.CreateIntToPtr(dsrc, getInt8PtrTy(dsrc->getContext()));
          if (seg_start != 0) {
            dsrc = BuilderZ.CreateConstInBoundsGEP1_64(
                Type::getInt8Ty(ddst->getContext()), dsrc, seg_start);
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

      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError)
        applyChainRule(BuilderZ, fwd_rule, shadow_dst, shadow_src);
      else
        applyChainRule(BuilderZ, rev_rule, shadow_dst, shadow_src);
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
      bool emitReverse = true;
      if (EnzymeJuliaAddrLoad) {
        if (auto prev = dyn_cast_or_null<CallBase>(FI.getPrevNode())) {
          if (auto F = prev->getCalledFunction())
            if (F->getName() == "julia.safepoint")
              emitReverse = false;
        }
        if (auto prev = dyn_cast_or_null<CallBase>(FI.getNextNode())) {
          if (auto F = prev->getCalledFunction())
            if (F->getName() == "julia.safepoint")
              emitReverse = false;
        }
      }
      if (emitReverse) {
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
        II.getIntrinsicID() == Intrinsic::lifetime_end ||
        II.getCalledFunction()->getName() == "llvm.enzyme.lifetime_end") {
      eraseIfUnused(II, /*erase*/ true, /*check*/ false);
      return;
    }
#if LLVM_VERSION_MAJOR >= 20
    if (II.getIntrinsicID() == Intrinsic::experimental_memset_pattern) {
      visitMemSetCommon(II);
      return;
    }
#endif

    // When compiling Enzyme against standard LLVM, and not Intel's
    // modified version of LLVM, the intrinsic `llvm.intel.subscript` is
    // not fully understood by LLVM. One of the results of this is that the ID
    // of the intrinsic is set to Intrinsic::not_intrinsic - hence we are
    // handling the intrinsic here.
    if (isIntelSubscriptIntrinsic(II)) {
      if (Mode == DerivativeMode::ForwardModeSplit ||
          Mode == DerivativeMode::ForwardModeError ||
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
                                getIndex(&II, CacheType::Self, BuilderZ));
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
#if LLVM_VERSION_MAJOR < 20
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f:
#endif
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f: {
      auto CI = cast<ConstantInt>(I.getOperand(1));
      visitLoadLike(I, /*Align*/ MaybeAlign(CI->getZExtValue()),
                    /*constantval*/ false);
      return false;
    }
    default:
      break;
    }

    if (ID == Intrinsic::masked_store) {
      auto align0 = cast<ConstantInt>(I.getOperand(2))->getZExtValue();
      auto align = MaybeAlign(align0);
      visitCommonStore(I, /*orig_ptr*/ I.getOperand(1),
                       /*orig_val*/ I.getOperand(0), align,
                       /*isVolatile*/ false, llvm::AtomicOrdering::NotAtomic,
                       SyncScope::SingleThread,
                       /*mask*/ gutils->getNewFromOriginal(I.getOperand(3)));
      return false;
    }
    if (ID == Intrinsic::masked_load) {
      auto align0 = cast<ConstantInt>(I.getOperand(1))->getZExtValue();
      auto align = MaybeAlign(align0);
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      bool constantval = parseTBAA(I, DL, nullptr)[{-1}].isIntegral();
      visitLoadLike(I, align, constantval,
                    /*mask*/ gutils->getNewFromOriginal(I.getOperand(2)),
                    /*orig_maskInit*/ I.getOperand(3));
      return false;
    }

    auto mod = I.getParent()->getParent()->getParent();
    auto called = cast<CallInst>(&I)->getCalledFunction();
    (void)called;
    switch (ID) {
#include "IntrinsicDerivatives.inc"
    default:
      break;
    }

    switch (Mode) {
    case DerivativeMode::ReverseModePrimal: {
      switch (ID) {
#if LLVM_VERSION_MAJOR <= 20
      case Intrinsic::nvvm_barrier0:
#else
      case Intrinsic::nvvm_barrier_cta_sync_aligned_all:
#endif
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
        if (ID == Intrinsic::umax || ID == Intrinsic::smax ||
            ID == Intrinsic::abs || ID == Intrinsic::sadd_with_overflow ||
            ID == Intrinsic::uadd_with_overflow ||
            ID == Intrinsic::smul_with_overflow ||
            ID == Intrinsic::umul_with_overflow ||
            ID == Intrinsic::ssub_with_overflow ||
            ID == Intrinsic::usub_with_overflow)
          if (looseTypeAnalysis) {
            EmitWarning("CannotDeduceType", I,
                        "failed to deduce type of intrinsic ", I);
            return false;
          }
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        ss << "cannot handle (augmented) unknown intrinsic\n" << I;
        IRBuilder<> BuilderZ(&I);
        getForwardBuilder(BuilderZ);
        EmitNoDerivativeError(ss.str(), I, gutils, BuilderZ);
        return false;
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
      (void)vdiff;

      switch (ID) {

      case Intrinsic::nvvm_barrier0_popc:
      case Intrinsic::nvvm_barrier0_and:
      case Intrinsic::nvvm_barrier0_or: {
        SmallVector<Value *, 1> args = {};
#if LLVM_VERSION_MAJOR > 20
        auto cal = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(
                M, Intrinsic::nvvm_barrier_cta_sync_aligned_all),
            args));
        cal->setCallingConv(getIntrinsicDeclaration(
                                M, Intrinsic::nvvm_barrier_cta_sync_aligned_all)
                                ->getCallingConv());
#else
        auto cal = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(M, Intrinsic::nvvm_barrier0), args));
        cal->setCallingConv(getIntrinsicDeclaration(M, Intrinsic::nvvm_barrier0)
                                ->getCallingConv());
#endif
        cal->setDebugLoc(gutils->getNewFromOriginal(I.getDebugLoc()));
        return false;
      }

#if LLVM_VERSION_MAJOR <= 20
      case Intrinsic::nvvm_barrier0:
#else
      case Intrinsic::nvvm_barrier_cta_sync_aligned_all:
#endif
      case Intrinsic::amdgcn_s_barrier:
      case Intrinsic::nvvm_membar_cta:
      case Intrinsic::nvvm_membar_gl:
      case Intrinsic::nvvm_membar_sys: {
        SmallVector<Value *, 1> args = {};
        auto cal = cast<CallInst>(
            Builder2.CreateCall(getIntrinsicDeclaration(M, ID), args));
        cal->setCallingConv(getIntrinsicDeclaration(M, ID)->getCallingConv());
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
            getIntrinsicDeclaration(M, Intrinsic::lifetime_end, tys), args);
        cal->setCallingConv(
            getIntrinsicDeclaration(M, Intrinsic::lifetime_end, tys)
                ->getCallingConv());
        return false;
      }

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
      default:
        if (gutils->isConstantInstruction(&I))
          return false;
        if (ID == Intrinsic::umax || ID == Intrinsic::smax ||
            ID == Intrinsic::abs || ID == Intrinsic::sadd_with_overflow ||
            ID == Intrinsic::uadd_with_overflow ||
            ID == Intrinsic::smul_with_overflow ||
            ID == Intrinsic::umul_with_overflow ||
            ID == Intrinsic::ssub_with_overflow ||
            ID == Intrinsic::usub_with_overflow)
          if (looseTypeAnalysis) {
            EmitWarning("CannotDeduceType", I,
                        "failed to deduce type of intrinsic ", I);
            return false;
          }
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << *gutils->newFunc << "\n";
        if (Intrinsic::isOverloaded(ID))
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
        else
          ss << "cannot handle (reverse) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        EmitNoDerivativeError(ss.str(), I, gutils, Builder2);
        return false;
      }
      return false;
    }
    case DerivativeMode::ForwardModeSplit:
    case DerivativeMode::ForwardModeError:
    case DerivativeMode::ForwardMode: {

      IRBuilder<> Builder2(&I);
      getForwardBuilder(Builder2);

      switch (ID) {

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
      default:
        if (gutils->isConstantInstruction(&I))
          return false;
        if (ID == Intrinsic::umax || ID == Intrinsic::smax ||
            ID == Intrinsic::abs || ID == Intrinsic::sadd_with_overflow ||
            ID == Intrinsic::uadd_with_overflow ||
            ID == Intrinsic::smul_with_overflow ||
            ID == Intrinsic::umul_with_overflow ||
            ID == Intrinsic::ssub_with_overflow ||
            ID == Intrinsic::usub_with_overflow)
          if (looseTypeAnalysis) {
            EmitWarning("CannotDeduceType", I,
                        "failed to deduce type of intrinsic ", I);
            return false;
          }
        std::string s;
        llvm::raw_string_ostream ss(s);
        if (Intrinsic::isOverloaded(ID))
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID, ArrayRef<Type *>(),
                                   gutils->oldFunc->getParent(), nullptr)
             << "\n"
             << I;
        else
          ss << "cannot handle (forward) unknown intrinsic\n"
             << Intrinsic::getName(ID) << "\n"
             << I;
        EmitNoDerivativeError(ss.str(), I, gutils, Builder2);
        if (!gutils->isConstantValue(&I))
          setDiffe(&I,
                   Constant::getNullValue(gutils->getShadowType(I.getType())),
                   Builder2);
        return false;
      }
      return false;
    }
    }

    return false;
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

    auto found_ow = overwritten_args_map.find(&call);
    assert(found_ow != overwritten_args_map.end());
    const bool subsequent_calls_may_write = found_ow->second.first;
    const std::vector<bool> &overwritten_args = found_ow->second.second;

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

    for (unsigned i = 3; i < call.arg_size(); ++i) {

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
        assert(TR.query(call.getArgOperand(i))[{-1}].isFloat());
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
            RequestContext(&call, &BuilderZ), cast<Function>(called),
            subretType, argsInverted, TR.analyzer->interprocedural,
            /*return is used*/ false,
            /*shadowReturnUsed*/ false, nextTypeInfo,
            subsequent_calls_may_write, overwritten_args, false,
            gutils->runtimeActivity, gutils->strongZero, gutils->getWidth(),
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
                                  getIndex(&call, CacheType::Tape, BuilderZ));
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

    {
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (!subdata && !isMemFreeLibMFunction(getFuncNameFromCall(&call), &ID)) {
        llvm::errs() << *gutils->oldFunc->getParent() << "\n";
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
        llvm::errs() << *called << "\n";
        llvm_unreachable("no subdata");
      }
    }

    if (subdata) {
      auto found = subdata->returns.find(AugmentedStruct::DifferentialReturn);
      assert(found == subdata->returns.end());
    }
    if (subdata) {
      auto found = subdata->returns.find(AugmentedStruct::Return);
      assert(found == subdata->returns.end());
    }

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
        if (subdata && subdata->returns.find(AugmentedStruct::Tape) !=
                           subdata->returns.end()) {
          if (Mode == DerivativeMode::ReverseModeGradient) {
            if (tape == nullptr) {
#if LLVM_VERSION_MAJOR >= 18
              auto It = BuilderZ.GetInsertPoint();
              It.setHeadBit(true);
              BuilderZ.SetInsertPoint(It);
#endif
              tape = BuilderZ.CreatePHI(subdata->tapeType, 0, "tapeArg");
            }
            tape = gutils->cacheForReverse(
                BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));
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
            RequestContext(&call, &Builder2),
            (ReverseCacheKey){
                .todiff = cast<Function>(called),
                .retType = subretType,
                .constant_args = argsInverted,
                .subsequent_calls_may_write = subsequent_calls_may_write,
                .overwritten_args = overwritten_args,
                .returnUsed = false,
                .shadowReturnUsed = false,
                .mode = DerivativeMode::ReverseModeGradient,
                .width = gutils->getWidth(),
                .freeMemory = true,
                .AtomicAdd = true,
                .additionalType =
                    tape ? PointerType::getUnqual(tape->getType()) : nullptr,
                .forceAnonymousTape = false,
                .typeInfo = nextTypeInfo,
                .runtimeActivity = gutils->runtimeActivity,
                .strongZero = gutils->strongZero},
            TR.analyzer->interprocedural, subdata,
            /*omp*/ true);

        if (subdata && subdata->returns.find(AugmentedStruct::Tape) !=
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
          Function *F =
              Function::Create(FT, GlobalVariable::InternalLinkage,
                               cast<Function>(newcalled)->getName() + "#out",
                               *task->getParent());
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

            MaybeAlign align;
            AtomicRMWInst::BinOp op = AtomicRMWInst::FAdd;
            if (auto vt = dyn_cast<VectorType>(dif->getType())) {
              assert(!vt->getElementCount().isScalable());
              size_t numElems = vt->getElementCount().getKnownMinValue();
              for (size_t i = 0; i < numElems; ++i) {
                auto vdif = B.CreateExtractElement(dif, i);
                Value *Idxs[] = {
                    ConstantInt::get(Type::getInt64Ty(vt->getContext()), 0),
                    ConstantInt::get(Type::getInt32Ty(vt->getContext()), i)};
                auto vptr = B.CreateInBoundsGEP(vt, ptr, Idxs);
                B.CreateAtomicRMW(op, vptr, vdif, align,
                                  AtomicOrdering::Monotonic, SyncScope::System);
              }
            } else {
              B.CreateAtomicRMW(op, ptr, dif, align, AtomicOrdering::Monotonic,
                                SyncScope::System);
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
#if LLVM_VERSION_MAJOR < 17
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
      TR.dump();
      EmitFailure("CannotDeduceType", call.getDebugLoc(), &call,
                  "failed to deduce type of copy ", call);
    }
#if LLVM_VERSION_MAJOR < 17
  knownF:
#endif
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
          dsto =
              Builder2.CreateIntToPtr(dsto, getInt8PtrTy(dsto->getContext()));
        unsigned dstaddr =
            cast<PointerType>(dsto->getType())->getAddressSpace();
        auto secretpt = PointerType::get(secretty, dstaddr);
        if (offset != 0) {
          dsto = Builder2.CreateConstInBoundsGEP1_64(
              Type::getInt8Ty(dsto->getContext()), dsto, offset);
        }
        if (srco->getType()->isIntegerTy())
          srco =
              Builder2.CreateIntToPtr(srco, getInt8PtrTy(dsto->getContext()));
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

  void recursivelyHandleSubfunction(llvm::CallInst &call,
                                    llvm::Function *called,
                                    bool subsequent_calls_may_write,
                                    const std::vector<bool> &overwritten_args,
                                    bool shadowReturnUsed,
                                    DIFFE_TYPE subretType, bool subretused) {
    using namespace llvm;

    IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
    BuilderZ.setFastMathFlags(getFast());

    CallInst *newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    Module &M = *call.getParent()->getParent()->getParent();

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
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      SmallVector<Value *, 8> args;
      std::vector<DIFFE_TYPE> argsInverted;
      std::map<int, Type *> gradByVal;
      std::map<int, std::vector<Attribute>> structAttrs;

      for (unsigned i = 0; i < call.arg_size(); ++i) {

        if (call.paramHasAttr(i, Attribute::StructRet)) {
          structAttrs[args.size()].push_back(
              Attribute::get(call.getContext(), "enzyme_sret"));
          // TODO
          // structAttrs[args.size()].push_back(Attribute::get(
          //     call.getContext(), Attribute::AttrKind::ElementType,
          //     call.getParamAttr(i, Attribute::StructRet).getValueAsType()));
        }
        for (auto attr : {"enzymejl_returnRoots", "enzymejl_parmtype",
                          "enzymejl_parmtype_ref", "enzyme_type"})
          if (call.getAttributes().hasParamAttr(i, attr)) {
            structAttrs[args.size()].push_back(call.getParamAttr(i, attr));
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
          argi = getUndefinedValueForType(M, argi->getType());
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

        for (auto attr : {"enzymejl_returnRoots", "enzymejl_parmtype",
                          "enzymejl_parmtype_ref", "enzyme_type"})
          if (call.getAttributes().hasParamAttr(i, attr)) {
            if (gutils->getWidth() == 1) {
              structAttrs[args.size()].push_back(call.getParamAttr(i, attr));
            } else if (attr == std::string("enzymejl_returnRoots")) {
              structAttrs[args.size()].push_back(
                  Attribute::get(call.getContext(), "enzymejl_returnRoots_v"));
            }
          }
        if (call.paramHasAttr(i, Attribute::StructRet)) {
          if (gutils->getWidth() == 1) {
            structAttrs[args.size()].push_back(
                Attribute::get(call.getContext(), "enzyme_sret")
                // orig->getParamAttr(i,
                // Attribute::StructRet).getValueAsType());
            );
            // TODO
            // structAttrs[args.size()].push_back(Attribute::get(
            //     call.getContext(), Attribute::AttrKind::ElementType,
            //     call.getParamAttr(i,
            //     Attribute::StructRet).getValueAsType()));
          } else {
            structAttrs[args.size()].push_back(
                Attribute::get(call.getContext(), "enzyme_sret_v"));
            // TODO
            // structAttrs[args.size()].push_back(Attribute::get(
            //     call.getContext(), Attribute::AttrKind::ElementType,
            //     call.getParamAttr(i,
            //     Attribute::StructRet).getValueAsType()));
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
      if (tapeIdx) {

        auto idx = *tapeIdx;
        FunctionType *FT = subdata->fn->getFunctionType();
#if LLVM_VERSION_MAJOR >= 18
        auto It = BuilderZ.GetInsertPoint();
        It.setHeadBit(true);
        BuilderZ.SetInsertPoint(It);
#endif
        tape = BuilderZ.CreatePHI(
            (tapeIdx == -1)
                ? FT->getReturnType()
                : cast<StructType>(FT->getReturnType())->getElementType(idx),
            1, "tapeArg");

        assert(!tape->getType()->isEmptyTy());
        gutils->TapesToPreventRecomputation.insert(cast<Instruction>(tape));
        tape = gutils->cacheForReverse(
            BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));
        args.push_back(tape);
      }

      Value *newcalled = nullptr;
      FunctionType *FT = nullptr;

      if (called) {
        newcalled = gutils->Logic.CreateForwardDiff(
            RequestContext(&call, &BuilderZ), cast<Function>(called),
            subretType, argsInverted, TR.analyzer->interprocedural,
            /*returnValue*/ subretused, Mode,
            ((DiffeGradientUtils *)gutils)->FreeMemory, gutils->runtimeActivity,
            gutils->strongZero, gutils->getWidth(),
            tape ? tape->getType() : nullptr, nextTypeInfo,
            subsequent_calls_may_write, overwritten_args,
            /*augmented*/ subdata);
        FT = cast<Function>(newcalled)->getFunctionType();
      } else {
        auto callval = call.getCalledOperand();
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

        FT = getFunctionTypeForClone(
            ft, Mode, gutils->getWidth(), tape ? tape->getType() : nullptr,
            argsInverted, false, /*returnTape*/ false,
            /*returnPrimal*/ subretused, /*returnShadow*/ retActive);
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

    SmallVector<ValueType, 2> PreBundleTypes;
    SmallVector<ValueType, 2> BundleTypes;

    for (unsigned i = 0; i < call.arg_size(); ++i) {

      auto argi = gutils->getNewFromOriginal(call.getArgOperand(i));

      if (call.isByValArgument(i)) {
        preByVal[pre_args.size()] = call.getParamByValType(i);
      }
      for (auto attr : {"enzymejl_returnRoots", "enzymejl_parmtype",
                        "enzymejl_parmtype_ref", "enzyme_type"})
        if (call.getAttributes().hasParamAttr(i, attr)) {
          structAttrs[pre_args.size()].push_back(call.getParamAttr(i, attr));
        }
      if (call.paramHasAttr(i, Attribute::StructRet)) {
        structAttrs[pre_args.size()].push_back(
            // TODO persist types
            Attribute::get(call.getContext(), "enzyme_sret")
            // Attribute::get(orig->getContext(), "enzyme_sret",
            // orig->getParamAttr(ii, Attribute::StructRet).getValueAsType());
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

      ValueType preType = ValueType::Primal;
      ValueType revType = ValueType::Primal;

      // Keep the existing passed value if coming from outside.
      if (readNoneNoCapture ||
          (argTy == DIFFE_TYPE::DUP_NONEED &&
           (writeOnlyNoCapture ||
            !isa<Argument>(getBaseObject(call.getArgOperand(i)))))) {
        prearg = getUndefinedValueForType(M, argi->getType());
        preType = ValueType::None;
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
          argi = getUndefinedValueForType(M, argi->getType());
          revType = ValueType::None;
        }
        args.push_back(lookup(argi, Builder2));
      }

      argsInverted.push_back(argTy);

      if (argTy == DIFFE_TYPE::CONSTANT) {
        PreBundleTypes.push_back(preType);
        BundleTypes.push_back(revType);
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

        for (auto attr : {"enzymejl_returnRoots", "enzymejl_parmtype",
                          "enzymejl_parmtype_ref", "enzyme_type"})
          if (call.getAttributes().hasParamAttr(i, attr)) {
            if (gutils->getWidth() == 1) {
              structAttrs[pre_args.size()].push_back(
                  call.getParamAttr(i, attr));
            } else if (attr == std::string("enzymejl_returnRoots")) {
              structAttrs[pre_args.size()].push_back(
                  Attribute::get(call.getContext(), "enzymejl_returnRoots_v"));
            }
          }
        if (call.paramHasAttr(i, Attribute::StructRet)) {
          if (gutils->getWidth() == 1) {
            structAttrs[pre_args.size()].push_back(
                // TODO persist types
                Attribute::get(call.getContext(), "enzyme_sret")
                // Attribute::get(orig->getContext(), "enzyme_sret",
                // orig->getParamAttr(ii,
                // Attribute::StructRet).getValueAsType());
            );
          } else {
            structAttrs[pre_args.size()].push_back(
                // TODO persist types
                Attribute::get(call.getContext(), "enzyme_sret_v")
                // Attribute::get(orig->getContext(), "enzyme_sret_v",
                // gutils->getShadowType(orig->getParamAttr(ii,
                // Attribute::StructRet).getValueAsType()));
            );
          }
        }
        if (Mode != DerivativeMode::ReverseModePrimal) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);

          Value *darg = nullptr;

          if (((writeOnlyNoCapture && TR.query(call.getArgOperand(
                                          i))[{-1, -1}] == BaseType::Pointer) ||
               gutils->isConstantInstruction(&call)) &&
              !replaceFunction) {
            darg = getUndefinedValueForType(
                M, gutils->getShadowType(argi->getType()));
          } else {
            darg = gutils->invertPointerM(call.getArgOperand(i), Builder2);
            revType = (revType == ValueType::None) ? ValueType::Shadow
                                                   : ValueType::Both;
          }
          args.push_back(lookup(darg, Builder2));
        }
        if (Mode == DerivativeMode::ReverseModeGradient && !replaceFunction) {
          pre_args.push_back(getUndefinedValueForType(M, argi->getType()));
        } else {
          pre_args.push_back(
              gutils->invertPointerM(call.getArgOperand(i), BuilderZ));
        }
        preType =
            (preType == ValueType::None) ? ValueType::Shadow : ValueType::Both;

        // Note sometimes whattype mistakenly says something should be
        // constant [because composed of integer pointers alone]
        (void)argType;
        assert(whatType(argType, Mode) == DIFFE_TYPE::DUP_ARG ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      } else {
        if (foreignFunction)
          assert(!argType->isIntOrIntVectorTy());
        assert(whatType(argType, Mode) == DIFFE_TYPE::OUT_DIFF ||
               whatType(argType, Mode) == DIFFE_TYPE::CONSTANT);
      }
      PreBundleTypes.push_back(preType);
      BundleTypes.push_back(revType);
    }
    if (called) {
      if (call.arg_size() !=
          cast<Function>(called)->getFunctionType()->getNumParams()) {
        llvm::errs() << *gutils->oldFunc->getParent() << "\n";
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << call << "\n";
        llvm::errs() << " number of arg operands != function parameters\n";
        EmitFailure("MismatchArgs", call.getDebugLoc(), &call,
                    "Number of arg operands != function parameters\n", call);
      }
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
        auto callval = call.getCalledOperand();
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
          EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
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
              RequestContext(&call, &BuilderZ), cast<Function>(called),
              subretType, argsInverted, TR.analyzer->interprocedural,
              /*return is used*/ subretused, shadowReturnUsed, nextTypeInfo,
              subsequent_calls_may_write, overwritten_args, false,
              gutils->runtimeActivity, gutils->strongZero, gutils->getWidth(),
              gutils->AtomicAdd);
          if (Mode == DerivativeMode::ReverseModePrimal) {
            assert(augmentedReturn);
            auto subaugmentations =
                (std::map<const llvm::CallInst *, AugmentedReturn *>
                     *)&augmentedReturn->subaugmentations;
            insert_or_assign2<const llvm::CallInst *, AugmentedReturn *>(
                *subaugmentations, &call, (AugmentedReturn *)subdata);
          }
        }
        {
          Intrinsic::ID ID = Intrinsic::not_intrinsic;
          if (!subdata &&
              !isMemFreeLibMFunction(getFuncNameFromCall(&call), &ID)) {
            llvm::errs() << *gutils->oldFunc->getParent() << "\n";
            llvm::errs() << *gutils->oldFunc << "\n";
            llvm::errs() << *gutils->newFunc << "\n";
            llvm::errs() << *called << "\n";
            assert(subdata);
          }
        }

        if (subdata) {
          fnandtapetype = subdata;
          newcalled = subdata->fn;
          FT = cast<Function>(newcalled)->getFunctionType();

          auto found =
              subdata->returns.find(AugmentedStruct::DifferentialReturn);
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
      }
      // sub_index_map = fnandtapetype.tapeIndices;

      // llvm::errs() << "seeing sub_index_map of " << sub_index_map->size()
      // << " in ap " << cast<Function>(called)->getName() << "\n";
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModePrimal) {

        assert(newcalled);
        assert(FT);

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
            gutils->getInvertedBundles(&call, PreBundleTypes, BuilderZ,
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

        if (tapeIdx) {
          auto tval = *tapeIdx;
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
          tape = gutils->cacheForReverse(
              BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));
        }

        if (subretused) {
          Value *dcall = nullptr;
          assert(returnIdx);
          assert(augmentcall);
          auto rval = *returnIdx;
          dcall = (rval < 0) ? augmentcall
                             : BuilderZ.CreateExtractValue(augmentcall,
                                                           {(unsigned)rval});
          gutils->originalToNewFn[&call] = dcall;
          gutils->newToOriginalFn.erase(newCall);
          gutils->newToOriginalFn[dcall] = &call;

          assert(dcall->getType() == call.getType());
          assert(dcall);

          if (!gutils->isConstantValue(&call)) {
            if (!call.getType()->isFPOrFPVectorTy() && TR.anyPointer(&call)) {
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
                  Seen[UsageKey(pair.first, QueryType::Primal)] = false;
                }
              }
            if (!primalNeededInReverse) {

              auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                    ? DerivativeMode::ReverseModeGradient
                                    : Mode;
              primalNeededInReverse =
                  DifferentialUseAnalysis::is_value_needed_in_reverse<
                      QueryType::Primal>(gutils, &call, minCutMode, Seen,
                                         oldUnreachable);
            }
            if (primalNeededInReverse)
              gutils->cacheForReverse(
                  BuilderZ, dcall, getIndex(&call, CacheType::Self, BuilderZ));
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
          if (FT) {
            if (!tape) {
              assert(tapeIdx);
              auto tval = *tapeIdx;
#if LLVM_VERSION_MAJOR >= 18
              auto It = BuilderZ.GetInsertPoint();
              It.setHeadBit(true);
              BuilderZ.SetInsertPoint(It);
#endif
              tape = BuilderZ.CreatePHI(
                  (tapeIdx == -1) ? FT->getReturnType()
                                  : cast<StructType>(FT->getReturnType())
                                        ->getElementType(tval),
                  1, "tapeArg");
            }
            tape = gutils->cacheForReverse(
                BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));
          }
        }

        if (subretused) {
          Intrinsic::ID ID = Intrinsic::not_intrinsic;
          if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, &call, Mode, oldUnreachable) &&
              !gutils->unnecessaryIntermediates.count(&call)) {

            if (!isMemFreeLibMFunction(getFuncNameFromCall(&call), &ID)) {

#if LLVM_VERSION_MAJOR >= 18
              auto It = BuilderZ.GetInsertPoint();
              It.setHeadBit(true);
              BuilderZ.SetInsertPoint(It);
#endif
              auto idx = getIndex(&call, CacheType::Self, BuilderZ);
              if (idx == IndexMappingError) {
                std::string str;
                raw_string_ostream ss(str);
                ss << "Failed to compute consistent cache index for operation: "
                   << call << "\n";
                if (CustomErrorHandler) {
                  CustomErrorHandler(str.c_str(), wrap(&call),
                                     ErrorType::InternalError, nullptr, nullptr,
                                     nullptr);
                } else {
                  EmitFailure("GetIndexError", call.getDebugLoc(), &call,
                              ss.str());
                }
              } else {
                if (Mode == DerivativeMode::ReverseModeCombined)
                  cachereplace = newCall;
                else
                  cachereplace = BuilderZ.CreatePHI(
                      call.getType(), 1, call.getName() + "_tmpcacheB");
                cachereplace =
                    gutils->cacheForReverse(BuilderZ, cachereplace, idx);
              }
            }
          } else {
#if LLVM_VERSION_MAJOR >= 18
            auto It = BuilderZ.GetInsertPoint();
            It.setHeadBit(true);
            BuilderZ.SetInsertPoint(It);
#endif
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
              !isa<ReturnInst>(use)) {
            hasNonReturnUse = true;
          }
        }

        if (subcheck && hasNonReturnUse) {

          Value *newip = nullptr;
          if (Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ReverseModePrimal) {

            if (!differetIdx) {
              std::string str;
              raw_string_ostream ss(str);
              ss << "Did not have return index set when differentiating "
                    "function\n";
              ss << " call" << call << "\n";
              ss << " augmentcall" << *augmentcall << "\n";
              if (CustomErrorHandler) {
                CustomErrorHandler(str.c_str(), wrap(&call),
                                   ErrorType::InternalError, nullptr, nullptr,
                                   nullptr);
              } else {
                EmitFailure("GetIndexError", call.getDebugLoc(), &call,
                            ss.str());
              }
              placeholder->replaceAllUsesWith(
                  UndefValue::get(placeholder->getType()));
              if (placeholder == &*BuilderZ.GetInsertPoint()) {
                BuilderZ.SetInsertPoint(placeholder->getNextNode());
              }
              gutils->erase(placeholder);
            } else {
              auto drval = *differetIdx;
              newip = (drval < 0)
                          ? augmentcall
                          : BuilderZ.CreateExtractValue(augmentcall,
                                                        {(unsigned)drval},
                                                        call.getName() + "'ac");
              assert(newip->getType() == placeholder->getType());
              placeholder->replaceAllUsesWith(newip);
              if (placeholder == &*BuilderZ.GetInsertPoint()) {
                BuilderZ.SetInsertPoint(placeholder->getNextNode());
              }
              gutils->erase(placeholder);
            }
          } else {
            newip = placeholder;
          }

          newip = gutils->cacheForReverse(
              BuilderZ, newip, getIndex(&call, CacheType::Shadow, BuilderZ));

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
                QueryType::Primal>(gutils, &call, Mode, oldUnreachable) &&
            !gutils->unnecessaryIntermediates.count(&call)) {
          assert(!replaceFunction);
#if LLVM_VERSION_MAJOR >= 18
          auto It = BuilderZ.GetInsertPoint();
          It.setHeadBit(true);
          BuilderZ.SetInsertPoint(It);
#endif
          cachereplace = BuilderZ.CreatePHI(call.getType(), 1,
                                            call.getName() + "_cachereplace2");
          cachereplace = gutils->cacheForReverse(
              BuilderZ, cachereplace,
              getIndex(&call, CacheType::Self, BuilderZ));
        } else {
#if LLVM_VERSION_MAJOR >= 18
          auto It = BuilderZ.GetInsertPoint();
          It.setHeadBit(true);
          BuilderZ.SetInsertPoint(It);
#endif
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
          RequestContext(&call, &Builder2),
          (ReverseCacheKey){
              .todiff = cast<Function>(called),
              .retType = subretType,
              .constant_args = argsInverted,
              .subsequent_calls_may_write = subsequent_calls_may_write,
              .overwritten_args = overwritten_args,
              .returnUsed = replaceFunction && subretused,
              .shadowReturnUsed = shadowReturnUsed && replaceFunction,
              .mode = subMode,
              .width = gutils->getWidth(),
              .freeMemory = true,
              .AtomicAdd = gutils->AtomicAdd,
              .additionalType = tape ? tape->getType() : nullptr,
              .forceAnonymousTape = false,
              .typeInfo = nextTypeInfo,
              .runtimeActivity = gutils->runtimeActivity,
              .strongZero = gutils->strongZero},
          TR.analyzer->interprocedural, subdata);
      if (!newcalled)
        return;
      FT = cast<Function>(newcalled)->getFunctionType();
    } else {

      assert(subMode != DerivativeMode::ReverseModeCombined);

      auto callval = call.getCalledOperand();

      if (gutils->isConstantValue(callval)) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << *gutils->oldFunc << "\n";
        ss << "in Mode: " << to_string(Mode) << "\n";
        ss << " orig: " << call << " callval: " << *callval << "\n";
        ss << " constant function being called, but active call instruction\n";
        auto val = EmitNoDerivativeError(ss.str(), call, gutils, Builder2);
        if (val)
          newcalled = val;
        else
          newcalled =
              UndefValue::get(gutils->getShadowType(callval->getType()));
      } else {
        newcalled = lookup(gutils->invertPointerM(callval, Builder2), Builder2);
      }

      auto ft = call.getFunctionType();

      auto res =
          getDefaultFunctionTypeForGradient(ft, /*subretType*/ subretType);
      // TODO Note there is empty tape added here, replace with generic
      res.first.push_back(getInt8PtrTy(newcalled->getContext()));
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

    for (unsigned i = 0; i < call.arg_size(); ++i) {
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

      SmallPtrSet<Value *, 2> postCreateSet(postCreate.begin(),
                                            postCreate.end());
      for (auto a : postCreate) {
        a->moveBefore(*Builder2.GetInsertBlock(), Builder2.GetInsertPoint());
        for (size_t i = 0; i < a->getNumOperands(); i++) {
          auto op = dyn_cast<Instruction>(a->getOperand(i));
          if (!op || postCreateSet.count(op))
            continue;
          if (gutils->isOriginal(op->getParent())) {
            IRBuilder<> BuilderA(a);
            a->setOperand(i, gutils->lookupM(op, BuilderA));
          }
        }
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
          if (!call.getType()->isFPOrFPVectorTy() && TR.anyPointer(&call)) {
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

  void handleMPI(llvm::CallInst &call, llvm::Function *called,
                 llvm::StringRef funcName);

  bool handleKnownCallDerivatives(llvm::CallInst &call, llvm::Function *called,
                                  llvm::StringRef funcName,
                                  bool subsequent_calls_may_write,
                                  const std::vector<bool> &overwritten_args,
                                  llvm::CallInst *const newCall);

  // Return
  void visitCallInst(llvm::CallInst &call) {
    using namespace llvm;

    StringRef funcName = getFuncNameFromCall(&call);

    // When compiling Enzyme against standard LLVM, and not Intel's
    // modified version of LLVM, the intrinsic `llvm.intel.subscript` is
    // not fully understood by LLVM. One of the results of this is that the
    // visitor dispatches to visitCallInst, rather than visitIntrinsicInst, when
    // presented with the intrinsic - hence why we are handling it here.
    if (startsWith(funcName, ("llvm.intel.subscript"))) {
      assert(isa<IntrinsicInst>(call));
      visitIntrinsicInst(cast<IntrinsicInst>(call));
      return;
    }

    if (funcName == "llvm.enzyme.lifetime_start") {
      visitIntrinsicInst(cast<IntrinsicInst>(call));
      return;
    }
    if (funcName == "llvm.enzyme.lifetime_end") {
      SmallVector<Value *, 2> orig_ops(call.getNumOperands());
      for (unsigned i = 0; i < call.getNumOperands(); ++i) {
        orig_ops[i] = call.getOperand(i);
      }
      handleAdjointForIntrinsic(Intrinsic::lifetime_end, call, orig_ops);
      eraseIfUnused(call);
      return;
    }

    CallInst *const newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
    IRBuilder<> BuilderZ(newCall);
    BuilderZ.setFastMathFlags(getFast());

    if (overwritten_args_map.find(&call) == overwritten_args_map.end() &&
        Mode != DerivativeMode::ForwardMode &&
        Mode != DerivativeMode::ForwardModeError) {
      llvm::errs() << " call: " << call << "\n";
      for (auto &pair : overwritten_args_map) {
        llvm::errs() << " + " << *pair.first << "\n";
      }
    }

    assert(overwritten_args_map.find(&call) != overwritten_args_map.end() ||
           Mode == DerivativeMode::ForwardMode ||
           Mode == DerivativeMode::ForwardModeError);
    const bool subsequent_calls_may_write =
        (Mode == DerivativeMode::ForwardMode ||
         Mode == DerivativeMode::ForwardModeError)
            ? false
            : overwritten_args_map.find(&call)->second.first;
    const std::vector<bool> &overwritten_args =
        (Mode == DerivativeMode::ForwardMode ||
         Mode == DerivativeMode::ForwardModeError)
            ? std::vector<bool>()
            : overwritten_args_map.find(&call)->second.second;

    auto called = getFunctionFromCall(&call);

    bool subretused = false;
    bool shadowReturnUsed = false;
    auto smode = Mode;
    if (smode == DerivativeMode::ReverseModeGradient)
      smode = DerivativeMode::ReverseModePrimal;
    DIFFE_TYPE subretType = gutils->getReturnDiffeType(
        &call, &subretused, &shadowReturnUsed, smode);

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
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

        ifound = gutils->invertedPointers.find(&call);
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
                                    getIndex(&call, CacheType::Tape, BuilderZ));
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

#if LLVM_VERSION_MAJOR >= 18
            auto It = BuilderZ.GetInsertPoint();
            It.setHeadBit(true);
            BuilderZ.SetInsertPoint(It);
#endif
            tape = BuilderZ.CreatePHI(tapeType, 0);
            tape = gutils->cacheForReverse(
                BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ),
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
              invertedReturn = gutils->cacheForReverse(
                  BuilderZ, invertedReturn,
                  getIndex(&call, CacheType::Shadow, BuilderZ));
            } else {
              auto idx = getIndex(&call, CacheType::Shadow, BuilderZ);
              invertedReturn =
                  gutils->cacheForReverse(BuilderZ, placeholder, idx);
              if (idx == IndexMappingError) {
                if (placeholder->getType() != invertedReturn->getType())
                  llvm::errs() << " place: " << *placeholder
                               << "  invRet: " << *invertedReturn;
                placeholder->replaceAllUsesWith(invertedReturn);
                gutils->erase(placeholder);
              }
            }

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
              Seen[UsageKey(pair.first, QueryType::Primal)] = false;
          primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
        }
        if (subretused && primalNeededInReverse) {
          if (normalReturn != newCall) {
            assert(normalReturn->getType() == newCall->getType());
            gutils->replaceAWithB(newCall, normalReturn);
            BuilderZ.SetInsertPoint(newCall->getNextNode());
            gutils->erase(newCall);
          }
          normalReturn = gutils->cacheForReverse(
              BuilderZ, normalReturn,
              getIndex(&call, CacheType::Self, BuilderZ));
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

    if (called) {
      if (funcName == "__kmpc_fork_call") {
        visitOMPCall(call);
        return;
      }
    }

    if (handleKnownCallDerivatives(call, called, funcName,
                                   subsequent_calls_may_write, overwritten_args,
                                   newCall))
      return;

    bool useConstantFallback =
        DifferentialUseAnalysis::callShouldNotUseDerivative(gutils, call);
    if (!useConstantFallback) {
      if (gutils->isConstantInstruction(&call) &&
          gutils->isConstantValue(&call)) {
        EmitWarning("ConstnatFallback", call,
                    "Call was deduced inactive but still doing differential "
                    "rewrite as it may escape an allocation",
                    call);
      }
    }
    if (useConstantFallback) {
      if (!gutils->isConstantValue(&call)) {
        auto found = gutils->invertedPointers.find(&call);
        if (found != gutils->invertedPointers.end()) {
          PHINode *placeholder = cast<PHINode>(&*found->second);
          gutils->invertedPointers.erase(found);
          gutils->erase(placeholder);
        }
      }
      bool noFree = Mode == DerivativeMode::ForwardMode ||
                    Mode == DerivativeMode::ForwardModeError;
      noFree |= call.hasFnAttr(Attribute::NoFree);
      if (!noFree && called) {
        noFree |= called->hasFnAttribute(Attribute::NoFree);
      }

      std::map<UsageKey, bool> CacheResults;
      for (auto pair : gutils->knownRecomputeHeuristic) {
        if (!pair.second || gutils->unnecessaryIntermediates.count(
                                cast<Instruction>(pair.first))) {
          CacheResults[UsageKey(pair.first, QueryType::Primal)] = false;
        }
      }

      if (!noFree && !EnzymeGlobalActivity) {
        bool mayActiveFree = false;
        for (unsigned i = 0; i < call.arg_size(); ++i) {
          Value *a = call.getOperand(i);

          if (EnzymeJuliaAddrLoad && isSpecialPtr(a->getType()))
            continue;
          // if could not be a pointer, it cannot be freed
          if (!TR.query(a)[{-1}].isPossiblePointer())
            continue;
          // if active value, we need to do memory preservation
          if (!gutils->isConstantValue(a)) {
            mayActiveFree = true;
            break;
          }
          // if used in reverse (even if just primal), need to do
          // memory preservation
          const auto obj = getBaseObject(a);
          // If not allocation/allocainst, it is possible this aliases
          // a pointer needed in the reverse pass
          bool isAllocation = false;
          for (auto objv = obj;;) {
            if (isAllocationCall(objv, gutils->TLI)) {
              isAllocation = true;
              break;
            }
            if (auto objC = dyn_cast<CallBase>(objv))
              if (auto F = getFunctionFromCall(objC))
                if (!F->empty()) {
                  SmallPtrSet<Value *, 1> set;
                  for (auto &B : *F) {
                    if (auto RI = dyn_cast<ReturnInst>(B.getTerminator())) {
                      auto v = getBaseObject(RI->getOperand(0));
                      if (isa<ConstantPointerNull>(v))
                        continue;
                      set.insert(v);
                    }
                  }
                  if (set.size() == 1) {
                    objv = *set.begin();
                    continue;
                  }
                }
            break;
          }
          if (!isAllocation) {
            mayActiveFree = true;
            break;
          }
          {
            auto found = gutils->knownRecomputeHeuristic.find(obj);
            if (found != gutils->knownRecomputeHeuristic.end()) {
              if (!found->second) {
                auto CacheResults2(CacheResults);
                CacheResults2.erase(UsageKey(obj, QueryType::Primal));
                if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                        QueryType::Primal>(gutils, obj,
                                           DerivativeMode::ReverseModeGradient,
                                           CacheResults2, oldUnreachable)) {
                  mayActiveFree = true;
                  break;
                }
              }
              continue;
            }
          }
          auto CacheResults2(CacheResults);
          if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, obj,
                                     DerivativeMode::ReverseModeGradient,
                                     CacheResults2, oldUnreachable)) {
            mayActiveFree = true;
            break;
          }
        }
        if (!mayActiveFree)
          noFree = true;
      }
      if (!noFree) {
        auto callval = call.getCalledOperand();
        if (!isa<Constant>(callval))
          callval = gutils->getNewFromOriginal(callval);
        newCall->setCalledOperand(gutils->Logic.CreateNoFree(
            RequestContext(&call, &BuilderZ), callval));
      }
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self, BuilderZ));
          eraseIfUnused(call);
          return;
        }
      }

      // If we need this value and it is illegal to recompute it (it writes or
      // may load overwritten data)
      //    Store and reload it
      if (Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode &&
          Mode != DerivativeMode::ForwardModeError && subretused &&
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
                Seen[UsageKey(pair.first, QueryType::Primal)] = false;
              }
            }
          if (!primalNeededInReverse) {

            auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                                  ? DerivativeMode::ReverseModeGradient
                                  : Mode;
            primalNeededInReverse =
                DifferentialUseAnalysis::is_value_needed_in_reverse<
                    QueryType::Primal>(gutils, &call, minCutMode, Seen,
                                       oldUnreachable);
          }
          if (primalNeededInReverse) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self, BuilderZ));
            eraseIfUnused(call);
            return;
          }
        }
        // Force erasure in reverse pass, since cached if needed
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit)
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        else
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

    return recursivelyHandleSubfunction(
        call, called, subsequent_calls_may_write, overwritten_args,
        shadowReturnUsed, subretType, subretused);
  }
};

#endif // ENZYME_ADJOINT_GENERATOR_H
