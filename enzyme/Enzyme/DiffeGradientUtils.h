//===- DiffeGradientUtils.h - Helper class and utilities for AD ---------===//
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

#ifndef ENZYME_DIFFEGRADIENTUTILS_H_
#define ENZYME_DIFFEGRADIENTUTILS_H_

#include "GradientUtils.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"

#include "ActivityAnalysis.h"
#include "EnzymeLogic.h"
#include "Utils.h"

#include "llvm-c/Core.h"

#if LLVM_VERSION_MAJOR <= 16
#include "llvm/ADT/Triple.h"
#endif

class DiffeGradientUtils final : public GradientUtils {
  DiffeGradientUtils(
      EnzymeLogic &Logic, llvm::Function *newFunc_, llvm::Function *oldFunc_,
      llvm::TargetLibraryInfo &TLI, TypeAnalysis &TA, TypeResults TR,
      llvm::ValueToValueMapTy &invertedPointers_,
      const llvm::SmallPtrSetImpl<llvm::Value *> &constantvalues_,
      const llvm::SmallPtrSetImpl<llvm::Value *> &returnvals_,
      DIFFE_TYPE ActiveReturn, llvm::ArrayRef<DIFFE_TYPE> constant_values,
      llvm::ValueMap<const llvm::Value *, AssertingReplacingVH> &origToNew_,
      DerivativeMode mode, unsigned width, bool omp);

public:
  /// Whether to free memory in reverse pass or split forward.
  bool FreeMemory;
  llvm::ValueMap<const llvm::Value *, llvm::TrackingVH<llvm::AllocaInst>>
      differentials;
  static DiffeGradientUtils *
  CreateFromClone(EnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  llvm::Function *todiff, llvm::TargetLibraryInfo &TLI,
                  TypeAnalysis &TA, FnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, llvm::Type *additionalArg, bool omp);

  llvm::AllocaInst *getDifferential(llvm::Value *val);

public:
  llvm::Value *diffe(llvm::Value *val, llvm::IRBuilder<> &BuilderM);

  /// Returns created select instructions, if any
  llvm::SmallVector<llvm::SelectInst *, 4>
  addToDiffe(llvm::Value *val, llvm::Value *dif, llvm::IRBuilder<> &BuilderM,
             llvm::Type *addingType, llvm::ArrayRef<llvm::Value *> idxs = {},
             llvm::Value *mask = nullptr);

  void setDiffe(llvm::Value *val, llvm::Value *toset,
                llvm::IRBuilder<> &BuilderM);

  llvm::CallInst *
  freeCache(llvm::BasicBlock *forwardPreheader, const SubLimitType &sublimits,
            int i, llvm::AllocaInst *alloc, llvm::ConstantInt *byteSizeOfType,
            llvm::Value *storeInto, llvm::MDNode *InvariantMD) override;

/// align is the alignment that should be specified for load/store to pointer
#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(llvm::Instruction *orig, llvm::Value *origVal,
                             llvm::Type *addingType, unsigned start,
                             unsigned size, llvm::Value *origptr,
                             llvm::Value *dif, llvm::IRBuilder<> &BuilderM,
                             llvm::MaybeAlign align = llvm::MaybeAlign(),
                             llvm::Value *mask = nullptr);
#else
  void addToInvertedPtrDiffe(llvm::Instruction *orig, llvm::Value *origVal,
                             llvm::Type *addingType, unsigned start,
                             unsigned size, llvm::Value *origptr,
                             llvm::Value *dif, llvm::IRBuilder<> &BuilderM,
                             unsigned align = 0, llvm::Value *mask = nullptr);
#endif

#if LLVM_VERSION_MAJOR >= 10
  void addToInvertedPtrDiffe(llvm::Instruction *orig, llvm::Value *origVal,
                             TypeTree vd, unsigned size, llvm::Value *origptr,
                             llvm::Value *prediff, llvm::IRBuilder<> &Builder2,
                             llvm::MaybeAlign align = llvm::MaybeAlign(),
                             llvm::Value *premask = nullptr);
#else
  void addToInvertedPtrDiffe(llvm::Instruction *orig, llvm::Value *origVal,
                             TypeTree vd, unsigned size, llvm::Value *origptr,
                             llvm::Value *prediff, llvm::IRBuilder<> &Builder2,
                             unsigned align = 0,
                             llvm::Value *premask = nullptr);
#endif
};

#endif
