//===- DifferentialUseAnalysis.h - Determine values needed in reverse pass-===//
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
// This file contains the declaration of Differential USe Analysis -- an
// AD-specific analysis that deduces if a given value is needed in the reverse
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_DIFFERENTIALUSEANALYSIS_H_
#define ENZYME_DIFFERENTIALUSEANALYSIS_H_

#include <map>
#include <set>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

#include "DiffeGradientUtils.h"
#include "GradientUtils.h"
#include "LibraryFuncs.h"

extern "C" {
extern llvm::cl::opt<bool> EnzymePrintDiffUse;
}

/// Classification of what type of use is requested
enum class QueryType {
  // The original value is needed for the derivative
  Primal = 0,
  // The shadow value is needed for the derivative
  Shadow = 1,
  // The primal value is needed to stand in for the shadow
  // value and compute the derivative of an instruction
  ShadowByConstPrimal = 2
};

static inline std::string to_string(QueryType mode) {
  switch (mode) {
  case QueryType::Primal:
    return "Primal";
  case QueryType::Shadow:
    return "Shadow";
  case QueryType::ShadowByConstPrimal:
    return "ShadowByConstPrimal";
  }
  llvm_unreachable("illegal QueryType");
}

typedef std::pair<const llvm::Value *, QueryType> UsageKey;

namespace DifferentialUseAnalysis {

/// Determine if a value is needed directly to compute the adjoint
/// of the given instruction user. `shadow` denotes whether we are considering
/// the shadow of the value (shadow=true) or the primal of the value
/// (shadow=false).
/// Recursive use is only usable in shadow mode.
bool is_use_directly_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *val, DerivativeMode mode,
    const llvm::Instruction *user,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable,
    QueryType shadow, bool *recursiveUse = nullptr);

template <QueryType VT, bool OneLevel = false>
inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *inst, DerivativeMode mode,
    std::map<UsageKey, bool> &seen,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable) {
  using namespace llvm;

  TypeResults const &TR = gutils->TR;
  static_assert(VT == QueryType::Primal || VT == QueryType::Shadow ||
                VT == QueryType::ShadowByConstPrimal);
  auto idx = UsageKey(inst, VT);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  if (VT == QueryType::Primal) {
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (op->getOpcode() == Instruction::FDiv) {
        if (!gutils->isConstantValue(const_cast<Value *>(inst)) &&
            !gutils->isConstantValue(op->getOperand(1))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as is active div\n";
          return seen[idx] = true;
        }
      }
    }
  }

  if (auto CI = dyn_cast<CallInst>(inst)) {
    StringRef funcName = getFuncNameFromCall(const_cast<CallInst *>(CI));
    if (funcName == "julia.get_pgcstack" || funcName == "julia.ptls_states")
      return seen[idx] = true;
  }

  bool inst_cv = gutils->isConstantValue(const_cast<Value *>(inst));

  // Consider all users of this value, do any of them need this in the reverse?
  for (auto use : inst->users()) {
    if (use == inst)
      continue;

    const Instruction *user = dyn_cast<Instruction>(use);

    // A shadow value is only needed in reverse if it or one of its descendants
    // is used in an active instruction.
    // If inst is a constant value, the primal may be used in its place and
    // thus required.
    if (VT == QueryType::Shadow || VT == QueryType::ShadowByConstPrimal ||
        inst_cv) {
      bool recursiveUse = false;
      if (is_use_directly_needed_in_reverse(
              gutils, inst, mode, user, oldUnreachable,
              (VT == QueryType::Shadow) ? QueryType::Shadow
                                        : QueryType::ShadowByConstPrimal,
              &recursiveUse)) {
        return seen[idx] = true;
      }

      if (recursiveUse && !OneLevel) {
        bool val;
        if (VT == QueryType::Shadow)
          val = is_value_needed_in_reverse<QueryType::Shadow>(
              gutils, user, mode, seen, oldUnreachable);
        else
          val = is_value_needed_in_reverse<QueryType::ShadowByConstPrimal>(
              gutils, user, mode, seen, oldUnreachable);
        if (val) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow sub-need " << *user << "\n";
          return seen[idx] = true;
        }
      }

      if (VT != QueryType::Primal)
        continue;
    }

    assert(VT == QueryType::Primal);

    // If a sub user needs, we need
    if (!OneLevel && is_value_needed_in_reverse<VT>(gutils, user, mode, seen,
                                                    oldUnreachable)) {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                     << " in reverse as sub-need " << *user << "\n";
      return seen[idx] = true;
    }

    // Anything we may try to rematerialize requires its store operands for
    // the reverse pass.
    if (!OneLevel) {
      if (isa<StoreInst>(user) || isa<MemTransferInst>(user) ||
          isa<MemSetInst>(user)) {
        for (auto pair : gutils->rematerializableAllocations) {
          // If caching the outer allocation and have already set that this is
          // not needed return early. This is necessary to avoid unnecessarily
          // deciding stored values are needed if we have already decided to
          // cache the whole allocation.
          auto found = seen.find(std::make_pair(pair.first, QueryType::Primal));
          if (found != seen.end() && !found->second)
            continue;

          // Directly consider all the load uses to avoid an illegal inductive
          // recurrence. Specifically if we're asking if the alloca is used,
          // we'll set it to unused, then check the gep, then here we'll
          // directly say unused by induction instead of checking the final
          // loads.
          if (pair.second.stores.count(user)) {
            for (LoadInst *L : pair.second.loads)
              if (is_value_needed_in_reverse<VT>(gutils, L, mode, seen,
                                                 oldUnreachable)) {
                if (EnzymePrintDiffUse)
                  llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                               << " in reverse as rematload " << *L << "\n";
                return seen[idx] = true;
              }
            for (auto &pair : pair.second.loadLikeCalls)
              if (is_use_directly_needed_in_reverse(
                      gutils, pair.operand, mode, pair.loadCall, oldUnreachable,
                      QueryType::Primal) ||
                  is_value_needed_in_reverse<VT>(gutils, pair.loadCall, mode,
                                                 seen, oldUnreachable)) {
                if (EnzymePrintDiffUse)
                  llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                               << " in reverse as rematloadcall "
                               << *pair.loadCall << "\n";
                return seen[idx] = true;
              }
          }
        }
      }
    }

    // One may need to this value in the computation of loop
    // bounds/comparisons/etc (which even though not active -- will be used for
    // the reverse pass)
    //   We could potentially optimize this to avoid caching if in combined mode
    //   and the instruction dominates all returns
    //   otherwise it will use the local cache (rather than save for a separate
    //   backwards cache)
    //   We also don't need this if looking at the shadow rather than primal
    {
      // Proving that none of the uses (or uses' uses) are used in control flow
      // allows us to safely not do this load

      // TODO save loop bounds for dynamic loop

      // TODO make this more aggressive and dont need to save loop latch
      if (isa<BranchInst>(use) || isa<SwitchInst>(use)) {
        size_t num = 0;
        for (auto suc : successors(cast<Instruction>(use)->getParent())) {
          if (!oldUnreachable.count(suc)) {
            num++;
          }
        }
        if (num <= 1)
          continue;
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                       << " in reverse as control-flow " << *user << "\n";
        return seen[idx] = true;
      }

      if (auto CI = dyn_cast<CallInst>(use)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                           << " in reverse as omp init " << *user << "\n";
            return seen[idx] = true;
          }
        }
      }
    }

    // The following are types we know we don't need to compute adjoints

    // If a primal value is needed to compute a shadow pointer (e.g. int offset
    // in gep), it needs preserving.
    bool primalUsedInShadowPointer = true;
    if (isa<CastInst>(user) || isa<LoadInst>(user))
      primalUsedInShadowPointer = false;
    if (auto CI = dyn_cast<CallInst>(user)) {
      auto funcName = getFuncNameFromCall(CI);
      if (funcName == "julia.pointer_from_objref") {
        primalUsedInShadowPointer = false;
      }
      if (funcName.contains("__enzyme_todense")) {
        primalUsedInShadowPointer = false;
      }
    }
    if (auto GEP = dyn_cast<GetElementPtrInst>(user)) {
      bool idxUsed = false;
      for (auto &idx : GEP->indices()) {
        if (idx.get() == inst)
          idxUsed = true;
      }
      if (!idxUsed)
        primalUsedInShadowPointer = false;
    }
    if (auto II = dyn_cast<IntrinsicInst>(user)) {
      if (isIntelSubscriptIntrinsic(*II)) {
        const std::array<size_t, 4> idxArgsIndices{{0, 1, 2, 4}};
        bool idxUsed = false;
        for (auto i : idxArgsIndices) {
          if (II->getOperand(i) == inst)
            idxUsed = true;
        }
        if (!idxUsed)
          primalUsedInShadowPointer = false;
      }
    }
    if (auto IVI = dyn_cast<InsertValueInst>(user)) {
      bool valueIsIndex = false;
      for (unsigned i = 2; i < IVI->getNumOperands(); ++i) {
        if (IVI->getOperand(i) == inst) {
          if (inst == IVI->getInsertedValueOperand() &&
              TR.query(
                    const_cast<Value *>(IVI->getInsertedValueOperand()))[{-1}]
                  .isFloat()) {
            continue;
          }
          valueIsIndex = true;
        }
      }
      primalUsedInShadowPointer = valueIsIndex;
    }
    if (auto EVI = dyn_cast<ExtractValueInst>(user)) {
      bool valueIsIndex = false;
      for (unsigned i = 1; i < EVI->getNumOperands(); ++i) {
        if (EVI->getOperand(i) == inst) {
          valueIsIndex = true;
        }
      }
      primalUsedInShadowPointer = valueIsIndex;
    }

    if (primalUsedInShadowPointer)
      if (!user->getType()->isVoidTy() &&
          TR.query(const_cast<Instruction *>(user))
              .Inner0()
              .isPossiblePointer()) {
        if (is_value_needed_in_reverse<QueryType::Shadow>(
                gutils, user, mode, seen, oldUnreachable)) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as used to compute shadow ptr "
                         << *user << "\n";
          return seen[idx] = true;
        }
      }

    bool direct = is_use_directly_needed_in_reverse(
        gutils, inst, mode, user, oldUnreachable, QueryType::Primal);
    if (!direct)
      continue;

    if (inst->getType()->isTokenTy()) {
      llvm::errs() << " need " << *inst << " via " << *user << "\n";
    }
    assert(!inst->getType()->isTokenTy());

    return seen[idx] = true;
  }
  return false;
}

template <QueryType VT>
static inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *inst, DerivativeMode mode,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable) {
  static_assert(VT == QueryType::Primal || VT == QueryType::Shadow);
  std::map<UsageKey, bool> seen;
  return is_value_needed_in_reverse<VT>(gutils, inst, mode, seen,
                                        oldUnreachable);
}

struct Node {
  llvm::Value *V;
  bool outgoing;
  Node(llvm::Value *V, bool outgoing) : V(V), outgoing(outgoing){};
  bool operator<(const Node N) const {
    if (V < N.V)
      return true;
    return !(N.V < V) && outgoing < N.outgoing;
  }
  void dump() {
    if (V)
      llvm::errs() << "[" << *V << ", " << (int)outgoing << "]\n";
    else
      llvm::errs() << "[" << V << ", " << (int)outgoing << "]\n";
  }
};

using Graph = std::map<Node, std::set<Node>>;

void dump(std::map<Node, std::set<Node>> &G);

/* Returns true if there is a path from source 's' to sink 't' in
 residual graph. Also fills parent[] to store the path */
void bfs(const std::map<Node, std::set<Node>> &G,
         const llvm::SmallPtrSetImpl<llvm::Value *> &Recompute,
         std::map<Node, Node> &parent);

// Return 1 if next is better
// 0 if equal
// -1 if prev is better, or unknown
int cmpLoopNest(llvm::Loop *prev, llvm::Loop *next);

void minCut(const llvm::DataLayout &DL, llvm::LoopInfo &OrigLI,
            const llvm::SmallPtrSetImpl<llvm::Value *> &Recomputes,
            const llvm::SmallPtrSetImpl<llvm::Value *> &Intermediates,
            llvm::SmallPtrSetImpl<llvm::Value *> &Required,
            llvm::SmallPtrSetImpl<llvm::Value *> &MinReq,
            const llvm::ValueMap<llvm::Value *, GradientUtils::Rematerializer>
                &rematerializableAllocations,
            llvm::TargetLibraryInfo &TLI);

}; // namespace DifferentialUseAnalysis

#endif
