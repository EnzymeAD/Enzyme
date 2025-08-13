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
#include "Poseidon/Poseidon.h"

extern "C" {
extern llvm::cl::opt<bool> EnzymePrintDiffUse;
}

extern llvm::StringMap<
    std::function<bool(const llvm::CallInst *, const GradientUtils *,
                       const llvm::Value *, bool, DerivativeMode, bool &)>>
    customDiffUseHandlers;

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
#ifdef ENZYME_ENABLE_FPOPT
    if ((FPProfileGenerate ||
         gutils->mode == DerivativeMode::ForwardModeError) &&
        !gutils->isConstantValue(const_cast<Value *>(inst))) {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                     << " in reverse as FPOpt profiler mode or forward mode "
                        "error always needs result\n";
      return seen[idx] = true;
    }
#endif
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

      if (!TR.anyFloat(const_cast<Value *>(inst)))
        if (auto IVI = dyn_cast<Instruction>(user)) {
          bool inserted = false;
          if (auto II = dyn_cast<InsertValueInst>(IVI))
            inserted = II->getInsertedValueOperand() == inst ||
                       II->getAggregateOperand() == inst;
          if (auto II = dyn_cast<ExtractValueInst>(IVI))
            inserted = II->getAggregateOperand() == inst;
          if (auto II = dyn_cast<InsertElementInst>(IVI))
            inserted = II->getOperand(1) == inst || II->getOperand(0) == inst;
          if (auto II = dyn_cast<ExtractElementInst>(IVI))
            inserted = II->getOperand(0) == inst;
          if (inserted) {
            SmallVector<const Instruction *, 1> todo;
            todo.push_back(IVI);
            while (todo.size()) {
              auto cur = todo.pop_back_val();
              for (auto u : cur->users()) {
                if (auto IVI2 = dyn_cast<InsertValueInst>(u)) {
                  todo.push_back(IVI2);
                  continue;
                }
                if (auto IVI2 = dyn_cast<ExtractValueInst>(u)) {
                  todo.push_back(IVI2);
                  continue;
                }
                if (auto IVI2 = dyn_cast<InsertElementInst>(u)) {
                  todo.push_back(IVI2);
                  continue;
                }
                if (auto IVI2 = dyn_cast<ExtractElementInst>(u)) {
                  todo.push_back(IVI2);
                  continue;
                }

                bool partial = false;
                if (!gutils->isConstantValue(const_cast<Instruction *>(cur))) {
                  partial = is_value_needed_in_reverse<QueryType::Shadow>(
                      gutils, user, mode, seen, oldUnreachable);
                }
                if (partial) {

                  if (EnzymePrintDiffUse)
                    llvm::errs()
                        << " Need (partial) direct " << to_string(VT) << " of "
                        << *inst << " in reverse from insertelem " << *user
                        << " via " << *cur << " in " << *u << "\n";
                  return seen[idx] = true;
                }
              }
            }
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
      bool isStored = false;
      if (auto SI = dyn_cast<StoreInst>(user))
        isStored = inst == SI->getValueOperand();
      else if (auto MTI = dyn_cast<MemTransferInst>(user)) {
        isStored = inst == MTI->getSource() || inst == MTI->getLength();
      } else if (auto MS = dyn_cast<MemSetInst>(user)) {
        isStored = inst == MS->getLength() || inst == MS->getValue();
      } else if (auto CB = dyn_cast<CallBase>(user)) {
        auto name = getFuncNameFromCall(CB);
        if (name == "julia.write_barrier" ||
            name == "julia.write_barrier_binding") {
          auto sz = CB->arg_size();
          // First pointer is the destination
          for (size_t i = 1; i < sz; i++)
            isStored |= inst == CB->getArgOperand(i);
        }
      }
      if (isStored) {
        for (auto pair : gutils->rematerializableAllocations) {
          // If already decided to cache the whole allocation, ignore
          if (gutils->needsCacheWholeAllocation(pair.first)) {
            continue;
          }

          // If caching the outer allocation and have already set that this is
          // not needed return early. This is necessary to avoid unnecessarily
          // deciding stored values are needed if we have already decided to
          // cache the whole allocation.
          auto found = seen.find(std::make_pair(pair.first, QueryType::Primal));
          if (found != seen.end() && !found->second) {
            continue;
          }

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

            if (is_value_needed_in_reverse<VT>(gutils, pair.first, mode, seen,
                                               oldUnreachable)) {
              if (EnzymePrintDiffUse)
                llvm::errs()
                    << " Need: " << to_string(VT) << " of " << *inst
                    << " in reverse as rematalloc " << *pair.first << "\n";
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
      if (funcName == "julia.gc_loaded") {
        primalUsedInShadowPointer = false;
      }
      if (funcName.contains("__enzyme_todense")) {
        primalUsedInShadowPointer = false;
      }
      if (funcName.contains("__enzyme_ignore_derivatives")) {
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
    // No need for insert/extractvalue since indices are unsigned
    //  not llvm runtime values
    if (isa<InsertValueInst>(user) || isa<ExtractValueInst>(user))
      primalUsedInShadowPointer = false;

    if (primalUsedInShadowPointer)
      if (!user->getType()->isVoidTy() &&
          TR.anyPointer(const_cast<Instruction *>(user))) {
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
         const llvm::SetVector<llvm::Value *> &Recompute,
         std::map<Node, Node> &parent);

// Return 1 if next is better
// 0 if equal
// -1 if prev is better, or unknown
int cmpLoopNest(llvm::Loop *prev, llvm::Loop *next);

void minCut(const llvm::DataLayout &DL, llvm::LoopInfo &OrigLI,
            const llvm::SetVector<llvm::Value *> &Recomputes,
            const llvm::SetVector<llvm::Value *> &Intermediates,
            llvm::SetVector<llvm::Value *> &Required,
            llvm::SetVector<llvm::Value *> &MinReq, const GradientUtils *gutils,
            llvm::TargetLibraryInfo &TLI);

__attribute__((always_inline)) static inline void
forEachDirectInsertUser(llvm::function_ref<void(llvm::Instruction *)> f,
                        const GradientUtils *gutils, llvm::Instruction *IVI,
                        llvm::Value *val, bool useCheck) {
  using namespace llvm;
  if (!gutils->isConstantValue(IVI))
    return;
  bool inserted = false;
  if (auto II = dyn_cast<InsertValueInst>(IVI))
    inserted = II->getInsertedValueOperand() == val ||
               II->getAggregateOperand() == val;
  if (auto II = dyn_cast<ExtractValueInst>(IVI))
    inserted = II->getAggregateOperand() == val;
  if (auto II = dyn_cast<InsertElementInst>(IVI))
    inserted = II->getOperand(1) == val || II->getOperand(0) == val;
  if (auto II = dyn_cast<ExtractElementInst>(IVI))
    inserted = II->getOperand(0) == val;
  if (inserted) {
    SmallVector<Instruction *, 1> todo;
    todo.push_back(IVI);
    while (todo.size()) {
      auto cur = todo.pop_back_val();
      for (auto u : cur->users()) {
        if (isa<InsertValueInst>(u) || isa<InsertElementInst>(u) ||
            isa<ExtractValueInst>(u) || isa<ExtractElementInst>(u)) {
          auto I2 = cast<Instruction>(u);
          bool subCheck = useCheck;
          if (!subCheck) {
            subCheck = is_value_needed_in_reverse<QueryType::Shadow>(
                gutils, I2, gutils->mode, gutils->notForAnalysis);
          }
          if (subCheck)
            f(I2);
          todo.push_back(I2);
          continue;
        }
      }
    }
  }
}

__attribute__((always_inline)) static inline void
forEachDifferentialUser(llvm::function_ref<void(llvm::Value *)> f,
                        const GradientUtils *gutils, llvm::Value *V,
                        bool useCheck = false) {
  for (auto V2 : V->users()) {
    if (auto Inst = llvm::dyn_cast<llvm::Instruction>(V2)) {
      for (const auto &pair : gutils->rematerializableAllocations) {
        if (pair.second.stores.count(Inst)) {
          f(llvm::cast<llvm::Instruction>(pair.first));
        }
      }
      f(Inst);
      forEachDirectInsertUser(f, gutils, Inst, V, useCheck);
    }
  }
}

//! Return whether or not this is a constant and should use reverse pass
bool callShouldNotUseDerivative(const GradientUtils *gutils,
                                llvm::CallBase &orig);

}; // namespace DifferentialUseAnalysis

#endif
