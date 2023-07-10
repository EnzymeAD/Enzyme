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

namespace DifferentialUseAnalysis {

/// Determine if a value is needed directly to compute the adjoint
/// of the given instruction user
bool is_use_directly_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *val,
    const llvm::Instruction *user,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable);

template <ValueType VT, bool OneLevel = false>
inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *inst, DerivativeMode mode,
    std::map<UsageKey, bool> &seen,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable) {
  using namespace llvm;

  TypeResults const &TR = gutils->TR;
  static_assert(VT == ValueType::Primal || VT == ValueType::Shadow);
  auto idx = UsageKey(inst, VT);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  if (VT != ValueType::Shadow) {
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
    if (VT == ValueType::Shadow || inst_cv) {

      // Floating point numbers cannot be used as a shadow pointer/etc
      if (inst_cv || (mode != DerivativeMode::ForwardMode &&
                      mode != DerivativeMode::ForwardModeSplit))
        if (TR.query(const_cast<Value *>(inst))[{-1}].isFloat())
          goto endShadow;

      if (!user) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                       << " in reverse as unknown user " << *use << "\n";
        return seen[idx] = true;
      }

      if (auto SI = dyn_cast<StoreInst>(user)) {
        if (mode == DerivativeMode::ReverseModeGradient ||
            mode == DerivativeMode::ForwardModeSplit) {

          bool rematerialized = false;
          for (auto pair : gutils->backwardsOnlyShadows)
            if (pair.second.stores.count(SI)) {
              rematerialized = true;
              break;
            }

          if (SI->getValueOperand() == inst) {
            // storing an active pointer into a location
            // doesn't require the shadow pointer for the
            // reverse pass
            // Unless the store is into a backwards store, which would
            // would then be performed in the reverse if the stored value was
            // a possible pointer.
            if (!rematerialized)
              goto endShadow;
          } else {
            // Likewise, if not rematerializing in reverse pass, you
            // don't need to keep the pointer operand for known pointers
            if (!rematerialized &&
                TR.query(const_cast<Value *>(SI->getValueOperand()))[{-1}] ==
                    BaseType::Pointer)
              goto endShadow;
          }
        }

        if (!gutils->isConstantValue(
                const_cast<Value *>(SI->getPointerOperand()))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow store  " << *SI << "\n";
          return seen[idx] = true;
        } else
          goto endShadow;
      }

      if (auto MTI = dyn_cast<MemTransferInst>(user)) {
        if (MTI->getArgOperand(0) != inst && MTI->getArgOperand(1) != inst)
          goto endShadow;

        if (!gutils->isConstantValue(
                const_cast<Value *>(MTI->getArgOperand(0)))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow MTI  " << *MTI << "\n";
          return seen[idx] = true;
        } else
          goto endShadow;
      }

      if (auto MS = dyn_cast<MemSetInst>(user)) {
        if (MS->getArgOperand(0) != inst)
          goto endShadow;

        if (!gutils->isConstantValue(
                const_cast<Value *>(MS->getArgOperand(0)))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow MS  " << *MS << "\n";
          return seen[idx] = true;
        } else
          goto endShadow;
      }

      if (auto CI = dyn_cast<CallInst>(user)) {
        {
          SmallVector<OperandBundleDef, 2> OrigDefs;
          CI->getOperandBundlesAsDefs(OrigDefs);
          SmallVector<OperandBundleDef, 2> Defs;
          for (auto bund : OrigDefs) {
            for (auto inp : bund.inputs()) {
              if (inp == inst)
                return seen[idx] = true;
            }
          }
        }
        StringRef funcName = getFuncNameFromCall(const_cast<CallInst *>(CI));

        // Don't need shadow inputs for alloc function
        if (isAllocationFunction(funcName, gutils->TLI))
          goto endShadow;

        // Even though inactive, keep the shadow pointer around in forward mode
        // to perform the same memory free behavior on the shadow.
        if (mode == DerivativeMode::ForwardMode &&
            isDeallocationFunction(funcName, gutils->TLI)) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow free " << *CI << "\n";
          return seen[idx] = true;
        }

        // Only need shadow request for reverse
        if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
          if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
            goto endShadow;
          // Need shadow request
          if (inst == CI->getArgOperand(6)) {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                           << " in reverse as shadow MPI " << *CI << "\n";
            return seen[idx] = true;
          }
          // Need shadow buffer in forward pass
          if (mode != DerivativeMode::ReverseModeGradient)
            if (inst == CI->getArgOperand(0)) {
              if (EnzymePrintDiffUse)
                llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                             << " in reverse as shadow MPI " << *CI << "\n";
              return seen[idx] = true;
            }
          goto endShadow;
        }
        if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
            goto endShadow;
          // Need shadow request
          if (inst == CI->getArgOperand(6)) {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                           << " in reverse as shadow MPI " << *CI << "\n";
            return seen[idx] = true;
          }
          // Need shadow buffer in reverse pass or forward mode
          if (inst == CI->getArgOperand(0)) {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                           << " in reverse as shadow MPI " << *CI << "\n";
            return seen[idx] = true;
          }
          goto endShadow;
        }

        // Don't need shadow of anything (all via cache for reverse),
        // but need shadow of request for primal.
        if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
          if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
            goto endShadow;
          // Need shadow request in forward pass only
          if (mode != DerivativeMode::ReverseModeGradient)
            if (inst == CI->getArgOperand(0)) {
              if (EnzymePrintDiffUse)
                llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                             << " in reverse as shadow MPI " << *CI << "\n";
              return seen[idx] = true;
            }
          goto endShadow;
        }

        // Don't need shadow of anything (all via cache for reverse),
        // but need shadow of request for primal.
        if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
          if (gutils->isConstantInstruction(const_cast<Instruction *>(user)))
            goto endShadow;
          // Need shadow request in forward pass
          if (mode != DerivativeMode::ReverseModeGradient)
            if (inst == CI->getArgOperand(1)) {
              if (EnzymePrintDiffUse)
                llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                             << " in reverse as shadow MPI " << *CI << "\n";
              return seen[idx] = true;
            }
          goto endShadow;
        }

        // Use in a write barrier requires the shadow in the forward, even
        // though the instruction is active.
        if (mode != DerivativeMode::ReverseModeGradient &&
            funcName == "julia.write_barrier") {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow write_barrier " << *CI
                         << "\n";
          return seen[idx] = true;
        }

        bool writeOnlyNoCapture = true;
#if LLVM_VERSION_MAJOR >= 14
        for (size_t i = 0; i < CI->arg_size(); i++)
#else
        for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
        {
          if (inst == CI->getArgOperand(i)) {
            if (!isNoCapture(CI, i)) {
              writeOnlyNoCapture = false;
              break;
            }
            if (!isWriteOnly(CI, i)) {
              writeOnlyNoCapture = false;
              break;
            }
          }
        }
        // Don't need the shadow argument if it is a pointer to pointers, which
        // is only written since the shadow pointer store will have been
        // completed in the forward pass.
        if (writeOnlyNoCapture &&
            TR.query(const_cast<Value *>(inst))[{-1, -1}] ==
                BaseType::Pointer &&
            mode == DerivativeMode::ReverseModeGradient)
          return false;

#if LLVM_VERSION_MAJOR >= 11
        const Value *FV = CI->getCalledOperand();
#else
        const Value *FV = CI->getCalledValue();
#endif
        if (FV == inst) {
          if (!gutils->isConstantInstruction(const_cast<Instruction *>(user)) ||
              !gutils->isConstantValue(const_cast<Value *>((Value *)user))) {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                           << " in reverse as shadow call " << *CI << "\n";
            return seen[idx] = true;
          }
        }
      }

      if (isa<ReturnInst>(user)) {
        if ((gutils->ATA->ActiveReturns == DIFFE_TYPE::DUP_ARG ||
             gutils->ATA->ActiveReturns == DIFFE_TYPE::DUP_NONEED) &&
            ((inst_cv && VT == ValueType::Primal) ||
             (!inst_cv && VT == ValueType::Shadow))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow return " << *user << "\n";
          return seen[idx] = true;
        } else
          goto endShadow;
      }

      // With certain exceptions, assume active instructions require the
      // shadow of the operand.
      if (mode == DerivativeMode::ForwardMode ||
          mode == DerivativeMode::ForwardModeSplit ||
          (!isa<ExtractValueInst>(user) && !isa<ExtractElementInst>(user) &&
           !isa<InsertValueInst>(user) && !isa<InsertElementInst>(user) &&
           !isPointerArithmeticInst(user, /*includephi*/ false,
                                    /*includebin*/ false))) {
        if (!inst_cv &&
            !gutils->isConstantInstruction(const_cast<Instruction *>(user))) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as shadow inst " << *user << "\n";
          return seen[idx] = true;
        }
      }

      // Now the remaining instructions are inactive, however note that
      // a constant instruction may still require the use of the shadow
      // in the forward pass, for example double* x = load double** y
      // is a constant instruction, but needed in the forward
      if (user->getType()->isVoidTy())
        goto endShadow;

      if (!TR.query(const_cast<Instruction *>(user))[{-1}]
               .isPossiblePointer()) {
        goto endShadow;
      }

      if (!OneLevel && is_value_needed_in_reverse<ValueType::Shadow>(
                           gutils, user, mode, seen, oldUnreachable)) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                       << " in reverse as shadow sub-need " << *user << "\n";
        return seen[idx] = true;
      }
    endShadow:
      if (VT != ValueType::Primal)
        continue;
    }

    assert(VT == ValueType::Primal);

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
          auto found = seen.find(std::make_pair(pair.first, ValueType::Primal));
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
                      gutils, pair.operand, pair.loadCall, oldUnreachable) ||
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
        if (is_value_needed_in_reverse<ValueType::Shadow>(
                gutils, user, mode, seen, oldUnreachable)) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need: " << to_string(VT) << " of " << *inst
                         << " in reverse as used to compute shadow ptr "
                         << *user << "\n";
          return seen[idx] = true;
        }
      }

    bool direct =
        is_use_directly_needed_in_reverse(gutils, inst, user, oldUnreachable);
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

template <ValueType VT>
static inline bool is_value_needed_in_reverse(
    const GradientUtils *gutils, const llvm::Value *inst, DerivativeMode mode,
    const llvm::SmallPtrSetImpl<llvm::BasicBlock *> &oldUnreachable) {
  static_assert(VT == ValueType::Primal || VT == ValueType::Shadow);
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
