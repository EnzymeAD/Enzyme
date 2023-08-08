//===- DifferentialUseAnalysis.cpp - Determine values needed in reverse
// pass-===//
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

#include <deque>
#include <map>
#include <set>

#include "DifferentialUseAnalysis.h"

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

using namespace llvm;

bool DifferentialUseAnalysis::is_use_directly_needed_in_reverse(
    const GradientUtils *gutils, const Value *val, const Instruction *user,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  TypeResults const &TR = gutils->TR;
  if (auto ainst = dyn_cast<Instruction>(val)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }
  assert(user->getParent()->getParent() == gutils->oldFunc);

  if (oldUnreachable.count(user->getParent()))
    return false;

  if (isPointerArithmeticInst(user, /*includephi*/ true,
                              /*includebin*/ false)) {
    return false;
  }

  if (auto LI = dyn_cast<LoadInst>(user)) {
    if (EnzymeRuntimeActivityCheck) {
      auto vd = TR.query(const_cast<llvm::Instruction *>(user));
      if (!vd.isKnown()) {
        auto ET = LI->getType();
        // It verbatim needs to replicate the same behavior as adjointgenerator.
        // From reverse mode type analysis
        // (https://github.com/EnzymeAD/Enzyme/blob/194875cbccd73d63cacfefbfa85c1f583c2fa1fe/enzyme/Enzyme/AdjointGenerator.h#L556)
        if (looseTypeAnalysis || true) {
          vd = defaultTypeTreeForLLVM(ET, const_cast<LoadInst *>(LI));
        }
      }
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      auto LoadSize = (DL.getTypeSizeInBits(LI->getType()) + 1) / 8;
      bool hasFloat = true;
      for (ssize_t i = -1; i < (ssize_t)LoadSize; ++i) {
        if (vd[{(int)i}].isFloat()) {
          hasFloat = true;
          break;
        }
      }
      if (hasFloat && !gutils->isConstantInstruction(
                          const_cast<llvm::Instruction *>(user))) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need direct primal of " << *val
                       << " in reverse from runtime active load " << *user
                       << "\n";
        return true;
      }
    }
    return false;
  }

  // We don't need any of the input operands to compute the adjoint of a store
  // instance
  if (auto SI = dyn_cast<StoreInst>(user)) {
    // The one exception to this is stores to the loop bounds.
    if (SI->getValueOperand() == val) {
      for (auto U : SI->getPointerOperand()->users()) {
        if (auto CI = dyn_cast<CallInst>(U)) {
          if (auto F = CI->getCalledFunction()) {
            if (F->getName() == "__kmpc_for_static_init_4" ||
                F->getName() == "__kmpc_for_static_init_4u" ||
                F->getName() == "__kmpc_for_static_init_8" ||
                F->getName() == "__kmpc_for_static_init_8u") {
              if (CI->getArgOperand(4) == val || CI->getArgOperand(5) == val ||
                  CI->getArgOperand(6)) {
                if (EnzymePrintDiffUse)
                  llvm::errs() << " Need direct primal of " << *val
                               << " in reverse from omp " << *user << "\n";
                return true;
              }
            }
          }
        }
      }

      // Preserve any non-floating point values that are stored in an active
      // backwards creation shadow.
      if (!TR.query(const_cast<Value *>(SI->getValueOperand()))[{-1}].isFloat())
        for (auto pair : gutils->backwardsOnlyShadows)
          if (pair.second.stores.count(SI)) {
            if (EnzymePrintDiffUse)
              llvm::errs() << " Need direct primal of " << *val
                           << " in reverse from remat store " << *user << "\n";
            return true;
          }
    }
    return false;
  }

  // If memtransfer, only the size may need preservation for the reverse pass
  if (auto MTI = dyn_cast<MemTransferInst>(user)) {
    // Unless we're storing into a backwards only shadow store
    if (MTI->getArgOperand(1) == val || MTI->getArgOperand(2) == val) {
      for (auto pair : gutils->backwardsOnlyShadows)
        if (pair.second.stores.count(MTI)) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need direct primal of " << *val
                         << " in reverse from remat memtransfer " << *user
                         << "\n";
          return true;
        }
    }
    if (MTI->getArgOperand(2) != val)
      return false;
    bool res = !gutils->isConstantValue(MTI->getArgOperand(0));
    if (res) {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need direct primal of " << *val
                     << " in reverse from memtransfer " << *user << "\n";
    }
    return res;
  }

  // Preserve the length of memsets of backward creation shadows,
  // or if float-like and non constant value.
  if (auto MS = dyn_cast<MemSetInst>(user)) {
    if (MS->getArgOperand(1) == val || MS->getArgOperand(2) == val) {
      for (auto pair : gutils->backwardsOnlyShadows)
        if (pair.second.stores.count(MS)) {
          if (EnzymePrintDiffUse)
            llvm::errs() << " Need direct primal of " << *val
                         << " in reverse from remat memset " << *user << "\n";
          return true;
        }
      bool res = !gutils->isConstantValue(MS->getArgOperand(0));
      if (res) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need direct primal of " << *val
                       << " in reverse from memset " << *user << "\n";
      }
      return res;
    }
  }

  if (isa<CmpInst>(user) || isa<BranchInst>(user) || isa<ReturnInst>(user) ||
      isa<FPExtInst>(user) || isa<FPTruncInst>(user)
#if LLVM_VERSION_MAJOR >= 10
      || isa<FreezeInst>(user)
#endif
      // isa<ExtractElement>(use) ||
      // isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
      // isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
      // || isa<StoreInst>(use)
  ) {
    return false;
  }

  if (auto IEI = dyn_cast<InsertElementInst>(user)) {
    // Only need the index in the reverse, so if the value is not
    // the index, short circuit and say we don't need
    if (IEI->getOperand(2) != val) {
      return false;
    }
    // The index is only needed in the reverse if the value being inserted
    // is a possible active floating point value
    if (gutils->isConstantValue(const_cast<InsertElementInst *>(IEI)) ||
        TR.query(const_cast<InsertElementInst *>(IEI))[{-1}] ==
            BaseType::Pointer)
      return false;
    // Otherwise, we need the value.
    if (EnzymePrintDiffUse)
      llvm::errs() << " Need direct primal of " << *val
                   << " in reverse from non-pointer insertelem " << *user << " "
                   << TR.query(const_cast<InsertElementInst *>(IEI)).str()
                   << "\n";
    return true;
  }
  if (auto EEI = dyn_cast<ExtractElementInst>(user)) {
    // Only need the index in the reverse, so if the value is not
    // the index, short circuit and say we don't need
    if (EEI->getIndexOperand() != val) {
      return false;
    }
    // The index is only needed in the reverse if the value being inserted
    // is a possible active floating point value
    if (gutils->isConstantValue(const_cast<ExtractElementInst *>(EEI)) ||
        TR.query(const_cast<ExtractElementInst *>(EEI))[{-1}] ==
            BaseType::Pointer)
      return false;
    // Otherwise, we need the value.
    if (EnzymePrintDiffUse)
      llvm::errs() << " Need direct primal of " << *val
                   << " in reverse from non-pointer extractelem " << *user
                   << " "
                   << TR.query(const_cast<ExtractElementInst *>(EEI)).str()
                   << "\n";
    return true;
  }

  if (auto IVI = dyn_cast<InsertValueInst>(user)) {
    // Only need the index in the reverse, so if the value is not
    // the index, short circuit and say we don't need
    bool valueIsIndex = false;
    for (unsigned i = 2; i < IVI->getNumOperands(); ++i) {
      if (IVI->getOperand(i) == val) {
        valueIsIndex = true;
      }
    }

    if (!valueIsIndex)
      return false;

    // The index is only needed in the reverse if the value being inserted
    // is a possible active floating point value
    if (gutils->isConstantValue(const_cast<InsertValueInst *>(IVI)) ||
        TR.query(const_cast<InsertValueInst *>(IVI))[{-1}] == BaseType::Pointer)
      return false;
    // Otherwise, we need the value.
    if (EnzymePrintDiffUse)
      llvm::errs() << " Need direct primal of " << *val
                   << " in reverse from non-pointer insertval " << *user << " "
                   << TR.query(const_cast<InsertValueInst *>(IVI)).str()
                   << "\n";
    return true;
  }

  if (auto EVI = dyn_cast<ExtractValueInst>(user)) {
    // Only need the index in the reverse, so if the value is not
    // the index, short circuit and say we don't need
    bool valueIsIndex = false;
    for (unsigned i = 2; i < EVI->getNumOperands(); ++i) {
      if (EVI->getOperand(i) == val) {
        valueIsIndex = true;
      }
    }

    if (!valueIsIndex)
      return false;

    // The index is only needed in the reverse if the value being inserted
    // is a possible active floating point value
    if (gutils->isConstantValue(const_cast<ExtractValueInst *>(EVI)) ||
        TR.query(const_cast<ExtractValueInst *>(EVI))[{-1}] ==
            BaseType::Pointer)
      return false;
    // Otherwise, we need the value.
    if (EnzymePrintDiffUse)
      llvm::errs() << " Need direct primal of " << *val
                   << " in reverse from non-pointer extractval " << *user << " "
                   << TR.query(const_cast<ExtractValueInst *>(EVI)).str()
                   << "\n";
    return true;
  }

  Intrinsic::ID ID = Intrinsic::not_intrinsic;
  if (auto II = dyn_cast<IntrinsicInst>(user)) {
    ID = II->getIntrinsicID();
  } else if (auto CI = dyn_cast<CallInst>(user)) {
    StringRef funcName = getFuncNameFromCall(const_cast<CallInst *>(CI));
    isMemFreeLibMFunction(funcName, &ID);
  }

  if (ID != Intrinsic::not_intrinsic) {
    if (ID == Intrinsic::lifetime_start || ID == Intrinsic::lifetime_end ||
        ID == Intrinsic::stacksave || ID == Intrinsic::stackrestore) {
      return false;
    }
    if (ID == Intrinsic::fma || ID == Intrinsic::fmuladd) {
      bool needed = false;
      if (user->getOperand(0) == val &&
          !gutils->isConstantValue(user->getOperand(1)))
        needed = true;
      if (user->getOperand(1) == val &&
          !gutils->isConstantValue(user->getOperand(0)))
        needed = true;
      if (needed) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need direct primal of " << *val
                       << " in reverse from fma " << *user << "\n";
      }
      return needed;
    }
  }

  if (auto op = dyn_cast<BinaryOperator>(user)) {
    if (op->getOpcode() == Instruction::FAdd ||
        op->getOpcode() == Instruction::FSub) {
      return false;
    } else if (op->getOpcode() == Instruction::FMul) {
      bool needed = false;
      if (op->getOperand(0) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(0)))
        needed = true;
      if (needed) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need direct primal of " << *val
                       << " in reverse from fmul " << *user << "\n";
      }
      return needed;
    } else if (op->getOpcode() == Instruction::FDiv) {
      bool needed = false;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      if (op->getOperand(1) == val &&
          !gutils->isConstantValue(op->getOperand(0)))
        needed = true;
      if (op->getOperand(0) == val &&
          !gutils->isConstantValue(op->getOperand(1)))
        needed = true;
      if (needed) {
        if (EnzymePrintDiffUse)
          llvm::errs() << " Need direct primal of " << *val
                       << " in reverse from fdiv " << *user << "\n";
      }
      return needed;
    }
  }

  if (auto si = dyn_cast<SelectInst>(user)) {
    // Only would potentially need the condition
    if (si->getCondition() != val) {
      return false;
    }

    // only need the condition if select is active
    bool needed = !gutils->isConstantValue(const_cast<SelectInst *>(si));
    if (needed) {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need direct primal of " << *val
                     << " in reverse from select " << *user << "\n";
    }
    return needed;
  }

  if (auto CI = dyn_cast<CallInst>(user)) {

    auto Mode = gutils->mode;
    const bool cacheMode = (Mode != DerivativeMode::ForwardMode);

    auto funcName = getFuncNameFromCall(CI);
    auto blasMetaData = extractBLAS(funcName);
#if LLVM_VERSION_MAJOR >= 16
    if (blasMetaData.has_value())
#else
    if (blasMetaData.hasValue())
#endif
    {
      const std::vector<bool> *overwritten_args_ptr = nullptr;
      if (gutils->overwritten_args_map_ptr) {
        auto found =
            gutils->overwritten_args_map_ptr->find(const_cast<CallInst *>(CI));
        assert(found != gutils->overwritten_args_map_ptr->end());
        overwritten_args_ptr = &found->second;
      }
#if LLVM_VERSION_MAJOR >= 16
      BlasInfo blas = blasMetaData.value();
#else
      BlasInfo blas = blasMetaData.getValue();
#endif
#include "BlasDiffUse.inc"
    }

    // Only need primal (and shadow) request for reverse
    if (funcName == "MPI_Isend" || funcName == "MPI_Irecv" ||
        funcName == "PMPI_Isend" || funcName == "PMPI_Irecv") {
      if (val != CI->getArgOperand(6)) {
        return false;
      }
    }

    // Only need the primal request.
    if (funcName == "MPI_Wait" || funcName == "PMPI_Wait")
      if (val != CI->getArgOperand(0))
        return false;

    // Only need element count for reverse of waitall
    if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall")
      if (val != CI->getArgOperand(0) || val != CI->getOperand(1))
        return false;
    // Since adjoint of barrier is another barrier in reverse
    // we still need even if instruction is inactive
    if (funcName == "__kmpc_barrier" || funcName == "MPI_Barrier") {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need direct primal of " << *val
                     << " in reverse from barrier " << *user << "\n";
      return true;
    }

    // Since adjoint of GC preserve is another preserve in reverse
    // we still need even if instruction is inactive
    if (funcName == "llvm.julia.gc_preserve_begin") {
      if (EnzymePrintDiffUse)
        llvm::errs() << " Need direct primal of " << *val
                     << " in reverse from gc " << *user << "\n";
      return true;
    }

    bool writeOnlyNoCapture = true;

    if (shouldDisableNoWrite(CI)) {
      writeOnlyNoCapture = false;
    }
#if LLVM_VERSION_MAJOR >= 14
    for (size_t i = 0; i < CI->arg_size(); i++)
#else
    for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
    {
      if (val == CI->getArgOperand(i)) {
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

    // Don't need the primal argument if it is write only and not captured
    if (writeOnlyNoCapture)
      return false;
  }

  bool neededFB = !gutils->isConstantInstruction(user) ||
                  !gutils->isConstantValue(const_cast<Instruction *>(user));
  if (neededFB) {
    if (EnzymePrintDiffUse)
      llvm::errs() << " Need direct primal of " << *val
                   << " in reverse from fallback " << *user << "\n";
  }
  return neededFB;
}

void DifferentialUseAnalysis::dump(Graph &G) {
  for (auto &pair : G) {
    llvm::errs() << "[" << *pair.first.V << ", " << (int)pair.first.outgoing
                 << "]\n";
    for (auto N : pair.second) {
      llvm::errs() << "\t[" << *N.V << ", " << (int)N.outgoing << "]\n";
    }
  }
}

/* Returns true if there is a path from source 's' to sink 't' in
  residual graph. Also fills parent[] to store the path */
void DifferentialUseAnalysis::bfs(const Graph &G,
                                  const SmallPtrSetImpl<Value *> &Recompute,
                                  std::map<Node, Node> &parent) {
  std::deque<Node> q;
  for (auto V : Recompute) {
    Node N(V, false);
    parent.emplace(N, Node(nullptr, true));
    q.push_back(N);
  }

  // Standard BFS Loop
  while (!q.empty()) {
    auto u = q.front();
    q.pop_front();
    auto found = G.find(u);
    if (found == G.end())
      continue;
    for (auto v : found->second) {
      if (parent.find(v) == parent.end()) {
        q.push_back(v);
        parent.emplace(v, u);
      }
    }
  }
}

// Return 1 if next is better
// 0 if equal
// -1 if prev is better, or unknown
int DifferentialUseAnalysis::cmpLoopNest(Loop *prev, Loop *next) {
  if (next == prev)
    return 0;
  if (next == nullptr)
    return 1;
  else if (prev == nullptr)
    return -1;
  for (Loop *L = prev; L != nullptr; L = L->getParentLoop()) {
    if (L == next)
      return 1;
  }
  return -1;
}

void DifferentialUseAnalysis::minCut(
    const DataLayout &DL, LoopInfo &OrigLI,
    const SmallPtrSetImpl<Value *> &Recomputes,
    const SmallPtrSetImpl<Value *> &Intermediates,
    SmallPtrSetImpl<Value *> &Required, SmallPtrSetImpl<Value *> &MinReq,
    const ValueMap<Value *, GradientUtils::Rematerializer>
        &rematerializableAllocations,
    llvm::TargetLibraryInfo &TLI) {
  Graph G;
  for (auto V : Intermediates) {
    G[Node(V, false)].insert(Node(V, true));
    for (auto U : V->users()) {
      if (auto I = dyn_cast<Instruction>(U)) {
        for (auto pair : rematerializableAllocations) {
          if (Intermediates.count(pair.first) && pair.second.stores.count(I)) {
            if (V != pair.first)
              G[Node(V, true)].insert(Node(pair.first, false));
          }
        }
      }
      if (Intermediates.count(U)) {
        if (V != U)
          G[Node(V, true)].insert(Node(U, false));
      }
    }
  }
  for (auto pair : rematerializableAllocations) {
    if (Intermediates.count(pair.first)) {
      for (LoadInst *L : pair.second.loads) {
        if (Intermediates.count(L)) {
          if (L != pair.first)
            G[Node(pair.first, true)].insert(Node(L, false));
        }
      }
      for (auto L : pair.second.loadLikeCalls) {
        if (Intermediates.count(L.loadCall)) {
          if (L.loadCall != pair.first)
            G[Node(pair.first, true)].insert(Node(L.loadCall, false));
        }
      }
    }
  }
  for (auto R : Required) {
    assert(Intermediates.count(R));
  }
  for (auto R : Recomputes) {
    assert(Intermediates.count(R));
  }

  Graph Orig = G;

  // Augment the flow while there is a path from source to sink
  while (1) {
    std::map<Node, Node> parent;
    bfs(G, Recomputes, parent);
    Node end(nullptr, false);
    for (auto req : Required) {
      if (parent.find(Node(req, true)) != parent.end()) {
        end = Node(req, true);
        break;
      }
    }
    if (end.V == nullptr)
      break;
    // update residual capacities of the edges and reverse edges
    // along the path
    Node v = end;
    while (1) {
      assert(parent.find(v) != parent.end());
      Node u = parent.find(v)->second;
      assert(u.V != nullptr);
      assert(G[u].count(v) == 1);
      assert(G[v].count(u) == 0);
      G[u].erase(v);
      G[v].insert(u);
      if (Recomputes.count(u.V) && u.outgoing == false)
        break;
      v = u;
    }
  }

  // Flow is maximum now, find vertices reachable from s

  std::map<Node, Node> parent;
  bfs(G, Recomputes, parent);

  // Print all edges that are from a reachable vertex to
  // non-reachable vertex in the original graph
  for (auto &pair : Orig) {
    if (parent.find(pair.first) != parent.end())
      for (auto N : pair.second) {
        if (parent.find(N) == parent.end()) {
          assert(pair.first.outgoing == 0 && N.outgoing == 1);
          assert(pair.first.V == N.V);
          MinReq.insert(N.V);
        }
      }
  }

  // When ambiguous, push to cache the last value in a computation chain
  // This should be considered in a cost for the max flow
  std::deque<Value *> todo(MinReq.begin(), MinReq.end());
  while (todo.size()) {
    auto V = todo.front();
    todo.pop_front();
    auto found = Orig.find(Node(V, true));
    if (found->second.size() == 1 && !Required.count(V)) {
      bool potentiallyRecursive =
          isa<PHINode>((*found->second.begin()).V) &&
          OrigLI.isLoopHeader(
              cast<PHINode>((*found->second.begin()).V)->getParent());
      int moreOuterLoop = cmpLoopNest(
          OrigLI.getLoopFor(cast<Instruction>(V)->getParent()),
          OrigLI.getLoopFor(
              cast<Instruction>(((*found->second.begin()).V))->getParent()));
      if (potentiallyRecursive)
        continue;
      if (moreOuterLoop == -1)
        continue;
      if (auto ASC = dyn_cast<AddrSpaceCastInst>((*found->second.begin()).V)) {
        if (ASC->getDestAddressSpace() == 11 ||
            ASC->getDestAddressSpace() == 13)
          continue;
        if (ASC->getSrcAddressSpace() == 10 && ASC->getDestAddressSpace() == 0)
          continue;
      }
      // If an allocation call, we cannot cache any "capturing" users
      if (isAllocationCall(V, TLI)) {
        auto next = (*found->second.begin()).V;
        bool noncapture = false;
        if (isa<LoadInst>(next)) {
          noncapture = true;
        } else if (auto II = dyn_cast<IntrinsicInst>(next)) {
          if (II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_i ||
              II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_p ||
              II->getIntrinsicID() == Intrinsic::nvvm_ldu_global_f ||
              II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_i ||
              II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_p ||
              II->getIntrinsicID() == Intrinsic::nvvm_ldg_global_f ||
              II->getIntrinsicID() == Intrinsic::masked_load)
            noncapture = true;
        } else if (auto CI = dyn_cast<CallInst>(next)) {
          bool captures = false;
#if LLVM_VERSION_MAJOR >= 14
          for (size_t i = 0; i < CI->arg_size(); i++)
#else
          for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
          {
            if (CI->getArgOperand(i) == V && !isNoCapture(CI, i)) {
              captures = true;
              break;
            }
          }
          noncapture = !captures;
        }

        if (!noncapture)
          continue;
      }

      if (moreOuterLoop == 1 ||
          (moreOuterLoop == 0 &&
           DL.getTypeSizeInBits(V->getType()) >=
               DL.getTypeSizeInBits((*found->second.begin()).V->getType()))) {
        MinReq.erase(V);
        MinReq.insert((*found->second.begin()).V);
        todo.push_back((*found->second.begin()).V);
      }
    }
  }
  return;
}
