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
#include <deque>
#include <map>
#include <set>

#include "GradientUtils.h"

enum class ValueType { Primal, ShadowPtr };

typedef std::tuple<const Value *, bool, ValueType> UsageKey;

// Determine if a value is needed directly to compute the adjoint
// of the given instruction user
static inline bool is_use_directly_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *val,
    const Instruction *user,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  if (auto ainst = dyn_cast<Instruction>(val)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }
  assert(user->getParent()->getParent() == gutils->oldFunc);

  if (oldUnreachable.count(user->getParent()))
    return false;

  if (isa<LoadInst>(user) || isa<CastInst>(user) || isa<PHINode>(user) ||
      isa<GetElementPtrInst>(user)) {
    return false;
  }

  // We don't need any of the input operands to compute the adjoint of a store
  // instance
  if (isa<StoreInst>(user)) {
    return false;
  }

  // If memtransfer, only the size may need preservation for the reverse pass
  if (auto MTI = dyn_cast<MemTransferInst>(user)) {
    if (MTI->getArgOperand(2) != val)
      return false;
  }

  if (isa<CmpInst>(user) || isa<BranchInst>(user) || isa<ReturnInst>(user) ||
      isa<FPExtInst>(user) || isa<FPTruncInst>(user) ||
      (isa<InsertElementInst>(user) &&
       cast<InsertElementInst>(user)->getOperand(2) != val) ||
      (isa<ExtractElementInst>(user) &&
       cast<ExtractElementInst>(user)->getIndexOperand() != val)
      // isa<ExtractElement>(use) ||
      // isa<InsertElementInst>(use) || isa<ShuffleVectorInst>(use) ||
      // isa<ExtractValueInst>(use) || isa<AllocaInst>(use)
      /*|| isa<StoreInst>(use)*/) {
    return false;
  }

  if (auto II = dyn_cast<IntrinsicInst>(user)) {
    if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
        II->getIntrinsicID() == Intrinsic::lifetime_end ||
        II->getIntrinsicID() == Intrinsic::stacksave ||
        II->getIntrinsicID() == Intrinsic::stackrestore) {
      return false;
    }
    if (II->getIntrinsicID() == Intrinsic::fma) {
      bool needed = false;
      if (II->getArgOperand(0) == val &&
          !gutils->isConstantValue(II->getArgOperand(1)))
        needed = true;
      if (II->getArgOperand(1) == val &&
          !gutils->isConstantValue(II->getArgOperand(0)))
        needed = true;
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
      return needed;
    }
  }

  if (auto si = dyn_cast<SelectInst>(user)) {
    // Only would potentially need the condition
    if (si->getCondition() != val) {
      return false;
    }

    // only need the condition if select is active
    return !gutils->isConstantValue(const_cast<SelectInst *>(si));
  }

  return !gutils->isConstantInstruction(user) ||
         !gutils->isConstantValue(const_cast<Instruction *>(user));
}

template <ValueType VT, bool OneLevel = false>
static inline bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, std::map<UsageKey, bool> &seen,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  auto idx = UsageKey(inst, topLevel, VT);
  if (seen.find(idx) != seen.end())
    return seen[idx];
  if (auto ainst = dyn_cast<Instruction>(inst)) {
    assert(ainst->getParent()->getParent() == gutils->oldFunc);
  }

  // Inductively claim we aren't needed (and try to find contradiction)
  seen[idx] = false;

  if (VT != ValueType::ShadowPtr) {
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (op->getOpcode() == Instruction::FDiv) {
        if (!gutils->isConstantValue(const_cast<Value *>(inst)) &&
            !gutils->isConstantValue(op->getOperand(1))) {
          return seen[idx] = true;
        }
      }
    }
  }

  // Consider all users of this value, do any of them need this in the reverse?
  for (auto use : inst->users()) {
    if (use == inst)
      continue;

    const Instruction *user = dyn_cast<Instruction>(use);

    // A shadow value is only needed in reverse if it or one of its descendants
    // is used in an active instruction
    if (VT == ValueType::ShadowPtr) {
      if (!user)
        return seen[idx] = true;

      if (auto SI = dyn_cast<StoreInst>(user)) {
        // storing an active pointer into a location
        // doesn't require the shadow pointer for the
        // reverse pass
        if (SI->getPointerOperand() != inst)
          continue;

        if (!gutils->isConstantValue(
                const_cast<Value *>(SI->getPointerOperand())))
          return seen[idx] = true;
        else
          continue;
      }

      if (auto MTI = dyn_cast<MemTransferInst>(user)) {
        if (MTI->getArgOperand(0) != inst && MTI->getArgOperand(1) != inst)
          continue;

        if (!gutils->isConstantValue(
                const_cast<Value *>(MTI->getArgOperand(0))))
          return seen[idx] = true;
        else
          continue;
      }

      if (isa<ReturnInst>(user)) {
        if (gutils->ATA->ActiveReturns)
          return seen[idx] = true;
        else
          continue;
      }

      if (!gutils->isConstantInstruction(const_cast<Instruction *>(user)))
        return seen[idx] = true;

      if (user->getType()->isVoidTy())
        continue;

      if (!TR.query(const_cast<Instruction *>(user))
               .Inner0()
               .isPossiblePointer())
        continue;

      if (!OneLevel && is_value_needed_in_reverse<ValueType::ShadowPtr>(
                           TR, gutils, user, topLevel, seen, oldUnreachable)) {
        return seen[idx] = true;
      }
      continue;
    }

    assert(VT == ValueType::Primal);

    // If a sub user needs, we need
    if (!OneLevel && is_value_needed_in_reverse<VT>(TR, gutils, user, topLevel,
                                                    seen, oldUnreachable)) {
      return seen[idx] = true;
    }

    // One may need to this value in the computation of loop
    // bounds/comparisons/etc (which even though not active -- will be used for
    // the reverse pass)
    //   We only need this if we're not doing the combined forward/reverse since
    //   otherwise it will use the local cache (rather than save for a separate
    //   backwards cache)
    //   We also don't need this if looking at the shadow rather than primal
    if (!topLevel) {
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
        return seen[idx] = true;
      }

      if (auto CI = dyn_cast<CallInst>(use)) {
        if (auto F = CI->getCalledFunction()) {
          if (F->getName() == "__kmpc_for_static_init_4" ||
              F->getName() == "__kmpc_for_static_init_4u" ||
              F->getName() == "__kmpc_for_static_init_8" ||
              F->getName() == "__kmpc_for_static_init_8u") {
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
    if (auto GEP = dyn_cast<GetElementPtrInst>(user)) {
      bool idxUsed = false;
      for (auto &idx : GEP->indices()) {
        if (idx.get() == inst)
          idxUsed = true;
      }
      if (!idxUsed)
        primalUsedInShadowPointer = false;
    }

    if (primalUsedInShadowPointer)
      if (!user->getType()->isVoidTy() &&
          TR.query(const_cast<Instruction *>(user))
              .Inner0()
              .isPossiblePointer()) {
        if (!OneLevel &&
            is_value_needed_in_reverse<ValueType::ShadowPtr>(
                TR, gutils, user, topLevel, seen, oldUnreachable)) {
          return seen[idx] = true;
        }
      }

    bool direct = is_use_directly_needed_in_reverse(TR, gutils, inst, user,
                                                    oldUnreachable);
    if (!direct)
      continue;

    return seen[idx] = true;
  }
  return false;
}

template <ValueType VT>
static inline bool is_value_needed_in_reverse(
    TypeResults &TR, const GradientUtils *gutils, const Value *inst,
    bool topLevel, const SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  std::map<UsageKey, bool> seen;
  return is_value_needed_in_reverse<VT>(TR, gutils, inst, topLevel, seen,
                                        oldUnreachable);
}

struct Node {
  Value *V;
  bool outgoing;
  Node(Value *V, bool outgoing) : V(V), outgoing(outgoing){};
  bool operator<(const Node N) const {
    if (V < N.V)
      return true;
    return !(N.V < V) && outgoing < N.outgoing;
  }
  void dump() { llvm::errs() << "[" << *V << ", " << (int)outgoing << "]\n"; }
};

typedef std::map<Node, std::set<Node>> Graph;

static inline void dump(Graph &G) {
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
static inline void bfs(const Graph &G,
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

static inline void minCut(const SmallPtrSetImpl<Value *> &Recomputes,
                          const SmallPtrSetImpl<Value *> &Intermediates,
                          SmallPtrSetImpl<Value *> &Required,
                          SmallPtrSetImpl<Value *> &MinReq) {
  Graph G;
  for (auto V : Intermediates) {
    G[Node(V, false)].insert(Node(V, true));
    for (auto U : V->users()) {
      if (Intermediates.count(U)) {
        G[Node(V, true)].insert(Node(U, false));
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
      G[u].erase(v);
      assert(G[v].count(u) == 0);
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
    if (found->second.size() == 1 &&
        !isa<PHINode>((*found->second.begin()).V)) {
      MinReq.erase(V);
      MinReq.insert((*found->second.begin()).V);
      todo.push_back((*found->second.begin()).V);
    }
  }
  return;
}