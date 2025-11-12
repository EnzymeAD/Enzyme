//===- RemovalUtils.cpp - Utilities to remove Enzyme ops -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RemovalUtils.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Utils.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/PatternMatch.h"
#include <cassert>
#include <deque>

#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::enzyme;

#define DEBUG_TYPE "enzyme-mincut"

void mlir::enzyme::localizeGradients(OpBuilder &builder,
                                     MGradientUtilsReverse *gutils,
                                     Block *fwd) {
  Operation *parent = fwd->getParentOp();

  auto localizeGradientValue = [&](Value val) {
    if (gutils->isConstantValue(val))
      return;
    auto iface = dyn_cast<AutoDiffTypeInterface>(val.getType());
    if (iface && !iface.isMutable()) {
      auto grad = gutils->getDifferential(val);

      enzyme::SetOp initialSet = nullptr;
      for (auto user : grad.getUsers()) {
        if (!parent->isProperAncestor(user)) {
          assert(!initialSet);
          initialSet = dyn_cast<enzyme::SetOp>(user);
          assert(initialSet);
        }
      }

      auto initOp = grad.getDefiningOp<enzyme::InitOp>();

      {
        OpBuilder::InsertionGuard g(builder);
        Value zero =
            iface.createNullValue(builder, initialSet.getValue().getLoc());
        builder.setInsertionPointAfter(zero.getDefiningOp());
        enzyme::SetOp::create(builder, initialSet.getLoc(), grad, zero);
        initialSet->erase();
      }

      builder.setInsertionPointToStart(builder.getBlock());
      initOp->remove();
      builder.insert(initOp);
    }
  };

  for (auto operand : fwd->getArguments()) {
    localizeGradientValue(operand);
  }

  for (auto &it : fwd->getOperations()) {
    for (auto res : it.getResults()) {
      localizeGradientValue(res);
    }
  }
}

void mlir::enzyme::removalBlockExplore(
    Block *block, IRMapping &mapping, PatternRewriter &rewriter,
    llvm::SetVector<Value> &gradients,
    llvm::MapVector<Value, CacheInfo> &caches) {
  for (auto it = block->begin(), e = block->end(); it != e;) {
    Operation *op = &*it;

    if (auto setOp = dyn_cast<enzyme::SetOp>(op)) {
      auto grad = setOp.getGradient();
      auto value = setOp.getValue();
      mapping.map(grad, value);
      gradients.insert(grad);
    }

    if (auto getOp = dyn_cast<enzyme::GetOp>(op)) {
      auto grad = getOp.getGradient();
      Value value = mapping.lookupOrNull(getOp.getGradient());
      if (!value) {
        value = enzyme::GetOp::create(rewriter, getOp->getLoc(),
                                      getOp.getResult().getType(), grad);
        mapping.map(grad, value);
      }
      rewriter.replaceAllUsesWith(getOp.getResult(), value);
    }

    if (auto pushOp = dyn_cast<enzyme::PushOp>(op)) {
      CacheInfo info(pushOp.getCache());

      Value pushedValue = info.pushedValue();

      // Then we can push the value before the if, if it is defined before the
      // if
      if (pushedValue.getParentBlock() != block) {
        enzyme::PushOp::create(rewriter, pushOp->getLoc(), pushOp.getCache(),
                               pushedValue);

        ++it; // Increment iterator to allow in place deletion
        rewriter.eraseOp(pushOp);

        // Move the pop before the other if
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(info.popOp->getParentOp());

        auto newPop =
            enzyme::PopOp::create(rewriter, info.popOp->getLoc(),
                                  pushedValue.getType(), info.popOp.getCache());
        rewriter.replaceAllUsesWith(info.popOp.getResult(), newPop);
        rewriter.eraseOp(info.popOp);

        continue;
      }

      if (caches.contains(pushedValue)) {
        info = info.merge(caches.lookup(pushedValue), rewriter);
      }
      caches[pushedValue] = info;
    }

    ++it;
  }
}

typedef llvm::PointerUnion<Operation *, Value> Node;

void dump(const Node &n) {
  if (isa<Value>(n))
    llvm::errs() << "[" << cast<Value>(n) << ", "
                 << "Value"
                 << "]\n";
  else if (isa<Operation *>(n))
    llvm::errs() << "[" << *cast<Operation *>(n) << ", "
                 << "Operation"
                 << "]\n";
  else
    llvm::errs() << "["
                 << "NULL"
                 << ", "
                 << "None"
                 << "]\n";
}

struct Graph : public llvm::MapVector<Node, SmallPtrSet<Node, 2>> {
  const SmallPtrSet<Node, 2> &at(const Node &n) {
    auto found = find(n);
    assert(found != end());
    return found->second;
  }
};

static void dump(Graph &G) {
  for (auto &pair : G) {
    dump(pair.first);
    for (const auto &N : pair.second) {
      llvm::errs() << "\t";
      dump(N);
    }
  }
}

// A node in the compute graph.
// Operation nodes have outgoing edges to value nodes that they produce and
// incoming nodes from values they take as operands.

// parent is populated with a path from each connected leaf node of G to one
// of the Value in Source.
static inline void bfs(const Graph &G, const llvm::SetVector<Value> &Sources,
                       DenseMap<Node, Node> &parent) {
  std::deque<Node> q;
  for (const auto &V : Sources) {
    Node N(V);
    parent.try_emplace(N, Node());
    q.push_back(N);
  }

  // Standard BFS Loop

  SmallPtrSet<Node, 2> done;

  while (!q.empty()) {
    auto u = q.front();
    q.pop_front();
    auto found = G.find(u);
    if (found == G.end())
      continue;

    if (!done.insert(u).second)
      continue;

    for (const auto &v : found->second) {
      if (parent.try_emplace(v, u).second) {
        q.push_back(v);
      }
    }
  }
}

// Whether or not an operation can be moved from the forward region to the
// reverse region or vice-versa.
static inline bool isMovable(Operation *op) {
  return op->getNumRegions() == 0 && op->getBlock()->getTerminator() != op &&
         mlir::isPure(op);
}

// Given a graph `G`, construct a new graph `G2`, where all paths must terminate
// in a node in the set `Required` and start at `Root`.
template <typename T>
static Graph filterGraph(const Graph &Orig, const SetVector<Value> &Roots,
                         const SetVector<T> &Required) {
  Graph inverted;

  // Compute the graph with inverted edges by a floodfill, stopping at the first
  // `required`. This is required in the case of a root -> required -> required
  // edge. We do not want to contain the required->required subgraph.
  if (false) {
    std::deque<Node> worklist;
    for (auto val : Roots) {
      worklist.push_back(val);
    }

    SmallPtrSet<Node, 2> done;
    for (auto src : Required) {
      done.insert(src);
    }

    while (!worklist.empty()) {
      Node N = worklist.front();
      worklist.pop_front();

      if (!done.insert(N).second)
        continue;

      auto pair = Orig.find(N);
      if (pair == Orig.end()) {
        continue;
      }

      for (const auto &NN : pair->second) {

        inverted[NN].insert(N);
        if (!done.contains(NN)) {
          worklist.push_back(NN);
        }
      }
    }

  } else {
    for (auto &pair : Orig) {
      for (auto N : pair.second) {
        inverted[N].insert(pair.first);
      }
    }
  }

  std::deque<Node> worklist;
  for (auto snk : Required) {
    worklist.emplace_back(snk);
  }

  SmallPtrSet<Node, 2> done;
  for (auto src : Roots) {
    done.insert(src);
  }

  Graph G;

  while (!worklist.empty()) {
    Node N = worklist.front();
    worklist.pop_front();

    if (!done.insert(N).second)
      continue;

    auto pair = inverted.find(N);
    if (pair == inverted.end()) {
      continue;
    }

    for (const auto &NN : pair->second) {

      G[NN].insert(N);
      if (!done.contains(NN)) {
        worklist.push_back(NN);
      }
    }
  }

  return G;
}

static int64_t computeSizeOfType(Value val) {
  auto T = dyn_cast<AutoDiffTypeInterface>(val.getType());
  return T ? T.getApproxSize() : INT64_MAX;
};

static int64_t computeRankOfType(Value val) {
  auto TT = dyn_cast<RankedTensorType>(val.getType());
  return TT ? TT.getRank() : 0;
}

/// Find a common IsolatedFromAbove ancestor of the given ops. If at least one
/// op is a top-level module op (which is expected to be isolated from above),
/// return that op.
static Operation *findCommonAncestor(ArrayRef<Operation *> ops) {
  // Check if there is a top-level operation within `ops`. If so, return that
  // op.
  for (Operation *op : ops) {
    if (!op->getParentOp()) {
#ifndef NDEBUG
      assert(op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
             "expected top-level op to be isolated from above");
      for (Operation *other : ops)
        assert(op->isAncestor(other) &&
               "expected ops to have a common ancestor");
#endif // NDEBUG
      return op;
    }
  }

  // No top-level op. Find a common ancestor.
  Operation *commonAncestor =
      ops.front()->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
  for (Operation *op : ops.drop_front()) {
    while (!commonAncestor->isProperAncestor(op)) {
      commonAncestor =
          commonAncestor->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
      assert(commonAncestor &&
             "expected to find a common isolated from above ancestor");
    }
  }

  return commonAncestor;
}

// Given the full forward/backward compute graph, the push/pop can be seen
// as a special cut of this graph. This function tries to modifies the
// boundary of the push/pop to minimize the amount of memory that is live
// across different loops.
// The insertion point of rewriter must be in the reverse block, after any
// fwdrevmap settings have been created.
void mlir::enzyme::minCutCache(Block *forward, Block *reverse,
                               SmallVector<CacheInfo> &caches0,
                               PatternRewriter &rewriter,
                               const IRMapping &fwdrevmap, Operation *lastFwd) {
  assert(rewriter.getInsertionBlock() == reverse);
  assert(rewriter.getInsertionPoint()->getBlock() == reverse);
  if (caches0.empty())
    return;

  // where to build the new inits
  Operation *entry = caches0[0].initOp;

  IRMapping mapping = fwdrevmap;
  SmallVector<CacheInfo> caches;
  for (auto &info : caches0) {
    auto todo = info.pushedValue();
    if (todo.getParentBlock() != forward) {
      rewriter.modifyOpInPlace(info.pushOp, [&]() {
        if (&*rewriter.getInsertionPoint() == info.pushOp)
          rewriter.setInsertionPoint(info.pushOp->getNextNode());

        info.pushOp->moveBefore(forward->getParentOp());
      });
      rewriter.modifyOpInPlace(info.popOp, [&]() {
        if (&*rewriter.getInsertionPoint() == info.popOp)
          rewriter.setInsertionPoint(info.popOp->getNextNode());
        info.popOp->moveBefore(reverse->getParentOp());
      });
      mapping.map(info.pushedValue(), info.popOp);
      continue;
    }
    caches.push_back(info);
  }
  assert(rewriter.getInsertionPoint()->getBlock() == reverse);

  if (caches.empty()) {
    caches0.clear();
    return;
  }

  Graph G;

  LLVM_DEBUG(llvm::dbgs() << "trying min/cut\n");
  LLVM_DEBUG(
      findCommonAncestor({forward->getParentOp(), reverse->getParentOp()})
          ->dump());

  LLVM_DEBUG(llvm::dbgs() << "forward: " << *forward << "\n";);
  LLVM_DEBUG(llvm::dbgs() << "reverse: " << *reverse << "\n";);

  SmallVector<Value> worklist;
  for (auto &cache : caches) {
    worklist.push_back(cache.pushedValue());
  }

  // nodes that cannot be recomputed
  SetVector<Value> roots;

  // Walk Backward
  //
  // Roots (sources) are either block arguments or values which are defined
  // outside of forward.
  while (!worklist.empty()) {
    Value todo = worklist.pop_back_val();

    if (todo.getParentBlock() != forward || fwdrevmap.contains(todo)) {
      continue;
    }

    Operation *owner = todo.getDefiningOp();
    if (!owner || !isMovable(owner)) {
      roots.insert(todo);
      continue;
    }

    auto &&[_, inserted] = G[Node(owner)].insert(Node(todo));
    if (inserted) {
      for (Value operand : owner->getOperands()) {
        G[Node(operand)].insert(Node(owner));
        worklist.push_back(operand);
      }
    }
  }

  worklist.clear();

  // The operation whose use of a value forces a value to be available
  // in the reverse pass
  SetVector<Operation *> Required;

  {
    for (auto &info : caches) {
      Value poped = info.popOp.getResult();

      bool isRequired = false;
      for (auto user : poped.getUsers()) {
        if (user->getBlock() != reverse || !isMovable(user)) {
          G[info.pushedValue()].insert(Node(user));
          Required.insert(user);
          isRequired = true;
          break;
        }
      }
      if (!isRequired)
        for (auto user : poped.getUsers()) {
          G[Node(info.pushedValue())].insert(user);
          for (Value res : user->getResults()) {
            G[Node(user)].insert(res);
            worklist.push_back(res);
          }
        }
    }

    // Walk Forward
    while (!worklist.empty()) {
      Value todo = worklist.pop_back_val();

      bool isRequired = false;
      for (auto user : todo.getUsers()) {
        if (user->getBlock() != reverse || !isMovable(user)) {
          G[todo].insert(Node(user));
          Required.insert(user);
          isRequired = true;
          break;
        }
      }
      if (isRequired)
        continue;

      for (auto user : todo.getUsers()) {
        Node N(user);
        auto &&[_, inserted] = G[Node(todo)].insert(N);
        if (inserted) {
          for (Value res : user->getResults()) {
            G[N].insert(Node(res));
            worklist.push_back(res);
          }
        }
      }
    }

    for (auto N : G) {
      if (!isa<Operation *>(N.first))
        continue;
      auto op = cast<Operation *>(N.first);
      if (op->getBlock() != reverse)
        continue;
      for (auto v : op->getOperands()) {
        if (v.getParentBlock() != reverse) {
          continue;
        }
        if (G.contains(Node(v))) {
          continue;
        }
        Required.insert(op);
        break;
      }
    }
    assert(rewriter.getInsertionPoint()->getBlock() == reverse);

    LLVM_DEBUG(llvm::dbgs() << "Required: \n";);
    LLVM_DEBUG(for (auto R : Required) llvm::dbgs() << " + " << *R << "\n";);

    LLVM_DEBUG(llvm::dbgs() << "Roots: \n";);
    LLVM_DEBUG(for (auto R : roots) llvm::dbgs() << " + " << R << "\n";);
  }

  LLVM_DEBUG(llvm::dbgs() << "pre filter graph: \n";);
  LLVM_DEBUG(dump(G));
  G = filterGraph(G, roots, Required);
  LLVM_DEBUG(llvm::dbgs() << "post filter graph: \n";);
  LLVM_DEBUG(dump(G));

  Graph Orig = G;

  // Augment the flow while there is a path from source to sink
  while (1) {
    DenseMap<Node, Node> parent;
    bfs(G, roots, parent);
    Node end;
    for (auto req : Required) {
      if (parent.find(Node(req)) != parent.end()) {
        end = Node(req);
        break;
      }
    }
    if (end.isNull())
      break;
    // update residual capacities of the edges and reverse edges
    // along the path
    Node v = end;
    while (1) {
      assert(parent.find(v) != parent.end());
      Node u = parent.find(v)->second;
      assert(!u.isNull());
      assert(G[u].count(v) == 1);
      assert(G[v].count(u) == 0);
      G[u].erase(v);
      G[v].insert(u);
      if (isa<Value>(u) && roots.contains(cast<Value>(u)))
        break;
      v = u;
    }
  }
  assert(rewriter.getInsertionPoint()->getBlock() == reverse);
  // Flow is maximum now, find vertices reachable from s

  DenseMap<Node, Node> parent;
  bfs(G, roots, parent);

  LLVM_DEBUG(llvm::dbgs() << "residual graph: \n";);
  LLVM_DEBUG(dump(G));

  // Those are the new values to cache
  SetVector<Value> newCaches;

  // All edges that are from a reachable vertex to non-reachable vertex in
  // the original graph are edges for the minimum cut. The set of values to
  // cache are the values transported along those edges (either. Value ->
  // Operation or Operation -> Value).
  //
  // Note: we could use more heuristics here to select the actual cached
  // value
  //       based on sizes, existing caches, number of users in the fwd as to
  //       not duplicate work, etc...
  for (auto &pair : Orig) {
    if (parent.find(pair.first) != parent.end()) {
      for (auto N : pair.second) {
        if (parent.find(N) == parent.end()) {
          Value newCache;
          if (isa<Value>(pair.first)) {
            assert(isa<Operation *>(N));
            newCache = cast<Value>(pair.first);
          } else {
            assert(isa<Operation *>(pair.first));
            assert(isa<Value>(N));
            newCache = cast<Value>(N);
          }
          newCaches.insert(newCache);
        }
      }
    }
  }

  // compute path from new caches to required
  parent.clear();
  bfs(Orig, newCaches, parent);

  LLVM_DEBUG({
    llvm::dbgs() << "initial new caches: \n";
    for (Value v : newCaches) {
      v.dump();
    }
  });

  // The cachegraph is a sub graph of Orig with only pathes new caches
  // to Required nodes.
  Graph cacheGraph = filterGraph(Orig, newCaches, Required);

  LLVM_DEBUG(llvm::dbgs() << "cacheGraph:\n");
  LLVM_DEBUG(dump(cacheGraph));

  SmallVector<CacheInfo> newCacheInfos;

  // We guard here so then the IP after this is immediately before the new pop's
  Operation *firstClone = nullptr;

  // Refine cached values based on some heuristics
  if (newCaches.size()) {

    // sort caches to provide determinism.
    // llvm::sort(newCaches.getArrayRef().begin(),
    // newCaches.getArrayRef().end(), mlir::enzyme::valueCmp);

    SmallVector<Value> todo(newCaches.begin(), newCaches.end());
    while (todo.size()) {
      auto cur = todo.pop_back_val();

      auto &next = cacheGraph.at(Node(cur));

      if (next.size() > 1)
        continue;

      auto nextF = *next.begin();
      assert(isa<Operation *>(nextF));
      auto opNext = cast<Operation *>(nextF);

      if (Required.count(opNext))
        continue;

      if (opNext->getNumResults() != 1)
        continue;

      Value candidate = opNext->getResult(0);

      int64_t curSize = computeSizeOfType(cur),
              curRank = computeRankOfType(cur);

      int64_t newSize = computeSizeOfType(candidate),
              newRank = computeRankOfType(candidate);

      if (newRank < curRank || (newRank == curRank && newSize < curSize)) {
        newCaches.remove(cur);
        newCaches.insert(candidate);
        todo.push_back(candidate);
        cacheGraph.erase(cur);
        cacheGraph.erase(opNext);
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "refined cacheGraph:\n");
    LLVM_DEBUG(dump(cacheGraph));
    LLVM_DEBUG({
      llvm::dbgs() << "refined new caches: \n";
      for (Value v : newCaches) {
        v.dump();
      }
    });

    SetVector<Value> reverseCaches;
    for (Value newCache : newCaches) {
      if (newCache.getParentBlock() != forward) {
        reverseCaches.insert(newCache);
        continue;
      }
      assert(rewriter.getInsertionBlock() == reverse);

      enzyme::InitOp initOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(entry);
        enzyme::InitOp::create(
            rewriter, newCache.getLoc(),
            enzyme::CacheType::get(newCache.getContext(), newCache.getType()));
      });

      enzyme::PushOp pushOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        if (lastFwd && isa<BlockArgument>(newCache)) {
          rewriter.setInsertionPointAfter(lastFwd);
        } else {
          rewriter.setInsertionPointAfterValue(newCache);
        }
        enzyme::PushOp::create(rewriter, newCache.getLoc(), initOp.getResult(),
                               newCache);
      });

      assert(rewriter.getInsertionBlock() == reverse);
      assert(rewriter.getInsertionPoint()->getBlock() == reverse);
      enzyme::PopOp popOp = enzyme::PopOp::create(
          rewriter, newCache.getLoc(), newCache.getType(), initOp.getResult());
      if (!firstClone)
        firstClone = popOp;
      mapping.map(newCache, popOp.getResult());

      CacheInfo info;
      info.initOp = initOp;
      info.pushOp = pushOp;
      info.popOp = popOp;
      newCacheInfos.push_back(info);
    }

    if (reverseCaches.size()) {
      Graph fwdGraph = filterGraph(Orig, roots, newCaches);

      IRMapping fwdmap;
      for (auto &info : caches) {
        fwdmap.map(info.popOp->getResult(0), info.pushedValue());
      }

      SmallVector<Operation *> toErase;
      for (auto &op : llvm::make_early_inc_range(*reverse)) {
        if (!fwdGraph.contains(Node(&op)))
          continue;

        Operation *newO = ({
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPoint(forward->getTerminator());
          rewriter.clone(op, fwdmap);
        });

        bool hasUse = false;
        for (auto &&[res, newRes] :
             llvm::zip_equal(op.getResults(), newO->getResults())) {
          if (newCaches.contains(res)) {
            enzyme::InitOp initOp = ({
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(entry);
              enzyme::InitOp::create(rewriter, newRes.getLoc(),
                                     enzyme::CacheType::get(newRes.getContext(),
                                                            newRes.getType()));
            });

            enzyme::PushOp pushOp = ({
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(forward->getTerminator());
              enzyme::PushOp::create(rewriter, newRes.getLoc(),
                                     initOp.getResult(), newRes);
            });

            enzyme::PopOp popOp = ({
              OpBuilder::InsertionGuard guard(rewriter);
              rewriter.setInsertionPoint(&op);
              enzyme::PopOp::create(rewriter, newRes.getLoc(), newRes.getType(),
                                    initOp.getResult());
            });

            rewriter.replaceAllUsesWith(res, popOp->getResult(0));

            CacheInfo info;
            info.initOp = initOp;
            info.pushOp = pushOp;
            info.popOp = popOp;
            newCacheInfos.push_back(info);
          }
          if (!hasUse) {
            for (auto user : res.getUsers()) {
              if (!fwdGraph.contains(Node(user))) {
                hasUse = true;
                break;
              }
            }
          }
        }

        if (!hasUse && !op.hasAttr("enzyme.no_erase")) {
          toErase.push_back(&op);
        }
      }
      for (auto op : llvm::reverse(toErase)) {
        rewriter.eraseOp(op);
      }
    }
  }

  for (auto &op : llvm::make_early_inc_range(*forward)) {
    if (!cacheGraph.contains(Node(&op)))
      continue;
    bool hasUse = false;
    for (auto res : op.getResults()) {
      if (newCaches.contains(res)) {
        continue;
      }
      hasUse = true;
    }
    if (!hasUse)
      continue;
    for (auto v : op.getOperands()) {
      if (mapping.contains(v))
        continue;
      if (v.getParentBlock() == forward)
        continue;

      enzyme::InitOp initOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(entry);
        enzyme::InitOp::create(
            rewriter, v.getLoc(),
            enzyme::CacheType::get(v.getContext(), v.getType()));
      });

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(forward->getParentOp());
        enzyme::PushOp::create(rewriter, v.getLoc(), initOp.getResult(), v);
      };

      enzyme::PopOp popOp = ({
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(reverse->getParentOp());
        enzyme::PopOp::create(rewriter, v.getLoc(), v.getType(),
                              initOp.getResult());
      });
      mapping.map(v, popOp->getResult(0));
    }
    auto cop = rewriter.clone(op, mapping);
    if (!firstClone)
      firstClone = cop;
  }
  if (firstClone)
    rewriter.setInsertionPoint(firstClone);

  // Remove old caches
  for (auto &info : caches) {
    if (mapping.contains(info.pushedValue())) {
      rewriter.replaceOp(info.popOp, mapping.lookup(info.pushedValue()));
    } else {
      rewriter.eraseOp(info.popOp);
    }
    rewriter.eraseOp(info.pushOp);
    rewriter.eraseOp(info.initOp);
  }

  LLVM_DEBUG(llvm::dbgs() << "post min/cut\n");
  LLVM_DEBUG(
      findCommonAncestor({forward->getParentOp(), reverse->getParentOp()})
          ->dump());

  // Set new caches
  caches0 = std::move(newCacheInfos);
}

mlir::enzyme::CacheInfo
mlir::enzyme::CacheInfo::merge(mlir::enzyme::CacheInfo other,
                               mlir::PatternRewriter &rewriter) {
  assert(other.pushOp->getBlock() == pushOp->getBlock());
  assert(other.popOp->getBlock() == popOp->getBlock());

  enzyme::InitOp newInitOp;
  if (other.initOp->isBeforeInBlock(initOp)) {
    newInitOp = other.initOp;
    rewriter.replaceAllUsesWith(initOp.getResult(), newInitOp.getResult());
    rewriter.eraseOp(initOp);
  } else {
    newInitOp = initOp;
    rewriter.replaceAllUsesWith(other.initOp.getResult(),
                                newInitOp.getResult());
    rewriter.eraseOp(other.initOp);
  }

  rewriter.eraseOp(other.pushOp);

  enzyme::PopOp newPopOp;
  if (other.popOp->isBeforeInBlock(popOp)) {
    newPopOp = other.popOp;
    rewriter.replaceAllUsesWith(popOp.getResult(), newPopOp.getResult());
    rewriter.eraseOp(popOp);
  } else {
    newPopOp = popOp;
    rewriter.replaceAllUsesWith(other.popOp.getResult(), newPopOp.getResult());
    rewriter.eraseOp(other.popOp);
  }

  CacheInfo newInfo{newInitOp};
  return newInfo;
}
