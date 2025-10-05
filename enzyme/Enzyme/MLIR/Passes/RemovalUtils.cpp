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
#include "Utils.h"
#include "mlir/IR/PatternMatch.h"
#include <cassert>
#include <deque>

using namespace mlir;
using namespace mlir::enzyme;

#define DEBUG_TYPE "enzyme-mincut"

typedef llvm::PointerUnion<Operation *, Value> Node;

void dump(const Node &n) {
  if (n.is<Value>())
    llvm::errs() << "[" << n.get<Value>() << ", "
                 << "Value"
                 << "]\n";
  else if (n.is<Operation *>())
    llvm::errs() << "[" << *n.get<Operation *>() << ", "
                 << "Operation"
                 << "]\n";
  else
    llvm::errs() << "["
                 << "NULL"
                 << ", "
                 << "None"
                 << "]\n";
}

typedef DenseMap<Node, SmallPtrSet<Node, 2>> Graph;

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
  return op->getNumRegions() == 0 && mlir::isPure(op);
}

static Graph reverseGraph(const Graph &Orig, const SetVector<Value> &sources,
                          const SetVector<Value> &sinks) {
  Graph inverted, revGraph;

  // Compute the graph with inverted edges
  for (auto &pair : Orig) {
    for (auto N : pair.second) {
      inverted[N].insert(pair.first);
    }
  }

  std::deque<Node> worklist;
  for (auto snk : sinks) {
    worklist.emplace_back(snk);
  }

  SmallPtrSet<Node, 2> done;
  for (auto src : sources) {
    done.insert(src);
  }

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

      revGraph[NN].insert(N);
      if (!done.contains(NN)) {
        worklist.push_back(NN);
      }
    }
  }

  return revGraph;
}

// Given the full forward/backward compute graph, the push/pop can be seen
// as a special cut of this graph. This function tries to modifies the
// boundary of the push/pop to minimize the amount of memory that is live
// across different loops.
void mlir::enzyme::minCutCache(Block *forward, Block *reverse,
                               SmallVector<CacheInfo> &caches,
                               PatternRewriter &rewriter) {
  if (caches.empty())
    return;

  // where to build the new inits
  Operation *entry = caches[0].initOp;

  Graph G;

  LLVM_DEBUG(llvm::dbgs() << "trying min/cut\n");
  LLVM_DEBUG(forward->getParentOp()->getParentOp()->dump());

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

    if (todo.getParentBlock() != forward) {
      roots.insert(todo);
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

  for (auto &info : caches) {
    // insert use of the push through the pop. These define the existing
    // forward/reverse cut that the min cut is trying to improve.
    //
    // Given the following IR:
    //
    // %cache = "enzyme.init"() : () -> !enzyme.Cache<f32>
    // ^forward:
    //   %pushed = "operation.someop"(%somevalue) : (f32) -> f32
    //   "enzyme.push"(%cache, %pushed) : (!enzyme.Cache<f32>, f32) -> ()
    // ^backward:
    //   %poped = "enzyme.pop"(%cache) : (!enzyme.Cache<f32>) -> f32
    //   %use = "operation.use"(%poped) : (f32) -> f32
    //
    // will result in the following graph:
    //
    // [%somevalue, Value]
    //   [%pushed, Operation]
    //     [%pushed, Value]
    //       [%poped, Operation]
    //         [%poped, Value]
    //           [%use, Operation]
    //             [%use, Value]
    //
    Node popNode = Node(static_cast<Operation *>(info.popOp));
    Value poped = info.popOp.getResult();
    G[Node(info.pushedValue())].insert(popNode);
    G[popNode].insert(Node(poped));
    worklist.push_back(poped);
  }

  SetVector<Value> Required;

  // Walk Forward
  while (!worklist.empty()) {
    Value todo = worklist.pop_back_val();

    for (auto user : todo.getUsers()) {
      if (user->getBlock() != reverse && !isMovable(user)) {
        Required.insert(todo);
        continue;
      }

      if (!llvm::all_of(user->getOperands(), [&G, &todo](Value operand) {
            return operand == todo || G.count(Node(operand));
          })) {
        Required.insert(todo);
        continue;
      }

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

  if (G.empty())
    return;

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
      if (u.is<Value>() && roots.contains(u.get<Value>()))
        break;
      v = u;
    }
  }
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
          if (pair.first.is<Value>()) {
            assert(N.is<Operation *>());
            newCache = pair.first.get<Value>();
          } else {
            assert(pair.first.is<Operation *>());
            assert(N.is<Value>());
            newCache = N.get<Value>();
          }
          newCaches.insert(newCache);
        }
      }
    }
  }

  // compute path from new caches to required
  parent.clear();
  bfs(Orig, newCaches, parent);

  // The reverse graph is a sub graph of Orig with only pathes from Required
  // to "dominating" caches.
  Graph revGraph = reverseGraph(Orig, newCaches, Required);

  LLVM_DEBUG(llvm::dbgs() << "revGraph:\n");
  LLVM_DEBUG(dump(revGraph));

  // Refine cached values based on some heuristics
  auto newCacheVec = newCaches.takeVector();

  // sort caches to provide determinism.
  llvm::sort(newCacheVec.begin(), newCacheVec.end(), mlir::enzyme::valueCmp);

  for (Value newCache : newCacheVec) {
    worklist.clear();
    worklist.push_back(newCache);

    auto computeSizeOfType = [](Value val) -> int64_t {
      auto T = cast<RankedTensorType>(val.getType());
      if (!T.getElementType().isIntOrFloat())
        return INT64_MAX;
      int64_t sz = T.getElementType().getIntOrFloatBitWidth();
      for (auto sh : T.getShape())
        sz *= sh;
      return sz;
    };

    Value picked = newCache;
    int64_t curSize = computeSizeOfType(picked),
            curRank = cast<RankedTensorType>(picked.getType()).getRank();

    while (!worklist.empty()) {
      Value candidate = worklist.pop_back_val();

      auto C = revGraph.find(Node(candidate));
      if (C == revGraph.end())
        continue;

      if (C->second.size() > 1)
        continue;

      if (candidate.getParentBlock() == reverse)
        continue; // TODO: support this

      int64_t newSize = computeSizeOfType(candidate),
              newRank = cast<RankedTensorType>(candidate.getType()).getRank();
      if (newSize < curSize || (newSize == curSize && newRank < curRank) ||
          candidate.getDefiningOp<enzyme::PopOp>() != nullptr) {
        curSize = newSize;
        curRank = newRank;
        picked = candidate;
      }

      for (auto &N : C->second) {
        // not eligible
        if (N.get<Operation *>()->getNumResults() > 1)
          continue;

        worklist.append(N.get<Operation *>()->getResults().begin(),
                        N.get<Operation *>()->getResults().end());
      }
    }

    auto p = parent.find(Node(picked));
    while (p != parent.end()) {
      revGraph.erase(p->second);
      p = parent.find(p->second);
    }

    newCaches.insert(picked);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "new caches: \n";
    for (Value v : newCaches) {
      v.dump();
    }
  });

  SmallVector<CacheInfo> newCacheInfos;
  IRMapping mapping;

  // For all new caches, materialize the path either by moving ops from
  // forward to reverse or reverse to forward.
  for (Value newCache : newCaches) {
    enzyme::InitOp initOp = ({
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(entry);
      rewriter.create<enzyme::InitOp>(
          newCache.getLoc(),
          enzyme::CacheType::get(newCache.getContext(), newCache.getType()));
    });
    enzyme::PushOp pushOp;
    enzyme::PopOp popOp;

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfterValue(newCache);

    // TODO: This newCache value might not be available here since it might
    // be
    //       a part of the reverse. The operations needed to create newCache
    //       in the forward should be cloned from forward to reverse.
    assert(newCache.getParentBlock() != reverse && "todo");

    pushOp = rewriter.create<enzyme::PushOp>(newCache.getLoc(),
                                             initOp.getResult(), newCache);

    rewriter.setInsertionPointToStart(reverse);
    popOp = rewriter.create<enzyme::PopOp>(
        newCache.getLoc(), newCache.getType(), initOp.getResult());

    mapping.map(newCache, popOp.getResult());

    CacheInfo info;
    info.initOp = initOp;
    info.pushOp = pushOp;
    info.popOp = popOp;
    newCacheInfos.push_back(info);
  }

  worklist.clear();
  worklist.assign(newCaches.begin(), newCaches.end());

  SetVector<Operation *> cloned;

  // Clone ops in the reverse graph to make sure all edges have been mapped.
  while (!worklist.empty()) {
    Value todo = worklist.pop_back_val();

    if (Required.count(todo)) {
      rewriter.replaceAllUsesWith(todo, mapping.lookup(todo));
      continue;
    }

    auto found = revGraph.find(Node(todo));
    assert(found != revGraph.end());

    for (auto N : found->second) {
      assert(N.is<Operation *>());

      // Special case for across forward/reverse boundary.
      if (isa<enzyme::PopOp>(N.get<Operation *>())) {
        rewriter.replaceAllOpUsesWith(N.get<Operation *>(),
                                      mapping.lookup(todo));
        continue;
      }

      if (!llvm::all_of(N.get<Operation *>()->getOperands(),
                        [&mapping](Value operand) {
                          return mapping.contains(operand);
                        })) {
        continue;
      }

      OpBuilder::InsertionGuard guard(rewriter);

      Value lastVal = mapping.lookup(todo);
      Operation *lastValOp = lastVal.getDefiningOp();

      rewriter.setInsertionPointAfterValue(lastVal);
      Operation *newO = rewriter.clone(*N.get<Operation *>(), mapping);
      cloned.insert(newO);

      for (auto [oldRes, newRes] : llvm::zip_equal(
               N.get<Operation *>()->getResults(), newO->getResults()))
        mapping.map(oldRes, newRes);

      auto pair = revGraph.find(N);
      if (pair == revGraph.end())
        continue;

      for (auto NN : pair->second) {
        assert(NN.is<Value>());
        worklist.push_back(NN.get<Value>());
      }
    }
  }

  if (cloned.size()) {
    mlir::sortTopologically(cloned[0]->getBlock());
  }

  // TODO do all the moves for existing ops, then do the ones within
  // dependencies here

  // Remove old caches
  for (auto &info : caches) {
    rewriter.eraseOp(info.popOp);
    rewriter.eraseOp(info.pushOp);
    rewriter.eraseOp(info.initOp);
  }

  // Set new caches
  caches.assign(newCacheInfos.begin(), newCacheInfos.end());
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
