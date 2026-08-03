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
#include "mlir/Transforms/RegionUtils.h"
#include <cassert>
#include <deque>

// loop-invariant cache requires a copy, which is implemented using an scf.for
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::enzyme;

#define DEBUG_TYPE "enzyme-mincut"

static llvm::cl::opt<bool>
    DebugGraphviz("mincut-print-graphviz", llvm::cl::init(false),
                  llvm::cl::Hidden,
                  llvm::cl::desc("Use with DEBUG_TYPE 'enzyme-mincut' to print "
                                 "the mincut graphs in GraphViz"));

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

namespace {

// A node in the compute/flow graph. It identifies either an operation or a
// value, plus a one-bit "outgoing" side selector used by the node-split min-cut
// (see minCutValues): a value is split into an incoming endpoint
// (outgoing=false) and an outgoing endpoint (outgoing=true); an operation is
// never split (outgoing is always false).
//
// The op-or-value pointer and the bit are packed into a PointerIntPair so that
// Node stays pointer-sized and pointer-like -- usable directly in
// SmallPtrSet/DenseMap/MapVector via the trait specializations below. Use the
// isValue()/isOperation()/getValue()/getOperation() accessors to inspect it.
struct Node {
  using Base = llvm::PointerUnion<Operation *, Value>;
  llvm::PointerIntPair<Base, 1, bool> pair;

  Node() : pair(Base(), false) {}
  Node(Operation *op) : pair(Base(op), false) {}
  Node(Value v, bool outgoing = false) : pair(Base(v), outgoing) {}
  Node(Base base, bool outgoing) : pair(base, outgoing) {
    assert((base.isNull() || isa<Value>(base) || !outgoing) &&
           "operations are never split: they have a single endpoint");
  }

  Base getBase() const { return pair.getPointer(); }
  bool outgoing() const { return pair.getInt(); }
  bool isNull() const { return getBase().isNull(); }

  bool isValue() const {
    Base b = getBase();
    return isa<Value>(b);
  }
  bool isOperation() const {
    Base b = getBase();
    return isa<Operation *>(b);
  }
  Value getValue() const {
    Base b = getBase();
    return cast<Value>(b);
  }
  Operation *getOperation() const {
    Base b = getBase();
    return cast<Operation *>(b);
  }
  // The value if this node is one, otherwise a null Value.
  Value dynValue() const {
    Base b = getBase();
    return dyn_cast_if_present<Value>(b);
  }

  bool operator==(const Node &o) const { return pair == o.pair; }
  bool operator!=(const Node &o) const { return pair != o.pair; }

  void *getOpaqueValue() const { return pair.getOpaqueValue(); }
  static Node getFromOpaqueValue(void *p) {
    Node n;
    n.pair = llvm::PointerIntPair<Base, 1, bool>::getFromOpaqueValue(p);
    return n;
  }
};

} // namespace

// Make Node usable as a pointer-like key/element (SmallPtrSet, DenseMap, ...)
// by delegating to the packed PointerIntPair. DenseMap tracks empty/tombstone
// buckets out-of-band, so only getHashValue/isEqual are required (matching the
// DenseMapInfo LLVM provides for raw pointers and PointerIntPair).
//
// These name templates in namespace llvm, so they cannot sit inside the
// anonymous namespace above; specializing on an internal-linkage type from
// here is fine, and keeps the specializations internal too.
template <> struct llvm::PointerLikeTypeTraits<Node> {
  using Inner = llvm::PointerIntPair<Node::Base, 1, bool>;
  static inline void *getAsVoidPointer(Node n) { return n.getOpaqueValue(); }
  static inline Node getFromVoidPointer(void *p) {
    return Node::getFromOpaqueValue(p);
  }
  static constexpr int NumLowBitsAvailable =
      llvm::PointerLikeTypeTraits<Inner>::NumLowBitsAvailable;
};

template <> struct llvm::DenseMapInfo<Node> {
  using Inner = llvm::PointerIntPair<Node::Base, 1, bool>;
  static unsigned getHashValue(const Node &n) {
    return llvm::DenseMapInfo<Inner>::getHashValue(n.pair);
  }
  static bool isEqual(const Node &a, const Node &b) { return a.pair == b.pair; }
};

namespace {

void dump(const Node &n) {
  if (n.isValue())
    llvm::errs() << "[" << n.getValue() << ", "
                 << (n.outgoing() ? "Value(out)" : "Value(in)") << "]\n";
  else if (n.isOperation())
    llvm::errs() << "[" << *n.getOperation() << ", "
                 << "Operation"
                 << "]\n";
  else
    llvm::errs() << "["
                 << "NULL"
                 << ", "
                 << "None"
                 << "]\n";
}

// The adjacency sets are insertion-ordered rather than hashed. A Node wraps a
// Value or Operation pointer, so iterating a SmallPtrSet of them visits
// neighbours in address order, which varies run to run under ASLR. The max
// flow below then explores augmenting paths in a different order and, whenever
// several minimum cuts have the same capacity, settles on a different one --
// producing correct but different caching decisions on every run.
using NodeSet = llvm::SmallSetVector<Node, 2>;

struct Graph : public llvm::MapVector<Node, NodeSet> {
  const NodeSet &at(const Node &n) {
    auto found = find(n);
    assert(found != end());
    return found->second;
  }
};

static void dumpGraphviz(Graph &G) {
  auto serialize = [&](Node n) -> std::string {
    std::string s;
    llvm::raw_string_ostream ss(s);
    if (n.isValue()) {
      auto v = n.getValue();
      const char *side = n.outgoing() ? "[val:out]" : "[val:in]";
      if (isa<OpResult>(v)) {
        auto res = cast<OpResult>(v);
        ss << side << "(" << res.getResultNumber() << ")";
        if (res.getOwner()->hasAttr("dbg")) {
          auto dbg = res.getOwner()->getAttrOfType<StringAttr>("dbg");
          ss << dbg.getValue();
        } else {
          ss << res.getOwner()->getName().getStringRef();
        }
      } else {
        ss << side << v;
      }
    } else if (n.isOperation()) {
      auto op = n.getOperation();
      ss << "[op]";
      if (op->hasAttr("dbg")) {
        auto dbg = op->getAttrOfType<StringAttr>("dbg");
        ss << dbg.getValue();
      } else {
        ss << op->getName().getStringRef();
      }
    } else {
      ss << "none";
    }
    return s;
  };

  using llvm::errs;
  errs() << "digraph G {\n";
  for (auto &pair : G) {
    for (const auto &N : pair.second) {
      errs() << "  \"" << serialize(pair.first) << "\" -> \"" << serialize(N)
             << "\";\n";
    }
  }

  errs() << "}\n";
}

static void dump(Graph &G) {
  if (DebugGraphviz) {
    dumpGraphviz(G);
  } else {
    for (auto &pair : G) {
      dump(pair.first);
      for (const auto &N : pair.second) {
        llvm::errs() << "\t";
        dump(N);
      }
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

static inline bool isLoadMovable(Operation *op) {
  if (!hasSingleEffect<MemoryEffects::Read>(op)) {
    return false;
  }
  return op->hasAttr("enzyme.readonly");
}

static inline bool isMovable(Operation *op);
static inline bool isRegionBranchMovable(RegionBranchOpInterface regionBranch) {
  // We can move region ops that only contain pure operations. As a heuristic,
  // we only consider non-looping ops.
  if (regionBranch.hasLoop())
    return false;
  for (auto &region : regionBranch->getRegions()) {
    // Regions with multiple blocks potentially contain loops
    if (!region.hasOneBlock())
      return false;

    for (auto &bodyOp : region.front()) {
      if (auto bodyRegionBranch = dyn_cast<RegionBranchOpInterface>(&bodyOp)) {
        if (!isRegionBranchMovable(bodyRegionBranch))
          return false;
        // Terminators are considered not movable, but do not impact our ability
        // to move their enclosing region op.
      } else if (&bodyOp != region.front().getTerminator() &&
                 !isMovable(&bodyOp)) {
        return false;
      }
    }
  }
  return true;
}

// Whether or not an operation can be moved from the forward region to the
// reverse region or vice-versa.
static inline bool isMovable(Operation *op) {
  if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(op))
    return isRegionBranchMovable(regionBranch);
  return op->getNumRegions() == 0 && op->getBlock()->getTerminator() != op &&
         (mlir::isPure(op) || isLoadMovable(op));
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

// Annotate operations with a debug attribute. This makes the GraphViz printing
// nicer.
static void annotate_ops(Block *forward, Block *reverse) {
  unsigned counter = 0;
  forward->walk([&](Operation *op) {
    auto debugName =
        StringAttr::get(op->getContext(),
                        op->getName().stripDialect() + llvm::Twine(counter++));
    op->setAttr("dbg", debugName);
  });
  reverse->walk([&](Operation *op) {
    auto debugName =
        StringAttr::get(op->getContext(),
                        op->getName().stripDialect() + llvm::Twine(counter++));
    op->setAttr("dbg", debugName);
  });
}

// The node used when a graph node is the outgoing (tail) endpoint of an edge: a
// value uses its outgoing endpoint (outgoing=true), an operation uses its
// single node.
static Node edgeTail(Node n) {
  if (Value v = n.dynValue())
    return Node(v, /*outgoing=*/true);
  return n;
}
// The node used when a graph node is the incoming (head) endpoint of an edge: a
// value uses its incoming endpoint (outgoing=false), an operation uses its
// single node.
static Node edgeHead(Node n) {
  if (Value v = n.dynValue())
    return Node(v, /*outgoing=*/false);
  return n;
}

// Given the compute graph `Orig` (with `roots` the non-recomputable sources and
// `Required` the operations that force a value to be live in reverse), return
// the minimal set of values to cache.
//
// The cut is computed on a node-split version of the graph: every value V is
// split into an incoming node (outgoing=false) and an outgoing node
// (outgoing=true) joined by a single unit-capacity edge, so that all of V's
// flow funnels through it and caching V costs exactly one regardless of how
// many operations consume it. Operations are not split. Since the split graph
// is a plain `Graph` over the same `Node` type, the existing bfs/dump utilities
// apply to it directly.
static SetVector<Value> minCutValues(const Graph &Orig,
                                     const SetVector<Value> &roots,
                                     const SetVector<Operation *> &Required) {
  Graph G;
  // Build the node-split graph. Each original edge is either operation->value
  // (a definition) or value->operation (a use).
  for (const auto &pair : Orig) {
    Node A = pair.first;
    // Internal split edge for the tail value (no-op for an operation).
    if (Value va = A.dynValue())
      G[Node(va, /*outgoing=*/false)].insert(Node(va, /*outgoing=*/true));
    for (Node B : pair.second) {
      G[edgeTail(A)].insert(edgeHead(B));
      // Internal split edge for the head value (no-op for an operation).
      if (Value vb = B.dynValue())
        G[Node(vb, /*outgoing=*/false)].insert(Node(vb, /*outgoing=*/true));
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "node-split flow graph: \n";);
  LLVM_DEBUG(dump(G));

  // Save the original edges for cut extraction; the max-flow below mutates G
  // into its residual graph. The flow sources are the incoming endpoints of the
  // roots, which is exactly what bfs() seeds from a set of root values.
  Graph Split = G;

  // Edmonds-Karp: repeatedly augment along a shortest source->sink path in the
  // residual graph until no augmenting path remains. All edges have unit
  // capacity, represented by set membership.
  while (true) {
    DenseMap<Node, Node> parent;
    bfs(G, roots, parent);
    Node end;
    for (Operation *req : Required) {
      if (parent.count(Node(req))) {
        end = Node(req);
        break;
      }
    }
    if (end.isNull())
      break;
    // Flip the residual edges along the found path, stopping at the source (a
    // root's incoming endpoint).
    Node v = end;
    while (true) {
      assert(parent.count(v));
      Node u = parent.find(v)->second;
      assert(!u.isNull());
      G[u].remove(v);
      G[v].insert(u);
      if (u.isValue() && !u.outgoing() && roots.contains(u.getValue()))
        break;
      v = u;
    }
  }

  // Reachable set from the sources in the residual graph.
  DenseMap<Node, Node> parent;
  bfs(G, roots, parent);

  LLVM_DEBUG(llvm::dbgs() << "residual flow graph: \n";);
  LLVM_DEBUG(dump(G));

  // The min cut is the set of edges from a reachable node to a non-reachable
  // node in the original graph. Each such edge names exactly one value to
  // cache: the value being transported across it. Because values are split,
  // this edge is one of
  //   - the internal split edge V_in -> V_out  (=> cache V), or
  //   - a use edge V_out -> op                 (=> cache V), or
  //   - a def edge op -> V_in                  (=> cache the produced value).
  // Operations are never split, so at least one endpoint of the cut edge is a
  // value, and that value is the one to cache.
  SetVector<Value> newCaches;
  for (const auto &pair : Split) {
    if (!parent.count(pair.first))
      continue;
    for (const Node &N : pair.second) {
      if (parent.count(N))
        continue;
      assert((pair.first.isValue() || N.isValue()) &&
             "min-cut edge must transport a value");
      Value cache = pair.first.isValue() ? pair.first.getValue() : N.getValue();
      newCaches.insert(cache);
    }
  }
  return newCaches;
}

// When the min cut is ambiguous, prefer caching the LAST value in a computation
// chain.
//
// Max flow fixes the *capacity* of the cut but not which of the equal-capacity
// cuts is returned, and `minCutValues` extracts the one reachable from the
// sources -- i.e. the cut nearest the roots, which caches the earliest value in
// every chain and leaves the longest possible tail to recompute in reverse.
// Sliding a cut edge downstream past an operation that has a single graph user
// and a single result yields another cut of the same capacity (the slid edge is
// the only way from that value to a required op), so this never costs an extra
// cache; it just shrinks what the reverse pass rebuilds.
//
// The size guard keeps the slide from trading a small buffer for a larger one:
// capacities here are unit, so the flow minimizes the NUMBER of cached values,
// not their bytes, and two equal-capacity cuts can differ in size.
//
// This mirrors the "push to cache the last value in a computation chain"
// heuristic in the LLVM Enzyme mincut (Enzyme/DifferentialUseAnalysis.cpp).
static void pushCachesDownstream(const Graph &Orig,
                                 const SetVector<Operation *> &Required,
                                 SetVector<Value> &newCaches) {
  SmallVector<Value> todo(newCaches.begin(), newCaches.end());

  while (!todo.empty()) {
    Value cur = todo.pop_back_val();
    // May have been slid away already by an earlier iteration.
    if (!newCaches.contains(cur))
      continue;

    // `cur` must feed exactly one operation; otherwise moving the cut past it
    // would mean caching several values in place of one.
    auto users = Orig.find(Node(cur));
    if (users == Orig.end() || users->second.size() != 1)
      continue;
    Node userNode = *users->second.begin();
    if (!userNode.isOperation())
      continue;
    Operation *user = userNode.getOperation();

    // A required operation consumes `cur` itself in the reverse pass, so there
    // is nothing downstream of it to slide to.
    if (Required.contains(user))
      continue;

    // ... and that operation must produce exactly one value.
    auto results = Orig.find(userNode);
    if (results == Orig.end() || results->second.size() != 1)
      continue;
    Node resNode = *results->second.begin();
    if (!resNode.isValue())
      continue;
    Value next = resNode.getValue();

    // Never slide into a deeper region: a value defined inside a nested block
    // is live more often than the one it would replace. (The LLVM mincut makes
    // the same check by loop nest, and additionally slides *outwards*; here we
    // only ever move within one block.)
    if (next.getParentBlock() != cur.getParentBlock())
      continue;

    // Never trade a cached buffer for a larger one.
    if (computeSizeOfType(cur) < computeSizeOfType(next))
      continue;

    newCaches.remove(cur);
    newCaches.insert(next);
    // Keep sliding down the chain.
    todo.push_back(next);
  }
}

} // namespace

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

  LLVM_DEBUG(if (DebugGraphviz) annotate_ops(forward, reverse));

  // where to build the new inits
  Operation *entry = caches0[0].initOp;

  IRMapping mapping = fwdrevmap;
  SmallVector<CacheInfo> caches;
  // Hoist out pushes of values that are defined outside of the block
  for (auto &info : caches0) {
    auto todo = info.pushedValue();
    bool isDefinedOutside =
        !forward->getParent()->isAncestor(todo.getParentRegion());
    if (isDefinedOutside) {
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

  // Maintain a mapping of forward to reverse blocks. We later use this to place
  // the new cache pops and cloned ops to the correct blocks.
  DenseMap<Block *, OpBuilder::InsertPoint> insertionPointMap;
  for (const auto &info : caches) {
    Block *fwdBlock = info.pushOp->getBlock();
    Block *revBlock = info.popOp->getBlock();
    // For the top-level reverse block, we use the provided rewriter's insertion
    // point (to skip over things like IV calculations). New operations in inner
    // blocks should be inserted at the beginning of those blocks.
    if (revBlock == reverse) {
      insertionPointMap[fwdBlock] = rewriter.saveInsertionPoint();
    } else {
      insertionPointMap[fwdBlock] =
          OpBuilder::InsertPoint(revBlock, revBlock->begin());
    }
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

    bool isDefinedOutside =
        !forward->getParent()->isAncestor(todo.getParentRegion());
    if (isDefinedOutside || fwdrevmap.contains(todo)) {
      continue;
    }

    Operation *owner = todo.getDefiningOp();
    if (!owner || !isMovable(owner)) {
      roots.insert(todo);
      continue;
    }

    bool inserted = G[Node(owner)].insert(Node(todo));
    if (inserted) {
      for (Value operand : owner->getOperands()) {
        G[Node(operand)].insert(Node(owner));
        worklist.push_back(operand);
      }
    }
    if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(owner)) {
      SetVector<Value> valuesDefinedAbove;
      mlir::getUsedValuesDefinedAbove(regionBranch->getRegions(),
                                      valuesDefinedAbove);
      for (Value val : valuesDefinedAbove) {
        G[Node(val)].insert(Node(regionBranch));
        worklist.push_back(val);
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
        bool inserted = G[Node(todo)].insert(N);
        if (inserted) {
          for (Value res : user->getResults()) {
            G[N].insert(Node(res));
            worklist.push_back(res);
          }
        }
      }
    }

    for (auto N : G) {
      if (!N.first.isOperation())
        continue;
      auto op = N.first.getOperation();
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

  // Compute the values to cache via a minimum cut. The cut is computed on a
  // node-split version of the graph (see minCutValues), so that caching a value
  // costs exactly one regardless of how many operations consume it. Without the
  // split, a value used by N operations is charged N (one per outgoing edge),
  // which causes the mincut to prefer caching several downstream values over a
  // single cheaper upstream one (e.g. an input reused many times). This mirrors
  // the node-splitting done by the LLVM Enzyme mincut in
  // DifferentialUseAnalysis.cpp.
  SetVector<Value> newCaches = minCutValues(Orig, roots, Required);

  // The cut above is only one of several equal-capacity min cuts; slide it as
  // far downstream as is free so the reverse pass recomputes as little as
  // possible.
  pushCachesDownstream(Orig, Required, newCaches);

  assert(rewriter.getInsertionPoint()->getBlock() == reverse);

  // compute path from new caches to required
  DenseMap<Node, Node> parent;
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
      assert(nextF.isOperation());
      auto opNext = nextF.getOperation();

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
      if (!forward->getParent()->isAncestor(newCache.getParentRegion())) {
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

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.restoreInsertionPoint(
          insertionPointMap.lookup(newCache.getParentBlock()));
      enzyme::PopOp popOp = enzyme::PopOp::create(
          rewriter, newCache.getLoc(), newCache.getType(), initOp.getResult());
      insertionPointMap[newCache.getParentBlock()] =
          rewriter.saveInsertionPoint();
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

  forward->walk([&](Operation *op) {
    if (!cacheGraph.contains(Node(op)))
      return;
    bool hasUse = false;
    for (auto res : op->getResults()) {
      if (newCaches.contains(res)) {
        continue;
      }
      hasUse = true;
    }
    if (!hasUse)
      return;
    for (auto v : op->getOperands()) {
      if (mapping.contains(v))
        continue;
      if (forward->getParent()->isAncestor(v.getParentRegion()))
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
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.restoreInsertionPoint(insertionPointMap.lookup(op->getBlock()));
    auto cop = rewriter.clone(*op, mapping);
    insertionPointMap[op->getBlock()] = rewriter.saveInsertionPoint();
    if (!firstClone)
      firstClone = cop;
  });

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

  caches0 = std::move(newCacheInfos);
}

static LogicalResult loopInvariantCacheImpl(CacheInfo info,
                                            LoopLikeOpInterface fwdLoop,
                                            LoopLikeOpInterface revLoop,
                                            IRMapping &mapping,
                                            int64_t threshold) {
  Operation *definingOp = info.pushedValue().getDefiningOp();
  if (!definingOp)
    return failure();
  auto loadOp = dyn_cast<memref::LoadOp>(definingOp);
  if (!loadOp)
    return failure();
  auto alloc = loadOp.getMemRef();
  auto allocaOp = dyn_cast_if_present<memref::AllocaOp>(alloc.getDefiningOp());
  if (!allocaOp)
    return failure();
  // This transformation will store the entire allocation, so only store when
  // it has a known "small" size.
  if (!(allocaOp.getType().hasStaticShape() &&
        allocaOp.getType().getNumElements() <= threshold)) {
    return failure();
  }

  // Check if the allocation is written to inside the loop body
  // TODO: need alias analysis for full correctness
  auto walkResult = fwdLoop->walk([&alloc](Operation *op) {
    if (hasEffect<MemoryEffects::Write>(op, alloc) || hasUnknownEffects(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  ImplicitLocOpBuilder builder(alloc.getLoc(), info.initOp);
  // Find the values that are necessary to re-compute the indices
  SmallVector<Value> pushedValues;
  SmallVector<Operation *> toCopy;
  std::queue<Value> worklist;
  DenseSet<Value> visited;
  for (Value idx : loadOp.getIndices()) {
    worklist.push(idx);
    visited.insert(idx);
  }
  while (!worklist.empty()) {
    Value curr = worklist.front();
    worklist.pop();

    if (mapping.contains(curr))
      continue;
    DominanceInfo dom;
    // Check if we can just re-use the same value in forward and reverse
    if (dom.dominates(curr, revLoop))
      continue;

    if (Operation *definingOp = curr.getDefiningOp()) {
      // Push values that are defined outside the fwd loop while copying ops
      // defined inside
      if (fwdLoop->isProperAncestor(definingOp)) {
        toCopy.push_back(definingOp);
        for (Value operand : definingOp->getOperands()) {
          if (!visited.contains(operand)) {
            visited.insert(operand);
            worklist.push(operand);
          }
        }
      } else {
        pushedValues.push_back(curr);
      }
    } else {
      return failure();
    }
  }

  // Make new caches for the values needed to store the indices
  builder.setInsertionPoint(info.initOp);
  SmallVector<Value> newInits =
      llvm::map_to_vector(pushedValues, [&](Value val) -> Value {
        return enzyme::InitOp::create(
            builder, CacheType::get(val.getContext(), val.getType()));
      });
  builder.setInsertionPoint(fwdLoop);
  for (auto &&[cache, val] : llvm::zip(newInits, pushedValues)) {
    enzyme::PushOp::create(builder, cache, val);
  }

  builder.setInsertionPoint(revLoop);
  for (auto &&[cache, val] : llvm::zip(newInits, pushedValues)) {
    Value popped = enzyme::PopOp::create(builder, val.getType(), cache);
    mapping.map(val, popped);
  }

  // Clone the ops necessary to re-compute the load indices
  builder.setInsertionPoint(info.popOp);
  for (Operation *op : llvm::reverse(toCopy)) {
    builder.clone(*op, mapping);
  }

  Value cachedMemRef;
  if (mapping.contains(alloc)) {
    cachedMemRef = mapping.lookup(alloc);
  } else {
    builder.setInsertionPoint(info.initOp);
    Value newCache = enzyme::InitOp::create(
        builder, enzyme::CacheType::get(builder.getContext(), alloc.getType()));

    builder.setInsertionPoint(fwdLoop);
    Value copyAlloc = memref::AllocOp::create(
        builder, allocaOp.getType(), allocaOp.getDynamicSizes(),
        allocaOp.getSymbolOperands(), allocaOp.getAlignmentAttr());
    // Copy over the contents in a loop nest so any resulting subviews may be
    // folded
    Value zero = arith::ConstantIndexOp::create(builder, 0);
    Value one = arith::ConstantIndexOp::create(builder, 1);

    int64_t rank = loadOp.getMemRefType().getRank();
    SmallVector<Value> lbs(rank, zero);
    SmallVector<Value> steps(rank, one);
    SmallVector<Value> ubs(rank);
    for (int64_t i = 0; i < rank; i++)
      ubs[i] = memref::DimOp::create(builder, alloc, i);

    scf::buildLoopNest(
        builder, builder.getLoc(), lbs, ubs, steps,
        [&](OpBuilder &bodyBuilder, Location loc, ValueRange ivs) {
          Value loaded = memref::LoadOp::create(bodyBuilder, loc, alloc, ivs);
          memref::StoreOp::create(bodyBuilder, loc, loaded, copyAlloc, ivs);
        });

    enzyme::PushOp::create(builder, newCache, copyAlloc);

    builder.setInsertionPoint(revLoop);
    cachedMemRef = enzyme::PopOp::create(builder, alloc.getType(), newCache);
    mapping.map(alloc, cachedMemRef);
  }

  builder.setInsertionPoint(info.popOp);
  SmallVector<Value> revIndices =
      llvm::map_to_vector(loadOp.getIndices(), [&](Value idx) -> Value {
        return mapping.lookupOrDefault(idx);
      });

  Value newLoad = memref::LoadOp::create(builder, cachedMemRef, revIndices);
  info.popOp.replaceAllUsesWith(newLoad);
  return success();
}

void mlir::enzyme::loopInvariantCache(SmallVectorImpl<CacheInfo> &caches,
                                      LoopLikeOpInterface fwdLoop,
                                      LoopLikeOpInterface revLoop,
                                      const IRMapping &fwdrevmap,
                                      int64_t threshold) {
  SmallVector<CacheInfo> newCaches;
  IRMapping mapping = fwdrevmap;
  for (auto info : caches) {
    if (succeeded(loopInvariantCacheImpl(info, fwdLoop, revLoop, mapping,
                                         threshold))) {
      info.pushOp->erase();
      info.popOp->erase();
      info.initOp->erase();
    } else {
      newCaches.push_back(info);
    }
  }
  caches = std::move(newCaches);
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
