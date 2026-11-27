//===- FPNode.h - the expression graph of a site --------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A site's floating-point operations as an expression graph: a node per value
// a subgraph reads or computes (FPLLValue), a node per literal (FPConst), and
// the operator nodes a parsed rewrite adds (FPNode); the FP subgraphs over it.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_FPNODE_H
#define POSEIDON_CORE_FPNODE_H

#include "Core/ErrorMoment.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace poseidon {

struct FPCoreOperator;
struct NumberFormatInfo;
class FPGraph;
enum class NumberFormat;

/// Whether V is a floating-point operation the pass profiles and rewrites: an
/// instruction it can model, whose operands are float or double arguments and
/// literals, integer literals, or scalar floating-point or integer
/// instructions.
bool isOptimizable(const llvm::Value &V);

/// A node of the expression graph: an operation, with its operands, of a
/// parsed rewrite. Values of the program and literals are the subclasses
/// FPLLValue and FPConst. Every node is owned by the FPGraph that created it.
class FPNode {
public:
  /// The subclass: an FPNode of a parsed rewrite, an FPLLValue or an FPConst.
  enum class NodeType { Node, LLValue, Const };

private:
  const NodeType ntype;

public:
  /// The operator of an operation: a row of the FPCore operator table. Null
  /// for a value, a literal, and an operation of the program the table does
  /// not model.
  const FPCoreOperator *op;
  /// The format of the node's value; null for a value that is not a
  /// floating-point number of a format of the pass.
  const NumberFormatInfo *format;
  /// The FPCore variable ("v0") a Herbie core names a subgraph input by;
  /// empty for any other node.
  std::string symbol;
  /// The operands, owned by the same graph.
  llvm::SmallVector<FPNode *, 2> operands;
  /// Sensitivity score: |gradient x value| summed over the profiled
  /// executions, the gradient being that of the declared metric; NaN for a
  /// node the profile has no record of.
  double sens = std::numeric_limits<double>::quiet_NaN();
  /// The accuracy weight: the gradient of the declared metric with respect to
  /// the value, summed over the profiled executions in magnitude, or with its
  /// sign under -poseidon-error-sum=signed (ProfileInfo::weight); NaN without
  /// a record.
  double grad = std::numeric_limits<double>::quiet_NaN();
  /// The second moment's weight, gradient^2 summed over the profiled
  /// executions; NaN when the profile records no second moments.
  double sqGrad = std::numeric_limits<double>::quiet_NaN();
  /// How often the profiling run executed the operation.
  uint64_t executions = 0;

  /// An operation of a parsed rewrite; to be created through an FPGraph.
  explicit FPNode(const FPCoreOperator &op, const NumberFormatInfo *format)
      : ntype(NodeType::Node), op(&op), format(format) {}
  explicit FPNode(NodeType ntype, const FPCoreOperator *op,
                  const NumberFormatInfo *format)
      : ntype(ntype), op(op), format(format) {}
  virtual ~FPNode() = default;

  NodeType getType() const;
  /// Whether the node's operator is the row `O`.
  bool is(const FPCoreOperator &O) const { return op == &O; }
  /// Whether the node's value has the format of precision `P`.
  bool hasPrecision(NumberFormat P) const;
  /// Appends `operand`, a node of the same graph.
  void addOperand(FPNode *operand);
  bool hasSymbol() const { return !symbol.empty(); }
  /// The Herbie expression of a node of the original graph, a value or a
  /// literal: a subgraph input as its symbol, anything else spelled out over
  /// its operands. An expression nested deeper than the pass spells out is an
  /// error at the instruction.
  llvm::Expected<std::string>
  fullExpression(const FPGraph &G,
                 const llvm::SetVector<llvm::Value *> &subgraphInputs,
                 unsigned depth = 0);
  /// Widens the profiled range of a value to [lower, upper]; a literal's range
  /// stays its value.
  void updateBounds(double lower, double upper);
  /// The least value the profile recorded; +inf before any was recorded.
  double getLowerBound() const { return lb; }
  /// The greatest value the profile recorded; -inf before any was recorded.
  double getUpperBound() const { return ub; }
  /// The profiled range as [lower, upper] when it is an interval of finite
  /// values; none when the profile recorded no value or an infinite one.
  std::optional<std::pair<double, double>> profiledRange() const {
    if (!std::isfinite(lb) || !std::isfinite(ub) || lb > ub)
      return std::nullopt;
    return std::make_pair(lb, ub);
  }
  /// Emits the node at `builder`, its operands first and each cast to the
  /// node's format, and returns its value. `VMap`, when given, maps each value
  /// of the program a leaf names to the value to read instead.
  virtual llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr);

protected:
  /// An if: a select between its arms, both at the node's precision.
  llvm::Value *lowerChoice(llvm::IRBuilder<> &builder,
                           const llvm::ValueToValueMapTy *VMap);

  /// The profiled range of the value; empty (lb > ub) before any is recorded.
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
};

/// The weight of `node`'s sampled error in moment `m`: |grad|, or its summed
/// squared gradient.
inline double accuracyWeight(const FPNode &node, ErrorMoment m) {
  return momentWeight(node.grad, node.sqGrad, m);
}

class FPLLValue;
/// The number of operations input `V` is spelled as in a Herbie core, or 0 if
/// it stays a variable. An input whose definition is a few operations over the
/// subgraph's other inputs and constants is spelled as that definition, so
/// Herbie sees the relation (log T next to T); the count includes the inputs
/// spelled out in turn.
unsigned inlinedOps(const FPLLValue &V,
                    const llvm::SetVector<llvm::Value *> &inputs);
/// The node of the existing value the parsed rewrite node `N` recomputes, if
/// `N` is one of the operations an input of `inputs` was spelled as; null
/// otherwise. The node returned belongs to `G`.
FPNode *existingDefinition(const FPNode &N,
                           const llvm::SetVector<llvm::Value *> &inputs,
                           const FPGraph &G);

/// A value of the program: an operation of a subgraph or a value it reads.
class FPLLValue : public FPNode {
public:
  /// The value; null once the value was erased by a rewrite.
  llvm::Value *value;

  /// `op` is the operator of the operation `value` is, null for any other
  /// value.
  explicit FPLLValue(llvm::Value *value, const FPCoreOperator *op,
                     const NumberFormatInfo *format)
      : FPNode(NodeType::LLValue, op, format), value(value) {}

  /// The value, or its image under `VMap`, which must map it; emits nothing.
  /// The value must not have been erased.
  llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

/// A literal; both its bounds are its value as a double, read once, as the
/// evaluators read it at every sample and a rational literal is parsed in MPFR.
class FPConst : public FPNode {
private:
  std::string strValue;

  /// An FPCore literal, a decimal, an infinity or an exact rational, as the
  /// nearest double.
  static double literalValue(const std::string &strValue);

public:
  /// The literal `literal` of `format`; to be created through an FPGraph.
  explicit FPConst(std::string literal, const NumberFormatInfo *format)
      : FPNode(NodeType::Const, nullptr, format), strValue(std::move(literal)) {
    lb = ub = literalValue(strValue);
  }

  /// The literal as written: a decimal, an exact rational p/q or an
  /// infinity.
  const std::string &literal() const { return strValue; }
  /// The literal as a constant of the node's format; emits nothing.
  llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

/// Lowers one expression per output, element k at the builder `at(k)` returns,
/// computing a subterm several elements share once wherever that copy
/// dominates the later use (every earlier copy, when DT is null). The nodes
/// that stand for a shared copy are created in `G`, which owns the elements.
llvm::SmallVector<llvm::Value *, 4>
emitSharedElements(FPGraph &G, llvm::ArrayRef<FPNode *> elements,
                   llvm::function_ref<llvm::IRBuilder<> &(size_t)> at,
                   llvm::ValueToValueMapTy *VMap,
                   const llvm::DominatorTree *DT);

class SubgraphSamples;

/// An FP subgraph: a connected set of a site's floating-point operations, the
/// values it reads from outside and the operations whose values are read
/// outside it.
struct Subgraph {
  /// The values other than floating-point constants that the operations read
  /// and the subgraph does not compute.
  llvm::SetVector<llvm::Value *> inputs;
  /// The operations whose values are read outside the subgraph.
  llvm::SetVector<llvm::Instruction *> outputs;
  /// Every operation of the subgraph, its outputs included.
  llvm::SetVector<llvm::Instruction *> operations;
  /// How many outputs the materialized rewrites replaced; the operations are
  /// erased when it reaches the output count.
  size_t outputsRewritten = 0;
  /// The points its units are priced at, while they are priced.
  std::shared_ptr<SubgraphSamples> samples;
  /// The moment its accuracy costs are summed in, its site's.
  ErrorMoment moment = ErrorMoment::First;

  Subgraph() = default;
  explicit Subgraph(llvm::SetVector<llvm::Value *> inputs,
                    llvm::SetVector<llvm::Instruction *> outputs,
                    llvm::SetVector<llvm::Instruction *> operations)
      : inputs(inputs), outputs(outputs), operations(operations) {}
};

/// The expression graph of one site: the function it is built over, a node
/// per value (the subgraphs' inputs and operations), and the value each symbol
/// of a Herbie expression names. The graph owns every node created through it,
/// those of the parsed rewrites included.
class FPGraph {
public:
  /// The site's function.
  llvm::Function *F = nullptr;
  /// The node of each value of the subgraphs, and of a materialized rewrite's
  /// new output.
  llvm::DenseMap<llvm::Value *, FPNode *> valueToNodeMap;
  /// The value each symbol of a Herbie core names.
  std::unordered_map<std::string, llvm::Value *> symbolToValueMap;

  FPGraph() = default;
  FPGraph(FPGraph &&) = default;
  FPGraph &operator=(FPGraph &&) = default;

  /// A node of type T, built from `args`, that the graph owns.
  template <typename T, typename... Args> T *create(Args &&...args) {
    owned.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    return static_cast<T *>(owned.back().get());
  }

  /// A graph over this one's maps that owns only the nodes created through it:
  /// what is parsed into it leaves this graph as it is. It reads this graph's
  /// nodes, so it must not outlive it.
  FPGraph overlay() const;

  /// `erased` was erased and its users read `now` instead: a later rewrite
  /// whose expression names `erased` reads `now`.
  void replaceErased(llvm::Value *erased, llvm::Value *now);

private:
  std::vector<std::unique_ptr<FPNode>> owned;
};

} // namespace poseidon
#endif // POSEIDON_CORE_FPNODE_H
