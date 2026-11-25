//===- FPNode.h - the expression graph of a site --------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
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

#ifndef POSEIDON_MODEL_FP_NODE_H
#define POSEIDON_MODEL_FP_NODE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace poseidon {

class FPNode {
public:
  enum class NodeType { Node, LLValue, Const };

private:
  const NodeType ntype;

public:
  std::string op;
  std::string dtype;
  std::string symbol;
  llvm::SmallVector<std::shared_ptr<FPNode>, 2> operands;
  // Sensitivity score, the sum of |grad * value|.
  double sens = std::numeric_limits<double>::quiet_NaN();
  // The sum of the gradients, signed.
  double grad = std::numeric_limits<double>::quiet_NaN();
  uint64_t executions = 0;

  explicit FPNode(const std::string &op, const std::string &dtype)
      : ntype(NodeType::Node), op(op), dtype(dtype) {}
  explicit FPNode(NodeType ntype, const std::string &op,
                  const std::string &dtype)
      : ntype(ntype), op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  NodeType getType() const;
  void addOperand(std::shared_ptr<FPNode> operand);
  bool hasSymbol() const { return !symbol.empty(); }
  // The Herbie expression of a node of the original graph, a value or a
  // literal: a subgraph input as its symbol, anything else spelled out over
  // its operands.
  std::string
  toFullExpression(std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                       &valueToNodeMap,
                   const llvm::SetVector<llvm::Value *> &subgraphInputs,
                   unsigned depth = 0);
  // Widens the profiled range of a value to [lower, upper]; a literal's range
  // stays its value.
  void updateBounds(double lower, double upper);
  double getLowerBound() const { return lb; }
  double getUpperBound() const { return ub; }
  virtual llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr);

protected:
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
};

class FPLLValue;
// The number of operations input `V` is spelled as in a Herbie core, or 0 if
// it stays a variable. An input whose definition is a few operations over the
// subgraph's other inputs and constants is spelled as that definition, so
// Herbie sees the relation (log T next to T); the count includes the inputs
// spelled out in turn.
unsigned inlinedOps(const FPLLValue &V,
                    const llvm::SetVector<llvm::Value *> &inputs);
// The existing value a parsed rewrite node recomputes, if the node is one of
// the operations an input was spelled as: the rewrite reuses it.
std::shared_ptr<FPNode> existingDefinition(
    const FPNode &N, const llvm::SetVector<llvm::Value *> &inputs,
    std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> &valueToNodeMap);

class FPLLValue : public FPNode {
public:
  llvm::Value *value;

  explicit FPLLValue(llvm::Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(NodeType::LLValue, op, dtype), value(value) {}

  llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

// A literal; both its bounds are its value as a double, read once, as the
// evaluators read it at every sample and a rational literal is parsed in MPFR.
class FPConst : public FPNode {
private:
  std::string strValue;

  static double literalValue(const std::string &strValue);

public:
  explicit FPConst(const std::string &strValue, const std::string &dtype)
      : FPNode(NodeType::Const, "__const", dtype), strValue(strValue) {
    lb = ub = literalValue(strValue);
  }

  const std::string &literal() const { return strValue; }
  llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

// Lowers one expression per output, element k at the builder `at(k)` returns,
// computing a subterm several elements share once wherever that copy
// dominates the later use (every earlier copy, when DT is null).
llvm::SmallVector<llvm::Value *, 4>
emitSharedElements(llvm::ArrayRef<std::shared_ptr<FPNode>> elements,
                   llvm::function_ref<llvm::IRBuilder<> &(size_t)> at,
                   llvm::ValueToValueMapTy *VMap,
                   const llvm::DominatorTree *DT);

// The fcmp predicate of FPCore comparison operator `Op`: IEEE's, under which
// only != holds when an operand is NaN. None for another operator.
std::optional<llvm::CmpInst::Predicate> fpcoreComparison(llvm::StringRef Op);

class SubgraphSamples;

struct Subgraph {
  llvm::SetVector<llvm::Value *> inputs;
  llvm::SetVector<llvm::Instruction *> outputs;
  llvm::SetVector<llvm::Instruction *> operations;
  size_t outputs_rewritten = 0;
  // The points its units are priced at, while they are priced.
  std::shared_ptr<SubgraphSamples> samples;
  // Frequency scale for synthetic output-boundary casts: they execute at the
  // insertion point's frequency (the output's consumer), not the body's.
  // min(1, threadsPerLaunch * launchCount / exec(output)); 1 for elementwise.
  double outBoundaryFreqScale = 1.0;

  // Measured execution counts of this subgraph's instructions (`opExec`) and
  // the same counts by loop depth (`depthExec`), which survive
  // materialization. Empty prices exactly as an unweighted walk.
  llvm::DenseMap<const llvm::Instruction *, uint64_t> opExec;
  std::map<unsigned, uint64_t> depthExec;
  uint64_t execNormalizer = 0;

  // The measured count of the deepest class at or below `d` (the shallowest
  // class outside every recorded loop) over the normalizer; 1.0 without one.
  double freqWeightAtDepth(unsigned d) const {
    if (depthExec.empty() || execNormalizer == 0)
      return 1.0;
    auto it = depthExec.upper_bound(d);
    uint64_t e = (it == depthExec.begin()) ? depthExec.begin()->second
                                           : std::prev(it)->second;
    return (double)e / (double)execNormalizer;
  }

  Subgraph() = default;
  explicit Subgraph(llvm::SetVector<llvm::Value *> inputs,
                    llvm::SetVector<llvm::Instruction *> outputs,
                    llvm::SetVector<llvm::Instruction *> operations)
      : inputs(inputs), outputs(outputs), operations(operations) {}
};

// The expression graph of one site: the function it is built over, a node per
// value (the subgraphs' inputs and operations), and the value each symbol of a
// Herbie expression names.
struct FPGraph {
  llvm::Function *F = nullptr;
  std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>> valueToNodeMap;
  std::unordered_map<std::string, llvm::Value *> symbolToValueMap;

  // `erased` was erased and its users read `now` instead: a later rewrite
  // whose expression names `erased` reads `now`.
  void replaceErased(llvm::Value *erased, llvm::Value *now);
};

} // namespace poseidon
#endif // POSEIDON_MODEL_FP_NODE_H
