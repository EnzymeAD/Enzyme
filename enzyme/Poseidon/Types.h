//=- Types.h - AST node declarations for Poseidon -------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AST node classes for representing floating-point
// expressions in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TYPES_H
#define POSEIDON_TYPES_H

#include "CostModel.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>

#include "Utils.h"

#include "Precision.h"

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
  double sens =
      std::numeric_limits<double>::quiet_NaN(); // Sensitivity score, sum of
                                                // |grad * value|
  double grad =
      std::numeric_limits<double>::quiet_NaN(); // Sum of gradients (not abs)
  unsigned executions = 0;

  explicit FPNode(const std::string &op, const std::string &dtype)
      : ntype(NodeType::Node), op(op), dtype(dtype) {}
  explicit FPNode(NodeType ntype, const std::string &op,
                  const std::string &dtype)
      : ntype(ntype), op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  NodeType getType() const;
  void addOperand(std::shared_ptr<FPNode> operand);
  virtual bool hasSymbol() const;
  virtual std::string
  toFullExpression(std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                       &valueToNodeMap,
                   const llvm::SetVector<llvm::Value *> &subgraphInputs,
                   unsigned depth = 0);
  unsigned getMPFRPrec() const;
  virtual void updateBounds(double lower, double upper);
  virtual double getLowerBound() const;
  virtual double getUpperBound() const;
  virtual llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr);
};

class FPLLValue : public FPNode {
private:
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();

public:
  llvm::Value *value;

  explicit FPLLValue(llvm::Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(NodeType::LLValue, op, dtype), value(value) {}

  bool hasSymbol() const override;
  std::string
  toFullExpression(std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                       &valueToNodeMap,
                   const llvm::SetVector<llvm::Value *> &subgraphInputs,
                   unsigned depth = 0) override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
  llvm::Value *
  getLLValue(llvm::IRBuilder<> &builder,
             const llvm::ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

class FPConst : public FPNode {
private:
  std::string strValue;

public:
  explicit FPConst(const std::string &strValue, const std::string &dtype)
      : FPNode(NodeType::Const, "__const", dtype), strValue(strValue) {}

  std::string
  toFullExpression(std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                       &valueToNodeMap,
                   const llvm::SetVector<llvm::Value *> &subgraphInputs,
                   unsigned depth = 0) override;
  bool hasSymbol() const override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
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

struct Subgraph {
  llvm::SetVector<llvm::Value *> inputs;
  llvm::SetVector<llvm::Instruction *> outputs;
  llvm::SetVector<llvm::Instruction *> operations;
  size_t outputs_rewritten = 0;
  // Frequency scale for synthetic output-boundary casts: they execute at the
  // insertion point's frequency (the output's consumer), not the body's.
  // min(1, threadsPerLaunch * launchCount / exec(output)); 1 for elementwise.
  double outBoundaryFreqScale = 1.0;

  Subgraph() = default;
  explicit Subgraph(llvm::SetVector<llvm::Value *> inputs,
                    llvm::SetVector<llvm::Instruction *> outputs,
                    llvm::SetVector<llvm::Instruction *> operations)
      : inputs(inputs), outputs(outputs), operations(operations) {}
};

struct SolutionStep;
class CandidateMatmul;

struct RewriteCandidate {
  // Double, not InstructionCost: cost-model rows are reciprocal throughputs
  // below 1.0, and rounding them to an integral cost per op would make every
  // elementwise candidate cost exactly 0.
  double CompCost = std::numeric_limits<double>::max();
  double herbieCost = std::numeric_limits<double>::quiet_NaN();
  double herbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  std::string expr;

  RewriteCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), herbieAccuracy(accuracy), expr(expression) {}
};

class CandidateOutput {
public:
  Subgraph *subgraph;
  llvm::Value *oldOutput;
  std::string expr;
  double grad = std::numeric_limits<double>::quiet_NaN();
  unsigned executions = 0;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  llvm::SmallVector<RewriteCandidate> candidates;
  llvm::SetVector<llvm::Instruction *> erasableInsts;
  // The outputs one candidate replaces, in array element order: the single
  // oldOutput for a per-output core, all of them for an array core.
  llvm::SmallVector<llvm::Value *, 1> roots;
  llvm::SmallVector<double, 1> rootGrads;
  bool herbieAnswered = false;

  explicit CandidateOutput(Subgraph &subgraph, llvm::Value *oldOutput,
                           std::string expr, double grad, unsigned executions)
      : subgraph(&subgraph), oldOutput(oldOutput), expr(expr), grad(grad),
        executions(executions), roots({oldOutput}), rootGrads({grad}) {
    initialCompCost = getCompCost({oldOutput}, subgraph.inputs);
    findErasableInstructions();
  }

  explicit CandidateOutput(Subgraph &subgraph,
                           llvm::ArrayRef<llvm::Value *> outputs,
                           llvm::ArrayRef<double> grads, std::string expr,
                           unsigned executions)
      : subgraph(&subgraph), oldOutput(outputs.front()), expr(expr),
        executions(executions), roots(outputs.begin(), outputs.end()),
        rootGrads(grads.begin(), grads.end()) {
    grad = 0.0;
    for (double g : grads)
      grad += std::fabs(g);
    initialCompCost = getCompCost(
        llvm::SmallVector<llvm::Value *>(outputs.begin(), outputs.end()),
        subgraph.inputs);
    findErasableInstructions();
  }

  bool isArray() const { return roots.size() > 1; }
  bool sharesRoot(const CandidateOutput &other) const;

  void apply(size_t candidateIndex,
             std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
                 &valueToNodeMap,
             std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);
  llvm::InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);

private:
  void findErasableInstructions();
};

class CandidateSubgraph {
public:
  Subgraph *subgraph;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  double initialCompCost = std::numeric_limits<double>::quiet_NaN();
  unsigned executions = 0;
  llvm::MapVector<FPNode *, double> perOutputInitialAccCost;
  llvm::SmallVector<PTCandidate, 8> candidates;

  using CandidateOutputSet = std::set<CandidateOutput *>;
  struct CacheKey {
    size_t candidateIndex;
    CandidateOutputSet CandidateOutputs;
    bool operator==(const CacheKey &other) const;
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey &key) const;
  };

  std::unordered_map<CacheKey, llvm::InstructionCost, CacheKeyHash>
      compCostDeltaCache;
  std::unordered_map<CacheKey, double, CacheKeyHash> accCostDeltaCache;

  explicit CandidateSubgraph(Subgraph &subgraph) : subgraph(&subgraph) {
    initialCompCost = getCompCost(
        {subgraph.outputs.begin(), subgraph.outputs.end()}, subgraph.inputs);
  }

  void apply(size_t candidateIndex);
  llvm::InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  llvm::InstructionCost
  getAdjustedCompCostDelta(size_t candidateIndex,
                           const llvm::SmallVectorImpl<SolutionStep> &steps);
  double getAdjustedAccCostDelta(
      size_t candidateIndex, const llvm::SmallVectorImpl<SolutionStep> &steps,
      std::unordered_map<llvm::Value *, std::shared_ptr<FPNode>>
          &valueToNodeMap,
      std::unordered_map<std::string, llvm::Value *> &symbolToValueMap);
};

struct SolutionStep {
  std::variant<CandidateOutput *, CandidateSubgraph *, CandidateMatmul *> item;
  size_t candidateIndex;

  SolutionStep(CandidateOutput *ao_, size_t idx)
      : item(ao_), candidateIndex(idx) {}
  SolutionStep(CandidateSubgraph *acc_, size_t idx)
      : item(acc_), candidateIndex(idx) {}
  SolutionStep(CandidateMatmul *cm_, size_t idx)
      : item(cm_), candidateIndex(idx) {}
};

} // namespace poseidon
#endif // POSEIDON_TYPES_H