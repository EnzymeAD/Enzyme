//=- PoseidonTypes.h - AST node declarations for Poseidon -----------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AST node classes for representing floating-point
// expressions in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_TYPES_H
#define ENZYME_POSEIDON_TYPES_H

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>

#include "PoseidonUtils.h"

#include "PoseidonPrecUtils.h"

using namespace llvm;

class FPNode {
public:
  enum class NodeType { Node, LLValue, Const };

private:
  const NodeType ntype;

public:
  std::string op;
  std::string dtype;
  std::string symbol;
  SmallVector<std::shared_ptr<FPNode>, 2> operands;
  double sens; // Sensitivity score, sum of |grad * value|
  double grad; // Sum of gradients (not abs)
  unsigned executions;

  explicit FPNode(const std::string &op, const std::string &dtype)
      : ntype(NodeType::Node), op(op), dtype(dtype) {}
  explicit FPNode(NodeType ntype, const std::string &op,
                  const std::string &dtype)
      : ntype(ntype), op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  NodeType getType() const;
  void addOperand(std::shared_ptr<FPNode> operand);
  virtual bool hasSymbol() const;
  virtual std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0);
  unsigned getMPFRPrec() const;
  virtual void markAsInput();
  virtual void updateBounds(double lower, double upper);
  virtual double getLowerBound() const;
  virtual double getUpperBound() const;
  virtual Value *getLLValue(IRBuilder<> &builder,
                            const ValueToValueMapTy *VMap = nullptr);
};

class FPLLValue : public FPNode {
private:
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
  bool input = false;

public:
  Value *value;

  explicit FPLLValue(Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(NodeType::LLValue, op, dtype), value(value) {}

  bool hasSymbol() const override;
  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0) override;
  void markAsInput() override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
  Value *getLLValue(IRBuilder<> &builder,
                    const ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

class FPConst : public FPNode {
private:
  std::string strValue;

public:
  explicit FPConst(const std::string &strValue, const std::string &dtype)
      : FPNode(NodeType::Const, "__const", dtype), strValue(strValue) {}

  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0) override;
  bool hasSymbol() const override;
  void markAsInput() override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
  Value *getLLValue(IRBuilder<> &builder,
                    const ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

struct Subgraph {
  SetVector<Value *> inputs;
  SetVector<Instruction *> outputs;
  SetVector<Instruction *> operations;
  size_t outputs_rewritten = 0;

  Subgraph() = default;
  explicit Subgraph(SetVector<Value *> inputs, SetVector<Instruction *> outputs,
                    SetVector<Instruction *> operations)
      : inputs(inputs), outputs(outputs), operations(operations) {}
};

struct SolutionStep;

struct RewriteCandidate {
  InstructionCost CompCost = std::numeric_limits<InstructionCost>::max();
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
  Value *oldOutput;
  std::string expr;
  double grad = std::numeric_limits<double>::quiet_NaN();
  unsigned executions = 0;
  const TargetTransformInfo *TTI = nullptr;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost initialCompCost =
      std::numeric_limits<InstructionCost>::quiet_NaN();
  double initialHerbieCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  SmallVector<RewriteCandidate> candidates;
  SmallPtrSet<Instruction *, 8> erasableInsts;

  explicit CandidateOutput(Subgraph &subgraph, Value *oldOutput,
                           std::string expr, double grad, unsigned executions,
                           const TargetTransformInfo &TTI)
      : subgraph(&subgraph), oldOutput(oldOutput), expr(expr), grad(grad),
        executions(executions), TTI(&TTI) {
    initialCompCost = getCompCost({oldOutput}, subgraph.inputs, TTI);
    findErasableInstructions();
  }

  void
  apply(size_t candidateIndex,
        std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
        std::unordered_map<std::string, Value *> &symbolToValueMap);
  InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);

private:
  void findErasableInstructions();
};

class CandidateSubgraph {
public:
  Subgraph *subgraph;
  const TargetTransformInfo &TTI;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost initialCompCost =
      std::numeric_limits<InstructionCost>::quiet_NaN();
  unsigned executions = 0;
  std::unordered_map<FPNode *, double> perOutputInitialAccCost;
  SmallVector<PTCandidate, 8> candidates;

  using CandidateOutputSet = std::set<CandidateOutput *>;
  struct CacheKey {
    size_t candidateIndex;
    CandidateOutputSet CandidateOutputs;
    bool operator==(const CacheKey &other) const;
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey &key) const;
  };

  std::unordered_map<CacheKey, InstructionCost, CacheKeyHash>
      compCostDeltaCache;
  std::unordered_map<CacheKey, double, CacheKeyHash> accCostDeltaCache;

  explicit CandidateSubgraph(Subgraph &subgraph, const TargetTransformInfo &TTI)
      : subgraph(&subgraph), TTI(TTI) {
    initialCompCost =
        getCompCost({subgraph.outputs.begin(), subgraph.outputs.end()},
                    subgraph.inputs, TTI);
  }

  void apply(size_t candidateIndex);
  InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  InstructionCost
  getAdjustedCompCostDelta(size_t candidateIndex,
                           const SmallVectorImpl<SolutionStep> &steps);
  double getAdjustedAccCostDelta(
      size_t candidateIndex, SmallVectorImpl<SolutionStep> &steps,
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      std::unordered_map<std::string, Value *> &symbolToValueMap);
};

struct SolutionStep {
  std::variant<CandidateOutput *, CandidateSubgraph *> item;
  size_t candidateIndex;

  SolutionStep(CandidateOutput *ao_, size_t idx)
      : item(ao_), candidateIndex(idx) {}
  SolutionStep(CandidateSubgraph *acc_, size_t idx)
      : item(acc_), candidateIndex(idx) {}
};

void getSampledPoints(
    ArrayRef<Value *> inputs,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints);

void getSampledPoints(
    const std::string &expr,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<MapVector<Value *, double>, 4> &sampledPoints);

#endif // ENZYME_POSEIDON_TYPES_H