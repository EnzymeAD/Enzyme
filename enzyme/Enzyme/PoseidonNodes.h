//=- PoseidonNodes.h - AST node declarations for Poseidon -----------------=//
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

#ifndef ENZYME_POSEIDON_NODES_H
#define ENZYME_POSEIDON_NODES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

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
  double grad;
  double geoMean;
  double arithMean;
  double maxAbs;
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

#endif // ENZYME_POSEIDON_NODES_H