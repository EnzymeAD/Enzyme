//===- Evaluators.h - Expression evaluators for Poseidon ------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The values of an expression graph at a point: as the program computes them,
// at the precisions a candidate's changes give its nodes, and exactly.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MODEL_EVALUATORS_H
#define POSEIDON_MODEL_EVALUATORS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include <unordered_map>

#include "FPFormat.h"
#include "FPNode.h"
#include "Techniques/Precision/ExpansionModel.h"

namespace poseidon {

/// The highest MPFR precision getMPFRValues evaluates at, in bits.
inline constexpr unsigned kMaxMPFRPrecision = 1024;

/// Evaluates expression graphs in floating point at one input point, each node
/// at the format a candidate's precision changes give it and the others at
/// their own, keeping every node's value once computed. One evaluator serves
/// one point and one candidate.
class FPEvaluator {
private:
  llvm::DenseMap<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;
  /// The position, among the candidate's changes, of the change a node
  /// belongs to.
  std::unordered_map<const FPNode *, size_t> nodeChange;
  /// The components of each node evaluated as an FP32 expansion.
  std::unordered_map<const FPNode *, FloatExpansion> components;
  /// The relative error bound of the square roots a change makes approximate.
  std::unordered_map<const FPNode *, double> approxSqrtError;

  bool evaluateExpansionNode(const FPNode *node, unsigned n);
  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  /// An if, and, or, not, TRUE or FALSE; false for any other node.
  bool evaluateLogic(const FPNode *node,
                     const llvm::MapVector<llvm::Value *, double> &inputValues);
  /// Any other operation, its operands evaluated, at `nodePrec`.
  double evaluateOperation(const FPNode *node,
                           PrecisionChangeType nodePrec) const;

public:
  /// An evaluator of the candidate whose precision changes are `changes`;
  /// none evaluates the graph as written.
  FPEvaluator(llvm::ArrayRef<PrecisionChange> changes = {});

  /// Computes `node` and its operands at the point `inputValues`, which gives
  /// a value to every input the node reads.
  void evaluateNode(const FPNode *node,
                    const llvm::MapVector<llvm::Value *, double> &inputValues);
  /// The value of `node`, which evaluateNode has computed.
  double getResult(const FPNode *node) const;
};

/// The value of each of `outputs` at the point `inputValues`, under the
/// precision changes `changes`, into `results`.
void getFPValues(llvm::ArrayRef<FPNode *> outputs,
                 const llvm::MapVector<llvm::Value *, double> &inputValues,
                 llvm::SmallVectorImpl<double> &results,
                 llvm::ArrayRef<PrecisionChange> changes = {});

/// The exact values, evaluated in MPFR at increasing precision until the first
/// 53 significand bits of each output stop changing; NaN for an output that
/// does not settle by 1024 bits. Returns how many outputs did not.
unsigned
getMPFRValues(llvm::ArrayRef<FPNode *> outputs,
              const llvm::MapVector<llvm::Value *, double> &inputValues,
              llvm::SmallVectorImpl<double> &results);

} // namespace poseidon
#endif // POSEIDON_MODEL_EVALUATORS_H