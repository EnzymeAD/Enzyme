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

#ifndef POSEIDON_EVALUATORS_H
#define POSEIDON_EVALUATORS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include <unordered_map>

#include "Core/ExpansionModel.h"
#include "FPFormat.h"
#include "FPNode.h"

namespace poseidon {

// The highest MPFR precision getMPFRValues evaluates at, in bits.
inline constexpr unsigned kMaxMPFRPrecision = 1024;

class FPEvaluator {
private:
  llvm::DenseMap<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;
  // The change of the candidate a node belongs to, and the components of the
  // nodes evaluated as an FP32 expansion.
  std::unordered_map<const FPNode *, size_t> nodeChange;
  std::unordered_map<const FPNode *, ExpansionLimbs> limbs;
  // The relative error bound of the square roots a change makes approximate.
  std::unordered_map<const FPNode *, double> approxSqrtError;

  bool evaluateExpansionNode(const FPNode *node, unsigned n);
  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  // An if, and, or, not, TRUE or FALSE; false for any other node.
  bool evaluateLogic(const FPNode *node,
                     const llvm::MapVector<llvm::Value *, double> &inputValues);
  // Any other operation, its operands evaluated, at `nodePrec`.
  double evaluateOperation(const FPNode *node,
                           PrecisionChangeType nodePrec) const;

public:
  FPEvaluator(llvm::ArrayRef<PrecisionChange> changes = {});

  void evaluateNode(const FPNode *node,
                    const llvm::MapVector<llvm::Value *, double> &inputValues);
  double getResult(const FPNode *node) const;
};

void getFPValues(llvm::ArrayRef<FPNode *> outputs,
                 const llvm::MapVector<llvm::Value *, double> &inputValues,
                 llvm::SmallVectorImpl<double> &results,
                 llvm::ArrayRef<PrecisionChange> changes = {});

// The exact values, evaluated in MPFR at increasing precision until the first
// 53 significand bits of each output stop changing; NaN for an output that
// does not settle by 1024 bits. Returns how many outputs did not.
unsigned
getMPFRValues(llvm::ArrayRef<FPNode *> outputs,
              const llvm::MapVector<llvm::Value *, double> &inputValues,
              llvm::SmallVectorImpl<double> &results);

} // namespace poseidon
#endif // POSEIDON_EVALUATORS_H