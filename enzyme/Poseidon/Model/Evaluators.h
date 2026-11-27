//===- Evaluators.h - Expression evaluators for Poseidon ------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "Core/FPNode.h"
#include "Core/NumberFormat.h"

namespace poseidon {

/// The highest MPFR precision getMPFRValues evaluates at, in bits.
inline constexpr unsigned kMaxMPFRPrecision = 1024;

/// The arithmetic of a format the target has no instructions for, as the code
/// emitted for it computes it. An evaluator makes one for each change to such
/// a format and asks it for every operation of the change.
class EmulatedArithmetic {
public:
  /// Makes the arithmetic of the format `format`.
  using Factory = std::unique_ptr<EmulatedArithmetic> (*)(NumberFormat format);

  virtual ~EmulatedArithmetic();
  /// The value of `node`, computed in the format and restored to the node's
  /// own type; none for an operation the format has no routine for, which the
  /// evaluator then computes as written. An operand this object computed is
  /// read from what it kept; any other through `valueOf`, at its own type.
  virtual std::optional<double>
  evaluate(const FPNode &node,
           llvm::function_ref<double(const FPNode &)> valueOf) = 0;
};

/// Operations the evaluator computes in a format other than the one they are
/// written in, and with which arithmetic. The nodes belong to the site's graph.
struct FormatChange {
  /// The operations.
  llvm::SetVector<FPLLValue *> nodes;
  /// The format they are computed in as written.
  NumberFormat oldType;
  /// The format the change computes them in.
  NumberFormat newType;
  /// The arithmetic of `newType` when the target has no instructions for it;
  /// null when its operations are rounded to it.
  EmulatedArithmetic::Factory arithmetic;

  explicit FormatChange(llvm::SetVector<FPLLValue *> &nodes,
                        NumberFormat oldType, NumberFormat newType,
                        EmulatedArithmetic::Factory arithmetic = nullptr)
      : nodes(nodes), oldType(oldType), newType(newType),
        arithmetic(arithmetic) {}
};

/// Evaluates expression graphs in floating point at one input point, each node
/// at the format a candidate's precision changes give it and the others at
/// their own, keeping every node's value once computed. One evaluator serves
/// one point and one candidate.
class FPEvaluator {
private:
  llvm::DenseMap<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, NumberFormat> nodePrecisions;
  /// The position, among the candidate's changes, of the change a node
  /// belongs to.
  std::unordered_map<const FPNode *, size_t> nodeChange;
  /// The arithmetic of each change, by position; null for a change whose
  /// operations are rounded to its format.
  std::vector<std::unique_ptr<EmulatedArithmetic>> emulated;
  /// The relative error bound of the square roots a change makes approximate.
  std::unordered_map<const FPNode *, double> approxSqrtError;

  NumberFormat getNodePrecision(const FPNode *node) const;
  /// An if, and, or, not, TRUE or FALSE; false for any other node.
  bool evaluateLogic(const FPNode *node,
                     const llvm::MapVector<llvm::Value *, double> &inputValues);
  /// Any other operation, its operands evaluated, at `nodePrec`.
  double evaluateOperation(const FPNode *node, NumberFormat nodePrec) const;

public:
  /// An evaluator of the candidate whose precision changes are `changes`;
  /// none evaluates the graph as written.
  FPEvaluator(llvm::ArrayRef<FormatChange> changes = {});

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
                 llvm::ArrayRef<FormatChange> changes = {});

/// The exact values, evaluated in MPFR at increasing precision until the first
/// 53 significand bits of each output stop changing; NaN for an output that
/// does not settle by 1024 bits. Returns how many outputs did not.
unsigned
getMPFRValues(llvm::ArrayRef<FPNode *> outputs,
              const llvm::MapVector<llvm::Value *, double> &inputValues,
              llvm::SmallVectorImpl<double> &results);

} // namespace poseidon
#endif // POSEIDON_MODEL_EVALUATORS_H
