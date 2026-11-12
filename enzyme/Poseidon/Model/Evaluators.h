//=- Evaluators.h - Expression evaluators for Poseidon --------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the evaluator classes for floating-point expressions
// in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_EVALUATORS_H
#define POSEIDON_EVALUATORS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"
#include <mpfr.h>
#include <unordered_map>

#include "Core/ExpansionModel.h"
#include "FPFormat.h"
#include "FPNode.h"

namespace poseidon {

class FPEvaluator {
private:
  std::unordered_map<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;
  // The change of the candidate a node belongs to, and the components of the
  // nodes evaluated as an FP32 expansion.
  std::unordered_map<const FPNode *, size_t> nodeChange;
  std::unordered_map<const FPNode *, ExpansionLimbs> limbs;
  // The relative error bound of the square roots a change makes approximate.
  std::unordered_map<const FPNode *, double> approxSqrtError;

  bool evaluateExpansionNode(const FPNode *node, unsigned n);

public:
  FPEvaluator(llvm::ArrayRef<PrecisionChange> changes = {});

  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  void evaluateNode(const FPNode *node,
                    const llvm::MapVector<llvm::Value *, double> &inputValues);
  double getResult(const FPNode *node) const;
};

class MPFREvaluator {
private:
  struct CachedValue {
    mpfr_t value;
    unsigned prec;

    CachedValue(unsigned prec);
    CachedValue(const CachedValue &) = delete;
    CachedValue &operator=(const CachedValue &) = delete;
    CachedValue(CachedValue &&other) noexcept;
    CachedValue &operator=(CachedValue &&other) noexcept;
    virtual ~CachedValue();
  };

  std::unordered_map<const FPNode *, CachedValue> cache;
  unsigned prec;
  std::unordered_map<const FPNode *, unsigned> nodeToNewPrec;

public:
  MPFREvaluator(unsigned prec, llvm::ArrayRef<PrecisionChange> changes = {});
  virtual ~MPFREvaluator() = default;

  unsigned getNodePrecision(const FPNode *node, bool groundTruth) const;
  void evaluateNode(const FPNode *node,
                    const llvm::MapVector<llvm::Value *, double> &inputValues,
                    bool groundTruth);
  mpfr_t &getResult(FPNode *node);
};

void getFPValues(llvm::ArrayRef<FPNode *> outputs,
                 const llvm::MapVector<llvm::Value *, double> &inputValues,
                 llvm::SmallVectorImpl<double> &results,
                 llvm::ArrayRef<PrecisionChange> changes = {});

void getMPFRValues(llvm::ArrayRef<FPNode *> outputs,
                   const llvm::MapVector<llvm::Value *, double> &inputValues,
                   llvm::SmallVectorImpl<double> &results, bool groundTruth,
                   const unsigned groundTruthPrec = 53,
                   llvm::ArrayRef<PrecisionChange> changes = {});

} // namespace poseidon
#endif // POSEIDON_EVALUATORS_H