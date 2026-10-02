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

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include <mpfr.h>
#include <unordered_map>

#include "Precision.h"
#include "Types.h"

namespace poseidon {

class FPEvaluator {
private:
  std::unordered_map<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;
  // Exact limbs of nodes evaluated as an FP32 expansion of three or more
  // components; without them an expansion op feeding another would receive an
  // operand already rounded to 53 bits.
  std::unordered_map<const FPNode *, llvm::SmallVector<float, 4>> expCache;
  // Set only under -poseidon-accuracy-reference-bits.
  bool exactExpansion = false;

public:
  FPEvaluator(PTCandidate *pt = nullptr);

  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  void evaluateNode(const FPNode *node,
                    const llvm::MapVector<llvm::Value *, double> &inputValues);
  double getResult(const FPNode *node) const;
  // Limbs recorded for `node`, or nullptr when it was not evaluated as a
  // >= 3-component expansion. The value is the unevaluated sum of the limbs.
  const llvm::SmallVector<float, 4> *getResultLimbs(const FPNode *node) const;
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
  MPFREvaluator(unsigned prec, PTCandidate *pt = nullptr);
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
                 PTCandidate *pt = nullptr);

void getMPFRValues(llvm::ArrayRef<FPNode *> outputs,
                   const llvm::MapVector<llvm::Value *, double> &inputValues,
                   llvm::SmallVectorImpl<double> &results, bool groundTruth,
                   const unsigned groundTruthPrec = 53,
                   PTCandidate *pt = nullptr);

// Per-output error of `pt` (or the original subgraph) against an MPFR reference
// at `refBits`, with the subtraction itself in MPFR so the difference of two
// nearly equal values keeps its width. `candNonFinite` marks outputs where the
// candidate is non-finite while the reference is finite (the strict-mode
// signal).
void getSampleErrorsWide(
    llvm::ArrayRef<FPNode *> outputs,
    const llvm::MapVector<llvm::Value *, double> &inputValues,
    llvm::SmallVectorImpl<double> &errors, unsigned refBits,
    PTCandidate *pt = nullptr,
    llvm::SmallVectorImpl<char> *candNonFinite = nullptr);

} // namespace poseidon
#endif // POSEIDON_EVALUATORS_H