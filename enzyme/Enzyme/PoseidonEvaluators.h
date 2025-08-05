//=- PoseidonEvaluators.h - Expression evaluators for Poseidon ------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the evaluator classes for floating-point expressions
// in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_EVALUATORS_H
#define ENZYME_POSEIDON_EVALUATORS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/Value.h"
#include <mpfr.h>
#include <unordered_map>

using namespace llvm;

class FPNode;
struct PTCandidate;
enum class PrecisionChangeType;

class FPEvaluator {
private:
  std::unordered_map<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;

public:
  FPEvaluator(PTCandidate *pt = nullptr);

  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues);
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
  MPFREvaluator(unsigned prec, PTCandidate *pt = nullptr);
  virtual ~MPFREvaluator() = default;

  unsigned getNodePrecision(const FPNode *node, bool groundTruth) const;
  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues,
                    bool groundTruth);
  mpfr_t &getResult(FPNode *node);
};

#endif // ENZYME_POSEIDON_EVALUATORS_H