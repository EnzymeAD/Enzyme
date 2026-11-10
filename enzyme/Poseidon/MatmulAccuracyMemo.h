//=- MatmulAccuracyMemo.h - Stored results of the matrix accuracy model ---=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A matrix accuracy model's result is a deterministic function of its inputs,
// so it is kept in flags::Cache, one entry file per key. The key holds the
// model's inputs and the model's identity, never its results: hosts differ in
// the last digits of a result for equal inputs. An entry is filed under every
// part but the identity, so one computed by another model is found and
// reported, and one computed by a model declared equivalent is reused.
//
// This code is kept out of the files the model's identity is a digest of, so
// changing it leaves every stored result valid.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMULACCURACYMEMO_H
#define POSEIDON_MATMULACCURACYMEMO_H

#include "CacheKey.h"
#include "FPKind.h"

#include <string>

namespace poseidon {

struct AbstractMatmul;
struct MatmulProfile;

// The memo entry of one evaluation of a matrix accuracy model.
class MatmulAccuracyMemo {
public:
  // For getMatmulAccuracyCost on these inputs.
  static MatmulAccuracyMemo matmul(const AbstractMatmul &m,
                                   const MatmulProfile &prof, double confidence,
                                   unsigned sampleLogBits, unsigned numSamples,
                                   FPKind inputPrec, FPKind accPrec,
                                   FPKind exponentPrec, unsigned orderTileK,
                                   unsigned inputMantBits);
  // For getOzakiIIAccuracyCost on these inputs.
  static MatmulAccuracyMemo ozakiII(const AbstractMatmul &m,
                                    const MatmulProfile &prof,
                                    double confidence, unsigned sampleLogBits,
                                    unsigned numSamples, unsigned capturedBits);

  // The stored result, or one imported by -poseidon-import-accuracy-memo;
  // false when the model has to compute it.
  bool lookup(double &accuracyCost, double &domainError) const;
  void store(double accuracyCost, double domainError) const;

private:
  CacheKey key;
  std::string path;
  std::string legacyName;
};

} // namespace poseidon
#endif // POSEIDON_MATMULACCURACYMEMO_H
