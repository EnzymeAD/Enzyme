//===- MatmulAccuracyMemo.h - Stored results of the matrix accuracy model -===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A matrix accuracy model's result is a deterministic function of its inputs,
// so it is kept in -poseidon-cache, one entry file per key. The key holds the
// model's inputs and the compiler that computed it, never its results: hosts
// differ in the last digits of a result for equal inputs. An entry is filed
// under every part but the compiler, so one computed by another build is found
// and reported.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMULACCURACYMEMO_H
#define POSEIDON_MATMULACCURACYMEMO_H

#include "Core/CacheKey.h"
#include "Core/FPKind.h"

#include <string>

namespace poseidon {

struct AbstractMatmul;
struct MatmulProfile;
struct Options;
class Session;

// The memo entry of one evaluation of a matrix accuracy model.
class MatmulAccuracyMemo {
public:
  // For a MatmulAccuracyQuery::Model::Rounded query on these inputs.
  static MatmulAccuracyMemo matmul(const Options &O, const AbstractMatmul &m,
                                   const MatmulProfile &prof, double confidence,
                                   unsigned sampleLogBits, unsigned numSamples,
                                   FPKind inputPrec, FPKind accPrec,
                                   FPKind exponentPrec, unsigned orderTileK,
                                   unsigned inputMantBits);
  // For a MatmulAccuracyQuery::Model::OzakiII query on these inputs.
  static MatmulAccuracyMemo ozakiII(const Options &O, const AbstractMatmul &m,
                                    const MatmulProfile &prof,
                                    double confidence, unsigned sampleLogBits,
                                    unsigned numSamples, unsigned moduli);
  // For a MatmulAccuracyQuery::Model::TcecLibrary query on these inputs.
  static MatmulAccuracyMemo
  tcecLibrary(const Options &O, const AbstractMatmul &m,
              const MatmulProfile &prof, double confidence,
              unsigned sampleLogBits, unsigned numSamples);

  // The stored result, or one imported by -poseidon-import-accuracy-memo;
  // false when the model has to compute it.
  bool lookup(Session &S, double &accuracyCost, double &domainError) const;
  void store(const Options &O, double accuracyCost, double domainError) const;
  // The entry's file in -poseidon-cache: two memos of one file hold one
  // result.
  const std::string &file() const { return path; }

private:
  CacheKey key;
  std::string path;
  std::string legacyName;
};

} // namespace poseidon
#endif // POSEIDON_MATMULACCURACYMEMO_H
