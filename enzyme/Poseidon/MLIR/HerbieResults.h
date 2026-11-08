//===- HerbieResults.h - Herbie's answers for poseidon-rewrite ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What poseidon-rewrite reads back from Herbie: its results.json, and the
// expressions in it. Herbie is not run here yet; the results come from a file
// in the format a real run writes.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MLIR_HERBIERESULTS_H
#define POSEIDON_MLIR_HERBIERESULTS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir::poseidon {

// An expression as Herbie writes it, e.g. (+.f64 (*.f64 x0 x1) #s(literal 1/2
// binary64)).
struct HerbieExpr {
  enum class Kind { Op, Var, Number };
  Kind kind;
  // Op: the operator without its precision suffix ("+", "log1p", "if").
  // Var: the variable name. Number: the literal as Herbie wrote it ("1/2",
  // "-inf.0", "3.5").
  std::string name;
  // "f64" or "f32": the precision an operator computes in, or a literal is
  // rounded to. Empty when Herbie gave none.
  std::string precision;
  std::vector<HerbieExpr> args;

  // The expression without precisions or approx wrappers, e.g. (+ (* x0 x1)
  // 1/2).
  void print(llvm::raw_ostream &os) const;
};

// Parses one Herbie expression; on failure, says why in `error`.
std::optional<HerbieExpr> parseHerbieExpr(llvm::StringRef text,
                                          std::string &error);

// One rewrite Herbie proposes for a core.
struct HerbieCandidate {
  std::string expr;
  // Herbie's own estimates: its average error in bits, and its cost relative
  // to the original expression's.
  double errorBits;
  double cost;
};

// Herbie's answer for one core.
struct HerbieAnswer {
  // e.g. "imp-start" (improved), "ex-start" (already accurate), "timeout".
  std::string status;
  // The original expression's average error in bits.
  double initialErrorBits;
  // Most accurate first: Herbie's output, then its alternatives. Empty when
  // Herbie found nothing ("output": "#f").
  std::vector<HerbieCandidate> candidates;
};

// Herbie's results.json, by core name; on failure, says why in `error`.
std::optional<llvm::StringMap<HerbieAnswer>>
parseHerbieResults(llvm::StringRef json, std::string &error);

// Stands in for running Herbie: the results.json text of an earlier run, read
// from `path`.
std::optional<std::string> readHerbieResults(llvm::StringRef path,
                                             std::string &error);

} // namespace mlir::poseidon

#endif // POSEIDON_MLIR_HERBIERESULTS_H
