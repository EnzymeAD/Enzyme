//===- ErrorMoment.h - how the accuracy model sums rounding errors --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A candidate's accuracy cost sums, over the outputs of what it rewrites, the
// error a sample of the output carries times a weight the profile summed over
// the output's executions, and is read against the site's summed sensitivity.
//
// First moment: |error| times sum |gradient|, against sum |gradient x value|.
// Every execution's error counts with its full magnitude, as if all of them
// had one sign. Second moment: error^2 times sum gradient^2, against
// sum (gradient x value)^2, the first-order variance of the seeded output
// when the rounding errors of distinct executions are independent; the
// relative error is the root of the quotient, a root-sum-of-squares. Only the
// second moment ranks two candidates as their errors do when one spreads its
// rounding over far more executions than the other.
//
// -poseidon-error-sum names the weight: the signed sum of an operation's
// gradients or the sum of their magnitudes, both in the first moment, or the
// sum of their squares in the second. Without it the profile decides: the
// squares when it records them, else the magnitudes, which every profile
// records.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_CORE_ERROR_MOMENT_H
#define POSEIDON_CORE_ERROR_MOMENT_H

#include "llvm/Support/ErrorHandling.h"

#include <cmath>

namespace poseidon {

/// The moment the accuracy model sums a site's rounding errors in.
enum class ErrorMoment { First, Second };

/// How an operation's rounding errors are summed over its executions: weighted
/// by the signed sum of its gradients, by the sum of their magnitudes, or
/// squared and weighted by the sum of their squares.
enum class ErrorSum { Signed, Absolute, Squares };

/// The moment the sum `s` is in.
inline ErrorMoment momentOf(ErrorSum s) {
  return s == ErrorSum::Squares ? ErrorMoment::Second : ErrorMoment::First;
}

/// The value of -poseidon-error-sum that names `s`.
inline const char *errorSumName(ErrorSum s) {
  switch (s) {
  case ErrorSum::Signed:
    return "signed";
  case ErrorSum::Absolute:
    return "absolute";
  case ErrorSum::Squares:
    return "squares";
  }
  llvm_unreachable("every error sum has a name");
}

/// What the error of `result` against `goldVal` contributes to an accuracy
/// cost in moment `m`: its magnitude, or its square.
inline double momentError(double goldVal, double result, ErrorMoment m) {
  const double d = std::fabs(goldVal - result);
  return m == ErrorMoment::First ? d : d * d;
}

/// The relative error the accuracy cost `accCost` stands for against the
/// scale `accScale` it is read against, both in moment `m`.
inline double relativeError(double accCost, double accScale, ErrorMoment m) {
  const double r = accCost / accScale;
  return m == ErrorMoment::First ? r : std::sqrt(r);
}

/// The accuracy cost in moment `m` that stands for the relative error `rel`
/// against the scale `accScale`; relativeError's inverse.
inline double accuracyCostOf(double rel, double accScale, ErrorMoment m) {
  return (m == ErrorMoment::First ? rel : rel * rel) * accScale;
}

} // namespace poseidon
#endif // POSEIDON_CORE_ERROR_MOMENT_H
