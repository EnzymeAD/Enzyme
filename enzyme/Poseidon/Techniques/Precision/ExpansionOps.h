//===- ExpansionOps.h - the operations on floating-point expansions -------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_PRECISION_EXPANSION_OPS_H
#define POSEIDON_TECHNIQUES_PRECISION_EXPANSION_OPS_H

#include "Techniques/Precision/ExpansionRoutines.h"

#include <cassert>
#include <utility>

namespace poseidon {

/// Each operation on the FP32 expansions `Values::Expansion`, computed by the
/// routines of ExpansionRoutines<Arith>: the ds* routine for a pair, the
/// expansion* one for a wider expansion. `Values` gives an expansion's
/// component count (size), its components, most significant first
/// (components), and an expansion of n components to fill (sized).
template <typename Arith, typename Values> class ExpansionOps {
  using Routines = ExpansionRoutines<Arith>;
  using F32 = typename Arith::F32;
  using Expansion = typename Values::Expansion;

public:
  explicit ExpansionOps(Arith A) : R(std::move(A)) {}

  Expansion neg(const Expansion &a) {
    return unary(a, &Routines::dsNeg, &Routines::expansionNeg);
  }
  Expansion sqrt(const Expansion &a) {
    return unary(a, &Routines::dsSqrt, &Routines::expansionSqrt);
  }
  Expansion add(const Expansion &a, const Expansion &b) {
    return binary(a, b, &Routines::dsAdd, &Routines::expansionAdd);
  }
  Expansion sub(const Expansion &a, const Expansion &b) {
    return binary(a, b, &Routines::dsSub, &Routines::expansionSub);
  }
  Expansion mul(const Expansion &a, const Expansion &b) {
    return binary(a, b, &Routines::dsMul, &Routines::expansionMul);
  }
  Expansion div(const Expansion &a, const Expansion &b) {
    return binary(a, b, &Routines::dsDiv, &Routines::expansionDiv);
  }
  /// a * a: the general product for a pair.
  Expansion sqr(const Expansion &a) {
    const unsigned n = Values::size(a);
    if (n == 2)
      return mul(a, a);
    Expansion r = Values::sized(n);
    R.expansionSqr(n, Values::components(a), Values::components(r));
    return r;
  }
  /// fma(a, b, c): a product and then a sum for n >= 3.
  Expansion fma(const Expansion &a, const Expansion &b, const Expansion &c) {
    if (Values::size(a) == 2)
      return ofPair(R.dsFma(pairOf(a), pairOf(b), pairOf(c)));
    return add(mul(a, b), c);
  }
  /// fma(a, a, c): a square and then a sum for n >= 3.
  Expansion fmaSqr(const Expansion &a, const Expansion &c) {
    return Values::size(a) == 2 ? fma(a, a, c) : add(sqr(a), c);
  }
  Expansion fromF32(F32 v, unsigned n) {
    return from(v, n, &Routines::dsFromF32, &Routines::expansionFromF32);
  }
  Expansion fromF64(typename Arith::F64 v, unsigned n) {
    return from(v, n, &Routines::dsFromF64, &Routines::expansionFromF64);
  }
  F32 toF32(const Expansion &a) {
    return to(a, &Routines::dsToF32, &Routines::expansionToF32);
  }
  typename Arith::F64 toF64(const Expansion &a) {
    return to(a, &Routines::dsToF64, &Routines::expansionToF64);
  }

private:
  Routines R;

  static typename Arith::Pair pairOf(const Expansion &a) {
    const F32 *x = Values::components(a);
    return {x[0], x[1]};
  }
  static Expansion ofPair(typename Arith::Pair p) {
    Expansion r = Values::sized(2);
    Values::components(r)[0] = p.hi;
    Values::components(r)[1] = p.lo;
    return r;
  }
  template <typename PairOp, typename WideOp>
  Expansion unary(const Expansion &a, PairOp pairOp, WideOp wideOp) {
    const unsigned n = Values::size(a);
    if (n == 2)
      return ofPair((R.*pairOp)(pairOf(a)));
    Expansion r = Values::sized(n);
    (R.*wideOp)(n, Values::components(a), Values::components(r));
    return r;
  }
  template <typename PairOp, typename WideOp>
  Expansion binary(const Expansion &a, const Expansion &b, PairOp pairOp,
                   WideOp wideOp) {
    const unsigned n = Values::size(a);
    assert(Values::size(b) == n && "expansion operands agree on size");
    if (n == 2)
      return ofPair((R.*pairOp)(pairOf(a), pairOf(b)));
    Expansion r = Values::sized(n);
    (R.*wideOp)(n, Values::components(a), Values::components(b),
                Values::components(r));
    return r;
  }
  template <typename V, typename PairOp, typename WideOp>
  Expansion from(V v, unsigned n, PairOp pairOp, WideOp wideOp) {
    if (n == 2)
      return ofPair((R.*pairOp)(v));
    Expansion r = Values::sized(n);
    (R.*wideOp)(n, v, Values::components(r));
    return r;
  }
  template <typename PairOp, typename WideOp>
  auto to(const Expansion &a, PairOp pairOp, WideOp wideOp) {
    const unsigned n = Values::size(a);
    return n == 2 ? (R.*pairOp)(pairOf(a))
                  : (R.*wideOp)(n, Values::components(a));
  }
};

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_PRECISION_EXPANSION_OPS_H
