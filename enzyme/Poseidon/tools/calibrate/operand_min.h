//===- operand_min.h - smallest operand a reduced-precision product keeps -===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The smallest magnitude of an operand in its narrow format that a product
// keeps rather than reading as zero, measured on the product a calibrator
// runs, at the shape it is timed at. A library or matrix unit may flush the
// format's subnormals, or more, and only on the kernels it picks for products
// of a realistic size.
//
// Each binade from the format's smallest subnormal up to kNormalBinades above
// its smallest normal fills one operand, and the other is a sparse matrix of
// +-2^s with kTerms entries per output. The binade's values carry four
// significant bits, so every product and every partial sum of an output is
// exact in an accumulator of 11 significand bits or more, and the result is
// compared bit for bit with the host's. 2^s puts the smallest nonzero partial
// sum at the accumulator's smallest normal or above: what is measured is the
// operand, not an accumulator that flushes its own subnormals. The floor is the
// lowest binade from which every binade up is kept, in both operands.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TOOLS_CALIBRATE_OPERAND_MIN_H
#define POSEIDON_TOOLS_CALIBRATE_OPERAND_MIN_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace operand_min {

struct Format {
  const char *name;
  int precision;   // significand bits, the leading one included
  int minExponent; // exponent of the smallest normal
};

inline constexpr Format kFormats[] = {
    {"f16", 11, -14}, {"bf16", 8, -126}, {"tf32", 11, -126}, {"f32", 24, -126}};

inline const Format *formatNamed(const char *name) {
  for (const Format &f : kFormats)
    if (!std::strcmp(f.name, name))
      return &f;
  return nullptr;
}

inline constexpr int kTerms = 8;
inline constexpr int kNormalBinades = 10;

inline int subnormalExponent(const Format &f) {
  return f.minExponent - f.precision + 1;
}

class Rng {
public:
  uint64_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return state >> 11;
  }
  double sign() { return (next() >> 20) & 1 ? -1.0 : 1.0; }

private:
  uint64_t state = 0x9e3779b97f4a7c15ull;
};

/// Column k of the sparse operand's row i (operand A) or row k of its column i
/// (operand B), for t < kTerms: distinct for every t.
inline int sparseIndex(int i, int t, int n) {
  return (i + t * (n / kTerms)) % n;
}

/// Whether C = A * B (n x n, row-major), computed by `product`, keeps every
/// operand of binade `e` in A (`side` 0) or in B (`side` 1), the other operand
/// holding +-`other`. With `anchor` nonzero, the binade operand's element
/// (0, 0) is `anchor` instead, for a product that scales each operand by its
/// largest magnitude, and the outputs it reaches are not compared; `offset` is
/// then the binade of the scaled value minus that of the operand.
template <typename Product>
bool keeps(const Format &f, int e, int side, int n, double other, double anchor,
           int offset, Product &&product, FILE *log) {
  const int subnormal = subnormalExponent(f);
  const int raw = e - offset;
  const double unit =
      std::ldexp(1.0, (e - 3 > subnormal ? e - 3 : subnormal) - offset);
  const uint64_t steps = (uint64_t)(std::ldexp(1.0, raw) / unit);
  const size_t nn = (size_t)n * n;
  std::vector<double> X(nn), S(nn, 0.0), C(nn);
  Rng rng;
  for (double &x : X)
    x = rng.sign() *
        (std::ldexp(1.0, raw) + unit * (double)(rng.next() % steps));
  if (anchor != 0.0)
    X[0] = anchor;
  // S's nonzeros: row i of A has them in columns sparseIndex(i, t), column j
  // of B in rows sparseIndex(j, t).
  std::vector<double> s((size_t)n * kTerms);
  for (int i = 0; i < n; ++i)
    for (int t = 0; t < kTerms; ++t) {
      double v = rng.sign() * other;
      s[(size_t)i * kTerms + t] = v;
      int k = sparseIndex(i, t, n);
      if (side == 0)
        S[(size_t)k * n + i] = v;
      else
        S[(size_t)i * n + k] = v;
    }
  if (side == 0)
    product(X.data(), S.data(), C.data());
  else
    product(S.data(), X.data(), C.data());
  size_t lost = 0, compared = 0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      if (anchor != 0.0 && (side == 0 ? i == 0 : j == 0))
        continue;
      double exact = 0.0;
      for (int t = 0; t < kTerms; ++t) {
        if (side == 0)
          exact += X[(size_t)i * n + sparseIndex(j, t, n)] *
                   s[(size_t)j * kTerms + t];
        else
          exact += s[(size_t)i * kTerms + t] *
                   X[(size_t)sparseIndex(i, t, n) * n + j];
      }
      ++compared;
      lost += C[(size_t)i * n + j] != exact;
    }
  if (log)
    std::fprintf(
        log, "[operand-min] %s %c 2^%d: %s (%zu of %zu outputs differ)\n",
        f.name, side ? 'B' : 'A', e, lost ? "lost" : "kept", lost, compared);
  return lost == 0;
}

/// The floor of `f` in `product`, whose accumulator is `acc`: the smallest
/// power of two from which every binade up to kNormalBinades above the
/// smallest normal is kept in both operands; 0 when the top binade itself is
/// not kept, which is no measurement. Every binade is put through the product
/// and logged.
template <typename Product>
double measure(const Format &f, const Format &acc, int n, Product &&product,
               FILE *log, double anchor = 0.0, int offset = 0) {
  const int lo = subnormalExponent(f), hi = f.minExponent + kNormalBinades;
  const int scale = std::max(0, acc.minExponent - lo);
  int floor = hi + 1;
  for (int e = lo; e <= hi; ++e) {
    bool kept = true;
    for (int side = 0; side < 2; ++side)
      kept &= keeps(f, e, side, n, std::ldexp(1.0, scale), anchor, offset,
                    product, log);
    if (!kept)
      floor = hi + 1;
    else if (floor > hi)
      floor = e;
  }
  if (log)
    std::fprintf(
        log,
        "[operand-min] %s, %s accumulator, other operand +-2^%d: %s 2^%d\n",
        f.name, acc.name, scale, floor > hi ? "no binade kept up to" : "floor",
        floor > hi ? hi : floor);
  return floor > hi ? 0.0 : std::ldexp(1.0, floor);
}

} // namespace operand_min

#endif // POSEIDON_TOOLS_CALIBRATE_OPERAND_MIN_H
