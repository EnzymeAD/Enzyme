// RUN: rm -rf %t && mkdir %t
// RUN: %clang++ $(llvm-config --cxxflags) -I%S/../../../Poseidon %S/../../../Poseidon/Core/FPKind.cpp %s $(llvm-config --ldflags --libs support --system-libs) -o %t/round
// RUN: %t/round | FileCheck %s

// roundToPrec rounds with integer arithmetic on the binary64 encoding; it has
// to return APFloat's conversion to the format and back, bit for bit, for
// every double: around each format's exponent range, at and next to every kind
// of tie, through underflow and overflow, and for the special values.

#include "Core/FPKind.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstdint>
#include <random>

using namespace llvm;
using namespace poseidon;

static double viaAPFloat(double x, const fltSemantics &sem) {
  bool losesInfo;
  APFloat ap(x);
  ap.convert(sem, APFloat::rmNearestTiesToEven, &losesInfo);
  ap.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
  return ap.convertToDouble();
}

int main() {
  const std::pair<FPKind, const fltSemantics *> kinds[] = {
      {FPKind::F16, &APFloat::IEEEhalf()},
      {FPKind::BF16, &APFloat::BFloat()},
      {FPKind::TF32, &APFloat::FloatTF32()},
      {FPKind::F32, &APFloat::IEEEsingle()}};
  std::mt19937_64 rng(1);
  for (const auto &[kind, sem] : kinds) {
    const int p = APFloat::semanticsPrecision(*sem);
    const int emin = APFloat::semanticsMinExponent(*sem);
    const int emax = APFloat::semanticsMaxExponent(*sem);
    unsigned long checked = 0, mismatches = 0;
    auto check = [&](uint64_t bits) {
      const double x = bit_cast<double>(bits);
      const double got = roundToPrec(x, kind), want = viaAPFloat(x, *sem);
      ++checked;
      if (bit_cast<uint64_t>(got) != bit_cast<uint64_t>(want) &&
          !(std::isnan(got) && std::isnan(want)))
        ++mismatches;
    };
    for (int i = 0; i < 200000; ++i)
      check(rng());
    for (int e = emin - p - 3; e <= emax + 2; ++e) {
      // The binary64 fraction bits below the format's last kept bit.
      const int drop = 52 - (p - 1) + std::max(0, emin - e);
      for (int i = 0; i < 400; ++i) {
        const uint64_t sign = rng() & (uint64_t(1) << 63);
        const uint64_t head = (sign | uint64_t(e + 1023) << 52);
        const uint64_t frac = rng() & ((uint64_t(1) << 52) - 1);
        check(head | frac);
        if (drop >= 52)
          continue;
        const uint64_t kept = frac >> drop << drop;
        const uint64_t tie = kept | uint64_t(1) << (drop - 1);
        for (uint64_t f : {kept, tie - 1, tie, tie + 1})
          check(head | f);
      }
    }
    const double specials[] = {
        0.0,         INFINITY,     NAN,          0x1p-1074,
        0x1p-1022,   0x1p+1023,    65504.0,      65520.0,
        0x1p-24,     0x1p-25,      0x1.8p-25,    0x1p-149,
        0x1p-150,    0x1.8p-150,   0x1.ffp+127,  0x1.ff8p+127,
        0x1.fffffep+127, 0x1.ffffffp+127, 0x1.fffffffffffffp+1023};
    for (double s : specials) {
      check(bit_cast<uint64_t>(s));
      check(bit_cast<uint64_t>(-s));
    }
    outs() << fpKindName(kind) << ": " << checked << " values, " << mismatches
           << " mismatches\n";
  }
}

// CHECK: f16: {{[0-9]+}} values, 0 mismatches
// CHECK: bf16: {{[0-9]+}} values, 0 mismatches
// CHECK: tf32: {{[0-9]+}} values, 0 mismatches
// CHECK: f32: {{[0-9]+}} values, 0 mismatches
