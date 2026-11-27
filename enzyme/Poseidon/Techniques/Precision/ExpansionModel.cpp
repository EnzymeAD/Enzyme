//===- ExpansionModel.cpp - Host emulation of FP32 expansions -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The routines of ExpansionRoutines.h, which Expansion.cpp emits, on host
// floats: std::fma exactly where the emitter emits llvm.fma.f32. The file is
// compiled without floating-point contraction (CMakeLists.txt), so that no
// build of the compiler fuses a product and a sum the emitted code keeps
// apart.
//
//===----------------------------------------------------------------------===//

#include "ExpansionModel.h"
#include "Core/FPCore.h"
#include "Core/FPNode.h"
#include "Techniques/Precision/ExpansionOps.h"

#include "llvm/ADT/DenseMap.h"

#include <cassert>
#include <cfloat>
#include <cmath>

#if defined(__FAST_MATH__)
#error "ExpansionModel.cpp must be compiled without fast-math"
#endif
static_assert(FLT_EVAL_METHOD == 0,
              "ExpansionModel.cpp needs each float operation rounded to float");

namespace poseidon {

namespace {

struct F2 {
  float hi, lo;
};

// Host floats; the names of the emitted instructions are not used.
struct HostArith {
  using F32 = float;
  using F64 = double;
  using Bool = bool;
  using Pair = F2;

  float add(float a, float b, const char *) { return a + b; }
  double add(double a, double b, const char *) { return a + b; }
  float sub(float a, float b, const char *) { return a - b; }
  double sub(double a, double b, const char *) { return a - b; }
  float mul(float a, float b, const char *) { return a * b; }
  float div(float a, float b, const char *) { return a / b; }
  float neg(float a, const char *) { return -a; }
  // Up to LLVM 22, clang takes fma(x, x, y) for never -0, although a negative
  // sum that underflows is -0, and then rewrites a later b - (s - a) into
  // b + (a - s), which turns a -0 into +0. Read through a volatile, the result
  // is what IEEE arithmetic gives whatever compiler builds the pass.
  static float unanalyzed(float v) {
    volatile float r = v;
    return r;
  }
  float fma(float a, float b, float c, const char *) {
    return unanalyzed(std::fma(a, b, c));
  }
  float fmaCall(float a, float b, float c, const char *) {
    return unanalyzed(std::fma(a, b, c));
  }
  float sqrt(float a, const char *) { return std::sqrt(a); }
  float sqrtCall(float a, const char *) { return std::sqrt(a); }
  float constant(double v) { return (float)v; }
  bool equal(float a, float b, const char *) { return a == b; }
  bool both(bool a, bool b, const char *) { return a && b; }
  float select(bool c, float t, float f, const char *) { return c ? t : f; }
  float truncate(double v, const char *) { return (float)v; }
  double extend(float v, const char *) { return (double)v; }
};

// The model's expansions, for ExpansionOps.
struct HostExpansions {
  using Expansion = FloatExpansion;
  static unsigned size(const FloatExpansion &v) { return v.n; }
  static const float *components(const FloatExpansion &v) { return v.x; }
  static float *components(FloatExpansion &v) { return v.x; }
  static FloatExpansion sized(unsigned n) {
    FloatExpansion c;
    c.n = n;
    return c;
  }
};

using Ops = ExpansionOps<HostArith, HostExpansions>;

} // namespace

static Ops ops() { return Ops(HostArith()); }

FloatExpansion FloatExpansion::of(const float *components, unsigned n) {
  FloatExpansion r;
  r.n = n;
  for (unsigned i = 0; i < n; ++i)
    r.x[i] = components[i];
  return r;
}

FloatExpansion expansionFromF64(double v, unsigned n) {
  return ops().fromF64(v, n);
}
FloatExpansion expansionFromF32(float v, unsigned n) {
  return ops().fromF32(v, n);
}
double expansionToF64(const FloatExpansion &a) { return ops().toF64(a); }
float expansionToF32(const FloatExpansion &a) { return ops().toF32(a); }
FloatExpansion expansionNeg(const FloatExpansion &a) { return ops().neg(a); }
FloatExpansion expansionAdd(const FloatExpansion &a, const FloatExpansion &b) {
  return ops().add(a, b);
}
FloatExpansion expansionSub(const FloatExpansion &a, const FloatExpansion &b) {
  return ops().sub(a, b);
}
FloatExpansion expansionMul(const FloatExpansion &a, const FloatExpansion &b) {
  return ops().mul(a, b);
}
FloatExpansion expansionSqr(const FloatExpansion &a) { return ops().sqr(a); }
FloatExpansion expansionDiv(const FloatExpansion &a, const FloatExpansion &b) {
  return ops().div(a, b);
}
FloatExpansion expansionSqrt(const FloatExpansion &a) { return ops().sqrt(a); }
FloatExpansion expansionFma(const FloatExpansion &a, const FloatExpansion &b,
                            const FloatExpansion &c) {
  return ops().fma(a, b, c);
}
FloatExpansion expansionFmaSqr(const FloatExpansion &a,
                               const FloatExpansion &c) {
  return ops().fmaSqr(a, c);
}

namespace {
// The arithmetic of one change to an n-component expansion: the components of
// every operation of the change it has computed.
class ExpansionArithmetic final : public EmulatedArithmetic {
public:
  explicit ExpansionArithmetic(unsigned n) : n(n) {}
  std::optional<double>
  evaluate(const FPNode &node,
           llvm::function_ref<double(const FPNode &)> valueOf) override;

private:
  unsigned n;
  llvm::DenseMap<const FPNode *, FloatExpansion> components;
};
} // namespace

// As materialized: the result's components are kept for the change's later
// operations.
std::optional<double> ExpansionArithmetic::evaluate(
    const FPNode &node, llvm::function_ref<double(const FPNode &)> valueOf) {
  const bool supported = node.is(fpcore::Neg) || node.is(fpcore::Add) ||
                         node.is(fpcore::Sub) || node.is(fpcore::Mul) ||
                         node.is(fpcore::Div) || node.is(fpcore::Sqrt) ||
                         node.is(fpcore::Fma);
  if (!supported)
    return std::nullopt;
  auto operand = [&](unsigned i) -> FloatExpansion {
    const FPNode *o = node.operands[i];
    auto kept = components.find(o);
    if (kept != components.end())
      return kept->second;
    double v = valueOf(*o);
    return o->hasPrecision(NumberFormat::FP32) ? expansionFromF32((float)v, n)
                                               : expansionFromF64(v, n);
  };
  // A value multiplied by itself is materialized as a square.
  const bool sameFactor =
      node.operands.size() >= 2 && node.operands[0] == node.operands[1];
  FloatExpansion r;
  if (node.is(fpcore::Neg))
    r = expansionNeg(operand(0));
  else if (node.is(fpcore::Add))
    r = expansionAdd(operand(0), operand(1));
  else if (node.is(fpcore::Sub))
    r = expansionSub(operand(0), operand(1));
  else if (node.is(fpcore::Mul))
    r = sameFactor ? expansionSqr(operand(0))
                   : expansionMul(operand(0), operand(1));
  else if (node.is(fpcore::Div))
    r = expansionDiv(operand(0), operand(1));
  else if (node.is(fpcore::Sqrt))
    r = expansionSqrt(operand(0));
  else
    r = sameFactor ? expansionFmaSqr(operand(0), operand(2))
                   : expansionFma(operand(0), operand(1), operand(2));
  components[&node] = r;
  return node.hasPrecision(NumberFormat::FP32) ? (double)expansionToF32(r)
                                               : expansionToF64(r);
}

std::unique_ptr<EmulatedArithmetic> expansionArithmetic(NumberFormat format) {
  const unsigned n = expansionComponents(format);
  assert(n && "not an expansion format");
  return std::make_unique<ExpansionArithmetic>(n);
}

} // namespace poseidon
