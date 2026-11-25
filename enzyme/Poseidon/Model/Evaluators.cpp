//===- Evaluators.cpp - Expression evaluators for Poseidon ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Evaluators.h"
#include "FPFormat.h"
#include "FPNode.h"
#include "Targets/Target.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <mpfr.h>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

static constexpr unsigned kMaxMPFRPrec = 1024;
// The significand bits of the reference that have to agree between two
// precisions for it to count as converged: those of a binary64 result.
static constexpr unsigned kGroundTruthPrec = 53;

FPEvaluator::FPEvaluator(ArrayRef<PrecisionChange> changes) {
  for (const auto &[index, change] : enumerate(changes)) {
    for (auto node : change.nodes) {
      nodePrecisions[node] = change.newType;
      nodeChange[node] = index;
      auto *CI = dyn_cast<CallInst>(node->value);
      if (change.newType == PrecisionChangeType::FP32 && CI &&
          approxSqrtF32Of(*CI) != Intrinsic::not_intrinsic)
        approxSqrtError[node] =
            Target::of(*CI->getModule()).approxSqrtF32Error();
    }
  }
}

// The device's approximate root is known only to lie within `bound` of the
// exact root, and its bits differ between GPU generations, so the model
// returns the FP32 value at least `bound` above the exact root: the worst
// error at the operation, in a fixed direction. Where that error meets
// rounding errors of opposite sign, an output can show less than the device's.
static float approxSqrtModel(float x, double bound) {
  double far = std::sqrt((double)x) * (1 + bound);
  float r = (float)far;
  return r < far ? std::nextafter(r, INFINITY) : r;
}

// Evaluates `node` as an operation of an `n`-component expansion change, as
// it is materialized: an operand's components are kept from an operation of
// the same change and otherwise split from the operand's value at its type;
// the result's components are kept for the change's later operations, and the
// node's value is the result restored to its own type. Returns false for an
// operation with no expansion form.
bool FPEvaluator::evaluateExpansionNode(const FPNode *node, unsigned n) {
  const std::string &op = node->op;
  const bool supported = op == "neg" || op == "+" || op == "-" || op == "*" ||
                         op == "/" || op == "sqrt" || op == "fma";
  if (!supported)
    return false;
  auto operand = [&](unsigned i) -> ExpansionLimbs {
    const FPNode *o = node->operands[i].get();
    auto kept = limbs.find(o);
    if (kept != limbs.end() && nodeChange.at(o) == nodeChange.at(node))
      return kept->second;
    double v = getResult(o);
    return o->dtype == "f32" ? expansionFromF32((float)v, n)
                             : expansionFromF64(v, n);
  };
  // A value multiplied by itself is materialized as a square.
  const bool sameFactor =
      node->operands.size() >= 2 && node->operands[0] == node->operands[1];
  ExpansionLimbs r;
  if (op == "neg")
    r = expansionNeg(operand(0));
  else if (op == "+")
    r = expansionAdd(operand(0), operand(1));
  else if (op == "-")
    r = expansionSub(operand(0), operand(1));
  else if (op == "*")
    r = sameFactor ? expansionSqr(operand(0))
                   : expansionMul(operand(0), operand(1));
  else if (op == "/")
    r = expansionDiv(operand(0), operand(1));
  else if (op == "sqrt")
    r = expansionSqrt(operand(0));
  else
    r = sameFactor ? expansionFmaSqr(operand(0), operand(2))
                   : expansionFma(operand(0), operand(1), operand(2));
  limbs[node] = r;
  cache.try_emplace(node, node->dtype == "f32" ? (double)expansionToF32(r)
                                               : expansionToF64(r));
  return true;
}

PrecisionChangeType FPEvaluator::getNodePrecision(const FPNode *node) const {
  auto it = nodePrecisions.find(node);
  if (it != nodePrecisions.end())
    return it->second;
  const FPFormatInfo *F = formatOfDtype(node->dtype);
  if (!F || F->Kind == FPKind::Invalid)
    llvm_unreachable(
        ("Operator " + node->op + " has unexpected dtype: " + node->dtype)
            .c_str());
  return F->Precision;
}

namespace {
struct UnaryFn {
  StringLiteral Op;
  double (*F64)(double);
  float (*F32)(float);
};
struct BinaryFn {
  StringLiteral Op;
  double (*F64)(double, double);
  float (*F32)(float, float);
};
struct ConstantOp {
  StringLiteral Op;
  double Value;
};
} // namespace

static double identity(double x) { return x; }
static float identityF(float x) { return x; }

// The functions an operator is evaluated with in binary64 and in binary32. A
// conversion is the identity at the precision of its node, its destination.
static constexpr UnaryFn kUnaryFns[] = {
    {"neg", [](double x) { return -x; }, [](float x) { return -x; }},
    {"sin", std::sin, sinf},
    {"cos", std::cos, cosf},
    {"tan", std::tan, tanf},
    {"exp", std::exp, expf},
    {"expm1", std::expm1, expm1f},
    {"log", std::log, logf},
    {"log1p", std::log1p, log1pf},
    {"sqrt", std::sqrt, sqrtf},
    {"cbrt", std::cbrt, cbrtf},
    {"asin", std::asin, asinf},
    {"acos", std::acos, acosf},
    {"atan", std::atan, atanf},
    {"sinh", std::sinh, sinhf},
    {"cosh", std::cosh, coshf},
    {"tanh", std::tanh, tanhf},
    {"asinh", std::asinh, asinhf},
    {"acosh", std::acosh, acoshf},
    {"atanh", std::atanh, atanhf},
    {"ceil", std::ceil, ceilf},
    {"floor", std::floor, floorf},
    {"exp2", std::exp2, exp2f},
    {"log10", std::log10, log10f},
    {"log2", std::log2, log2f},
    {"rint", std::rint, rintf},
    {"round", std::round, roundf},
    {"trunc", std::trunc, truncf},
    {"fabs", std::fabs, fabsf},
    {"erf", std::erf, erff},
    {"lgamma", std::lgamma, lgammaf},
    {"tgamma", std::tgamma, tgammaf},
    {"binary64->binary32", identity, identityF},
    {"binary32->binary64", identity, identityF},
};

static constexpr BinaryFn kBinaryFns[] = {
    {"+", [](double a, double b) { return a + b; },
     [](float a, float b) { return a + b; }},
    {"-", [](double a, double b) { return a - b; },
     [](float a, float b) { return a - b; }},
    {"*", [](double a, double b) { return a * b; },
     [](float a, float b) { return a * b; }},
    {"/", [](double a, double b) { return a / b; },
     [](float a, float b) { return a / b; }},
    {"pow", std::pow, powf},
    {"hypot", std::hypot, hypotf},
    {"atan2", std::atan2, atan2f},
    {"copysign", std::copysign, copysignf},
    {"fmax", std::fmax, fmaxf},
    {"fmin", std::fmin, fminf},
    {"fdim", std::fdim, fdimf},
    {"fmod", std::fmod, fmodf},
    {"remainder", std::remainder, remainderf},
};

static constexpr StringLiteral kComparisons[] = {"==", "!=", "<",
                                                 ">",  "<=", ">="};

// The comparison `op` of `a` and `b`, as C compares them.
template <typename T> static bool compare(StringRef op, T a, T b) {
  return StringSwitch<bool>(op)
      .Case("==", a == b)
      .Case("!=", a != b)
      .Case("<", a < b)
      .Case(">", a > b)
      .Case("<=", a <= b)
      .Case(">=", a >= b);
}

static constexpr ConstantOp kConstants[] = {
    {"PI", M_PI},
    {"E", M_E},
    {"INFINITY", INFINITY},
    {"NAN", NAN},
};

template <typename Row, size_t N>
static const Row *findOp(const Row (&Table)[N], StringRef Op) {
  const Row *R = find_if(Table, [&](const Row &R) { return R.Op == Op; });
  return R == std::end(Table) ? nullptr : R;
}

void FPEvaluator::evaluateNode(const FPNode *node,
                               const MapVector<Value *, double> &inputValues) {
  if (cache.find(node) != cache.end())
    return;

  if (isa<FPConst>(node)) {
    double constVal = node->getLowerBound();
    cache.try_emplace(node, constVal);
    return;
  }

  if (isa<FPLLValue>(node) && inputValues.count(cast<FPLLValue>(node)->value)) {
    double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
    cache.try_emplace(node, inputValue);
    return;
  }

  if (node->op == "if") {
    evaluateNode(node->operands[0].get(), inputValues);
    double cond = getResult(node->operands[0].get());

    if (cond == 1.0) {
      evaluateNode(node->operands[1].get(), inputValues);
      double then_val = getResult(node->operands[1].get());
      cache.try_emplace(node, then_val);
    } else {
      evaluateNode(node->operands[2].get(), inputValues);
      double else_val = getResult(node->operands[2].get());
      cache.try_emplace(node, else_val);
    }
    return;
  }
  if (node->op == "and") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 != 1.0) {
      cache.try_emplace(node, 0.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 != 1.0) {
      cache.try_emplace(node, 0.0);
      return;
    }
    cache.try_emplace(node, 1.0);
    return;
  }
  if (node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 == 1.0) {
      cache.try_emplace(node, 1.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 == 1.0) {
      cache.try_emplace(node, 1.0);
      return;
    }
    cache.try_emplace(node, 0.0);
    return;
  }
  if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op = getResult(node->operands[0].get());
    cache.try_emplace(node, (op == 1.0) ? 0.0 : 1.0);
    return;
  }
  if (node->op == "TRUE") {
    cache.try_emplace(node, 1.0);
    return;
  }
  if (node->op == "FALSE") {
    cache.try_emplace(node, 0.0);
    return;
  }

  PrecisionChangeType nodePrec = getNodePrecision(node);

  for (const auto &operand : node->operands) {
    evaluateNode(operand.get(), inputValues);
  }

  if (nodePrec == PrecisionChangeType::FP80 ||
      nodePrec == PrecisionChangeType::FP128)
    llvm_unreachable("FPEvaluator: FP80/FP128 evaluation not implemented");

  if (const unsigned nExp = expansionComponents(nodePrec))
    if (evaluateExpansionNode(node, nExp))
      return;

  double res = 0.0;

  bool useReducedFloat = (nodePrec == PrecisionChangeType::FP32 ||
                          nodePrec == PrecisionChangeType::FP16 ||
                          nodePrec == PrecisionChangeType::BF16);

  auto truncToPrec = [&](float val) -> float {
    FPKind k = formatOf(nodePrec).Kind;
    if (k != FPKind::BF16 && k != FPKind::F16)
      return val;
    return static_cast<float>(roundToPrec((double)val, k));
  };

  auto toF = [&](double v) -> float {
    return truncToPrec(static_cast<float>(v));
  };

  auto operand = [&](unsigned i) { return getResult(node->operands[i].get()); };
  auto approx = approxSqrtError.find(node);
  if (node->op == "sqrt" && approx != approxSqrtError.end()) {
    res = approxSqrtModel(toF(operand(0)), approx->second);
  } else if (const UnaryFn *F = findOp(kUnaryFns, node->op)) {
    res = useReducedFloat ? (double)truncToPrec(F->F32(toF(operand(0))))
                          : F->F64(operand(0));
  } else if (const BinaryFn *F = findOp(kBinaryFns, node->op)) {
    res = useReducedFloat
              ? (double)truncToPrec(F->F32(toF(operand(0)), toF(operand(1))))
              : F->F64(operand(0), operand(1));
  } else if (node->op == "fma") {
    res = useReducedFloat
              ? (double)truncToPrec(
                    fmaf(toF(operand(0)), toF(operand(1)), toF(operand(2))))
              : std::fma(operand(0), operand(1), operand(2));
  } else if (is_contained(kComparisons, node->op)) {
    bool r = useReducedFloat
                 ? compare(node->op, toF(operand(0)), toF(operand(1)))
                 : compare(node->op, operand(0), operand(1));
    res = r ? 1.0 : 0.0;
  } else if (const ConstantOp *C = findOp(kConstants, node->op)) {
    res = C->Value;
  } else {
    std::string msg = "FPEvaluator: Unexpected operator " + node->op;
    llvm_unreachable(msg.c_str());
  }

  cache.try_emplace(node, res);
}

double FPEvaluator::getResult(const FPNode *node) const {
  auto it = cache.find(node);
  assert(it != cache.end() && "Node not evaluated yet");
  return it->second;
}

namespace {

// Evaluates an expression in MPFR at one precision: every operation rounds its
// operands to it and computes its result at it.
class MPFREvaluator {
  struct CachedValue {
    mpfr_t value;
    unsigned prec;

    explicit CachedValue(unsigned prec);
    CachedValue(const CachedValue &) = delete;
    CachedValue &operator=(const CachedValue &) = delete;
    CachedValue(CachedValue &&other) noexcept;
    CachedValue &operator=(CachedValue &&other) noexcept;
    ~CachedValue();
  };

  std::unordered_map<const FPNode *, CachedValue> cache;
  unsigned prec;

public:
  explicit MPFREvaluator(unsigned prec) : prec(prec) {}

  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues);
  mpfr_t &getResult(const FPNode *node);
};

using MPFRUnary = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
using MPFRBinary = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);

struct MPFRUnaryFn {
  StringLiteral Op;
  MPFRUnary Fn;
};
struct MPFRBinaryFn {
  StringLiteral Op;
  MPFRBinary Fn;
};

} // namespace

// lgamma is the logarithm of |Gamma|, as C's lgamma.
static constexpr MPFRUnaryFn kMPFRUnaryFns[] = {
    {"neg", mpfr_neg},
    {"sin", mpfr_sin},
    {"cos", mpfr_cos},
    {"tan", mpfr_tan},
    {"asin", mpfr_asin},
    {"acos", mpfr_acos},
    {"atan", mpfr_atan},
    {"exp", mpfr_exp},
    {"expm1", mpfr_expm1},
    {"log", mpfr_log},
    {"log1p", mpfr_log1p},
    {"sqrt", mpfr_sqrt},
    {"cbrt", mpfr_cbrt},
    {"fabs", mpfr_abs},
    {"asinh", mpfr_asinh},
    {"acosh", mpfr_acosh},
    {"atanh", mpfr_atanh},
    {"sinh", mpfr_sinh},
    {"cosh", mpfr_cosh},
    {"tanh", mpfr_tanh},
    {"ceil",
     [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_ceil(r, x); }},
    {"floor",
     [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_floor(r, x); }},
    {"erf", mpfr_erf},
    {"exp2", mpfr_exp2},
    {"log10", mpfr_log10},
    {"log2", mpfr_log2},
    {"rint", mpfr_rint},
    {"round",
     [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_round(r, x); }},
    {"trunc",
     [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_trunc(r, x); }},
    {"lgamma",
     [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t rnd) {
       int sign;
       return mpfr_lgamma(r, &sign, x, rnd);
     }},
    {"tgamma", mpfr_gamma},
};

static constexpr MPFRBinaryFn kMPFRBinaryFns[] = {
    {"+", mpfr_add},
    {"-", mpfr_sub},
    {"*", mpfr_mul},
    {"/", mpfr_div},
    {"atan2", mpfr_atan2},
    {"pow", mpfr_pow},
    {"hypot", mpfr_hypot},
    {"copysign", mpfr_copysign},
    {"fdim", mpfr_dim},
    {"fmod", mpfr_fmod},
    {"remainder", mpfr_remainder},
    {"fmax", mpfr_max},
    {"fmin", mpfr_min},
};

MPFREvaluator::CachedValue::CachedValue(unsigned prec) : prec(prec) {
  mpfr_init2(value, prec);
  mpfr_set_zero(value, 1);
}

MPFREvaluator::CachedValue::CachedValue(CachedValue &&other) noexcept
    : prec(other.prec) {
  mpfr_init2(value, other.prec);
  mpfr_swap(value, other.value);
}

MPFREvaluator::CachedValue &
MPFREvaluator::CachedValue::operator=(CachedValue &&other) noexcept {
  if (this != &other) {
    mpfr_set_prec(value, other.prec);
    prec = other.prec;
    mpfr_swap(value, other.value);
  }
  return *this;
}

MPFREvaluator::CachedValue::~CachedValue() { mpfr_clear(value); }

void MPFREvaluator::evaluateNode(
    const FPNode *node, const MapVector<Value *, double> &inputValues) {
  if (cache.find(node) != cache.end())
    return;

  if (isa<FPConst>(node)) {
    double constVal = node->getLowerBound();
    CachedValue cv(53);
    mpfr_set_d(cv.value, constVal, MPFR_RNDN);
    cache.emplace(node, CachedValue(std::move(cv)));
    return;
  }

  if (isa<FPLLValue>(node) && inputValues.count(cast<FPLLValue>(node)->value)) {
    double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
    CachedValue cv(53);
    mpfr_set_d(cv.value, inputValue, MPFR_RNDN);
    cache.emplace(node, std::move(cv));
    return;
  }

  if (node->op == "if") {
    evaluateNode(node->operands[0].get(), inputValues);
    mpfr_t &cond = getResult(node->operands[0].get());
    if (0 == mpfr_cmp_ui(cond, 1)) {
      evaluateNode(node->operands[1].get(), inputValues);
      mpfr_t &then_val = getResult(node->operands[1].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[1].get()).prec));
      mpfr_set(cache.at(node).value, then_val, MPFR_RNDN);
    } else {
      evaluateNode(node->operands[2].get(), inputValues);
      mpfr_t &else_val = getResult(node->operands[2].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[2].get()).prec));
      mpfr_set(cache.at(node).value, else_val, MPFR_RNDN);
    }
    return;
  }

  unsigned nodePrec = prec;
  cache.emplace(node, CachedValue(nodePrec));
  mpfr_t &res = cache.at(node).value;

  // The operands, evaluated in order and rounded to this precision.
  auto rounded = [&]() {
    SmallVector<mpfr_ptr, 3> ops;
    for (const auto &operand : node->operands)
      evaluateNode(operand.get(), inputValues);
    for (const auto &operand : node->operands) {
      mpfr_ptr op = getResult(operand.get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      ops.push_back(op);
    }
    return ops;
  };

  if (const MPFRUnaryFn *F = findOp(kMPFRUnaryFns, node->op)) {
    F->Fn(res, rounded()[0], MPFR_RNDN);
  } else if (const MPFRBinaryFn *F = findOp(kMPFRBinaryFns, node->op)) {
    SmallVector<mpfr_ptr, 3> ops = rounded();
    F->Fn(res, ops[0], ops[1], MPFR_RNDN);
  } else if (node->op == "fma") {
    SmallVector<mpfr_ptr, 3> ops = rounded();
    mpfr_fma(res, ops[0], ops[1], ops[2], MPFR_RNDN);
  } else if (is_contained(kComparisons, node->op)) {
    evaluateNode(node->operands[0].get(), inputValues);
    evaluateNode(node->operands[1].get(), inputValues);
    int order = mpfr_cmp(getResult(node->operands[0].get()),
                         getResult(node->operands[1].get()));
    mpfr_set_ui(res, compare(node->op, order, 0), MPFR_RNDN);
  } else if (node->op == "and" || node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues);
    evaluateNode(node->operands[1].get(), inputValues);
    bool a = 0 == mpfr_cmp_ui(getResult(node->operands[0].get()), 1);
    bool b = 0 == mpfr_cmp_ui(getResult(node->operands[1].get()), 1);
    mpfr_set_ui(res, node->op == "and" ? a && b : a || b, MPFR_RNDN);
  } else if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_set_prec(res, nodePrec);
    mpfr_set_ui(res, 0 == mpfr_cmp_ui(op, 1) ? 0 : 1, MPFR_RNDN);
  } else if (node->op == "TRUE") {
    mpfr_set_ui(res, 1, MPFR_RNDN);
  } else if (node->op == "FALSE") {
    mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "PI") {
    mpfr_const_pi(res, MPFR_RNDN);
  } else if (node->op == "E") {
    mpfr_set_ui(res, 1, MPFR_RNDN);
    mpfr_exp(res, res, MPFR_RNDN);
  } else if (node->op == "INFINITY") {
    mpfr_set_inf(res, 1);
  } else if (node->op == "NAN") {
    mpfr_set_nan(res);
  } else if (node->op == "binary64->binary32" ||
             node->op == "binary32->binary64") {
    // Identity on the value: the ground truth does not narrow.
    evaluateNode(node->operands[0].get(), inputValues);
    mpfr_set_prec(res, nodePrec);
    mpfr_set(res, getResult(node->operands[0].get()), MPFR_RNDN);
  } else {
    llvm::errs() << "MPFREvaluator: Unexpected operator '" << node->op << "'\n";
    llvm_unreachable("Unexpected operator encountered");
  }
}

mpfr_t &MPFREvaluator::getResult(const FPNode *node) {
  assert(cache.count(node) > 0 && "MPFREvaluator: Unexpected unevaluated node");
  return cache.at(node).value;
}

void getFPValues(ArrayRef<FPNode *> outputs,
                 const MapVector<Value *, double> &inputValues,
                 SmallVectorImpl<double> &results,
                 ArrayRef<PrecisionChange> changes) {
  assert(!outputs.empty() && "no outputs to evaluate");
  results.resize(outputs.size());

  FPEvaluator evaluator(changes);

  for (const auto *output : outputs) {
    evaluator.evaluateNode(output, inputValues);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    results[i] = evaluator.getResult(outputs[i]);
  }
}

void getMPFRValues(ArrayRef<FPNode *> outputs,
                   const MapVector<Value *, double> &inputValues,
                   SmallVectorImpl<double> &results) {
  assert(!outputs.empty() && "no outputs to evaluate");
  results.resize(outputs.size());

  unsigned curPrec = 64;
  std::vector<mpfr_exp_t> prevResExp(outputs.size(), 0);
  std::vector<char *> prevResStr(outputs.size(), nullptr);
  std::vector<int> prevResSign(outputs.size(), 0);
  BitVector converged(outputs.size());
  size_t numConverged = 0;

  while (true) {
    MPFREvaluator evaluator(curPrec);

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (converged[i])
        continue;

      mpfr_t &res = evaluator.getResult(outputs[i]);
      int resSign = mpfr_sgn(res);
      mpfr_exp_t resExp;
      char *resStr =
          mpfr_get_str(nullptr, &resExp, 2, kGroundTruthPrec, res, MPFR_RNDN);

      if (prevResStr[i] != nullptr && resSign == prevResSign[i] &&
          resExp == prevResExp[i] && strcmp(resStr, prevResStr[i]) == 0) {
        converged[i] = true;
        numConverged++;
        mpfr_free_str(resStr);
        mpfr_free_str(prevResStr[i]);
        prevResStr[i] = nullptr;
        continue;
      }

      if (prevResStr[i]) {
        mpfr_free_str(prevResStr[i]);
      }
      prevResStr[i] = resStr;
      prevResExp[i] = resExp;
      prevResSign[i] = resSign;
    }

    if (numConverged == outputs.size()) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
      }
      break;
    }

    curPrec *= 2;

    if (curPrec > kMaxMPFRPrec) {
      llvm::errs() << "getMPFRValues: MPFR precision limit reached for some "
                      "outputs, returning NaN\n";
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!converged[i]) {
          mpfr_free_str(prevResStr[i]);
          results[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
          results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
        }
      }
      return;
    }
  }
}

} // namespace poseidon
