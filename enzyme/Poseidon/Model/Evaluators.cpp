//===- Evaluators.cpp - Expression evaluators for Poseidon ----------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Evaluators.h"
#include "FPCore.h"
#include "FPFormat.h"
#include "FPNode.h"
#include "Targets/Target.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <limits>
#include <mpfr.h>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

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
  const bool supported = node->is(fpcore::Neg) || node->is(fpcore::Add) ||
                         node->is(fpcore::Sub) || node->is(fpcore::Mul) ||
                         node->is(fpcore::Div) || node->is(fpcore::Sqrt) ||
                         node->is(fpcore::Fma);
  if (!supported)
    return false;
  auto operand = [&](unsigned i) -> FloatExpansion {
    const FPNode *o = node->operands[i];
    auto kept = components.find(o);
    if (kept != components.end() && nodeChange.at(o) == nodeChange.at(node))
      return kept->second;
    double v = getResult(o);
    return o->hasPrecision(PrecisionChangeType::FP32)
               ? expansionFromF32((float)v, n)
               : expansionFromF64(v, n);
  };
  // A value multiplied by itself is materialized as a square.
  const bool sameFactor =
      node->operands.size() >= 2 && node->operands[0] == node->operands[1];
  FloatExpansion r;
  if (node->is(fpcore::Neg))
    r = expansionNeg(operand(0));
  else if (node->is(fpcore::Add))
    r = expansionAdd(operand(0), operand(1));
  else if (node->is(fpcore::Sub))
    r = expansionSub(operand(0), operand(1));
  else if (node->is(fpcore::Mul))
    r = sameFactor ? expansionSqr(operand(0))
                   : expansionMul(operand(0), operand(1));
  else if (node->is(fpcore::Div))
    r = expansionDiv(operand(0), operand(1));
  else if (node->is(fpcore::Sqrt))
    r = expansionSqrt(operand(0));
  else
    r = sameFactor ? expansionFmaSqr(operand(0), operand(2))
                   : expansionFma(operand(0), operand(1), operand(2));
  components[node] = r;
  cache.try_emplace(node, node->hasPrecision(PrecisionChangeType::FP32)
                              ? (double)expansionToF32(r)
                              : expansionToF64(r));
  return true;
}

PrecisionChangeType FPEvaluator::getNodePrecision(const FPNode *node) const {
  auto it = nodePrecisions.find(node);
  if (it != nodePrecisions.end())
    return it->second;
  const FPFormatInfo *F = node->format;
  assert(F && F->Kind != FPKind::Invalid &&
         "an evaluated node has a format the matrix model has a kind for");
  return F->Precision;
}

// Whether comparison `pred` holds between two values whose order is `order`
// (negative, zero or positive), or which are unordered when it is none.
static bool comparisonHolds(CmpInst::Predicate pred, std::optional<int> order) {
  if (!order)
    return CmpInst::isUnordered(pred);
  switch (pred) {
  case CmpInst::FCMP_OEQ:
    return *order == 0;
  case CmpInst::FCMP_UNE:
    return *order != 0;
  case CmpInst::FCMP_OLT:
    return *order < 0;
  case CmpInst::FCMP_OGT:
    return *order > 0;
  case CmpInst::FCMP_OLE:
    return *order <= 0;
  case CmpInst::FCMP_OGE:
    return *order >= 0;
  default:
    llvm_unreachable("FPCore has no other comparison");
  }
}

template <typename T> static std::optional<int> orderOf(T a, T b) {
  if (std::isnan(a) || std::isnan(b))
    return std::nullopt;
  return a < b ? -1 : a > b ? 1 : 0;
}

bool FPEvaluator::evaluateLogic(const FPNode *node,
                                const MapVector<Value *, double> &inputValues) {
  auto truth = [&](unsigned i) {
    evaluateNode(node->operands[i], inputValues);
    return getResult(node->operands[i]) == 1.0;
  };
  double res;
  if (node->is(fpcore::If)) {
    unsigned arm = truth(0) ? 1 : 2;
    evaluateNode(node->operands[arm], inputValues);
    res = getResult(node->operands[arm]);
  } else if (node->is(fpcore::And)) {
    res = truth(0) && truth(1);
  } else if (node->is(fpcore::Or)) {
    res = truth(0) || truth(1);
  } else if (node->is(fpcore::Not)) {
    res = !truth(0);
  } else if (node->is(fpcore::True) || node->is(fpcore::False)) {
    res = node->is(fpcore::True);
  } else {
    return false;
  }
  cache.try_emplace(node, res);
  return true;
}

double FPEvaluator::evaluateOperation(const FPNode *node,
                                      PrecisionChangeType nodePrec) const {
  const bool useReducedFloat = nodePrec == PrecisionChangeType::FP32 ||
                               nodePrec == PrecisionChangeType::FP16 ||
                               nodePrec == PrecisionChangeType::BF16;
  const FPKind kind = formatOf(nodePrec).Kind;
  // A binary16 or bfloat16 operation is computed in binary32 and rounded.
  auto truncToPrec = [&](float val) -> float {
    if (kind != FPKind::BF16 && kind != FPKind::FP16)
      return val;
    return static_cast<float>(roundToPrec((double)val, kind));
  };
  auto toF = [&](double v) { return truncToPrec(static_cast<float>(v)); };
  auto operand = [&](unsigned i) { return getResult(node->operands[i]); };

  if (auto approx = approxSqrtError.find(node);
      node->is(fpcore::Sqrt) && approx != approxSqrtError.end())
    return approxSqrtModel(toF(operand(0)), approx->second);
  const FPCoreOperator *O = node->op;
  assert(O && "the parser admits only modelled operators");
  if (O->F64Unary)
    return useReducedFloat ? (double)truncToPrec(O->F32Unary(toF(operand(0))))
                           : O->F64Unary(operand(0));
  if (O->F64Binary)
    return useReducedFloat ? (double)truncToPrec(
                                 O->F32Binary(toF(operand(0)), toF(operand(1))))
                           : O->F64Binary(operand(0), operand(1));
  if (node->is(fpcore::Fma))
    return useReducedFloat
               ? (double)truncToPrec(
                     fmaf(toF(operand(0)), toF(operand(1)), toF(operand(2))))
               : std::fma(operand(0), operand(1), operand(2));
  if (O->Predicate != CmpInst::BAD_FCMP_PREDICATE) {
    std::optional<int> order = useReducedFloat
                                   ? orderOf(toF(operand(0)), toF(operand(1)))
                                   : orderOf(operand(0), operand(1));
    return comparisonHolds(O->Predicate, order) ? 1.0 : 0.0;
  }
  assert(O->MPFRConstant && "every other operator is a constant");
  return O->Constant;
}

void FPEvaluator::evaluateNode(const FPNode *node,
                               const MapVector<Value *, double> &inputValues) {
  if (cache.count(node))
    return;
  if (isa<FPConst>(node)) {
    cache.try_emplace(node, node->getLowerBound());
    return;
  }
  if (auto *LL = dyn_cast<FPLLValue>(node);
      LL && inputValues.count(LL->value)) {
    cache.try_emplace(node, inputValues.lookup(LL->value));
    return;
  }
  if (evaluateLogic(node, inputValues))
    return;

  PrecisionChangeType nodePrec = getNodePrecision(node);
  for (const auto &operand : node->operands)
    evaluateNode(operand, inputValues);
  if (nodePrec == PrecisionChangeType::FP80 ||
      nodePrec == PrecisionChangeType::FP128)
    llvm_unreachable("FPEvaluator: FP80/FP128 evaluation not implemented");
  if (const unsigned nExp = expansionComponents(nodePrec))
    if (evaluateExpansionNode(node, nExp))
      return;
  cache.try_emplace(node, evaluateOperation(node, nodePrec));
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
    CachedValue &operator=(CachedValue &&) = delete;
    ~CachedValue();
  };

  std::unordered_map<const FPNode *, CachedValue> cache;
  unsigned prec;

public:
  explicit MPFREvaluator(unsigned prec) : prec(prec) {}

  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues);
  mpfr_t &getResult(const FPNode *node);

private:
  // An if: the value of the arm its condition picks, at that arm's
  // precision.
  void evaluateChoice(const FPNode *node,
                      const MapVector<Value *, double> &inputValues);
  // Any other operation, into `res`.
  void evaluateOperation(const FPNode *node,
                         const MapVector<Value *, double> &inputValues,
                         mpfr_t &res);
};

} // namespace

MPFREvaluator::CachedValue::CachedValue(unsigned prec) : prec(prec) {
  mpfr_init2(value, prec);
  mpfr_set_zero(value, 1);
}

MPFREvaluator::CachedValue::CachedValue(CachedValue &&other) noexcept
    : prec(other.prec) {
  mpfr_init2(value, other.prec);
  mpfr_swap(value, other.value);
}

MPFREvaluator::CachedValue::~CachedValue() { mpfr_clear(value); }

void MPFREvaluator::evaluateNode(
    const FPNode *node, const MapVector<Value *, double> &inputValues) {
  if (cache.count(node))
    return;
  double leaf;
  if (isa<FPConst>(node)) {
    leaf = node->getLowerBound();
  } else if (auto *LL = dyn_cast<FPLLValue>(node);
             LL && inputValues.count(LL->value)) {
    leaf = inputValues.lookup(LL->value);
  } else if (node->is(fpcore::If)) {
    evaluateChoice(node, inputValues);
    return;
  } else {
    cache.emplace(node, CachedValue(prec));
    evaluateOperation(node, inputValues, cache.at(node).value);
    return;
  }
  CachedValue cv(std::numeric_limits<double>::digits);
  mpfr_set_d(cv.value, leaf, MPFR_RNDN);
  cache.emplace(node, std::move(cv));
}

void MPFREvaluator::evaluateChoice(
    const FPNode *node, const MapVector<Value *, double> &inputValues) {
  evaluateNode(node->operands[0], inputValues);
  const FPNode *arm =
      node->operands[0 == mpfr_cmp_ui(getResult(node->operands[0]), 1) ? 1 : 2];
  evaluateNode(arm, inputValues);
  cache.emplace(node, CachedValue(cache.at(arm).prec));
  mpfr_set(cache.at(node).value, getResult(arm), MPFR_RNDN);
}

void MPFREvaluator::evaluateOperation(
    const FPNode *node, const MapVector<Value *, double> &inputValues,
    mpfr_t &res) {
  // The operands, evaluated in order and rounded to this precision.
  auto rounded = [&]() {
    SmallVector<mpfr_ptr, 3> ops;
    for (const auto &operand : node->operands)
      evaluateNode(operand, inputValues);
    for (const auto &operand : node->operands) {
      mpfr_ptr op = getResult(operand);
      mpfr_prec_round(op, prec, MPFR_RNDN);
      ops.push_back(op);
    }
    return ops;
  };
  // The operands as they are, evaluated in order.
  auto operands = [&]() {
    SmallVector<mpfr_ptr, 2> ops;
    for (const auto &operand : node->operands)
      evaluateNode(operand, inputValues);
    for (const auto &operand : node->operands)
      ops.push_back(getResult(operand));
    return ops;
  };
  auto truth = [](mpfr_srcptr x) { return 0 == mpfr_cmp_ui(x, 1); };

  const FPCoreOperator *O = node->op;
  assert(O && "the parser admits only modelled operators");
  if (O->MPFRUnary) {
    O->MPFRUnary(res, rounded()[0], MPFR_RNDN);
  } else if (O->MPFRBinary) {
    SmallVector<mpfr_ptr, 3> ops = rounded();
    O->MPFRBinary(res, ops[0], ops[1], MPFR_RNDN);
  } else if (node->is(fpcore::Fma)) {
    SmallVector<mpfr_ptr, 3> ops = rounded();
    mpfr_fma(res, ops[0], ops[1], ops[2], MPFR_RNDN);
  } else if (O->Predicate != CmpInst::BAD_FCMP_PREDICATE) {
    SmallVector<mpfr_ptr, 2> ops = operands();
    std::optional<int> order;
    if (!mpfr_unordered_p(ops[0], ops[1]))
      order = mpfr_cmp(ops[0], ops[1]);
    mpfr_set_ui(res, comparisonHolds(O->Predicate, order), MPFR_RNDN);
  } else if (O->MPFRConstant) {
    O->MPFRConstant(res, MPFR_RNDN);
  } else if (node->is(fpcore::And) || node->is(fpcore::Or)) {
    SmallVector<mpfr_ptr, 2> ops = operands();
    bool a = truth(ops[0]), b = truth(ops[1]);
    mpfr_set_ui(res, node->is(fpcore::And) ? a && b : a || b, MPFR_RNDN);
  } else if (node->is(fpcore::Not)) {
    mpfr_set_ui(res, !truth(operands()[0]), MPFR_RNDN);
  } else if (node->is(fpcore::True) || node->is(fpcore::False)) {
    mpfr_set_ui(res, node->is(fpcore::True), MPFR_RNDN);
  } else {
    // A conversion is the identity on the value: the reference does not
    // narrow.
    assert((node->is(fpcore::ToBinary32) || node->is(fpcore::ToBinary64)) &&
           "every other operator has a row the evaluator reads");
    mpfr_set(res, operands()[0], MPFR_RNDN);
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

namespace {
// What has to agree between two precisions for a value to count as settled:
// its sign, its exponent and its leading kGroundTruthPrec bits.
struct LeadingBits {
  int sign;
  mpfr_exp_t exponent;
  std::string digits;
  bool operator==(const LeadingBits &other) const {
    return sign == other.sign && exponent == other.exponent &&
           digits == other.digits;
  }
};
} // namespace

static LeadingBits leadingBits(mpfr_srcptr x) {
  mpfr_exp_t exponent;
  char *digits =
      mpfr_get_str(nullptr, &exponent, 2, kGroundTruthPrec, x, MPFR_RNDN);
  LeadingBits bits{mpfr_sgn(x), exponent, digits};
  mpfr_free_str(digits);
  return bits;
}

// The MPFR precision getMPFRValues evaluates at first, in bits.
static constexpr unsigned kFirstMPFRPrecision = 64;

unsigned getMPFRValues(ArrayRef<FPNode *> outputs,
                       const MapVector<Value *, double> &inputValues,
                       SmallVectorImpl<double> &results) {
  assert(!outputs.empty() && "no outputs to evaluate");
  const size_t n = outputs.size();
  results.resize(n);
  SmallVector<std::optional<LeadingBits>, 4> previous(n);
  BitVector settled(n);
  for (unsigned prec = kFirstMPFRPrecision;; prec *= 2) {
    MPFREvaluator evaluator(prec);
    for (const FPNode *output : outputs)
      evaluator.evaluateNode(output, inputValues);
    for (size_t i = 0; i < n; ++i) {
      if (settled[i])
        continue;
      LeadingBits now = leadingBits(evaluator.getResult(outputs[i]));
      if (previous[i] == now)
        settled.set(i);
      else
        previous[i] = std::move(now);
    }
    if (!settled.all() && prec * 2 <= kMaxMPFRPrecision)
      continue;
    // Every value is read at the last precision evaluated.
    for (size_t i = 0; i < n; ++i)
      results[i] = settled[i]
                       ? mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN)
                       : std::numeric_limits<double>::quiet_NaN();
    return n - settled.count();
  }
}

} // namespace poseidon
