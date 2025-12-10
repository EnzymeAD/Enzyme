//=- PoseidonEvaluators.cpp - Expression evaluators for Poseidon ----------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the evaluator classes for floating-point expressions
// in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"

#include <cmath>

#include "PoseidonEvaluators.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonTypes.h"

using namespace llvm;

extern "C" {
cl::opt<bool> FPOptStrictMode(
    "fpopt-strict-mode", cl::init(false), cl::Hidden,
    cl::desc(
        "Discard all FPOpt candidates that produce NaN or inf outputs for any "
        "input point that originally produced finite outputs"));
cl::opt<double> FPOptGeoMeanEps(
    "fpopt-geo-mean-eps", cl::init(0.0), cl::Hidden,
    cl::desc("The offset used in the geometric mean "
             "calculation; if = 0, zeros are replaced with ULPs"));
}

FPEvaluator::FPEvaluator(PTCandidate *pt) {
  if (pt) {
    for (const auto &change : pt->changes) {
      for (auto node : change.nodes) {
        nodePrecisions[node] = change.newType;
      }
    }
  }
}

PrecisionChangeType FPEvaluator::getNodePrecision(const FPNode *node) const {
  // If the node has a new precision from PT, use it
  PrecisionChangeType precType;

  auto it = nodePrecisions.find(node);
  if (it != nodePrecisions.end()) {
    precType = it->second;
  } else {
    // Otherwise, use the node's original precision
    if (node->dtype == "f32") {
      precType = PrecisionChangeType::FP32;
    } else if (node->dtype == "f64") {
      precType = PrecisionChangeType::FP64;
    } else {
      llvm_unreachable(
          ("Operator " + node->op + " has unexpected dtype: " + node->dtype)
              .c_str());
    }
  }

  if (precType != PrecisionChangeType::FP32 &&
      precType != PrecisionChangeType::FP64) {
    llvm_unreachable("Unsupported FP precision");
  }

  return precType;
}

void FPEvaluator::evaluateNode(const FPNode *node,
                               const MapVector<Value *, double> &inputValues) {
  if (cache.find(node) != cache.end())
    return;

  if (isa<FPConst>(node)) {
    double constVal = node->getLowerBound();
    cache.emplace(node, constVal);
    return;
  }

  if (isa<FPLLValue>(node) && inputValues.count(cast<FPLLValue>(node)->value)) {
    double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
    cache.emplace(node, inputValue);
    return;
  }

  if (node->op == "if") {
    evaluateNode(node->operands[0].get(), inputValues);
    double cond = getResult(node->operands[0].get());

    if (cond == 1.0) {
      evaluateNode(node->operands[1].get(), inputValues);
      double then_val = getResult(node->operands[1].get());
      cache.emplace(node, then_val);
    } else {
      evaluateNode(node->operands[2].get(), inputValues);
      double else_val = getResult(node->operands[2].get());
      cache.emplace(node, else_val);
    }
    return;
  } else if (node->op == "and") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 != 1.0) {
      cache.emplace(node, 0.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 != 1.0) {
      cache.emplace(node, 0.0);
      return;
    }
    cache.emplace(node, 1.0);
    return;
  } else if (node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op0 = getResult(node->operands[0].get());
    if (op0 == 1.0) {
      cache.emplace(node, 1.0);
      return;
    }
    evaluateNode(node->operands[1].get(), inputValues);
    double op1 = getResult(node->operands[1].get());
    if (op1 == 1.0) {
      cache.emplace(node, 1.0);
      return;
    }
    cache.emplace(node, 0.0);
    return;
  } else if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues);
    double op = getResult(node->operands[0].get());
    cache.emplace(node, (op == 1.0) ? 0.0 : 1.0);
    return;
  } else if (node->op == "TRUE") {
    cache.emplace(node, 1.0);
    return;
  } else if (node->op == "FALSE") {
    cache.emplace(node, 0.0);
    return;
  }

  PrecisionChangeType nodePrec = getNodePrecision(node);

  for (const auto &operand : node->operands) {
    evaluateNode(operand.get(), inputValues);
  }

  double res = 0.0;

  auto evalUnary = [&](auto doubleFunc, auto floatFunc) -> double {
    double op = getResult(node->operands[0].get());
    if (nodePrec == PrecisionChangeType::FP32)
      return floatFunc(static_cast<float>(op));
    else
      return doubleFunc(op);
  };

  auto evalBinary = [&](auto doubleFunc, auto floatFunc) -> double {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    if (nodePrec == PrecisionChangeType::FP32)
      return floatFunc(static_cast<float>(op0), static_cast<float>(op1));
    else
      return doubleFunc(op0, op1);
  };

  auto evalTernary = [&](auto doubleFunc, auto floatFunc) -> double {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    double op2 = getResult(node->operands[2].get());
    if (nodePrec == PrecisionChangeType::FP32)
      return floatFunc(static_cast<float>(op0), static_cast<float>(op1),
                       static_cast<float>(op2));
    else
      return doubleFunc(op0, op1, op2);
  };

  if (node->op == "neg") {
    double op = getResult(node->operands[0].get());
    res =
        (nodePrec == PrecisionChangeType::FP32) ? -static_cast<float>(op) : -op;
  } else if (node->op == "+") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    res = (nodePrec == PrecisionChangeType::FP32)
              ? static_cast<float>(op0) + static_cast<float>(op1)
              : op0 + op1;
  } else if (node->op == "-") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    res = (nodePrec == PrecisionChangeType::FP32)
              ? static_cast<float>(op0) - static_cast<float>(op1)
              : op0 - op1;
  } else if (node->op == "*") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    res = (nodePrec == PrecisionChangeType::FP32)
              ? static_cast<float>(op0) * static_cast<float>(op1)
              : op0 * op1;
  } else if (node->op == "/") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    res = (nodePrec == PrecisionChangeType::FP32)
              ? static_cast<float>(op0) / static_cast<float>(op1)
              : op0 / op1;
  } else if (node->op == "sin") {
    res = evalUnary(static_cast<double (*)(double)>(std::sin),
                    static_cast<float (*)(float)>(sinf));
  } else if (node->op == "cos") {
    res = evalUnary(static_cast<double (*)(double)>(std::cos),
                    static_cast<float (*)(float)>(cosf));
  } else if (node->op == "tan") {
    res = evalUnary(static_cast<double (*)(double)>(std::tan),
                    static_cast<float (*)(float)>(tanf));
  } else if (node->op == "exp") {
    res = evalUnary(static_cast<double (*)(double)>(std::exp),
                    static_cast<float (*)(float)>(expf));
  } else if (node->op == "expm1") {
    res = evalUnary(static_cast<double (*)(double)>(std::expm1),
                    static_cast<float (*)(float)>(expm1f));
  } else if (node->op == "log") {
    res = evalUnary(static_cast<double (*)(double)>(std::log),
                    static_cast<float (*)(float)>(logf));
  } else if (node->op == "log1p") {
    res = evalUnary(static_cast<double (*)(double)>(std::log1p),
                    static_cast<float (*)(float)>(log1pf));
  } else if (node->op == "sqrt") {
    res = evalUnary(static_cast<double (*)(double)>(std::sqrt),
                    static_cast<float (*)(float)>(sqrtf));
  } else if (node->op == "cbrt") {
    res = evalUnary(static_cast<double (*)(double)>(std::cbrt),
                    static_cast<float (*)(float)>(cbrtf));
  } else if (node->op == "asin") {
    res = evalUnary(static_cast<double (*)(double)>(std::asin),
                    static_cast<float (*)(float)>(asinf));
  } else if (node->op == "acos") {
    res = evalUnary(static_cast<double (*)(double)>(std::acos),
                    static_cast<float (*)(float)>(acosf));
  } else if (node->op == "atan") {
    res = evalUnary(static_cast<double (*)(double)>(std::atan),
                    static_cast<float (*)(float)>(atanf));
  } else if (node->op == "sinh") {
    res = evalUnary(static_cast<double (*)(double)>(std::sinh),
                    static_cast<float (*)(float)>(sinhf));
  } else if (node->op == "cosh") {
    res = evalUnary(static_cast<double (*)(double)>(std::cosh),
                    static_cast<float (*)(float)>(coshf));
  } else if (node->op == "tanh") {
    res = evalUnary(static_cast<double (*)(double)>(std::tanh),
                    static_cast<float (*)(float)>(tanhf));
  } else if (node->op == "asinh") {
    res = evalUnary(static_cast<double (*)(double)>(std::asinh),
                    static_cast<float (*)(float)>(asinhf));
  } else if (node->op == "acosh") {
    res = evalUnary(static_cast<double (*)(double)>(std::acosh),
                    static_cast<float (*)(float)>(acoshf));
  } else if (node->op == "atanh") {
    res = evalUnary(static_cast<double (*)(double)>(std::atanh),
                    static_cast<float (*)(float)>(atanhf));
  } else if (node->op == "ceil") {
    res = evalUnary(static_cast<double (*)(double)>(std::ceil),
                    static_cast<float (*)(float)>(ceilf));
  } else if (node->op == "floor") {
    res = evalUnary(static_cast<double (*)(double)>(std::floor),
                    static_cast<float (*)(float)>(floorf));
  } else if (node->op == "exp2") {
    res = evalUnary(static_cast<double (*)(double)>(std::exp2),
                    static_cast<float (*)(float)>(exp2f));
  } else if (node->op == "log10") {
    res = evalUnary(static_cast<double (*)(double)>(std::log10),
                    static_cast<float (*)(float)>(log10f));
  } else if (node->op == "log2") {
    res = evalUnary(static_cast<double (*)(double)>(std::log2),
                    static_cast<float (*)(float)>(log2f));
  } else if (node->op == "rint") {
    res = evalUnary(static_cast<double (*)(double)>(std::rint),
                    static_cast<float (*)(float)>(rintf));
  } else if (node->op == "round") {
    res = evalUnary(static_cast<double (*)(double)>(std::round),
                    static_cast<float (*)(float)>(roundf));
  } else if (node->op == "trunc") {
    res = evalUnary(static_cast<double (*)(double)>(std::trunc),
                    static_cast<float (*)(float)>(truncf));
  } else if (node->op == "pow") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::pow),
                     static_cast<float (*)(float, float)>(powf));
  } else if (node->op == "fabs") {
    res = evalUnary(static_cast<double (*)(double)>(std::fabs),
                    static_cast<float (*)(float)>(fabsf));
  } else if (node->op == "hypot") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::hypot),
                     static_cast<float (*)(float, float)>(hypotf));
  } else if (node->op == "atan2") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::atan2),
                     static_cast<float (*)(float, float)>(atan2f));
  } else if (node->op == "copysign") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::copysign),
                     static_cast<float (*)(float, float)>(copysignf));
  } else if (node->op == "fmax") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmax),
                     static_cast<float (*)(float, float)>(fmaxf));
  } else if (node->op == "fmin") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmin),
                     static_cast<float (*)(float, float)>(fminf));
  } else if (node->op == "fdim") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fdim),
                     static_cast<float (*)(float, float)>(fdimf));
  } else if (node->op == "fmod") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::fmod),
                     static_cast<float (*)(float, float)>(fmodf));
  } else if (node->op == "remainder") {
    res = evalBinary(static_cast<double (*)(double, double)>(std::remainder),
                     static_cast<float (*)(float, float)>(remainderf));
  } else if (node->op == "fma") {
    res = evalTernary(static_cast<double (*)(double, double, double)>(std::fma),
                      static_cast<float (*)(float, float, float)>(fmaf));
  } else if (node->op == "lgamma") {
    res = evalUnary(static_cast<double (*)(double)>(std::lgamma),
                    static_cast<float (*)(float)>(lgammaf));
  } else if (node->op == "tgamma") {
    res = evalUnary(static_cast<double (*)(double)>(std::tgamma),
                    static_cast<float (*)(float)>(tgammaf));
  } else if (node->op == "==") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) == static_cast<float>(op1)
                      : op0 == op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == "!=") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) != static_cast<float>(op1)
                      : op0 != op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == "<") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) < static_cast<float>(op1)
                      : op0 < op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == ">") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) > static_cast<float>(op1)
                      : op0 > op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == "<=") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) <= static_cast<float>(op1)
                      : op0 <= op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == ">=") {
    double op0 = getResult(node->operands[0].get());
    double op1 = getResult(node->operands[1].get());
    bool result = (nodePrec == PrecisionChangeType::FP32)
                      ? static_cast<float>(op0) >= static_cast<float>(op1)
                      : op0 >= op1;
    res = result ? 1.0 : 0.0;
  } else if (node->op == "PI") {
    res = M_PI;
  } else if (node->op == "E") {
    res = M_E;
  } else if (node->op == "INFINITY") {
    res = INFINITY;
  } else if (node->op == "NAN") {
    res = NAN;
  } else {
    std::string msg = "FPEvaluator: Unexpected operator " + node->op;
    llvm_unreachable(msg.c_str());
  }

  cache.emplace(node, res);
}

double FPEvaluator::getResult(const FPNode *node) const {
  auto it = cache.find(node);
  assert(it != cache.end() && "Node not evaluated yet");
  return it->second;
}

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

MPFREvaluator::MPFREvaluator(unsigned prec, PTCandidate *pt) : prec(prec) {
  if (pt) {
    for (const auto &change : pt->changes) {
      for (auto node : change.nodes) {
        nodeToNewPrec[node] = getMPFRPrec(change.newType);
      }
    }
  }
}

unsigned MPFREvaluator::getNodePrecision(const FPNode *node,
                                         bool groundTruth) const {
  // If trying to evaluate the ground truth, use the current MPFR precision
  if (groundTruth)
    return prec;

  // If the node has a new precision for PT, use it
  auto it = nodeToNewPrec.find(node);
  if (it != nodeToNewPrec.end()) {
    return it->second;
  }

  // Otherwise, use the original precision
  return node->getMPFRPrec();
}

// Compute the expression with MPFR at `prec` precision
// recursively. When operand is a FPConst, use its lower
// bound. When operand is a FPLLValue, get its inputs from
// `inputs`.
void MPFREvaluator::evaluateNode(const FPNode *node,
                                 const MapVector<Value *, double> &inputValues,
                                 bool groundTruth) {
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
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &cond = getResult(node->operands[0].get());
    if (0 == mpfr_cmp_ui(cond, 1)) {
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &then_val = getResult(node->operands[1].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[1].get()).prec));
      mpfr_set(cache.at(node).value, then_val, MPFR_RNDN);
    } else {
      evaluateNode(node->operands[2].get(), inputValues, groundTruth);
      mpfr_t &else_val = getResult(node->operands[2].get());
      cache.emplace(node, CachedValue(cache.at(node->operands[2].get()).prec));
      mpfr_set(cache.at(node).value, else_val, MPFR_RNDN);
    }
    return;
  }

  unsigned nodePrec = getNodePrecision(node, groundTruth);
  cache.emplace(node, CachedValue(nodePrec));
  mpfr_t &res = cache.at(node).value;

  if (node->op == "neg") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_neg(res, op, MPFR_RNDN);
  } else if (node->op == "+") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_add(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "-") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_sub(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "*") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_mul(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "/") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_div(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "sin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sin(res, op, MPFR_RNDN);
  } else if (node->op == "cos") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cos(res, op, MPFR_RNDN);
  } else if (node->op == "tan") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_tan(res, op, MPFR_RNDN);
  } else if (node->op == "asin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_asin(res, op, MPFR_RNDN);
  } else if (node->op == "acos") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_acos(res, op, MPFR_RNDN);
  } else if (node->op == "atan") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_atan(res, op, MPFR_RNDN);
  } else if (node->op == "atan2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_atan2(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "exp") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_exp(res, op, MPFR_RNDN);
  } else if (node->op == "expm1") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_expm1(res, op, MPFR_RNDN);
  } else if (node->op == "log") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log(res, op, MPFR_RNDN);
  } else if (node->op == "log1p") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log1p(res, op, MPFR_RNDN);
  } else if (node->op == "sqrt") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sqrt(res, op, MPFR_RNDN);
  } else if (node->op == "cbrt") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cbrt(res, op, MPFR_RNDN);
  } else if (node->op == "pow") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_pow(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fma") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    evaluateNode(node->operands[2].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_t &op2 = getResult(node->operands[2].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op2, nodePrec, MPFR_RNDN);
    mpfr_fma(res, op0, op1, op2, MPFR_RNDN);
  } else if (node->op == "fabs") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_abs(res, op, MPFR_RNDN);
  } else if (node->op == "hypot") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_hypot(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "asinh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_asinh(res, op, MPFR_RNDN);
  } else if (node->op == "acosh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_acosh(res, op, MPFR_RNDN);
  } else if (node->op == "atanh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_atanh(res, op, MPFR_RNDN);
  } else if (node->op == "sinh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_sinh(res, op, MPFR_RNDN);
  } else if (node->op == "cosh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_cosh(res, op, MPFR_RNDN);
  } else if (node->op == "tanh") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_tanh(res, op, MPFR_RNDN);
  } else if (node->op == "ceil") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_ceil(res, op);
  } else if (node->op == "floor") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_floor(res, op);
  } else if (node->op == "erf") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_erf(res, op, MPFR_RNDN);
  } else if (node->op == "exp2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_exp2(res, op, MPFR_RNDN);
  } else if (node->op == "log10") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log10(res, op, MPFR_RNDN);
  } else if (node->op == "log2") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_log2(res, op, MPFR_RNDN);
  } else if (node->op == "rint") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_rint(res, op, MPFR_RNDN);
  } else if (node->op == "round") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_round(res, op);
  } else if (node->op == "trunc") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_prec_round(op, nodePrec, MPFR_RNDN);
    mpfr_trunc(res, op);
  } else if (node->op == "copysign") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_copysign(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fdim") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_dim(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmod") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_fmod(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "remainder") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_remainder(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmax") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_max(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "fmin") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
    mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
    mpfr_min(res, op0, op1, MPFR_RNDN);
  } else if (node->op == "==") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "!=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 != mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "<") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 > mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == ">") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 < mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "<=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 >= mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == ">=") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 <= mpfr_cmp(op0, op1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "and") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp_ui(op0, 1) && 0 == mpfr_cmp_ui(op1, 1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "or") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    evaluateNode(node->operands[1].get(), inputValues, groundTruth);
    mpfr_t &op0 = getResult(node->operands[0].get());
    mpfr_t &op1 = getResult(node->operands[1].get());
    if (0 == mpfr_cmp_ui(op0, 1) || 0 == mpfr_cmp_ui(op1, 1))
      mpfr_set_ui(res, 1, MPFR_RNDN);
    else
      mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "not") {
    evaluateNode(node->operands[0].get(), inputValues, groundTruth);
    mpfr_t &op = getResult(node->operands[0].get());
    mpfr_set_prec(res, nodePrec);
    if (0 == mpfr_cmp_ui(op, 1))
      mpfr_set_ui(res, 0, MPFR_RNDN);
    else
      mpfr_set_ui(res, 1, MPFR_RNDN);
  } else if (node->op == "TRUE") {
    mpfr_set_ui(res, 1, MPFR_RNDN);
  } else if (node->op == "FALSE") {
    mpfr_set_ui(res, 0, MPFR_RNDN);
  } else if (node->op == "PI") {
    mpfr_const_pi(res, MPFR_RNDN);
  } else if (node->op == "E") {
    mpfr_const_euler(res, MPFR_RNDN);
  } else if (node->op == "INFINITY") {
    mpfr_set_inf(res, 1);
  } else if (node->op == "NAN") {
    mpfr_set_nan(res);
  } else {
    llvm::errs() << "MPFREvaluator: Unexpected operator '" << node->op << "'\n";
    llvm_unreachable("Unexpected operator encountered");
  }
}

mpfr_t &MPFREvaluator::getResult(FPNode *node) {
  assert(cache.count(node) > 0 && "MPFREvaluator: Unexpected unevaluated node");
  return cache.at(node).value;
}

// Emulate computation using native floating-point types
void getFPValues(ArrayRef<FPNode *> outputs,
                 const MapVector<Value *, double> &inputValues,
                 SmallVectorImpl<double> &results, PTCandidate *pt) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  FPEvaluator evaluator(pt);

  for (const auto *output : outputs) {
    evaluator.evaluateNode(output, inputValues);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    results[i] = evaluator.getResult(outputs[i]);
  }
}

// If looking for ground truth, compute a "correct" answer with MPFR.
//   For each sampled input configuration:
//     0. Ignore `FPNode.dtype`.
//     1. Compute the expression with MPFR at `prec` precision
//        by calling `MPFRValueHelper`. When operand is a FPConst, use its
//        lower bound. When operand is a FPLLValue, get its inputs from
//        `inputs`.
//     2. Dynamically extend precisions
//        until the first `groundTruthPrec` bits of significand don't change.
// Otherwise, compute the expression with MPFR at precisions specified within
// `FPNode`s or new precisions specified by `pt`.
void getMPFRValues(ArrayRef<FPNode *> outputs,
                   const MapVector<Value *, double> &inputValues,
                   SmallVectorImpl<double> &results, bool groundTruth,
                   const unsigned groundTruthPrec, PTCandidate *pt) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  if (!groundTruth) {
    MPFREvaluator evaluator(0, pt);
    // if (pt) {
    //   llvm::errs() << "getMPFRValues: PT candidate detected: " << pt->desc
    //                << "\n";
    // } else {
    //   llvm::errs() << "getMPFRValues: emulating original computation\n";
    // }

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues, false);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
    }
    return;
  }

  unsigned curPrec = 64;
  std::vector<mpfr_exp_t> prevResExp(outputs.size(), 0);
  std::vector<char *> prevResStr(outputs.size(), nullptr);
  std::vector<int> prevResSign(outputs.size(), 0);
  std::vector<bool> converged(outputs.size(), false);
  size_t numConverged = 0;

  while (true) {
    MPFREvaluator evaluator(curPrec, nullptr);

    // llvm::errs() << "getMPFRValues: computing ground truth with precision "
    //              << curPrec << "\n";

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues, true);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (converged[i])
        continue;

      mpfr_t &res = evaluator.getResult(outputs[i]);
      int resSign = mpfr_sgn(res);
      mpfr_exp_t resExp;
      char *resStr =
          mpfr_get_str(nullptr, &resExp, 2, groundTruthPrec, res, MPFR_RNDN);

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

    if (curPrec > FPOptMaxMPFRPrec) {
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