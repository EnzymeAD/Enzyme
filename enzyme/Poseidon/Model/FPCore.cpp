//===- FPCore.cpp - FP expressions as FPCore and as C ---------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPCore.h"
#include "FPFormat.h"
#include "Targets/MathLibrary.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <regex>
#include <string>

using namespace llvm;

namespace poseidon {

// The items of `S` separated by spaces outside parentheses, from its first
// non-space character on; each further space separates another item.
static SmallVector<StringRef, 4> splitTopLevel(StringRef S) {
  SmallVector<StringRef, 4> Items;
  S = S.ltrim(' ');
  int Depth = 0;
  size_t Start = 0;
  for (size_t I = 0; I < S.size(); ++I) {
    if (S[I] == '(') {
      ++Depth;
    } else if (S[I] == ')') {
      --Depth;
    } else if (S[I] == ' ' && Depth == 0) {
      Items.push_back(S.slice(Start, I));
      Start = I + 1;
    }
  }
  if (Start < S.size())
    Items.push_back(S.drop_front(Start));
  return Items;
}

using MPFRUnary = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
using MPFRBinary = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
using MPFRConstant = int (*)(mpfr_ptr, mpfr_rnd_t);

// A math function: its intrinsic (none for a library call), and its binary64,
// binary32 and MPFR forms.
static constexpr FPCoreOperator unary(StringLiteral Name, Intrinsic::ID ID,
                                      double (*F64)(double),
                                      float (*F32)(float), MPFRUnary MPFR) {
  FPCoreOperator O{Name, 1};
  O.Intrinsic = ID;
  O.LibraryCall = ID == Intrinsic::not_intrinsic;
  O.F64Unary = F64;
  O.F32Unary = F32;
  O.MPFRUnary = MPFR;
  return O;
}

static constexpr FPCoreOperator binary(StringLiteral Name, Intrinsic::ID ID,
                                       double (*F64)(double, double),
                                       float (*F32)(float, float),
                                       MPFRBinary MPFR) {
  FPCoreOperator O{Name, 2};
  O.Intrinsic = ID;
  O.LibraryCall = ID == Intrinsic::not_intrinsic;
  O.F64Binary = F64;
  O.F32Binary = F32;
  O.MPFRBinary = MPFR;
  return O;
}

// An operation the lowering emits by name, with its evaluation.
static constexpr FPCoreOperator arithmetic(StringLiteral Name,
                                           double (*F64)(double),
                                           float (*F32)(float),
                                           MPFRUnary MPFR) {
  FPCoreOperator O = unary(Name, Intrinsic::not_intrinsic, F64, F32, MPFR);
  O.LibraryCall = false;
  return O;
}

static constexpr FPCoreOperator arithmetic(StringLiteral Name,
                                           double (*F64)(double, double),
                                           float (*F32)(float, float),
                                           MPFRBinary MPFR) {
  FPCoreOperator O = binary(Name, Intrinsic::not_intrinsic, F64, F32, MPFR);
  O.LibraryCall = false;
  return O;
}

static constexpr FPCoreOperator comparison(StringLiteral Name,
                                           CmpInst::Predicate Predicate,
                                           StringLiteral LoweredName) {
  FPCoreOperator O{Name, 2, /*Boolean=*/true};
  O.Predicate = Predicate;
  O.LoweredName = LoweredName;
  return O;
}

static constexpr FPCoreOperator constant(StringLiteral Name, double Value,
                                         MPFRConstant MPFR) {
  FPCoreOperator O{Name, 0};
  O.Constant = Value;
  O.MPFRConstant = MPFR;
  return O;
}

// An operator every consumer treats by name.
static constexpr FPCoreOperator special(StringLiteral Name, unsigned Arity,
                                        bool Boolean = false) {
  return FPCoreOperator{Name, Arity, Boolean};
}

static double identity(double x) { return x; }
static float identityF(float x) { return x; }

// Herbie's operators, as the parser, the evaluators and the lowering model
// them. A conversion is the identity at the precision of its node, its
// destination; lgamma is the logarithm of |Gamma|, as C's lgamma.
static constexpr FPCoreOperator kOperators[] = {
    arithmetic(
        "neg", [](double x) { return -x; }, [](float x) { return -x; },
        mpfr_neg),
    arithmetic(
        "+", [](double a, double b) { return a + b; },
        [](float a, float b) { return a + b; }, mpfr_add),
    arithmetic(
        "-", [](double a, double b) { return a - b; },
        [](float a, float b) { return a - b; }, mpfr_sub),
    arithmetic(
        "*", [](double a, double b) { return a * b; },
        [](float a, float b) { return a * b; }, mpfr_mul),
    arithmetic(
        "/", [](double a, double b) { return a / b; },
        [](float a, float b) { return a / b; }, mpfr_div),
    arithmetic("pow", std::pow, powf, mpfr_pow),
    unary("sin", Intrinsic::sin, std::sin, sinf, mpfr_sin),
    unary("cos", Intrinsic::cos, std::cos, cosf, mpfr_cos),
    unary("tan", Intrinsic::tan, std::tan, tanf, mpfr_tan),
    unary("exp", Intrinsic::exp, std::exp, expf, mpfr_exp),
    unary("log", Intrinsic::log, std::log, logf, mpfr_log),
    unary("sqrt", Intrinsic::sqrt, std::sqrt, sqrtf, mpfr_sqrt),
    unary("fabs", Intrinsic::fabs, std::fabs, fabsf, mpfr_abs),
    unary("asin", Intrinsic::asin, std::asin, asinf, mpfr_asin),
    unary("acos", Intrinsic::acos, std::acos, acosf, mpfr_acos),
    unary("atan", Intrinsic::atan, std::atan, atanf, mpfr_atan),
    unary("sinh", Intrinsic::sinh, std::sinh, sinhf, mpfr_sinh),
    unary("cosh", Intrinsic::cosh, std::cosh, coshf, mpfr_cosh),
    unary("tanh", Intrinsic::tanh, std::tanh, tanhf, mpfr_tanh),
    unary(
        "ceil", Intrinsic::ceil, std::ceil, ceilf,
        [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_ceil(r, x); }),
    unary(
        "floor", Intrinsic::floor, std::floor, floorf,
        [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_floor(r, x); }),
    unary("exp2", Intrinsic::exp2, std::exp2, exp2f, mpfr_exp2),
    unary("log10", Intrinsic::log10, std::log10, log10f, mpfr_log10),
    unary("log2", Intrinsic::log2, std::log2, log2f, mpfr_log2),
    unary("rint", Intrinsic::rint, std::rint, rintf, mpfr_rint),
    unary(
        "round", Intrinsic::round, std::round, roundf,
        [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_round(r, x); }),
    unary(
        "trunc", Intrinsic::trunc, std::trunc, truncf,
        [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t) { return mpfr_trunc(r, x); }),
    unary("expm1", Intrinsic::not_intrinsic, std::expm1, expm1f, mpfr_expm1),
    unary("log1p", Intrinsic::not_intrinsic, std::log1p, log1pf, mpfr_log1p),
    unary("cbrt", Intrinsic::not_intrinsic, std::cbrt, cbrtf, mpfr_cbrt),
    unary("asinh", Intrinsic::not_intrinsic, std::asinh, asinhf, mpfr_asinh),
    unary("acosh", Intrinsic::not_intrinsic, std::acosh, acoshf, mpfr_acosh),
    unary("atanh", Intrinsic::not_intrinsic, std::atanh, atanhf, mpfr_atanh),
    unary("erf", Intrinsic::not_intrinsic, std::erf, erff, mpfr_erf),
    unary("lgamma", Intrinsic::not_intrinsic, std::lgamma, lgammaf,
          [](mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t rnd) {
            int sign;
            return mpfr_lgamma(r, &sign, x, rnd);
          }),
    unary("tgamma", Intrinsic::not_intrinsic, std::tgamma, tgammaf, mpfr_gamma),
    binary("fmin", Intrinsic::minnum, std::fmin, fminf, mpfr_min),
    binary("fmax", Intrinsic::maxnum, std::fmax, fmaxf, mpfr_max),
    binary("atan2", Intrinsic::atan2, std::atan2, atan2f, mpfr_atan2),
    binary("copysign", Intrinsic::copysign, std::copysign, copysignf,
           mpfr_copysign),
    binary("hypot", Intrinsic::not_intrinsic, std::hypot, hypotf, mpfr_hypot),
    binary("fdim", Intrinsic::not_intrinsic, std::fdim, fdimf, mpfr_dim),
    binary("fmod", Intrinsic::not_intrinsic, std::fmod, fmodf, mpfr_fmod),
    binary("remainder", Intrinsic::not_intrinsic, std::remainder, remainderf,
           mpfr_remainder),
    special("fma", 3),
    // IEEE comparisons, as FPCore's: with a NaN operand only != holds.
    comparison("==", CmpInst::FCMP_OEQ, "eq"),
    comparison("!=", CmpInst::FCMP_UNE, "ne"),
    comparison("<", CmpInst::FCMP_OLT, "lt"),
    comparison(">", CmpInst::FCMP_OGT, "gt"),
    comparison("<=", CmpInst::FCMP_OLE, "le"),
    comparison(">=", CmpInst::FCMP_OGE, "ge"),
    constant("PI", M_PI, mpfr_const_pi),
    constant("E", M_E,
             [](mpfr_ptr r, mpfr_rnd_t rnd) {
               mpfr_set_ui(r, 1, rnd);
               return mpfr_exp(r, r, rnd);
             }),
    constant("INFINITY", INFINITY,
             [](mpfr_ptr r, mpfr_rnd_t) {
               mpfr_set_inf(r, 1);
               return 0;
             }),
    constant("NAN", NAN,
             [](mpfr_ptr r, mpfr_rnd_t) {
               mpfr_set_nan(r);
               return 0;
             }),
    arithmetic("binary64->binary32", identity, identityF, nullptr),
    arithmetic("binary32->binary64", identity, identityF, nullptr),
    special("and", 2, /*Boolean=*/true),
    special("or", 2, /*Boolean=*/true),
    special("not", 1, /*Boolean=*/true),
    special("TRUE", 0, /*Boolean=*/true),
    special("FALSE", 0, /*Boolean=*/true),
    special("if", 3),
    special("array", 0),
};

const FPCoreOperator *fpcoreOperator(StringRef Name) {
  static const StringMap<const FPCoreOperator *> ByName = [] {
    StringMap<const FPCoreOperator *> Map;
    for (const FPCoreOperator &O : kOperators)
      Map[O.Name] = &O;
    return Map;
  }();
  return ByName.lookup(Name);
}

// Whether `N` is a truth value: a comparison, a logical operation, or a choice
// between truth values.
static bool isBoolean(const FPNode &N) {
  if (N.getType() != FPNode::NodeType::Node)
    return false;
  if (N.op == "if")
    return isBoolean(*N.operands[1]);
  const FPCoreOperator *O = fpcoreOperator(N.op);
  return O && O->Boolean;
}

static Error malformed(const Twine &Why, StringRef Expr) {
  return createStringError(inconvertibleErrorCode(),
                           Why + " in '" + Expr + "'");
}

// How an approx form and an element projection begin.
static constexpr StringLiteral kApproxHead = "#s(approx";
static constexpr StringLiteral kProjectionHead = "ref.";

namespace {
struct FPCoreParser {
  std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap;
  std::unordered_map<std::string, Value *> &symbolToValueMap;
  SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs;
  const SetVector<Value *> *subgraphInputs;

  // `expr`, an array only when it is the whole expression (`top`).
  Expected<std::shared_ptr<FPNode>> parse(StringRef expr, bool top);
  // `expr` parsed as a number, not a truth value or an array.
  Expected<std::shared_ptr<FPNode>> parseNumber(StringRef expr) {
    Expected<std::shared_ptr<FPNode>> N = parse(expr, /*top=*/false);
    if (N && isBoolean(**N))
      return malformed("a truth value where a number is expected", expr);
    return N;
  }

private:
  // A literal, `#s(literal v format)` or a bare number; null for any other
  // text.
  Expected<std::shared_ptr<FPNode>> parseLiteral(const std::string &expr);
  // `#s(approx spec result)`: the result, the pair kept in approxSpecs.
  Expected<std::shared_ptr<FPNode>> parseApprox(StringRef expr, bool top);
  // `(ref.N.array<...> e)`: element N of the array or sincos e.
  Expected<std::shared_ptr<FPNode>>
  parseProjection(StringRef expr, StringRef head, StringRef args);
  // `(op.dtype operands...)`.
  Expected<std::shared_ptr<FPNode>>
  parseOperation(StringRef expr, StringRef head, StringRef args, bool top);
};
} // namespace

Expected<std::shared_ptr<FPNode>> FPCoreParser::parse(StringRef expr,
                                                      bool top) {
  const std::string trimmedExpr = expr.trim(' ').str();
  if (trimmedExpr.empty())
    return malformed("an empty expression", expr);
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#')
    if (std::shared_ptr<FPNode> node =
            valueToNodeMap[symbolToValueMap[trimmedExpr]])
      return node;
  Expected<std::shared_ptr<FPNode>> literal = parseLiteral(trimmedExpr);
  if (!literal || *literal)
    return literal;
  if (StringRef(trimmedExpr).starts_with(kApproxHead))
    return parseApprox(trimmedExpr, top);
  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')')
    return malformed("an unknown symbol or an unparenthesized operation",
                     trimmedExpr);
  StringRef body = StringRef(trimmedExpr).drop_front().drop_back();
  StringRef head = body.take_until([](char c) { return c == ' '; });
  StringRef args = body.drop_front(head.size());
  if (head.starts_with(kProjectionHead))
    return parseProjection(trimmedExpr, head, args);
  return parseOperation(trimmedExpr, head, args, top);
}

Expected<std::shared_ptr<FPNode>>
FPCoreParser::parseLiteral(const std::string &expr) {
  static const std::regex constantPattern(
      "^#s\\(literal\\s+([-+]?\\d+(/\\d+)?|[-+]?inf\\.0)\\s+(\\w+)\\)$");
  static const std::regex plainConstantPattern(
      R"(^([-+]?(\d+(\.\d+)?)(/\d+)?|[-+]?inf\.0))");
  // Herbie writes a literal as an exact rational of a representable value and
  // an infinity as inf.0, so a zero denominator is no value it can mean.
  auto zeroDenominator = [](StringRef Literal) {
    StringRef Denominator = Literal.split('/').second;
    return !Denominator.empty() &&
           Denominator.find_first_not_of('0') == StringRef::npos;
  };
  std::smatch matches;
  std::string dtype;
  if (std::regex_match(expr, matches, constantPattern)) {
    dtype = StringSwitch<std::string>(matches[3].str())
                .Case("binary64", "f64")
                .Case("binary32", "f32")
                .Default("");
    if (dtype.empty())
      return malformed("a literal of the unmodelled format " + matches[3].str(),
                       expr);
  } else if (std::regex_match(expr, matches, plainConstantPattern)) {
    dtype = "f64";
  } else {
    return nullptr;
  }
  std::string value = matches[1].str();
  if (zeroDenominator(value))
    return malformed("a rational literal with denominator 0", expr);
  return std::make_shared<FPConst>(value, dtype);
}

Expected<std::shared_ptr<FPNode>> FPCoreParser::parseApprox(StringRef expr,
                                                            bool top) {
  StringRef inner = expr.drop_front(kApproxHead.size()).drop_back().trim(' ');
  SmallVector<StringRef, 4> parts = splitTopLevel(inner);
  if (expr.back() != ')' || parts.size() < 2)
    return malformed("an approx form without its specification and result",
                     expr);
  StringRef resultPart = inner.drop_front(parts[0].size() + 1).trim(' ');
  if (approxSpecs)
    approxSpecs->push_back({parts[0].str(), resultPart.str()});
  return parse(resultPart, top);
}

Expected<std::shared_ptr<FPNode>>
FPCoreParser::parseProjection(StringRef expr, StringRef head, StringRef args) {
  unsigned idx;
  StringRef inner = args.trim(' ');
  if (!head.drop_front(kProjectionHead.size())
           .take_until([](char c) { return c == '.'; })
           .getAsInteger(10, idx) &&
      !inner.empty() && inner.front() == '(' && inner.back() == ')') {
    StringRef projected = inner.drop_front().drop_back();
    StringRef of = projected.take_until([](char c) { return c == ' '; });
    SmallVector<StringRef, 4> elems =
        splitTopLevel(projected.drop_front(of.size()));
    if (of == "array" && idx < elems.size())
      return parseNumber(elems[idx]);
    if ((of == "sincos.f64" || of == "sincos.f32") && elems.size() == 1 &&
        idx < 2) {
      auto node = std::make_shared<FPNode>(idx == 0 ? "sin" : "cos",
                                           of.split('.').second.str());
      Expected<std::shared_ptr<FPNode>> operand = parseNumber(elems[0]);
      if (!operand)
        return operand.takeError();
      node->addOperand(*operand);
      return node;
    }
  }
  return malformed("an array projection of no array element", expr);
}

Expected<std::shared_ptr<FPNode>> FPCoreParser::parseOperation(StringRef expr,
                                                               StringRef head,
                                                               StringRef args,
                                                               bool top) {
  std::string dtype;
  std::string op;
  if (head.starts_with("array<")) {
    op = "array";
  } else if (size_t pos = head.find('.'); pos != StringRef::npos) {
    op = head.substr(0, pos).str();
    dtype = head.substr(pos + 1).str();
    if (dtype != "f64" && dtype != "f32")
      return malformed("the operator " + head + " of an unmodelled format",
                       expr);
  } else {
    op = head.str();
  }

  const FPCoreOperator *O = fpcoreOperator(op);
  if (!O)
    return malformed("the operator " + op + ", which Poseidon does not model",
                     expr);
  SmallVector<StringRef, 4> operandTexts = splitTopLevel(args);
  if (op == "array" ? !top || operandTexts.empty()
                    : operandTexts.size() != O->Arity)
    return malformed(op == "array"
                         ? Twine("an array that is not a whole expression")
                         : "the operator " + op + " with " +
                               Twine(operandTexts.size()) +
                               " operand(s), not " + Twine(O->Arity),
                     expr);

  // Herbie's conversions carry no ".f32"/".f64" suffix; the node takes the
  // destination precision, which makes it an identity at that precision.
  if (dtype.empty()) {
    if (op == "binary64->binary32")
      dtype = "f32";
    else if (op == "binary32->binary64")
      dtype = "f64";
  }

  auto node = std::make_shared<FPNode>(op, dtype);
  // A logical operation, and the condition of a choice, take truth values;
  // every other operand is a number.
  const bool logical = op == "and" || op == "or" || op == "not";
  for (auto [k, text] : enumerate(operandTexts)) {
    const bool truth = logical || (op == "if" && k == 0);
    Expected<std::shared_ptr<FPNode>> operand =
        truth || op == "if" ? parse(text, /*top=*/false) : parseNumber(text);
    if (!operand)
      return operand.takeError();
    if (truth && !isBoolean(**operand))
      return malformed("a number where a truth value is expected", text);
    node->addOperand(*operand);
  }
  if (op == "if" &&
      isBoolean(*node->operands[1]) != isBoolean(*node->operands[2]))
    return malformed("a choice between a number and a truth value", expr);

  if (subgraphInputs)
    if (std::shared_ptr<FPNode> existing =
            existingDefinition(*node, *subgraphInputs, valueToNodeMap))
      return existing;
  return node;
}

Expected<std::shared_ptr<FPNode>> parseHerbieExpr(
    StringRef expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs,
    const SetVector<Value *> *subgraphInputs) {
  FPCoreParser P{valueToNodeMap, symbolToValueMap, approxSpecs, subgraphInputs};
  Expected<std::shared_ptr<FPNode>> N = P.parse(expr, /*top=*/true);
  if (N && isBoolean(**N))
    return malformed("a truth value where a number is expected", expr);
  return N;
}

std::string dtypeOfType(Type *T) {
  const FPFormatInfo *F = formatOf(T);
  return F ? F->Dtype.str() : "";
}

namespace {

struct CExpr {
  std::string text;
  int prec;
  std::string dtype;
};

enum : int {
  kPrecTernary = 1,
  kPrecOr,
  kPrecAnd,
  kPrecEq,
  kPrecRel,
  kPrecAdd,
  kPrecMul,
  kPrecUnary,
  kPrecPrimary
};

} // namespace

static bool isFPDtype(StringRef d) { return formatOfDtype(d); }

StringRef cTypeName(StringRef d) {
  const FPFormatInfo *F = formatOfDtype(d);
  return F ? StringRef(F->CType) : "double";
}

// `v` printed with "%.*g" at `digits` significant digits.
static std::string printG(int digits, double v) {
  std::string s;
  raw_string_ostream(s) << format("%.*g", digits, v);
  return s;
}

static std::string cLiteral(double v, StringRef dtype) {
  if (std::isnan(v))
    return "NAN";
  if (std::isinf(v))
    return v > 0 ? "INFINITY" : "-INFINITY";
  std::string s;
  if (dtype == "f32") {
    float f = (float)v;
    for (int p = 1; p <= std::numeric_limits<float>::max_digits10; ++p) {
      s = printG(p, (double)f);
      if (strtof(s.c_str(), nullptr) == f)
        break;
    }
  } else {
    for (int p = 1; p <= std::numeric_limits<double>::max_digits10; ++p) {
      s = printG(p, v);
      if (strtod(s.c_str(), nullptr) == v)
        break;
    }
  }
  if (s.find_first_of(".e") == std::string::npos)
    s += ".0";
  if (dtype == "f32")
    s += "f";
  return s;
}

static std::string paren(const CExpr &e, int minPrec) {
  return e.prec < minPrec ? "(" + e.text + ")" : e.text;
}

static CExpr castTo(const CExpr &e, StringRef dtype) {
  return {"(" + cTypeName(dtype).str() + ")" + paren(e, kPrecUnary), kPrecUnary,
          dtype.str()};
}

static constexpr unsigned kMaxRenderedNodes = 512;

namespace {

class CRenderer {
  const CRendering &how;
  unsigned budget = kMaxRenderedNodes;

  std::string dtypeOf(const FPNode &N) const {
    if (how.dtypes) {
      auto it = how.dtypes->find(&N);
      if (it != how.dtypes->end())
        return it->second;
    }
    if (auto *LL = dyn_cast<FPLLValue>(&N))
      if (LL->value)
        return dtypeOfType(LL->value->getType());
    return N.dtype;
  }

  bool isLeaf(const FPNode &N) const {
    auto *LL = dyn_cast<FPLLValue>(&N);
    if (!LL)
      return false;
    return N.operands.empty() ||
           (how.inputs && how.inputs->contains(LL->value));
  }

  static bool isFMulAdd(const FPNode &N) {
    auto *LL = dyn_cast<FPLLValue>(&N);
    auto *II = LL ? dyn_cast_or_null<IntrinsicInst>(LL->value) : nullptr;
    return II && II->getIntrinsicID() == Intrinsic::fmuladd;
  }

  CExpr literal(const FPConst &C, StringRef dtype) {
    std::string t = cLiteral(C.getLowerBound(), dtype);
    return {t, t[0] == '-' ? (int)kPrecUnary : (int)kPrecPrimary, dtype.str()};
  }

  CExpr operand(const FPNode &N, StringRef want) {
    if (auto *C = dyn_cast<FPConst>(&N))
      if (isFPDtype(want))
        return literal(*C, want);
    CExpr e = render(N);
    if (isFPDtype(want) && isFPDtype(e.dtype) && e.dtype != want)
      return castTo(e, want);
    return e;
  }

  SmallVector<CExpr, 3> operands(const FPNode &N, StringRef want) {
    SmallVector<CExpr, 3> out;
    for (const auto &op : N.operands)
      out.push_back(operand(*op, want));
    return out;
  }

  static std::string firstFP(ArrayRef<CExpr> es) {
    for (const CExpr &e : es)
      if (isFPDtype(e.dtype))
        return e.dtype;
    return "";
  }

public:
  explicit CRenderer(const CRendering &how) : how(how) {}

  bool exhausted() const { return budget == 0; }

  CExpr render(const FPNode &N) {
    if (budget == 0)
      return {"", kPrecPrimary, ""};
    --budget;
    if (auto *C = dyn_cast<FPConst>(&N))
      return literal(*C, C->dtype.empty() ? StringRef("f64") : C->dtype);
    if (isLeaf(N))
      return leaf(cast<FPLLValue>(N));
    std::string dt = dtypeOf(N);
    return renderOperation(N, isFPDtype(dt) ? dt : "");
  }

private:
  CExpr leaf(const FPLLValue &LL) {
    std::string name = how.leafName && LL.value ? how.leafName(LL.value) : "";
    if (name.empty())
      name = LL.symbol.empty() ? "?" : LL.symbol;
    return {name, kPrecPrimary, dtypeOf(LL)};
  }

  // The operation `N`, its result wanted as `want`, a format's dtype or empty.
  CExpr renderOperation(const FPNode &N, const std::string &want) {
    const std::string &op = N.op;
    if (op == "binary64->binary32" || op == "binary32->binary64")
      return renderConversion(N);
    if (op == "if" && N.operands.size() == 3)
      return renderChoice(N, want);
    if (int P = infixPrecedence(op); P && N.operands.size() >= 2)
      return renderInfix(N, want, P);
    if (op == "not" && N.operands.size() == 1)
      return {"!" + paren(render(*N.operands[0]), kPrecUnary), kPrecUnary,
              "bool"};
    if (N.operands.empty())
      return renderConstant(op, want);
    if (op == "neg" || (op == "-" && N.operands.size() == 1))
      return renderNegation(N, want);
    if (op == "fma" && N.operands.size() == 3 && isFMulAdd(N))
      return renderFMulAdd(N, want);
    if (std::optional<CExpr> e = renderNegativeLiteralTerm(N, want))
      return *e;
    if ((op == "+" || op == "-" || op == "*" || op == "/") &&
        N.operands.size() >= 2)
      return renderArithmetic(N, want);
    if (op == "array")
      return renderArray(N);
    return renderCall(N, want);
  }

  static int infixPrecedence(StringRef op) {
    return StringSwitch<int>(op)
        .Case("<", kPrecRel)
        .Case("<=", kPrecRel)
        .Case(">", kPrecRel)
        .Case(">=", kPrecRel)
        .Case("==", kPrecEq)
        .Case("!=", kPrecEq)
        .Case("and", kPrecAnd)
        .Case("or", kPrecOr)
        .Default(0);
  }

  CExpr renderConversion(const FPNode &N) {
    CExpr inner = render(*N.operands[0]);
    std::string target = N.op == "binary64->binary32" ? "f32" : "f64";
    return inner.dtype == target ? inner : castTo(inner, target);
  }

  CExpr renderChoice(const FPNode &N, const std::string &want) {
    CExpr cond = render(*N.operands[0]);
    CExpr a = operand(*N.operands[1], want);
    CExpr b = operand(*N.operands[2], want.empty() ? a.dtype : want);
    return {paren(cond, kPrecOr) + " ? " + paren(a, kPrecOr) + " : " +
                paren(b, kPrecTernary),
            kPrecTernary, want.empty() ? a.dtype : want};
  }

  CExpr renderInfix(const FPNode &N, const std::string &want, int P) {
    SmallVector<CExpr, 3> es = operands(N, want);
    StringRef sym = N.op == "and"  ? "&&"
                    : N.op == "or" ? "||"
                                   : StringRef(N.op);
    std::string t;
    for (auto [k, e] : llvm::enumerate(es))
      t += (k ? " " + sym.str() + " " : "") + paren(e, P + 1);
    return {t, P, "bool"};
  }

  CExpr renderConstant(const std::string &op, const std::string &want) {
    std::string t = StringSwitch<std::string>(op)
                        .Case("PI", "M_PI")
                        .Case("E", "M_E")
                        .Case("INFINITY", "INFINITY")
                        .Case("NAN", "NAN")
                        .Case("TRUE", "true")
                        .Case("FALSE", "false")
                        .Default(op);
    bool isBool = op == "TRUE" || op == "FALSE";
    CExpr e{t, kPrecPrimary, isBool ? "bool" : "f64"};
    if (!isBool && !want.empty() && want != "f64")
      e = castTo(e, want);
    return e;
  }

  CExpr renderNegation(const FPNode &N, const std::string &want) {
    CExpr e = operand(*N.operands[0], want);
    std::string t = paren(e, kPrecUnary);
    if (t[0] == '-')
      t = "(" + t + ")";
    return {"-" + t, kPrecUnary, want.empty() ? e.dtype : want};
  }

  CExpr renderFMulAdd(const FPNode &N, const std::string &want) {
    SmallVector<CExpr, 3> es = operands(N, want);
    CExpr m{paren(es[0], kPrecMul) + " * " + paren(es[1], kPrecMul + 1),
            kPrecMul, firstFP(es)};
    return {paren(m, kPrecAdd) + " + " + paren(es[2], kPrecAdd + 1), kPrecAdd,
            want.empty() ? firstFP(es) : want};
  }

  // `a + c` or `a - c` with a negative literal c, as `a - |c|` or `a + |c|`.
  std::optional<CExpr> renderNegativeLiteralTerm(const FPNode &N,
                                                 const std::string &want) {
    if ((N.op != "+" && N.op != "-") || N.operands.size() != 2)
      return std::nullopt;
    auto *C = dyn_cast<FPConst>(N.operands[1].get());
    if (!C || !(C->getLowerBound() < 0.0))
      return std::nullopt;
    CExpr a = operand(*N.operands[0], want);
    std::string rdt = want.empty() ? a.dtype : want;
    std::string lit =
        cLiteral(-C->getLowerBound(), rdt.empty() ? StringRef("f64") : rdt);
    return CExpr{paren(a, kPrecAdd) + (N.op == "+" ? " - " : " + ") + lit,
                 kPrecAdd, rdt};
  }

  CExpr renderArithmetic(const FPNode &N, const std::string &want) {
    int P = (N.op == "+" || N.op == "-") ? kPrecAdd : kPrecMul;
    SmallVector<CExpr, 3> es = operands(N, want);
    std::string rdt = want.empty() ? firstFP(es) : want;
    CExpr acc = es[0];
    for (size_t k = 1; k < es.size(); ++k)
      acc = {paren(acc, P) + " " + N.op + " " + paren(es[k], P + 1), P, rdt};
    return acc;
  }

  CExpr renderArray(const FPNode &N) {
    std::string t = "{";
    for (auto [k, e] : llvm::enumerate(N.operands))
      t += (k ? ", " : "") + render(*e).text;
    return {t + "}", kPrecPrimary, ""};
  }

  CExpr renderCall(const FPNode &N, const std::string &want) {
    SmallVector<CExpr, 3> es = operands(N, want);
    std::string rdt = want.empty() ? firstFP(es) : want;
    std::string name = N.op;
    if (hasFloatSuffixInC(name) && rdt == "f32")
      name += "f";
    std::string t = name + "(";
    for (auto [k, e] : llvm::enumerate(es))
      t += (k ? ", " : "") + e.text;
    return {t + ")", kPrecPrimary, rdt};
  }
};

} // namespace

std::string renderC(const FPNode &node, const CRendering &how,
                    StringRef resultDtype) {
  CRenderer R(how);
  CExpr e = R.render(node);
  if (R.exhausted())
    return "";
  if (isFPDtype(resultDtype) && isFPDtype(e.dtype) && e.dtype != resultDtype)
    e = castTo(e, resultDtype);
  return e.text;
}

static bool isRenderableFPCore(
    StringRef expr,
    const std::unordered_map<std::string, Value *> &symbolToValueMap) {
  static const std::regex number(R"(^[-+]?(\d+(\.\d+)?(/\d+)?|inf\.0)$)");
  SmallVector<std::pair<bool, bool>, 8> groups;
  bool expectHead = false;
  size_t i = 0;
  while (i < expr.size()) {
    char c = expr[i];
    if (std::isspace((unsigned char)c)) {
      ++i;
      continue;
    }
    if (c == '(' || expr.substr(i).starts_with("#s(")) {
      i += c == '(' ? 1 : 3;
      groups.push_back({false, !groups.empty() && groups.back().first});
      expectHead = true;
      continue;
    }
    if (c == ')') {
      if (groups.empty())
        return false;
      groups.pop_back();
      expectHead = false;
      ++i;
      continue;
    }
    size_t j = i;
    while (j < expr.size() && !std::isspace((unsigned char)expr[j]) &&
           expr[j] != '(' && expr[j] != ')')
      ++j;
    std::string atom = expr.substr(i, j - i).str();
    i = j;
    if (expectHead) {
      expectHead = false;
      if (atom == "literal" && !groups.empty())
        groups.back().first = true;
      continue;
    }
    bool inLiteral =
        !groups.empty() && (groups.back().first || groups.back().second);
    if (inLiteral || symbolToValueMap.count(atom) ||
        std::regex_match(atom, number))
      continue;
    return false;
  }
  return groups.empty();
}

SmallVector<std::string, 1>
fpcoreToC(StringRef expr,
          std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
          std::unordered_map<std::string, Value *> &symbolToValueMap,
          const CRendering &how, StringRef resultDtype,
          SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs) {
  SmallVector<std::string, 1> out;
  if (!isRenderableFPCore(expr, symbolToValueMap))
    return out;
  Expected<std::shared_ptr<FPNode>> parsed = parseHerbieExpr(
      expr, valueToNodeMap, symbolToValueMap, approxSpecs, how.inputs);
  if (!parsed) {
    consumeError(parsed.takeError());
    return out;
  }
  std::shared_ptr<FPNode> node = *parsed;
  if (node->op == "array")
    for (const auto &element : node->operands)
      out.push_back(renderC(*element, how, resultDtype));
  else
    out.push_back(renderC(*node, how, resultDtype));
  return out;
}

std::string getHerbieOperator(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FNeg:
    return "neg";
  case Instruction::FAdd:
    return "+";
  case Instruction::FSub:
    return "-";
  case Instruction::FMul:
    return "*";
  case Instruction::FDiv:
    return "/";
  case Instruction::FRem:
    return "fmod";
  case Instruction::Call: {
    const Function *Callee = cast<CallInst>(I).getCalledFunction();
    assert(Callee && "getHerbieOperator: Call without a function");
    StringRef Name = Callee->getName();
    if (Callee->isIntrinsic())
      return StringSwitch<StringRef>(intrinsicBaseName(*Callee))
          .Case("fmuladd", "fma")
          .Case("maxnum", "fmax")
          .Case("minnum", "fmin")
          .Case("powi", "pow")
          .Default(intrinsicBaseName(*Callee))
          .str();
    StringRef Fn = mathFunctionOf(Name);
    return (Fn.empty() ? Name : Fn).str();
  }
  default:
    report_fatal_error("getHerbieOperator: Unknown operator");
  }
}

StringRef intrinsicBaseName(const Function &Callee) {
  StringRef Name = Intrinsic::getBaseName(Callee.getIntrinsicID());
  Name.consume_front("llvm.");
  return Name;
}

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args) {
  std::regex argPattern("v\\d+");

  std::sregex_iterator begin(expr.begin(), expr.end(), argPattern);
  std::sregex_iterator end;

  while (begin != end) {
    args.insert(begin->str());
    ++begin;
  }
}

} // namespace poseidon
