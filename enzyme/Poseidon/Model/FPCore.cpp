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

namespace {
struct FPCoreOperator {
  StringLiteral Op;
  // The operand count; an array takes any positive number.
  unsigned Arity;
  bool Boolean;
};
} // namespace

// The operators the evaluators and the lowering model, as Herbie names them.
static constexpr FPCoreOperator kOperators[] = {
    {"neg", 1, false},
    {"sin", 1, false},
    {"cos", 1, false},
    {"tan", 1, false},
    {"exp", 1, false},
    {"expm1", 1, false},
    {"log", 1, false},
    {"log1p", 1, false},
    {"sqrt", 1, false},
    {"cbrt", 1, false},
    {"asin", 1, false},
    {"acos", 1, false},
    {"atan", 1, false},
    {"sinh", 1, false},
    {"cosh", 1, false},
    {"tanh", 1, false},
    {"asinh", 1, false},
    {"acosh", 1, false},
    {"atanh", 1, false},
    {"ceil", 1, false},
    {"floor", 1, false},
    {"exp2", 1, false},
    {"log10", 1, false},
    {"log2", 1, false},
    {"rint", 1, false},
    {"round", 1, false},
    {"trunc", 1, false},
    {"fabs", 1, false},
    {"erf", 1, false},
    {"lgamma", 1, false},
    {"tgamma", 1, false},
    {"binary64->binary32", 1, false},
    {"binary32->binary64", 1, false},
    {"+", 2, false},
    {"-", 2, false},
    {"*", 2, false},
    {"/", 2, false},
    {"pow", 2, false},
    {"hypot", 2, false},
    {"atan2", 2, false},
    {"copysign", 2, false},
    {"fmax", 2, false},
    {"fmin", 2, false},
    {"fdim", 2, false},
    {"fmod", 2, false},
    {"remainder", 2, false},
    {"fma", 3, false},
    {"PI", 0, false},
    {"E", 0, false},
    {"INFINITY", 0, false},
    {"NAN", 0, false},
    {"==", 2, true},
    {"!=", 2, true},
    {"<", 2, true},
    {">", 2, true},
    {"<=", 2, true},
    {">=", 2, true},
    {"and", 2, true},
    {"or", 2, true},
    {"not", 1, true},
    {"TRUE", 0, true},
    {"FALSE", 0, true},
    {"if", 3, false},
    {"array", 0, false},
};

static const FPCoreOperator *fpcoreOperator(StringRef Op) {
  const FPCoreOperator *O =
      find_if(kOperators, [&](const FPCoreOperator &O) { return O.Op == Op; });
  return O == std::end(kOperators) ? nullptr : O;
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

namespace {
struct FPCoreParser {
  std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap;
  std::unordered_map<std::string, Value *> &symbolToValueMap;
  SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs;
  const SetVector<Value *> *subgraphInputs;

  Expected<std::shared_ptr<FPNode>> parse(StringRef expr, bool top);
  // `expr` parsed as a number, not a truth value or an array.
  Expected<std::shared_ptr<FPNode>> parseNumber(StringRef expr) {
    Expected<std::shared_ptr<FPNode>> N = parse(expr, /*top=*/false);
    if (N && isBoolean(**N))
      return malformed("a truth value where a number is expected", expr);
    return N;
  }
};
} // namespace

Expected<std::shared_ptr<FPNode>> FPCoreParser::parse(StringRef expr,
                                                      bool top) {
  const std::string trimmedExpr = expr.trim(' ').str();
  if (trimmedExpr.empty())
    return malformed("an empty expression", expr);

  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    if (auto node = valueToNodeMap[symbolToValueMap[trimmedExpr]]) {
      return node;
    }
  }

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
  {
    std::smatch matches;
    if (std::regex_match(trimmedExpr, matches, constantPattern)) {
      std::string value = matches[1].str();
      std::string dtype = matches[3].str();
      if (zeroDenominator(value))
        return malformed("a rational literal with denominator 0", trimmedExpr);
      if (dtype == "binary64")
        dtype = "f64";
      else if (dtype == "binary32")
        dtype = "f32";
      else
        return malformed("a literal of the unmodelled format " + dtype,
                         trimmedExpr);
      return std::make_shared<FPConst>(value, dtype);
    }
    if (std::regex_match(trimmedExpr, matches, plainConstantPattern)) {
      std::string value = matches[1].str();
      if (zeroDenominator(value))
        return malformed("a rational literal with denominator 0", trimmedExpr);
      std::string dtype = "f64";
      return std::make_shared<FPConst>(value, dtype);
    }
  }

  if (StringRef(trimmedExpr).starts_with("#s(approx")) {
    StringRef inner =
        StringRef(trimmedExpr).drop_front(9).drop_back().trim(' ');
    SmallVector<StringRef, 4> parts = splitTopLevel(inner);
    if (trimmedExpr.back() != ')' || parts.size() < 2)
      return malformed("an approx form without its specification and result",
                       trimmedExpr);
    StringRef resultPart = inner.drop_front(parts[0].size() + 1).trim(' ');
    if (approxSpecs)
      approxSpecs->push_back({parts[0].str(), resultPart.str()});
    return parse(resultPart, top);
  }

  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')')
    return malformed("an unknown symbol or an unparenthesized operation",
                     trimmedExpr);

  StringRef body = StringRef(trimmedExpr).drop_front().drop_back();
  StringRef fullOp = body.take_until([](char c) { return c == ' '; });
  StringRef args = body.drop_front(fullOp.size());

  // An array projection, `(ref.N.array<...> (array e0 e1 ...))` or
  // `(ref.N.array<...> (sincos.fXX x))`: the N-th element is what is wanted.
  if (fullOp.starts_with("ref.")) {
    unsigned idx;
    StringRef inner = args.trim(' ');
    if (!fullOp.drop_front(4)
             .take_until([](char c) { return c == '.'; })
             .getAsInteger(10, idx) &&
        !inner.empty() && inner.front() == '(' && inner.back() == ')') {
      StringRef projected = inner.drop_front().drop_back();
      StringRef head = projected.take_until([](char c) { return c == ' '; });
      SmallVector<StringRef, 4> elems =
          splitTopLevel(projected.drop_front(head.size()));
      if (head == "array" && idx < elems.size())
        return parseNumber(elems[idx]);
      if ((head == "sincos.f64" || head == "sincos.f32") && elems.size() == 1 &&
          idx < 2) {
        auto node = std::make_shared<FPNode>(idx == 0 ? "sin" : "cos",
                                             head.drop_front(7).str());
        Expected<std::shared_ptr<FPNode>> operand = parseNumber(elems[0]);
        if (!operand)
          return operand.takeError();
        node->addOperand(*operand);
        return node;
      }
    }
    return malformed("an array projection of no array element", trimmedExpr);
  }

  std::string dtype;
  std::string op;
  if (fullOp.starts_with("array<")) {
    op = "array";
  } else if (size_t pos = fullOp.find('.'); pos != StringRef::npos) {
    op = fullOp.substr(0, pos).str();
    dtype = fullOp.substr(pos + 1).str();
    if (dtype != "f64" && dtype != "f32")
      return malformed("the operator " + fullOp + " of an unmodelled format",
                       trimmedExpr);
  } else {
    op = fullOp.str();
  }

  const FPCoreOperator *O = fpcoreOperator(op);
  if (!O)
    return malformed("the operator " + op + ", which Poseidon does not model",
                     trimmedExpr);
  SmallVector<StringRef, 4> operandTexts = splitTopLevel(args);
  if (op == "array" ? !top || operandTexts.empty()
                    : operandTexts.size() != O->Arity)
    return malformed(op == "array"
                         ? Twine("an array that is not a whole expression")
                         : "the operator " + op + " with " +
                               Twine(operandTexts.size()) +
                               " operand(s), not " + Twine(O->Arity),
                     trimmedExpr);

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
    return malformed("a choice between a number and a truth value",
                     trimmedExpr);

  if (subgraphInputs)
    if (auto existing =
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
    for (int p = 1; p <= 9; ++p) {
      s = printG(p, (double)f);
      if (strtof(s.c_str(), nullptr) == f)
        break;
    }
  } else {
    for (int p = 1; p <= 17; ++p) {
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
    if (isLeaf(N)) {
      auto *LL = cast<FPLLValue>(&N);
      std::string name =
          how.leafName && LL->value ? how.leafName(LL->value) : "";
      if (name.empty())
        name = N.symbol.empty() ? "?" : N.symbol;
      return {name, kPrecPrimary, dtypeOf(N)};
    }

    const std::string &op = N.op;
    std::string dt = dtypeOf(N);
    const std::string want = isFPDtype(dt) ? dt : "";

    if (op == "binary64->binary32" || op == "binary32->binary64") {
      CExpr inner = render(*N.operands[0]);
      std::string target = op == "binary64->binary32" ? "f32" : "f64";
      return inner.dtype == target ? inner : castTo(inner, target);
    }
    if (op == "if" && N.operands.size() == 3) {
      CExpr cond = render(*N.operands[0]);
      CExpr a = operand(*N.operands[1], want);
      CExpr b = operand(*N.operands[2], want.empty() ? a.dtype : want);
      return {paren(cond, kPrecOr) + " ? " + paren(a, kPrecOr) + " : " +
                  paren(b, kPrecTernary),
              kPrecTernary, want.empty() ? a.dtype : want};
    }
    static const StringMap<int> infix = {
        {"<", kPrecRel}, {"<=", kPrecRel}, {">", kPrecRel},   {">=", kPrecRel},
        {"==", kPrecEq}, {"!=", kPrecEq},  {"and", kPrecAnd}, {"or", kPrecOr}};
    auto inf = infix.find(op);
    if (inf != infix.end() && N.operands.size() >= 2) {
      SmallVector<CExpr, 3> es = operands(N, want);
      StringRef sym = op == "and" ? "&&" : op == "or" ? "||" : StringRef(op);
      std::string t;
      for (auto [k, e] : llvm::enumerate(es))
        t += (k ? " " + sym.str() + " " : "") + paren(e, inf->second + 1);
      return {t, inf->second, "bool"};
    }
    if (op == "not" && N.operands.size() == 1)
      return {"!" + paren(render(*N.operands[0]), kPrecUnary), kPrecUnary,
              "bool"};
    if (N.operands.empty()) {
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
    if (op == "neg" || (op == "-" && N.operands.size() == 1)) {
      CExpr e = operand(*N.operands[0], want);
      std::string t = paren(e, kPrecUnary);
      if (t[0] == '-')
        t = "(" + t + ")";
      return {"-" + t, kPrecUnary, want.empty() ? e.dtype : want};
    }
    if (op == "fma" && N.operands.size() == 3 && isFMulAdd(N)) {
      SmallVector<CExpr, 3> es = operands(N, want);
      CExpr m{paren(es[0], kPrecMul) + " * " + paren(es[1], kPrecMul + 1),
              kPrecMul, firstFP(es)};
      return {paren(m, kPrecAdd) + " + " + paren(es[2], kPrecAdd + 1), kPrecAdd,
              want.empty() ? firstFP(es) : want};
    }
    if ((op == "+" || op == "-") && N.operands.size() == 2)
      if (auto *C = dyn_cast<FPConst>(N.operands[1].get()))
        if (C->getLowerBound() < 0.0) {
          CExpr a = operand(*N.operands[0], want);
          std::string rdt = want.empty() ? a.dtype : want;
          std::string lit = cLiteral(-C->getLowerBound(),
                                     rdt.empty() ? StringRef("f64") : rdt);
          return {paren(a, kPrecAdd) + (op == "+" ? " - " : " + ") + lit,
                  kPrecAdd, rdt};
        }
    if ((op == "+" || op == "-" || op == "*" || op == "/") &&
        N.operands.size() >= 2) {
      int P = (op == "+" || op == "-") ? kPrecAdd : kPrecMul;
      SmallVector<CExpr, 3> es = operands(N, want);
      std::string rdt = want.empty() ? firstFP(es) : want;
      CExpr acc = es[0];
      for (size_t k = 1; k < es.size(); ++k)
        acc = {paren(acc, P) + " " + op + " " + paren(es[k], P + 1), P, rdt};
      return acc;
    }
    if (op == "array") {
      std::string t = "{";
      for (auto [k, e] : llvm::enumerate(N.operands))
        t += (k ? ", " : "") + render(*e).text;
      return {t + "}", kPrecPrimary, ""};
    }
    SmallVector<CExpr, 3> es = operands(N, want);
    std::string rdt = want.empty() ? firstFP(es) : want;
    std::string name = op;
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
  return Intrinsic::getBaseName(Callee.getIntrinsicID()).drop_front(5);
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
