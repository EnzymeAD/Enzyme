//=- FPCore.cpp - FP expressions as FPCore and as C -------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPCore.h"
#include "Targets/DeviceMath.h"
#include "Targets/MathFunctions.h"
#include "Utils.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
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

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs,
    const SetVector<Value *> *subgraphInputs) {
  std::string trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    if (auto node = valueToNodeMap[symbolToValueMap[trimmedExpr]]) {
      return node;
    }
  }

  static const std::regex constantPattern(
      "^#s\\(literal\\s+([-+]?\\d+(/\\d+)?|[-+]?inf\\.0)\\s+(\\w+)\\)$");
  static const std::regex plainConstantPattern(
      R"(^([-+]?(\d+(\.\d+)?)(/\d+)?|[-+]?inf\.0))");

  {
    std::smatch matches;
    if (std::regex_match(trimmedExpr, matches, constantPattern)) {
      std::string value = matches[1].str();
      std::string dtype = matches[3].str();
      if (dtype == "binary64") {
        dtype = "f64";
      } else if (dtype == "binary32") {
        dtype = "f32";
      } else {
        std::string msg =
            "Herbie expr parser: Unexpected constant dtype: " + dtype;
        llvm_unreachable(msg.c_str());
      }
      return std::make_shared<FPConst>(value, dtype);
    } else if (std::regex_match(trimmedExpr, matches, plainConstantPattern)) {
      std::string value = matches[1].str();
      std::string dtype = "f64"; // Assume f64 by default
      return std::make_shared<FPConst>(value, dtype);
    }
  }

  if (trimmedExpr.substr(0, 9) == "#s(approx") {
    if (trimmedExpr.back() != ')') {
      llvm_unreachable(("Malformed approx expression: " + trimmedExpr).c_str());
    }
    std::string inner = trimmedExpr.substr(9, trimmedExpr.size() - 9 - 1);
    inner.erase(0, inner.find_first_not_of(" "));
    inner.erase(inner.find_last_not_of(" ") + 1);

    int depth = 0;
    size_t splitPos = std::string::npos;
    for (size_t i = 0; i < inner.size(); ++i) {
      if (inner[i] == '(')
        depth++;
      else if (inner[i] == ')')
        depth--;
      else if (inner[i] == ' ' && depth == 0) {
        splitPos = i;
        break;
      }
    }
    if (splitPos == std::string::npos) {
      llvm_unreachable(("Malformed approx expression: " + trimmedExpr).c_str());
    }
    std::string resultPart = inner.substr(splitPos + 1);
    resultPart.erase(0, resultPart.find_first_not_of(" "));
    resultPart.erase(resultPart.find_last_not_of(" ") + 1);
    if (approxSpecs)
      approxSpecs->push_back({inner.substr(0, splitPos), resultPart});
    return parseHerbieExpr(resultPart, valueToNodeMap, symbolToValueMap,
                           approxSpecs, subgraphInputs);
  }

  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')') {
    llvm::errs() << "Unexpected subexpression: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

  trimmedExpr = trimmedExpr.substr(1, trimmedExpr.size() - 2);

  auto endOp = trimmedExpr.find(' ');
  std::string fullOp = trimmedExpr.substr(0, endOp);

  // An array projection, `(ref.N.array<...> (array e0 e1 ...))` or
  // `(ref.N.array<...> (sincos.fXX x))`: the N-th element is what is wanted.
  if (fullOp.rfind("ref.", 0) == 0) {
    size_t idxEnd = fullOp.find('.', 4);
    unsigned idx = std::stoul(fullOp.substr(4, idxEnd - 4));
    std::string inner = trimmedExpr.substr(endOp + 1);
    inner.erase(0, inner.find_first_not_of(" "));
    inner.erase(inner.find_last_not_of(" ") + 1);
    if (inner.front() == '(' && inner.back() == ')') {
      std::string body = inner.substr(1, inner.size() - 2);
      std::string head = body.substr(0, body.find(' '));
      SmallVector<std::string, 4> elems;
      int depth = 0;
      size_t start = body.find_first_not_of(" ", head.size());
      for (size_t i = start; i < body.size(); ++i) {
        if (body[i] == '(')
          depth++;
        else if (body[i] == ')')
          depth--;
        else if (body[i] == ' ' && depth == 0) {
          elems.push_back(body.substr(start, i - start));
          start = i + 1;
        }
      }
      if (start < body.size())
        elems.push_back(body.substr(start));
      if (head == "array" && idx < elems.size())
        return parseHerbieExpr(elems[idx], valueToNodeMap, symbolToValueMap,
                               approxSpecs, subgraphInputs);
      if (head.rfind("sincos.", 0) == 0 && elems.size() == 1 && idx < 2) {
        auto node =
            std::make_shared<FPNode>(idx == 0 ? "sin" : "cos", head.substr(7));
        node->addOperand(parseHerbieExpr(elems[0], valueToNodeMap,
                                         symbolToValueMap, approxSpecs,
                                         subgraphInputs));
        return node;
      }
    }
    llvm::errs() << "Unexpected array projection: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

  size_t pos = fullOp.find('.');
  std::string dtype;
  std::string op;
  if (fullOp.rfind("array<", 0) == 0) {
    op = "array";
  } else if (pos != std::string::npos) {
    op = fullOp.substr(0, pos);
    dtype = fullOp.substr(pos + 1);
    assert(dtype == "f64" || dtype == "f32");
  } else {
    op = fullOp;
  }

  // Herbie's explicit conversions arrive without the ".f32"/".f64" suffix that
  // gives every other operator its dtype; naming the result precision here
  // makes a conversion node an identity operator carrying the destination
  // precision, which every downstream stage rounds and coerces correctly.
  if (dtype.empty()) {
    if (op == "binary64->binary32")
      dtype = "f32";
    else if (op == "binary32->binary64")
      dtype = "f64";
  }

  auto node = std::make_shared<FPNode>(op, dtype);

  int depth = 0;
  auto start = trimmedExpr.find_first_not_of(" ", endOp);
  std::string::size_type curr;
  for (curr = start; curr < trimmedExpr.size(); ++curr) {
    if (trimmedExpr[curr] == '(')
      depth++;
    if (trimmedExpr[curr] == ')')
      depth--;
    if (depth == 0 && trimmedExpr[curr] == ' ') {
      node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                       valueToNodeMap, symbolToValueMap,
                                       approxSpecs, subgraphInputs));
      start = curr + 1;
    }
  }
  if (start < curr) {
    node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                     valueToNodeMap, symbolToValueMap,
                                     approxSpecs, subgraphInputs));
  }

  if (subgraphInputs)
    if (auto existing =
            existingDefinition(*node, *subgraphInputs, valueToNodeMap))
      return existing;

  return node;
}

std::string dtypeOfType(Type *T) {
  if (T->isHalfTy())
    return "f16";
  if (T->isBFloatTy())
    return "bf16";
  if (T->isFloatTy())
    return "f32";
  if (T->isDoubleTy())
    return "f64";
  if (T->isX86_FP80Ty())
    return "f80";
  if (T->isFP128Ty())
    return "f128";
  return "";
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

bool isFPDtype(StringRef d) {
  return d == "f16" || d == "bf16" || d == "f32" || d == "f64" || d == "f80" ||
         d == "f128";
}

StringRef cTypeName(StringRef d) {
  return StringSwitch<StringRef>(d)
      .Case("f16", "_Float16")
      .Case("bf16", "__bf16")
      .Case("f32", "float")
      .Case("f64", "double")
      .Case("f80", "long double")
      .Case("f128", "__float128")
      .Default("double");
}

std::string cLiteral(double v, StringRef dtype) {
  if (std::isnan(v))
    return "NAN";
  if (std::isinf(v))
    return v > 0 ? "INFINITY" : "-INFINITY";
  char buf[64];
  if (dtype == "f32") {
    float f = (float)v;
    for (int p = 1; p <= 9; ++p) {
      snprintf(buf, sizeof(buf), "%.*g", p, (double)f);
      if (strtof(buf, nullptr) == f)
        break;
    }
  } else {
    for (int p = 1; p <= 17; ++p) {
      snprintf(buf, sizeof(buf), "%.*g", p, v);
      if (strtod(buf, nullptr) == v)
        break;
    }
  }
  std::string s = buf;
  if (s.find_first_of(".e") == std::string::npos)
    s += ".0";
  if (dtype == "f32")
    s += "f";
  return s;
}

std::string paren(const CExpr &e, int minPrec) {
  return e.prec < minPrec ? "(" + e.text + ")" : e.text;
}

CExpr castTo(const CExpr &e, StringRef dtype) {
  return {"(" + cTypeName(dtype).str() + ")" + paren(e, kPrecUnary), kPrecUnary,
          dtype.str()};
}

constexpr unsigned kMaxRenderedNodes = 512;

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
                          .Case("NaN", "NAN")
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
    std::string name = op == "rem" ? "fmod" : op;
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

bool isRenderableFPCore(
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
fpcoreToC(const std::string &expr,
          std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
          std::unordered_map<std::string, Value *> &symbolToValueMap,
          const CRendering &how, StringRef resultDtype,
          SmallVectorImpl<std::pair<std::string, std::string>> *approxSpecs) {
  SmallVector<std::string, 1> out;
  if (!isRenderableFPCore(expr, symbolToValueMap))
    return out;
  std::shared_ptr<FPNode> node = parseHerbieExpr(
      expr, valueToNodeMap, symbolToValueMap, approxSpecs, how.inputs);
  if (!node)
    return out;
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
  case Instruction::Call: {
    const CallInst *CI = dyn_cast<CallInst>(&I);
    assert(CI && CI->getCalledFunction() &&
           "getHerbieOperator: Call without a function");

    StringRef funcName = CI->getCalledFunction()->getName();

    if (StringRef deviceMath = deviceMathName(funcName); !deviceMath.empty())
      return deviceMath.str();

    if (funcName.starts_with("llvm.")) {
      std::regex regex("llvm\\.(\\w+)\\.?.*");
      std::smatch matches;
      std::string nameStr = funcName.str();
      if (std::regex_search(nameStr, matches, regex) && matches.size() > 1) {
        std::string intrinsic = matches[1];
        if (intrinsic == "fmuladd")
          return "fma";
        if (intrinsic == "maxnum")
          return "fmax";
        if (intrinsic == "minnum")
          return "fmin";
        if (intrinsic == "powi")
          return "pow";
        return intrinsic;
      }
      assert(0 && "getHerbieOperator: Unknown LLVM intrinsic");
    } else {
      std::string name = funcName.str();
      if (!name.empty() && name.back() == 'f') {
        name.pop_back();
      }
      return name;
    }
  }
  default:
    assert(0 && "getHerbieOperator: Unknown operator");
  }
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
