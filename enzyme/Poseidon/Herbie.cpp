//=- Herbie.cpp - Herbie integration utilities ----------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities for integrating with the Herbie tool for
// floating-point expression optimization.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "CostModel.h"
#include "Evaluators.h"
#include "Flags.h"
#include "Herbie.h"
#include "HerbieRun.h"
#include "Optimize.h"
#include "Sampling.h"
#include "Types.h"
#include "Utils.h"

#include <iomanip>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

// The Herbie platform this compile searches against. It is the device's, so it
// is derived from the resolved cost model.
namespace {
const HerbiePlatform &getHerbiePlatform() {
  static HerbiePlatform P;
  static bool Resolved = false;
  if (Resolved)
    return P;
  Resolved = true;

  const std::string &NativeArch = getCostModelNativeArch();
  bool IsCUDA = StringRef(NativeArch).starts_with("sm_");
  bool IsHIP = StringRef(NativeArch).starts_with("gfx");
  if (IsCUDA) {
    std::string Stripped = NativeArch;
    size_t Pos;
    while ((Pos = Stripped.find('_')) != std::string::npos)
      Stripped.erase(Pos, 1);
    P.Name = "cuda-" + Stripped;
  } else if (IsHIP) {
    P.Name = "hip-" + NativeArch;
  } else {
    P.Name = "default";
  }

  if (!flags::HerbiePlatform.empty()) {
    // A value that looks like a file is the platform itself; anything else is
    // a platform name compiled into the Herbie binary. The name keeps naming
    // the device family either way, and the digest below distinguishes files.
    StringRef V(flags::HerbiePlatform);
    if (V.contains('/') || V.ends_with(".rkt"))
      P.Path = V.str();
    else
      P.Name = V.str();
  } else if (IsCUDA || IsHIP) {
    // Generated next to the CSV it was priced from, by
    // `poseidon-calibrate --only herbie-platform`.
    StringRef Csv(costModelPath());
    P.Path =
        (Csv.ends_with(".csv") ? Csv.drop_back(4) : Csv).str() + ".herbie.rkt";
    if (!sys::fs::exists(P.Path))
      report_fatal_error(
          Twine("Poseidon: the Herbie algebraic search needs the platform "
                "generated from this device's cost model, and ") +
          P.Path +
          " does not exist. Generate it with 'poseidon-calibrate --only "
          "herbie-platform --out " +
          costModelPath() +
          "'. Another architecture's platform is that architecture's prices "
          "and is never substituted.");
  }

  if (!P.Path.empty()) {
    P.Digest = readFileDigest(P.Path);
    if (P.Digest.empty())
      report_fatal_error(Twine("Poseidon: cannot read the Herbie platform ") +
                         P.Path);
  }
  return P;
}
} // namespace

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

bool isLibmName(StringRef op) {
  static const StringSet<> names = {
      "sin",  "cos",   "tan",   "asin",   "acos",   "atan",     "atan2",
      "sinh", "cosh",  "tanh",  "asinh",  "acosh",  "atanh",    "exp",
      "exp2", "expm1", "log",   "log2",   "log10",  "log1p",    "pow",
      "cbrt", "sqrt",  "fabs",  "fma",    "fmin",   "fmax",     "hypot",
      "ceil", "floor", "trunc", "round",  "rint",   "fmod",     "remainder",
      "fdim", "erf",   "erfc",  "lgamma", "tgamma", "copysign", "nearbyint"};
  return names.contains(op);
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
    if (isLibmName(name) && rdt == "f32")
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

// How this compile runs Herbie: its -poseidon-herbie-* flags.
static HerbieRunOptions herbieRunOptions() {
  HerbieRunOptions Opts;
  Opts.Program = flags::HerbieBinary.empty() ? std::string(HERBIE_BINARY)
                                             : flags::HerbieBinary;
  Opts.Platform = getHerbiePlatform();
  Opts.Seed = flags::RandomSeed;
  Opts.NumPoints = flags::HerbieNumPts;
  Opts.NumIters = flags::HerbieNumIters;
  Opts.NumEnodes = flags::HerbieNumEnodes;
  Opts.CoreTimeout = flags::HerbieTimeout;
  Opts.NumThreads = flags::HerbieNumThreads;
  Opts.Processes = flags::HerbieProcesses;
  Opts.SubgraphTimeout = flags::HerbieSubgraphTimeout;
  Opts.CacheDir = flags::Cache;
  Opts.Print = flags::Print;
  return Opts;
}

void addHerbieKeyParts(CacheKey &key) {
  addHerbieKeyParts(key, herbieRunOptions());
}

bool improveViaHerbie(const std::vector<std::string> &inputExprs,
                      std::vector<CandidateOutput> &COs, int subgraphIdx,
                      StringRef funcTag) {
  if (!flags::HerbieBinary.empty() && flags::Print)
    llvm::errs() << "Poseidon: using Herbie binary '" << flags::HerbieBinary
                 << "'\n";
  llvm::errs() << "random seed: " << std::to_string(flags::RandomSeed) << "\n";
  HerbieRunOptions Opts = herbieRunOptions();

  const std::string Label =
      "subgraph " + std::to_string(subgraphIdx) + " of " + funcTag.str();
  HerbieRunResult Run = runHerbie(inputExprs, Opts, Label);
  if (Run.Result == HerbieRunResult::Outcome::Timeout)
    for (CandidateOutput &CO : COs)
      CO.herbieStatus = "timeout";
  if (Run.Result == HerbieRunResult::Outcome::Crashed) {
    if (flags::StrictMode)
      report_fatal_error(Twine("Poseidon strict mode: Herbie crashed (") +
                         Run.Crash + ") on " + Label);
    for (CandidateOutput &CO : COs)
      CO.herbieStatus = "crashed";
  }
  if (Run.Result != HerbieRunResult::Outcome::Ok)
    return false;

  std::vector<std::unordered_set<std::string>> seenExprs(COs.size());

  auto processHerbieOutput = [&](const std::string &content) -> bool {
    Expected<json::Value> parsed = json::parse(content);
    if (!parsed) {
      llvm::errs() << "Failed to parse Herbie result!\n";
      return false;
    }

    json::Object *obj = parsed->getAsObject();
    json::Array &tests = *obj->getArray("tests");

    for (size_t testIndex = 0; testIndex < tests.size(); ++testIndex) {
      auto &test = *tests[testIndex].getAsObject();

      StringRef testName = test.getString("name").value_or(StringRef("?"));
      StringRef status = test.getString("status").value_or(StringRef("?"));
      llvm::errs() << "[poseidon] Herbie core \"" << testName << "\" (subgraph "
                   << subgraphIdx << " of " << funcTag
                   << "): status=" << status;
      if (auto t = test.getNumber("time"))
        llvm::errs() << " time=" << (*t / 1000.0) << "s";
      if (auto s = test.getNumber("start"))
        llvm::errs() << " start-err-bits=" << *s;
      if (auto e = test.getNumber("end"))
        llvm::errs() << " end-err-bits=" << *e;
      llvm::errs() << "\n";

      StringRef bestExpr = test.getString("output").value();
      if (bestExpr == "#f") {
        llvm::errs() << "[poseidon] Herbie core \"" << testName
                     << "\": no output expression (status " << status
                     << ") -- no algebraic candidate from this core.\n";
        size_t noOutput;
        if (!testName.getAsInteger(10, noOutput) && noOutput < COs.size())
          COs[noOutput].herbieStatus = status.str();
        continue;
      }

      StringRef ID = test.getString("name").value();
      size_t index = std::stoul(ID.str());
      if (index >= COs.size()) {
        llvm::errs() << "Invalid CO index: " << index << "\n";
        continue;
      }

      CandidateOutput &CO = COs[index];
      CO.herbieAnswered = true;
      CO.herbieStatus = status.str();
      auto &seenExprSet = seenExprs[index];

      double bits = test.getNumber("bits").value();
      CO.herbieBits = bits;
      json::Array &costAccuracy = *test.getArray("cost-accuracy");

      json::Array &initial = *costAccuracy[0].getAsArray();
      double initialCostVal = initial[0].getAsNumber().value();
      double initialAccuracy = 1.0 - initial[1].getAsNumber().value() / bits;
      double initialCost = 1.0;

      CO.initialHerbieCost = initialCost;
      CO.initialHerbieAccuracy = initialAccuracy;

      if (seenExprSet.count(bestExpr.str()) == 0) {
        seenExprSet.insert(bestExpr.str());

        json::Array &best = *costAccuracy[1].getAsArray();
        double bestCost = best[0].getAsNumber().value() / initialCostVal;
        double bestAccuracy = 1.0 - best[1].getAsNumber().value() / bits;

        RewriteCandidate bestCandidate(bestCost, bestAccuracy, bestExpr.str());
        bestCandidate.ordinal = CO.candidates.size();
        CO.candidates.push_back(bestCandidate);
      }

      json::Array &alternatives = *costAccuracy[2].getAsArray();

      for (size_t j = 0; j < alternatives.size(); ++j) {
        json::Array &entry = *alternatives[j].getAsArray();
        if (entry.size() < 3)
          continue;
        StringRef expr = entry[2].getAsString().value();

        if (seenExprSet.count(expr.str()) != 0) {
          continue;
        }
        seenExprSet.insert(expr.str());

        double cost = entry[0].getAsNumber().value() / initialCostVal;
        double accuracy = 1.0 - entry[1].getAsNumber().value() / bits;

        RewriteCandidate candidate(cost, accuracy, expr.str());
        candidate.ordinal = CO.candidates.size();
        CO.candidates.push_back(candidate);
      }
    }
    return true;
  };

  return processHerbieOutput(Run.ResultsJson);
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

std::string getPrecondition(
    const SmallSet<std::string, 8> &args,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string preconditions;

  for (const auto &arg : args) {
    const auto node = valueToNodeMap.at(symbolToValueMap.at(arg));
    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    if (upper - lower < 1e-10 && !std::isinf(lower) && !std::isinf(upper)) {
      double midpoint = (lower + upper) / 2.0;
      double tolerance = std::max(1e-10, std::abs(midpoint) * 1e-6);
      lower = midpoint - tolerance;
      upper = midpoint + tolerance;
    }

    std::ostringstream lowerStr, upperStr;
    lowerStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << lower;
    upperStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << upper;

    if (flags::HerbiePreFloorBits > 0 && lower < 0 && upper > 0 &&
        std::isfinite(lower) && std::isfinite(upper)) {
      double floorMag =
          std::ldexp(std::max(-lower, upper), -(int)flags::HerbiePreFloorBits);
      std::ostringstream floorStr;
      floorStr << std::setprecision(std::numeric_limits<double>::max_digits10)
               << std::scientific << floorMag;
      preconditions += " (or (<= " + lowerStr.str() + " " + arg + " -" +
                       floorStr.str() + ") (<= " + floorStr.str() + " " + arg +
                       " " + upperStr.str() + "))";
      continue;
    }

    preconditions +=
        " (<=" +
        (std::isinf(lower) ? (lower > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + lowerStr.str())) +
        " " + arg +
        (std::isinf(upper) ? (upper > 0 ? " INFINITY" : " (- INFINITY)")
                           : (" " + upperStr.str())) +
        ")";
  }

  return preconditions.empty() ? "TRUE" : "(and" + preconditions + ")";
}

namespace {

using RegimeArm = std::pair<const FPNode *, bool>;

void collectIfNodes(const FPNode *node, SmallPtrSetImpl<const FPNode *> &seen,
                    SmallVectorImpl<const FPNode *> &ifs) {
  if (!seen.insert(node).second)
    return;
  if (node->op == "if")
    ifs.push_back(node);
  for (const auto &operand : node->operands)
    collectIfNodes(operand.get(), seen, ifs);
}

// The arms the candidate actually evaluates at `point`, following each `if` the
// way the materialised select resolves it.
void collectTakenArms(const FPNode *node,
                      const MapVector<Value *, double> &point,
                      SmallPtrSetImpl<const FPNode *> &seen,
                      SmallVectorImpl<RegimeArm> &arms) {
  if (!seen.insert(node).second)
    return;
  if (node->op != "if") {
    for (const auto &operand : node->operands)
      collectTakenArms(operand.get(), point, seen, arms);
    return;
  }
  SmallVector<double, 1> cond;
  getFPValues({node->operands[0].get()}, point, cond);
  bool taken = cond[0] == 1.0;
  arms.push_back({node, taken});
  collectTakenArms(node->operands[0].get(), point, seen, arms);
  collectTakenArms(node->operands[taken ? 1 : 2].get(), point, seen, arms);
}

SmallVector<RegimeArm, 4> takenArms(const FPNode *root,
                                    const MapVector<Value *, double> &point) {
  SmallPtrSet<const FPNode *, 32> seen;
  SmallVector<RegimeArm, 4> arms;
  collectTakenArms(root, point, seen, arms);
  return arms;
}

struct ArmStats {
  double candSum = 0.0;
  double origSum = 0.0;
  unsigned count = 0;
};

} // namespace

namespace {

// One round of the regime-arm search. A round's points and the original's
// gold value and error at each depend only on the root and the round's seed, so
// every regime-split candidate of the root shares them.
struct ArmRound {
  SmallVector<MapVector<Value *, double>, 4> points;
  SmallVector<double, 4> gold, origError;
  SmallVector<bool, 4> priced;
};

struct RootSamples {
  Value *root = nullptr;
  double grad = 0.0;
  SmallVector<double, 4> goldVals, origErrors, origVals;
  double origCost = 0.0;
  mutable std::vector<ArmRound> armRounds;
};

bool sampleRoot(
    RootSamples &R, ArrayRef<MapVector<Value *, double>> sampledPoints,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap) {
  R.goldVals.resize(flags::NumSamples);
  R.origErrors.resize(flags::NumSamples);
  R.origVals.resize(flags::NumSamples);
  double sum = 0.0;
  unsigned count = 0;
  for (const auto &pair : enumerate(sampledPoints)) {
    std::shared_ptr<FPNode> node = valueToNodeMap[R.root];
    SmallVector<double, 1> results;
    getMPFRValues({node.get()}, pair.value(), results, true, 53);
    double goldVal = results[0];
    R.goldVals[pair.index()] = goldVal;

    getFPValues({node.get()}, pair.value(), results);
    double realVal = results[0];
    double error = sampleError(goldVal, realVal);
    R.origErrors[pair.index()] = error;
    R.origVals[pair.index()] = realVal;
    if (!std::isnan(error)) {
      sum += error;
      ++count;
    }
  }
  if (count == 0)
    return false;
  R.origCost = sum / count;
  return true;
}

std::optional<double>
priceRoot(const RootSamples &R, std::shared_ptr<FPNode> parsedNode,
          const std::string &exprForLog, const Subgraph &subgraph,
          ArrayRef<MapVector<Value *, double>> sampledPoints,
          std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
          std::unordered_map<std::string, Value *> &symbolToValueMap,
          std::string &why) {
  bool discardCandidate = false;
  double candCost = 0.0;

  SmallVector<const FPNode *, 2> ifNodes;
  if (flags::MinArmSamples > 0) {
    SmallPtrSet<const FPNode *, 32> seen;
    collectIfNodes(parsedNode.get(), seen, ifNodes);
  }
  std::map<RegimeArm, ArmStats> armStats;
  for (const FPNode *ifNode : ifNodes) {
    armStats[{ifNode, true}];
    armStats[{ifNode, false}];
  }

  double sum = 0.0;
  unsigned count = 0;
  unsigned broken = 0;
  for (const auto &pair : enumerate(sampledPoints)) {
    SmallVector<double, 1> results;
    getFPValues({parsedNode.get()}, pair.value(), results);
    double realVal = results[0];
    double goldVal = R.goldVals[pair.index()];
    if (std::isfinite(goldVal) && goldVal != 0.0) {
      double origRel =
          std::fabs(R.origVals[pair.index()] - goldVal) / std::fabs(goldVal);
      double candRel = std::fabs(realVal - goldVal) / std::fabs(goldVal);
      if (origRel < 1.0 && !(candRel < 1.0))
        ++broken;
    }

    if (std::isfinite(goldVal) && !std::isfinite(realVal)) {
      discardCandidate = true;
      why = "it returns a non-finite value at a sampled input where the "
            "exact result is finite";
      break;
    }

    double error = sampleError(goldVal, realVal);
    if (!std::isnan(error)) {
      sum += error;
      ++count;
      double origError = R.origErrors[pair.index()];
      if (!ifNodes.empty() && !std::isnan(origError))
        for (const RegimeArm &arm : takenArms(parsedNode.get(), pair.value())) {
          ArmStats &st = armStats[arm];
          st.candSum += error;
          st.origSum += origError;
          ++st.count;
        }
    }
  }
  if (!discardCandidate) {
    if (count == 0) {
      discardCandidate = true;
      why = "every sample of it scored NaN";
    } else {
      candCost = sum / count;
    }
  }
  if (!discardCandidate && flags::MaxBrokenShare >= 0.0 &&
      broken > flags::MaxBrokenShare * sampledPoints.size()) {
    if (flags::Print)
      llvm::errs() << "[poseidon] dropping candidate: relative error >= 1 on "
                   << broken << " of " << sampledPoints.size()
                   << " samples: " << exprForLog.substr(0, 120) << "\n";
    discardCandidate = true;
    why = formatv("its relative error is 1 or more on {0} of {1} samples",
                  broken, sampledPoints.size())
              .str();
  }

  // Uniform marginal sampling weights each regime by its share of the
  // profiled box, which need not be its share of the workload: correlated
  // inputs can put most of the run in an arm the box almost never reaches.
  // Price every arm on its own samples and charge the worst arm's excess
  // over the original.
  if (!discardCandidate && !ifNodes.empty()) {
    auto starved = [&] {
      for (const auto &kv : armStats)
        if (kv.second.count < flags::MinArmSamples)
          return true;
      return false;
    };
    size_t budget = static_cast<size_t>(flags::ArmSearchFactor) *
                    flags::NumSamples.getValue();
    size_t drawn = 0;
    unsigned round = 0;
    std::shared_ptr<FPNode> origNode = valueToNodeMap[R.root];
    while (starved() && drawn < budget) {
      if (R.armRounds.size() <= round) {
        ArmRound fresh;
        getSampledPoints(subgraph.inputs.getArrayRef(), valueToNodeMap,
                         symbolToValueMap, fresh.points, nullptr,
                         flags::NumSamples.getValue(), round + 1);
        fresh.gold.resize(fresh.points.size());
        fresh.origError.resize(fresh.points.size());
        fresh.priced.resize(fresh.points.size(), false);
        R.armRounds.push_back(std::move(fresh));
      }
      ArmRound &rd = R.armRounds[round++];
      if (rd.points.empty())
        break;
      drawn += rd.points.size();
      for (size_t p = 0; p < rd.points.size(); ++p) {
        const auto &point = rd.points[p];
        SmallVector<RegimeArm, 4> arms = takenArms(parsedNode.get(), point);
        bool wanted = false;
        for (const RegimeArm &arm : arms)
          if (armStats[arm].count < flags::MinArmSamples)
            wanted = true;
        if (!wanted)
          continue;
        SmallVector<double, 1> r;
        if (!rd.priced[p]) {
          getMPFRValues({origNode.get()}, point, r, true, 53);
          rd.gold[p] = r[0];
          getFPValues({origNode.get()}, point, r);
          rd.origError[p] = sampleError(rd.gold[p], r[0]);
          rd.priced[p] = true;
        }
        double goldVal = rd.gold[p];
        double origError = rd.origError[p];
        getFPValues({parsedNode.get()}, point, r);
        double error = sampleError(goldVal, r[0]);
        if (std::isnan(error) || std::isnan(origError))
          continue;
        for (const RegimeArm &arm : arms) {
          ArmStats &st = armStats[arm];
          if (st.count >= flags::MinArmSamples)
            continue;
          st.candSum += error;
          st.origSum += origError;
          ++st.count;
        }
      }
    }

    double worst = candCost;
    std::string armReport;
    raw_string_ostream os(armReport);
    for (const FPNode *ifNode : ifNodes)
      for (bool taken : {false, true}) {
        const ArmStats &st = armStats[{ifNode, taken}];
        os << " " << (taken ? "then" : "else") << ":" << st.count;
        if (st.count)
          os << ":" << format("%.3e", st.candSum / st.count) << "/"
             << format("%.3e", st.origSum / st.count);
        if (st.count < flags::MinArmSamples) {
          discardCandidate = true;
          why = formatv("a regime-split arm is reached by fewer than {0} "
                        "samples",
                        (unsigned)flags::MinArmSamples)
                    .str();
          continue;
        }
        worst =
            std::max(worst, R.origCost + (st.candSum - st.origSum) / st.count);
      }
    if (flags::Print)
      llvm::errs() << "[poseidon] regime-split candidate: arms (samples:cand/"
                      "orig mean error)"
                   << os.str() << "; " << drawn << " extra draws; box cost "
                   << format("%.3e", candCost) << " -> "
                   << (discardCandidate ? std::string("dropped (unpriced arm)")
                                        : formatv("{0:e}", worst).str())
                   << "; " << exprForLog.substr(0, 120) << "\n";
    if (!discardCandidate)
      candCost = worst;
  }
  if (discardCandidate)
    return std::nullopt;
  return candCost;
}

void collectInputs(const FPNode *node, const SetVector<Value *> &inputs,
                   SmallPtrSetImpl<const FPNode *> &seen,
                   SmallPtrSetImpl<Value *> &used) {
  if (!seen.insert(node).second)
    return;
  if (auto *LL = dyn_cast<FPLLValue>(node)) {
    used.insert(LL->value);
    if (inputs.contains(LL->value) && !inlinedOps(*LL, inputs))
      return;
  }
  for (const auto &operand : node->operands)
    collectInputs(operand.get(), inputs, seen, used);
}

} // namespace

void setUnifiedAccuracyCost(
    CandidateOutput &CO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  CancellationPlan cancelPlan =
      buildCancellationPlan(*CO.subgraph, valueToNodeMap);
  SmallVector<MapVector<Value *, double>, 4> sampledPoints;

  getSampledPoints(CO.subgraph->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints, &cancelPlan);

  // No usable sample means the output cannot be priced: drop its candidates and
  // keep the original body, without inventing a bound or an accuracy number.
  auto unpriceable = [&](const char *why) {
    CO.unpricedReason = why;
    llvm::errs() << "[poseidon] subgraph output cannot be priced (" << why
                 << "); discarding its " << CO.candidates.size()
                 << " Herbie candidate(s) and keeping the original body. "
                    "A rewrite may still exist here: this is an un-priced "
                    "site, not a rejected one.\n";
    CO.candidates.clear();
    CO.initialAccCost = 0.0;
  };
  if (sampledPoints.empty()) {
    unpriceable("the profiled input box is empty or unbounded");
    return;
  }

  SmallVector<RootSamples, 1> rootSamples(CO.roots.size());
  double initialAccCost = 0.0;
  for (auto [R, root, grad] : llvm::zip(rootSamples, CO.roots, CO.rootGrads)) {
    R.root = root;
    R.grad = grad;
    if (!sampleRoot(R, sampledPoints, valueToNodeMap)) {
      unpriceable("every sample of the original expression scored NaN");
      return;
    }
    initialAccCost += R.origCost * std::fabs(R.grad);
  }
  CO.initialAccCost = initialAccCost;
  if (!std::isfinite(CO.initialAccCost)) {
    // A non-finite baseline makes every candidate delta nan and the DP would
    // silently fall back to the no-op; drop the candidates and say so instead.
    for (const RootSamples &R : rootSamples)
      llvm::errs() << "  origCost = " << R.origCost << ", grad = " << R.grad
                   << "\n";
    unpriceable("the baseline accuracy cost is not finite; the site's profile "
                "is degenerate (a zero or null-space output shadow gives a "
                "gradient that cannot weight anything)");
    return;
  }

  // An element is spliced in at its own output, so it may only read the values
  // that output's original expression read: inputs, and what an input spelled
  // as its definition is computed from.
  SmallVector<SmallPtrSet<Value *, 8>, 1> rootInputs;
  for (Value *root : CO.roots) {
    SmallPtrSet<const FPNode *, 32> seen;
    SmallPtrSet<Value *, 8> used;
    collectInputs(valueToNodeMap[root].get(), CO.subgraph->inputs, seen, used);
    rootInputs.push_back(std::move(used));
  }

  SmallVector<RewriteCandidate, 4> newCandidates;
  for (auto &candidate : CO.candidates) {
    std::shared_ptr<FPNode> parsedNode =
        parseHerbieExpr(candidate.expr, valueToNodeMap, symbolToValueMap,
                        nullptr, &CO.subgraph->inputs);
    SmallVector<std::shared_ptr<FPNode>, 1> elements;
    if (CO.isArray()) {
      if (parsedNode->op != "array" ||
          parsedNode->operands.size() != CO.roots.size()) {
        llvm::errs() << "[poseidon] dropping array candidate with "
                     << parsedNode->operands.size() << " element(s) for "
                     << CO.roots.size() << " outputs\n";
        CO.dropped.push_back(
            {candidate.expr, "it does not have one element per output"});
        continue;
      }
      elements.append(parsedNode->operands.begin(), parsedNode->operands.end());
    } else {
      elements.push_back(parsedNode);
    }

    bool discardCandidate = false;
    double total = 0.0;
    std::string perElement;
    for (auto [k, element] : llvm::enumerate(elements)) {
      SmallPtrSet<const FPNode *, 32> seen;
      SmallPtrSet<Value *, 8> used;
      collectInputs(element.get(), CO.subgraph->inputs, seen, used);
      if (!llvm::all_of(used,
                        [&](Value *V) { return rootInputs[k].contains(V); })) {
        if (flags::Print)
          llvm::errs() << "[poseidon] dropping candidate: element " << k
                       << " reads a value its output does not\n";
        discardCandidate = true;
        CO.dropped.push_back(
            {candidate.expr, formatv("its element {0} reads a value that "
                                     "output does not",
                                     k)
                                 .str()});
        break;
      }
      std::string why;
      std::optional<double> cost =
          priceRoot(rootSamples[k], element, candidate.expr, *CO.subgraph,
                    sampledPoints, valueToNodeMap, symbolToValueMap, why);
      if (!cost) {
        discardCandidate = true;
        CO.dropped.push_back({candidate.expr, why});
        break;
      }
      total += *cost * std::fabs(rootSamples[k].grad);
      if (CO.isArray() && flags::Print)
        perElement +=
            formatv(" {0:e}/{1:e}", *cost * std::fabs(rootSamples[k].grad),
                    rootSamples[k].origCost * std::fabs(rootSamples[k].grad))
                .str();
    }
    if (CO.isArray() && flags::Print && !discardCandidate)
      llvm::errs() << "[poseidon] array candidate per element (cand/orig):"
                   << perElement << "\n";

    if (!discardCandidate) {
      candidate.accuracyCost = total;
      assert(!std::isnan(candidate.accuracyCost));
      newCandidates.push_back(std::move(candidate));
    }
  }
  CO.candidates = std::move(newCandidates);
}

double getCompCost(
    const std::string &expr, Module *M,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF, const SetVector<Value *> *subgraphInputs) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  SmallVector<Type *, 8> argTypes;
  SmallVector<std::string, 8> argNames;
  for (const auto &argStr : argStrSet) {
    Value *argValue = symbolToValueMap[argStr];
    args.insert(argValue);
    argTypes.push_back(argValue->getType());
    argNames.push_back(argStr);
  }

  auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap,
                                    nullptr, subgraphInputs);
  // An existing value a node was mapped back to is an argument no symbol names.
  SmallVector<Value *, 2> mappedArgs;
  if (subgraphInputs) {
    SmallVector<FPNode *, 16> stack{parsedNode.get()};
    SmallPtrSet<FPNode *, 16> seen;
    while (!stack.empty()) {
      FPNode *N = stack.pop_back_val();
      if (!seen.insert(N).second)
        continue;
      if (auto *L = dyn_cast<FPLLValue>(N)) {
        if (args.insert(L->value)) {
          argTypes.push_back(L->value->getType());
          mappedArgs.push_back(L->value);
        }
        continue;
      }
      for (auto &op : N->operands)
        stack.push_back(op.get());
    }
  }

  Type *ReturnType = nullptr;
  for (Type *ArgTy : argTypes) {
    if (ArgTy->isFloatingPointTy()) {
      ReturnType = ArgTy;
      break;
    }
    if (ArgTy->isVectorTy()) {
      if (auto *VT = dyn_cast<VectorType>(ArgTy)) {
        if (VT->getElementType()->isFloatingPointTy()) {
          ReturnType = ArgTy;
          break;
        }
      }
    }
  }
  if (!ReturnType) {
    if (parsedNode->dtype == "f32")
      ReturnType = Type::getFloatTy(M->getContext());
    else if (parsedNode->dtype == "f16")
      ReturnType = Type::getHalfTy(M->getContext());
    else
      ReturnType = Type::getDoubleTy(M->getContext());
  }

  // An array candidate returns all of its elements, so none is dead and the
  // subterms they share are simplified into one computation before costing.
  const bool isArray = parsedNode->op == "array";
  Type *FuncRetType = ReturnType;
  if (isArray)
    FuncRetType = StructType::get(
        M->getContext(),
        SmallVector<Type *>(parsedNode->operands.size(), ReturnType));

  FunctionType *FT = FunctionType::get(FuncRetType, argTypes, false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "tempFunc", M);

  ValueToValueMapTy VMap;
  Function::arg_iterator AI = tempFunction->arg_begin();
  for (const auto &argStr : argNames) {
    VMap[symbolToValueMap[argStr]] = &*AI;
    ++AI;
  }
  for (Value *arg : mappedArgs) {
    VMap[arg] = &*AI;
    ++AI;
  }

  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);

  IRBuilder<> builder(entry);

  builder.setFastMathFlags(FMF);
  if (isArray) {
    SmallVector<Value *, 4> elements = emitSharedElements(
        parsedNode->operands, [&](size_t) -> IRBuilder<> & { return builder; },
        &VMap, nullptr);
    Value *Agg = UndefValue::get(FuncRetType);
    for (auto [i, V] : llvm::enumerate(elements)) {
      if (V->getType() != ReturnType)
        V = builder.CreateFPCast(V, ReturnType);
      Agg = builder.CreateInsertValue(Agg, V, i);
    }
    builder.CreateRet(Agg);
  } else {
    Value *RetVal = parsedNode->getLLValue(builder, &VMap);
    assert(RetVal && "Parsed node did not produce a value");
    if (RetVal->getType() != ReturnType)
      RetVal = builder.CreateFPCast(RetVal, ReturnType);
    builder.CreateRet(RetVal);
  }

  simplifyFunction(*tempFunction, OptimizationLevel::O3);

  double cost = getCompCost(tempFunction);

  tempFunction->eraseFromParent();
  return cost;
}

} // namespace poseidon
