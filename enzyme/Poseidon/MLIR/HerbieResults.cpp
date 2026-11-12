//===- HerbieResults.cpp - Herbie's answers for poseidon-rewrite ----------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HerbieResults.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <limits>

using namespace llvm;

namespace mlir::poseidon {

namespace {

// Herbie's literal precisions, in the suffix form its operators use.
std::string precisionOf(StringRef herbieType) {
  if (herbieType == "binary64")
    return "f64";
  if (herbieType == "binary32")
    return "f32";
  return herbieType.str();
}

bool isNumber(StringRef atom) {
  StringRef rest = atom;
  if (rest.starts_with("+") || rest.starts_with("-"))
    rest = rest.drop_front();
  return !rest.empty() &&
         (isDigit(rest.front()) || rest == "inf.0" || rest == "nan.0");
}

// A recursive-descent reader over the text, one expression at a time.
class ExprParser {
public:
  ExprParser(StringRef text, std::string &error) : text(text), error(error) {}

  std::optional<HerbieExpr> parseAll() {
    std::optional<HerbieExpr> e = parse();
    skipSpace();
    if (e && !text.empty())
      return fail("unexpected text after the expression: '" + text.str() + "'");
    return e;
  }

private:
  StringRef text;
  std::string &error;

  std::nullopt_t fail(std::string why) {
    if (error.empty())
      error = std::move(why);
    return std::nullopt;
  }

  void skipSpace() { text = text.ltrim(); }

  StringRef atom() {
    skipSpace();
    size_t end = text.find_first_of(" \t\n()");
    StringRef a = text.take_front(end);
    text = text.drop_front(a.size());
    return a;
  }

  bool expect(char c) {
    skipSpace();
    if (!text.consume_front(StringRef(&c, 1))) {
      fail(std::string("expected '") + c + "' at '" + text.str() + "'");
      return false;
    }
    return true;
  }

  std::optional<HerbieExpr> parse() {
    skipSpace();
    if (text.empty())
      return fail("expression ends early");

    // #s(literal <value> <type>) and #s(approx <spec> <impl>).
    if (text.consume_front("#s(")) {
      StringRef head = atom();
      if (head == "literal") {
        HerbieExpr e{HerbieExpr::Kind::Number, atom().str(), "", {}};
        e.precision = precisionOf(atom());
        if (!expect(')'))
          return std::nullopt;
        return e;
      }
      if (head == "approx") {
        // The spec is the real-number expression Herbie approximated; the
        // impl is what to compute.
        if (!parse())
          return std::nullopt;
        std::optional<HerbieExpr> impl = parse();
        if (!impl || !expect(')'))
          return std::nullopt;
        return impl;
      }
      return fail("unknown #s(" + head.str() + " ...) form");
    }

    // (<op>[.<precision>] <arg>...)
    if (text.consume_front("(")) {
      StringRef op = atom();
      if (op.empty())
        return fail("missing operator at '" + text.str() + "'");
      HerbieExpr e{HerbieExpr::Kind::Op, op.str(), "", {}};
      size_t dot = op.rfind('.');
      if (dot != StringRef::npos) {
        e.name = op.take_front(dot).str();
        e.precision = op.drop_front(dot + 1).str();
      }
      while (true) {
        skipSpace();
        if (text.consume_front(")"))
          return e;
        std::optional<HerbieExpr> arg = parse();
        if (!arg)
          return std::nullopt;
        e.args.push_back(std::move(*arg));
      }
    }

    StringRef a = atom();
    if (a.empty())
      return fail("unexpected '" + text.take_front(1).str() + "'");
    return HerbieExpr{isNumber(a) ? HerbieExpr::Kind::Number
                                  : HerbieExpr::Kind::Var,
                      a.str(),
                      "",
                      {}};
  }
};

double numberOr(const json::Value *v, double fallback) {
  if (v)
    if (std::optional<double> d = v->getAsNumber())
      return *d;
  return fallback;
}

} // namespace

void HerbieExpr::print(raw_ostream &os) const {
  if (kind != Kind::Op) {
    os << name;
    return;
  }
  os << "(" << name;
  for (const HerbieExpr &arg : args) {
    os << " ";
    arg.print(os);
  }
  os << ")";
}

std::optional<HerbieExpr> parseHerbieExpr(StringRef text, std::string &error) {
  return ExprParser(text, error).parseAll();
}

// p/q, for p >= 0 and q > 0, rounded once to the nearest value of
// `semantics`, ties to even. Dividing in a wider format and then converting
// would round twice, which is wrong when the first rounding lands exactly
// halfway between two values of `semantics`.
static std::optional<APFloat> roundFraction(APInt p, APInt q,
                                            const fltSemantics &semantics) {
  // n = floor(p * 2^k / q), with k large enough that n has at least two bits
  // more than `semantics` keeps. When the division leaves a remainder, one
  // more bit, set, stands for it. n * 2^-k is then on the same side of every
  // point where rounding changes as p/q is, so rounding it rounds p/q.
  unsigned precision = APFloat::semanticsPrecision(semantics);
  unsigned pBits = p.getActiveBits(), qBits = q.getActiveBits();
  unsigned k =
      qBits + precision + 3 > pBits ? qBits + precision + 3 - pBits : 0;
  unsigned width = std::max(pBits + k, qBits) + 2;
  APInt n, remainder;
  APInt::udivrem(p.zextOrTrunc(width).shl(k), q.zextOrTrunc(width), n,
                 remainder);
  if (!remainder.isZero()) {
    n = n.shl(1);
    n.setBit(0);
    ++k;
  }
  // APFloat reads n * 2^-k, written in hexadecimal, exactly and rounds it
  // once, subnormal and overflowing results included.
  SmallString<64> hex("0x");
  n.toStringUnsigned(hex, 16);
  hex += "p-";
  hex += std::to_string(k);
  APFloat value(semantics);
  auto status = value.convertFromString(hex, APFloat::rmNearestTiesToEven);
  if (!status) {
    consumeError(status.takeError());
    return std::nullopt;
  }
  // Inexact, overflow and underflow: the rounding asked for.
  return value;
}

std::optional<APFloat> literalValue(StringRef text,
                                    const fltSemantics &semantics) {
  bool negative = text.consume_front("-");
  if (!negative)
    text.consume_front("+");
  APFloat value(semantics);
  if (text == "inf.0") {
    value = APFloat::getInf(semantics, negative);
  } else if (text == "nan.0") {
    value = APFloat::getNaN(semantics);
  } else if (auto [num, den] = text.split('/'); !den.empty()) {
    // A fraction: numerator and denominator as integers of any size.
    APInt p, q;
    if (num.getAsInteger(10, p) || den.getAsInteger(10, q) || q.isZero())
      return std::nullopt;
    std::optional<APFloat> fraction = roundFraction(p, q, semantics);
    if (!fraction)
      return std::nullopt;
    value = *fraction;
    if (negative)
      value.changeSign();
  } else {
    auto status = value.convertFromString(text, APFloat::rmNearestTiesToEven);
    if (!status) {
      consumeError(status.takeError());
      return std::nullopt;
    }
    if (negative)
      value.changeSign();
  }
  return value;
}

std::optional<StringMap<HerbieAnswer>> parseHerbieResults(StringRef json,
                                                          std::string &error) {
  Expected<json::Value> parsed = json::parse(json);
  if (!parsed) {
    error = "not JSON: " + toString(parsed.takeError());
    return std::nullopt;
  }
  const json::Object *root = parsed->getAsObject();
  const json::Array *tests = root ? root->getArray("tests") : nullptr;
  if (!tests) {
    error = "no \"tests\" array";
    return std::nullopt;
  }

  const double unknown = std::numeric_limits<double>::quiet_NaN();
  StringMap<HerbieAnswer> answers;
  for (const json::Value &t : *tests) {
    const json::Object *test = t.getAsObject();
    std::optional<StringRef> name =
        test ? test->getString("name") : std::nullopt;
    if (!name) {
      error = "a test without a \"name\"";
      return std::nullopt;
    }
    HerbieAnswer answer;
    answer.status = test->getString("status").value_or("?").str();
    answer.initialErrorBits = unknown;

    // "cost-accuracy": [[initial cost, initial error],
    //                   [best cost, best error],
    //                   [[cost, error, expr], ...alternatives]]
    // The best expression itself is "output".
    const json::Array *costAccuracy = test->getArray("cost-accuracy");
    auto entry = [&](size_t i) -> const json::Array * {
      return costAccuracy && i < costAccuracy->size()
                 ? (*costAccuracy)[i].getAsArray()
                 : nullptr;
    };
    if (const json::Array *initial = entry(0); initial && initial->size() >= 2)
      answer.initialErrorBits = numberOr(&(*initial)[1], unknown);

    StringRef output = test->getString("output").value_or("#f");
    if (output != "#f") {
      HerbieCandidate best{output.str(), unknown, unknown};
      if (const json::Array *b = entry(1); b && b->size() >= 2) {
        best.cost = numberOr(&(*b)[0], unknown);
        best.errorBits = numberOr(&(*b)[1], unknown);
      }
      answer.candidates.push_back(std::move(best));
      if (const json::Array *alternatives = entry(2))
        for (const json::Value &a : *alternatives) {
          const json::Array *alt = a.getAsArray();
          if (!alt || alt->size() < 3 || !(*alt)[2].getAsString())
            continue;
          answer.candidates.push_back({(*alt)[2].getAsString()->str(),
                                       numberOr(&(*alt)[1], unknown),
                                       numberOr(&(*alt)[0], unknown)});
        }
    }
    answers[*name] = std::move(answer);
  }
  return answers;
}

std::optional<std::string> readHerbieResults(StringRef path,
                                             std::string &error) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer = MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read " + path.str() + ": " + buffer.getError().message();
    return std::nullopt;
  }
  return (*buffer)->getBuffer().str();
}

} // namespace mlir::poseidon
