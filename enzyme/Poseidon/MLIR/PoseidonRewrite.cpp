//===- PoseidonRewrite.cpp - Rewrite elementwise FP chains ----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The poseidon-rewrite pass: chains of elementwise arith/math ops on
// floating-point tensors are written as FPCore, rewritten by Herbie, and
// replaced by the rewrite.
//
//===----------------------------------------------------------------------===//

#include "HerbieResults.h"
#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <cmath>
#include <memory>
#include <random>
#include <string>

namespace mlir::poseidon {
#define GEN_PASS_DEF_POSEIDONREWRITEPASS
#include "Passes.h.inc"
} // namespace mlir::poseidon

using namespace mlir;

namespace {

// The elementwise ops a chain is made of, and the FPCore operator each one
// computes. Every op here applies a scalar operation to each element of its
// tensors independently, so the scalar FPCore describes the whole tensor op.
// The same table, read the other way, says which op builds each operator of a
// rewrite.
struct ChainOp {
  StringRef mlir;
  StringRef fpcore;
  unsigned arity;
};
const ChainOp chainOps[] = {
    {"arith.addf", "+", 2},       {"arith.subf", "-", 2},
    {"arith.mulf", "*", 2},       {"arith.divf", "/", 2},
    {"arith.negf", "neg", 1},     {"arith.remf", "fmod", 2},
    {"arith.maxnumf", "fmax", 2}, {"arith.minnumf", "fmin", 2},
    {"math.absf", "fabs", 1},     {"math.sqrt", "sqrt", 1},
    {"math.cbrt", "cbrt", 1},     {"math.exp", "exp", 1},
    {"math.exp2", "exp2", 1},     {"math.expm1", "expm1", 1},
    {"math.log", "log", 1},       {"math.log2", "log2", 1},
    {"math.log10", "log10", 1},   {"math.log1p", "log1p", 1},
    {"math.powf", "pow", 2},      {"math.fma", "fma", 3},
    {"math.sin", "sin", 1},       {"math.cos", "cos", 1},
    {"math.tan", "tan", 1},       {"math.asin", "asin", 1},
    {"math.acos", "acos", 1},     {"math.atan", "atan", 1},
    {"math.atan2", "atan2", 2},   {"math.sinh", "sinh", 1},
    {"math.cosh", "cosh", 1},     {"math.tanh", "tanh", 1},
    {"math.asinh", "asinh", 1},   {"math.acosh", "acosh", 1},
    {"math.atanh", "atanh", 1},   {"math.erf", "erf", 1},
    {"math.erfc", "erfc", 1},     {"math.copysign", "copysign", 2},
    {"math.floor", "floor", 1},   {"math.ceil", "ceil", 1},
    {"math.round", "round", 1},   {"math.trunc", "trunc", 1},
};

const ChainOp *chainOpByMlirName(StringRef name) {
  for (const ChainOp &op : chainOps)
    if (op.mlir == name)
      return &op;
  return nullptr;
}

const ChainOp *chainOpByFPCore(StringRef name) {
  for (const ChainOp &op : chainOps)
    if (op.fpcore == name)
      return &op;
  return nullptr;
}

// The FPCore operator an op computes, or "" for an op that is not part of a
// chain.
StringRef fpcoreOperator(Operation *op) {
  const ChainOp *info = chainOpByMlirName(op->getName().getStringRef());
  return info ? info->fpcore : StringRef();
}

// A ranked tensor of f32 or f64: the precisions Herbie searches in.
bool isFloatTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor &&
         (tensor.getElementType().isF32() || tensor.getElementType().isF64());
}

bool isChainOp(Operation *op) {
  if (fpcoreOperator(op).empty())
    return false;
  return llvm::all_of(op->getOperandTypes(), isFloatTensor) &&
         llvm::all_of(op->getResultTypes(), isFloatTensor);
}

// A tensor whose elements are all the same float, written into the FPCore as
// a number instead of becoming an input.
std::optional<std::string> splatLiteral(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat())
    return std::nullopt;
  double v = dense.getSplatValue<APFloat>().convertToDouble();
  if (std::isnan(v))
    return std::string("NAN");
  if (std::isinf(v))
    return std::string(v > 0 ? "INFINITY" : "(- INFINITY)");
  // 17 significant digits name a double (and so a float) exactly.
  std::string text;
  llvm::raw_string_ostream os(text);
  os << llvm::format("%.17g", v);
  return text;
}

// A connected group of chain ops in one block, connected through the values
// they pass each other.
struct Chain {
  // In block order.
  SmallVector<Operation *> ops;
  llvm::SmallPtrSet<Operation *, 16> members;
  // The values the chain reads from outside it, in first-use order; input i
  // is the FPCore variable xi. Splat constants are not inputs.
  SmallVector<Value> inputs;
  llvm::DenseMap<Value, unsigned> inputIndex;
  // The chain's results that something outside the chain uses: each is
  // written as its own FPCore and is what a rewrite replaces.
  SmallVector<Value> outputs;
};

SmallVector<Chain, 4> findChains(Block &block) {
  SmallVector<Chain, 4> chains;
  llvm::SmallPtrSet<Operation *, 32> assigned;
  for (Operation &start : block) {
    if (!isChainOp(&start) || assigned.contains(&start))
      continue;

    // Everything reachable from `start` through def-use edges between chain
    // ops of this block, in either direction.
    Chain chain;
    SmallVector<Operation *> worklist = {&start};
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!chain.members.insert(op).second)
        continue;
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp())
          if (def->getBlock() == &block && isChainOp(def))
            worklist.push_back(def);
      for (Operation *user : op->getUsers())
        if (user->getBlock() == &block && isChainOp(user))
          worklist.push_back(user);
    }
    assigned.insert(chain.members.begin(), chain.members.end());

    chain.ops.assign(chain.members.begin(), chain.members.end());
    llvm::sort(chain.ops, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    for (Operation *op : chain.ops) {
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && chain.members.contains(def))
          continue;
        if (!splatLiteral(operand) &&
            chain.inputIndex.try_emplace(operand, chain.inputs.size()).second)
          chain.inputs.push_back(operand);
      }
      for (Value result : op->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !chain.members.contains(user);
            }))
          chain.outputs.push_back(result);
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

// The FPCore expression `value` computes in terms of the chain's inputs. A
// value used twice is written out twice: FPCore expressions are trees.
std::string renderExpr(Value value, const Chain &chain) {
  Operation *def = value.getDefiningOp();
  if (!def || !chain.members.contains(def)) {
    if (auto literal = splatLiteral(value))
      return *literal;
    return "x" + std::to_string(chain.inputIndex.lookup(value));
  }
  std::string expr = "(" + fpcoreOperator(def).str();
  for (Value operand : def->getOperands())
    expr += " " + renderExpr(operand, chain);
  return expr + ")";
}

// One Herbie problem: the output as a function of the chain's inputs.
// `name` comes back in Herbie's results and says which output a rewrite is
// for.
std::string renderFPCore(Value output, const Chain &chain, StringRef name) {
  std::string args;
  for (size_t i = 0; i < chain.inputs.size(); ++i)
    args += (i ? " x" : "x") + std::to_string(i);
  Type element = cast<RankedTensorType>(output.getType()).getElementType();
  return "(FPCore (" + args + ") :name \"" + name.str() + "\" :precision " +
         (element.isF64() ? "binary64" : "binary32") + " " +
         renderExpr(output, chain) + ")";
}

// The value of a Herbie literal ("1/2", "3.5", "-inf.0") in `semantics`,
// e.g. IEEE double for an f64 tensor.
std::optional<APFloat> literalValue(StringRef text,
                                    const llvm::fltSemantics &semantics) {
  bool negative = text.consume_front("-");
  if (!negative)
    text.consume_front("+");
  APFloat value(semantics);
  if (text == "inf.0") {
    value = APFloat::getInf(semantics, negative);
  } else if (text == "nan.0") {
    value = APFloat::getNaN(semantics);
  } else if (auto [num, den] = text.split('/'); !den.empty()) {
    // A fraction: numerator and denominator as integers of any size, divided
    // in quadruple precision, then rounded to the tensor's precision.
    APInt p, q;
    if (num.getAsInteger(10, p) || den.getAsInteger(10, q) || q.isZero())
      return std::nullopt;
    APFloat fp(APFloat::IEEEquad()), fq(APFloat::IEEEquad());
    fp.convertFromAPInt(p, /*IsSigned=*/false, APFloat::rmNearestTiesToEven);
    fq.convertFromAPInt(q, /*IsSigned=*/false, APFloat::rmNearestTiesToEven);
    fp.divide(fq, APFloat::rmNearestTiesToEven);
    bool losesInfo;
    fp.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
    value = fp;
    if (negative)
      value.changeSign();
  } else {
    auto status = value.convertFromString(text, APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      return std::nullopt;
    }
    if (negative)
      value.changeSign();
  }
  return value;
}

// Herbie's comparisons, as arith.cmpf predicates. The ordered ones are false
// when either side is NaN, as in C; != is true then.
std::optional<arith::CmpFPredicate> comparison(StringRef name) {
  if (name == "<")
    return arith::CmpFPredicate::OLT;
  if (name == "<=")
    return arith::CmpFPredicate::OLE;
  if (name == ">")
    return arith::CmpFPredicate::OGT;
  if (name == ">=")
    return arith::CmpFPredicate::OGE;
  if (name == "==")
    return arith::CmpFPredicate::OEQ;
  if (name == "!=")
    return arith::CmpFPredicate::UNE;
  return std::nullopt;
}

// The input an FPCore variable names: 3 for "x3".
std::optional<unsigned> inputIndex(StringRef name) {
  unsigned i;
  if (!name.consume_front("x") || name.getAsInteger(10, i))
    return std::nullopt;
  return i;
}

// Builds Herbie expressions as ops on tensors of one type. `inputs[i]` is the
// value of the FPCore variable xi, or null where that input cannot be used
// at the insertion point.
class ExprBuilder {
public:
  ExprBuilder(RankedTensorType type, ArrayRef<Value> inputs)
      : type(type), inputs(inputs) {}

  // Why `e` cannot be built, or "" when it can. Checked before building so
  // that a candidate that cannot be built leaves no ops behind. `boolean` is
  // whether `e` is a condition (an `if`'s first argument).
  std::string whyNot(const poseidon::HerbieExpr &e,
                     bool boolean = false) const {
    using Kind = poseidon::HerbieExpr::Kind;
    StringRef element = type.getElementType().isF64() ? "f64" : "f32";
    if (!e.precision.empty() && e.precision != element)
      return "computes in " + e.precision + ", not " + element.str();
    if (boolean) {
      if (e.kind == Kind::Op && comparison(e.name)) {
        if (e.args.size() != 2)
          return "'" + e.name + "' with other than two operands";
        return whyNotAll(e.args, false);
      }
      if (e.kind == Kind::Op && (e.name == "and" || e.name == "or")) {
        if (e.args.size() != 2)
          return "'" + e.name + "' with other than two operands";
        return whyNotAll(e.args, true);
      }
      return "not a condition";
    }
    switch (e.kind) {
    case Kind::Var: {
      std::optional<unsigned> i = inputIndex(e.name);
      if (!i || *i >= inputs.size())
        return "unknown variable " + e.name;
      if (!inputs[*i])
        return e.name + " is defined after the chain's output";
      return "";
    }
    case Kind::Number:
      if (!literalValue(e.name, semantics()))
        return "unreadable literal " + e.name;
      return "";
    case Kind::Op:
      if (e.name == "if") {
        if (e.args.size() != 3)
          return "'if' with other than three operands";
        if (std::string why = whyNot(e.args[0], true); !why.empty())
          return why;
        return whyNotAll(ArrayRef(e.args).drop_front(), false);
      }
      if (const ChainOp *op = chainOpByFPCore(e.name)) {
        if (e.args.size() != op->arity)
          return "'" + e.name + "' with " + std::to_string(e.args.size()) +
                 " operands";
        return whyNotAll(e.args, false);
      }
      return "no MLIR op computes '" + e.name + "'";
    }
    llvm_unreachable("unknown HerbieExpr kind");
  }

  // Builds `e`, which whyNot accepted, at the rewriter's insertion point.
  Value build(IRRewriter &rewriter, Location loc,
              const poseidon::HerbieExpr &e) const {
    using Kind = poseidon::HerbieExpr::Kind;
    if (e.kind == Kind::Var)
      return inputs[*inputIndex(e.name)];
    if (e.kind == Kind::Number) {
      // A tensor whose every element is the literal: a splat constant.
      APFloat value = *literalValue(e.name, semantics());
      return arith::ConstantOp::create(
          rewriter, loc,
          cast<TypedAttr>(DenseElementsAttr::get(type, ArrayRef(value))));
    }
    SmallVector<Value, 3> args;
    for (const poseidon::HerbieExpr &arg : e.args)
      args.push_back(build(rewriter, loc, arg));
    // Conditions are tensors of i1, compared and selected element by element.
    if (std::optional<arith::CmpFPredicate> pred = comparison(e.name))
      return arith::CmpFOp::create(rewriter, loc, *pred, args[0], args[1]);
    if (e.name == "and")
      return arith::AndIOp::create(rewriter, loc, args[0], args[1]);
    if (e.name == "or")
      return arith::OrIOp::create(rewriter, loc, args[0], args[1]);
    if (e.name == "if")
      return arith::SelectOp::create(rewriter, loc, args[0], args[1], args[2]);
    // Every op in the table is built the same way: its name, its operands,
    // and one result of the chain's tensor type.
    OperationState state(loc, chainOpByFPCore(e.name)->mlir);
    state.addOperands(args);
    state.addTypes(type);
    return rewriter.create(state)->getResult(0);
  }

private:
  RankedTensorType type;
  ArrayRef<Value> inputs;

  const llvm::fltSemantics &semantics() const {
    return cast<FloatType>(type.getElementType()).getFloatSemantics();
  }

  std::string whyNotAll(ArrayRef<poseidon::HerbieExpr> args,
                        bool boolean) const {
    for (const poseidon::HerbieExpr &arg : args)
      if (std::string why = whyNot(arg, boolean); !why.empty())
        return why;
    return "";
  }
};

// The chain's inputs that ops placed just before `output`'s op can use: an
// input of the chain is defined before some op of the chain, not necessarily
// before this one.
SmallVector<Value> usableInputs(const Chain &chain, Value output) {
  Operation *at = output.getDefiningOp();
  SmallVector<Value> usable;
  for (Value input : chain.inputs) {
    Operation *def = input.getDefiningOp();
    bool before =
        !def || def->getBlock() != at->getBlock() || def->isBeforeInBlock(at);
    usable.push_back(before ? input : Value());
  }
  return usable;
}

struct PoseidonRewrite
    : public poseidon::impl::PoseidonRewritePassBase<PoseidonRewrite> {
  using PoseidonRewritePassBase::PoseidonRewritePassBase;

  // Herbie's answers by core name, shared by the copies of the pass that run
  // on different functions at once. Null without herbie-results.
  std::shared_ptr<const llvm::StringMap<poseidon::HerbieAnswer>> answers;

  // Runs once, before the pass runs on any function.
  LogicalResult initialize(MLIRContext *context) override {
    if (herbieResults.empty())
      return success();
    std::string error;
    std::optional<std::string> json =
        poseidon::readHerbieResults(herbieResults, error);
    std::optional<llvm::StringMap<poseidon::HerbieAnswer>> parsed;
    if (json)
      parsed = poseidon::parseHerbieResults(*json, error);
    if (!parsed)
      return emitError(UnknownLoc::get(context))
             << "poseidon-rewrite: herbie-results " << herbieResults << ": "
             << error;
    answers = std::make_shared<const llvm::StringMap<poseidon::HerbieAnswer>>(
        std::move(*parsed));
    return success();
  }

  // One output and the rewrite chosen for it.
  struct Rewrite {
    Value output;
    poseidon::HerbieExpr expr;
    SmallVector<Value> inputs;
  };

  // Prints Herbie's answer for core `name` (when dumping), and chooses one of
  // its candidates that can be built, at random.
  std::optional<Rewrite> choose(StringRef name, const Chain &chain,
                                Value output) const {
    auto it = answers->find(name);
    if (it == answers->end()) {
      if (dumpFPCore)
        llvm::errs() << "    herbie: no result\n";
      return std::nullopt;
    }
    const poseidon::HerbieAnswer &answer = it->second;
    if (dumpFPCore)
      llvm::errs() << "    herbie: " << answer.status << ", "
                   << answer.candidates.size()
                   << " candidate(s), original error "
                   << llvm::format("%g", answer.initialErrorBits) << " bits\n";

    SmallVector<Value> inputs = usableInputs(chain, output);
    ExprBuilder builder(cast<RankedTensorType>(output.getType()), inputs);
    SmallVector<poseidon::HerbieExpr> buildable;
    for (auto [i, candidate] : llvm::enumerate(answer.candidates)) {
      std::string why;
      std::optional<poseidon::HerbieExpr> expr =
          poseidon::parseHerbieExpr(candidate.expr, why);
      if (expr)
        why = builder.whyNot(*expr);
      if (dumpFPCore) {
        llvm::errs() << "      " << i << ": "
                     << llvm::format("%g", candidate.errorBits)
                     << " bits, cost " << llvm::format("%g", candidate.cost)
                     << ": ";
        if (expr)
          expr->print(llvm::errs());
        else
          llvm::errs() << "unparseable: " << candidate.expr;
        if (!why.empty())
          llvm::errs() << " [cannot build: " << why << "]";
        llvm::errs() << "\n";
      }
      if (why.empty())
        buildable.push_back(std::move(*expr));
    }
    if (buildable.empty()) {
      if (dumpFPCore)
        llvm::errs() << "    no candidate can be built; left as is\n";
      return std::nullopt;
    }

    // Not the best candidate: any one that can be built. The draw depends on
    // the seed and the core's name only, so a run makes the same choices
    // whatever order the functions are processed in. std::mt19937_64's
    // output is fixed by the C++ standard, unlike the std distributions, so
    // the choice is the same with any standard library.
    std::mt19937_64 rng(seed ^ llvm::xxh3_64bits(name));
    size_t pick = rng() % buildable.size();
    if (dumpFPCore) {
      llvm::errs() << "    chose ";
      buildable[pick].print(llvm::errs());
      llvm::errs() << "\n";
    }
    return Rewrite{output, std::move(buildable[pick]), std::move(inputs)};
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Chain, 4> chains;
    func.walk([&](Block *block) {
      for (Chain &chain : findChains(*block))
        // One op alone is a single rounding, which no rewrite improves.
        if (chain.ops.size() >= 2 && !chain.outputs.empty())
          chains.push_back(std::move(chain));
    });

    // First decide every rewrite, then change the IR, so that what is
    // printed and decided is about the function as it came in.
    SmallVector<Rewrite> rewrites;
    unsigned core = 0;
    for (auto [index, chain] : llvm::enumerate(chains)) {
      if (dumpFPCore) {
        llvm::errs() << "poseidon-rewrite: @" << func.getSymName() << " chain "
                     << index << ": " << chain.ops.size() << " ops, "
                     << chain.outputs.size() << " output(s)\n";
        for (auto [i, input] : llvm::enumerate(chain.inputs)) {
          llvm::errs() << "  x" << i << " = ";
          input.printAsOperand(llvm::errs(), OpPrintingFlags());
          llvm::errs() << "\n";
        }
      }
      for (Value output : chain.outputs) {
        std::string name =
            (func.getSymName() + "/" + std::to_string(core++)).str();
        std::string fpcore = renderFPCore(output, chain, name);
        if (dumpFPCore)
          llvm::errs() << "  " << fpcore << "\n";
        if (answers)
          if (std::optional<Rewrite> rewrite = choose(name, chain, output))
            rewrites.push_back(std::move(*rewrite));
      }
    }
    if (rewrites.empty())
      return;

    // Each rewrite is built just before the op that computed the output, where
    // every input it uses is already defined, and takes over all of the
    // output's uses.
    IRRewriter rewriter(&getContext());
    for (Rewrite &rewrite : rewrites) {
      Operation *old = rewrite.output.getDefiningOp();
      rewriter.setInsertionPoint(old);
      ExprBuilder builder(cast<RankedTensorType>(rewrite.output.getType()),
                          rewrite.inputs);
      Value replacement = builder.build(rewriter, old->getLoc(), rewrite.expr);
      rewriter.replaceAllUsesWith(rewrite.output, replacement);
    }

    // The chains' old ops that nothing uses any more. Last to first, so that
    // an op's users are gone before it is looked at.
    for (Chain &chain : chains)
      for (Operation *op : llvm::reverse(chain.ops))
        if (op->use_empty())
          rewriter.eraseOp(op);
  }
};

} // namespace
