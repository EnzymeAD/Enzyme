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
#include "HerbieRun.h"
#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

namespace mlir::poseidon {
#define GEN_PASS_DEF_POSEIDONREWRITEPASS
#include "Passes.h.inc"
} // namespace mlir::poseidon

using namespace mlir;

namespace {

// Both this pass's namespace, mlir::poseidon, and the Herbie runner's,
// ::poseidon, would answer to `poseidon` here.
namespace poseidon = mlir::poseidon;
namespace herbie = ::poseidon;

// The elementwise ops a chain is made of, and the FPCore operator each one
// computes. Every op here applies a scalar operation to each element of its
// tensors independently, so the scalar FPCore describes the whole tensor op.
// The same table, read the other way, says which op builds each operator of a
// rewrite.
//
// FPCore's fmax and fmin return the other operand when one is NaN, as
// arith.maxnumf and arith.minnumf do; arith.maximumf and arith.minimumf
// return NaN instead, so they compute no FPCore operator.
struct ChainOp {
  StringRef mlir;
  StringRef fpcore;
  unsigned arity;
};
// The op's name comes from its class, so that the compiler checks it.
template <typename OpTy>
constexpr ChainOp chainOp(StringRef fpcore, unsigned arity) {
  return {OpTy::getOperationName(), fpcore, arity};
}
const ChainOp chainOps[] = {
    chainOp<arith::AddFOp>("+", 2),
    chainOp<arith::SubFOp>("-", 2),
    chainOp<arith::MulFOp>("*", 2),
    chainOp<arith::DivFOp>("/", 2),
    chainOp<arith::NegFOp>("neg", 1),
    chainOp<arith::RemFOp>("fmod", 2),
    chainOp<arith::MaxNumFOp>("fmax", 2),
    chainOp<arith::MinNumFOp>("fmin", 2),
    chainOp<math::AbsFOp>("fabs", 1),
    chainOp<math::SqrtOp>("sqrt", 1),
    chainOp<math::CbrtOp>("cbrt", 1),
    chainOp<math::ExpOp>("exp", 1),
    chainOp<math::Exp2Op>("exp2", 1),
    chainOp<math::ExpM1Op>("expm1", 1),
    chainOp<math::LogOp>("log", 1),
    chainOp<math::Log2Op>("log2", 1),
    chainOp<math::Log10Op>("log10", 1),
    chainOp<math::Log1pOp>("log1p", 1),
    chainOp<math::PowFOp>("pow", 2),
    chainOp<math::FmaOp>("fma", 3),
    chainOp<math::SinOp>("sin", 1),
    chainOp<math::CosOp>("cos", 1),
    chainOp<math::TanOp>("tan", 1),
    chainOp<math::AsinOp>("asin", 1),
    chainOp<math::AcosOp>("acos", 1),
    chainOp<math::AtanOp>("atan", 1),
    chainOp<math::Atan2Op>("atan2", 2),
    chainOp<math::SinhOp>("sinh", 1),
    chainOp<math::CoshOp>("cosh", 1),
    chainOp<math::TanhOp>("tanh", 1),
    chainOp<math::AsinhOp>("asinh", 1),
    chainOp<math::AcoshOp>("acosh", 1),
    chainOp<math::AtanhOp>("atanh", 1),
    chainOp<math::ErfOp>("erf", 1),
    chainOp<math::ErfcOp>("erfc", 1),
    chainOp<math::CopySignOp>("copysign", 2),
    chainOp<math::FloorOp>("floor", 1),
    chainOp<math::CeilOp>("ceil", 1),
    chainOp<math::RoundOp>("round", 1),
    chainOp<math::TruncOp>("trunc", 1),
};

// An op of the table, as one context names it.
struct KnownOp {
  KnownOp(OperationName name, const ChainOp *info) : name(name), info(info) {}
  OperationName name;
  const ChainOp *info;
};

// The table, looked up by an op's name in one context and by FPCore operator.
class ChainOpTable {
public:
  explicit ChainOpTable(MLIRContext *context) {
    for (const ChainOp &op : chainOps) {
      OperationName name(op.mlir, context);
      byName.try_emplace(name, &op);
      byFPCore.try_emplace(op.fpcore, name, &op);
    }
  }

  // The entry of `op`'s kind, or null for an op that is not part of a chain.
  const ChainOp *forOp(Operation *op) const {
    return byName.lookup(op->getName());
  }

  // The op that computes FPCore operator `fpcore`, or null.
  const KnownOp *forFPCore(StringRef fpcore) const {
    auto it = byFPCore.find(fpcore);
    return it == byFPCore.end() ? nullptr : &it->second;
  }

private:
  llvm::DenseMap<OperationName, const ChainOp *> byName;
  llvm::StringMap<KnownOp> byFPCore;
};

// A ranked tensor of f32 or f64: the precisions Herbie searches in.
bool isF32OrF64Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor &&
         (tensor.getElementType().isF32() || tensor.getElementType().isF64());
}

bool isChainOp(const ChainOpTable &table, Operation *op) {
  return table.forOp(op) &&
         llvm::all_of(op->getOperandTypes(), isF32OrF64Tensor) &&
         llvm::all_of(op->getResultTypes(), isF32OrF64Tensor);
}

// The element of a tensor whose elements are all the same float. Such a
// tensor is written into the FPCore as a number instead of becoming an input.
std::optional<APFloat> splatValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat())
    return std::nullopt;
  return dense.getSplatValue<APFloat>();
}

// `v` as an FPCore number, in its own precision's digits: 17 for f64 and 9
// for f32, enough to read back as the same value.
void printNumber(const APFloat &v, raw_ostream &os) {
  if (v.isNaN()) {
    os << "NAN";
    return;
  }
  if (v.isInfinity()) {
    os << (v.isNegative() ? "(- INFINITY)" : "INFINITY");
    return;
  }
  SmallString<32> text;
  v.toString(text);
  // FPCore writes the exponent with a lower-case e.
  std::replace(text.begin(), text.end(), 'E', 'e');
  os << text;
}

// A connected group of chain ops in one block, connected through the values
// they pass each other.
struct Chain {
  // In block order.
  llvm::SetVector<Operation *> ops;
  // The values the chain reads from outside it, in first-use order, each with
  // its position: input i is the FPCore variable xi. Splat constants are not
  // inputs.
  llvm::MapVector<Value, unsigned> inputs;
  // The chain's results that something outside the chain uses: each is
  // written as its own FPCore and is what a rewrite replaces.
  SmallVector<Value> outputs;
};

SmallVector<Chain> findChains(Block &block, const ChainOpTable &table) {
  SmallVector<Chain> chains;
  llvm::SmallPtrSet<Operation *, 32> assigned;
  for (Operation &start : block) {
    if (!isChainOp(table, &start) || assigned.contains(&start))
      continue;

    // Everything reachable from `start` through def-use edges between chain
    // ops of this block, in either direction.
    llvm::SetVector<Operation *> found;
    SmallVector<Operation *> worklist = {&start};
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!found.insert(op))
        continue;
      for (Value operand : op->getOperands())
        if (Operation *def = operand.getDefiningOp())
          if (def->getBlock() == &block && isChainOp(table, def))
            worklist.push_back(def);
      for (Operation *user : op->getUsers())
        if (user->getBlock() == &block && isChainOp(table, user))
          worklist.push_back(user);
    }
    assigned.insert(found.begin(), found.end());

    // In block order, so that the inputs are numbered the same way however
    // the chain was discovered.
    SmallVector<Operation *> sorted = found.takeVector();
    llvm::sort(sorted, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    Chain chain;
    chain.ops.insert(sorted.begin(), sorted.end());
    for (Operation *op : chain.ops) {
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && chain.ops.contains(def))
          continue;
        if (!splatValue(operand))
          chain.inputs.try_emplace(operand, chain.inputs.size());
      }
      for (Value result : op->getResults())
        if (llvm::any_of(result.getUsers(), [&](Operation *user) {
              return !chain.ops.contains(user);
            }))
          chain.outputs.push_back(result);
    }
    chains.push_back(std::move(chain));
  }
  return chains;
}

// Prints the FPCore expression `value` computes in terms of the chain's
// inputs. A value used twice is written out twice: FPCore expressions are
// trees.
void printExpr(Value value, const Chain &chain, const ChainOpTable &table,
               raw_ostream &os) {
  Operation *def = value.getDefiningOp();
  if (!def || !chain.ops.contains(def)) {
    if (std::optional<APFloat> literal = splatValue(value))
      printNumber(*literal, os);
    else
      os << "x" << chain.inputs.lookup(value);
    return;
  }
  os << "(" << table.forOp(def)->fpcore << " ";
  llvm::interleave(
      def->getOperands(), os,
      [&](Value operand) { printExpr(operand, chain, table, os); }, " ");
  os << ")";
}

// One Herbie problem: the output as a function of the chain's inputs.
// `name` comes back in Herbie's results and says which output a rewrite is
// for.
std::string renderFPCore(Value output, const Chain &chain,
                         const ChainOpTable &table, StringRef name) {
  std::string core;
  llvm::raw_string_ostream os(core);
  os << "(FPCore (";
  llvm::interleave(
      llvm::seq<size_t>(0, chain.inputs.size()), os,
      [&](size_t i) { os << "x" << i; }, " ");
  Type element = cast<RankedTensorType>(output.getType()).getElementType();
  os << llvm::formatv(") :name \"{0}\" :precision {1} ", name,
                      element.isF64() ? "binary64" : "binary32");
  printExpr(output, chain, table, os);
  os << ")";
  return core;
}

// The input an FPCore variable names, as renderFPCore names them: 3 for "x3".
std::optional<unsigned> inputIndex(StringRef name) {
  unsigned i;
  if (!name.consume_front("x") || name.getAsInteger(10, i))
    return std::nullopt;
  return i;
}

// Herbie's comparisons, as arith.cmpf predicates: MLIR's, so here rather than
// with the parser, which knows no MLIR. The ordered ones are false when either
// side is NaN, as in C; != is true then.
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

// Builds Herbie expressions as ops on tensors of one type. `inputs[i]` is the
// value of the FPCore variable xi, or null where that input cannot be used
// at the insertion point.
class ExprBuilder {
public:
  ExprBuilder(const ChainOpTable &table, RankedTensorType type,
              ArrayRef<Value> inputs)
      : table(table), type(type), inputs(inputs) {}

  // Whether `e` can be built; when it cannot, says why in `why`. Checked
  // before building so that a candidate that cannot be built leaves no ops
  // behind. `boolean` is whether `e` is a condition (an `if`'s first
  // argument).
  LogicalResult canBuild(const poseidon::HerbieExpr &e, raw_ostream &why,
                         bool boolean = false) const {
    using Kind = poseidon::HerbieExpr::Kind;
    StringRef element = type.getElementType().isF64() ? "f64" : "f32";
    if (!e.precision.empty() && e.precision != element) {
      why << "computes in " << e.precision << ", not " << element;
      return failure();
    }
    if (boolean) {
      bool comparing = e.kind == Kind::Op && comparison(e.name);
      bool joining = e.kind == Kind::Op && (e.name == "and" || e.name == "or");
      if (!comparing && !joining) {
        why << "not a condition";
        return failure();
      }
      if (e.args.size() != 2) {
        why << "'" << e.name << "' with other than two operands";
        return failure();
      }
      return canBuildAll(e.args, why, /*boolean=*/joining);
    }
    switch (e.kind) {
    case Kind::Var: {
      std::optional<unsigned> i = inputIndex(e.name);
      if (!i || *i >= inputs.size()) {
        why << "unknown variable " << e.name;
        return failure();
      }
      if (!inputs[*i]) {
        why << e.name << " is defined after the chain's output";
        return failure();
      }
      return success();
    }
    case Kind::Number:
      if (!poseidon::literalValue(e.name, semantics())) {
        why << "unreadable literal " << e.name;
        return failure();
      }
      return success();
    case Kind::Op:
      if (e.name == "if") {
        if (e.args.size() != 3) {
          why << "'if' with other than three operands";
          return failure();
        }
        if (failed(canBuild(e.args[0], why, /*boolean=*/true)))
          return failure();
        return canBuildAll(ArrayRef(e.args).drop_front(), why,
                           /*boolean=*/false);
      }
      if (const KnownOp *op = table.forFPCore(e.name)) {
        if (e.args.size() != op->info->arity) {
          why << "'" << e.name << "' with " << e.args.size() << " operands";
          return failure();
        }
        return canBuildAll(e.args, why, /*boolean=*/false);
      }
      why << "no MLIR op computes '" << e.name << "'";
      return failure();
    }
    llvm_unreachable("unknown HerbieExpr kind");
  }

  // Builds `e`, which canBuild accepted, at the rewriter's insertion point.
  Value build(IRRewriter &rewriter, Location loc,
              const poseidon::HerbieExpr &e) const {
    using Kind = poseidon::HerbieExpr::Kind;
    if (e.kind == Kind::Var)
      return inputs[*inputIndex(e.name)];
    if (e.kind == Kind::Number) {
      // A tensor whose every element is the literal: a splat constant.
      APFloat value = *poseidon::literalValue(e.name, semantics());
      return arith::ConstantOp::create(
          rewriter, loc,
          cast<TypedAttr>(DenseElementsAttr::get(type, ArrayRef(value))));
    }
    SmallVector<Value> args;
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
    OperationState state(loc, table.forFPCore(e.name)->name);
    state.addOperands(args);
    state.addTypes(type);
    return rewriter.create(state)->getResult(0);
  }

private:
  const ChainOpTable &table;
  RankedTensorType type;
  ArrayRef<Value> inputs;

  const llvm::fltSemantics &semantics() const {
    return cast<FloatType>(type.getElementType()).getFloatSemantics();
  }

  LogicalResult canBuildAll(ArrayRef<poseidon::HerbieExpr> args,
                            raw_ostream &why, bool boolean) const {
    for (const poseidon::HerbieExpr &arg : args)
      if (failed(canBuild(arg, why, boolean)))
        return failure();
    return success();
  }
};

// The chain's inputs that ops placed just before `output`'s op can use: an
// input of the chain is defined before some op of the chain, not necessarily
// before this one.
SmallVector<Value> usableInputs(const Chain &chain, Value output) {
  Operation *at = output.getDefiningOp();
  SmallVector<Value> usable;
  for (Value input : llvm::make_first_range(chain.inputs)) {
    Operation *def = input.getDefiningOp();
    bool before =
        !def || def->getBlock() != at->getBlock() || def->isBeforeInBlock(at);
    usable.push_back(before ? input : Value());
  }
  return usable;
}

// The Herbie executable configured at build time.
#ifndef HERBIE_BINARY
#define HERBIE_BINARY ""
#endif

struct PoseidonRewrite
    : public poseidon::impl::PoseidonRewritePassBase<PoseidonRewrite> {
  using PoseidonRewritePassBase::PoseidonRewritePassBase;

  using Answers = llvm::StringMap<poseidon::HerbieAnswer>;

  // The chain ops, as this pass's context names them.
  std::shared_ptr<const ChainOpTable> chainOpTable;

  // Herbie's answers by core name, read from herbie-results and shared by the
  // copies of the pass that run on different functions at once. Null without
  // herbie-results.
  std::shared_ptr<const Answers> answers;
  // How to run Herbie, when run-herbie asks for it and herbie-results does
  // not answer instead. Null otherwise.
  std::shared_ptr<const herbie::HerbieRunOptions> herbieRun;

  // Runs once, before the pass runs on any function.
  LogicalResult initialize(MLIRContext *context) override {
    chainOpTable = std::make_shared<const ChainOpTable>(context);
    if (!herbieResults.empty()) {
      std::string error;
      std::optional<std::string> json =
          poseidon::readHerbieResults(herbieResults, error);
      std::optional<Answers> parsed;
      if (json)
        parsed = poseidon::parseHerbieResults(*json, error);
      if (!parsed)
        return emitError(UnknownLoc::get(context))
               << "poseidon-rewrite: herbie-results " << herbieResults << ": "
               << error;
      answers = std::make_shared<const Answers>(std::move(*parsed));
      return success();
    }
    if (!runHerbie)
      return success();

    herbie::HerbieRunOptions opts;
    opts.Program = herbieBinary.empty() ? std::string(HERBIE_BINARY)
                                        : herbieBinary.getValue();
    if (opts.Program.empty())
      return emitError(UnknownLoc::get(context))
             << "poseidon-rewrite: run-herbie without herbie-binary, and no "
                "Herbie was configured at build time";
    // A platform given as a file is passed as its path, and its contents are
    // part of the cache key; otherwise it is a name Herbie knows.
    opts.Platform.Name = "default";
    StringRef platform = herbiePlatform.getValue();
    if (platform.contains('/') || platform.ends_with(".rkt")) {
      opts.Platform.Path = platform.str();
      opts.Platform.Digest = herbie::readFileDigest(platform);
      if (opts.Platform.Digest.empty())
        return emitError(UnknownLoc::get(context))
               << "poseidon-rewrite: herbie-platform " << platform
               << ": cannot read";
    } else if (!platform.empty()) {
      opts.Platform.Name = platform.str();
    }
    opts.Seed = herbieSeed;
    // The search the LLVM plugin runs by default.
    opts.NumPoints = 1024;
    opts.NumIters = 6;
    opts.NumEnodes = 8000;
    // One budget for the function, which each core's own timeout is also
    // held to.
    opts.SubgraphTimeout = herbieTimeout;
    opts.CoreTimeout = herbieTimeout ? herbieTimeout : 9999;
    opts.NumThreads = herbieThreads;
    opts.Processes = 0;
    opts.CacheDir = herbieCache;
    herbieRun = std::make_shared<const herbie::HerbieRunOptions>(opts);
    return success();
  }

  // Runs Herbie on one function's cores. Its answers by core name; none when
  // it ran out of time; failure when it could not run.
  FailureOr<Answers> askHerbie(func::FuncOp func,
                               const std::vector<std::string> &cores) const {
    herbie::HerbieRunResult run;
    {
      // One function at a time: Herbie already spreads one function's cores
      // over herbie-threads processes, and the runner's record of binary
      // digests is not safe to fill from two threads at once.
      static std::mutex herbieLock;
      std::lock_guard<std::mutex> guard(herbieLock);
      run =
          herbie::runHerbie(cores, *herbieRun, ("@" + func.getSymName()).str());
    }
    switch (run.Result) {
    case herbie::HerbieRunResult::Outcome::Ok:
      break;
    case herbie::HerbieRunResult::Outcome::Timeout:
      func.emitWarning() << "poseidon-rewrite: Herbie ran out of time ("
                         << herbieTimeout << " s); left as is";
      return Answers();
    case herbie::HerbieRunResult::Outcome::Failed:
      func.emitError() << "poseidon-rewrite: could not run Herbie ("
                       << herbieRun->Program << ")";
      return failure();
    }
    std::string error;
    std::optional<Answers> parsed =
        poseidon::parseHerbieResults(run.ResultsJson, error);
    if (!parsed) {
      func.emitError() << "poseidon-rewrite: Herbie's results: " << error;
      return failure();
    }
    return std::move(*parsed);
  }

  // One output and the rewrite chosen for it.
  struct Rewrite {
    Value output;
    poseidon::HerbieExpr expr;
    SmallVector<Value> inputs;
  };

  // Prints Herbie's answer for core `name` (when dumping), and chooses one of
  // its candidates that can be built, at random.
  std::optional<Rewrite> choose(const Answers &known, StringRef name,
                                const Chain &chain, Value output) const {
    auto it = known.find(name);
    if (it == known.end()) {
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
    ExprBuilder builder(*chainOpTable, cast<RankedTensorType>(output.getType()),
                        inputs);
    SmallVector<poseidon::HerbieExpr> buildable;
    for (auto [i, candidate] : llvm::enumerate(answer.candidates)) {
      // Why the candidate cannot be built, when it cannot: unparseable, or no
      // ops for it.
      std::string why;
      std::optional<poseidon::HerbieExpr> expr =
          poseidon::parseHerbieExpr(candidate.expr, why);
      llvm::raw_string_ostream whyOS(why);
      bool canBuild = expr && succeeded(builder.canBuild(*expr, whyOS));
      if (dumpFPCore) {
        llvm::errs() << "      " << i << ": "
                     << llvm::format("%g", candidate.errorBits)
                     << " bits, cost " << llvm::format("%g", candidate.cost)
                     << ": ";
        if (expr)
          expr->print(llvm::errs());
        else
          llvm::errs() << "unparseable: " << candidate.expr;
        if (!canBuild)
          llvm::errs() << " [cannot build: " << why << "]";
        llvm::errs() << "\n";
      }
      if (canBuild)
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
    SmallVector<Chain> chains;
    func.walk([&](Block *block) {
      for (Chain &chain : findChains(*block, *chainOpTable))
        // One op alone is a single rounding, which no rewrite improves.
        if (chain.ops.size() >= 2 && !chain.outputs.empty())
          chains.push_back(std::move(chain));
    });

    // One core per chain output, named after the function and the output's
    // position in it.
    std::vector<std::string> names, cores;
    for (Chain &chain : chains)
      for (Value output : chain.outputs) {
        names.push_back(
            llvm::formatv("{0}/{1}", func.getSymName(), names.size()).str());
        cores.push_back(
            renderFPCore(output, chain, *chainOpTable, names.back()));
      }

    // Herbie's answers: read from herbie-results, or Herbie run now.
    const Answers *found = answers.get();
    std::optional<Answers> ran;
    if (!found && herbieRun && !cores.empty()) {
      FailureOr<Answers> asked = askHerbie(func, cores);
      if (failed(asked))
        return signalPassFailure();
      ran = std::move(*asked);
      found = &*ran;
    }

    // First decide every rewrite, then change the IR, so that what is
    // printed and decided is about the function as it came in.
    SmallVector<Rewrite> rewrites;
    unsigned core = 0;
    for (auto [index, chain] : llvm::enumerate(chains)) {
      if (dumpFPCore) {
        llvm::errs() << "poseidon-rewrite: @" << func.getSymName() << " chain "
                     << index << ": " << chain.ops.size() << " ops, "
                     << chain.outputs.size() << " output(s)\n";
        for (auto [input, i] : chain.inputs) {
          llvm::errs() << "  x" << i << " = ";
          input.printAsOperand(llvm::errs(), OpPrintingFlags());
          llvm::errs() << "\n";
        }
      }
      for (Value output : chain.outputs) {
        const std::string &name = names[core];
        if (dumpFPCore)
          llvm::errs() << "  " << cores[core] << "\n";
        ++core;
        if (found)
          if (std::optional<Rewrite> rewrite =
                  choose(*found, name, chain, output))
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
      ExprBuilder builder(*chainOpTable,
                          cast<RankedTensorType>(rewrite.output.getType()),
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
