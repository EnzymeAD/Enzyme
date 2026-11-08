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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>
#include <memory>
#include <string>

namespace mlir::poseidon {
#define GEN_PASS_DEF_POSEIDONREWRITEPASS
#include "Passes.h.inc"
} // namespace mlir::poseidon

using namespace mlir;

namespace {

// The FPCore operator an op computes, or "" for an op that is not part of a
// chain. Every op here is elementwise: on tensors it applies the scalar
// operation to each element independently, so the scalar FPCore describes the
// whole tensor op.
StringRef fpcoreOperator(Operation *op) {
  return llvm::StringSwitch<StringRef>(op->getName().getStringRef())
      .Case("arith.addf", "+")
      .Case("arith.subf", "-")
      .Case("arith.mulf", "*")
      .Case("arith.divf", "/")
      .Case("arith.negf", "neg")
      .Case("arith.remf", "fmod")
      .Case("arith.maxnumf", "fmax")
      .Case("arith.minnumf", "fmin")
      .Case("math.absf", "fabs")
      .Case("math.sqrt", "sqrt")
      .Case("math.cbrt", "cbrt")
      .Case("math.exp", "exp")
      .Case("math.exp2", "exp2")
      .Case("math.expm1", "expm1")
      .Case("math.log", "log")
      .Case("math.log2", "log2")
      .Case("math.log10", "log10")
      .Case("math.log1p", "log1p")
      .Case("math.powf", "pow")
      .Case("math.fma", "fma")
      .Case("math.sin", "sin")
      .Case("math.cos", "cos")
      .Case("math.tan", "tan")
      .Case("math.asin", "asin")
      .Case("math.acos", "acos")
      .Case("math.atan", "atan")
      .Case("math.atan2", "atan2")
      .Case("math.sinh", "sinh")
      .Case("math.cosh", "cosh")
      .Case("math.tanh", "tanh")
      .Case("math.asinh", "asinh")
      .Case("math.acosh", "acosh")
      .Case("math.atanh", "atanh")
      .Case("math.erf", "erf")
      .Case("math.erfc", "erfc")
      .Case("math.copysign", "copysign")
      .Case("math.floor", "floor")
      .Case("math.ceil", "ceil")
      .Case("math.round", "round")
      .Case("math.trunc", "trunc")
      .Default("");
}

// A ranked tensor of f32 or f64: the precisions Herbie searches in.
bool isFloatTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && (tensor.getElementType().isF32() ||
                    tensor.getElementType().isF64());
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

  // Prints what Herbie answered for core `name`.
  void dumpAnswer(StringRef name) const {
    auto it = answers->find(name);
    if (it == answers->end()) {
      llvm::errs() << "    herbie: no result\n";
      return;
    }
    const poseidon::HerbieAnswer &answer = it->second;
    llvm::errs() << "    herbie: " << answer.status << ", "
                 << answer.candidates.size()
                 << " candidate(s), original error "
                 << llvm::format("%g", answer.initialErrorBits) << " bits\n";
    for (auto [i, candidate] : llvm::enumerate(answer.candidates)) {
      llvm::errs() << "      " << i << ": "
                   << llvm::format("%g", candidate.errorBits) << " bits, cost "
                   << llvm::format("%g", candidate.cost) << ": ";
      std::string error;
      if (std::optional<poseidon::HerbieExpr> expr =
              poseidon::parseHerbieExpr(candidate.expr, error))
        expr->print(llvm::errs());
      else
        llvm::errs() << "unparseable (" << error << "): " << candidate.expr;
      llvm::errs() << "\n";
    }
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
        if (!dumpFPCore)
          continue;
        llvm::errs() << "  " << fpcore << "\n";
        if (answers)
          dumpAnswer(name);
      }
    }
  }
};

} // namespace
