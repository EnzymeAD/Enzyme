//===- FPNode.cpp - the expression graph of a site ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPNode.h"
#include "FPFormat.h"
#include "IR/Utils.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <functional>

using namespace llvm;

namespace poseidon {

static constexpr unsigned kMaxExprDepth = 100;
static constexpr unsigned kMaxInlinedOps = 3;

static Type *fpTypeFromDtype(StringRef dtype, IRBuilder<> &builder) {
  const FPFormatInfo *F = formatOfDtype(dtype);
  return F ? F->LLVMType(builder.getContext()) : nullptr;
}

static std::string libmFuncName(Module *M, Type *Ty, StringRef base) {
  return Target::of(*M).mathFunction(base, Ty);
}

// A call to the math library's `name` on `ops`, all of the type of `ops[0]`,
// named "herbie.<name>".
static Value *mathLibraryCall(IRBuilder<> &b, Module *M, StringRef name,
                              ArrayRef<Value *> ops) {
  Type *Ty = ops[0]->getType();
  std::string funcName = libmFuncName(M, Ty, name);
  SmallVector<Type *, 2> argTys(ops.size(), Ty);
  FunctionCallee f =
      M->getOrInsertFunction(funcName, FunctionType::get(Ty, argTys, false));
  return createMathCall(b, f, ops, "herbie." + name);
}

FPNode::NodeType FPNode::getType() const { return ntype; }

void FPNode::addOperand(std::shared_ptr<FPNode> operand) {
  operands.push_back(operand);
}

void FPNode::updateBounds(double lower, double upper) {
  if (isa<FPConst>(this))
    return;
  lb = std::min(lb, lower);
  ub = std::max(ub, upper);
}

namespace {
struct IntrinsicOp {
  StringLiteral Op;
  Intrinsic::ID ID;
};
struct ComparisonOp {
  StringLiteral Op;
  CmpInst::Predicate Pred;
  StringLiteral Name;
};
struct ConstantOp {
  StringLiteral Op;
  double Value;
};
} // namespace

static constexpr IntrinsicOp kUnaryIntrinsics[] = {
    {"sin", Intrinsic::sin},     {"cos", Intrinsic::cos},
    {"tan", Intrinsic::tan},     {"exp", Intrinsic::exp},
    {"log", Intrinsic::log},     {"sqrt", Intrinsic::sqrt},
    {"fabs", Intrinsic::fabs},   {"asin", Intrinsic::asin},
    {"acos", Intrinsic::acos},   {"atan", Intrinsic::atan},
    {"sinh", Intrinsic::sinh},   {"cosh", Intrinsic::cosh},
    {"tanh", Intrinsic::tanh},   {"ceil", Intrinsic::ceil},
    {"floor", Intrinsic::floor}, {"exp2", Intrinsic::exp2},
    {"log10", Intrinsic::log10}, {"log2", Intrinsic::log2},
    {"rint", Intrinsic::rint},   {"round", Intrinsic::round},
    {"trunc", Intrinsic::trunc},
};

static constexpr IntrinsicOp kBinaryIntrinsics[] = {
    {"fmin", Intrinsic::minnum},
    {"fmax", Intrinsic::maxnum},
    {"atan2", Intrinsic::atan2},
    {"copysign", Intrinsic::copysign},
};

// Operators without an intrinsic every target lowers.
static constexpr StringLiteral kLibraryCalls[] = {
    "expm1", "log1p",  "cbrt",   "hypot", "fdim",  "fmod",  "remainder",
    "erf",   "lgamma", "tgamma", "asinh", "acosh", "atanh",
};

static constexpr ComparisonOp kComparisons[] = {
    {"==", CmpInst::FCMP_OEQ, "eq"}, {"!=", CmpInst::FCMP_ONE, "ne"},
    {"<", CmpInst::FCMP_OLT, "lt"},  {">", CmpInst::FCMP_OGT, "gt"},
    {"<=", CmpInst::FCMP_OLE, "le"}, {">=", CmpInst::FCMP_OGE, "ge"},
};

static constexpr ConstantOp kConstants[] = {
    {"PI", M_PI},
    {"E", M_E},
    {"INFINITY", INFINITY},
    {"NAN", NAN},
};

template <typename Row, size_t N>
static const Row *findOp(const Row (&Table)[N], StringRef Op) {
  const Row *R = find_if(Table, [&](const Row &R) { return R.Op == Op; });
  return R == std::end(Table) ? nullptr : R;
}

static Value *lowerPow(IRBuilder<> &b, Module *M, ArrayRef<Value *> ops) {
  if (auto *CF = dyn_cast<ConstantFP>(ops[1])) {
    double value = CF->getValueAPF().convertToDouble();
    if (value == std::floor(value) && value >= INT_MIN && value <= INT_MAX) {
      int exp = static_cast<int>(value);
      SmallVector<Type *, 1> overloadedTypes = {
          ops[0]->getType(), Type::getInt32Ty(M->getContext())};
      Function *powiFunc = Intrinsic::getOrInsertDeclaration(M, Intrinsic::powi,
                                                             overloadedTypes);
      Value *exponent =
          ConstantInt::get(Type::getInt32Ty(M->getContext()), exp);
      return createMathCall(b, powiFunc, {ops[0], exponent}, "herbie.powi");
    }
  }

  // The intrinsic, as clang emits pow without errno: a libcall may write
  // errno, which keeps the loop from vectorizing and turns pow(C, x) into an
  // errno-writing exp2 call.
  if (!Target::of(*M).isDevice())
    return b.CreateBinaryIntrinsic(Intrinsic::pow, ops[0], ops[1], nullptr,
                                   "herbie.pow");
  return mathLibraryCall(b, M, "pow", ops);
}

Value *FPNode::getLLValue(IRBuilder<> &builder, const ValueToValueMapTy *VMap) {
  Module *M = builder.GetInsertBlock()->getModule();
  if (op == "array")
    report_fatal_error("Poseidon: an array rewrite is lowered element by "
                       "element at its outputs, never as one value");
  if (op == "if") {
    Value *condValue = operands[0]->getLLValue(builder, VMap);
    Value *trueValue = operands[1]->getLLValue(builder, VMap);
    Value *falseValue = operands[2]->getLLValue(builder, VMap);
    // A regime split at one precision may fall back to a bare input of
    // another; both arms take the split's precision.
    Type *armTy = fpTypeFromDtype(dtype, builder);
    if (!armTy)
      armTy = trueValue->getType();
    if (trueValue->getType() != armTy)
      trueValue = builder.CreateFPCast(trueValue, armTy);
    if (falseValue->getType() != armTy)
      falseValue = builder.CreateFPCast(falseValue, armTy);
    return builder.CreateSelect(condValue, trueValue, falseValue,
                                "herbie.select");
  }

  SmallVector<Value *, 3> operandValues;
  Type *targetTy = fpTypeFromDtype(dtype, builder);
  for (auto operand : operands) {
    Value *val = operand->getLLValue(builder, VMap);
    assert(val && "Operand produced a null value!");
    if (targetTy && val->getType()->isFloatingPointTy() &&
        val->getType() != targetTy)
      val = builder.CreateFPCast(val, targetTy);
    operandValues.push_back(val);
  }

  // What the target's math library provides where its intrinsic would not
  // lower.
  if (!operandValues.empty() &&
      Target::of(*M).callsMathLibrary(op, operandValues[0]->getType()))
    return mathLibraryCall(builder, M, op, operandValues);

  if (const IntrinsicOp *I = findOp(kUnaryIntrinsics, op))
    return builder.CreateUnaryIntrinsic(I->ID, operandValues[0], nullptr,
                                        "herbie." + op);
  if (const IntrinsicOp *I = findOp(kBinaryIntrinsics, op))
    return builder.CreateBinaryIntrinsic(
        I->ID, operandValues[0], operandValues[1], nullptr, "herbie." + op);
  if (is_contained(kLibraryCalls, op))
    return mathLibraryCall(builder, M, op, operandValues);
  if (const ComparisonOp *C = findOp(kComparisons, op))
    return builder.CreateFCmp(C->Pred, operandValues[0], operandValues[1],
                              "herbie." + C->Name);
  if (const ConstantOp *C = findOp(kConstants, op))
    return ConstantFP::get(builder.getDoubleTy(), C->Value);
  if (op == "neg")
    return builder.CreateFNeg(operandValues[0], "herbie.neg");
  if (op == "+")
    return builder.CreateFAdd(operandValues[0], operandValues[1], "herbie.add");
  if (op == "-")
    return builder.CreateFSub(operandValues[0], operandValues[1], "herbie.sub");
  if (op == "*")
    return builder.CreateFMul(operandValues[0], operandValues[1], "herbie.mul");
  if (op == "/")
    return builder.CreateFDiv(operandValues[0], operandValues[1], "herbie.div");
  if (op == "pow")
    return lowerPow(builder, M, operandValues);
  if (op == "fma")
    return builder.CreateIntrinsic(Intrinsic::fma,
                                   {operandValues[0]->getType()}, operandValues,
                                   nullptr, "herbie.fma");
  if (op == "and")
    return builder.CreateAnd(operandValues[0], operandValues[1], "herbie.and");
  if (op == "or")
    return builder.CreateOr(operandValues[0], operandValues[1], "herbie.or");
  if (op == "not")
    return builder.CreateNot(operandValues[0], "herbie.not");
  if (op == "TRUE")
    return ConstantInt::getTrue(builder.getContext());
  if (op == "FALSE")
    return ConstantInt::getFalse(builder.getContext());
  // CreateFPCast rather than FPTrunc/FPExt: the operand may already carry the
  // destination type, coerced to the node's dtype above, and FPCast is then the
  // identity.
  if (op == "binary64->binary32")
    return builder.CreateFPCast(operandValues[0], builder.getFloatTy(),
                                "herbie.f64tof32");
  if (op == "binary32->binary64")
    return builder.CreateFPCast(operandValues[0], builder.getDoubleTy(),
                                "herbie.f32tof64");
  std::string msg = "FPNode getLLValue: Unexpected operator " + op;
  llvm_unreachable(msg.c_str());
}

// Memoized: an input is consulted once per input defined over it, so a chain
// of inputs one operation apart (T^2, T^3, ...) would otherwise be re-walked
// exponentially often.
static unsigned inlinedOps(const FPLLValue &V, const SetVector<Value *> &inputs,
                           DenseMap<const FPLLValue *, unsigned> &memo) {
  if (auto it = memo.find(&V); it != memo.end())
    return it->second;
  unsigned ops = 0;
  std::function<bool(const FPNode &)> over = [&](const FPNode &N) {
    if (isa<FPConst>(N))
      return true;
    if (&N != &V && inputs.contains(cast<FPLLValue>(N).value) &&
        !inlinedOps(cast<FPLLValue>(N), inputs, memo))
      return true;
    if (N.operands.empty() || ++ops > kMaxInlinedOps)
      return false;
    return llvm::all_of(N.operands, [&](const auto &O) { return over(*O); });
  };
  return memo[&V] = over(V) ? ops : 0;
}

unsigned inlinedOps(const FPLLValue &V, const SetVector<Value *> &inputs) {
  DenseMap<const FPLLValue *, unsigned> memo;
  return inlinedOps(V, inputs, memo);
}

std::shared_ptr<FPNode> existingDefinition(
    const FPNode &N, const SetVector<Value *> &inputs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap) {
  auto same = [](const FPNode &A, const FPNode &B) {
    auto *CA = dyn_cast<FPConst>(&A), *CB = dyn_cast<FPConst>(&B);
    return &A == &B || (CA && CB && CA->dtype == CB->dtype &&
                        CA->getLowerBound() == CB->getLowerBound());
  };
  auto matches = [&](const FPNode &W, bool swapped) {
    if (W.op != N.op || W.dtype != N.dtype ||
        W.operands.size() != N.operands.size())
      return false;
    for (size_t i = 0; i < W.operands.size(); ++i)
      if (!same(*W.operands[swapped ? W.operands.size() - 1 - i : i],
                *N.operands[i]))
        return false;
    return true;
  };
  bool commutes = (N.op == "+" || N.op == "*") && N.operands.size() == 2;
  for (Value *V : inputs) {
    auto *VN = dyn_cast<FPLLValue>(valueToNodeMap.at(V).get());
    if (!VN || !inlinedOps(*VN, inputs))
      continue;
    SmallVector<const FPLLValue *, 4> todo{VN};
    while (!todo.empty()) {
      const FPLLValue *W = todo.pop_back_val();
      // An erased value (its output was rewritten) is recomputed instead.
      if (W->value && (matches(*W, false) || (commutes && matches(*W, true))))
        return valueToNodeMap.at(W->value);
      for (const auto &O : W->operands)
        if (auto *L = dyn_cast<FPLLValue>(O.get());
            L && !L->operands.empty() && !inputs.contains(L->value))
          todo.push_back(L);
    }
  }
  return nullptr;
}

std::string FPNode::toFullExpression(
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &subgraphInputs, unsigned depth) {
  if (auto *C = dyn_cast<FPConst>(this))
    return C->literal();
  auto *V = dyn_cast<FPLLValue>(this);
  if (!V)
    llvm_unreachable("an operator node of a parsed rewrite has no expression "
                     "of the original graph");
  Value *value = V->value;
  if (subgraphInputs.contains(value) && !inlinedOps(*V, subgraphInputs)) {
    assert(hasSymbol() && "FPLLValue has no symbol!");
    return symbol;
  }
  assert(!operands.empty() && "FPNode has no operands!");

  if (depth > kMaxExprDepth)
    reportUserError(*cast<Instruction>(value),
                    "Expression depth exceeded maximum allowed depth of " +
                        Twine(kMaxExprDepth) + " for " + op +
                        "; consider disabling loop unrolling");

  std::string expr = "(" + (op == "neg" ? "-" : op);
  for (auto operand : operands) {
    expr += " " + operand->toFullExpression(valueToNodeMap, subgraphInputs,
                                            depth + 1);
  }
  expr += ")";
  return expr;
}

Value *FPLLValue::getLLValue(IRBuilder<> &builder,
                             const ValueToValueMapTy *VMap) {
  if (!value)
    report_fatal_error("Poseidon: a rewrite names a value an earlier step "
                       "erased");
  if (VMap) {
    assert(VMap->count(value) && "FPLLValue not found in passed-in VMap!");
    return VMap->lookup(value);
  }
  return value;
}

bool FPLLValue::classof(const FPNode *N) {
  return N->getType() == NodeType::LLValue;
}

void FPGraph::replaceErased(Value *erased, Value *now) {
  auto it = valueToNodeMap.find(erased);
  if (it == valueToNodeMap.end())
    return;
  auto *node = cast<FPLLValue>(it->second.get());
  if (node->hasSymbol()) {
    symbolToValueMap[node->symbol] = now;
    valueToNodeMap[now] = std::make_shared<FPLLValue>(now, "__no", node->dtype);
  }
  node->value = nullptr;
}

double FPConst::literalValue(const std::string &strValue) {
  if (strValue == "+inf.0") {
    return std::numeric_limits<double>::infinity();
  }
  if (strValue == "-inf.0") {
    return -std::numeric_limits<double>::infinity();
  }

  return literalToDouble(strValue);
}

Value *FPConst::getLLValue(IRBuilder<> &builder,
                           const ValueToValueMapTy *VMap) {
  Type *Ty = fpTypeFromDtype(dtype, builder);
  if (!Ty) {
    std::string msg = "FPConst getValue: Unexpected dtype: " + dtype;
    llvm_unreachable(msg.c_str());
  }
  if (strValue == "+inf.0") {
    return ConstantFP::getInfinity(Ty, false);
  }
  if (strValue == "-inf.0") {
    return ConstantFP::getInfinity(Ty, true);
  }

  return ConstantFP::get(Ty, literalToDouble(strValue));
}

bool FPConst::classof(const FPNode *N) {
  return N->getType() == NodeType::Const;
}

SmallVector<Value *, 4>
emitSharedElements(ArrayRef<std::shared_ptr<FPNode>> elements,
                   function_ref<IRBuilder<> &(size_t)> at,
                   ValueToValueMapTy *VMap, const DominatorTree *DT) {
  std::unordered_map<Value *, std::shared_ptr<FPNode>> noNodes;
  SetVector<Value *> noInputs;
  DenseMap<const FPNode *, std::string> keys;
  std::function<const std::string &(const FPNode *)> key =
      [&](const FPNode *N) -> const std::string & {
    auto It = keys.find(N);
    if (It != keys.end())
      return It->second;
    std::string k;
    if (auto *LL = dyn_cast<FPLLValue>(N))
      k = "%" + utohexstr((uintptr_t)LL->value);
    else if (isa<FPConst>(N))
      k = N->dtype + ":" +
          const_cast<FPNode *>(N)->toFullExpression(noNodes, noInputs);
    else {
      k = N->op + "." + N->dtype + "(";
      for (const auto &operand : N->operands)
        k += key(operand.get()) + ",";
      k += ")";
    }
    return keys[N] = std::move(k);
  };
  // Inputs lower to their own value and constants to themselves: only the
  // operator nodes of a rewrite are emitted.
  auto isLeaf = [](const FPNode *N) {
    return N->getType() != FPNode::NodeType::Node;
  };
  StringMap<unsigned> counts;
  std::function<void(const FPNode *)> count = [&](const FPNode *N) {
    if (isLeaf(N))
      return;
    ++counts[key(N)];
    for (const auto &operand : N->operands)
      count(operand.get());
  };
  for (const auto &element : elements)
    count(element.get());

  StringMap<Value *> emitted;
  auto placeholder = [&](Value *V, const std::string &dtype) {
    if (VMap)
      (*VMap)[V] = V;
    return std::make_shared<FPLLValue>(V, "__shared", dtype);
  };
  SmallVector<Value *, 4> out;
  for (auto [k, element] : llvm::enumerate(elements)) {
    IRBuilder<> &B = at(k);
    auto available = [&](Value *V) {
      auto *I = dyn_cast<Instruction>(V);
      if (!I || !DT)
        return true;
      return DT->dominates(I, &*B.GetInsertPoint());
    };
    std::function<std::shared_ptr<FPNode>(std::shared_ptr<FPNode>)> share =
        [&](std::shared_ptr<FPNode> N) -> std::shared_ptr<FPNode> {
      if (isLeaf(N.get()))
        return N;
      const std::string &k = key(N.get());
      for (auto &operand : N->operands)
        operand = share(operand);
      if (counts[k] < 2)
        return N;
      auto It = emitted.find(k);
      if (It != emitted.end() && available(It->second))
        return placeholder(It->second, N->dtype);
      Value *V = N->getLLValue(B, VMap);
      emitted[k] = V;
      return placeholder(V, N->dtype);
    };
    out.push_back(share(element)->getLLValue(B, VMap));
  }
  return out;
}

} // namespace poseidon
