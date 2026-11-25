//===- FPNode.cpp - the expression graph of a site ------------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPNode.h"
#include "FPCore.h"
#include "FPFormat.h"
#include "IR/Utils.h"
#include "Targets/MathLibrary.h"
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
#include <cstdlib>
#include <functional>
#include <mpfr.h>

using namespace llvm;

namespace poseidon {

static constexpr unsigned kMaxExprDepth = 100;
static constexpr unsigned kMaxInlinedOps = 3;

// The LLVM type of `format`; null for none.
static Type *llvmTypeOf(const FPFormatInfo *format, IRBuilder<> &builder) {
  return format ? format->LLVMType(builder.getContext()) : nullptr;
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

bool FPNode::hasPrecision(PrecisionChangeType P) const {
  return format && format->Precision == P;
}

void FPNode::addOperand(FPNode *operand) { operands.push_back(operand); }

void FPNode::updateBounds(double lower, double upper) {
  if (isa<FPConst>(this))
    return;
  lb = std::min(lb, lower);
  ub = std::max(ub, upper);
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

// The operation `O` on `ops`, for an operator the table lowers by name.
static Value *lowerByName(IRBuilder<> &b, Module *M, const FPCoreOperator &O,
                          ArrayRef<Value *> ops) {
  if (&O == &fpcore::Neg)
    return b.CreateFNeg(ops[0], "herbie.neg");
  if (&O == &fpcore::Add)
    return b.CreateFAdd(ops[0], ops[1], "herbie.add");
  if (&O == &fpcore::Sub)
    return b.CreateFSub(ops[0], ops[1], "herbie.sub");
  if (&O == &fpcore::Mul)
    return b.CreateFMul(ops[0], ops[1], "herbie.mul");
  if (&O == &fpcore::Div)
    return b.CreateFDiv(ops[0], ops[1], "herbie.div");
  if (&O == &fpcore::Pow)
    return lowerPow(b, M, ops);
  if (&O == &fpcore::Fma)
    return b.CreateIntrinsic(Intrinsic::fma, {ops[0]->getType()}, ops, nullptr,
                             "herbie.fma");
  if (&O == &fpcore::And)
    return b.CreateAnd(ops[0], ops[1], "herbie.and");
  if (&O == &fpcore::Or)
    return b.CreateOr(ops[0], ops[1], "herbie.or");
  if (&O == &fpcore::Not)
    return b.CreateNot(ops[0], "herbie.not");
  if (&O == &fpcore::True || &O == &fpcore::False)
    return ConstantInt::getBool(b.getContext(), &O == &fpcore::True);
  // CreateFPCast rather than FPTrunc/FPExt: the operand may already carry the
  // destination type, coerced to the node's format, and FPCast is then the
  // identity.
  if (&O == &fpcore::ToBinary32)
    return b.CreateFPCast(ops[0], b.getFloatTy(), "herbie.f64tof32");
  assert(&O == &fpcore::ToBinary64 && "the parser admits no other operator");
  return b.CreateFPCast(ops[0], b.getDoubleTy(), "herbie.f32tof64");
}

Value *FPNode::lowerChoice(IRBuilder<> &builder,
                           const ValueToValueMapTy *VMap) {
  Value *condValue = operands[0]->getLLValue(builder, VMap);
  Value *trueValue = operands[1]->getLLValue(builder, VMap);
  Value *falseValue = operands[2]->getLLValue(builder, VMap);
  // A regime split at one precision may fall back to a bare input of
  // another; both arms take the split's precision.
  Type *armTy = llvmTypeOf(format, builder);
  if (!armTy)
    armTy = trueValue->getType();
  if (trueValue->getType() != armTy)
    trueValue = builder.CreateFPCast(trueValue, armTy);
  if (falseValue->getType() != armTy)
    falseValue = builder.CreateFPCast(falseValue, armTy);
  return builder.CreateSelect(condValue, trueValue, falseValue,
                              "herbie.select");
}

Value *FPNode::getLLValue(IRBuilder<> &builder, const ValueToValueMapTy *VMap) {
  Module *M = builder.GetInsertBlock()->getModule();
  assert(op && "a parsed rewrite's operation has an operator");
  assert(!is(fpcore::Array) && "an array rewrite is lowered element by element "
                               "at its outputs, never as one value");
  if (is(fpcore::If))
    return lowerChoice(builder, VMap);

  SmallVector<Value *, 3> operandValues;
  Type *targetTy = llvmTypeOf(format, builder);
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
  const FPCoreOperator *O = op;
  if (!operandValues.empty() &&
      Target::of(*M).callsMathLibrary(O->Name, operandValues[0]->getType()))
    return mathLibraryCall(builder, M, O->Name, operandValues);

  if (O->Intrinsic != Intrinsic::not_intrinsic && O->Arity == 1)
    return builder.CreateUnaryIntrinsic(O->Intrinsic, operandValues[0], nullptr,
                                        "herbie." + O->Name);
  if (O->Intrinsic != Intrinsic::not_intrinsic)
    return builder.CreateBinaryIntrinsic(O->Intrinsic, operandValues[0],
                                         operandValues[1], nullptr,
                                         "herbie." + O->Name);
  if (O->LibraryCall)
    return mathLibraryCall(builder, M, O->Name, operandValues);
  if (O->Predicate != CmpInst::BAD_FCMP_PREDICATE)
    return builder.CreateFCmp(O->Predicate, operandValues[0], operandValues[1],
                              "herbie." + O->LoweredName);
  if (O->MPFRConstant)
    return ConstantFP::get(builder.getDoubleTy(), O->Constant);
  return lowerByName(builder, M, *O, operandValues);
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

FPNode *existingDefinition(const FPNode &N, const SetVector<Value *> &inputs,
                           const FPGraph &G) {
  auto same = [](const FPNode &A, const FPNode &B) {
    auto *CA = dyn_cast<FPConst>(&A), *CB = dyn_cast<FPConst>(&B);
    return &A == &B || (CA && CB && CA->format == CB->format &&
                        CA->getLowerBound() == CB->getLowerBound());
  };
  auto matches = [&](const FPNode &W, bool swapped) {
    if (W.op != N.op || W.format != N.format ||
        W.operands.size() != N.operands.size())
      return false;
    for (size_t i = 0; i < W.operands.size(); ++i)
      if (!same(*W.operands[swapped ? W.operands.size() - 1 - i : i],
                *N.operands[i]))
        return false;
    return true;
  };
  bool commutes =
      (N.is(fpcore::Add) || N.is(fpcore::Mul)) && N.operands.size() == 2;
  for (Value *V : inputs) {
    auto *VN = dyn_cast<FPLLValue>(G.valueToNodeMap.at(V));
    if (!VN || !inlinedOps(*VN, inputs))
      continue;
    SmallVector<const FPLLValue *, 4> todo{VN};
    while (!todo.empty()) {
      const FPLLValue *W = todo.pop_back_val();
      // An erased value (its output was rewritten) is recomputed instead.
      if (W->value && (matches(*W, false) || (commutes && matches(*W, true))))
        return G.valueToNodeMap.at(W->value);
      for (const auto &O : W->operands)
        if (auto *L = dyn_cast<FPLLValue>(O);
            L && !L->operands.empty() && !inputs.contains(L->value))
          todo.push_back(L);
    }
  }
  return nullptr;
}

std::string FPNode::toFullExpression(const FPGraph &G,
                                     const SetVector<Value *> &subgraphInputs,
                                     unsigned depth) {
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
                        Twine(kMaxExprDepth) + " for " + operationName(*this) +
                        "; consider disabling loop unrolling");

  std::string expr = "(" + (is(fpcore::Neg) ? "-" : operationName(*this));
  for (auto operand : operands) {
    expr += " " + operand->toFullExpression(G, subgraphInputs, depth + 1);
  }
  expr += ")";
  return expr;
}

Value *FPLLValue::getLLValue(IRBuilder<> &builder,
                             const ValueToValueMapTy *VMap) {
  assert(value && "a rewrite names a value an earlier step erased");
  if (VMap) {
    assert(VMap->count(value) && "FPLLValue not found in passed-in VMap!");
    return VMap->lookup(value);
  }
  return value;
}

bool FPLLValue::classof(const FPNode *N) {
  return N->getType() == NodeType::LLValue;
}

FPGraph FPGraph::overlay() const {
  FPGraph G;
  G.F = F;
  G.valueToNodeMap = valueToNodeMap;
  G.symbolToValueMap = symbolToValueMap;
  return G;
}

void FPGraph::replaceErased(Value *erased, Value *now) {
  auto it = valueToNodeMap.find(erased);
  if (it == valueToNodeMap.end())
    return;
  auto *node = cast<FPLLValue>(it->second);
  if (node->hasSymbol()) {
    symbolToValueMap[node->symbol] = now;
    valueToNodeMap[now] = create<FPLLValue>(now, nullptr, node->format);
  }
  node->value = nullptr;
}

// An FPCore literal, a decimal or an exact rational, as the nearest double.
static double literalToDouble(const std::string &str) {
  size_t div = str.find('/');
  if (div == std::string::npos)
    return std::strtod(str.c_str(), nullptr);
  mpfr_prec_t exact = 4 * static_cast<mpfr_prec_t>(str.size()) + 64;
  mpfr_t num, den, quot;
  mpfr_init2(num, exact);
  mpfr_init2(den, exact);
  mpfr_init2(quot, 128);
  mpfr_set_str(num, str.substr(0, div).c_str(), 10, MPFR_RNDN);
  mpfr_set_str(den, str.substr(div + 1).c_str(), 10, MPFR_RNDN);
  mpfr_div(quot, num, den, MPFR_RNDN);
  double result = mpfr_get_d(quot, MPFR_RNDN);
  mpfr_clear(num);
  mpfr_clear(den);
  mpfr_clear(quot);
  return result;
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
  Type *Ty = llvmTypeOf(format, builder);
  assert(Ty && "a literal has a format of an LLVM type");
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
emitSharedElements(FPGraph &G, ArrayRef<FPNode *> elements,
                   function_ref<IRBuilder<> &(size_t)> at,
                   ValueToValueMapTy *VMap, const DominatorTree *DT) {
  DenseMap<const FPNode *, std::string> keys;
  auto formatName = [](const FPNode *N) {
    return N->format ? N->format->Dtype.str() : std::string();
  };
  std::function<const std::string &(const FPNode *)> key =
      [&](const FPNode *N) -> const std::string & {
    auto It = keys.find(N);
    if (It != keys.end())
      return It->second;
    std::string k;
    if (auto *LL = dyn_cast<FPLLValue>(N))
      k = "%" + utohexstr((uintptr_t)LL->value);
    else if (isa<FPConst>(N))
      k = formatName(N) + ":" + cast<FPConst>(N)->literal();
    else {
      k = N->op->Name.str() + "." + formatName(N) + "(";
      for (const auto &operand : N->operands)
        k += key(operand) + ",";
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
      count(operand);
  };
  for (const auto &element : elements)
    count(element);

  StringMap<Value *> emitted;
  auto placeholder = [&](Value *V, const FPFormatInfo *format) {
    if (VMap)
      (*VMap)[V] = V;
    return G.create<FPLLValue>(V, nullptr, format);
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
    std::function<FPNode *(FPNode *)> share = [&](FPNode *N) -> FPNode * {
      if (isLeaf(N))
        return N;
      const std::string &k = key(N);
      for (auto &operand : N->operands)
        operand = share(operand);
      if (counts[k] < 2)
        return N;
      auto It = emitted.find(k);
      if (It != emitted.end() && available(It->second))
        return placeholder(It->second, N->format);
      Value *V = N->getLLValue(B, VMap);
      emitted[k] = V;
      return placeholder(V, N->format);
    };
    out.push_back(share(element)->getLLValue(B, VMap));
  }
  return out;
}

} // namespace poseidon
