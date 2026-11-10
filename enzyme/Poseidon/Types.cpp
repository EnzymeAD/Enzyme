//=- Types.cpp - AST node implementations for Poseidon --------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST node classes for representing floating-point
// expressions in the Poseidon optimization pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <optional>

#include "CostModel.h"
#include "Herbie.h"
#include "Optimize.h"
#include "Pass/Session.h"
#include "Solvers.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Techniques/Precision.h"
#include "Types.h"
#include "Utils.h"

using namespace llvm;

namespace poseidon {

static constexpr unsigned kMaxExprDepth = 100;
static constexpr unsigned kMaxInlinedOps = 3;

static Type *fpTypeFromDtype(StringRef dtype, IRBuilder<> &builder) {
  if (dtype == "f16")
    return builder.getHalfTy();
  if (dtype == "bf16")
    return builder.getBFloatTy();
  if (dtype == "f32")
    return builder.getFloatTy();
  if (dtype == "f64")
    return builder.getDoubleTy();
  return nullptr;
}

static std::string libmFuncName(Module *M, Type *Ty, StringRef base) {
  return Target::of(*M).mathFunction(base, Ty);
}

FPNode::NodeType FPNode::getType() const { return ntype; }

void FPNode::addOperand(std::shared_ptr<FPNode> operand) {
  operands.push_back(operand);
}

bool FPNode::hasSymbol() const {
  std::string msg = "Unexpected invocation of `hasSymbol` on an "
                    "unmaterialized " +
                    op + " FPNode";
  llvm_unreachable(msg.c_str());
}

std::string FPNode::toFullExpression(
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &subgraphInputs, unsigned depth) {
  std::string msg = "Unexpected invocation of `toFullExpression` on an "
                    "unmaterialized " +
                    op + " FPNode";
  llvm_unreachable(msg.c_str());
}

unsigned FPNode::getMPFRPrec() const {
  if (dtype == "f16")
    return 11;
  if (dtype == "bf16")
    return 8;
  if (dtype == "f32")
    return 24;
  if (dtype == "f64")
    return 53;
  // Herbie-dialect spellings (see FPEvaluator::getNodePrecision).
  if (dtype == "binary32")
    return 24;
  if (dtype == "binary64")
    return 53;
  std::string msg =
      "getMPFRPrec: operator " + op + " has unknown dtype " + dtype;
  llvm_unreachable(msg.c_str());
}

void FPNode::updateBounds(double lower, double upper, bool print) {
  std::string msg = "Unexpected invocation of `updateBounds` on an "
                    "unmaterialized " +
                    op + " FPNode";
  llvm_unreachable(msg.c_str());
}

double FPNode::getLowerBound() const {
  std::string msg = "Unexpected invocation of `getLowerBound` on an "
                    "unmaterialized " +
                    op + " FPNode";
  llvm_unreachable(msg.c_str());
}

double FPNode::getUpperBound() const {
  std::string msg = "Unexpected invocation of `getUpperBound` on an "
                    "unmaterialized " +
                    op + " FPNode";
  llvm_unreachable(msg.c_str());
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

  static const std::unordered_map<
      std::string, std::function<Value *(IRBuilder<> &, Module *,
                                         const SmallVectorImpl<Value *> &)>>
      opMap = {
          {"neg",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &ops)
               -> Value * { return b.CreateFNeg(ops[0], "herbie.neg"); }},
          {"+",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFAdd(ops[0], ops[1], "herbie.add");
           }},
          {"-",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFSub(ops[0], ops[1], "herbie.sub");
           }},
          {"*",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFMul(ops[0], ops[1], "herbie.mul");
           }},
          {"/",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFDiv(ops[0], ops[1], "herbie.div");
           }},
          {"fmin",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateBinaryIntrinsic(Intrinsic::minnum, ops[0], ops[1],
                                            nullptr, "herbie.fmin");
           }},
          {"fmax",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateBinaryIntrinsic(Intrinsic::maxnum, ops[0], ops[1],
                                            nullptr, "herbie.fmax");
           }},
          {"sin",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::sin, ops[0], nullptr,
                                           "herbie.sin");
           }},
          {"cos",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::cos, ops[0], nullptr,
                                           "herbie.cos");
           }},
          {"tan",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::tan, ops[0], nullptr,
                                           "herbie.tan");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "tan");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee tanFunc = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, tanFunc, {ops[0]}, "herbie.tan");
#endif
           }},
          {"exp",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::exp, ops[0], nullptr,
                                           "herbie.exp");
           }},
          {"expm1",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "expm1");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.expm1");
           }},
          {"log",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::log, ops[0], nullptr,
                                           "herbie.log");
           }},
          {"log1p",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "log1p");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.log1p");
           }},
          {"sqrt",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::sqrt, ops[0], nullptr,
                                           "herbie.sqrt");
           }},
          {"cbrt",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "cbrt");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.cbrt");
           }},
          {"pow",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             if (auto *CF = dyn_cast<ConstantFP>(ops[1])) {
               double value = CF->getValueAPF().convertToDouble();
               if (value == std::floor(value) && value >= INT_MIN &&
                   value <= INT_MAX) {
                 int exp = static_cast<int>(value);
                 SmallVector<Type *, 1> overloadedTypes = {
                     ops[0]->getType(), Type::getInt32Ty(M->getContext())};
                 Function *powiFunc = Intrinsic::getOrInsertDeclaration(
                     M, Intrinsic::powi, overloadedTypes);
                 Value *exponent =
                     ConstantInt::get(Type::getInt32Ty(M->getContext()), exp);
                 return createMathCall(b, powiFunc, {ops[0], exponent},
                                       "herbie.powi");
               }
             }

             // The intrinsic, as clang emits pow without errno: a libcall
             // may write errno, which keeps the loop from vectorizing and
             // turns pow(C, x) into an errno-writing exp2 call.
             if (!Target::of(*M).isDevice())
               return b.CreateBinaryIntrinsic(Intrinsic::pow, ops[0], ops[1],
                                              nullptr, "herbie.pow");
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "pow");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.pow");
           }},
          {"fma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateIntrinsic(Intrinsic::fma, {ops[0]->getType()},
                                      {ops[0], ops[1], ops[2]}, nullptr,
                                      "herbie.fma");
           }},
          {"fabs",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::fabs, ops[0], nullptr,
                                           "herbie.fabs");
           }},
          {"hypot",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "hypot");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.hypot");
           }},
          {"asin",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::asin, ops[0], nullptr,
                                           "herbie.asin");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "asin");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.asin");
#endif
           }},
          {"acos",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::acos, ops[0], nullptr,
                                           "herbie.acos");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "acos");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.acos");
#endif
           }},
          {"atan",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::atan, ops[0], nullptr,
                                           "herbie.atan");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "atan");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.atan");
#endif
           }},
          {"atan2",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateBinaryIntrinsic(Intrinsic::atan2, ops[0], ops[1],
                                            nullptr, "herbie.atan2");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "atan2");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.atan2");
#endif
           }},
          {"sinh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::sinh, ops[0], nullptr,
                                           "herbie.sinh");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "sinh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.sinh");
#endif
           }},
          {"cosh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::cosh, ops[0], nullptr,
                                           "herbie.cosh");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "cosh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.cosh");
#endif
           }},
          {"tanh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::tanh, ops[0], nullptr,
                                           "herbie.tanh");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "tanh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.tanh");
#endif
           }},
          {"copysign",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateBinaryIntrinsic(Intrinsic::copysign, ops[0], ops[1],
                                            nullptr, "herbie.copysign");
           }},
          {"rem",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFRem(ops[0], ops[1], "herbie.rem");
           }},
          {"ceil",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::ceil, ops[0], nullptr,
                                           "herbie.ceil");
           }},
          {"floor",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::floor, ops[0], nullptr,
                                           "herbie.floor");
           }},
          {"exp2",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::exp2, ops[0], nullptr,
                                           "herbie.exp2");
           }},
          {"log10",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::log10, ops[0], nullptr,
                                           "herbie.log10");
           }},
          {"log2",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::log2, ops[0], nullptr,
                                           "herbie.log2");
           }},
          {"rint",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::rint, ops[0], nullptr,
                                           "herbie.rint");
           }},
          {"round",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::round, ops[0], nullptr,
                                           "herbie.round");
           }},
          {"trunc",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateUnaryIntrinsic(Intrinsic::trunc, ops[0], nullptr,
                                           "herbie.trunc");
           }},
          {"fdim",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "fdim");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.fdim");
           }},
          {"fmod",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "fmod");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.fmod");
           }},
          {"remainder",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "remainder");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0], ops[1]}, "herbie.remainder");
           }},
          {"erf",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "erf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.erf");
           }},
          {"lgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "lgamma");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.lgamma");
           }},
          {"tgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "tgamma");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.tgamma");
           }},
          {"asinh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "asinh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.asinh");
           }},
          {"acosh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "acosh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.acosh");
           }},
          {"atanh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "atanh");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return createMathCall(b, f, {ops[0]}, "herbie.atanh");
           }},
          {"==",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpOEQ(ops[0], ops[1], "herbie.eq");
           }},
          {"!=",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpONE(ops[0], ops[1], "herbie.ne");
           }},
          {"<",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpOLT(ops[0], ops[1], "herbie.lt");
           }},
          {">",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpOGT(ops[0], ops[1], "herbie.gt");
           }},
          {"<=",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpOLE(ops[0], ops[1], "herbie.le");
           }},
          {">=",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFCmpOGE(ops[0], ops[1], "herbie.ge");
           }},
          {"and",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateAnd(ops[0], ops[1], "herbie.and");
           }},
          {"or",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &ops)
               -> Value * { return b.CreateOr(ops[0], ops[1], "herbie.or"); }},
          {"not",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &ops)
               -> Value * { return b.CreateNot(ops[0], "herbie.not"); }},
          {"TRUE",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &)
               -> Value * { return ConstantInt::getTrue(b.getContext()); }},
          {"FALSE",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &)
               -> Value * { return ConstantInt::getFalse(b.getContext()); }},
          {"PI",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &)
               -> Value * { return ConstantFP::get(b.getDoubleTy(), M_PI); }},
          {"E",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &)
               -> Value * { return ConstantFP::get(b.getDoubleTy(), M_E); }},
          {"INFINITY",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &) -> Value * {
             return ConstantFP::getInfinity(b.getDoubleTy(), false);
           }},
          {"NaN",
           [](IRBuilder<> &b, Module *M, const SmallVectorImpl<Value *> &)
               -> Value * { return ConstantFP::getNaN(b.getDoubleTy()); }},
          // Herbie may emit the explicit conversions the FPCores request via
          // :herbie-conversions. CreateFPCast rather than FPTrunc/FPExt: the
          // operand may already carry the destination type (operands are
          // coerced to the node's dtype above), and FPCast is then the
          // identity.
          {"binary64->binary32",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFPCast(ops[0], b.getFloatTy(), "herbie.f64tof32");
           }},
          {"binary32->binary64",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateFPCast(ops[0], b.getDoubleTy(), "herbie.f32tof64");
           }},
      };

  // What the target's math library provides where its intrinsic would not
  // lower; the rest falls through to opMap.
  if (const Target &T = Target::of(*M);
      !operandValues.empty() &&
      T.callsMathLibrary(op, operandValues[0]->getType())) {
    Type *Ty = operandValues[0]->getType();
    SmallVector<Type *, 2> argTys(operandValues.size(), Ty);
    FunctionCallee fn = M->getOrInsertFunction(
        T.mathFunction(op, Ty), FunctionType::get(Ty, argTys, false));
    return createMathCall(builder, fn, operandValues, "herbie." + op);
  }

  auto it = opMap.find(op);
  if (it != opMap.end())
    return it->second(builder, M, operandValues);
  else {
    std::string msg = "FPNode getLLValue: Unexpected operator " + op;
    llvm_unreachable(msg.c_str());
  }
}

bool FPLLValue::hasSymbol() const { return !symbol.empty(); }

// Memoized: an input is consulted once per input defined over it, so a chain
// of inputs one operation apart (T^2, T^3, ... from sympy CSE) would otherwise
// be re-walked exponentially often.
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

std::string FPLLValue::toFullExpression(
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &subgraphInputs, unsigned depth) {
  if (subgraphInputs.contains(value) && !inlinedOps(*this, subgraphInputs)) {
    assert(hasSymbol() && "FPLLValue has no symbol!");
    return symbol;
  } else {
    assert(!operands.empty() && "FPNode has no operands!");

    if (depth > kMaxExprDepth) {
      std::string msg = "Expression depth exceeded maximum allowed depth of " +
                        std::to_string(kMaxExprDepth) + " for " + op +
                        "; consider disabling loop unrolling";

      llvm_unreachable(msg.c_str());
    }

    std::string expr = "(" + (op == "neg" ? "-" : op);
    for (auto operand : operands) {
      expr += " " + operand->toFullExpression(valueToNodeMap, subgraphInputs,
                                              depth + 1);
    }
    expr += ")";
    return expr;
  }
}

void FPLLValue::updateBounds(double lower, double upper, bool print) {
  lb = std::min(lb, lower);
  ub = std::max(ub, upper);
  if (print)
    llvm::errs() << "Updated bounds for " << *value << ": [" << lb << ", " << ub
                 << "]\n";
}

double FPLLValue::getLowerBound() const { return lb; }
double FPLLValue::getUpperBound() const { return ub; }

Value *FPLLValue::getLLValue(IRBuilder<> &builder,
                             const ValueToValueMapTy *VMap) {
  if (VMap) {
    assert(VMap->count(value) && "FPLLValue not found in passed-in VMap!");
    return VMap->lookup(value);
  }
  return value;
}

bool FPLLValue::classof(const FPNode *N) {
  return N->getType() == NodeType::LLValue;
}

std::string FPConst::toFullExpression(
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &subgraphInputs, unsigned depth) {
  return strValue;
}

bool FPConst::hasSymbol() const {
  std::string msg = "Unexpected invocation of `hasSymbol` on an FPConst";
  llvm_unreachable(msg.c_str());
}

void FPConst::updateBounds(double lower, double upper, bool print) { return; }

double FPConst::getLowerBound() const {
  if (strValue == "+inf.0") {
    return std::numeric_limits<double>::infinity();
  } else if (strValue == "-inf.0") {
    return -std::numeric_limits<double>::infinity();
  }

  return literalToDouble(strValue);
}

double FPConst::getUpperBound() const { return getLowerBound(); }

Value *FPConst::getLLValue(IRBuilder<> &builder,
                           const ValueToValueMapTy *VMap) {
  Type *Ty = fpTypeFromDtype(dtype, builder);
  if (!Ty) {
    std::string msg = "FPConst getValue: Unexpected dtype: " + dtype;
    llvm_unreachable(msg.c_str());
  }
  if (strValue == "+inf.0") {
    return ConstantFP::getInfinity(Ty, false);
  } else if (strValue == "-inf.0") {
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

void RewriteUnit::apply(
    size_t candidateIndex,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap, bool print) {
  if (candidateIndex >= candidates.size()) {
    if (print)
      llvm::errs() << "RewriteUnit::apply: candidateIndex " << candidateIndex
                   << " out of range (size " << candidates.size()
                   << "), skipping\n";
    return;
  }
  for (Value *root : roots) {
    auto *oldInst = dyn_cast<Instruction>(root);
    if (!oldInst || !oldInst->getParent()) {
      if (print)
        llvm::errs() << "RewriteUnit::apply: oldOutput "
                     << "is not in a basic block (already erased?), skipping\n";
      return;
    }
  }

  auto parsedNode =
      parseHerbieExpr(candidates[candidateIndex].expr, valueToNodeMap,
                      symbolToValueMap, nullptr, &subgraph->inputs);
  SmallVector<std::shared_ptr<FPNode>, 1> elements;
  if (isArray()) {
    if (parsedNode->op != "array" ||
        parsedNode->operands.size() != roots.size())
      report_fatal_error("Poseidon: an array rewrite does not have one element "
                         "per output of its subgraph");
    elements.append(parsedNode->operands.begin(), parsedNode->operands.end());
  } else {
    elements.push_back(parsedNode);
  }

  SmallVector<Value *, 4> newOutputs;
  if (isArray()) {
    DominatorTree DT(*cast<Instruction>(oldOutput)->getFunction());
    std::optional<IRBuilder<>> builder;
    newOutputs = emitSharedElements(
        elements,
        [&](size_t k) -> IRBuilder<> & {
          auto *oldInst = cast<Instruction>(roots[k]);
          builder.emplace(oldInst->getParent(),
                          ++BasicBlock::iterator(oldInst));
          builder->setFastMathFlags(oldInst->getFastMathFlags());
          return *builder;
        },
        nullptr, &DT);
  }

  for (auto [k, root, element] : llvm::enumerate(roots, elements)) {
    auto *oldInst = cast<Instruction>(root);
    IRBuilder<> builder(oldInst->getParent(), ++BasicBlock::iterator(oldInst));
    builder.setFastMathFlags(oldInst->getFastMathFlags());

    Value *newOutput = isArray() ? newOutputs[k] : element->getLLValue(builder);
    assert(newOutput && "Failed to get value from parsed node");
    if (isArray() && isa<Instruction>(newOutput))
      builder.SetInsertPoint(cast<Instruction>(newOutput)->getNextNode());

    if (newOutput->getType() != root->getType())
      newOutput =
          builder.CreateFPCast(newOutput, root->getType(), "herbie.fpcast");

    root->replaceAllUsesWith(newOutput);
    symbolToValueMap[valueToNodeMap[root]->symbol] = newOutput;
    valueToNodeMap[newOutput] = std::make_shared<FPLLValue>(
        newOutput, "__no", valueToNodeMap[root]->dtype);
  }

  for (auto *I : erasableInsts) {
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
    subgraph->operations.remove(I); // Avoid a second removal
    cast<FPLLValue>(valueToNodeMap[I].get())->value = nullptr;
  }

  subgraph->outputs_rewritten += roots.size();
}

bool RewriteUnit::sharesRoot(const RewriteUnit &other) const {
  for (Value *root : roots)
    if (llvm::is_contained(other.roots, root))
      return true;
  return false;
}

// Lower is better
InstructionCost RewriteUnit::getCompCostDelta(CostModelState &CM,
                                              size_t candidateIndex) {
  double erasableCost = 0.0;

  for (auto *I : erasableInsts) {
    erasableCost += getInstructionCompCost(CM, I);
  }

  // Scale by the execution count BEFORE rounding to the integral
  // InstructionCost the DP compares (see RewriteCandidate::CompCost).
  double delta =
      (candidates[candidateIndex].CompCost - erasableCost) * (double)executions;
  return InstructionCost((int64_t)std::llround(delta));
}

void RewriteUnit::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  for (Value *root : roots)
    collectExprInsts(root, subgraph->inputs, exprInsts, visited);
  visited.clear();

  // Seeded in IR order, not in the order the pointers happened to hash: the
  // topological sort below is stable, so its input order decides the order the
  // report lists these instructions in and the order their costs are summed.
  SetVector<Instruction *> instsToProcess;
  for (Instruction &I :
       instructions(*cast<Instruction>(oldOutput)->getFunction()))
    if (exprInsts.contains(&I))
      instsToProcess.insert(&I);

  SmallVector<Instruction *, 8> instsToProcessSorted;
  reverseTopoSort(instsToProcess, instsToProcessSorted);

  erasableInsts.clear();
  for (Value *root : roots)
    erasableInsts.insert(cast<Instruction>(root));

  for (auto *I : instsToProcessSorted) {
    if (erasableInsts.contains(I))
      continue;

    bool usedOutside = false;
    for (auto user : I->users()) {
      if (auto *userI = dyn_cast<Instruction>(user)) {
        if (erasableInsts.contains(userI)) {
          continue;
        }
      }
      usedOutside = true;
      break;
    }

    if (!usedOutside) {
      erasableInsts.insert(I);
    }
  }
}

bool PrecisionUnit::CacheKey::operator==(const CacheKey &other) const {
  return candidateIndex == other.candidateIndex &&
         RewriteUnits == other.RewriteUnits;
}

std::size_t PrecisionUnit::CacheKeyHash::operator()(const CacheKey &key) const {
  std::size_t seed = std::hash<size_t>{}(key.candidateIndex);
  for (const auto *ao : key.RewriteUnits) {
    seed ^= std::hash<const RewriteUnit *>{}(ao) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
  }
  return seed;
}

void PrecisionUnit::apply(Session &S, size_t candidateIndex) {
  if (candidateIndex >= candidates.size()) {
    llvm_unreachable("Invalid candidate index");
  }

  candidates[candidateIndex].apply(S, *subgraph);
}

// Lower is better
InstructionCost PrecisionUnit::getCompCostDelta(size_t candidateIndex) {
  // TODO: adjust this based on erasured instructions
  double delta = (candidates[candidateIndex].CompCost - initialCompCost) *
                 (double)executions;
  return InstructionCost((int64_t)std::llround(delta));
}

// Lower is better
double PrecisionUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

// Lower is better
double RewriteUnit::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

InstructionCost
PrecisionUnit::getAdjustedCompCostDelta(Session &S, size_t candidateIndex,
                                        ArrayRef<SolutionStep> steps) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  CacheKey key{candidateIndex, RewriteUnits};

  auto cacheIt = compCostDeltaCache.find(key);
  if (cacheIt != compCostDeltaCache.end()) {
    return cacheIt->second;
  }

  Subgraph newSubgraph = *this->subgraph;

  for (auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      const auto &CO = *co;
      if (CO.subgraph == subgraph) {
        newSubgraph.operations.remove_if(
            [&CO](Instruction *I) { return CO.erasableInsts.contains(I); });
        for (Value *root : CO.roots)
          newSubgraph.outputs.remove(cast<Instruction>(root));
      }
    }
  }

  if (newSubgraph.outputs.empty()) {
    compCostDeltaCache[key] = 0;
    return 0;
  }

  double initialCompCost = getCompCost(
      S.CostModel, {newSubgraph.outputs.begin(), newSubgraph.outputs.end()},
      newSubgraph.inputs, &newSubgraph.opExec, newSubgraph.execNormalizer);

  double candidateCompCost =
      getCompCost(S, newSubgraph, candidates[candidateIndex]);

  InstructionCost adjustedCostDelta = InstructionCost((int64_t)std::llround(
      (candidateCompCost - initialCompCost) * (double)executions));
  // Per-candidate pricing audit: with a one-point Pareto table these are the
  // ONLY record of why each candidate lost (all-dominated frontiers must be
  // auditable, not silent).
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "CS candidate '" << candidates[candidateIndex].desc
                 << "': initial=" << initialCompCost
                 << " cand=" << candidateCompCost << " exec=" << executions
                 << " Δcost=" << adjustedCostDelta
                 << " Δacc=" << getAccCostDelta(candidateIndex) << "\n";

  compCostDeltaCache[key] = adjustedCostDelta;
  return adjustedCostDelta;
}

double PrecisionUnit::getAdjustedAccCostDelta(
    size_t candidateIndex, ArrayRef<SolutionStep> steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  RewriteUnitSet RewriteUnits;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      if (co->subgraph == subgraph) {
        RewriteUnits.insert(co);
      }
    }
  }

  CacheKey key{candidateIndex, RewriteUnits};

  auto cacheIt = accCostDeltaCache.find(key);
  if (cacheIt != accCostDeltaCache.end()) {
    return cacheIt->second;
  }

  double totalCandidateAccCost = 0.0;
  double totalInitialAccCost = 0.0;

  SmallPtrSet<FPNode *, 8> stepNodes;
  for (const auto &step : steps) {
    if (auto *co = dyn_cast<RewriteUnit>(step.unit)) {
      const auto &CO = *co;
      if (CO.subgraph == subgraph) {
        for (Value *root : CO.roots) {
          auto it = valueToNodeMap.find(root);
          assert(it != valueToNodeMap.end() && it->second);
          stepNodes.insert(it->second.get());
        }
      }
    }
  }

  for (auto &[node, cost] : perOutputInitialAccCost) {
    if (!stepNodes.count(node)) {
      totalInitialAccCost += cost;
    }
  }

  for (auto &[node, cost] : candidates[candidateIndex].perOutputAccCost) {
    if (!stepNodes.count(node)) {
      totalCandidateAccCost += cost;
    }
  }

  double adjustedAccCostDelta = totalCandidateAccCost - totalInitialAccCost;

  accCostDeltaCache[key] = adjustedAccCostDelta;
  return adjustedAccCostDelta;
}

InstructionCost RewriteUnit::costDelta(size_t i, ArrayRef<SolutionStep>) {
  return getCompCostDelta(session().CostModel, i);
}

double RewriteUnit::errorDelta(size_t i, ArrayRef<SolutionStep>) {
  return getAccCostDelta(i);
}

bool RewriteUnit::sameRewriteTargetAs(const OptimizationUnit &earlier) const {
  const auto *other = dyn_cast<RewriteUnit>(&earlier);
  return other && other->subgraph == subgraph &&
         other->roots.size() == roots.size() &&
         llvm::all_of(other->roots,
                      [&](Value *r) { return llvm::is_contained(roots, r); });
}

const SetVector<Instruction *> *RewriteUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

InstructionCost PrecisionUnit::costDelta(size_t i,
                                         ArrayRef<SolutionStep> prior) {
  return getAdjustedCompCostDelta(session(), i, prior);
}

double PrecisionUnit::errorDelta(size_t i, ArrayRef<SolutionStep> prior) {
  return getAdjustedAccCostDelta(i, prior, site().valueToNodeMap,
                                 site().symbolToValueMap);
}

bool PrecisionUnit::pricedAgainst(const OptimizationUnit &earlier) const {
  const auto *co = dyn_cast<RewriteUnit>(&earlier);
  return co && co->subgraph == subgraph;
}

const SetVector<Instruction *> *PrecisionUnit::footprint() const {
  return subgraph ? &subgraph->operations : nullptr;
}

std::string RewriteUnit::choiceText(size_t i) const {
  const RewriteCandidate &cand = candidates[i];
  return expr + " --(" + std::to_string(cand.ordinal) + ")-> " + cand.expr;
}

json::Object RewriteUnit::stepJSON(size_t i) {
  json::Object stepObj;
  stepObj["type"] = "rewrite";
  stepObj["original_expr"] = expr;
  auto &cand = candidates[i];
  stepObj["rewritten_expr"] = cand.expr;
  stepObj["herbie_cost"] = jsonFloat(cand.herbieCost);
  stepObj["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
  stepObj["initial_herbie_cost"] = jsonFloat(initialHerbieCost);
  stepObj["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);
  stepObj["computation_cost_delta"] =
      GET_INSTRUCTION_COST(getCompCostDelta(session().CostModel, i));
  stepObj["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(i));
  stepObj["gradient"] = jsonFloat(grad);
  stepObj["executions"] = static_cast<int64_t>(executions);

  SmallVector<Instruction *, 8> insts;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    insts.push_back(I);
  for (auto *I : erasableInsts)
    insts.push_back(I);
  stepObj["source_locations"] = getSourceLocationsJSON(insts);

  json::Array affectedIR;
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    affectedIR.push_back(getValueStr(I));
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      affectedIR.push_back(getValueStr(I));
  }
  stepObj["affected_instructions"] = std::move(affectedIR);
  return stepObj;
}

void RewriteUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Rewrite] " << expr << "  -->  " << cand.expr << "\n";
  if (!std::isnan(initialHerbieAccuracy) && !std::isnan(cand.herbieAccuracy))
    OS << pad << "  Herbie accuracy: " << initialHerbieAccuracy << " -> "
       << cand.herbieAccuracy << " bits\n";
  OS << pad << "  Gradient: " << grad << ", Executions: " << executions << "\n";

  SmallSet<std::string, 4> seen;
  auto printLoc = [&](Instruction *I) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty() && seen.insert(loc).second)
      OS << pad << "  Source: " << loc << "\n";
  };
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    printLoc(I);
  for (auto *I : erasableInsts)
    printLoc(I);

  OS << pad << "  Affected IR:\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput))
    OS << pad << "    " << *I << "\n";
  for (auto *I : erasableInsts) {
    if (I != oldOutput)
      OS << pad << "    " << *I << "\n";
  }
}

void RewriteUnit::materialize(size_t i) {
  apply(i, site().valueToNodeMap, site().symbolToValueMap,
        session().Opts.Diagnostics.Print);
}

std::string PrecisionUnit::choiceText(size_t i) const {
  const PTCandidate &cand = candidates[i];
  return "CS: " + cand.desc + " (#" + std::to_string(cand.ordinal) + ")";
}

json::Object PrecisionUnit::stepJSON(size_t i) {
  auto &cand = candidates[i];
  json::Object stepObj;
  stepObj["type"] = "precision_change";
  stepObj["description"] = cand.desc;
  stepObj["candidate_index"] = static_cast<int64_t>(cand.ordinal);

  json::Array changes;
  for (const auto &change : cand.changes) {
    json::Object changeObj;
    changeObj["from"] = getPrecisionChangeTypeString(change.oldType);
    changeObj["to"] = getPrecisionChangeTypeString(change.newType);
    changeObj["num_operations"] = static_cast<int64_t>(change.nodes.size());

    SmallVector<Instruction *, 8> insts;
    json::Array nodeIR;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        insts.push_back(I);
        nodeIR.push_back(getValueStr(I));
      }
    }
    changeObj["source_locations"] = getSourceLocationsJSON(insts);
    changeObj["affected_instructions"] = std::move(nodeIR);
    changes.push_back(std::move(changeObj));
  }
  stepObj["changes"] = std::move(changes);
  return stepObj;
}

void PrecisionUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  auto &cand = candidates[i];
  OS << pad << "[Precision] " << cand.desc << " (#" << cand.ordinal << ")\n";
  for (const auto &change : cand.changes) {
    OS << pad << "  " << getPrecisionChangeTypeString(change.oldType) << " -> "
       << getPrecisionChangeTypeString(change.newType) << " for "
       << change.nodes.size() << " operations\n";
    SmallSet<std::string, 4> seen;
    for (auto *node : change.nodes) {
      if (auto *I = dyn_cast<Instruction>(node->value)) {
        auto loc = getSourceLocationStr(I);
        if (!loc.empty() && seen.insert(loc).second)
          OS << pad << "    Source: " << loc << "\n";
      }
    }
  }
}

void PrecisionUnit::materialize(size_t i) { apply(session(), i); }

void RewriteUnit::price() {
  if (!herbieAnswered)
    return;
  FunctionFPState &st = site();
  Module *M = st.F->getParent();
  const FastMathFlags FMF = cast<Instruction>(oldOutput)->getFastMathFlags();
  for (RewriteCandidate &c : candidates)
    c.CompCost = getCompCost(session().CostModel, c.expr, M, st.valueToNodeMap,
                             st.symbolToValueMap, FMF, &subgraph->inputs);
  setUnifiedAccuracyCost(session().Opts, *this, st.valueToNodeMap,
                         st.symbolToValueMap);
}

void PrecisionUnit::price() {
  for (PTCandidate &c : candidates)
    c.CompCost = getCompCost(session(), *subgraph, c);
  setUnifiedAccuracyCost(session().Opts, *this, site().valueToNodeMap,
                         site().symbolToValueMap);
}

void RewriteUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &cand : candidates)
    cand.accuracyCost *= factor;
}

void PrecisionUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &[node, cost] : perOutputInitialAccCost)
    cost *= factor;
  for (auto &cand : candidates) {
    cand.accuracyCost *= factor;
    for (auto &[node, cost] : cand.perOutputAccCost)
      cost *= factor;
  }
}

void RewriteUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
  OS << "Initial HerbieCost: " << initialHerbieCost << "\n";
  OS << "Initial HerbieAccuracy: " << initialHerbieAccuracy << "\n";
  OS << "Initial Expression: " << expr << "\n";
  OS << "Grad: " << grad << "\n\n";
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ "
        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
  OS << "--------------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t"
       << getCompCostDelta(session().CostModel, i) << "\t\t"
       << candidate.herbieCost << "\t\t" << candidate.herbieAccuracy << "\t\t"
       << candidate.expr << "\n";
  }
  OS << "################################\n\n";
}

void RewriteUnit::addReportSummary(json::Array &summary) {
  json::Object co;
  co["original_expr"] = expr;
  co["num_candidates"] = static_cast<int64_t>(candidates.size());
  co["gradient"] = jsonFloat(grad);
  co["executions"] = static_cast<int64_t>(executions);
  co["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  co["initial_computation_cost"] = initialCompCost;
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationJSON(I);
    if (!loc.empty())
      co["source_location"] = std::move(loc);
  }
  summary.push_back(std::move(co));
}

void RewriteUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Expression #" << indexInKind() << ": " << expr << "\n";
  textOut << "  Gradient: " << grad << ", Executions: " << executions << "\n";
  if (auto *I = dyn_cast<Instruction>(oldOutput)) {
    auto loc = getSourceLocationStr(I);
    if (!loc.empty())
      textOut << "  Source: " << loc << "\n";
  }
  textOut << "  Candidates: " << candidates.size() << "\n";
  for (size_t j = 0; j < candidates.size(); ++j) {
    auto &cand = candidates[j];
    textOut << "    [" << cand.ordinal << "] " << cand.expr;
    if (!std::isnan(cand.herbieAccuracy))
      textOut << "  (accuracy: " << cand.herbieAccuracy << " bits)";
    textOut << "\n";
  }
}

void RewriteUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &cand = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(session().CostModel, ci);
    int64_t compDeltaVal = GET_INSTRUCTION_COST(compDelta);

    if (std::isnan(accDelta))
      continue;

    // comp < 0 means faster, acc < 0 means more accurate
    std::string category;
    double efficiency = 0.0;
    if (compDeltaVal <= 0 && accDelta <= 0) {
      category = "free_win";
      efficiency = std::abs(accDelta) + std::abs((double)compDeltaVal);
    } else if (compDeltaVal <= 0 && accDelta > 0) {
      category = "speed_for_accuracy";
      efficiency = (accDelta > 1e-30)
                       ? std::abs((double)compDeltaVal) / accDelta
                       : std::abs((double)compDeltaVal);
    } else if (compDeltaVal > 0 && accDelta <= 0) {
      category = "accuracy_for_speed";
      efficiency = (compDeltaVal > 0)
                       ? std::abs(accDelta) / (double)compDeltaVal
                       : std::abs(accDelta);
    } else {
      continue;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["original_expr"] = expr;
    rw["rewritten_expr"] = cand.expr;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);
    rw["gradient"] = jsonFloat(grad);
    rw["executions"] = static_cast<int64_t>(executions);
    rw["herbie_accuracy"] = jsonFloat(cand.herbieAccuracy);
    rw["initial_herbie_accuracy"] = jsonFloat(initialHerbieAccuracy);

    if (auto *I = dyn_cast<Instruction>(oldOutput)) {
      auto loc = getSourceLocationJSON(I);
      if (!loc.empty())
        rw["source_location"] = std::move(loc);
    }

    rewrites.push_back(std::move(rw));
  }
}

// Top-N per-candidate deltas keep an all-dominated frontier auditable (the
// table then collapses to the no-op point).
static constexpr size_t kMaxCandRows = 40;

// The candidates' positions by cost change, cheapest first; equal changes keep
// their positions' order.
static SmallVector<size_t, 40> byCostDelta(PrecisionUnit &CS) {
  SmallVector<size_t, 40> order(CS.candidates.size());
  for (size_t ci = 0; ci < order.size(); ++ci)
    order[ci] = ci;
  llvm::stable_sort(order, [&](size_t a, size_t b) {
    return GET_INSTRUCTION_COST(CS.getCompCostDelta(a)) <
           GET_INSTRUCTION_COST(CS.getCompCostDelta(b));
  });
  return order;
}

void PrecisionUnit::addReportSummary(json::Array &summary) {
  json::Object cs;
  cs["num_candidates"] = static_cast<int64_t>(candidates.size());
  cs["initial_accuracy_cost"] = jsonFloat(initialAccCost);
  cs["initial_computation_cost"] = initialCompCost;
  SmallVector<size_t, 40> order = byCostDelta(*this);
  json::Array cands;
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    json::Object c;
    c["id"] = candidateId(ci);
    c["description"] = candidates[ci].desc;
    c["computation_cost_delta"] = GET_INSTRUCTION_COST(getCompCostDelta(ci));
    c["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(ci));
    cands.push_back(std::move(c));
  }
  cs["candidates"] = std::move(cands);
  summary.push_back(std::move(cs));
}

void PrecisionUnit::writeReportSummary(raw_ostream &textOut) {
  textOut << "Subgraph #" << indexInKind() << ": " << candidates.size()
          << " candidates (Δcost, Δacc; sorted by Δcost)\n";
  SmallVector<size_t, 40> order = byCostDelta(*this);
  for (size_t k = 0; k < order.size() && k < kMaxCandRows; ++k) {
    size_t ci = order[k];
    textOut << "  [" << candidateId(ci) << "] " << candidates[ci].desc
            << ": Δcost=" << GET_INSTRUCTION_COST(getCompCostDelta(ci))
            << " Δacc=" << format("%.6e", getAccCostDelta(ci)) << "\n";
  }
}

void PrecisionUnit::addCuratedRewrites(json::Array &rewrites) {
  for (size_t ci = 0; ci < candidates.size(); ++ci) {
    auto &pt = candidates[ci];
    double accDelta = getAccCostDelta(ci);
    auto compDelta = getCompCostDelta(ci);
    int64_t compDeltaVal = GET_INSTRUCTION_COST(compDelta);

    if (std::isnan(accDelta))
      continue;

    std::string category;
    double efficiency = 0.0;
    if (compDeltaVal <= 0 && accDelta <= 0) {
      category = "free_win";
      efficiency = std::abs(accDelta) + std::abs((double)compDeltaVal);
    } else if (compDeltaVal <= 0 && accDelta > 0) {
      category = "speed_for_accuracy";
      efficiency = (accDelta > 1e-30)
                       ? std::abs((double)compDeltaVal) / accDelta
                       : std::abs((double)compDeltaVal);
    } else if (compDeltaVal > 0 && accDelta <= 0) {
      category = "accuracy_for_speed";
      efficiency = (compDeltaVal > 0)
                       ? std::abs(accDelta) / (double)compDeltaVal
                       : std::abs(accDelta);
    } else {
      // Emitted even though dominated: an all-dominated frontier must
      // leave a record of why each candidate lost.
      category = "dominated";
      efficiency = 0.0;
    }

    json::Object rw;
    rw["id"] = candidateId(ci);
    rw["type"] = "precision_change";
    rw["description"] = pt.desc;
    rw["category"] = category;
    rw["efficiency"] = jsonFloat(efficiency);
    rw["computation_cost_delta"] = compDeltaVal;
    rw["accuracy_cost_delta"] = jsonFloat(accDelta);

    SmallVector<Instruction *, 16> ptInsts;
    for (const auto &change : pt.changes) {
      for (auto *node : change.nodes) {
        if (auto *I = dyn_cast<Instruction>(node->value))
          ptInsts.push_back(I);
      }
    }
    rw["source_locations"] = getSourceLocationsJSON(ptInsts);
    json::Array affectedIR;
    for (auto *I : ptInsts)
      affectedIR.push_back(getValueStr(I));
    rw["affected_instructions"] = std::move(affectedIR);

    rewrites.push_back(std::move(rw));
  }
}

void PrecisionUnit::printPriced(raw_ostream &OS) {
  OS << "\n################################\n";
  OS << "Initial AccuracyCost: " << initialAccCost << "\n";
  OS << "Initial ComputationCost: " << initialCompCost << "\n";
  OS << "Candidates:\n";
  OS << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
     << "---------------------------\n";
  for (size_t i = 0; i < candidates.size(); ++i) {
    auto &candidate = candidates[i];
    OS << getAccCostDelta(i) << "\t\t" << getCompCostDelta(i) << "\t\t"
       << candidate.desc << "\n";
  }
  OS << "################################\n\n";
}

} // namespace poseidon
