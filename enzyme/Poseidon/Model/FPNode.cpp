//=- FPNode.cpp - the expression graph of a site ----------------------------=//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPNode.h"
#include "Targets/DeviceMath.h"
#include "Targets/Target.h"
#include "Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <optional>

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

} // namespace poseidon
