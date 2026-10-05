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

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cassert>
#include <cmath>
#include <functional>

#include "CostModel.h"
#include "Flags.h"
#include "Herbie.h"
#include "Optimize.h"
#include "Precision.h"
#include "Types.h"
#include "Utils.h"

using namespace llvm;

namespace poseidon {

static constexpr unsigned kMaxExprDepth = 100;

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

static std::string libmFuncName(Module *M, Type *Ty, StringRef dblBase,
                                StringRef fltBase) {
  StringRef base = Ty->isDoubleTy() ? dblBase : fltBase;
  if (Triple(M->getTargetTriple()).isNVPTX())
    return ("__nv_" + base).str();
  return base.str();
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

void FPNode::updateBounds(double lower, double upper) {
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
             std::string funcName = libmFuncName(M, Ty, "tan", "tanf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee tanFunc = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(tanFunc, {ops[0]}, "herbie.tan");
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
             std::string funcName = libmFuncName(M, Ty, "expm1", "expm1f");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.expm1");
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
             std::string funcName = libmFuncName(M, Ty, "log1p", "log1pf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.log1p");
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
             std::string funcName = libmFuncName(M, Ty, "cbrt", "cbrtf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.cbrt");
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
                 return b.CreateCall(powiFunc, {ops[0], exponent},
                                     "herbie.powi");
               }
             }

             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "pow", "powf");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.pow");
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
             std::string funcName = libmFuncName(M, Ty, "hypot", "hypotf");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.hypot");
           }},
          {"asin",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
#if LLVM_VERSION_MAJOR > 16
             return b.CreateUnaryIntrinsic(Intrinsic::asin, ops[0], nullptr,
                                           "herbie.asin");
#else
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "asin", "asinf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.asin");
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
             std::string funcName = libmFuncName(M, Ty, "acos", "acosf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.acos");
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
             std::string funcName = libmFuncName(M, Ty, "atan", "atanf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.atan");
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
             std::string funcName = libmFuncName(M, Ty, "atan2", "atan2f");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.atan2");
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
             std::string funcName = libmFuncName(M, Ty, "sinh", "sinhf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.sinh");
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
             std::string funcName = libmFuncName(M, Ty, "cosh", "coshf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.cosh");
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
             std::string funcName = libmFuncName(M, Ty, "tanh", "tanhf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.tanh");
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
             std::string funcName = libmFuncName(M, Ty, "fdim", "fdimf");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.fdim");
           }},
          {"fmod",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "fmod", "fmodf");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.fmod");
           }},
          {"remainder",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName =
                 libmFuncName(M, Ty, "remainder", "remainderf");
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.remainder");
           }},
          {"erf",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "erf", "erff");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.erf");
           }},
          {"lgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "lgamma", "lgammaf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.lgamma");
           }},
          {"tgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "tgamma", "tgammaf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.tgamma");
           }},
          {"asinh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "asinh", "asinhf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.asinh");
           }},
          {"acosh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "acosh", "acoshf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.acosh");
           }},
          {"atanh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = libmFuncName(M, Ty, "atanh", "atanhf");
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.atanh");
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

  // On NVPTX the generic math intrinsics emitted by opMap (llvm.sin/cos/exp/
  // log/...) have no native instruction and fail instruction selection
  // ("Cannot select: fNN = fsin"). libdevice provides __nv_<f> for f64 and
  // __nv_<f>f for f32. Algebraic / native ops (sqrt, fabs, floor, ceil, trunc,
  // rint, round, copysign, fma, fmin/fmax, +,-,*,/) DO lower on NVPTX, so they
  // fall through to opMap.
  if (isGPUMode(*builder.GetInsertBlock()->getParent())) {
    static const std::unordered_map<std::string,
                                    std::pair<const char *, const char *>>
        nvLibdevice = {
            {"sin", {"__nv_sin", "__nv_sinf"}},
            {"cos", {"__nv_cos", "__nv_cosf"}},
            {"tan", {"__nv_tan", "__nv_tanf"}},
            {"asin", {"__nv_asin", "__nv_asinf"}},
            {"acos", {"__nv_acos", "__nv_acosf"}},
            {"atan", {"__nv_atan", "__nv_atanf"}},
            {"atan2", {"__nv_atan2", "__nv_atan2f"}},
            {"sinh", {"__nv_sinh", "__nv_sinhf"}},
            {"cosh", {"__nv_cosh", "__nv_coshf"}},
            {"tanh", {"__nv_tanh", "__nv_tanhf"}},
            {"asinh", {"__nv_asinh", "__nv_asinhf"}},
            {"acosh", {"__nv_acosh", "__nv_acoshf"}},
            {"atanh", {"__nv_atanh", "__nv_atanhf"}},
            {"exp", {"__nv_exp", "__nv_expf"}},
            {"exp2", {"__nv_exp2", "__nv_exp2f"}},
            {"expm1", {"__nv_expm1", "__nv_expm1f"}},
            {"log", {"__nv_log", "__nv_logf"}},
            {"log2", {"__nv_log2", "__nv_log2f"}},
            {"log10", {"__nv_log10", "__nv_log10f"}},
            {"log1p", {"__nv_log1p", "__nv_log1pf"}},
            {"pow", {"__nv_pow", "__nv_powf"}},
            {"cbrt", {"__nv_cbrt", "__nv_cbrtf"}},
        };
    auto nvIt = nvLibdevice.find(op);
    if (nvIt != nvLibdevice.end() && !operandValues.empty() &&
        (operandValues[0]->getType()->isFloatTy() ||
         operandValues[0]->getType()->isDoubleTy())) {
      Type *Ty = operandValues[0]->getType();
      const char *fname =
          Ty->isFloatTy() ? nvIt->second.second : nvIt->second.first;
      SmallVector<Type *, 2> argTys(operandValues.size(), Ty);
      FunctionCallee fn =
          M->getOrInsertFunction(fname, FunctionType::get(Ty, argTys, false));
      return builder.CreateCall(fn, operandValues, "herbie." + op);
    }
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

std::string FPLLValue::toFullExpression(
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const SetVector<Value *> &subgraphInputs, unsigned depth) {
  if (subgraphInputs.contains(value)) {
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

void FPLLValue::updateBounds(double lower, double upper) {
  lb = std::min(lb, lower);
  ub = std::max(ub, upper);
  if (flags::Print)
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

void FPConst::updateBounds(double lower, double upper) { return; }

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

void CandidateOutput::apply(
    size_t candidateIndex,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  if (candidateIndex >= candidates.size()) {
    if (flags::Print)
      llvm::errs() << "CandidateOutput::apply: candidateIndex "
                   << candidateIndex << " out of range (size "
                   << candidates.size() << "), skipping\n";
    return;
  }
  auto *oldInst = dyn_cast<Instruction>(oldOutput);
  if (!oldInst || !oldInst->getParent()) {
    if (flags::Print)
      llvm::errs() << "CandidateOutput::apply: oldOutput "
                   << "is not in a basic block (already erased?), skipping\n";
    return;
  }

  auto parsedNode = parseHerbieExpr(candidates[candidateIndex].expr,
                                    valueToNodeMap, symbolToValueMap);

  IRBuilder<> builder(oldInst->getParent(), ++BasicBlock::iterator(oldInst));
  builder.setFastMathFlags(oldInst->getFastMathFlags());

  Value *newOutput = parsedNode->getLLValue(builder);
  assert(newOutput && "Failed to get value from parsed node");

  if (newOutput->getType() != oldOutput->getType())
    newOutput =
        builder.CreateFPCast(newOutput, oldOutput->getType(), "herbie.fpcast");

  oldOutput->replaceAllUsesWith(newOutput);
  symbolToValueMap[valueToNodeMap[oldOutput]->symbol] = newOutput;
  valueToNodeMap[newOutput] = std::make_shared<FPLLValue>(
      newOutput, "__no", valueToNodeMap[oldOutput]->dtype);

  for (auto *I : erasableInsts) {
    if (!I->use_empty())
      I->replaceAllUsesWith(UndefValue::get(I->getType()));
    I->eraseFromParent();
    subgraph->operations.remove(I); // Avoid a second removal
    cast<FPLLValue>(valueToNodeMap[I].get())->value = nullptr;
  }

  subgraph->outputs_rewritten++;
}

// Lower is better
InstructionCost CandidateOutput::getCompCostDelta(size_t candidateIndex) {
  double erasableCost = 0.0;

  for (auto *I : erasableInsts) {
    erasableCost += getInstructionCompCost(I);
  }

  // Scale by the execution count BEFORE rounding to the integral
  // InstructionCost the DP compares (see RewriteCandidate::CompCost).
  double delta =
      (candidates[candidateIndex].CompCost - erasableCost) * (double)executions;
  return InstructionCost((int64_t)std::llround(delta));
}

void CandidateOutput::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  collectExprInsts(oldOutput, subgraph->inputs, exprInsts, visited);
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
  erasableInsts.insert(cast<Instruction>(oldOutput));

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

bool CandidateSubgraph::CacheKey::operator==(const CacheKey &other) const {
  return candidateIndex == other.candidateIndex &&
         CandidateOutputs == other.CandidateOutputs;
}

std::size_t
CandidateSubgraph::CacheKeyHash::operator()(const CacheKey &key) const {
  std::size_t seed = std::hash<size_t>{}(key.candidateIndex);
  for (const auto *ao : key.CandidateOutputs) {
    seed ^= std::hash<const CandidateOutput *>{}(ao) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
  }
  return seed;
}

void CandidateSubgraph::apply(size_t candidateIndex) {
  if (candidateIndex >= candidates.size()) {
    llvm_unreachable("Invalid candidate index");
  }

  candidates[candidateIndex].apply(*subgraph);
}

// Lower is better
InstructionCost CandidateSubgraph::getCompCostDelta(size_t candidateIndex) {
  // TODO: adjust this based on erasured instructions
  double delta = (candidates[candidateIndex].CompCost - initialCompCost) *
                 (double)executions;
  return InstructionCost((int64_t)std::llround(delta));
}

// Lower is better
double CandidateSubgraph::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

// Lower is better
double CandidateOutput::getAccCostDelta(size_t candidateIndex) {
  return candidates[candidateIndex].accuracyCost - initialAccCost;
}

InstructionCost CandidateSubgraph::getAdjustedCompCostDelta(
    size_t candidateIndex, const SmallVectorImpl<SolutionStep> &steps) {
  CandidateOutputSet CandidateOutputs;
  for (const auto &step : steps) {
    if (auto *ptr = std::get_if<CandidateOutput *>(&step.item)) {
      if ((*ptr)->subgraph == subgraph) {
        CandidateOutputs.insert(*ptr);
      }
    }
  }

  CacheKey key{candidateIndex, CandidateOutputs};

  auto cacheIt = compCostDeltaCache.find(key);
  if (cacheIt != compCostDeltaCache.end()) {
    return cacheIt->second;
  }

  Subgraph newSubgraph = *this->subgraph;

  for (auto &step : steps) {
    if (auto *ptr = std::get_if<CandidateOutput *>(&step.item)) {
      const auto &CO = **ptr;
      if (CO.subgraph == subgraph) {
        newSubgraph.operations.remove_if(
            [&CO](Instruction *I) { return CO.erasableInsts.contains(I); });
        newSubgraph.outputs.remove(cast<Instruction>(CO.oldOutput));
      }
    }
  }

  if (newSubgraph.outputs.empty()) {
    compCostDeltaCache[key] = 0;
    return 0;
  }

  double initialCompCost =
      getCompCost({newSubgraph.outputs.begin(), newSubgraph.outputs.end()},
                  newSubgraph.inputs);

  double candidateCompCost =
      getCompCost(newSubgraph, candidates[candidateIndex]);

  InstructionCost adjustedCostDelta = InstructionCost((int64_t)std::llround(
      (candidateCompCost - initialCompCost) * (double)executions));
  // Per-candidate pricing audit: with a one-point Pareto table these are the
  // ONLY record of why each candidate lost (all-dominated frontiers must be
  // auditable, not silent).
  if (flags::Print)
    llvm::errs() << "CS candidate '" << candidates[candidateIndex].desc
                 << "': initial=" << initialCompCost
                 << " cand=" << candidateCompCost << " exec=" << executions
                 << " Δcost=" << adjustedCostDelta
                 << " Δacc=" << getAccCostDelta(candidateIndex) << "\n";

  compCostDeltaCache[key] = adjustedCostDelta;
  return adjustedCostDelta;
}

double CandidateSubgraph::getAdjustedAccCostDelta(
    size_t candidateIndex, const SmallVectorImpl<SolutionStep> &steps,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  CandidateOutputSet CandidateOutputs;
  for (const auto &step : steps) {
    if (auto *ptr = std::get_if<CandidateOutput *>(&step.item)) {
      if ((*ptr)->subgraph == subgraph) {
        CandidateOutputs.insert(*ptr);
      }
    }
  }

  CacheKey key{candidateIndex, CandidateOutputs};

  auto cacheIt = accCostDeltaCache.find(key);
  if (cacheIt != accCostDeltaCache.end()) {
    return cacheIt->second;
  }

  double totalCandidateAccCost = 0.0;
  double totalInitialAccCost = 0.0;

  SmallPtrSet<FPNode *, 8> stepNodes;
  for (const auto &step : steps) {
    if (auto *ptr = std::get_if<CandidateOutput *>(&step.item)) {
      const auto &CO = **ptr;
      if (CO.subgraph == subgraph) {
        auto it = valueToNodeMap.find(CO.oldOutput);
        assert(it != valueToNodeMap.end() && it->second);
        stepNodes.insert(it->second.get());
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

} // namespace poseidon
