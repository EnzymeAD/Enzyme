//=- PoseidonNodes.cpp - AST node implementations for Poseidon ------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
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
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cassert>
#include <cmath>
#include <functional>

#include "../Utils.h"
#include "Poseidon.h"
#include "PoseidonHerbieUtils.h"
#include "PoseidonPrecUtils.h"
#include "PoseidonTypes.h"
#include "PoseidonUtils.h"

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
  if (dtype == "f32")
    return 24;
  if (dtype == "f64")
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

    return builder.CreateSelect(condValue, trueValue, falseValue,
                                "herbie.select");
  }

  SmallVector<Value *, 3> operandValues;
  for (auto operand : operands) {
    Value *val = operand->getLLValue(builder, VMap);
    assert(val && "Operand produced a null value!");
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
             std::string funcName = Ty->isDoubleTy() ? "tan" : "tanf";
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
             std::string funcName = Ty->isDoubleTy() ? "expm1" : "expm1f";
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
             std::string funcName = Ty->isDoubleTy() ? "log1p" : "log1pf";
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
             std::string funcName = Ty->isDoubleTy() ? "cbrt" : "cbrtf";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.cbrt");
           }},
          {"pow",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             // Use powi when possible
             if (auto *CF = dyn_cast<ConstantFP>(ops[1])) {
               double value = CF->getValueAPF().convertToDouble();
               if (value == std::floor(value) && value >= INT_MIN &&
                   value <= INT_MAX) {
                 int exp = static_cast<int>(value);
                 SmallVector<Type *, 1> overloadedTypes = {
                     ops[0]->getType(), Type::getInt32Ty(M->getContext())};
                 Function *powiFunc = getIntrinsicDeclaration(
                     M, Intrinsic::powi, overloadedTypes);
                 Value *exponent =
                     ConstantInt::get(Type::getInt32Ty(M->getContext()), exp);
                 return b.CreateCall(powiFunc, {ops[0], exponent},
                                     "herbie.powi");
               }
             }

             return b.CreateBinaryIntrinsic(Intrinsic::pow, ops[0], ops[1],
                                            nullptr, "herbie.pow");
           }},
          {"fma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             return b.CreateIntrinsic(Intrinsic::fmuladd, {ops[0]->getType()},
                                      {ops[0], ops[1], ops[2]}, nullptr,
                                      "herbie.fmuladd");
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
             std::string funcName = Ty->isDoubleTy() ? "hypot" : "hypotf";
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
             std::string funcName = Ty->isDoubleTy() ? "asin" : "asinf";
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
             std::string funcName = Ty->isDoubleTy() ? "acos" : "acosf";
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
             std::string funcName = Ty->isDoubleTy() ? "atan" : "atanf";
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
             std::string funcName = Ty->isDoubleTy() ? "atan2" : "atan2f";
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
             std::string funcName = Ty->isDoubleTy() ? "sinh" : "sinhf";
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
             std::string funcName = Ty->isDoubleTy() ? "cosh" : "coshf";
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
             std::string funcName = Ty->isDoubleTy() ? "tanh" : "tanhf";
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
             std::string funcName = Ty->isDoubleTy() ? "fdim" : "fdimf";
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.fdim");
           }},
          {"fmod",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "fmod" : "fmodf";
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.fmod");
           }},
          {"remainder",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName =
                 Ty->isDoubleTy() ? "remainder" : "remainderf";
             FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.remainder");
           }},
          {"erf",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "erf" : "erff";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.erf");
           }},
          {"lgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "lgamma" : "lgammaf";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.lgamma");
           }},
          {"tgamma",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "tgamma" : "tgammaf";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.tgamma");
           }},
          {"asinh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "asinh" : "asinhf";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.asinh");
           }},
          {"acosh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "acosh" : "acoshf";
             FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
             FunctionCallee f = M->getOrInsertFunction(funcName, FT);
             return b.CreateCall(f, {ops[0]}, "herbie.acosh");
           }},
          {"atanh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "atanh" : "atanhf";
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
      };

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
  // Check if this value is an input to the current subgraph
  if (subgraphInputs.contains(value)) {
    assert(hasSymbol() && "FPLLValue has no symbol!");
    return symbol;
  } else {
    assert(!operands.empty() && "FPNode has no operands!");

    if (depth > FPOptMaxExprDepth) {
      std::string msg = "Expression depth exceeded maximum allowed depth of " +
                        std::to_string(FPOptMaxExprDepth) + " for " + op +
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
  if (FPOptPrint)
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

  double constantValue;
  size_t div = strValue.find('/');

  if (div != std::string::npos) {
    std::string numerator = strValue.substr(0, div);
    std::string denominator = strValue.substr(div + 1);
    double num = stringToDouble(numerator);
    double denom = stringToDouble(denominator);

    constantValue = num / denom;
  } else {
    constantValue = stringToDouble(strValue);
  }

  return constantValue;
}

double FPConst::getUpperBound() const { return getLowerBound(); }

Value *FPConst::getLLValue(IRBuilder<> &builder,
                           const ValueToValueMapTy *VMap) {
  Type *Ty;
  if (dtype == "f64") {
    Ty = builder.getDoubleTy();
  } else if (dtype == "f32") {
    Ty = builder.getFloatTy();
  } else {
    std::string msg = "FPConst getValue: Unexpected dtype: " + dtype;
    llvm_unreachable(msg.c_str());
  }
  if (strValue == "+inf.0") {
    return ConstantFP::getInfinity(Ty, false);
  } else if (strValue == "-inf.0") {
    return ConstantFP::getInfinity(Ty, true);
  }

  double constantValue;
  size_t div = strValue.find('/');

  if (div != std::string::npos) {
    std::string numerator = strValue.substr(0, div);
    std::string denominator = strValue.substr(div + 1);
    double num = stringToDouble(numerator);
    double denom = stringToDouble(denominator);

    constantValue = num / denom;
  } else {
    constantValue = stringToDouble(strValue);
  }

  // if (FPOptPrint)
  //   llvm::errs() << "Returning " << strValue << " as " << dtype
  //                << " constant: " << constantValue << "\n";
  return ConstantFP::get(Ty, constantValue);
}

bool FPConst::classof(const FPNode *N) {
  return N->getType() == NodeType::Const;
}

void CandidateOutput::apply(
    size_t candidateIndex,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  // 4) parse the output string solution from herbieland
  // 5) convert into a solution in llvm vals/instructions

  // if (FPOptPrint)
  //   llvm::errs() << "Parsing Herbie output: " << herbieOutput << "\n";
  auto parsedNode = parseHerbieExpr(candidates[candidateIndex].expr,
                                    valueToNodeMap, symbolToValueMap);
  // if (FPOptPrint)
  //   llvm::errs() << "Parsed Herbie output: "
  //                << parsedNode->toFullExpression(valueToNodeMap) << "\n";

  IRBuilder<> builder(cast<Instruction>(oldOutput)->getParent(),
                      ++BasicBlock::iterator(cast<Instruction>(oldOutput)));
  builder.setFastMathFlags(cast<Instruction>(oldOutput)->getFastMathFlags());

  // auto *F = cast<Instruction>(oldOutput)->getParent()->getParent();
  // llvm::errs() << "Before: " << *F << "\n";
  Value *newOutput = parsedNode->getLLValue(builder);
  assert(newOutput && "Failed to get value from parsed node");

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

  // llvm::errs() << "After: " << *F << "\n";

  subgraph->outputs_rewritten++;
}

// Lower is better
InstructionCost CandidateOutput::getCompCostDelta(size_t candidateIndex) {
  InstructionCost erasableCost = 0;

  for (auto *I : erasableInsts) {
    erasableCost += getInstructionCompCost(I, *TTI);
  }

  return (candidates[candidateIndex].CompCost - erasableCost) * executions;
}

void CandidateOutput::findErasableInstructions() {
  SmallPtrSet<Value *, 8> visited;
  SmallPtrSet<Instruction *, 8> exprInsts;
  collectExprInsts(oldOutput, subgraph->inputs, exprInsts, visited);
  visited.clear();

  SetVector<Instruction *> instsToProcess(exprInsts.begin(), exprInsts.end());

  SmallVector<Instruction *, 8> instsToProcessSorted;
  reverseTopoSort(instsToProcess, instsToProcessSorted);

  // `oldOutput` is trivially erasable
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
      // If the user is not an intruction or the user instruction is not an
      // erasable instruction, then the current instruction is not erasable
      // llvm::errs() << "Can't erase " << *I << " because of " << *user <<
      // "\n";
      usedOutside = true;
      break;
    }

    if (!usedOutside) {
      erasableInsts.insert(I);
    }
  }

  // llvm::errs() << "Erasable instructions:\n";
  // for (auto *I : erasableInsts) {
  //   llvm::errs() << *I << "\n";
  // }
  // llvm::errs() << "End of erasable instructions\n";
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

  // Traverse all the instructions to be changed precisions in a
  // topological order with respect to operand dependencies. Insert FP casts
  // between llvm::Value inputs and first level of instructions to be changed.
  // Restore precisions of the last level of instructions to be changed.
  candidates[candidateIndex].apply(*subgraph);
}

// Lower is better
InstructionCost CandidateSubgraph::getCompCostDelta(size_t candidateIndex) {
  // TODO: adjust this based on erasured instructions
  return (candidates[candidateIndex].CompCost - initialCompCost) * executions;
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
        // Eliminate erasadable instructions from the adjusted CS
        newSubgraph.operations.remove_if(
            [&CO](Instruction *I) { return CO.erasableInsts.contains(I); });
        newSubgraph.outputs.remove(cast<Instruction>(CO.oldOutput));
      }
    }
  }

  // If all outputs are rewritten, then the adjusted CS is empty
  if (newSubgraph.outputs.empty()) {
    compCostDeltaCache[key] = 0;
    return 0;
  }

  InstructionCost initialCompCost =
      getCompCost({newSubgraph.outputs.begin(), newSubgraph.outputs.end()},
                  newSubgraph.inputs, TTI);

  InstructionCost candidateCompCost =
      getCompCost(newSubgraph, TTI, candidates[candidateIndex]);

  InstructionCost adjustedCostDelta =
      (candidateCompCost - initialCompCost) * executions;
  // llvm::errs() << "Initial cost: " << initialCompCost << "\n";
  // llvm::errs() << "Candidate cost: " << candidateCompCost << "\n";
  // llvm::errs() << "Num executions: " << executions << "\n";
  // llvm::errs() << "Adjusted cost delta: " << adjustedCostDelta << "\n\n";

  compCostDeltaCache[key] = adjustedCostDelta;
  return adjustedCostDelta;
}

double CandidateSubgraph::getAdjustedAccCostDelta(
    size_t candidateIndex, SmallVectorImpl<SolutionStep> &steps,
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

  // Collect erased output nodes
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

  // Iterate over all output nodes and sum costs for nodes not erased
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
