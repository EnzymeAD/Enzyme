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

#include "PoseidonNodes.h"
#include "Poseidon.h"
#include "PoseidonUtils.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cassert>
#include <cmath>
#include <functional>

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
    unsigned depth) {
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

void FPNode::markAsInput() {
  std::string msg = "Unexpected invocation of `markAsInput` on an "
                    "unmaterialized " +
                    op + " FPNode";
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
    auto IP = builder.GetInsertPoint();

    Instruction *Then, *Else;
    SplitBlockAndInsertIfThenElse(condValue, &*IP, &Then, &Else);

    Then->getParent()->setName("herbie.then");
    builder.SetInsertPoint(Then);
    Value *ThenVal = operands[1]->getLLValue(builder, VMap);
    if (Instruction *I = dyn_cast<Instruction>(ThenVal))
      I->setName("herbie.then_val");

    Else->getParent()->setName("herbie.else");
    builder.SetInsertPoint(Else);
    Value *ElseVal = operands[2]->getLLValue(builder, VMap);
    if (Instruction *I = dyn_cast<Instruction>(ElseVal))
      I->setName("herbie.else_val");

    builder.SetInsertPoint(&*IP);
    auto Phi = builder.CreatePHI(ThenVal->getType(), 2);
    Phi->addIncoming(ThenVal, Then->getParent());
    Phi->addIncoming(ElseVal, Else->getParent());
    Phi->setName("herbie.merge");
    return Phi;
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
             Function *tanFunc = M->getFunction(funcName);
             if (!tanFunc) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               tanFunc =
                   Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
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
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
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
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
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
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
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
#if LLVM_VERSION_MAJOR >= 21
                 Function *powiFunc = Intrinsic::getOrInsertDeclaration(
                     M, Intrinsic::powi, overloadedTypes);
#else
                 Function *powiFunc = getIntrinsicDeclaration(
                     M, Intrinsic::powi, overloadedTypes);
#endif

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
             std::string funcName = Ty->isDoubleTy() ? "hypot" : "hypotf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.hypot");
           }},
          {"asin",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "asin" : "asinf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.asin");
           }},
          {"acos",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "acos" : "acosf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.acos");
           }},
          {"atan",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "atan" : "atanf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.atan");
           }},
          {"atan2",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "atan2" : "atan2f";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty, Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0], ops[1]}, "herbie.atan2");
           }},
          {"sinh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "sinh" : "sinhf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.sinh");
           }},
          {"cosh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "cosh" : "coshf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.cosh");
           }},
          {"tanh",
           [](IRBuilder<> &b, Module *M,
              const SmallVectorImpl<Value *> &ops) -> Value * {
             Type *Ty = ops[0]->getType();
             std::string funcName = Ty->isDoubleTy() ? "tanh" : "tanhf";
             Function *f = M->getFunction(funcName);
             if (!f) {
               FunctionType *FT = FunctionType::get(Ty, {Ty}, false);
               f = Function::Create(FT, Function::ExternalLinkage, funcName, M);
             }
             return b.CreateCall(f, {ops[0]}, "herbie.tanh");
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
    unsigned depth) {
  if (input) {
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
      expr += " " + operand->toFullExpression(valueToNodeMap, depth + 1);
    }
    expr += ")";
    return expr;
  }
}

void FPLLValue::markAsInput() { input = true; }

void FPLLValue::updateBounds(double lower, double upper) {
  lb = std::min(lb, lower);
  ub = std::max(ub, upper);
  if (EnzymePrintFPOpt)
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
    unsigned depth) {
  return strValue;
}

bool FPConst::hasSymbol() const {
  std::string msg = "Unexpected invocation of `hasSymbol` on an FPConst";
  llvm_unreachable(msg.c_str());
}

void FPConst::markAsInput() { return; }

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

  // if (EnzymePrintFPOpt)
  //   llvm::errs() << "Returning " << strValue << " as " << dtype
  //                << " constant: " << constantValue << "\n";
  return ConstantFP::get(Ty, constantValue);
}

bool FPConst::classof(const FPNode *N) {
  return N->getType() == NodeType::Const;
}