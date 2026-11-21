//===- InstructionBatcher.cpp
//--------------------------------------------------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file contains an instruction visitor InstructionBatcher that generates
// the batches all LLVM instructions.
//
//===----------------------------------------------------------------------===//

#include "InstructionBatcher.h"

#include "llvm/IR/InstVisitor.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/Casting.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "DiffeGradientUtils.h"
#include "GradientUtils.h"

using namespace llvm;

InstructionBatcher::InstructionBatcher(
    Function *oldFunc, Function *newFunc, unsigned width,
    ValueMap<const Value *, std::vector<Value *>> &vectorizedValues,
    ValueToValueMapTy &originalToNewFn, SmallPtrSetImpl<Value *> &toVectorize,
    EnzymeLogic &Logic)
    : hasError(false), vectorizedValues(vectorizedValues),
      originalToNewFn(originalToNewFn), toVectorize(toVectorize), width(width),
      Logic(Logic) {}

Value *InstructionBatcher::getNewOperand(unsigned int i, llvm::Value *op) {
  if (auto meta = dyn_cast<MetadataAsValue>(op)) {
    auto md = meta->getMetadata();
    if (auto val = dyn_cast<ValueAsMetadata>(md))
      return MetadataAsValue::get(
          op->getContext(),
          ValueAsMetadata::get(getNewOperand(i, val->getValue())));
    return meta;
  }

  if (auto *BA = dyn_cast<BlockAddress>(op)) {
    auto found = originalToNewFn.find(BA->getBasicBlock());
    if (found != originalToNewFn.end()) {
      auto *NewBB = cast<BasicBlock>(&*found->second);
      return BlockAddress::get(NewBB->getParent(), NewBB);
    }
    return BA;
  }

  if (isa<Constant>(op)) {
    return op;
  } else if (toVectorize.count(op) != 0) {
    auto found = vectorizedValues.find(op);
    assert(found != vectorizedValues.end() &&
           "Expected vectorized operand in mapping");
    return found->second[i];
  } else {
    auto found = originalToNewFn.find(op);
    assert(found != originalToNewFn.end() &&
           "Expected scalar operand in originalToNewFn mapping");
    return found->second;
  }
}

void InstructionBatcher::visitInstruction(llvm::Instruction &inst) {
  auto found = vectorizedValues.find(&inst);
  assert(found != vectorizedValues.end() &&
         "Expected vectorized instruction in mapping");
  auto placeholders = found->second;
  Instruction *placeholder = cast<Instruction>(placeholders[0]);

  for (unsigned i = 1; i < width; ++i) {
    ValueToValueMapTy vmap;
    Instruction *new_inst = placeholder->clone();
    vmap[placeholder] = new_inst;

    for (unsigned j = 0; j < inst.getNumOperands(); ++j) {
      Value *op = inst.getOperand(j);

      if (auto meta = dyn_cast<MetadataAsValue>(op))
        if (!isa<ValueAsMetadata>(meta->getMetadata()))
          continue;

      Value *new_op = getNewOperand(i, op);
      vmap[placeholder->getOperand(j)] = new_op;
    }

    if (placeholders.size() == width) {
      // Instructions which return a value
      Instruction *placeholder = cast<Instruction>(placeholders[i]);
      assert(!placeholder->getType()->isVoidTy() &&
             "Expected non-void placeholder for multi-lane instruction");

      ReplaceInstWithInst(placeholder, new_inst);
      vectorizedValues[&inst][i] = new_inst;
    } else if (placeholders.size() == 1) {
      // Instructions which don't return a value
      assert(placeholder->getType()->isVoidTy() &&
             "Expected void instruction for single placeholder");

      Instruction *prev = cast<Instruction>(vectorizedValues[&inst].back());
      IRBuilder<> Builder2 = prev->getNextNode()
                                 ? IRBuilder<>(prev->getNextNode())
                                 : IRBuilder<>(prev->getParent());
      Builder2.SetCurrentDebugLocation(DebugLoc());
      Builder2.Insert(new_inst);
      vectorizedValues[&inst].push_back(new_inst);
    } else {
      llvm_unreachable("Unexpected number of values in mapping");
    }

    RemapInstruction(new_inst, vmap, RF_NoModuleLevelChanges);

    if (!inst.getType()->isVoidTy() && inst.hasName())
      new_inst->setName(inst.getName() + Twine(i));
  }
}

void InstructionBatcher::visitPHINode(PHINode &phi) {
  PHINode *placeholder = cast<PHINode>(vectorizedValues[&phi][0]);

  for (unsigned i = 1; i < width; ++i) {
    ValueToValueMapTy vmap;
    Instruction *new_phi = placeholder->clone();
    vmap[placeholder] = new_phi;

    for (unsigned j = 0; j < phi.getNumIncomingValues(); ++j) {
      Value *orig_block = phi.getIncomingBlock(j);
      BasicBlock *new_block = cast<BasicBlock>(originalToNewFn[orig_block]);
      Value *orig_val = phi.getIncomingValue(j);
      Value *new_val = getNewOperand(i, orig_val);

      vmap[placeholder->getIncomingValue(j)] = new_val;
      vmap[new_block] = new_block;
    }

    RemapInstruction(new_phi, vmap, RF_NoModuleLevelChanges);
    Instruction *placeholder = cast<Instruction>(vectorizedValues[&phi][i]);
    ReplaceInstWithInst(placeholder, new_phi);
    new_phi->setName(phi.getName());
    vectorizedValues[&phi][i] = new_phi;
  }
}

void InstructionBatcher::visitSwitchInst(llvm::SwitchInst &inst) {
  // TODO: runtime check
  hasError = true;
  EmitFailure("SwitchConditionCannotBeVectorized", inst.getDebugLoc(), &inst,
              "switch conditions have to be scalar values", inst);
  return;
}

#if LLVM_VERSION_MAJOR >= 24
void InstructionBatcher::visitCondBrInst(llvm::CondBrInst &branch) {
  visitBranch(branch);
}
void InstructionBatcher::visitUncondBrInst(llvm::UncondBrInst &branch) {
  visitBranch(branch);
}
void InstructionBatcher::visitBranch(llvm::Instruction &branch) {
#else
void InstructionBatcher::visitBranchInst(llvm::BranchInst &branch) {
#endif
  // TODO: runtime check
  hasError = true;
  EmitFailure("BranchConditionCannotBeVectorized", branch.getDebugLoc(),
              &branch, "branch conditions have to be scalar values", branch);
  return;
}

void InstructionBatcher::visitReturnInst(llvm::ReturnInst &ret) {
  auto found = originalToNewFn.find(ret.getParent());
  assert(found != originalToNewFn.end());
  BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
  IRBuilder<> Builder2 = IRBuilder<>(nBB);
  Builder2.SetCurrentDebugLocation(DebugLoc());
  ReturnInst *placeholder = cast<ReturnInst>(nBB->getTerminator());
  SmallVector<Value *, 4> rets;

  for (unsigned j = 0; j < ret.getNumOperands(); ++j) {
    Value *op = ret.getOperand(j);
    for (unsigned i = 0; i < width; ++i) {
      Value *new_op = getNewOperand(i, op);
      rets.push_back(new_op);
    }
  }

  if (ret.getNumOperands() != 0) {
    if (nBB->getParent()->getReturnType() == ret.getOperand(0)->getType()) {
      hasError = true;
      EmitFailure("ReturnValueCannotBeVectorized", ret.getDebugLoc(), &ret,
                  "return value cannot be vectorized when scalar return is "
                  "requested",
                  ret);
      return;
    }
#if LLVM_VERSION_MAJOR > 22
    auto ret = Builder2.CreateAggregateRet(rets);
#else
    auto ret = Builder2.CreateAggregateRet(rets.data(), width);
#endif
    ret->setDebugLoc(placeholder->getDebugLoc());
    placeholder->eraseFromParent();
  }
}

void InstructionBatcher::visitCallInst(llvm::CallInst &call) {
  auto found = vectorizedValues.find(&call);
  assert(found != vectorizedValues.end() &&
         "Expected vectorized call in mapping");
  auto placeholders = found->second;
  Instruction *placeholder = cast<Instruction>(placeholders[0]);
  IRBuilder<> Builder2(placeholder);
  Builder2.SetCurrentDebugLocation(DebugLoc());
  auto orig_func = getFunctionFromCall(&call);

  bool canBatchCallee =
      orig_func && !orig_func->isDeclaration() && !orig_func->isVarArg() &&
      call.getFunctionType() == orig_func->getFunctionType() &&
      !call.hasOperandBundles();

  if (!canBatchCallee)
    return visitInstruction(call);

  SmallVector<Value *, 4> args;
  SmallVector<BATCH_TYPE, 4> arg_types;
#if LLVM_VERSION_MAJOR >= 14
  for (unsigned j = 0; j < call.arg_size(); ++j) {
#else
  for (unsigned j = 0; j < call.getNumArgOperands(); ++j) {
#endif
    Value *op = call.getArgOperand(j);

    if (toVectorize.count(op) != 0) {
      Type *aggTy = GradientUtils::getShadowType(op->getType(), width);
      Value *agg = UndefValue::get(aggTy);
      for (unsigned i = 0; i < width; i++) {
        auto found = vectorizedValues.find(op);
        assert(found != vectorizedValues.end() &&
               "Expected vectorized call operand in mapping");
        Value *new_op = found->second[i];
        agg = Builder2.CreateInsertValue(agg, new_op, {i});
      }
      args.push_back(agg);
      arg_types.push_back(BATCH_TYPE::VECTOR);
    } else {
      args.push_back(getNewOperand(0, op));
      arg_types.push_back(BATCH_TYPE::SCALAR);
    }
  }

  BATCH_TYPE ret_type = orig_func->getReturnType()->isVoidTy()
                            ? BATCH_TYPE::SCALAR
                            : BATCH_TYPE::VECTOR;

  Function *new_func = Logic.CreateBatch(RequestContext(&call, &Builder2),
                                         orig_func, width, arg_types, ret_type);
  if (!new_func) {
    hasError = true;
    return;
  }
  CallInst *new_call = Builder2.CreateCall(new_func->getFunctionType(),
                                           new_func, args, call.getName());
  new_call->setCallingConv(new_func->getCallingConv());
  new_call->setDebugLoc(placeholder->getDebugLoc());

  if (!call.getType()->isVoidTy()) {
    for (unsigned i = 0; i < width; ++i) {
      Instruction *placeholder = dyn_cast<Instruction>(placeholders[i]);
      ExtractValueInst *ret = ExtractValueInst::Create(
          new_call, {i},
          "unwrap" + (call.hasName() ? "." + call.getName() + Twine(i) : ""));
      ReplaceInstWithInst(placeholder, ret);
      vectorizedValues[&call][i] = ret;
    }
  } else {
    vectorizedValues[&call][0] = new_call;
    placeholder->replaceAllUsesWith(new_call);
    placeholder->eraseFromParent();
  }
}
