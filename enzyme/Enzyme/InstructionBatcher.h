//===- InstructionBatcher.h
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
// This file contains an instruction visitor AdjointGenerator that generates
// the corresponding augmented forward pass code, and adjoints for all
// LLVM instructions.
//
//===----------------------------------------------------------------------===//

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

#include "GradientUtils.h"

using namespace llvm;

class InstructionBatcher : public llvm::InstVisitor<InstructionBatcher> {
public:
  InstructionBatcher(
      Function *oldFunc, Function *newFunc, unsigned width,
      ValueMap<const Value *, std::vector<Value *>> &vectorizedValues,
      ValueToValueMapTy &originalToNewFn, SmallPtrSetImpl<Value *> &toVectorize,
      EnzymeLogic &Logic)
      : vectorizedValues(vectorizedValues), originalToNewFn(originalToNewFn),
        toVectorize(toVectorize), width(width), Logic(Logic) {}

private:
  ValueMap<const Value *, std::vector<Value *>> &vectorizedValues;
  ValueToValueMapTy &originalToNewFn;
  SmallPtrSetImpl<Value *> &toVectorize;
  unsigned width;
  EnzymeLogic &Logic;

public:
  void visitInstruction(llvm::Instruction &inst) {
    auto found = originalToNewFn.find(inst.getParent());
    assert(found != originalToNewFn.end());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);
    unsigned actualWidth = toVectorize.count(&inst) != 0 ? width : 1;

    for (unsigned i = 0; i < actualWidth; ++i) {
      ValueToValueMapTy vmap;
      Instruction *new_inst = inst.clone();
      vmap[&inst] = new_inst;

      for (unsigned j = 0; j < inst.getNumOperands(); ++j) {
        Value *op = inst.getOperand(j);
        Value *new_op;
        if (isa<Constant>(op)) {
          new_op = op;
        } else if (toVectorize.count(op) != 0) {
          new_op = vectorizedValues[op][i];
        } else {
          new_op = originalToNewFn[op];
        }
        vmap[op] = new_op;
      }

      if (!inst.getType()->isVoidTy() && inst.hasName()) {
        Builder2.Insert(new_inst, inst.getName() + std::to_string(i));
      } else {
        Builder2.Insert(new_inst);
      }
      RemapInstruction(new_inst, vmap, RF_NoModuleLevelChanges);
      if (toVectorize.count(&inst) != 0) {
        vectorizedValues[&inst].push_back(new_inst);
      } else {
        originalToNewFn[&inst] = new_inst;
      }
    }
  }

  void visitPHINode(PHINode &phi) {
    auto found = originalToNewFn.find(phi.getParent());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);

    unsigned actualWidth = toVectorize.count(&phi) != 0 ? width : 1;

    for (unsigned i = 0; i < actualWidth; ++i) {
      ValueToValueMapTy vmap;
      Instruction *new_phi = phi.clone();
      vmap[&phi] = new_phi;

      for (unsigned j = 0; j < phi.getNumIncomingValues(); ++j) {
        Value *orig_block = phi.getIncomingBlock(j);
        BasicBlock *new_block = cast<BasicBlock>(originalToNewFn[orig_block]);
        Value *orig_val = phi.getIncomingValue(j);
        Value *new_val;
        if (isa<Constant>(orig_val)) {
          new_val = orig_val;
        } else if (toVectorize.count(orig_val) != 0) {
          new_val = vectorizedValues[orig_val][i];
        } else {
          new_val = originalToNewFn[orig_val];
        }
        vmap[orig_val] = new_val;
        vmap[orig_block] = new_block;
      }

      RemapInstruction(new_phi, vmap, RF_NoModuleLevelChanges);
      Instruction *placeholder;
      if (toVectorize.count(&phi) != 0) {
        placeholder = cast<Instruction>(vectorizedValues[&phi][i]);
      } else {
        placeholder = cast<Instruction>(originalToNewFn[&phi]);
      }
      ReplaceInstWithInst(placeholder, new_phi);
      new_phi->setName(phi.getName());

      if (toVectorize.count(&phi) != 0) {
        vectorizedValues[&phi][i] = new_phi;
      } else {
        originalToNewFn[&phi] = new_phi;
      }
    }
  }

  void visitSwitchInst(llvm::SwitchInst &inst) {
    if (toVectorize.count(inst.getCondition()) != 0) {
      EmitFailure("SwitchConditionCannotBeVectorized", inst.getDebugLoc(),
                  &inst, "switch conditions have to be scalar values", inst);
      llvm_unreachable("vectorized control flow is not allowed");
    }

    auto found = originalToNewFn.find(inst.getParent());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);

    Instruction *new_switch = inst.clone();
    ValueToValueMapTy vmap;
    vmap[&inst] = new_switch;

    for (unsigned j = 0; j < inst.getNumOperands(); ++j) {
      Value *op = inst.getOperand(j);
      Value *new_op = originalToNewFn[op];
      vmap[op] = new_op;
    }

    Builder2.Insert(new_switch, inst.getName());
    RemapInstruction(new_switch, vmap, RF_NoModuleLevelChanges);
  }

  void visitBranchInst(llvm::BranchInst &branch) {
    if (branch.isConditional() &&
        toVectorize.count(branch.getCondition()) != 0) {
      EmitFailure("BranchConditionCannotBeVectorized", branch.getDebugLoc(),
                  &branch, "branch conditions have to be scalar values",
                  branch);
      llvm_unreachable("vectorized control flow is not allowed");
    }

    auto found = originalToNewFn.find(branch.getParent());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);

    Instruction *new_branch = branch.clone();
    ValueToValueMapTy vmap;
    vmap[&branch] = new_branch;

    for (unsigned j = 0; j < branch.getNumOperands(); ++j) {
      Value *op = branch.getOperand(j);
      Value *new_op = originalToNewFn[op];
      vmap[op] = new_op;
    }

    Builder2.Insert(new_branch);
    RemapInstruction(new_branch, vmap, RF_NoModuleLevelChanges);
  }

  void visitReturnInst(llvm::ReturnInst &ret) {
    auto found = originalToNewFn.find(ret.getParent());
    assert(found != originalToNewFn.end());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);
    SmallVector<Value *, 0> rets;

    for (unsigned j = 0; j < ret.getNumOperands(); ++j) {
      Value *op = ret.getOperand(j);
      for (unsigned i = 0; i < width; ++i) {
        Value *new_op;
        if (toVectorize.count(op) != 0) {
          new_op = vectorizedValues[op][i];
        } else {
          new_op = originalToNewFn[op];
        }
        rets.push_back(new_op);
      }
    }

    if (ret.getNumOperands() == 0) {
      Builder2.CreateRetVoid();
    } else {
      Builder2.CreateAggregateRet(rets.data(), width);
    }
  }

  void visitCallInst(llvm::CallInst &call) {
    auto found = originalToNewFn.find(call.getParent());
    assert(found != originalToNewFn.end());
    BasicBlock *nBB = dyn_cast<BasicBlock>(&*found->second);
    IRBuilder<> Builder2 = IRBuilder<>(nBB);

    if (call.isDebugOrPseudoInst()) {
      return;
    }

    if (!toVectorize.count(&call) != 0) {
      ValueToValueMapTy vmap;
      Instruction *new_call = call.clone();
      vmap[&call] = new_call;

      for (unsigned j = 0; j < call.getNumArgOperands(); ++j) {
        Value *arg = call.getArgOperand(j);
        Value *new_arg;
        if (isa<Constant>(arg)) {
          new_arg = arg;
        } else {
          new_arg = originalToNewFn[arg];
          assert(originalToNewFn.find(arg) != originalToNewFn.end());
        }
        vmap[arg] = new_arg;
      }

      Builder2.Insert(new_call, call.getName());
      RemapInstruction(new_call, vmap, RF_NoModuleLevelChanges);
      originalToNewFn[&call] = new_call;
      return;
    }

    SmallVector<Value *, 0> args;
    SmallVector<BATCH_TYPE> arg_types;

    for (unsigned j = 0; j < call.getNumArgOperands(); ++j) {
      // make sure this does not include the called func!
      Value *op = call.getArgOperand(j);

      if (toVectorize.count(op) != 0) {
        Type *aggTy = GradientUtils::getShadowType(op->getType(), width);
        Value *agg = UndefValue::get(aggTy);
        for (unsigned i = 0; i < width; i++) {
          Value *new_op = vectorizedValues[op][i];
          Builder2.CreateInsertValue(agg, new_op, {i});
        }
        args.push_back(agg);
        arg_types.push_back(BATCH_TYPE::VECTOR);
      } else if (isa<Constant>(op)) {
        args.push_back(op);
        arg_types.push_back(BATCH_TYPE::SCALAR);
      } else {
        Value *arg = originalToNewFn[op];
        args.push_back(arg);
        arg_types.push_back(BATCH_TYPE::SCALAR);
      }
    }

    Function *orig_func = call.getCalledFunction();
    Function *new_func = Logic.CreateBatch(orig_func, width, arg_types);
    CallInst *new_call = Builder2.CreateCall(new_func->getFunctionType(),
                                             new_func, args, call.getName());

    if (!call.getFunctionType()->getReturnType()->isVoidTy()) {
      for (unsigned i = 0; i < width; ++i) {
        Value *ret = Builder2.CreateExtractValue(
            new_call, {i}, "unwrap." + call.getName() + std::to_string(i));
        vectorizedValues[&call].push_back(ret);
      }
    }
  }
};
