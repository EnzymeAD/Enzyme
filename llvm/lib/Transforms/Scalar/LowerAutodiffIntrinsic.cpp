/*
 * LowerAutodiffIntrinsic.cpp - Lower autodiff intrinsic
 * 
 * Copyright (C) 2019 William S. Moses (enzyme@wsmoses.com) - All Rights Reserved
 *
 * For commercial use of this code please contact the author(s) above.
 *
 * For research use of the code please use the following citation.
 *
 * \misc{mosesenzyme,
    author = {William S. Moses, Tim Kaler},
    title = {Enzyme: LLVM Automatic Differentiation},
    year = {2019},
    howpublished = {\url{https://github.com/wsmoses/AutoDiff/}},
    note = {commit xxxxxxx}
 */

#include "llvm/Transforms/Scalar/LowerAutodiffIntrinsic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

#define DEBUG_TYPE "lower-autodiff-intrinsic"



#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"

void HandleAutoDiff(CallInst *CI) {
  LLVMContext &Context = CI->getContext();

  Value* fn = CI->getArgOperand(0);
  Value* arg0 = CI->getArgOperand(1);

  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BitCastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  fn->dump();
  //llvm::errs() << "fn"<<fn->getValueID()<<"|"<<Value::BlockAddress<<"|"<<Value::ConstantExpr<<"\n";
  Function* todiff = dyn_cast<Function>(fn);
  auto M = CI->getParent()->getParent()->getParent();
  llvm::errs() << todiff << "\n";
  todiff->dump();

  llvm::errs() << "autodifferentiating " << "\n";
  CI->dump();
  todiff->dump();

  ValueToValueMapTy VMap;
  auto newFunc = llvm::CloneFunction(todiff, VMap);

  llvm::ReturnInst* ret = nullptr;
  for (inst_iterator I = inst_begin(newFunc), E = inst_end(newFunc); I != E; ++I) {
    if (auto foundret = dyn_cast<ReturnInst>(&*I)) {
      if (ret) {
        llvm::errs() << "two returns found\n";
        assert(0 && "two returns");
        exit(1);
      }
      ret = foundret;
    }
  }

  ValueToValueMapTy differentials;
  {
  auto BB2 = BasicBlock::Create(Context, "invert", newFunc);
  auto retval = ret->getReturnValue();
  IRBuilder<> Builder2(ret);
  Builder2.CreateBr(BB2);
  ret->eraseFromParent();

  Builder2.SetInsertPoint(BB2);
  differentials[retval] = ConstantFP::get(retval->getType(), 1.0);

  for (inst_iterator I = inst_begin(newFunc), E = --inst_end(newFunc); I != E; ) {
    --E;
    Instruction* inst = &*E;
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      Value* dif0;
      Value* dif1;
      switch(op->getOpcode()) {
        case Instruction::FMul:
          dif0 = Builder2.CreateBinOp(op->getOpcode(), differentials[inst], op->getOperand(0));
          dif1 = Builder2.CreateBinOp(op->getOpcode(), differentials[inst], op->getOperand(1));
          break;
        case Instruction::FAdd:
          dif0 = differentials[inst];
          dif1 = differentials[inst];
          break;
        case Instruction::FSub:
          dif0 = differentials[inst];
          dif1 = Builder2.CreateFNeg(differentials[inst]);
          break;
        case Instruction::FDiv:
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, differentials[inst], op->getOperand(1));
          dif1 = Builder2.CreateFNeg(
              Builder2.CreateBinOp(Instruction::FDiv, differentials[inst],
                Builder2.CreateBinOp(Instruction::FMul, op->getOperand(1), op->getOperand(1))
              )
          );

        default:
          inst->dump();
          llvm::errs() << "cannot handle unknown binary operator\n";
          assert(0 && "unknown binary operator");
          exit(1);
      }

      auto f0 = differentials.find(op->getOperand(0));
      if (f0 != differentials.end()) {
        dif0 = Builder2.CreateFAdd(f0->second, dif0);
      }
      differentials[op->getOperand(0)] = dif0;

      auto f1 = differentials.find(op->getOperand(1));
      if (f1 != differentials.end()) {
        dif1 = Builder2.CreateFAdd(f1->second, dif1);
      }

      differentials[op->getOperand(1)] = dif1;

    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::sqrt: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, differentials[inst],
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.0), op)
          );
          break;
        }
        case Intrinsic::log: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, differentials[inst], op->getOperand(0));
          break;
        }
        case Intrinsic::log2: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, differentials[inst],
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 0.6931471805599453), op->getOperand(0))
          );
          break;
        }
        case Intrinsic::log10: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, differentials[inst],
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.302585092994046), op->getOperand(0))
          );
          break;
        }
        case Intrinsic::exp: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul, differentials[inst], op);
          break;
        }
        case Intrinsic::exp2: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, differentials[inst], op), ConstantFP::get(op->getType(), 0.6931471805599453)
          );
          break;
        }
        case Intrinsic::pow: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, differentials[inst],
              Builder2.CreateBinOp(Instruction::FDiv, op, op->getOperand(0))), op->getOperand(1)
          );

          Value *args[] = {op->getOperand(1)};
          Type *tys[] = {op->getOperand(1)->getType()};
          dif1 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, differentials[inst], op),
            Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
          );

          break;
        }
        case Intrinsic::sin: {
          Value *args[] = {op->getOperand(0)};
          Type *tys[] = {op->getOperand(0)->getType()};
          dif0 = Builder2.CreateBinOp(Instruction::FMul, differentials[inst],
            Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          break;
        }
        case Intrinsic::cos: {
          Value *args[] = {op->getOperand(0)};
          Type *tys[] = {op->getOperand(0)->getType()};
          dif0 = Builder2.CreateBinOp(Instruction::FMul, differentials[inst],
            Builder2.CreateFNeg(
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) )
          );
          break;
        }
        default:
          inst->dump();
          llvm::errs() << "cannot handle unknown intrinsic\n";
          assert(0 && "unknown intrinsic");
          exit(1);
      }

      auto f0 = differentials.find(op->getOperand(0));
      if (f0 != differentials.end()) {
        dif0 = Builder2.CreateFAdd(f0->second, dif0);
      }

      differentials[op->getOperand(0)] = dif0;

      if (dif1) {
        auto f1 = differentials.find(op->getOperand(1));
        if (f1 != differentials.end()) {
          dif1 = Builder2.CreateFAdd(f1->second, dif1);
        }
        differentials[op->getOperand(1)] = dif1;
      }

    } else {
      inst->dump();
      llvm::errs() << "cannot handle above inst\n";
      assert(0 && "unknown inst");
      exit(1);
    }
  }

  Value * retargs[CI->getNumArgOperands()-1];
  assert(CI->getNumArgOperands()-1 == 1);

  {
  int i=0;
  for (auto I = newFunc->arg_begin(), E = newFunc->arg_end(); I != E; I++) {
    I->dump();
    retargs[i] = differentials[I];
    i++;
  }
  }

  newFunc->dump();
  retargs[0]->dump();
  Builder2.CreateRet(retargs[0]);
  }

  newFunc->dump();

  IRBuilder<> Builder(CI);
  Value* args[1] = {arg0};
  Value* diffret = Builder.CreateCall(newFunc, args);
  CI->replaceAllUsesWith(diffret);
  CI->eraseFromParent();
}

static bool lowerAutodiffIntrinsic(Function &F) {
  bool Changed = false;

  for (BasicBlock &BB : F) {

    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE;) {
      Instruction *Inst = &*BI++;
      CallInst *CI = dyn_cast_or_null<CallInst>(Inst);
      if (!CI) continue;

      Function *Fn = CI->getCalledFunction();
      if (Fn && Fn->getIntrinsicID() == Intrinsic::autodiff) {
        HandleAutoDiff(CI);
        Changed = true;
      }
    }
  }

  return Changed;
}

PreservedAnalyses LowerAutodiffIntrinsicPass::run(Function &F,
                                                FunctionAnalysisManager &) {
  if (lowerAutodiffIntrinsic(F))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

namespace {
/// Legacy pass for lowering expect intrinsics out of the IR.
///
/// When this pass is run over a function it uses expect intrinsics which feed
/// branches and switches to provide branch weight metadata for those
/// terminators. It then removes the expect intrinsics from the IR so the rest
/// of the optimizer can ignore them.
class LowerAutodiffIntrinsic : public FunctionPass {
public:
  static char ID;
  LowerAutodiffIntrinsic() : FunctionPass(ID) {
    initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override { return lowerAutodiffIntrinsic(F); }
};
}

char LowerAutodiffIntrinsic::ID = 0;
INITIALIZE_PASS(LowerAutodiffIntrinsic, "lower-autodiff",
                "Lower 'autodiff' Intrinsics", false, false)

FunctionPass *llvm::createLowerAutodiffIntrinsicPass() {
  return new LowerAutodiffIntrinsic();
}
