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

Function *CloneFunctionWithReturns(Function *F, SmallVector<ReturnInst*, 8>& Returns) {
 std::vector<Type*> RetTypes;
 RetTypes.push_back(F->getReturnType());
 std::vector<Type*> ArgTypes;

 ValueToValueMapTy VMap;

 // The user might be deleting arguments to the function by specifying them in
 // the VMap.  If so, we need to not add the arguments to the arg ty vector
 //
 for (const Argument &I : F->args())
   if (VMap.count(&I) == 0) // Haven't mapped the argument to anything yet?
   {
     ArgTypes.push_back(I.getType());
     RetTypes.push_back(I.getType());
   }

 auto RetType = StructType::get(F->getContext(), RetTypes);

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), F->getName(), F->getParent());

 // Loop over the arguments, copying the names of the mapped arguments over...
 Function::arg_iterator DestI = NewF->arg_begin();


 for (const Argument & I : F->args())
   if (VMap.count(&I) == 0) {     // Is this argument preserved?
     DestI->setName(I.getName()); // Copy the name over...
     VMap[&I] = &*DestI++;        // Add mapping to VMap
   }

 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);

 return NewF;
}

#include "llvm/IR/Constant.h"
#include <deque>
#include "llvm/IR/CFG.h"

Function* CreatePrimalAndGradient(Function* todiff) {
  auto M = todiff->getParent();
  auto& Context = M->getContext();
  todiff->dump();

  llvm::errs() << "autodifferentiating " << "\n";
  todiff->dump();

  SmallVector<ReturnInst*, 8> Returns;
  auto newFunc = CloneFunctionWithReturns(todiff, Returns);

  ValueToValueMapTy differentials;

  std::deque<BasicBlock*> blockstodo;
  for(auto a:Returns) {
    blockstodo.push_back(a->getParent());
  }

  llvm::Value* retval = Returns[0]->getReturnValue();
  assert(Returns.size() == 1);

  SmallVector<BasicBlock*, 12> fnthings;
  for (BasicBlock &BB : *newFunc) {
     fnthings.push_back(&BB);
  }
  ValueMap<BasicBlock*,BasicBlock*> reverseBlocks;
  for (BasicBlock *BB : fnthings) {
    auto BB2 = BasicBlock::Create(Context, "invert" + BB->getName(), newFunc);
    reverseBlocks[BB] = BB2;
  }

  IRBuilder<> entryBuilder(&newFunc->getEntryBlock().front());

  ValueToValueMapTy scopeMap;

  while(blockstodo.size() > 0) {
    auto BB = blockstodo.front();
    blockstodo.pop_front();

    auto BB2 = reverseBlocks[BB];
    assert(BB2);
    BB2->dump();
    if (BB2->size() != 0) {
        llvm::errs() << "skipping block" << BB2->getName() << "\n";
        continue;
    }

    IRBuilder<> Builder2(BB2);
    differentials[BB] = BB2;

    auto lookup = [&](Value* val) -> Value* {
        if (auto inst = dyn_cast<Instruction>(val)) {
        if (scopeMap.find(val) == scopeMap.end()) {
            scopeMap[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'");
            IRBuilder <> v(inst);
            auto st = v.CreateStore(val, scopeMap[val]);
            st->moveAfter(inst);
        }
        return Builder2.CreateLoad(scopeMap[val]);
        }
        return val;
    };

  auto term = BB->getTerminator();

  if(auto op = dyn_cast<ReturnInst>(term)) {
      auto retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.CreateBr(BB2);
      op->eraseFromParent();
      differentials[retval] = entryBuilder.CreateAlloca(retval->getType(), nullptr, retval->getName()+"'");
      entryBuilder.CreateStore(ConstantFP::get(retval->getType(), 1.0), differentials[retval]);
  } else if (isa<BranchInst>(term)) {
        
  } else {
    llvm::errs() << "unknown return instance\n";
    assert(0 && "unknown return inst");
  }

  for(auto PB :successors(BB)) {
    
    for (auto I = PB->begin(), E = PB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {
            differentials[PN->getIncomingValueForBlock(BB)] = differentials[PN];
        } else break;
    }
  }
  

  for (auto I = BB->rbegin(), E = BB->rend(); I != E; I++) {
    Instruction* inst = &*I;
    inst->dump();
    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      Value* dif0;
      Value* dif1;
      switch(op->getOpcode()) {
        case Instruction::FMul:
          dif0 = Builder2.CreateBinOp(op->getOpcode(), Builder2.CreateLoad(differentials[inst]), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          dif1 = Builder2.CreateBinOp(op->getOpcode(), Builder2.CreateLoad(differentials[inst]), lookup(op->getOperand(0)), "diffe"+op->getOperand(1)->getName());
          break;
        case Instruction::FAdd:
          dif0 = Builder2.CreateLoad(differentials[inst]);
          dif1 = Builder2.CreateLoad(differentials[inst]);
          break;
        case Instruction::FSub:
          dif0 = Builder2.CreateLoad(differentials[inst]);
          dif1 = Builder2.CreateFNeg(Builder2.CreateLoad(differentials[inst]));
          break;
        case Instruction::FDiv:
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, Builder2.CreateLoad(differentials[inst]), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          dif1 = Builder2.CreateFNeg(
              Builder2.CreateBinOp(Instruction::FDiv, 
                Builder2.CreateFMul(Builder2.CreateLoad(differentials[inst]), op),
                lookup(op->getOperand(1)))
          );

        default:
          inst->dump();
          llvm::errs() << "cannot handle unknown binary operator\n";
          assert(0 && "unknown binary operator");
          exit(1);
      }

      auto f0 = differentials.find(op->getOperand(0));
      if (f0 == differentials.end()) {
        differentials[op->getOperand(0)] = entryBuilder.CreateAlloca(op->getOperand(0)->getType(), nullptr, op->getOperand(0)->getName()+"'");
        entryBuilder.CreateStore(Constant::getNullValue(op->getOperand(0)->getType()), differentials[op->getOperand(0)]);
      }
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(0)]), dif0), differentials[op->getOperand(0)]);

      auto f1 = differentials.find(op->getOperand(1));
      if (f1 == differentials.end()) {
        differentials[op->getOperand(1)] = entryBuilder.CreateAlloca(op->getOperand(1)->getType(), nullptr, op->getOperand(1)->getName()+"'");
        entryBuilder.CreateStore(Constant::getNullValue(op->getOperand(1)->getType()), differentials[op->getOperand(1)]);
      }
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(1)]), dif1), differentials[op->getOperand(1)]);

    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      op->dump();
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::sqrt: {
          newFunc->dump();
          differentials[inst]->dump();
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, Builder2.CreateLoad(differentials[inst]),
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.0), lookup(op))
          );
          break;
        }
        case Intrinsic::fabs: {
          auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), ConstantFP::get(op->getOperand(0)->getType(), 0));
          dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), -1), ConstantFP::get(op->getOperand(0)->getType(), 1));
          break;
        }
        case Intrinsic::log: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, Builder2.CreateLoad(differentials[inst]), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, Builder2.CreateLoad(differentials[inst]),
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
          );
          break;
        }
        case Intrinsic::log10: {
          dif0 = Builder2.CreateBinOp(Instruction::FDiv, Builder2.CreateLoad(differentials[inst]),
            Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
          );
          break;
        }
        case Intrinsic::exp: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
          );
          break;
        }
        case Intrinsic::pow: {
          dif0 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]),
              Builder2.CreateBinOp(Instruction::FDiv, lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
          );

          Value *args[] = {op->getOperand(1)};
          Type *tys[] = {op->getOperand(1)->getType()};
          dif1 = Builder2.CreateBinOp(Instruction::FMul,
            Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]), lookup(op)),
            Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
          );

          break;
        }
        case Intrinsic::sin: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          dif0 = Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]),
            Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          break;
        }
        case Intrinsic::cos: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          dif0 = Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]),
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
      if (f0 == differentials.end()) {
        differentials[op->getOperand(0)] = entryBuilder.CreateAlloca(op->getOperand(0)->getType(), nullptr, op->getOperand(0)->getName()+"'");
        entryBuilder.CreateStore(Constant::getNullValue(op->getOperand(0)->getType()), differentials[op->getOperand(0)]);
      }
      llvm::errs() << "diffe\n";
      differentials[op->getOperand(0)]->dump();
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(0)]), dif0), differentials[op->getOperand(0)]);

      if(dif1) {
      auto f1 = differentials.find(op->getOperand(1));
      if (f1 == differentials.end()) {
        differentials[op->getOperand(1)] = entryBuilder.CreateAlloca(op->getOperand(1)->getType(), nullptr, op->getOperand(1)->getName()+"'");
        entryBuilder.CreateStore(Constant::getNullValue(op->getOperand(1)->getType()), differentials[op->getOperand(1)]);
      }
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(1)]), dif1), differentials[op->getOperand(1)]);
      }
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {
        if(auto called = op->getCalledFunction()) {
              auto newcalled = CreatePrimalAndGradient(dyn_cast<Function>(called));
              SmallVector<Value*, 8> args;
              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                args.push_back(lookup(op->getArgOperand(i)));
              }
              auto diffes = Builder2.CreateCall(newcalled, args);
              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                unsigned idxs[] = {i+1};
                auto diffeadd = Builder2.CreateFMul(Builder2.CreateLoad(differentials[inst]), Builder2.CreateExtractValue(diffes, idxs));
                auto f1 = differentials.find(args[i]);
                if (f1 == differentials.end()) {
                  differentials[args[i]] = entryBuilder.CreateAlloca(args[i]->getType());
                  entryBuilder.CreateStore(Constant::getNullValue(args[i]->getType()), differentials[args[i]]);
                }
                Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[args[i]]), diffeadd), differentials[args[i]]);
              }
        } else {
            op->dump();
            M->dump();
            llvm::errs() << "cannot handle non const function\n";
            assert(0 && "unknown non const function");
            exit(1);
        }

    } else if(auto op = dyn_cast_or_null<SelectInst>(inst)) {

      auto dif1 = Builder2.CreateSelect(lookup(op->getOperand(0)), Builder2.CreateLoad(differentials[op]), Constant::getNullValue(op->getOperand(1)->getType()), "diffe"+op->getOperand(1)->getName());
      auto dif2 = Builder2.CreateSelect(lookup(op->getOperand(0)), Constant::getNullValue(op->getOperand(2)->getType()), Builder2.CreateLoad(differentials[op]), "diffe"+op->getOperand(2)->getName());
      
      auto f1 = differentials.find(op->getOperand(1));
      if (f1 == differentials.end()) {
        differentials[op->getOperand(1)] = entryBuilder.CreateAlloca(op->getOperand(1)->getType(), nullptr, op->getOperand(1)->getName()+"'");
        entryBuilder.CreateStore(Constant::getNullValue(op->getOperand(1)->getType()), differentials[op->getOperand(1)]);
      }
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(1)]), dif1), differentials[op->getOperand(1)]);

      auto f2 = differentials.find(op->getOperand(2));
      if (f2 == differentials.end()) {
        differentials[op->getOperand(2)] = entryBuilder.CreateAlloca(op->getOperand(2)->getType(), nullptr, op->getOperand(2)->getName()+"'");
        entryBuilder.CreateStore(differentials[op->getOperand(2)], Constant::getNullValue(op->getOperand(2)->getType()));
      }
      Builder2.CreateStore(differentials[op->getOperand(2)], Builder2.CreateFAdd(Builder2.CreateLoad(differentials[op->getOperand(2)]), dif2));

    } else if(isa<CmpInst>(inst) || isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<AllocaInst>(inst) || isa<StoreInst>(inst) ) {
        continue;
    } else {
      inst->dump();
      inst->getParent()->dump();
      inst->getParent()->getParent()->dump();
      llvm::errs() << "cannot handle above inst\n";
      assert(0 && "unknown inst");
      exit(1);
    }
  }

  unsigned predcount = 0;
  for (BasicBlock *Pred : predecessors(BB)) {
    predcount++;
    if (Pred->size() != 0)
      blockstodo.push_back(Pred);
  }

  if (predcount == 0) {
    Value * retargs[newFunc->getFunctionType()->getNumParams()+1] = {0};
    retargs[0] = retval;
    llvm::errs() << "retval" << retval << "\n";
    llvm::errs() << "retval" << *retval << "\n";
    assert(retargs[0]);
    retargs[0]->dump();
    {
    int i=1;
    for (auto& I: newFunc->args()) {
      retargs[i] = Builder2.CreateLoad(differentials[(Value*)&I]);
      retargs[i]->dump();
      i++;
    }
    }
    Value* toret = UndefValue::get(newFunc->getReturnType());
    toret->dump();
    for(unsigned i=0; i<newFunc->getFunctionType()->getNumParams()+1; i++) {
      unsigned idx[] = { i };
      toret = Builder2.CreateInsertValue(toret, retargs[i], idx);
    }
    Builder2.CreateRet(toret);
    continue;
  } 
  
  BasicBlock* preds[predcount] = {0};
  predcount = 0;
  for (BasicBlock *Pred : predecessors(BB)) {
    preds[predcount] = Pred;
    predcount++;
  }

  
  if (predcount == 1) {
    Builder2.CreateBr(reverseBlocks[preds[0]]);
  } else if (predcount == 2) {
    IRBuilder <> pbuilder(&BB->front());
    auto phi = pbuilder.CreatePHI(Type::getInt1Ty(Context), 2);
    phi->addIncoming(ConstantInt::getTrue(phi->getType()), preds[0]);
    phi->addIncoming(ConstantInt::getFalse(phi->getType()), preds[1]);
    Builder2.CreateCondBr(phi, reverseBlocks[preds[0]], reverseBlocks[preds[1]]);
  } else {
    newFunc->dump();
    BB->dump();
    printf("predcount = %d\n", predcount);
    assert(0 && "Need to determine where came from");
  }


  }

  newFunc->dump();

  return newFunc;
}

void HandleAutoDiff(CallInst *CI) {

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

  auto newFunc = CreatePrimalAndGradient(dyn_cast<Function>(fn));
  newFunc->dump();

  IRBuilder<> Builder(CI);
  Value* args[1] = {arg0};
  Value* diffret = Builder.CreateCall(newFunc, args);
  unsigned idxs[] = {1};
  diffret = Builder.CreateExtractValue(diffret, idxs);
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
