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

#include "llvm/ADT/SmallSet.h"
using namespace llvm;

#define DEBUG_TYPE "lower-autodiff-intrinsic"

#include <utility> 

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"

Function *CloneFunctionWithReturns(Function *F, SmallVector<ReturnInst*, 8>& Returns, ValueToValueMapTy& ptrInputs, const SmallSet<unsigned,4>& constant_args, SmallPtrSetImpl<Value*> &constants) {
 std::vector<Type*> RetTypes;
 RetTypes.push_back(F->getReturnType());
 std::vector<Type*> ArgTypes;

 ValueToValueMapTy VMap;

 // The user might be deleting arguments to the function by specifying them in
 // the VMap.  If so, we need to not add the arguments to the arg ty vector
 //
 unsigned argno = 0;
 for (const Argument &I : F->args()) {
     ArgTypes.push_back(I.getType());
     if (constant_args.count(argno)) {
        argno++;
        continue;
     }
     if (I.getType()->isPointerTy()) {
       ArgTypes.push_back(I.getType());
     } else {
       RetTypes.push_back(I.getType());
     }
     argno++;
  }

 auto RetType = StructType::get(F->getContext(), RetTypes);

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), F->getName(), F->getParent());

 bool hasPtrInput = false;

 unsigned ii = 0, jj = 0;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
     
     if (constant_args.count(ii)) {
        constants.insert(j);
        llvm::errs() << "inserting into constants: " << *j << "\n";
     } 
     
     if ( (constant_args.count(ii) == 0) && i->getType()->isPointerTy()) {
       VMap[i] = j;
       hasPtrInput = true;
       ptrInputs[j] = (j+1);
       NewF->addParamAttr(jj, Attribute::NoCapture);

       i++;
       ii++;

       j++;
       j++;
       jj+=2;
     } else {
       VMap[i] = j;
       i++;
       ii++;
       j++;
       jj++;
     }
 }

 // Loop over the arguments, copying the names of the mapped arguments over...
 Function::arg_iterator DestI = NewF->arg_begin();


 for (const Argument & I : F->args())
   if (VMap.count(&I) == 0) {     // Is this argument preserved?
     DestI->setName(I.getName()); // Copy the name over...
     VMap[&I] = &*DestI++;        // Add mapping to VMap
   }

 CloneFunctionInto(NewF, F, VMap, F->getSubprogram() != nullptr, Returns, "",
                   nullptr);

 if (hasPtrInput) {
    if (NewF->hasFnAttribute(Attribute::ReadNone)) {
    NewF->removeFnAttr(Attribute::ReadNone);
    }
    if (NewF->hasFnAttribute(Attribute::ReadOnly)) {
    NewF->removeFnAttr(Attribute::ReadOnly);
    }
 }

 return NewF;
}

#include "llvm/IR/Constant.h"
#include <deque>
#include "llvm/IR/CFG.h"

Function* CreatePrimalAndGradient(Function* todiff, const SmallSet<unsigned,4>& constant_args) {
  auto M = todiff->getParent();
  auto& Context = M->getContext();

  SmallVector<ReturnInst*, 8> Returns;
  ValueToValueMapTy ptrInputs;
  SmallPtrSet<Value*,4> constants;
  auto newFunc = CloneFunctionWithReturns(todiff, Returns, ptrInputs, constant_args, constants);

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

  auto inversionAllocs = BasicBlock::Create(Context, "allocsForInversion", newFunc);

  ValueMap<BasicBlock*,BasicBlock*> reverseBlocks;
  for (BasicBlock *BB : fnthings) {
    auto BB2 = BasicBlock::Create(Context, "invert" + BB->getName(), newFunc);
    reverseBlocks[BB] = BB2;
  }


  IRBuilder<> entryBuilder(inversionAllocs);

  ValueToValueMapTy scopeMap;
  SmallPtrSet<Value*, 10> addedStores;

  while(blockstodo.size() > 0) {
    auto BB = blockstodo.front();
    blockstodo.pop_front();

    auto BB2 = reverseBlocks[BB];
    assert(BB2);
    if (BB2->size() != 0) {
        //llvm::errs() << "skipping block" << BB2->getName() << "\n";
        continue;
    }

    IRBuilder<> Builder2(BB2);
    differentials[BB] = BB2;

    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> lookingfor;
    std::function<bool(Value*)> isconstant = [&](Value* val) -> bool {
        if(isa<AllocaInst>(val)) return false;

        if(isa<Constant>(val) || (constants.find(val) != constants.end())) {
            //llvm::errs() << "CACHED: derived that " << *val << "is constant\n";
            return true;
        }
        if((nonconstant.find(val) != nonconstant.end())) {
            //llvm::errs() << "CACHED: derived that " << *val << "is NOT constant\n";
            return false;
        }
        if((lookingfor.find(val) != lookingfor.end())) return true;
        if (auto inst = dyn_cast<PHINode>(val)) {
            lookingfor.insert(inst);
            for(auto& a: inst->incoming_values()) {
                if (!isconstant(a)) {
                    nonconstant.insert(val);
                    llvm::errs() << "derived that " << *val << "is NOT constant (pn)\n";
                    return false;
                }
            }
            lookingfor.erase(inst);
            constants.insert(val);
            llvm::errs() << "derived that " << *val << "is constant (pn)\n";
            return true;
        }
        if (auto inst = dyn_cast<Instruction>(val)) {
            lookingfor.insert(inst);
            for(auto& a: inst->operands()) {
                if (!isconstant(a)) {
                    nonconstant.insert(val);
                    llvm::errs() << "derived that " << *val << "is NOT constant (i)\n";
                    return false;
                }
            }
            lookingfor.erase(inst);
            constants.insert(val);
            llvm::errs() << "derived that " << *val << "is constant\n";
            return true;
        }
        llvm::errs() << "derived that " << *val << "is NOT constant\n";
        nonconstant.insert(val);
        return false;
    };

    auto lookup = [&](Value* val) -> Value* {
        if (auto inst = dyn_cast<Instruction>(val)) {
        if (scopeMap.find(val) == scopeMap.end()) {
            scopeMap[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"_cache");
            Instruction* putafter = isa<PHINode>(inst) ? (inst->getParent()->getFirstNonPHI() ): inst;
            IRBuilder <> v(putafter);
            auto st = v.CreateStore(val, scopeMap[val]);
            if (!isa<PHINode>(inst))
                st->moveAfter(putafter);
            addedStores.insert(st);
        }
        return Builder2.CreateLoad(scopeMap[val]);
        }
        return val;
    };

    ValueToValueMapTy antiallocas;
    std::function<std::pair<Value*,Value*>(Value*)> lookupOrAllocate = [&](Value* val) -> std::pair<Value*,Value*> {
        if (auto inst = dyn_cast<AllocaInst>(val)) {
            if (antiallocas.find(val) == antiallocas.end()) {
                antiallocas[val] = entryBuilder.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), inst->getArraySize(), inst->getName()+"'");
                cast<AllocaInst>(antiallocas[val])->setAlignment(inst->getAlignment()); 
                Value *args[] = {entryBuilder.CreateBitCast(antiallocas[val],Type::getInt8PtrTy(Context)), ConstantInt::get(Type::getInt8Ty(val->getContext()), 0), entryBuilder.CreateMul(
                entryBuilder.CreateZExtOrTrunc(inst->getArraySize(),Type::getInt64Ty(Context)), 
                    ConstantInt::get(Type::getInt64Ty(Context), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8 ) ), ConstantInt::getFalse(Context) };
                Type *tys[] = {args[0]->getType(), args[2]->getType()};
                Intrinsic::getDeclaration(M, Intrinsic::memset, tys)->dump();
                Intrinsic::getDeclaration(M, Intrinsic::memset, tys)->getFunctionType()->dump();
                for (auto t: tys) t->dump();
                entryBuilder.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
            }
            return std::pair<Value*,Value*>(val, antiallocas[val]);
        } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
          auto ptr = lookupOrAllocate(inst->getPointerOperand());
          ptr.first = Builder2.CreateGEP(ptr.first, SmallVector<Value*,4>(inst->indices()));
          if (ptr.second)
            ptr.second = Builder2.CreateGEP(ptr.second, SmallVector<Value*,4>(inst->indices()));
          return ptr;
        } else if (auto inst = dyn_cast<Instruction>(val)) {
          if (scopeMap.find(val) == scopeMap.end()) {
            scopeMap[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"_loacache");
            Instruction* putafter = isa<PHINode>(inst) ? (inst->getParent()->getFirstNonPHI() ): inst;
            IRBuilder <> v(putafter);
            auto st = v.CreateStore(val, scopeMap[val]);
            if (!isa<PHINode>(inst))
                st->moveAfter(putafter);
            addedStores.insert(st);
          }
          return std::pair<Value*,Value*>(Builder2.CreateLoad(scopeMap[val]),nullptr);
        }
        return std::pair<Value*,Value*>(val,nullptr);
    };

    auto diffe = [&](Value* val) -> Value* {
      if (differentials.find(val) == differentials.end()) {
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'de");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
      }
      return Builder2.CreateLoad(differentials[val]);
    };

    auto addToDiffe = [&](Value* val, Value* dif) {
      auto old = diffe(val);
      Builder2.CreateStore(Builder2.CreateFAdd(old, dif), differentials[val]);
    };

    auto setDiffe = [&](Value* val, Value* toset) {
      if (differentials.find(val) == differentials.end()) {
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'de");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
      }
      Builder2.CreateStore(toset, differentials[val]);
    };

    auto addToDiffeIndexed = [&](Value* val, Value* dif, ArrayRef<Value*> idxs) {
      if (differentials.find(val) == differentials.end()) {
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'di");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
      }
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for(auto i : idxs)
        sv.push_back(i);
      auto ptr = Builder2.CreateGEP(differentials[val], sv);
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(ptr), dif), ptr);
    };

    std::function<Value*(Value*)> invertPointer = [&](Value* val) -> Value* {
      if (auto arg = dyn_cast<Argument>(val)) {
        return ptrInputs[arg];
      } else if (auto arg = dyn_cast<CastInst>(val)) {
        return Builder2.CreateCast(arg->getOpcode(), invertPointer(arg->getOperand(0)), arg->getDestTy(), arg->getName()+"'ip");
      } else if (auto arg = dyn_cast<GetElementPtrInst>(val)) {
        SmallVector<Value*,4> invertargs;
        for(auto &a: arg->indices()) {
            invertargs.push_back(lookup(a));
        }
        return Builder2.CreateGEP(invertPointer(arg->getPointerOperand()), invertargs, arg->getName()+"'ip");
      } else if (auto arg = dyn_cast<AllocaInst>(val)) {
        return antiallocas[arg];
      } else {
        newFunc->dump();
        val->dump();
        assert(0 && "cannot find deal with ptr that isnt arg");
      }
    };

    auto addToPtrDiffe = [&](Value* val, Value* dif) {
      auto ptr = invertPointer(val);
      Builder2.CreateStore(Builder2.CreateFAdd(Builder2.CreateLoad(ptr), dif), ptr);
    };

    auto setPtrDiffe = [&](Value* val, Value* toset) {
      auto ptr = invertPointer(val);
      Builder2.CreateStore(toset, ptr);
    };

  auto term = BB->getTerminator();

  if(auto op = dyn_cast<ReturnInst>(term)) {
      auto retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.CreateBr(BB2);
      op->eraseFromParent();
      differentials[retval] = entryBuilder.CreateAlloca(retval->getType(), nullptr, retval->getName()+"'ret");
      entryBuilder.CreateStore(ConstantFP::get(retval->getType(), 1.0), differentials[retval]);
  } else if (isa<BranchInst>(term)) {
        
  } else {
    llvm::errs() << "unknown return instance\n";
    assert(0 && "unknown return inst");
  }

  for(auto PB :successors(BB)) {
    
    for (auto I = PB->begin(), E = PB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {
            if (!isconstant(PN->getIncomingValueForBlock(BB)) && !isconstant(PN)) {
              setDiffe(PN->getIncomingValueForBlock(BB), diffe(PN) );
            }
        } else break;
    }
  }
  

  for (auto I = BB->rbegin(), E = BB->rend(); I != E;) {
    Instruction* inst = &*I;
    I++;
    if (addedStores.find(inst) != addedStores.end()) continue;
    if (isconstant(inst)) continue;

    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getOpcode()) {
        case Instruction::FMul:
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(op->getOpcode(), diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateBinOp(op->getOpcode(), diffe(inst), lookup(op->getOperand(0)), "diffe"+op->getOperand(1)->getName());
          break;
        case Instruction::FAdd:
          if (!isconstant(op->getOperand(0)))
            dif0 = diffe(inst);
          if (!isconstant(op->getOperand(1)))
            dif1 = diffe(inst);
          break;
        case Instruction::FSub:
          if (!isconstant(op->getOperand(0)))
            dif0 = diffe(inst);
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(diffe(inst));
          break;
        case Instruction::FDiv:
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(
              Builder2.CreateBinOp(Instruction::FDiv, 
                Builder2.CreateFMul(diffe(inst), op),
                lookup(op->getOperand(1)))
            );

        default:
          llvm::errs() << "cannot handle unknown binary operator\n";
          assert(0 && "unknown binary operator");
          exit(1);
      }

      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (!dif0 && !dif1) {
        constants.insert(op);
      }
    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::memset: {
            if (!isconstant(op->getOperand(1))) {
                llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n";
                inst->dump();
                exit(1);
            }
            auto ptx = invertPointer(op->getOperand(0));
            SmallVector<Value*, 4> args;
            args.push_back(ptx);
            args.push_back(op->getOperand(1));
            args.push_back(op->getOperand(2));
            args.push_back(op->getOperand(3));

            Type *tys[] = {args[0]->getType(), args[2]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args);
            cal->setAttributes(op->getAttributes());
            break;
        }
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        case Intrinsic::dbg_label:
        case Intrinsic::dbg_addr:
            break;
        case Intrinsic::lifetime_start:
        case Intrinsic::lifetime_end:
          op->eraseFromParent();
          break;
        case Intrinsic::sqrt: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.0), lookup(op))
            );
          break;
        }
        case Intrinsic::fabs: {
          auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), ConstantFP::get(op->getOperand(0)->getType(), 0));
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), -1), ConstantFP::get(op->getOperand(0)->getType(), 1));
          break;
        }
        case Intrinsic::log: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::log10: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateBinOp(Instruction::FMul, ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::exp: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FMul, diffe(inst), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FMul,
              Builder2.CreateBinOp(Instruction::FMul, diffe(inst), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
            );
          break;
        }
        case Intrinsic::pow: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FMul,
              Builder2.CreateBinOp(Instruction::FMul, diffe(inst),
                Builder2.CreateBinOp(Instruction::FDiv, lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
            );

          Value *args[] = {op->getOperand(1)};
          Type *tys[] = {op->getOperand(1)->getType()};

          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateBinOp(Instruction::FMul,
              Builder2.CreateBinOp(Instruction::FMul, diffe(inst), lookup(op)),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );

          break;
        }
        case Intrinsic::sin: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FMul, Builder2.CreateLoad(differentials[inst]),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          break;
        }
        case Intrinsic::cos: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FMul, diffe(inst),
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

      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (!dif0 && !dif1) {
        constants.insert(op);
      }
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {
        if(auto called = op->getCalledFunction()) {

              SmallSet<unsigned,4> constant_args;

              SmallVector<Value*, 8> args;
              SmallVector<bool, 8> argsInverted;

              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (isconstant(op->getArgOperand(i))) {
                    llvm::errs() << "operand " << i << "for called->getName()" << "is const\n";
                    constant_args.insert(i);
                    args.push_back(lookup(op->getArgOperand(i)));
                    argsInverted.push_back(false);
                    break;
                }

                std::pair<Value*, Value*> loa = lookupOrAllocate(op->getArgOperand(i));
                args.push_back(loa.first);
                argsInverted.push_back(loa.second == nullptr);
                if(loa.second) args.push_back(loa.second);
              }
              if (constant_args.size() == args.size()) break;

              auto newcalled = CreatePrimalAndGradient(dyn_cast<Function>(called), constant_args);

              auto diffes = Builder2.CreateCall(newcalled, args);
              diffes->setDebugLoc(inst->getDebugLoc());
              unsigned structidx = 1;
              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (argsInverted[i]) {
                  unsigned idxs[] = {structidx};
                  auto diffeadd = Builder2.CreateFMul( diffe(inst), Builder2.CreateExtractValue(diffes, idxs));
                  structidx++;
                  addToDiffe(args[i], diffeadd);
                }
              }

        } else {
            llvm::errs() << "cannot handle non const function\n";
            assert(0 && "unknown non const function");
            exit(1);
        }

    } else if(auto op = dyn_cast_or_null<SelectInst>(inst)) {

      Value* dif1 = nullptr;
      Value* dif2 = nullptr;

      if (!isconstant(op->getOperand(1)))
        dif1 = Builder2.CreateSelect(lookup(op->getOperand(0)), diffe(inst), Constant::getNullValue(op->getOperand(1)->getType()), "diffe"+op->getOperand(1)->getName());
      if (!isconstant(op->getOperand(2)))
        dif2 = Builder2.CreateSelect(lookup(op->getOperand(0)), Constant::getNullValue(op->getOperand(2)->getType()), diffe(inst), "diffe"+op->getOperand(2)->getName());
      
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (dif2) addToDiffe(op->getOperand(2), dif2);
      if (!dif1 && !dif2) {
        constants.insert(op);
      }
    } else if(auto op = dyn_cast<LoadInst>(inst)) {
      if (!isconstant(op->getOperand(0))) {
        auto dif1 = diffe(inst);
        addToPtrDiffe(op->getOperand(0), dif1);
      } else {
        constants.insert(op);
      }
    } else if(auto op = dyn_cast<StoreInst>(inst)) {
      //TODO const
      if (!isconstant(op->getValueOperand())) {
        auto dif1 = Builder2.CreateLoad(invertPointer(op->getPointerOperand()));
        addToDiffe(op->getValueOperand(), dif1);
      }
      setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()));

    } else if(auto op = dyn_cast<ExtractValueInst>(inst)) {
      //todo const
      SmallVector<Value*,4> sv;
      for(auto i : op->getIndices())
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), i));
      addToDiffeIndexed(op->getOperand(0), diffe(inst), sv);
    } else if (auto op = dyn_cast<ShuffleVectorInst>(inst)) {
      //TODO const
      auto loaded = diffe(inst);
      auto l1 = cast<VectorType>(op->getOperand(0)->getType())->getNumElements();
      uint64_t instidx = 0;
      for( auto idx : op->getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx-l1);
        SmallVector<Value*,4> sv;
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), opidx));
        addToDiffeIndexed(op->getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv);
        instidx++;
      }
    } else if(auto op = dyn_cast<ExtractElementInst>(inst)) {
      //TODO const
      SmallVector<Value*,4> sv;
      sv.push_back(op->getIndexOperand());
      addToDiffeIndexed(op->getVectorOperand(), diffe(inst), sv);
    } else if(auto op = dyn_cast<InsertElementInst>(inst)) {
      //TODO const
      auto dif1 = diffe(inst);
      addToDiffe(op->getOperand(0), Builder2.CreateInsertElement(dif1, Constant::getNullValue(op->getOperand(1)->getType()), lookup(op->getOperand(2)) ));

      addToDiffe(op->getOperand(1), Builder2.CreateExtractElement(dif1, lookup(op->getOperand(2))));
    } else if(isa<CastInst>(inst)) {
      if (isconstant(inst->getOperand(0))) {
        constants.insert(inst->getOperand(0));
      }
    } else if(isa<CmpInst>(inst) || isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<AllocaInst>(inst) || isa<CastInst>(inst) || isa<GetElementPtrInst>(inst)) {
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
    newFunc->dump();
    retargs[0]->dump();
    assert(retargs[0]);
    unsigned num_args;
    {
    int i=1;
    for (auto& I: newFunc->args()) {
        if (I.getType()->isPointerTy() || isconstant(&I)) {
            continue;
        }
      retargs[i] = diffe((Value*)&I);
      retargs[i]->dump();
      i++;
    }
    num_args = i;
    }

    Value* toret = UndefValue::get(newFunc->getReturnType());
    for(unsigned i=0; i<num_args; i++) {
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
    Builder2.CreateCondBr(lookup(phi), reverseBlocks[preds[0]], reverseBlocks[preds[1]]);
  } else {
    newFunc->dump();
    BB->dump();
    printf("predcount = %d\n", predcount);
    assert(0 && "Need to determine where came from");
  }


  }

  newFunc->dump();
  while(inversionAllocs->size() > 0) {
    inversionAllocs->back().moveBefore(&newFunc->getEntryBlock().front());
  }

  inversionAllocs->eraseFromParent();

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

  SmallSet<unsigned,4> constants;

  auto newFunc = CreatePrimalAndGradient(dyn_cast<Function>(fn), constants);

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
