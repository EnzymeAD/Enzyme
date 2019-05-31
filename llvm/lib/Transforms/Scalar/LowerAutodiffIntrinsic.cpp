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
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PhiValues.h"


#include "llvm/IR/InlineAsm.h"
//#include "llvm/Transforms/Utils/EaryCSE.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Operator.h"
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
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"

#include "llvm/ADT/SmallSet.h"
using namespace llvm;

#define DEBUG_TYPE "lower-autodiff-intrinsic"

#include <utility>
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"



enum class DIFFE_TYPE {
  OUT_DIFF, // add differential to output struct
  DUP_ARG,  // duplicate the argument and store differential inside
  CONSTANT  // no differential
};

static inline DIFFE_TYPE whatType(llvm::Type* arg) {
  if (arg->isPointerTy()) {
    switch(whatType(cast<llvm::PointerType>(arg)->getElementType())) {
      case DIFFE_TYPE::OUT_DIFF:
        return DIFFE_TYPE::DUP_ARG;
      case DIFFE_TYPE::CONSTANT:
        return DIFFE_TYPE::CONSTANT;
      case DIFFE_TYPE::DUP_ARG:
        return DIFFE_TYPE::DUP_ARG;
    }
    arg->dump();
    assert(0 && "Cannot handle type0");
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isArrayTy()) {
    return whatType(cast<llvm::ArrayType>(arg)->getElementType());
  } else if (arg->isStructTy()) {
    auto st = cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0) return DIFFE_TYPE::CONSTANT;

    auto ty = whatType(st->getElementType(0));
    for(unsigned i=1; i<st->getNumElements(); i++) {
      switch(whatType(st->getElementType(i))) {
        case DIFFE_TYPE::OUT_DIFF:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                case DIFFE_TYPE::CONSTANT:
                  ty = DIFFE_TYPE::OUT_DIFF;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
              }
        case DIFFE_TYPE::CONSTANT:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                  ty = DIFFE_TYPE::OUT_DIFF;
                case DIFFE_TYPE::CONSTANT:
                  ty = DIFFE_TYPE::CONSTANT;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
              }
        case DIFFE_TYPE::DUP_ARG:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                  ty = DIFFE_TYPE::DUP_ARG;
                case DIFFE_TYPE::CONSTANT:
                  ty = DIFFE_TYPE::DUP_ARG;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
              }
      }
    }

    return ty;
  } else if (arg->isIntOrIntVectorTy() || arg->isFunctionTy ()) {
    return DIFFE_TYPE::CONSTANT;
  } else if  (arg->isFPOrFPVectorTy()) {
    return DIFFE_TYPE::OUT_DIFF;
  } else {
    arg->dump();
    assert(0 && "Cannot handle type");
    return DIFFE_TYPE::CONSTANT;
  }
}

bool isReturned(Instruction *inst) {
	for (const auto &a:inst->users()) {
		if(isa<ReturnInst>(a))
			return true;
	}
	return false;
}

bool isConstantValue(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant) {
	if(isa<Constant>(val) || isa<BasicBlock>(val) || isa<InlineAsm>(val) || (constants.find(val) != constants.end())) {
    	return true;
    }
	if (val->getType()->isVoidTy()) return true;


}

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant) {
	if(isa<Constant>(val) || isa<BasicBlock>(val) || isa<InlineAsm>(val) || (constants.find(val) != constants.end())) {
    	return true;
    }

    if((nonconstant.find(val) != nonconstant.end())) {
        return false;
    }

	if (auto op = dyn_cast<CallInst>(val)) {
		if(auto called = op->getCalledFunction()) {
			if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "__assert_fail") {
				nonconstant.insert(val);
				return false;
			}
		}
	}

    if (auto op = dyn_cast<IntrinsicInst>(val)) {
		switch(op->getIntrinsicID()) {
			case Intrinsic::lifetime_start:
			case Intrinsic::lifetime_end:
			case Intrinsic::dbg_addr:
			case Intrinsic::dbg_declare:
			case Intrinsic::dbg_value:
			case Intrinsic::invariant_start:
			case Intrinsic::invariant_end:
			case Intrinsic::var_annotation:
			case Intrinsic::ptr_annotation:
			case Intrinsic::annotation:
			case Intrinsic::codeview_annotation:
			case Intrinsic::expect:
			case Intrinsic::type_test:
			case Intrinsic::donothing:
			//case Intrinsic::is_constant:
				constants.insert(val);
				return true;
			default:
				break;
		}
	}

    llvm::errs() << "checking if is constant " << *val << "\n";

	//Proceed assuming this is constant, can we prove this should be constant otherwise
	SmallPtrSet<Value*, 20> constants2;
	constants2.insert(constants.begin(), constants.end());
	SmallPtrSet<Value*, 20> nonconstant2;
	nonconstant2.insert(nonconstant.begin(), nonconstant.end());
	constants2.insert(val);

	if (val->getType()->isPointerTy()) {
	  if (auto inst = dyn_cast<Instruction>(val)) {

		// If object (such as eigen matrix) is composed of constant (size) and differential (doubles)
		// make sure the integer portions of the object are treated as constants
		if (cast<PointerType>(val->getType())->getElementType()->isIntegerTy()) {

		  bool failed = false;
		  SmallVector<Value*, 4> todo;
		  SmallPtrSet<Value*, 4> visited;
		  todo.push_back(inst);
		  while(!failed && todo.size() > 0) {
			Value* cur = todo.back();
			todo.pop_back();
			if (!visited.insert(cur).second) continue;

			if (auto gep = dyn_cast<GetElementPtrInst>(cur)) {
				todo.push_back(gep->getPointerOperand());
				continue;
			} else if (auto li = dyn_cast<LoadInst>(cur)) {
				todo.push_back(li->getPointerOperand());
				continue;
			} else if (isa<Argument>(cur)) {
				continue;
			} else if (isa<AllocaInst>(cur)) {
				continue;
			} else if (auto ci = dyn_cast<CastInst>(cur)) {
				if (auto src = dyn_cast<PointerType>(ci->getSrcTy())) 
					if (auto dst = dyn_cast<PointerType>(ci->getDestTy())) 
						if (src->getElementType()->isStructTy() && dst->getElementType()->isStructTy()) {
							todo.push_back(ci->getOperand(0));
							continue;
						}
				llvm::errs() << "unable to int fix (cast):" << *val << " cur is " << *cur << "\n";
				failed = true;
				break;
			} else if (auto pn = dyn_cast<PHINode>(cur)) {
				for(unsigned i=0; i<pn->getNumIncomingValues(); i++)
					todo.push_back(pn->getIncomingValue(i));
			} else {
				llvm::errs() << "unable to int fix:" << *val << " cur is " << *cur << "\n";
				failed = true;
				break;
			}
		  }
		}

		for (const auto &a:inst->users()) {
		  if(auto store = dyn_cast<StoreInst>(a)) {
			if (!isconstantM(store->getValueOperand(), constants2, nonconstant2)) {
				nonconstant.insert(val);
				llvm::errs() << "memory erase 1: " << *val << "\n";
				return false;
			}
		  } else if (isa<LoadInst>(a)) continue;
		  else {
			if (!isconstantM(a, constants2, nonconstant2)) {
				nonconstant.insert(val);
				llvm::errs() << "memory erase 2: " << *val << "\n";
				return false;
			}
		  }

		}
		//llvm::errs() << "memory erase 3: " << *val << "\n";

	  }
	}

	if (auto inst = dyn_cast<PHINode>(val)) {
		//lookingfor.insert(inst);
		for(auto& a: inst->incoming_values()) {
			if (!isconstantM(a, constants2, nonconstant2)) {
				nonconstant.insert(val);
				//lookingfor.erase(val);
				llvm::errs() << "nonconstant phi " << *val << " op " << *a << "\n";
				return false;
			}
		}
		constants.insert(val);
		llvm::errs() << "constant phi:" << *val << "\n";
		return true;
	}

	//assert(!isa<GetElementPtrInst>(val));
	// Handled uses above
	if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
		if (!isconstantM(inst->getPointerOperand(), constants2, nonconstant2)) {
			nonconstant.insert(val);
			llvm::errs() << "nonconstant gep " << *val << " op " << *inst->getPointerOperand() << "\n";
			return false;
		}
		constants.insert(val);
		llvm::errs() << "constant gep:" << *val << "\n";
		return true;
	}

	if (auto inst = dyn_cast<Instruction>(val)) {
		//lookingfor.insert(val);
		for(auto& a: inst->operands()) {
			if (!isconstantM(a, constants2, nonconstant2)) {
				nonconstant.insert(val);
				llvm::errs() << "nonconstant inst " << *val << " op " << *a << "\n";
				return false;
			}
		}

		constants.insert(val);
		llvm::errs() << "constant inst:" << *val << "\n";
		return true;
	}

	nonconstant.insert(val);
	llvm::errs() << "couldnt decide nonconstants:" << *val << "\n";
	return false;
}


 static bool promoteMemoryToRegister(Function &F, DominatorTree &DT,
                                     AssumptionCache &AC) {
   std::vector<AllocaInst *> Allocas;
   BasicBlock &BB = F.getEntryBlock(); // Get the entry node for the function
   bool Changed = false;
 
   while (true) {
     Allocas.clear();
 
     // Find allocas that are safe to promote, by looking at all instructions in
     // the entry node
     for (BasicBlock::iterator I = BB.begin(), E = --BB.end(); I != E; ++I)
       if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) // Is it an alloca?
         if (isAllocaPromotable(AI))
           Allocas.push_back(AI);
 
     if (Allocas.empty())
       break;
 
     PromoteMemToReg(Allocas, DT, &AC);
     Changed = true;
   }
   return Changed;
 }

Function *CloneFunctionWithReturns(Function *F, ValueToValueMapTy& ptrInputs, const SmallSet<unsigned,4>& constant_args, SmallPtrSetImpl<Value*> &constants, bool returnValue) {
 std::vector<Type*> RetTypes;
 if (returnValue)
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
     auto wt = whatType(I.getType());
     if (wt == DIFFE_TYPE::DUP_ARG) {
        ArgTypes.push_back(I.getType());
        /*
        if (I.getType()->isPointerTy() && !(I.hasAttribute(Attribute::ReadOnly) || I.hasAttribute(Attribute::ReadNone) ) ) {
          llvm::errs() << "Cannot take derivative of function " <<F->getName()<< " input argument to function " << I.getName() << " is not marked read-only\n";
          exit(1);
        }
        */
     } else if (wt == DIFFE_TYPE::OUT_DIFF) {
       RetTypes.push_back(I.getType());
     }
     argno++;
  }

 auto RetType = StructType::get(F->getContext(), RetTypes);

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), "diffe"+F->getName(), F->getParent());

 bool hasPtrInput = false;

 unsigned ii = 0, jj = 0;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {

   auto wt = whatType(i->getType());

   bool isconstant = (constant_args.count(ii) > 0) || wt == DIFFE_TYPE::CONSTANT;
   if (isconstant) {
      constants.insert(j);
   }

   if (!isconstant && wt == DIFFE_TYPE::DUP_ARG) {
     VMap[i] = j;
     hasPtrInput = true;
     ptrInputs[j] = (j+1);
     if (F->hasParamAttribute(ii, Attribute::NoCapture)) {
       NewF->addParamAttr(jj, Attribute::NoCapture);
       NewF->addParamAttr(jj+1, Attribute::NoCapture);
     }
     if (F->hasParamAttribute(ii, Attribute::NoAlias)) {
       NewF->addParamAttr(jj, Attribute::NoAlias);
       NewF->addParamAttr(jj+1, Attribute::NoAlias);
     }

     j->setName(i->getName());
     j++;
     j->setName(i->getName()+"'");
     j++;
     jj+=2;

     i++;
     ii++;

   } else {
     VMap[i] = j;
     j->setName(i->getName());

     j++;
     jj++;
     i++;
     ii++;
   }
 }

 // Loop over the arguments, copying the names of the mapped arguments over...
 Function::arg_iterator DestI = NewF->arg_begin();


 for (const Argument & I : F->args())
   if (VMap.count(&I) == 0) {     // Is this argument preserved?
     DestI->setName(I.getName()); // Copy the name over...
     VMap[&I] = &*DestI++;        // Add mapping to VMap
   }
 SmallVector <ReturnInst*,4> Returns;
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
 NewF->setLinkage(Function::LinkageTypes::InternalLinkage);
 assert(NewF->hasLocalLinkage());

 SmallPtrSet<Value*,4> constants2;
 for (auto a :constants){
    constants2.insert(a);
 }
 SmallPtrSet<Value*,20> nonconstant2;
 //for (auto a :nonconstant){
 //   nonconstant2.insert(a);
 //}
 
  {
   remover:
   for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I)
     if (auto call = dyn_cast<CallInst>(&*I)) {
        if (isconstantM(call, constants2, nonconstant2)) continue;
        if (call->getCalledFunction() == nullptr) continue;
        if (call->getCalledFunction()->empty()) continue;
        if (call->getCalledFunction()->hasFnAttribute(Attribute::NoInline)) {
            llvm::errs() << "can't inline noinline " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        if (call->getCalledFunction()->hasFnAttribute(Attribute::ReturnsTwice)) continue;
        if (call->getCalledFunction() == F || call->getCalledFunction() == NewF) {
            llvm::errs() << "can't inline recursive " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        llvm::errs() << "inlining " << call->getCalledFunction()->getName() << "\n";
        InlineFunctionInfo IFI;
        InlineFunction(call, IFI);
        goto remover;
     }
 }

 {
 DominatorTree DT(*NewF);
 AssumptionCache AC(*NewF);
 promoteMemoryToRegister(*NewF, DT, AC);

 GVN gvn;

 FunctionAnalysisManager AM;
 AM.registerPass([] { return AAManager(); });
 AM.registerPass([] { return AssumptionAnalysis(); });
 AM.registerPass([] { return TargetLibraryAnalysis(); });
 AM.registerPass([] { return DominatorTreeAnalysis(); });
 AM.registerPass([] { return MemoryDependenceAnalysis(); });
 AM.registerPass([] { return LoopAnalysis(); });
 AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
 AM.registerPass([] { return PhiValuesAnalysis(); });
 //AM.registerPass([] { return DominatorTreeWrapperPass() });
 gvn.run(*NewF, AM);

 SROA().run(*NewF, AM);
 //gvn.run(*NewF, AM);

 //gvn.runImpl(*NewF, AC, DT, TLI, AA);
 /*
     auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
     auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
     auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
     auto &AC = getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
     auto *MSSA =
         UseMemorySSA ? &getAnalysis<MemorySSAWrapperPass>().getMSSA() : nullptr;
 
   EarlyCSE CSE(F.getParent()->getDataLayout(), TLI, TTI, DT, AC, MSSA);
   CSE.run();
*/
 }

 return NewF;
}

#include "llvm/IR/Constant.h"
#include <deque>
#include "llvm/IR/CFG.h"

PHINode* canonicalizeIVs(Type *Ty, Loop *L, ScalarEvolution &SE, DominatorTree &DT) {

  BasicBlock* Header = L->getHeader();
  Module* M = Header->getParent()->getParent();
  const DataLayout &DL = M->getDataLayout();

  SCEVExpander Exp(SE, DL, "ls");

  PHINode *CanonicalIV = Exp.getOrInsertCanonicalInductionVariable(L, Ty);
  if (!CanonicalIV) {
    L->dump();
    Ty->dump();
  }
  assert (CanonicalIV && "canonicalizing IV");
  //DEBUG(dbgs() << "Canonical induction variable " << *CanonicalIV << "\n");

  SmallVector<WeakTrackingVH, 16> DeadInsts;
  Exp.replaceCongruentIVs(L, &DT, DeadInsts);

  
  for (WeakTrackingVH V : DeadInsts) {
    //DEBUG(dbgs() << "erasing dead inst " << *V << "\n");
    Instruction *I = cast<Instruction>(V);
    I->eraseFromParent();
  }
  

  return CanonicalIV;
}

/// \brief Replace the latch of the loop to check that IV is always less than or
/// equal to the limit.
///
/// This method assumes that the loop has a single loop latch.
Value* canonicalizeLoopLatch(PHINode *IV, Value *Limit, Loop* L, ScalarEvolution &SE, BasicBlock* ExitBlock) {
  Value *NewCondition;
  BasicBlock *Header = L->getHeader();
  BasicBlock *Latch = L->getLoopLatch();
  assert(Latch && "No single loop latch found for loop.");

  IRBuilder<> Builder(&*Latch->getFirstInsertionPt());
  Builder.setFastMathFlags(FastMathFlags::getFast());

  // This process assumes that IV's increment is in Latch.

  // Create comparison between IV and Limit at top of Latch.
  NewCondition = Builder.CreateICmpULT(IV, Limit);

  // Replace the conditional branch at the end of Latch.
  BranchInst *LatchBr = dyn_cast_or_null<BranchInst>(Latch->getTerminator());
  assert(LatchBr && LatchBr->isConditional() &&
         "Latch does not terminate with a conditional branch.");
  Builder.SetInsertPoint(Latch->getTerminator());
  Builder.CreateCondBr(NewCondition, Header, ExitBlock);

  // Erase the old conditional branch.
  Value *OldCond = LatchBr->getCondition();
  LatchBr->eraseFromParent();
  
  if (!OldCond->hasNUsesOrMore(1))
    if (Instruction *OldCondInst = dyn_cast<Instruction>(OldCond))
      OldCondInst->eraseFromParent();
  

  return NewCondition;
}

typedef struct {
  PHINode* var;
  PHINode* antivar;
  BasicBlock* latch;
  BasicBlock* header;
  BasicBlock* preheader;
  bool dynamic;
  //limit is last value, iters is number of iters (thus iters = limit + 1)
  Value* limit;
  BasicBlock* exit;
  Loop* parent;
} LoopContext;

Function* CreatePrimalAndGradient(Function* todiff, const SmallSet<unsigned,4>& constant_args, TargetLibraryInfo &TLI, bool returnValue) {
  auto M = todiff->getParent();
  auto& Context = M->getContext();

  ValueToValueMapTy ptrInputs;
  SmallPtrSet<Value*,4> constants;
  auto newFunc = CloneFunctionWithReturns(todiff, ptrInputs, constant_args, constants, returnValue);

  DominatorTree DT(*newFunc);
  LoopInfo LI(DT);
  AssumptionCache AC(*newFunc);
  ScalarEvolution SE(*newFunc, TLI, AC, DT, LI);

  ValueToValueMapTy differentials;
  ValueToValueMapTy antiallocas;
  SmallVector<Value*, 4> mallocCalls;
  SmallVector<Value*, 4> dynamicMallocCalls;

    SmallPtrSet<Value*,20> nonconstant;
    SmallPtrSet<Value*,2> lookingfor;
    SmallPtrSet<Value*,2> memorylookingfor;

  auto isconstant = [&](Value* val) -> bool {
    return isconstantM(val, constants, nonconstant);
  };

  llvm::AllocaInst* retAlloca = nullptr;
  if (returnValue && !todiff->getReturnType()->isVoidTy()) {
	retAlloca = IRBuilder<>(&newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
  }

  SmallVector<BasicBlock*, 12> fnthings;
  for (BasicBlock &BB : *newFunc) {
     fnthings.push_back(&BB);
  }

  auto returnMerged = BasicBlock::Create(Context, "returnMerged", newFunc);

  auto inversionAllocs = BasicBlock::Create(Context, "allocsForInversion", newFunc);

  ValueMap<BasicBlock*,BasicBlock*> reverseBlocks;
  std::deque<BasicBlock*> blockstodo;
  for (BasicBlock *BB : fnthings) {
    auto BB2 = BasicBlock::Create(Context, "invert" + BB->getName(), newFunc);
    reverseBlocks[BB] = BB2;
    blockstodo.push_back(BB);//->getParent());
  }

  IRBuilder<> entryBuilder(inversionAllocs);
  entryBuilder.setFastMathFlags(FastMathFlags::getFast());

  IRBuilder<> returnBuilder(returnMerged);
  returnBuilder.setFastMathFlags(FastMathFlags::getFast());

  ValueToValueMapTy scopeMap;
  SmallPtrSet<Value*, 10> addedStores;
  std::map<Loop*, LoopContext> loopContexts;

  // returns if in loop
  auto getContext = [&](BasicBlock* BB, LoopContext & loopContext) -> bool {

    if (auto L = LI.getLoopFor(BB)) {
        if (loopContexts.find(L) != loopContexts.end()) {
            loopContext = loopContexts.find(L)->second;
            return true;
        }

        SmallVector<BasicBlock *, 8> PotentialExitBlocks;
        SmallVector<BasicBlock *, 8> ExitBlocks;
        L->getExitBlocks(PotentialExitBlocks);
        for(auto a:PotentialExitBlocks) {

            SmallVector<BasicBlock*, 4> tocheck;
            SmallPtrSet<BasicBlock*, 4> checked;
            tocheck.push_back(a);

            bool isExit = false;

            while(tocheck.size()) {
                auto foo = tocheck.back();
                tocheck.pop_back();
                if (checked.count(foo)) {
                    //llvm::errs() << "looping in: " << *foo << "\n";
                    isExit = true;
                    goto exitblockcheck;
                }
                checked.insert(foo);
                if(auto bi = dyn_cast<BranchInst>(foo->getTerminator())) {
                    for(auto nb : bi->successors()) {
                        if (L->contains(nb)) continue;
                        tocheck.push_back(nb);
                    }
                } else if (isa<UnreachableInst>(foo->getTerminator())) {
                    continue;
                } else {
                    //llvm::errs() << "unknown ending in: " << *foo << "\n";
                    isExit = true;
                    goto exitblockcheck;
                }
            }

            
            exitblockcheck:
            if (isExit) {
				ExitBlocks.push_back(a);
            }
        }

        if (ExitBlocks.size() != 1) {
			BB->getParent()->dump();
			L->dump();
			for(auto b:ExitBlocks)
				b->dump();
			llvm::errs() << "offending: \n";
			llvm::errs() << "No unique exit block (1)\n";
			exit(1);
        }

        BasicBlock* ExitBlock = ExitBlocks[0];

        BasicBlock *Header = L->getHeader();
        BasicBlock *Preheader = L->getLoopPreheader();
        assert(Preheader && "requires preheader");
        BasicBlock *Latch = L->getLoopLatch();

        const SCEV *Limit = SE.getExitCount(L, Latch);
        
		SCEVExpander Exp(SE, M->getDataLayout(), "ad");

		PHINode *CanonicalIV = nullptr;
		Value *LimitVar = nullptr;
		if (SE.getCouldNotCompute() != Limit) {

        	CanonicalIV = canonicalizeIVs(Limit->getType(), L, SE, DT);
        	if (!CanonicalIV) {
            	llvm::errs() << "Could not get canonical IV.\n";
            	exit(1);
        	}
        	
			const SCEVAddRecExpr *CanonicalSCEV = cast<const SCEVAddRecExpr>(SE.getSCEV(CanonicalIV));

        	assert(SE.isLoopBackedgeGuardedByCond(L, ICmpInst::ICMP_ULT,
                                              CanonicalSCEV, Limit) &&
               "Loop backedge is not guarded by canonical comparison with limit.");
        
			LimitVar = Exp.expandCodeFor(Limit, CanonicalIV->getType(),
                                            Preheader->getTerminator());

        	// Canonicalize the loop latch.
			canonicalizeLoopLatch(CanonicalIV, LimitVar, L, SE, ExitBlock);

			loopContext.dynamic = false;
		} else {
          llvm::errs() << "Se has any info: " << SE.getBackedgeTakenInfo(L).hasAnyInfo() << "\n";
          llvm::errs() << "SE could not compute loop limit.\n";

		  IRBuilder <>B(&Header->front());
		  //TODO replace 2 in below phi with |predecessors(Header)|
		  CanonicalIV = B.CreatePHI(Type::getInt64Ty(Context), 2);
		  B.SetInsertPoint(Header->getTerminator());
		  auto inc = B.CreateAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1));
		  CanonicalIV->addIncoming(inc, Latch);
		  for (BasicBlock *Pred : predecessors(Header)) {
			  if (Pred != Latch) {
				  CanonicalIV->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
			  }
		  }

		  B.SetInsertPoint(&ExitBlock->front());
		  LimitVar = B.CreatePHI(CanonicalIV->getType(), 1);

		  for (BasicBlock *Pred : predecessors(ExitBlock)) {
    		if (LI.getLoopFor(Pred) == L)
		    	cast<PHINode>(LimitVar)->addIncoming(CanonicalIV, Pred);
			else
				cast<PHINode>(LimitVar)->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
		  }
		  loopContext.dynamic = true;
		}
	
		// Remove Canonicalizable IV's
		{
		  SmallVector<PHINode*, 8> IVsToRemove;
		  for (BasicBlock::iterator II = Header->begin(); isa<PHINode>(II); ++II) {
			PHINode *PN = cast<PHINode>(II);
			if (PN == CanonicalIV) continue;
			if (!SE.isSCEVable(PN->getType())) continue;
			const SCEV *S = SE.getSCEV(PN);
			if (SE.getCouldNotCompute() == S) continue;
			Value *NewIV = Exp.expandCodeFor(S, S->getType(), CanonicalIV);
			if (NewIV == PN) {
				llvm::errs() << "TODO: odd case need to ensure replacement\n";
				continue;
			}
			PN->replaceAllUsesWith(NewIV);
			IVsToRemove.push_back(PN);
		  }
		  for (PHINode *PN : IVsToRemove) {
			//llvm::errs() << "ERASING: " << *PN << "\n";
			PN->eraseFromParent();
		  }
		}

        //if (SE.getCouldNotCompute() == Limit) {
        //Limit = SE.getMaxBackedgeTakenCount(L);
        //}
		assert(CanonicalIV);
		assert(LimitVar);
        loopContext.var = CanonicalIV;
        loopContext.limit = LimitVar;
        loopContext.antivar = PHINode::Create(CanonicalIV->getType(), CanonicalIV->getNumIncomingValues(), CanonicalIV->getName()+"'phi");
        loopContext.exit = ExitBlock;
        loopContext.latch = Latch;
        loopContext.preheader = Preheader;
		loopContext.header = Header;
        loopContext.parent = L->getParentLoop();

        loopContexts[L] = loopContext;
        return true;
    }
    return false;
  };

  // Force loop canonicalization everywhere
  for(auto BB:blockstodo) {
    LoopContext loopContext;
    getContext(BB, loopContext);
  }

  while(blockstodo.size() > 0) {
    auto BB = blockstodo.back();
    blockstodo.pop_back();

    LoopContext loopContext;
    bool inLoop = getContext(BB, loopContext);

    auto BB2 = reverseBlocks[BB];
    assert(BB2);

    std::function<Value*(Value*,IRBuilder<>&)> lookupM;
    std::function<Value*(Value*,IRBuilder<>&, const ValueToValueMapTy&)> unwrapM = [&](Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy& available) -> Value* {
          if (available.count(val)) {
            return available.lookup(val);
          } if (isa<Argument>(val) || isa<Constant>(val)) {
            return val;
          } if (isa<AllocaInst>(val)) {
            return val;
          } else if (auto arg = dyn_cast<CastInst>(val)) {
            return BuilderM.CreateCast(arg->getOpcode(), unwrapM(arg->getOperand(0), BuilderM, available), arg->getDestTy(), arg->getName()+"_unwrap");
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            return BuilderM.CreateBinOp(op->getOpcode(), unwrapM(op->getOperand(0), BuilderM, available), unwrapM(op->getOperand(1), BuilderM, available));
          } else if (auto op = dyn_cast<ICmpInst>(val)) {
            return BuilderM.CreateICmp(op->getPredicate(), unwrapM(op->getOperand(0), BuilderM, available), unwrapM(op->getOperand(1), BuilderM, available));
          } else if (auto op = dyn_cast<FCmpInst>(val)) {
            return BuilderM.CreateFCmp(op->getPredicate(), unwrapM(op->getOperand(0), BuilderM, available), unwrapM(op->getOperand(1), BuilderM, available));
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            return BuilderM.CreateSelect(unwrapM(op->getOperand(0), BuilderM, available), unwrapM(op->getOperand(1), BuilderM, available), unwrapM(op->getOperand(2), BuilderM, available));
          } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
              auto ptr = unwrapM(inst->getPointerOperand(), BuilderM, available);
              SmallVector<Value*,4> ind;
              for(auto& a : inst->indices()) {
                ind.push_back(unwrapM(a, BuilderM,available));
              }
              return BuilderM.CreateGEP(ptr, ind);
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = unwrapM(load->getOperand(0), BuilderM, available);
                return BuilderM.CreateLoad(idx);
          } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
            switch(op->getIntrinsicID()) {
                case Intrinsic::sin: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available)};
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args);
                }
                case Intrinsic::cos: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available)};
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args);
                }
                default:;

            }
          }
          
            llvm::errs() << "cannot unwrap following " << *val << "\n";
            return lookupM(val, BuilderM);
            //val->dump();
            //assert(0 && "unable to unwrap");
            //exit(1);
    };

    std::function<bool(Value*,const ValueToValueMapTy&)> shouldRecompute = [&](Value* val, const ValueToValueMapTy& available) -> bool {
          if (available.count(val)) return false;
          if (isa<Argument>(val) || isa<Constant>(val) ) {
            return false;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            return shouldRecompute(op->getOperand(0), available);
          } else if (isa<AllocaInst>(val)) {
            return false;
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            bool a0 = shouldRecompute(op->getOperand(0), available);
            if (a0) {
                //llvm::errs() << "need recompute: " << *op->getOperand(0) << "\n";
            }
            bool a1 = shouldRecompute(op->getOperand(1), available);
            if (a1) {
                //llvm::errs() << "need recompute: " << *op->getOperand(1) << "\n";
            }
            return a0 || a1;
          } else if (auto op = dyn_cast<CmpInst>(val)) {
            return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available);
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            return shouldRecompute(op->getOperand(0), available) || shouldRecompute(op->getOperand(1), available) || shouldRecompute(op->getOperand(2), available);
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = load->getOperand(0);
                while (!isa<Argument>(idx)) {
                    if (auto gep = dyn_cast<GetElementPtrInst>(idx)) {
                        for(auto &a : gep->indices()) {
                            if (shouldRecompute(a, available)) {
                                //llvm::errs() << "not recomputable: " << *a << "\n";
                                return true;
                            }
                        }
                        idx = gep->getPointerOperand();
                    } else if(auto cast = dyn_cast<CastInst>(idx)) {
                        idx = cast->getOperand(0);
                    } else if(auto call = dyn_cast<CallInst>(idx)) {
						(void*)call;
                        //if (call->getCalledFunction()->getName() == "malloc")
                        //    return false;
                        //else
                        {
                            //llvm::errs() << "unknown call " << *call << "\n";
                            return true;
                        }
                    } else {
                      //llvm::errs() << "not a gep " << *idx << "\n";
                      return true;
                    }
                }
                Argument* arg = cast<Argument>(idx);
                if (! ( arg->hasAttribute(Attribute::ReadOnly) || arg->hasAttribute(Attribute::ReadNone)) ) {
                    //llvm::errs() << "argument " << *arg << " not marked read only\n";
                    return true;
                }
                return false;
          } else if (auto phi = dyn_cast<PHINode>(val)) {
            if (phi->getNumIncomingValues () == 1) {
                bool b = shouldRecompute(phi->getIncomingValue(0) , available);
                if (b) {
                    //llvm::errs() << "phi need recompute: " <<*phi->getIncomingValue(0) << "\n";
                }
                return b;
            }

            return true;
          } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
            switch(op->getIntrinsicID()) {
                case Intrinsic::sin:
                case Intrinsic::cos:
                    return false;
                    return shouldRecompute(op->getOperand(0), available);
                default:
                    return true;
            }
        }

          //llvm::errs() << "unknown inst " << *val << " unable to recompute\n";
          return true;
    };

    lookupM = [&](Value* val, IRBuilder<>& BuilderM) -> Value* {
        if (auto inst = dyn_cast<Instruction>(val)) {

            LoopContext lc;
            bool inLoop = getContext(inst->getParent(), lc);

            ValueToValueMapTy available;
            if (inLoop) {
                for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
                  available[idx.var] = idx.antivar;
                  if (idx.parent == nullptr) break;
                }
            }
            if (!shouldRecompute(inst, available)) {
                return unwrapM(inst, BuilderM, available);
            }

            //if (inLoop && inst == loopContext.var) {
            //  return loopContext.antivar;
            //}

            if (!inLoop) {
                if (scopeMap.find(val) == scopeMap.end()) {
                    //TODO add indexing
                    scopeMap[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"_cache");
                    Instruction* putafter = isa<PHINode>(inst) ? (inst->getParent()->getFirstNonPHI() ): inst;
                    if (cast<Instruction>(scopeMap[val])->getParent() == putafter->getParent()) {
                        //ensure putafter = later of putafter and scopeMap[val]
                        for(Instruction& I : *putafter->getParent()) {
                            if (&I == scopeMap[val]) {
                                break;
                            } else if (&I == putafter) {
                                putafter = cast<Instruction>(scopeMap[val]);
                                break;
                            } else {}
                        }
                    }
                    IRBuilder <> v(putafter);
                    v.setFastMathFlags(FastMathFlags::getFast());
                    auto st = v.CreateStore(val, scopeMap[val]);
                    if (!isa<PHINode>(inst))
                        st->moveAfter(putafter);
                    addedStores.insert(st);
                }
                return BuilderM.CreateLoad(scopeMap[val]);
            } else {
                if (scopeMap.find(val) == scopeMap.end()) {

                    ValueToValueMapTy valmap;
                    Value* size = nullptr;
					bool dynamic = false;
                    for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
					  //TODO handle allocations for dynamic loops
					  if (idx.dynamic && idx.parent != nullptr) {
          				newFunc->dump();
         				idx.var->dump();
         				idx.limit->dump();
						assert(0 && "cannot handle non-outermost dynamic loop");
					  }
                      Value* ns = nullptr;
					  if (idx.dynamic) {
						ns = ConstantInt::get(idx.limit->getType(), 1);
						dynamic = true;
					  } else {
						ns = entryBuilder.CreateAdd(unwrapM(idx.limit, entryBuilder, available), ConstantInt::get(idx.limit->getType(), 1));
					  }
                      if (size == nullptr) size = ns;
                      else size = entryBuilder.CreateMul(size, ns);
                      if (idx.parent == nullptr) break;
                    }

                    if (false && !dynamic) {
                      scopeMap[val] = entryBuilder.CreateAlloca(val->getType(), size, val->getName()+"_arraycache");
                    } else {
					  auto allocation = CallInst::CreateMalloc(entryBuilder.GetInsertBlock(), size->getType(), val->getType(), ConstantInt::get(size->getType(), M->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), size, nullptr, val->getName()+"_arraycache");
                      entryBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));

					  if (dynamic) {
                      	scopeMap[val] = entryBuilder.CreateAlloca(allocation->getType(), nullptr, val->getName()+"_dyncache");
					    entryBuilder.CreateStore(allocation, scopeMap[val]);	
                      	dynamicMallocCalls.push_back(scopeMap[val]);
					  } else {
					  	scopeMap[val] = allocation;
                      	mallocCalls.push_back(allocation);
                      }
                    }

                    Instruction* putafter = isa<PHINode>(inst) ? (inst->getParent()->getFirstNonPHI() ): inst->getNextNonDebugInstruction();
                    IRBuilder <> v(putafter);
                    v.setFastMathFlags(FastMathFlags::getFast());

                    SmallVector<Value*,3> indices;
                    SmallVector<Value*,3> limits;
					BasicBlock* dynamicHeader = nullptr;
					PHINode* dynamicPHI = nullptr;
                    for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                      indices.push_back(idx.var);

					  if (idx.dynamic) {
						dynamicHeader = idx.header;
						dynamicPHI = idx.var;
						llvm::errs() << "saw idx.dynamic:" << *dynamicPHI << "\n";
						assert(idx.parent == nullptr);
						break;
					  }

                      if (idx.parent == nullptr) break;
                      auto lim = v.CreateAdd(unwrapM(idx.limit, v, available), ConstantInt::get(idx.limit->getType(), 1));
                      if (limits.size() != 0) {
                        lim = v.CreateMul(lim, limits.back());
                        addedStores.insert(lim);
                      }
                      limits.push_back(lim);
                    }

                    Value* idx = nullptr;
                    for(unsigned i=0; i<indices.size(); i++) {
                      if (i == 0) {
                        idx = indices[i];
                      } else {
                        auto mul = v.CreateMul(indices[i], limits[i-1]);
                        addedStores.insert(mul);
                        idx = v.CreateAdd(idx, mul);
                        addedStores.insert(idx);
                      }
                    }

					Value* allocation = nullptr;
					if (dynamicHeader == nullptr) {
						allocation = scopeMap[val];
					} else {
						IRBuilder<> headB(dynamicHeader->getFirstNonPHIOrDbgOrLifetime ());
						Type *BPTy = Type::getInt8PtrTy(dynamicHeader->getContext());
						auto realloc = M->getOrInsertFunction("realloc", BPTy, BPTy, size->getType());
						allocation = headB.CreateLoad(scopeMap[val]);
						auto foo = headB.CreateAdd(dynamicPHI, ConstantInt::get(dynamicPHI->getType(), 1));
						foo->dump();
						Value* idxs[2] = {
							headB.CreatePointerCast(allocation, BPTy),
							headB.CreateMul(
								ConstantInt::get(size->getType(), M->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), 
								headB.CreateMul(
									size, foo
								) 
							)
						};

						allocation = headB.CreatePointerCast(headB.CreateCall(realloc, idxs, val->getName()+"_realloccache"), allocation->getType());
						headB.CreateStore(allocation, scopeMap[val]);
						cast<Instruction>(foo)->getParent()->getParent()->dump();
					}

                    Value* idxs[] = {idx};
					allocation->dump();
                    auto gep = v.CreateGEP(allocation, idxs);
                    addedStores.insert(gep);
					val->dump();
					gep->dump();
                    addedStores.insert(v.CreateStore(val, gep));
                }
                assert(inLoop);
                assert(lc.antivar == loopContext.antivar);

                SmallVector<Value*,3> indices;
                SmallVector<Value*,3> limits;
				bool dynamic = false;
                for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                  indices.push_back(idx.antivar);
				  if (idx.dynamic) dynamic = true;
                  if (idx.parent == nullptr) break;

                  auto lim = BuilderM.CreateAdd(unwrapM(idx.limit, BuilderM, available), ConstantInt::get(idx.limit->getType(), 1));
                  if (limits.size() != 0) {
                    lim = BuilderM.CreateMul(lim, limits.back());
                  }
                  limits.push_back(lim);
                }

                Value* idx = nullptr;
                for(unsigned i=0; i<indices.size(); i++) {
                  if (i == 0) {
                    idx = indices[i];
                  } else {
                    idx = BuilderM.CreateAdd(idx, BuilderM.CreateMul(indices[i], limits[i-1]));
                  }
                }

                Value* idxs[] = {idx};
				Value* tolookup = nullptr;
				if (dynamic)
					tolookup = BuilderM.CreateLoad(scopeMap[val]);
				else
					tolookup = scopeMap[val];
                return BuilderM.CreateLoad(BuilderM.CreateGEP(tolookup, idxs));
            }
        }

        return val;
    };

    IRBuilder<> Builder2(BB2);
    Builder2.setFastMathFlags(FastMathFlags::getFast());

    std::map<Value*,Value*> alreadyLoaded;

    std::function<Value*(Value*)> lookup = [&](Value* val) -> Value* {
      if (alreadyLoaded.find(val) != alreadyLoaded.end()) {
        return alreadyLoaded[val];
      }
      return alreadyLoaded[val] = lookupM(val, Builder2);
    };

    std::function<Value*(Value*,IRBuilder<>&)> lookupOrAllocateM = [&](Value* val,IRBuilder<> &BuilderM) -> Value* {
        if (auto inst = dyn_cast<AllocaInst>(val)) {
            if (antiallocas.find(val) == antiallocas.end()) {
                auto sz = lookupM(inst->getArraySize(), entryBuilder);
                antiallocas[val] = entryBuilder.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), sz, inst->getName()+"'loa");
                cast<AllocaInst>(antiallocas[val])->setAlignment(inst->getAlignment());
                entryBuilder.CreateStore(Constant::getNullValue(inst->getAllocatedType()), antiallocas[val]);
                /*
                Value *args[] = {entryBuilder.CreateBitCast(antiallocas[val],Type::getInt8PtrTy(Context)), ConstantInt::get(Type::getInt8Ty(val->getContext()), 0), entryBuilder.CreateMul(
                entryBuilder.CreateZExtOrTrunc(sz,Type::getInt64Ty(Context)),
                    ConstantInt::get(Type::getInt64Ty(Context), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8 ) ), ConstantInt::getFalse(Context) };
                Type *tys[] = {args[0]->getType(), args[2]->getType()};
                auto memset = cast<CallInst>(entryBuilder.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
                memset->addParamAttr(0, Attribute::getWithAlignment(Context, inst->getAlignment()));
                memset->addParamAttr(0, Attribute::NonNull);
                */
            }
            return antiallocas[val];
        } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
          auto ptr = lookupOrAllocateM(inst->getPointerOperand(),BuilderM);
          SmallVector<Value*,4> ind;
          for(auto& a : inst->indices()) {
            ind.push_back(lookup(a));
          }
          return BuilderM.CreateGEP(ptr, ind);
        } else if (auto inst = dyn_cast<CastInst>(val)) {
          auto ptr = lookupOrAllocateM(inst->getOperand(0),BuilderM);
          return BuilderM.CreateCast(inst->getOpcode(), ptr, inst->getDestTy());
        } else if (isa<Argument>(val)) {
          return ptrInputs[val];
        } else if (auto inst = dyn_cast<LoadInst>(val)) {
          //TODO this really should be looked up earlier 
          //  in case there was a modification between original
          //  and here
          auto ptr = lookupOrAllocateM(inst->getOperand(0),BuilderM);
          return BuilderM.CreateLoad(ptr);
        } else if (auto phi = dyn_cast<PHINode>(val)) {
            IRBuilder<> B(phi);
            PHINode* antiphi = B.CreatePHI(phi->getType(), phi->getNumIncomingValues());
            for(unsigned i=0; i<phi->getNumIncomingValues(); i++) {
                IRBuilder<> B2(phi->getIncomingBlock(i));
                auto val = lookupOrAllocateM(phi->getIncomingValue(i), B2);
                antiphi->addIncoming(val, phi->getIncomingBlock(i));
            }
            return lookupM(antiphi, BuilderM);
        }

        if( !val->getType()->isFPOrFPVectorTy()) {
            val->dump();
            val->getType()->dump();
        };
        assert(val->getType()->isFPOrFPVectorTy());
        return nullptr;
    };

    std::function<Value*(Value*)> lookupOrAllocate = [&](Value* val) -> Value* {
      return lookupOrAllocateM(val, Builder2);
    };

    auto diffe = [&](Value* val) -> Value* {
      assert(!val->getType()->isPointerTy());
      if (differentials.find(val) == differentials.end()) {
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'de");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
      }
      return Builder2.CreateLoad(differentials[val]);
    };

    std::function<Type*(Type*)> FloatToIntTy = [&](Type* T) -> Type* {
        assert(T->isFPOrFPVectorTy());
        if (auto ty = dyn_cast<VectorType>(T)) {
            return VectorType::get(FloatToIntTy(ty->getElementType()), ty->getNumElements());
        }
        if (T->isHalfTy()) return IntegerType::get(T->getContext(), 16); 
        if (T->isFloatTy()) return IntegerType::get(T->getContext(), 32); 
        if (T->isDoubleTy()) return IntegerType::get(T->getContext(), 64);
        assert(0 && "unknown floating point type");
        return nullptr;
    };

    std::function<Type*(Type*)> IntToFloatTy = [&](Type* T) -> Type* {
        assert(T->isIntOrIntVectorTy());
        if (auto ty = dyn_cast<VectorType>(T)) {
            return VectorType::get(IntToFloatTy(ty->getElementType()), ty->getNumElements());
        }
        if (auto ty = dyn_cast<IntegerType>(T)) {
            switch(ty->getBitWidth()) {
                case 16: return Type::getHalfTy(T->getContext());
                case 32: return Type::getFloatTy(T->getContext());
                case 64: return Type::getDoubleTy(T->getContext());
            }
        }
        assert(0 && "unknown int to floating point type");
        return nullptr;

    };

    auto addToDiffe = [&](Value* val, Value* dif) {
      assert(!isconstant(val));
      assert(val->getType() == dif->getType());
      auto old = diffe(val);
      assert(val->getType() == old->getType());
      Value* res;
      if (val->getType()->isIntOrIntVectorTy()) {
        res = Builder2.CreateFAdd(Builder2.CreateBitCast(old, IntToFloatTy(old->getType())), Builder2.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = Builder2.CreateBitCast(res, val->getType());
      } else {
        res = Builder2.CreateFAdd(old, dif);
      }
      Builder2.CreateStore(res, differentials[val]);
    };

    auto setDiffe = [&](Value* val, Value* toset) {
      assert(!isconstant(val));
      if (differentials.find(val) == differentials.end()) {
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'ds");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
      }
      Builder2.CreateStore(cast<Value>(toset), cast<Value>(differentials[val]));
    };

    auto addToDiffeIndexed = [&](Value* val, Value* dif, ArrayRef<Value*> idxs) {
      assert(!isconstant(val));
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

    std::function<Value*(Value*,IRBuilder<>&)> invertPointerM = [&](Value* val, IRBuilder<>& BuilderM) -> Value* {
      assert(val);
	  if (false && !val->getType()->isPointerTy()) {
		cast<Instruction>(val)->getParent()->getParent()->dump();
		cast<Instruction>(val)->getParent()->dump();
		val->dump();
        assert(val->getType()->isPointerTy());
	  }
      if (auto arg = dyn_cast<Argument>(val)) {
        return cast<Argument>(ptrInputs[arg]);
      } else if (auto arg = dyn_cast<CastInst>(val)) {
        return BuilderM.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), BuilderM), arg->getDestTy(), arg->getName()+"'ip");
      } else if (auto arg = dyn_cast<LoadInst>(val)) {
        auto li = BuilderM.CreateLoad(invertPointerM(arg->getOperand(0), BuilderM), arg->getName()+"'ip");
        li->setAlignment(arg->getAlignment());
        //llvm::errs() << "inverting pointer: " << *arg << "\n";
        //llvm::errs() << "inverted to pointer: " << *li << "\n";
        return li;
      } else if (auto arg = dyn_cast<GetElementPtrInst>(val)) {
        SmallVector<Value*,4> invertargs;
        for(auto &a: arg->indices()) {
            auto b = lookupM(a, BuilderM);
            invertargs.push_back(b);
        }
        return BuilderM.CreateGEP(invertPointerM(arg->getPointerOperand(), BuilderM), invertargs, arg->getName()+"'ip");
      } else if (auto inst = dyn_cast<AllocaInst>(val)) {
        if (antiallocas.find(val) == antiallocas.end()) {
            auto sz = lookupM(inst->getArraySize(), entryBuilder);
            antiallocas[val] = entryBuilder.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), sz, inst->getName()+"'ip");
            cast<AllocaInst>(antiallocas[val])->setAlignment(inst->getAlignment());
            Value *args[] = {entryBuilder.CreateBitCast(antiallocas[val],Type::getInt8PtrTy(Context)), ConstantInt::get(Type::getInt8Ty(val->getContext()), 0), entryBuilder.CreateMul(
            entryBuilder.CreateZExtOrTrunc(sz,Type::getInt64Ty(Context)),
                ConstantInt::get(Type::getInt64Ty(Context), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8 ) ), ConstantInt::getFalse(Context) };
            Type *tys[] = {args[0]->getType(), args[2]->getType()};
            auto memset = cast<CallInst>(entryBuilder.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
            memset->addParamAttr(0, Attribute::getWithAlignment(Context, inst->getAlignment()));
            memset->addParamAttr(0, Attribute::NonNull);
        }
        return lookupM(cast<AllocaInst>(antiallocas[inst]), BuilderM);
      } else if (auto call = dyn_cast<CallInst>(val)) {
        if (call->getCalledFunction()->getName() == "malloc") {
            if (antiallocas.find(val) == antiallocas.end()) {
                SmallVector<Value*, 8> args;
                for(unsigned i=0;i < call->getCalledFunction()->getFunctionType()->getNumParams(); i++) {
                    args.push_back(call->getArgOperand(i));
                }
                IRBuilder<> bb(call);
                antiallocas[val] = bb.CreateCall(call->getCalledFunction(), args, call->getName()+"'mi");
                cast<CallInst>(antiallocas[val])->setAttributes(call->getAttributes());
                if ( std::find(blockstodo.begin(), blockstodo.end(), call->getParent()) == blockstodo.end() ) {
                    assert(0);
                } else {
                
                }

                {
            Value *nargs[] = {bb.CreateBitCast(antiallocas[val],Type::getInt8PtrTy(Context)), ConstantInt::get(Type::getInt8Ty(val->getContext()), 0), bb.CreateZExtOrTrunc(args[0], Type::getInt64Ty(Context)), ConstantInt::getFalse(Context) };
            Type *tys[] = {nargs[0]->getType(), nargs[2]->getType()};

            auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), nargs));
            //memset->addParamAttr(0, Attribute::getWithAlignment(Context, inst->getAlignment()));
            memset->addParamAttr(0, Attribute::NonNull);
            addedStores.insert(memset);
                }

            }
            return lookupM(antiallocas[val], BuilderM);
        }
      
      } else if (auto phi = dyn_cast<PHINode>(val)) {
         /*if (antiallocas.find(val) == antiallocas.end()) {
            IRBuilder <> bb(phi);
            auto nphi = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
            for(size_t i = 0; i < phi->getNumIncomingValues(); i++) {
                nphi->addIncoming(invertPointer(phi->getIncomingValue(i)), phi->getIncomingBlock(i));
            }
            antiallocas[val] = nphi;
         }
         return antiallocas[val];*/

		 std::map<Value*,std::set<BasicBlock*>> mapped;
		 for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
			mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
		 }

		 if (false && mapped.size() == 1) {
         	return invertPointerM(phi->getIncomingValue(0), BuilderM);
		 } else if (false && mapped.size() == 2) {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(Type::getInt1Ty(Context), phi->getNumIncomingValues());

			 int cnt = 0;
			 Value* vals[2];
			 for(auto v : mapped) {
				assert( cnt <= 1 );
				vals[cnt] = v.first;
				for (auto b : v.second) {
					which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
				}
				cnt++;
			 }
			 
			 auto which2 = lookupM(which, BuilderM);
			 return BuilderM.CreateSelect(which2, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
		 } else {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());

			 for(auto v : mapped) {
				for (auto b : v.second) {
					IRBuilder <>pre(b->getTerminator());
					which->addIncoming(invertPointerM(v.first, pre), b);
				}
			 }
			 return lookupM(which, BuilderM);

		 }
      } else if (isa<ConstantPointerNull>(val)) {
         return val;
      } else if (auto cint = dyn_cast<ConstantInt>(val)) {
		 if (cint->isZero()) return cint;
	  }

        newFunc->dump();
        val->dump();
        assert(0 && "cannot find deal with ptr that isnt arg");
      
    };

    std::function<Value*(Value*)> invertPointer = [&](Value* val) -> Value* {
		return invertPointerM(val, Builder2);
	};

    auto addToPtrDiffe = [&](Value* val, Value* dif) {
      auto ptr = invertPointer(val);
      Value* res;
      Value* old = Builder2.CreateLoad(ptr);
      if (old->getType()->isIntOrIntVectorTy()) {
        res = Builder2.CreateFAdd(Builder2.CreateBitCast(old, IntToFloatTy(old->getType())), Builder2.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = Builder2.CreateBitCast(res, old->getType());
      } else if(old->getType()->isFPOrFPVectorTy()) {
        res = Builder2.CreateFAdd(old, dif);
      } else {
        newFunc->dump();
        llvm::errs() << "cannot handle type " << *old << "\n" << *dif;
        exit(1);
      }
      Builder2.CreateStore(res, ptr);
    };

    auto setPtrDiffe = [&](Value* val, Value* toset) {
      auto ptr = invertPointer(val);
      Builder2.CreateStore(toset, ptr);
    };

  auto term = BB->getTerminator();

  if(auto op = dyn_cast<ReturnInst>(term)) {
      auto retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.setFastMathFlags(FastMathFlags::getFast());
	  if (retAlloca)
		rb.CreateStore(retval, retAlloca);
	 
      rb.CreateBr(returnMerged);

      returnBuilder.CreateBr(BB2);
      returnBuilder.SetInsertPoint(&returnMerged->front());

      op->eraseFromParent();

      if (retval) {
        setDiffe(retval, ConstantFP::get(retval->getType(), 1.0));
      } else {
		assert (retAlloca == nullptr);
      }
  } else if (isa<BranchInst>(term)) {

  } else if (isa<UnreachableInst>(term)) {
  
  } else {
    BB->getParent()->dump();
    term->dump();
    llvm::errs() << "unknown return instance\n";
    assert(0 && "unknown return inst");
  }

  if (inLoop && loopContext.latch==BB) {
    BB2->getInstList().push_front(loopContext.antivar);

    IRBuilder<> tbuild(reverseBlocks[loopContext.exit]);

    // ensure we are before the terminator if it exists
    if (reverseBlocks[loopContext.exit]->size()) {
      tbuild.SetInsertPoint(&reverseBlocks[loopContext.exit]->back());
    }

    tbuild.setFastMathFlags(FastMathFlags::getFast());
    loopContext.antivar->addIncoming(lookupM(loopContext.limit, tbuild), reverseBlocks[loopContext.exit]);
    auto sub = Builder2.CreateSub(loopContext.antivar, ConstantInt::get(loopContext.antivar->getType(), 1));
    for(BasicBlock* in: successors(loopContext.latch) ) {
        if (LI.getLoopFor(in) == LI.getLoopFor(BB)) {
            loopContext.antivar->addIncoming(sub, reverseBlocks[in]);
        }
    }
    //TODO consider conditional and latch
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
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFMul(diffe(inst), lookup(op->getOperand(0)), "diffe"+op->getOperand(1)->getName());
          break;
        case Instruction::FAdd:{
          auto idiff = diffe(inst);
          if (!isconstant(op->getOperand(0)))
            dif0 = idiff;
          if (!isconstant(op->getOperand(1)))
            dif1 = idiff;
          break;
        }
        case Instruction::FSub:
          if (!isconstant(op->getOperand(0)))
            dif0 = diffe(inst);
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(diffe(inst));
          break;
        case Instruction::FDiv:
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(
              Builder2.CreateFDiv(
                Builder2.CreateFMul(diffe(inst), op),
                lookup(op->getOperand(1)))
            );
          break;

        default:
          newFunc->dump();
          op->dump();
          llvm::errs() << "cannot handle unknown binary operator\n";
          assert(0 && "unknown binary operator");
          exit(1);
      }

      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (!dif0 && !dif1) {
        constants.insert(op);
      }
    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::memcpy: {
            SmallVector<Value*, 4> args;
            args.push_back(invertPointer(op->getOperand(0)));
            args.push_back(invertPointer(op->getOperand(1)));
            args.push_back(lookup(op->getOperand(2)));
            args.push_back(lookup(op->getOperand(3)));

            Type *tys[] = {args[0]->getType(), args[1]->getType(), args[2]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memcpy, tys), args);
            cal->setAttributes(op->getAttributes());
            break;
        }
        case Intrinsic::memset: {
            if (!isconstant(op->getOperand(1))) {
                llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n";
                inst->dump();
                exit(1);
            }
            auto ptx = invertPointer(op->getOperand(0));
            SmallVector<Value*, 4> args;
            args.push_back(ptx);
            args.push_back(lookup(op->getOperand(1)));
            args.push_back(lookup(op->getOperand(2)));
            args.push_back(lookup(op->getOperand(3)));

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
        case Intrinsic::lifetime_start:{
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), lookup(op->getOperand(1))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::lifetime_end, tys), args);
            cal->setAttributes(op->getAttributes());
            break;
        }
        case Intrinsic::lifetime_end:
            op->eraseFromParent();
            break;
        case Intrinsic::sqrt: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.0), lookup(op))
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
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::log10: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::exp: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
            );
          break;
        }
        case Intrinsic::pow: {
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst),
                Builder2.CreateFDiv(lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
            );

          Value *args[] = {op->getOperand(1)};
          Type *tys[] = {op->getOperand(1)->getType()};

          if (!isconstant(op->getOperand(1)))
            dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );

          break;
        }
        case Intrinsic::sin: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          break;
        }
        case Intrinsic::cos: {
          Value *args[] = {lookup(op->getOperand(0))};
          Type *tys[] = {op->getOperand(0)->getType()};
          if (!isconstant(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateFNeg(
                Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args) )
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
      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {
        if(auto called = op->getCalledFunction()) {
            if (called->getName() == "printf" || called->getName() == "puts") {
                SmallVector<Value*, 4> args;
                for(unsigned i=0; i<op->getNumArgOperands(); i++) {
                    args.push_back(lookup(op->getArgOperand(i)));
                }
                auto cal = Builder2.CreateCall(called, args);
                cal->setAttributes(op->getAttributes());
            } else if (!op->getCalledFunction()->empty()) {
              SmallSet<unsigned,4> constant_args;

              SmallVector<Value*, 8> args;
              SmallVector<bool, 8> argsInverted;
              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (isconstant(op->getArgOperand(i))) {
                    constant_args.insert(i);
                    args.push_back(lookup(op->getArgOperand(i)));
                    argsInverted.push_back(false);
                    continue;
                }

                std::pair<Value*, Value*> loa;
                
                loa.first = lookup(op->getArgOperand(i));
                loa.second = lookupOrAllocate(op->getArgOperand(i));
                args.push_back(loa.first);
                argsInverted.push_back(loa.second == nullptr);
                /*
                if( (loa.second != nullptr) != (whatType(op->getArgOperand(i)->getType()) == DIFFE_TYPE::DUP_ARG) ) {
                    op->dump();
                    llvm::errs() << "loa.second: " << loa.second << "\n";
                    op->getArgOperand(i)->dump();
                    op->getArgOperand(i)->getType()->dump();
                }
                */
                //assert( (loa.second != nullptr) == (whatType(op->getArgOperand(i)->getType()) == DIFFE_TYPE::DUP_ARG) );
                if(loa.second) args.push_back(loa.second);
              }
              if (constant_args.size() == args.size()) break;

			  bool retUsed = false;
              for (const User *U : inst->users()) {
                if (auto si = dyn_cast<StoreInst>(U)) {
					if (si->getPointerOperand() == retAlloca && si->getValueOperand() == inst) {
						retUsed = true;
						continue;
					}
				}
				retUsed = false;
				break;
              }

              auto newcalled = CreatePrimalAndGradient(dyn_cast<Function>(called), constant_args, TLI, retUsed);//, LI, DT);

              auto diffes = Builder2.CreateCall(newcalled, args);
              diffes->setDebugLoc(inst->getDebugLoc());
              unsigned structidx = retUsed ? 1 : 0;
              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (argsInverted[i]) {
                  unsigned idxs[] = {structidx};
                  auto diffeadd = Builder2.CreateFMul( diffe(inst), Builder2.CreateExtractValue(diffes, idxs));
                  structidx++;
                  addToDiffe(op->getArgOperand(i), diffeadd);
                }
              }

              if (retUsed) {
                unsigned idxs[] = {0};
                auto retval = Builder2.CreateExtractValue(diffes, idxs);
				Builder2.CreateStore(retval, retAlloca);

				startremove:
              	for (User *U : inst->users()) {
                	if (auto si = dyn_cast<StoreInst>(U)) {
						if (si->getPointerOperand() == retAlloca && si->getValueOperand() == inst) {
							si->eraseFromParent();
							goto startremove;
						}
					}
                }

                // Previously this would be done, but really DCE should remove this if it doesn't affect anything
				// inst->eraseFromParent();
              }

			  if (inst->getNumUses() != 0)
              	setDiffe(inst, Constant::getNullValue(inst->getType()));
            }
            else if(called->getName()=="malloc") {
              auto ci = CallInst::CreateFree(Builder2.CreatePointerCast(lookup(inst), Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock());
              if (ci->getParent()==nullptr) {
                Builder2.GetInsertBlock()->getInstList().push_back(cast<Instruction>(ci));
              }

            }
            else if(called->getName()=="free") {
                if (auto dc = dyn_cast<CallInst>(op->getArgOperand(0))) {
                    if (dc->getCalledFunction()->getName() ==  "malloc") {
                        op->eraseFromParent();
                    }
                }
                //TODO HANDLE FREE
            } else {
              llvm::errs() << "cannot handle non invertible function\n" << *op << "\n";
              assert(0 && "unknown non invertible function");
              exit(1);
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
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<LoadInst>(inst)) {
       //TODO IF OP IS POINTER
      if (!op->getType()->isPointerTy()) {
        addToPtrDiffe(op->getOperand(0), diffe(inst));
        setDiffe(inst, Constant::getNullValue(inst->getType()));
      } else {
        //Builder2.CreateStore(diffe(inst), invertPointer(op->getOperand(0)));//, op->getName()+"'psweird");
        //addToNPtrDiffe(op->getOperand(0), diffe(inst));
        //assert(0 && "cannot handle non const pointer load inversion");
      }
    } else if(auto op = dyn_cast<StoreInst>(inst)) {
      //TODO const
      if (!isconstant(op->getValueOperand())) {
        auto dif1 = Builder2.CreateLoad(invertPointer(op->getPointerOperand()));
        addToDiffe(op->getValueOperand(), dif1);
      }
      setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()));

      //necessary if pointer is readwrite
      /*
      IRBuilder<> BuilderZ(inst);
      Builder2.CreateStore(
        lookup(BuilderZ.CreateLoad(op->getPointerOperand())), lookup(op->getPointerOperand()));
      */
    } else if(auto op = dyn_cast<ExtractValueInst>(inst)) {
      //todo const
      SmallVector<Value*,4> sv;
      for(auto i : op->getIndices())
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), i));
      addToDiffeIndexed(op->getOperand(0), diffe(inst), sv);
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if (auto op = dyn_cast<ShuffleVectorInst>(inst)) {
      //TODO const
      auto loaded = diffe(inst);
      size_t l1 = cast<VectorType>(op->getOperand(0)->getType())->getNumElements();
      uint64_t instidx = 0;
      for( auto idx : op->getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx-l1);
        SmallVector<Value*,4> sv;
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), opidx));
        addToDiffeIndexed(op->getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv);
        instidx++;
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<ExtractElementInst>(inst)) {
      //TODO const
      SmallVector<Value*,4> sv;
      sv.push_back(op->getIndexOperand());
      addToDiffeIndexed(op->getVectorOperand(), diffe(inst), sv);
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertElementInst>(inst)) {
      //TODO const
      auto dif1 = diffe(inst);
      addToDiffe(op->getOperand(0), Builder2.CreateInsertElement(dif1, Constant::getNullValue(op->getOperand(1)->getType()), lookup(op->getOperand(2)) ));

      addToDiffe(op->getOperand(1), Builder2.CreateExtractElement(dif1, lookup(op->getOperand(2))));
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<CastInst>(inst)) {
      if (op->getOpcode()==CastInst::CastOps::FPTrunc || op->getOpcode()==CastInst::CastOps::FPExt) {
        addToDiffe(op->getOperand(0), Builder2.CreateFPCast(diffe(inst), op->getOperand(0)->getType()));
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
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

  SmallVector<BasicBlock*,4> preds;
  for(auto B : predecessors(BB)) {
    preds.push_back(B);
  }

  if (preds.size() == 0) {
    SmallVector<Value *,4> retargs;

    if (returnValue) {
      retargs.push_back(Builder2.CreateLoad(retAlloca));
      assert(retargs[0]);
    }

    for (auto& I: newFunc->args()) {
      if (!isconstant(&I) && whatType(I.getType()) == DIFFE_TYPE::OUT_DIFF ) {
        retargs.push_back(diffe((Value*)&I));
      }
    }

    Value* toret = UndefValue::get(newFunc->getReturnType());
    for(unsigned i=0; i<retargs.size(); i++) {
      unsigned idx[] = { i };
      toret = Builder2.CreateInsertValue(toret, retargs[i], idx);
    }
    for(auto b : dynamicMallocCalls) {
	  auto a = Builder2.CreateLoad(b);
      auto ci = CallInst::CreateFree(Builder2.CreatePointerCast(a, Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock());
      if (ci->getParent()==nullptr) {
        Builder2.GetInsertBlock()->getInstList().push_back(cast<Instruction>(ci));
      }
    }
    for(auto a : mallocCalls) {
      auto ci = CallInst::CreateFree(Builder2.CreatePointerCast(a, Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock());
      if (ci->getParent()==nullptr) {
        Builder2.GetInsertBlock()->getInstList().push_back(cast<Instruction>(ci));
      }
    }
    Builder2.CreateRet(toret);
  } else if (preds.size() == 1) {
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {
            if (isconstant(PN)) continue;
            if (!isconstant(PN->getIncomingValueForBlock(preds[0]))) {
                setDiffe(PN->getIncomingValueForBlock(preds[0]), diffe(PN) );
            }
            setDiffe(PN, Constant::getNullValue(PN->getType()));
        } else break;
    }

    Builder2.CreateBr(reverseBlocks[preds[0]]);

  } else if (preds.size() == 2) {
    IRBuilder <> pbuilder(&BB->front());
    pbuilder.setFastMathFlags(FastMathFlags::getFast());
    Value* phi = nullptr;

    if (inLoop && BB2 == reverseBlocks[loopContext.var->getParent()]) {
      assert( ((preds[0] == loopContext.latch) && (preds[1] == loopContext.preheader)) || ((preds[1] == loopContext.latch) && (preds[0] == loopContext.preheader)) );
      if (preds[0] == loopContext.latch)
        phi = Builder2.CreateICmpNE(loopContext.antivar, Constant::getNullValue(loopContext.antivar->getType()));
      else if(preds[1] == loopContext.latch)
        phi = Builder2.CreateICmpEQ(loopContext.antivar, Constant::getNullValue(loopContext.antivar->getType()));
      else {
        llvm::errs() << "weird behavior for loopContext\n";
        assert(0 && "illegal loopcontext behavior");
      }
    } else {
      std::map<BasicBlock*,std::set<unsigned>> seen;
      std::map<BasicBlock*,std::set<BasicBlock*>> done;
      std::deque<std::tuple<BasicBlock*,unsigned,BasicBlock*>> Q; // newblock, prednum, pred
      Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(preds[0], 0, BB));
      Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(preds[1], 1, BB));
      //done.insert(BB);

      while(Q.size()) {
            auto trace = Q.front();
            auto block = std::get<0>(trace);
            auto num = std::get<1>(trace);
            auto predblock = std::get<2>(trace);
            Q.pop_front();

            if (seen[block].count(num) && done[block].count(predblock)) {
              continue;
            }

            seen[block].insert(num);
            done[block].insert(predblock);

            if (seen[block].size() == 1) {
              for (BasicBlock *Pred : predecessors(block)) {
                Q.push_back(std::tuple<BasicBlock*,unsigned,BasicBlock*>(Pred, (*seen[block].begin()), block ));
              }
            }

            SmallVector<BasicBlock*,4> succs;
            bool allDone = true;
            for (BasicBlock *Succ : successors(block)) {
                succs.push_back(Succ);
                if (done[block].count(Succ) == 0) {
                  allDone = false;
                }
            }

            if (!allDone) {
              continue;
            }

            if (seen[block].size() == preds.size() && succs.size() == preds.size()) {
			  //TODO below doesnt generalize past 2
			  bool hasSingle = false;
              for(auto a : succs) {
                if (seen[a].size() == 1) {
				  hasSingle = true;
                }
              }
			  if (!hasSingle)
                  goto continueloop;
              if (auto branch = dyn_cast<BranchInst>(block->getTerminator())) {
                assert(branch->getCondition());
                phi = lookup(branch->getCondition());
				//llvm::errs() << "looke dup phi is " << *phi << " p0:" << preds[0]->getName() << " p1:" << preds[1]->getName() <<"\n";
				for(unsigned i=0; i<preds.size(); i++) {
					auto s = branch->getSuccessor(i);
					assert(s == succs[i]);
					if (seen[s].size() == 1) {
						if ( (*seen[s].begin()) != i) {
							//llvm::errs() << "creating not based off of " << i << " " << s->getName() << "\n";
							phi = Builder2.CreateNot(phi);
							break;
						} else {
							//llvm::errs() << "creating based off of " << i << " " << s->getName() << "\n";
							break;
						}
					}
				}
                goto endPHI;
              }

              break;
            }
			continueloop:;
      }

      phi = pbuilder.CreatePHI(Type::getInt1Ty(Context), 2);
      cast<PHINode>(phi)->addIncoming(ConstantInt::getTrue(phi->getType()), preds[0]);
      cast<PHINode>(phi)->addIncoming(ConstantInt::getFalse(phi->getType()), preds[1]);
      phi = lookup(phi);
      goto endPHI;

      endPHI:;
    }

    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {

            // POINTER TYPE THINGS
            if (PN->getType()->isPointerTy()) continue;
            if (isconstant(PN)) continue; 
            if (!isconstant(PN->getIncomingValueForBlock(preds[0]))) {
                auto dif = Builder2.CreateSelect(phi, diffe(PN), diffe(PN->getIncomingValueForBlock(preds[0])));
                setDiffe(PN->getIncomingValueForBlock(preds[0]), dif );
            }
            if (!isconstant(PN->getIncomingValueForBlock(preds[1]))) {
                auto dif = Builder2.CreateSelect(phi, diffe(PN->getIncomingValueForBlock(preds[1])), diffe(PN));
                setDiffe(PN->getIncomingValueForBlock(preds[1]), dif);
            }
            setDiffe(PN, Constant::getNullValue(PN->getType()));
        } else break;
    }
    auto f0 = cast<BasicBlock>(reverseBlocks[preds[0]]);
    auto f1 = cast<BasicBlock>(reverseBlocks[preds[1]]);
    Builder2.CreateCondBr(phi, f0, f1);
  } else {
    IRBuilder <> pbuilder(&BB->front());
    pbuilder.setFastMathFlags(FastMathFlags::getFast());
    Value* phi = nullptr;

    if (true) {
      phi = pbuilder.CreatePHI(Type::getInt8Ty(Context), preds.size());
      for(unsigned i=0; i<preds.size(); i++) {
        cast<PHINode>(phi)->addIncoming(ConstantInt::get(phi->getType(), i), preds[i]);
      }
      phi = lookup(phi);
    }

    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {
          if (isconstant(PN)) continue;

          // POINTER TYPE THINGS
          if (PN->getType()->isPointerTy()) continue;

          for(unsigned i=0; i<preds.size(); i++) {
            if (!isconstant(PN->getIncomingValueForBlock(preds[i]))) {
                auto cond = Builder2.CreateICmpEQ(phi, ConstantInt::get(phi->getType(), i));
                auto dif = Builder2.CreateSelect(cond, diffe(PN), diffe(PN->getIncomingValueForBlock(preds[i])));
                setDiffe(PN->getIncomingValueForBlock(preds[i]), dif);
            }
          }
          setDiffe(PN, Constant::getNullValue(PN->getType()));
        } else break;
    }

    auto swit = Builder2.CreateSwitch(phi, reverseBlocks[preds.back()], preds.size()-1);
    for(unsigned i=0; i<preds.size()-1; i++) {
      swit->addCase(ConstantInt::get(cast<IntegerType>(phi->getType()), i), reverseBlocks[preds[i]]);
    }
  }


  }
  while(inversionAllocs->size() > 0) {
    inversionAllocs->back().moveBefore(&newFunc->getEntryBlock().front());
  }

  inversionAllocs->eraseFromParent();

  return newFunc;
}

void HandleAutoDiff(CallInst *CI, TargetLibraryInfo &TLI) {//, LoopInfo& LI, DominatorTree& DT) {
  Value* fn = CI->getArgOperand(0);

  while (auto ci = dyn_cast<CastInst>(fn)) {
    fn = ci->getOperand(0);
  }
  while (auto ci = dyn_cast<BlockAddress>(fn)) {
    fn = ci->getFunction();
  }
  while (auto ci = dyn_cast<ConstantExpr>(fn)) {
    fn = ci->getOperand(0);
  }
  if (!isa<Function>(fn)) {
    CI->getParent()->dump();
    CI->dump();
    fn->dump();
  }
  auto FT = cast<Function>(fn)->getFunctionType();

  SmallSet<unsigned,4> constants;
  SmallVector<Value*,2> args;

  unsigned truei = 0;
  IRBuilder<> Builder(CI);

  for(unsigned i=1; i<CI->getNumArgOperands(); i++) {
    Value* res = CI->getArgOperand(i);

    auto PTy = FT->getParamType(truei);
    auto ty = whatType(PTy);

    if (ty == DIFFE_TYPE::CONSTANT)
      constants.insert(truei);

    assert(truei < FT->getNumParams());
    if (PTy != res->getType()) {
      if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
        llvm::errs() << "Cannot cast __builtin_autodiff argument " << i << " " << *res << " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
        exit(1);
      }
      res = Builder.CreateBitCast(res, PTy);
    }

    args.push_back(res);
    if (ty == DIFFE_TYPE::DUP_ARG) {
      i++;

      Value* res = CI->getArgOperand(i);
      if (PTy != res->getType()) {
        if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
          llvm::errs() << "Cannot cast __builtin_autodiff argument " << i << " " << *res << " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
          exit(1);
        }
        res = Builder.CreateBitCast(res, PTy);
      }
      args.push_back(res);
    }

    truei++;
  }

  auto newFunc = CreatePrimalAndGradient(cast<Function>(fn), constants, TLI, /*should return*/false);//, LI, DT);

  Builder.setFastMathFlags(FastMathFlags::getFast());

  Value* diffret = Builder.CreateCall(newFunc, args);
  if (cast<StructType>(diffret->getType())->getNumElements()>0) {
    unsigned idxs[] = {0};
    diffret = Builder.CreateExtractValue(diffret, idxs);
    CI->replaceAllUsesWith(diffret);
  } else {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  }
  CI->eraseFromParent();
}

static bool lowerAutodiffIntrinsic(Function &F, TargetLibraryInfo &TLI) {//, LoopInfo& LI, DominatorTree& DT) {
  bool Changed = false;

  for (BasicBlock &BB : F) {

    for (auto BI = BB.rbegin(), BE = BB.rend(); BI != BE;) {
      Instruction *Inst = &*BI++;
      CallInst *CI = dyn_cast_or_null<CallInst>(Inst);
      if (!CI) continue;

      Function *Fn = CI->getCalledFunction();
      if (Fn && Fn->getIntrinsicID() == Intrinsic::autodiff) {
        HandleAutoDiff(CI, TLI);//, LI, DT);
        Changed = true;
      }
    }
  }

  return Changed;
}

PreservedAnalyses LowerAutodiffIntrinsicPass::run(Function &F,
                                                FunctionAnalysisManager &) {
                                                llvm::errs() << "running via run\n";
  //if (lowerAutodiffIntrinsic(F, this->getAnalysis<TargetLibraryInfoWrapperPass>().getTargetLibraryInfo()))
    return PreservedAnalyses::none();

  //return PreservedAnalyses::all();
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
    return lowerAutodiffIntrinsic(F, TLI);
  }
};
}

char LowerAutodiffIntrinsic::ID = 0;
INITIALIZE_PASS_BEGIN(LowerAutodiffIntrinsic, "lower-autodiff",
                "Lower 'autodiff' Intrinsics", false, false)

INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(LowerAutodiffIntrinsic, "lower-autodiff",
                "Lower 'autodiff' Intrinsics", false, false)

FunctionPass *llvm::createLowerAutodiffIntrinsicPass() {
  return new LowerAutodiffIntrinsic();
}
