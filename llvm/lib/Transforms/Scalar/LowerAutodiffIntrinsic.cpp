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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/PhiValues.h"

#include "llvm/Transforms/Utils.h"

#include "llvm/InitializePasses.h"

#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Verifier.h"
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

static cl::opt<bool> autodiff_inline(
            "autodiff_inline", cl::init(false), cl::Hidden,
                cl::desc("Force inlining of autodiff"));

static cl::opt<bool> printconst(
            "autodiff_printconst", cl::init(false), cl::Hidden,
                cl::desc("Print constant detection algorithm"));

static cl::opt<bool> autodiff_print(
            "autodiff_print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

enum class DIFFE_TYPE {
  OUT_DIFF=0, // add differential to output struct
  DUP_ARG=1,  // duplicate the argument and store differential inside
  CONSTANT=2  // no differential
};

//note this doesn't handle recursive types!
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
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
    assert(0 && "Cannot handle type0");
    return DIFFE_TYPE::CONSTANT;
  } else if (arg->isArrayTy()) {
    return whatType(cast<llvm::ArrayType>(arg)->getElementType());
  } else if (arg->isStructTy()) {
    auto st = cast<llvm::StructType>(arg);
    if (st->getNumElements() == 0) return DIFFE_TYPE::CONSTANT;

    auto ty = DIFFE_TYPE::CONSTANT;
    for(unsigned i=0; i<st->getNumElements(); i++) {
      switch(whatType(st->getElementType(i))) {
        case DIFFE_TYPE::OUT_DIFF:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                case DIFFE_TYPE::CONSTANT:
                  ty = DIFFE_TYPE::OUT_DIFF;
                  break;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
                  return ty;
              }
        case DIFFE_TYPE::CONSTANT:
              switch(ty) {
                case DIFFE_TYPE::OUT_DIFF:
                  ty = DIFFE_TYPE::OUT_DIFF;
                  break;
                case DIFFE_TYPE::CONSTANT:
                  break;
                case DIFFE_TYPE::DUP_ARG:
                  ty = DIFFE_TYPE::DUP_ARG;
                  return ty;
              }
        case DIFFE_TYPE::DUP_ARG:
            return DIFFE_TYPE::DUP_ARG;
      }
    }

    return ty;
  } else if (arg->isIntOrIntVectorTy() || arg->isFunctionTy ()) {
    return DIFFE_TYPE::CONSTANT;
  } else if  (arg->isFPOrFPVectorTy()) {
    return DIFFE_TYPE::OUT_DIFF;
  } else {
    assert(arg);
    llvm::errs() << "arg: " << *arg << "\n";
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

#if 0
bool isConstantValue(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant) {
	if(isa<Constant>(val) || isa<BasicBlock>(val) || isa<InlineAsm>(val) || (constants.find(val) != constants.end())) {
    	return true;
    }
	if (val->getType()->isVoidTy()) return true;
	
	SmallPtrSet<Value*, 20> constants2;
	constants2.insert(constants.begin(), constants.end());
	SmallPtrSet<Value*, 20> nonconstant2;
	nonconstant2.insert(nonconstant.begin(), nonconstant.end());
	constants2.insert(val);

	for(auto user : val->users()) {
		if (Instruction *inst = dyn_cast<Instruction>(user)) {
			if (auto pn = dyn_cast<PHINode>(inst)) {
				if (!isConstantValue(pn, constants2, nonconstant2)) {
					nonconstants.insert(val);
					return false;
				}
			} else if(isa<CmpInst>(inst)) {
				continue;
			} else if (auto bo = dyn_cast<BinaryOperator>(inst)) {
				if (!isConstantValue(bo, constants2, nonconstant2)) {
					nonconstants.insert(val);
					return false;
				}
			}

			assert(0 && "unknown using instruction");
			return false;
		} else {
			assert(0 && "cannot handle non instruction user");
		}
	}

	constants.insert(val);
	return true;
}
#endif

// TODO separate if the instruction is constant (i.e. could change things)
//    from if the value is constant (the value is something that could be differentiated)
bool isconstantM(Value* val, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, const SmallPtrSetImpl<Instruction*> &originalInstructions, uint8_t directions=3) {
    assert(val);
	constexpr uint8_t UP = 1;
	constexpr uint8_t DOWN = 2;
	//assert(directions >= 0);
	assert(directions <= 3);

	if(isa<Constant>(val) || isa<BasicBlock>(val) || isa<UnreachableInst>(val) || isa<BranchInst>(val) || isa<InlineAsm>(val) || (constants.find(val) != constants.end()) || (isa<Instruction>(val) && (originalInstructions.find(cast<Instruction>(val)) == originalInstructions.end()) ) ) {
    	return true;
    }

    if((nonconstant.find(val) != nonconstant.end())) {
        return false;
    }

    //All arguments should be marked constant/nonconstant ahead of time
    assert(!isa<Argument>(val));

	if (auto op = dyn_cast<CallInst>(val)) {
		if(auto called = op->getCalledFunction()) {
			if (called->getName() == "printf" || called->getName() == "puts") {
			//if (called->getName() == "printf" || called->getName() == "puts" || called->getName() == "__assert_fail") {
				nonconstant.insert(val);
				return false;
			}
		}
	}
	
    if (auto op = dyn_cast<CallInst>(val)) {
		if(auto called = op->getCalledFunction()) {
			if (called->getName() == "__assert_fail" || called->getName() == "free" || called->getName() == "_ZdlPv") {
			//if (called->getName() == "__assert_fail" || called->getName() == "malloc" || called->getName() == "free" || called->getName() =="_Znwm" || called->getName() == "_ZdlPv") {
				constants.insert(val);
				return true;
			}
		}
	}

    if (auto op = dyn_cast<IntrinsicInst>(val)) {
		switch(op->getIntrinsicID()) {
			case Intrinsic::stacksave:
			case Intrinsic::stackrestore:
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

	if (isa<CmpInst>(val)) {
		constants.insert(val);
		return true;
	}

    if (printconst)
	  llvm::errs() << "checking if is constant " << *val << "\n";

	if (val->getType()->isPointerTy()) {
	  if (auto inst = dyn_cast<Instruction>(val)) {

		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		constants2.insert(val);

		if (printconst)
			llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *val << "\n";

		for (const auto &a:inst->users()) {
		  if(auto store = dyn_cast<StoreInst>(a)) {
			if (!isconstantM(store->getValueOperand(), constants2, nonconstant2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(val);
    			if (printconst)
				  llvm::errs() << "memory erase 1: " << *val << "\n";
				return false;
			}
		  } else if (isa<LoadInst>(a)) continue;
		  else {
			if (!isconstantM(a, constants2, nonconstant2, originalInstructions, directions)) {
				if (directions == 3)
				  nonconstant.insert(val);
    			if (printconst)
				  llvm::errs() << "memory erase 2: " << *val << " op " << *a << "\n";
				return false;
			}
		  }

		}
		
		if (printconst)
			llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *val << "\n";
	  }
	}

	if (!val->getType()->isPointerTy() && !cast<Instruction>(val)->mayWriteToMemory() && (directions & DOWN) ) { 
		//Proceed assuming this is constant, can we prove this should be constant otherwise
		SmallPtrSet<Value*, 20> constants2;
		constants2.insert(constants.begin(), constants.end());
		SmallPtrSet<Value*, 20> nonconstant2;
		nonconstant2.insert(nonconstant.begin(), nonconstant.end());
		constants2.insert(val);

		if (printconst)
			llvm::errs() << " < USESEARCH" << (int)directions << ">" << *val << "\n";

		assert(!cast<Instruction>(val)->mayWriteToMemory());
		assert(!isa<StoreInst>(val));
		bool seenuse = false;
		for (const auto &a:val->users()) {
			if (auto gep = dyn_cast<GetElementPtrInst>(a)) {
				assert(val != gep->getPointerOperand());
				continue;
			}
			if (auto call = dyn_cast<CallInst>(a)) {
                auto fnp = call->getCalledFunction();
                if (fnp) {
                    auto fn = fnp->getName();
                    // todo realloc consider?
                    if (fn == "malloc" || fn == "_Znwm")
				        continue;
                }
			}
		  	if (!isconstantM(a, constants2, nonconstant2, originalInstructions, DOWN)) {
    			if (printconst)
			      llvm::errs() << "nonconstant inst (uses):" << *val << " user " << *a << "\n";
				seenuse = true;
				break;
			} else {
               if (printconst)
			   llvm::errs() << "found constant inst use:" << *val << " user " << *a << "\n";
			}
		}
		if (!seenuse) {
			constants.insert(val);
			constants.insert(constants2.begin(), constants2.end());
			// not here since if had full updown might not have been nonconstant
			//nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    		if (printconst)
			  llvm::errs() << "constant inst (uses):" << *val << "\n";
			return true;
		}
		
        if (printconst)
			llvm::errs() << " </USESEARCH" << (int)directions << ">" << *val << "\n";
	}

	SmallPtrSet<Value*, 20> constants2;
	constants2.insert(constants.begin(), constants.end());
	SmallPtrSet<Value*, 20> nonconstant2;
	nonconstant2.insert(nonconstant.begin(), nonconstant.end());
	constants2.insert(val);
		
    if (printconst)
		llvm::errs() << " < PRESEARCH" << (int)directions << ">" << *val << "\n";

	if (directions & UP) {

	if (auto inst = dyn_cast<PHINode>(val)) {
		for(auto& a: inst->incoming_values()) {
			if (!isconstantM(a, constants2, nonconstant2, originalInstructions, UP)) {
				if (directions == 3)
				  nonconstant.insert(val);
    			if (printconst)
				  llvm::errs() << "nonconstant phi " << *val << " op " << *a << "\n";
				return false;
			}
		}
		constants.insert(val);
		constants.insert(constants2.begin(), constants2.end());
		if (directions == 3)
		  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    	if (printconst)
		  llvm::errs() << "constant phi:" << *val << "\n";
		return true;
	} else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
	    // Handled uses above
		if (!isconstantM(inst->getPointerOperand(), constants2, nonconstant2, originalInstructions, UP)) {
            if (directions == 3)
			  nonconstant.insert(val);
    		if (printconst)
			  llvm::errs() << "nonconstant gep " << *val << " op " << *inst->getPointerOperand() << "\n";
			return false;
		}
		constants.insert(val);
		constants.insert(constants2.begin(), constants2.end());
		if (directions == 3)
		  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    	if (printconst)
		  llvm::errs() << "constant gep:" << *val << "\n";
		return true;
	} else if (auto inst = dyn_cast<Instruction>(val)) {
		for(auto& a: inst->operands()) {
			if (!isconstantM(a, constants2, nonconstant2, originalInstructions, UP)) {
                if (directions == 3)
				  nonconstant.insert(val);
    			if (printconst)
				  llvm::errs() << "nonconstant inst " << *val << " op " << *a << "\n";
				return false;
			}
		}

		constants.insert(val);
		constants.insert(constants2.begin(), constants2.end());
		if (directions == 3)
		  nonconstant.insert(nonconstant2.begin(), nonconstant2.end());
    	if (printconst)
		  llvm::errs() << "constant inst:" << *val << "\n";
		return true;
	}

	}

    if (printconst)
		llvm::errs() << " </PRESEARCH" << (int)directions << ">" << *val << "\n";

    if (directions == 3)
	  nonconstant.insert(val);
    if (printconst)
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

Function *CloneFunctionWithReturns(Function *F, ValueToValueMapTy& ptrInputs, const SmallSet<unsigned,4>& constant_args, SmallPtrSetImpl<Value*> &constants, SmallPtrSetImpl<Value*> &nonconstant, bool returnValue, bool differentialReturn) {
 assert(!F->empty());
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
     if (I.getType()->isPointerTy()) {
        ArgTypes.push_back(I.getType());
        /*
        if (I.getType()->isPointerTy() && !(I.hasAttribute(Attribute::ReadOnly) || I.hasAttribute(Attribute::ReadNone) ) ) {
          llvm::errs() << "Cannot take derivative of function " <<F->getName()<< " input argument to function " << I.getName() << " is not marked read-only\n";
          exit(1);
        }
        */
     } else { 
       RetTypes.push_back(I.getType());
     }
     argno++;
  }
 if (differentialReturn) {
    assert(!F->getReturnType()->isVoidTy());
    ArgTypes.push_back(F->getReturnType());
 }
 auto RetType = StructType::get(F->getContext(), RetTypes);

 // Create a new function type...
 FunctionType *FTy = FunctionType::get(RetType,
                                   ArgTypes, F->getFunctionType()->isVarArg());

 // Create the new function...
 Function *NewF = Function::Create(FTy, F->getLinkage(), "diffe"+F->getName(), F->getParent());
 if (differentialReturn) {
    auto I = NewF->arg_end();
    I--;
    I->setName("differeturn");
 }
 bool hasPtrInput = false;

 unsigned ii = 0, jj = 0;
 for (auto i=F->arg_begin(), j=NewF->arg_begin(); i != F->arg_end(); ) {
   bool isconstant = (constant_args.count(ii) > 0);

   if (isconstant) {
      constants.insert(j);
   } else {
	  nonconstant.insert(j);
   }


   if (!isconstant && i->getType()->isPointerTy()) {
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
	 nonconstant.insert(j);
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

 //SmallPtrSet<Value*,4> constants2;
 //for (auto a :constants){
 //   constants2.insert(a);
// }
 for(auto& r : Returns) {
   if (auto a = r->getReturnValue()) {
     nonconstant.insert(a);
   }
 }
 //for (auto a :nonconstant){
 //   nonconstant2.insert(a);
 //}
 if (true) {
    FunctionAnalysisManager AM;
 AM.registerPass([] { return AAManager(); });
 AM.registerPass([] { return ScalarEvolutionAnalysis(); });
 AM.registerPass([] { return AssumptionAnalysis(); });
 AM.registerPass([] { return TargetLibraryAnalysis(); });
 AM.registerPass([] { return DominatorTreeAnalysis(); });
 AM.registerPass([] { return MemoryDependenceAnalysis(); });
 AM.registerPass([] { return LoopAnalysis(); });
 AM.registerPass([] { return OptimizationRemarkEmitterAnalysis(); });
 AM.registerPass([] { return PhiValuesAnalysis(); });

    LoopSimplifyPass().run(*NewF, AM);

 }

  if(autodiff_inline) {
      llvm::errs() << "running inlining process\n";
   remover:
     SmallPtrSet<Instruction*, 10> originalInstructions;
     for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I) {
         originalInstructions.insert(&*I);
     }
   for (inst_iterator I = inst_begin(NewF), E = inst_end(NewF); I != E; ++I)
     if (auto call = dyn_cast<CallInst>(&*I)) {
        if (isconstantM(call, constants, nonconstant, originalInstructions)) continue;
        if (call->getCalledFunction() == nullptr) continue;
        if (call->getCalledFunction()->empty()) continue;
        /*
        if (call->getCalledFunction()->hasFnAttribute(Attribute::NoInline)) {
            llvm::errs() << "can't inline noinline " << call->getCalledFunction()->getName() << "\n";
            continue;
        }
        */
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

 if (autodiff_inline) {
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
 //LCSSAPass().run(*NewF, AM);
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

bool shouldRecompute(Value* val, const ValueToValueMapTy& available) {
          if (available.count(val)) return false;
          if (isa<Argument>(val) || isa<Constant>(val) ) {
            return false;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            return shouldRecompute(op->getOperand(0), available);
          } else if (isa<AllocaInst>(val)) {
            return true;
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
                    } else if(isa<CallInst>(idx)) {
                    //} else if(auto call = dyn_cast<CallInst>(idx)) {
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
}

    Type* FloatToIntTy(Type* T) {
        assert(T->isFPOrFPVectorTy());
        if (auto ty = dyn_cast<VectorType>(T)) {
            return VectorType::get(FloatToIntTy(ty->getElementType()), ty->getNumElements());
        }
        if (T->isHalfTy()) return IntegerType::get(T->getContext(), 16); 
        if (T->isFloatTy()) return IntegerType::get(T->getContext(), 32); 
        if (T->isDoubleTy()) return IntegerType::get(T->getContext(), 64);
        assert(0 && "unknown floating point type");
        return nullptr;
    }

    Type* IntToFloatTy(Type* T) {
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

bool operator==(const LoopContext& lhs, const LoopContext &rhs) {
    return lhs.parent == rhs.parent;
}

bool getContextM(BasicBlock *BB, LoopContext &loopContext, std::map<Loop*,LoopContext> &loopContexts, LoopInfo &LI,ScalarEvolution &SE,DominatorTree &DT) {
    if (auto L = LI.getLoopFor(BB)) {
        if (loopContexts.find(L) != loopContexts.end()) {
            loopContext = loopContexts.find(L)->second;
            return true;
        }

        SmallVector<BasicBlock *, 8> PotentialExitBlocks;
        SmallPtrSet<BasicBlock *, 8> ExitBlocks;
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
                    isExit = true;
                    goto exitblockcheck;
                }
            }

            
            exitblockcheck:
            if (isExit) {
				ExitBlocks.insert(a);
            }
        }

        if (ExitBlocks.size() != 1) {
            assert(BB);
            assert(BB->getParent());
            assert(L);
            llvm::errs() << *BB->getParent() << "\n";
            llvm::errs() << *L << "\n";
			for(auto b:ExitBlocks) {
                assert(b);
                llvm::errs() << *b << "\n";
            }
			llvm::errs() << "offending: \n";
			llvm::errs() << "No unique exit block (1)\n";
        }

        BasicBlock* ExitBlock = *ExitBlocks.begin(); //[0];

        BasicBlock *Header = L->getHeader();
        BasicBlock *Preheader = L->getLoopPreheader();
        assert(Preheader && "requires preheader");
        BasicBlock *Latch = L->getLoopLatch();

        const SCEV *Limit = SE.getExitCount(L, Latch);
        
		SCEVExpander Exp(SE, Preheader->getParent()->getParent()->getDataLayout(), "ad");

		PHINode *CanonicalIV = nullptr;
		Value *LimitVar = nullptr;
		if (SE.getCouldNotCompute() != Limit) {

        	CanonicalIV = canonicalizeIVs(Limit->getType(), L, SE, DT);
        	if (!CanonicalIV) {
                report_fatal_error("Couldn't get canonical IV.");
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
		  CanonicalIV = B.CreatePHI(Type::getInt64Ty(Header->getContext()), 1); // should be Header->getNumPredecessors());

		  B.SetInsertPoint(Header->getTerminator());
		  auto inc = B.CreateNUWAdd(CanonicalIV, ConstantInt::get(CanonicalIV->getType(), 1));
		  CanonicalIV->addIncoming(inc, Latch);
		  for (BasicBlock *Pred : predecessors(Header)) {
			  if (Pred != Latch) {
				  CanonicalIV->addIncoming(ConstantInt::get(CanonicalIV->getType(), 0), Pred);
			  }
		  }

		  B.SetInsertPoint(&ExitBlock->front());
		  LimitVar = B.CreatePHI(CanonicalIV->getType(), 1); // should be ExitBlock->getNumPredecessors());

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
  }

class GradientUtils {
public:
  llvm::Function *newFunc;
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Value*,4> constants;
  SmallPtrSet<Value*,20> nonconstant;
  DominatorTree DT;
  LoopInfo LI;
  AssumptionCache AC;
  ScalarEvolution SE;
  std::map<Loop*, LoopContext> loopContexts;
  SmallPtrSet<Instruction*, 10> originalInstructions;
  SmallVector<BasicBlock*, 12> originalBlocks;
  ValueMap<BasicBlock*,BasicBlock*> reverseBlocks;
  BasicBlock* inversionAllocs;
  ValueToValueMapTy scopeMap;
  std::vector<Instruction*> addedFrees;

protected:
  GradientUtils(Function* newFunc_, TargetLibraryInfo &TLI, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_) :
      newFunc(newFunc_), invertedPointers(), constants(constants_.begin(), constants_.end()), nonconstant(nonconstant_.begin(), nonconstant_.end()), DT(*newFunc_), LI(DT), AC(*newFunc_), SE(*newFunc_, TLI, AC, DT, LI), inversionAllocs(nullptr) {
        invertedPointers.insert(invertedPointers_.begin(), invertedPointers_.end());  
          for (BasicBlock &BB: *newFunc) {
            originalBlocks.emplace_back(&BB);
            for(Instruction &I : BB) {
                originalInstructions.insert(&I);
            }
          }
        assert(originalBlocks.size() > 0);
    }

public:
  static GradientUtils* CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, const SmallSet<unsigned,4> & constant_args, bool returnValue, bool differentialReturn) {
    assert(!todiff->empty());
    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    auto newFunc = CloneFunctionWithReturns(todiff, invertedPointers, constant_args, constants, nonconstant, returnValue, differentialReturn);
    return new GradientUtils(newFunc, TLI, invertedPointers, constants, nonconstant);
  }

  void prepareForReverse() {
    assert(reverseBlocks.size() == 0);
    for (BasicBlock *BB: originalBlocks) {
      reverseBlocks[BB] = BasicBlock::Create(BB->getContext(), "invert" + BB->getName(), newFunc);
    }
    assert(reverseBlocks.size() != 0);
  }

  BasicBlock* originalForReverseBlock(BasicBlock& BB2) const {
    assert(reverseBlocks.size() != 0);
    for(auto BB : originalBlocks) {
        auto it = reverseBlocks.find(BB);
        assert(it != reverseBlocks.end());
        if (it->second == &BB2) {
            return BB;
        }
    }
    report_fatal_error("could not find original block for given reverse block");
  }
 
  bool getContext(BasicBlock* BB, LoopContext& loopContext) {
    return getContextM(BB, loopContext, this->loopContexts, this->LI, this->SE, this->DT);
  }

  bool isOriginalBlock(const BasicBlock &BB) const {
    for(auto A : originalBlocks) {
        if (A == &BB) return true;
    }
    return false;
  }

  bool isConstantValue(Value* val) {
    if (val->getType()->isVoidTy()) return true;
    return isconstantM(val, constants, nonconstant, originalInstructions);
  };
  
  Value* unwrapM(Value* val, IRBuilder<>& BuilderM, const ValueToValueMapTy& available, bool lookupIfAble) {
          assert(val);
          if (available.count(val)) {
            return available.lookup(val);
          } 

          if (isa<Argument>(val) || isa<Constant>(val)) {
            return val;
          } else if (isa<AllocaInst>(val)) {
            return val;
          } else if (auto op = dyn_cast<CastInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) return nullptr;
            return BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(), op->getName()+"_unwrap");
          } else if (auto op = dyn_cast<BinaryOperator>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) return nullptr;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) return nullptr;
            return BuilderM.CreateBinOp(op->getOpcode(), op0, op1);
          } else if (auto op = dyn_cast<ICmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) return nullptr;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) return nullptr;
            return BuilderM.CreateICmp(op->getPredicate(), op0, op1);
          } else if (auto op = dyn_cast<FCmpInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) return nullptr;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) return nullptr;
            return BuilderM.CreateFCmp(op->getPredicate(), op0, op1);
          } else if (auto op = dyn_cast<SelectInst>(val)) {
            auto op0 = unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble);
            if (op0 == nullptr) return nullptr;
            auto op1 = unwrapM(op->getOperand(1), BuilderM, available, lookupIfAble);
            if (op1 == nullptr) return nullptr;
            auto op2 = unwrapM(op->getOperand(2), BuilderM, available, lookupIfAble);
            if (op2 == nullptr) return nullptr;
            return BuilderM.CreateSelect(op0, op1, op2);
          } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
              auto ptr = unwrapM(inst->getPointerOperand(), BuilderM, available, lookupIfAble);
              if (ptr == nullptr) return nullptr;
              SmallVector<Value*,4> ind;
              for(auto& a : inst->indices()) {
                auto op = unwrapM(a, BuilderM,available, lookupIfAble);
                if (op == nullptr) return nullptr;
                ind.push_back(op);
              }
              return BuilderM.CreateGEP(ptr, ind);
          } else if (auto load = dyn_cast<LoadInst>(val)) {
                Value* idx = unwrapM(load->getOperand(0), BuilderM, available, lookupIfAble);
                if (idx == nullptr) return nullptr;
                return BuilderM.CreateLoad(idx);
          } else if (auto op = dyn_cast<IntrinsicInst>(val)) {
            switch(op->getIntrinsicID()) {
                case Intrinsic::sin: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble)};
                  if (args[0] == nullptr) return nullptr;
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::sin, tys), args);
                }
                case Intrinsic::cos: {
                  Value *args[] = {unwrapM(op->getOperand(0), BuilderM, available, lookupIfAble)};
                  if (args[0] == nullptr) return nullptr;
                  Type *tys[] = {op->getOperand(0)->getType()};
                  return BuilderM.CreateCall(Intrinsic::getDeclaration(op->getParent()->getParent()->getParent(), Intrinsic::cos, tys), args);
                }
                default:;

            }
          } else if (auto phi = dyn_cast<PHINode>(val)) {
            if (phi->getNumIncomingValues () == 1) {
                return unwrapM(phi->getIncomingValue(0), BuilderM, available, lookupIfAble);
            }
          }

            assert(val);
            llvm::errs() << "cannot unwrap following " << *val << "\n";
            if (lookupIfAble)
                return lookupM(val, BuilderM);
          
          if (auto inst = dyn_cast<Instruction>(val)) {
            //LoopContext lc;
            // if (BuilderM.GetInsertBlock() != inversionAllocs && !( (reverseBlocks.find(BuilderM.GetInsertBlock()) != reverseBlocks.end())  && /*inLoop*/getContext(inst->getParent(), lc)) ) {
            if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        return inst;
                    }
                } else {
                    if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        return inst;
                    }
                }
            }
          }
            return nullptr;
            report_fatal_error("unable to unwrap");
    }
    Value* lookupM(Value* val, IRBuilder<>& BuilderM) {
        if (isa<Constant>(val)) return val;
        auto M = BuilderM.GetInsertBlock()->getParent()->getParent();
        if (auto inst = dyn_cast<Instruction>(val)) {
            if (this->inversionAllocs && inst->getParent() == this->inversionAllocs) {
                return val;
            }
            
            if (this->isOriginalBlock(*BuilderM.GetInsertBlock())) {
                if (BuilderM.GetInsertBlock()->size() && BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
                    if (this->DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
                        //llvm::errs() << "allowed " << *inst << "from domination\n";
                        return inst;
                    }
                } else {
                    if (this->DT.dominates(inst, BuilderM.GetInsertBlock())) {
                        //llvm::errs() << "allowed " << *inst << "from block domination\n";
                        return inst;
                    }
                }
            }

            assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));
            LoopContext lc;
            bool inLoop = getContext(inst->getParent(), lc);

            ValueToValueMapTy emptyMap;
            
            ValueToValueMapTy available;
            if (inLoop) {
                for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx)) {
                  available[idx.var] = idx.antivar;
                  if (idx.parent == nullptr) break;
                }
                
                bool isChildLoop = false;

                auto builderLoop = LI.getLoopFor(originalForReverseBlock(*BuilderM.GetInsertBlock()));
                while (builderLoop) {
                  if (builderLoop->getHeader() == lc.header) {
                    isChildLoop = true;
                    break;
                  }
                  builderLoop = builderLoop->getParentLoop();
                }
                if (!isChildLoop) {
                    llvm::errs() << "manually performing lcssa for instruction" << *inst << " in block " << BuilderM.GetInsertBlock()->getName() << "\n";
                    assert(DT.dominates(inst, originalForReverseBlock(*BuilderM.GetInsertBlock())));
                    IRBuilder<> lcssa(&lc.exit->front());
                    auto lcssaPHI = lcssa.CreatePHI(inst->getType(), 1, inst->getName()+"!manual_lcssa");
                    for(auto pred : predecessors(lc.exit))
                        lcssaPHI->addIncoming(inst, pred);
                    return lookupM(lcssaPHI, BuilderM);
                }
            }
            if (!shouldRecompute(inst, available)) {
                auto op = unwrapM(inst, BuilderM, available, /*lookupIfAble*/true);
                assert(op);
                return op;
            }

            assert(inversionAllocs && "must be able to allocate inverted caches");
            IRBuilder<> entryBuilder(inversionAllocs);
            entryBuilder.setFastMathFlags(FastMathFlags::getFast());

            if (!inLoop) {
                if (scopeMap.find(val) == scopeMap.end()) {
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
                }
                auto result = BuilderM.CreateLoad(scopeMap[val]);
                return result;
            } else {
                if (scopeMap.find(val) == scopeMap.end()) {

                    ValueToValueMapTy valmap;
                    Value* size = nullptr;
                    bool dynamic = false;

                    BasicBlock* outermostPreheader = nullptr;

                    for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                        if (idx.parent == nullptr) {
                            outermostPreheader = idx.preheader;
                        }
                        if (idx.parent == nullptr) break;
                    }
                    assert(outermostPreheader);

                    IRBuilder <> allocationBuilder(&outermostPreheader->back());

                    for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
					  //TODO handle allocations for dynamic loops
					  if (idx.dynamic && idx.parent != nullptr) {
                        assert(idx.var);
                        assert(idx.var->getParent());
                        assert(idx.var->getParent()->getParent());
                        llvm::errs() << *idx.var->getParent()->getParent() << "\n"
                            << "idx.var=" <<*idx.var << "\n"
                            << "idx.limit=" <<*idx.limit << "\n";
                        llvm::errs() << "cannot handle non-outermost dynamic loop\n";
						assert(0 && "cannot handle non-outermost dynamic loop");
					  }
                      Value* ns = nullptr;
					  if (idx.dynamic) {
						ns = ConstantInt::get(idx.limit->getType(), 1);
                        dynamic = true;
					  } else {
                        Value* limitm1 = nullptr;
                        limitm1 = unwrapM(idx.limit, allocationBuilder, emptyMap, /*lookupIfAble*/false);
                        if (limitm1 == nullptr) {
                            assert(outermostPreheader);
                            assert(outermostPreheader->getParent());
                            llvm::errs() << *outermostPreheader->getParent() << "\n";
                            llvm::errs() << "needed value " << *idx.limit << " at " << allocationBuilder.GetInsertBlock()->getName() << "\n";
                        }
                        assert(limitm1);
						ns = allocationBuilder.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
					  }
                      if (size == nullptr) size = ns;
                      else size = allocationBuilder.CreateNUWMul(size, ns);
                      if (idx.parent == nullptr) break;
                    }

                      if (dynamic) {
					    auto allocation = CallInst::CreateMalloc(entryBuilder.GetInsertBlock(), size->getType(), val->getType(), ConstantInt::get(size->getType(), entryBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), size, nullptr, val->getName()+"_malloccache");
                        entryBuilder.Insert(cast<Instruction>(allocation));

                      	scopeMap[val] = entryBuilder.CreateAlloca(allocation->getType(), nullptr, val->getName()+"_dyncache");
					    entryBuilder.CreateStore(allocation, scopeMap[val]);	
                      } else {
					    auto allocation = CallInst::CreateMalloc(&allocationBuilder.GetInsertBlock()->back(), size->getType(), val->getType(), ConstantInt::get(size->getType(), allocationBuilder.GetInsertBlock()->getParent()->getParent()->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), size, nullptr, val->getName()+"_malloccache");
                        //allocationBuilder.GetInsertBlock()->getInstList().push_back(cast<Instruction>(allocation));
                        cast<Instruction>(allocation)->moveBefore(allocationBuilder.GetInsertBlock()->getTerminator());
                        scopeMap[val] = entryBuilder.CreateAlloca(allocation->getType(), nullptr, val->getName()+"_mdyncache");
					    allocationBuilder.CreateStore(allocation, scopeMap[val]);	
                      }

                      if (reverseBlocks.size() != 0) {
                        IRBuilder<> tbuild(reverseBlocks[outermostPreheader]);
                        tbuild.setFastMathFlags(FastMathFlags::getFast());

                        // ensure we are before the terminator if it exists
                        if (tbuild.GetInsertBlock()->size()) {
                              tbuild.SetInsertPoint(tbuild.GetInsertBlock()->getFirstNonPHI());
                        }
                      	
                        auto ci = CallInst::CreateFree(tbuild.CreatePointerCast(tbuild.CreateLoad(scopeMap[val]), Type::getInt8PtrTy(outermostPreheader->getContext())), tbuild.GetInsertBlock());
                        if (ci->getParent()==nullptr) {
                            tbuild.Insert(ci);
                        }
                      } else {
                          llvm::errs() << "warning not freeing lookupM allocation\n";
                          report_fatal_error("not freeing lookupM allocation");
                      }

                    Instruction* putafter = isa<PHINode>(inst) ? (inst->getParent()->getFirstNonPHI() ): inst->getNextNonDebugInstruction();
                    IRBuilder <> v(putafter);
                    v.setFastMathFlags(FastMathFlags::getFast());

                    SmallVector<Value*,3> indices;
                    SmallVector<Value*,3> limits;
					PHINode* dynamicPHI = nullptr;

                    for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                      indices.push_back(idx.var);
                        
					  if (idx.dynamic) {
						dynamicPHI = idx.var;
                        assert(dynamicPHI);
						llvm::errs() << "saw idx.dynamic:" << *dynamicPHI << "\n";
						assert(idx.parent == nullptr);
						break;
					  }

                      if (idx.parent == nullptr) break;
                      auto limitm1 = unwrapM(idx.limit, v, emptyMap, /*lookupIfAble*/false);
                      assert(limitm1);
                      auto lim = v.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
                      if (limits.size() != 0) {
                        lim = v.CreateNUWMul(lim, limits.back());
                      }
                      limits.push_back(lim);
                    }

                    Value* idx = nullptr;
                    for(unsigned i=0; i<indices.size(); i++) {
                      if (i == 0) {
                        idx = indices[i];
                      } else {
                        auto mul = v.CreateNUWMul(indices[i], limits[i-1]);
                        idx = v.CreateNUWAdd(idx, mul);
                      }
                    }

					Value* allocation = nullptr;
					if (dynamicPHI == nullptr) {
						//allocation = scopeMap[val];
						IRBuilder<> outerBuilder(&outermostPreheader->back());
						allocation = outerBuilder.CreateLoad(scopeMap[val]);
					} else {
						Type *BPTy = Type::getInt8PtrTy(v.GetInsertBlock()->getContext());
						auto realloc = M->getOrInsertFunction("realloc", BPTy, BPTy, size->getType());
						allocation = v.CreateLoad(scopeMap[val]);
						auto foo = v.CreateNUWAdd(dynamicPHI, ConstantInt::get(dynamicPHI->getType(), 1));
						Value* idxs[2] = {
							v.CreatePointerCast(allocation, BPTy),
							v.CreateNUWMul(
								ConstantInt::get(size->getType(), M->getDataLayout().getTypeAllocSizeInBits(val->getType())/8), 
								v.CreateNUWMul(
									size, foo
								) 
							)
						};

                        Value* realloccall = nullptr;
						allocation = v.CreatePointerCast(realloccall = v.CreateCall(realloc, idxs, val->getName()+"_realloccache"), allocation->getType());
						v.CreateStore(allocation, scopeMap[val]);
					}

                    Value* idxs[] = {idx};
                    auto gep = v.CreateGEP(allocation, idxs);
					v.CreateStore(val, gep);
                }
                assert(inLoop);

                SmallVector<Value*,3> indices;
                SmallVector<Value*,3> limits;
                for(LoopContext idx = lc; ; getContext(idx.parent->getHeader(), idx) ) {
                  indices.push_back(idx.antivar);
                  if (idx.parent == nullptr) break;

                  auto limitm1 = unwrapM(idx.limit, BuilderM, available, /*lookupIfAble*/true);
                  assert(limitm1);
                  auto lim = BuilderM.CreateNUWAdd(limitm1, ConstantInt::get(idx.limit->getType(), 1));
                  if (limits.size() != 0) {
                    lim = BuilderM.CreateNUWMul(lim, limits.back());
                  }
                  limits.push_back(lim);
                }

                Value* idx = nullptr;
                for(unsigned i=0; i<indices.size(); i++) {
                  if (i == 0) {
                    idx = indices[i];
                  } else {
                    idx = BuilderM.CreateNUWAdd(idx, BuilderM.CreateNUWMul(indices[i], limits[i-1]));
                  }
                }

                Value* idxs[] = {idx};
				Value* tolookup = BuilderM.CreateLoad(scopeMap[val]);
                return BuilderM.CreateLoad(BuilderM.CreateGEP(tolookup, idxs));
            }
        }

        return val;
    };
    
    Value* invertPointerM(Value* val, IRBuilder<>& BuilderM) {
      if (isa<ConstantPointerNull>(val)) {
         return val;
      } else if (auto cint = dyn_cast<ConstantInt>(val)) {
		 if (cint->isZero()) return cint;
	  }

      assert(!isConstantValue(val));
      auto M = BuilderM.GetInsertBlock()->getParent()->getParent();
      assert(val);

      if (invertedPointers.find(val) != invertedPointers.end()) {
        return lookupM(invertedPointers[val], BuilderM);
      }

      if (auto arg = dyn_cast<CastInst>(val)) {
        auto result = BuilderM.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), BuilderM), arg->getDestTy(), arg->getName()+"'ipc");
        return result;
      } else if (auto arg = dyn_cast<LoadInst>(val)) {
        auto li = BuilderM.CreateLoad(invertPointerM(arg->getOperand(0), BuilderM), arg->getName()+"'ipl");
        li->setAlignment(arg->getAlignment());
        return li;
      } else if (auto arg = dyn_cast<GetElementPtrInst>(val)) {
        SmallVector<Value*,4> invertargs;
        for(auto &a: arg->indices()) {
            auto b = lookupM(a, BuilderM);
            invertargs.push_back(b);
        }
        auto result = BuilderM.CreateGEP(invertPointerM(arg->getPointerOperand(), BuilderM), invertargs, arg->getName()+"'ipg");
        return result;
      } else if (auto inst = dyn_cast<AllocaInst>(val)) {
            IRBuilder<> bb(inst);
            AllocaInst* antialloca = bb.CreateAlloca(inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(), inst->getArraySize(), inst->getName()+"'ipa");
            invertedPointers[val] = antialloca;
            antialloca->setAlignment(inst->getAlignment());
            Value *args[] = {bb.CreateBitCast(antialloca,Type::getInt8PtrTy(val->getContext())), ConstantInt::get(Type::getInt8Ty(val->getContext()), 0), bb.CreateNUWMul(
            bb.CreateZExtOrTrunc(inst->getArraySize(),Type::getInt64Ty(val->getContext())),
                ConstantInt::get(Type::getInt64Ty(val->getContext()), M->getDataLayout().getTypeAllocSizeInBits(inst->getAllocatedType())/8 ) ), ConstantInt::getFalse(val->getContext()) };
            Type *tys[] = {args[0]->getType(), args[2]->getType()};
            auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
            memset->addParamAttr(0, Attribute::getWithAlignment(inst->getContext(), inst->getAlignment()));
            memset->addParamAttr(0, Attribute::NonNull);
            return lookupM(invertedPointers[inst], BuilderM);
      } else if (auto call = dyn_cast<CallInst>(val)) {
        if (call->getCalledFunction() && call->getCalledFunction()->getName() == "malloc") {
                IRBuilder<> bb(call);
                {
                SmallVector<Value*, 8> args;
                for(unsigned i=0;i < call->getCalledFunction()->getFunctionType()->getNumParams(); i++) {
                    args.push_back(call->getArgOperand(i));
                }
                invertedPointers[val] = bb.CreateCall(call->getCalledFunction(), args, call->getName()+"'mi");
                }

                cast<CallInst>(invertedPointers[val])->setAttributes(call->getAttributes());

                {
            Value *nargs[] = {
                bb.CreateBitCast(invertedPointers[val],Type::getInt8PtrTy(val->getContext())),
                ConstantInt::get(Type::getInt8Ty(val->getContext()), 0),
                bb.CreateZExtOrTrunc(call->getArgOperand(0), Type::getInt64Ty(val->getContext())), ConstantInt::getFalse(val->getContext())
            };
            Type *tys[] = {nargs[0]->getType(), nargs[2]->getType()};

            auto memset = cast<CallInst>(bb.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::memset, tys), nargs));
            //memset->addParamAttr(0, Attribute::getWithAlignment(Context, inst->getAlignment()));
            memset->addParamAttr(0, Attribute::NonNull);
                }

                if (reverseBlocks.size()) {
                    IRBuilder<> freeBuilder(reverseBlocks[call->getParent()]);
                    if (auto term = freeBuilder.GetInsertBlock()->getTerminator()) {
                        freeBuilder.SetInsertPoint(term);
                    }
                    freeBuilder.setFastMathFlags(FastMathFlags::getFast());
                    auto ci = CallInst::CreateFree(freeBuilder.CreatePointerCast(lookupM(invertedPointers[val], freeBuilder), Type::getInt8PtrTy(call->getContext())), freeBuilder.GetInsertBlock());
                    if (ci->getParent()==nullptr) {
                      freeBuilder.Insert(ci);
                    }
                    addedFrees.push_back(ci);
                } else {
                    llvm::errs() << "warning in duplication not creating free for inverted pointer\n";
                }

            return lookupM(invertedPointers[val], BuilderM);
        }
      
      } else if (auto phi = dyn_cast<PHINode>(val)) {
		 std::map<Value*,std::set<BasicBlock*>> mapped;
		 for(unsigned int i=0; i<phi->getNumIncomingValues(); i++) {
			mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
		 }

		 if (false && mapped.size() == 1) {
         	return invertPointerM(phi->getIncomingValue(0), BuilderM);
		 }    
#if 0
         else if (false && mapped.size() == 2) {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
             //TODO this is not recursive

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
			 auto result = BuilderM.CreateSelect(which2, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
             return result;
		 } 
#endif
         
         else {
			 IRBuilder <> bb(phi);
			 auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
             invertedPointers[val] = which;

			 for(auto v : mapped) {
				for (auto b : v.second) {
					IRBuilder <>pre(b->getTerminator());
					which->addIncoming(invertPointerM(v.first, pre), b);
				}
			 }
			 return lookupM(which, BuilderM);
		 }
        }
        assert(BuilderM.GetInsertBlock());
        assert(BuilderM.GetInsertBlock()->getParent());
        assert(val);
        llvm::errs() << "fn:" << *BuilderM.GetInsertBlock()->getParent() << "\nval=" << *val << "\n";
        report_fatal_error("cannot find deal with ptr that isnt arg");
      
    };
};
  
class DiffeGradientUtils : public GradientUtils {
  DiffeGradientUtils(Function* newFunc_, TargetLibraryInfo &TLI, ValueToValueMapTy& invertedPointers_, const SmallPtrSetImpl<Value*> &constants_, const SmallPtrSetImpl<Value*> &nonconstant_)
      : GradientUtils(newFunc_, TLI, invertedPointers_, constants_, nonconstant_) {
        prepareForReverse();
        inversionAllocs = BasicBlock::Create(newFunc_->getContext(), "allocsForInversion", newFunc);
    }
  ValueToValueMapTy differentials;

public:
  static DiffeGradientUtils* CreateFromClone(Function *todiff, TargetLibraryInfo &TLI, const SmallSet<unsigned,4> & constant_args, bool returnValue, bool differentialReturn) {
    assert(!todiff->empty());
    ValueToValueMapTy invertedPointers;
    SmallPtrSet<Value*,4> constants;
    SmallPtrSet<Value*,20> nonconstant;
    auto newFunc = CloneFunctionWithReturns(todiff, invertedPointers, constant_args, constants, nonconstant, returnValue, differentialReturn);
    return new DiffeGradientUtils(newFunc, TLI, invertedPointers, constants, nonconstant);
  }

private:
  Value* getDifferential(Value *val) {
    assert(inversionAllocs);
    if (differentials.find(val) == differentials.end()) {
        IRBuilder<> entryBuilder(inversionAllocs);
        entryBuilder.setFastMathFlags(FastMathFlags::getFast());
        differentials[val] = entryBuilder.CreateAlloca(val->getType(), nullptr, val->getName()+"'de");
        entryBuilder.CreateStore(Constant::getNullValue(val->getType()), differentials[val]);
    }
    return differentials[val];
  }

public:
  Value* diffe(Value* val, IRBuilder<> &BuilderM) {
      assert(!val->getType()->isPointerTy());
      assert(!val->getType()->isVoidTy());
      return BuilderM.CreateLoad(getDifferential(val));
  }

  void addToDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
      assert(!val->getType()->isPointerTy());
      assert(!isConstantValue(val));
      assert(val->getType() == dif->getType());
      auto old = diffe(val, BuilderM);
      assert(val->getType() == old->getType());
      Value* res;
      if (val->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, val->getType());
      } else {
        res = BuilderM.CreateFAdd(old, dif);
      }
      BuilderM.CreateStore(res, getDifferential(val));
  }

  void setDiffe(Value* val, Value* toset, IRBuilder<> &BuilderM) {
      assert(!isConstantValue(val));
      BuilderM.CreateStore(toset, getDifferential(val));
  }

  void addToDiffeIndexed(Value* val, Value* dif, ArrayRef<Value*> idxs, IRBuilder<> &BuilderM) {
      assert(!isConstantValue(val));
      SmallVector<Value*,4> sv;
      sv.push_back(ConstantInt::get(Type::getInt32Ty(val->getContext()), 0));
      for(auto i : idxs)
        sv.push_back(i);
      auto ptr = BuilderM.CreateGEP(getDifferential(val), sv);
      BuilderM.CreateStore(BuilderM.CreateFAdd(BuilderM.CreateLoad(ptr), dif), ptr);
  }

  void addToPtrDiffe(Value* val, Value* dif, IRBuilder<> &BuilderM) {
      auto ptr = invertPointerM(val, BuilderM);
      Value* res;
      Value* old = BuilderM.CreateLoad(ptr);
      if (old->getType()->isIntOrIntVectorTy()) {
        res = BuilderM.CreateFAdd(BuilderM.CreateBitCast(old, IntToFloatTy(old->getType())), BuilderM.CreateBitCast(dif, IntToFloatTy(dif->getType())));
        res = BuilderM.CreateBitCast(res, old->getType());
      } else if(old->getType()->isFPOrFPVectorTy()) {
        res = BuilderM.CreateFAdd(old, dif);
      } else {
        assert(old);
        assert(dif);
        llvm::errs() << *newFunc << "\n" << "cannot handle type " << *old << "\n" << *dif;
        report_fatal_error("cannot handle type");
      }
      BuilderM.CreateStore(res, ptr);
  }

};

Function* CreatePrimalAndGradient(Function* todiff, const SmallSet<unsigned,4>& constant_args, TargetLibraryInfo &TLI, bool returnValue, bool differentialReturn) {
  assert(!todiff->empty());
  auto M = todiff->getParent();
  auto& Context = M->getContext();

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(todiff, TLI, constant_args, returnValue, differentialReturn);
 
  auto isconstant = [&gutils](Value* val) -> bool {
    return isconstantM(val, gutils->constants, gutils->nonconstant, gutils->originalInstructions);
  };

  llvm::AllocaInst* retAlloca = nullptr;
  if (returnValue && differentialReturn) {
	retAlloca = IRBuilder<>(&gutils->newFunc->getEntryBlock().front()).CreateAlloca(todiff->getReturnType(), nullptr, "toreturn");
  }

  // Force loop canonicalization everywhere
  for(BasicBlock* BB: gutils->originalBlocks) {
    LoopContext loopContext;
    gutils->getContext(BB, loopContext);
  }

  for(BasicBlock* BB: gutils->originalBlocks) {

    LoopContext loopContext;
    bool inLoop = gutils->getContext(BB, loopContext);

    auto BB2 = gutils->reverseBlocks[BB];
    assert(BB2);

    IRBuilder<> Builder2(BB2);
    if (BB2->size() > 0) {
        Builder2.SetInsertPoint(BB2->getFirstNonPHI());
    }
    Builder2.setFastMathFlags(FastMathFlags::getFast());

    std::map<Value*,Value*> alreadyLoaded;

    std::function<Value*(Value*)> lookup = [&](Value* val) -> Value* {
      if (alreadyLoaded.find(val) != alreadyLoaded.end()) {
        return alreadyLoaded[val];
      }
      return alreadyLoaded[val] = gutils->lookupM(val, Builder2);
    };

    auto diffe = [&Builder2,&gutils](Value* val) -> Value* {
        return gutils->diffe(val, Builder2);
    };

    auto addToDiffe = [&Builder2,&gutils](Value* val, Value* dif) -> void {
      gutils->addToDiffe(val, dif, Builder2);
    };

    auto setDiffe = [&](Value* val, Value* toset) -> void {
      gutils->setDiffe(val, toset, Builder2);
    };

    auto addToDiffeIndexed = [&](Value* val, Value* dif, ArrayRef<Value*> idxs) -> void{
      gutils->addToDiffeIndexed(val, dif, idxs, Builder2);
    };

    auto invertPointer = [&](Value* val) -> Value* {
        return gutils->invertPointerM(val, Builder2);
    };

    auto addToPtrDiffe = [&](Value* val, Value* dif) {
      gutils->addToPtrDiffe(val, dif, Builder2);
    };

  auto term = BB->getTerminator();
  assert(term);
  if(auto op = dyn_cast<ReturnInst>(term)) {
      auto retval = op->getReturnValue();
      IRBuilder<> rb(op);
      rb.setFastMathFlags(FastMathFlags::getFast());
	  if (retAlloca)
		rb.CreateStore(retval, retAlloca);
	 
      rb.CreateBr(BB2);

      op->eraseFromParent();

      if (differentialReturn && !gutils->isConstantValue(retval)) {
        //setDiffe(retval, ConstantFP::get(retval->getType(), 1.0));
        auto endarg = gutils->newFunc->arg_end();
        endarg--;
        setDiffe(retval, endarg);
      } else {
		assert (retAlloca == nullptr);
      }
  } else if (isa<BranchInst>(term) || isa<SwitchInst>(term)) {

  } else if (isa<UnreachableInst>(term)) {
  
  } else {
    assert(BB);
    assert(BB->getParent());
    assert(term);
    llvm::errs() << *BB->getParent() << "\n";
    llvm::errs() << "unknown terminator instance " << *term << "\n";
    assert(0 && "unknown terminator inst");
  }

  if (inLoop && loopContext.latch==BB) {
    BB2->getInstList().push_front(loopContext.antivar);

    IRBuilder<> tbuild(gutils->reverseBlocks[loopContext.exit]);
    tbuild.setFastMathFlags(FastMathFlags::getFast());

    // ensure we are before the terminator if it exists
    if (gutils->reverseBlocks[loopContext.exit]->size()) {
      tbuild.SetInsertPoint(&gutils->reverseBlocks[loopContext.exit]->back());
    }

    loopContext.antivar->addIncoming(gutils->lookupM(loopContext.limit, tbuild), gutils->reverseBlocks[loopContext.exit]);
    auto sub = Builder2.CreateSub(loopContext.antivar, ConstantInt::get(loopContext.antivar->getType(), 1));
    for(BasicBlock* in: successors(loopContext.latch) ) {
        if (gutils->LI.getLoopFor(in) == gutils->LI.getLoopFor(BB)) {
            loopContext.antivar->addIncoming(sub, gutils->reverseBlocks[in]);
        }
    }
  }

  if (!isa<UnreachableInst>(term))
  for (auto I = BB->rbegin(), E = BB->rend(); I != E;) {
    Instruction* inst = &*I;
    assert(inst);
    I++;
    if (gutils->originalInstructions.find(inst) == gutils->originalInstructions.end()) continue;
    //if (isconstant(inst)) continue;

    if (auto op = dyn_cast<BinaryOperator>(inst)) {
      if (isconstant(inst)) continue;
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getOpcode()) {
        case Instruction::FMul:
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFMul(diffe(inst), lookup(op->getOperand(0)), "diffe"+op->getOperand(1)->getName());
          break;
        case Instruction::FAdd:{
          auto idiff = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = idiff;
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = idiff;
          break;
        }
        case Instruction::FSub:
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = diffe(inst);
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(diffe(inst));
          break;
        case Instruction::FDiv:
          if (!gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(1)), "diffe"+op->getOperand(0)->getName());
          if (!gutils->isConstantValue(op->getOperand(1)))
            dif1 = Builder2.CreateFNeg(
              Builder2.CreateFDiv(
                Builder2.CreateFMul(diffe(inst), lookup(op)),
                lookup(op->getOperand(1)))
            );
          break;

        default:
          assert(op);
          llvm::errs() << *gutils->newFunc << "\n";
          llvm::errs() << "cannot handle unknown binary operator: " << *op << "\n";
          report_fatal_error("unknown binary operator");
      }

      setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<IntrinsicInst>(inst)) {
      Value* dif0 = nullptr;
      Value* dif1 = nullptr;
      switch(op->getIntrinsicID()) {
        case Intrinsic::memcpy: {
            if (isconstant(inst)) continue;
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
            if (isconstant(inst)) continue;
            if (!gutils->isConstantValue(op->getOperand(1))) {
                assert(inst);
                llvm::errs() << "couldn't handle non constant inst in memset to propagate differential to\n" << *inst;
                report_fatal_error("non constant in memset");
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
        case Intrinsic::stacksave:
        case Intrinsic::stackrestore:
        case Intrinsic::dbg_declare:
        case Intrinsic::dbg_value:
        case Intrinsic::dbg_label:
        case Intrinsic::dbg_addr:
            break;
        case Intrinsic::lifetime_start:{
            if (isconstant(inst)) continue;
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
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateBinOp(Instruction::FDiv, diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.0), lookup(op))
            );
          break;
        }
        case Intrinsic::fabs: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0))) {
            auto cmp = Builder2.CreateFCmpOLT(lookup(op->getOperand(0)), ConstantFP::get(op->getOperand(0)->getType(), 0));
            dif0 = Builder2.CreateSelect(cmp, ConstantFP::get(op->getOperand(0)->getType(), -1), ConstantFP::get(op->getOperand(0)->getType(), 1));
          }
          break;
        }
        case Intrinsic::log: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst), lookup(op->getOperand(0)));
          break;
        }
        case Intrinsic::log2: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 0.6931471805599453), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::log10: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFDiv(diffe(inst),
              Builder2.CreateFMul(ConstantFP::get(op->getType(), 2.302585092994046), lookup(op->getOperand(0)))
            );
          break;
        }
        case Intrinsic::exp: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(diffe(inst), lookup(op));
          break;
        }
        case Intrinsic::exp2: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0)))
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)), ConstantFP::get(op->getType(), 0.6931471805599453)
            );
          break;
        }
        case Intrinsic::pow: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0))) {

            /*
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst),
                Builder2.CreateFDiv(lookup(op), lookup(op->getOperand(0)))), lookup(op->getOperand(1))
            );
            */
            SmallVector<Value*, 2> args = {lookup(op->getOperand(0)), Builder2.CreateFSub(lookup(op->getOperand(1)), ConstantFP::get(op->getType(), 1.0))};
            Type *tys[] = {args[1]->getType()};
            auto cal = Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::pow, tys), args);
            cal->setAttributes(op->getAttributes());
            dif0 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), cal)
              , lookup(op->getOperand(1))
            );
          }

          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(1))) {
            Value *args[] = {lookup(op->getOperand(1))};
            Type *tys[] = {op->getOperand(1)->getType()};

            dif1 = Builder2.CreateFMul(
              Builder2.CreateFMul(diffe(inst), lookup(op)),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::log, tys), args)
            );
          }

          break;
        }
        case Intrinsic::sin: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::cos, tys), args) );
          }
          break;
        }
        case Intrinsic::cos: {
          if (!isconstant(op) && !gutils->isConstantValue(op->getOperand(0))) {
            Value *args[] = {lookup(op->getOperand(0))};
            Type *tys[] = {op->getOperand(0)->getType()};
            dif0 = Builder2.CreateFMul(diffe(inst),
              Builder2.CreateFNeg(
                Builder2.CreateCall(Intrinsic::getDeclaration(M, Intrinsic::sin, tys), args) )
            );
          }
          break;
        }
        default:
          assert(inst);
          llvm::errs() << "cannot handle unknown intrinsic\n" << *inst;
          report_fatal_error("unknown intrinsic");
      }

      if (dif0 || dif1) setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif0) addToDiffe(op->getOperand(0), dif0);
      if (dif1) addToDiffe(op->getOperand(1), dif1);
    } else if(auto op = dyn_cast_or_null<CallInst>(inst)) {

        Function *called = op->getCalledFunction();
        
        if (auto castinst = dyn_cast<ConstantExpr>(op->getCalledValue())) {
            if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                if (fn->getName() == "malloc" || fn->getName() == "free") {
                    called = fn;
                }
            }
        }

        if(called) {
            if (called->getName() == "printf" || called->getName() == "puts") {
                SmallVector<Value*, 4> args;
                for(unsigned i=0; i<op->getNumArgOperands(); i++) {
                    args.push_back(lookup(op->getArgOperand(i)));
                }
                auto cal = Builder2.CreateCall(called, args);
                cal->setAttributes(op->getAttributes());
            } else if(called->getName()=="malloc") {
              if (true) {
                 auto ci = CallInst::CreateFree(Builder2.CreatePointerCast(lookup(inst), Type::getInt8PtrTy(Context)), Builder2.GetInsertBlock());
                 if (ci->getParent()==nullptr) {
                   Builder2.Insert(ci);
                 }
              }

            } else if(called->getName()=="free") {
                llvm::Value* val = op->getArgOperand(0);
                while(auto cast = dyn_cast<CastInst>(val)) val = cast->getOperand(0);
                if (auto dc = dyn_cast<CallInst>(val)) {
                    if (dc->getCalledFunction()->getName() == "malloc") {
                        op->eraseFromParent();
                        continue;
                    }
                }
                assert(0 && "free not freeing a malloc");
                //TODO HANDLE FREE
                //
            } else if (!op->getCalledFunction()->empty()) {
                if (isconstant(op))
			        continue;
              SmallSet<unsigned,4> subconstant_args;

              SmallVector<Value*, 8> args;
              SmallVector<DIFFE_TYPE, 8> argsInverted;
              bool modifyPrimal = false;

              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (gutils->isConstantValue(op->getArgOperand(i))) {
                    subconstant_args.insert(i);
                    args.push_back(lookup(op->getArgOperand(i)));
                    argsInverted.push_back(DIFFE_TYPE::CONSTANT);
                    continue;
                }

                args.push_back(lookup(op->getArgOperand(i)));

				auto argType = op->getArgOperand(i)->getType();

				if (argType->isPointerTy() || argType->isIntegerTy()) {
					argsInverted.push_back(DIFFE_TYPE::DUP_ARG);
					args.push_back(invertPointer(op->getArgOperand(i)));

                    //TODO this check should consider whether this pointer has allocation/etc modifications and so on
                    if (! ( called->hasParamAttribute(i, Attribute::ReadOnly) || called->hasParamAttribute(i, Attribute::ReadNone)) ) {
					    modifyPrimal = true;
                    }

					//Note sometimes whattype mistakenly says something should be constant [because composed of integer pointers alone]
					assert(whatType(argType) == DIFFE_TYPE::DUP_ARG || whatType(argType) == DIFFE_TYPE::CONSTANT);
				} else {
					argsInverted.push_back(DIFFE_TYPE::OUT_DIFF);
					assert(whatType(argType) == DIFFE_TYPE::OUT_DIFF || whatType(argType) == DIFFE_TYPE::CONSTANT);
				}
              }
              if (subconstant_args.size() == args.size()) break;

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

              //TODO create augmented primal
              if (modifyPrimal) {
                called->dump();
                assert(0 && "need to modify primal of function but this isn't yet implemented");
                report_fatal_error("need to modify primal of function but this isn't yet implemented");
              }

              auto newcalled = CreatePrimalAndGradient(dyn_cast<Function>(called), subconstant_args, TLI, retUsed, !gutils->isConstantValue(inst));//, LI, DT);

              if (!gutils->isConstantValue(inst)) {
                args.push_back(diffe(inst));
              }

              auto diffes = Builder2.CreateCall(newcalled, args);
              diffes->setCallingConv(op->getCallingConv());
              diffes->setDebugLoc(inst->getDebugLoc());
              unsigned structidx = retUsed ? 1 : 0;

              for(unsigned i=0;i<called->getFunctionType()->getNumParams(); i++) {
                if (argsInverted[i] == DIFFE_TYPE::OUT_DIFF) {
                  unsigned idxs[] = {structidx};
                  Value* diffeadd = Builder2.CreateExtractValue(diffes, idxs);
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

              //TODO this shouldn't matter because this can't use itself, but setting null should be done before other sets but after load of diffe
			  if (inst->getNumUses() != 0 && !gutils->isConstantValue(inst))
              	setDiffe(inst, Constant::getNullValue(inst->getType()));
            } else {
             if (isconstant(op))
			    continue;
              assert(op);
              llvm::errs() << "cannot handle non invertible function\n" << *op << "\n";
              report_fatal_error("unknown noninvertible function");
            }
        } else {
            if (isconstant(op))
			    continue;
            assert(op);
            llvm::errs() << "cannot handle non const function in" << *op << "\n";
            report_fatal_error("unknown non constant function");
        }

    } else if(auto op = dyn_cast_or_null<SelectInst>(inst)) {
      if (isconstant(inst)) continue;

      Value* dif1 = nullptr;
      Value* dif2 = nullptr;

      if (!gutils->isConstantValue(op->getOperand(1)))
        dif1 = Builder2.CreateSelect(lookup(op->getOperand(0)), diffe(inst), Constant::getNullValue(op->getOperand(1)->getType()), "diffe"+op->getOperand(1)->getName());
      if (!gutils->isConstantValue(op->getOperand(2)))
        dif2 = Builder2.CreateSelect(lookup(op->getOperand(0)), Constant::getNullValue(op->getOperand(2)->getType()), diffe(inst), "diffe"+op->getOperand(2)->getName());

      setDiffe(inst, Constant::getNullValue(inst->getType()));
      if (dif1) addToDiffe(op->getOperand(1), dif1);
      if (dif2) addToDiffe(op->getOperand(2), dif2);
    } else if(auto op = dyn_cast<LoadInst>(inst)) {
      if (isconstant(inst)) continue;

       //TODO IF OP IS POINTER
      if (!op->getType()->isPointerTy()) {
        auto prediff = diffe(inst);
        setDiffe(inst, Constant::getNullValue(inst->getType()));
        addToPtrDiffe(op->getOperand(0), prediff);
      } else {
        //Builder2.CreateStore(diffe(inst), invertPointer(op->getOperand(0)));//, op->getName()+"'psweird");
        //addToNPtrDiffe(op->getOperand(0), diffe(inst));
        //assert(0 && "cannot handle non const pointer load inversion");
        assert(op);
		llvm::errs() << "ignoring load bc pointer of " << *op << "\n";
      }
    } else if(auto op = dyn_cast<StoreInst>(inst)) {
      if (isconstant(inst)) continue;

      //TODO const
       //TODO IF OP IS POINTER
      if (!op->getValueOperand()->getType()->isPointerTy()) {
		  if (!gutils->isConstantValue(op->getValueOperand())) {
			auto dif1 = Builder2.CreateLoad(invertPointer(op->getPointerOperand()));
			addToDiffe(op->getValueOperand(), dif1);
		  }
		  //setPtrDiffe(op->getPointerOperand(), Constant::getNullValue(op->getValueOperand()->getType()));
	  } else {
        IRBuilder <> storeBuilder(op);
        storeBuilder.CreateStore(gutils->invertPointerM(op->getValueOperand(),storeBuilder), gutils->invertPointerM(op->getPointerOperand(), storeBuilder) );
		//llvm::errs() << "ignoring store bc pointer of " << *op << "\n";
	  }
      //?necessary if pointer is readwrite
      /*
      IRBuilder<> BuilderZ(inst);
      Builder2.CreateStore(
        lookup(BuilderZ.CreateLoad(op->getPointerOperand())), lookup(op->getPointerOperand()));
      */
    } else if(auto op = dyn_cast<ExtractValueInst>(inst)) {
      if (isconstant(inst)) continue;
     
      auto prediff = diffe(inst);
      //todo const
      if (!gutils->isConstantValue(op->getOperand(0))) {
		SmallVector<Value*,4> sv;
      	for(auto i : op->getIndices())
        	sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), i));
        addToDiffeIndexed(op->getOperand(0), prediff, sv);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if (auto op = dyn_cast<ShuffleVectorInst>(inst)) {
      if (isconstant(inst)) continue;

      auto loaded = diffe(inst);
      size_t l1 = cast<VectorType>(op->getOperand(0)->getType())->getNumElements();
      uint64_t instidx = 0;
      for( size_t idx : op->getShuffleMask()) {
        auto opnum = (idx < l1) ? 0 : 1;
        auto opidx = (idx < l1) ? idx : (idx-l1);
        SmallVector<Value*,4> sv;
        sv.push_back(ConstantInt::get(Type::getInt32Ty(Context), opidx));
		if (!gutils->isConstantValue(op->getOperand(opnum)))
          addToDiffeIndexed(op->getOperand(opnum), Builder2.CreateExtractElement(loaded, instidx), sv);
        instidx++;
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<ExtractElementInst>(inst)) {
      if (isconstant(inst)) continue;

	  if (!gutils->isConstantValue(op->getVectorOperand())) {
        SmallVector<Value*,4> sv;
        sv.push_back(op->getIndexOperand());
        addToDiffeIndexed(op->getVectorOperand(), diffe(inst), sv);
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<InsertElementInst>(inst)) {
      if (isconstant(inst)) continue;

      auto dif1 = diffe(inst);

      if (!gutils->isConstantValue(op->getOperand(0)))
        addToDiffe(op->getOperand(0), Builder2.CreateInsertElement(dif1, Constant::getNullValue(op->getOperand(1)->getType()), lookup(op->getOperand(2)) ));

      if (!gutils->isConstantValue(op->getOperand(1)))
        addToDiffe(op->getOperand(1), Builder2.CreateExtractElement(dif1, lookup(op->getOperand(2))));

      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(auto op = dyn_cast<CastInst>(inst)) {
      if (isconstant(inst)) continue;

	  if (!gutils->isConstantValue(op->getOperand(0))) {
        if (op->getOpcode()==CastInst::CastOps::FPTrunc || op->getOpcode()==CastInst::CastOps::FPExt) {
          addToDiffe(op->getOperand(0), Builder2.CreateFPCast(diffe(inst), op->getOperand(0)->getType()));
        }
      }
      setDiffe(inst, Constant::getNullValue(inst->getType()));
    } else if(isa<CmpInst>(inst) || isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<SwitchInst>(inst) || isa<AllocaInst>(inst) || isa<CastInst>(inst) || isa<GetElementPtrInst>(inst)) {
        continue;
    } else {
      assert(inst);
      assert(inst->getParent());
      assert(inst->getParent()->getParent());
      llvm::errs() << *inst->getParent()->getParent() << "\n" << *inst->getParent() << "\n";
      llvm::errs() << "cannot handle above inst " << *inst << "\n";
      report_fatal_error("unknown instruction");
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

    for (auto& I: gutils->newFunc->args()) {
      // the differential return input value should be ignored
      auto idx = gutils->newFunc->arg_end();
      idx--;
      if (differentialReturn && &I == idx) continue;
      if (!gutils->isConstantValue(&I) && whatType(I.getType()) == DIFFE_TYPE::OUT_DIFF ) {
        retargs.push_back(diffe((Value*)&I));
      }
    }

    Value* toret = UndefValue::get(gutils->newFunc->getReturnType());
    for(unsigned i=0; i<retargs.size(); i++) {
      unsigned idx[] = { i };
      toret = Builder2.CreateInsertValue(toret, retargs[i], idx);
    }
    Builder2.SetInsertPoint(Builder2.GetInsertBlock());
    Builder2.CreateRet(toret);
  } else if (preds.size() == 1) {
    for (auto I = BB->begin(), E = BB->end(); I != E; I++) {
        if(auto PN = dyn_cast<PHINode>(&*I)) {
            if (gutils->isConstantValue(PN)) continue;
            //TODO consider whether indeed we can skip differential phi pointers
            if (PN->getType()->isPointerTy()) continue;
            auto prediff = diffe(PN);
            setDiffe(PN, Constant::getNullValue(PN->getType()));
            if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[0]))) {
                addToDiffe(PN->getIncomingValueForBlock(preds[0]), prediff );
            }
        } else break;
    }

    Builder2.SetInsertPoint(Builder2.GetInsertBlock());
    Builder2.CreateBr(gutils->reverseBlocks[preds[0]]);

  } else if (preds.size() == 2) {
    IRBuilder <> pbuilder(&BB->front());
    pbuilder.setFastMathFlags(FastMathFlags::getFast());
    Value* phi = nullptr;

    if (inLoop && BB2 == gutils->reverseBlocks[loopContext.var->getParent()]) {
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
				for(unsigned i=0; i<preds.size(); i++) {
					auto s = branch->getSuccessor(i);
					assert(s == succs[i]);
					if (seen[s].size() == 1) {
						if ( (*seen[s].begin()) != i) {
							phi = Builder2.CreateNot(phi);
							break;
						} else {
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
            if (gutils->isConstantValue(PN)) continue; 
            auto prediff = diffe(PN);
            setDiffe(PN, Constant::getNullValue(PN->getType()));
            if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[0]))) {
                auto dif = Builder2.CreateSelect(phi, prediff, Constant::getNullValue(prediff->getType()));
                addToDiffe(PN->getIncomingValueForBlock(preds[0]), dif );
            }
            if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[1]))) {
                auto dif = Builder2.CreateSelect(phi, Constant::getNullValue(prediff->getType()), prediff);
                addToDiffe(PN->getIncomingValueForBlock(preds[1]), dif);
            }
        } else break;
    }
    auto f0 = cast<BasicBlock>(gutils->reverseBlocks[preds[0]]);
    auto f1 = cast<BasicBlock>(gutils->reverseBlocks[preds[1]]);
    Builder2.SetInsertPoint(Builder2.GetInsertBlock());
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
          if (gutils->isConstantValue(PN)) continue;

          // POINTER TYPE THINGS
          if (PN->getType()->isPointerTy()) continue;
          auto prediff = diffe(PN);
          setDiffe(PN, Constant::getNullValue(PN->getType()));
          for(unsigned i=0; i<preds.size(); i++) {
            if (!gutils->isConstantValue(PN->getIncomingValueForBlock(preds[i]))) {
                auto cond = Builder2.CreateICmpEQ(phi, ConstantInt::get(phi->getType(), i));
                auto dif = Builder2.CreateSelect(cond, prediff, Constant::getNullValue(prediff->getType()));
                addToDiffe(PN->getIncomingValueForBlock(preds[i]), dif);
            }
          }
        } else break;
    }

    Builder2.SetInsertPoint(Builder2.GetInsertBlock());
    auto swit = Builder2.CreateSwitch(phi, gutils->reverseBlocks[preds.back()], preds.size()-1);
    for(unsigned i=0; i<preds.size()-1; i++) {
      swit->addCase(ConstantInt::get(cast<IntegerType>(phi->getType()), i), gutils->reverseBlocks[preds[i]]);
    }
  }


  }

  for(auto ci:gutils->addedFrees) {
    ci->moveBefore(ci->getParent()->getTerminator());
  }

  while(gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetimeOrAlloca());
  }

  DeleteDeadBlock(gutils->inversionAllocs);
  for(auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second) == pred_end(BBs.second))
        DeleteDeadBlock(BBs.second);
  }

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    gutils->newFunc->dump();
    report_fatal_error("function failed verification");
  }
  auto nf = gutils->newFunc;
  delete gutils;
  return nf;
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
  auto FT = cast<Function>(fn)->getFunctionType();
  assert(fn);
  
  if (autodiff_print)
      llvm::errs() << "prefn:\n" << *fn << "\n";

  SmallSet<unsigned,4> constants;
  SmallVector<Value*,2> args;

  unsigned truei = 0;
  IRBuilder<> Builder(CI);

  for(unsigned i=1; i<CI->getNumArgOperands(); i++) {
    Value* res = CI->getArgOperand(i);

    auto PTy = FT->getParamType(truei);
    DIFFE_TYPE ty = DIFFE_TYPE::CONSTANT;

    if (auto av = dyn_cast<MetadataAsValue>(res)) {
        auto MS = cast<MDString>(av->getMetadata())->getString();
        if (MS == "diffe_dup") {
            ty = DIFFE_TYPE::DUP_ARG;
        } else if(MS == "diffe_out") {
            ty = DIFFE_TYPE::OUT_DIFF;
        } else if (MS == "diffe_const") {
            ty = DIFFE_TYPE::CONSTANT;
        } else {
            assert(0 && "illegal diffe metadata string");
        }
        i++;
        res = CI->getArgOperand(i);
    } else 
      ty = whatType(PTy);

    if (ty == DIFFE_TYPE::CONSTANT)
      constants.insert(truei);

    assert(truei < FT->getNumParams());
    if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(1) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
      if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
        llvm::errs() << "Cannot cast(1) __builtin_autodiff argument " << i << " " << *res << "|"<< *res->getType() << " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
        report_fatal_error("Illegal cast(1)");
      }
      res = Builder.CreateBitCast(res, PTy);
    }

    args.push_back(res);
    if (ty == DIFFE_TYPE::DUP_ARG) {
      i++;

      Value* res = CI->getArgOperand(i);
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
            if (auto PT = dyn_cast<PointerType>(PTy)) {
                if (ptr->getAddressSpace() != PT->getAddressSpace()) {
                    res = Builder.CreateAddrSpaceCast(res, PointerType::get(ptr->getElementType(), PT->getAddressSpace()));
                    assert(res);
                    assert(PTy);
                    assert(FT);
                    llvm::errs() << "Warning cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
                }
            }
        }
        if (!res->getType()->canLosslesslyBitCastTo(PTy)) {
          assert(res);
          assert(res->getType());
          assert(PTy);
          assert(FT);
          llvm::errs() << "Cannot cast(2) __builtin_autodiff argument " << i << " " << *res <<"|" << *res->getType()<< " to argument " << truei << " " << *PTy << "\n" << "orig: " << *FT << "\n";
          report_fatal_error("Illegal cast(2)");
        }
        res = Builder.CreateBitCast(res, PTy);
      }
      args.push_back(res);
    }

    truei++;
  }

  bool differentialReturn = cast<Function>(fn)->getReturnType()->isFPOrFPVectorTy();
  auto newFunc = CreatePrimalAndGradient(cast<Function>(fn), constants, TLI, /*should return*/false, differentialReturn);//, LI, DT);
  llvm::errs() << "return type: " << *cast<Function>(fn)->getReturnType() << "\n";
  llvm::errs() << "newFunc type: " << *cast<Function>(fn)->getFunctionType() << "\n";
  
  if (differentialReturn)
    args.push_back(ConstantFP::get(cast<Function>(fn)->getReturnType(), 1.0));
  assert(newFunc);
  if (autodiff_print)
    llvm::errs() << "postfn:\n" << *newFunc << "\n";
  Builder.setFastMathFlags(FastMathFlags::getFast());

  CallInst* diffret = cast<CallInst>(Builder.CreateCall(newFunc, args));
  diffret->setCallingConv(CI->getCallingConv());
  diffret->setDebugLoc(CI->getDebugLoc());
  if (cast<StructType>(diffret->getType())->getNumElements()>0) {
    unsigned idxs[] = {0};
    auto diffreti = Builder.CreateExtractValue(diffret, idxs);
    CI->replaceAllUsesWith(diffreti);
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
    AU.addRequiredID(LoopSimplifyID);
    AU.addRequiredID(LCSSAID);
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

INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(LowerAutodiffIntrinsic, "lower-autodiff",
                "Lower 'autodiff' Intrinsics", false, false)

FunctionPass *llvm::createLowerAutodiffIntrinsicPass() {
  return new LowerAutodiffIntrinsic();
}
