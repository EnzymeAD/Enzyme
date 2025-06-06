//===- PreserveNVVM.cpp - Mark NVVM attributes for preservation.  -------===//
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
// This file contains createPreserveNVVM, a transformation pass that marks
// calls to __nv_* functions, marking them as noinline as implementing the llvm
// intrinsic.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/TimeProfiler.h"

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"

#include <map>

#include "PreserveNVVM.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "preserve-nvvm"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif

//! Returns whether changed.
bool preserveLinkage(bool Begin, Function &F, bool Inlining = true) {
  if (Begin && !F.hasFnAttribute("prev_fixup")) {
    F.addFnAttr("prev_fixup");
    if (F.hasFnAttribute(Attribute::AlwaysInline))
      F.addFnAttr("prev_always_inline");
    if (F.hasFnAttribute(Attribute::NoInline))
      F.addFnAttr("prev_no_inline");
    if (Inlining) {
      F.removeFnAttr(Attribute::AlwaysInline);
      F.addFnAttr(Attribute::NoInline);
    }
    F.addFnAttr("prev_linkage", std::to_string(F.getLinkage()));
    F.setLinkage(Function::LinkageTypes::ExternalLinkage);
    return true;
  }
  return false;
}

template <const char *handlername, DerivativeMode Mode, int numargs>
static void
handleCustomDerivative(llvm::Module &M, llvm::GlobalVariable &g,
                       SmallVectorImpl<GlobalVariable *> &globalsToErase) {
  if (g.hasInitializer()) {
    if (auto CA = dyn_cast<ConstantAggregate>(g.getInitializer())) {
      if (CA->getNumOperands() < numargs) {
        llvm::errs() << M << "\n";
        llvm::errs() << "Use of " << handlername
                     << " must be a "
                        "constant of size at least "
                     << numargs << " " << g << "\n";
        llvm_unreachable(handlername);
      } else {
        Function *Fs[numargs];
        for (size_t i = 0; i < numargs; i++) {
          Value *V = CA->getOperand(i);
          while (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V))
            V = CA->getOperand(0);
          while (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
          }
          if (auto F = dyn_cast<Function>(V)) {
            Fs[i] = F;
          } else {
            llvm::errs() << M << "\n";
            llvm::errs() << "Param of " << handlername
                         << " must be a "
                            "function"
                         << g << "\n"
                         << *V << "\n";
            llvm_unreachable(handlername);
          }
        }

        SmallSet<size_t, 1> byref;

        if constexpr (Mode == DerivativeMode::ReverseModeGradient) {
          assert(numargs >= 3);
          for (size_t i = numargs; i < CA->getNumOperands(); i++) {
            Value *V = CA->getOperand(i);
            while (auto CE = dyn_cast<ConstantExpr>(V)) {
              V = CE->getOperand(0);
            }
            if (auto CA = dyn_cast<ConstantAggregate>(V))
              V = CA->getOperand(0);
            while (auto CE = dyn_cast<ConstantExpr>(V)) {
              V = CE->getOperand(0);
            }
            if (auto GV = dyn_cast<GlobalVariable>(V)) {
              if (GV->isConstant())
                if (auto C = GV->getInitializer())
                  if (auto CA = dyn_cast<ConstantDataArray>(C))
                    if (CA->getType()->getElementType()->isIntegerTy(8) &&
                        CA->isCString()) {

                      auto str = CA->getAsCString();
                      bool legal = startsWith(str, "byref_");
                      size_t argnum = 0;
                      if (legal) {
                        for (size_t i = str.size() - 1, len = strlen("byref_");
                             i >= len; i--) {
                          char c = str[i];
                          if (c < '0' || c > '9') {
                            legal = false;
                            break;
                          }
                          argnum *= 10;
                          argnum += c - '0';
                        }
                      }
                      if (legal) {
                        byref.insert(argnum);
                        continue;
                      }
                    }
            }
            llvm::errs() << M << "\n";
            llvm::errs() << "Use of " << handlername
                         << " possible post args include 'byref_ret'"
                         << "\n";
            llvm_unreachable(handlername);
          }

          if (byref.size())
            for (size_t fn = 1; fn <= 2; fn++) {
              Function *F = Fs[fn];
              bool need = false;
              size_t nonSRetSize = 0;
              for (size_t i = 0; i < F->arg_size(); i++)
                if (!F->hasParamAttribute(i, Attribute::StructRet))
                  nonSRetSize++;
              for (auto r : byref)
                if (r < nonSRetSize)
                  need = true;
              if (!need)
                continue;

              SmallVector<Type *, 3> args;
              Type *sretTy = nullptr;
              size_t realidx = 0;
              size_t i = 0;
              for (auto &arg : F->args()) {
                if (!F->hasParamAttribute(i, Attribute::StructRet)) {
                  if (!byref.count(realidx))
                    args.push_back(arg.getType());
                  else {
                    // TODO in opaque pointers
                    Type *subTy = nullptr;
#if LLVM_VERSION_MAJOR < 17
                    subTy = arg.getType()->getPointerElementType();
#endif
                    assert(subTy);
                    args.push_back(subTy);
                  }
                  realidx++;
                } else {
                  llvm::Type *T = nullptr;
#if LLVM_VERSION_MAJOR > 12
                  T = F->getParamAttribute(i, Attribute::StructRet)
                          .getValueAsType();
#else
                  T = arg.getType()->getPointerElementType();
#endif
                  sretTy = T;
                }
                i++;
              }
              Type *RT = F->getReturnType();
              if (sretTy) {
                assert(RT->isVoidTy());
                RT = sretTy;
              }
              FunctionType *FTy =
                  FunctionType::get(RT, args, F->getFunctionType()->isVarArg());
              Function *NewF =
                  Function::Create(FTy, Function::LinkageTypes::InternalLinkage,
                                   "fixbyval_" + F->getName(), F->getParent());

              AllocaInst *AI = nullptr;
              BasicBlock *BB =
                  BasicBlock::Create(NewF->getContext(), "entry", NewF);
              IRBuilder<> bb(BB);
              if (sretTy)
                AI = bb.CreateAlloca(sretTy);
              SmallVector<Value *, 3> argVs;
              auto arg = NewF->arg_begin();
              realidx = 0;
              for (size_t i = 0; i < F->arg_size(); i++) {
                if (!F->hasParamAttribute(i, Attribute::StructRet)) {
                  arg->setName("arg" + Twine(realidx));
                  if (!byref.count(realidx))
                    argVs.push_back(arg);
                  else {
                    auto A = bb.CreateAlloca(arg->getType());
                    bb.CreateStore(arg, A);
                    argVs.push_back(A);
                  }
                  realidx++;
                  ++arg;
                } else {
                  argVs.push_back(AI);
                }
              }
              auto cal = bb.CreateCall(F, argVs);
              cal->setCallingConv(F->getCallingConv());

              if (sretTy) {
                Value *res = bb.CreateLoad(sretTy, AI);
                bb.CreateRet(res);
              } else if (!RT->isVoidTy()) {
                bb.CreateRet(cal);
              } else
                bb.CreateRetVoid();

              Fs[fn] = NewF;
            }

          preserveLinkage(true, *Fs[1], false);
          Fs[0]->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
          preserveLinkage(true, *Fs[2], false);
          Fs[0]->setMetadata(
              "enzyme_gradient",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[2])}));
        } else if (Mode == DerivativeMode::ForwardMode) {
          assert(numargs == 2);
          preserveLinkage(true, *Fs[1], false);
          Fs[0]->setMetadata(
              "enzyme_derivative",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
        } else if (Mode == DerivativeMode::ForwardModeSplit) {
          assert(numargs == 3);
          preserveLinkage(true, *Fs[1], false);
          Fs[0]->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[1])}));
          preserveLinkage(true, *Fs[2], false);
          Fs[0]->setMetadata(
              "enzyme_splitderivative",
              llvm::MDTuple::get(Fs[0]->getContext(),
                                 {llvm::ValueAsMetadata::get(Fs[2])}));
        } else
          assert("Unknown mode");
      }
    } else if (isTargetNVPTX(M)) {
      llvm::errs() << M << "\n";
      llvm::errs() << "Use of " << handlername
                   << " must be a "
                      "constant aggregate "
                   << g << "\n";
      llvm_unreachable(handlername);
    }
  } else {
    llvm::errs() << M << "\n";
    llvm::errs() << "Use of " << handlername
                 << " must be a "
                    "constant array of size "
                 << numargs << " " << g << "\n";
    llvm_unreachable(handlername);
  }
  globalsToErase.push_back(&g);
}

bool preserveNVVM(bool Begin, Module &M) {
  bool changed = false;
  constexpr static const char gradient_handler_name[] =
      "__enzyme_register_gradient";
  constexpr static const char derivative_handler_name[] =
      "__enzyme_register_derivative";
  constexpr static const char splitderivative_handler_name[] =
      "__enzyme_register_splitderivative";

  if (Begin)
    if (GlobalVariable *GA = M.getGlobalVariable("llvm.global.annotations")) {
      if (GA->hasInitializer()) {
        auto AOp = GA->getInitializer();
        // all metadata are stored in an array of struct of metadata
        if (ConstantArray *CA = dyn_cast<ConstantArray>(AOp)) {
          // so iterate over the operands
          SmallVector<Constant *, 1> replacements;
          for (Value *CAOp : CA->operands()) {
            // get the struct, which holds a pointer to the annotated function
            // as first field, and the annotation as second field
            ConstantStruct *CS = dyn_cast<ConstantStruct>(CAOp);
            if (!CS)
              continue;

            if (CS->getNumOperands() < 2)
              continue;

            // the second field is a pointer to a global constant Array that
            // holds the string
            GlobalVariable *GAnn =
                dyn_cast<GlobalVariable>(CS->getOperand(1)->getOperand(0));

            ConstantDataArray *A = nullptr;

            if (GAnn)
              A = dyn_cast<ConstantDataArray>(GAnn->getOperand(0));
            else
              A = dyn_cast<ConstantDataArray>(CS->getOperand(1)->getOperand(0));

            if (!A)
              continue;

            // we have the annotation! Check it's an epona annotation
            // and process
            StringRef AS = A->getAsCString();

            Constant *Val = cast<Constant>(CS->getOperand(0));
            while (auto CE = dyn_cast<ConstantExpr>(Val))
              Val = CE->getOperand(0);

            Function *Func = dyn_cast<Function>(Val);
            GlobalVariable *Glob = dyn_cast<GlobalVariable>(Val);

            if (AS == "enzyme_inactive" && Func) {
              Func->addAttribute(
                  AttributeList::FunctionIndex,
                  Attribute::get(Func->getContext(), "enzyme_inactive"));
              changed = true;
              preserveLinkage(Begin, *Func);
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }

            if (AS == "enzyme_shouldrecompute" && Func) {
              Func->addAttribute(
                  AttributeList::FunctionIndex,
                  Attribute::get(Func->getContext(), "enzyme_shouldrecompute"));
              changed = true;
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }

            if (AS == "enzyme_inactive" && Glob) {
              Glob->setMetadata("enzyme_inactive",
                                MDNode::get(Glob->getContext(), {}));
              changed = true;
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }

            if (AS == "enzyme_nofree" && Func) {
              Func->addAttribute(
                  AttributeList::FunctionIndex,
                  Attribute::get(Func->getContext(), Attribute::NoFree));
              changed = true;
              preserveLinkage(Begin, *Func);
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }

            if (startsWith(AS, "enzyme_function_like") && Func) {
              auto val = AS.substr(1 + AS.find('='));
              Func->addAttribute(
                  AttributeList::FunctionIndex,
                  Attribute::get(Func->getContext(), "enzyme_math", val));
              changed = true;
              preserveLinkage(Begin, *Func);
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }

            if (AS == "enzyme_sparse_accumulate" && Func) {
              Func->addAttribute(AttributeList::FunctionIndex,
                                 Attribute::get(Func->getContext(),
                                                "enzyme_sparse_accumulate"));
              changed = true;
              preserveLinkage(Begin, *Func);
              replacements.push_back(Constant::getNullValue(CAOp->getType()));
              continue;
            }
            replacements.push_back(cast<Constant>(CAOp));
          }
          GA->setInitializer(ConstantArray::get(CA->getType(), replacements));
        }
      }
    }

  for (GlobalVariable &g : M.globals()) {
    if (g.getName().contains(gradient_handler_name) ||
        g.getName().contains(derivative_handler_name) ||
        g.getName().contains(splitderivative_handler_name) ||
        g.getName().contains("__enzyme_nofree") ||
        g.getName().contains("__enzyme_inactivefn") ||
        g.getName().contains("__enzyme_sparse_accumulate") ||
        g.getName().contains("__enzyme_function_like") ||
        g.getName().contains("__enzyme_allocation_like")) {
      if (g.hasInitializer()) {
        Value *V = g.getInitializer();
        while (1) {
          if (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
            continue;
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V)) {
            V = CA->getOperand(0);
            continue;
          }
          break;
        }
        if (auto F = dyn_cast<Function>(V))
          changed |= preserveLinkage(Begin, *F);
      }
    }
  }
  SmallVector<GlobalVariable *, 1> toErase;
  for (GlobalVariable &g : M.globals()) {
    if (g.getName().contains(gradient_handler_name)) {
      handleCustomDerivative<gradient_handler_name,
                             DerivativeMode::ReverseModeGradient, 3>(M, g,
                                                                     toErase);
      changed = true;
    } else if (g.getName().contains(derivative_handler_name)) {
      handleCustomDerivative<derivative_handler_name,
                             DerivativeMode::ForwardMode, 2>(M, g, toErase);
      changed = true;
    } else if (g.getName().contains(splitderivative_handler_name)) {
      handleCustomDerivative<splitderivative_handler_name,
                             DerivativeMode::ForwardModeSplit, 3>(M, g,
                                                                  toErase);
      changed = true;
    }
    if (g.getName().contains("__enzyme_inactive_global")) {
      if (g.hasInitializer()) {
        Value *V = g.getInitializer();
        while (1) {
          if (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
            continue;
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V)) {
            V = CA->getOperand(0);
            continue;
          }
          break;
        }
        if (auto GV = cast<GlobalVariable>(V)) {
          GV->setMetadata("enzyme_inactive", MDNode::get(g.getContext(), {}));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __enzyme_inactive_global must be a "
                          "global variable"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_inactive_global");
        }
      }
    }
    if (g.getName().contains("__enzyme_inactivefn")) {
      if (g.hasInitializer()) {
        Value *V = g.getInitializer();
        while (1) {
          if (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
            continue;
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V)) {
            V = CA->getOperand(0);
            continue;
          }
          break;
        }
        if (auto F = cast<Function>(V)) {
          F->addAttribute(AttributeList::FunctionIndex,
                          Attribute::get(g.getContext(), "enzyme_inactive"));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __enzyme_inactivefn must be a "
                          "constant function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_inactivefn");
        }
      }
    }
    if (g.getName().contains("__enzyme_sparse_accumulate")) {
      if (g.hasInitializer()) {
        Value *V = g.getInitializer();
        while (1) {
          if (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
            continue;
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V)) {
            V = CA->getOperand(0);
            continue;
          }
          break;
        }
        if (auto F = cast<Function>(V)) {
          F->addAttribute(
              AttributeList::FunctionIndex,
              Attribute::get(g.getContext(), "enzyme_sparse_accumulate"));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __enzyme_sparse_accumulate must be a "
                          "constant function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_sparse_accumulate");
        }
      }
    }
    if (g.getName().contains("__enzyme_nofree")) {
      if (g.hasInitializer()) {
        Value *V = g.getInitializer();
        while (1) {
          if (auto CE = dyn_cast<ConstantExpr>(V)) {
            V = CE->getOperand(0);
            continue;
          }
          if (auto CA = dyn_cast<ConstantAggregate>(V)) {
            V = CA->getOperand(0);
            continue;
          }
          break;
        }
        if (auto F = cast<Function>(V)) {
          F->addAttribute(AttributeList::FunctionIndex,
                          Attribute::get(g.getContext(), Attribute::NoFree));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __enzyme_nofree must be a "
                          "constant function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_nofree");
        }
      }
    }
    if (g.getName().contains("__enzyme_function_like")) {
      if (g.hasInitializer()) {
        auto CA = dyn_cast<ConstantAggregate>(g.getInitializer());
        if (!CA || CA->getNumOperands() < 2) {
          llvm::errs() << "Use of "
                       << "enzyme_function_like"
                       << " must be a "
                          "constant of size at least "
                       << 2 << " " << g << "\n";
          llvm_unreachable("enzyme_function_like");
        }
        Value *V = CA->getOperand(0);
        Value *name = CA->getOperand(1);
        while (auto CE = dyn_cast<ConstantExpr>(V)) {
          V = CE->getOperand(0);
        }
        while (auto CE = dyn_cast<ConstantExpr>(name)) {
          name = CE->getOperand(0);
        }
        StringRef nameVal;
        if (auto GV = dyn_cast<GlobalVariable>(name))
          if (GV->isConstant())
            if (auto C = GV->getInitializer())
              if (auto CA = dyn_cast<ConstantDataArray>(C))
                if (CA->getType()->getElementType()->isIntegerTy(8) &&
                    CA->isCString())
                  nameVal = CA->getAsCString();

        if (nameVal == "") {
          llvm::errs() << *name << "\n";
          llvm::errs() << "Use of "
                       << "enzyme_function_like"
                       << "requires a non-empty function name"
                       << "\n";
          llvm_unreachable("enzyme_function_like");
        }
        if (auto F = cast<Function>(V)) {
          F->addAttribute(
              AttributeList::FunctionIndex,
              Attribute::get(g.getContext(), "enzyme_math", nameVal));
          toErase.push_back(&g);
          changed = true;
        } else {
          llvm::errs() << "Param of __enzyme_function_like must be a "
                          "constant function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_function_like");
        }
      }
    }
    if (g.getName().contains("__enzyme_allocation_like")) {
      if (g.hasInitializer()) {
        auto CA = dyn_cast<ConstantAggregate>(g.getInitializer());
        if (!CA || CA->getNumOperands() != 4) {
          llvm::errs() << "Use of "
                       << "enzyme_allocation_like"
                       << " must be a "
                          "constant of size at least "
                       << 4 << " " << g << "\n";
          llvm_unreachable("enzyme_allocation_like");
        }
        Value *V = CA->getOperand(0);
        Value *name = CA->getOperand(1);
        while (auto CE = dyn_cast<ConstantExpr>(V)) {
          V = CE->getOperand(0);
        }
        while (auto CE = dyn_cast<ConstantExpr>(name)) {
          name = CE->getOperand(0);
        }
        Value *deallocind = CA->getOperand(2);
        while (auto CE = dyn_cast<ConstantExpr>(deallocind)) {
          deallocind = CE->getOperand(0);
        }
        Value *deallocfn = CA->getOperand(3);
        while (auto CE = dyn_cast<ConstantExpr>(deallocfn)) {
          deallocfn = CE->getOperand(0);
        }
        size_t index = 0;
        if (isa<ConstantPointerNull>(name)) {
          // An integer 0 may have been implicitly converted to a null pointer
          index = 0;
        } else if (auto CI = dyn_cast<ConstantInt>(name)) {
          index = CI->getZExtValue();
        } else {
          llvm::errs() << *name << "\n";
          llvm::errs() << "Use of "
                       << "enzyme_allocation_like"
                       << "requires an integer index"
                       << "\n";
          llvm_unreachable("enzyme_allocation_like");
        }

        StringRef deallocIndStr;
        bool foundInd = false;
        if (auto GV = dyn_cast<GlobalVariable>(deallocind))
          if (GV->isConstant())
            if (auto C = GV->getInitializer())
              if (auto CA = dyn_cast<ConstantDataArray>(C))
                if (CA->getType()->getElementType()->isIntegerTy(8) &&
                    CA->isCString()) {
                  deallocIndStr = CA->getAsCString();
                  foundInd = true;
                }

        if (!foundInd) {
          llvm::errs() << *deallocind << "\n";
          llvm::errs() << "Use of "
                       << "enzyme_allocation_like"
                       << "requires a deallocation index string"
                       << "\n";
          llvm_unreachable("enzyme_allocation_like");
        }
        if (auto F = dyn_cast<Function>(V)) {
          F->addAttribute(AttributeList::FunctionIndex,
                          Attribute::get(g.getContext(), "enzyme_allocator",
                                         std::to_string(index)));
        } else {
          llvm::errs() << "Param of __enzyme_allocation_like must be a "
                          "function"
                       << g << "\n"
                       << *V << "\n";
          llvm_unreachable("__enzyme_allocation_like");
        }
        cast<Function>(V)->addAttribute(AttributeList::FunctionIndex,
                                        Attribute::get(g.getContext(),
                                                       "enzyme_deallocator",
                                                       deallocIndStr));

        if (auto F = dyn_cast<Function>(deallocfn)) {
          cast<Function>(V)->setMetadata(
              "enzyme_deallocator_fn",
              llvm::MDTuple::get(F->getContext(),
                                 {llvm::ValueAsMetadata::get(F)}));
          changed |= preserveLinkage(Begin, *F);
        } else {
          llvm::errs() << "Free fn of __enzyme_allocation_like must be a "
                          "function"
                       << g << "\n"
                       << *deallocfn << "\n";
          llvm_unreachable("__enzyme_allocation_like");
        }
        toErase.push_back(&g);
        changed = true;
      }
    }
  }

  for (auto G : toErase) {
    for (auto name : {"llvm.used", "llvm.compiler.used"}) {
      if (auto V = M.getGlobalVariable(name)) {
        auto C = cast<ConstantArray>(V->getInitializer());
        SmallVector<Constant *, 1> toKeep;
        bool found = false;
        for (unsigned i = 0; i < C->getNumOperands(); i++) {
          Value *Op = C->getOperand(i)->stripPointerCasts();
          if (Op == G)
            found = true;
          else
            toKeep.push_back(C->getOperand(i));
        }
        if (found) {
          if (toKeep.size()) {
            auto CA = ConstantArray::get(
                ArrayType::get(C->getType()->getElementType(), toKeep.size()),
                toKeep);
            GlobalVariable *NGV = new GlobalVariable(
                CA->getType(), V->isConstant(), V->getLinkage(), CA, "",
                V->getThreadLocalMode());
#if LLVM_VERSION_MAJOR > 16
            V->getParent()->insertGlobalVariable(V->getIterator(), NGV);
#else
            V->getParent()->getGlobalList().insert(V->getIterator(), NGV);
#endif
            NGV->takeName(V);

            // Nuke the old list, replacing any uses with the new one.
            if (!V->use_empty()) {
              Constant *VV = NGV;
              if (VV->getType() != V->getType())
                VV = ConstantExpr::getBitCast(VV, V->getType());
              V->replaceAllUsesWith(VV);
            }
          }
          V->eraseFromParent();
        }
      }
    }
    changed = true;
    G->replaceAllUsesWith(ConstantPointerNull::get(G->getType()));
    G->eraseFromParent();
  }

  StringMap<std::pair<std::string, std::string>> Implements;
  for (std::string T : {"", "f"}) {
    // CUDA
    // sincos, sinpi, cospi, sincospi, cyl_bessel_i1
    for (std::string name :
         {"sin",        "cos",     "tan",       "log2",   "exp",    "exp2",
          "exp10",      "cosh",    "sinh",      "tanh",   "atan2",  "atan",
          "asin",       "acos",    "log",       "log10",  "log1p",  "acosh",
          "asinh",      "atanh",   "expm1",     "hypot",  "rhypot", "norm3d",
          "rnorm3d",    "norm4d",  "rnorm4d",   "norm",   "rnorm",  "cbrt",
          "rcbrt",      "j0",      "j1",        "y0",     "y1",     "yn",
          "jn",         "erf",     "erfinv",    "erfc",   "erfcx",  "erfcinv",
          "normcdfinv", "normcdf", "lgamma",    "ldexp",  "scalbn", "frexp",
          "modf",       "fmod",    "remainder", "remquo", "powi",   "tgamma",
          "round",      "fdim",    "ilogb",     "logb",   "isinf",  "pow",
          "sqrt",       "finite",  "fabs",      "fmax"}) {
      std::string nvname = "__nv_" + name;
      std::string llname = "llvm." + name + ".";
      std::string mathname = name;

      if (T == "f") {
        mathname += "f";
        nvname += "f";
        llname += "f32";
      } else {
        llname += "f64";
      }

      Implements[nvname] = std::make_pair(mathname, llname);
    }
    // ROCM
    // sincos, sinpi, cospi, sincospi, cyl_bessel_i1
    for (std::string name : {"acos",         "acosh",        "asin",
                             "asinh",        "atan2",        "atan",
                             "atanh",        "cbrt",         "ceil",
                             "copysign",     "cos",          "native_cos",
                             "cosh",         "cospi",        "i0",
                             "i1",           "erfc",         "erfcinv",
                             "erfcx",        "erf",          "erfinv",
                             "exp10",        "native_exp10", "exp2",
                             "exp",          "native_exp",   "expm1",
                             "fabs",         "fdim",         "floor",
                             "fma",          "fmax",         "fmin",
                             "fmod",         "frexp",        "hypot",
                             "ilogb",        "isfinite",     "isinf",
                             "isnan",        "j0",           "j1",
                             "ldexp",        "lgamma",       "log10",
                             "native_log10", "log1p",        "log2",
                             "log2",         "logb",         "log",
                             "native_log",   "modf",         "nearbyint",
                             "nextafter",    "len3",         "len4",
                             "ncdf",         "ncdfinv",      "pow",
                             "pown",         "rcbrt",        "remainder",
                             "remquo",       "rhypot",       "rint",
                             "rlen3",        "rlen4",        "round",
                             "rsqrt",        "scalb",        "scalbn",
                             "signbit",      "sincos",       "sincospi",
                             "sin",          "native_sin",   "sinh",
                             "sinpi",        "sqrt",         "native_sqrt",
                             "tan",          "tanh",         "tgamma",
                             "trunc",        "y0",           "y1"}) {
      std::string nvname = "__ocml_" + name + "_";
      std::string llname = "llvm." + name + ".";
      std::string mathname = name;

      if (T == "f") {
        mathname += "f";
        nvname += "f32";
        llname += "f32";
      } else {
        nvname += "f64";
        llname += "f64";
      }

      Implements[nvname] = std::make_pair(mathname, llname);
    }
  }
  for (auto &F : llvm::make_early_inc_range(M)) {
    if (Begin) {
      changed |= attributeKnownFunctions(F);
    }
  }
  for (auto &F : M) {
    auto found = Implements.find(F.getName());
    if (found != Implements.end()) {
      changed = true;
      if (Begin) {
        // As a side effect, enforces arguments
        // cannot be erased.
        F.addFnAttr("implements", found->second.second);
        F.addFnAttr("implements2", found->second.first);
        F.addFnAttr("enzyme_math", found->second.first);
        changed |= preserveLinkage(Begin, F);
      }
    }
    if (!Begin && F.hasFnAttribute("prev_fixup")) {
      changed = true;
      F.removeFnAttr("prev_fixup");
      if (F.hasFnAttribute("prev_always_inline")) {
        F.addFnAttr(Attribute::AlwaysInline);
        F.removeFnAttr("prev_always_inline");
      }
      if (F.hasFnAttribute("prev_no_inline")) {
        F.removeFnAttr("prev_no_inline");
      } else {
        F.removeFnAttr(Attribute::NoInline);
      }
      int64_t val;
      F.getFnAttribute("prev_linkage").getValueAsString().getAsInteger(10, val);
      F.setLinkage((Function::LinkageTypes)val);
    }
  }
  return changed;
}

namespace {

class PreserveNVVM final : public ModulePass {
public:
  static char ID;
  bool Begin;
  PreserveNVVM(bool Begin = true) : ModulePass(ID), Begin(Begin) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnModule(Module &M) override { return preserveNVVM(Begin, M); }
};

class PreserveNVVMFn final : public FunctionPass {
public:
  static char ID;
  bool Begin;
  PreserveNVVMFn(bool Begin = true) : FunctionPass(ID), Begin(Begin) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnFunction(Function &F) override {
    return preserveNVVM(Begin, *F.getParent());
  }
};

} // namespace

char PreserveNVVM::ID = 0;

char PreserveNVVMFn::ID = 0;

static RegisterPass<PreserveNVVM> X("preserve-nvvm", "Preserve NVVM Pass");

static RegisterPass<PreserveNVVMFn> XFn("preserve-nvvm-fn",
                                        "Preserve NVVM Pass");

ModulePass *createPreserveNVVMPass(bool Begin) {
  return new PreserveNVVM(Begin);
}

FunctionPass *createPreserveNVVMFnPass(bool Begin) {
  return new PreserveNVVMFn(Begin);
}

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddPreserveNVVMPass(LLVMPassManagerRef PM, uint8_t Begin) {
  unwrap(PM)->add(createPreserveNVVMPass((bool)Begin));
}

PreserveNVVMNewPM::Result
PreserveNVVMNewPM::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  bool changed = preserveNVVM(Begin, M);
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey PreserveNVVMNewPM::Key;
