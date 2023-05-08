//===- Enzyme.cpp - Automatic Differentiation Transformation Pass  -------===//
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
// This file contains Enzyme, a transformation pass that takes replaces calls
// to function calls to *__enzyme_autodiff* with a call to the derivative of
// the function passed as the first argument.
//
//===----------------------------------------------------------------------===//
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"

#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"

#include "llvm/Passes/PassBuilder.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#if LLVM_VERSION_MAJOR >= 11
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/IR/AbstractCallSite.h"
#endif
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

#include "ActivityAnalysis.h"
#include "DiffeGradientUtils.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "TraceInterface.h"
#include "TraceUtils.h"
#include "Utils.h"

#include "InstructionBatcher.h"

#include "llvm/Transforms/Utils.h"

#if LLVM_VERSION_MAJOR >= 13
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"
#endif

#include "CApi.h"
using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-enzyme-intrinsic"

llvm::cl::opt<bool>
    EnzymePostOpt("enzyme-postopt", cl::init(false), cl::Hidden,
                  cl::desc("Run enzymepostprocessing optimizations"));

llvm::cl::opt<bool> EnzymeAttributor("enzyme-attributor", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Run attributor post Enzyme"));

llvm::cl::opt<bool> EnzymeOMPOpt("enzyme-omp-opt", cl::init(false), cl::Hidden,
                                 cl::desc("Whether to enable openmp opt"));

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#endif
void attributeKnownFunctions(llvm::Function &F) {
  if (F.getName().contains("__enzyme_todense"))
    F.addFnAttr(Attribute::ReadNone);

  if (F.getName().contains("__enzyme_float") ||
      F.getName().contains("__enzyme_double") ||
      F.getName().contains("__enzyme_integer") ||
      F.getName().contains("__enzyme_pointer") ||
      F.getName().contains("__enzyme_virtualreverse")) {
    F.addFnAttr(Attribute::ReadNone);
    for (auto &arg : F.args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(Attribute::ReadNone);
        arg.addAttr(Attribute::NoCapture);
      }
    }
  }
  if (F.getName() == "memcmp") {
    F.addFnAttr(Attribute::ReadOnly);
    F.addFnAttr(Attribute::ArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#else
    F.addFnAttr("nofree");
#endif
    for (int i = 0; i < 2; i++)
      if (F.getFunctionType()->getParamType(i)->isPointerTy()) {
        F.addParamAttr(i, Attribute::NoCapture);
        F.addParamAttr(i, Attribute::WriteOnly);
      }
  }
  if (F.getName() ==
      "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm") {
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::NoFree);
#else
    F.addFnAttr("nofree");
#endif
  }
  if (F.getName() == "MPI_Irecv" || F.getName() == "PMPI_Irecv") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::WriteOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::WriteOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Isend" || F.getName() == "PMPI_Isend") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::ReadOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::ReadOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Comm_rank" || F.getName() == "PMPI_Comm_rank" ||
      F.getName() == "MPI_Comm_size" || F.getName() == "PMPI_Comm_size") {
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    if (F.getFunctionType()->getParamType(0)->isPointerTy()) {
      F.addParamAttr(0, Attribute::NoCapture);
      F.addParamAttr(0, Attribute::ReadOnly);
    }
    if (F.getFunctionType()->getParamType(1)->isPointerTy()) {
      F.addParamAttr(1, Attribute::WriteOnly);
      F.addParamAttr(1, Attribute::NoCapture);
    }
  }
  if (F.getName() == "MPI_Wait" || F.getName() == "PMPI_Wait") {
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(0, Attribute::NoCapture);
    F.addParamAttr(1, Attribute::WriteOnly);
    F.addParamAttr(1, Attribute::NoCapture);
  }
  if (F.getName() == "MPI_Waitall" || F.getName() == "PMPI_Waitall") {
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
#if LLVM_VERSION_MAJOR >= 9
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
#endif
    F.addParamAttr(1, Attribute::NoCapture);
    F.addParamAttr(2, Attribute::WriteOnly);
    F.addParamAttr(2, Attribute::NoCapture);
  }
  if (F.getName() == "omp_get_max_threads" ||
      F.getName() == "omp_get_thread_num") {
    F.addFnAttr(Attribute::ReadOnly);
    F.addFnAttr(Attribute::InaccessibleMemOnly);
  }
  if (F.getName() == "frexp" || F.getName() == "frexpf" ||
      F.getName() == "frexpl") {
    F.addFnAttr(Attribute::ArgMemOnly);
    F.addParamAttr(1, Attribute::WriteOnly);
  }
  if (F.getName() == "__fd_sincos_1" || F.getName() == "__fd_cos_1" ||
      F.getName() == "__mth_i_ipowi") {
    F.addFnAttr(Attribute::ReadNone);
  }
}

namespace {
static Value *
castToDiffeFunctionArgType(IRBuilder<> &Builder, llvm::CallInst *CI,
                           llvm::FunctionType *FT, llvm::Type *destType,
                           unsigned int i, DerivativeMode mode,
                           llvm::Value *value, unsigned int truei) {
  auto res = value;
  if (auto ptr = dyn_cast<PointerType>(res->getType())) {
    if (auto PT = dyn_cast<PointerType>(destType)) {
      if (ptr->getAddressSpace() != PT->getAddressSpace()) {
        res = Builder.CreateAddrSpaceCast(
            res, PointerType::get(ptr->getPointerElementType(),
                                  PT->getAddressSpace()));
        assert(value);
        assert(destType);
        assert(FT);
        llvm::errs() << "Warning cast(2) __enzyme_autodiff argument " << i
                     << " " << *res << "|" << *res->getType() << " to argument "
                     << truei << " " << *destType << "\n"
                     << "orig: " << *FT << "\n";
        return res;
      }
    }
  }

  if (!res->getType()->canLosslesslyBitCastTo(destType)) {
    assert(value);
    assert(value->getType());
    assert(destType);
    assert(FT);
    auto loc = CI->getDebugLoc();
    if (auto arg = dyn_cast<Instruction>(res)) {
      loc = arg->getDebugLoc();
    }
    EmitFailure("IllegalArgCast", loc, CI,
                "Cannot cast __enzyme_autodiff shadow argument ", i, ", found ",
                *res, ", type ", *res->getType(), " - to arg ", truei, " ",
                *destType);
    return nullptr;
  }
  return Builder.CreateBitCast(value, destType);
}

static Optional<StringRef> getMetadataName(llvm::Value *res);

// if all phi arms are (recursively) based on the same metaString, use that
static Optional<StringRef> recursePhiReads(PHINode *val) {
  Optional<StringRef> finalMetadata;
  SmallVector<PHINode *, 1> todo = {val};
  SmallSet<PHINode *, 1> done;
  while (todo.size()) {
    auto phiInst = todo.back();
    todo.pop_back();
    if (done.count(phiInst))
      continue;
    done.insert(phiInst);
    for (unsigned j = 0; j < phiInst->getNumIncomingValues(); ++j) {
      auto newVal = phiInst->getIncomingValue(j);
      if (auto phi = dyn_cast<PHINode>(newVal)) {
        todo.push_back(phi);
      } else {
        auto metaString = getMetadataName(newVal);
        if (metaString) {
          if (!finalMetadata) {
            finalMetadata = metaString;
          } else if (finalMetadata != metaString) {
            return {};
          }
        }
      }
    }
  }
  return finalMetadata;
}

static Optional<StringRef> getMetadataName(llvm::Value *res) {
  if (auto av = dyn_cast<MetadataAsValue>(res)) {
    return cast<MDString>(av->getMetadata())->getString();
  } else if ((isa<LoadInst>(res) || isa<CastInst>(res)) &&
             isa<GlobalVariable>(cast<Instruction>(res)->getOperand(0))) {
    GlobalVariable *gv =
        cast<GlobalVariable>(cast<Instruction>(res)->getOperand(0));
    return gv->getName();
  } else if (isa<LoadInst>(res) &&
             isa<ConstantExpr>(cast<LoadInst>(res)->getOperand(0)) &&
             cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))->isCast() &&
             isa<GlobalVariable>(
                 cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))
                     ->getOperand(0))) {
    auto gv = cast<GlobalVariable>(
        cast<ConstantExpr>(cast<LoadInst>(res)->getOperand(0))->getOperand(0));
    return gv->getName();
  } else if (auto gv = dyn_cast<GlobalVariable>(res)) {
    return gv->getName();
  } else if (isa<ConstantExpr>(res) && cast<ConstantExpr>(res)->isCast() &&
             isa<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0))) {
    auto gv = cast<GlobalVariable>(cast<ConstantExpr>(res)->getOperand(0));
    return gv->getName();
  } else if (isa<CastInst>(res) && cast<CastInst>(res) &&
             isa<AllocaInst>(cast<CastInst>(res)->getOperand(0))) {
    auto gv = cast<AllocaInst>(cast<CastInst>(res)->getOperand(0));
    return gv->getName();
  } else if (auto gv = dyn_cast<AllocaInst>(res)) {
    return gv->getName();
  } else {
    if (isa<PHINode>(res)) {
      return recursePhiReads(cast<PHINode>(res));
    }
    return {};
  }
}

static Value *adaptReturnedVector(Value *ret, Value *diffret,
                                  IRBuilder<> &Builder, unsigned width) {
  Type *returnType = ret->getType();

  if (StructType *sty = dyn_cast<StructType>(returnType)) {
    Value *agg = ConstantAggregateZero::get(sty);

    for (unsigned int i = 0; i < width; i++) {
      Value *elem = Builder.CreateExtractValue(diffret, {i});
#if LLVM_VERSION_MAJOR >= 11
      if (auto vty = dyn_cast<FixedVectorType>(elem->getType())) {
#else
      if (auto vty = dyn_cast<VectorType>(elem->getType())) {
#endif
        for (unsigned j = 0; j < vty->getNumElements(); ++j) {
          Value *vecelem = Builder.CreateExtractElement(elem, j);
          agg = Builder.CreateInsertValue(agg, vecelem, {i * j});
        }
      } else {
        agg = Builder.CreateInsertValue(agg, elem, {i});
      }
    }
    diffret = agg;
  }
  return diffret;
}

static bool ReplaceOriginalCall(IRBuilder<> &Builder, Value *ret,
                                Type *retElemType, Value *diffret,
                                Instruction *CI, DerivativeMode mode) {
  Type *retType = ret->getType();
  Type *diffretType = diffret->getType();
  auto &DL = CI->getModule()->getDataLayout();

  if (diffretType->isEmptyTy() || diffretType->isVoidTy() ||
      retType->isEmptyTy() || retType->isVoidTy()) {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
    return true;
  }

  if (retType == diffretType) {
    CI->replaceAllUsesWith(diffret);
    CI->eraseFromParent();
    return true;
  }

  if (auto sretType = dyn_cast<StructType>(retType),
      diffsretType = dyn_cast<StructType>(diffretType);
      sretType && diffsretType && sretType->isLayoutIdentical(diffsretType)) {
    Value *newStruct = UndefValue::get(sretType);
    for (unsigned int i = 0; i < sretType->getStructNumElements(); i++) {
      Value *elem = Builder.CreateExtractValue(diffret, {i});
      newStruct = Builder.CreateInsertValue(newStruct, elem, {i});
    }
    CI->replaceAllUsesWith(newStruct);
    CI->eraseFromParent();
    return true;
  }

  if (isa<PointerType>(retType)) {
    retType = retElemType;

    if (auto sretType = dyn_cast<StructType>(retType),
        diffsretType = dyn_cast<StructType>(diffretType);
        sretType && diffsretType && sretType->isLayoutIdentical(diffsretType)) {
      for (unsigned int i = 0; i < sretType->getStructNumElements(); i++) {
#if LLVM_VERSION_MAJOR > 7
        Value *sgep = Builder.CreateStructGEP(retType, ret, i);
#else
        Value *sgep = Builder.CreateStructGEP(ret, i);
#endif
        Builder.CreateStore(Builder.CreateExtractValue(diffret, {i}), sgep);
      }
      CI->eraseFromParent();
      return true;
    }

    if (DL.getTypeSizeInBits(retType) >= DL.getTypeSizeInBits(diffretType)) {
      Builder.CreateStore(
          diffret,
          Builder.CreatePointerCast(ret, PointerType::getUnqual(diffretType)));
      CI->eraseFromParent();
      return true;
    }
  }

  if (mode == DerivativeMode::ReverseModePrimal &&
      DL.getTypeSizeInBits(retType) >= DL.getTypeSizeInBits(diffretType)) {
    IRBuilder<> EB(CI->getFunction()->getEntryBlock().getFirstNonPHI());
    auto AL = EB.CreateAlloca(retType);
    Builder.CreateStore(diffret, Builder.CreatePointerCast(
                                     AL, PointerType::getUnqual(diffretType)));
#if LLVM_VERSION_MAJOR > 7
    Value *cload = Builder.CreateLoad(retType, AL);
#else
    Value *cload = Builder.CreateLoad(AL);
#endif
    CI->replaceAllUsesWith(cload);
    CI->eraseFromParent();
    return true;
  }

  if (mode != DerivativeMode::ReverseModePrimal) {
    auto diffreti = Builder.CreateExtractValue(diffret, {0});
    if (diffreti->getType() == retType) {
      CI->replaceAllUsesWith(diffreti);
      CI->eraseFromParent();
      return true;
    } else if (diffretType == retType) {
      CI->replaceAllUsesWith(diffret);
      CI->eraseFromParent();
      return true;
    }
  }

  EmitFailure("IllegalReturnCast", CI->getDebugLoc(), CI,
              "Cannot cast return type of gradient ", *diffretType, *diffret,
              ", to desired type ", *retType);
  return false;
}

class EnzymeBase {
public:
  EnzymeLogic Logic;
  EnzymeBase(bool PostOpt)
      : Logic(EnzymePostOpt.getNumOccurrences() ? EnzymePostOpt : PostOpt) {
    // initializeLowerAutodiffIntrinsicPass(*PassRegistry::getPassRegistry());
  }

  Function *parseFunctionParameter(CallInst *CI) {
    Value *fn = CI->getArgOperand(0);

    // determine function to differentiate
    if (CI->hasStructRetAttr()) {
      fn = CI->getArgOperand(1);
    }

    fn = GetFunctionFromValue(fn);

    if (!fn || !isa<Function>(fn)) {
      EmitFailure("NoFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
      return nullptr;
    }
    if (cast<Function>(fn)->empty()) {
      EmitFailure("EmptyFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *fn);
      return nullptr;
    }

    return cast<Function>(fn);
  }

  static Optional<unsigned> parseWidthParameter(CallInst *CI) {
    unsigned width = 1;

#if LLVM_VERSION_MAJOR >= 14
    for (auto [i, found] = std::tuple{0u, false}; i < CI->arg_size(); ++i)
#else
    for (auto [i, found] = std::tuple{0u, false}; i < CI->getNumArgOperands();
         ++i)
#endif
    {
      Value *arg = CI->getArgOperand(i);

      if (auto MDName = getMetadataName(arg)) {
        if (*MDName == "enzyme_width") {
          if (found) {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "vector width declared more than once",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }

#if LLVM_VERSION_MAJOR >= 14
          if (i + 1 >= CI->arg_size())
#else
          if (i + 1 >= CI->getNumArgOperands())
#endif
          {
            EmitFailure("MissingVectorWidth", CI->getDebugLoc(), CI,
                        "constant integer followong enzyme_width is missing",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }

          Value *width_arg = CI->getArgOperand(i + 1);
          if (auto cint = dyn_cast<ConstantInt>(width_arg)) {
            width = cint->getZExtValue();
            found = true;
          } else {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "enzyme_width must be a constant integer",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }

          if (!found) {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "illegal enzyme vector argument width ",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }
        }
      }
    }
    return width;
  }

  struct Options {
    Value *differet;
    Value *tape;
    Value *dynamic_interface;
    std::pair<Value *, Value *> trace;
    std::pair<Value *, Value *> observations;
    unsigned width;
    int allocatedTapeSize;
    bool freeMemory;
    bool returnUsed;
    bool tapeIsPointer;
    bool differentialReturn;
    DIFFE_TYPE retType;
  };

  static Optional<Options> handleArguments(IRBuilder<> &Builder, CallInst *CI,
                                           Function *fn, DerivativeMode mode,
                                           bool sizeOnly,
                                           std::vector<DIFFE_TYPE> &constants,
                                           SmallVectorImpl<Value *> &args,
                                           std::map<int, Type *> &byVal) {
    FunctionType *FT = fn->getFunctionType();

    Value *differet = nullptr;
    Value *tape = nullptr;
    Value *dynamic_interface = nullptr;
    std::pair<Value *, Value *> trace = {nullptr, nullptr};
    std::pair<Value *, Value *> observations = {nullptr, nullptr};
    unsigned width = 1;
    int allocatedTapeSize = -1;
    bool freeMemory = true;
    bool tapeIsPointer = false;
    unsigned truei = 0;
    unsigned byRefSize = 0;

    DIFFE_TYPE retType = whatType(fn->getReturnType(), mode);

    bool returnUsed =
        !fn->getReturnType()->isVoidTy() && !fn->getReturnType()->isEmptyTy();

    bool differentialReturn = (mode == DerivativeMode::ReverseModeCombined ||
                               mode == DerivativeMode::ReverseModeGradient) &&
                              (retType == DIFFE_TYPE::OUT_DIFF);

    bool sret = CI->hasStructRetAttr() ||
                fn->hasParamAttribute(0, Attribute::StructRet);

    // find and handle enzyme_width
    if (auto parsedWidth = parseWidthParameter(CI)) {
      width = *parsedWidth;
    } else {
      return {};
    }

    // handle different argument order for struct return.
    if (fn->hasParamAttribute(0, Attribute::StructRet)) {
      Type *fnsrety = cast<PointerType>(FT->getParamType(0));

      truei = 1;

      const DataLayout &DL = CI->getParent()->getModule()->getDataLayout();
      Type *Ty = fnsrety->getPointerElementType();
#if LLVM_VERSION_MAJOR >= 11
      AllocaInst *primal = new AllocaInst(Ty, DL.getAllocaAddrSpace(), nullptr,
                                          DL.getPrefTypeAlign(Ty));
#else
      AllocaInst *primal = new AllocaInst(Ty, DL.getAllocaAddrSpace(), nullptr);
#endif

      primal->insertBefore(CI);

      Value *shadow;
      switch (mode) {
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        Value *sretPt = CI->getArgOperand(0);
        if (width > 1) {
          PointerType *pty = cast<PointerType>(sretPt->getType());
          if (auto sty = dyn_cast<StructType>(pty->getPointerElementType())) {
            Value *acc = UndefValue::get(
                ArrayType::get(PointerType::get(sty->getElementType(0),
                                                pty->getAddressSpace()),
                               width));
            for (size_t i = 0; i < width; ++i) {
#if LLVM_VERSION_MAJOR > 7
              Value *elem = Builder.CreateStructGEP(
                  sretPt->getType()->getPointerElementType(), sretPt, i);
#else
              Value *elem = Builder.CreateStructGEP(sretPt, i);
#endif
              acc = Builder.CreateInsertValue(acc, elem, i);
            }
            shadow = acc;
          } else {
            EmitFailure(
                "IllegalReturnType", CI->getDebugLoc(), CI,
                "Return type of __enzyme_autodiff has to be a struct with",
                width, "elements of the same type.");
            return {};
          }
        } else {
          shadow = sretPt;
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:
      case DerivativeMode::ReverseModeCombined:
      case DerivativeMode::ReverseModeGradient: {
        shadow = CI->getArgOperand(1);
        sret = true;
        break;
      }
      }

      args.push_back(primal);
      args.push_back(shadow);
      constants.push_back(DIFFE_TYPE::DUP_ARG);
    }

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 1 + sret; i < CI->arg_size(); ++i)
#else
    for (unsigned i = 1 + sret; i < CI->getNumArgOperands(); ++i)
#endif
    {
      Value *res = CI->getArgOperand(i);
      Optional<DIFFE_TYPE> opt_ty;
      auto metaString = getMetadataName(res);
      Optional<Value *> batchOffset;

      // handle metadata
      if (metaString && metaString->startswith("enzyme_")) {
        if (*metaString == "enzyme_byref") {
          ++i;
          if (!isa<ConstantInt>(CI->getArgOperand(i))) {
            EmitFailure("IllegalAllocatedSize", CI->getDebugLoc(), CI,
                        "illegal enzyme byref size ", *CI->getArgOperand(i),
                        "in", *CI);
            return {};
          }
          byRefSize = cast<ConstantInt>(CI->getArgOperand(i))->getZExtValue();
          assert(byRefSize > 0);
          continue;
        }
        if (*metaString == "enzyme_dup") {
          opt_ty = DIFFE_TYPE::DUP_ARG;
        } else if (*metaString == "enzyme_dupv") {
          opt_ty = DIFFE_TYPE::DUP_ARG;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }
        } else if (*metaString == "enzyme_dupnoneed") {
          opt_ty = DIFFE_TYPE::DUP_NONEED;
        } else if (*metaString == "enzyme_dupnoneedv") {
          opt_ty = DIFFE_TYPE::DUP_NONEED;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }
        } else if (*metaString == "enzyme_out") {
          opt_ty = DIFFE_TYPE::OUT_DIFF;
        } else if (*metaString == "enzyme_const") {
          opt_ty = DIFFE_TYPE::CONSTANT;
        } else if (*metaString == "enzyme_noret") {
          returnUsed = false;
          continue;
        } else if (*metaString == "enzyme_allocated") {
          assert(!sizeOnly);
          ++i;
          if (!isa<ConstantInt>(CI->getArgOperand(i))) {
            EmitFailure("IllegalAllocatedSize", CI->getDebugLoc(), CI,
                        "illegal enzyme allocated size ", *CI->getArgOperand(i),
                        "in", *CI);
            return {};
          }
          allocatedTapeSize =
              cast<ConstantInt>(CI->getArgOperand(i))->getZExtValue();
          continue;
        } else if (*metaString == "enzyme_tape") {
          assert(!sizeOnly);
          ++i;
          tape = CI->getArgOperand(i);
          tapeIsPointer = true;
          continue;
        } else if (*metaString == "enzyme_nofree") {
          assert(!sizeOnly);
          freeMemory = false;
          continue;
        } else if (*metaString == "enzyme_width") {
          ++i;
          continue;
        } else if (*metaString == "enzyme_interface") {
          ++i;
          dynamic_interface = CI->getArgOperand(i);
          continue;
        } else if (*metaString == "enzyme_trace") {
          trace.first = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::CONSTANT;
          continue;
        } else if (*metaString == "enzyme_duptrace") {
          trace.first = CI->getArgOperand(++i);
          trace.second = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::DUP_ARG;
          continue;
        } else if (*metaString == "enzyme_observations") {
          observations.first = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::CONSTANT;
          continue;
        } else if (*metaString == "enzyme_dupobservations") {
          observations.first = CI->getArgOperand(++i);
          observations.second = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::DUP_ARG;
          continue;
        } else {
          EmitFailure("IllegalDiffeType", CI->getDebugLoc(), CI,
                      "illegal enzyme metadata classification ", *CI,
                      *metaString);
          return {};
        }
        if (sizeOnly) {
          assert(opt_ty);
          constants.push_back(*opt_ty);
          truei++;
          continue;
        }
        ++i;
        res = CI->getArgOperand(i);
      }

      if (byRefSize) {
        Type *subTy = res->getType()->getPointerElementType();
        auto &DL = fn->getParent()->getDataLayout();
        auto BitSize = DL.getTypeSizeInBits(subTy);
        if (BitSize / 8 != byRefSize) {
          EmitFailure("IllegalByRefSize", CI->getDebugLoc(), CI,
                      "illegal enzyme pointer type size ", *res, " expected ",
                      byRefSize, " (bytes) actual size ", BitSize,
                      " (bits) in ", *CI);
        }
#if LLVM_VERSION_MAJOR > 7
        res = Builder.CreateLoad(subTy, res);
#else
        res = Builder.CreateLoad(res);
#endif
        byRefSize = 0;
      }

      if (truei >= FT->getNumParams()) {
        if (!isa<MetadataAsValue>(res) &&
            (mode == DerivativeMode::ReverseModeGradient ||
             mode == DerivativeMode::ForwardModeSplit)) {
          if (differentialReturn && differet == nullptr) {
            differet = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
#if LLVM_VERSION_MAJOR > 7
              differet = Builder.CreateLoad(
                  differet->getType()->getPointerElementType(), differet);
#else
              differet = Builder.CreateLoad(differet);
#endif
            }
            if (differet->getType() != fn->getReturnType())
              if (auto ST0 = dyn_cast<StructType>(differet->getType()))
                if (auto ST1 = dyn_cast<StructType>(fn->getReturnType()))
                  if (ST0->isLayoutIdentical(ST1)) {
                    IRBuilder<> B(&Builder.GetInsertBlock()
                                       ->getParent()
                                       ->getEntryBlock()
                                       .front());
                    auto AI = B.CreateAlloca(ST1);
                    Builder.CreateStore(differet,
                                        Builder.CreatePointerCast(
                                            AI, PointerType::getUnqual(ST0)));
#if LLVM_VERSION_MAJOR > 7
                    differet = Builder.CreateLoad(ST1, AI);
#else
                    differet = Builder.CreateLoad(AI);
#endif
                  }

            if (differet->getType() != fn->getReturnType()) {
              EmitFailure("BadDiffRet", CI->getDebugLoc(), CI,
                          "Bad DiffRet type ", *differet, " expected ",
                          *fn->getReturnType());
              return {};
            }
            continue;
          } else if (tape == nullptr) {
            tape = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
#if LLVM_VERSION_MAJOR > 7
              tape = Builder.CreateLoad(
                  tape->getType()->getPointerElementType(), tape);
#else
              tape = Builder.CreateLoad(tape);
#endif
            }
            continue;
          }
        }
        EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                    "Had too many arguments to __enzyme_autodiff", *CI,
                    " - extra arg - ", *res);
        return {};
      }
      assert(truei < FT->getNumParams());

      auto PTy = FT->getParamType(truei);
      DIFFE_TYPE ty = opt_ty ? *opt_ty : whatType(PTy, mode);

      constants.push_back(ty);

      assert(truei < FT->getNumParams());
      // cast primal
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
          if (auto PT = dyn_cast<PointerType>(PTy)) {
            if (ptr->getAddressSpace() != PT->getAddressSpace()) {
              res = Builder.CreateAddrSpaceCast(
                  res, PointerType::get(ptr->getPointerElementType(),
                                        PT->getAddressSpace()));
              assert(res);
              assert(PTy);
              assert(FT);
              llvm::errs() << "Warning cast(1) __enzyme_autodiff argument " << i
                           << " " << *res << "|" << *res->getType()
                           << " to argument " << truei << " " << *PTy << "\n"
                           << "orig: " << *FT << "\n";
            }
          }
        }
        if (res->getType()->canLosslesslyBitCastTo(PTy)) {
          res = Builder.CreateBitCast(res, PTy);
        }
        if (res->getType() != PTy && res->getType()->isIntegerTy() &&
            PTy->isIntegerTy(1)) {
          res = Builder.CreateTrunc(res, PTy);
        }
        if (res->getType() != PTy) {
          auto loc = CI->getDebugLoc();
          if (auto arg = dyn_cast<Instruction>(res)) {
            loc = arg->getDebugLoc();
          }
          EmitFailure("IllegalArgCast", loc, CI,
                      "Cannot cast __enzyme_autodiff primal argument ", i,
                      ", found ", *res, ", type ", *res->getType(),
                      " - to arg ", truei, " ", *PTy);
          return {};
        }
      }
#if LLVM_VERSION_MAJOR >= 9
      if (CI->isByValArgument(i)) {
        byVal[args.size()] = CI->getParamByValType(i);
      }
#endif
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;

        Value *res = nullptr;
        bool batch = batchOffset.hasValue();

        for (unsigned v = 0; v < width; ++v) {
#if LLVM_VERSION_MAJOR >= 14
          if (i >= CI->arg_size())
#else
          if (i >= CI->getNumArgOperands())
#endif
          {
            EmitFailure("MissingArgShadow", CI->getDebugLoc(), CI,
                        "__enzyme_autodiff missing argument shadow at index ",
                        i, ", need shadow of type ", *PTy,
                        " to shadow primal argument ", *args.back(),
                        " at call ", *CI);
            return {};
          }

          // cast diffe
          Value *element = CI->getArgOperand(i);
          if (batch) {
            if (auto elementPtrTy = dyn_cast<PointerType>(element->getType())) {
              element = Builder.CreateBitCast(
                  element, PointerType::get(Type::getInt8Ty(CI->getContext()),
                                            elementPtrTy->getAddressSpace()));
#if LLVM_VERSION_MAJOR >= 7
              element = Builder.CreateGEP(
                  Type::getInt8Ty(CI->getContext()), element,
                  Builder.CreateMul(
                      *batchOffset,
                      ConstantInt::get((*batchOffset)->getType(), v)));
#else
              element = Builder.CreateGEP(
#if LLVM_VERSION_MAJOR >= 14
                  elementPtrTy,
#endif
                  element,
                  Builder.CreateMul(
                      *batchOffset,
                      ConstantInt::get((*batchOffset)->getType(), v)));
#endif
              element = Builder.CreateBitCast(element, elementPtrTy);
            } else {
              EmitFailure(
                  "NonPointerBatch", CI->getDebugLoc(), CI,
                  "Batched argument at index ", i,
                  " must be of pointer type, found: ", *element->getType());
              return {};
            }
          }
          if (PTy != element->getType()) {
            element = castToDiffeFunctionArgType(Builder, CI, FT, PTy, i, mode,
                                                 element, truei);
            if (!element) {
              return {};
            }
          }

          if (width > 1) {
            res =
                res ? Builder.CreateInsertValue(res, element, {v})
                    : Builder.CreateInsertValue(UndefValue::get(ArrayType::get(
                                                    element->getType(), width)),
                                                element, {v});

            if (v < width - 1 && !batch) {
              ++i;
            }

          } else {
            res = element;
          }
        }

        args.push_back(res);
      }

      ++truei;
    }
    if (truei < FT->getNumParams()) {
      auto numParams = FT->getNumParams();
      EmitFailure(
          "EnzymeInsufficientArgs", CI->getDebugLoc(), CI,
          "Insufficient number of args passed to derivative call required ",
          numParams, " primal args, found ", truei);
      return {};
    }

    return Optional<Options>({differet, tape, dynamic_interface, trace,
                              observations, width, allocatedTapeSize,
                              freeMemory, returnUsed, tapeIsPointer,
                              differentialReturn, retType});
  }

  static FnTypeInfo
  populate_overwritten_args(TypeAnalysis &TA, llvm::Function *fn,
                            DerivativeMode mode,
                            std::vector<bool> &overwritten_args) {
    FnTypeInfo type_args(fn);
    for (auto &a : type_args.Function->args()) {
      overwritten_args.push_back(
          !(mode == DerivativeMode::ReverseModeCombined));
      TypeTree dt;
      if (a.getType()->isFPOrFPVectorTy()) {
        dt = ConcreteType(a.getType()->getScalarType());
      } else if (a.getType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR >= 15
        if (a.getContext().supportsTypedPointers()) {
#endif
          auto et = a.getType()->getPointerElementType();
          if (et->isFPOrFPVectorTy()) {
            dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
          } else if (et->isPointerTy()) {
            dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
          }
#if LLVM_VERSION_MAJOR >= 15
        }
#endif
        dt.insert({}, BaseType::Pointer);
      } else if (a.getType()->isIntOrIntVectorTy()) {
        dt = ConcreteType(BaseType::Integer);
      }
      type_args.Arguments.insert(
          std::pair<Argument *, TypeTree>(&a, dt.Only(-1, nullptr)));
      // TODO note that here we do NOT propagate constants in type info (and
      // should consider whether we should)
      type_args.KnownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&a, {}));
    }
    TypeTree dt;
    if (fn->getReturnType()->isFPOrFPVectorTy()) {
      dt = ConcreteType(fn->getReturnType()->getScalarType());
    }
    type_args.Return = dt.Only(-1, nullptr);

    type_args = TA.analyzeFunction(type_args).getAnalyzedTypeInfo();
    return type_args;
  }

  bool HandleBatch(CallInst *CI) {
    unsigned width = 1;
    unsigned truei = 0;
    std::map<unsigned, Value *> batchOffset;
    SmallVector<Value *, 4> args;
    SmallVector<BATCH_TYPE, 4> arg_types;
    IRBuilder<> Builder(CI);
    Function *F = parseFunctionParameter(CI);
    if (!F)
      return false;

    assert(F);
    FunctionType *FT = F->getFunctionType();

    // find and handle enzyme_width
    if (auto parsedWidth = parseWidthParameter(CI)) {
      width = *parsedWidth;
    } else {
      return false;
    }

    // handle different argument order for struct return.
    bool sret =
        CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);

    if (F->hasParamAttribute(0, Attribute::StructRet)) {
      truei = 1;
      Value *sretPt = CI->getArgOperand(0);

      args.push_back(sretPt);
      arg_types.push_back(BATCH_TYPE::VECTOR);
    }

#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 1 + sret; i < CI->arg_size(); ++i)
#else
    for (unsigned i = 1 + sret; i < CI->getNumArgOperands(); ++i)
#endif
    {
      Value *res = CI->getArgOperand(i);

      if (truei >= FT->getNumParams()) {
        EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                    "Had too many arguments to __enzyme_batch", *CI,
                    " - extra arg - ", *res);
        return false;
      }
      assert(truei < FT->getNumParams());
      auto PTy = FT->getParamType(truei);

      BATCH_TYPE ty = width == 1 ? BATCH_TYPE::SCALAR : BATCH_TYPE::VECTOR;
      auto metaString = getMetadataName(res);

      // handle metadata
      if (metaString && metaString->startswith("enzyme_")) {
        if (*metaString == "enzyme_scalar") {
          ty = BATCH_TYPE::SCALAR;
        } else if (*metaString == "enzyme_vector") {
          ty = BATCH_TYPE::VECTOR;
        } else if (*metaString == "enzyme_buffer") {
          ty = BATCH_TYPE::VECTOR;
          ++i;
          Value *offset_arg = CI->getArgOperand(i);
          if (offset_arg->getType()->isIntegerTy()) {
            batchOffset[i + 1] = offset_arg;
          } else {
            EmitFailure("IllegalVectorOffset", CI->getDebugLoc(), CI,
                        "enzyme_batch must be followd by an integer "
                        "offset.",
                        *CI->getArgOperand(i), " in", *CI);
            return false;
          }
          continue;
        } else if (*metaString == "enzyme_width") {
          ++i;
          continue;
        } else {
          EmitFailure("IllegalDiffeType", CI->getDebugLoc(), CI,
                      "illegal enzyme metadata classification ", *CI,
                      *metaString);
          return false;
        }
        ++i;
        res = CI->getArgOperand(i);
      }

      arg_types.push_back(ty);

      // wrap vector
      if (ty == BATCH_TYPE::VECTOR) {
        Value *res = nullptr;
        bool batch = batchOffset.count(i - 1) != 0;

        for (unsigned v = 0; v < width; ++v) {
#if LLVM_VERSION_MAJOR >= 14
          if (i >= CI->arg_size())
#else
          if (i >= CI->getNumArgOperands())
#endif
          {
            EmitFailure("MissingVectorArg", CI->getDebugLoc(), CI,
                        "__enzyme_batch missing vector argument at index ", i,
                        ", need argument of type ", *PTy, " at call ", *CI);
            return false;
          }

          // vectorize pointer
          Value *element = CI->getArgOperand(i);
          if (batch) {
            if (auto elementPtrTy = dyn_cast<PointerType>(element->getType())) {
              element = Builder.CreateBitCast(
                  element, PointerType::get(Type::getInt8Ty(CI->getContext()),
                                            elementPtrTy->getAddressSpace()));
#if LLVM_VERSION_MAJOR >= 7
              element = Builder.CreateGEP(
                  Type::getInt8Ty(CI->getContext()), element,
                  Builder.CreateMul(
                      batchOffset[i - 1],
                      ConstantInt::get(batchOffset[i - 1]->getType(), v)));
#else
              element = Builder.CreateGEP(
                  element,
                  Builder.CreateMul(
                      batchOffset[i - 1],
                      ConstantInt::get(batchOffset[i - 1]->getType(), v)));
#endif
              element = Builder.CreateBitCast(element, elementPtrTy);
            } else {
              return false;
            }
          }

          if (width > 1) {
            res =
                res ? Builder.CreateInsertValue(res, element, {v})
                    : Builder.CreateInsertValue(UndefValue::get(ArrayType::get(
                                                    element->getType(), width)),
                                                element, {v});

            if (v < width - 1 && !batch) {
              ++i;
            }

          } else {
            res = element;
          }
        }

        args.push_back(res);

      } else if (ty == BATCH_TYPE::SCALAR) {
        args.push_back(res);
      }

      truei++;
    }

    BATCH_TYPE ret_type = (F->getReturnType()->isVoidTy() || width == 1)
                              ? BATCH_TYPE::SCALAR
                              : BATCH_TYPE::VECTOR;

    auto newFunc = Logic.CreateBatch(F, width, arg_types, ret_type);

    if (!newFunc)
      return false;

    Value *batch =
        Builder.CreateCall(newFunc->getFunctionType(), newFunc, args);

    batch = adaptReturnedVector(CI, batch, Builder, width);

    Value *ret = CI;
    Type *retElemType = nullptr;
    if (CI->hasStructRetAttr()) {
      ret = CI->getArgOperand(0);
#if LLVM_VERSION_MAJOR >= 15
      retElemType = CI->getParamStructRetType(0);
#else
      retElemType = ret->getType()->getPointerElementType();
#endif
    }
    ReplaceOriginalCall(Builder, ret, retElemType, batch, CI,
                        DerivativeMode::ForwardMode);

    return true;
  }

  bool HandleAutoDiff(Instruction *CI, CallingConv::ID CallingConv, Value *ret,
                      Type *retElemType, SmallVectorImpl<Value *> &args,
                      const std::map<int, Type *> &byVal,
                      const std::vector<DIFFE_TYPE> &constants, Function *fn,
                      DerivativeMode mode, Options &options, bool sizeOnly) {
    auto &differet = options.differet;
    auto &tape = options.tape;
    auto &width = options.width;
    auto &allocatedTapeSize = options.allocatedTapeSize;
    auto &freeMemory = options.freeMemory;
    auto &returnUsed = options.returnUsed;
    auto &tapeIsPointer = options.tapeIsPointer;
    auto &differentialReturn = options.differentialReturn;
    auto &retType = options.retType;

    auto Arch = Triple(CI->getModule()->getTargetTriple()).getArch();
    bool AtomicAdd = Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
                     Arch == Triple::amdgcn;

    TypeAnalysis TA(Logic.PPC.FAM);
    std::vector<bool> overwritten_args;
    FnTypeInfo type_args =
        populate_overwritten_args(TA, fn, mode, overwritten_args);

    IRBuilder Builder(CI);

    // differentiate fn
    Function *newFunc = nullptr;
    Type *tapeType = nullptr;
    const AugmentedReturn *aug;
    switch (mode) {
    case DerivativeMode::ForwardMode:
      newFunc = Logic.CreateForwardDiff(
          fn, retType, constants, TA,
          /*should return*/ false, mode, freeMemory, width,
          /*addedType*/ nullptr, type_args, overwritten_args,
          /*augmented*/ nullptr);
      break;
    case DerivativeMode::ForwardModeSplit: {
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      aug = &Logic.CreateAugmentedPrimal(
          fn, retType, constants, TA,
          /*returnUsed*/ false, /*shadowReturnUsed*/ false, type_args,
          overwritten_args, forceAnonymousTape, width, /*atomicAdd*/ AtomicAdd);
      auto &DL = fn->getParent()->getDataLayout();
      if (!forceAnonymousTape) {
        assert(!aug->tapeType);
        if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
          auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
          tapeType = (tapeIdx == -1)
                         ? aug->fn->getReturnType()
                         : cast<StructType>(aug->fn->getReturnType())
                               ->getElementType(tapeIdx);
        } else {
          if (sizeOnly) {
            CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0, false));
            CI->eraseFromParent();
            return true;
          }
        }
        if (sizeOnly) {
          auto size = DL.getTypeSizeInBits(tapeType) / 8;
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), size, false));
          CI->eraseFromParent();
          return true;
        }
        if (tapeType &&
            DL.getTypeSizeInBits(tapeType) < 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = PointerType::getInt8PtrTy(fn->getContext());
      }
      newFunc = Logic.CreateForwardDiff(
          fn, retType, constants, TA,
          /*should return*/ false, mode, freeMemory, width,
          /*addedType*/ tapeType, type_args, overwritten_args, aug);
      break;
    }
    case DerivativeMode::ReverseModeCombined:
      assert(freeMemory);
      newFunc = Logic.CreatePrimalAndGradient(
          (ReverseCacheKey){.todiff = fn,
                            .retType = retType,
                            .constant_args = constants,
                            .overwritten_args = overwritten_args,
                            .returnUsed = false,
                            .shadowReturnUsed = false,
                            .mode = mode,
                            .width = width,
                            .freeMemory = freeMemory,
                            .AtomicAdd = AtomicAdd,
                            .additionalType = nullptr,
                            .forceAnonymousTape = false,
                            .typeInfo = type_args},
          TA, /*augmented*/ nullptr);
      break;
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient: {
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      bool shadowReturnUsed = returnUsed && (retType == DIFFE_TYPE::DUP_ARG ||
                                             retType == DIFFE_TYPE::DUP_NONEED);
      aug = &Logic.CreateAugmentedPrimal(
          fn, retType, constants, TA, returnUsed, shadowReturnUsed, type_args,
          overwritten_args, forceAnonymousTape, width,
          /*atomicAdd*/ AtomicAdd);
      auto &DL = fn->getParent()->getDataLayout();
      if (!forceAnonymousTape) {
        assert(!aug->tapeType);
        if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
          auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
          tapeType = (tapeIdx == -1)
                         ? aug->fn->getReturnType()
                         : cast<StructType>(aug->fn->getReturnType())
                               ->getElementType(tapeIdx);
        } else {
          if (sizeOnly) {
            CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), 0, false));
            CI->eraseFromParent();
            return true;
          }
        }
        if (sizeOnly) {
          auto size = DL.getTypeSizeInBits(tapeType) / 8;
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), size, false));
          CI->eraseFromParent();
          return true;
        }
        if (tapeType &&
            DL.getTypeSizeInBits(tapeType) < 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = PointerType::getInt8PtrTy(fn->getContext());
      }
      if (mode == DerivativeMode::ReverseModePrimal)
        newFunc = aug->fn;
      else
        newFunc = Logic.CreatePrimalAndGradient(
            (ReverseCacheKey){.todiff = fn,
                              .retType = retType,
                              .constant_args = constants,
                              .overwritten_args = overwritten_args,
                              .returnUsed = false,
                              .shadowReturnUsed = false,
                              .mode = mode,
                              .width = width,
                              .freeMemory = freeMemory,
                              .AtomicAdd = AtomicAdd,
                              .additionalType = tapeType,
                              .forceAnonymousTape = forceAnonymousTape,
                              .typeInfo = type_args},
            TA, aug);
    }
    }

    if (!newFunc) {
      StringRef n = fn->getName();
      EmitFailure("FailedToDifferentiate", fn->getSubprogram(),
                  &*fn->getEntryBlock().begin(),
                  "Could not generate derivative function of ", n);
      return false;
    }

    if (differentialReturn) {
      if (differet)
        args.push_back(differet);
      else if (fn->getReturnType()->isFPOrFPVectorTy()) {
        Constant *seed = ConstantFP::get(fn->getReturnType(), 1.0);
        if (width == 1) {
          args.push_back(seed);
        } else {
          ArrayType *arrayType = ArrayType::get(fn->getReturnType(), width);
          args.push_back(ConstantArray::get(
              arrayType, SmallVector<Constant *, 3>(width, seed)));
        }
      } else if (auto ST = dyn_cast<StructType>(fn->getReturnType())) {
        SmallVector<Constant *, 2> csts;
        for (auto e : ST->elements()) {
          csts.push_back(ConstantFP::get(e, 1.0));
        }
        args.push_back(ConstantStruct::get(ST, csts));
      }
    }

    if ((mode == DerivativeMode::ReverseModeGradient ||
         mode == DerivativeMode::ForwardModeSplit) &&
        tape && tapeType) {
      auto &DL = fn->getParent()->getDataLayout();
      if (tapeIsPointer) {
        tape = Builder.CreateBitCast(
            tape, PointerType::get(
                      tapeType,
                      cast<PointerType>(tape->getType())->getAddressSpace()));
#if LLVM_VERSION_MAJOR > 7
        tape = Builder.CreateLoad(tapeType, tape);
#else
        tape = Builder.CreateLoad(tape);
#endif
      } else if (tapeType != tape->getType() &&
                 DL.getTypeSizeInBits(tapeType) <=
                     DL.getTypeSizeInBits(tape->getType())) {
        IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
        auto AL = EB.CreateAlloca(tape->getType());
        Builder.CreateStore(tape, AL);
#if LLVM_VERSION_MAJOR > 7
        tape = Builder.CreateLoad(
            tapeType,
            Builder.CreatePointerCast(AL, PointerType::getUnqual(tapeType)));
#else
        tape = Builder.CreateLoad(
            Builder.CreatePointerCast(AL, PointerType::getUnqual(tapeType)));
#endif
      }
      assert(tape->getType() == tapeType);
      args.push_back(tape);
    }

    if (EnzymePrint) {
      llvm::errs() << "postfn:\n" << *newFunc << "\n";
    }
    Builder.setFastMathFlags(getFast());

    // call newFunc with the provided arguments.
    if (args.size() != newFunc->getFunctionType()->getNumParams()) {
      llvm::errs() << *CI << "\n";
      llvm::errs() << *newFunc << "\n";
      for (auto arg : args) {
        llvm::errs() << " + " << *arg << "\n";
      }
      auto modestr = to_string(mode);
      EmitFailure(
          "TooFewArguments", CI->getDebugLoc(), CI,
          "Too few arguments passed to __enzyme_autodiff mode=", modestr);
      return false;
    }
    assert(args.size() == newFunc->getFunctionType()->getNumParams());
    CallInst *diffretc = cast<CallInst>(Builder.CreateCall(newFunc, args));
    diffretc->setCallingConv(CallingConv);
    diffretc->setDebugLoc(CI->getDebugLoc());
#if LLVM_VERSION_MAJOR >= 9
    for (auto &&[attr, ty] : byVal) {
      diffretc->addParamAttr(
          attr, Attribute::getWithByValType(diffretc->getContext(), ty));
    }
#endif
    Value *diffret = diffretc;
    if (mode == DerivativeMode::ReverseModePrimal && tape) {
      if (aug->returns.find(AugmentedStruct::Tape) != aug->returns.end()) {
        auto tapeIdx = aug->returns.find(AugmentedStruct::Tape)->second;
        tapeType = (tapeIdx == -1) ? aug->fn->getReturnType()
                                   : cast<StructType>(aug->fn->getReturnType())
                                         ->getElementType(tapeIdx);
        unsigned idxs[] = {(unsigned)tapeIdx};
        Value *tapeRes = (tapeIdx == -1)
                             ? diffret
                             : Builder.CreateExtractValue(diffret, idxs);
        Builder.CreateStore(
            tapeRes,
            Builder.CreateBitCast(
                tape,
                PointerType::get(
                    tapeRes->getType(),
                    cast<PointerType>(tape->getType())->getAddressSpace())));
        if (tapeIdx != -1) {
          auto ST = cast<StructType>(diffret->getType());
          SmallVector<Type *, 2> tys(ST->elements().begin(),
                                     ST->elements().end());
          tys.erase(tys.begin());
          auto ST0 = StructType::get(ST->getContext(), tys);
          Value *out = UndefValue::get(ST0);
          for (unsigned i = 0; i < tys.size(); i++) {
            out = Builder.CreateInsertValue(
                out, Builder.CreateExtractValue(diffret, {i + 1}), {i});
          }
          diffret = out;
        } else {
          auto ST0 = StructType::get(tape->getContext(), {});
          diffret = UndefValue::get(ST0);
        }
      }
    }

    // Adapt the returned vector type to the struct type expected by our calling
    // convention.
    if (width > 1 && !diffret->getType()->isEmptyTy() &&
        !diffret->getType()->isVoidTy() &&
        (mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit)) {

      diffret = adaptReturnedVector(ret, diffret, Builder, width);
    }

    ReplaceOriginalCall(Builder, ret, retElemType, diffret, CI, mode);

    if (Logic.PostOpt) {
#if LLVM_VERSION_MAJOR >= 11
      auto Params = llvm::getInlineParams();

      llvm::SetVector<CallInst *> Q;
      Q.insert(diffretc);
      while (Q.size()) {
        auto cur = *Q.begin();
        Function *outerFunc = cur->getParent()->getParent();
        llvm::OptimizationRemarkEmitter ORE(outerFunc);
        Q.erase(Q.begin());
        if (auto F = cur->getCalledFunction()) {
          if (!F->empty()) {
            // Garbage collect AC's created
            SmallVector<AssumptionCache *, 2> ACAlloc;
            auto getAC = [&](Function &F) -> llvm::AssumptionCache & {
              auto AC = new AssumptionCache(F);
              ACAlloc.push_back(AC);
              return *AC;
            };
            auto GetTLI =
                [&](llvm::Function &F) -> const llvm::TargetLibraryInfo & {
              return Logic.PPC.FAM.getResult<TargetLibraryAnalysis>(F);
            };

            auto GetInlineCost = [&](CallBase &CB) {
              TargetTransformInfo TTI(F->getParent()->getDataLayout());
              auto cst = llvm::getInlineCost(CB, Params, TTI, getAC, GetTLI);
              return cst;
            };
            if (llvm::shouldInline(*cur, GetInlineCost, ORE)) {
              InlineFunctionInfo IFI;
              InlineResult IR =
#if LLVM_VERSION_MAJOR >= 11
                  InlineFunction(*cur, IFI);
#else
                  InlineFunction(cur, IFI);
#endif
              if (IR.isSuccess()) {
                LowerSparsification(outerFunc, /*replaceAll*/ false);
                for (auto U : outerFunc->users()) {
                  if (auto CI = dyn_cast<CallInst>(U)) {
                    if (CI->getCalledFunction() == outerFunc) {
                      Q.insert(CI);
                    }
                  }
                }
              }
            }
            for (auto AC : ACAlloc) {
              delete AC;
            }
          }
        }
      }
#endif
    }
    return true;
  }

  /// Return whether successful
  bool HandleAutoDiffArguments(CallInst *CI, DerivativeMode mode,
                               bool sizeOnly) {

    // determine function to differentiate
    Function *fn = parseFunctionParameter(CI);
    if (!fn)
      return false;

    IRBuilder<> Builder(CI);

    if (EnzymePrint)
      llvm::errs() << "prefn:\n" << *fn << "\n";

    std::map<int, Type *> byVal;
    std::vector<DIFFE_TYPE> constants;
    SmallVector<Value *, 2> args;

    auto options = handleArguments(Builder, CI, fn, mode, sizeOnly, constants,
                                   args, byVal);

    if (!options) {
      return false;
    }

    Value *ret = CI;
    Type *retElemType = nullptr;
    if (CI->hasStructRetAttr()) {
      ret = CI->getArgOperand(0);
#if LLVM_VERSION_MAJOR >= 15
      retElemType = CI->getParamStructRetType(0);
#else
      retElemType = ret->getType()->getPointerElementType();
#endif
    }

    return HandleAutoDiff(CI, CI->getCallingConv(), ret, retElemType, args,
                          byVal, constants, fn, mode, options.getValue(),
                          sizeOnly);
  }

  bool HandleProbProg(CallInst *CI, ProbProgMode mode) {
    IRBuilder<> Builder(CI);
    Function *F = parseFunctionParameter(CI);
    if (!F)
      return false;

    assert(F);

    std::vector<DIFFE_TYPE> constants;
    std::map<int, Type *> byVal;
    SmallVector<Value *, 4> args;

    auto diffeMode = DerivativeMode::ReverseModeCombined;

    auto opt = handleArguments(Builder, CI, F, diffeMode, false, constants,
                               args, byVal);

    SmallVector<Value *, 6> dargs = SmallVector(args);

    if (!opt.hasValue())
      return false;

    auto dynamic_interface = opt->dynamic_interface;
    auto trace = opt->trace.first;
    auto dtrace = opt->trace.second;
    auto observations = opt->observations.first;
    auto dobservations = opt->observations.second;

    // Interface
    bool has_dynamic_interface = dynamic_interface != nullptr;
    TraceInterface *interface;
    if (has_dynamic_interface) {
      interface =
          new DynamicTraceInterface(dynamic_interface, CI->getFunction());
    } else {
      interface = new StaticTraceInterface(F->getParent());
    }

    if (mode == ProbProgMode::Condition) {
      args.push_back(observations);
      dargs.push_back(observations);
      if (dobservations) {
        dargs.push_back(dobservations);
        constants.push_back(DIFFE_TYPE::DUP_ARG);
      } else {
        constants.push_back(DIFFE_TYPE::CONSTANT);
      }
    }

    args.push_back(trace);
    dargs.push_back(trace);
    if (dtrace) {
      dargs.push_back(dtrace);
      constants.push_back(DIFFE_TYPE::DUP_ARG);
    } else {
      constants.push_back(DIFFE_TYPE::CONSTANT);
    }

    // Determine generative functions
    SmallPtrSet<Function *, 4> generativeFunctions;
    SetVector<Function *, std::deque<Function *>> workList;
    workList.insert(interface->getSampleFunction());
    generativeFunctions.insert(interface->getSampleFunction());

    while (!workList.empty()) {
      auto todo = *workList.begin();
      workList.erase(workList.begin());

#if LLVM_VERSION_MAJOR > 10
      for (auto &&U : todo->uses()) {
        if (auto ACS = AbstractCallSite(&U)) {
          auto fun = ACS.getInstruction()->getParent()->getParent();
          auto [it, inserted] = generativeFunctions.insert(fun);
          if (inserted)
            workList.insert(fun);
        }
      }
#else
      for (auto &&U : todo->uses()) {
        if (auto &&call = dyn_cast<CallInst>(U.getUser())) {
          auto &&fun = call->getParent()->getParent();
          auto &&[it, inserted] = generativeFunctions.insert(fun);
          if (inserted)
            workList.insert(fun);
        }
      }
#endif
    }

    bool autodiff = dtrace || dobservations;

    auto newFunc =
        Logic.CreateTrace(F, generativeFunctions, mode, autodiff, interface);

    if (!autodiff) {
      auto call = CallInst::Create(newFunc->getFunctionType(), newFunc, args);
      ReplaceInstWithInst(CI, call);
      return true;
    }

    Value *ret = CI;
    Type *retElemType = nullptr;
    if (CI->hasStructRetAttr()) {
      ret = CI->getArgOperand(0);
#if LLVM_VERSION_MAJOR >= 15
      retElemType = CI->getParamStructRetType(0);
#else
      retElemType = ret->getType()->getPointerElementType();
#endif
    }

    bool status = HandleAutoDiff(
        CI, CI->getCallingConv(), ret, retElemType, dargs, byVal, constants,
        newFunc, DerivativeMode::ReverseModeCombined, opt.getValue(), false);

    delete interface;

    return status;
  }

  bool lowerEnzymeCalls(Function &F, std::set<Function *> &done) {
    if (done.count(&F))
      return false;
    done.insert(&F);

    if (F.empty())
      return false;

    bool Changed = false;

    for (BasicBlock &BB : F)
      if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator())) {

        Function *Fn = II->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(II->getCalledOperand()))
#else
        if (auto castinst = dyn_cast<ConstantExpr>(II->getCalledValue()))
#endif
        {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
              Fn = fn;
        }
        if (!Fn)
          continue;

        if (!(Fn->getName().contains("__enzyme_float") ||
              Fn->getName().contains("__enzyme_double") ||
              Fn->getName().contains("__enzyme_integer") ||
              Fn->getName().contains("__enzyme_pointer") ||
              Fn->getName().contains("__enzyme_virtualreverse") ||
              Fn->getName().contains("__enzyme_call_inactive") ||
              Fn->getName().contains("__enzyme_autodiff") ||
              Fn->getName().contains("__enzyme_fwddiff") ||
              Fn->getName().contains("__enzyme_fwdsplit") ||
              Fn->getName().contains("__enzyme_augmentfwd") ||
              Fn->getName().contains("__enzyme_augmentsize") ||
              Fn->getName().contains("__enzyme_reverse") ||
              Fn->getName().contains("__enzyme_batch") ||
              Fn->getName().contains("__enzyme_trace") ||
              Fn->getName().contains("__enzyme_condition")))
          continue;

        SmallVector<Value *, 16> CallArgs(II->arg_begin(), II->arg_end());
        SmallVector<OperandBundleDef, 1> OpBundles;
        II->getOperandBundlesAsDefs(OpBundles);
// Insert a normal call instruction...
#if LLVM_VERSION_MAJOR >= 8
        CallInst *NewCall =
            CallInst::Create(II->getFunctionType(), II->getCalledOperand(),
                             CallArgs, OpBundles, "", II);
#else
        CallInst *NewCall =
            CallInst::Create(II->getFunctionType(), II->getCalledValue(),
                             CallArgs, OpBundles, "", II);
#endif
        NewCall->takeName(II);
        NewCall->setCallingConv(II->getCallingConv());
        NewCall->setAttributes(II->getAttributes());
        NewCall->setDebugLoc(II->getDebugLoc());
        II->replaceAllUsesWith(NewCall);

        // Insert an unconditional branch to the normal destination.
        BranchInst::Create(II->getNormalDest(), II);

        // Remove any PHI node entries from the exception destination.
        II->getUnwindDest()->removePredecessor(&BB);

        // Remove the invoke instruction now.
        BB.getInstList().erase(II);
        Changed = true;
      }

    MapVector<CallInst *, DerivativeMode> toLower;
    MapVector<CallInst *, DerivativeMode> toVirtual;
    MapVector<CallInst *, DerivativeMode> toSize;
    SmallVector<CallInst *, 4> toBatch;
    MapVector<CallInst *, ProbProgMode> toProbProg;
    SetVector<CallInst *> InactiveCalls;
    SetVector<CallInst *> IterCalls;
  retry:;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);

        if (!CI)
          continue;

        Function *Fn = CI->getCalledFunction();

#if LLVM_VERSION_MAJOR >= 11
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
        if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
        {
          if (castinst->isCast())
            if (auto fn = dyn_cast<Function>(castinst->getOperand(0)))
              Fn = fn;
        }

        if (!Fn)
          continue;

#if LLVM_VERSION_MAJOR >= 14
        size_t num_args = CI->arg_size();
#else
        size_t num_args = CI->getNumArgOperands();
#endif

        if (Fn->getName().contains("__enzyme_todense")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("__enzyme_float")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_integer")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_double")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_pointer")) {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_virtualreverse")) {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("__enzyme_iter")) {
          Fn->addFnAttr(Attribute::ReadNone);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("__enzyme_call_inactive")) {
          InactiveCalls.insert(CI);
        }
        if (Fn->getName() == "omp_get_max_threads" ||
            Fn->getName() == "omp_get_thread_num") {
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
        }
        if ((Fn->getName() == "cblas_ddot" || Fn->getName() == "cblas_sdot") &&
            Fn->isDeclaration()) {
          Fn->addFnAttr(Attribute::ReadOnly);
          Fn->addFnAttr(Attribute::ArgMemOnly);
          CI->addParamAttr(1, Attribute::ReadOnly);
          CI->addParamAttr(1, Attribute::NoCapture);
          CI->addParamAttr(3, Attribute::ReadOnly);
          CI->addParamAttr(3, Attribute::NoCapture);
        }
        if (Fn->getName() == "frexp" || Fn->getName() == "frexpf" ||
            Fn->getName() == "frexpl") {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
          CI->addParamAttr(1, Attribute::WriteOnly);
        }
        if (Fn->getName() == "__fd_sincos_1" || Fn->getName() == "__fd_cos_1" ||
            Fn->getName() == "__mth_i_ipowi") {
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
        }
        if (Fn->getName().contains("strcmp")) {
          Fn->addParamAttr(0, Attribute::ReadOnly);
          Fn->addParamAttr(1, Attribute::ReadOnly);
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
        }
        if (Fn->getName() == "f90io_fmtw_end" ||
            Fn->getName() == "f90io_unf_end") {
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
        }
        if (Fn->getName() == "f90io_open2003a") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i : {0, 1, 2, 3, 4, 5, 6, 7, /*8, */ 9, 10, 11, 12, 13}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }
          // todo more
          for (size_t i : {0, 1}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName() == "f90io_fmtw_inita") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 2}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 2}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn->getName() == "f90io_unf_init") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0, 1, 2, 3}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        if (Fn->getName() == "f90io_src_info03a") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          // todo more
          for (size_t i : {0, 1}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
            }
          }

          // todo more
          for (size_t i : {0}) {
            if (i < num_args &&
                CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName() == "f90io_sc_d_fmt_write" ||
            Fn->getName() == "f90io_sc_i_fmt_write" ||
            Fn->getName() == "ftnio_fmt_write64" ||
            Fn->getName() == "f90io_fmt_write64_aa" ||
            Fn->getName() == "f90io_fmt_writea" ||
            Fn->getName() == "f90io_unf_writea" ||
            Fn->getName() == "f90_pausea") {
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadOnly);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }

        bool enableEnzyme = false;
        bool virtualCall = false;
        bool sizeOnly = false;
        bool batch = false;
        bool probProg = false;
        DerivativeMode derivativeMode;
        ProbProgMode probProgMode;
        if (Fn->getName().contains("__enzyme_autodiff")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ReverseModeCombined;
        } else if (Fn->getName().contains("__enzyme_fwddiff")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ForwardMode;
        } else if (Fn->getName().contains("__enzyme_fwdsplit")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ForwardModeSplit;
        } else if (Fn->getName().contains("__enzyme_augmentfwd")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ReverseModePrimal;
        } else if (Fn->getName().contains("__enzyme_augmentsize")) {
          enableEnzyme = true;
          sizeOnly = true;
          derivativeMode = DerivativeMode::ReverseModePrimal;
        } else if (Fn->getName().contains("__enzyme_reverse")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ReverseModeGradient;
        } else if (Fn->getName().contains("__enzyme_virtualreverse")) {
          enableEnzyme = true;
          virtualCall = true;
          derivativeMode = DerivativeMode::ReverseModeCombined;
        } else if (Fn->getName().contains("__enzyme_batch")) {
          enableEnzyme = true;
          batch = true;
        } else if (Fn->getName().contains("__enzyme_trace")) {
          enableEnzyme = true;
          probProgMode = ProbProgMode::Trace;
          probProg = true;
        } else if (Fn->getName().contains("__enzyme_condition")) {
          enableEnzyme = true;
          probProgMode = ProbProgMode::Condition;
          probProg = true;
        }

        if (enableEnzyme) {

          Value *fn = CI->getArgOperand(0);
          while (auto ci = dyn_cast<CastInst>(fn)) {
            fn = ci->getOperand(0);
          }
          while (auto ci = dyn_cast<BlockAddress>(fn)) {
            fn = ci->getFunction();
          }
          while (auto ci = dyn_cast<ConstantExpr>(fn)) {
            fn = ci->getOperand(0);
          }
          if (auto si = dyn_cast<SelectInst>(fn)) {
            BasicBlock *post = BB.splitBasicBlock(CI);
            BasicBlock *sel1 = BasicBlock::Create(BB.getContext(), "sel1", &F);
            BasicBlock *sel2 = BasicBlock::Create(BB.getContext(), "sel2", &F);
            BB.getTerminator()->eraseFromParent();
            IRBuilder<> PB(&BB);
            PB.CreateCondBr(si->getCondition(), sel1, sel2);
            IRBuilder<> S1(sel1);
            auto B1 = S1.CreateBr(post);
            CallInst *cloned = cast<CallInst>(CI->clone());
            cloned->insertBefore(B1);
            cloned->setOperand(0, si->getTrueValue());
            IRBuilder<> S2(sel2);
            auto B2 = S2.CreateBr(post);
            CI->moveBefore(B2);
            CI->setOperand(0, si->getFalseValue());
            if (CI->getNumUses() != 0) {
              IRBuilder<> P(post->getFirstNonPHI());
              auto merge = P.CreatePHI(CI->getType(), 2);
              merge->addIncoming(cloned, sel1);
              merge->addIncoming(CI, sel2);
              CI->replaceAllUsesWith(merge);
            }
            goto retry;
          }
          if (virtualCall)
            toVirtual[CI] = derivativeMode;
          else if (sizeOnly)
            toSize[CI] = derivativeMode;
          else if (batch)
            toBatch.push_back(CI);
          else if (probProg) {
            toProbProg[CI] = probProgMode;
          } else
            toLower[CI] = derivativeMode;

          if (auto dc = dyn_cast<Function>(fn)) {
            // Force postopt on any inner functions in the nested
            // AD case.
            bool tmp = Logic.PostOpt;
            Logic.PostOpt = true;
            Changed |= lowerEnzymeCalls(*dc, done);
            Logic.PostOpt = tmp;
          }
        }
      }
    }

    for (auto CI : InactiveCalls) {
      IRBuilder<> B(CI);
      Value *fn = CI->getArgOperand(0);
      SmallVector<Value *, 4> Args;
      SmallVector<Type *, 4> ArgTypes;
#if LLVM_VERSION_MAJOR >= 14
      for (size_t i = 1; i < CI->arg_size(); ++i)
#else
      for (size_t i = 1; i < CI->getNumArgOperands(); ++i)
#endif
      {
        Args.push_back(CI->getArgOperand(i));
        ArgTypes.push_back(CI->getArgOperand(i)->getType());
      }
      auto FT = FunctionType::get(CI->getType(), ArgTypes, /*varargs*/ false);
      if (fn->getType() != FT) {
        fn = B.CreatePointerCast(fn, PointerType::getUnqual(FT));
      }
      auto Rep = B.CreateCall(FT, fn, Args);
      Rep->addAttribute(AttributeList::FunctionIndex,
                        Attribute::get(Rep->getContext(), "enzyme_inactive"));
      CI->replaceAllUsesWith(Rep);
      CI->eraseFromParent();
      Changed = true;
    }

    // Perform all the size replacements first to create constants
    for (auto pair : toSize) {
      bool successful = HandleAutoDiffArguments(pair.first, pair.second,
                                                /*sizeOnly*/ true);
      Changed = true;
      if (!successful)
        break;
    }
    for (auto pair : toLower) {
      bool successful = HandleAutoDiffArguments(pair.first, pair.second,
                                                /*sizeOnly*/ false);
      Changed = true;
      if (!successful)
        break;
    }

    for (auto pair : toVirtual) {
      auto CI = pair.first;
      Constant *fn = dyn_cast<Constant>(CI->getArgOperand(0));
      if (!fn) {
        EmitFailure("IllegalVirtual", CI->getDebugLoc(), CI,
                    "Cannot create virtual version of non-constant value ", *CI,
                    *CI->getArgOperand(0));
        return false;
      }
      TypeAnalysis TA(Logic.PPC.FAM);

      auto Arch =
          llvm::Triple(
              CI->getParent()->getParent()->getParent()->getTargetTriple())
              .getArch();

      bool AtomicAdd = Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
                       Arch == Triple::amdgcn;

      auto val = GradientUtils::GetOrCreateShadowConstant(
          Logic, Logic.PPC.FAM.getResult<TargetLibraryAnalysis>(F), TA, fn,
          pair.second, /*width*/ 1, AtomicAdd);
      CI->replaceAllUsesWith(ConstantExpr::getPointerCast(val, CI->getType()));
      CI->eraseFromParent();
      Changed = true;
    }

    for (auto call : toBatch) {
      HandleBatch(call);
    }

    for (auto &&[call, mode] : toProbProg) {
      HandleProbProg(call, mode);
    }

    if (Changed && EnzymeAttributor) {
      // TODO consider enabling when attributor does not delete
      // dead internal functions, which invalidates Enzyme's cache
      // code left here to re-enable upon Attributor patch

#if LLVM_VERSION_MAJOR >= 13 && !defined(FLANG) && !defined(ROCM)

      AnalysisGetter AG(Logic.PPC.FAM);
      SetVector<Function *> Functions;
      for (Function &F2 : *F.getParent()) {
        Functions.insert(&F2);
      }

      CallGraphUpdater CGUpdater;
      BumpPtrAllocator Allocator;
      InformationCache InfoCache(*F.getParent(), AG, Allocator,
                                 /* CGSCC */ nullptr);

      DenseSet<const char *> Allowed = {
          &AAHeapToStack::ID,     &AANoCapture::ID,

          &AAMemoryBehavior::ID,  &AAMemoryLocation::ID, &AANoUnwind::ID,
          &AANoSync::ID,          &AANoRecurse::ID,      &AAWillReturn::ID,
          &AANoReturn::ID,        &AANonNull::ID,        &AANoAlias::ID,
          &AADereferenceable::ID, &AAAlign::ID,

          &AAReturnedValues::ID,  &AANoFree::ID,         &AANoUndef::ID,

          //&AAValueSimplify::ID,
          //&AAReachability::ID,
          //&AAValueConstantRange::ID,
          //&AAUndefinedBehavior::ID,
          //&AAPotentialValues::ID,
      };

#if LLVM_VERSION_MAJOR >= 15
      AttributorConfig aconfig(CGUpdater);
      aconfig.Allowed = &Allowed;
      aconfig.DeleteFns = false;
      Attributor A(Functions, InfoCache, aconfig);
#else

      Attributor A(Functions, InfoCache, CGUpdater, &Allowed,
                   /*DeleteFns*/ false);
#endif
      for (Function *F : Functions) {
        // Populate the Attributor with abstract attribute opportunities in
        // the function and the information cache with IR information.
        A.identifyDefaultAbstractAttributes(*F);
      }
      A.run();
#endif
    }

    return Changed;
  }

  bool run(Module &M) {
    Logic.clear();

    bool changed = false;
    for (Function &F : M) {
      attributeKnownFunctions(F);
      if (F.empty())
        continue;
      SmallVector<Instruction *, 4> toErase;
      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F && F->getName() == "f90_mzero8") {
              toErase.push_back(CI);
              IRBuilder<> B(CI);

              SmallVector<Value *, 4> args;
              args.push_back(CI->getArgOperand(0));
              args.push_back(
                  ConstantInt::get(Type::getInt8Ty(M.getContext()), 0));
              args.push_back(B.CreateMul(
                  CI->getArgOperand(1),
                  ConstantInt::get(CI->getArgOperand(1)->getType(), 8)));
#if LLVM_VERSION_MAJOR <= 6
              args.push_back(
                  ConstantInt::get(Type::getInt32Ty(M.getContext()), 1U));
#endif
              args.push_back(ConstantInt::getFalse(M.getContext()));

              Type *tys[] = {args[0]->getType(), args[2]->getType()};
              auto memsetIntr =
                  Intrinsic::getDeclaration(&M, Intrinsic::memset, tys);
              B.CreateCall(memsetIntr, args);
            }
          }
        }
      }
      for (Instruction *I : toErase) {
        I->eraseFromParent();
      }
    }

#if LLVM_VERSION_MAJOR >= 13
    if (Logic.PostOpt && EnzymeOMPOpt) {
      OpenMPOptPass().run(M, Logic.PPC.MAM);
      /// Attributor is run second time for promoted args to get attributes.
      AttributorPass().run(M, Logic.PPC.MAM);
      for (auto &F : M)
        if (!F.empty())
          PromotePass().run(F, Logic.PPC.FAM);
      changed = true;
    }
#endif

    std::set<Function *> done;
    for (Function &F : M) {
      if (F.empty())
        continue;

      changed |= lowerEnzymeCalls(F, done);
    }

    SmallVector<CallInst *, 4> toErase;
    for (Function &F : M) {
      if (F.empty())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
#if LLVM_VERSION_MAJOR >= 11
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
#else
            if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
#endif
            {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F) {
              if (F->getName().contains("__enzyme_float") ||
                  F->getName().contains("__enzyme_double") ||
                  F->getName().contains("__enzyme_integer") ||
                  F->getName().contains("__enzyme_pointer")) {
                toErase.push_back(CI);
              }
              if (F->getName() == "__enzyme_iter") {
                CI->replaceAllUsesWith(CI->getArgOperand(0));
                toErase.push_back(CI);
              }
            }
          }
        }
      }
    }
    for (auto I : toErase) {
      I->eraseFromParent();
      changed = true;
    }

    SmallPtrSet<CallInst *, 4> sample_calls;
    for (auto &&func : M) {
      for (auto &&BB : func) {
        for (auto &&Inst : BB) {
          if (auto CI = dyn_cast<CallInst>(&Inst)) {
            Function *enzyme_sample = CI->getCalledFunction();
            if (enzyme_sample && enzyme_sample->getName().contains(
                                     TraceInterface::sampleFunctionName)) {
              if (CI->getNumOperands() < 3) {
                EmitFailure(
                    "IllegalNumberOfArguments", CI->getDebugLoc(), CI,
                    "Not enough arguments passed to call to __enzyme_sample");
              }
              Function *samplefn = GetFunctionFromValue(CI->getOperand(0));
              unsigned expected =
                  samplefn->getFunctionType()->getNumParams() + 3;
#if LLVM_VERSION_MAJOR >= 14
              unsigned actual = CI->arg_size();
#else
              unsigned actual = CI->getNumArgOperands();
#endif
              if (actual - 3 != samplefn->getFunctionType()->getNumParams()) {
                EmitFailure("IllegalNumberOfArguments", CI->getDebugLoc(), CI,
                            "Illegal number of arguments passed to call to "
                            "__enzyme_sample.",
                            " Expected: ", expected, " got: ", actual);
              }
              Function *pdf = GetFunctionFromValue(CI->getArgOperand(1));

              for (unsigned i = 0;
                   i < samplefn->getFunctionType()->getNumParams(); ++i) {
                Value *ci_arg = CI->getArgOperand(i + 3);
                Value *sample_arg = samplefn->arg_begin() + i;
                Value *pdf_arg = pdf->arg_begin() + i;

                if (ci_arg->getType() != sample_arg->getType()) {
                  EmitFailure(
                      "IllegalSampleType", CI->getDebugLoc(), CI,
                      "Type of: ", *ci_arg, " (", *ci_arg->getType(), ")",
                      " does not match the argument type of the sample "
                      "function: ",
                      *samplefn, " at: ", i, " (", *sample_arg->getType(), ")");
                }
                if (ci_arg->getType() != pdf_arg->getType()) {
                  EmitFailure("IllegalSampleType", CI->getDebugLoc(), CI,
                              "Type of: ", *ci_arg, " (", *ci_arg->getType(),
                              ")",
                              " does not match the argument type of the "
                              "density function: ",
                              *pdf, " at: ", i, " (", *pdf_arg->getType(), ")");
                }
              }

              if ((pdf->arg_end() - 1)->getType() !=
                  samplefn->getReturnType()) {
                EmitFailure(
                    "IllegalSampleType", CI->getDebugLoc(), CI,
                    "Return type of ", *samplefn, " (",
                    *samplefn->getReturnType(), ")",
                    " does not match the last argument type of the density "
                    "function: ",
                    *pdf, " (", *(pdf->arg_end() - 1)->getType(), ")");
              }
              sample_calls.insert(CI);
            }
          }
        }
      }
    }

    // Replace calls to __enzyme_sample with the actual sample calls after
    // running prob prog
    for (auto call : sample_calls) {
      Function *samplefn = GetFunctionFromValue(call->getArgOperand(0));

      SmallVector<Value *, 2> args;
      for (auto it = call->arg_begin() + 3; it != call->arg_end(); it++) {
        args.push_back(*it);
      }
      CallInst *choice =
          CallInst::Create(samplefn->getFunctionType(), samplefn, args);

      ReplaceInstWithInst(call, choice);
    }

    for (const auto &pair : Logic.PPC.cache)
      pair.second->eraseFromParent();
    Logic.clear();

    if (changed && Logic.PostOpt) {
      PassBuilder PB;
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;
      PB.registerModuleAnalyses(MAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
#if LLVM_VERSION_MAJOR >= 14
      auto PM = PB.buildModuleSimplificationPipeline(OptimizationLevel::O2,
                                                     ThinOrFullLTOPhase::None);
#elif LLVM_VERSION_MAJOR >= 12
      auto PM = PB.buildModuleSimplificationPipeline(
          PassBuilder::OptimizationLevel::O2, ThinOrFullLTOPhase::None);
#else
      auto PM = PB.buildModuleSimplificationPipeline(
          PassBuilder::OptimizationLevel::O2, PassBuilder::ThinLTOPhase::None);
#endif
      PM.run(M, MAM);
#if LLVM_VERSION_MAJOR >= 13
      if (EnzymeOMPOpt) {
        OpenMPOptPass().run(M, MAM);
        /// Attributor is run second time for promoted args to get attributes.
        AttributorPass().run(M, MAM);
        for (auto &F : M)
          if (!F.empty())
            PromotePass().run(F, FAM);
      }
#endif
    }

    for (auto &F : M) {
      if (!F.empty())
        changed |= LowerSparsification(&F);
    }
    return changed;
  }
};

class EnzymeOldPM : public EnzymeBase, public ModulePass {
public:
  static char ID;
  EnzymeOldPM(bool PostOpt = false) : EnzymeBase(PostOpt), ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();

    // AU.addRequiredID(LCSSAID);

    // LoopInfo is required to ensure that all loops have preheaders
    // AU.addRequired<LoopInfoWrapperPass>();

    // AU.addRequiredID(llvm::LoopSimplifyID);//<LoopSimplifyWrapperPass>();
  }
  bool runOnModule(Module &M) override { return run(M); }
};

} // namespace

char EnzymeOldPM::ID = 0;

static RegisterPass<EnzymeOldPM> X("enzyme", "Enzyme Pass");

ModulePass *createEnzymePass(bool PostOpt) { return new EnzymeOldPM(PostOpt); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddEnzymePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEnzymePass(/*PostOpt*/ false));
}

#include "llvm/Passes/PassPlugin.h"

class EnzymeNewPM final : public EnzymeBase,
                          public AnalysisInfoMixin<EnzymeNewPM> {
  friend struct llvm::AnalysisInfoMixin<EnzymeNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  EnzymeNewPM(bool PostOpt = false) : EnzymeBase(PostOpt) {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
    return EnzymeBase::run(M) ? PreservedAnalyses::none()
                              : PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

#undef DEBUG_TYPE
AnalysisKey EnzymeNewPM::Key;

#include "PreserveNVVM.h"
#ifdef ENZYME_RUNPASS
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SROA.h"

#if LLVM_VERSION_MAJOR >= 12
#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
namespace llvm {
// extern cl::opt<bool> EnableMatrix;
#define EnableMatrix false
#if LLVM_VERSION_MAJOR <= 14
// extern cl::opt<bool> EnableFunctionSpecialization;
#define EnableFunctionSpecialization false
// extern cl::opt<bool> RunPartialInlining;
#define RunPartialInlining false
#endif
} // namespace llvm
#if LLVM_VERSION_MAJOR <= 14
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#if LLVM_VERSION_MAJOR >= 12
#include "llvm/Transforms/Coroutines/CoroCleanup.h"
#endif
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/PartialInlining.h"
#if LLVM_VERSION_MAJOR <= 12
#include "llvm/Transforms/Utils/Mem2Reg.h"
#endif
#endif
#endif
#endif

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "EnzymeNewPM", "v0.1",
          [](llvm::PassBuilder &PB) {
#ifdef ENZYME_RUNPASS
#if LLVM_VERSION_MAJOR < 14
            using OptimizationLevel = llvm::PassBuilder::OptimizationLevel;
#endif

            auto PB0 = new llvm::PassBuilder(PB);
#if LLVM_VERSION_MAJOR >= 12
            auto prePass =
                [PB0](ModulePassManager &MPM, OptimizationLevel Level)
#else
            auto prePass = [PB0](ModulePassManager &MPM)
#endif
            {

#if LLVM_VERSION_MAJOR < 12
              llvm_unreachable(
                  "New Pass manager pipeline unsupported at version <= 11");
#else
#if LLVM_VERSION_MAJOR < 15
    ////// End of Module simplification
    // Specialize functions with IPSCCP.
#if LLVM_VERSION_MAJOR >= 13
              if (EnableFunctionSpecialization &&
                  Level == OptimizationLevel::O3)
                MPM.addPass(FunctionSpecializationPass());
#endif

              // Interprocedural constant propagation now that basic cleanup has
              // occurred and prior to optimizing globals.
              // FIXME: This position in the pipeline hasn't been carefully
              // considered in years, it should be re-analyzed.
              MPM.addPass(IPSCCPPass());

              // Attach metadata to indirect call sites indicating the set of
              // functions they may target at run-time. This should follow
              // IPSCCP.
              MPM.addPass(CalledValuePropagationPass());

              // Optimize globals to try and fold them into constants.
              MPM.addPass(GlobalOptPass());

              // Promote any localized globals to SSA registers.
              // FIXME: Should this instead by a run of SROA?
              // FIXME: We should probably run instcombine and simplifycfg
              // afterward to delete control flows that are dead once globals
              // have been folded to constants.
              MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

              // Remove any dead arguments exposed by cleanups and constant
              // folding globals.
              MPM.addPass(DeadArgumentEliminationPass());

              // Create a small function pass pipeline to cleanup after all the
              // global optimizations.
              FunctionPassManager GlobalCleanupPM;
              GlobalCleanupPM.addPass(InstCombinePass());

#if LLVM_VERSION_MAJOR >= 14
              GlobalCleanupPM.addPass(SimplifyCFGPass(
                  SimplifyCFGOptions().convertSwitchRangeToICmp(true)));
#else
              GlobalCleanupPM.addPass(SimplifyCFGPass(SimplifyCFGOptions()));
#endif
              MPM.addPass(createModuleToFunctionPassAdaptor(
                  std::move(GlobalCleanupPM)));

              ThinOrFullLTOPhase Phase = ThinOrFullLTOPhase::None;
#if LLVM_VERSION >= 13
              bool EnableModuleInliner = false;
              if (EnableModuleInliner)
                MPM.addPass(PB0->buildModuleInlinerPipeline(Level, Phase));
              else
#endif
                MPM.addPass(PB0->buildInlinerPipeline(Level, Phase));

              FunctionPassManager CoroCleanupPM;
              CoroCleanupPM.addPass(CoroCleanupPass());
              MPM.addPass(
                  createModuleToFunctionPassAdaptor(std::move(CoroCleanupPM)));

              ////// Finished Module simplification, starting ModuleOptimization
              //
              // Optimize globals now that the module is fully simplified.
              MPM.addPass(GlobalOptPass());
              MPM.addPass(GlobalDCEPass());

              // Run partial inlining pass to partially inline functions that
              // have large bodies.
              if (RunPartialInlining)
                MPM.addPass(PartialInlinerPass());

              // Do RPO function attribute inference across the module to
              // forward-propagate attributes where applicable.
              // FIXME: Is this really an optimization rather than a
              // canonicalization?
              MPM.addPass(ReversePostOrderFunctionAttrsPass());
#endif
              FunctionPassManager OptimizePM;
              OptimizePM.addPass(Float2IntPass());
              OptimizePM.addPass(LowerConstantIntrinsicsPass());

              if (EnableMatrix) {
                OptimizePM.addPass(LowerMatrixIntrinsicsPass());
                OptimizePM.addPass(EarlyCSEPass());
              }

              LoopPassManager LPM;
              bool LTOPreLink = false;
      // First rotate loops that may have been un-rotated by prior passes.
      // Disable header duplication at -Oz.
#if LLVM_VERSION_MAJOR >= 11
              LPM.addPass(
                  LoopRotatePass(Level != OptimizationLevel::Oz, LTOPreLink));
#endif
              // Some loops may have become dead by now. Try to delete them.
              // FIXME: see discussion in https://reviews.llvm.org/D112851,
              //        this may need to be revisited once we run GVN before
              //        loop deletion in the simplification pipeline.
              LPM.addPass(LoopDeletionPass());

              LPM.addPass(llvm::LoopFullUnrollPass());
              OptimizePM.addPass(
                  createFunctionToLoopPassAdaptor(std::move(LPM)));

              MPM.addPass(
                  createModuleToFunctionPassAdaptor(std::move(OptimizePM)));
#endif
            };

#if LLVM_VERSION_MAJOR >= 12
            auto loadPass =
                [prePass](ModulePassManager &MPM, OptimizationLevel Level)
#else
            auto loadPass = [prePass](ModulePassManager &MPM)
#endif
            {
              MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));

#if LLVM_VERSION_MAJOR >= 12
              if (Level != OptimizationLevel::O0)
                prePass(MPM, Level);
#else
              prePass(MPM);
#endif
              FunctionPassManager OptimizerPM;
              FunctionPassManager OptimizerPM2;
#if LLVM_VERSION_MAJOR >= 14
              OptimizerPM.addPass(llvm::GVNPass());
              OptimizerPM.addPass(llvm::SROAPass());
#else
              OptimizerPM.addPass(llvm::GVN());
              OptimizerPM.addPass(llvm::SROA());
#endif
              MPM.addPass(
                  createModuleToFunctionPassAdaptor(std::move(OptimizerPM)));
              MPM.addPass(EnzymeNewPM(/*PostOpt=*/true));
              MPM.addPass(PreserveNVVMNewPM(/*Begin*/ false));
#if LLVM_VERSION_MAJOR >= 14
              OptimizerPM2.addPass(llvm::GVNPass());
              OptimizerPM2.addPass(llvm::SROAPass());
#else
              OptimizerPM2.addPass(llvm::GVN());
              OptimizerPM2.addPass(llvm::SROA());
#endif

              LoopPassManager LPM1;
              LPM1.addPass(LoopDeletionPass());
              OptimizerPM2.addPass(
                  createFunctionToLoopPassAdaptor(std::move(LPM1)));

              MPM.addPass(
                  createModuleToFunctionPassAdaptor(std::move(OptimizerPM2)));
              MPM.addPass(GlobalOptPass());
            };
// TODO need for perf reasons to move Enzyme pass to the pre vectorization.
#if LLVM_VERSION_MAJOR >= 15
            PB.registerOptimizerEarlyEPCallback(loadPass);
#elif LLVM_VERSION_MAJOR >= 12
            PB.registerPipelineEarlySimplificationEPCallback(loadPass);
#else
            PB.registerPipelineStartEPCallback(loadPass);
#endif

#if LLVM_VERSION_MAJOR >= 12
            auto loadNVVM = [](ModulePassManager &MPM, OptimizationLevel)
#else
            auto loadNVVM = [](ModulePassManager &MPM)
#endif
            { MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true)); };

            // We should register at vectorizer start for consistency, however,
            // that requires a functionpass, and we have a modulepass.
            // PB.registerVectorizerStartEPCallback(loadPass);
            PB.registerPipelineStartEPCallback(loadNVVM);
#if LLVM_VERSION_MAJOR >= 15
            PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadNVVM);
#endif
#endif
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "enzyme") {
                    MPM.addPass(EnzymeNewPM());
                    return true;
                  }
                  if (Name == "preserve-nvvm") {
                    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));
                    return true;
                  }
                  return false;
                });
          }};
}
