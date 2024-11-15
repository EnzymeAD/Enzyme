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
#include "llvm/IR/PassManager.h"
#include <llvm/Config/llvm-config.h>
#include <memory>
#include <string>
#include <utility>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include <optional>
#if LLVM_VERSION_MAJOR <= 16
#include "llvm/ADT/Optional.h"
#endif
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "ActivityAnalysis.h"
#include "DiffeGradientUtils.h"
#include "EnzymeLogic.h"
#include "GradientUtils.h"
#include "TraceInterface.h"
#include "TraceUtils.h"
#include "Utils.h"

#include "InstructionBatcher.h"

#include "llvm/Transforms/Utils.h"

#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include "BlasAttributor.inc"

#include "CApi.h"
using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "lower-enzyme-intrinsic"

llvm::cl::opt<bool>
    EnzymeDisablePreOpt("enzyme-disable-preopt", cl::init(true), cl::Hidden,
                        cl::desc("Do not run any pre-processing passes"));

llvm::cl::opt<bool> EnzymeEnable("enzyme-enable", cl::init(true), cl::Hidden,
                                 cl::desc("Run the Enzyme pass"));

llvm::cl::opt<bool>
    EnzymePostOpt("enzyme-postopt", cl::init(false), cl::Hidden,
                  cl::desc("Run enzymepostprocessing optimizations"));

llvm::cl::opt<bool> EnzymeAttributor("enzyme-attributor", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Run attributor post Enzyme"));

llvm::cl::opt<bool> EnzymeOMPOpt("enzyme-omp-opt", cl::init(false), cl::Hidden,
                                 cl::desc("Whether to enable openmp opt"));

llvm::cl::opt<std::string> EnzymeTruncateAll(
    "enzyme-truncate-all", cl::init(""), cl::Hidden,
    cl::desc(
        "Truncate all floating point operations. "
        "E.g. \"64to32\" or \"64to<exponent_width>-<significand_width>\"."));

llvm::cl::opt<bool>
    FPOptExtraMemOpt("fpopt-extra-memopt", cl::init(false), cl::Hidden,
                     cl::desc("Run some memory optimizations to aid the "
                              "flood-fill algo before running FPOpt"));

llvm::cl::opt<bool> FPOptExtraPreReassoc(
    "fpopt-extra-pre-reassoc", cl::init(false), cl::Hidden,
    cl::desc("Run LLVM -reassiociate before running FPOpt"));

llvm::cl::opt<bool> FPOptExtraIfConversion(
    "fpopt-extra-ifconv", cl::init(false), cl::Hidden,
    cl::desc("Push speculative phi node folding to increase number of select "
             "instructions for graph capture"));

llvm::cl::opt<bool> FPOptExtraPreCSE("fpopt-extra-pre-cse", cl::init(false),
                                     cl::Hidden,
                                     cl::desc("Run CSE before FPOpt"));

llvm::cl::opt<bool> FPOptExtraPostCSE("fpopt-extra-post-cse", cl::init(false),
                                      cl::Hidden,
                                      cl::desc("Run CSE after FPOpt"));

#define addAttribute addAttributeAtIndex
#define getAttribute getAttributeAtIndex
bool attributeKnownFunctions(llvm::Function &F) {
  bool changed = false;
  if (F.getName() == "fprintf") {
    for (auto &arg : F.args()) {
      if (arg.getType()->isPointerTy()) {
        arg.addAttr(Attribute::NoCapture);
        changed = true;
      }
    }
  }
  if (F.getName().contains("__enzyme_float") ||
      F.getName().contains("__enzyme_double") ||
      F.getName().contains("__enzyme_integer") ||
      F.getName().contains("__enzyme_pointer") ||
      F.getName().contains("__enzyme_todense") ||
      F.getName().contains("__enzyme_iter") ||
      F.getName().contains("__enzyme_virtualreverse")) {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyReadsMemory();
    F.setOnlyWritesMemory();
#else
    F.addFnAttr(Attribute::ReadNone);
#endif
    if (!F.getName().contains("__enzyme_todense"))
      for (auto &arg : F.args()) {
        if (arg.getType()->isPointerTy()) {
          arg.addAttr(Attribute::ReadNone);
          arg.addAttr(Attribute::NoCapture);
        }
      }
  }
  if (F.getName() == "memcmp") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesArgMemory();
    F.setOnlyReadsMemory();
#else
    F.addFnAttr(Attribute::ArgMemOnly);
    F.addFnAttr(Attribute::ReadOnly);
#endif
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
    for (int i = 0; i < 2; i++)
      if (F.getFunctionType()->getParamType(i)->isPointerTy()) {
        F.addParamAttr(i, Attribute::NoCapture);
        F.addParamAttr(i, Attribute::WriteOnly);
      }
  }

  if (F.getName() ==
      "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm") {
    changed = true;
    F.addFnAttr(Attribute::NoFree);
  }
  if (F.getName() == "MPI_Irecv" || F.getName() == "PMPI_Irecv") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesInaccessibleMemOrArgMem();
#else
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
#endif
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
    F.addParamAttr(0, Attribute::WriteOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::WriteOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Isend" || F.getName() == "PMPI_Isend") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesInaccessibleMemOrArgMem();
#else
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
#endif
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
    F.addParamAttr(0, Attribute::ReadOnly);
    if (F.getFunctionType()->getParamType(2)->isPointerTy()) {
      F.addParamAttr(2, Attribute::NoCapture);
      F.addParamAttr(2, Attribute::ReadOnly);
    }
    F.addParamAttr(6, Attribute::WriteOnly);
  }
  if (F.getName() == "MPI_Comm_rank" || F.getName() == "PMPI_Comm_rank" ||
      F.getName() == "MPI_Comm_size" || F.getName() == "PMPI_Comm_size") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesInaccessibleMemOrArgMem();
#else
    F.addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
#endif
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);

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
    changed = true;
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
    F.addParamAttr(0, Attribute::NoCapture);
    F.addParamAttr(1, Attribute::WriteOnly);
    F.addParamAttr(1, Attribute::NoCapture);
  }
  if (F.getName() == "MPI_Waitall" || F.getName() == "PMPI_Waitall") {
    changed = true;
    F.addFnAttr(Attribute::NoUnwind);
    F.addFnAttr(Attribute::NoRecurse);
    F.addFnAttr(Attribute::WillReturn);
    F.addFnAttr(Attribute::NoFree);
    F.addFnAttr(Attribute::NoSync);
    F.addParamAttr(1, Attribute::NoCapture);
    F.addParamAttr(2, Attribute::WriteOnly);
    F.addParamAttr(2, Attribute::NoCapture);
  }
  // Map of MPI function name to the arg index of its type argument
  std::map<std::string, int> MPI_TYPE_ARGS = {
      {"MPI_Send", 2},      {"MPI_Ssend", 2},     {"MPI_Bsend", 2},
      {"MPI_Recv", 2},      {"MPI_Brecv", 2},     {"PMPI_Send", 2},
      {"PMPI_Ssend", 2},    {"PMPI_Bsend", 2},    {"PMPI_Recv", 2},
      {"PMPI_Brecv", 2},

      {"MPI_Isend", 2},     {"MPI_Irecv", 2},     {"PMPI_Isend", 2},
      {"PMPI_Irecv", 2},

      {"MPI_Reduce", 3},    {"PMPI_Reduce", 3},

      {"MPI_Allreduce", 3}, {"PMPI_Allreduce", 3}};
  {
    auto found = MPI_TYPE_ARGS.find(F.getName().str());
    if (found != MPI_TYPE_ARGS.end()) {
      for (auto user : F.users()) {
        if (auto CI = dyn_cast<CallBase>(user))
          if (CI->getCalledFunction() == &F) {
            if (Constant *C =
                    dyn_cast<Constant>(CI->getArgOperand(found->second))) {
              while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
                C = CE->getOperand(0);
              }
              if (auto GV = dyn_cast<GlobalVariable>(C)) {
                if (GV->getName() == "ompi_mpi_cxx_bool") {
                  changed = true;
                  CI->addAttribute(
                      AttributeList::FunctionIndex,
                      Attribute::get(CI->getContext(), "enzyme_inactive"));
                }
              }
            }
          }
      }
    }
  }

  if (F.getName() == "omp_get_max_threads" ||
      F.getName() == "omp_get_thread_num") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesInaccessibleMemory();
    F.setOnlyReadsMemory();
#else
    F.addFnAttr(Attribute::InaccessibleMemOnly);
    F.addFnAttr(Attribute::ReadOnly);
#endif
  }
  if (F.getName() == "frexp" || F.getName() == "frexpf" ||
      F.getName() == "frexpl") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyAccessesArgMemory();
#else
    F.addFnAttr(Attribute::ArgMemOnly);
#endif
    F.addParamAttr(1, Attribute::WriteOnly);
  }
  if (F.getName() == "__fd_sincos_1" || F.getName() == "__fd_cos_1" ||
      F.getName() == "__mth_i_ipowi") {
    changed = true;
#if LLVM_VERSION_MAJOR >= 16
    F.setOnlyReadsMemory();
    F.setOnlyWritesMemory();
#else
    F.addFnAttr(Attribute::ReadNone);
#endif
  }
  auto name = F.getName();

  const char *NonEscapingFns[] = {
      "julia.ptls_states",
      "julia.get_pgcstack",
      "lgamma_r",
      "memcmp",
      "_ZNSt6chrono3_V212steady_clock3nowEv",
      "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_"
      "createERmm",
      "_ZNKSt8__detail20_Prime_rehash_policy14_M_need_rehashEmmm",
      "fprintf",
      "fwrite",
      "fputc",
      "strtol",
      "getenv",
      "memchr",
      "cublasSetMathMode",
      "cublasSetStream_v2",
      "cuMemPoolTrimTo",
      "cuDeviceGetMemPool",
      "cuStreamSynchronize",
      "cuStreamDestroy",
      "cuStreamQuery",
      "cuCtxGetCurrent",
      "cuDeviceGet",
      "cuDeviceGetName",
      "cuDriverGetVersion",
      "cudaRuntimeGetVersion",
      "cuDeviceGetCount",
      "cuMemPoolGetAttribute",
      "cuMemGetInfo_v2",
      "cuDeviceGetAttribute",
      "cuDevicePrimaryCtxRetain",
  };
  for (auto fname : NonEscapingFns)
    if (name == fname) {
      changed = true;
      F.addAttribute(
          AttributeList::FunctionIndex,
          Attribute::get(F.getContext(), "enzyme_no_escaping_allocation"));
    }
  changed |= attributeTablegen(F);
  return changed;
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
#if LLVM_VERSION_MAJOR < 17
        if (CI->getContext().supportsTypedPointers()) {
          res = Builder.CreateAddrSpaceCast(
              res, PointerType::get(ptr->getPointerElementType(),
                                    PT->getAddressSpace()));
        } else {
          res = Builder.CreateAddrSpaceCast(res, PT);
        }
#else
        res = Builder.CreateAddrSpaceCast(res, PT);
#endif
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

#if LLVM_VERSION_MAJOR > 16
static std::optional<StringRef> getMetadataName(llvm::Value *res);
#else
static Optional<StringRef> getMetadataName(llvm::Value *res);
#endif

// if all phi arms are (recursively) based on the same metaString, use that
#if LLVM_VERSION_MAJOR > 16
static std::optional<StringRef> recursePhiReads(PHINode *val)
#else
static Optional<StringRef> recursePhiReads(PHINode *val)
#endif
{
#if LLVM_VERSION_MAJOR > 16
  std::optional<StringRef> finalMetadata;
#else
  Optional<StringRef> finalMetadata;
#endif
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

#if LLVM_VERSION_MAJOR > 16
std::optional<StringRef> getMetadataName(llvm::Value *res)
#else
Optional<StringRef> getMetadataName(llvm::Value *res)
#endif
{
  if (auto S = simplifyLoad(res))
    return getMetadataName(S);

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
  } else if (isa<PHINode>(res)) {
    return recursePhiReads(cast<PHINode>(res));
  }

  return {};
}

static Value *adaptReturnedVector(Value *ret, Value *diffret,
                                  IRBuilder<> &Builder, unsigned width) {
  Type *returnType = ret->getType();

  if (StructType *sty = dyn_cast<StructType>(returnType)) {
    Value *agg = ConstantAggregateZero::get(sty);

    for (unsigned int i = 0; i < width; i++) {
      Value *elem = Builder.CreateExtractValue(diffret, {i});
      if (auto vty = dyn_cast<FixedVectorType>(elem->getType())) {
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
        Value *sgep = Builder.CreateStructGEP(retType, ret, i);
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

  if ((mode == DerivativeMode::ReverseModePrimal &&
       DL.getTypeSizeInBits(retType) >= DL.getTypeSizeInBits(diffretType)) ||
      ((mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeError) &&
       DL.getTypeSizeInBits(retType) == DL.getTypeSizeInBits(diffretType))) {
    IRBuilder<> EB(CI->getFunction()->getEntryBlock().getFirstNonPHI());
    auto AL = EB.CreateAlloca(retType);
    Builder.CreateStore(diffret, Builder.CreatePointerCast(
                                     AL, PointerType::getUnqual(diffretType)));
    Value *cload = Builder.CreateLoad(retType, AL);
    CI->replaceAllUsesWith(cload);
    CI->eraseFromParent();
    return true;
  }

  if (mode != DerivativeMode::ReverseModePrimal &&
      diffret->getType()->isAggregateType()) {
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

  auto diffretsize = DL.getTypeSizeInBits(diffretType);
  auto retsize = DL.getTypeSizeInBits(retType);
  EmitFailure("IllegalReturnCast", CI->getDebugLoc(), CI,
              "Cannot cast return type of gradient ", *diffretType, *diffret,
              " of size ", diffretsize, " bits ", ", to desired type ",
              *retType, " of size ", retsize, " bits");
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

    Value *ofn = fn;
    fn = GetFunctionFromValue(fn);

    if (!fn || !isa<Function>(fn)) {
      assert(ofn);
      EmitFailure("NoFunctionToDifferentiate", CI->getDebugLoc(), CI,
                  "failed to find fn to differentiate", *CI, " - found - ",
                  *ofn);
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

#if LLVM_VERSION_MAJOR > 16
  static std::optional<unsigned> parseWidthParameter(CallInst *CI)
#else
  static Optional<unsigned> parseWidthParameter(CallInst *CI)
#endif
  {
    unsigned width = 1;

    for (auto [i, found] = std::tuple{0u, false}; i < CI->arg_size(); ++i) {
      Value *arg = CI->getArgOperand(i);

      if (auto MDName = getMetadataName(arg)) {
        if (*MDName == "enzyme_width") {
          if (found) {
            EmitFailure("IllegalVectorWidth", CI->getDebugLoc(), CI,
                        "vector width declared more than once",
                        *CI->getArgOperand(i), " in", *CI);
            return {};
          }

          if (i + 1 >= CI->arg_size()) {
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
    Value *trace;
    Value *observations;
    Value *likelihood;
    Value *diffeLikelihood;
    unsigned width;
    int allocatedTapeSize;
    bool freeMemory;
    bool returnUsed;
    bool tapeIsPointer;
    bool differentialReturn;
    bool diffeTrace;
    DIFFE_TYPE retType;
    bool primalReturn;
    StringSet<> ActiveRandomVariables;
    std::vector<bool> overwritten_args;
    bool runtimeActivity;
  };

#if LLVM_VERSION_MAJOR > 16
  static std::optional<Options>
  handleArguments(IRBuilder<> &Builder, CallInst *CI, Function *fn,
                  DerivativeMode mode, bool sizeOnly,
                  std::vector<DIFFE_TYPE> &constants,
                  SmallVectorImpl<Value *> &args, std::map<int, Type *> &byVal)
#else
  static Optional<Options>
  handleArguments(IRBuilder<> &Builder, CallInst *CI, Function *fn,
                  DerivativeMode mode, bool sizeOnly,
                  std::vector<DIFFE_TYPE> &constants,
                  SmallVectorImpl<Value *> &args, std::map<int, Type *> &byVal)
#endif
  {
    FunctionType *FT = fn->getFunctionType();

    Value *differet = nullptr;
    Value *tape = nullptr;
    Value *dynamic_interface = nullptr;
    Value *trace = nullptr;
    Value *observations = nullptr;
    Value *likelihood = nullptr;
    Value *diffeLikelihood = nullptr;
    unsigned width = 1;
    int allocatedTapeSize = -1;
    bool freeMemory = true;
    bool tapeIsPointer = false;
    bool diffeTrace = false;
    unsigned truei = 0;
    unsigned byRefSize = 0;
    bool primalReturn = false;
    bool runtimeActivity = false;
    StringSet<> ActiveRandomVariables;

    DIFFE_TYPE retType = whatType(fn->getReturnType(), mode);

    if (fn->hasParamAttribute(0, Attribute::StructRet)) {
      Type *Ty = nullptr;
      Ty = fn->getParamAttribute(0, Attribute::StructRet).getValueAsType();
      if (whatType(Ty, mode) != DIFFE_TYPE::CONSTANT) {
        retType = DIFFE_TYPE::DUP_ARG;
      }
    }

    bool returnUsed =
        !fn->getReturnType()->isVoidTy() && !fn->getReturnType()->isEmptyTy();

    bool sret = CI->hasStructRetAttr() ||
                fn->hasParamAttribute(0, Attribute::StructRet);

    std::vector<bool> overwritten_args(
        fn->getFunctionType()->getNumParams(),
        !(mode == DerivativeMode::ReverseModeCombined));

    for (unsigned i = 1 + sret; i < CI->arg_size(); ++i) {
      Value *res = CI->getArgOperand(i);
      auto metaString = getMetadataName(res);
      // handle metadata
      if (metaString && startsWith(*metaString, "enzyme_")) {
        if (*metaString == "enzyme_const_return") {
          retType = DIFFE_TYPE::CONSTANT;
          continue;
        } else if (*metaString == "enzyme_active_return") {
          retType = DIFFE_TYPE::OUT_DIFF;
          continue;
        } else if (*metaString == "enzyme_dup_return") {
          retType = DIFFE_TYPE::DUP_ARG;
          continue;
        } else if (*metaString == "enzyme_noret") {
          returnUsed = false;
          continue;
        } else if (*metaString == "enzyme_primal_return") {
          primalReturn = true;
          continue;
        }
      }
    }
    bool differentialReturn = (mode == DerivativeMode::ReverseModeCombined ||
                               mode == DerivativeMode::ReverseModeGradient) &&
                              (retType == DIFFE_TYPE::OUT_DIFF);

    // find and handle enzyme_width
    if (auto parsedWidth = parseWidthParameter(CI)) {
      width = *parsedWidth;
    } else {
      return {};
    }

    // handle different argument order for struct return.
    if (fn->hasParamAttribute(0, Attribute::StructRet)) {
      truei = 1;

      const DataLayout &DL = CI->getParent()->getModule()->getDataLayout();
      Type *Ty = nullptr;
      Ty = fn->getParamAttribute(0, Attribute::StructRet).getValueAsType();
      Type *CTy = nullptr;
      CTy = CI->getAttribute(AttributeList::FirstArgIndex, Attribute::StructRet)
                .getValueAsType();
      auto FnSize = (DL.getTypeSizeInBits(Ty) / 8);
      auto CSize = CTy ? (DL.getTypeSizeInBits(CTy) / 8) : 0;
      auto count = ((mode == DerivativeMode::ForwardMode ||
                     mode == DerivativeMode::ForwardModeSplit ||
                     mode == DerivativeMode::ForwardModeError) &&
                    (retType == DIFFE_TYPE::DUP_ARG ||
                     retType == DIFFE_TYPE::DUP_NONEED)) *
                       width +
                   primalReturn;
      if (CSize < count * FnSize) {
        EmitFailure(
            "IllegalByRefSize", CI->getDebugLoc(), CI, "Struct return type ",
            *CTy, " (", CSize, " bytes), not large enough to store ", count,
            " returns of type ", *Ty, " (", FnSize, " bytes), width=", width,
            " primal requested=", primalReturn);
      }
      Value *primal = nullptr;
      if (primalReturn) {
        Value *sretPt = CI->getArgOperand(0);
        PointerType *pty = cast<PointerType>(sretPt->getType());
        primal = Builder.CreatePointerCast(
            sretPt, PointerType::get(Ty, pty->getAddressSpace()));
      } else {
        AllocaInst *primalA = new AllocaInst(Ty, DL.getAllocaAddrSpace(),
                                             nullptr, DL.getPrefTypeAlign(Ty));
        primalA->insertBefore(CI);
        primal = primalA;
      }

      Value *shadow = nullptr;
      switch (mode) {
      case DerivativeMode::ForwardModeError:
      case DerivativeMode::ForwardModeSplit:
      case DerivativeMode::ForwardMode: {
        if (retType != DIFFE_TYPE::CONSTANT) {
          Value *sretPt = CI->getArgOperand(0);
          PointerType *pty = cast<PointerType>(sretPt->getType());
          auto shadowPtr = Builder.CreatePointerCast(
              sretPt, PointerType::get(Ty, pty->getAddressSpace()));
          if (width == 1) {
            if (primalReturn)
              shadowPtr = Builder.CreateConstGEP1_64(Ty, shadowPtr, 1);
            shadow = shadowPtr;
          } else {
            Value *acc = UndefValue::get(ArrayType::get(
                PointerType::get(Ty, pty->getAddressSpace()), width));
            for (size_t i = 0; i < width; ++i) {
              Value *elem =
                  Builder.CreateConstGEP1_64(Ty, shadowPtr, i + primalReturn);
              acc = Builder.CreateInsertValue(acc, elem, i);
            }
            shadow = acc;
          }
        }
        break;
      }
      case DerivativeMode::ReverseModePrimal:
      case DerivativeMode::ReverseModeCombined:
      case DerivativeMode::ReverseModeGradient: {
        if (retType != DIFFE_TYPE::CONSTANT)
          shadow = CI->getArgOperand(1);
        sret = true;
        break;
      }
      }

      args.push_back(primal);
      if (retType != DIFFE_TYPE::CONSTANT)
        args.push_back(shadow);
      if (retType == DIFFE_TYPE::DUP_ARG && !primalReturn && isWriteOnly(fn, 0))
        retType = DIFFE_TYPE::DUP_NONEED;
      constants.push_back(retType);
      retType = DIFFE_TYPE::CONSTANT;
      primalReturn = false;
    }

    ssize_t interleaved = -1;

    size_t maxsize;
    maxsize = CI->arg_size();
    size_t num_args = maxsize;
    for (unsigned i = 1 + sret; i < maxsize; ++i) {
      Value *res = CI->getArgOperand(i);
      auto metaString = getMetadataName(res);
      if (metaString && startsWith(*metaString, "enzyme_")) {
        if (*metaString == "enzyme_interleave") {
          maxsize = i;
          interleaved = i + 1;
          break;
        }
      }
    }

    DIFFE_TYPE last_ty = DIFFE_TYPE::DUP_ARG;

    for (ssize_t i = 1 + sret; (size_t)i < maxsize; ++i) {
      Value *res = CI->getArgOperand(i);
      auto metaString = getMetadataName(res);
#if LLVM_VERSION_MAJOR > 16
      std::optional<Value *> batchOffset;
      std::optional<DIFFE_TYPE> opt_ty;
#else
      Optional<Value *> batchOffset;
      Optional<DIFFE_TYPE> opt_ty;
#endif

      bool overwritten = !(mode == DerivativeMode::ReverseModeCombined);

      bool skipArg = false;

      // handle metadata
      while (metaString && startsWith(*metaString, "enzyme_")) {
        if (*metaString == "enzyme_not_overwritten") {
          overwritten = false;
        } else if (*metaString == "enzyme_byref") {
          ++i;
          if (!isa<ConstantInt>(CI->getArgOperand(i))) {
            EmitFailure("IllegalAllocatedSize", CI->getDebugLoc(), CI,
                        "illegal enzyme byref size ", *CI->getArgOperand(i),
                        "in", *CI);
            return {};
          }
          byRefSize = cast<ConstantInt>(CI->getArgOperand(i))->getZExtValue();
          assert(byRefSize > 0);
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_dup") {
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
          skipArg = true;
          break;
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
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_tape") {
          assert(!sizeOnly);
          ++i;
          tape = CI->getArgOperand(i);
          tapeIsPointer = true;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_nofree") {
          assert(!sizeOnly);
          freeMemory = false;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_runtime_activity") {
          runtimeActivity = true;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_primal_return") {
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_const_return") {
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_active_return") {
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_dup_return") {
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_width") {
          ++i;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_interface") {
          ++i;
          dynamic_interface = CI->getArgOperand(i);
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_trace") {
          trace = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::CONSTANT;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_duptrace") {
          trace = CI->getArgOperand(++i);
          diffeTrace = true;
          opt_ty = DIFFE_TYPE::CONSTANT;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_likelihood") {
          likelihood = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::CONSTANT;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_duplikelihood") {
          likelihood = CI->getArgOperand(++i);
          diffeLikelihood = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::DUP_ARG;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_observations") {
          observations = CI->getArgOperand(++i);
          opt_ty = DIFFE_TYPE::CONSTANT;
          skipArg = true;
          break;
        } else if (*metaString == "enzyme_active_rand_var") {
          Value *string = CI->getArgOperand(++i);
          StringRef const_string;
          if (getConstantStringInfo(string, const_string)) {
            ActiveRandomVariables.insert(const_string);
          } else {
            EmitFailure(
                "IllegalStringType", CI->getDebugLoc(), CI,
                "active variable address must be a compile-time constant", *CI,
                *metaString);
          }
          skipArg = true;
          break;
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
          skipArg = true;
          break;
        }
        ++i;
        if (i == CI->arg_size()) {
          EmitFailure("EnzymeCallingError", CI->getDebugLoc(), CI,
                      "Too few arguments to Enzyme call ", *CI);
          return {};
        }
        res = CI->getArgOperand(i);
        metaString = getMetadataName(res);
      }

      if (skipArg)
        continue;

      if (byRefSize) {
        Type *subTy = nullptr;
        if (truei < FT->getNumParams()) {
          subTy = FT->getParamType(i);
        } else if ((mode == DerivativeMode::ReverseModeGradient ||
                    mode == DerivativeMode::ForwardModeSplit)) {
          if (differentialReturn && differet == nullptr) {
            subTy = FT->getReturnType();
          }
        }

        if (!subTy) {
          EmitFailure("IllegalByVal", CI->getDebugLoc(), CI,
                      "illegal enzyme byval arg", truei, " ", *res);
          return {};
        }

        auto &DL = fn->getParent()->getDataLayout();
        auto BitSize = DL.getTypeSizeInBits(subTy);
        if (BitSize / 8 != byRefSize) {
          EmitFailure("IllegalByRefSize", CI->getDebugLoc(), CI,
                      "illegal enzyme pointer type size ", *res, " expected ",
                      byRefSize, " (bytes) actual size ", BitSize,
                      " (bits) in ", *CI);
        }
        res = Builder.CreateBitCast(
            res,
            PointerType::get(
                subTy, cast<PointerType>(res->getType())->getAddressSpace()));
        res = Builder.CreateLoad(subTy, res);
        byRefSize = 0;
      }

      if (truei >= FT->getNumParams()) {
        if (!isa<MetadataAsValue>(res) &&
            (mode == DerivativeMode::ReverseModeGradient ||
             mode == DerivativeMode::ForwardModeSplit)) {
          if (differentialReturn && differet == nullptr) {
            differet = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
              Type *T = nullptr;
              T = CI->getParamAttr(i, Attribute::ByVal).getValueAsType();
              differet = Builder.CreateLoad(T, differet);
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
                    differet = Builder.CreateLoad(ST1, AI);
                  }

            if (differet->getType() !=
                GradientUtils::getShadowType(fn->getReturnType(), width)) {
              EmitFailure("BadDiffRet", CI->getDebugLoc(), CI,
                          "Bad DiffRet type ", *differet, " expected ",
                          *fn->getReturnType());
              return {};
            }
            continue;
          } else if (tape == nullptr) {
            tape = res;
            if (CI->paramHasAttr(i, Attribute::ByVal)) {
              Type *T = nullptr;
              T = CI->getParamAttr(i, Attribute::ByVal).getValueAsType();
              tape = Builder.CreateLoad(T, tape);
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
      overwritten_args[truei] = overwritten;

      auto PTy = FT->getParamType(truei);
      DIFFE_TYPE ty =
          opt_ty ? *opt_ty
                 : ((interleaved == -1) ? whatType(PTy, mode) : last_ty);
      last_ty = ty;

      constants.push_back(ty);

      assert(truei < FT->getNumParams());
      // cast primal
      if (PTy != res->getType()) {
        if (auto ptr = dyn_cast<PointerType>(res->getType())) {
          if (auto PT = dyn_cast<PointerType>(PTy)) {
            if (ptr->getAddressSpace() != PT->getAddressSpace()) {
#if LLVM_VERSION_MAJOR < 17
              if (CI->getContext().supportsTypedPointers()) {
                res = Builder.CreateAddrSpaceCast(
                    res, PointerType::get(ptr->getPointerElementType(),
                                          PT->getAddressSpace()));
              } else {
                res = Builder.CreateAddrSpaceCast(res, PT);
              }
#else
              res = Builder.CreateAddrSpaceCast(res, PT);
#endif
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
          auto S = simplifyLoad(res);
          if (!S)
            S = res;
          EmitFailure("IllegalArgCast", loc, CI,
                      "Cannot cast __enzyme_autodiff primal argument ", i,
                      ", found ", *res, ", type ", *res->getType(),
                      " (simplified to ", *S, " ) ", " - to arg ", truei, ", ",
                      *PTy);
          return {};
        }
      }
      if (CI->isByValArgument(i)) {
        byVal[args.size()] = CI->getParamByValType(i);
      }

      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        if (interleaved == -1)
          ++i;

        Value *res = nullptr;
#if LLVM_VERSION_MAJOR >= 16
        bool batch = batchOffset.has_value();
#else
        bool batch = batchOffset.hasValue();
#endif

        for (unsigned v = 0; v < width; ++v) {
          if ((size_t)((interleaved == -1) ? i : interleaved) >= num_args) {
            EmitFailure("MissingArgShadow", CI->getDebugLoc(), CI,
                        "__enzyme_autodiff missing argument shadow at index ",
                        *((interleaved == -1) ? &i : &interleaved),
                        ", need shadow of type ", *PTy,
                        " to shadow primal argument ", *args.back(),
                        " at call ", *CI);
            return {};
          }

          // cast diffe
          Value *element =
              CI->getArgOperand((interleaved == -1) ? i : interleaved);
          if (batch) {
            if (auto elementPtrTy = dyn_cast<PointerType>(element->getType())) {
              element = Builder.CreateBitCast(
                  element, PointerType::get(Type::getInt8Ty(CI->getContext()),
                                            elementPtrTy->getAddressSpace()));
              element = Builder.CreateGEP(
                  Type::getInt8Ty(CI->getContext()), element,
                  Builder.CreateMul(
                      *batchOffset,
                      ConstantInt::get((*batchOffset)->getType(), v)));
              element = Builder.CreateBitCast(element, elementPtrTy);
            } else {
              EmitFailure(
                  "NonPointerBatch", CI->getDebugLoc(), CI,
                  "Batched argument at index ",
                  *((interleaved == -1) ? &i : &interleaved),
                  " must be of pointer type, found: ", *element->getType());
              return {};
            }
          }
          if (PTy != element->getType()) {
            element = castToDiffeFunctionArgType(
                Builder, CI, FT, PTy, (interleaved == -1) ? i : interleaved,
                mode, element, truei);
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

            if (v < width - 1 && !batch && (interleaved == -1)) {
              ++i;
            }

          } else {
            res = element;
          }

          if (interleaved != -1)
            interleaved++;
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

    return Options({differet, tape, dynamic_interface, trace, observations,
                    likelihood, diffeLikelihood, width, allocatedTapeSize,
                    freeMemory, returnUsed, tapeIsPointer, differentialReturn,
                    diffeTrace, retType, primalReturn, ActiveRandomVariables,
                    overwritten_args, runtimeActivity});
  }

  static FnTypeInfo populate_type_args(TypeAnalysis &TA, llvm::Function *fn,
                                       DerivativeMode mode) {
    FnTypeInfo type_args(fn);
    for (auto &a : type_args.Function->args()) {
      TypeTree dt;
      if (a.getType()->isFPOrFPVectorTy()) {
        dt = ConcreteType(a.getType()->getScalarType());
      } else if (a.getType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR < 17
        if (a.getContext().supportsTypedPointers()) {
          auto et = a.getType()->getPointerElementType();
          if (et->isFPOrFPVectorTy()) {
            dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
          } else if (et->isPointerTy()) {
            dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
          }
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

  static FloatRepresentation getDefaultFloatRepr(unsigned width) {
    switch (width) {
    case 16:
      return FloatRepresentation(5, 10);
    case 32:
      return FloatRepresentation(8, 23);
    case 64:
      return FloatRepresentation(11, 52);
    default:
      llvm_unreachable("Invalid float width");
    }
  };

  bool HandleTruncateFunc(CallInst *CI, TruncateMode mode) {
    IRBuilder<> Builder(CI);
    Function *F = parseFunctionParameter(CI);
    if (!F)
      return false;
    unsigned ArgSize = CI->arg_size();
    if (ArgSize != 4 && ArgSize != 3) {
      EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                  "Had incorrect number of args to __enzyme_truncate_func", *CI,
                  " - expected 3 or 4");
      return false;
    }
    FloatTruncation truncation = [&]() -> FloatTruncation {
      if (ArgSize == 3) {
        auto Cfrom = cast<ConstantInt>(CI->getArgOperand(1));
        assert(Cfrom);
        auto Cto = cast<ConstantInt>(CI->getArgOperand(2));
        assert(Cto);
        return FloatTruncation(
            getDefaultFloatRepr((unsigned)Cfrom->getValue().getZExtValue()),
            getDefaultFloatRepr((unsigned)Cto->getValue().getZExtValue()),
            mode);
      } else if (ArgSize == 4) {
        auto Cfrom = cast<ConstantInt>(CI->getArgOperand(1));
        assert(Cfrom);
        auto Cto_exponent = cast<ConstantInt>(CI->getArgOperand(2));
        assert(Cto_exponent);
        auto Cto_significand = cast<ConstantInt>(CI->getArgOperand(3));
        assert(Cto_significand);
        return FloatTruncation(
            getDefaultFloatRepr((unsigned)Cfrom->getValue().getZExtValue()),
            FloatRepresentation(
                (unsigned)Cto_exponent->getValue().getZExtValue(),
                (unsigned)Cto_significand->getValue().getZExtValue()),
            mode);
      }
      llvm_unreachable("??");
    }();

    RequestContext context(CI, &Builder);
    llvm::Value *res = Logic.CreateTruncateFunc(context, F, truncation, mode);
    if (!res)
      return false;
    res = Builder.CreatePointerCast(res, CI->getType());
    CI->replaceAllUsesWith(res);
    CI->eraseFromParent();
    return true;
  }

  bool HandleTruncateValue(CallInst *CI, bool isTruncate) {
    IRBuilder<> Builder(CI);
    if (CI->arg_size() != 3) {
      EmitFailure("TooManyArgs", CI->getDebugLoc(), CI,
                  "Had incorrect number of args to __enzyme_truncate_value",
                  *CI, " - expected 3");
      return false;
    }
    auto Cfrom = cast<ConstantInt>(CI->getArgOperand(1));
    assert(Cfrom);
    auto Cto = cast<ConstantInt>(CI->getArgOperand(2));
    assert(Cto);
    auto Addr = CI->getArgOperand(0);
    RequestContext context(CI, &Builder);
    bool res = Logic.CreateTruncateValue(
        context, Addr,
        getDefaultFloatRepr((unsigned)Cfrom->getValue().getZExtValue()),
        getDefaultFloatRepr((unsigned)Cto->getValue().getZExtValue()),
        isTruncate);
    if (!res)
      return false;
    return true;
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

    for (unsigned i = 1 + sret; i < CI->arg_size(); ++i) {
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
      if (metaString && startsWith(*metaString, "enzyme_")) {
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
          if (i >= CI->arg_size()) {
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
              element = Builder.CreateGEP(
                  Type::getInt8Ty(CI->getContext()), element,
                  Builder.CreateMul(
                      batchOffset[i - 1],
                      ConstantInt::get(batchOffset[i - 1]->getType(), v)));
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

    auto newFunc = Logic.CreateBatch(RequestContext(CI, &Builder), F, width,
                                     arg_types, ret_type);

    if (!newFunc)
      return false;

    Value *batch =
        Builder.CreateCall(newFunc->getFunctionType(), newFunc, args);

    batch = adaptReturnedVector(CI, batch, Builder, width);

    Value *ret = CI;
    Type *retElemType = nullptr;
    if (CI->hasStructRetAttr()) {
      ret = CI->getArgOperand(0);
      retElemType =
          CI->getAttribute(AttributeList::FirstArgIndex, Attribute::StructRet)
              .getValueAsType();
    }
    ReplaceOriginalCall(Builder, ret, retElemType, batch, CI,
                        DerivativeMode::ForwardMode);

    return true;
  }

  bool HandleAutoDiff(Instruction *CI, CallingConv::ID CallingConv, Value *ret,
                      Type *retElemType, SmallVectorImpl<Value *> &args,
                      const std::map<int, Type *> &byVal,
                      const std::vector<DIFFE_TYPE> &constants, Function *fn,
                      DerivativeMode mode, Options &options, bool sizeOnly,
                      SmallVectorImpl<CallInst *> &calls) {
    auto &differet = options.differet;
    auto &tape = options.tape;
    auto &width = options.width;
    auto &allocatedTapeSize = options.allocatedTapeSize;
    auto &freeMemory = options.freeMemory;
    auto &returnUsed = options.returnUsed;
    auto &tapeIsPointer = options.tapeIsPointer;
    auto &differentialReturn = options.differentialReturn;
    auto &retType = options.retType;
    auto &overwritten_args = options.overwritten_args;
    auto primalReturn = options.primalReturn;

    auto Arch = Triple(CI->getModule()->getTargetTriple()).getArch();
    bool AtomicAdd = Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
                     Arch == Triple::amdgcn;

    TypeAnalysis TA(Logic.PPC.FAM);
    FnTypeInfo type_args = populate_type_args(TA, fn, mode);

    IRBuilder Builder(CI);
    RequestContext context(CI, &Builder);

    // differentiate fn
    Function *newFunc = nullptr;
    Type *tapeType = nullptr;
    const AugmentedReturn *aug;
    switch (mode) {
    case DerivativeMode::ForwardModeError:
    case DerivativeMode::ForwardMode:
      if (primalReturn && fn->getReturnType()->isVoidTy()) {
        auto fnname = fn->getName();
        EmitFailure("PrimalRetOfVoid", CI->getDebugLoc(), CI,
                    "Requested primal result of void-returning function type ",
                    *fn->getFunctionType(), " ", fnname, " ", *CI);
      } else
        newFunc = Logic.CreateForwardDiff(
            context, fn, retType, constants, TA,
            /*should return*/ primalReturn, mode, freeMemory,
            options.runtimeActivity, width,
            /*addedType*/ nullptr, type_args, overwritten_args,
            /*augmented*/ nullptr);
      break;
    case DerivativeMode::ForwardModeSplit: {
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      aug = &Logic.CreateAugmentedPrimal(
          context, fn, retType, constants, TA,
          /*returnUsed*/ false, /*shadowReturnUsed*/ false, type_args,
          overwritten_args, forceAnonymousTape, options.runtimeActivity, width,
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
            DL.getTypeSizeInBits(tapeType) > 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = getInt8PtrTy(fn->getContext());
      }
      newFunc = Logic.CreateForwardDiff(
          context, fn, retType, constants, TA,
          /*should return*/ primalReturn, mode, freeMemory,
          options.runtimeActivity, width,
          /*addedType*/ tapeType, type_args, overwritten_args, aug);
      break;
    }
    case DerivativeMode::ReverseModeCombined:
      assert(freeMemory);
      newFunc = Logic.CreatePrimalAndGradient(
          context,
          (ReverseCacheKey){.todiff = fn,
                            .retType = retType,
                            .constant_args = constants,
                            .overwritten_args = overwritten_args,
                            .returnUsed = primalReturn,
                            .shadowReturnUsed = false,
                            .mode = mode,
                            .width = width,
                            .freeMemory = freeMemory,
                            .AtomicAdd = AtomicAdd,
                            .additionalType = nullptr,
                            .forceAnonymousTape = false,
                            .typeInfo = type_args,
                            .runtimeActivity = options.runtimeActivity},
          TA, /*augmented*/ nullptr);
      break;
    case DerivativeMode::ReverseModePrimal:
    case DerivativeMode::ReverseModeGradient: {
      if (primalReturn) {
        EmitFailure(
            "SplitPrimalRet", CI->getDebugLoc(), CI,
            "Option enzyme_primal_return not available in reverse split mode");
      }
      bool forceAnonymousTape = !sizeOnly && allocatedTapeSize == -1;
      bool shadowReturnUsed = returnUsed && (retType == DIFFE_TYPE::DUP_ARG ||
                                             retType == DIFFE_TYPE::DUP_NONEED);
      aug = &Logic.CreateAugmentedPrimal(
          context, fn, retType, constants, TA, returnUsed, shadowReturnUsed,
          type_args, overwritten_args, forceAnonymousTape,
          options.runtimeActivity, width,
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
            DL.getTypeSizeInBits(tapeType) > 8 * (size_t)allocatedTapeSize) {
          auto bytes = DL.getTypeSizeInBits(tapeType) / 8;
          EmitFailure("Insufficient tape allocation size", CI->getDebugLoc(),
                      CI, "need ", bytes, " bytes have ", allocatedTapeSize,
                      " bytes");
        }
      } else {
        tapeType = getInt8PtrTy(fn->getContext());
      }
      if (mode == DerivativeMode::ReverseModePrimal)
        newFunc = aug->fn;
      else
        newFunc = Logic.CreatePrimalAndGradient(
            context,
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
                              .typeInfo = type_args,
                              .runtimeActivity = options.runtimeActivity},
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
      } else if (auto AT = dyn_cast<ArrayType>(fn->getReturnType())) {
        SmallVector<Constant *, 2> csts(
            AT->getNumElements(), ConstantFP::get(AT->getElementType(), 1.0));
        args.push_back(ConstantArray::get(AT, csts));
      } else {
        auto RT = fn->getReturnType();
        EmitFailure("EnzymeCallingError", CI->getDebugLoc(), CI,
                    "Differential return required for call ", *CI,
                    " but one of type ", *RT, " could not be auto deduced");
        return false;
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
        tape = Builder.CreateLoad(tapeType, tape);
      } else if (tapeType != tape->getType() &&
                 DL.getTypeSizeInBits(tapeType) <=
                     DL.getTypeSizeInBits(tape->getType())) {
        IRBuilder<> EB(&CI->getParent()->getParent()->getEntryBlock().front());
        auto AL = EB.CreateAlloca(tape->getType());
        Builder.CreateStore(tape, AL);
        tape = Builder.CreateLoad(
            tapeType,
            Builder.CreatePointerCast(AL, PointerType::getUnqual(tapeType)));
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

    for (auto &&[attr, ty] : byVal) {
      diffretc->addParamAttr(
          attr, Attribute::getWithByValType(diffretc->getContext(), ty));
    }

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
    calls.push_back(diffretc);
    return diffret;
  }

  /// Return whether successful
  bool HandleAutoDiffArguments(CallInst *CI, DerivativeMode mode, bool sizeOnly,
                               SmallVectorImpl<CallInst *> &calls) {

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
      retElemType =
          CI->getAttribute(AttributeList::FirstArgIndex, Attribute::StructRet)
              .getValueAsType();
    }

    return HandleAutoDiff(CI, CI->getCallingConv(), ret, retElemType, args,
                          byVal, constants, fn, mode, *options, sizeOnly,
                          calls);
  }

  bool HandleProbProg(CallInst *CI, ProbProgMode mode,
                      SmallVectorImpl<CallInst *> &calls) {
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

    SmallVector<Value *, 6> dargs(args.begin(), args.end());

#if LLVM_VERSION_MAJOR >= 16
    if (!opt.has_value())
      return false;
#else
    if (!opt.hasValue())
      return false;
#endif

    auto dynamic_interface = opt->dynamic_interface;
    auto trace = opt->trace;
    auto dtrace = opt->diffeTrace;
    auto observations = opt->observations;
    auto likelihood = opt->likelihood;
    auto dlikelihood = opt->diffeLikelihood;

    // Interface
    bool has_dynamic_interface = dynamic_interface != nullptr;
    bool needs_interface =
        mode == ProbProgMode::Trace || mode == ProbProgMode::Condition;
    std::unique_ptr<TraceInterface> interface;
    if (has_dynamic_interface) {
      interface = std::make_unique<DynamicTraceInterface>(dynamic_interface,
                                                          CI->getFunction());
    } else if (needs_interface) {
      interface = std::make_unique<StaticTraceInterface>(F->getParent());
    }

    // Find sample function
    SmallPtrSet<Function *, 4> sampleFunctions;
    SmallPtrSet<Function *, 4> observeFunctions;
    for (auto &func : F->getParent()->functions()) {
      if (func.getName().contains("__enzyme_sample")) {
        assert(func.getFunctionType()->getNumParams() >= 3);
        sampleFunctions.insert(&func);
      } else if (func.getName().contains("__enzyme_observe")) {
        assert(func.getFunctionType()->getNumParams() >= 3);
        observeFunctions.insert(&func);
      }
    }

    assert(!sampleFunctions.empty() || !observeFunctions.empty());

    bool autodiff = dtrace || dlikelihood;
    IRBuilder<> AllocaBuilder(CI->getParent()->getFirstNonPHI());

    if (!likelihood) {
      likelihood = AllocaBuilder.CreateAlloca(AllocaBuilder.getDoubleTy(),
                                              nullptr, "likelihood");
      Builder.CreateStore(ConstantFP::getNullValue(Builder.getDoubleTy()),
                          likelihood);
    }
    args.push_back(likelihood);

    if (autodiff && !dlikelihood) {
      dlikelihood = AllocaBuilder.CreateAlloca(AllocaBuilder.getDoubleTy(),
                                               nullptr, "dlikelihood");
      Builder.CreateStore(ConstantFP::get(Builder.getDoubleTy(), 1.0),
                          dlikelihood);
    }

    if (autodiff) {
      dargs.push_back(likelihood);
      dargs.push_back(dlikelihood);
      constants.push_back(DIFFE_TYPE::DUP_ARG);
      opt->overwritten_args.push_back(false);
    } else {
      constants.push_back(DIFFE_TYPE::CONSTANT);
      opt->overwritten_args.push_back(false);
    }

    if (mode == ProbProgMode::Condition) {
      opt->overwritten_args.push_back(false);
      args.push_back(observations);
      dargs.push_back(observations);
      constants.push_back(DIFFE_TYPE::CONSTANT);
    }

    if (mode == ProbProgMode::Trace || mode == ProbProgMode::Condition) {
      opt->overwritten_args.push_back(false);
      args.push_back(trace);
      dargs.push_back(trace);
      constants.push_back(DIFFE_TYPE::CONSTANT);
    }

    auto newFunc = Logic.CreateTrace(
        RequestContext(CI, &Builder), F, sampleFunctions, observeFunctions,
        opt->ActiveRandomVariables, mode, autodiff, interface.get());

    if (!autodiff) {
      auto call = CallInst::Create(newFunc->getFunctionType(), newFunc, args);
      ReplaceInstWithInst(CI, call);
      return true;
    }

    Value *ret = CI;
    Type *retElemType = nullptr;
    if (CI->hasStructRetAttr()) {
      ret = CI->getArgOperand(0);
      retElemType =
          CI->getAttribute(AttributeList::FirstArgIndex, Attribute::StructRet)
              .getValueAsType();
    }

    bool status = HandleAutoDiff(
        CI, CI->getCallingConv(), ret, retElemType, dargs, byVal, constants,
        newFunc, DerivativeMode::ReverseModeCombined, *opt, false, calls);

    return status;
  }

  bool handleFullModuleTrunc(Function &F) {
    if (startsWith(F.getName(), EnzymeFPRTPrefix))
      return false;
    typedef std::vector<FloatTruncation> TruncationsTy;
    static TruncationsTy FullModuleTruncs = []() -> TruncationsTy {
      StringRef ConfigStr(EnzymeTruncateAll);
      auto Invalid = [=]() {
        // TODO emit better diagnostic
        llvm::report_fatal_error("error: invalid format for truncation config");
      };

      // "64" or "11-52"
      auto parseFloatRepr = [&]() -> std::optional<FloatRepresentation> {
        unsigned Tmp = 0;
        if (ConfigStr.consumeInteger(10, Tmp))
          return {};
        if (ConfigStr.consume_front("-")) {
          unsigned Tmp2 = 0;
          if (ConfigStr.consumeInteger(10, Tmp2))
            Invalid();
          return FloatRepresentation(Tmp, Tmp2);
        }
        return getDefaultFloatRepr(Tmp);
      };

      // Parse "64to32;32to16;5-10to4-9"
      TruncationsTy Tmp;
      while (true) {
        auto From = parseFloatRepr();
        if (!From && !ConfigStr.empty())
          Invalid();
        if (!From)
          break;
        if (!ConfigStr.consume_front("to"))
          Invalid();
        auto To = parseFloatRepr();
        if (!To)
          Invalid();
        Tmp.push_back({*From, *To, TruncOpFullModuleMode});
        ConfigStr.consume_front(";");
      }
      return Tmp;
    }();

    if (FullModuleTruncs.empty())
      return false;

    // TODO sort truncations (64to32, then 32to16 will make everything 16)
    for (auto Truncation : FullModuleTruncs) {
      IRBuilder<> Builder(F.getContext());
      RequestContext context(&*F.getEntryBlock().begin(), &Builder);
      Function *TruncatedFunc = Logic.CreateTruncateFunc(
          context, &F, Truncation, TruncOpFullModuleMode);

      ValueToValueMapTy Mapping;
      for (auto &&[Arg, TArg] : llvm::zip(F.args(), TruncatedFunc->args()))
        Mapping[&TArg] = &Arg;

      // Move the truncated body into the original function
      F.deleteBody();
#if LLVM_VERSION_MAJOR >= 16
      F.splice(F.begin(), TruncatedFunc);
#else
      F.getBasicBlockList().splice(F.begin(),
                                   TruncatedFunc->getBasicBlockList());
#endif
      RemapFunction(F, Mapping,
                    RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
      TruncatedFunc->deleteBody();
    }
    return true;
  }

  bool lowerEnzymeCalls(Function &F, std::set<Function *> &done) {
    if (done.count(&F))
      return false;
    done.insert(&F);

    if (F.empty())
      return false;

    if (handleFullModuleTrunc(F))
      return true;

    bool Changed = false;

    for (BasicBlock &BB : F)
      if (InvokeInst *II = dyn_cast<InvokeInst>(BB.getTerminator())) {

        Function *Fn = II->getCalledFunction();

        if (auto castinst = dyn_cast<ConstantExpr>(II->getCalledOperand())) {
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
              Fn->getName().contains("__enzyme_truncate") ||
              Fn->getName().contains("__enzyme_batch") ||
              Fn->getName().contains("__enzyme_error_estimate") ||
              Fn->getName().contains("__enzyme_trace") ||
              Fn->getName().contains("__enzyme_condition")))
          continue;

        SmallVector<Value *, 16> CallArgs(II->arg_begin(), II->arg_end());
        SmallVector<OperandBundleDef, 1> OpBundles;
        II->getOperandBundlesAsDefs(OpBundles);
        // Insert a normal call instruction...
        CallInst *NewCall =
            CallInst::Create(II->getFunctionType(), II->getCalledOperand(),
                             CallArgs, OpBundles, "", II);
        NewCall->takeName(II);
        NewCall->setCallingConv(II->getCallingConv());
        NewCall->setAttributes(II->getAttributes());
        NewCall->setDebugLoc(II->getDebugLoc());
        II->replaceAllUsesWith(NewCall);

        // Insert an unconditional branch to the normal destination.
        BranchInst::Create(II->getNormalDest(), II);

        // Remove any PHI node entries from the exception destination.
        II->getUnwindDest()->removePredecessor(&BB);

        II->eraseFromParent();
        Changed = true;
      }

    MapVector<CallInst *, DerivativeMode> toLower;
    MapVector<CallInst *, DerivativeMode> toVirtual;
    MapVector<CallInst *, DerivativeMode> toSize;
    SmallVector<CallInst *, 4> toBatch;
    SmallVector<CallInst *, 4> toTruncateFuncMem;
    SmallVector<CallInst *, 4> toTruncateFuncOp;
    SmallVector<CallInst *, 4> toTruncateValue;
    SmallVector<CallInst *, 4> toExpandValue;
    MapVector<CallInst *, ProbProgMode> toProbProg;
    SetVector<CallInst *> InactiveCalls;
    SetVector<CallInst *> IterCalls;
  retry:;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        CallInst *CI = dyn_cast<CallInst>(&I);

        if (!CI)
          continue;

        Function *Fn = nullptr;

        Value *FnOp = CI->getCalledOperand();
        while (true) {
          if ((Fn = dyn_cast<Function>(FnOp)))
            break;
          if (auto castinst = dyn_cast<ConstantExpr>(FnOp)) {
            if (castinst->isCast()) {
              FnOp = castinst->getOperand(0);
              continue;
            }
          }
          break;
        }

        if (!Fn)
          continue;

        size_t num_args = CI->arg_size();

        if (Fn->getName().contains("__enzyme_todense")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
        }
        if (Fn->getName().contains("__enzyme_float")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_integer")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_double")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_pointer")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
          for (size_t i = 0; i < num_args; ++i) {
            if (CI->getArgOperand(i)->getType()->isPointerTy()) {
              CI->addParamAttr(i, Attribute::ReadNone);
              CI->addParamAttr(i, Attribute::NoCapture);
            }
          }
        }
        if (Fn->getName().contains("__enzyme_virtualreverse")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
        }
        if (Fn->getName().contains("__enzyme_iter")) {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
        }
        if (Fn->getName().contains("__enzyme_call_inactive")) {
          InactiveCalls.insert(CI);
        }
        if (Fn->getName() == "omp_get_max_threads" ||
            Fn->getName() == "omp_get_thread_num") {
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemory();
          CI->setOnlyAccessesInaccessibleMemory();
          Fn->setOnlyReadsMemory();
          CI->setOnlyReadsMemory();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
#endif
        }
        if ((Fn->getName() == "cblas_ddot" || Fn->getName() == "cblas_sdot") &&
            Fn->isDeclaration()) {
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesArgMemory();
          Fn->setOnlyReadsMemory();
          CI->setOnlyReadsMemory();
#else
          Fn->addFnAttr(Attribute::ArgMemOnly);
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
#endif
          CI->addParamAttr(1, Attribute::ReadOnly);
          CI->addParamAttr(1, Attribute::NoCapture);
          CI->addParamAttr(3, Attribute::ReadOnly);
          CI->addParamAttr(3, Attribute::NoCapture);
        }
        if (Fn->getName() == "frexp" || Fn->getName() == "frexpf" ||
            Fn->getName() == "frexpl") {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyAccessesArgMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ArgMemOnly);
#endif
          CI->addParamAttr(1, Attribute::WriteOnly);
        }
        if (Fn->getName() == "__fd_sincos_1" || Fn->getName() == "__fd_cos_1" ||
            Fn->getName() == "__mth_i_ipowi") {
#if LLVM_VERSION_MAJOR >= 16
          CI->setOnlyReadsMemory();
          CI->setOnlyWritesMemory();
#else
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#endif
        }
        if (getFuncName(Fn) == "strcmp") {
          Fn->addParamAttr(0, Attribute::ReadOnly);
          Fn->addParamAttr(1, Attribute::ReadOnly);
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyReadsMemory();
          CI->setOnlyReadsMemory();
#else
          Fn->addFnAttr(Attribute::ReadOnly);
          CI->addAttribute(AttributeList::FunctionIndex, Attribute::ReadOnly);
#endif
        }
        if (Fn->getName() == "f90io_fmtw_end" ||
            Fn->getName() == "f90io_unf_end") {
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemory();
          CI->setOnlyAccessesInaccessibleMemory();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOnly);
#endif
        }
        if (Fn->getName() == "f90io_open2003a") {
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemOrArgMem();
          CI->setOnlyAccessesInaccessibleMemOrArgMem();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
#endif
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
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemOrArgMem();
          CI->setOnlyAccessesInaccessibleMemOrArgMem();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
#endif
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
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemOrArgMem();
          CI->setOnlyAccessesInaccessibleMemOrArgMem();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
#endif
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
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemOrArgMem();
          CI->setOnlyAccessesInaccessibleMemOrArgMem();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
#endif
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
#if LLVM_VERSION_MAJOR >= 16
          Fn->setOnlyAccessesInaccessibleMemOrArgMem();
          CI->setOnlyAccessesInaccessibleMemOrArgMem();
#else
          Fn->addFnAttr(Attribute::InaccessibleMemOrArgMemOnly);
          CI->addAttribute(AttributeList::FunctionIndex,
                           Attribute::InaccessibleMemOrArgMemOnly);
#endif
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
        bool truncateFuncOp = false;
        bool truncateFuncMem = false;
        bool truncateValue = false;
        bool expandValue = false;
        bool probProg = false;
        DerivativeMode derivativeMode;
        ProbProgMode probProgMode;
        if (Fn->getName().contains("__enzyme_autodiff")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ReverseModeCombined;
        } else if (Fn->getName().contains("__enzyme_fwddiff")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ForwardMode;
        } else if (Fn->getName().contains("__enzyme_error_estimate")) {
          enableEnzyme = true;
          derivativeMode = DerivativeMode::ForwardModeError;
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
        } else if (Fn->getName().contains("__enzyme_truncate_mem_func")) {
          enableEnzyme = true;
          truncateFuncMem = true;
        } else if (Fn->getName().contains("__enzyme_truncate_op_func")) {
          enableEnzyme = true;
          truncateFuncOp = true;
        } else if (Fn->getName().contains("__enzyme_truncate_mem_value")) {
          enableEnzyme = true;
          truncateValue = true;
        } else if (Fn->getName().contains("__enzyme_expand_mem_value")) {
          enableEnzyme = true;
          expandValue = true;
        } else if (Fn->getName().contains("__enzyme_likelihood")) {
          enableEnzyme = true;
          probProgMode = ProbProgMode::Likelihood;
          probProg = true;
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
          else if (truncateFuncOp)
            toTruncateFuncOp.push_back(CI);
          else if (truncateFuncMem)
            toTruncateFuncMem.push_back(CI);
          else if (truncateValue)
            toTruncateValue.push_back(CI);
          else if (expandValue)
            toExpandValue.push_back(CI);
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
      for (size_t i = 1; i < CI->arg_size(); ++i) {
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

    SmallVector<CallInst *, 1> calls;

    // Perform all the size replacements first to create constants
    for (auto pair : toSize) {
      bool successful = HandleAutoDiffArguments(pair.first, pair.second,
                                                /*sizeOnly*/ true, calls);
      Changed = true;
      if (!successful)
        break;
    }
    for (auto pair : toLower) {
      bool successful = HandleAutoDiffArguments(pair.first, pair.second,
                                                /*sizeOnly*/ false, calls);
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

      IRBuilder<> Builder(CI);
      auto val = GradientUtils::GetOrCreateShadowConstant(
          RequestContext(CI, &Builder), Logic,
          Logic.PPC.FAM.getResult<TargetLibraryAnalysis>(F), TA, fn,
          pair.second, /*runtimeActivity*/ false, /*width*/ 1, AtomicAdd);
      CI->replaceAllUsesWith(ConstantExpr::getPointerCast(val, CI->getType()));
      CI->eraseFromParent();
      Changed = true;
    }

    for (auto call : toBatch) {
      HandleBatch(call);
    }
    for (auto call : toTruncateFuncMem) {
      HandleTruncateFunc(call, TruncMemMode);
    }
    for (auto call : toTruncateFuncOp) {
      HandleTruncateFunc(call, TruncOpMode);
    }
    for (auto call : toTruncateValue) {
      HandleTruncateValue(call, true);
    }
    for (auto call : toExpandValue) {
      HandleTruncateValue(call, false);
    }

    for (auto &&[call, mode] : toProbProg) {
      HandleProbProg(call, mode, calls);
    }

    if (Logic.PostOpt) {
      auto Params = llvm::getInlineParams();

      llvm::SetVector<CallInst *> Q;
      for (auto call : calls)
        Q.insert(call);
      while (Q.size()) {
        auto cur = *Q.begin();
        Function *outerFunc = cur->getParent()->getParent();
        llvm::OptimizationRemarkEmitter ORE(outerFunc);
        Q.erase(Q.begin());
        if (auto F = cur->getCalledFunction()) {
          if (!F->empty()) {
            // Garbage collect AC's created
            SmallVector<std::unique_ptr<AssumptionCache>, 2> ACAlloc;
            auto getAC = [&](Function &F) -> llvm::AssumptionCache & {
              auto AC = std::make_unique<AssumptionCache>(F);
              ACAlloc.push_back(std::move(AC));
              return *ACAlloc.back();
            };
            auto GetTLI =
                [&](llvm::Function &F) -> const llvm::TargetLibraryInfo & {
              return Logic.PPC.FAM.getResult<TargetLibraryAnalysis>(F);
            };

            TargetTransformInfo TTI(F->getParent()->getDataLayout());
            auto GetInlineCost = [&](CallBase &CB) {
              auto cst = llvm::getInlineCost(CB, Params, TTI, getAC, GetTLI);
              return cst;
            };
#if LLVM_VERSION_MAJOR >= 20
            if (llvm::shouldInline(*cur, TTI, GetInlineCost, ORE))
#else
            if (llvm::shouldInline(*cur, GetInlineCost, ORE))
#endif
            {
              InlineFunctionInfo IFI;
              InlineResult IR = InlineFunction(*cur, IFI);
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
          }
        }
      }
    }

    if (Changed && EnzymeAttributor) {
      // TODO consider enabling when attributor does not delete
      // dead internal functions, which invalidates Enzyme's cache
      // code left here to re-enable upon Attributor patch

#if !defined(FLANG) && !defined(ROCM)

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
#if LLVM_VERSION_MAJOR < 17
          &AAReturnedValues::ID,
#endif
          &AANoFree::ID,          &AANoUndef::ID,

          //&AAValueSimplify::ID,
          //&AAReachability::ID,
          //&AAValueConstantRange::ID,
          //&AAUndefinedBehavior::ID,
          //&AAPotentialValues::ID,
      };

      AttributorConfig aconfig(CGUpdater);
      aconfig.Allowed = &Allowed;
      aconfig.DeleteFns = false;
      Attributor A(Functions, InfoCache, aconfig);
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

    for (Function &F : make_early_inc_range(M)) {
      attributeKnownFunctions(F);
    }

    bool changed = false;
    for (Function &F : M) {
      if (F.empty())
        continue;
      for (BasicBlock &BB : F) {
        for (Instruction &I : make_early_inc_range(BB)) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
            if (auto castinst =
                    dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
              if (castinst->isCast())
                if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                  F = fn;
                }
            }
            if (F && F->getName() == "f90_mzero8") {
              IRBuilder<> B(CI);

              Value *args[3];
              args[0] = CI->getArgOperand(0);
              args[1] = ConstantInt::get(Type::getInt8Ty(M.getContext()), 0);
              args[2] = B.CreateMul(
                  CI->getArgOperand(1),
                  ConstantInt::get(CI->getArgOperand(1)->getType(), 8));
              B.CreateMemSet(args[0], args[1], args[2], MaybeAlign());

              CI->eraseFromParent();
            }
          }
        }
      }
    }

    if (Logic.PostOpt && EnzymeOMPOpt) {
      OpenMPOptPass().run(M, Logic.PPC.MAM);
      /// Attributor is run second time for promoted args to get attributes.
      AttributorPass().run(M, Logic.PPC.MAM);
      for (auto &F : M)
        if (!F.empty())
          PromotePass().run(F, Logic.PPC.FAM);
      changed = true;
    }

    std::set<Function *> done;
    for (Function &F : M) {
      if (F.empty())
        continue;

      changed |= lowerEnzymeCalls(F, done);
    }

    for (Function &F : M) {
      if (F.empty())
        continue;

      for (BasicBlock &BB : F) {
        for (Instruction &I : make_early_inc_range(BB)) {
          if (auto CI = dyn_cast<CallInst>(&I)) {
            Function *F = CI->getCalledFunction();
            if (auto castinst =
                    dyn_cast<ConstantExpr>(CI->getCalledOperand())) {
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
                CI->eraseFromParent();
                changed = true;
              }
              if (F->getName() == "__enzyme_iter") {
                CI->replaceAllUsesWith(CI->getArgOperand(0));
                CI->eraseFromParent();
                changed = true;
              }
            }
          }
        }
      }
    }

    SmallPtrSet<CallInst *, 16> sample_calls;
    SmallPtrSet<CallInst *, 16> observe_calls;
    for (auto &&func : M) {
      for (auto &&BB : func) {
        for (auto &&Inst : BB) {
          if (auto CI = dyn_cast<CallInst>(&Inst)) {
            Function *fun = CI->getCalledFunction();
            if (!fun)
              continue;

            if (fun->getName().contains("__enzyme_sample")) {
              if (CI->getNumOperands() < 3) {
                EmitFailure(
                    "IllegalNumberOfArguments", CI->getDebugLoc(), CI,
                    "Not enough arguments passed to call to __enzyme_sample");
              }
              Function *samplefn = GetFunctionFromValue(CI->getOperand(0));
              unsigned expected =
                  samplefn->getFunctionType()->getNumParams() + 3;
              unsigned actual = CI->arg_size();
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

            } else if (fun->getName().contains("__enzyme_observe")) {
              if (CI->getNumOperands() < 3) {
                EmitFailure(
                    "IllegalNumberOfArguments", CI->getDebugLoc(), CI,
                    "Not enough arguments passed to call to __enzyme_sample");
              }
              Value *observed = CI->getOperand(0);
              Function *pdf = GetFunctionFromValue(CI->getArgOperand(1));
              unsigned expected = pdf->getFunctionType()->getNumParams() - 1;

              unsigned actual = CI->arg_size();
              if (actual - 3 != expected) {
                EmitFailure("IllegalNumberOfArguments", CI->getDebugLoc(), CI,
                            "Illegal number of arguments passed to call to "
                            "__enzyme_observe.",
                            " Expected: ", expected, " got: ", actual);
              }

              for (unsigned i = 0;
                   i < pdf->getFunctionType()->getNumParams() - 1; ++i) {
                Value *ci_arg = CI->getArgOperand(i + 3);
                Value *pdf_arg = pdf->arg_begin() + i;

                if (ci_arg->getType() != pdf_arg->getType()) {
                  EmitFailure("IllegalSampleType", CI->getDebugLoc(), CI,
                              "Type of: ", *ci_arg, " (", *ci_arg->getType(),
                              ")",
                              " does not match the argument type of the "
                              "density function: ",
                              *pdf, " at: ", i, " (", *pdf_arg->getType(), ")");
                }
              }

              if ((pdf->arg_end() - 1)->getType() != observed->getType()) {
                EmitFailure(
                    "IllegalSampleType", CI->getDebugLoc(), CI,
                    "Return type of ", *observed, " (", *observed->getType(),
                    ")",
                    " does not match the last argument type of the density "
                    "function: ",
                    *pdf, " (", *(pdf->arg_end() - 1)->getType(), ")");
              }
              observe_calls.insert(CI);
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

    for (auto call : observe_calls) {
      Value *observed = call->getArgOperand(0);

      if (!call->getType()->isVoidTy())
        call->replaceAllUsesWith(observed);
      call->eraseFromParent();
    }

    for (const auto &pair : Logic.PPC.cache)
      pair.second->eraseFromParent();
    Logic.clear();

    if (changed && Logic.PostOpt) {
      TimeTraceScope timeScope("Enzyme PostOpt", M.getName());

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
      auto PM = PB.buildModuleSimplificationPipeline(OptimizationLevel::O2,
                                                     ThinOrFullLTOPhase::None);
      PM.run(M, MAM);
      if (EnzymeOMPOpt) {
        OpenMPOptPass().run(M, MAM);
        /// Attributor is run second time for promoted args to get attributes.
        AttributorPass().run(M, MAM);
        for (auto &F : M)
          if (!F.empty())
            PromotePass().run(F, FAM);
      }
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

#include "ActivityAnalysisPrinter.h"
#include "JLInstSimplify.h"
#include "PreserveNVVM.h"
#ifdef ENZYME_ENABLE_FPOPT
#include "Herbie.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#endif

#include "TypeAnalysis/TypeAnalysisPrinter.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/CalledValuePropagation.h"
#include "llvm/Transforms/IPO/ConstantMerge.h"
#include "llvm/Transforms/IPO/CrossDSOCFI.h"
#include "llvm/Transforms/IPO/DeadArgumentElimination.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/GlobalOpt.h"
#include "llvm/Transforms/IPO/GlobalSplit.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
#include "llvm/Transforms/IPO/SCCP.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/CallSiteSplitting.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/Float2Int.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopDeletion.h"
#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Scalar/SROA.h"
// #include "llvm/Transforms/IPO/MemProfContextDisambiguation.h"
#include "llvm/Transforms/IPO/ArgumentPromotion.h"
#include "llvm/Transforms/Scalar/ConstraintElimination.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Scalar/MemCpyOptimizer.h"
#include "llvm/Transforms/Scalar/NewGVN.h"
#include "llvm/Transforms/Scalar/TailRecursionElimination.h"
#if LLVM_VERSION_MAJOR >= 17
#include "llvm/Transforms/Utils/MoveAutoInit.h"
#endif
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Scalar/LoopFlatten.h"
#include "llvm/Transforms/Scalar/MergedLoadStoreMotion.h"

static InlineParams getInlineParamsFromOptLevel(OptimizationLevel Level) {
  return getInlineParams(Level.getSpeedupLevel(), Level.getSizeLevel());
}

#include "llvm/Transforms/Scalar/LowerConstantIntrinsics.h"
#include "llvm/Transforms/Scalar/LowerMatrixIntrinsics.h"
namespace llvm {
extern cl::opt<unsigned> SetLicmMssaNoAccForPromotionCap;
extern cl::opt<unsigned> SetLicmMssaOptCap;
#define EnableLoopFlatten false
#define EagerlyInvalidateAnalyses false
#define RunNewGVN false
#define EnableConstraintElimination true
#define UseInlineAdvisor InliningAdvisorMode::Default
#define EnableMemProfContextDisambiguation false
// extern cl::opt<bool> EnableMatrix;
#define EnableMatrix false
#define EnableModuleInliner false
} // namespace llvm

void augmentPassBuilder(llvm::PassBuilder &PB) {

  auto prePass = [](ModulePassManager &MPM, OptimizationLevel Level) {
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
    LPM.addPass(LoopRotatePass(Level != OptimizationLevel::Oz, LTOPreLink));
    // Some loops may have become dead by now. Try to delete them.
    // FIXME: see discussion in https://reviews.llvm.org/D112851,
    //        this may need to be revisited once we run GVN before
    //        loop deletion in the simplification pipeline.
    LPM.addPass(LoopDeletionPass());

    LPM.addPass(llvm::LoopFullUnrollPass());
    OptimizePM.addPass(createFunctionToLoopPassAdaptor(std::move(LPM)));

    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizePM)));
  };

#if LLVM_VERSION_MAJOR >= 20
  auto loadPass = [prePass](ModulePassManager &MPM, OptimizationLevel Level,
                            ThinOrFullLTOPhase)
#else
  auto loadPass = [prePass](ModulePassManager &MPM, OptimizationLevel Level)
#endif
  {
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));

    if (!EnzymeEnable)
      return;

    if (EnzymeDisablePreOpt) {
      if (Level != OptimizationLevel::O0)
        prePass(MPM, Level);
    }

    MPM.addPass(llvm::AlwaysInlinerPass());
    FunctionPassManager OptimizerPM;
    FunctionPassManager OptimizerPM2;
#if LLVM_VERSION_MAJOR >= 16
    OptimizerPM.addPass(llvm::GVNPass());
    OptimizerPM.addPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
#else
    OptimizerPM.addPass(llvm::GVNPass());
    OptimizerPM.addPass(llvm::SROAPass());
#endif
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizerPM)));

#ifdef ENZYME_ENABLE_FPOPT
    // All of these ablations are designed to be run at -O0
    FunctionPassManager herbieFPM;
    if (FPOptExtraMemOpt) {
      llvm::dbgs() << "Running mem2reg" << "\n";
      herbieFPM.addPass(llvm::PromotePass());
    }

    // check if we need to queue reassociations
    if (FPOptExtraPreReassoc) {
      herbieFPM.addPass(llvm::ReassociatePass());
    }

    if (FPOptExtraIfConversion) {
      llvm::SimplifyCFGOptions o;
      o.ConvertSwitchToLookupTable = true;
      o.ConvertSwitchRangeToICmp = true;
      o.NeedCanonicalLoop = false;
      o.SinkCommonInsts = true;
      o.HoistCommonInsts = true;
      o.ForwardSwitchCondToPhi = false;
      o.FoldTwoEntryPHINode = true; // Important for PHI->select
      o.SimplifyCondBranch = true;  // Important for if-conversion
      herbieFPM.addPass(llvm::SimplifyCFGPass(o));
    }

    if (FPOptExtraPreCSE) {
      // easy cases
      herbieFPM.addPass(llvm::EarlyCSEPass(true));
      // 'harder'/edge cases
      herbieFPM.addPass(llvm::GVNPass());
    }

    if (FPOptExtraPreReassoc || FPOptExtraIfConversion || FPOptExtraPreCSE)
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(herbieFPM)));

    if (EnzymeEnableFPOpt)
      MPM.addPass(FPOptNewPM());

    FunctionPassManager herbieFPM2;
    if (FPOptExtraPostCSE) {
      // easy cases
      herbieFPM2.addPass(llvm::EarlyCSEPass(true));
      // 'harder'/edge cases
      herbieFPM2.addPass(llvm::GVNPass());
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(herbieFPM2)));
    }
#endif
    MPM.addPass(EnzymeNewPM(/*PostOpt=*/true));
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ false));
#if LLVM_VERSION_MAJOR >= 16
    OptimizerPM2.addPass(llvm::GVNPass());
    OptimizerPM2.addPass(llvm::SROAPass(llvm::SROAOptions::PreserveCFG));
#else
    OptimizerPM2.addPass(llvm::GVNPass());
    OptimizerPM2.addPass(llvm::SROAPass());
#endif

    LoopPassManager LPM1;
    LPM1.addPass(LoopDeletionPass());
    OptimizerPM2.addPass(createFunctionToLoopPassAdaptor(std::move(LPM1)));

    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(OptimizerPM2)));
    MPM.addPass(GlobalOptPass());
  };
  // TODO need for perf reasons to move Enzyme pass to the pre vectorization.
  PB.registerOptimizerEarlyEPCallback(loadPass);

  auto loadNVVM = [](ModulePassManager &MPM, OptimizationLevel) {
    MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));
  };

  // We should register at vectorizer start for consistency, however,
  // that requires a functionpass, and we have a modulepass.
  // PB.registerVectorizerStartEPCallback(loadPass);
  PB.registerPipelineStartEPCallback(loadNVVM);
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadNVVM);

  auto preLTOPass = [](ModulePassManager &MPM, OptimizationLevel Level) {
    // Create a function that performs CFI checks for cross-DSO calls with
    // targets in the current module.
    MPM.addPass(CrossDSOCFIPass());

    if (Level == OptimizationLevel::O0) {
      return;
    }

    // Try to run OpenMP optimizations, quick no-op if no OpenMP metadata
    // present.
#if LLVM_VERSION_MAJOR >= 16
    MPM.addPass(OpenMPOptPass(ThinOrFullLTOPhase::FullLTOPostLink));
#else
    MPM.addPass(OpenMPOptPass());
#endif

    // Remove unused virtual tables to improve the quality of code
    // generated by whole-program devirtualization and bitset lowering.
    MPM.addPass(GlobalDCEPass());

    // Do basic inference of function attributes from known properties of
    // system libraries and other oracles.
    MPM.addPass(InferFunctionAttrsPass());

    if (Level.getSpeedupLevel() > 1) {
      MPM.addPass(createModuleToFunctionPassAdaptor(CallSiteSplittingPass(),
                                                    EagerlyInvalidateAnalyses));

      // Indirect call promotion. This should promote all the targets that
      // are left by the earlier promotion pass that promotes intra-module
      // targets. This two-step promotion is to save the compile time. For
      // LTO, it should produce the same result as if we only do promotion
      // here.
      // MPM.addPass(PGOIndirectCallPromotion(
      //	true /* InLTO */, PGOOpt && PGOOpt->Action ==
      // PGOOptions::SampleUse));

      // Propagate constants at call sites into the functions they call.
      // This opens opportunities for globalopt (and inlining) by
      // substituting function pointers passed as arguments to direct uses
      // of functions.
#if LLVM_VERSION_MAJOR >= 16
      MPM.addPass(IPSCCPPass(IPSCCPOptions(/*AllowFuncSpec=*/
                                           Level != OptimizationLevel::Os &&
                                           Level != OptimizationLevel::Oz)));
#else
      MPM.addPass(IPSCCPPass());
#endif

      // Attach metadata to indirect call sites indicating the set of
      // functions they may target at run-time. This should follow IPSCCP.
      MPM.addPass(CalledValuePropagationPass());
    }

    // Now deduce any function attributes based in the current code.
    MPM.addPass(
        createModuleToPostOrderCGSCCPassAdaptor(PostOrderFunctionAttrsPass()));

    // Do RPO function attribute inference across the module to
    // forward-propagate attributes where applicable.
    // FIXME: Is this really an optimization rather than a
    // canonicalization?
    MPM.addPass(ReversePostOrderFunctionAttrsPass());

    // Use in-range annotations on GEP indices to split globals where
    // beneficial.
    MPM.addPass(GlobalSplitPass());

    // Run whole program optimization of virtual call when the list of
    // callees is fixed. MPM.addPass(WholeProgramDevirtPass(ExportSummary,
    // nullptr));

    // Stop here at -O1.
    if (Level == OptimizationLevel::O1) {
      return;
    }

    // Optimize globals to try and fold them into constants.
    MPM.addPass(GlobalOptPass());

    // Promote any localized globals to SSA registers.
    MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));

    // Linking modules together can lead to duplicate global constant,
    // only keep one copy of each constant.
    MPM.addPass(ConstantMergePass());

    // Remove unused arguments from functions.
    MPM.addPass(DeadArgumentEliminationPass());

    // Reduce the code after globalopt and ipsccp.  Both can open up
    // significant simplification opportunities, and both can propagate
    // functions through function pointers.  When this happens, we often
    // have to resolve varargs calls, etc, so let instcombine do this.
    FunctionPassManager PeepholeFPM;
    PeepholeFPM.addPass(InstCombinePass());
    if (Level.getSpeedupLevel() > 1)
      PeepholeFPM.addPass(AggressiveInstCombinePass());

    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(PeepholeFPM),
                                                  EagerlyInvalidateAnalyses));

    // Note: historically, the PruneEH pass was run first to deduce
    // nounwind and generally clean up exception handling overhead. It
    // isn't clear this is valuable as the inliner doesn't currently care
    // whether it is inlining an invoke or a call. Run the inliner now.
    if (EnableModuleInliner) {
      MPM.addPass(ModuleInlinerPass(getInlineParamsFromOptLevel(Level),
                                    UseInlineAdvisor,
                                    ThinOrFullLTOPhase::FullLTOPostLink));
    } else {
      MPM.addPass(ModuleInlinerWrapperPass(
          getInlineParamsFromOptLevel(Level),
          /* MandatoryFirst */ true,
          InlineContext{ThinOrFullLTOPhase::FullLTOPostLink,
                        InlinePass::CGSCCInliner}));
    }

    // Perform context disambiguation after inlining, since that would
    // reduce the amount of additional cloning required to distinguish the
    // allocation contexts. if (EnableMemProfContextDisambiguation)
    //	MPM.addPass(MemProfContextDisambiguation());

    // Optimize globals again after we ran the inliner.
    MPM.addPass(GlobalOptPass());

    // Run the OpenMPOpt pass again after global optimizations.
#if LLVM_VERSION_MAJOR >= 16
    MPM.addPass(OpenMPOptPass(ThinOrFullLTOPhase::FullLTOPostLink));
#else
    MPM.addPass(OpenMPOptPass());
#endif

    // Garbage collect dead functions.
    MPM.addPass(GlobalDCEPass());

    // If we didn't decide to inline a function, check to see if we can
    // transform it to pass arguments by value instead of by reference.
    MPM.addPass(
        createModuleToPostOrderCGSCCPassAdaptor(ArgumentPromotionPass()));

    FunctionPassManager FPM;
    // The IPO Passes may leave cruft around. Clean up after them.
    FPM.addPass(InstCombinePass());

    if (EnableConstraintElimination)
      FPM.addPass(ConstraintEliminationPass());

    FPM.addPass(JumpThreadingPass());

    // Do a post inline PGO instrumentation and use pass. This is a context
    // sensitive PGO pass.
#if 0
		  if (PGOOpt) {
			if (PGOOpt->CSAction == PGOOptions::CSIRInstr)
			  addPGOInstrPasses(MPM, Level, /* RunProfileGen */ true,
								/* IsCS */ true, PGOOpt->CSProfileGenFile,
								PGOOpt->ProfileRemappingFile,
								ThinOrFullLTOPhase::FullLTOPostLink, PGOOpt->FS);
			else if (PGOOpt->CSAction == PGOOptions::CSIRUse)
			  addPGOInstrPasses(MPM, Level, /* RunProfileGen */ false,
								/* IsCS */ true, PGOOpt->ProfileFile,
								PGOOpt->ProfileRemappingFile,
								ThinOrFullLTOPhase::FullLTOPostLink, PGOOpt->FS);
		  }
#endif

    // Break up allocas
#if LLVM_VERSION_MAJOR >= 16
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
#else
    FPM.addPass(SROAPass());
#endif

    // LTO provides additional opportunities for tailcall elimination due
    // to link-time inlining, and visibility of nocapture attribute.
    FPM.addPass(TailCallElimPass());

    // Run a few AA driver optimizations here and now to cleanup the code.
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM),
                                                  EagerlyInvalidateAnalyses));

    MPM.addPass(
        createModuleToPostOrderCGSCCPassAdaptor(PostOrderFunctionAttrsPass()));

    // Require the GlobalsAA analysis for the module so we can query it
    // within MainFPM.
    MPM.addPass(RequireAnalysisPass<GlobalsAA, Module>());
  };

  auto loadLTO = [preLTOPass, loadPass](ModulePassManager &MPM,
                                        OptimizationLevel Level) {
    preLTOPass(MPM, Level);
    MPM.addPass(
        createModuleToPostOrderCGSCCPassAdaptor(PostOrderFunctionAttrsPass()));

    // Require the GlobalsAA analysis for the module so we can query it
    // within MainFPM.
    MPM.addPass(RequireAnalysisPass<GlobalsAA, Module>());

    // Invalidate AAManager so it can be recreated and pick up the newly
    // available GlobalsAA.
    MPM.addPass(
        createModuleToFunctionPassAdaptor(InvalidateAnalysisPass<AAManager>()));

    FunctionPassManager MainFPM;
    MainFPM.addPass(createFunctionToLoopPassAdaptor(
        LICMPass(SetLicmMssaOptCap, SetLicmMssaNoAccForPromotionCap,
                 /*AllowSpeculation=*/true),
        /*USeMemorySSA=*/true, /*UseBlockFrequencyInfo=*/false));

    if (RunNewGVN)
      MainFPM.addPass(NewGVNPass());
    else
      MainFPM.addPass(GVNPass());

    // Remove dead memcpy()'s.
    MainFPM.addPass(MemCpyOptPass());

    // Nuke dead stores.
    MainFPM.addPass(DSEPass());
#if LLVM_VERSION_MAJOR >= 17
    MainFPM.addPass(MoveAutoInitPass());
#endif
    MainFPM.addPass(MergedLoadStoreMotionPass());

    LoopPassManager LPM;
    if (EnableLoopFlatten && Level.getSpeedupLevel() > 1)
      LPM.addPass(LoopFlattenPass());
    LPM.addPass(IndVarSimplifyPass());
    LPM.addPass(LoopDeletionPass());
    // FIXME: Add loop interchange.

#if LLVM_VERSION_MAJOR >= 20
    loadPass(MPM, Level, ThinOrFullLTOPhase::None);
#else
    loadPass(MPM, Level);
#endif
  };
  PB.registerFullLinkTimeOptimizationEarlyEPCallback(loadLTO);
}

void registerEnzyme(llvm::PassBuilder &PB) {
#ifdef ENZYME_RUNPASS
  augmentPassBuilder(PB);
#endif
  PB.registerPipelineParsingCallback(
      [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "enzyme") {
          MPM.addPass(EnzymeNewPM());
          return true;
        }
#ifdef ENZYME_ENABLE_FPOPT
        if (Name == "fp-opt") {
          MPM.addPass(FPOptNewPM());
          return true;
        }
#endif
        if (Name == "preserve-nvvm") {
          MPM.addPass(PreserveNVVMNewPM(/*Begin*/ true));
          return true;
        }
        if (Name == "print-type-analysis") {
          MPM.addPass(TypeAnalysisPrinterNewPM());
          return true;
        }
        return false;
      });
  PB.registerPipelineParsingCallback(
      [](llvm::StringRef Name, llvm::FunctionPassManager &FPM,
         llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
        if (Name == "print-activity-analysis") {
          FPM.addPass(ActivityAnalysisPrinterNewPM());
          return true;
        }
        if (Name == "jl-inst-simplify") {
          FPM.addPass(JLInstSimplifyNewPM());
          return true;
        }
        return false;
      });
}

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "EnzymeNewPM", "v0.1", registerEnzyme};
}
