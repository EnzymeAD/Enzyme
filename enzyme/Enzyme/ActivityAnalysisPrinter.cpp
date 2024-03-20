// ActivityAnalysisPrinter.cpp - Printer utility pass for Activity Analysis =//
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
// This file contains a utility LLVM pass for printing derived Activity Analysis
// results of a given function.
//
//===----------------------------------------------------------------------===//
#include <llvm/Config/llvm-config.h>
#include <map>
#include <set>
#include <stdint.h>
#include <string>
#include <utility>

#if LLVM_VERSION_MAJOR >= 16
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "ActivityAnalysis.h"
#include "ActivityAnalysisPrinter.h"
#include "FunctionUtils.h"
#include "TypeAnalysis/BaseType.h"
#include "TypeAnalysis/ConcreteType.h"
#include "TypeAnalysis/TypeAnalysis.h"
#include "TypeAnalysis/TypeTree.h"
#include "Utils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "activity-analysis-results"

/// Function TypeAnalysis will be starting its run from
static llvm::cl::opt<std::string>
    FunctionToAnalyze("activity-analysis-func", cl::init(""), cl::Hidden,
                      cl::desc("Which function to analyze/print"));

static llvm::cl::opt<bool>
    InactiveArgs("activity-analysis-inactive-args", cl::init(false), cl::Hidden,
                 cl::desc("Whether all args are inactive"));

static llvm::cl::opt<bool>
    DuplicatedRet("activity-analysis-duplicated-ret", cl::init(false),
                  cl::Hidden, cl::desc("Whether the return is duplicated"));
namespace {

bool printActivityAnalysis(llvm::Function &F, TargetLibraryInfo &TLI) {
  if (F.getName() != FunctionToAnalyze)
    return /*changed*/ false;

  FnTypeInfo type_args(&F);
  for (auto &a : type_args.Function->args()) {
    TypeTree dt;
    if (a.getType()->isFPOrFPVectorTy()) {
      dt = ConcreteType(a.getType()->getScalarType());
    } else if (a.getType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR < 17
#if LLVM_VERSION_MAJOR >= 13
      if (a.getContext().supportsTypedPointers()) {
#endif
        auto et = a.getType()->getPointerElementType();
        if (et->isFPOrFPVectorTy()) {
          dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
        } else if (et->isPointerTy()) {
          dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
        }
#if LLVM_VERSION_MAJOR >= 13
      }
#endif
#endif
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
  if (F.getReturnType()->isFPOrFPVectorTy()) {
    dt = ConcreteType(F.getReturnType()->getScalarType());
  } else if (F.getReturnType()->isPointerTy()) {
#if LLVM_VERSION_MAJOR < 17
#if LLVM_VERSION_MAJOR >= 13
    if (F.getContext().supportsTypedPointers()) {
#endif
      auto et = F.getReturnType()->getPointerElementType();
      if (et->isFPOrFPVectorTy()) {
        dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
      } else if (et->isPointerTy()) {
        dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
      }
#if LLVM_VERSION_MAJOR >= 13
    }
#endif
#endif
  } else if (F.getReturnType()->isIntOrIntVectorTy()) {
    dt = ConcreteType(BaseType::Integer);
  }
  type_args.Return = dt.Only(-1, nullptr);

  PreProcessCache PPC;
  TypeAnalysis TA(PPC.FAM);
  TypeResults TR = TA.analyzeFunction(type_args);

  llvm::SmallPtrSet<llvm::Value *, 4> ConstantValues;
  llvm::SmallPtrSet<llvm::Value *, 4> ActiveValues;
  for (auto &a : type_args.Function->args()) {
    if (InactiveArgs) {
      ConstantValues.insert(&a);
    } else if (a.getType()->isIntOrIntVectorTy()) {
      ConstantValues.insert(&a);
    } else {
      ActiveValues.insert(&a);
    }
  }

  DIFFE_TYPE ActiveReturns = F.getReturnType()->isFPOrFPVectorTy()
                                 ? DIFFE_TYPE::OUT_DIFF
                                 : DIFFE_TYPE::CONSTANT;
  if (DuplicatedRet)
    ActiveReturns = DIFFE_TYPE::DUP_ARG;
  SmallPtrSet<BasicBlock *, 4> notForAnalysis(getGuaranteedUnreachable(&F));
  ActivityAnalyzer ATA(PPC, PPC.FAM.getResult<AAManager>(F), notForAnalysis,
                       TLI, ConstantValues, ActiveValues, ActiveReturns);

  for (auto &a : F.args()) {
    ATA.isConstantValue(TR, &a);
    llvm::errs().flush();
  }
  for (auto &BB : F) {
    for (auto &I : BB) {
      ATA.isConstantInstruction(TR, &I);
      ATA.isConstantValue(TR, &I);
      llvm::errs().flush();
    }
  }

  for (auto &a : F.args()) {
    bool icv = ATA.isConstantValue(TR, &a);
    llvm::errs().flush();
    llvm::outs() << a << ": icv:" << icv << "\n";
    llvm::outs().flush();
  }
  for (auto &BB : F) {
    llvm::outs() << BB.getName() << "\n";
    for (auto &I : BB) {
      bool ici = ATA.isConstantInstruction(TR, &I);
      bool icv = ATA.isConstantValue(TR, &I);
      llvm::errs().flush();
      llvm::outs() << I << ": icv:" << icv << " ici:" << ici << "\n";
      llvm::outs().flush();
    }
  }
  return /*changed*/ false;
}

class ActivityAnalysisPrinter final : public FunctionPass {
public:
  static char ID;
  ActivityAnalysisPrinter() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
  }

  bool runOnFunction(Function &F) override {

    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);

    return printActivityAnalysis(F, TLI);
  }
};

} // namespace

char ActivityAnalysisPrinter::ID = 0;

static RegisterPass<ActivityAnalysisPrinter>
    X("print-activity-analysis", "Print Activity Analysis Results");

ActivityAnalysisPrinterNewPM::Result
ActivityAnalysisPrinterNewPM::run(llvm::Function &F,
                                  llvm::FunctionAnalysisManager &FAM) {
  bool changed = false;
  changed = printActivityAnalysis(F, FAM.getResult<TargetLibraryAnalysis>(F));
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey ActivityAnalysisPrinterNewPM::Key;
