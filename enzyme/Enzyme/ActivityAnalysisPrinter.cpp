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

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

#include "ActivityAnalysis.h"
#include "ActivityAnalysisPrinter.h"
#include "EnzymeLogic.h"
#include "FunctionUtils.h"
#include "TypeAnalysis/TypeAnalysis.h"
#include "Utils.h"

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
      if (a.getContext().supportsTypedPointers()) {
        auto et = a.getType()->getPointerElementType();
        if (et->isFPOrFPVectorTy()) {
          dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
        } else if (et->isPointerTy()) {
          dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
        }
      }
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
    if (F.getContext().supportsTypedPointers()) {
      auto et = F.getReturnType()->getPointerElementType();
      if (et->isFPOrFPVectorTy()) {
        dt = TypeTree(ConcreteType(et->getScalarType())).Only(-1, nullptr);
      } else if (et->isPointerTy()) {
        dt = TypeTree(ConcreteType(BaseType::Pointer)).Only(-1, nullptr);
      }
    }
#endif
  } else if (F.getReturnType()->isIntOrIntVectorTy()) {
    dt = ConcreteType(BaseType::Integer);
  }
  type_args.Return = dt.Only(-1, nullptr);

  EnzymeLogic Logic(false);
  TypeAnalysis TA(Logic);
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
  ActivityAnalyzer ATA(Logic.PPC, Logic.PPC.FAM.getResult<AAManager>(F),
                       notForAnalysis, TLI, ConstantValues, ActiveValues,
                       ActiveReturns);

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

class ActivityAnalysisPrinter final : public ModulePass {
public:
  static char ID;
  ActivityAnalysisPrinter() : ModulePass(ID) {}

  bool runOnModule(Module &M) override {
    // Check if function name is specified
    if (FunctionToAnalyze.empty()) {
      // Use the first function in the module as context for the diagnostic
      Function *FirstFunc = nullptr;
      for (auto &F : M) {
        if (!F.isDeclaration()) {
          FirstFunc = &F;
          break;
        }
      }
      
      if (FirstFunc) {
        EmitFailure("NoFunctionSpecified", FirstFunc->getSubprogram(),
                    FirstFunc,
                    "No function specified for -activity-analysis-func");
      } else {
        report_fatal_error("No function specified for -activity-analysis-func");
      }
      return false;
    }
    
    // Check if the specified function exists
    Function *TargetFunc = M.getFunction(FunctionToAnalyze);
    
    if (!TargetFunc) {
      // Use the first function in the module as context for the diagnostic
      Function *FirstFunc = nullptr;
      for (auto &F : M) {
        if (!F.isDeclaration()) {
          FirstFunc = &F;
          break;
        }
      }
      
      if (FirstFunc) {
        EmitFailure("FunctionNotFound", FirstFunc->getSubprogram(),
                    FirstFunc,
                    "Function '", FunctionToAnalyze,
                    "' specified by -activity-analysis-func not found in module");
      } else {
        // Fallback if no functions in module
        std::string msg = "Function '" + FunctionToAnalyze + 
                          "' specified by -activity-analysis-func not found in module";
        report_fatal_error(StringRef(msg));
      }
      return false;
    }
    
    // Run analysis only on the target function
    auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(*TargetFunc);
    return printActivityAnalysis(*TargetFunc, TLI);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.setPreservesAll();
  }
};

} // namespace

char ActivityAnalysisPrinter::ID = 0;

static RegisterPass<ActivityAnalysisPrinter>
    X("print-activity-analysis", "Print Activity Analysis Results");

ActivityAnalysisPrinterNewPM::Result
ActivityAnalysisPrinterNewPM::run(llvm::Module &M,
                                  llvm::ModuleAnalysisManager &MAM) {
  // Check if function name is specified
  if (FunctionToAnalyze.empty()) {
    // Use the first function in the module as context for the diagnostic
    Function *FirstFunc = nullptr;
    for (auto &F : M) {
      if (!F.isDeclaration()) {
        FirstFunc = &F;
        break;
      }
    }
    
    if (FirstFunc) {
      EmitFailure("NoFunctionSpecified", FirstFunc->getSubprogram(),
                  FirstFunc,
                  "No function specified for -activity-analysis-func");
    } else {
      report_fatal_error("No function specified for -activity-analysis-func");
    }
    return PreservedAnalyses::all();
  }
  
  // Check if the specified function exists
  Function *TargetFunc = M.getFunction(FunctionToAnalyze);
  
  if (!TargetFunc) {
    // Use the first function in the module as context for the diagnostic
    Function *FirstFunc = nullptr;
    for (auto &F : M) {
      if (!F.isDeclaration()) {
        FirstFunc = &F;
        break;
      }
    }
    
    if (FirstFunc) {
      EmitFailure("FunctionNotFound", FirstFunc->getSubprogram(),
                  FirstFunc,
                  "Function '", FunctionToAnalyze,
                  "' specified by -activity-analysis-func not found in module");
    } else {
      // Fallback if no functions in module
      std::string msg = "Function '" + FunctionToAnalyze + 
                        "' specified by -activity-analysis-func not found in module";
      report_fatal_error(StringRef(msg));
    }
    return PreservedAnalyses::all();
  }
  
  // Run analysis only on the target function
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool changed = printActivityAnalysis(*TargetFunc, FAM.getResult<TargetLibraryAnalysis>(*TargetFunc));
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey ActivityAnalysisPrinterNewPM::Key;
