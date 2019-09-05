//===- Enzyme.cpp - Example code from "Writing an LLVM Pass" ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements two versions of the LLVM "Enzyme World" pass described
// in docs/WritingAnLLVMPass.html
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "enzyme"

STATISTIC(EnzymeCounter, "Counts number of functions greeted");

namespace {
  // Enzyme - The first implementation, without getAnalysisUsage.
  struct Enzyme : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    Enzyme() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
      ++EnzymeCounter;
      errs() << "Enzyme: ";
      errs().write_escaped(F.getName()) << '\n';
      return false;
    }
  };
}

char Enzyme::ID = 0;
static RegisterPass<Enzyme> X("enzyme", "Enzyme World Pass");
