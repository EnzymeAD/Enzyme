#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include <llvm/ADT/StringRef.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/Program.h>

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "Herbie.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "fp-opt"

void runViaHerbie(const std::string &cmd) {
  std::string tmpin = "/tmp/herbie_input";
  std::string tmpout = "/tmp/herbie_output";

  std::remove(tmpout.c_str());
  std::ofstream input(tmpin);
  if (!input) {
    llvm::errs() << "Failed to open input file.\n";
    return;
  }
  input << cmd;
  input.close();

  std::string Program = HERBIE_BINARY;
  llvm::StringRef Args[] = {Program, "improve", tmpin, tmpout};
  std::string ErrMsg;
  bool ExecutionFailed = false;

  llvm::errs() << "Executing: " << Program << "\n";

  llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/llvm::None,
                            /*Redirects=*/llvm::None,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg,
                            &ExecutionFailed);

  if (ExecutionFailed) {
    llvm::errs() << "Execution failed: " << ErrMsg << "\n";
    return;
  }

  std::ifstream output(tmpout);
  if (!output) {
    llvm::errs() << "Failed to open output file.\n";
    return;
  }

  std::string line;
  llvm::errs() << "Herbie output:\n";
  while (std::getline(output, line)) {
    llvm::errs() << line << "\n";
  }
  output.close();
}

std::string getHerbieOperator(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FAdd:
    return "+";
  case Instruction::FSub:
    return "-";
  case Instruction::FMul:
    return "*";
  case Instruction::FDiv:
    return "/";
  default:
    return "UnknownOp";
  }
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F) {
  bool changed = false;
  std::string herbieInput;
  std::map<Value *, std::string> valueToSymbolMap;
  std::map<std::string, Value *> symbolToValueMap;
  std::set<std::string> arguments;
  int symbolCounter = 0;

  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting  llvm instructions into herbie string (FPNode ....)
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *op = dyn_cast<BinaryOperator>(&I)) {
        if (op->getType()->isFloatingPointTy()) {
          std::string lhs =
              valueToSymbolMap.count(op->getOperand(0))
                  ? valueToSymbolMap[op->getOperand(0)]
                  : (valueToSymbolMap[op->getOperand(0)] = getNextSymbol());
          std::string rhs =
              valueToSymbolMap.count(op->getOperand(1))
                  ? valueToSymbolMap[op->getOperand(1)]
                  : (valueToSymbolMap[op->getOperand(1)] = getNextSymbol());

          arguments.insert(lhs);
          arguments.insert(rhs);

          std::string symbol = getNextSymbol();
          valueToSymbolMap[&I] = symbol;
          symbolToValueMap[symbol] = &I;

          std::string herbieNode = "(";
          herbieNode += getHerbieOperator(I);
          herbieNode += " ";
          herbieNode += lhs;
          herbieNode += " ";
          herbieNode += rhs;
          herbieNode += ")";
          herbieInput += herbieNode;
        }
      }
    }
  }

  if (herbieInput.empty()) {
    return changed;
  }

  std::string argumentsStr = "(";
  for (const auto &arg : arguments) {
    argumentsStr += arg + " ";
  }
  argumentsStr.pop_back();
  argumentsStr += ")";

  herbieInput = "(FPCore " + argumentsStr + " " + herbieInput + ")";

  llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

  // 3) run fancy opts
  runViaHerbie(herbieInput);

  // 4) parse the output string solution from herbieland
  // 5) convert into a solution in llvm vals/instructions
  return changed;
}

namespace {

class FPOpt final : public FunctionPass {
public:
  static char ID;
  FPOpt() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {}
  bool runOnFunction(Function &F) override { return fpOptimize(F); }
};

} // namespace

char FPOpt::ID = 0;

static RegisterPass<FPOpt> X("fp-opt",
                             "Run Enzyme/Herbie Floating point optimizations");

FunctionPass *createFPOptPass() { return new FPOpt(); }

#include <llvm-c/Core.h>
#include <llvm-c/Types.h>

#include "llvm/IR/LegacyPassManager.h"

extern "C" void AddFPOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFPOptPass());
}

FPOptNewPM::Result FPOptNewPM::run(llvm::Module &M,
                                   llvm::ModuleAnalysisManager &MAM) {
  bool changed = false;
  for (auto &F : M)
    changed |= fpOptimize(F);
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey FPOptNewPM::Key;
