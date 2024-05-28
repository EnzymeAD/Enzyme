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
#include <regex>
#include <sstream>
#include <string>
#include <vector> // TODO: SmallVector??

#include "Herbie.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "fp-opt"

class FPNode {
public:
  std::string op;
  std::string symbol;
  std::vector<FPNode *> children;

  FPNode(const std::string &op) : op(op) {}

  FPNode(const std::string &op, const std::string &symbol)
      : op(op), symbol(symbol) {}

  void addChild(FPNode *child) { children.push_back(child); }

  std::string
  toFullExpression(std::map<std::string, FPNode *> &symbolToNodeMap) {
    if (!children.empty()) {
      std::string expr = "(" + op;
      for (FPNode *child : children) {
        if (symbolToNodeMap.count(child->symbol)) {
          expr += " " + symbolToNodeMap[child->symbol]->toFullExpression(
                            symbolToNodeMap);
        } else {
          expr += " " + child->symbol;
        }
      }
      expr += ")";
      return expr;
    } else {
      return symbol;
    }
  }
};

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
  std::string content((std::istreambuf_iterator<char>(output)),
                      std::istreambuf_iterator<char>());
  output.close();

  llvm::errs() << "Herbie output:\n" << content << "\n";

  std::string token;
  std::regex fpcoreRegex(":alt\\s*\\(\\)\\s*(.*)\\s*\\)");
  std::smatch matches;
  std::string args, properties, optimizedExpr;

  if (std::regex_search(content, matches, fpcoreRegex)) {
    optimizedExpr = matches[1].str();
    llvm::errs() << "Optimized expression: " << optimizedExpr
                 << "\n"; // TODO: Constant?
  } else {
    llvm::errs() << "Failed to parse Herbie output!\n";
  }
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

unsigned getLLVMOpcode(const std::string &herbieOp) {
  if (herbieOp == "+.f64")
    return Instruction::FAdd;
  if (herbieOp == "-.f64")
    return Instruction::FSub;
  if (herbieOp == "*.f64")
    return Instruction::FMul;
  if (herbieOp == "/.f64")
    return Instruction::FDiv;
  return Instruction::UserOp1;
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F) {
  bool changed = false;
  std::string herbieInput;
  std::map<Value *, std::string> valueToSymbolMap;
  std::map<std::string, Value *> symbolToValueMap;
  std::map<std::string, FPNode *> symbolToNodeMap;

  std::map<BasicBlock *, std::string>
      blockToHerbieExprMap; // BB to be optimized --> Herbie expressions
  std::map<std::string, std::vector<Instruction *>>
      herbieExprToInstMap; // Herbie expressions --> original instructions

  std::set<std::string> arguments; // TODO: for different basic blocks
  int symbolCounter = 0;

  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "__v" + std::to_string(symbolCounter++);
  };

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting  llvm instructions into herbie string (FPNode ....)
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *op = dyn_cast<BinaryOperator>(&I)) { // TODO: Other operators?
        if (op->getType()->isFloatingPointTy()) {
          FPNode *node = new FPNode(getHerbieOperator(I));
          for (unsigned i = 0; i < op->getNumOperands(); ++i) {
            Value *operand = op->getOperand(i);
            std::string operandSymbol =
                valueToSymbolMap.count(operand)
                    ? valueToSymbolMap[operand]
                    : (valueToSymbolMap[operand] = getNextSymbol());
            symbolToValueMap[operandSymbol] = operand;

            FPNode *childNode = symbolToNodeMap.count(operandSymbol)
                                    ? symbolToNodeMap[operandSymbol]
                                    : (symbolToNodeMap[operandSymbol] =
                                           new FPNode("__arg", operandSymbol));
            node->addChild(childNode);

            if (childNode->op == "__arg") {
              arguments.insert(operandSymbol);
            }
          }

          std::string symbol = getNextSymbol();
          node->symbol = symbol;
          valueToSymbolMap[&I] = symbol;
          symbolToNodeMap[symbol] = node;
        }
      }
    }
  }

  for (auto &BB : F) {
    // Get last instruction in the basic block which is FP instruction
    // Get the largest Herbie expression (i.e., Herbie expression of the last
    // instruction in BB) of BB using valueToSymbolMap and toFullExpression
    Value *lastFPInst = nullptr;
    for (auto I = BB.rbegin(); I != BB.rend(); ++I) {
      if (valueToSymbolMap.count(&*I)) {
        lastFPInst = &*I;
        break;
      }
    }
    if (lastFPInst) {
      std::string bbHerbieExpr =
          symbolToNodeMap[valueToSymbolMap[lastFPInst]]->toFullExpression(
              symbolToNodeMap);
      blockToHerbieExprMap[&BB] = bbHerbieExpr;
      for (auto &I : BB) {
        // Map all FP instructions to the largest herbie expression of BB.
        if (valueToSymbolMap.count(&I)) {
          herbieExprToInstMap[bbHerbieExpr].push_back(&I);
        }
      }
    }
  }

  for (auto &BB : F) {
    if (blockToHerbieExprMap.count(&BB)) {
      // TODO: Assume same arguments for all basic blocks
      std::string argumentsStr = "(";
      for (const auto &arg : arguments) {
        argumentsStr += arg + " ";
      }
      argumentsStr.pop_back();
      argumentsStr += ")";

      std::string herbieInput =
          "(FPCore " + argumentsStr + " " + blockToHerbieExprMap[&BB] + ")";
      llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

      // 3) run fancy opts
      runViaHerbie(herbieInput);
    }
  }

  // llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

  // // 3) run fancy opts
  // runViaHerbie(herbieInput);

  // // 4) parse the output string solution from herbieland
  // // 5) convert into a solution in llvm vals/instructions

  // // Extract the Herbie operator and operands

  // std::istringstream exprStream(optimizedExpr);
  // std::string herbieOp, op1, op2;
  // exprStream >> herbieOp >> op1 >> op2;

  // llvm::errs() << "Op: " << herbieOp << ", op1: " << op1 << ", op2: " <<
  // op2
  //              << "\n";

  // // Find the corresponding LLVM values
  // Value *val1 = symbolToValueMap[op1];
  // Value *val2 = symbolToValueMap[op2];
  // assert(val1 && val2);

  // // Map Herbie operator back to LLVM opcode
  // unsigned llvmOpcode = getLLVMOpcode(herbieOp);
  // Instruction *newOp = nullptr;

  // switch (llvmOpcode) {
  // case Instruction::FAdd:
  //   newOp = BinaryOperator::CreateFAdd(val1, val2, "opt");
  //   break;
  // case Instruction::FSub:
  //   newOp = BinaryOperator::CreateFSub(val1, val2, "opt");
  //   break;
  // case Instruction::FMul:
  //   newOp = BinaryOperator::CreateFMul(val1, val2, "opt");
  //   break;
  // case Instruction::FDiv:
  //   newOp = BinaryOperator::CreateFDiv(val1, val2, "opt");
  //   break;
  // default:
  //   llvm::errs() << "Unknown operator: " << herbieOp << "\n";
  // }

  // if (newOp) {
  //   llvm::errs() << "Optimized: " << *val1 << " " << herbieOp << " " <<
  //   *val2
  //                << " -> " << *newOp << "\n";
  //   herbieExprToOptInstsMap[optimizedExpr].push_back(newOp);
  //   changed = true;
  // }

  // for (auto &instMapPair : instToHerbieExprMap) {
  //   auto inst = instMapPair.first;
  //   auto herbieExpr = instMapPair.second;
  //   llvm::errs() << "Checking Inst: " << *inst
  //                << ", Herbie expr: " << herbieExpr << "\n";
  //   if (0 != herbieExprToOptInstsMap.count(herbieExpr)) {
  //     llvm::errs() << "Replacing: " << *inst << " with "
  //                  << *herbieExprToOptInstsMap[herbieExpr] << "\n";
  //     auto *optInst = herbieExprToOptInstsMap[herbieExpr];
  //     inst->replaceAllUsesWith(optInst);
  //     inst->getParent()->getInstList().insert(inst->getIterator(),
  //     optInst); inst->eraseFromParent();
  //   }
  // }

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
