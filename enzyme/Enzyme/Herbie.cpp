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

#include <chrono>
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
      : op(op), symbol(symbol) {
    // llvm::errs() << "Creating FPNode: " << op << " " << symbol << "\n";
  }

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

FPNode *parseHerbieExpr(const std::string &expr) {
  llvm::errs() << "Parsing: " << expr << "\n";
  auto trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  // Arguments
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    // llvm::errs() << "Base case: " << trimmedExpr << "\n";
    return new FPNode("__arg", trimmedExpr);
  }

  // Constants
  std::regex constantPattern("^#s\\(literal\\s+([\\d\\.]+)\\s+\\w+\\)$");
  std::smatch matches;
  if (std::regex_match(trimmedExpr, matches, constantPattern)) {
    llvm::errs() << "Found __const " << matches[1].str() << "\n";
    return new FPNode("__const", matches[1].str());
  }

  assert(trimmedExpr.front() == '(' && trimmedExpr.back() == ')');
  trimmedExpr = trimmedExpr.substr(1, trimmedExpr.size() - 2);

  // Get the operator
  auto endOp = trimmedExpr.find(' ');
  std::string op = trimmedExpr.substr(0, endOp);

  // TODO: Simply remove the type for now
  size_t pos = op.find('.');
  if (pos != std::string::npos) {
    op = op.substr(0, pos);
  }

  FPNode *node = new FPNode(op);

  int depth = 0;
  auto start = trimmedExpr.find_first_not_of(" ", endOp);
  std::string::size_type curr;
  for (curr = start; curr < trimmedExpr.size(); ++curr) {
    // llvm::errs() << "Curr: " << trimmedExpr[curr] << "\n";
    if (trimmedExpr[curr] == '(')
      depth++;
    if (trimmedExpr[curr] == ')')
      depth--;
    if (depth == 0 && trimmedExpr[curr] == ' ') {
      // llvm::errs() << "Adding child for " << trimmedExpr << ": "
      //              << trimmedExpr.substr(start, curr - start) << "\n";
      node->addChild(parseHerbieExpr(trimmedExpr.substr(start, curr - start)));
      start = curr + 1;
    }
  }
  if (start < curr) {
    node->addChild(parseHerbieExpr(trimmedExpr.substr(start, curr - start)));
  }

  return node;
}

Value *herbieExprToValue(FPNode *node, Instruction *insertBefore,
                         IRBuilder<> &builder,
                         std::map<std::string, Value *> &symbolToValueMap) {
  assert(node);

  if (node->op == "__arg") {
    llvm::errs() << "Returning: " << node->symbol << "\n";
    return symbolToValueMap[node->symbol];
  }

  if (node->op == "__const") {
    llvm::errs() << "Returning constant: " << node->symbol << "\n";
    double constantValue = std::stod(node->symbol);
    return ConstantFP::get(builder.getDoubleTy(), constantValue);
  }

  std::vector<Value *> operands;
  for (FPNode *child : node->children) {
    operands.push_back(
        herbieExprToValue(child, insertBefore, builder, symbolToValueMap));
  }

  Value *val = nullptr;
  builder.SetInsertPoint(insertBefore);

  std::string &op = node->op;

  if (op == "+") {
    assert(operands[0]);
    assert(operands[1]);
    val = builder.CreateFAdd(operands[0], operands[1], "faddtmp");
  } else if (op == "-") {
    val = builder.CreateFSub(operands[0], operands[1], "fsubtmp");
  } else if (op == "*") {
    val = builder.CreateFMul(operands[0], operands[1], "fmultmp");
  } else if (op == "/") {
    val = builder.CreateFDiv(operands[0], operands[1], "fdivtmp");
  } else {
    llvm::errs() << "Unknown operator: " << node->op << "\n";
  }

  return val;
}

Value *getLastFPInst(BasicBlock &BB,
                     std::map<Value *, std::string> &valueToSymbolMap) {
  Value *lastFPInst = nullptr;
  for (auto I = BB.rbegin(); I != BB.rend(); ++I) {
    if (valueToSymbolMap.count(&*I)) {
      lastFPInst = &*I;
      break;
    }
  }
  return lastFPInst;
}

bool improveViaHerbie(std::string &expr) {
  auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
  auto millis =
      std::chrono::duration_cast<std::chrono::milliseconds>(now).count();

  std::string tmpin = "/tmp/herbie_input_" + std::to_string(millis);
  std::string tmpout = "/tmp/herbie_output_" + std::to_string(millis);

  std::remove(tmpout.c_str());
  std::ofstream input(tmpin);
  if (!input) {
    llvm::errs() << "Failed to open input file.\n";
    return 1;
  }
  input << expr;
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
    return false;
  }
  std::remove(tmpin.c_str());

  std::ifstream output(tmpout);
  if (!output) {
    llvm::errs() << "Failed to open output file.\n";
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(output)),
                      std::istreambuf_iterator<char>());
  output.close();
  std::remove(tmpout.c_str());

  llvm::errs() << "Herbie output:\n" << content << "\n";

  std::string token;
  std::regex fpcoreRegex(":alt\\s*\\(\\)\\s*(.*)\\s*\\)");
  std::smatch matches;
  std::string optimizedExpr;

  if (std::regex_search(content, matches, fpcoreRegex)) {
    llvm::errs() << "Optimized expression: " << optimizedExpr << "\n";
    expr = matches[1].str();
    return true;
  } else {
    llvm::errs() << "Failed to parse Herbie output!\n";
    return false;
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
          symbolToValueMap[symbol] = &I;
        }
      }
    }
  }

  for (auto &BB : F) {
    // Get last instruction in the basic block which is FP instruction
    // Get the largest Herbie expression (i.e., Herbie expression of the last
    // instruction in BB) of BB using valueToSymbolMap and toFullExpression
    Value *lastFPInst = getLastFPInst(BB, valueToSymbolMap);
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

      std::string herbieExpr =
          "(FPCore " + argumentsStr + " " + blockToHerbieExprMap[&BB] + ")";
      llvm::errs() << "Herbie input:\n" << herbieExpr << "\n";

      // 3) run fancy opts
      if (!improveViaHerbie(herbieExpr)) {
        llvm::errs() << "Failed to optimize " << blockToHerbieExprMap[&BB]
                     << " using Herbie!\n";
        return changed;
      } else {
        llvm::errs() << "Optimized: " << blockToHerbieExprMap[&BB] << " -> "
                     << herbieExpr << "\n";
      }

      // 4) parse the output string solution from herbieland
      // 5) convert into a solution in llvm vals/instructions
      llvm::errs() << "Parsing Herbie Expr: " << herbieExpr << "\n";
      FPNode *parsedNode = parseHerbieExpr(herbieExpr);
      llvm::errs() << "Parsed Herbie Expr: "
                   << parsedNode->toFullExpression(symbolToNodeMap) << "\n";

      Instruction *insertBefore = BB.getTerminator();
      IRBuilder<> builder(&BB);
      builder.setFastMathFlags(getFast());
      builder.SetInsertPoint(insertBefore);

      // Convert the parsed expression to LLVM values/instructions
      Value *newRootValue = herbieExprToValue(parsedNode, insertBefore, builder,
                                              symbolToValueMap);
      Value *oldRootValue = getLastFPInst(BB, valueToSymbolMap);
      llvm::errs() << "Replacing: " << *oldRootValue << " with "
                   << *newRootValue << "\n";
      oldRootValue->replaceAllUsesWith(newRootValue);

      auto &eraseList = herbieExprToInstMap[blockToHerbieExprMap[&BB]];
      for (auto it = eraseList.rbegin(); it != eraseList.rend(); ++it) {
        llvm::errs() << "Removing: " << **it << "\n";
        (*it)->eraseFromParent();
      }

      changed = true;
    }
  }

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
