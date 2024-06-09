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
  SmallVector<FPNode *, 1> operands;

  FPNode(const std::string &op) : op(op), symbol() {}
  virtual ~FPNode() = default;

  void addOperand(FPNode *operand) { operands.push_back(operand); }

  bool hasSymbol() const { return !symbol.empty(); }

  virtual std::string
  toFullExpression(std::unordered_map<Value *, FPNode *> &valueToNodeMap) {
    assert(!operands.empty() && "FPNode has no operands!");
    std::string expr = "(" + op;
    for (auto operand : operands) {
      expr += " " + operand->toFullExpression(valueToNodeMap);
    }
    expr += ")";
    return expr;
  }

  virtual Value *getValue(Instruction *insertBefore, IRBuilder<> &builder) {
    std::vector<Value *> operandValues;
    for (auto operand : operands) {
      operandValues.push_back(operand->getValue(insertBefore, builder));
    }

    Value *val = nullptr;
    builder.SetInsertPoint(insertBefore);

    llvm::errs() << "Generating new instruction for op: " << op << "\n";
    if (op == "+") {
      val = builder.CreateFAdd(operandValues[0], operandValues[1]);
    } else if (op == "-") {
      val = builder.CreateFSub(operandValues[0], operandValues[1]);
    } else if (op == "*") {
      val = builder.CreateFMul(operandValues[0], operandValues[1]);
    } else if (op == "/") {
      val = builder.CreateFDiv(operandValues[0], operandValues[1]);
    } else if (op == "sin") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::sin, operandValues[0]);
    } else if (op == "cos") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::cos, operandValues[0]);
#if LLVM_VERSION_MAJOR >= 16 // TODO: Double check version
    } else if (op == "tan") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::tan, operandValues[0]);
#endif
    } else if (op == "exp") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::exp, operandValues[0]);
    } else if (op == "log") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::log, operandValues[0]);
    } else if (op == "sqrt") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::sqrt, operandValues[0]);
    } else if (op == "pow") {
      val = builder.CreateBinaryIntrinsic(Intrinsic::pow, operandValues[0],
                                          operandValues[1]);
    } else if (op == "fma") {
      val = builder.CreateIntrinsic(
          Intrinsic::fma, {operandValues[0]->getType()},
          {operandValues[0], operandValues[1], operandValues[2]});
    } else if (op == "fabs") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::fabs, operandValues[0]);
    } else {
      assert(0 && "FPNode.getValue: Unknown operator");
    }

    return val;
  }

  virtual bool isExpression() const { return true; }
};

// Represents a true LLVM Value
class FPLLValue : public FPNode {
  Value *value;

public:
  FPLLValue(Value *value) : FPNode("__arg"), value(value) {}

  virtual std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    assert(hasSymbol() && "FPLLValue has no symbol!");
    return symbol;
  }

  virtual Value *getValue(Instruction *insertBefore,
                          IRBuilder<> &builder) override {
    return value;
  }

  bool isExpression() const override { return false; }
};

class FPConst : public FPNode {
  std::string value;

public:
  FPConst(std::string value) : FPNode("__const"), value(value) {}

  virtual std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    return value;
  }

  virtual Value *getValue(Instruction *insertBefore,
                          IRBuilder<> &builder) override {
    llvm::errs() << "Returning constant: " << value << "\n";
    double constantValue = std::stod(value);
    // TODO eventually have this be typed
    return ConstantFP::get(builder.getDoubleTy(), constantValue);
  }

  bool isExpression() const override { return false; }
};

FPNode *
parseHerbieExpr(const std::string &expr,
                std::unordered_map<Value *, FPNode *> &valueToNodeMap,
                std::unordered_map<std::string, Value *> &symbolToValueMap) {
  llvm::errs() << "Parsing: " << expr << "\n";
  auto trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  // Arguments
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    // llvm::errs() << "Base case: " << trimmedExpr << "\n";
    return valueToNodeMap[symbolToValueMap[trimmedExpr]];
  }

  // Constants
  std::regex constantPattern("^#s\\(literal\\s+([\\d\\.]+)\\s+\\w+\\)$");
  std::smatch matches;
  if (std::regex_match(trimmedExpr, matches, constantPattern)) {
    llvm::errs() << "Found __const " << matches[1].str() << "\n";
    return new FPConst(matches[1].str());
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
      node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                       valueToNodeMap, symbolToValueMap));
      start = curr + 1;
    }
  }
  if (start < curr) {
    node->addOperand(parseHerbieExpr(trimmedExpr.substr(start, curr - start),
                                     valueToNodeMap, symbolToValueMap));
  }

  return node;
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
  llvm::StringRef Args[] = {Program,     "improve", "--seed",
                            "239778888", tmpin,     tmpout};
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
  case Instruction::Call: {
    const CallInst *CI = dyn_cast<CallInst>(&I);
    assert(CI && CI->getCalledFunction() &&
           "getHerbieOperator: Call without a function");

    std::string funcName = CI->getCalledFunction()->getName().str();
    std::regex regex("llvm\\.(\\w+)\\.?.*");
    std::smatch matches;
    if (std::regex_search(funcName, matches, regex) && matches.size() > 1) {
      return matches[1].str();
    }
    assert(0 && "getHerbieOperator: Unknown callee");
  }
  default:
    assert(0 && "getHerbieOperator: Unknown operator");
  }
}

bool herbiable(const Value &Val) {
  const Instruction *I = dyn_cast<Instruction>(&Val);
  if (!I)
    return false;

  switch (I->getOpcode()) {
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
    return I->getType()->isFloatTy() || I->getType()->isDoubleTy();
  case Instruction::Call: {
    const CallInst *CI = dyn_cast<CallInst>(I);
    if (CI && CI->getCalledFunction() &&
        (CI->getType()->isFloatTy() || CI->getType()->isDoubleTy())) {
      StringRef funcName = CI->getCalledFunction()->getName();
      return funcName.startswith("llvm.sin") ||
             funcName.startswith("llvm.cos") ||
             funcName.startswith("llvm.tan") ||
             funcName.startswith("llvm.exp") ||
             funcName.startswith("llvm.log") ||
             funcName.startswith("llvm.sqrt") ||
             funcName.startswith("llvm.pow") ||
             funcName.startswith("llvm.fma") ||
             funcName.startswith("llvm.fabs");
    }
    return false;
  }
  default:
    return false;
  }
}

struct HerbieComponents {
  SetVector<Value *> inputs;
  SetVector<Value *> outputs;
  SetVector<Instruction *> operations;

  HerbieComponents(SetVector<Value *> inputs, SetVector<Value *> outputs,
                   SetVector<Instruction *> operations)
      : inputs(std::move(inputs)), outputs(std::move(outputs)),
        operations(std::move(operations)) {}
};

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F) {
  bool changed = false;

  int symbolCounter = 0;
  auto getNextSymbol = [&symbolCounter]() -> std::string {
    return "v" + std::to_string(symbolCounter++);
  };

  // Extract change:

  // E1) create map<Value, FPNode> for all instructions I, map[I] = FPLLValue(I)
  // E2) for all instructions, if herbiable(I), map[I] = FPNode(operation(I),
  // map[operands(I)])
  // E3) floodfill for all starting locations I to find all distinct graphs /
  // outputs.

  /*
  B1:
    x = sin(arg)

  B2:
    y = 1 - x * x


  -> result y = cos(arg)^2

B1:
  nothing

B2:
  costmp = cos(arg)
  y = costmp * costmp

  */

  std::unordered_map<Value *, FPNode *> valueToNodeMap;
  std::unordered_map<std::string, Value *> symbolToValueMap;

  for (auto &arg : F.args()) {
    valueToNodeMap[&arg] = new FPLLValue(&arg);
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      valueToNodeMap[&I] = new FPLLValue(&I);
    }
  }

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (herbiable(I)) {
        llvm::errs() << "Herbie Operator: " << getHerbieOperator(I) << "\n";
        auto node = new FPNode(getHerbieOperator(I));

        auto operands =
            isa<CallInst>(I) ? cast<CallInst>(I).args() : I.operands();
        for (auto &operand : operands) {
          if (!valueToNodeMap.count(operand)) {
            if (auto C = dyn_cast<ConstantFP>(operand)) {
              llvm::SmallVector<char, 10> value;
              C->getValueAPF().toString(value);
              std::string valueStr(value.begin(), value.end());
              valueToNodeMap[operand] = new FPConst(valueStr);
              llvm::errs() << "Registered FPNode for constant: " << valueStr
                           << "\n";
            } else if (auto GV = dyn_cast<GlobalVariable>(operand)) {
              valueToNodeMap[operand] = new FPLLValue(GV);
              llvm::errs() << "Registered FPNode for global variable: " << *GV
                           << "\n";
            } else {
              assert(0 && "Unknown operand");
            }
          }
          node->addOperand(valueToNodeMap[operand]);
        }
        valueToNodeMap[&I] = node;
      }
    }
  }

  for (auto &[value, node] : valueToNodeMap) {
    llvm::errs() << "Value: " << *value
                 << " isExpression: " << valueToNodeMap[value]->isExpression()
                 << "\n";
  }

  SmallSet<Value *, 1> component_seen;
  SmallVector<HerbieComponents, 1> connected_components;
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Not a herbiable instruction, doesn't make sense to create graph node
      // out of.
      if (!herbiable(I)) {
        llvm::errs() << "Skipping non-herbiable instruction: " << I << "\n";
        continue;
      }

      // Instruction is already in a set
      if (component_seen.contains(&I)) {
        llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      llvm::errs() << "Starting floodfill from: " << I << "\n";

      SmallVector<Value *, 1> todo;
      SetVector<Value *> input_seen;
      SetVector<Value *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        auto node = valueToNodeMap[cur];
        assert(node && "Node not found in valueToNodeMap");

        // We now can assume that this is a herbiable expression
        // Since we can only herbify instructions, let's assert that
        assert(isa<Instruction>(cur));
        auto I2 = cast<Instruction>(cur);

        // Don't repeat any instructions we've already seen (to avoid loops for
        // phi nodes)
        if (operation_seen.contains(I2)) {
          llvm::errs() << "Skipping already seen instruction: " << *I2 << "\n";
          continue;
        }

        // Assume that a herbiable expression can only be in one connected
        // component.
        assert(!component_seen.contains(cur));

        llvm::errs() << "Insert to operation_seen and component_seen: " << *I2
                     << "\n";
        operation_seen.insert(I2);
        component_seen.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (auto &operand : operands) {
          if (!herbiable(*operand)) {
            llvm::errs() << "Non-herbiable input found: " << *operand << "\n";
            input_seen.insert(operand);
          } else {
            llvm::errs() << "Adding operand to todo list: " << *operand << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!herbiable(*I3)) {
              llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      llvm::errs() << "Finished floodfill\n\n";

      // Don't bother with graphs without any herbiable operations
      if (!operation_seen.empty()) {
        llvm::errs() << "Found connected component with "
                     << operation_seen.size() << " operations and "
                     << input_seen.size() << " inputs and "
                     << output_seen.size() << " outputs\n";

        llvm::errs() << "Inputs:\n";
        for (auto &input : input_seen) {
          llvm::errs() << *input << "\n";
        }

        llvm::errs() << "Outputs:\n";
        for (auto &output : output_seen) {
          llvm::errs() << *output << "\n";
        }

        llvm::errs() << "Operations:\n";
        for (auto &operation : operation_seen) {
          llvm::errs() << *operation << "\n";
        }

        connected_components.emplace_back(std::move(input_seen),
                                          std::move(output_seen),
                                          std::move(operation_seen));
      }
    }
  }

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting llvm instructions into herbie string (FPNode ....)
  if (connected_components.empty()) {
    llvm::errs() << "No herbiable connected components found\n";
    return false;
  }

  for (auto &component : connected_components) {
    std::string argumentsStr = "(";
    for (const auto &input : component.inputs) {
      auto node = valueToNodeMap[input];
      if (node->op == "__const") {
        // Constants don't need a symbol
        continue;
      }
      argumentsStr +=
          node->hasSymbol() ? node->symbol : (node->symbol = getNextSymbol());
      symbolToValueMap[node->symbol] = input;
      llvm::errs() << "assigning symbol: " << node->symbol << " to " << *input
                   << "\n";
      argumentsStr += " ";
    }
    argumentsStr.pop_back();
    argumentsStr += ")";

    for (const auto &output : component.outputs) {
      std::string herbieExpr =
          "(FPCore " + argumentsStr + " " +
          valueToNodeMap[output]->toFullExpression(valueToNodeMap) + ")";
      llvm::errs() << "Herbie input:\n" << herbieExpr << "\n";

      // 3) run fancy opts
      if (!improveViaHerbie(herbieExpr)) {
        llvm::errs() << "Failed to optimize " << herbieExpr
                     << " using Herbie!\n";
        continue;
      } else {
        llvm::errs() << "Optimized: " << herbieExpr << "\n";
      }

      // 4) parse the output string solution from herbieland
      // 5) convert into a solution in llvm vals/instructions
      llvm::errs() << "Parsing Herbie Expr: " << herbieExpr << "\n";
      FPNode *parsedNode =
          parseHerbieExpr(herbieExpr, valueToNodeMap, symbolToValueMap);
      llvm::errs() << "Parsed Herbie Expr: "
                   << parsedNode->toFullExpression(valueToNodeMap) << "\n";

      Instruction *insertBefore = component.operations.back();
      IRBuilder<> builder(insertBefore);
      // TODO ponder fast math
      builder.setFastMathFlags(getFast());
      builder.SetInsertPoint(insertBefore);

      // Convert the parsed expression to LLVM values/instructions
      Value *newRootValue = parsedNode->getValue(insertBefore, builder);
      assert(newRootValue && "Failed to get value from parsed node");
      llvm::errs() << "Replacing: " << *output << " with " << *newRootValue
                   << "\n";
      output->replaceAllUsesWith(newRootValue);

      // TODO: better cleanup
      for (auto I = component.operations.rbegin();
           I != component.operations.rend(); ++I) {
        if ((*I)->use_empty()) {
          llvm::errs() << "Removing: " << **I << "\n";
          (*I)->eraseFromParent();
        }
      }

      changed = true;
    }
  }

  for (auto &[_, node] : valueToNodeMap) {
    delete node;
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
