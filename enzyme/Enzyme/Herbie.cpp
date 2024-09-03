#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

#include <llvm/Demangle/Demangle.h>

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Support/Host.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/JSON.h>

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <utility>

#include "Herbie.h"
#include "Utils.h"

using namespace llvm;
#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "fp-opt"

extern "C" {
cl::opt<bool> EnzymeEnableFPOpt("enzyme-enable-fpopt", cl::init(false),
                                cl::Hidden, cl::desc("Run the FPOpt pass"));
static cl::opt<bool>
    EnzymePrintFPOpt("enzyme-print-fpopt", cl::init(false), cl::Hidden,
                     cl::desc("Enable Enzyme to print FPOpt info"));
static cl::opt<bool>
    EnzymePrintHerbie("enzyme-print-herbie", cl::init(false), cl::Hidden,
                      cl::desc("Enable Enzyme to print Herbie expressions"));
static cl::opt<std::string>
    FPOptLogPath("fpopt-log-path", cl::init(""), cl::Hidden,
                 cl::desc("Which log to use in the FPOpt pass"));
static cl::opt<std::string> FPOptTargetFuncRegex(
    "fpopt-target-func-regex", cl::init(".*"), cl::Hidden,
    cl::desc("Regex pattern to match target functions in the FPOpt pass"));
static cl::opt<bool> HerbieDisableNumerics(
    "herbie-disable-numerics", cl::init(false), cl::Hidden,
    cl::desc("Disable Herbie rewrite rules that produce numerical shorthands "
             "expm1, log1p, fma, and hypot"));
static cl::opt<bool>
    HerbieDisableTaylor("herbie-disable-taylor", cl::init(false), cl::Hidden,
                        cl::desc("Disable Herbie's series expansion"));
static cl::opt<bool> HerbieDisableSetupSimplify(
    "herbie-disable-setup-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from pre-simplifying expressions"));
static cl::opt<bool> HerbieDisableGenSimplify(
    "herbie-disable-gen-simplify", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from simplifying expressions "
             "during the main improvement loop"));
static cl::opt<bool> HerbieDisableRegime(
    "herbie-disable-regime", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching between expressions candidates"));
static cl::opt<bool> HerbieDisableBranchExpr(
    "herbie-disable-branch-expr", cl::init(false), cl::Hidden,
    cl::desc("Stop Herbie from branching on expressions"));
static cl::opt<bool> HerbieDisableAvgError(
    "herbie-disable-avg-error", cl::init(false), cl::Hidden,
    cl::desc("Make Herbie choose the candidates with the least maximum error"));
static cl::opt<bool> FPOptEnableSolver(
    "fpopt-enable-solver", cl::init(false), cl::Hidden,
    cl::desc("Use the solver to select desirable rewrite candidates; when "
             "disabled, apply all Herbie's first choices"));
static cl::opt<std::string> FPOptSolverType("fpopt-solver-type", cl::init("dp"),
                                            cl::Hidden,
                                            cl::desc("Which solver to use"));
static cl::opt<int64_t> FPOptComputationCostBudget(
    "fpopt-comp-cost-budget", cl::init(100000000000L), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
static cl::opt<int> FPOptMaxFPCCDepth(
    "fpopt-max-fpcc-depth", cl::init(10), cl::Hidden,
    cl::desc("The maximum depth of a floating-point connected component"));
}

class FPNode {
public:
  std::string op;
  std::string dtype;
  std::string symbol;
  SmallVector<FPNode *, 2> operands;
  double grad;
  unsigned executions;

  FPNode(const std::string &op) = delete;
  explicit FPNode(const std::string &op, const std::string &dtype)
      : op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  void addOperand(FPNode *operand) { operands.push_back(operand); }

  bool hasSymbol() const { return !symbol.empty(); }

  virtual std::string
  toFullExpression(std::unordered_map<Value *, FPNode *> &valueToNodeMap) {
    std::string msg = "Unexpected invocation of `toFullExpression` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }

  virtual void markAsInput() {
    std::string msg = "Unexpected invocation of `markAsInput` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }

  virtual void updateBounds(double lower, double upper) {
    std::string msg = "Unexpected invocation of `updateBounds` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }
  virtual double getLowerBound() const {
    std::string msg = "Unexpected invocation of `getLowerBound` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }
  virtual double getUpperBound() const {
    std::string msg = "Unexpected invocation of `getUpperBound` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }

  virtual Value *getValue(IRBuilder<> &builder) {
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Generating new instruction for op: " << op << "\n";
    Module *M = builder.GetInsertBlock()->getModule();

    if (op == "if") {
      Value *condValue = operands[0]->getValue(builder);
      auto IP = builder.GetInsertPoint();

      Instruction *Then, *Else;
      SplitBlockAndInsertIfThenElse(condValue, &*IP, &Then, &Else);

      Then->getParent()->setName("herbie.then");
      builder.SetInsertPoint(Then);
      Value *ThenVal = operands[1]->getValue(builder);
      if (Instruction *I = dyn_cast<Instruction>(ThenVal)) {
        I->setName("herbie.then_val");
      }

      Else->getParent()->setName("herbie.else");
      builder.SetInsertPoint(Else);
      Value *ElseVal = operands[2]->getValue(builder);
      if (Instruction *I = dyn_cast<Instruction>(ElseVal)) {
        I->setName("herbie.else_val");
      }

      builder.SetInsertPoint(&*IP);
      auto Phi = builder.CreatePHI(ThenVal->getType(), 2);
      Phi->addIncoming(ThenVal, Then->getParent());
      Phi->addIncoming(ElseVal, Else->getParent());
      Phi->setName("herbie.merge");

      return Phi;
    }

    SmallVector<Value *, 2> operandValues;
    for (auto *operand : operands) {
      operandValues.push_back(operand->getValue(builder));
    }

    Value *val = nullptr;

    if (op == "neg") {
      val = builder.CreateFNeg(operandValues[0], "herbie.neg");
    } else if (op == "+") {
      val =
          builder.CreateFAdd(operandValues[0], operandValues[1], "herbie.add");
    } else if (op == "-") {
      val =
          builder.CreateFSub(operandValues[0], operandValues[1], "herbie.sub");
    } else if (op == "*") {
      val =
          builder.CreateFMul(operandValues[0], operandValues[1], "herbie.mul");
    } else if (op == "/") {
      val =
          builder.CreateFDiv(operandValues[0], operandValues[1], "herbie.div");
    } else if (op == "sin") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::sin, operandValues[0],
                                         nullptr, "herbie.sin");
    } else if (op == "cos") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::cos, operandValues[0],
                                         nullptr, "herbie.cos");
    } else if (op == "tan") {
#if LLVM_VERSION_MAJOR >= 16 // TODO: Double check version
      val = builder.CreateUnaryIntrinsic(Intrinsic::tan, operandValues[0],
                                         "herbie.tan");
#else
      // Using std::tan(f) for lower versions of LLVM.
      auto *Ty = operandValues[0]->getType();
      std::string funcName = Ty->isDoubleTy() ? "tan" : "tanf";
      llvm::Function *tanFunc = M->getFunction(funcName);
      if (!tanFunc) {
        auto *funcTy = FunctionType::get(Ty, {Ty}, false);
        tanFunc =
            Function::Create(funcTy, Function::ExternalLinkage, funcName, M);
      }
      if (tanFunc) {
        val = builder.CreateCall(tanFunc, {operandValues[0]}, "herbie.tan");
      } else {
        std::string msg =
            "Failed to find or declare " + funcName + " in the module.";
        llvm_unreachable(msg.c_str());
      }

#endif
    } else if (op == "exp") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::exp, operandValues[0],
                                         nullptr, "herbie.exp");
    } else if (op == "expm1") {
      auto *Ty = operandValues[0]->getType();
      std::string funcName = Ty->isDoubleTy() ? "expm1" : "expm1f";
      llvm::Function *expm1Func = M->getFunction(funcName);
      if (!expm1Func) {
        auto *funcTy = FunctionType::get(Ty, {Ty}, false);
        expm1Func =
            Function::Create(funcTy, Function::ExternalLinkage, funcName, M);
      }
      if (expm1Func) {
        val = builder.CreateCall(expm1Func, {operandValues[0]}, "herbie.expm1");
      } else {
        std::string msg = "Failed to find or declare " + funcName +
                          " in the module. Consider disabling Herbie rules for "
                          "numerics (-herbie-disable-numerics).";
        llvm_unreachable(msg.c_str());
      }
    } else if (op == "log") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::log, operandValues[0],
                                         nullptr, "herbie.log");
    } else if (op == "log1p") {
      auto *Ty = operandValues[0]->getType();
      std::string funcName = Ty->isDoubleTy() ? "log1p" : "log1pf";
      llvm::Function *log1pFunc = M->getFunction(funcName);
      if (!log1pFunc) {
        auto *funcTy = FunctionType::get(Ty, {Ty}, false);
        log1pFunc =
            Function::Create(funcTy, Function::ExternalLinkage, funcName, M);
      }
      if (log1pFunc) {
        val = builder.CreateCall(log1pFunc, {operandValues[0]}, "herbie.log1p");
      } else {
        std::string msg =
            "Failed to find or declare log1p in the module. Consider disabling "
            "Herbie rules for numerics (-herbie-disable-numerics).";
        llvm_unreachable(msg.c_str());
      }
    } else if (op == "sqrt") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::sqrt, operandValues[0],
                                         nullptr, "herbie.sqrt");
    } else if (op == "cbrt") {
      auto *Ty = operandValues[0]->getType();
      std::string funcName = Ty->isDoubleTy() ? "cbrt" : "cbrtf";
      llvm::Function *cbrtFunc = M->getFunction(funcName);
      if (!cbrtFunc) {
        auto *funcTy = FunctionType::get(Ty, {Ty}, false);
        cbrtFunc =
            Function::Create(funcTy, Function::ExternalLinkage, funcName, M);
      }
      if (cbrtFunc) {
        val = builder.CreateCall(cbrtFunc, {operandValues[0]}, "herbie.cbrt");
      } else {
        std::string msg =
            "Failed to find or declare " + funcName +
            " in the module. Consider disabling "
            "Herbie rules for numerics (-herbie-disable-numerics).";
        llvm_unreachable(msg.c_str());
      }
    } else if (op == "pow") {
      val = builder.CreateBinaryIntrinsic(Intrinsic::pow, operandValues[0],
                                          operandValues[1], nullptr,
                                          "herbie.pow");
    } else if (op == "fma") {
      val = builder.CreateIntrinsic(
          Intrinsic::fmuladd, {operandValues[0]->getType()},
          {operandValues[0], operandValues[1], operandValues[2]}, nullptr,
          "herbie.fma");
    } else if (op == "fabs") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::fabs, operandValues[0],
                                         nullptr, "herbie.fabs");
    } else if (op == "hypot") {
      auto *Ty = operandValues[0]->getType();
      std::string funcName = Ty->isDoubleTy() ? "hypot" : "hypotf";
      llvm::Function *hypotFunc = M->getFunction(funcName);
      if (!hypotFunc) {
        auto *funcTy = FunctionType::get(Ty, {Ty, Ty}, false);
        hypotFunc =
            Function::Create(funcTy, Function::ExternalLinkage, funcName, M);
      }
      if (hypotFunc) {
        val = builder.CreateCall(
            hypotFunc, {operandValues[0], operandValues[1]}, "herbie.hypot");
      } else {
        std::string msg =
            "Failed to find or declare " + funcName +
            " in the module. Consider disabling "
            "Herbie rules for numerics (-herbie-disable-numerics).";
        llvm_unreachable(msg.c_str());
      }
    } else if (op == "==") {
      val = builder.CreateFCmpOEQ(operandValues[0], operandValues[1],
                                  "herbie.if.eq");
    } else if (op == "!=") {
      val = builder.CreateFCmpONE(operandValues[0], operandValues[1],
                                  "herbie.if.ne");
    } else if (op == "<") {
      val = builder.CreateFCmpOLT(operandValues[0], operandValues[1],
                                  "herbie.if.lt");
    } else if (op == ">") {
      val = builder.CreateFCmpOGT(operandValues[0], operandValues[1],
                                  "herbie.if.gt");
    } else if (op == "<=") {
      val = builder.CreateFCmpOLE(operandValues[0], operandValues[1],
                                  "herbie.if.le");
    } else if (op == ">=") {
      val = builder.CreateFCmpOGE(operandValues[0], operandValues[1],
                                  "herbie.if.ge");
    } else if (op == "and") {
      val = builder.CreateAnd(operandValues[0], operandValues[1],
                              "herbie.if.and");
    } else if (op == "or") {
      val =
          builder.CreateOr(operandValues[0], operandValues[1], "herbie.if.or");
    } else if (op == "not") {
      val = builder.CreateNot(operandValues[0], "herbie.if.not");
    } else if (op == "TRUE") {
      val = ConstantInt::getTrue(builder.getContext());
    } else if (op == "FALSE") {
      val = ConstantInt::getFalse(builder.getContext());
    } else {
      std::string msg = "FPNode getValue: Unexpected operator " + op;
      llvm_unreachable(msg.c_str());
    }

    return val;
  }

  virtual bool isExpression() const { return true; }
};

// Represents a true LLVM Value
class FPLLValue : public FPNode {
  Value *value;
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
  bool input = false; // Whether `llvm::Value` is an input of an FPCC

public:
  explicit FPLLValue(Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(op, dtype), value(value) {}

  std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    if (input) {
      assert(hasSymbol() && "FPLLValue has no symbol!");
      return symbol;
    } else {
      assert(!operands.empty() && "FPNode has no operands!");
      std::string expr = "(" + op;
      for (auto operand : operands) {
        expr += " " + operand->toFullExpression(valueToNodeMap);
      }
      expr += ")";
      return expr;
    }
  }

  void markAsInput() override { input = true; }

  void updateBounds(double lower, double upper) override {
    lb = std::min(lb, lower);
    ub = std::max(ub, upper);
    if (EnzymePrintFPOpt)
      llvm::errs() << "Updated bounds for " << *value << ": [" << lb << ", "
                   << ub << "]\n";
  }

  double getLowerBound() const override { return lb; }
  double getUpperBound() const override { return ub; }

  Value *getValue(IRBuilder<> &builder) override { return value; }

  bool isExpression() const override { return false; }
};

double stringToDouble(const std::string &str) {
  char *end;
  errno = 0;
  double result = std::strtod(str.c_str(), &end);

  if (errno == ERANGE) {
    if (result == HUGE_VAL) {
      result = std::numeric_limits<double>::infinity();
    } else if (result == -HUGE_VAL) {
      result = -std::numeric_limits<double>::infinity();
    }
  }

  return result; // Denormalized values are fine
}

class FPConst : public FPNode {
  std::string strValue;

public:
  explicit FPConst(const std::string &strValue, const std::string &dtype)
      : FPNode("__const", dtype), strValue(strValue) {}

  std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    return strValue;
  }

  void markAsInput() override { return; }

  void updateBounds(double lower, double upper) override { return; }

  double getLowerBound() const override {
    if (strValue == "+inf.0") {
      return std::numeric_limits<double>::infinity();
    } else if (strValue == "-inf.0") {
      return -std::numeric_limits<double>::infinity();
    }

    double constantValue;
    size_t div = strValue.find('/');

    if (div != std::string::npos) {
      std::string numerator = strValue.substr(0, div);
      std::string denominator = strValue.substr(div + 1);
      double num = stringToDouble(numerator);
      double denom = stringToDouble(denominator);

      constantValue = num / denom;
    } else {
      constantValue = stringToDouble(strValue);
    }

    return constantValue;
  }

  double getUpperBound() const override { return getLowerBound(); }

  virtual Value *getValue(IRBuilder<> &builder) override {
    Type *Ty;
    if (dtype == "f64") {
      Ty = builder.getDoubleTy();
    } else if (dtype == "f32") {
      Ty = builder.getFloatTy();
    } else {
      std::string msg = "FPConst getValue: Unexpected dtype: " + dtype;
      llvm_unreachable(msg.c_str());
    }
    if (strValue == "+inf.0") {
      return ConstantFP::getInfinity(Ty, false);
    } else if (strValue == "-inf.0") {
      return ConstantFP::getInfinity(Ty, true);
    }

    double constantValue;
    size_t div = strValue.find('/');

    if (div != std::string::npos) {
      std::string numerator = strValue.substr(0, div);
      std::string denominator = strValue.substr(div + 1);
      double num = stringToDouble(numerator);
      double denom = stringToDouble(denominator);

      constantValue = num / denom;
    } else {
      constantValue = stringToDouble(strValue);
    }

    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Returning " << strValue << " as " << dtype
    //                << " constant: " << constantValue << "\n";
    return ConstantFP::get(Ty, constantValue);
  }

  bool isExpression() const override { return false; }
};

FPNode *
parseHerbieExpr(const std::string &expr,
                std::unordered_map<Value *, FPNode *> &valueToNodeMap,
                std::unordered_map<std::string, Value *> &symbolToValueMap) {
  // if (EnzymePrintFPOpt)
  //   llvm::errs() << "Parsing: " << expr << "\n";
  std::string trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  // Arguments
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    return valueToNodeMap[symbolToValueMap[trimmedExpr]];
  }

  // Constants
  std::regex constantPattern(
      "^#s\\(literal\\s+([-+]?\\d+(/\\d+)?|[-+]?inf\\.0)\\s+(\\w+)\\)$");

  std::smatch matches;
  if (std::regex_match(trimmedExpr, matches, constantPattern)) {
    std::string value = matches[1].str();
    std::string dtype = matches[3].str();
    if (dtype == "binary64") {
      dtype = "f64";
    } else if (dtype == "binary32") {
      dtype = "f32";
    } else {
      std::string msg =
          "Herbie expr parser: Unexpected constant dtype: " + dtype;
      llvm_unreachable(msg.c_str());
    }
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Herbie expr parser: Found __const " << value
    //                << " with dtype " << dtype << "\n";
    return new FPConst(value, dtype);
  }

  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')') {
    llvm::errs() << "Unexpected subexpression: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

  trimmedExpr = trimmedExpr.substr(1, trimmedExpr.size() - 2);

  // Get the operator
  auto endOp = trimmedExpr.find(' ');
  std::string fullOp = trimmedExpr.substr(0, endOp);

  size_t pos = fullOp.find('.');

  std::string dtype;
  std::string op;
  if (pos != std::string::npos) {
    op = fullOp.substr(0, pos);
    dtype = fullOp.substr(pos + 1);
    assert(dtype == "f64" || dtype == "f32");
    // llvm::errs() << "Herbie expr parser: Found operator " << op
    //              << " with dtype " << dtype << "\n";
  } else {
    op = fullOp;
    // llvm::errs() << "Herbie expr parser: Found operator " << op << "\n";
  }

  auto node = new FPNode(op, dtype);

  int depth = 0;
  auto start = trimmedExpr.find_first_not_of(" ", endOp);
  std::string::size_type curr;
  for (curr = start; curr < trimmedExpr.size(); ++curr) {
    if (trimmedExpr[curr] == '(')
      depth++;
    if (trimmedExpr[curr] == ')')
      depth--;
    if (depth == 0 && trimmedExpr[curr] == ' ') {
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

void getUniqueArgs(const std::string &expr, SmallSet<std::string, 8> &args) {
  // TODO: Update it if we use let expr in the future
  std::regex argPattern("v\\d+");

  std::sregex_iterator begin(expr.begin(), expr.end(), argPattern);
  std::sregex_iterator end;

  while (begin != end) {
    args.insert(begin->str());
    ++begin;
  }
}

// Sum up the cost of `output` and its FP operands recursively up to `inputs`
// (exclusive).
InstructionCost getTTICost(Value *output, const SetVector<Value *> &inputs,
                           const TargetTransformInfo &TTI) {
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  InstructionCost cost = 0;

  todo.push_back(output);
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      // TODO: unfair to ignore branches when calculating cost
      auto instCost = TTI.getInstructionCost(
          I, TargetTransformInfo::TCK_SizeAndLatency); // TODO: What metric?
      // auto instCost =
      //     TTI.getInstructionCost(I,
      //     TargetTransformInfo::TCK_RecipThroughput);

      // if (EnzymePrintFPOpt)
      //   llvm::errs() << "Cost of " << *I << " is: " << instCost << "\n";

      // Only add the cost of the instruction if it is not an input
      cost += instCost;

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  return cost;
}

InstructionCost
getTTICost(const std::string &expr, Module *M, const TargetTransformInfo &TTI,
           std::unordered_map<Value *, FPNode *> &valueToNodeMap,
           std::unordered_map<std::string, Value *> &symbolToValueMap) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  for (const auto &argStr : argStrSet) {
    args.insert(symbolToValueMap[argStr]);
  }

  FPNode *parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);

  // Materialize the expression in a temporary function
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M->getContext()), false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "getTTICost_temp", M);
  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);
  Instruction *ReturnInst = ReturnInst::Create(M->getContext(), entry);

  IRBuilder<> builder(ReturnInst);

  builder.setFastMathFlags(getFast());
  Value *newOutput = parsedNode->getValue(builder);

  // tempFunction->print(llvm::errs());

  InstructionCost cost = getTTICost(newOutput, args, TTI);

  tempFunction->eraseFromParent();
  return cost;
}

struct RewriteCandidate {
  // Only one rewrite candidate per output `llvm::Value` can be applied
  InstructionCost TTICost;
  double herbieCost; // Unused for now
  double accuracy;
  std::string expr;

  RewriteCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), accuracy(accuracy), expr(expression) {}
};

// Floating-Point Connected Component
struct FPCC {
  SetVector<Value *> inputs;
  SetVector<Instruction *> outputs;
  SetVector<Instruction *> operations;
  size_t outputs_rewritten = 0;

  FPCC() = default;
  explicit FPCC(SetVector<Value *> inputs, SetVector<Instruction *> outputs,
                SetVector<Instruction *> operations)
      : inputs(std::move(inputs)), outputs(std::move(outputs)),
        operations(std::move(operations)) {}
};

void splitFPCC(FPCC &CC, SmallVector<FPCC, 1> &newCCs) {
  std::unordered_map<Instruction *, int> shortestDistances;

  for (auto &op : CC.operations) {
    shortestDistances[op] = std::numeric_limits<int>::max();
  }

  // find the shortest distance from inputs to each operation
  for (auto &input : CC.inputs) {
    SmallVector<std::pair<Instruction *, int>, 8> todo;
    for (auto user : input->users()) {
      if (auto *I = dyn_cast<Instruction>(user); I && CC.operations.count(I)) {
        todo.emplace_back(I, 1);
      }
    }

    while (!todo.empty()) {
      auto [cur, dist] = todo.pop_back_val();
      if (dist < shortestDistances[cur]) {
        shortestDistances[cur] = dist;
        for (auto user : cur->users()) {
          if (auto *I = dyn_cast<Instruction>(user);
              I && CC.operations.count(I)) {
            todo.emplace_back(I, dist + 1);
          }
        }
      }
    }
  }

  llvm::errs() << "Shortest distances:\n";
  for (auto &[op, dist] : shortestDistances) {
    llvm::errs() << *op << ": " << dist << "\n";
  }

  int maxDepth =
      std::max_element(shortestDistances.begin(), shortestDistances.end(),
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.second < rhs.second;
                       })
          ->second;

  if (maxDepth <= FPOptMaxFPCCDepth) {
    newCCs.push_back(CC);
    return;
  }

  newCCs.resize(maxDepth / FPOptMaxFPCCDepth + 1);

  // Split `operations` based on the shortest distance
  for (const auto &[op, dist] : shortestDistances) {
    newCCs[dist / FPOptMaxFPCCDepth].operations.insert(op);
  }

  // Reconstruct `inputs` and `outputs` for new components
  for (auto &newCC : newCCs) {
    for (auto &op : newCC.operations) {
      auto operands =
          isa<CallInst>(op) ? cast<CallInst>(op)->args() : op->operands();
      for (auto &operand : operands) {
        if (newCC.inputs.count(operand)) {
          continue;
        }

        // Original non-herbiable operands or herbiable intermediate operations
        if (CC.inputs.count(operand) ||
            !newCC.operations.count(cast<Instruction>(operand))) {
          newCC.inputs.insert(operand);
        }
      }

      for (auto user : op->users()) {
        if (auto *I = dyn_cast<Instruction>(user);
            I && !newCC.operations.count(I)) {
          newCC.outputs.insert(op);
        }
      }
    }
  }

  if (EnzymePrintFPOpt) {
    llvm::errs() << "Splitting the FPCC into " << newCCs.size()
                 << " components\n";
  }
}

class ApplicableOutput {
public:
  FPCC &component;
  Value *oldOutput;
  std::string expr;
  double grad;
  unsigned executions;
  InstructionCost initialTTICost;    // Requires manual initialization
  InstructionCost initialHerbieCost; // Requires manual initialization
  double initialAccuracy;            // Requires manual initialization
  SmallVector<RewriteCandidate> candidates;

  explicit ApplicableOutput(FPCC &component, Value *oldOutput, std::string expr,
                            double grad, unsigned executions,
                            const TargetTransformInfo &TTI)
      : component(component), oldOutput(oldOutput), expr(expr), grad(grad),
        executions(executions) {
    initialTTICost = getTTICost(oldOutput, component.inputs, TTI);
  }

  void apply(size_t candidateIndex,
             std::unordered_map<Value *, FPNode *> &valueToNodeMap,
             std::unordered_map<std::string, Value *> &symbolToValueMap) {
    // 4) parse the output string solution from herbieland
    // 5) convert into a solution in llvm vals/instructions

    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Parsing Herbie output: " << herbieOutput << "\n";
    FPNode *parsedNode = parseHerbieExpr(candidates[candidateIndex].expr,
                                         valueToNodeMap, symbolToValueMap);
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Parsed Herbie output: "
    //                << parsedNode->toFullExpression(valueToNodeMap) << "\n";

    Instruction *insertBefore = dyn_cast<Instruction>(oldOutput);
    IRBuilder<> builder(insertBefore);
    // TODO ponder fast math
    builder.setFastMathFlags(getFast());

    Value *newOutput = parsedNode->getValue(builder);
    assert(newOutput && "Failed to get value from parsed node");

    if (EnzymePrintFPOpt)
      llvm::errs() << "Applying Herbie rewrite (#" << candidateIndex
                   << "): " << expr << "\n --> "
                   << candidates[candidateIndex].expr << "\n";

    oldOutput->replaceAllUsesWith(newOutput);
    symbolToValueMap[valueToNodeMap[oldOutput]->symbol] = newOutput;
    valueToNodeMap[newOutput] =
        new FPLLValue(newOutput, "__no", valueToNodeMap[oldOutput]->dtype);
    component.outputs_rewritten++;
  }

  // Lower is better
  InstructionCost getComputationCost(size_t candidateIndex) {
    // TODO: consider erasure of the old output
    return candidates[candidateIndex].TTICost * executions;
  }

  // Lower is better
  double getAccuracyCost(size_t candidateIndex) {
    return (initialAccuracy - candidates[candidateIndex].accuracy) *
           std::fabs(grad);
  }
};

bool improveViaHerbie(
    const std::string &inputExpr, ApplicableOutput &AO, Module *M,
    const TargetTransformInfo &TTI,
    std::unordered_map<Value *, FPNode *> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  SmallString<32> tmpin, tmpout;

  if (llvm::sys::fs::createUniqueFile("herbie_input_%%%%%%%%%%%%%%%%", tmpin,
                                      llvm::sys::fs::perms::owner_all)) {
    llvm::errs() << "Failed to create a unique input file.\n";
    return false;
  }

  if (llvm::sys::fs::createUniqueDirectory("herbie_output_%%%%%%%%%%%%%%%%",
                                           tmpout)) {
    llvm::errs() << "Failed to create a unique output directory.\n";
    return false;
  }

  std::ofstream input(tmpin.c_str());
  if (!input) {
    llvm::errs() << "Failed to open input file.\n";
    return 1;
  }
  input << inputExpr;
  input.close();

  std::string Program = HERBIE_BINARY;
  SmallVector<llvm::StringRef> Args = {Program,     "report",    "--seed",
                                       "239778888", "--timeout", "60"};

  Args.push_back("--disable");
  Args.push_back("generate:proofs"); // We can't show HTML reports

  if (HerbieDisableNumerics) {
    Args.push_back("--disable");
    Args.push_back("rules:numerics");
  }

  if (HerbieDisableTaylor) {
    Args.push_back("--disable");
    Args.push_back("generate:taylor");
  }

  if (HerbieDisableSetupSimplify) {
    Args.push_back("--disable");
    Args.push_back("setup:simplify");
  }

  if (HerbieDisableGenSimplify) {
    Args.push_back("--disable");
    Args.push_back("generate:simplify");
  }

  if (HerbieDisableRegime) {
    Args.push_back("--disable");
    Args.push_back("reduce:regimes");
  }

  if (HerbieDisableBranchExpr) {
    Args.push_back("--disable");
    Args.push_back("reduce:branch-expressions");
  }

  if (HerbieDisableAvgError) {
    Args.push_back("--disable");
    Args.push_back("reduce:avg-error");
  }

  Args.push_back(tmpin);
  Args.push_back(tmpout);

  std::string ErrMsg;
  bool ExecutionFailed = false;

  if (EnzymePrintFPOpt)
    llvm::errs() << "Executing: " << Program << "\n";

  llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/llvm::None,
                            /*Redirects=*/llvm::None,
                            /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg,
                            &ExecutionFailed);

  std::remove(tmpin.c_str());
  if (ExecutionFailed) {
    llvm::errs() << "Execution failed: " << ErrMsg << "\n";
    return false;
  }

  std::ifstream output((tmpout + "/results.json").str());
  if (!output) {
    llvm::errs() << "Failed to open output file.\n";
    return false;
  }
  std::string content((std::istreambuf_iterator<char>(output)),
                      std::istreambuf_iterator<char>());
  output.close();
  std::remove(tmpout.c_str());

  llvm::errs() << "Herbie output: " << content << "\n";

  Expected<json::Value> parsed = json::parse(content);
  if (!parsed) {
    llvm::errs() << "Failed to parse Herbie result!\n";
    return false;
  }

  json::Object *obj = parsed->getAsObject();
  json::Array &tests = *obj->getArray("tests");
  StringRef bestExpr = tests[0].getAsObject()->getString("output").getValue();

  if (bestExpr == "#f") {
    return false;
  }

  double bits = tests[0].getAsObject()->getNumber("bits").getValue();
  json::Array &costAccuracy =
      *tests[0].getAsObject()->getArray("cost-accuracy");

  json::Array &initial = *costAccuracy[0].getAsArray();
  double initialCostVal = initial[0].getAsNumber().getValue();
  double initialCost = 1.0;
  double initialAccuracy = 1.0 - initial[1].getAsNumber().getValue() / bits;
  AO.initialHerbieCost = initialCost;
  AO.initialAccuracy = initialAccuracy;

  json::Array &best = *costAccuracy[1].getAsArray();
  double bestCost = best[0].getAsNumber().getValue() / initialCostVal;
  double bestAccuracy = 1.0 - best[1].getAsNumber().getValue() / bits;

  RewriteCandidate bestCandidate(bestCost, bestAccuracy, bestExpr.str());
  bestCandidate.TTICost =
      getTTICost(bestExpr.str(), M, TTI, valueToNodeMap, symbolToValueMap);
  AO.candidates.push_back(bestCandidate);

  if (EnzymePrintHerbie) {
    llvm::errs() << "Initial: TTICost = " << AO.initialTTICost
                 << ", HerbieCost = " << initialCost
                 << ", Accuracy = " << initialAccuracy << "\n";
    llvm::errs() << "Best: TTICost = " << bestCandidate.TTICost
                 << ", HerbieCost = " << bestCost
                 << ", Accuracy = " << bestAccuracy
                 << ", Expression = " << bestExpr << "\n";
  }

  json::Array &alternatives = *costAccuracy[2].getAsArray();

  // Handle alternatives
  for (size_t i = 0; i < alternatives.size(); ++i) {
    json::Array &entry = *alternatives[i].getAsArray();
    double cost = entry[0].getAsNumber().getValue() / initialCostVal;
    double accuracy = 1.0 - entry[1].getAsNumber().getValue() / bits;
    StringRef expr = entry[2].getAsString().getValue();
    RewriteCandidate candidate(cost, accuracy, expr.str());
    candidate.TTICost =
        getTTICost(expr.str(), M, TTI, valueToNodeMap, symbolToValueMap);
    AO.candidates.push_back(candidate);
    if (EnzymePrintHerbie)
      llvm::errs() << "Alternative " << i + 1
                   << ": TTICost = " << candidate.TTICost
                   << ", HerbieCost = " << cost << ", Accuracy = " << accuracy
                   << ", Expression = " << expr << "\n";
  }

  return true;
}

std::string getHerbieOperator(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::FNeg:
    return "neg";
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

    // Special cases
    if (startsWith(funcName, "cbrt"))
      return "cbrt";

    std::regex regex("llvm\\.(\\w+)\\.?.*");
    std::smatch matches;
    if (std::regex_search(funcName, matches, regex) && matches.size() > 1) {
      if (matches[1].str() == "fmuladd")
        return "fma";
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
  case Instruction::FNeg:
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
             funcName.startswith("llvm.sqrt") || funcName.startswith("cbrt") ||
             funcName.startswith("llvm.pow") ||
             funcName.startswith("llvm.fma") ||
             funcName.startswith("llvm.fmuladd");
      // llvm.fabs is deliberately excluded
    }
    return false;
  }
  default:
    return false;
  }
}

struct ValueInfo {
  double minRes;
  double maxRes;
  unsigned executions;
  SmallVector<double, 2> lower;
  SmallVector<double, 2> upper;
};

void extractValueFromLog(const std::string &logPath,
                         const std::string &functionName, size_t blockIdx,
                         size_t instIdx, ValueInfo &data) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex valuePattern("^Value:" + functionName + ":" +
                          std::to_string(blockIdx) + ":" +
                          std::to_string(instIdx));
  std::regex newEntryPattern("^(Value|Grad):");

  while (getline(file, line)) {
    if (std::regex_search(line, valuePattern)) {
      std::string minResLine, maxResLine, executionsLine;
      if (getline(file, minResLine) && getline(file, maxResLine) &&
          getline(file, executionsLine)) {
        std::regex minResPattern(R"(MinRes = ([\d\.eE+-]+))");
        std::regex maxResPattern(R"(MaxRes = ([\d\.eE+-]+))");
        std::regex executionsPattern(R"(Executions = (\d+))");

        std::smatch minResMatch, maxResMatch, executionsMatch;
        if (std::regex_search(minResLine, minResMatch, minResPattern) &&
            std::regex_search(maxResLine, maxResMatch, maxResPattern) &&
            std::regex_search(executionsLine, executionsMatch,
                              executionsPattern)) {
          data.minRes = stringToDouble(minResMatch[1]);
          data.maxRes = stringToDouble(maxResMatch[1]);
          data.executions = std::stol(executionsMatch[1]);
        } else {
          std::string error =
              "Failed to parse stats for: Function: " + functionName +
              ", BlockIdx: " + std::to_string(blockIdx) +
              ", InstIdx: " + std::to_string(instIdx);
          llvm_unreachable(error.c_str());
        }
      }

      std::regex rangePattern(
          R"(Operand\[\d+\] = \[([\d\.eE+-]+), ([\d\.eE+-]+)\])");
      while (getline(file, line)) {
        if (std::regex_search(line, newEntryPattern)) {
          // All operands have been extracted
          return;
        }

        std::smatch rangeMatch;
        if (std::regex_search(line, rangeMatch, rangePattern)) {
          data.lower.push_back(stringToDouble(rangeMatch[1]));
          data.upper.push_back(stringToDouble(rangeMatch[2]));
        }
      }
    }
  }

  std::string error =
      "Failed to extract value info for: Function: " + functionName +
      ", BlockIdx: " + std::to_string(blockIdx) +
      ", InstIdx: " + std::to_string(instIdx);
  llvm_unreachable(error.c_str());
}

bool extractGradFromLog(const std::string &logPath,
                        const std::string &functionName, size_t blockIdx,
                        size_t instIdx, double &grad) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex gradPattern("^Grad:" + functionName + ":" +
                         std::to_string(blockIdx) + ":" +
                         std::to_string(instIdx));

  while (getline(file, line)) {
    if (std::regex_search(line, gradPattern)) {

      // Extract Grad data
      std::regex gradExtractPattern(R"(Grad = ([\d\.eE+-]+))");
      std::smatch gradMatch;
      if (getline(file, line) &&
          std::regex_search(line, gradMatch, gradExtractPattern)) {
        grad = stringToDouble(gradMatch[1]);
        return true;
      }
    }
  }

  llvm::errs() << "Failed to extract gradient for: Function: " << functionName
               << ", BlockIdx: " << blockIdx << ", InstIdx: " << instIdx
               << "\n";
  return false;
}

bool isLogged(const std::string &logPath, const std::string &functionName) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    assert(0 && "Failed to open log file");
  }

  std::regex functionRegex("^Value:" + functionName);

  std::string line;
  while (std::getline(file, line)) {
    if (std::regex_search(line, functionRegex)) {
      return true;
    }
  }

  return false;
}

std::string getPrecondition(
    const SmallSet<std::string, 8> &args,
    const std::unordered_map<Value *, FPNode *> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string preconditions;

  for (const auto &arg : args) {
    const auto *node = valueToNodeMap.at(symbolToValueMap.at(arg));
    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    std::ostringstream lowerStr, upperStr;
    lowerStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << lower;
    upperStr << std::setprecision(std::numeric_limits<double>::max_digits10)
             << std::scientific << upper;

    preconditions += " (<=" + (std::isinf(lower) ? "" : " " + lowerStr.str()) +
                     " " + arg +
                     (std::isinf(upper) ? "" : " " + upperStr.str()) + ")";
  }

  return preconditions.empty() ? "TRUE" : "(and" + preconditions + ")";
}

// Given the cost budget `FPOptComputationCostBudget`, we want to minimize the
// accuracy cost of the rewritten expressions.
bool accuracyGreedySolver(
    SmallVector<ApplicableOutput> &AOs,
    std::unordered_map<Value *, FPNode *> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy greedy solver with computation budget: "
               << FPOptComputationCostBudget << "\n";
  InstructionCost totalComputationCost = 0;

  for (auto &AO : AOs) {
    int bestCandidateIndex = -1;
    double bestAccuracyCost = std::numeric_limits<double>::infinity();
    InstructionCost bestCandidateComputationCost;

    for (auto &candidate : enumerate(AO.candidates)) {
      size_t i = candidate.index();
      auto candidateComputationCost = AO.getComputationCost(i);
      auto candidateAccuracyCost = AO.getAccuracyCost(i);
      llvm::errs() << "Candidate " << i << " for " << AO.expr
                   << " has accuracy cost: " << candidateAccuracyCost
                   << " and computation cost: " << candidateComputationCost
                   << "\n";

      // See if the candidate fits within the computation cost budget
      if (totalComputationCost + candidateComputationCost <=
          FPOptComputationCostBudget) {
        // Select the candidate with the lowest accuracy cost
        if (candidateAccuracyCost < bestAccuracyCost) {
          llvm::errs() << "Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candidateAccuracyCost;
          bestCandidateComputationCost = candidateComputationCost;
        }
      }
    }

    if (bestCandidateIndex != -1) {
      AO.apply(bestCandidateIndex, valueToNodeMap, symbolToValueMap);
      changed = true;
      totalComputationCost += bestCandidateComputationCost;
      llvm::errs() << "Updated total computation cost: " << totalComputationCost
                   << "\n\n";
    }
  }

  return changed;
}

bool accuracyDPSolver(
    SmallVector<ApplicableOutput> &AOs,
    std::unordered_map<Value *, FPNode *> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap =
      std::map<InstructionCost,
               SmallVector<std::pair<ApplicableOutput *, size_t>>>;

  CostMap costToAccuracyMap;
  costToAccuracyMap[0] = std::numeric_limits<double>::infinity();
  SolutionMap costToSolutionMap;
  costToSolutionMap[0] = {};

  for (auto &AO : AOs) {
    CostMap newCostToAccuracyMap = costToAccuracyMap;
    SolutionMap newCostToSolutionMap = costToSolutionMap;

    llvm::errs() << "Processing " << AO.expr << "\n";
    for (const auto &pair : costToAccuracyMap) {
      for (auto &candidate : enumerate(AO.candidates)) {
        InstructionCost currentComputationCost = pair.first;
        double currentAccuracyCost = pair.second;

        size_t i = candidate.index();
        auto candidateComputationCost = AO.getComputationCost(i);
        auto candidateAccuracyCost = AO.getAccuracyCost(i);

        InstructionCost newComputationCost =
            currentComputationCost + candidateComputationCost;
        double newAccuracyCost = currentAccuracyCost + candidateAccuracyCost;

        if (newComputationCost <= FPOptComputationCostBudget) {
          if (costToAccuracyMap.find(newComputationCost) ==
                  costToAccuracyMap.end() ||
              costToAccuracyMap[newComputationCost] > newAccuracyCost) {
            // Maintain the way to achieve the lowest accuracy cost for each
            // achievable computation cost
            newCostToAccuracyMap[newComputationCost] = newAccuracyCost;
            newCostToSolutionMap[newComputationCost] =
                costToSolutionMap[currentComputationCost];
            newCostToSolutionMap[newComputationCost].emplace_back(&AO, i);
            llvm::errs() << "Updating accuracy map (candidate " << i
                         << "): computation cost " << newComputationCost
                         << " -> accuracy cost " << newAccuracyCost << "\n";
          }
        }
      }
    }

    // Accuracy costs should be non-increasing
    for (auto it = std::next(newCostToAccuracyMap.begin());
         it != newCostToAccuracyMap.end(); ++it) {
      auto prev = std::prev(it);
      if (it->second > prev->second) {
        // Lower accuracy cost is achieved by a lower computation cost; inherit
        // the solution of the lower computation cost
        it->second = prev->second;
        newCostToSolutionMap[it->first] = newCostToSolutionMap[prev->first];
        llvm::errs() << "Correcting accuracy cost for computation cost "
                     << it->first << " to " << it->second
                     << " which comes from " << prev->first << "\n";
      }
    }

    costToAccuracyMap.swap(newCostToAccuracyMap);
    costToSolutionMap.swap(newCostToSolutionMap);
  }

  llvm::errs() << "DP Table: \n";
  for (const auto &entry : costToAccuracyMap) {
    llvm::errs() << "Computation cost: " << entry.first
                 << ", Accuracy cost: " << entry.second << "\n";
  }

  double minAccuracyCost = std::numeric_limits<double>::infinity();
  InstructionCost bestCost = 0;
  for (const auto &pair : costToAccuracyMap) {
    if (pair.second < minAccuracyCost) {
      minAccuracyCost = pair.second;
      bestCost = pair.first;
    }
  }

  llvm::errs() << "Minimum accuracy cost within budget: " << minAccuracyCost
               << "\n";
  llvm::errs() << "Computation cost budget used: " << bestCost << "\n";

  assert(costToSolutionMap.find(bestCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");
  for (const auto &solution : costToSolutionMap[bestCost]) {
    auto *AO = solution.first;
    size_t i = solution.second;
    AO->apply(i, valueToNodeMap, symbolToValueMap);
    changed = true;
  }

  return changed;
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F, const TargetTransformInfo &TTI) {
  const std::string functionName = F.getName().str();
  const std::string demangledName = llvm::demangle(functionName);

  // TODO: Finer control
  llvm::errs() << "Regex: " << FPOptTargetFuncRegex << "\n";
  std::regex targetFuncRegex(FPOptTargetFuncRegex);
  if (!std::regex_match(demangledName, targetFuncRegex)) {
    if (EnzymePrintFPOpt)
      llvm::errs() << "Skipping function: " << demangledName
                   << " (demangled) since it does not match the target regex\n";
    return false;
  }

  if (!FPOptLogPath.empty()) {
    if (!isLogged(FPOptLogPath, functionName)) {
      if (EnzymePrintFPOpt)
        llvm::errs()
            << "Skipping matched function: " << functionName
            << " since a log is provided but this function is not logged\n";
      return false;
    }
  }

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

  llvm::errs() << "FPOpt: Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!herbiable(I)) {
        valueToNodeMap[&I] = new FPLLValue(&I, "__nh", "__nh"); // Non-herbiable
        if (EnzymePrintFPOpt)
          llvm::errs() << "Registered FPLLValue for non-herbiable instruction: "
                       << I << "\n";
        continue;
      }

      std::string dtype;
      if (I.getType()->isFloatTy()) {
        dtype = "f32";
      } else if (I.getType()->isDoubleTy()) {
        dtype = "f64";
      } else {
        llvm_unreachable("Unexpected floating point type for instruction");
      }
      auto node = new FPLLValue(&I, getHerbieOperator(I), dtype);

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I).args() : I.operands();
      for (auto &operand : operands) {
        if (!valueToNodeMap.count(operand)) {
          if (auto Arg = dyn_cast<Argument>(operand)) {
            std::string dtype;
            if (Arg->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (Arg->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for argument");
            }
            valueToNodeMap[operand] = new FPLLValue(Arg, "__arg", dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for argument: " << *Arg
                           << "\n";
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            std::string dtype;
            if (C->getType()->isFloatTy()) {
              dtype = "f32";
            } else if (C->getType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable("Unexpected floating point type for constant");
            }
            valueToNodeMap[operand] = new FPConst(value.c_str(), dtype);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for " << dtype
                           << " constant: " << value << "\n";
          } else if (auto GV = dyn_cast<GlobalVariable>(operand)) {
            assert(
                GV->getType()->getPointerElementType()->isFloatingPointTy() &&
                "Global variable is not floating point type");
            std::string dtype;
            if (GV->getType()->getPointerElementType()->isFloatTy()) {
              dtype = "f32";
            } else if (GV->getType()->getPointerElementType()->isDoubleTy()) {
              dtype = "f64";
            } else {
              llvm_unreachable(
                  "Unexpected floating point type for global variable");
            }
            valueToNodeMap[operand] = new FPLLValue(GV, "__gv", dtype);
            if (EnzymePrintFPOpt)
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

  SmallSet<Value *, 8> component_seen;
  SmallVector<FPCC, 1> connected_components;
  for (auto &BB : F) {
    for (auto &I : BB) {
      // Not a herbiable instruction, doesn't make sense to create graph node
      // out of.
      if (!herbiable(I)) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skipping non-herbiable instruction: " << I << "\n";
        continue;
      }

      // Instruction is already in a set
      if (component_seen.contains(&I)) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skipping already seen instruction: " << I << "\n";
        continue;
      }

      if (EnzymePrintFPOpt)
        llvm::errs() << "Starting floodfill from: " << I << "\n";

      SmallVector<Value *, 8> todo;
      SetVector<Value *> input_seen;
      SetVector<Instruction *> output_seen;
      SetVector<Instruction *> operation_seen;
      todo.push_back(&I);
      while (!todo.empty()) {
        auto cur = todo.pop_back_val();
        assert(valueToNodeMap.count(cur) && "Node not found in valueToNodeMap");

        // We now can assume that this is a herbiable expression
        // Since we can only herbify instructions, let's assert that
        assert(isa<Instruction>(cur));
        auto I2 = cast<Instruction>(cur);

        // Don't repeat any instructions we've already seen (to avoid loops
        // for phi nodes)
        if (operation_seen.contains(I2)) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "Skipping already seen instruction: " << *I2
                         << "\n";
          continue;
        }

        // Assume that a herbiable expression can only be in one connected
        // component.
        assert(!component_seen.contains(cur));

        if (EnzymePrintFPOpt)
          llvm::errs() << "Insert to operation_seen and component_seen: " << *I2
                       << "\n";
        operation_seen.insert(I2);
        component_seen.insert(cur);

        auto operands =
            isa<CallInst>(I2) ? cast<CallInst>(I2)->args() : I2->operands();

        for (auto &operand_ : enumerate(operands)) {
          auto &operand = operand_.value();
          auto i = operand_.index();
          if (!herbiable(*operand)) {
            if (EnzymePrintFPOpt)
              llvm::errs() << "Non-herbiable input found: " << *operand << "\n";
            input_seen.insert(operand);

            // look up error log to get bounds of non-herbiable inputs
            if (!FPOptLogPath.empty()) {
              ValueInfo valueInfo;
              auto blockIt = std::find_if(
                  I2->getFunction()->begin(), I2->getFunction()->end(),
                  [&](const auto &block) { return &block == I2->getParent(); });
              assert(blockIt != I2->getFunction()->end() && "Block not found");
              size_t blockIdx =
                  std::distance(I2->getFunction()->begin(), blockIt);
              auto instIt =
                  std::find_if(I2->getParent()->begin(), I2->getParent()->end(),
                               [&](const auto &curr) { return &curr == I2; });
              assert(instIt != I2->getParent()->end() &&
                     "Instruction not found");
              size_t instIdx = std::distance(I2->getParent()->begin(), instIt);

              extractValueFromLog(FPOptLogPath, functionName, blockIdx, instIdx,
                                  valueInfo);
              auto *node = valueToNodeMap[operand];
              node->updateBounds(valueInfo.lower[i], valueInfo.upper[i]);

              if (EnzymePrintFPOpt) {
                llvm::errs() << "Range of " << *operand << " is ["
                             << node->getLowerBound() << ", "
                             << node->getUpperBound() << "]\n";
              }
            }
          } else {
            if (EnzymePrintFPOpt)
              llvm::errs() << "Adding operand to todo list: " << *operand
                           << "\n";
            todo.push_back(operand);
          }
        }

        for (auto U : I2->users()) {
          if (auto I3 = dyn_cast<Instruction>(U)) {
            if (!herbiable(*I3)) {
              if (EnzymePrintFPOpt)
                llvm::errs() << "Output instruction found: " << *I2 << "\n";
              output_seen.insert(I2);
            } else {
              if (EnzymePrintFPOpt)
                llvm::errs() << "Adding user to todo list: " << *I3 << "\n";
              todo.push_back(I3);
            }
          }
        }
      }

      // Don't bother with graphs without any herbiable operations
      if (!operation_seen.empty()) {
        if (EnzymePrintFPOpt) {
          llvm::errs() << "Found a connected component with "
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
        }

        // TODO: Further check
        if (operation_seen.size() == 1) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "Skipping trivial connected component\n";
          continue;
        }

        FPCC origCC{input_seen, output_seen, operation_seen};
        SmallVector<FPCC, 1> newCCs;
        splitFPCC(origCC, newCCs);

        for (auto &CC : newCCs) {
          for (auto *input : CC.inputs) {
            valueToNodeMap[input]->markAsInput();
          }
        }

        if (!FPOptLogPath.empty()) {
          for (auto &CC : newCCs) {
            // Extract grad and value info for all outputs. This implicitly
            // extracts the value info for herbiable intermediate `inputs` since
            // they are also `outputs` of a previous FPCC.
            for (auto &output : CC.outputs) {
              double grad = 0;
              auto blockIt = std::find_if(
                  output->getFunction()->begin(), output->getFunction()->end(),
                  [&](const auto &block) {
                    return &block == output->getParent();
                  });
              assert(blockIt != output->getFunction()->end() &&
                     "Block not found");
              size_t blockIdx =
                  std::distance(output->getFunction()->begin(), blockIt);
              auto instIt = std::find_if(
                  output->getParent()->begin(), output->getParent()->end(),
                  [&](const auto &curr) { return &curr == output; });
              assert(instIt != output->getParent()->end() &&
                     "Instruction not found");
              size_t instIdx =
                  std::distance(output->getParent()->begin(), instIt);
              bool found = extractGradFromLog(FPOptLogPath, functionName,
                                              blockIdx, instIdx, grad);

              auto *node = valueToNodeMap[output];

              if (found) {
                node->grad = grad;

                ValueInfo valueInfo;
                extractValueFromLog(FPOptLogPath, functionName, blockIdx,
                                    instIdx, valueInfo);
                node->executions = valueInfo.executions;
                node->updateBounds(valueInfo.minRes, valueInfo.maxRes);

                if (EnzymePrintFPOpt) {
                  llvm::errs() << "Range of " << *output << " is ["
                               << node->getLowerBound() << ", "
                               << node->getUpperBound() << "]\n";
                }

                if (EnzymePrintFPOpt)
                  llvm::errs()
                      << "Grad of " << *output << " is: " << node->grad << "\n"
                      << "Execution count of " << *output
                      << " is: " << node->executions << "\n";
              } else { // Unknown bounds
                if (EnzymePrintFPOpt)
                  llvm::errs()
                      << "Grad of " << *output << " are not found in the log\n";
              }
            }
          }
        }

        connected_components.insert(connected_components.end(), newCCs.begin(),
                                    newCCs.end());
      }
    }
  }

  llvm::errs() << "FPOpt: Found " << connected_components.size()
               << " connected components in " << F.getName() << "\n";

  // 1) Identify subgraphs of the computation which can be entirely represented
  // in herbie-style arithmetic
  // 2) Make the herbie FP-style expression by
  // converting llvm instructions into herbie string (FPNode ....)
  if (connected_components.empty()) {
    if (EnzymePrintFPOpt)
      llvm::errs() << "No herbiable connected components found\n";
    return false;
  }

  SmallVector<ApplicableOutput> AOs;

  for (auto &component : connected_components) {
    assert(component.inputs.size() > 0 && "No inputs found for component");
    for (const auto &input : component.inputs) {
      auto node = valueToNodeMap[input];
      if (node->op == "__const") {
        // Constants don't need a symbol
        continue;
      }
      if (!node->hasSymbol()) {
        node->symbol = getNextSymbol();
      }
      symbolToValueMap[node->symbol] = input;
      if (EnzymePrintFPOpt)
        llvm::errs() << "assigning symbol: " << node->symbol << " to " << *input
                     << "\n";
    }

    assert(component.outputs.size() > 0 && "No outputs found for component");
    for (auto &output : component.outputs) {
      // 3) run fancy opts
      double grad = valueToNodeMap[output]->grad;
      unsigned executions = valueToNodeMap[output]->executions;

      // TODO: For now just skip if grad is 0
      if (!FPOptLogPath.empty() && grad == 0.) {
        continue;
      }

      // TODO: Herbie properties
      std::string expr =
          valueToNodeMap[output]->toFullExpression(valueToNodeMap);
      SmallSet<std::string, 8> args;
      getUniqueArgs(expr, args);

      std::string properties = ":herbie-conversions ([binary64 binary32])";
      if (valueToNodeMap[output]->dtype == "f32") {
        properties += " :precision binary32";
      } else if (valueToNodeMap[output]->dtype == "f64") {
        properties += " :precision binary64";
      } else {
        llvm_unreachable("Unexpected dtype");
      }

      if (!FPOptLogPath.empty()) {
        std::string precondition =
            getPrecondition(args, valueToNodeMap, symbolToValueMap);
        properties += " :pre " + precondition;
      }

      std::string argStr;
      for (const auto &arg : args) {
        if (!argStr.empty())
          argStr += " ";
        argStr += arg;
      }

      std::string herbieInput =
          "(FPCore (" + argStr + ") " + properties + " " + expr + ")";
      if (EnzymePrintHerbie)
        llvm::errs() << "Herbie input:\n" << herbieInput << "\n";

      ApplicableOutput AO(component, output, expr, grad, executions, TTI);
      if (!improveViaHerbie(herbieInput, AO, F.getParent(), TTI, valueToNodeMap,
                            symbolToValueMap)) {
        if (EnzymePrintHerbie)
          llvm::errs() << "Failed to optimize an expression using Herbie!\n";
        continue;
      }

      AOs.push_back(std::move(AO));
    }
  }

  // Perform rewrites
  if (EnzymePrintFPOpt) {
    for (auto &AO : AOs) {
      // TODO: Solver
      // Available Parameters:
      // 1. gradients at the output llvm::Value
      // 2. costs of the potential rewrites from Herbie (lower is preferred)
      // 3. percentage accuracies of potential rewrites (higher is better)
      // 4*. TTI costs of potential rewrites (TODO: need to consider branches)
      // 5*. Custom error estimates of potential rewrites (TODO)

      llvm::errs() << "\n################################\n";
      llvm::errs() << "Initial TTICost: " << AO.initialTTICost << "\n";
      llvm::errs() << "Initial HerbieCost: " << AO.initialHerbieCost << "\n";
      llvm::errs() << "Initial Accuracy: " << AO.initialAccuracy << "\n";
      llvm::errs() << "Initial Expression: " << AO.expr << "\n";
      llvm::errs() << "Grad: " << AO.grad << "\n\n";
      llvm::errs() << "Candidates:\n";
      llvm::errs() << "TTICost\tHerbieCost\tAccuracy\tExpression\n";
      llvm::errs() << "--------------------------------\n";
      for (const auto &candidate : AO.candidates) {
        llvm::errs() << candidate.TTICost << "\t" << candidate.herbieCost
                     << "\t" << candidate.accuracy << "\t" << candidate.expr
                     << "\n";
      }
      llvm::errs() << "################################\n\n";
    }
  }

  if (!FPOptEnableSolver) {
    for (auto &AO : AOs) {
      AO.apply(0, valueToNodeMap, symbolToValueMap);
      changed = true;
    }
  } else {
    // TODO: Solver
    if (FPOptLogPath.empty()) {
      llvm::errs() << "FPOpt: Solver enabled but no log file is provided\n";
      return false;
    }
    if (FPOptSolverType == "greedy") {
      changed = accuracyGreedySolver(AOs, valueToNodeMap, symbolToValueMap);
    } else if (FPOptSolverType == "dp") {
      changed = accuracyDPSolver(AOs, valueToNodeMap, symbolToValueMap);
    } else {
      llvm::errs() << "FPOpt: Unknown solver type: " << FPOptSolverType << "\n";
      return false;
    }
  }

  llvm::errs() << "FPOpt: Finished optimizing " << F.getName() << "\n";

  for (auto &[_, node] : valueToNodeMap) {
    delete node;
  }

  // Cleanup
  if (changed) {
    for (auto &component : connected_components) {
      if (component.outputs_rewritten != component.outputs.size()) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Skip erasing a connect component: only rewrote "
                       << component.outputs_rewritten << " of "
                       << component.outputs.size() << " outputs\n";
        continue; // Intermediate operations cannot be erased safely
      }
      for (auto *I : component.operations) {
        if (EnzymePrintFPOpt)
          llvm::errs() << "Erasing: " << *I << "\n";
        if (!I->use_empty()) {
          I->replaceAllUsesWith(UndefValue::get(I->getType()));
        }
        I->eraseFromParent();
      }
    }

    llvm::errs() << "FPOpt: Finished cleaning up " << F.getName() << "\n";
  }

  if (EnzymePrintFPOpt) {
    llvm::errs() << "Finished fpOptimize\n";
    F.print(llvm::errs());
  }

  return changed;
}

namespace {

class FPOpt final : public FunctionPass {
public:
  static char ID;
  FPOpt() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    FunctionPass::getAnalysisUsage(AU);
  }
  bool runOnFunction(Function &F) override {
    auto &TTI = getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    return fpOptimize(F, TTI);
  }
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
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      const auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
      changed |= fpOptimize(F, TTI);
    }
  }

  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
llvm::AnalysisKey FPOptNewPM::Key;
