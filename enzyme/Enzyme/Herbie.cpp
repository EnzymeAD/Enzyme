#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

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
    LogPath("log-path", cl::init(""), cl::Hidden,
            cl::desc("Which log to use in the FPOpt pass"));
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
static cl::opt<bool>
    HerbieDisableRegime("herbie-disable-regime", cl::init(false), cl::Hidden,
                        cl::desc("Stop Herbie from simplifying expressions "
                                 "during the main improvement loop"));
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
static cl::opt<InstructionCost> FPOptComputationCostBudget(
    "fpopt-comp-cost-budget", cl::init(1000000000), cl::Hidden,
    cl::desc("The maximum computation cost budget for the solver"));
}

class FPNode {
public:
  std::string op;
  std::string symbol;
  SmallVector<FPNode *, 1> operands;
  double grad;
  unsigned executions;

  FPNode(const std::string &op) : op(op) {}
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

  virtual void updateBounds(double lower, double upper) {
    assert(0 && "Trying to update bounds of a non-input node!");
  }
  virtual double getLowerBound() const {
    assert(0 && "Trying to get lower bound of a non-input node!");
  }
  virtual double getUpperBound() const {
    assert(0 && "Trying to get upper bound of a non-input node!");
  }

  virtual Value *getValue(IRBuilder<> &builder) {
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Generating new instruction for op: " << op << "\n";

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
      // Lower versions do not have tan intrinsic
      val = builder.CreateFDiv(
          builder.CreateUnaryIntrinsic(Intrinsic::sin, operandValues[0]),
          builder.CreateUnaryIntrinsic(Intrinsic::cos, operandValues[0]),
          "herbie.tan");
#endif
    } else if (op == "exp") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::exp, operandValues[0],
                                         nullptr, "herbie.exp");
    } else if (op == "expm1") {
      val = builder.CreateFSub(
          builder.CreateUnaryIntrinsic(Intrinsic::exp, operandValues[0]),
          ConstantFP::get(operandValues[0]->getType(), 1.0), "herbie.expm1");
    } else if (op == "log") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::log, operandValues[0],
                                         nullptr, "herbie.log");
    } else if (op == "log1p") {
      val = builder.CreateUnaryIntrinsic(
          Intrinsic::log,
          builder.CreateFAdd(ConstantFP::get(operandValues[0]->getType(), 1.0),
                             operandValues[0]),
          nullptr, "herbie.log1p");
    } else if (op == "sqrt") {
      val = builder.CreateUnaryIntrinsic(Intrinsic::sqrt, operandValues[0],
                                         nullptr, "herbie.sqrt");
    } else if (op == "cbrt") {
      val = builder.CreateBinaryIntrinsic(
          Intrinsic::pow, operandValues[0],
          ConstantFP::get(operandValues[0]->getType(), 1.0 / 3.0), nullptr,
          "herbie.cbrt");
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
      val = builder.CreateUnaryIntrinsic(
          Intrinsic::sqrt,
          builder.CreateFAdd(
              builder.CreateFMul(operandValues[0], operandValues[0]),
              builder.CreateFMul(operandValues[1], operandValues[1])),
          nullptr, "herbie.hypot");
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
      llvm::errs() << "Unknown operator: " << op << "\n";
      assert(0 && "Failed to generate optimized IR");
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

public:
  FPLLValue(Value *value) : FPNode("__arg"), value(value) {}

  virtual std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    assert(hasSymbol() && "FPLLValue has no symbol!");
    return symbol;
  }

  virtual void updateBounds(double lower, double upper) override {
    lb = std::min(lb, lower);
    ub = std::max(ub, upper);
    if (EnzymePrintFPOpt)
      llvm::errs() << "Updated bounds for " << *value << ": [" << lb << ", "
                   << ub << "]\n";
  }

  virtual double getLowerBound() const override { return lb; }
  virtual double getUpperBound() const override { return ub; }

  virtual Value *getValue(IRBuilder<> &builder) override { return value; }

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
  FPConst(std::string strValue) : FPNode("__const"), strValue(strValue) {}

  virtual std::string toFullExpression(
      std::unordered_map<Value *, FPNode *> &valueToNodeMap) override {
    return strValue;
  }

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
    if (strValue == "+inf.0") {
      return ConstantFP::getInfinity(builder.getDoubleTy(), false);
    } else if (strValue == "-inf.0") {
      return ConstantFP::getInfinity(builder.getDoubleTy(), true);
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

    // TODO eventually have this be typed
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Returning " << strValue
    //                << " as constant: " << constantValue << "\n";
    return ConstantFP::get(builder.getDoubleTy(), constantValue);
  }

  bool isExpression() const override { return false; }
};

FPNode *
parseHerbieExpr(const std::string &expr,
                std::unordered_map<Value *, FPNode *> &valueToNodeMap,
                std::unordered_map<std::string, Value *> &symbolToValueMap) {
  // if (EnzymePrintFPOpt)
  //   llvm::errs() << "Parsing: " << expr << "\n";
  auto trimmedExpr = expr;
  trimmedExpr.erase(0, trimmedExpr.find_first_not_of(" "));
  trimmedExpr.erase(trimmedExpr.find_last_not_of(" ") + 1);

  // Arguments
  if (trimmedExpr.front() != '(' && trimmedExpr.front() != '#') {
    return valueToNodeMap[symbolToValueMap[trimmedExpr]];
  }

  // Constants
  std::regex constantPattern(
      "^#s\\(literal\\s+([-+]?\\d+(/\\d+)?|[-+]?inf\\.0)\\s+\\w+\\)$");

  std::smatch matches;
  if (std::regex_match(trimmedExpr, matches, constantPattern)) {
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Found __const " << matches[1].str() << "\n";
    return new FPConst(matches[1].str());
  }

  if (trimmedExpr.front() != '(' || trimmedExpr.back() != ')') {
    llvm::errs() << "Unexpected subexpression: " << trimmedExpr << "\n";
    assert(0 && "Failed to parse Herbie expression");
  }

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

      if (EnzymePrintFPOpt)
        llvm::errs() << "Cost of " << *I << " is: " << instCost << "\n";

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

  tempFunction->print(llvm::errs());

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

struct FPComponent {
  SetVector<Value *> inputs;
  SetVector<Value *> outputs;
  SetVector<Instruction *> operations;
  size_t outputs_rewritten = 0;

  explicit FPComponent(SetVector<Value *> inputs, SetVector<Value *> outputs,
                       SetVector<Instruction *> operations)
      : inputs(std::move(inputs)), outputs(std::move(outputs)),
        operations(std::move(operations)) {}
};

class ApplicableOutput {
public:
  FPComponent &component;
  Value *oldOutput;
  std::string expr;
  double grad;
  unsigned executions;
  InstructionCost initialTTICost;    // Requires manual initialization
  InstructionCost initialHerbieCost; // Requires manual initialization
  double initialAccuracy;            // Requires manual initialization
  SmallVector<RewriteCandidate> candidates;

  explicit ApplicableOutput(FPComponent &component, Value *oldOutput,
                            std::string expr, double grad, unsigned executions,
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
             funcName.startswith("llvm.fmuladd") ||
             funcName.startswith("llvm.fabs");
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
  SmallVector<double, 4> lower;
  SmallVector<double, 4> upper;
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

          llvm::errs() << "Extracted value info: MinRes = " << data.minRes
                       << ", MaxRes = " << data.maxRes
                       << ", Executions = " << data.executions << "\n";
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

    if (std::isinf(lower) && std::isinf(upper))
      continue;

    if (std::isinf(lower)) {
      preconditions += " (<= " + arg + " " + std::to_string(upper) + ")";
      continue;
    }

    if (std::isinf(upper)) {
      preconditions += " (>= " + arg + " " + std::to_string(lower) + ")";
      continue;
    }

    preconditions += " (<= " + std::to_string(lower) + " " + arg + " " +
                     std::to_string(upper) + ")";
  }

  return preconditions.empty() ? "TRUE" : "(and" + preconditions + ")";
}

bool getErrorsWithJIT(const Value *oldOutput, const Value *newOutput,
                      const Function *F, double &oldError, double &newError) {
  // LLVMContext &Context = oldOutput->getContext();

  std::string errStr;
  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();

  std::unique_ptr<Module> M = CloneModule(*F->getParent());
  if (!M) {
    llvm::errs() << "Failed to clone the module.\n";
    return false;
  }

  Function *clonedFunction =
      Function::Create(F->getFunctionType(), Function::ExternalLinkage,
                       F->getName() + "_cloned", M.get());

  ValueToValueMapTy VMap;
  auto destArgIt = clonedFunction->arg_begin();
  for (auto &arg : F->args()) {
    VMap[&arg] = &*destArgIt++;
  }

  SmallVector<ReturnInst *, 8> Returns;
  CloneFunctionInto(clonedFunction, F, VMap,
                    CloneFunctionChangeType::DifferentModule, Returns);

  assert(VMap.count(oldOutput) && "Old output not found in VMap");
  VMap[oldOutput]->replaceAllUsesWith(VMap[newOutput]);

  llvm::errs() << "Cloned module: \n";
  M->print(llvm::errs(), nullptr);

  auto JIT = orc::LLJITBuilder().create();
  if (!JIT) {
    llvm::errs() << "Failed to create LLJIT: " << toString(JIT.takeError())
                 << "\n";
    return false;
  }

  auto &J = *JIT;
  J->getMainJITDylib().addGenerator(
      cantFail(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          J->getDataLayout().getGlobalPrefix())));

  auto TSM =
      orc::ThreadSafeModule(std::move(M), std::make_unique<LLVMContext>());
  if (auto Err = J->addIRModule(std::move(TSM))) {
    llvm::errs() << "Failed to add module: " << toString(std::move(Err))
                 << "\n";
    return false;
  }

  llvm::errs() << "Looking up function\n";
  auto Sym = J->lookup(clonedFunction->getName());
  if (!Sym) {
    llvm::errs() << "Failed to find symbol: " << toString(Sym.takeError())
                 << "\n";
    return false;
  }

  // TODO: Different for LLVM 15 and above
  llvm::errs() << "JITting function\n";
  auto *FP = (double (*)())(uintptr_t)Sym->getAddress();
  double result = FP();

  llvm::errs() << "Result of function: " << result << "\n";

  return true;
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
    size_t bestCandidateIndex = -1;
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

  CostMap accuracy;
  accuracy[0] = 0.0;
  SolutionMap solutions;
  solutions[0] = {};

  for (auto &AO : AOs) {
    CostMap newAccuracy = accuracy;
    SolutionMap newSolutions = solutions;

    llvm::errs() << "Processing " << AO.expr << "\n";
    for (const auto &pair : accuracy) {
      for (auto &candidate : enumerate(AO.candidates)) {
        size_t i = candidate.index();
        auto candidateComputationCost = AO.getComputationCost(i);
        auto candidateAccuracyCost = AO.getAccuracyCost(i);

        InstructionCost newComputationCost =
            pair.first + candidateComputationCost;
        double newAccuracyCost = pair.second + candidateAccuracyCost;

        if (newComputationCost <= FPOptComputationCostBudget) {
          if (newAccuracy.find(newComputationCost) == newAccuracy.end() ||
              newAccuracy[newComputationCost] > newAccuracyCost) {
            newAccuracy[newComputationCost] = newAccuracyCost;
            newSolutions[newComputationCost] = solutions[pair.first];
            newSolutions[newComputationCost].emplace_back(&AO, i);
            llvm::errs() << "Updating accuracy map (candidate " << i
                         << "): computation cost " << newComputationCost
                         << " -> accuracy cost " << newAccuracyCost << "\n";
            // llvm::errs() << "Current available solutions: ";
            // for (const auto &solution : newSolutions[newComputationCost]) {
            //   llvm::errs() << "\t" << solution.first->expr << " --> "
            //                <<
            //                solution.first->candidates[solution.second].expr
            //                << "\n";
            // }
          }
        }
      }
    }

    // Accuracy costs should be non-increasing
    for (auto it = std::next(newAccuracy.begin()); it != newAccuracy.end();
         ++it) {
      auto prev = std::prev(it);
      if (it->second > prev->second) {
        it->second = prev->second;
        newSolutions[it->first] = newSolutions[prev->first];
        llvm::errs() << "Correcting accuracy cost for computation cost "
                     << it->first << " to " << it->second
                     << " which comes from " << prev->first << "\n";
      }
    }

    accuracy.swap(newAccuracy);
    solutions.swap(newSolutions);
  }

  llvm::errs() << "DP Table: \n";
  for (const auto &entry : accuracy) {
    llvm::errs() << "Computation cost: " << entry.first
                 << ", Accuracy cost: " << entry.second << "\n";
  }

  double minAccuracyCost = std::numeric_limits<double>::infinity();
  InstructionCost bestCost = 0;
  for (const auto &entry : accuracy) {
    if (entry.second < minAccuracyCost) {
      minAccuracyCost = entry.second;
      bestCost = entry.first;
    }
  }

  llvm::errs() << "Minimum accuracy cost within budget: " << minAccuracyCost
               << "\n";
  llvm::errs() << "Computation cost budget used: " << bestCost << "\n";

  assert(solutions.find(bestCost) != solutions.end() &&
         "FPOpt DP solver: expected a solution!");
  for (const auto &solution : solutions[bestCost]) {
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
  std::string functionName = F.getName().str();

  // TODO: Finer control
  if (!LogPath.empty()) {
    if (!isLogged(LogPath, functionName)) {
      if (EnzymePrintFPOpt)
        llvm::errs() << "Skipping function: " << F.getName()
                     << " since it is not logged\n";
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
        valueToNodeMap[&I] = new FPLLValue(&I);
        if (EnzymePrintFPOpt)
          llvm::errs() << "Registered FPLLValue for non-herbiable instruction: "
                       << I << "\n";
        continue;
      }

      auto node = new FPNode(getHerbieOperator(I));

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I).args() : I.operands();
      for (auto &operand : operands) {
        if (!valueToNodeMap.count(operand)) {
          if (auto Arg = dyn_cast<Argument>(operand)) {
            valueToNodeMap[operand] = new FPLLValue(Arg);
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for argument: " << *Arg
                           << "\n";
          } else if (auto C = dyn_cast<ConstantFP>(operand)) {
            SmallString<10> value;
            C->getValueAPF().toString(value);
            valueToNodeMap[operand] = new FPConst(value.c_str());
            if (EnzymePrintFPOpt)
              llvm::errs() << "Registered FPNode for constant: " << value
                           << "\n";
          } else if (auto GV = dyn_cast<GlobalVariable>(operand)) {
            valueToNodeMap[operand] = new FPLLValue(GV);
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
  SmallVector<FPComponent, 1> connected_components;
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

            // look up error log to get bounds of the operand of I2
            if (!LogPath.empty()) {
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

              extractValueFromLog(LogPath, functionName, blockIdx, instIdx,
                                  valueInfo);
              auto *node = valueToNodeMap[operand];
              node->updateBounds(valueInfo.lower[i], valueInfo.upper[i]);
              node->executions = valueInfo.executions;

              if (EnzymePrintFPOpt)
                llvm::errs()
                    << "Range of " << *operand << " is [" << valueInfo.lower[i]
                    << ", " << valueInfo.upper[i] << "]\n";
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

              // Look up grad log to get grad of output I2
              if (!LogPath.empty()) {
                double grad = 0;
                auto blockIt = std::find_if(I2->getFunction()->begin(),
                                            I2->getFunction()->end(),
                                            [&](const auto &block) {
                                              return &block == I2->getParent();
                                            });
                assert(blockIt != I2->getFunction()->end() &&
                       "Block not found");
                size_t blockIdx =
                    std::distance(I2->getFunction()->begin(), blockIt);
                auto instIt = std::find_if(
                    I2->getParent()->begin(), I2->getParent()->end(),
                    [&](const auto &curr) { return &curr == I2; });
                assert(instIt != I2->getParent()->end() &&
                       "Instruction not found");
                size_t instIdx =
                    std::distance(I2->getParent()->begin(), instIt);
                bool found = extractGradFromLog(LogPath, functionName, blockIdx,
                                                instIdx, grad);

                auto *node = valueToNodeMap[I2];
                if (found) {
                  node->grad = grad;
                  if (EnzymePrintFPOpt)
                    llvm::errs()
                        << "Grad of " << *I2 << " is: " << grad << "\n";
                } else { // Unknown bounds
                  if (EnzymePrintFPOpt)
                    llvm::errs()
                        << "Grad of " << *I2 << " are not found in the log\n";
                }
              }
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

        connected_components.emplace_back(std::move(input_seen),
                                          std::move(output_seen),
                                          std::move(operation_seen));
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
      // TODO: Herbie properties
      std::string expr =
          valueToNodeMap[output]->toFullExpression(valueToNodeMap);
      SmallSet<std::string, 8> args;
      getUniqueArgs(expr, args);

      std::string properties =
          ":precision binary64 :herbie-conversions ([binary64 binary32])";

      if (!LogPath.empty()) {
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

      // 3) run fancy opts
      double grad = valueToNodeMap[output]->grad;
      unsigned executions = valueToNodeMap[output]->executions;

      // TODO: For now just skip if grad is 0
      if (grad == 0.) {
        continue;
      }

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
    if (LogPath.empty()) {
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
