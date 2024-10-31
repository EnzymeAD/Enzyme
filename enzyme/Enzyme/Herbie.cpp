#include <llvm/Config/llvm-config.h>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "llvm/Analysis/TargetTransformInfo.h"

#include "llvm/Demangle/Demangle.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/JSON.h>

#include "llvm/Pass.h"

#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <mpfr.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

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
static cl::opt<std::string>
    FPOptCostModelPath("fpopt-cost-model-path", cl::init(""), cl::Hidden,
                       cl::desc("Use a custom cost model in the FPOpt pass"));
static cl::opt<std::string> FPOptTargetFuncRegex(
    "fpopt-target-func-regex", cl::init(".*"), cl::Hidden,
    cl::desc("Regex pattern to match target functions in the FPOpt pass"));
static cl::opt<bool> FPOptEnableHerbie(
    "fpopt-enable-herbie", cl::init(true), cl::Hidden,
    cl::desc("Use Herbie to rewrite floating-point expressions"));
static cl::opt<bool> FPOptEnablePT(
    "fpopt-enable-pt", cl::init(false), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
static cl::opt<int> HerbieTimeout("herbie-timeout", cl::init(60), cl::Hidden,
                                  cl::desc("Herbie's timeout to use for each "
                                           "candidate expressions."));
static cl::opt<int>
    HerbieNumPoints("herbie-num-pts", cl::init(512), cl::Hidden,
                    cl::desc("Number of input points Herbie uses to evaluate "
                             "candidate expressions."));
static cl::opt<int> HerbieNumIters(
    "herbie-num-iters", cl::init(6), cl::Hidden,
    cl::desc("Number of times Herbie attempts to improve accuracy."));
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
static cl::opt<unsigned> FPOptMaxFPCCDepth(
    "fpopt-max-fpcc-depth", cl::init(10), cl::Hidden,
    cl::desc("The maximum depth of a floating-point connected component"));
static cl::opt<unsigned>
    FPOptRandomSeed("fpopt-random-seed", cl::init(239778888), cl::Hidden,
                    cl::desc("The random seed used in the FPOpt pass"));
static cl::opt<unsigned>
    FPOptNumSamples("fpopt-num-samples", cl::init(10), cl::Hidden,
                    cl::desc("Number of sampled points for input hypercube"));
static cl::opt<unsigned>
    FPOptMaxMPFRPrec("fpopt-max-mpfr-prec", cl::init(1024), cl::Hidden,
                     cl::desc("Max precision for MPFR gold value computation"));
static cl::opt<bool> FPOptEarlyPrune(
    "fpopt-early-prune", cl::init(true), cl::Hidden,
    cl::desc("Prune dominated candidates in expression transformation phases"));
static cl::opt<double> FPOptCostDominanceThreshold(
    "fpopt-cost-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for cost dominance in DP solver"));
static cl::opt<double> FPOptAccuracyDominanceThreshold(
    "fpopt-acc-dom-thres", cl::init(0.05), cl::Hidden,
    cl::desc("The threshold for accuracy dominance in DP solver"));
}

class FPNode {
public:
  enum class NodeType { Node, LLValue, Const };

private:
  const NodeType ntype;

public:
  std::string op;
  std::string dtype;
  std::string symbol;
  SmallVector<std::shared_ptr<FPNode>, 2> operands;
  double grad;
  double geometricAvg;
  unsigned executions;

  explicit FPNode(const std::string &op, const std::string &dtype)
      : ntype(NodeType::Node), op(op), dtype(dtype) {}
  explicit FPNode(NodeType ntype, const std::string &op,
                  const std::string &dtype)
      : ntype(ntype), op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  NodeType getType() const { return ntype; }

  void addOperand(std::shared_ptr<FPNode> operand) {
    operands.push_back(operand);
  }

  virtual bool hasSymbol() const {
    std::string msg = "Unexpected invocation of `hasSymbol` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }

  virtual std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap) {
    std::string msg = "Unexpected invocation of `toFullExpression` on an "
                      "unmaterialized " +
                      op + " FPNode";
    llvm_unreachable(msg.c_str());
  }

  unsigned getMPFRPrec() const {
    if (dtype == "f16")
      return 11;
    if (dtype == "f32")
      return 24;
    if (dtype == "f64")
      return 53;
    std::string msg =
        "getMPFRPrec: operator " + op + " has unknown dtype " + dtype;
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

  virtual Value *getLLValue(IRBuilder<> &builder,
                            const ValueToValueMapTy *VMap = nullptr) {
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Generating new instruction for op: " << op << "\n";
    Module *M = builder.GetInsertBlock()->getModule();

    if (op == "if") {
      Value *condValue = operands[0]->getLLValue(builder, VMap);
      auto IP = builder.GetInsertPoint();

      Instruction *Then, *Else;
      SplitBlockAndInsertIfThenElse(condValue, &*IP, &Then, &Else);

      Then->getParent()->setName("herbie.then");
      builder.SetInsertPoint(Then);
      Value *ThenVal = operands[1]->getLLValue(builder, VMap);
      if (Instruction *I = dyn_cast<Instruction>(ThenVal)) {
        I->setName("herbie.then_val");
      }

      Else->getParent()->setName("herbie.else");
      builder.SetInsertPoint(Else);
      Value *ElseVal = operands[2]->getLLValue(builder, VMap);
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
    for (auto operand : operands) {
      operandValues.push_back(operand->getLLValue(builder, VMap));
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
          Intrinsic::fma, {operandValues[0]->getType()},
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
      std::string msg = "FPNode getLLValue: Unexpected operator " + op;
      llvm_unreachable(msg.c_str());
    }

    return val;
  }
};

// Represents a true LLVM Value
class FPLLValue : public FPNode {
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
  bool input = false; // Whether `llvm::Value` is an input of an FPCC

public:
  Value *value;

  explicit FPLLValue(Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(NodeType::LLValue, op, dtype), value(value) {}

  bool hasSymbol() const override { return !symbol.empty(); }

  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap)
      override {
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

  Value *getLLValue(IRBuilder<> &builder,
                    const ValueToValueMapTy *VMap = nullptr) override {
    if (VMap) {
      assert(VMap->count(value) && "FPLLValue not found in passed-in VMap!");
      return VMap->lookup(value);
    }
    return value;
  }

  static bool classof(const FPNode *N) {
    return N->getType() == NodeType::LLValue;
  }
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
      : FPNode(NodeType::Const, "__const", dtype), strValue(strValue) {}

  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap)
      override {
    return strValue;
  }

  bool hasSymbol() const override {
    std::string msg = "Unexpected invocation of `hasSymbol` on an FPConst";
    llvm_unreachable(msg.c_str());
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

  virtual Value *getLLValue(IRBuilder<> &builder,
                            const ValueToValueMapTy *VMap = nullptr) override {
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

  static bool classof(const FPNode *N) {
    return N->getType() == NodeType::Const;
  }
};

void topoSort(const SetVector<Instruction *> &insts,
              SmallVectorImpl<Instruction *> &instsSorted) {
  SmallPtrSet<Instruction *, 8> visited;
  SmallPtrSet<Instruction *, 8> onStack;

  std::function<void(Instruction *)> dfsVisit = [&](Instruction *I) {
    if (visited.count(I))
      return;
    visited.insert(I);
    onStack.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
    for (auto &op : operands) {
      if (isa<Instruction>(op)) {
        Instruction *oI = cast<Instruction>(op);
        if (insts.contains(oI)) {
          if (onStack.count(oI)) {
            llvm_unreachable(
                "topoSort: Cycle detected in instruction dependencies!");
          }
          dfsVisit(oI);
        }
      }
    }

    onStack.erase(I);
    instsSorted.push_back(I);
  };

  for (auto *I : insts) {
    if (!visited.count(I)) {
      dfsVisit(I);
    }
  }

  llvm::reverse(instsSorted);
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
             funcName.startswith("hypot") || funcName.startswith("expm1") ||
             funcName.startswith("log1p");
      // llvm.fabs is deliberately excluded
    }
    return false;
  }
  default:
    return false;
  }
}

enum class PrecisionChangeType { BF16, FP16, FP32, FP64, FP80, FP128 };

unsigned getMPFRPrec(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return 8;
  case PrecisionChangeType::FP16:
    return 11;
  case PrecisionChangeType::FP32:
    return 24;
  case PrecisionChangeType::FP64:
    return 53;
  case PrecisionChangeType::FP80:
    return 64;
  case PrecisionChangeType::FP128:
    return 113;
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return Type::getBFloatTy(context);
  case PrecisionChangeType::FP16:
    return Type::getHalfTy(context);
  case PrecisionChangeType::FP32:
    return Type::getFloatTy(context);
  case PrecisionChangeType::FP64:
    return Type::getDoubleTy(context);
  case PrecisionChangeType::FP80:
    return Type::getX86_FP80Ty(context);
  case PrecisionChangeType::FP128:
    return Type::getFP128Ty(context);
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

PrecisionChangeType getPrecisionChangeType(Type *type) {
  if (type->isHalfTy()) {
    return PrecisionChangeType::BF16;
  } else if (type->isHalfTy()) {
    return PrecisionChangeType::FP16;
  } else if (type->isFloatTy()) {
    return PrecisionChangeType::FP32;
  } else if (type->isDoubleTy()) {
    return PrecisionChangeType::FP64;
  } else if (type->isX86_FP80Ty()) {
    return PrecisionChangeType::FP80;
  } else if (type->isFP128Ty()) {
    return PrecisionChangeType::FP128;
  } else {
    llvm_unreachable("Unsupported FP precision");
  }
}

StringRef getPrecisionChangeTypeString(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::BF16:
    return "BF16";
  case PrecisionChangeType::FP16:
    return "FP16";
  case PrecisionChangeType::FP32:
    return "FP32";
  case PrecisionChangeType::FP64:
    return "FP64";
  case PrecisionChangeType::FP80:
    return "FP80";
  case PrecisionChangeType::FP128:
    return "FP128";
  default:
    return "Unknown PT type";
  }
}

std::string getLibmFunctionForPrecision(StringRef funcName, Type *newType) {
  static const std::unordered_set<std::string> libmFunctions = {
      "sin",  "cos", "exp", "log",   "sqrt",  "tan",
      "cbrt", "pow", "fma", "hypot", "expm1", "log1p"};

  std::string baseName = funcName.str();
  if (baseName.back() == 'f' || baseName.back() == 'l') {
    baseName.pop_back();
  }

  if (libmFunctions.count(baseName)) {
    if (newType->isFloatTy()) {
      return baseName + "f";
    } else if (newType->isDoubleTy()) {
      return baseName;
    } else if (newType->isFP128Ty() || newType->isX86_FP80Ty()) {
      return baseName + "l";
    }
  }

  return "";
}

// Floating-Point Connected Component
struct FPCC {
  SetVector<Value *> inputs;
  SetVector<Instruction *> outputs;
  SetVector<Instruction *> operations;
  size_t outputs_rewritten = 0;

  FPCC() = default;
  explicit FPCC(SetVector<Value *> inputs, SetVector<Instruction *> outputs,
                SetVector<Instruction *> operations)
      : inputs(inputs), outputs(outputs), operations(operations) {}
};

struct PrecisionChange {
  SetVector<FPLLValue *>
      nodes; // Only nodes with existing `llvm::Value`s can be changed
  PrecisionChangeType oldType;
  PrecisionChangeType newType;

  explicit PrecisionChange(SetVector<FPLLValue *> &nodes,
                           PrecisionChangeType oldType,
                           PrecisionChangeType newType)
      : nodes(nodes), oldType(oldType), newType(newType) {}
};

void changePrecision(Instruction *I, PrecisionChange &change,
                     MapVector<Value *, Value *> &oldToNew) {
  if (!herbiable(*I)) {
    llvm_unreachable("Trying to tune an instruction is not herbiable");
  }

  IRBuilder<> Builder(I);
  Builder.setFastMathFlags(I->getFastMathFlags());
  Type *newType = getLLVMFPType(change.newType, I->getContext());
  Value *newI = nullptr;

  if (isa<UnaryOperator>(I) || isa<BinaryOperator>(I)) {
    // llvm::errs() << "PT Changing: " << *I << " to " << *newType << "\n";
    SmallVector<Value *, 2> newOps;
    for (auto &operand : I->operands()) {
      Value *newOp = nullptr;
      if (oldToNew.count(operand)) {
        newOp = oldToNew[operand];
      } else {
        newOp = Builder.CreateFPCast(operand, newType, "fpopt.fpcast");
        oldToNew[operand] = newOp;
      }
      newOps.push_back(newOp);
    }
    newI = Builder.CreateNAryOp(I->getOpcode(), newOps);
  } else if (auto *CI = dyn_cast<CallInst>(I)) {
    SmallVector<Value *, 4> newArgs;
    for (auto &arg : CI->args()) {
      Value *newArg = nullptr;
      if (oldToNew.count(arg)) {
        newArg = oldToNew[arg];
      } else {
        newArg = Builder.CreateFPCast(arg, newType, "fpopt.fpcast");
        oldToNew[arg] = newArg;
      }
      newArgs.push_back(newArg);
    }
    auto *calledFunc = CI->getCalledFunction();
    if (calledFunc && calledFunc->isIntrinsic()) {
      Intrinsic::ID intrinsicID = calledFunc->getIntrinsicID();
      if (intrinsicID != Intrinsic::not_intrinsic) {
        Function *newFunc =
            Intrinsic::getDeclaration(CI->getModule(), intrinsicID, {newType});
        newI = Builder.CreateCall(newFunc, newArgs);
      } else {
        llvm::errs() << "PT: Unknown intrinsic: " << *CI << "\n";
        llvm_unreachable("changePrecision: Unknown intrinsic call to change");
      }
    } else {
      StringRef funcName = calledFunc->getName();
      std::string newFuncName = getLibmFunctionForPrecision(funcName, newType);

      if (!newFuncName.empty()) {
        Module *M = CI->getModule();
        SmallVector<Type *, 4> newArgTypes(newArgs.size(), newType);

        FunctionCallee newFuncCallee = M->getOrInsertFunction(
            newFuncName, FunctionType::get(newType, newArgTypes, false));

        if (Function *newFunc = dyn_cast<Function>(newFuncCallee.getCallee())) {
          newI = Builder.CreateCall(newFunc, newArgs);
        } else {
          llvm::errs() << "PT: Failed to get "
                       << getPrecisionChangeTypeString(change.newType)
                       << " libm function for: " << *CI << "\n";
          llvm_unreachable("changePrecision: Failed to get libm function");
        }
      } else {
        llvm::errs() << "PT: Unknown function call: " << *CI << "\n";
        llvm_unreachable("changePrecision: Unknown function call to change");
      }
    }

  } else {
    llvm_unreachable("Unknown herbiable instruction");
  }

  oldToNew[I] = newI;
  // llvm::errs() << "PT Changing: " << *I << " to " << *newI << "\n";
}

struct PTCandidate {
  // Only one PT candidate per FPCC can be applied
  SmallVector<PrecisionChange, 1> changes;
  double accuracyCost;
  InstructionCost CompCost;
  std::string desc;
  std::unordered_map<FPNode *, double> perOutputAccCost;

  // TODO:
  explicit PTCandidate(SmallVector<PrecisionChange> &changes,
                       const Twine &desc = "")
      : changes(changes), desc(desc.str()) {}

  // If `VMap` is passed, map `llvm::Value`s in `component` to their cloned
  // values and change outputs in VMap to new casted outputs.
  void apply(FPCC &component, ValueToValueMapTy *VMap = nullptr) {
    SetVector<Instruction *> operations;
    ValueToValueMapTy clonedToOriginal; // Maps cloned outputs to old outputs
    if (VMap) {
      for (auto *I : component.operations) {
        assert(VMap->count(I));
        operations.insert(cast<Instruction>(VMap->lookup(I)));

        clonedToOriginal[VMap->lookup(I)] = I;
        // llvm::errs() << "Mapping back: " << *VMap->lookup(I) << " (in "
        //              << cast<Instruction>(VMap->lookup(I))
        //                     ->getParent()
        //                     ->getParent()
        //                     ->getName()
        //              << ") --> " << *I << " (in "
        //              << I->getParent()->getParent()->getName() << ")\n";
      }
    } else {
      operations = component.operations;
    }

    for (auto &change : changes) {
      SmallPtrSet<Instruction *, 8> seen;
      SmallVector<Instruction *, 8> todo;
      MapVector<Value *, Value *> oldToNew;

      SetVector<Instruction *> instsToChange;
      for (auto node : change.nodes) {
        assert(isa<Instruction>(node->value));
        auto *I = cast<Instruction>(node->value);
        if (VMap) {
          assert(VMap->count(I));
          I = cast<Instruction>(VMap->lookup(I));
        }
        if (!operations.contains(I)) {
          // Already erased by `AO.apply()`.
          continue;
        }
        instsToChange.insert(I);
      }

      SmallVector<Instruction *, 8> instsToChangeSorted;
      topoSort(instsToChange, instsToChangeSorted);

      for (auto *I : instsToChangeSorted) {
        changePrecision(I, change, oldToNew);
      }

      // Restore the precisions of the last level of instructions to be changed.
      // Clean up old instructions.
      for (auto &[oldV, newV] : oldToNew) {
        if (!isa<Instruction>(oldV)) {
          continue;
        }

        if (!instsToChange.contains(cast<Instruction>(oldV))) {
          continue;
        }

        SmallPtrSet<Instruction *, 8> users;
        for (auto *user : oldV->users()) {
          assert(
              isa<Instruction>(user) &&
              "PT: Unexpected non-instruction user of a changed instruction");
          if (!instsToChange.contains(cast<Instruction>(user))) {
            users.insert(cast<Instruction>(user));
          }
        }

        Value *casted = nullptr;
        if (!users.empty()) {
          IRBuilder<> builder(cast<Instruction>(oldV)->getParent(),
                              ++BasicBlock::iterator(cast<Instruction>(oldV)));
          casted = builder.CreateFPCast(
              newV, getLLVMFPType(change.oldType, builder.getContext()));

          if (VMap) {
            assert(VMap->count(clonedToOriginal[oldV]));
            (*VMap)[clonedToOriginal[oldV]] = casted;
          }
        }

        for (auto *user : users) {
          user->replaceUsesOfWith(oldV, casted);
        }

        // Assumes no external uses of the old value since all corresponding new
        // values are already restored to original precision and used to replace
        // uses of their old value. This is also advantageous to the solvers.
        for (auto *user : oldV->users()) {
          assert(instsToChange.contains(cast<Instruction>(user)) &&
                 "PT: Unexpected external user of a changed instruction");
        }

        if (!oldV->use_empty()) {
          oldV->replaceAllUsesWith(UndefValue::get(oldV->getType()));
        }

        cast<Instruction>(oldV)->eraseFromParent();

        // The change is being materialized to the original component
        if (!VMap)
          component.operations.remove(cast<Instruction>(oldV));
      }
    }
  }
};

class FPEvaluator {
  std::unordered_map<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;

public:
  FPEvaluator(PTCandidate *pt = nullptr) {
    if (pt) {
      for (const auto &change : pt->changes) {
        for (auto node : change.nodes) {
          nodePrecisions[node] = change.newType;
        }
      }
    }
  }

  PrecisionChangeType getNodePrecision(const FPNode *node) const {
    // If the node has a new precision from PT, use it
    PrecisionChangeType precType;

    auto it = nodePrecisions.find(node);
    if (it != nodePrecisions.end()) {
      precType = it->second;
    } else {
      // Otherwise, use the node's original precision
      if (node->dtype == "f32") {
        precType = PrecisionChangeType::FP32;
      } else if (node->dtype == "f64") {
        precType = PrecisionChangeType::FP64;
      } else {
        llvm_unreachable("Unsupported FP node precision type");
      }
    }

    if (precType != PrecisionChangeType::FP32 &&
        precType != PrecisionChangeType::FP64) {
      llvm_unreachable("Unsupported FP precision");
    }

    return precType;
  }

  void evaluateNode(const FPNode *node,
                    const SmallMapVector<Value *, double, 4> &inputValues) {
    if (cache.find(node) != cache.end())
      return;

    if (isa<FPConst>(node)) {
      double constVal = node->getLowerBound(); // TODO: Can be improved
      cache.emplace(node, constVal);
      return;
    }

    if (isa<FPLLValue>(node) &&
        inputValues.count(cast<FPLLValue>(node)->value)) {
      double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
      cache.emplace(node, inputValue);
      return;
    }

    if (node->op == "if") {
      evaluateNode(node->operands[0].get(), inputValues);
      double cond = getResult(node->operands[0].get());

      if (cond == 1.0) {
        evaluateNode(node->operands[1].get(), inputValues);
        double then_val = getResult(node->operands[1].get());
        cache.emplace(node, then_val);
      } else {
        evaluateNode(node->operands[2].get(), inputValues);
        double else_val = getResult(node->operands[2].get());
        cache.emplace(node, else_val);
      }
      return;
    }

    PrecisionChangeType nodePrec = getNodePrecision(node);

    for (const auto &operand : node->operands) {
      evaluateNode(operand.get(), inputValues);
    }

    double res = 0.0;

    if (node->op == "neg") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32) ? -static_cast<float>(op)
                                                    : -op;
    } else if (node->op == "+") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? static_cast<float>(op0) + static_cast<float>(op1)
                : op0 + op1;
    } else if (node->op == "-") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? static_cast<float>(op0) - static_cast<float>(op1)
                : op0 - op1;
    } else if (node->op == "*") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? static_cast<float>(op0) * static_cast<float>(op1)
                : op0 * op1;
    } else if (node->op == "/") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? static_cast<float>(op0) / static_cast<float>(op1)
                : op0 / op1;
    } else if (node->op == "sin") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::sin(static_cast<float>(op))
                : std::sin(op);
    } else if (node->op == "cos") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::cos(static_cast<float>(op))
                : std::cos(op);
    } else if (node->op == "tan") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::tan(static_cast<float>(op))
                : std::tan(op);
    } else if (node->op == "exp") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::exp(static_cast<float>(op))
                : std::exp(op);
    } else if (node->op == "expm1") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::expm1(static_cast<float>(op))
                : std::expm1(op);
    } else if (node->op == "log") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::log(static_cast<float>(op))
                : std::log(op);
    } else if (node->op == "log1p") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::log1p(static_cast<float>(op))
                : std::log1p(op);
    } else if (node->op == "sqrt") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::sqrt(static_cast<float>(op))
                : std::sqrt(op);
    } else if (node->op == "cbrt") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::cbrt(static_cast<float>(op))
                : std::cbrt(op);
    } else if (node->op == "pow") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::pow(static_cast<float>(op0), static_cast<float>(op1))
                : std::pow(op0, op1);
    } else if (node->op == "fabs") {
      double op = getResult(node->operands[0].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::fabs(static_cast<float>(op))
                : std::fabs(op);
    } else if (node->op == "fma") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      double op2 = getResult(node->operands[2].get());
      res = (nodePrec == PrecisionChangeType::FP32)
                ? std::fma(static_cast<float>(op0), static_cast<float>(op1),
                           static_cast<float>(op2))
                : std::fma(op0, op1, op2);
    } else if (node->op == "==") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) == static_cast<float>(op1)
                        : op0 == op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == "!=") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) != static_cast<float>(op1)
                        : op0 != op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == "<") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) < static_cast<float>(op1)
                        : op0 < op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == ">") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) > static_cast<float>(op1)
                        : op0 > op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == "<=") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) <= static_cast<float>(op1)
                        : op0 <= op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == ">=") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      bool result = (nodePrec == PrecisionChangeType::FP32)
                        ? static_cast<float>(op0) >= static_cast<float>(op1)
                        : op0 >= op1;
      res = result ? 1.0 : 0.0;
    } else if (node->op == "and") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (op0 == 1.0 && op1 == 1.0) ? 1.0 : 0.0;
    } else if (node->op == "or") {
      double op0 = getResult(node->operands[0].get());
      double op1 = getResult(node->operands[1].get());
      res = (op0 == 1.0 || op1 == 1.0) ? 1.0 : 0.0;
    } else if (node->op == "not") {
      double op = getResult(node->operands[0].get());
      res = (op == 1.0) ? 0.0 : 1.0;
    } else if (node->op == "TRUE") {
      res = 1.0;
    } else if (node->op == "FALSE") {
      res = 0.0;
    } else {
      std::string msg = "FPEvaluator: Unexpected operator " + node->op;
      llvm_unreachable(msg.c_str());
    }

    cache.emplace(node, res);
  }

  double getResult(const FPNode *node) const {
    auto it = cache.find(node);
    assert(it != cache.end() && "Node not evaluated yet");
    return it->second;
  }
};

// Emulate computation using native floating-point types
void getFPValues(ArrayRef<FPNode *> outputs,
                 const SmallMapVector<Value *, double, 4> &inputValues,
                 SmallVectorImpl<double> &results, PTCandidate *pt = nullptr) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  FPEvaluator evaluator(pt);

  for (const auto *output : outputs) {
    evaluator.evaluateNode(output, inputValues);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    results[i] = evaluator.getResult(outputs[i]);
  }
}

class MPFREvaluator {
  struct CachedValue {
    mpfr_t value;
    unsigned prec;

    CachedValue(unsigned prec) : prec(prec) {
      mpfr_init2(value, prec);
      mpfr_set_zero(value, 1);
    }

    CachedValue(const CachedValue &) = delete;
    CachedValue &operator=(const CachedValue &) = delete;

    CachedValue(CachedValue &&other) noexcept : prec(other.prec) {
      mpfr_init2(value, other.prec);
      mpfr_swap(value, other.value);
    }

    CachedValue &operator=(CachedValue &&other) noexcept {
      if (this != &other) {
        mpfr_set_prec(value, other.prec);
        prec = other.prec;
        mpfr_swap(value, other.value);
      }
      return *this;
    }

    virtual ~CachedValue() { mpfr_clear(value); }
  };

  std::unordered_map<const FPNode *, CachedValue> cache;
  unsigned prec; // Used only for ground truth evaluation
  std::unordered_map<const FPNode *, unsigned> nodeToNewPrec;

public:
  MPFREvaluator(unsigned prec, PTCandidate *pt = nullptr) : prec(prec) {
    if (pt) {
      for (const auto &change : pt->changes) {
        for (auto node : change.nodes) {
          nodeToNewPrec[node] = getMPFRPrec(change.newType);
        }
      }
    }
  }

  virtual ~MPFREvaluator() = default;

  unsigned getNodePrecision(const FPNode *node, bool groundTruth) const {
    // If trying to evaluate the ground truth, use the current MPFR precision
    if (groundTruth)
      return prec;

    // If the node has a new precision for PT, use it
    auto it = nodeToNewPrec.find(node);
    if (it != nodeToNewPrec.end()) {
      return it->second;
    }

    // Otherwise, use the original precision
    return node->getMPFRPrec();
  }

  // Compute the expression with MPFR at `prec` precision
  // recursively. When operand is a FPConst, use its lower
  // bound. When operand is a FPLLValue, get its inputs from
  // `inputs`.
  void evaluateNode(const FPNode *node,
                    const SmallMapVector<Value *, double, 4> &inputValues,
                    bool groundTruth) {
    if (isa<FPConst>(node)) {
      if (cache.find(node) != cache.end())
        return;

      double constVal = node->getLowerBound(); // TODO: Can be improved
      CachedValue cv(53);
      mpfr_set_d(cv.value, constVal, MPFR_RNDN);

      cache.emplace(node, std::move(cv));
      return;
    }

    if (isa<FPLLValue>(node) &&
        inputValues.count(cast<FPLLValue>(node)->value)) {
      if (cache.find(node) != cache.end())
        return;

      double inputValue = inputValues.lookup(cast<FPLLValue>(node)->value);
      // llvm::errs() << "Input value for " << *cast<FPLLValue>(node)->value
      //              << ": " << inputValue << "\n";
      CachedValue cv(53);
      mpfr_set_d(cv.value, inputValue, MPFR_RNDN);

      cache.emplace(node, std::move(cv));
      return;
    }

    // Type of results of if nodes depend on the evaluated branches
    if (node->op == "if") {
      if (cache.find(node) != cache.end())
        return;

      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &cond = getResult(node->operands[0].get());

      if (0 == mpfr_cmp_ui(cond, 1)) {
        evaluateNode(node->operands[1].get(), inputValues, groundTruth);
        mpfr_t &then_val = getResult(node->operands[1].get());
        cache.emplace(node,
                      CachedValue(cache.at(node->operands[1].get()).prec));
        mpfr_set(cache.at(node).value, then_val, MPFR_RNDN);
      } else {
        evaluateNode(node->operands[2].get(), inputValues, groundTruth);
        mpfr_t &else_val = getResult(node->operands[2].get());
        cache.emplace(node,
                      CachedValue(cache.at(node->operands[2].get()).prec));
        mpfr_set(cache.at(node).value, else_val, MPFR_RNDN);
      }
      return;
    }

    auto it = cache.find(node);

    unsigned nodePrec = getNodePrecision(node, groundTruth);
    // llvm::errs() << "Precision for " << node->op << " set to " << nodePrec
    //              << "\n";

    if (it != cache.end()) {
      assert(cache.at(node).prec == nodePrec && "Unexpected precision change");
      return;
    } else {
      // Prepare for recomputation
      cache.emplace(node, CachedValue(nodePrec));
    }

    mpfr_t &res = cache.at(node).value;

    // Cast operands to dest precision first -- this agrees with our
    // precision tuner behavior, i.e., fpcasting operands first
    if (node->op == "neg") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_neg(res, op, MPFR_RNDN);
    } else if (node->op == "+") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_add(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "-") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_sub(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "*") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_mul(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "/") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_div(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "sin") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_sin(res, op, MPFR_RNDN);
    } else if (node->op == "cos") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_cos(res, op, MPFR_RNDN);
    } else if (node->op == "tan") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_tan(res, op, MPFR_RNDN);
    } else if (node->op == "exp") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_exp(res, op, MPFR_RNDN);
    } else if (node->op == "expm1") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_expm1(res, op, MPFR_RNDN);
    } else if (node->op == "log") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_log(res, op, MPFR_RNDN);
    } else if (node->op == "log1p") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_log1p(res, op, MPFR_RNDN);
    } else if (node->op == "sqrt") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_sqrt(res, op, MPFR_RNDN);
    } else if (node->op == "cbrt") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_cbrt(res, op, MPFR_RNDN);
    } else if (node->op == "pow") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_pow(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "fma") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      evaluateNode(node->operands[2].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_t &op2 = getResult(node->operands[2].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op2, nodePrec, MPFR_RNDN);
      mpfr_fma(res, op0, op1, op2, MPFR_RNDN);
    } else if (node->op == "fabs") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_prec_round(op, nodePrec, MPFR_RNDN);
      mpfr_abs(res, op, MPFR_RNDN);
    } else if (node->op == "hypot") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_prec_round(op0, nodePrec, MPFR_RNDN);
      mpfr_prec_round(op1, nodePrec, MPFR_RNDN);
      mpfr_hypot(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "==") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 == mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "!=") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 != mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "<") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 > mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == ">") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 < mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "<=") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 >= mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == ">=") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 <= mpfr_cmp(op0, op1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "and") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 == mpfr_cmp_ui(op0, 1) && 0 == mpfr_cmp_ui(op1, 1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "or") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());

      if (0 == mpfr_cmp_ui(op0, 1) || 0 == mpfr_cmp_ui(op1, 1)) {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      }
    } else if (node->op == "not") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_set_prec(res, nodePrec);
      if (0 == mpfr_cmp_ui(op, 1)) {
        mpfr_set_ui(res, 0, MPFR_RNDN);
      } else {
        mpfr_set_ui(res, 1, MPFR_RNDN);
      }
    } else if (node->op == "TRUE") {
      mpfr_set_ui(res, 1, MPFR_RNDN);
    } else if (node->op == "FALSE") {
      mpfr_set_ui(res, 0, MPFR_RNDN);
    } else {
      std::string msg = "MPFREvaluator: Unexpected operator " + node->op;
      llvm_unreachable(msg.c_str());
    }

    // if (mpfr_nan_p(res)) {
    //   llvm::errs() << "WARNING MPFREvaluator: NaN detected for node "
    //                << node->op << "\n";
    //   llvm::errs() << "Problematic operand(s):";
    //   for (const auto &operand : node->operands) {
    //     mpfr_t &op = getResult(operand.get());
    //     llvm::errs() << " " << mpfr_get_d(op, MPFR_RNDN);
    //   }
    //   llvm::errs() << "\n";
    //   llvm::errs() << "Sampled input values:";
    //   for (const auto &[_, input] : inputValues) {
    //     llvm::errs() << " " << input;
    //   }
    //   llvm::errs() << "\n";
    // }
  }

  mpfr_t &getResult(FPNode *node) {
    assert(cache.count(node) > 0 &&
           "MPFREvaluator: Unexpected unevaluated node");
    return cache.at(node).value;
  }
};

// If looking for ground truth, compute a "correct" answer with MPFR.
//   For each sampled input configuration:
//     0. Ignore `FPNode.dtype`.
//     1. Compute the expression with MPFR at `prec` precision
//        by calling `MPFRValueHelper`. When operand is a FPConst, use its
//        lower bound. When operand is a FPLLValue, get its inputs from
//        `inputs`.
//     2. Dynamically extend precisions
//        until the first `groundTruthPrec` bits of significand don't change.
// Otherwise, compute the expression with MPFR at precisions specified within
// `FPNode`s or new precisions specified by `pt`.
void getMPFRValues(ArrayRef<FPNode *> outputs,
                   const SmallMapVector<Value *, double, 4> &inputValues,
                   SmallVectorImpl<double> &results, bool groundTruth = false,
                   const unsigned groundTruthPrec = 53,
                   PTCandidate *pt = nullptr) {
  assert(!outputs.empty());
  results.resize(outputs.size());

  if (!groundTruth) {
    MPFREvaluator evaluator(0, pt);
    // if (pt) {
    //   llvm::errs() << "getMPFRValues: PT candidate detected: " << pt->desc
    //                << "\n";
    // } else {
    //   llvm::errs() << "getMPFRValues: emulating original computation\n";
    // }

    for (const auto *output : outputs) {
      evaluator.evaluateNode(output, inputValues, false);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
    }
    return;
  }

  unsigned curPrec = 64;
  std::vector<mpfr_exp_t> prevResExp(outputs.size(), 0);
  std::vector<char *> prevResStr(outputs.size(), nullptr);
  std::vector<int> prevResSign(outputs.size(), 0);
  std::vector<bool> converged(outputs.size(), false);
  size_t numConverged = 0;

  while (true) {
    MPFREvaluator evaluator(curPrec, nullptr);

    // llvm::errs() << "getMPFRValues: computing ground truth with precision "
    //              << curPrec << "\n";

    for (const auto &output : outputs) {
      evaluator.evaluateNode(output, inputValues, true);
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (converged[i])
        continue;

      mpfr_t &res = evaluator.getResult(outputs[i]);
      int resSign = mpfr_sgn(res);
      mpfr_exp_t resExp;
      char *resStr =
          mpfr_get_str(nullptr, &resExp, 2, groundTruthPrec, res, MPFR_RNDN);

      if (prevResStr[i] != nullptr && resSign == prevResSign[i] &&
          resExp == prevResExp[i] && strcmp(resStr, prevResStr[i]) == 0) {
        converged[i] = true;
        numConverged++;
        mpfr_free_str(resStr);
        mpfr_free_str(prevResStr[i]);
        prevResStr[i] = nullptr;
        continue;
      }

      if (prevResStr[i]) {
        mpfr_free_str(prevResStr[i]);
      }
      prevResStr[i] = resStr;
      prevResExp[i] = resExp;
      prevResSign[i] = resSign;
    }

    if (numConverged == outputs.size()) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
      }
      break;
    }

    curPrec *= 2;

    if (curPrec > FPOptMaxMPFRPrec) {
      llvm::errs() << "getMPFRValues: MPFR precision limit reached for some "
                      "outputs, returning NaN\n";
      for (size_t i = 0; i < outputs.size(); ++i) {
        if (!converged[i]) {
          mpfr_free_str(prevResStr[i]);
          results[i] = std::numeric_limits<double>::quiet_NaN();
        } else {
          results[i] = mpfr_get_d(evaluator.getResult(outputs[i]), MPFR_RNDN);
        }
      }
      return;
    }
  }
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

void getSampledPoints(
    ArrayRef<Value *> inputs,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<SmallMapVector<Value *, double, 4>, 4> &sampledPoints) {
  std::default_random_engine gen;
  gen.seed(FPOptRandomSeed);
  std::uniform_real_distribution<> dis;

  SmallMapVector<Value *, SmallVector<double, 2>, 4> hypercube;
  for (const auto input : inputs) {
    const auto node = valueToNodeMap.at(input);

    double lower = node->getLowerBound();
    double upper = node->getUpperBound();

    hypercube.insert({input, {lower, upper}});
  }

  // llvm::errs() << "Hypercube:\n";
  // for (const auto &entry : hypercube) {
  //   Value *val = entry.first;
  //   double lower = entry.second[0];
  //   double upper = entry.second[1];
  //   llvm::errs() << valueToNodeMap.at(val)->symbol << ": [" << lower << ", "
  //                << upper << "]\n";
  // }

  // Sample `FPOptNumSamples` points from the hypercube. Store it in
  // `sampledPoints`.
  sampledPoints.clear();
  sampledPoints.resize(FPOptNumSamples);
  for (int i = 0; i < FPOptNumSamples; ++i) {
    SmallMapVector<Value *, double, 4> point;
    for (const auto &entry : hypercube) {
      Value *val = entry.first;
      double lower = entry.second[0];
      double upper = entry.second[1];
      double sample = dis(gen, decltype(dis)::param_type{lower, upper});
      point.insert({val, sample});
    }
    sampledPoints[i] = point;
    // llvm::errs() << "Sample " << i << ":\n";
    // for (const auto &entry : point) {
    //   llvm::errs() << valueToNodeMap.at(entry.first)->symbol << ": "
    //                << entry.second << "\n";
    // }
  }
}

void getSampledPoints(
    const std::string &expr,
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap,
    SmallVector<SmallMapVector<Value *, double, 4>, 4> &sampledPoints) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SmallVector<Value *, 4> inputs;
  for (const auto &argStr : argStrSet) {
    inputs.push_back(symbolToValueMap.at(argStr));
  }

  getSampledPoints(inputs, valueToNodeMap, symbolToValueMap, sampledPoints);
}

std::shared_ptr<FPNode> parseHerbieExpr(
    const std::string &expr,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
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
    return std::make_shared<FPConst>(value, dtype);
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

  auto node = std::make_shared<FPNode>(op, dtype);

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

TargetTransformInfo::OperandValueKind getOperandValueKind(const Value *V) {
  if (isa<Constant>(V)) {
    assert(!isa<UndefValue>(V));
    return TargetTransformInfo::OK_UniformConstantValue;
  }
  return TargetTransformInfo::OK_AnyValue;
}

TargetTransformInfo::OperandValueProperties
getOperandValueProperties(const Value *V) {
  // TODO: Power of 2?
  return TargetTransformInfo::OP_None;
}

InstructionCost getInstructionCompCost(const Instruction *I,
                                       const TargetTransformInfo &TTI) {
  if (!FPOptCostModelPath.empty()) {
    static std::map<std::pair<std::string, std::string>, InstructionCost>
        CostModel;
    static bool Loaded = false;

    if (!Loaded) {
      std::ifstream CostFile(FPOptCostModelPath);
      if (!CostFile.is_open()) {
        std::string msg =
            "Cost model file could not be opened: " + FPOptCostModelPath;
        llvm_unreachable(msg.c_str());
      }

      std::string Line;
      while (std::getline(CostFile, Line)) {
        std::istringstream SS(Line);
        std::string OpcodeStr, PrecisionStr;
        std::string CostStr;

        if (!std::getline(SS, OpcodeStr, ',')) {
          std::string msg = "Unexpected line in custom cost model: " + Line;
          llvm_unreachable(msg.c_str());
        }
        if (!std::getline(SS, PrecisionStr, ',')) {
          std::string msg = "Unexpected line in custom cost model: " + Line;
          llvm_unreachable(msg.c_str());
        }
        if (!std::getline(SS, CostStr)) {
          std::string msg = "Unexpected line in custom cost model: " + Line;
          llvm_unreachable(msg.c_str());
        }
        // llvm::errs() << "Cost model: " << OpcodeStr << ", " << PrecisionStr
        //              << ", " << CostStr << "\n";

        CostModel[{OpcodeStr, PrecisionStr}] = std::stoi(CostStr);
      }

      Loaded = true;
    }

    std::string OpcodeName;
    switch (I->getOpcode()) {
    case Instruction::FNeg:
      OpcodeName = "fneg";
      break;
    case Instruction::FAdd:
      OpcodeName = "fadd";
      break;
    case Instruction::FSub:
      OpcodeName = "fsub";
      break;
    case Instruction::FMul:
      OpcodeName = "fmul";
      break;
    case Instruction::FDiv:
      OpcodeName = "fdiv";
      break;
    case Instruction::FCmp:
      OpcodeName = "fcmp";
      break;
    case Instruction::FPExt:
      OpcodeName = "fpext";
      break;
    case Instruction::FPTrunc:
      OpcodeName = "fptrunc";
      break;
    case Instruction::PHI:
      return 0;
    case Instruction::Call: {
      auto *Call = cast<CallInst>(I);
      if (auto CalledFunc = Call->getCalledFunction()) {
        if (CalledFunc->isIntrinsic()) {
          switch (CalledFunc->getIntrinsicID()) {
          case Intrinsic::sin:
            OpcodeName = "sin";
            break;
          case Intrinsic::cos:
            OpcodeName = "cos";
            break;
          case Intrinsic::exp:
            OpcodeName = "exp";
            break;
          case Intrinsic::log:
            OpcodeName = "log";
            break;
          case Intrinsic::sqrt:
            OpcodeName = "sqrt";
            break;
          case Intrinsic::fabs:
            OpcodeName = "fabs";
            break;
          case Intrinsic::fma:
            OpcodeName = "fma";
            break;
          case Intrinsic::pow:
            OpcodeName = "pow";
            break;
          default: {
            std::string msg = "Custom cost model: unsupported intrinsic " +
                              CalledFunc->getName().str();
            llvm_unreachable(msg.c_str());
          }
          }
        } else {
          std::string FuncName = CalledFunc->getName().str();
          if (FuncName.back() == 'f' || FuncName.back() == 'l') {
            FuncName.pop_back();
          }

          if (FuncName == "sin") {
            OpcodeName = "sin";
          } else if (FuncName == "cos") {
            OpcodeName = "cos";
          } else if (FuncName == "tan") {
            OpcodeName = "tan";
          } else if (FuncName == "exp") {
            OpcodeName = "exp";
          } else if (FuncName == "log") {
            OpcodeName = "log";
          } else if (FuncName == "sqrt") {
            OpcodeName = "sqrt";
          } else if (FuncName == "expm1") {
            OpcodeName = "expm1";
          } else if (FuncName == "log1p") {
            OpcodeName = "log1p";
          } else if (FuncName == "cbrt") {
            OpcodeName = "cbrt";
          } else if (FuncName == "pow") {
            OpcodeName = "pow";
          } else if (FuncName == "fabs") {
            OpcodeName = "fabs";
          } else if (FuncName == "fma") {
            OpcodeName = "fma";
          } else if (FuncName == "hypot") {
            OpcodeName = "hypot";
          } else {
            std::string msg =
                "Custom cost model: unknown function call " + FuncName;
            llvm_unreachable(msg.c_str());
          }
        }
      } else {
        llvm_unreachable("Custom cost model: unknown function call");
      }
      break;
    }
    default:
      std::string msg = "Custom cost model: unexpected opcode " +
                        std::string(I->getOpcodeName());
      llvm_unreachable(msg.c_str());
    }

    std::string PrecisionName;
    Type *Ty = I->getType();
    if (I->getOpcode() == Instruction::FCmp) {
      Ty = I->getOperand(0)->getType();
    }

    if (Ty->isBFloatTy()) {
      PrecisionName = "bf16";
    } else if (Ty->isHalfTy()) {
      PrecisionName = "half";
    } else if (Ty->isFloatTy()) {
      PrecisionName = "float";
    } else if (Ty->isDoubleTy()) {
      PrecisionName = "double";
    } else if (Ty->isX86_FP80Ty()) {
      PrecisionName = "fp80";
    } else if (Ty->isFP128Ty()) {
      PrecisionName = "fp128";
    } else {
      std::string msg = "Custom cost model: unsupported precision type!";
      llvm_unreachable(msg.c_str());
    }

    if (I->getOpcode() == Instruction::FPExt ||
        I->getOpcode() == Instruction::FPTrunc) {
      Type *SrcTy = I->getOperand(0)->getType();
      std::string SrcPrecisionName;

      if (SrcTy->isBFloatTy()) {
        SrcPrecisionName = "bf16";
      } else if (SrcTy->isHalfTy()) {
        SrcPrecisionName = "half";
      } else if (SrcTy->isFloatTy()) {
        SrcPrecisionName = "float";
      } else if (SrcTy->isDoubleTy()) {
        SrcPrecisionName = "double";
      } else if (SrcTy->isX86_FP80Ty()) {
        SrcPrecisionName = "fp80";
      } else if (SrcTy->isFP128Ty()) {
        SrcPrecisionName = "fp128";
      } else {
        std::string msg = "Custom cost model: unsupported precision type!";
        llvm_unreachable(msg.c_str());
      }

      OpcodeName += "_" + SrcPrecisionName + "_to_" + PrecisionName;
      PrecisionName = SrcPrecisionName;
    }

    auto Key = std::make_pair(OpcodeName, PrecisionName);
    auto It = CostModel.find(Key);
    if (It != CostModel.end()) {
      return It->second;
    }

    std::string msg = "Custom cost model: entry not found for " + OpcodeName +
                      " @ " + PrecisionName;
    llvm::errs() << "Unexpected Intruction: " << *I << "\n";
    llvm_unreachable(msg.c_str());
  }

  unsigned Opcode = I->getOpcode();
  switch (Opcode) {
  case Instruction::FNeg: {
    SmallVector<const Value *, 1> Args(I->operands());
    return TTI.getArithmeticInstrCost(
        Opcode, I->getType(), TargetTransformInfo::TCK_Latency,
        getOperandValueKind(I->getOperand(0)), TargetTransformInfo::OK_AnyValue,
        getOperandValueProperties(I->getOperand(0)),
        TargetTransformInfo::OP_None, Args, I);
  }
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv: {
    SmallVector<const Value *, 2> Args(I->operands());
    return TTI.getArithmeticInstrCost(
        Opcode, I->getType(), TargetTransformInfo::TCK_Latency,
        getOperandValueKind(I->getOperand(0)),
        getOperandValueKind(I->getOperand(1)),
        getOperandValueProperties(I->getOperand(0)),
        getOperandValueProperties(I->getOperand(1)), Args, I);
  }
  case Instruction::FCmp: {
    const auto *FCI = cast<FCmpInst>(I);
    return TTI.getCmpSelInstrCost(Opcode, FCI->getType(), /* CondTy */ nullptr,
                                  FCI->getPredicate(),
                                  TargetTransformInfo::TCK_Latency, I);
  }
  case Instruction::PHI: {
    return TTI.getInstructionCost(I, TargetTransformInfo::TCK_Latency);
  }
  default: {
    if (const auto *Call = dyn_cast<CallInst>(I)) {
      if (Function *CalledFunc = Call->getCalledFunction()) {
        if (CalledFunc->isIntrinsic()) {
          auto IID = CalledFunc->getIntrinsicID();
          SmallVector<Type *, 4> OperandTypes;
          SmallVector<const Value *, 4> Args;
          for (auto &Arg : Call->args()) {
            OperandTypes.push_back(Arg->getType());
            Args.push_back(Arg.get());
          }

          IntrinsicCostAttributes ICA(IID, Call->getType(), Args, OperandTypes,
                                      Call->getFastMathFlags(),
                                      cast<IntrinsicInst>(I));
          return TTI.getIntrinsicInstrCost(ICA,
                                           TargetTransformInfo::TCK_Latency);
        } else {
          SmallVector<Type *, 4> ArgTypes;
          for (auto &Arg : Call->args())
            ArgTypes.push_back(Arg->getType());

          return TTI.getCallInstrCost(CalledFunc, Call->getType(), ArgTypes,
                                      TargetTransformInfo::TCK_Latency);
        }
      }
    }
    llvm::errs() << "WARNING: Using default cost for " << *I << "\n";
    return TTI.getInstructionCost(I, TargetTransformInfo::TCK_Latency);
  }
  }
}

InstructionCost computeMaxCost(
    BasicBlock *BB, std::unordered_map<BasicBlock *, InstructionCost> &MaxCost,
    std::unordered_set<BasicBlock *> &Visited, const TargetTransformInfo &TTI) {
  if (MaxCost.find(BB) != MaxCost.end())
    return MaxCost[BB];

  if (!Visited.insert(BB).second)
    return 0;

  InstructionCost BBCost = 0;
  for (const Instruction &I : *BB) {
    if (I.isTerminator())
      continue;

    auto instCost = getInstructionCompCost(&I, TTI);

    if (EnzymePrintFPOpt)
      // llvm::errs() << "Cost of " << I << " is: " << instCost << "\n";

      BBCost += instCost;
  }

  InstructionCost succCost = 0;

  if (!succ_empty(BB)) {
    InstructionCost maxSuccCost = 0;
    for (BasicBlock *Succ : successors(BB)) {
      InstructionCost succBBCost = computeMaxCost(Succ, MaxCost, Visited, TTI);
      if (succBBCost > maxSuccCost)
        maxSuccCost = succBBCost;
    }
    // llvm::errs() << "Max succ cost: " << maxSuccCost << "\n";
    succCost = maxSuccCost;
  }

  InstructionCost totalCost = BBCost + succCost;
  // llvm::errs() << "BB " << BB->getName() << " cost: " << totalCost << "\n";
  MaxCost[BB] = totalCost;
  Visited.erase(BB);
  return totalCost;
}

InstructionCost getCompCost(Function *F, const TargetTransformInfo &TTI) {
  std::unordered_map<BasicBlock *, InstructionCost> MaxCost;
  std::unordered_set<BasicBlock *> Visited;

  BasicBlock *EntryBB = &F->getEntryBlock();
  InstructionCost TotalCost = computeMaxCost(EntryBB, MaxCost, Visited, TTI);
  // llvm::errs() << "Total cost: " << TotalCost << "\n";
  return TotalCost;
}

// Sum up the cost of `output` and its FP operands recursively up to `inputs`
// (exclusive).
InstructionCost getCompCost(const SmallVector<Value *> &outputs,
                            const SetVector<Value *> &inputs,
                            const TargetTransformInfo &TTI) {
  assert(!outputs.empty());
  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;
  InstructionCost cost = 0;

  todo.insert(todo.end(), outputs.begin(), outputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (inputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      // TODO: unfair to ignore branches when calculating cost
      auto instCost = getInstructionCompCost(I, TTI);

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

InstructionCost getCompCost(
    const std::string &expr, Module *M, const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap,
    const FastMathFlags &FMF) {
  // llvm::errs() << "Evaluating cost of " << expr << "\n";
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  SmallVector<Type *, 8> argTypes;
  SmallVector<std::string, 8> argNames;
  for (const auto &argStr : argStrSet) {
    Value *argValue = symbolToValueMap[argStr];
    args.insert(argValue);
    argTypes.push_back(argValue->getType());
    argNames.push_back(argStr);
  }

  auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);

  // Materialize the expression in a temporary function
  FunctionType *FT =
      FunctionType::get(Type::getVoidTy(M->getContext()), argTypes, false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "tempFunc", M);

  ValueToValueMapTy VMap;
  Function::arg_iterator AI = tempFunction->arg_begin();
  for (const auto &argStr : argNames) {
    VMap[symbolToValueMap[argStr]] = &*AI;
    ++AI;
  }

  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);
  Instruction *ReturnInst = ReturnInst::Create(M->getContext(), entry);

  IRBuilder<> builder(ReturnInst);

  builder.setFastMathFlags(FMF);
  parsedNode->getLLValue(builder, &VMap);

  InstructionCost cost = getCompCost(tempFunction, TTI);

  tempFunction->eraseFromParent();
  return cost;
}

InstructionCost getCompCost(FPCC &component, const TargetTransformInfo &TTI,
                            PTCandidate &pt) {
  assert(!component.outputs.empty());

  InstructionCost cost = 0;

  Function *F = cast<Instruction>(component.outputs[0])->getFunction();

  ValueToValueMapTy VMap;
  Function *FClone = CloneFunction(F, VMap);
  FClone->setName(F->getName() + "_clone");

  pt.apply(component, &VMap);
  // output values in VMap are changed to the new casted values
  // FClone->print(llvm::errs());

  SmallPtrSet<Value *, 8> clonedInputs;
  for (auto &input : component.inputs) {
    clonedInputs.insert(VMap[input]);
  }

  SmallPtrSet<Value *, 8> clonedOutputs;
  for (auto &output : component.outputs) {
    clonedOutputs.insert(VMap[output]);
  }

  SmallPtrSet<Value *, 8> seen;
  SmallVector<Value *, 8> todo;

  todo.insert(todo.end(), clonedOutputs.begin(), clonedOutputs.end());
  while (!todo.empty()) {
    auto cur = todo.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    if (clonedInputs.contains(cur))
      continue;

    if (auto *I = dyn_cast<Instruction>(cur)) {
      auto instCost = getInstructionCompCost(I, TTI);
      // llvm::errs() << "Cost of " << *I << " is: " << instCost << "\n";

      cost += instCost;

      auto operands =
          isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
      for (auto &operand : operands) {
        todo.push_back(operand);
      }
    }
  }

  // llvm::errs() << "DEBUG: " << pt.desc << "\n";
  // FClone->print(llvm::errs());

  FClone->eraseFromParent();

  return cost;
}

struct RewriteCandidate {
  // Only one rewrite candidate per output `llvm::Value` can be applied
  InstructionCost CompCost;
  double herbieCost; // Unused for now
  double herbieAccuracy;
  double accuracyCost;
  std::string expr;

  RewriteCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), herbieAccuracy(accuracy), expr(expression) {}
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

  // llvm::errs() << "Shortest distances:\n";
  // for (auto &[op, dist] : shortestDistances) {
  //   llvm::errs() << *op << ": " << dist << "\n";
  // }

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

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSetImpl<Instruction *> &exprInsts,
                      SmallPtrSetImpl<Value *> &visited) {
  if (!V || inputs.contains(V) || visited.contains(V)) {
    return;
  }

  visited.insert(V);

  if (auto *I = dyn_cast<Instruction>(V)) {
    exprInsts.insert(I);

    auto operands =
        isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();

    for (auto &op : operands) {
      collectExprInsts(op, inputs, exprInsts, visited);
    }
  }
}

class ApplicableOutput;
class ApplicableFPCC;

struct SolutionStep {
  std::variant<ApplicableOutput *, ApplicableFPCC *> item;
  size_t candidateIndex;

  SolutionStep(ApplicableOutput *ao_, size_t idx)
      : item(ao_), candidateIndex(idx) {}

  SolutionStep(ApplicableFPCC *acc_, size_t idx)
      : item(acc_), candidateIndex(idx) {}
};

class ApplicableOutput {
public:
  FPCC *component;
  Value *oldOutput;
  std::string expr;
  double grad;
  unsigned executions;
  const TargetTransformInfo &TTI;
  double initialAccCost;           // Requires manual initialization
  InstructionCost initialCompCost; // Requires manual initialization
  double initialHerbieCost;        // Requires manual initialization
  double initialHerbieAccuracy;    // Requires manual initialization
  SmallVector<RewriteCandidate> candidates;
  SmallPtrSet<Instruction *, 8> erasableInsts;

  explicit ApplicableOutput(FPCC &component, Value *oldOutput, std::string expr,
                            double grad, unsigned executions,
                            const TargetTransformInfo &TTI)
      : component(&component), oldOutput(oldOutput), expr(expr), grad(grad),
        executions(executions), TTI(TTI) {
    initialCompCost = getCompCost({oldOutput}, component.inputs, TTI);
    findErasableInstructions();
  }

  void
  apply(size_t candidateIndex,
        std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
        std::unordered_map<std::string, Value *> &symbolToValueMap) {
    // 4) parse the output string solution from herbieland
    // 5) convert into a solution in llvm vals/instructions

    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Parsing Herbie output: " << herbieOutput << "\n";
    auto parsedNode = parseHerbieExpr(candidates[candidateIndex].expr,
                                      valueToNodeMap, symbolToValueMap);
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Parsed Herbie output: "
    //                << parsedNode->toFullExpression(valueToNodeMap) << "\n";

    Instruction *insertBefore = dyn_cast<Instruction>(oldOutput);
    IRBuilder<> builder(insertBefore);
    builder.setFastMathFlags(insertBefore->getFastMathFlags());

    Value *newOutput = parsedNode->getLLValue(builder);
    assert(newOutput && "Failed to get value from parsed node");

    if (EnzymePrintFPOpt)
      llvm::errs() << "Applying Herbie rewrite (#" << candidateIndex
                   << "): " << expr << "\n --> "
                   << candidates[candidateIndex].expr << "\n";

    oldOutput->replaceAllUsesWith(newOutput);
    symbolToValueMap[valueToNodeMap[oldOutput]->symbol] = newOutput;
    valueToNodeMap[newOutput] = std::make_shared<FPLLValue>(
        newOutput, "__no", valueToNodeMap[oldOutput]->dtype);

    for (auto *I : erasableInsts) {
      if (!I->use_empty())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
      component->operations.remove(I); // Avoid a second removal
    }

    component->outputs_rewritten++;
  }

  // Lower is better
  InstructionCost getCompCostDelta(size_t candidateIndex) {
    InstructionCost erasableCost = 0;

    for (auto *I : erasableInsts) {
      erasableCost += getInstructionCompCost(I, TTI);
    }

    return (candidates[candidateIndex].CompCost - erasableCost) * executions;
  }

  // Lower is better
  double getAccCostDelta(size_t candidateIndex) {
    return candidates[candidateIndex].accuracyCost - initialAccCost;
  }

  void findErasableInstructions() {
    SmallPtrSet<Value *, 8> visited;
    SmallPtrSet<Instruction *, 8> exprInsts;
    collectExprInsts(oldOutput, component->inputs, exprInsts, visited);
    visited.clear();

    SetVector<Instruction *> instsToProcess(exprInsts.begin(), exprInsts.end());

    SmallVector<Instruction *, 8> instsToProcessSorted;
    topoSort(instsToProcess, instsToProcessSorted);

    // `oldOutput` is trivially erasable
    erasableInsts.clear();
    erasableInsts.insert(cast<Instruction>(oldOutput));

    for (auto *I : reverse(instsToProcessSorted)) {
      if (erasableInsts.contains(I))
        continue;

      bool usedOutside = false;
      for (auto user : I->users()) {
        if (auto *userI = dyn_cast<Instruction>(user)) {
          if (erasableInsts.contains(userI)) {
            continue;
          }
        }
        // If the user is not an intruction or the user instruction is not an
        // erasable instruction, then the current instruction is not erasable
        // llvm::errs() << "Can't erase " << *I << " because of " << *user <<
        // "\n";
        usedOutside = true;
        break;
      }

      if (!usedOutside) {
        erasableInsts.insert(I);
      }
    }

    // llvm::errs() << "Erasable instructions:\n";
    // for (auto *I : erasableInsts) {
    //   llvm::errs() << *I << "\n";
    // }
    // llvm::errs() << "End of erasable instructions\n";
  }
};

void setUnifiedAccuracyCost(
    ApplicableFPCC &ACC,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap);

class ApplicableFPCC {
public:
  FPCC *component;
  const TargetTransformInfo &TTI;
  double initialAccCost; // Requires manual initialization
  InstructionCost initialCompCost;
  unsigned executions; // Requires manual initialization
  std::unordered_map<FPNode *, double> perOutputInitialAccCost;

  SmallVector<PTCandidate, 8> candidates;

  // Caches for adjusted cost calculations
  using ApplicableOutputSet = std::set<ApplicableOutput *>;
  struct CacheKey {
    size_t candidateIndex;
    ApplicableOutputSet applicableOutputs;

    bool operator==(const CacheKey &other) const {
      return candidateIndex == other.candidateIndex &&
             applicableOutputs == other.applicableOutputs;
    }
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey &key) const {
      std::size_t seed = std::hash<size_t>{}(key.candidateIndex);
      for (const auto *ao : key.applicableOutputs) {
        seed ^= std::hash<const ApplicableOutput *>{}(ao) + 0x9e3779b9 +
                (seed << 6) + (seed >> 2);
      }
      return seed;
    }
  };

  std::unordered_map<CacheKey, InstructionCost, CacheKeyHash>
      compCostDeltaCache;
  std::unordered_map<CacheKey, double, CacheKeyHash> accCostDeltaCache;

  explicit ApplicableFPCC(FPCC &fpcc, const TargetTransformInfo &TTI)
      : component(&fpcc), TTI(TTI) {
    initialCompCost =
        getCompCost({component->outputs.begin(), component->outputs.end()},
                    component->inputs, TTI);
  }

  void apply(size_t candidateIndex) {
    if (candidateIndex >= candidates.size()) {
      llvm_unreachable("Invalid candidate index");
    }

    // Traverse all the instructions to be changed precisions in a
    // topological order with respect to operand dependencies. Insert FP casts
    // between llvm::Value inputs and first level of instructions to be changed.
    // Restore precisions of the last level of instructions to be changed.
    llvm::errs() << "Applying PT candidate #" << candidateIndex << ": "
                 << candidates[candidateIndex].desc << "\n";
    candidates[candidateIndex].apply(*component);
  }

  // Lower is better
  InstructionCost getCompCostDelta(size_t candidateIndex) {
    // TODO: adjust this based on erasured instructions
    return (candidates[candidateIndex].CompCost - initialCompCost) * executions;
  }

  // Lower is better
  double getAccCostDelta(size_t candidateIndex) {
    return candidates[candidateIndex].accuracyCost - initialAccCost;
  }

  InstructionCost
  getAdjustedCompCostDelta(size_t candidateIndex,
                           const SmallVectorImpl<SolutionStep> &steps) {
    ApplicableOutputSet applicableOutputs;
    for (const auto &step : steps) {
      if (auto *ptr = std::get_if<ApplicableOutput *>(&step.item)) {
        if ((*ptr)->component == component) {
          applicableOutputs.insert(*ptr);
        }
      }
    }

    CacheKey key{candidateIndex, applicableOutputs};

    auto cacheIt = compCostDeltaCache.find(key);
    if (cacheIt != compCostDeltaCache.end()) {
      return cacheIt->second;
    }

    FPCC newComponent = *this->component;

    for (auto &step : steps) {
      if (auto *ptr = std::get_if<ApplicableOutput *>(&step.item)) {
        const auto &AO = **ptr;
        if (AO.component == component) {
          // Eliminate erasadable instructions from the adjusted ACC
          newComponent.operations.remove_if(
              [&AO](Instruction *I) { return AO.erasableInsts.contains(I); });
          newComponent.outputs.remove(cast<Instruction>(AO.oldOutput));
        }
      }
    }

    // If all outputs are rewritten, then the adjusted ACC is empty
    if (newComponent.outputs.empty()) {
      compCostDeltaCache[key] = 0;
      return 0;
    }

    InstructionCost initialCompCost =
        getCompCost({newComponent.outputs.begin(), newComponent.outputs.end()},
                    newComponent.inputs, TTI);

    InstructionCost candidateCompCost =
        getCompCost(newComponent, TTI, candidates[candidateIndex]);

    InstructionCost adjustedCostDelta =
        (candidateCompCost - initialCompCost) * executions;

    compCostDeltaCache[key] = adjustedCostDelta;
    return adjustedCostDelta;
  }

  double getAdjustedAccCostDelta(
      size_t candidateIndex, SmallVectorImpl<SolutionStep> &steps,
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      std::unordered_map<std::string, Value *> &symbolToValueMap) {
    ApplicableOutputSet applicableOutputs;
    for (const auto &step : steps) {
      if (auto *ptr = std::get_if<ApplicableOutput *>(&step.item)) {
        if ((*ptr)->component == component) {
          applicableOutputs.insert(*ptr);
        }
      }
    }

    CacheKey key{candidateIndex, applicableOutputs};

    auto cacheIt = accCostDeltaCache.find(key);
    if (cacheIt != accCostDeltaCache.end()) {
      return cacheIt->second;
    }

    double totalCandidateAccCost = 0.0;
    double totalInitialAccCost = 0.0;

    // Collect erased output nodes
    SmallPtrSet<FPNode *, 8> stepNodes;
    for (const auto &step : steps) {
      if (auto *ptr = std::get_if<ApplicableOutput *>(&step.item)) {
        const auto &AO = **ptr;
        if (AO.component == component) {
          auto it = valueToNodeMap.find(AO.oldOutput);
          assert(it != valueToNodeMap.end() && it->second);
          stepNodes.insert(it->second.get());
        }
      }
    }

    // Iterate over all output nodes and sum costs for nodes not erased
    for (auto &[node, cost] : perOutputInitialAccCost) {
      if (!stepNodes.count(node)) {
        totalInitialAccCost += cost;
      }
    }

    for (auto &[node, cost] : candidates[candidateIndex].perOutputAccCost) {
      if (!stepNodes.count(node)) {
        totalCandidateAccCost += cost;
      }
    }

    double adjustedAccCostDelta = totalCandidateAccCost - totalInitialAccCost;

    accCostDeltaCache[key] = adjustedAccCostDelta;
    return adjustedAccCostDelta;
  }
};

void setUnifiedAccuracyCost(
    ApplicableOutput &AO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  SmallVector<SmallMapVector<Value *, double, 4>, 4> sampledPoints;
  getSampledPoints(AO.component->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints);

  SmallVector<double, 4> goldVals;
  goldVals.resize(FPOptNumSamples);
  double initAC = 0.;

  unsigned numValidSamples = 0;
  for (const auto &pair : enumerate(sampledPoints)) {
    ArrayRef<FPNode *> outputs = {valueToNodeMap[AO.oldOutput].get()};
    SmallVector<double, 1> results;
    getMPFRValues(outputs, pair.value(), results, true, 53);
    double goldVal = results[0];
    // llvm::errs() << "DEBUG AO gold value: " << goldVal << "\n";
    goldVals[pair.index()] = goldVal;

    getFPValues(outputs, pair.value(), results);
    double realVal = results[0];
    // llvm::errs() << "DEBUG AO real value: " << realVal << "\n";

    if (!std::isnan(goldVal) && !std::isnan(realVal)) {
      initAC += std::fabs(goldVal - realVal);
      numValidSamples++;
    }
  }

  AO.initialAccCost = initAC / numValidSamples * std::fabs(AO.grad);
  // llvm::errs() << "DEBUG calculated AO initial accuracy cost: "
  //              << AO.initialAccCost << "\n";
  assert(numValidSamples && "No valid samples for AO -- try increasing the "
                            "number of samples");
  assert(!std::isnan(AO.initialAccCost));

  for (auto &candidate : AO.candidates) {
    const auto &expr = candidate.expr;
    auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);
    double ac = 0.;

    numValidSamples = 0;
    for (const auto &pair : enumerate(sampledPoints)) {
      // Compute the "gold" value & real value for each sampled point
      // Compute an average of (difference * gradient)
      // TODO: Consider geometric average???
      assert(valueToNodeMap.count(AO.oldOutput));

      // llvm::errs() << "Computing real output for candidate: " << expr <<
      // "\n";

      // llvm::errs() << "Current input values:\n";
      // for (const auto &entry : pair.value()) {
      //   llvm::errs() << valueToNodeMap[entry.first]->symbol << ": "
      //                << entry.second << "\n";
      // }

      // llvm::errs() << "Gold value: " << goldVals[pair.index()] << "\n";

      ArrayRef<FPNode *> outputs = {parsedNode.get()};
      SmallVector<double, 1> results;
      getFPValues(outputs, pair.value(), results);
      double realVal = results[0];

      // llvm::errs() << "Real value: " << realVal << "\n";
      double goldVal = goldVals[pair.index()];
      if (!std::isnan(goldVal) && !std::isnan(realVal)) {
        ac += std::fabs(goldVal - realVal);
        numValidSamples++;
      }
    }
    assert(numValidSamples && "No valid samples for AO -- try increasing the "
                              "number of samples");
    candidate.accuracyCost = ac / numValidSamples * std::fabs(AO.grad);
    assert(!std::isnan(candidate.accuracyCost));
  }
}

void setUnifiedAccuracyCost(
    ApplicableFPCC &ACC,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  SmallVector<SmallMapVector<Value *, double, 4>, 4> sampledPoints;
  getSampledPoints(ACC.component->inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints);

  SmallMapVector<FPNode *, SmallVector<double, 4>, 4>
      goldVals; // output -> gold vals
  for (auto *output : ACC.component->outputs) {
    auto *node = valueToNodeMap[output].get();
    goldVals[node].resize(FPOptNumSamples);
    ACC.perOutputInitialAccCost[node] = 0.;
  }

  SmallVector<FPNode *, 4> outputs;
  for (auto *output : ACC.component->outputs) {
    outputs.push_back(valueToNodeMap[output].get());
  }

  std::unordered_map<FPNode *, unsigned> numValidSamplesPerOutput;
  for (auto *output : outputs) {
    numValidSamplesPerOutput[output] = 0;
  }

  for (const auto &pair : enumerate(sampledPoints)) {
    SmallVector<double, 8> results;

    // Get ground truth values for all outputs
    getMPFRValues(outputs, pair.value(), results, true, 53);
    for (const auto &[output, result] : zip(outputs, results)) {
      goldVals[output][pair.index()] = result;
      // llvm::errs() << "DEBUG ACC gold value: " << result << "\n";
    }

    // Emulate FPCC with parsed precision
    getFPValues(outputs, pair.value(), results);

    for (const auto &[output, result] : zip(outputs, results)) {
      // llvm::errs() << "DEBUG ACC real value: " << result << "\n";
      double goldVal = goldVals[output][pair.index()];
      if (!std::isnan(goldVal) && !std::isnan(result)) {
        double diff = std::fabs(goldVal - result);
        ACC.perOutputInitialAccCost[output] += diff;
        numValidSamplesPerOutput[output]++;
      }
    }
  }

  // Normalize accuracy costs and compute aggregated initialAccCost
  ACC.initialAccCost = 0.0;
  for (auto *output : outputs) {
    unsigned numValidSamples = numValidSamplesPerOutput[output];
    assert(numValidSamples && "No valid samples for at least one output node "
                              "-- try increasing the number of samples");
    ACC.perOutputInitialAccCost[output] /= numValidSamples;
    // Local error --> global error
    ACC.perOutputInitialAccCost[output] *= std::fabs(output->grad);
    // llvm::errs() << "DEBUG calculated ACC per output initial accuracy cost: "
    //              << ACC.perOutputInitialAccCost[output] << "\n";
    ACC.initialAccCost += ACC.perOutputInitialAccCost[output];
  }
  assert(!std::isnan(ACC.initialAccCost));

  // Compute accuracy costs for each PT candidate
  for (auto &candidate : ACC.candidates) {
    std::unordered_map<FPNode *, unsigned> numValidSamplesPerOutput;
    for (auto *output : outputs) {
      candidate.perOutputAccCost[output] = 0.;
      numValidSamplesPerOutput[output] = 0;
    }

    for (const auto &pair : enumerate(sampledPoints)) {
      SmallVector<double, 8> results;
      getFPValues(outputs, pair.value(), results, &candidate);

      for (const auto &[output, result] : zip(outputs, results)) {
        double goldVal = goldVals[output][pair.index()];
        if (!std::isnan(goldVal) && !std::isnan(result)) {
          double diff = std::fabs(goldVal - result);
          // Sum up local errors
          candidate.perOutputAccCost[output] += diff;
          numValidSamplesPerOutput[output]++;
        }
      }
    }

    // Normalize accuracy costs and compute aggregated accuracyCost
    candidate.accuracyCost = 0.0;
    for (auto *output : outputs) {
      unsigned numValidSamples = numValidSamplesPerOutput[output];
      assert(numValidSamples && "No valid samples for output -- try increasing "
                                "the number of samples");
      candidate.perOutputAccCost[output] /= numValidSamples;
      // Local error --> global error
      candidate.perOutputAccCost[output] *= std::fabs(output->grad);
      // llvm::errs()
      //     << "DEBUG calculated ACC per output candidate accuracy cost: "
      //     << candidate.perOutputAccCost[output] << "\n";
      candidate.accuracyCost += candidate.perOutputAccCost[output];
    }
    assert(!std::isnan(candidate.accuracyCost));
  }
}

bool improveViaHerbie(
    const std::string &inputExpr, ApplicableOutput &AO, Module *M,
    const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string Program = HERBIE_BINARY;
  llvm::errs() << "random seed: " << std::to_string(FPOptRandomSeed) << "\n";

  SmallVector<llvm::StringRef> BaseArgs = {
      Program,        "report",
      "--seed",       std::to_string(FPOptRandomSeed),
      "--timeout",    std::to_string(HerbieTimeout),
      "--num-points", std::to_string(HerbieNumPoints),
      "--num-iters",  std::to_string(HerbieNumIters)};

  BaseArgs.push_back("--disable");
  BaseArgs.push_back("generate:proofs");

  if (HerbieDisableNumerics) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("rules:numerics");
  }

  if (HerbieDisableSetupSimplify) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("setup:simplify");
  }

  if (HerbieDisableGenSimplify) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("generate:simplify");
  }

  if (HerbieDisableTaylor) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("generate:taylor");
  }

  if (HerbieDisableRegime) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:regimes");
  }

  if (HerbieDisableBranchExpr) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:branch-expressions");
  }

  if (HerbieDisableAvgError) {
    BaseArgs.push_back("--disable");
    BaseArgs.push_back("reduce:avg-error");
  }

  SmallVector<SmallVector<llvm::StringRef>> BaseArgsList;

  if (!HerbieDisableTaylor) {
    SmallVector<llvm::StringRef> Args1 = BaseArgs;
    BaseArgsList.push_back(Args1);

    SmallVector<llvm::StringRef> Args2 = BaseArgs;
    Args2.push_back("--disable");
    Args2.push_back("generate:taylor");
    BaseArgsList.push_back(Args2);
  }

  bool InitialValuesSet = false;

  for (const auto &BaseArgs : BaseArgsList) {
    SmallString<32> tmpin, tmpout;

    if (llvm::sys::fs::createUniqueFile("herbie_input_%%%%%%%%%%%%%%%%", tmpin,
                                        llvm::sys::fs::perms::owner_all)) {
      llvm::errs() << "Failed to create a unique input file.\n";
      continue;
    }

    if (llvm::sys::fs::createUniqueDirectory("herbie_output_%%%%%%%%%%%%%%%%",
                                             tmpout)) {
      llvm::errs() << "Failed to create a unique output directory.\n";
      llvm::sys::fs::remove(tmpin);
      continue;
    }

    std::ofstream input(tmpin.c_str());
    if (!input) {
      llvm::errs() << "Failed to open input file.\n";
      llvm::sys::fs::remove(tmpin);
      llvm::sys::fs::remove(tmpout);
      continue;
    }
    input << inputExpr;
    input.close();

    SmallVector<llvm::StringRef> Args = BaseArgs;
    Args.push_back(tmpin);
    Args.push_back(tmpout);

    std::string ErrMsg;
    bool ExecutionFailed = false;

    if (EnzymePrintFPOpt) {
      llvm::errs() << "Executing Herbie with arguments: ";
      for (const auto &arg : Args) {
        llvm::errs() << arg << " ";
      }
      llvm::errs() << "\n";
    }

    llvm::sys::ExecuteAndWait(Program, Args, /*Env=*/llvm::None,
                              /*Redirects=*/llvm::None,
                              /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg,
                              &ExecutionFailed);

    std::remove(tmpin.c_str());
    if (ExecutionFailed) {
      llvm::errs() << "Execution failed: " << ErrMsg << "\n";
      llvm::sys::fs::remove(tmpout);
      continue;
    }

    std::ifstream output((tmpout + "/results.json").str());
    if (!output) {
      llvm::errs() << "Failed to open output file.\n";
      llvm::sys::fs::remove(tmpout);
      continue;
    }
    std::string content((std::istreambuf_iterator<char>(output)),
                        std::istreambuf_iterator<char>());
    output.close();
    llvm::sys::fs::remove(tmpout.c_str());

    llvm::errs() << "Herbie output: " << content << "\n";

    Expected<json::Value> parsed = json::parse(content);
    if (!parsed) {
      llvm::errs() << "Failed to parse Herbie result!\n";
      continue;
    }

    json::Object *obj = parsed->getAsObject();
    json::Array &tests = *obj->getArray("tests");
    StringRef bestExpr = tests[0].getAsObject()->getString("output").getValue();

    if (bestExpr == "#f") {
      continue;
    }

    double bits = tests[0].getAsObject()->getNumber("bits").getValue();
    json::Array &costAccuracy =
        *tests[0].getAsObject()->getArray("cost-accuracy");

    json::Array &initial = *costAccuracy[0].getAsArray();
    double initialCostVal = initial[0].getAsNumber().getValue();
    double initialCost = 1.0;
    double initialAccuracy = 1.0 - initial[1].getAsNumber().getValue() / bits;

    if (!InitialValuesSet) {
      AO.initialHerbieCost = initialCost;
      AO.initialHerbieAccuracy = initialAccuracy;
      InitialValuesSet = true;
    }

    json::Array &best = *costAccuracy[1].getAsArray();
    double bestCost = best[0].getAsNumber().getValue() / initialCostVal;
    double bestAccuracy = 1.0 - best[1].getAsNumber().getValue() / bits;

    RewriteCandidate bestCandidate(bestCost, bestAccuracy, bestExpr.str());
    bestCandidate.CompCost =
        getCompCost(bestExpr.str(), M, TTI, valueToNodeMap, symbolToValueMap,
                    cast<Instruction>(AO.oldOutput)->getFastMathFlags());
    AO.candidates.push_back(bestCandidate);

    json::Array &alternatives = *costAccuracy[2].getAsArray();

    // Handle alternatives
    for (size_t i = 0; i < alternatives.size(); ++i) {
      json::Array &entry = *alternatives[i].getAsArray();
      double cost = entry[0].getAsNumber().getValue() / initialCostVal;
      double accuracy = 1.0 - entry[1].getAsNumber().getValue() / bits;
      StringRef expr = entry[2].getAsString().getValue();
      RewriteCandidate candidate(cost, accuracy, expr.str());
      candidate.CompCost =
          getCompCost(expr.str(), M, TTI, valueToNodeMap, symbolToValueMap,
                      cast<Instruction>(AO.oldOutput)->getFastMathFlags());
      AO.candidates.push_back(candidate);
    }
  }

  if (AO.candidates.empty()) {
    return false;
  }

  setUnifiedAccuracyCost(AO, valueToNodeMap, symbolToValueMap);
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

struct ValueInfo {
  double minRes;
  double maxRes;
  unsigned executions;
  double geometricAvg;
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
      std::string minResLine, maxResLine, executionsLine, geometricAvgLine;
      if (getline(file, minResLine) && getline(file, maxResLine) &&
          getline(file, executionsLine) && getline(file, geometricAvgLine)) {
        std::regex minResPattern(R"(MinRes = ([\d\.eE+-]+))");
        std::regex maxResPattern(R"(MaxRes = ([\d\.eE+-]+))");
        std::regex executionsPattern(R"(Executions = (\d+))");
        std::regex geometricAvgPattern(R"(Geometric Average = ([\d\.eE+-]+))");

        std::smatch minResMatch, maxResMatch, executionsMatch,
            geometricAvgMatch;
        if (std::regex_search(minResLine, minResMatch, minResPattern) &&
            std::regex_search(maxResLine, maxResMatch, maxResPattern) &&
            std::regex_search(executionsLine, executionsMatch,
                              executionsPattern) &&
            std::regex_search(geometricAvgLine, geometricAvgMatch,
                              geometricAvgPattern)) {
          data.minRes = stringToDouble(minResMatch[1]);
          data.maxRes = stringToDouble(maxResMatch[1]);
          data.executions = std::stol(executionsMatch[1]);
          data.geometricAvg = stringToDouble(geometricAvgMatch[1]);
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
    const std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    const std::unordered_map<std::string, Value *> &symbolToValueMap) {
  std::string preconditions;

  for (const auto &arg : args) {
    const auto node = valueToNodeMap.at(symbolToValueMap.at(arg));
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
    SmallVector<ApplicableOutput, 4> &AOs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
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
      auto candCompCost = AO.getCompCostDelta(i);
      auto candAccCost = AO.getAccCostDelta(i);
      llvm::errs() << "Candidate " << i << " for " << AO.expr
                   << " has accuracy cost: " << candAccCost
                   << " and computation cost: " << candCompCost << "\n";

      // See if the candidate fits within the computation cost budget
      if (totalComputationCost + candCompCost <= FPOptComputationCostBudget) {
        // Select the candidate with the lowest accuracy cost
        if (candAccCost < bestAccuracyCost) {
          llvm::errs() << "Candidate " << i << " selected!\n";
          bestCandidateIndex = i;
          bestAccuracyCost = candAccCost;
          bestCandidateComputationCost = candCompCost;
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
    SmallVector<ApplicableOutput, 4> &AOs, SmallVector<ApplicableFPCC, 4> &ACCs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap = std::map<InstructionCost, SmallVector<SolutionStep>>;

  CostMap costToAccuracyMap;
  costToAccuracyMap[0] = 0;
  SolutionMap costToSolutionMap;
  costToSolutionMap[0] = {};

  for (auto &AO : AOs) {
    CostMap newCostToAccuracyMap;
    SolutionMap newCostToSolutionMap;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost currCompCost = pair.first;
      double currAccCost = pair.second;

      // It is possible to apply zero candidate for an AO
      if (newCostToAccuracyMap.find(currCompCost) ==
              newCostToAccuracyMap.end() ||
          newCostToAccuracyMap[currCompCost] > currAccCost) {
        newCostToAccuracyMap[currCompCost] = currAccCost;
        newCostToSolutionMap[currCompCost] = costToSolutionMap[currCompCost];
      }

      for (auto &candidate : enumerate(AO.candidates)) {
        size_t i = candidate.index();
        auto candCompCost = AO.getCompCostDelta(i);
        auto candAccCost = AO.getAccCostDelta(i);

        InstructionCost newCompCost = currCompCost + candCompCost;
        double newAccCost = currAccCost + candAccCost;

        if (EnzymePrintFPOpt)
          llvm::errs() << "AO candidate " << i
                       << " has accuracy cost: " << candAccCost
                       << " and computation cost: " << candCompCost << "\n";

        if (newCostToAccuracyMap.find(newCompCost) ==
                newCostToAccuracyMap.end() ||
            newCostToAccuracyMap[newCompCost] > newAccCost) {
          newCostToAccuracyMap[newCompCost] = newAccCost;
          newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
          newCostToSolutionMap[newCompCost].emplace_back(&AO, i);
          if (EnzymePrintFPOpt)
            llvm::errs() << "Updating accuracy map (AO candidate " << i
                         << "): computation cost " << newCompCost
                         << " -> accuracy cost " << newAccCost << "\n";
        }
      }
    }

    // TODO: Do not prune AO parts of the DP table since AOs influence ACCs
    if (!FPOptEarlyPrune) {
      costToAccuracyMap.swap(newCostToAccuracyMap);
      costToSolutionMap.swap(newCostToSolutionMap);

      continue;
    }

    CostMap prunedCostToAccuracyMap;
    SolutionMap prunedCostToSolutionMap;

    for (const auto &l : newCostToAccuracyMap) {
      InstructionCost currCompCost = l.first;
      double currAccCost = l.second;

      bool dominated = false;
      for (const auto &r : newCostToAccuracyMap) {
        InstructionCost otherCompCost = r.first;
        double otherAccCost = r.second;

        if (currCompCost - otherCompCost >
                std::fabs(FPOptCostDominanceThreshold *
                          otherCompCost.getValue().getValue()) &&
            currAccCost - otherAccCost >
                std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "AO candidate with computation cost: "
                         << currCompCost
                         << " and accuracy cost: " << currAccCost
                         << " is dominated by candidate with computation cost:"
                         << otherCompCost
                         << " and accuracy cost: " << otherAccCost << "\n";
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        prunedCostToAccuracyMap[currCompCost] = currAccCost;
        prunedCostToSolutionMap[currCompCost] =
            newCostToSolutionMap[currCompCost];
      }
    }

    costToAccuracyMap.swap(prunedCostToAccuracyMap);
    costToSolutionMap.swap(prunedCostToSolutionMap);
  }

  for (auto &ACC : ACCs) {
    CostMap newCostToAccuracyMap;
    SolutionMap newCostToSolutionMap;

    for (const auto &pair : costToAccuracyMap) {
      InstructionCost currCompCost = pair.first;
      double currAccCost = pair.second;

      // It is possible to apply zero candidate for an ACC
      if (newCostToAccuracyMap.find(currCompCost) ==
              newCostToAccuracyMap.end() ||
          newCostToAccuracyMap[currCompCost] > currAccCost) {
        newCostToAccuracyMap[currCompCost] = currAccCost;
        newCostToSolutionMap[currCompCost] = costToSolutionMap[currCompCost];
      }

      for (auto &candidate : enumerate(ACC.candidates)) {
        size_t i = candidate.index();
        auto candCompCost =
            ACC.getAdjustedCompCostDelta(i, costToSolutionMap[currCompCost]);
        auto candAccCost =
            ACC.getAdjustedAccCostDelta(i, costToSolutionMap[currCompCost],
                                        valueToNodeMap, symbolToValueMap);

        InstructionCost newCompCost = currCompCost + candCompCost;
        double newAccCost = currAccCost + candAccCost;

        // if (EnzymePrintFPOpt)
        llvm::errs() << "ACC candidate " << i << " (" << candidate.value().desc
                     << ") has accuracy cost: " << candAccCost
                     << " and computation cost: " << candCompCost << "\n";

        if (newCostToAccuracyMap.find(newCompCost) ==
                newCostToAccuracyMap.end() ||
            newCostToAccuracyMap[newCompCost] > newAccCost) {
          newCostToAccuracyMap[newCompCost] = newAccCost;
          newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
          newCostToSolutionMap[newCompCost].emplace_back(&ACC, i);
          if (EnzymePrintFPOpt)
            llvm::errs() << "Updating accuracy map (ACC candidate " << i
                         << "): computation cost " << newCompCost
                         << " -> accuracy cost " << newAccCost << "\n";
        }
      }
    }

    CostMap prunedCostToAccuracyMap;
    SolutionMap prunedCostToSolutionMap;

    for (const auto &l : newCostToAccuracyMap) {
      InstructionCost currCompCost = l.first;
      double currAccCost = l.second;

      bool dominated = false;
      for (const auto &r : newCostToAccuracyMap) {
        InstructionCost otherCompCost = r.first;
        double otherAccCost = r.second;

        if (currCompCost - otherCompCost >
                std::fabs(FPOptCostDominanceThreshold *
                          otherCompCost.getValue().getValue()) &&
            currAccCost - otherAccCost >
                std::fabs(FPOptAccuracyDominanceThreshold * otherAccCost)) {
          if (EnzymePrintFPOpt)
            llvm::errs() << "ACC candidate with computation cost: "
                         << currCompCost
                         << " and accuracy cost: " << currAccCost
                         << " is dominated by candidate with computation cost:"
                         << otherCompCost
                         << " and accuracy cost: " << otherAccCost << "\n";
          dominated = true;
          break;
        }
      }

      if (!dominated) {
        prunedCostToAccuracyMap[currCompCost] = currAccCost;
        prunedCostToSolutionMap[currCompCost] =
            newCostToSolutionMap[currCompCost];
      }
    }

    costToAccuracyMap.swap(prunedCostToAccuracyMap);
    costToSolutionMap.swap(prunedCostToSolutionMap);
  }

  llvm::errs() << "\n*** DP Table ***\n";
  for (const auto &pair : costToAccuracyMap) {
    llvm::errs() << "Computation cost: " << pair.first
                 << ", Accuracy cost: " << pair.second << "\n";
    llvm::errs() << "\tSolution steps: \n";
    for (const auto &step : costToSolutionMap[pair.first]) {
      std::visit(
          [&](auto *item) {
            using T = std::decay_t<decltype(*item)>;
            if constexpr (std::is_same_v<T, ApplicableOutput>) {
              llvm::errs() << "\t\t" << item->expr << " --("
                           << step.candidateIndex << ")-> "
                           << item->candidates[step.candidateIndex].expr
                           << "\n";
            } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
              llvm::errs() << "\t\tACC: "
                           << item->candidates[step.candidateIndex].desc
                           << " (#" << step.candidateIndex << ")\n";
            } else {
              llvm_unreachable(
                  "accuracyDPSolver: Unexpected type of solution step");
            }
          },
          step.item);
    }
  }
  llvm::errs() << "*** End of DP Table ***\n\n";
  llvm::errs() << "*** Critical Computation Costs ***\n";
  // Just print all computation costs in the DP table
  for (const auto &pair : costToAccuracyMap) {
    llvm::errs() << pair.first << ",";
  }
  llvm::errs() << "\n";
  llvm::errs() << "*** End of Critical Computation Costs ***\n\n";

  double minAccCost = std::numeric_limits<double>::infinity();
  InstructionCost bestCompCost = 0;
  for (const auto &pair : costToAccuracyMap) {
    InstructionCost compCost = pair.first;
    double accCost = pair.second;

    if (compCost <= FPOptComputationCostBudget && accCost < minAccCost) {
      minAccCost = accCost;
      bestCompCost = compCost;
    }
  }

  if (minAccCost == std::numeric_limits<double>::infinity()) {
    llvm::errs() << "No solution found within the computation cost budget!\n";
    return changed;
  }

  llvm::errs() << "Minimum accuracy cost within budget: " << minAccCost << "\n";
  llvm::errs() << "Computation cost budget used: " << bestCompCost << "\n";

  if (bestCompCost == 0 && minAccCost == 0) {
    llvm::errs() << "WARNING: DP Solver recommended no optimization given the "
                    "current computation cost budget.\n";
    return changed;
  }

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");

  llvm::errs() << "\n!!! DP solver: Applying solution ... !!!\n";
  for (const auto &solution : costToSolutionMap[bestCompCost]) {
    std::visit(
        [&](auto *item) {
          using T = std::decay_t<decltype(*item)>;
          if constexpr (std::is_same_v<T, ApplicableOutput>) {
            item->apply(solution.candidateIndex, valueToNodeMap,
                        symbolToValueMap);
          } else if constexpr (std::is_same_v<T, ApplicableFPCC>) {
            item->apply(solution.candidateIndex);
          } else {
            llvm_unreachable(
                "accuracyDPSolver: Unexpected type of solution step");
          }
        },
        solution.item);
    changed = true;
  }
  llvm::errs() << "!!! DP Solver: Solution applied !!!\n\n";

  return changed;
}

// Run (our choice of) floating point optimizations on function `F`.
// Return whether or not we change the function.
bool fpOptimize(Function &F, const TargetTransformInfo &TTI) {
  const std::string functionName = F.getName().str();
  std::string demangledName = llvm::demangle(functionName);
  size_t pos = demangledName.find('(');
  if (pos != std::string::npos) {
    demangledName = demangledName.substr(0, pos);
  }

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
        llvm::errs() << "Skipping matched function: " << functionName
                     << " since this function is not found in the log\n";
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

  std::unordered_map<Value *, std::shared_ptr<FPNode>> valueToNodeMap;
  std::unordered_map<std::string, Value *> symbolToValueMap;

  llvm::errs() << "FPOpt: Starting Floodfill for " << F.getName() << "\n";

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (!herbiable(I)) {
        valueToNodeMap[&I] =
            std::make_shared<FPLLValue>(&I, "__nh", "__nh"); // Non-herbiable
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
      auto node = std::make_shared<FPLLValue>(&I, getHerbieOperator(I), dtype);

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
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(Arg, "__arg", dtype);
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
            valueToNodeMap[operand] =
                std::make_shared<FPConst>(value.c_str(), dtype);
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
            valueToNodeMap[operand] =
                std::make_shared<FPLLValue>(GV, "__gv", dtype);
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

            // Don't mark constants as input `llvm::Value`s
            if (!isa<ConstantFP>(operand))
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
              auto node = valueToNodeMap[operand];
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
            // Extract grad and value info for all outputs.
            for (auto &op : CC.operations) {
              double grad = 0;
              auto blockIt = std::find_if(
                  op->getFunction()->begin(), op->getFunction()->end(),
                  [&](const auto &block) { return &block == op->getParent(); });
              assert(blockIt != op->getFunction()->end() && "Block not found");
              size_t blockIdx =
                  std::distance(op->getFunction()->begin(), blockIt);
              auto instIt =
                  std::find_if(op->getParent()->begin(), op->getParent()->end(),
                               [&](const auto &curr) { return &curr == op; });
              assert(instIt != op->getParent()->end() &&
                     "Instruction not found");
              size_t instIdx = std::distance(op->getParent()->begin(), instIt);
              bool found = extractGradFromLog(FPOptLogPath, functionName,
                                              blockIdx, instIdx, grad);

              auto node = valueToNodeMap[op];

              if (found) {
                node->grad = grad;

                ValueInfo valueInfo;
                extractValueFromLog(FPOptLogPath, functionName, blockIdx,
                                    instIdx, valueInfo);
                node->executions = valueInfo.executions;
                node->geometricAvg = valueInfo.geometricAvg;
                node->updateBounds(valueInfo.minRes, valueInfo.maxRes);

                if (EnzymePrintFPOpt) {
                  llvm::errs()
                      << "Range of " << *op << " is [" << node->getLowerBound()
                      << ", " << node->getUpperBound() << "]\n";
                }

                if (EnzymePrintFPOpt)
                  llvm::errs()
                      << "Grad of " << *op << " is: " << node->grad << "\n"
                      << "Execution count of " << *op
                      << " is: " << node->executions << "\n";
              } else { // Unknown bounds
                if (EnzymePrintFPOpt)
                  llvm::errs()
                      << "Grad of " << *op << " are not found in the log\n";
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

  SmallVector<ApplicableOutput, 4> AOs;
  SmallVector<ApplicableFPCC, 4> ACCs;

  for (auto &component : connected_components) {
    assert(component.inputs.size() > 0 && "No inputs found for component");
    if (FPOptEnableHerbie) {
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
          llvm::errs() << "assigning symbol: " << node->symbol << " to "
                       << *input << "\n";
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
        if (!improveViaHerbie(herbieInput, AO, F.getParent(), TTI,
                              valueToNodeMap, symbolToValueMap)) {
          if (EnzymePrintHerbie)
            llvm::errs() << "Failed to optimize an expression using Herbie!\n";
          continue;
        }

        AOs.push_back(std::move(AO));
      }
    }

    if (FPOptEnablePT) {
      // Sort `component.operations` by the gradient and construct
      // `PrecisionChange`s.
      ApplicableFPCC ACC(component, TTI);
      auto *o0 = component.outputs[0];
      ACC.executions = valueToNodeMap[o0]->executions;

      const SmallVector<PrecisionChangeType> precTypes{
          PrecisionChangeType::FP32,
          PrecisionChangeType::FP64,
      };

      // TODO: since we are only doing FP64 -> FP32, we can skip more expensive
      // operations for now.
      static const std::unordered_set<std::string> Funcs = {
          "sin",   "cos",  "tan", "exp",  "log",   "sqrt", "expm1",
          "log1p", "cbrt", "pow", "fabs", "hypot", "fma"};

      SmallVector<FPLLValue *, 8> operations;
      for (auto *I : component.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        if (Funcs.count(node->op) != 0) {
          operations.push_back(node);
        }
      }

      // Sort operations by the gradient
      llvm::sort(operations, [](const auto &a, const auto &b) {
        return std::fabs(a->grad * a->geometricAvg) <
               std::fabs(b->grad * b->geometricAvg);
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = operations.size() * percent / 100;

        SetVector<FPLLValue *> opsToChange(operations.begin(),
                                           operations.begin() + numToChange);

        if (EnzymePrintFPOpt && !opsToChange.empty()) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of Funcs (" << numToChange << ")\n";
          llvm::errs() << "Subset gradient range: ["
                       << std::fabs(opsToChange.front()->grad) << ", "
                       << std::fabs(opsToChange.back()->grad) << "]\n";
        }

        for (auto prec : precTypes) {
          StringRef precStr = getPrecisionChangeTypeString(prec);
          Twine desc =
              Twine("Funcs 0% -- ") + Twine(percent) + "% -> " + precStr;

          PrecisionChange change(
              opsToChange,
              getPrecisionChangeType(component.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate(changes, desc);
          candidate.CompCost = getCompCost(component, TTI, candidate);

          ACC.candidates.push_back(std::move(candidate));
        }
      }

      // Create candidates by considering all operations without filtering
      SmallVector<FPLLValue *, 8> allOperations;
      for (auto *I : component.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        auto node = cast<FPLLValue>(valueToNodeMap[I].get());
        allOperations.push_back(node);
      }

      // Sort all operations by the gradient
      llvm::sort(allOperations, [](const auto &a, const auto &b) {
        return std::fabs(a->grad * a->geometricAvg) <
               std::fabs(b->grad * b->geometricAvg);
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = allOperations.size() * percent / 100;

        SetVector<FPLLValue *> opsToChange(allOperations.begin(),
                                           allOperations.begin() + numToChange);

        if (EnzymePrintFPOpt && !opsToChange.empty()) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of all operations (" << numToChange << ")\n";
          llvm::errs() << "Subset gradient range: ["
                       << std::fabs(opsToChange.front()->grad) << ", "
                       << std::fabs(opsToChange.back()->grad) << "]\n";
        }

        for (auto prec : precTypes) {
          StringRef precStr = getPrecisionChangeTypeString(prec);
          Twine desc = Twine("All 0% -- ") + Twine(percent) + "% -> " + precStr;

          PrecisionChange change(
              opsToChange,
              getPrecisionChangeType(component.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate(changes, desc);
          candidate.CompCost = getCompCost(component, TTI, candidate);

          ACC.candidates.push_back(std::move(candidate));
        }
      }

      setUnifiedAccuracyCost(ACC, valueToNodeMap, symbolToValueMap);

      ACCs.push_back(std::move(ACC));
    }
  }

  // Perform rewrites
  if (EnzymePrintFPOpt) {
    if (FPOptEnableHerbie) {
      for (auto &AO : AOs) {
        // TODO: Solver
        // Available Parameters:
        // 1. gradients at the output llvm::Value
        // 2. costs of the potential rewrites from Herbie (lower is preferred)
        // 3. percentage accuracies of potential rewrites (higher is better)
        // 4*. TTI costs of potential rewrites (TODO: need to consider branches)
        // 5*. Custom error estimates of potential rewrites (TODO)

        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << AO.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << AO.initialCompCost
                     << "\n";
        llvm::errs() << "Initial HerbieCost: " << AO.initialHerbieCost << "\n";
        llvm::errs() << "Initial HerbieAccuracy: " << AO.initialHerbieAccuracy
                     << "\n";
        llvm::errs() << "Initial Expression: " << AO.expr << "\n";
        llvm::errs() << "Grad: " << AO.grad << "\n\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ "
                        "CompCost\t\tHerbieCost\t\tAccuracy\t\tExpression\n";
        llvm::errs() << "--------------------------------\n";
        for (size_t i = 0; i < AO.candidates.size(); ++i) {
          auto &candidate = AO.candidates[i];
          llvm::errs() << AO.getAccCostDelta(i) << "\t\t"
                       << AO.getCompCostDelta(i) << "\t\t"
                       << candidate.herbieCost << "\t\t"
                       << candidate.herbieAccuracy << "\t\t" << candidate.expr
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
    if (FPOptEnablePT) {
      for (auto &ACC : ACCs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << ACC.initialAccCost << "\n";
        llvm::errs() << "Initial ComputationCost: " << ACC.initialCompCost
                     << "\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "Δ AccCost\t\tΔ CompCost\t\tDescription\n"
                     << "---------------------------\n";
        for (size_t i = 0; i < ACC.candidates.size(); ++i) {
          auto &candidate = ACC.candidates[i];
          llvm::errs() << ACC.getAccCostDelta(i) << "\t\t"
                       << ACC.getCompCostDelta(i) << "\t\t" << candidate.desc
                       << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
  }

  if (!FPOptEnableSolver) {
    if (FPOptEnableHerbie) {
      for (auto &AO : AOs) {
        AO.apply(0, valueToNodeMap, symbolToValueMap);
        changed = true;
      }
    }

    // TODO: just for testing
    if (FPOptEnablePT) {
      for (auto &ACC : ACCs) {
        ACC.apply(0);
        changed = true;
      }
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
      changed = accuracyDPSolver(AOs, ACCs, valueToNodeMap, symbolToValueMap);
    } else {
      llvm::errs() << "FPOpt: Unknown solver type: " << FPOptSolverType << "\n";
      return false;
    }
  }

  llvm::errs() << "FPOpt: Finished optimizing " << F.getName() << "\n";

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
