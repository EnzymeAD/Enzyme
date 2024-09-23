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
#include <map>
#include <numeric>
#include <random>
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
static cl::opt<bool> FPOptEnableHerbie(
    "fpopt-enable-herbie", cl::init(true), cl::Hidden,
    cl::desc("Use Herbie to rewrite floating-point expressions"));
static cl::opt<bool> FPOptEnablePT(
    "fpopt-enable-pt", cl::init(false), cl::Hidden,
    cl::desc("Consider precision changes of floating-point expressions"));
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

  virtual Value *getLLValue(IRBuilder<> &builder) {
    // if (EnzymePrintFPOpt)
    //   llvm::errs() << "Generating new instruction for op: " << op << "\n";
    Module *M = builder.GetInsertBlock()->getModule();

    if (op == "if") {
      Value *condValue = operands[0]->getLLValue(builder);
      auto IP = builder.GetInsertPoint();

      Instruction *Then, *Else;
      SplitBlockAndInsertIfThenElse(condValue, &*IP, &Then, &Else);

      Then->getParent()->setName("herbie.then");
      builder.SetInsertPoint(Then);
      Value *ThenVal = operands[1]->getLLValue(builder);
      if (Instruction *I = dyn_cast<Instruction>(ThenVal)) {
        I->setName("herbie.then_val");
      }

      Else->getParent()->setName("herbie.else");
      builder.SetInsertPoint(Else);
      Value *ElseVal = operands[2]->getLLValue(builder);
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
      operandValues.push_back(operand->getLLValue(builder));
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

  Value *getLLValue(IRBuilder<> &builder) override { return value; }

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

  virtual Value *getLLValue(IRBuilder<> &builder) override {
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

enum class PrecisionChangeType { FP16, FP32, FP64 };

unsigned getMPFRPrec(PrecisionChangeType type) {
  switch (type) {
  case PrecisionChangeType::FP16:
    return 11;
  case PrecisionChangeType::FP32:
    return 24;
  case PrecisionChangeType::FP64:
    return 53;
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

Type *getLLVMFPType(PrecisionChangeType type, LLVMContext &context) {
  switch (type) {
  case PrecisionChangeType::FP16:
    return Type::getHalfTy(context);
  case PrecisionChangeType::FP32:
    return Type::getFloatTy(context);
  case PrecisionChangeType::FP64:
    return Type::getDoubleTy(context);
  default:
    llvm_unreachable("Unsupported FP precision");
  }
}

PrecisionChangeType getPrecisionChangeType(Type *type) {
  if (type->isHalfTy()) {
    return PrecisionChangeType::FP16;
  } else if (type->isFloatTy()) {
    return PrecisionChangeType::FP32;
  } else if (type->isDoubleTy()) {
    return PrecisionChangeType::FP64;
  } else {
    llvm_unreachable("Unsupported FP precision");
  }
}

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

struct PTCandidate {
  // Only one PT candidate per FPCC can be applied
  SmallVector<PrecisionChange, 1> changes;
  double accuracyCost;
  InstructionCost TTICost;

  // TODO:
  explicit PTCandidate(SmallVector<PrecisionChange> &changes)
      : changes(changes) {
    // TTICost = getTTICost(changes);
  }
};

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

    if (it != cache.end()) {
      assert(cache.at(node).prec == nodePrec && "Unexpected precision change");
      return;
    } else {
      // Prepare for recomputation
      cache.emplace(node, CachedValue(nodePrec));
    }

    mpfr_t &res = cache.at(node).value;

    if (node->op == "neg") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_neg(res, op, MPFR_RNDN);
    } else if (node->op == "+") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_add(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "-") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_sub(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "*") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_mul(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "/") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_div(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "sin") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_sin(res, op, MPFR_RNDN);
    } else if (node->op == "cos") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_cos(res, op, MPFR_RNDN);
    } else if (node->op == "tan") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_tan(res, op, MPFR_RNDN);
    } else if (node->op == "exp") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_exp(res, op, MPFR_RNDN);
    } else if (node->op == "expm1") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_expm1(res, op, MPFR_RNDN);
    } else if (node->op == "log") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_log(res, op, MPFR_RNDN);
    } else if (node->op == "log1p") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_log1p(res, op, MPFR_RNDN);
    } else if (node->op == "sqrt") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_sqrt(res, op, MPFR_RNDN);
    } else if (node->op == "cbrt") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_cbrt(res, op, MPFR_RNDN);
    } else if (node->op == "pow") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_pow(res, op0, op1, MPFR_RNDN);
    } else if (node->op == "fma") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      evaluateNode(node->operands[2].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
      mpfr_t &op2 = getResult(node->operands[2].get());
      mpfr_fma(res, op0, op1, op2, MPFR_RNDN);
    } else if (node->op == "fabs") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      mpfr_t &op = getResult(node->operands[0].get());
      mpfr_abs(res, op, MPFR_RNDN);
    } else if (node->op == "hypot") {
      evaluateNode(node->operands[0].get(), inputValues, groundTruth);
      evaluateNode(node->operands[1].get(), inputValues, groundTruth);
      mpfr_t &op0 = getResult(node->operands[0].get());
      mpfr_t &op1 = getResult(node->operands[1].get());
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
  assert(outputs.size() > 0);
  results.resize(outputs.size());

  if (!groundTruth) {
    MPFREvaluator evaluator(0, pt);
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

// Sum up the cost of `output` and its FP operands recursively up to `inputs`
// (exclusive).
InstructionCost getTTICost(const SmallVector<Value *> &outputs,
                           const SetVector<Value *> &inputs,
                           const TargetTransformInfo &TTI) {
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
           std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
           std::unordered_map<std::string, Value *> &symbolToValueMap) {
  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(expr, argStrSet);

  SetVector<Value *> args;
  for (const auto &argStr : argStrSet) {
    args.insert(symbolToValueMap[argStr]);
  }

  auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);

  // Materialize the expression in a temporary function
  FunctionType *FT = FunctionType::get(Type::getVoidTy(M->getContext()), false);
  Function *tempFunction =
      Function::Create(FT, Function::InternalLinkage, "tempFunc", M);
  BasicBlock *entry =
      BasicBlock::Create(M->getContext(), "entry", tempFunction);
  Instruction *ReturnInst = ReturnInst::Create(M->getContext(), entry);

  IRBuilder<> builder(ReturnInst);

  builder.setFastMathFlags(getFast());
  Value *newOutput = parsedNode->getLLValue(builder);

  // tempFunction->print(llvm::errs());

  InstructionCost cost = getTTICost({newOutput}, args, TTI);

  tempFunction->eraseFromParent();
  return cost;
}

struct RewriteCandidate {
  // Only one rewrite candidate per output `llvm::Value` can be applied
  InstructionCost TTICost;
  double herbieCost; // Unused for now
  double herbieAccuracy;
  double accuracyCost;
  std::string expr;

  RewriteCandidate(double cost, double accuracy, std::string expression)
      : herbieCost(cost), herbieAccuracy(accuracy), expr(expression) {}
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

void collectExprInsts(Value *V, const SetVector<Value *> &inputs,
                      SmallPtrSet<Instruction *, 16> &exprInsts,
                      SmallPtrSet<Value *, 16> &visited) {
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

class ApplicableOutput {
public:
  FPCC &component;
  Value *oldOutput;
  std::string expr;
  double grad;
  unsigned executions;
  const TargetTransformInfo &TTI;
  InstructionCost initialAccuracyCost; // Requires manual initialization
  InstructionCost initialTTICost;      // Requires manual initialization
  InstructionCost initialHerbieCost;   // Requires manual initialization
  double initialHerbieAccuracy;        // Requires manual initialization
  SmallVector<RewriteCandidate> candidates;
  SmallPtrSet<Instruction *, 8> erasableInsts;

  explicit ApplicableOutput(FPCC &component, Value *oldOutput, std::string expr,
                            double grad, unsigned executions,
                            const TargetTransformInfo &TTI)
      : component(component), oldOutput(oldOutput), expr(expr), grad(grad),
        executions(executions), TTI(TTI) {
    initialTTICost = getTTICost({oldOutput}, component.inputs, TTI);
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
    // TODO ponder fast math
    builder.setFastMathFlags(getFast());

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
      component.operations.remove(I); // Avoid a second removal
    }

    component.outputs_rewritten++;
  }

  // Lower is better
  InstructionCost getComputationCost(size_t candidateIndex) {
    // TODO: Better cost model
    InstructionCost erasableCost = 0;

    for (auto *I : erasableInsts) {
      erasableCost +=
          TTI.getInstructionCost(I, TargetTransformInfo::TCK_SizeAndLatency);
    }

    return (candidates[candidateIndex].TTICost - erasableCost) * executions;
  }

  // Lower is better
  double getAccuracyCost(size_t candidateIndex) {
    // TODO: Update this accuracy
    return candidates[candidateIndex].accuracyCost;
  }

  void findErasableInstructions() {
    SmallPtrSet<Instruction *, 16> exprInsts;
    SmallPtrSet<Value *, 16> visited;
    collectExprInsts(oldOutput, component.inputs, exprInsts, visited);

    for (auto *I : exprInsts) {
      bool usedOutside = false;

      for (auto user : I->users()) {
        if (auto *userI = dyn_cast<Instruction>(user);
            userI && exprInsts.contains(userI)) {
          // Use is within the expression
          continue;
        } else {
          // Can't erase an llvm::Value or an instruction used outside
          // the expression

          // llvm::errs() << "Can't erase: " << *I << " -- used by: " << *user
          //              << "\n";
          usedOutside = true;
          break;
        }
      }

      if (!usedOutside) {
        erasableInsts.insert(I);
      }
    }

    llvm::errs() << "Erasable instructions:\n";
    for (auto *I : erasableInsts) {
      llvm::errs() << *I << "\n";
    }
    llvm::errs() << "End of erasable instructions\n";
  }
};

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

void changePrecision(Instruction *I, PrecisionChange &change,
                     MapVector<Value *, Value *> &oldToNew) {
  if (!herbiable(*I)) {
    llvm_unreachable("Trying to tune an instruction is not herbiable");
  }

  IRBuilder<> Builder(I);
  Builder.setFastMathFlags(getFast());
  Type *newType = getLLVMFPType(change.newType, I->getContext());
  Value *newI = nullptr;

  if (isa<UnaryOperator>(I) || isa<BinaryOperator>(I)) {
    llvm::errs() << "PT Changing: " << *I << " to " << *newType << "\n";
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
    Function *newFunc = Intrinsic::getDeclaration(
        CI->getModule(), CI->getCalledFunction()->getIntrinsicID(), {newType});
    newI = Builder.CreateCall(newFunc, newArgs);
  } else {
    llvm_unreachable("Unknown herbiable instruction");
  }

  oldToNew[I] = newI;
  llvm::errs() << "PT Changing: " << *I << " to " << *newI << "\n";
}

class ApplicableFPCC {
public:
  FPCC &component;
  const TargetTransformInfo &TTI;
  InstructionCost initialAccuracyCost; // Requires manual initialization
  InstructionCost initialTTICost;

  SmallVector<PTCandidate> candidates;

  explicit ApplicableFPCC(FPCC &fpcc, const TargetTransformInfo &TTI)
      : component(fpcc), TTI(TTI) {
    initialTTICost =
        getTTICost({component.outputs.begin(), component.outputs.end()},
                   component.inputs, TTI);
  }

  void apply(size_t candidateIndex) {
    if (candidateIndex >= candidates.size()) {
      llvm_unreachable("Invalid candidate index");
    }

    // Traverse all the instructions to be changed precisions in a
    // topological order with respect to operand dependencies. Insert FP casts
    // between llvm::Value inputs and first level of instructions to be changed.
    // Restore precisions of the last level of instructions to be changed.

    for (auto &change : candidates[candidateIndex].changes) {
      SmallPtrSet<Instruction *, 8> seen;
      SmallVector<Instruction *, 8> todo;
      MapVector<Value *, Value *> oldToNew;

      SetVector<Instruction *> instsToChange;
      for (auto node : change.nodes) {
        assert(isa<Instruction>(node->value));
        instsToChange.insert(cast<Instruction>(node->value));
      }

      // For implicit topo ordering wrt operand dependencies
      MapVector<Instruction *, int> operandCount;
      for (auto *I : instsToChange) {
        // We only change precisions of instructions
        int count = 0;
        auto operands =
            isa<CallInst>(I) ? cast<CallInst>(I)->args() : I->operands();
        for (auto &op : operands) {
          if (isa<Instruction>(op) &&
              instsToChange.contains(cast<Instruction>(op))) {
            count++;
          }
        }
        operandCount[I] = count;

        if (0 == count) {
          todo.push_back(I);
        }
      }

      while (!todo.empty()) {
        auto *cur = todo.pop_back_val();
        llvm::errs() << "PT Processing: " << *cur << "\n";
        if (!seen.insert(cur).second)
          continue;

        if (isa<Instruction>(cur) &&
            component.operations.contains(cast<Instruction>(cur))) {
          changePrecision(cast<Instruction>(cur), change, oldToNew);
        }

        for (auto user : cur->users()) {
          if (isa<Instruction>(user) &&
              operandCount.count(cast<Instruction>(user))) {
            if (0 == --operandCount[cast<Instruction>(user)]) {
              llvm::errs() << "PT Adding: " << *cast<Instruction>(user) << "\n";
              todo.push_back(cast<Instruction>(user));
            }
          }
        }
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

        for (auto user : oldV->users()) {
          if (isa<Instruction>(user) &&
              !instsToChange.contains(cast<Instruction>(user))) {
            IRBuilder<> builder(cast<Instruction>(user));

            newV = builder.CreateFPCast(
                newV, getLLVMFPType(change.oldType, builder.getContext()));

            user->replaceUsesOfWith(oldV, newV);
          }
        }

        // Assumes no external uses of the old value since all corresponding new
        // values are already restored to original precision and used to replace
        // uses of their old value. This is also advantageous to the solvers.
        if (!oldV->use_empty()) {
          oldV->replaceAllUsesWith(UndefValue::get(oldV->getType()));
        }
        cast<Instruction>(oldV)->eraseFromParent();
      }
    }
  }

  // TODO: Update
  // Lower is better
  // InstructionCost getComputationCost(size_t candidateIndex) {
  //   // TODO: consider erasure of the old output
  //   return candidates[candidateIndex].TTICost * executions;
  // }

  // // Lower is better
  // double getAccuracyCost(size_t candidateIndex) {
  //   return (initialHerbieAccuracy - candidates[candidateIndex].accuracy) *
  //          std::fabs(grad);
  // }
};

void setUnifiedAccuracyCost(
    ApplicableOutput &AO,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {

  SmallSet<std::string, 8> argStrSet;
  getUniqueArgs(AO.expr, argStrSet);

  SmallVector<SmallMapVector<Value *, double, 4>, 4> sampledPoints;
  getSampledPoints(AO.expr, valueToNodeMap, symbolToValueMap, sampledPoints);

  SmallVector<double, 4> goldVals;
  goldVals.resize(FPOptNumSamples);
  double initialAC = 0.;

  for (const auto &pair : enumerate(sampledPoints)) {
    ArrayRef<FPNode *> outputs = {valueToNodeMap[AO.oldOutput].get()};
    SmallVector<double, 1> results;
    getMPFRValues(outputs, pair.value(), results, true, 53);
    double goldVal = results[0];
    goldVals[pair.index()] = goldVal;

    getMPFRValues(outputs, pair.value(), results, false);
    double realVal = results[0];

    initialAC += std::fabs((goldVal - realVal) * AO.grad);
  }

  AO.initialAccuracyCost = initialAC;

  for (auto &candidate : AO.candidates) {
    const auto &expr = candidate.expr;
    auto parsedNode = parseHerbieExpr(expr, valueToNodeMap, symbolToValueMap);
    double ac = 0.;

    for (const auto &pair : enumerate(sampledPoints)) {
      // Compute the "gold" value & real value for each sampled point
      // Compute an average of (difference * gradient)
      // TODO: Consider geometric average???
      assert(valueToNodeMap.count(AO.oldOutput));

      llvm::errs() << "Computing real output for candidate: " << expr << "\n";

      llvm::errs() << "Current input values:\n";
      for (const auto &entry : pair.value()) {
        llvm::errs() << valueToNodeMap[entry.first]->symbol << ": "
                     << entry.second << "\n";
      }

      llvm::errs() << "Gold value: " << goldVals[pair.index()] << "\n";

      ArrayRef<FPNode *> outputs = {parsedNode.get()};
      SmallVector<double, 1> results;
      getMPFRValues(outputs, pair.value(), results, false);
      double realVal = results[0];

      llvm::errs() << "Real value: " << realVal << "\n";
      ac += std::fabs((goldVals[pair.index()] - realVal) * AO.grad);
    }
    candidate.accuracyCost = ac;
  }
}

void setUnifiedAccuracyCost(
    ApplicableFPCC &ACC,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  SmallVector<SmallMapVector<Value *, double, 4>, 4> sampledPoints;
  getSampledPoints(ACC.component.inputs.getArrayRef(), valueToNodeMap,
                   symbolToValueMap, sampledPoints);

  double initialAC = 0.;
  SmallMapVector<FPNode *, double, 4> goldVals; // output -> gold val

  SmallVector<FPNode *, 4> outputs;
  for (auto *output : ACC.component.outputs) {
    outputs.push_back(valueToNodeMap[output].get());
  }

  for (const auto &pair : enumerate(sampledPoints)) {
    SmallVector<double, 1> results;
    getMPFRValues(outputs, pair.value(), results, true, 53);
    for (const auto &[output, result] : zip(outputs, results)) {
      goldVals[output] = result;
    }

    getMPFRValues(outputs, pair.value(), results, false);
    for (const auto &[output, result] : zip(outputs, results)) {
      initialAC += std::fabs((goldVals[output] - result) * output->grad);
    }
  }

  ACC.initialAccuracyCost = initialAC;
  llvm::errs() << "Initial ACC accuracy cost: " << ACC.initialAccuracyCost
               << "\n";

  for (auto &candidate : ACC.candidates) {
    double ac = 0.;
    for (const auto &pair : enumerate(sampledPoints)) {
      SmallVector<double, 1> results;

      getMPFRValues(outputs, pair.value(), results, false, 0, &candidate);
      for (const auto &[output, result] : zip(outputs, results)) {
        llvm::errs() << "DEBUG gold value: " << goldVals[output] << "\n";
        llvm::errs() << "DEBUG real value: " << goldVals[output] << "\n";
        ac += std::fabs((goldVals[output] - result) * output->grad);
      }
    }
    candidate.accuracyCost = ac;
    llvm::errs() << "Accuracy cost for PT candidate: " << ac << "\n";
  }
}

bool improveViaHerbie(
    const std::string &inputExpr, ApplicableOutput &AO, Module *M,
    const TargetTransformInfo &TTI,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
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
  llvm::errs() << "random seed: " << std::to_string(FPOptRandomSeed) << "\n";
  SmallVector<llvm::StringRef> Args = {
      Program,     "report", "--seed", std::to_string(FPOptRandomSeed),
      "--timeout", "60"};

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
  AO.initialHerbieAccuracy = initialAccuracy;

  json::Array &best = *costAccuracy[1].getAsArray();
  double bestCost = best[0].getAsNumber().getValue() / initialCostVal;
  double bestAccuracy = 1.0 - best[1].getAsNumber().getValue() / bits;

  RewriteCandidate bestCandidate(bestCost, bestAccuracy, bestExpr.str());
  bestCandidate.TTICost =
      getTTICost(bestExpr.str(), M, TTI, valueToNodeMap, symbolToValueMap);
  AO.candidates.push_back(bestCandidate);

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
  }

  setUnifiedAccuracyCost(AO, valueToNodeMap, symbolToValueMap);

  // if (EnzymePrintHerbie) {
  //   llvm::errs() << "Initial: "
  //                << "AccuracyCost = " << AO.initialAccuracyCost
  //                << ", ComputationCost = " << 0
  //                << ", TTICost = " << AO.initialTTICost
  //                << ", HerbieCost = " << initialCost
  //                << ", HerbieAccuracy = " << initialAccuracy << "\n";
  //   // The best candidate from Herbie is also printed below
  //   for (size_t i = 0; i < AO.candidates.size(); ++i) {
  //     auto &candidate = AO.candidates[i];
  //     llvm::errs() << "Alternative " << i + 1
  //                  << ": AccuracyCost = " << candidate.accuracyCost
  //                  << ", ComputationCost = " << AO.getComputationCost(i)
  //                  << ", TTICost = " << candidate.TTICost
  //                  << ", HerbieCost = " << candidate.herbieCost
  //                  << ", HerbieAccuracy = " << candidate.herbieAccuracy
  //                  << ", Expression = " << candidate.expr << "\n";
  //   }
  // }

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
      auto candCompCost = AO.getComputationCost(i);
      auto candAccCost = AO.getAccuracyCost(i);
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
    SmallVector<ApplicableOutput, 4> &AOs,
    std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
    std::unordered_map<std::string, Value *> &symbolToValueMap) {
  bool changed = false;
  llvm::errs() << "Starting accuracy DP solver with computation budget: "
               << FPOptComputationCostBudget << "\n";

  using CostMap = std::map<InstructionCost, double>;
  using SolutionMap =
      std::map<InstructionCost,
               SmallVector<std::pair<ApplicableOutput *, size_t>>>;

  CostMap costToAccuracyMap;
  costToAccuracyMap[0] = 0;
  SolutionMap costToSolutionMap;
  costToSolutionMap[0] = {};

  for (auto &AO : AOs) {
    CostMap newCostToAccuracyMap;
    SolutionMap newCostToSolutionMap;

    llvm::errs() << "Processing AO: " << AO.expr << "\n";

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
        auto candCompCost = AO.getComputationCost(i);
        auto candAccCost = AO.getAccuracyCost(i);

        InstructionCost newCompCost = currCompCost + candCompCost;
        double newAccCost = currAccCost + candAccCost;

        llvm::errs() << "Candidate " << i
                     << " has accuracy cost: " << candAccCost
                     << " and computation cost: " << candCompCost << "\n";

        if (newCostToAccuracyMap.find(newCompCost) ==
                newCostToAccuracyMap.end() ||
            newCostToAccuracyMap[newCompCost] > newAccCost) {
          newCostToAccuracyMap[newCompCost] = newAccCost;
          newCostToSolutionMap[newCompCost] = costToSolutionMap[currCompCost];
          newCostToSolutionMap[newCompCost].emplace_back(&AO, i);
          llvm::errs() << "Updating accuracy map (candidate " << i
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

        if (currCompCost > otherCompCost && currAccCost >= otherAccCost) {
          llvm::errs() << "Candidate with computation cost: " << currCompCost
                       << " and accuracy cost: " << currAccCost
                       << " is dominated by candidate with computation cost: "
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

  llvm::errs() << "DP Table: \n";
  for (const auto &pair : costToAccuracyMap) {
    llvm::errs() << "Computation cost: " << pair.first
                 << ", Accuracy cost: " << pair.second << "\n";
  }

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
    llvm::errs()
        << "WARNING: DP Solver recommended no expression-level optimization.\n";
    return changed;
  }

  assert(costToSolutionMap.find(bestCompCost) != costToSolutionMap.end() &&
         "FPOpt DP solver: expected a solution!");

  for (const auto &solution : costToSolutionMap[bestCompCost]) {
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

      SmallVector<FPLLValue *, 8> operations;
      for (auto *I : component.operations) {
        assert(isa<FPLLValue>(valueToNodeMap[I].get()) &&
               "Corrupted FPNode for original instructions");
        operations.push_back(cast<FPLLValue>(valueToNodeMap[I].get()));
      }

      // TODO: computation cost conflicts with Herbie rewrites

      // Sort the operations by the gradient
      llvm::sort(operations, [](const auto &a, const auto &b) {
        llvm::errs() << "Gradient of " << *(a->value) << " is " << a->grad
                     << "\n";
        llvm::errs() << "Gradient of " << *(b->value) << " is " << b->grad
                     << "\n";
        assert(!std::isnan(a->grad) && "Gradient is NaN for an operation");
        assert(!std::isnan(b->grad) && "Gradient is NaN for an operation");
        return std::fabs(a->grad) < std::fabs(b->grad);
      });

      // Create PrecisionChanges for 0-10%, 0-20%, ..., up to 0-100%
      for (int percent = 10; percent <= 100; percent += 10) {
        size_t numToChange = operations.size() * percent / 100;

        SetVector<FPLLValue *> opsToChange(operations.begin(),
                                           operations.begin() + numToChange);

        if (!opsToChange.empty()) {
          llvm::errs() << "Created PrecisionChange for " << percent
                       << "% of operations (" << numToChange << ")\n";
          llvm::errs() << "Subset gradient range: ["
                       << std::fabs(opsToChange.front()->grad) << ", "
                       << std::fabs(opsToChange.back()->grad) << "]\n";
        }

        SmallVector<PrecisionChangeType> precTypes{PrecisionChangeType::FP16,
                                                   PrecisionChangeType::FP32,
                                                   PrecisionChangeType::FP64};

        for (auto prec : precTypes) {
          PrecisionChange change(
              opsToChange,
              getPrecisionChangeType(component.outputs[0]->getType()), prec);

          SmallVector<PrecisionChange, 1> changes{std::move(change)};
          PTCandidate candidate(changes);

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
        llvm::errs() << "Initial AccuracyCost: " << AO.initialAccuracyCost
                     << "\n";
        llvm::errs() << "Initial ComputationCost: " << 0 << "\n";
        llvm::errs() << "Initial TTICost: " << AO.initialTTICost << "\n";
        llvm::errs() << "Initial HerbieCost: " << AO.initialHerbieCost << "\n";
        llvm::errs() << "Initial HerbieAccuracy: " << AO.initialHerbieAccuracy
                     << "\n";
        llvm::errs() << "Initial Expression: " << AO.expr << "\n";
        llvm::errs() << "Grad: " << AO.grad << "\n\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "AccuracyCost\t\tComputationCost\t\tTTICost\t\tHerbieCo"
                        "st\t\tAccu"
                        "racy\t\tExpression\n";
        llvm::errs() << "--------------------------------\n";
        for (size_t i = 0; i < AO.candidates.size(); ++i) {
          auto &candidate = AO.candidates[i];
          llvm::errs() << candidate.accuracyCost << "\t\t"
                       << AO.getComputationCost(i) << "\t\t"
                       << candidate.TTICost << "\t\t" << candidate.herbieCost
                       << "\t\t" << candidate.herbieAccuracy << "\t\t"
                       << candidate.expr << "\n";
        }
        llvm::errs() << "################################\n\n";
      }
    }
    if (FPOptEnablePT) {
      for (auto &ACC : ACCs) {
        llvm::errs() << "\n################################\n";
        llvm::errs() << "Initial AccuracyCost: " << ACC.initialAccuracyCost
                     << "\n";
        llvm::errs() << "Initial ComputationCost: " << 0 << "\n";
        llvm::errs() << "Initial TTICost: " << ACC.initialTTICost << "\n";
        llvm::errs() << "Candidates:\n";
        llvm::errs() << "AccuracyCost\t\tComputationCost\t\tTTICost\n"
                     << "--------------------------------\n";
        for (size_t i = 0; i < ACC.candidates.size(); ++i) {
          auto &candidate = ACC.candidates[i];
          llvm::errs() << candidate.accuracyCost
                       << "\t\t"
                       //  << ACC.getComputationCost(i) << "\t\t"
                       << candidate.TTICost << "\n";
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
      changed = accuracyDPSolver(AOs, valueToNodeMap, symbolToValueMap);
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
