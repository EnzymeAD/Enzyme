#ifndef ENZYME_POSEIDON_H
#define ENZYME_POSEIDON_H

#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"

#include <mpfr.h>

namespace llvm {
class FunctionPass;
class TargetTransformInfo;
class Function;
class Module;
class AnalysisUsage;
} // namespace llvm

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> EnzymeEnableFPOpt;
extern llvm::cl::opt<bool> EnzymePrintFPOpt;
extern llvm::cl::opt<bool> FPOptPrintPreproc;
}

extern llvm::cl::opt<bool> EnzymePrintHerbie;
extern llvm::cl::opt<std::string> FPOptLogPath;
extern llvm::cl::opt<std::string> FPOptCostModelPath;
extern llvm::cl::opt<std::string> FPOptTargetFuncRegex;
extern llvm::cl::opt<bool> FPOptEnableHerbie;
extern llvm::cl::opt<bool> FPOptEnablePT;
extern llvm::cl::opt<int> HerbieNumThreads;
extern llvm::cl::opt<int> HerbieTimeout;
extern llvm::cl::opt<std::string> FPOptCachePath;
extern llvm::cl::opt<int> HerbieNumPoints;
extern llvm::cl::opt<int> HerbieNumIters;
extern llvm::cl::opt<bool> HerbieDisableNumerics;
extern llvm::cl::opt<bool> HerbieDisableArithmetic;
extern llvm::cl::opt<bool> HerbieDisableFractions;
extern llvm::cl::opt<bool> HerbieDisableTaylor;
extern llvm::cl::opt<bool> HerbieDisableSetupSimplify;
extern llvm::cl::opt<bool> HerbieDisableGenSimplify;
extern llvm::cl::opt<bool> HerbieDisableRegime;
extern llvm::cl::opt<bool> HerbieDisableBranchExpr;
extern llvm::cl::opt<bool> HerbieDisableAvgError;
extern llvm::cl::opt<bool> FPOptEnableSolver;
extern llvm::cl::opt<std::string> FPOptSolverType;
extern llvm::cl::opt<bool> FPOptStrictMode;
extern llvm::cl::opt<std::string> FPOptReductionProf;
extern llvm::cl::opt<std::string> FPOptReductionEval;
extern llvm::cl::opt<double> FPOptGeoMeanEps;
extern llvm::cl::opt<bool> FPOptLooseCoverage;
extern llvm::cl::opt<bool> FPOptShowTable;
extern llvm::cl::list<int64_t> FPOptShowTableCosts;
extern llvm::cl::opt<bool> FPOptShowPTDetails;
extern llvm::cl::opt<int64_t> FPOptComputationCostBudget;
extern llvm::cl::opt<unsigned> FPOptMaxFPCCDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprDepth;
extern llvm::cl::opt<unsigned> FPOptMaxExprLength;
extern llvm::cl::opt<unsigned> FPOptRandomSeed;
extern llvm::cl::opt<unsigned> FPOptNumSamples;
extern llvm::cl::opt<unsigned> FPOptMaxMPFRPrec;
extern llvm::cl::opt<double> FPOptWidenRange;
extern llvm::cl::opt<bool> FPOptEarlyPrune;
extern llvm::cl::opt<double> FPOptCostDominanceThreshold;
extern llvm::cl::opt<double> FPOptAccuracyDominanceThreshold;

// LibM functions set
extern const std::unordered_set<std::string> LibmFuncs;

// Forward declarations
class FPNode;
class FPLLValue;
class FPConst;
struct FPCC;
struct PTCandidate;
struct RewriteCandidate;
struct PrecisionChange;
class ApplicableOutput;
class ApplicableFPCC;
struct SolutionStep;

// Enums
enum class PrecisionChangeType { BF16, FP16, FP32, FP64, FP80, FP128 };

// Utility functions
double getOneULP(double value);
unsigned getMPFRPrec(PrecisionChangeType type);

// Structs
struct FPCC {
  SetVector<Value *> inputs;
  SetVector<Instruction *> outputs;
  SetVector<Instruction *> operations;
  size_t outputs_rewritten = 0;

  FPCC() = default;
  explicit FPCC(SetVector<Value *> inputs, SetVector<Instruction *> outputs,
                SetVector<Instruction *> operations);
};

struct PrecisionChange {
  SetVector<FPLLValue *> nodes;
  PrecisionChangeType oldType;
  PrecisionChangeType newType;

  explicit PrecisionChange(SetVector<FPLLValue *> &nodes,
                           PrecisionChangeType oldType,
                           PrecisionChangeType newType);
};

struct PTCandidate {
  SmallVector<PrecisionChange, 1> changes;
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost CompCost = std::numeric_limits<InstructionCost>::max();
  std::string desc;
  std::unordered_map<FPNode *, double> perOutputAccCost;
  std::unordered_map<FPNode *, SmallVector<double, 4>> errors;

  explicit PTCandidate(SmallVector<PrecisionChange> changes,
                       const std::string &desc);
  void apply(FPCC &component, ValueToValueMapTy *VMap = nullptr);
};

struct RewriteCandidate {
  InstructionCost CompCost = std::numeric_limits<InstructionCost>::max();
  double herbieCost = std::numeric_limits<double>::quiet_NaN();
  double herbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  double accuracyCost = std::numeric_limits<double>::quiet_NaN();
  std::string expr;

  RewriteCandidate(double cost, double accuracy, std::string expression);
};

struct SolutionStep {
  std::variant<ApplicableOutput *, ApplicableFPCC *> item;
  size_t candidateIndex;

  SolutionStep(ApplicableOutput *ao_, size_t idx);
  SolutionStep(ApplicableFPCC *acc_, size_t idx);
};

// Classes
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
  double geoMean;
  double arithMean;
  double maxAbs;
  unsigned executions;

  explicit FPNode(const std::string &op, const std::string &dtype)
      : ntype(NodeType::Node), op(op), dtype(dtype) {}
  explicit FPNode(NodeType ntype, const std::string &op,
                  const std::string &dtype)
      : ntype(ntype), op(op), dtype(dtype) {}
  virtual ~FPNode() = default;

  NodeType getType() const;
  void addOperand(std::shared_ptr<FPNode> operand);
  virtual bool hasSymbol() const;
  virtual std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0);
  unsigned getMPFRPrec() const;
  virtual void markAsInput();
  virtual void updateBounds(double lower, double upper);
  virtual double getLowerBound() const;
  virtual double getUpperBound() const;
  virtual Value *getLLValue(IRBuilder<> &builder,
                            const ValueToValueMapTy *VMap = nullptr);
};

class FPLLValue : public FPNode {
private:
  double lb = std::numeric_limits<double>::infinity();
  double ub = -std::numeric_limits<double>::infinity();
  bool input = false;

public:
  Value *value;

  explicit FPLLValue(Value *value, const std::string &op,
                     const std::string &dtype)
      : FPNode(NodeType::LLValue, op, dtype), value(value) {}

  bool hasSymbol() const override;
  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0) override;
  void markAsInput() override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
  Value *getLLValue(IRBuilder<> &builder,
                    const ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

class FPConst : public FPNode {
private:
  std::string strValue;

public:
  explicit FPConst(const std::string &strValue, const std::string &dtype)
      : FPNode(NodeType::Const, "__const", dtype), strValue(strValue) {}

  std::string toFullExpression(
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      unsigned depth = 0) override;
  bool hasSymbol() const override;
  void markAsInput() override;
  void updateBounds(double lower, double upper) override;
  double getLowerBound() const override;
  double getUpperBound() const override;
  Value *getLLValue(IRBuilder<> &builder,
                    const ValueToValueMapTy *VMap = nullptr) override;

  static bool classof(const FPNode *N);
};

class FPEvaluator {
private:
  std::unordered_map<const FPNode *, double> cache;
  std::unordered_map<const FPNode *, PrecisionChangeType> nodePrecisions;

public:
  FPEvaluator(PTCandidate *pt = nullptr);

  PrecisionChangeType getNodePrecision(const FPNode *node) const;
  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues);
  double getResult(const FPNode *node) const;
};

class MPFREvaluator {
private:
  struct CachedValue {
    mpfr_t value;
    unsigned prec;

    CachedValue(unsigned prec);
    CachedValue(const CachedValue &) = delete;
    CachedValue &operator=(const CachedValue &) = delete;
    CachedValue(CachedValue &&other) noexcept;
    CachedValue &operator=(CachedValue &&other) noexcept;
    virtual ~CachedValue();
  };

  std::unordered_map<const FPNode *, CachedValue> cache;
  unsigned prec;
  std::unordered_map<const FPNode *, unsigned> nodeToNewPrec;

public:
  MPFREvaluator(unsigned prec, PTCandidate *pt = nullptr);
  virtual ~MPFREvaluator() = default;

  unsigned getNodePrecision(const FPNode *node, bool groundTruth) const;
  void evaluateNode(const FPNode *node,
                    const MapVector<Value *, double> &inputValues,
                    bool groundTruth);
  mpfr_t &getResult(FPNode *node);
};

class ApplicableOutput {
public:
  FPCC *component;
  Value *oldOutput;
  std::string expr;
  double grad = std::numeric_limits<double>::quiet_NaN();
  unsigned executions = 0;
  const TargetTransformInfo *TTI = nullptr;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost initialCompCost =
      std::numeric_limits<InstructionCost>::quiet_NaN();
  double initialHerbieCost = std::numeric_limits<double>::quiet_NaN();
  double initialHerbieAccuracy = std::numeric_limits<double>::quiet_NaN();
  SmallVector<RewriteCandidate> candidates;
  SmallPtrSet<Instruction *, 8> erasableInsts;

  explicit ApplicableOutput(FPCC &component, Value *oldOutput, std::string expr,
                            double grad, unsigned executions,
                            const TargetTransformInfo &TTI);

  void
  apply(size_t candidateIndex,
        std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
        std::unordered_map<std::string, Value *> &symbolToValueMap);
  InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);

private:
  void findErasableInstructions();
};

class ApplicableFPCC {
public:
  FPCC *component;
  const TargetTransformInfo &TTI;
  double initialAccCost = std::numeric_limits<double>::quiet_NaN();
  InstructionCost initialCompCost =
      std::numeric_limits<InstructionCost>::quiet_NaN();
  unsigned executions = 0;
  std::unordered_map<FPNode *, double> perOutputInitialAccCost;
  SmallVector<PTCandidate, 8> candidates;

  using ApplicableOutputSet = std::set<ApplicableOutput *>;
  struct CacheKey {
    size_t candidateIndex;
    ApplicableOutputSet applicableOutputs;
    bool operator==(const CacheKey &other) const;
  };

  struct CacheKeyHash {
    std::size_t operator()(const CacheKey &key) const;
  };

  std::unordered_map<CacheKey, InstructionCost, CacheKeyHash>
      compCostDeltaCache;
  std::unordered_map<CacheKey, double, CacheKeyHash> accCostDeltaCache;

  explicit ApplicableFPCC(FPCC &fpcc, const TargetTransformInfo &TTI);

  void apply(size_t candidateIndex);
  InstructionCost getCompCostDelta(size_t candidateIndex);
  double getAccCostDelta(size_t candidateIndex);
  InstructionCost
  getAdjustedCompCostDelta(size_t candidateIndex,
                           const SmallVectorImpl<SolutionStep> &steps);
  double getAdjustedAccCostDelta(
      size_t candidateIndex, SmallVectorImpl<SolutionStep> &steps,
      std::unordered_map<Value *, std::shared_ptr<FPNode>> &valueToNodeMap,
      std::unordered_map<std::string, Value *> &symbolToValueMap);
};

struct GradInfo {
  double geoMean;
  double arithMean;
  double maxAbs;

  GradInfo() : geoMean(0.0), arithMean(0.0), maxAbs(0.0) {}
};

struct ValueInfo {
  double minRes;
  double maxRes;
  unsigned executions;
  double geoMean;
  double arithMean;
  double maxAbs;

  SmallVector<double, 2> minOperands;
  SmallVector<double, 2> maxOperands;

  ValueInfo()
      : minRes(std::numeric_limits<double>::max()),
        maxRes(std::numeric_limits<double>::lowest()), executions(0),
        geoMean(0.0), arithMean(0.0), maxAbs(0.0) {}
};

class FPOpt final : public FunctionPass {
public:
  static char ID;
  FPOpt();

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;
};

llvm::FunctionPass *createFPOptPass();

class FPOptNewPM final : public llvm::AnalysisInfoMixin<FPOptNewPM> {
  friend struct llvm::AnalysisInfoMixin<FPOptNewPM>;

private:
  static llvm::AnalysisKey Key;

public:
  using Result = llvm::PreservedAnalyses;
  FPOptNewPM() {}

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM);

  static bool isRequired() { return true; }
};

#endif // ENZYME_POSEIDON_H
