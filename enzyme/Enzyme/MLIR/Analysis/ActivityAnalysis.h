#ifndef ENZYME_MLIR_ANALYSIS_ACTIVITYANALYSIS_H
#define ENZYME_MLIR_ANALYSIS_ACTIVITYANALYSIS_H

#include "../../Utils.h"
#include "mlir/IR/Block.h"

namespace mlir {

class CallOpInterface;

namespace enzyme {

// class TypeResults {};

class MTypeResults;

/// Helper class to analyze the differential activity
class ActivityAnalyzer {
  // PreProcessCache &PPC;

  // TODO: MLIR aliasing Information
  // llvm::AAResults &AA;

  // Blocks not to be analyzed
  const llvm::SmallPtrSetImpl<Block *> &notForAnalysis;

  /// Library Information
  // llvm::TargetLibraryInfo &TLI;

public:
  /// Whether the returns of the function being analyzed are active
  const DIFFE_TYPE ActiveReturns;

private:
  /// Direction of current analysis
  const uint8_t directions;
  /// Analyze up based off of operands
  static constexpr uint8_t UP = 1;
  /// Analyze down based off uses
  static constexpr uint8_t DOWN = 2;

  /// Operations that don't propagate adjoints
  /// These operations could return an active pointer, but
  /// do not propagate adjoints themselves
  llvm::SmallPtrSet<Operation *, 4> ConstantOperations;

  /// Operations that could propagate adjoints
  llvm::SmallPtrSet<Operation *, 20> ActiveOperations;

  /// Values that do not contain derivative information, either
  /// directly or as a pointer to
  llvm::SmallPtrSet<Value, 4> ConstantValues;

  /// Values that may contain derivative information
  llvm::SmallPtrSet<Value, 2> ActiveValues;

  /// Intermediate pointers which are created by inactive instructions
  /// but are marked as active values to inductively determine their
  /// activity.
  llvm::SmallPtrSet<Value, 1> DeducingPointers;

public:
  /// Construct the analyzer from the a previous set of constant and active
  /// values and whether returns are active. The all arguments of the functions
  /// being analyzed must be in the set of constant and active values, lest an
  /// error occur during analysis
  ActivityAnalyzer(
      // PreProcessCache &PPC, llvm::AAResults &AA_,
      const llvm::SmallPtrSetImpl<Block *> &notForAnalysis_,
      // llvm::TargetLibraryInfo &TLI_,
      const llvm::SmallPtrSetImpl<Value> &ConstantValues,
      const llvm::SmallPtrSetImpl<Value> &ActiveValues,
      DIFFE_TYPE ActiveReturns)
      : notForAnalysis(notForAnalysis_), ActiveReturns(ActiveReturns),
        directions(UP | DOWN),
        ConstantValues(ConstantValues.begin(), ConstantValues.end()),
        ActiveValues(ActiveValues.begin(), ActiveValues.end()) {}

  /// Return whether this operation is known not to propagate adjoints
  /// Note that operations could return an active pointer, but
  /// do not propagate adjoints themselves
  bool isConstantOperation(MTypeResults const &TR, Operation *op);

  /// Return whether this values is known not to contain derivative
  /// information, either directly or as a pointer to
  bool isConstantValue(MTypeResults const &TR, Value val);

private:
  DenseMap<Operation *, llvm::SmallPtrSet<Value, 4>>
      ReEvaluateValueIfInactiveOp;
  DenseMap<Value, llvm::SmallPtrSet<Value, 4>> ReEvaluateValueIfInactiveValue;
  DenseMap<Value, llvm::SmallPtrSet<Operation *, 4>>
      ReEvaluateOpIfInactiveValue;

  void InsertConstantOperation(MTypeResults const &TR, Operation *op);
  void InsertConstantValue(MTypeResults const &TR, Value V);

  /// Create a new analyzer starting from an existing Analyzer
  /// This is used to perform inductive assumptions
  ActivityAnalyzer(ActivityAnalyzer &Other, uint8_t directions)
      : notForAnalysis(Other.notForAnalysis),
        ActiveReturns(Other.ActiveReturns), directions(directions),
        ConstantOperations(Other.ConstantOperations),
        ActiveOperations(Other.ActiveOperations),
        ConstantValues(Other.ConstantValues), ActiveValues(Other.ActiveValues) {
    // DeducingPointers(Other.DeducingPointers) {
    assert(directions != 0);
    assert((directions & Other.directions) == directions);
    assert((directions & Other.directions) != 0);
  }

  /// Import known constants from an existing analyzer
  void insertConstantsFrom(MTypeResults const &TR,
                           ActivityAnalyzer &Hypothesis) {
    for (auto I : Hypothesis.ConstantOperations) {
      InsertConstantOperation(TR, I);
    }
    for (auto V : Hypothesis.ConstantValues) {
      InsertConstantValue(TR, V);
    }
  }

  /// Import known data from an existing analyzer
  void insertAllFrom(MTypeResults const &TR, ActivityAnalyzer &Hypothesis,
                     Value Orig) {
    insertConstantsFrom(TR, Hypothesis);
    for (auto I : Hypothesis.ActiveOperations) {
      bool inserted = ActiveOperations.insert(I).second;
      if (inserted && directions == 3) {
        ReEvaluateOpIfInactiveValue[Orig].insert(I);
      }
    }
    for (auto V : Hypothesis.ActiveValues) {
      bool inserted = ActiveValues.insert(V).second;
      if (inserted && directions == 3) {
        ReEvaluateValueIfInactiveValue[Orig].insert(V);
      }
    }
  }

  /// Is the use of value val as an argument of call CI known to be inactive
  bool isFunctionArgumentConstant(mlir::CallOpInterface CI, Value val);

  /// Is the value guaranteed to be inactive because of how it's produced.
  bool isValueInactiveFromOrigin(MTypeResults const &TR, Value val);
  /// Is the operation guaranteed to be inactive because of how its operands are
  /// produced.
  bool
  isOperationInactiveFromOrigin(MTypeResults const &TR, Operation *op,
                                llvm::Optional<unsigned> resultNo = llvm::None);

public:
  enum class UseActivity {
    // No Additional use activity info
    None = 0,

    // Only consider loads of memory
    OnlyLoads = 1,

    // Only consider active stores into
    OnlyStores = 2,

    // Only consider active stores and pointer-style loads
    OnlyNonPointerStores = 3,

    // Only consider any (active or not) stores into
    AllStores = 4
  };
  /// Is the value free of any active uses
  bool isValueInactiveFromUsers(MTypeResults const &TR, Value val,
                                UseActivity UA,
                                Operation **FoundInst = nullptr);

  /// Is the value potentially actively returned or stored
  bool isValueActivelyStoredOrReturned(MTypeResults const &TR, Value val,
                                       bool outside = false);

private:
  /// StoredOrReturnedCache acts as an inductive cache of results for
  /// isValueActivelyStoredOrReturned
  std::map<std::pair<bool, Value>, bool> StoredOrReturnedCache;
};

} // namespace enzyme

inline bool operator<(const Value &lhs, const Value &rhs) {
  return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
}
} // namespace mlir

#endif // ENZYME_MLIR_ANALYSIS_ACTIVITYANALYSIS_H
