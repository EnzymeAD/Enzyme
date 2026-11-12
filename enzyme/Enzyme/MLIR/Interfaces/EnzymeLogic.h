#pragma once

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/CommandLine.h"

#include "../../TypeAnalysis/TypeTree.h"
#include "../../Utils.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace mlir {
namespace enzyme {

typedef void(buildReturnFunction)(OpBuilder &, mlir::Block *);

class MGradientUtilsReverse;

class MFnTypeInfo {
public:
  inline bool operator<(const MFnTypeInfo &rhs) const { return false; }
};

class MTypeAnalysis {
public:
  MFnTypeInfo getAnalyzedTypeInfo(FunctionOpInterface op) const {
    return MFnTypeInfo();
  }
};

} // namespace enzyme
} // namespace mlir

template <> struct ConcreteTypeTraits<mlir::Type> {
  static bool isFloatingPoint(mlir::Type T) { return static_cast<bool>(T); }

  static void print(llvm::raw_ostream &OS, mlir::Type T) { OS << T; }

  static std::string floatName(mlir::Type T) {
    std::string S;
    llvm::raw_string_ostream OS(S);
    OS << T;
    return S;
  }

  static bool less(mlir::Type A, mlir::Type B) {
    return A.getAsOpaquePointer() < B.getAsOpaquePointer();
  }
};

template <> struct TypeTreeTraits<mlir::Type> {
  using Origin = mlir::Operation *;

  template <typename T>
  static llvm::cl::opt<T> *registeredOption(llvm::StringRef Name) {
    auto &Opts = llvm::cl::getRegisteredOptions();
    auto It = Opts.find(Name);
    return It == Opts.end() ? nullptr
                            : static_cast<llvm::cl::opt<T> *>(It->second);
  }

  static unsigned maxDepth() {
    static auto *Opt = registeredOption<unsigned>("enzyme-max-type-depth");
    return Opt ? *Opt : 6;
  }

  static int maxOffset() {
    static auto *Opt = registeredOption<int>("enzyme-max-type-offset");
    return Opt ? *Opt : 500;
  }

  template <typename TreeT>
  static void depthLimitExceeded(const TreeT &Tree, Origin Orig,
                                 const llvm::Twine &Detail) {
    std::string Msg = ("not handling more than " + llvm::Twine(maxDepth()) +
                       " pointer lookups deep dt:" + Tree.str() + Detail)
                          .str();
    if (Orig)
      Orig->emitWarning(Msg);
    else
      llvm::errs() << Msg << "\n";
  }
};

namespace mlir {
namespace enzyme {

using MConcreteType = GenericConcreteType<mlir::Type>;

using MTypeTree = GenericTypeTree<mlir::Type>;

// The MLIR counterpart of TypeResults. There is no type analysis behind it.
class MTypeResults {
public:
  MTypeTree query(Value val) const;

  /// What the value itself is.
  MConcreteType intType(size_t num, Value val, bool errIfNotFound = true,
                        bool pointerIntSame = false) const;

  MConcreteType firstPointer(size_t num, Value val, bool errIfNotFound = true,
                             bool pointerIntSame = false) const;
};

class MEnzymeLogic {
public:
  struct MForwardCacheKey {
    FunctionOpInterface todiff;
    const std::vector<DIFFE_TYPE> retType;
    const std::vector<DIFFE_TYPE> constant_args;
    // std::map<llvm::Argument *, bool> uncacheable_args;
    std::vector<bool> returnUsed;
    DerivativeMode mode;
    unsigned width;
    mlir::Type additionalType;
    const MFnTypeInfo typeInfo;
    bool omp;
    bool strongZero;

    inline bool operator<(const MForwardCacheKey &rhs) const {
      if (todiff < rhs.todiff)
        return true;
      if (rhs.todiff < todiff)
        return false;

      if (retType < rhs.retType)
        return true;
      if (rhs.retType < retType)
        return false;

      if (std::lexicographical_compare(
              constant_args.begin(), constant_args.end(),
              rhs.constant_args.begin(), rhs.constant_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.constant_args.begin(), rhs.constant_args.end(),
              constant_args.begin(), constant_args.end()))
        return false;

      if (returnUsed < rhs.returnUsed)
        return true;
      if (rhs.returnUsed < returnUsed)
        return false;

      if (mode < rhs.mode)
        return true;
      if (rhs.mode < mode)
        return false;

      if (width < rhs.width)
        return true;
      if (rhs.width < width)
        return false;

      if (additionalType.getImpl() < rhs.additionalType.getImpl())
        return true;
      if (rhs.additionalType.getImpl() < additionalType.getImpl())
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;

      if (omp < rhs.omp)
        return true;
      if (rhs.omp < omp)
        return false;

      if (strongZero < rhs.strongZero)
        return true;
      if (rhs.strongZero < strongZero)
        return false;

      // equal
      return false;
    }
  };

  struct MReverseCacheKey {
    FunctionOpInterface todiff;
    const std::vector<DIFFE_TYPE> retActivity;
    const std::vector<DIFFE_TYPE> argActivity;
    const std::vector<bool> returnPrimals;
    const std::vector<bool> returnShadows;
    DerivativeMode mode;
    bool freeMemory;
    bool atomicAdd;
    unsigned width;
    mlir::Type additionalType;
    const MFnTypeInfo typeInfo;
    const std::vector<bool> volatileArgs;
    bool omp;
    bool strongZero;

    inline bool operator<(const MReverseCacheKey &rhs) const {
      if (todiff < rhs.todiff)
        return true;
      if (rhs.todiff < todiff)
        return false;

      if (std::lexicographical_compare(retActivity.begin(), retActivity.end(),
                                       rhs.retActivity.begin(),
                                       rhs.retActivity.end()))
        return true;
      if (std::lexicographical_compare(rhs.retActivity.begin(),
                                       rhs.retActivity.end(),
                                       retActivity.begin(), retActivity.end()))
        return false;

      if (std::lexicographical_compare(argActivity.begin(), argActivity.end(),
                                       rhs.argActivity.begin(),
                                       rhs.argActivity.end()))
        return true;
      if (std::lexicographical_compare(rhs.argActivity.begin(),
                                       rhs.argActivity.end(),
                                       argActivity.begin(), argActivity.end()))
        return false;

      if (returnPrimals < rhs.returnPrimals)
        return true;
      if (rhs.returnPrimals < rhs.returnPrimals)
        return false;

      if (returnShadows < rhs.returnShadows)
        return true;
      if (rhs.returnShadows < rhs.returnShadows)
        return false;

      if (mode < rhs.mode)
        return true;
      if (rhs.mode < mode)
        return false;

      if (freeMemory < rhs.freeMemory)
        return true;
      if (rhs.freeMemory < freeMemory)
        return false;

      if (width < rhs.width)
        return true;
      if (rhs.width < width)
        return false;

      if (additionalType.getImpl() < rhs.additionalType.getImpl())
        return true;
      if (rhs.additionalType.getImpl() < additionalType.getImpl())
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;

      if (volatileArgs < rhs.volatileArgs)
        return true;
      if (rhs.volatileArgs < volatileArgs)
        return false;

      if (omp < rhs.omp)
        return true;
      if (rhs.omp < omp)
        return false;

      if (strongZero < rhs.strongZero)
        return true;
      if (rhs.strongZero < strongZero)
        return false;

      // equal
      return false;
    }
  };

  std::map<MForwardCacheKey, FunctionOpInterface> ForwardCachedFunctions;
  std::map<MReverseCacheKey, FunctionOpInterface> ReverseCachedFunctions;

  // solver may be null, in which case the older inductive hypothesis activity
  // analysis will be used.
  std::unique_ptr<DataFlowSolver> solver;
  MEnzymeLogic(bool dataflowActivity);

  FunctionOpInterface
  CreateForwardDiff(FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    std::vector<bool> returnPrimals, DerivativeMode mode,
                    bool freeMemory, size_t width, mlir::Type addedType,
                    MFnTypeInfo type_args, std::vector<bool> overwritten_args,
                    void *augmented, bool omp, llvm::StringRef postpasses,
                    bool verifyPostPasses, bool strongZero);

  FunctionOpInterface
  CreateReverseDiff(FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    std::vector<bool> returnPrimals,
                    std::vector<bool> returnShadows, DerivativeMode mode,
                    bool freeMemory, bool atomicAdd, size_t width,
                    mlir::Type addedType, MFnTypeInfo type_args,
                    std::vector<bool> overwritten_args, void *augmented,
                    bool omp, llvm::StringRef postpasses, bool verifyPostPasses,
                    bool strongZero, bool markReadonly);

  void
  initializeShadowValues(SmallVector<mlir::Block *> &dominatorToposortBlocks,
                         MGradientUtilsReverse *gutils);
  void
  handlePredecessors(Block *oBB, Block *newBB, Block *reverseBB,
                     MGradientUtilsReverse *gutils,
                     llvm::function_ref<buildReturnFunction> buildReturnOp);
  LogicalResult visitChildren(Block *oBB, Block *reverseBB,
                              MGradientUtilsReverse *gutils);
  LogicalResult visitChild(Operation *op, OpBuilder &builder,
                           MGradientUtilsReverse *gutils);
  void mapInvertArguments(Block *oBB, Block *reverseBB,
                          MGradientUtilsReverse *gutils);
  LogicalResult
  differentiate(MGradientUtilsReverse *gutils, Region &oldRegion,
                Region &newRegion,
                llvm::function_ref<buildReturnFunction> buildFuncRetrunOp,
                std::function<std::pair<Value, Value>(Type)> cacheCreator);
};

} // Namespace enzyme
} // Namespace mlir
