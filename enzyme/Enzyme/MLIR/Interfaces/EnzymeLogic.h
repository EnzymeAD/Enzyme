#pragma once

#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "../../TypeAnalysis/TypeAnalysis.h"
#include "../../Utils.h"
#include <functional>

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

class MTypeResults {
public:
  // TODO
  TypeTree getReturnAnalysis() { return TypeTree(); }
  TypeTree query(Value) const { return TypeTree(); }
  ConcreteType intType(size_t num, Value val, bool errIfNotFound = true,
                       bool pointerIntSame = false) const {
    if (isa<IntegerType, IndexType>(val.getType())) {
      return BaseType::Integer;
    }
    if (errIfNotFound) {
      llvm_unreachable("something happened");
    }
    return BaseType::Unknown;
  }
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

  FunctionOpInterface
  CreateForwardDiff(FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    std::vector<bool> returnPrimals, DerivativeMode mode,
                    bool freeMemory, size_t width, mlir::Type addedType,
                    MFnTypeInfo type_args, std::vector<bool> volatile_args,
                    void *augmented, bool omp, llvm::StringRef postpasses,
                    bool verifyPostPasses, bool strongZero);

  FunctionOpInterface
  CreateReverseDiff(FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    std::vector<bool> returnPrimals,
                    std::vector<bool> returnShadows, DerivativeMode mode,
                    bool freeMemory, size_t width, mlir::Type addedType,
                    MFnTypeInfo type_args, std::vector<bool> volatile_args,
                    void *augmented, bool omp, llvm::StringRef postpasses,
                    bool verifyPostPasses, bool strongZero);

  FlatSymbolRefAttr
  CreateSplitModeDiff(FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
                      std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                      std::vector<bool> returnPrimals,
                      std::vector<bool> returnShadows, DerivativeMode mode,
                      bool freeMemory, size_t width, mlir::Type addedType,
                      MFnTypeInfo type_args, std::vector<bool> volatile_args,
                      void *augmented, bool omp, llvm::StringRef postpasses,
                      bool verifyPostPasses, bool strongZero);

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
