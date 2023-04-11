//===- GradientUtils.h - Utilities for gradient interfaces -------* C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/ActivityAnalysis.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/FunctionInterfaces.h"

// TODO: no relative includes.
#include "../../EnzymeLogic.h"

namespace mlir {
namespace enzyme {

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
    if (val.getType().isa<IntegerType, IndexType>()) {
      return BaseType::Integer;
    }
    if (errIfNotFound) {
      llvm_unreachable("something happened");
    }
    return BaseType::Unknown;
  }
};

class MEnzymeLogic;

class MGradientUtils {
public:
  // From CacheUtility
  FunctionOpInterface newFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  FunctionOpInterface oldFunc;
  BlockAndValueMapping invertedPointers;
  BlockAndValueMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  SmallPtrSet<Block *, 4> blocksNotForAnalysis;
  std::unique_ptr<enzyme::ActivityAnalyzer> activityAnalyzer;

  MTypeAnalysis &TA;
  MTypeResults TR;
  bool omp;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const;
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const;
  Operation *getNewFromOriginal(Operation *originst) const;

  MGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 MTypeResults TR_, BlockAndValueMapping &invertedPointers_,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 DIFFE_TYPE ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                 BlockAndValueMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeMode mode, unsigned width, bool omp);
  void erase(Operation *op) { op->erase(); }
  void eraseIfUnused(Operation *op, bool erase = true, bool check = true) {
    // TODO
  }
  bool isConstantValue(mlir::Value v) const;
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &Builder2);
  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM);
  void forceAugmentedReturns();

  Operation *cloneWithNewOperands(OpBuilder &B, Operation *op);

  LogicalResult visitChild(Operation *op);
};

class MEnzymeLogic {
public:
  MDiffeGradientUtils(MEnzymeLogic &Logic, FunctionOpInterface newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {}

      if (additionalType.getImpl() < rhs.additionalType.getImpl())
        return true;
      if (rhs.additionalType.getImpl() < additionalType.getImpl())
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;
      // equal
      return false;
    }
  };

  std::map<MForwardCacheKey, FunctionOpInterface> ForwardCachedFunctions;

    BlockAndValueMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    BlockAndValueMapping invertedPointers;
    FunctionOpInterface newFunc = CloneFunctionWithReturns(
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);
    MTypeResults TR; // TODO
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

} // namespace enzyme
} // namespace mlir
