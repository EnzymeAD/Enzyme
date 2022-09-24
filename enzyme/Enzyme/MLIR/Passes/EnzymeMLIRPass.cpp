//===- EnzymeMLIRPass.cpp - //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
//#include "PassDetails.h"

#include "../../EnzymeLogic.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace enzyme;

class MFnTypeInfo {};
class MTypeAnalysis {
public:
  MFnTypeInfo getAnalyzedTypeInfo(FunctionOpInterface op) const {}
};

Type getShadowType(Type T, unsigned width) { return T; }

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, DerivativeMode mode, unsigned width,
    mlir::Type additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE returnType) {
  SmallVector<mlir::Type, 4> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::Return) {
    assert(FTy.getNumResults() == 1);
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  } else if (returnValue == ReturnType::ArgsWithTwoReturns ||
             returnValue == ReturnType::TwoReturns) {
    assert(FTy.getNumResults() == 1);
    RetTypes.push_back(FTy.getResult(0));
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  }

  SmallVector<mlir::Type, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto I : FTy.getInputs()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(I, width));
    }
    ++argno;
  }

  if (diffeReturnArg) {
    ArgTypes.push_back(getShadowType(FTy.getResult(0), width));
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }

  OpBuilder builder(FTy.getContext());
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.insert(RetTypes.begin(),
                    LLVM::LLVMPointerType::get(builder.getIntegerType(8)));
  }

  // Create a new function type...
  return builder.getFunctionType(ArgTypes, RetTypes);
}

mlir::func::FuncOp CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnType returnValue,
    DIFFE_TYPE returnType, Twine name, BlockAndValueMapping &VMap,
    bool diffeReturnArg, mlir::Type additionalArg) {
  assert(!F.getBody().empty());
  // F = preprocessForClone(F, mode);
  // llvm::ValueToValueMapTy VMap;
  auto FTy = getFunctionTypeForClone(
      F.getFunctionType().cast<mlir::FunctionType>(), mode, width,
      additionalArg, constant_args, diffeReturnArg, returnValue, returnType);

  /*
  for (Block &BB : F.getBody().getBlocks()) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (auto rv = ri->getReturnValue()) {
        returnvals.insert(rv);
      }
    }
  }
  */

  // Create the new function...
  auto NewF = mlir::func::FuncOp::create(F.getLoc(), name.str(), FTy);
  ((Operation *)F)->getParentOfType<ModuleOp>().push_back(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  F.getBody().cloneInto(&NewF.getBody(), VMap);

  {
    unsigned ii = 0;
    auto &blk = NewF.getBody().front();
    for (ssize_t i = constant_args.size() - 1; i >= 0; i--) {
      if (constant_args[i] == DIFFE_TYPE::DUP_ARG ||
          constant_args[i] == DIFFE_TYPE::DUP_NONEED) {
        if (i == constant_args.size() - 1)
          blk.addArgument(blk.getArgument(i).getType(),
                          blk.getArgument(i).getLoc());
        else
          blk.insertArgument(blk.args_begin() + i + 1,
                             blk.getArgument(i).getType(),
                             blk.getArgument(i).getLoc());
      }
    }
  }

  return NewF;
}

class MEnzymeLogic {
public:
  mlir::func::FuncOp
  CreateForwardDiff(FunctionOpInterface fn, DIFFE_TYPE retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    bool returnUsed, DerivativeMode mode, bool freeMemory,
                    size_t width, mlir::Type addedType, MFnTypeInfo type_args,
                    std::vector<bool> volatile_args, void *augmented) {
    if (fn.getBody().empty()) {
      llvm::errs() << fn << "\n";
      llvm_unreachable("Differentiating empty function");
    }

    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      prefix = "fwddiffe";
      break;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      prefix = "diffe";
      break;
    case DerivativeMode::ReverseModePrimal:
      llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
    }

    if (width > 1)
      prefix += std::to_string(width);

    bool diffeReturnArg = false;
    mlir::Type additionalArg = nullptr;
    BlockAndValueMapping originalToNew;

    bool retActive = retType != DIFFE_TYPE::CONSTANT;
    ReturnType returnValue =
        returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                   : (retActive ? ReturnType::Return : ReturnType::Void);

    SmallPtrSet<mlir::Value, 1> returnvals;
    auto newFunc = CloneFunctionWithReturns(
        mode, width, fn, /*invertedPointers, */ constants, /* constant_values,*/
        /*nonconstant_values,*/ returnvals, returnValue, retType,
        prefix + fn.getName(), originalToNew,
        /*diffeReturnArg*/ diffeReturnArg, additionalArg);

    // TODO derivative internal handling

    return newFunc;
  }
};

namespace {
struct DifferentiatePass : public DifferentiatePassBase<DifferentiatePass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  template <typename T>
  void HandleAutoDiff(SymbolTableCollection &symbolTable, T CI) {
    std::vector<DIFFE_TYPE> constants;
    SmallVector<mlir::Value, 2> args;

    size_t truei = 0;
    auto activityAttr = CI.getActivity();

    for (unsigned i = 0; i < CI.getInputs().size(); ++i) {
      mlir::Value res = CI.getInputs()[i];

      auto mop = activityAttr[truei];
      auto iattr = cast<mlir::enzyme::ActivityAttr>(mop);
      DIFFE_TYPE ty = (DIFFE_TYPE)(iattr.getValue());

      constants.push_back(ty);
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;
        res = CI.getInputs()[i];
        args.push_back(res);
      }

      truei++;
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    DIFFE_TYPE retType =
        fn.getNumResults() == 0 ? DIFFE_TYPE::CONSTANT : DIFFE_TYPE::DUP_ARG;

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    auto mode = DerivativeMode::ForwardMode;
    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getBody().getArguments()) {
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    auto newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr);

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc, args);
    CI.replaceAllUsesWith(dCI);
    CI->erase();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    SmallVector<Operation *> toLower;
    op->walk([&](enzyme::ForwardDiffOp dop) {
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto callableOp = cast<FunctionOpInterface>(symbolOp);

      lowerEnzymeCalls(symbolTable, callableOp);
      toLower.push_back(dop);
    });

    for (auto T : toLower) {
      if (auto F = dyn_cast<enzyme::ForwardDiffOp>(T)) {
        HandleAutoDiff(symbolTable, F);
      } else {
        llvm_unreachable("Illegal type");
      }
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createDifferentiatePass() {
  new DifferentiatePass();
  return std::make_unique<DifferentiatePass>();
}
} // namespace enzyme
} // namespace mlir

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

void DifferentiatePass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  ConversionPatternRewriter B(getOperation()->getContext());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeCalls(symbolTable, op); });
}
