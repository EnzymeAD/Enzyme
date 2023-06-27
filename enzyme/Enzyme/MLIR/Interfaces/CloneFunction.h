#pragma once

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/BreadthFirstIterator.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace mlir::enzyme;

Type getShadowType(Type type, unsigned width = 1);

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, DerivativeMode mode, unsigned width,
    mlir::Type additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE ReturnType);

void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               IRMapping &mapper, std::map<Operation *, Operation *> &opMap);

void cloneInto(Region *src, Region *dest, IRMapping &mapper,
               std::map<mlir::Operation *, mlir::Operation *> &opMap);

Operation *clone(Operation *src, IRMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap);

FunctionOpInterface CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    IRMapping &ptrInputs, ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnType returnValue,
    DIFFE_TYPE ReturnType, Twine name, IRMapping &VMap,
    std::map<Operation *, Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg);