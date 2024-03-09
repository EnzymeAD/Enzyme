#pragma once

#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/BreadthFirstIterator.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "CloneFunction.h"
#include "EnzymeLogic.h"

using namespace mlir;
using namespace mlir::enzyme;

Type getShadowType(Type type, unsigned width = 1);

mlir::FunctionType
getFunctionTypeForClone(mlir::FunctionType FTy, DerivativeMode mode,
                        unsigned width, mlir::Type additionalArg,
                        llvm::ArrayRef<bool> returnPrimals,
                        llvm::ArrayRef<bool> returnShadows,
                        llvm::ArrayRef<DIFFE_TYPE> ReturnActivity,
                        llvm::ArrayRef<DIFFE_TYPE> ArgActivity);

void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               IRMapping &mapper, std::map<Operation *, Operation *> &opMap);

void cloneInto(Region *src, Region *dest, IRMapping &mapper,
               std::map<mlir::Operation *, mlir::Operation *> &opMap);

Operation *clone(Operation *src, IRMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap);

FunctionOpInterface CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    IRMapping &ptrInputs, ArrayRef<DIFFE_TYPE> ArgActivity,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals,
    const std::vector<bool> &returnPrimals,
    const std::vector<bool> &returnShadows, ArrayRef<DIFFE_TYPE> ReturnActivity,
    Twine name, IRMapping &VMap, std::map<Operation *, Operation *> &OpMap,
    mlir::Type additionalArg);
