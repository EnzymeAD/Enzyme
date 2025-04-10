#include "Dialect/Ops.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/BreadthFirstIterator.h"

#include "EnzymeLogic.h"
#include "GradientUtils.h"

using namespace mlir;
using namespace mlir::enzyme;

//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

FunctionOpInterface
mlir::enzyme::MEnzymeLogic::CreateTrace(FunctionOpInterface fn,
                                        MTypeAnalysis &TA, bool freeMemory,
                                        MFnTypeInfo type_args) {
  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Tracing empty function");
  }

  // Assume the same trace object base type as the traced function return type.
  auto traceType = enzyme::TraceType::get(
      fn.getContext(),
      fn.getFunctionType().cast<mlir::FunctionType>().getResult(0));

  auto originalInputs =
      fn.getFunctionType().cast<mlir::FunctionType>().getInputs();
  SmallVector<mlir::Type, 4> ArgTypes(originalInputs.begin(),
                                      originalInputs.end());
  ArgTypes.insert(ArgTypes.begin(), traceType);

  auto originalResults =
      fn.getFunctionType().cast<mlir::FunctionType>().getResults();
  SmallVector<mlir::Type, 4> RetTypes(originalResults.begin(),
                                      originalResults.end());
  RetTypes.insert(RetTypes.begin(), traceType);

  OpBuilder builder(fn.getContext());
  auto FTy = builder.getFunctionType(ArgTypes, RetTypes);

  auto NewF = cast<FunctionOpInterface>(fn->cloneWithoutRegions());
  SymbolTable::setSymbolName(NewF, fn.getName().str() + ".trace");
  NewF.setType(FTy);

  Operation *parent = fn->getParentWithTrait<OpTrait::SymbolTable>();
  SymbolTable table(parent);
  table.insert(NewF);
  // SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;

  cloneInto(&fn.getFunctionBody(), &NewF.getFunctionBody(), originalToNew,
            originalToNewOps);

  // Ensure the execution trace is passed through.
  for (auto &block : NewF.getFunctionBody()) {
    block.insertArgument(block.args_begin(), traceType,
                         block.getTerminator()->getLoc());

    OpBuilder b(&block, block.end());
    auto term = block.getTerminator();
    llvm::errs() << "Handling terminator: " << *term << "\n";

    SmallVector<Value, 4> newOperands(term->getOperands().begin(),
                                      term->getOperands().end());
    newOperands.insert(newOperands.begin(), block.getArgument(0));
    auto retloc = block.getTerminator()->getLoc();

    if (auto brOp = dyn_cast<cf::BranchOp>(term)) {
      brOp->replaceAllUsesWith(
          b.create<cf::BranchOp>(retloc, brOp.getDest(), newOperands));
    } else if (auto retOp = dyn_cast<func::ReturnOp>(term)) {
      retOp->replaceAllUsesWith(b.create<func::ReturnOp>(retloc, newOperands));
    } else {
      fn.emitError() << "Unsupported terminator found in traced function: "
                     << *term;
      return nullptr;
    }

    term->erase();
  }

  return NewF;
}
