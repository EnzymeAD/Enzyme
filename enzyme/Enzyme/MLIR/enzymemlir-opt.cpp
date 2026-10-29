//===- enzymemlir-opt.cpp - The enzymemlir-opt driver ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'enzymemlir-opt' tool, which is the enzyme analog
// of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/LLVMExt/LLVMExt.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/GradientUtils.h"
#include "Passes/Passes.h"

#include "Dialect/Ops.h"

#include "llvm/Support/CommandLine.h"

using namespace mlir;

// For tests of registered forward call handlers, which a frontend would
// otherwise supply. The handler registered for `<callee>` calls
// `@<callee>_jvp` instead of differentiating `<callee>`; that function takes
// every operand followed by its shadow (a zero one for a constant operand) and
// returns every result followed by its shadow. With `<callee>:tangent` the
// call's clone stays to compute the primal results, and `@<callee>_tangent`
// takes the same arguments and returns only the shadows.
static LogicalResult testForwardCallHandler(Operation *orig, OpBuilder &builder,
                                            enzyme::MGradientUtils *gutils,
                                            StringRef rule, bool keepPrimal) {
  auto ruleFn = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
      orig, StringAttr::get(orig->getContext(), rule));
  if (!ruleFn)
    return orig->emitError()
           << "test forward call handler: no function @" << rule;

  SmallVector<Value> args;
  for (Value operand : orig->getOperands()) {
    args.push_back(gutils->getNewFromOriginal(operand));
    args.push_back(gutils->invertPointerM(operand, builder));
  }
  SmallVector<Type> resultTypes;
  for (Value res : orig->getResults()) {
    if (!keepPrimal)
      resultTypes.push_back(res.getType());
    resultTypes.push_back(gutils->getShadowType(res.getType()));
  }
  auto call =
      func::CallOp::create(builder, orig->getLoc(), rule, resultTypes, args);

  SmallVector<Value> primals;
  unsigned idx = 0;
  for (Value res : orig->getResults()) {
    if (!keepPrimal)
      primals.push_back(call.getResult(idx++));
    Value shadow = call.getResult(idx++);
    if (!gutils->isConstantValue(res))
      gutils->setDiffe(res, shadow, builder);
  }
  if (!keepPrimal) {
    Operation *newOp = gutils->getNewFromOriginal(orig);
    gutils->replaceOrigOpWith(orig, primals);
    gutils->erase(newOp);
  }
  return success();
}

static llvm::cl::list<std::string> TestForwardCallHandlers(
    "enzyme-test-forward-call-handler", llvm::cl::Hidden,
    llvm::cl::desc("Register a test forward-mode handler for calls to "
                   "<callee> (or <callee>:tangent)"),
    llvm::cl::callback([](const std::string &spec) {
      StringRef callee = spec;
      bool keepPrimal = callee.consume_back(":tangent");
      std::string rule = (callee + (keepPrimal ? "_tangent" : "_jvp")).str();
      enzyme::registerForwardCallHandler(
          callee, [rule, keepPrimal](Operation *orig, OpBuilder &builder,
                                     enzyme::MGradientUtils *gutils) {
            return testForwardCallHandler(orig, builder, gutils, rule,
                                          keepPrimal);
          });
    }));

class MemRefInsider
    : public mlir::MemRefElementTypeInterface::FallbackModel<MemRefInsider> {};

template <typename T>
struct PtrElementModel
    : public mlir::LLVM::PointerElementTypeInterface::ExternalModel<
          PtrElementModel<T>, T> {};

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::async::AsyncDialect>();
  registry.insert<mlir::complex::ComplexDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::gpu::GPUDialect>();
  registry.insert<mlir::NVVM::NVVMDialect>();
  registry.insert<mlir::omp::OpenMPDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::ub::UBDialect>();
  registry.insert<DLTIDialect>();

  registry.insert<mlir::enzyme::EnzymeDialect>();
  registry.insert<mlir::impulse::ImpulseDialect>();
  registry.insert<mlir::enzyme::llvm_ext::LLVMExtDialect>();

  mlir::enzyme::registerenzymePasses();

  mlir::func::registerInlinerExtension(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerMem2RegPass();
  mlir::registerLowerAffinePass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerSymbolDCEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerConvertSCFToOpenMPPass();
  mlir::registerSCFToControlFlowPass();
  mlir::affine::registerAffinePasses();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerLinalgPasses();

  registry.addExtension(+[](MLIRContext *ctx, LLVM::LLVMDialect *dialect) {
    LLVM::LLVMFunctionType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMArrayType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMPointerType::attachInterface<MemRefInsider>(*ctx);
    LLVM::LLVMStructType::attachInterface<MemRefInsider>(*ctx);
    MemRefType::attachInterface<PtrElementModel<MemRefType>>(*ctx);
    LLVM::LLVMStructType::attachInterface<
        PtrElementModel<LLVM::LLVMStructType>>(*ctx);
    LLVM::LLVMPointerType::attachInterface<
        PtrElementModel<LLVM::LLVMPointerType>>(*ctx);
    LLVM::LLVMArrayType::attachInterface<PtrElementModel<LLVM::LLVMArrayType>>(
        *ctx);
  });

  // Register the autodiff interface implementations for upstream dialects.
  enzyme::registerCoreDialectAutodiffInterfaces(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Enzyme modular optimizer driver", registry));
}
