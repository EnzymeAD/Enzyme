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
// otherwise supply. The handler registered under `<key>` (a callee's name or
// an `enzyme_math` value) calls `@<key>_jvp` instead of differentiating the
// callee; that function takes every operand followed by its shadow (a zero one
// for a constant operand) and returns every result followed by its shadow.
// With `<key>:tangent` the call's clone stays to compute the primal results,
// and `@<key>_tangent` takes the same arguments and returns only the shadows.
// With `<key>:primal-first` the primal results are first replaced by those of
// `@<key>_primal`, which takes the operands only, and `@<key>_tangent` is
// called after that.
enum class TestForwardCallHandlerKind { JVP, Tangent, PrimalFirst };

static FailureOr<func::CallOp> callTestRule(Operation *orig, OpBuilder &builder,
                                            StringRef name, TypeRange types,
                                            ValueRange args) {
  if (!SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(
          orig, StringAttr::get(orig->getContext(), name))) {
    orig->emitError() << "test forward call handler: no function @" << name;
    return failure();
  }
  return func::CallOp::create(builder, orig->getLoc(), name, types, args);
}

static LogicalResult testForwardCallHandler(Operation *orig, OpBuilder &builder,
                                            enzyme::MGradientUtils *gutils,
                                            StringRef key,
                                            TestForwardCallHandlerKind kind) {
  SmallVector<Value> primalArgs, args;
  for (Value operand : orig->getOperands()) {
    Value primal = gutils->getNewFromOriginal(operand);
    primalArgs.push_back(primal);
    args.push_back(primal);
    args.push_back(gutils->invertPointerM(operand, builder));
  }
  SmallVector<Type> types, shadowTypes, jvpTypes;
  for (Value res : orig->getResults()) {
    types.push_back(res.getType());
    shadowTypes.push_back(gutils->getShadowType(res.getType()));
    jvpTypes.push_back(types.back());
    jvpTypes.push_back(shadowTypes.back());
  }

  SmallVector<Value> primals, shadows;
  switch (kind) {
  case TestForwardCallHandlerKind::JVP: {
    auto call =
        callTestRule(orig, builder, (key + "_jvp").str(), jvpTypes, args);
    if (failed(call))
      return failure();
    for (unsigned i = 0, e = orig->getNumResults(); i < e; i++) {
      primals.push_back(call->getResult(2 * i));
      shadows.push_back(call->getResult(2 * i + 1));
    }
    break;
  }
  case TestForwardCallHandlerKind::Tangent: {
    auto call = callTestRule(orig, builder, (key + "_tangent").str(),
                             shadowTypes, args);
    if (failed(call))
      return failure();
    llvm::append_range(shadows, call->getResults());
    break;
  }
  case TestForwardCallHandlerKind::PrimalFirst: {
    auto primalCall =
        callTestRule(orig, builder, (key + "_primal").str(), types, primalArgs);
    if (failed(primalCall))
      return failure();
    enzyme::replaceForwardCall(orig, primalCall->getResults(), builder, gutils);
    auto call = callTestRule(orig, builder, (key + "_tangent").str(),
                             shadowTypes, args);
    if (failed(call))
      return failure();
    llvm::append_range(shadows, call->getResults());
    break;
  }
  }

  for (auto &&[res, shadow] : llvm::zip_equal(orig->getResults(), shadows))
    if (!gutils->isConstantValue(res))
      gutils->setDiffe(res, shadow, builder);
  if (kind == TestForwardCallHandlerKind::JVP)
    enzyme::replaceForwardCall(orig, primals, builder, gutils);
  return success();
}

static llvm::cl::list<std::string> TestForwardCallHandlers(
    "enzyme-test-forward-call-handler", llvm::cl::Hidden,
    llvm::cl::desc("Register a test forward-mode call handler under <key> "
                   "(or <key>:tangent, <key>:primal-first)"),
    llvm::cl::callback([](const std::string &spec) {
      StringRef key = spec;
      auto kind = TestForwardCallHandlerKind::JVP;
      if (key.consume_back(":tangent"))
        kind = TestForwardCallHandlerKind::Tangent;
      else if (key.consume_back(":primal-first"))
        kind = TestForwardCallHandlerKind::PrimalFirst;
      std::string name = key.str();
      enzyme::registerForwardCallHandler(
          key, [name, kind](Operation *orig, OpBuilder &builder,
                            enzyme::MGradientUtils *gutils) {
            return testForwardCallHandler(orig, builder, gutils, name, kind);
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
