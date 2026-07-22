//===- EnzymeMLIR.cpp - C API for Enzyme MLIR dialect ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EnzymeMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/CAPI/Registration.h"

#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Passes/Passes.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Enzyme, enzyme,
                                      mlir::enzyme::EnzymeDialect)

MLIR_CAPI_EXPORTED void enzymeRegisterPasses(void) {
  mlir::enzyme::registerDifferentiatePass();
  mlir::enzyme::registerExpandImpulsePass();
  mlir::enzyme::registerBatchPass();
  mlir::enzyme::registerBatchDiffPass();
  mlir::enzyme::registerDifferentiateWrapperPass();
  mlir::enzyme::registerInlineEnzymeIntoRegionPass();
  mlir::enzyme::registerOutlineEnzymeFromRegionPass();
  mlir::enzyme::registerPrintActivityAnalysisPass();
  mlir::enzyme::registerPrintAliasAnalysisPass();
  mlir::enzyme::registerRemoveUnusedEnzymeOpsPass();
}

MLIR_CAPI_EXPORTED void
enzymeRegisterDialectExtensions(MlirDialectRegistry registry) {
  mlir::enzyme::registerCoreDialectAutodiffInterfaces(*unwrap(registry));
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateDifferentiatePass(void) {
  return wrap(mlir::enzyme::createDifferentiatePass().release());
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateDifferentiatePassWithOptions(
    MlirStringRef postpasses, bool verifyPostPasses) {
  mlir::enzyme::DifferentiatePassOptions opts;
  opts.postpasses = std::string(postpasses.data, postpasses.length);
  opts.verifyPostPasses = verifyPostPasses;
  return wrap(mlir::enzyme::createDifferentiatePass(opts).release());
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateConvertEnzymeToMemRefPass(void) {
  return wrap(mlir::enzyme::createEnzymeOpsToMemRefPass().release());
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateBatchPass(void) {
  return wrap(mlir::enzyme::createBatchPass().release());
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateBatchDiffPass(void) {
  return wrap(mlir::enzyme::createBatchDiffPass().release());
}

MLIR_CAPI_EXPORTED MlirPass enzymeCreateRemoveUnusedEnzymeOpsPass(void) {
  return wrap(mlir::enzyme::createRemoveUnusedEnzymeOpsPass().release());
}

MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx,
                                                       uint32_t activity) {
  return wrap(mlir::enzyme::ActivityAttr::get(
      unwrap(ctx), (mlir::enzyme::Activity)activity));
}

static mlir::ArrayAttr
activityArrayAttr(MlirContext ctx, const MlirAttribute *activity, intptr_t n) {
  return mlir::ArrayAttr::get(
      unwrap(ctx), llvm::ArrayRef<mlir::Attribute>(
                       reinterpret_cast<const mlir::Attribute *>(activity), n));
}

MLIR_CAPI_EXPORTED MlirOperation enzymeAutoDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc) {
  auto op =
      mlir::OpBuilder(unwrap(ctx))
          .create<mlir::enzyme::AutoDiffOp>(
              unwrap(loc),
              mlir::TypeRange(llvm::ArrayRef<mlir::Type>(
                  reinterpret_cast<const mlir::Type *>(resultTypes), nResults)),
              llvm::StringRef(fn.data, fn.length),
              mlir::ValueRange(llvm::ArrayRef<mlir::Value>(
                  reinterpret_cast<const mlir::Value *>(inputs), nInputs)),
              activityArrayAttr(ctx, activity, nActivity),
              activityArrayAttr(ctx, retActivity, nRetActivity),
              (uint64_t)width, strongZero);
  return wrap(op.getOperation());
}

MLIR_CAPI_EXPORTED MlirOperation enzymeForwardDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc) {
  auto op =
      mlir::OpBuilder(unwrap(ctx))
          .create<mlir::enzyme::ForwardDiffOp>(
              unwrap(loc),
              mlir::TypeRange(llvm::ArrayRef<mlir::Type>(
                  reinterpret_cast<const mlir::Type *>(resultTypes), nResults)),
              llvm::StringRef(fn.data, fn.length),
              mlir::ValueRange(llvm::ArrayRef<mlir::Value>(
                  reinterpret_cast<const mlir::Value *>(inputs), nInputs)),
              activityArrayAttr(ctx, activity, nActivity),
              activityArrayAttr(ctx, retActivity, nRetActivity),
              (uint64_t)width, strongZero);
  return wrap(op.getOperation());
}

MLIR_CAPI_EXPORTED MlirOperation enzymeJacobianOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc) {
  auto op =
      mlir::OpBuilder(unwrap(ctx))
          .create<mlir::enzyme::JacobianOp>(
              unwrap(loc),
              mlir::TypeRange(llvm::ArrayRef<mlir::Type>(
                  reinterpret_cast<const mlir::Type *>(resultTypes), nResults)),
              llvm::StringRef(fn.data, fn.length),
              mlir::ValueRange(llvm::ArrayRef<mlir::Value>(
                  reinterpret_cast<const mlir::Value *>(inputs), nInputs)),
              activityArrayAttr(ctx, activity, nActivity),
              activityArrayAttr(ctx, retActivity, nRetActivity),
              (uint64_t)width, strongZero);
  return wrap(op.getOperation());
}

MLIR_CAPI_EXPORTED MlirOperation enzymeBatchOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const int64_t *batchShape, intptr_t nBatchShape, MlirLocation loc) {
  auto op =
      mlir::OpBuilder(unwrap(ctx))
          .create<mlir::enzyme::BatchOp>(
              unwrap(loc),
              mlir::TypeRange(llvm::ArrayRef<mlir::Type>(
                  reinterpret_cast<const mlir::Type *>(resultTypes), nResults)),
              llvm::StringRef(fn.data, fn.length),
              mlir::ValueRange(llvm::ArrayRef<mlir::Value>(
                  reinterpret_cast<const mlir::Value *>(inputs), nInputs)),
              llvm::ArrayRef<int64_t>(batchShape, nBatchShape));
  return wrap(op.getOperation());
}

MLIR_CAPI_EXPORTED MlirAttribute
enzymeRngDistributionAttrGet(MlirContext ctx, EnzymeRngDistribution dist) {
  mlir::impulse::RngDistribution rngDist;
  switch (dist) {
  case EnzymeRngDistribution_Uniform:
    rngDist = mlir::impulse::RngDistribution::UNIFORM;
    break;
  case EnzymeRngDistribution_Normal:
    rngDist = mlir::impulse::RngDistribution::NORMAL;
    break;
  case EnzymeRngDistribution_MultiNormal:
    rngDist = mlir::impulse::RngDistribution::MULTINORMAL;
    break;
  }
  return wrap(mlir::impulse::RngDistributionAttr::get(unwrap(ctx), rngDist));
}

MLIR_CAPI_EXPORTED MlirAttribute enzymeSupportAttrGet(
    MlirContext ctx, EnzymeSupportKind kind, bool hasLowerBound,
    double lowerBound, bool hasUpperBound, double upperBound) {
  auto *mlirCtx = unwrap(ctx);

  mlir::impulse::SupportKind supportKind;
  switch (kind) {
  case EnzymeSupportKind_Real:
    supportKind = mlir::impulse::SupportKind::REAL;
    break;
  case EnzymeSupportKind_Positive:
    supportKind = mlir::impulse::SupportKind::POSITIVE;
    break;
  case EnzymeSupportKind_UnitInterval:
    supportKind = mlir::impulse::SupportKind::UNIT_INTERVAL;
    break;
  case EnzymeSupportKind_Interval:
    supportKind = mlir::impulse::SupportKind::INTERVAL;
    break;
  case EnzymeSupportKind_GreaterThan:
    supportKind = mlir::impulse::SupportKind::GREATER_THAN;
    break;
  case EnzymeSupportKind_LessThan:
    supportKind = mlir::impulse::SupportKind::LESS_THAN;
    break;
  }

  mlir::FloatAttr lowerAttr;
  if (hasLowerBound)
    lowerAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), lowerBound);

  mlir::FloatAttr upperAttr;
  if (hasUpperBound)
    upperAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), upperBound);

  return wrap(mlir::impulse::SupportAttr::get(mlirCtx, supportKind, lowerAttr,
                                              upperAttr));
}

MLIR_CAPI_EXPORTED MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx,
                                                        double trajectoryLength,
                                                        bool adaptStepSize,
                                                        bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);
  auto trajectoryLengthAttr =
      mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), trajectoryLength);

  return wrap(mlir::impulse::HMCConfigAttr::get(
      mlirCtx, trajectoryLengthAttr, adaptStepSize, adaptMassMatrix));
}

MLIR_CAPI_EXPORTED MlirAttribute enzymeNUTSConfigAttrGet(
    MlirContext ctx, int64_t maxTreeDepth, bool hasMaxDeltaEnergy,
    double maxDeltaEnergy, bool adaptStepSize, bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);

  mlir::FloatAttr maxDeltaEnergyAttr;
  if (hasMaxDeltaEnergy)
    maxDeltaEnergyAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), maxDeltaEnergy);

  return wrap(mlir::impulse::NUTSConfigAttr::get(
      mlirCtx, maxTreeDepth, maxDeltaEnergyAttr, adaptStepSize,
      adaptMassMatrix));
}

MLIR_CAPI_EXPORTED MlirAttribute enzymeSymbolAttrGet(MlirContext ctx,
                                                     uint64_t ptr) {
  return wrap(mlir::impulse::SymbolAttr::get(unwrap(ctx), ptr));
}
