//===- EnzymeMLIR.cpp - C API for Enzyme MLIR dialect ---------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EnzymeMLIR.h"
#include "EnzymeMLIRPasses.h"

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

using namespace mlir::enzyme;

extern "C" {
#include "Passes/Passes.capi.cpp.inc"
}

void enzymeRegisterDialectExtensions(MlirDialectRegistry registry) {
  mlir::enzyme::registerCoreDialectAutodiffInterfaces(*unwrap(registry));
}

MlirPass enzymeCreateDifferentiatePassWithOptions(MlirStringRef postpasses,
                                                  bool verifyPostPasses) {
  mlir::enzyme::DifferentiatePassOptions opts;
  opts.postpasses = std::string(postpasses.data, postpasses.length);
  opts.verifyPostPasses = verifyPostPasses;
  return wrap(mlir::enzyme::createDifferentiatePass(opts).release());
}

MlirAttribute enzymeActivityAttrGet(MlirContext ctx, uint32_t activity) {
  return wrap(mlir::enzyme::ActivityAttr::get(
      unwrap(ctx), (mlir::enzyme::Activity)activity));
}

static mlir::ArrayAttr
activityArrayAttr(MlirContext ctx, const MlirAttribute *activity, intptr_t n) {
  return mlir::ArrayAttr::get(
      unwrap(ctx), llvm::ArrayRef<mlir::Attribute>(
                       reinterpret_cast<const mlir::Attribute *>(activity), n));
}

MlirOperation
enzymeAutoDiffOpCreate(MlirContext ctx, MlirStringRef fn,
                       const MlirType *resultTypes, intptr_t nResults,
                       const MlirValue *inputs, intptr_t nInputs,
                       const MlirAttribute *activity, intptr_t nActivity,
                       const MlirAttribute *retActivity, intptr_t nRetActivity,
                       int64_t width, bool strongZero, MlirLocation loc) {
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

MlirOperation enzymeForwardDiffOpCreate(
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

MlirOperation
enzymeJacobianOpCreate(MlirContext ctx, MlirStringRef fn,
                       const MlirType *resultTypes, intptr_t nResults,
                       const MlirValue *inputs, intptr_t nInputs,
                       const MlirAttribute *activity, intptr_t nActivity,
                       const MlirAttribute *retActivity, intptr_t nRetActivity,
                       int64_t width, bool strongZero, MlirLocation loc) {
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

MlirOperation enzymeBatchOpCreate(MlirContext ctx, MlirStringRef fn,
                                  const MlirType *resultTypes,
                                  intptr_t nResults, const MlirValue *inputs,
                                  intptr_t nInputs, const int64_t *batchShape,
                                  intptr_t nBatchShape, MlirLocation loc) {
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

static_assert(static_cast<int>(EnzymeRngDistribution_Uniform) ==
                  static_cast<int>(mlir::impulse::RngDistribution::UNIFORM),
              "EnzymeRngDistribution out of sync with ODS RngDistribution");
static_assert(static_cast<int>(EnzymeRngDistribution_Normal) ==
                  static_cast<int>(mlir::impulse::RngDistribution::NORMAL),
              "EnzymeRngDistribution out of sync with ODS RngDistribution");
static_assert(static_cast<int>(EnzymeRngDistribution_MultiNormal) ==
                  static_cast<int>(mlir::impulse::RngDistribution::MULTINORMAL),
              "EnzymeRngDistribution out of sync with ODS RngDistribution");

static_assert(static_cast<int>(EnzymeSupportKind_Real) ==
                  static_cast<int>(mlir::impulse::SupportKind::REAL),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_Positive) ==
                  static_cast<int>(mlir::impulse::SupportKind::POSITIVE),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_UnitInterval) ==
                  static_cast<int>(mlir::impulse::SupportKind::UNIT_INTERVAL),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_Interval) ==
                  static_cast<int>(mlir::impulse::SupportKind::INTERVAL),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_GreaterThan) ==
                  static_cast<int>(mlir::impulse::SupportKind::GREATER_THAN),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_LessThan) ==
                  static_cast<int>(mlir::impulse::SupportKind::LESS_THAN),
              "EnzymeSupportKind out of sync with ODS SupportKind");
static_assert(static_cast<int>(EnzymeSupportKind_Simplex) ==
                  static_cast<int>(mlir::impulse::SupportKind::SIMPLEX),
              "EnzymeSupportKind out of sync with ODS SupportKind");

static_assert(static_cast<int>(EnzymeGuideKind_AutoNormal) ==
                  static_cast<int>(mlir::impulse::GuideKind::AUTO_NORMAL),
              "EnzymeGuideKind out of sync with ODS GuideKind");
static_assert(
    static_cast<int>(EnzymeGuideKind_AutoMultivariateNormal) ==
        static_cast<int>(mlir::impulse::GuideKind::AUTO_MULTIVARIATE_NORMAL),
    "EnzymeGuideKind out of sync with ODS GuideKind");
static_assert(static_cast<int>(EnzymeGuideKind_Custom) ==
                  static_cast<int>(mlir::impulse::GuideKind::CUSTOM),
              "EnzymeGuideKind out of sync with ODS GuideKind");
static_assert(static_cast<int>(EnzymeGuideKind_AutoDelta) ==
                  static_cast<int>(mlir::impulse::GuideKind::AUTO_DELTA),
              "EnzymeGuideKind out of sync with ODS GuideKind");

static_assert(static_cast<int>(EnzymeOptimizerKind_Adam) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::ADAM),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(static_cast<int>(EnzymeOptimizerKind_SGD) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::SGD),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(static_cast<int>(EnzymeOptimizerKind_ClippedAdam) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::CLIPPED_ADAM),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(static_cast<int>(EnzymeOptimizerKind_Momentum) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::MOMENTUM),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(static_cast<int>(EnzymeOptimizerKind_RMSProp) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::RMSPROP),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(static_cast<int>(EnzymeOptimizerKind_Adagrad) ==
                  static_cast<int>(mlir::impulse::OptimizerKind::ADAGRAD),
              "EnzymeOptimizerKind out of sync with ODS OptimizerKind");
static_assert(
    static_cast<int>(EnzymeOptimizerKind_RMSPropMomentum) ==
        static_cast<int>(mlir::impulse::OptimizerKind::RMSPROP_MOMENTUM),
    "EnzymeOptimizerKind out of sync with ODS OptimizerKind");

MlirAttribute enzymeRngDistributionAttrGet(MlirContext ctx,
                                           EnzymeRngDistribution dist) {
  return wrap(mlir::impulse::RngDistributionAttr::get(
      unwrap(ctx), static_cast<mlir::impulse::RngDistribution>(dist)));
}

MlirAttribute enzymeSupportAttrGet(MlirContext ctx, EnzymeSupportKind kind,
                                   bool hasLowerBound, double lowerBound,
                                   bool hasUpperBound, double upperBound,
                                   int64_t eventSize) {
  auto *mlirCtx = unwrap(ctx);

  mlir::FloatAttr lowerAttr;
  if (hasLowerBound)
    lowerAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), lowerBound);

  mlir::FloatAttr upperAttr;
  if (hasUpperBound)
    upperAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), upperBound);

  mlir::IntegerAttr eventSizeAttr;
  if (eventSize > 0)
    eventSizeAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), eventSize);

  return wrap(mlir::impulse::SupportAttr::get(
      mlirCtx, static_cast<mlir::impulse::SupportKind>(kind), lowerAttr,
      upperAttr, eventSizeAttr));
}

MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx, double trajectoryLength,
                                     bool adaptStepSize, bool adaptMassMatrix,
                                     int64_t numSteps) {
  auto *mlirCtx = unwrap(ctx);
  auto trajectoryLengthAttr =
      mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), trajectoryLength);
  return wrap(mlir::impulse::HMCConfigAttr::get(
      mlirCtx, trajectoryLengthAttr, adaptStepSize, adaptMassMatrix, numSteps));
}

MlirAttribute enzymeNUTSConfigAttrGet(MlirContext ctx, int64_t maxTreeDepth,
                                      bool hasMaxDeltaEnergy,
                                      double maxDeltaEnergy, bool adaptStepSize,
                                      bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);

  mlir::FloatAttr maxDeltaEnergyAttr;
  if (hasMaxDeltaEnergy)
    maxDeltaEnergyAttr =
        mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), maxDeltaEnergy);

  return wrap(mlir::impulse::NUTSConfigAttr::get(
      mlirCtx, maxTreeDepth, maxDeltaEnergyAttr, adaptStepSize,
      adaptMassMatrix));
}

MlirAttribute enzymeVIConfigAttrGet(
    MlirContext ctx, EnzymeGuideKind guideKind,
    EnzymeOptimizerKind optimizerKind, int64_t numParticles, bool hasInitScale,
    double initScale, bool hasAdamB1, double adamB1, bool hasAdamB2,
    double adamB2, bool hasAdamEps, double adamEps, bool hasClipNorm,
    double clipNorm, bool hasInitRadius, double initRadius,
    bool hasMomentumMass, double momentumMass, bool hasRmspropGamma,
    double rmspropGamma, bool hasRmspropEps, double rmspropEps,
    bool stableUpdate) {
  auto *mlirCtx = unwrap(ctx);
  auto f64Ty = mlir::Float64Type::get(mlirCtx);

  auto guideAttr = mlir::impulse::GuideKindAttr::get(
      mlirCtx, static_cast<mlir::impulse::GuideKind>(guideKind));
  auto optimAttr = mlir::impulse::OptimizerKindAttr::get(
      mlirCtx, static_cast<mlir::impulse::OptimizerKind>(optimizerKind));

  auto maybeAttr = [&](bool has, double val) -> mlir::FloatAttr {
    if (!has)
      return {};
    return mlir::FloatAttr::get(f64Ty, val);
  };

  return wrap(mlir::impulse::VIConfigAttr::get(
      mlirCtx, guideAttr, optimAttr, numParticles,
      maybeAttr(hasInitScale, initScale), maybeAttr(hasAdamB1, adamB1),
      maybeAttr(hasAdamB2, adamB2), maybeAttr(hasAdamEps, adamEps),
      maybeAttr(hasClipNorm, clipNorm), maybeAttr(hasInitRadius, initRadius),
      maybeAttr(hasMomentumMass, momentumMass),
      maybeAttr(hasRmspropGamma, rmspropGamma),
      maybeAttr(hasRmspropEps, rmspropEps), stableUpdate));
}

MlirAttribute enzymeMHConfigAttrGet(MlirContext ctx) {
  return wrap(mlir::impulse::MHConfigAttr::get(unwrap(ctx)));
}

MlirAttribute enzymeGibbsConfigAttrGet(MlirContext ctx) {
  return wrap(mlir::impulse::GibbsConfigAttr::get(unwrap(ctx)));
}

MlirAttribute enzymeSymbolAttrGet(MlirContext ctx, uint64_t ptr) {
  return wrap(mlir::impulse::SymbolAttr::get(unwrap(ctx), ptr));
}
