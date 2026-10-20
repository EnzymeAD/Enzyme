#include "EnzymeMLIR.h"

#include "mlir/CAPI/IR.h"

#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"

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

MlirAttribute enzymeSupportAttrGet(MlirContext ctx, EnzymeSupportKind kind,
                                   bool hasLowerBound, double lowerBound,
                                   bool hasUpperBound, double upperBound,
                                   int64_t eventSize) {
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
  case EnzymeSupportKind_Simplex:
    supportKind = mlir::impulse::SupportKind::SIMPLEX;
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

  mlir::IntegerAttr eventSizeAttr;
  if (eventSize > 0)
    eventSizeAttr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(mlirCtx, 64), eventSize);

  return wrap(mlir::impulse::SupportAttr::get(mlirCtx, supportKind, lowerAttr,
                                              upperAttr, eventSizeAttr));
}

MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx, double trajectoryLength,
                                     bool adaptStepSize, bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);
  auto trajectoryLengthAttr =
      mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), trajectoryLength);

  return wrap(mlir::impulse::HMCConfigAttr::get(mlirCtx, trajectoryLengthAttr,
                                                adaptStepSize, adaptMassMatrix,
                                                /*num_steps=*/0));
}

MlirAttribute enzymeHMCConfigAttrGetWithNumSteps(MlirContext ctx,
                                                 double trajectoryLength,
                                                 bool adaptStepSize,
                                                 bool adaptMassMatrix,
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
    MlirContext ctx, int32_t guideKind, int32_t optimizerKind,
    int64_t numParticles, bool hasInitScale, double initScale, bool hasAdamB1,
    double adamB1, bool hasAdamB2, double adamB2, bool hasAdamEps,
    double adamEps, bool hasClipNorm, double clipNorm, bool hasInitRadius,
    double initRadius, bool hasMomentumMass, double momentumMass,
    bool hasRmspropGamma, double rmspropGamma, bool hasRmspropEps,
    double rmspropEps, bool stableUpdate) {
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

MlirAttribute enzymeSymbolAttrGet(MlirContext ctx, uint64_t ptr) {
  return wrap(mlir::impulse::SymbolAttr::get(unwrap(ctx), ptr));
}
