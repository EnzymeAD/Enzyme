#include "EnzymeMLIR.h"

#include "mlir/CAPI/IR.h"

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"

MlirType enzymeTraceTypeGet(MlirContext ctx) {
  return wrap(mlir::enzyme::TraceType::get(unwrap(ctx)));
}

MlirType enzymeConstraintTypeGet(MlirContext ctx) {
  return wrap(mlir::enzyme::ConstraintType::get(unwrap(ctx)));
}

MlirAttribute enzymeRngDistributionAttrGet(MlirContext ctx,
                                           EnzymeRngDistribution dist) {
  mlir::enzyme::RngDistribution rngDist;
  switch (dist) {
  case EnzymeRngDistribution_Uniform:
    rngDist = mlir::enzyme::RngDistribution::UNIFORM;
    break;
  case EnzymeRngDistribution_Normal:
    rngDist = mlir::enzyme::RngDistribution::NORMAL;
    break;
  case EnzymeRngDistribution_MultiNormal:
    rngDist = mlir::enzyme::RngDistribution::MULTINORMAL;
    break;
  }
  return wrap(mlir::enzyme::RngDistributionAttr::get(unwrap(ctx), rngDist));
}

MlirAttribute enzymeSupportAttrGet(MlirContext ctx, EnzymeSupportKind kind,
                                   bool hasLowerBound, double lowerBound,
                                   bool hasUpperBound, double upperBound) {
  auto *mlirCtx = unwrap(ctx);

  mlir::enzyme::SupportKind supportKind;
  switch (kind) {
  case EnzymeSupportKind_Real:
    supportKind = mlir::enzyme::SupportKind::REAL;
    break;
  case EnzymeSupportKind_Positive:
    supportKind = mlir::enzyme::SupportKind::POSITIVE;
    break;
  case EnzymeSupportKind_UnitInterval:
    supportKind = mlir::enzyme::SupportKind::UNIT_INTERVAL;
    break;
  case EnzymeSupportKind_Interval:
    supportKind = mlir::enzyme::SupportKind::INTERVAL;
    break;
  case EnzymeSupportKind_GreaterThan:
    supportKind = mlir::enzyme::SupportKind::GREATER_THAN;
    break;
  case EnzymeSupportKind_LessThan:
    supportKind = mlir::enzyme::SupportKind::LESS_THAN;
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

  return wrap(mlir::enzyme::SupportAttr::get(mlirCtx, supportKind, lowerAttr,
                                             upperAttr));
}

MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx, double trajectoryLength,
                                     bool adaptStepSize, bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);
  auto trajectoryLengthAttr =
      mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), trajectoryLength);

  return wrap(mlir::enzyme::HMCConfigAttr::get(mlirCtx, trajectoryLengthAttr,
                                               adaptStepSize, adaptMassMatrix));
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

  return wrap(mlir::enzyme::NUTSConfigAttr::get(
      mlirCtx, maxTreeDepth, maxDeltaEnergyAttr, adaptStepSize,
      adaptMassMatrix));
}

MlirAttribute enzymeSymbolAttrGet(MlirContext ctx, uint64_t ptr) {
  return wrap(mlir::enzyme::SymbolAttr::get(unwrap(ctx), ptr));
}
