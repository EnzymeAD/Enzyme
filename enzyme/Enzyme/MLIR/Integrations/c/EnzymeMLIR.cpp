#include "EnzymeMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Enzyme, enzyme,
                                      mlir::enzyme::EnzymeDialect)

MlirAttribute enzymeActivityAttrGet(MlirContext ctx, EnzymeActivity activity) {
  mlir::enzyme::Activity act;
  switch (activity) {
  case EnzymeActivity_enzyme_active:
    act = mlir::enzyme::Activity::enzyme_active;
    break;
  case EnzymeActivity_enzyme_dup:
    act = mlir::enzyme::Activity::enzyme_dup;
    break;
  case EnzymeActivity_enzyme_const:
    act = mlir::enzyme::Activity::enzyme_const;
    break;
  case EnzymeActivity_enzyme_dupnoneed:
    act = mlir::enzyme::Activity::enzyme_dupnoneed;
    break;
  case EnzymeActivity_enzyme_activenoneed:
    act = mlir::enzyme::Activity::enzyme_activenoneed;
    break;
  case EnzymeActivity_enzyme_constnoneed:
    act = mlir::enzyme::Activity::enzyme_constnoneed;
    break;
  }
  return wrap(mlir::enzyme::ActivityAttr::get(unwrap(ctx), act));
}

static mlir::ArrayAttr activityArrayAttr(MlirContext ctx,
                                         MlirAttribute *activity, intptr_t n) {
  llvm::SmallVector<mlir::Attribute> attrs;
  attrs.reserve(n);
  for (intptr_t i = 0; i < n; ++i)
    attrs.push_back(unwrap(activity[i]));
  return mlir::ArrayAttr::get(unwrap(ctx), attrs);
}

static void collectTypes(MlirType *src, intptr_t n,
                         llvm::SmallVectorImpl<mlir::Type> &out) {
  out.reserve(n);
  for (intptr_t i = 0; i < n; ++i)
    out.push_back(unwrap(src[i]));
}

static void collectValues(MlirValue *src, intptr_t n,
                          llvm::SmallVectorImpl<mlir::Value> &out) {
  out.reserve(n);
  for (intptr_t i = 0; i < n; ++i)
    out.push_back(unwrap(src[i]));
}

MlirOperation
enzymeAutoDiffOpCreate(MlirContext ctx, MlirStringRef fn, MlirType *resultTypes,
                       intptr_t nResults, MlirValue *inputs, intptr_t nInputs,
                       MlirAttribute *activity, intptr_t nActivity,
                       MlirAttribute *retActivity, intptr_t nRetActivity,
                       int64_t width, bool strongZero, MlirLocation loc) {
  auto *mlirCtx = unwrap(ctx);
  llvm::SmallVector<mlir::Type> results;
  collectTypes(resultTypes, nResults, results);
  llvm::SmallVector<mlir::Value> operands;
  collectValues(inputs, nInputs, operands);
  auto op = mlir::OpBuilder(mlirCtx).create<mlir::enzyme::AutoDiffOp>(
      unwrap(loc), mlir::TypeRange(results),
      llvm::StringRef(fn.data, fn.length), mlir::ValueRange(operands),
      activityArrayAttr(ctx, activity, nActivity),
      activityArrayAttr(ctx, retActivity, nRetActivity), (uint64_t)width,
      strongZero);
  return wrap(op.getOperation());
}

MlirOperation enzymeForwardDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, MlirType *resultTypes, intptr_t nResults,
    MlirValue *inputs, intptr_t nInputs, MlirAttribute *activity,
    intptr_t nActivity, MlirAttribute *retActivity, intptr_t nRetActivity,
    int64_t width, bool strongZero, MlirLocation loc) {
  auto *mlirCtx = unwrap(ctx);
  llvm::SmallVector<mlir::Type> results;
  collectTypes(resultTypes, nResults, results);
  llvm::SmallVector<mlir::Value> operands;
  collectValues(inputs, nInputs, operands);
  auto op = mlir::OpBuilder(mlirCtx).create<mlir::enzyme::ForwardDiffOp>(
      unwrap(loc), mlir::TypeRange(results),
      llvm::StringRef(fn.data, fn.length), mlir::ValueRange(operands),
      activityArrayAttr(ctx, activity, nActivity),
      activityArrayAttr(ctx, retActivity, nRetActivity), (uint64_t)width,
      strongZero);
  return wrap(op.getOperation());
}

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
                                   bool hasUpperBound, double upperBound) {
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

MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx, double trajectoryLength,
                                     bool adaptStepSize, bool adaptMassMatrix) {
  auto *mlirCtx = unwrap(ctx);
  auto trajectoryLengthAttr =
      mlir::FloatAttr::get(mlir::Float64Type::get(mlirCtx), trajectoryLength);

  return wrap(mlir::impulse::HMCConfigAttr::get(
      mlirCtx, trajectoryLengthAttr, adaptStepSize, adaptMassMatrix));
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

MlirAttribute enzymeSymbolAttrGet(MlirContext ctx, uint64_t ptr) {
  return wrap(mlir::impulse::SymbolAttr::get(unwrap(ctx), ptr));
}
