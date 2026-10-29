#include "EnzymeMLIR.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Builders.h"

#include "Dialect/Dialect.h"
#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/GradientUtils.h"

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

//===----------------------------------------------------------------------===//
// Custom forward-mode derivatives of calls
//===----------------------------------------------------------------------===//

struct EnzymeMLIRGradientUtils {
  mlir::Operation *orig;
  mlir::OpBuilder &builder;
  mlir::enzyme::MGradientUtils *gutils;
};

void enzymeRegisterForwardCallHandler(MlirStringRef callee,
                                      EnzymeMLIRForwardCallHandler handler,
                                      void *userData) {
  mlir::enzyme::registerForwardCallHandler(
      unwrap(callee),
      [handler, userData](mlir::Operation *orig, mlir::OpBuilder &builder,
                          mlir::enzyme::MGradientUtils *gutils) {
        EnzymeMLIRGradientUtils ref{orig, builder, gutils};
        return unwrap(handler(wrap(orig), &ref, userData));
      });
}

void enzymeUnregisterForwardCallHandler(MlirStringRef callee) {
  mlir::enzyme::unregisterForwardCallHandler(unwrap(callee));
}

int64_t enzymeGradientUtilsGetWidth(EnzymeMLIRGradientUtilsRef gutils) {
  return gutils->gutils->width;
}

bool enzymeGradientUtilsIsConstantValue(EnzymeMLIRGradientUtilsRef gutils,
                                        MlirValue value) {
  return gutils->gutils->isConstantValue(unwrap(value));
}

MlirValue
enzymeGradientUtilsGetNewFromOriginal(EnzymeMLIRGradientUtilsRef gutils,
                                      MlirValue value) {
  return wrap(gutils->gutils->getNewFromOriginal(unwrap(value)));
}

MlirValue enzymeGradientUtilsInvertPointer(EnzymeMLIRGradientUtilsRef gutils,
                                           MlirValue value) {
  return wrap(gutils->gutils->invertPointerM(unwrap(value), gutils->builder));
}

MlirType enzymeGradientUtilsGetShadowType(EnzymeMLIRGradientUtilsRef gutils,
                                          MlirType type) {
  return wrap(gutils->gutils->getShadowType(unwrap(type)));
}

void enzymeGradientUtilsInsertOperation(EnzymeMLIRGradientUtilsRef gutils,
                                        MlirOperation op) {
  gutils->builder.insert(unwrap(op));
}

void enzymeGradientUtilsSetDiffe(EnzymeMLIRGradientUtilsRef gutils,
                                 MlirValue result, MlirValue shadow) {
  gutils->gutils->setDiffe(unwrap(result), unwrap(shadow), gutils->builder);
}

void enzymeGradientUtilsReplaceOrigOpWith(EnzymeMLIRGradientUtilsRef gutils,
                                          intptr_t nValues,
                                          MlirValue const *values) {
  assert(static_cast<unsigned>(nValues) == gutils->orig->getNumResults() &&
         "one value per result of the call");
  llvm::SmallVector<mlir::Value> vals;
  for (intptr_t i = 0; i < nValues; i++)
    vals.push_back(unwrap(values[i]));
  mlir::Operation *newOp = gutils->gutils->getNewFromOriginal(gutils->orig);
  gutils->gutils->replaceOrigOpWith(gutils->orig, vals);
  gutils->gutils->erase(newOp);
}
