#ifndef ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
#define ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Enzyme, enzyme);

//===----------------------------------------------------------------------===//
// Activity attribute
//===----------------------------------------------------------------------===//

typedef enum {
  EnzymeActivity_enzyme_active = 0,
  EnzymeActivity_enzyme_dup = 1,
  EnzymeActivity_enzyme_const = 2,
  EnzymeActivity_enzyme_dupnoneed = 3,
  EnzymeActivity_enzyme_activenoneed = 4,
  EnzymeActivity_enzyme_constnoneed = 5,
} EnzymeActivity;

MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx,
                                                       EnzymeActivity activity);

//===----------------------------------------------------------------------===//
// Core AD ops
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED MlirOperation enzymeAutoDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, MlirType *resultTypes, intptr_t nResults,
    MlirValue *inputs, intptr_t nInputs, MlirAttribute *activity,
    intptr_t nActivity, MlirAttribute *retActivity, intptr_t nRetActivity,
    int64_t width, bool strongZero, MlirLocation loc);

MLIR_CAPI_EXPORTED MlirOperation enzymeForwardDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, MlirType *resultTypes, intptr_t nResults,
    MlirValue *inputs, intptr_t nInputs, MlirAttribute *activity,
    intptr_t nActivity, MlirAttribute *retActivity, intptr_t nRetActivity,
    int64_t width, bool strongZero, MlirLocation loc);

//===----------------------------------------------------------------------===//
// Probabilistic Programming Ops
//===----------------------------------------------------------------------===//

typedef enum {
  EnzymeRngDistribution_Uniform = 0,
  EnzymeRngDistribution_Normal = 1,
  EnzymeRngDistribution_MultiNormal = 2,
} EnzymeRngDistribution;

MLIR_CAPI_EXPORTED MlirAttribute
enzymeRngDistributionAttrGet(MlirContext ctx, EnzymeRngDistribution dist);

typedef enum {
  EnzymeSupportKind_Real = 0,
  EnzymeSupportKind_Positive = 1,
  EnzymeSupportKind_UnitInterval = 2,
  EnzymeSupportKind_Interval = 3,
  EnzymeSupportKind_GreaterThan = 4,
  EnzymeSupportKind_LessThan = 5,
} EnzymeSupportKind;

MLIR_CAPI_EXPORTED MlirAttribute enzymeSupportAttrGet(
    MlirContext ctx, EnzymeSupportKind kind, bool hasLowerBound,
    double lowerBound, bool hasUpperBound, double upperBound);

MLIR_CAPI_EXPORTED MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx,
                                                        double trajectoryLength,
                                                        bool adaptStepSize,
                                                        bool adaptMassMatrix);

MLIR_CAPI_EXPORTED MlirAttribute enzymeNUTSConfigAttrGet(
    MlirContext ctx, int64_t maxTreeDepth, bool hasMaxDeltaEnergy,
    double maxDeltaEnergy, bool adaptStepSize, bool adaptMassMatrix);

MLIR_CAPI_EXPORTED MlirAttribute enzymeSymbolAttrGet(MlirContext ctx,
                                                     uint64_t ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
