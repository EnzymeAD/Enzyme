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
  EnzymeSupportKind_Simplex = 6,
} EnzymeSupportKind;

MLIR_CAPI_EXPORTED MlirAttribute
enzymeSupportAttrGet(MlirContext ctx, EnzymeSupportKind kind,
                     bool hasLowerBound, double lowerBound, bool hasUpperBound,
                     double upperBound, int64_t eventSize);

MLIR_CAPI_EXPORTED MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx,
                                                        double trajectoryLength,
                                                        bool adaptStepSize,
                                                        bool adaptMassMatrix);

MLIR_CAPI_EXPORTED MlirAttribute enzymeNUTSConfigAttrGet(
    MlirContext ctx, int64_t maxTreeDepth, bool hasMaxDeltaEnergy,
    double maxDeltaEnergy, bool adaptStepSize, bool adaptMassMatrix);

typedef enum {
  EnzymeGuideKind_AutoNormal = 0,
  EnzymeGuideKind_AutoMultivariateNormal = 1,
  EnzymeGuideKind_Custom = 2,
  EnzymeGuideKind_AutoDelta = 3,
} EnzymeGuideKind;

typedef enum {
  EnzymeOptimizerKind_Adam = 0,
  EnzymeOptimizerKind_SGD = 1,
  EnzymeOptimizerKind_ClippedAdam = 2,
  EnzymeOptimizerKind_Momentum = 3,
  EnzymeOptimizerKind_RMSProp = 4,
  EnzymeOptimizerKind_Adagrad = 5,
  EnzymeOptimizerKind_RMSPropMomentum = 6,
} EnzymeOptimizerKind;

MLIR_CAPI_EXPORTED MlirAttribute enzymeVIConfigAttrGet(
    MlirContext ctx, int32_t guideKind, int32_t optimizerKind,
    int64_t numParticles, bool hasInitScale, double initScale, bool hasAdamB1,
    double adamB1, bool hasAdamB2, double adamB2, bool hasAdamEps,
    double adamEps, bool hasClipNorm, double clipNorm, bool hasInitRadius,
    double initRadius, bool hasMomentumMass, double momentumMass,
    bool hasRmspropGamma, double rmspropGamma, bool hasRmspropEps,
    double rmspropEps, bool stableUpdate);

MLIR_CAPI_EXPORTED MlirAttribute enzymeMHConfigAttrGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute enzymeSymbolAttrGet(MlirContext ctx,
                                                     uint64_t ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
