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

//===----------------------------------------------------------------------===//
// Custom forward-mode derivatives of calls
//===----------------------------------------------------------------------===//

/// Derivative being built; valid only while a forward call handler runs.
typedef struct EnzymeMLIRGradientUtils *EnzymeMLIRGradientUtilsRef;

/// Builds the forward-mode derivative of `call` before its clone. It must set
/// the shadow of every active result, and may replace the primal results with
/// enzymeGradientUtilsReplaceOrigOpWith.
typedef MlirLogicalResult (*EnzymeMLIRForwardCallHandler)(
    MlirOperation call, EnzymeMLIRGradientUtilsRef gutils, void *userData);

/// Registers `handler` under `key` (see mlir::enzyme::forwardCallHandlers).
/// Not synchronized. `userData` is passed to the handler.
MLIR_CAPI_EXPORTED void enzymeRegisterForwardCallHandler(
    MlirStringRef key, EnzymeMLIRForwardCallHandler handler, void *userData);

MLIR_CAPI_EXPORTED int64_t
enzymeGradientUtilsGetWidth(EnzymeMLIRGradientUtilsRef gutils);

MLIR_CAPI_EXPORTED bool
enzymeGradientUtilsIsConstantValue(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirValue value);

MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsGetNewFromOriginal(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

/// Returns the shadow of `value`, zero if `value` is inactive.
MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsInvertPointer(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

MLIR_CAPI_EXPORTED MlirType enzymeGradientUtilsGetShadowType(
    EnzymeMLIRGradientUtilsRef gutils, MlirType type);

/// Inserts the detached `op` at the handler's insertion point and takes
/// ownership of it.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsInsertOperation(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirOperation op);

MLIR_CAPI_EXPORTED void
enzymeGradientUtilsSetDiffe(EnzymeMLIRGradientUtilsRef gutils, MlirValue result,
                            MlirValue shadow);

/// Replaces the call's results with `values` and erases its clone.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsReplaceOrigOpWith(EnzymeMLIRGradientUtilsRef gutils,
                                     intptr_t nValues, MlirValue const *values);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
