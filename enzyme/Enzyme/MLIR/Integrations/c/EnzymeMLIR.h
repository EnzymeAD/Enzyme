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

/// The derivative under construction, as a forward call handler sees it. It is
/// valid only during the handler's invocation.
typedef struct EnzymeMLIRGradientUtils *EnzymeMLIRGradientUtilsRef;

/// Constructs the forward-mode derivative of `call`, an op of the function
/// being differentiated, in place of differentiating its callee. It sets the
/// shadow of every active result of `call` (enzymeGradientUtilsSetDiffe), and
/// either leaves the call's clone in the derivative to compute the primal
/// results or replaces them (enzymeGradientUtilsReplaceOrigOpWith). New ops go
/// before that clone, through enzymeGradientUtilsInsertOperation.
typedef MlirLogicalResult (*EnzymeMLIRForwardCallHandler)(
    MlirOperation call, EnzymeMLIRGradientUtilsRef gutils, void *userData);

/// Registers `handler` for the forward-mode derivative of every call op whose
/// callee is the flat symbol `callee`, replacing any handler registered for
/// it. The registry is process-wide; `userData` is passed to the handler.
MLIR_CAPI_EXPORTED void enzymeRegisterForwardCallHandler(
    MlirStringRef callee, EnzymeMLIRForwardCallHandler handler, void *userData);

MLIR_CAPI_EXPORTED void
enzymeUnregisterForwardCallHandler(MlirStringRef callee);

/// The number of tangents computed at once (the batch width).
MLIR_CAPI_EXPORTED int64_t
enzymeGradientUtilsGetWidth(EnzymeMLIRGradientUtilsRef gutils);

/// Whether `value`, of the original function, has no derivative.
MLIR_CAPI_EXPORTED bool
enzymeGradientUtilsIsConstantValue(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirValue value);

/// The value of the derivative that `value`, of the original function, became.
MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsGetNewFromOriginal(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

/// The shadow (tangent) of `value`, of the original function: zero for a
/// constant value.
MLIR_CAPI_EXPORTED MlirValue enzymeGradientUtilsInvertPointer(
    EnzymeMLIRGradientUtilsRef gutils, MlirValue value);

/// The type of the shadow of a value of type `type`.
MLIR_CAPI_EXPORTED MlirType enzymeGradientUtilsGetShadowType(
    EnzymeMLIRGradientUtilsRef gutils, MlirType type);

/// Inserts the detached operation `op` into the derivative, after the ops the
/// handler inserted before it. Ownership passes to the derivative.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsInsertOperation(EnzymeMLIRGradientUtilsRef gutils,
                                   MlirOperation op);

/// Sets `shadow` as the shadow of the active result `result` of the call.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsSetDiffe(EnzymeMLIRGradientUtilsRef gutils, MlirValue result,
                            MlirValue shadow);

/// Replaces the call's results in the derivative by `values` (one per result)
/// and erases the call's clone. Operations inserted afterwards go where the
/// clone was.
MLIR_CAPI_EXPORTED void
enzymeGradientUtilsReplaceOrigOpWith(EnzymeMLIRGradientUtilsRef gutils,
                                     intptr_t nValues, MlirValue const *values);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
