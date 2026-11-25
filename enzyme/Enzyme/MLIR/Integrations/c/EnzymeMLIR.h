//===- EnzymeMLIR.h - C API for Enzyme MLIR dialect ---------------*- C -*-===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to the Enzyme MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
#define ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_

#include <stdbool.h>
#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
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

/// Construct an `enzyme.activity` attribute from an activity code matching the
/// `Activity` enum ordinals (0=active, 1=dup, 2=const, 3=dupnoneed, ...).
MLIR_CAPI_EXPORTED MlirAttribute enzymeActivityAttrGet(MlirContext ctx,
                                                       uint32_t activity);

//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

/// Register Enzyme's external model AD implementations for the core MLIR
/// dialects (arith, func, scf, ...) into the given DialectRegistry.
/// Must be called before appending the registry to an MLIRContext so that
/// the Differentiate pass knows how to lower ops in those dialects.
MLIR_CAPI_EXPORTED void
enzymeRegisterDialectExtensions(MlirDialectRegistry registry);

/// Create the core differentiation pass with options.
/// postpasses: semicolon-separated list of passes to run after differentiation
/// (may be NULL or empty). verifyPostPasses: run the verifier after each
/// post-pass.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateDifferentiatePassWithOptions(
    MlirStringRef postpasses, bool verifyPostPasses);

//===----------------------------------------------------------------------===//
// Core AD ops
//===----------------------------------------------------------------------===//

/// Construct an `enzyme.autodiff` (reverse-mode) op referencing the function
/// named `fn`. `activity`/`retActivity` are arrays of `enzyme.activity`
/// attributes of length `nActivity`/`nRetActivity`.
MLIR_CAPI_EXPORTED MlirOperation enzymeAutoDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc);

/// Construct an `enzyme.fwddiff` (forward-mode) op referencing the function
/// named `fn`. See `enzymeAutoDiffOpCreate` for parameter semantics.
MLIR_CAPI_EXPORTED MlirOperation enzymeForwardDiffOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc);

/// Construct an `enzyme.jacobian` op referencing the function named `fn`.
/// See `enzymeAutoDiffOpCreate` for parameter semantics.
MLIR_CAPI_EXPORTED MlirOperation enzymeJacobianOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const MlirAttribute *activity, intptr_t nActivity,
    const MlirAttribute *retActivity, intptr_t nRetActivity, int64_t width,
    bool strongZero, MlirLocation loc);

/// Construct an `enzyme.batch` op that maps the function named `fn` over a
/// batch of inputs. `batchShape` is an array of `nBatchShape` batch dimensions.
MLIR_CAPI_EXPORTED MlirOperation enzymeBatchOpCreate(
    MlirContext ctx, MlirStringRef fn, const MlirType *resultTypes,
    intptr_t nResults, const MlirValue *inputs, intptr_t nInputs,
    const int64_t *batchShape, intptr_t nBatchShape, MlirLocation loc);

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

/// `numSteps` = 0 derives the leapfrog step count from `trajectoryLength`.
MLIR_CAPI_EXPORTED MlirAttribute enzymeHMCConfigAttrGet(MlirContext ctx,
                                                        double trajectoryLength,
                                                        bool adaptStepSize,
                                                        bool adaptMassMatrix,
                                                        int64_t numSteps);

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
    MlirContext ctx, EnzymeGuideKind guideKind,
    EnzymeOptimizerKind optimizerKind, int64_t numParticles, bool hasInitScale,
    double initScale, bool hasAdamB1, double adamB1, bool hasAdamB2,
    double adamB2, bool hasAdamEps, double adamEps, bool hasClipNorm,
    double clipNorm, bool hasInitRadius, double initRadius,
    bool hasMomentumMass, double momentumMass, bool hasRmspropGamma,
    double rmspropGamma, bool hasRmspropEps, double rmspropEps,
    bool stableUpdate);

MLIR_CAPI_EXPORTED MlirAttribute enzymeMHConfigAttrGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute enzymeGibbsConfigAttrGet(MlirContext ctx);

MLIR_CAPI_EXPORTED MlirAttribute enzymeSymbolAttrGet(MlirContext ctx,
                                                     uint64_t ptr);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENZYME_MLIR_INTEGRATIONS_C_ENZYMEMLIR_H_
