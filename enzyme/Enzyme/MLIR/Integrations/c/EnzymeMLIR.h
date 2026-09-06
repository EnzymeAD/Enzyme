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

/// Register all Enzyme MLIR passes with the global registry.
MLIR_CAPI_EXPORTED void enzymeRegisterPasses(void);

/// Register Enzyme's external model AD implementations for the core MLIR
/// dialects (arith, func, scf, ...) into the given DialectRegistry.
/// Must be called before appending the registry to an MLIRContext so that
/// the Differentiate pass knows how to lower ops in those dialects.
MLIR_CAPI_EXPORTED void
enzymeRegisterDialectExtensions(MlirDialectRegistry registry);

/// Create the core differentiation pass (`--enzyme`).
MLIR_CAPI_EXPORTED MlirPass enzymeCreateDifferentiatePass(void);

/// Create the core differentiation pass with options.
/// postpasses: semicolon-separated list of passes to run after differentiation
/// (may be NULL or empty). verifyPostPasses: run the verifier after each
/// post-pass.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateDifferentiatePassWithOptions(
    MlirStringRef postpasses, bool verifyPostPasses);

/// Create the `--enzyme-ops-to-memref` pass, which lowers `enzyme.init`,
/// `enzyme.get`, and `enzyme.set` to `memref` + `scf.if`. Required before
/// lowering to LLVM.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateConvertEnzymeToMemRefPass(void);

/// Create the `--enzyme-batch` pass, which lowers `enzyme.batch` ops to
/// calls to generated `@batched_*` functions.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateBatchPass(void);

/// Create the `--enzyme-batch-diff` pass, which differentiates batched
/// functions produced by `--enzyme-batch`.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateBatchDiffPass(void);

/// Create the `--remove-unnecessary-enzyme-ops` pass, which removes dead
/// `enzyme.*` ops after differentiation.
MLIR_CAPI_EXPORTED MlirPass enzymeCreateRemoveUnusedEnzymeOpsPass(void);

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
