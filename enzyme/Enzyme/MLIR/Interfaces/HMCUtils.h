//===- HMCUtils.h - Utilities for HMC/NUTS algorithms -------* C++ -*-===//
//
// This file declares utility functions for Hamiltonian Monte Carlo (HMC) and
// No-U-Turn Sampler (NUTS) implementations.
//
// Reference:
// https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/hmc_util.py
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_HMC_UTILS_H
#define ENZYME_MLIR_INTERFACES_HMC_UTILS_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace impulse {

/// Where the values of one inferred site sit in the position vector, in
/// constrained and in unconstrained coordinates, and in the trace.
struct SupportInfo {
  /// First position column of the site's constrained values.
  int64_t constrainedOffset;
  /// First trace column of the site's values.
  int64_t traceOffset;
  /// Number of constrained values of the site.
  int64_t constrainedSize;
  /// First position column of the site's unconstrained values.
  int64_t unconstrainedOffset;
  /// Number of unconstrained values of the site; differs from
  /// `constrainedSize` only for SIMPLEX.
  int64_t unconstrainedSize;
  /// The support of the site's values; null means all of R.
  impulse::SupportAttr support;

  SupportInfo(int64_t constrainedOffset, int64_t traceOffset,
              int64_t constrainedSize, int64_t unconstrainedOffset,
              int64_t unconstrainedSize, impulse::SupportAttr support)
      : constrainedOffset(constrainedOffset), traceOffset(traceOffset),
        constrainedSize(constrainedSize),
        unconstrainedOffset(unconstrainedOffset),
        unconstrainedSize(unconstrainedSize), support(support) {}
};

/// The state of an HMC or NUTS chain between two transitions: the `1xN`
/// position, the gradient and the potential energy there, and the RNG state
/// the next transition draws from.
struct ChainPoint {
  Value q;
  Value grad;
  Value potentialEnergy;
  Value rng;
};

/// The outcome of one HMC or NUTS transition.
struct MCMCKernelResult {
  ChainPoint next;
  /// Whether the proposal was accepted.
  Value accepted;
  /// Whether the trajectory diverged; false for HMC.
  Value divergent;
  /// Mean acceptance probability of the proposals.
  Value acceptProb;
};

struct DualAveragingState {
  static constexpr unsigned kNumValues = 5;

  Value log_step_size;
  Value log_step_size_avg;
  Value gradient_avg;
  Value step_count;
  Value prox_center;

  SmallVector<Value> toValues() const {
    return {log_step_size, log_step_size_avg, gradient_avg, step_count,
            prox_center};
  }
  static DualAveragingState fromValues(ArrayRef<Value> values) {
    return {values[0], values[1], values[2], values[3], values[4]};
  }
  SmallVector<Type> getTypes() const {
    return {log_step_size.getType(), log_step_size_avg.getType(),
            gradient_avg.getType(), step_count.getType(),
            prox_center.getType()};
  }
};

/// The density an MCMC or VI lowering targets: the model `fn` evaluated on
/// `originalTrace` with the position written into the columns of `supports`,
/// or the log density `logpdfFn` of the position, or the `logpdfRegion` of an
/// `impulse.infer_region`.
struct ModelContext {
  FlatSymbolRefAttr fn;
  FlatSymbolRefAttr logpdfFn;
  /// When set, the log density is cloned from this region instead of
  /// evaluated through `fn`.
  Region *logpdfRegion = nullptr;
  /// Block argument k of `logpdfRegion` reads its value from the constrained
  /// position starting at column `logpdfArgOffsets[k]`.
  SmallVector<int64_t> logpdfArgOffsets;
  /// The operands of `fn` or `logpdfFn` after the RNG state or the position.
  SmallVector<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr allAddresses;
  SmallVector<SupportInfo> supports;
  /// Attributes added to every `enzyme.autodiff_region` of the lowering.
  DictionaryAttr autodiffAttrs;
  /// Number of elements of the position the inference moves.
  int64_t positionSize = 0;
  Type elementType;
  /// Whether the lowering dumps intermediate values for debugging.
  bool debugDump = false;

  bool hasCustomLogpdf() const { return logpdfFn != nullptr; }

  bool hasLogpdfRegion() const { return logpdfRegion != nullptr; }

  int64_t getFullTraceSize() const {
    auto traceType = cast<RankedTensorType>(originalTrace.getType());
    return traceType.getShape()[1];
  }

  /// Number of elements of the position mapped onto the supports of the
  /// sites, which differs from `positionSize` only for SIMPLEX.
  int64_t getConstrainedPositionSize() const {
    if (supports.empty())
      return positionSize;
    int64_t size = 0;
    for (const SupportInfo &info : supports)
      size += info.constrainedSize;
    return size;
  }

  RankedTensorType getPositionType() const {
    return RankedTensorType::get({1, positionSize}, elementType);
  }

  RankedTensorType getScalarType() const {
    return RankedTensorType::get({}, elementType);
  }
};

/// The model and parameters of an HMC or NUTS transition.
struct HMCContext : public ModelContext {
  Value stepSize;
  Value invMass;
  Value massMatrixSqrt;
  /// Sets the number of leapfrog steps of HMC unless `numSteps` is positive.
  Value trajectoryLength;
  int64_t numSteps = 0;

  HMCContext withStepSize(Value newStepSize) const {
    HMCContext copy = *this;
    copy.stepSize = newStepSize;
    return copy;
  }
};

/// The model and parameters of a NUTS transition.
struct NUTSContext : public HMCContext {
  /// The Hamiltonian at the start of the transition.
  Value H0;
  Value maxDeltaEnergy;
  int64_t maxTreeDepth = 0;

  NUTSContext withH0(Value newH0) const {
    NUTSContext copy = *this;
    copy.H0 = newH0;
    return copy;
  }
};

/// Computes the square root of the mass matrix from the inverse mass matrix.
Value computeMassMatrixSqrt(OpBuilder &builder, Location loc, Value invMass,
                            RankedTensorType positionType);

/// Returns the log density of the model at the constrained `1xN` position
/// `qConstrained`: the logpdf region of `ctx` cloned at the insertion point
/// when there is one, else the weight of a generate op of `ctx.fn` that draws
/// from `rng`. `rng` advances to the first output of the generate op, by
/// convention the model's RNG state; the region and a model without outputs
/// leave it as it is.
Value emitModelLogDensity(OpBuilder &builder, Location loc, Value qConstrained,
                          Value &rng, const ModelContext &ctx);

/// Returns the starting position, its potential energy and gradient, and the
/// key for the first transition. The position is `initialPosition` with a
/// custom log density, else read from the trace and unconstrained.
ChainPoint InitHMC(OpBuilder &builder, Location loc, Value rng,
                   const HMCContext &ctx, Value initialPosition = Value());

/// Single HMC iteration: momentum sampling + leapfrog + MH accept/reject
MCMCKernelResult SampleHMC(OpBuilder &builder, Location loc,
                           const ChainPoint &start, const HMCContext &ctx);

/// Single NUTS iteration: momentum sampling + tree building
MCMCKernelResult SampleNUTS(OpBuilder &builder, Location loc,
                            const ChainPoint &start, const NUTSContext &ctx);

// TODO: Proper customization
struct DualAveragingConfig {
  double t0 = 10.0;    // Stabilization
  double kappa = 0.75; // Weight decay
  double gamma = 0.05; // Convergence
  double target_accept_prob = 0.8;
};

/// Initialize dual averaging state from initial step size. The prox center is
/// log(10 * stepSize) if `logOfProduct`, else log(10) + log(stepSize); the two
/// round differently.
DualAveragingState initDualAveraging(OpBuilder &builder, Location loc,
                                     Value stepSize, bool logOfProduct = false);

/// Update dual averaging state with observed acceptance probability.
DualAveragingState updateDualAveraging(OpBuilder &builder, Location loc,
                                       const DualAveragingState &state,
                                       Value acceptProb,
                                       const DualAveragingConfig &config);

/// Get step size from dual averaging state.
/// If `final` is true, returns the averaged step size.
/// Otherwise, returns the updated step size.
Value getStepSizeFromDualAveraging(OpBuilder &builder, Location loc,
                                   const DualAveragingState &state,
                                   bool final = false);

/// State for Welford covariance estimation
struct WelfordState {
  static constexpr unsigned kNumValues = 3;

  Value mean;
  Value m2; // Sum of squared deviations
  Value n;

  SmallVector<Value> toValues() const { return {mean, m2, n}; }
  static WelfordState fromValues(ArrayRef<Value> values) {
    return {values[0], values[1], values[2]};
  }
  SmallVector<Type> getTypes() const {
    return {mean.getType(), m2.getType(), n.getType()};
  }
};

/// Configuration for Welford covariance estimation
struct WelfordConfig {
  bool diagonal = true;
  bool regularize = true; // Optional regularization (Stan's shrinkage)
};

/// Initialize state for Welford covariance estimation.
WelfordState initWelford(OpBuilder &builder, Location loc, int64_t positionSize,
                         bool diagonal);

/// Update Welford state with a new sample.
WelfordState updateWelford(OpBuilder &builder, Location loc,
                           const WelfordState &state, Value sample,
                           const WelfordConfig &config);

/// Finalize Welford state to produce sample covariance (returned as inverse
/// mass matrix).
Value finalizeWelford(OpBuilder &builder, Location loc,
                      const WelfordState &state, const WelfordConfig &config);

struct AdaptWindow {
  int64_t start;
  int64_t end;
};

/// Build warmup adaptation schedule.
/// TODO: Make customizable
SmallVector<AdaptWindow> buildAdaptationSchedule(int64_t numSteps);

/// Transform an entire position vector from unconstrained to constrained space.
/// based on the support information.
Value constrainPosition(OpBuilder &builder, Location loc, Value unconstrained,
                        ArrayRef<SupportInfo> supports);

/// Returns log|det J| of constrainPosition at `unconstrained`, summed over
/// the sites to a rank-0 tensor.
Value constrainPositionLogAbsDetJacobian(OpBuilder &builder, Location loc,
                                         Value unconstrained,
                                         ArrayRef<SupportInfo> supports);
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_HMC_UTILS_H
