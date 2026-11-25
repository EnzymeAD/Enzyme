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

struct IntegrationResult {
  Value q;
  Value p;
  Value grad;
  Value U;
  Value rng;
};

struct GradientResult {
  Value U;    // U(q) = -log p(q)
  Value grad; // dU/dq
  Value rng;  // Updated RNG state
};

struct InitialHMCState {
  Value q0;    // Flattened position vector
  Value U0;    // Initial potential energy
  Value grad0; // Initial gradient at position q0
  Value rng;   // RNG state for SampleHMC
};

/// Result of one MCMC kernel step
struct MCMCKernelResult {
  Value q;           // New position vector
  Value grad;        // Gradient at new position
  Value U;           // Potential energy at new position
  Value accepted;    // Whether proposal was accepted
  Value divergent;   // Whether the trajectory diverged; false for HMC
  Value accept_prob; // Mean acceptance probability
  Value rng;         // Updated RNG state
};

struct DualAveragingState {
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

struct IntegratorState {
  Value q;
  Value p;
  Value grad;
};

/// The density an MCMC or VI lowering targets: the model `fn` evaluated on
/// `originalTrace` with the position written into the columns of `supports`,
/// or the log density `logpdfFn` of the position, or the `logpdfRegion` of an
/// `impulse.infer_region`. The element type is that of `stepSize`.
struct ModelContext {
  FlatSymbolRefAttr fn;
  ArrayRef<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr allAddresses;
  Value stepSize;
  int64_t positionSize;
  SmallVector<SupportInfo> supports;
  FlatSymbolRefAttr logpdfFn;
  DictionaryAttr autodiffAttrs;
  /// Whether the lowering dumps intermediate values for debugging.
  bool debugDump = false;

  /// When set, the log density is cloned from this region, whose block
  /// arguments take slices of the constrained position, instead of evaluated
  /// through `fn`.
  Region *logpdfRegion = nullptr;

  bool hasCustomLogpdf() const { return logpdfFn != nullptr; }

  bool hasLogpdfRegion() const { return logpdfRegion != nullptr; }

  int64_t getFullTraceSize() const {
    auto traceType = cast<RankedTensorType>(originalTrace.getType());
    return traceType.getShape()[1];
  }

  Type getElementType() const {
    return cast<RankedTensorType>(stepSize.getType()).getElementType();
  }

  RankedTensorType getPositionType() const {
    return RankedTensorType::get({1, positionSize}, getElementType());
  }

  RankedTensorType getScalarType() const {
    return RankedTensorType::get({}, getElementType());
  }
};

struct HMCContext : public ModelContext {
  ArrayAttr selection;
  Value invMass;
  Value massMatrixSqrt;
  Value trajectoryLength;
  // 0 derives the step count from trajectoryLength.
  int64_t numSteps = 0;

  HMCContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
             ArrayRef<Type> fnResultTypes, Value originalTrace,
             ArrayAttr selection, ArrayAttr allAddresses, Value invMass,
             Value massMatrixSqrt, Value stepSize, Value trajectoryLength,
             int64_t positionSize, ArrayRef<SupportInfo> supports,
             DictionaryAttr autodiffAttrs = {})
      : ModelContext{fn,
                     fnInputs,
                     SmallVector<Type>(fnResultTypes),
                     originalTrace,
                     allAddresses,
                     stepSize,
                     positionSize,
                     SmallVector<SupportInfo>(supports),
                     /*logpdfFn=*/{},
                     autodiffAttrs},
        selection(selection), invMass(invMass), massMatrixSqrt(massMatrixSqrt),
        trajectoryLength(trajectoryLength) {}

  HMCContext(FlatSymbolRefAttr logpdfFn, ArrayRef<Value> fnInputs,
             Value invMass, Value massMatrixSqrt, Value stepSize,
             Value trajectoryLength, int64_t positionSize,
             DictionaryAttr autodiffAttrs = {})
      : ModelContext{/*fn=*/{},
                     fnInputs,
                     /*fnResultTypes=*/{},
                     /*originalTrace=*/{},
                     /*allAddresses=*/{},
                     stepSize,
                     positionSize,
                     /*supports=*/{},
                     logpdfFn,
                     autodiffAttrs},
        invMass(invMass), massMatrixSqrt(massMatrixSqrt),
        trajectoryLength(trajectoryLength) {}

  bool hasConstrainedSupports() const {
    for (const auto &info : supports) {
      if (info.support && info.support.getKind() != impulse::SupportKind::REAL)
        return true;
    }
    return false;
  }

  HMCContext withStepSize(Value newStepSize) const {
    HMCContext copy = *this;
    copy.stepSize = newStepSize;
    return copy;
  }
};

struct NUTSContext : public HMCContext {
  Value H0;
  Value maxDeltaEnergy;
  int64_t maxTreeDepth;

  NUTSContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
              ArrayRef<Type> fnResultTypes, Value originalTrace,
              ArrayAttr selection, ArrayAttr allAddresses, Value invMass,
              Value massMatrixSqrt, Value stepSize, int64_t positionSize,
              ArrayRef<SupportInfo> supports, Value H0, Value maxDeltaEnergy,
              int64_t maxTreeDepth, DictionaryAttr autodiffAttrs = {})
      : HMCContext(fn, fnInputs, fnResultTypes, originalTrace, selection,
                   allAddresses, invMass, massMatrixSqrt, stepSize,
                   /* Unused trajectoryLength */ Value(), positionSize,
                   supports, autodiffAttrs),
        H0(H0), maxDeltaEnergy(maxDeltaEnergy), maxTreeDepth(maxTreeDepth) {}

  NUTSContext(FlatSymbolRefAttr logpdfFn, ArrayRef<Value> fnInputs,
              Value invMass, Value massMatrixSqrt, Value stepSize,
              int64_t positionSize, Value H0, Value maxDeltaEnergy,
              int64_t maxTreeDepth, DictionaryAttr autodiffAttrs = {})
      : HMCContext(logpdfFn, fnInputs, invMass, massMatrixSqrt, stepSize,
                   /* Unused trajectoryLength */ Value(), positionSize,
                   autodiffAttrs),
        H0(H0), maxDeltaEnergy(maxDeltaEnergy), maxTreeDepth(maxTreeDepth) {}

  NUTSContext withH0(Value newH0) const {
    NUTSContext copy = *this;
    copy.H0 = newH0;
    return copy;
  }
};

struct NUTSTreeState {
  Value q_left, p_left, grad_left;
  Value q_right, p_right, grad_right;
  Value q_proposal, grad_proposal, U_proposal, H_proposal;
  Value depth, weight, turning, diverging;
  Value sum_accept_probs, num_proposals, p_sum;
  Value rng;

  SmallVector<Value> toValues() const;
  static NUTSTreeState fromValues(ArrayRef<Value> values);
  SmallVector<Type> getTypes() const;

  IntegratorState getLeftLeaf() const { return {q_left, p_left, grad_left}; }
  IntegratorState getRightLeaf() const {
    return {q_right, p_right, grad_right};
  }
};

struct SubtreeBuildResult {
  NUTSTreeState tree;
  Value pCkpts;
  Value pSumCkpts;
};

/// Conditionally dump a value for debugging.
/// Emits an enzyme::DumpOp if `debugDump` is true; otherwise has no effect.
Value conditionalDump(OpBuilder &builder, Location loc, Value value,
                      StringRef label, bool debugDump);

/// Computes `v = M^-1 @ p`
/// If `invMass` is nullptr, returns `momentum` unchanged (assumes identity)
/// If `invMass` is a diagonal matrix, computes `v = invMass * momentum`
/// If `invMass` is a dense matrix, computes `v = invMass @ momentum`
Value applyInverseMassMatrix(OpBuilder &builder, Location loc, Value invMass,
                             Value momentum, RankedTensorType positionType);

/// Computes `K = 0.5 * p^T @ M^-1 @ p`
Value computeKineticEnergy(OpBuilder &builder, Location loc, Value momentum,
                           Value invMass, RankedTensorType positionType);

/// Computes the square root of the mass matrix from the inverse mass matrix.
Value computeMassMatrixSqrt(OpBuilder &builder, Location loc, Value invMass,
                            RankedTensorType positionType);

/// Samples momentum from `N(0, M)` where M is the mass matrix, drawing from
/// `rng`.
Value sampleMomentum(OpBuilder &builder, Location loc, Value rng, Value invMass,
                     Value massMatrixSqrt, RankedTensorType positionType,
                     bool debugDump);

/// Writes `qConstrained` into the original trace and emits a generate op of
/// the model on that trace.
GenerateOp emitGenerateAt(OpBuilder &builder, Location loc, Value qConstrained,
                          Value rng, const ModelContext &ctx);

/// Computes potential energy `U(q) = -log p(q)` and its gradient `dU/dq`
GradientResult computePotentialAndGradient(OpBuilder &builder, Location loc,
                                           Value position, Value rng,
                                           const HMCContext &ctx);

/// Computes a single leapfrog integration step.
/// Specifically:
///   - `p_half = p - (eps/2) * grad`
///   - `q_new = q + eps * M^-1 * p_half`
///   - `grad_new = dU/dq(q_new)`
///   - `p_new = p_half - (eps/2) * grad_new`
IntegrationResult computeIntegrationStep(OpBuilder &builder, Location loc,
                                         const IntegratorState &leaf, Value rng,
                                         Value direction,
                                         const HMCContext &ctx);

/// U-turn termination criterion.
/// Returns `true` if U-turn detected in the trajectory.
Value checkTurning(OpBuilder &builder, Location loc, Value pLeft, Value pRight,
                   Value pSum, const NUTSContext &ctx);

/// Computes the uniform transition probability for subtree combination.
/// Specifically: `sigmoid(new_weight - current_weight)`.
Value computeUniformTransitionProb(OpBuilder &builder, Location loc,
                                   Value currentWeight, Value newWeight);

/// Computes the biased transition probability for main tree combination.
/// Specifically: `min(1, exp(new_weight - current_weight))`, zeroed if
/// turning or diverging.
Value computeBiasedTransitionProb(OpBuilder &builder, Location loc,
                                  Value currentWeight, Value newWeight,
                                  Value turning, Value diverging);

/// Combines a tree with a newly-built subtree during NUTS doubling process.
///
/// The `subtree` is merged into `tree` based on `direction`:
///   - If going right (`direction` is true), then we keep `tree`'s left
///   boundary, takes `subtree`'s right boundary
///   - If going left (`direction` is false), then we take `subtree`'s left
///   boundary, keeps `tree`'s right boundary
///
/// Proposal selection has two options:
///   - Biased kernel (`biased` is true):
///     `prob = min(1, exp(subtree.weight - tree.weight))`,
///     zeroed if `turning` or `diverging`
///   - Uniform kernel (`biased` is false):
///     `prob = sigmoid(subtree.weight - tree.weight)`
///
/// Also checks the U-turn criterion on the combined momentum sum.
NUTSTreeState combineTrees(OpBuilder &builder, Location loc,
                           const NUTSTreeState &tree,
                           const NUTSTreeState &subTree, Value direction,
                           Value rng, bool biased, const NUTSContext &ctx);

/// Returns the starting position, its potential energy and gradient, and the
/// key for the first transition. The position is `initialPosition` with a
/// custom log density, else read from the trace and unconstrained.
InitialHMCState InitHMC(OpBuilder &builder, Location loc, Value rng,
                        const HMCContext &ctx, Value initialPosition = Value());

/// Single HMC iteration: momentum sampling + leapfrog + MH accept/reject
MCMCKernelResult SampleHMC(OpBuilder &builder, Location loc, Value q,
                           Value grad, Value U, Value rng,
                           const HMCContext &ctx);

/// Single NUTS iteration: momentum sampling + tree building
MCMCKernelResult SampleNUTS(OpBuilder &builder, Location loc, Value q,
                            Value grad, Value U, Value rng,
                            const NUTSContext &ctx);

/// Builds a base tree (leaf node) by taking one leapfrog step.
NUTSTreeState buildBaseTree(OpBuilder &builder, Location loc,
                            const IntegratorState &leaf, Value rng,
                            Value direction, const NUTSContext &ctx);

/// Extracts an appropriate leaf based on direction.
IntegratorState getLeafFromTree(OpBuilder &builder, Location loc,
                                const NUTSTreeState &tree, Value direction,
                                const NUTSContext &ctx);

/// Builds a subtree iteratively by appending leaves one at a time.
SubtreeBuildResult buildIterativeSubtree(OpBuilder &builder, Location loc,
                                         const NUTSTreeState &initialTree,
                                         Value direction, Value pCkpts,
                                         Value pSumCkpts,
                                         const NUTSContext &ctx);

/// Tree doubling by building a subtree of same depth and combining.
SubtreeBuildResult doubleTree(OpBuilder &builder, Location loc,
                              const NUTSTreeState &tree, Value direction,
                              Value pCkpts, Value pSumCkpts,
                              const NUTSContext &ctx);

/// Main NUTS tree building loop.
NUTSTreeState buildTree(OpBuilder &builder, Location loc,
                        const NUTSTreeState &initialTree,
                        const NUTSContext &ctx);

/// Computes checkpoint indices from leaf index for iterative turning check.
std::pair<Value, Value> leafIdxToCheckpointIdxs(OpBuilder &builder,
                                                Location loc, Value leafIdx);

/// Checkpoint-based iterative turning check.
Value checkIterativeTurning(OpBuilder &builder, Location loc, Value p,
                            Value pSum, Value pCkpts, Value pSumCkpts,
                            Value idxMin, Value idxMax, const NUTSContext &ctx);

/// Update checkpoint arrays at even leaf indices.
std::pair<Value, Value> updateCheckpoints(OpBuilder &builder, Location loc,
                                          Value leafIdx, Value ckptIdxMax,
                                          Value p, Value pSum, Value pCkpts,
                                          Value pSumCkpts);

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

/// Transform an entire position vector from constrained to unconstrained space
/// based on the support information.
Value unconstrainPosition(OpBuilder &builder, Location loc, Value constrained,
                          ArrayRef<SupportInfo> supports);

/// Transform an entire position vector from unconstrained to constrained space.
/// based on the support information.
Value constrainPosition(OpBuilder &builder, Location loc, Value unconstrained,
                        ArrayRef<SupportInfo> supports);

/// Compute total Jacobian correction for the constrain transform over all
/// position vector slices.
Value computeTotalJacobianCorrection(OpBuilder &builder, Location loc,
                                     Value unconstrained,
                                     ArrayRef<SupportInfo> supports);
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_HMC_UTILS_H
