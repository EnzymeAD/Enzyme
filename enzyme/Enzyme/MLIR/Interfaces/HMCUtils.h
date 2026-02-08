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

#include "Dialect/Ops.h"
#include "Interfaces/TransformUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace enzyme {
namespace MCMC {

struct SupportInfo {
  int64_t offset;
  int64_t traceOffset;
  int64_t size;
  enzyme::SupportAttr support;

  SupportInfo(int64_t offset, int64_t traceOffset, int64_t size,
              enzyme::SupportAttr support)
      : offset(offset), traceOffset(traceOffset), size(size), support(support) {
  }
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

struct HMCContext {
  FlatSymbolRefAttr fn;
  ArrayRef<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr selection;
  ArrayAttr allAddresses;
  Value invMass;
  Value massMatrixSqrt;
  Value stepSize;
  Value trajectoryLength;
  int64_t positionSize;
  SmallVector<SupportInfo> supports;

  HMCContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
             ArrayRef<Type> fnResultTypes, Value originalTrace,
             ArrayAttr selection, ArrayAttr allAddresses, Value invMass,
             Value massMatrixSqrt, Value stepSize, Value trajectoryLength,
             int64_t positionSize, ArrayRef<SupportInfo> supports)
      : fn(fn), fnInputs(fnInputs),
        fnResultTypes(fnResultTypes.begin(), fnResultTypes.end()),
        originalTrace(originalTrace), selection(selection),
        allAddresses(allAddresses), invMass(invMass),
        massMatrixSqrt(massMatrixSqrt), stepSize(stepSize),
        trajectoryLength(trajectoryLength), positionSize(positionSize),
        supports(supports.begin(), supports.end()) {}

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

  bool hasConstrainedSupports() const {
    for (const auto &info : supports) {
      if (info.support && info.support.getKind() != enzyme::SupportKind::REAL)
        return true;
    }
    return false;
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
              int64_t maxTreeDepth)
      : HMCContext(fn, fnInputs, fnResultTypes, originalTrace, selection,
                   allAddresses, invMass, massMatrixSqrt, stepSize,
                   /* Unused trajectoryLength */ Value(), positionSize,
                   supports),
        H0(H0), maxDeltaEnergy(maxDeltaEnergy), maxTreeDepth(maxTreeDepth) {}
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

/// Samples momentum from `N(0, M)` where M is the mass matrix.
/// Returns `(momentum, updated_rng_state)`.
std::pair<Value, Value> sampleMomentum(OpBuilder &builder, Location loc,
                                       Value rng, Value invMass,
                                       Value massMatrixSqrt,
                                       RankedTensorType positionType,
                                       bool debugDump = false);

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

/// Initializes HMC/NUTS state from a trace
/// Specifically:
///   - Extracts position from trace
///   - Computes initial potential energy U0 = -weight
///   - Samples initial momentum p0 ~ N(0, M)
///   - Computes initial kinetic energy and Hamiltonian
///   - Computes initial gradient via AutoDiffRegionOp
InitialHMCState InitHMC(OpBuilder &builder, Location loc, Value rng,
                        const HMCContext &ctx, bool debugDump = false);

/// Single HMC iteration: momentum sampling + leapfrog + MH accept/reject
MCMCKernelResult SampleHMC(OpBuilder &builder, Location loc, Value q,
                           Value grad, Value U, Value rng,
                           const HMCContext &ctx, bool debugDump = false);

/// Single NUTS iteration: momentum sampling + tree building
MCMCKernelResult SampleNUTS(OpBuilder &builder, Location loc, Value q,
                            Value grad, Value U, Value rng,
                            const NUTSContext &ctx, bool debugDump = false);

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
                                         const NUTSContext &ctx,
                                         bool debugDump = false);

/// Tree doubling by building a subtree of same depth and combining.
SubtreeBuildResult doubleTree(OpBuilder &builder, Location loc,
                              const NUTSTreeState &tree, Value direction,
                              Value pCkpts, Value pSumCkpts,
                              const NUTSContext &ctx, bool debugDump = false);

/// Main NUTS tree building loop.
NUTSTreeState buildTree(OpBuilder &builder, Location loc,
                        const NUTSTreeState &initialTree,
                        const NUTSContext &ctx, bool debugDump = false);

/// Computes checkpoint indices from leaf index for iterative turning check.
std::pair<Value, Value> leafIdxToCheckpointIdxs(OpBuilder &builder,
                                                Location loc, Value leafIdx);

/// Checkpoint-based iterative turning check.
Value checkIterativeTurning(OpBuilder &builder, Location loc, Value p,
                            Value pSum, Value pCkpts, Value pSumCkpts,
                            Value idxMin, Value idxMax, const NUTSContext &ctx,
                            bool debugDump = false);

/// Update checkpoint arrays at even leaf indices.
std::pair<Value, Value> updateCheckpoints(OpBuilder &builder, Location loc,
                                          Value leafIdx, Value ckptIdxMax,
                                          Value p, Value pSum, Value pCkpts,
                                          Value pSumCkpts,
                                          const NUTSContext &ctx,
                                          bool debugDump = false);

// TODO: Proper customization
struct DualAveragingConfig {
  double t0 = 10.0;    // Stabilization
  double kappa = 0.75; // Weight decay
  double gamma = 0.05; // Convergence
  double target_accept_prob = 0.8;
};

/// Initialize dual averaging state from initial step size.
DualAveragingState initDualAveraging(OpBuilder &builder, Location loc,
                                     Value stepSize);

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
} // namespace MCMC
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_HMC_UTILS_H
