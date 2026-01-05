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

struct HMCResult {
  Value trace;
  Value accepted;
  Value rng;
};

/// Result of one MCMC kernel step
struct MCMCKernelResult {
  Value q;        // New position vector
  Value grad;     // Gradient at new position
  Value U;        // Potential energy at new position
  Value accepted; // Whether proposal was accepted
  Value rng;      // Updated RNG state
};

struct IntegratorState {
  Value q;
  Value p;
  Value grad;
};

struct HMCContext {
  FlatSymbolRefAttr fn;
  ArrayRef<Value> fnInputs;
  Value originalTrace;
  ArrayAttr selection;
  Value invMass;
  Value stepSize;
  int64_t positionSize;

  HMCContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
             Value originalTrace, ArrayAttr selection, Value invMass,
             Value stepSize, int64_t positionSize)
      : fn(fn), fnInputs(fnInputs), originalTrace(originalTrace),
        selection(selection), invMass(invMass), stepSize(stepSize),
        positionSize(positionSize) {}
};

struct NUTSContext : public HMCContext {
  Value energyCurrent;
  Value maxDeltaEnergy;
  int64_t maxTreeDepth;

  NUTSContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
              Value originalTrace, ArrayAttr selection, Value invMass,
              Value stepSize, int64_t positionSize, Value energyCurrent,
              Value maxDeltaEnergy, int64_t maxTreeDepth)
      : HMCContext(fn, fnInputs, originalTrace, selection, invMass, stepSize,
                   positionSize),
        energyCurrent(energyCurrent), maxDeltaEnergy(maxDeltaEnergy),
        maxTreeDepth(maxTreeDepth) {}
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

/// Create sigmoid function: 1 / (1 + exp(-x))
Value createSigmoid(OpBuilder &builder, Location loc, Value x);

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

/// Samples momentum from `N(0, M)`
/// If `invMass` is nullptr, samples from `N(0, I)` (assumes identity).
/// Otherwise, handles either a diagonal or dense inverse mass matrix.
/// Returns `(momentum, updated_rng_state)`.
std::pair<Value, Value> sampleMomentum(OpBuilder &builder, Location loc,
                                       Value rng, Value invMass,
                                       RankedTensorType positionType);

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
                           int64_t numLeapfrogSteps, const HMCContext &ctx,
                           bool debugDump = false);

/// Single NUTS iteration: momentum sampling + tree building
MCMCKernelResult SampleNUTS(OpBuilder &builder, Location loc, Value q,
                            Value grad, Value U, Value rng,
                            const NUTSContext &ctx, bool debugDump = false);

/// PostProcess HMC samples
HMCResult PostProcessHMC(OpBuilder &builder, Location loc, Value q,
                         Value accepted, Value rng, const HMCContext &ctx);

/// PostProcess NUTS samples
HMCResult PostProcessNUTS(OpBuilder &builder, Location loc, Value q, Value rng,
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
} // namespace MCMC
} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_HMC_UTILS_H
