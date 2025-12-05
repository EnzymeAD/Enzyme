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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace enzyme {

/// Result of a single leapfrog integration step.
struct LeapfrogResult {
  Value q_new;
  Value p_new;
  Value grad_new;
  Value U_new;
  Value rng_out;
};

/// Result of computing potential energy and its gradient.
struct GradientResult {
  Value potential; // U(q) = -log p(q)
  Value gradient;  // dU/dq
  Value rngOut;    // Updated RNG state
};

/// Initial state for HMC/NUTS algorithms.
struct InitialHMCState {
  Value position;    // q0 - flattened position vector
  Value momentum;    // p0 - sampled or provided momentum
  Value potential;   // U0 - initial potential energy
  Value kinetic;     // K0 - initial kinetic energy
  Value hamiltonian; // H0 = U0 + K0
  Value gradient;    // grad0 - gradient at initial position
  Value rng;         // Updated RNG state after sampling momentum
};

/// Runtime state of a NUTS tree.
struct NUTSTreeState {
  Value q_left, p_left, grad_left;
  Value q_right, p_right, grad_right;
  Value q_proposal, grad_proposal, U_proposal, H_proposal;
  Value depth, weight, turning, diverging;
  Value sum_accept_probs, num_proposals, p_sum;

  static constexpr size_t NUM_FIELDS = 17;

  SmallVector<Value> toValues() const;
  static NUTSTreeState fromValues(ArrayRef<Value> values);
  SmallVector<Type> getTypes() const;
};

/// Create an identity matrix of the given type.
Value createIdentityMatrix(OpBuilder &builder, Location loc,
                           RankedTensorType matrixType);

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
                           Value invMass, Value halfConst,
                           RankedTensorType scalarType,
                           RankedTensorType positionType);

/// Samples momentum from `N(0, M)`
/// If `invMass` is nullptr, samples from `N(0, I)` (assumes identity).
/// Otherwise, handles either a diagonal or dense inverse mass matrix.
/// Returns `(momentum, updated_rng_state)`.
std::pair<Value, Value> sampleMomentum(OpBuilder &builder, Location loc,
                                       Value rngState, Value invMass,
                                       Value zeroConst, Value oneConst,
                                       RankedTensorType positionType);

/// Computes potential energy `U(q) = -log p(q)` and its gradient `dU/dq`
/// by constructing an `AutoDiffRegionOp` containing an `UpdateOp`.
GradientResult computePotentialAndGradient(
    OpBuilder &builder, Location loc, Value position, Value rng,
    FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs, Value originalTrace,
    ArrayAttr selection, enzyme::TraceType traceType,
    RankedTensorType scalarType, RankedTensorType positionType);

/// Emits a single leapfrog integration step (aka velocity Verlet).
/// Specifically:
///   - `p_half = p - (eps/2) * grad`
///   - `q_new = q + eps * M^-1 * p_half`
///   - `grad_new = dU/dq(q_new)`
///   - `p_new = p_half - (eps/2) * grad_new`
LeapfrogResult
emitLeapfrogStep(OpBuilder &builder, Location loc, Value q, Value p, Value grad,
                 Value rng, Value stepSize, Value invMass, FlatSymbolRefAttr fn,
                 ArrayRef<Value> fnInputs, Value originalTrace,
                 ArrayAttr selection, enzyme::TraceType traceType,
                 RankedTensorType scalarType, RankedTensorType positionType);

/// U-turn termination criterion.
/// Returns `true` if U-turn detected in the trajectory.
Value checkTurning(OpBuilder &builder, Location loc, Value invMass, Value pLeft,
                   Value pRight, Value pSum, Value zeroConst,
                   RankedTensorType scalarType, RankedTensorType positionType);

/// Computes the uniform transition probability for subtree combination.
/// Specifically: `sigmoid(new_weight - current_weight)`.
Value computeUniformTransitionProb(OpBuilder &builder, Location loc,
                                   Value currentWeight, Value newWeight);

/// Computes the biased transition probability for main tree combination.
/// Specifically: `min(1, exp(new_weight - current_weight))`, zeroed if
/// turning or diverging.
Value computeBiasedTransitionProb(OpBuilder &builder, Location loc,
                                  Value currentWeight, Value newWeight,
                                  Value turning, Value diverging,
                                  Value zeroConst, Value oneConst);

/// Combines two trees during NUTS doubling.
NUTSTreeState
combineTrees(OpBuilder &builder, Location loc, const NUTSTreeState &currentTree,
             const NUTSTreeState &newTree, Value invMass, Value goingRight,
             Value rngKey, bool biasedTransition, Value zeroConst,
             Value oneConst, RankedTensorType scalarType,
             RankedTensorType positionType, RankedTensorType i64TensorType,
             RankedTensorType i1TensorType);

/// Initializes HMC/NUTS state from a trace.
/// Specifically:
///   - Extracts position
///   - Samples momentum
///   - Computes initial energies and gradient
InitialHMCState
initializeHMCState(OpBuilder &builder, Location loc, Value rngState,
                   Value originalTrace, ArrayAttr selection, Value invMass,
                   Value initialMomentum, // Debug
                   FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
                   RankedTensorType scalarType, RankedTensorType positionType,
                   bool debugDump = false);

/// Creates the final trace and computes acceptance for HMC.
/// Specifically:
///   - Generates the final trace
///   - Computes the acceptance probability
/// Returns `(selectedTrace, accepted, finalRng)`.
std::tuple<Value, Value, Value>
finalizeHMCStep(OpBuilder &builder, Location loc, Value proposedPosition,
                Value proposedMomentum, Value H0, Value invMass, Value rngState,
                Value originalTrace, ArrayAttr selection, FlatSymbolRefAttr fn,
                ArrayRef<Value> fnInputs, RankedTensorType scalarType,
                RankedTensorType positionType, bool debugDump = false);

/// Creates the final trace and computes acceptance for NUTS.
/// Specifically:
///   - Generates the final trace
///   - Always accepts the proposal
/// Returns `(selectedTrace, accepted, finalRng)`.
std::tuple<Value, Value, Value>
finalizeNUTSStep(OpBuilder &builder, Location loc, Value proposedPosition,
                 Value rngState, Value originalTrace, ArrayAttr selection,
                 FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
                 RankedTensorType scalarType, RankedTensorType positionType,
                 RankedTensorType i1TensorType);

} // namespace enzyme
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_HMC_UTILS_H
