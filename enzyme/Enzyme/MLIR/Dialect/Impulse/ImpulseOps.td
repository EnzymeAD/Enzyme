//===- ImpulseOps.td - Impulse dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPULSE_OPS
#define IMPULSE_OPS

include "ImpulseEnums.td"
include "Dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

def SymbolAttr : Impulse_Attr<"Symbol", "symbol"> {
  let summary = "Symbol associated with a Sample op";
  let description = [{
  Symbol associated with a Sample op.
  }];
  let parameters = (ins "uint64_t":$ptr);
  let assemblyFormat = "`<` $ptr `>`";
}

def AddressAttr : TypedArrayAttrBase<SymbolAttr, "Address as an array of symbols">;
def AddressArrayAttr : TypedArrayAttrBase<AddressAttr, "Array of addresses">;

//===----------------------------------------------------------------------===//
// Probabilistic programming ops
//===----------------------------------------------------------------------===//

def SampleOp : Impulse_Op<"sample",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Sample from a distribution";
  let description = [{
  Sample from a distribution. By convention, the 0th operand in `inputs`
  or `outputs` is the initial RNG state (seed).

  The sampler symbol `fn` is optional: a site that carries a `#impulse.dist`
  attribute derives its sampler and logpdf from the shared distribution builder,
  so it needs no traced `fn`/`logpdf` symbols and its result types are the ones
  the frontend constructs directly from the distribution's shape. A site with a
  traced sampler (the rejection-loop natives, generative sub-calls, or the
  custom-logpdf escape hatch) carries `fn`; a custom-logpdf site additionally
  carries `logpdf`. The verifier requires either `fn` or `dist` to be present.

  Round-trip limitation: with `fn` absent the pretty form prints as
  `impulse.sample(%inputs) ...`, which the generic MLIR parser cannot round-trip
  (the leading `(` after the op name is read as the start of a type). Passes
  operate on the in-memory op, so this affects only textual `.mlir` inputs; write
  an fn-absent site in generic op form (`"impulse.sample"(...) {...} : ...`) when
  a test must parse one from text.
  }];

  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$fn,
    Variadic<AnyType>:$inputs,
    OptionalAttr<FlatSymbolRefAttr>:$logpdf,
    OptionalAttr<SymbolAttr>:$symbol,
    OptionalAttr<SupportAttr>:$support,
    OptionalAttr<DistAttr>:$dist,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    ($fn^)? `(` $inputs `)` prop-dict attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
}

def UntracedCallOp : Impulse_Op<"untracedCall"> {
  let summary = "Call a probabilistic function without tracing";
  let description = [{
  Call a probabilistic function without tracing. By convention, the 0th operand in `inputs`
  or `outputs` is the initial RNG state (seed).
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` prop-dict attr-dict `:` functional-type($inputs, results)
  }];
}

def SimulateOp : Impulse_Op<"simulate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Simulate a generative function";
  let description = [{
    Simulates a generative function, building a trace tensor containing all
    sampled values and computing the accumulated log probability weight.

    The `selection` attribute specifies all sample addresses in order,
    determining the trace tensor layout.

    Returns: (trace, weight, rng, retvals...)
    - trace: tensor<1 x position_size x f64> - flattened samples
    - weight: tensor<f64> - accumulated log probability
    - rng: updated RNG state
    - retvals: original function return values
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    AddressArrayAttr:$selection,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$trace,
    AnyRankedTensor:$weight,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` prop-dict attr-dict `:` functional-type($inputs, results)
  }];
}

def GenerateOp : Impulse_Op<"generate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Constrained generation from a generative function";
  let description = [{
    Generates from a generative function with some addresses constrained.
    The constraint tensor contains flattened constrained values in the order
    specified by constrained_addresses.

    Returns: (trace, weight, rng, retvals...)
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    AnyRankedTensor:$constraint,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$constrained_addresses,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$trace,
    AnyRankedTensor:$weight,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $constraint prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def RegenerateOp : Impulse_Op<"regenerate",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]> {
  let summary = "Regenerate selected addresses in a trace";
  let description = [{
    Regenerates selected addresses while keeping others fixed.
    Used internally by MH.

    Takes explicit old_trace and returns new trace with weight.

    An optional `constraint` tensor (with `constrained_addresses` describing its
    layout) supplies observed values for constrained sample sites. Constrained
    sites are always scored/written from the constraint value when their branch
    is taken, rather than reused from the old trace slot. This is what keeps MH
    correct when a proposed value flips which `impulse.if` branch executes: a
    site that was previously not taken (its trace slot is stale) is refreshed
    from the constraint instead of the stale slot.

    The trace TYPE selects the weight semantics. A bare trace
    (tensor<1 x position_size x f64>) yields the full joint log probability.
    A score-tail trace (width = position_size + one memoized score scalar per
    site) yields the accept-ratio DELTA sum_dirty(logp_new - stored_score): a
    density term is dirty iff its site is regenerated or its parameters are
    dataflow-reachable from a regenerated site's value; clean terms cancel in
    the Metropolis--Hastings ratio and are never emitted, and old scores are
    read from the tail instead of recomputed. The caller maintains the carried
    weight as `w + delta` on accept, and the returned trace carries refreshed
    scores.

    Returns: (new_trace, weight, retvals...)
    - new_trace: same type as the given trace - flattened samples
      (plus refreshed site scores for score-tail traces)
    - weight: tensor<f64> - full joint or delta, per the trace type
    - retvals: original function return values
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    AnyRankedTensor:$original_trace,
    Optional<AnyRankedTensor>:$constraint,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$regenerate_addresses,
    OptionalAttr<AddressArrayAttr>:$constrained_addresses,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$new_trace,
    AnyRankedTensor:$weight,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace (`constrained_by` $constraint^)? prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def MHOp : Impulse_Op<"mh",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]> {
  let summary = "Metropolis-Hastings step for probabilistic inference";
  let description = [{
    Performs one MH step: regenerates selected addresses and accepts/rejects
    based on weight ratio.

    An optional `constraint` tensor (with `constrained_addresses`) supplies
    observed values, forwarded verbatim to the internal `regenerate`.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    AnyRankedTensor:$original_trace,
    AnyRankedTensor:$original_weight,
    Variadic<AnyType>:$inputs,
    Optional<AnyRankedTensor>:$constraint,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$regenerate_addresses,
    OptionalAttr<AddressArrayAttr>:$constrained_addresses,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$new_trace,
    AnyRankedTensor:$new_weight,
    AnyRankedTensor:$accepted,
    AnyType:$output_rng
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace `weight` $original_weight (`constrained_by` $constraint^)? prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

// The `init_*` operands are read only on a cold start (empty `state_in`).
class Impulse_InferSignatureArgs {
  dag args = (ins
    OptionalAttr<FlatSymbolRefAttr>:$fn,
    Variadic<AnyType>:$inputs,
    Variadic<AnyRankedTensor>:$state_in,
    Optional<AnyRankedTensor>:$original_trace,

    // MH-only: observed values for constrained sample sites (with
    // `constrained_addresses` describing the layout), forwarded verbatim to the
    // internal regenerate so a branch-flipping proposal re-scores the
    // newly-taken observation from the constraint. Null for HMC/NUTS/VI.
    Optional<AnyRankedTensor>:$constraint,

    Optional<AnyRankedTensor>:$init_inverse_mass_matrix,
    Optional<AnyRankedTensor>:$init_step_size,
    Optional<AnyRankedTensor>:$init_position,
    // VI guide scale in constrained (positive) space, not softplus space.
    Optional<AnyRankedTensor>:$init_scale,

    AddressArrayAttr:$selection,
    AddressArrayAttr:$all_addresses,

    DefaultValuedAttr<I64Attr, "0">:$num_warmup,
    DefaultValuedAttr<I64Attr, "1">:$num_samples,
    DefaultValuedAttr<I64Attr, "1">:$thinning,

    // Values > 1 add a leading chain dim to trace, diagnostics and state_out.
    DefaultValuedAttr<I64Attr, "1">:$num_chains,

    OptionalAttr<I64Attr>:$total_warmup,

    OptionalAttr<HMCConfigAttr>:$hmc_config,
    OptionalAttr<NUTSConfigAttr>:$nuts_config,
    OptionalAttr<VIConfigAttr>:$vi_config,
    OptionalAttr<MHConfigAttr>:$mh_config,

    // MH-only: layout of the `constraint` operand (see above). Null otherwise.
    OptionalAttr<AddressArrayAttr>:$constrained_addresses,

    OptionalAttr<FlatSymbolRefAttr>:$logpdf_fn,

    OptionalAttr<StrArrayAttr>:$state_schema,

    OptionalAttr<DictionaryAttr>:$autodiff_attrs,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );
}

// `diagnostics` columns are [accepted, diverging].
class Impulse_InferSignatureResults {
  dag outputs = (outs
    AnyRankedTensor:$trace,
    AnyRankedTensor:$diagnostics,
    AnyRankedTensor:$log_densities,
    AnyType:$output_rng_state,
    Variadic<AnyRankedTensor>:$state_out
  );
}

def InferOp : Impulse_Op<"infer",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]>,
    Impulse_InferSignatureArgs, Impulse_InferSignatureResults {
  let summary = "MCMC inference for probabilistic programs";
  let description = [{
    Runs MCMC inference on selected addresses.

    Two modes of operation:
    1. Trace-based mode: `fn` and `original_trace` are provided. The model
       function with `impulse.sample` ops defines the density.
    2. Custom logpdf mode: `logpdf_fn` and `init_position` are provided.
       The logpdf function maps position -> scalar log-density directly.

    The `selection` attribute determines which addresses to sample via HMC/NUTS.
    All sample addresses are included in the trace tensor for consistency.

    Resumable algorithm state (adapted step size, inverse mass matrix, dual
    averaging / Welford accumulators, warmup progress, and the final
    position/gradient/potential energy) is carried opaquely as a bundle:
    `state_in` warm-starts a run (empty = cold start), `state_out` returns it,
    and `state_schema` names each slot. `trace`, `diagnostics`,
    `log_densities`, and `output_rng_state` are the only universal results.
  }];

  let arguments = args;
  let results = outputs;

  let assemblyFormat = [{
    ($fn^)?
    `(` $inputs `)`
    (`given` $original_trace^)?
    (`constrained_by` $constraint^)?
    (`init_inverse_mass_matrix` `=` $init_inverse_mass_matrix^)?
    (`init_step_size` `=` $init_step_size^)?
    (`logpdf_fn` `=` $logpdf_fn^)?
    (`init_position` `=` $init_position^)?
    (`init_scale` `=` $init_scale^)?
    (`state` `=` `[` $state_in^ `]`)?
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Region-form ops
//===----------------------------------------------------------------------===//

def SampleRegionOp : Impulse_Op<"sample_region", [AutomaticAllocationScope]> {
  let summary = "Sample from a distribution with inlined sampler and logpdf regions";
  let description = [{
    Region-form of `impulse.sample`: the sampler and logpdf function bodies are
    inlined as nested regions, enabling sample-invariant code motion (SICM) to
    analyze across sample sites. The `logpdf` region is cleared once its body has
    been merged into the enclosing `infer_region`'s joint logpdf.

    The `address` attribute is the site's full address as an array of symbols;
    submodel dissolution prepends the enclosing site's address, so entries of
    the `infer_region`'s `selection`/`all_addresses` match by attribute equality.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<FlatSymbolRefAttr>:$fn,
    OptionalAttr<FlatSymbolRefAttr>:$logpdf_fn,
    OptionalAttr<AddressAttr>:$address,
    OptionalAttr<SupportAttr>:$support,
    OptionalAttr<DistAttr>:$dist,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let regions = (region AnyRegion:$sampler, AnyRegion:$logpdf);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $inputs `)` `sampler` $sampler `logpdf` $logpdf attr-dict
    `:` functional-type($inputs, $outputs)
  }];
}

def InferRegionOp : Impulse_Op<"infer_region",
    [AutomaticAllocationScope, AttrSizedOperandSegments]>,
    Impulse_InferSignatureArgs, Impulse_InferSignatureResults {
  let summary = "MCMC inference region with sampler and joint logpdf";
  let description = [{
    Region-form of `impulse.infer`, produced by `inline-infer-regions` and
    consumed directly by `expand-impulse`:

    - `sampler`: the model computation (with `sample_region` ops), the target of
      sampler-region SICM and the source for proposal-based algorithms.
    - `logpdf`: the joint logpdf, constructed by the `sicm` pass (after
      sampler-region SICM, so it is born optimized). Its block arguments are
      the position values, one per non-RNG sample result of each selected
      address in `selection` order; everything else it references is captured
      from the enclosing scope and is sample-invariant by construction
      (hoisted invariants, model inputs, and pinned trace values of unselected
      sites). SICM runs a second time on this region; `expand-impulse` then
      clones it as the potential-energy body inside the materialized
      HMC/NUTS loop.

    Mirrors `impulse.infer`'s operands/results/attributes (created by a
    wholesale attribute copy) plus the two regions and the
    `num_position_args`/`position_size` bookkeeping for the joint logpdf.
  }];

  let arguments = !con(args, (ins
    DefaultValuedAttr<I64Attr, "0">:$num_position_args,
    DefaultValuedAttr<I64Attr, "0">:$position_size
  ));

  let results = outputs;

  let regions = (region AnyRegion:$sampler, AnyRegion:$logpdf);

  let assemblyFormat = [{
    ($fn^)?
    `(` $inputs `)`
    (`given` $original_trace^)?
    (`constrained_by` $constraint^)?
    (`init_inverse_mass_matrix` `=` $init_inverse_mass_matrix^)?
    (`init_step_size` `=` $init_step_size^)?
    (`logpdf_fn` `=` $logpdf_fn^)?
    (`init_position` `=` $init_position^)?
    (`init_scale` `=` $init_scale^)?
    (`state` `=` `[` $state_in^ `]`)?
    `sampler` $sampler `logpdf` $logpdf
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// RNG ops
//===----------------------------------------------------------------------===//

def RandomOp : Impulse_Op<"random"> {
  let summary = "Generate random numbers using specified distribution";
  let description = [{
  Draws a result tensor of random numbers from `rng_state` and returns the
  advanced RNG state alongside the samples. Uniform bits are produced by a
  counter-based bit generator keyed by `rng_state`; the parametric distributions
  below are constructed deterministically from those bits, so the same input
  state reproduces the same samples.

  If rng_distribution = UNIFORM, samples are drawn from the uniform distribution
  over [a, b). A base draw in [0, 1) is formed from the raw bits (assembling the
  mantissa of a float in [1, 2) and subtracting 1) and affine-mapped to
  [a, b). If a >= b, the behavior is undefined.

  If rng_distribution = NORMAL, samples follow the normal distribution with
  mean = a and standard deviation = b, formed from a uniform draw u in [0, 1)
  via the probit `sqrt(2) * erfinv(2u - 1)`, then scaled by b and shifted by a.
  If b < 0, the behavior is undefined.

  If rng_distribution = MULTINORMAL, samples follow the multivariate normal with
  mean = a (scalar or vector) and covariance matrix = b: a standard-normal vector
  is drawn as above and mapped through the Cholesky factor of b, then shifted by
  a. The parameter b must be positive definite.

  By convention, the 0th operand in inputs is the initial RNG state and the
  0th operand in results is the updated RNG state.

  Note: lowered using `stablehlo.rng_bit_generator` (DEFAULT algorithm) for the
  bits, with `chlo.erf_inv` for the probit and `stablehlo.cholesky` for
  MULTINORMAL.
  }];

  let arguments = (ins
    AnyType:$rng_state,
    AnyType:$a,
    AnyType:$b,
    RngDistributionAttr:$rng_distribution
  );

  let results = (outs AnyType:$output_rng_state, AnyType:$result);

  let assemblyFormat = [{
    $rng_state `,` $a `,` $b  prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def RandomSplitOp : Impulse_Op<"randomSplit"> {
  let summary = "Split RNG state into multiple independent states";
  let description = [{
    Derives N independent RNG states (N = number of results) from a single input
    `rng_state` by hashing it with the Threefry2x32 counter-based PRF (five
    rounds), matching JAX's `random.split`. Each output state is statistically
    independent and deterministically reproducible from the input state.
    Reference: https://github.com/jax-ml/jax/blob/c25e095fcec9678a4ce5f723afce0c6a3c48a5e7/jax/_src/random.py#L281-L294

    Note: lowered to the inlined Threefry2x32 round sequence over StableHLO
    integer ops (no single StableHLO op).
  }];

  let arguments = (ins AnyType:$rng_state);
  let results = (outs Variadic<AnyType>:$output_rng_states);

  let assemblyFormat = [{
    $rng_state attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Control flow ops
//===----------------------------------------------------------------------===//

def ForOp : Impulse_Op<"for", [AutomaticAllocationScope]> {
  let summary = "Counted loop for probabilistic programming";
  let description = [{
    A counted loop operation that iterates from `lowerBound` to `upperBound`
    by `step`, carrying `iter_args` through each iteration. The iteration
    variable and iter_args are passed to the body region.
  }];

  let arguments = (ins
    AnyType:$lowerBound,
    AnyType:$upperBound,
    AnyType:$step,
    Variadic<AnyType>:$initArgs
  );

  let regions = (region SizedRegion<1>:$region);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `(` $lowerBound `:` type($lowerBound) `)`
    `to` `(` $upperBound `:` type($upperBound) `)`
    `step` `(` $step `:` type($step) `)`
    (`iter_args` `(` $initArgs^ `:` type($initArgs) `)`)?
    `->` type(results) $region attr-dict
  }];
}

def WhileOp : Impulse_Op<"while", [AutomaticAllocationScope]> {
  let summary = "While loop with condition";
  let description = [{
    A while loop operation that continues iterating as long as the condition
    evaluates to true. Intended to be lowered to `stablehlo.while`.
  }];

  let arguments = (ins Variadic<AnyType>:$initArgs);
  let regions = (region SizedRegion<1>:$conditionRegion,
                        SizedRegion<1>:$bodyRegion);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `(` $initArgs `:` type($initArgs) `)`
    `->` type(results)
    `condition` $conditionRegion
    `body` $bodyRegion
    attr-dict
  }];
}

def IfOp : Impulse_Op<"if", [AutomaticAllocationScope, RecursiveMemoryEffects]> {
  let summary = "Conditional execution with two branches";
  let description = [{
    A conditional operation that executes exactly one of two branches based on a
    boolean predicate.
  }];

  let arguments = (ins AnyType:$predicate);
  let regions = (region SizedRegion<1>:$trueBranch,
                        SizedRegion<1>:$falseBranch);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    `(` $predicate `)` `(` $trueBranch `,` $falseBranch `)`
    attr-dict `:` functional-type($predicate, results)
  }];
}

def YieldOp : Impulse_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["ForOp", "WhileOp", "IfOp", "InferRegionOp", "SampleRegionOp"]>]> {
  let summary = "Yield values at the end of a loop or if op";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

//===----------------------------------------------------------------------===//
// Linear algebra ops
//===----------------------------------------------------------------------===//

def DotOp : Impulse_Op<"dot", [Pure]> {
  let summary = "Computes a general dot product operation";
  let description = [{
    Generalized tensor contraction of `lhs` and `rhs`. Contracts the paired
    `lhs_contracting_dimensions`/`rhs_contracting_dimensions` (which must have
    equal size in each pair) by summing the elementwise products over them, and
    batches over the paired `lhs_batching_dimensions`/`rhs_batching_dimensions`
    (also equal-sized per pair). The result shape is, in order: the batch
    dimensions, the remaining (free) `lhs` dimensions, then the remaining (free)
    `rhs` dimensions. A dimension index may appear at most once across the
    batching and contracting lists for a given operand.

    Note: lowered 1:1 to `stablehlo.dot_general` with the same dimension numbers.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    DenseI64ArrayAttr:$lhs_batching_dimensions,
    DenseI64ArrayAttr:$rhs_batching_dimensions,
    DenseI64ArrayAttr:$lhs_contracting_dimensions,
    DenseI64ArrayAttr:$rhs_contracting_dimensions
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def CholeskyOp : Impulse_Op<"cholesky", [Pure]> {
  let summary = "Compute Cholesky decomposition of a symmetric positive definite matrix";
  let description = [{
    Computes the Cholesky factorization of the symmetric positive-definite
    matrix in the trailing two dimensions of `input` (any leading dimensions are
    treated as a batch). If `lower` is true, returns the lower-triangular L with
    A = L @ L^T; if false, returns the upper-triangular U with A = U^T @ U. The
    strictly-opposite triangle of the result is zero. The input must be symmetric
    positive definite; the result is otherwise unspecified. Result shape equals
    the input shape.

    Note: lowered 1:1 to `stablehlo.cholesky` with the same `lower` flag.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$lower
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def TriangularSolveOp : Impulse_Op<"triangular_solve", [Pure]> {
  let summary = "Solve a triangular linear system";
  let description = [{
    Solves a linear system whose coefficient matrix `a` is triangular, returning
    X with the shape of `b`. Only the triangle of `a` selected by `lower` is
    read (lower-triangular if `lower` is true, else upper); when `unit_diagonal`
    is true the diagonal of `a` is taken to be all ones and its stored diagonal
    is ignored. `op(a)` is `a`, `a^T`, or `a^H` according to `transpose_a`
    (NO_TRANSPOSE, TRANSPOSE, ADJOINT). If `left_side` is true, solves
    `op(a) @ X = b`; if false, solves `X @ op(a) = b`. `a` operates on the
    trailing two dimensions; leading dimensions are batched.

    Note: lowered 1:1 to `stablehlo.triangular_solve` with the same attributes.
  }];

  let arguments = (ins
    AnyRankedTensor:$a,
    AnyRankedTensor:$b,
    DefaultValuedAttr<BoolAttr, "true">:$left_side,
    DefaultValuedAttr<BoolAttr, "true">:$lower,
    DefaultValuedAttr<BoolAttr, "false">:$unit_diagonal,
    DefaultValuedAttr<TransposeAttr, "Transpose::NO_TRANSPOSE">:$transpose_a
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $a `,` $b prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Tensor manipulation ops
//===----------------------------------------------------------------------===//

def SelectOp : Impulse_Op<"select", [Pure]> {
  let summary = "Exteneded select operation";
  let description = [{
    Elementwise selection between `true_value` and `false_value`: each result
    element is the corresponding `true_value` element where `condition` is true,
    otherwise the `false_value` element. As an extension over `arith.select`, a
    rank-0 `tensor<i1>` `condition` selects between the two operands as whole
    tensors, so the operands may have a shape that differs from a per-element
    condition. When `condition` matches the operand shape, selection is per
    element. `true_value` and `false_value` share the result's shape and element
    type.

    Note: lowered 1:1 to `stablehlo.select`.
  }];

  let arguments = (ins
    AnyRankedTensor:$condition,
    AnyType:$true_value,
    AnyType:$false_value
  );

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $condition `,` $true_value `,` $false_value attr-dict `:` functional-type(operands, results)
  }];
}

def ReshapeOp : Impulse_Op<"reshape", [Pure]> {
  let summary = "Reshape a tensor to a new static shape";
  let description = [{
    Reinterprets `input` as a tensor of the result type's static shape,
    preserving the linear (row-major, last-dimension-fastest) order of elements.
    The result shape is fully static and its total element count must equal that
    of `input`; no elements are added, dropped, or reordered.

    Note: lowered 1:1 to `stablehlo.reshape`.
  }];
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];
}

def SliceOp : Impulse_Op<"slice", [Pure]> {
  let summary = "Extract a static slice from a tensor";
  let description = [{
    Extracts a statically-shaped sub-tensor of `operand`. In each dimension `d`
    it takes the elements at indices `start_indices[d] + k*strides[d]` for
    `k = 0, 1, ...` while the index stays below `limit_indices[d]`, yielding
    `ceil((limit_indices[d] - start_indices[d]) / strides[d])` elements. All
    three index arrays are static and have one entry per operand dimension; the
    result shape follows from them and must be in bounds
    (`0 <= start <= limit <= dim_size`, `strides >= 1`).

    Note: lowered 1:1 to `stablehlo.slice`.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand prop-dict attr-dict `:` functional-type($operand, $result)
  }];
}

def DynamicSliceOp : Impulse_Op<"dynamic_slice", [Pure]> {
  let summary = "Extract a slice from a tensor at dynamic start indices";
  let description = [{
    Extracts a sub-tensor of static shape `slice_sizes` from `operand`, starting
    at the runtime `start_indices` (one scalar index tensor per operand
    dimension). Before use, each start index is clamped into
    `[0, operand_dim - slice_sizes[d]]` so the whole slice stays in bounds, so
    out-of-range indices shift the window to the nearest fully-in-bounds position
    rather than reading out of bounds. The result shape is `slice_sizes`.

    Note: lowered 1:1 to `stablehlo.dynamic_slice`, whose clamping semantics this
    matches.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    Variadic<AnyRankedTensor>:$start_indices,
    DenseI64ArrayAttr:$slice_sizes
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand `,` $start_indices prop-dict attr-dict `:` functional-type(operands, results)
  }];
}

def DynamicUpdateSliceOp : Impulse_Op<"dynamic_update_slice", [Pure]> {
  let summary = "Update a slice in a tensor at dynamic start indices";
  let description = [{
    Produces a copy of `operand` with `update` written in starting at the runtime
    `start_indices` (one scalar index tensor per operand dimension). Before use,
    each start index is clamped into `[0, operand_dim - update_dim[d]]` so the
    entire `update` lands in bounds, so out-of-range indices shift the write
    window to the nearest fully-in-bounds position rather than writing out of
    bounds. The result has the shape of `operand`.

    Note: lowered 1:1 to `stablehlo.dynamic_update_slice`, whose clamping
    semantics this matches.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    AnyRankedTensor:$update,
    Variadic<AnyRankedTensor>:$start_indices
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand `,` $update `,` $start_indices attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Math ops
//===----------------------------------------------------------------------===//

def LogAddExpOp : Impulse_Op<"log_add_exp", [Pure]> {
  let summary = "Computes log(exp(x) + exp(y))";
  let description = [{
    Elementwise `log(exp(lhs) + exp(rhs))`, computed in a numerically stable way
    as `max(lhs, rhs) + log1p(exp(-|lhs - rhs|))`. When the difference is not
    finite (e.g. an infinite operand) the result falls back to `lhs + rhs`.
    `lhs` and `rhs` share the result shape and element type.

    Note: there is no single StableHLO op; this lowers to the max/log1p
    expansion above.
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def LogisticOp : Impulse_Op<"logistic", [Pure]> {
  let summary = "Computes logistic (sigmoid) function: 1 / (1 + exp(-x))";
  let description = [{
    Elementwise logistic (sigmoid) function `1 / (1 + exp(-x))` over `operand`.
    Result and operand share shape and element type.

    Note: lowered 1:1 to `stablehlo.logistic`.
  }];

  let arguments = (ins AnyRankedTensor:$operand);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

def LgammaOp : Impulse_Op<"lgamma", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultShape]> {
  let summary = "Computes the log of the absolute value of the Gamma function";
  let description = [{
    Elementwise log|Gamma(x)|. Lowered to `chlo.lgamma` in Enzyme-JAX; emitted by
    the distribution builder for logpdf normalizers that need it (Gamma, StudentT,
    Poisson).
  }];

  let arguments = (ins AnyRankedTensor:$operand);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

def PopcountOp : Impulse_Op<"popcount", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultShape]> {
  let summary = "Computes population count";
  let description = [{
    Elementwise population count: each result element is the number of set bits
    (1-bits) in the corresponding integer `operand` element. Operand and result
    are integer tensors of the same shape and element type.

    Note: lowered 1:1 to `stablehlo.popcnt`.
  }];

  let arguments = (ins AnyRankedTensor:$operand);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

#endif // IMPULSE_OPS
