//===- ImpulseOps.td - Impulse dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPULSE_OPS
#define IMPULSE_OPS

include "ImpulseAttributes.td"
include "Dialect.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Probabilistic programming ops
//===----------------------------------------------------------------------===//

def SampleOp : Impulse_Op<"sample",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Sample from a distribution";
  let description = [{
    Draws the value of one sample site of a model. `inputs` are the RNG state
    followed by the parameters of the distribution, and `outputs` are the
    advanced RNG state followed by the sampled values.

    With `dist`, the distribution is `dist`, and `fn`, when present, is a
    sampler for it with the signature of this op. Without `dist`, `fn` is
    required. If `logpdf` is present, `fn` is the sampler and `logpdf` its log
    density: a function of the sampled values followed by the parameters that
    returns a scalar. If not, `fn` is a model that this site runs, and the
    sites of `fn` are nested under the address of this site.

    `symbol` names the site within its model. `support`, when present, is the
    support of the sampled values; with `dist`, it must be of the same kind as
    the support of `dist`.
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
    `` custom<OptionalCallee>($fn) `(` $inputs `)` prop-dict attr-dict `:` functional-type($inputs, results)
  }];

  let hasVerifier = 1;
}

def UntracedCallOp : Impulse_Op<"untraced_call"> {
  let summary = "Call a probabilistic function without tracing";
  let description = [{
    Calls the model `fn` on `inputs`; `outputs` are its results. Every site of
    `fn` draws a new value, and no trace or weight is computed.
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
    Runs the model `fn` on `inputs`, every site drawing a new value. `trace`
    is the trace of `selection` and `weight` the joint log density of the run.
    `outputs` are the results of `fn`.
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
    Runs the model `fn` on `inputs`. A site in `constrained_addresses` takes
    its value from `constraint`, a trace of `constrained_addresses`; every
    other site draws a new value. `trace` is the trace of `selection` and
    `weight` the joint log density of the run. `outputs` are the results of
    `fn`.
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
    Runs the model `fn` on `inputs` again, the sites in `regenerate_addresses`
    drawing new values. Every other site takes its value from
    `original_trace`, a trace of `selection`, except that a site in
    `constrained_addresses` takes it from `constraint`, a trace of
    `constrained_addresses`, whenever the site runs. `new_trace` has the type
    of `original_trace` and holds the values of the new run; `outputs` are the
    results of `fn`.

    With a plain trace, `weight` is the joint log density of the new run. With
    a score-tail trace, `weight` is the joint log density of the new run minus
    that of `original_trace`, taken from the log densities it holds, and
    `new_trace` holds the log densities of the new run.
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
    One Metropolis-Hastings step on `original_trace`, a trace of `selection`
    whose joint log density is `original_weight`. The proposal is the
    `impulse.regenerate` of `original_trace` with the same `fn`, `inputs`,
    `selection`, `regenerate_addresses`, `constraint` and
    `constrained_addresses`, and it is accepted with the Metropolis-Hastings
    probability for a proposal that draws the regenerated sites from their
    distributions. If `accepted` is true, `new_trace` and `new_weight` are the
    trace of the proposal and its joint log density; otherwise they are
    `original_trace` and `original_weight`. `output_rng` is the advanced RNG
    state.
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

class Impulse_InferSignatureArgs {
  dag args = (ins
    OptionalAttr<FlatSymbolRefAttr>:$fn,
    Variadic<AnyType>:$inputs,
    Variadic<AnyRankedTensor>:$state_in,
    Optional<AnyRankedTensor>:$original_trace,
    Optional<AnyRankedTensor>:$constraint,

    Optional<AnyRankedTensor>:$init_inverse_mass_matrix,
    Optional<AnyRankedTensor>:$init_step_size,
    Optional<AnyRankedTensor>:$init_position,
    Optional<AnyRankedTensor>:$init_scale,

    AddressArrayAttr:$selection,
    AddressArrayAttr:$all_addresses,

    DefaultValuedAttr<I64Attr, "0">:$num_warmup,
    DefaultValuedAttr<I64Attr, "1">:$num_samples,
    DefaultValuedAttr<I64Attr, "1">:$thinning,
    DefaultValuedAttr<I64Attr, "1">:$num_chains,

    OptionalAttr<I64Attr>:$total_warmup,

    OptionalAttr<HMCConfigAttr>:$hmc_config,
    OptionalAttr<NUTSConfigAttr>:$nuts_config,
    OptionalAttr<VIConfigAttr>:$vi_config,
    OptionalAttr<MHConfigAttr>:$mh_config,
    OptionalAttr<GibbsConfigAttr>:$gibbs_config,

    OptionalAttr<AddressArrayAttr>:$constrained_addresses,

    OptionalAttr<FlatSymbolRefAttr>:$logpdf_fn,

    OptionalAttr<StrArrayAttr>:$state_schema,

    OptionalAttr<DictionaryAttr>:$autodiff_attrs,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );
}

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
  let summary = "Inference over the sample sites of a model";
  let description = [{
    Runs the inference algorithm selected by exactly one of `hmc_config`,
    `nuts_config`, `mh_config`, `gibbs_config` and `vi_config`. `inputs[0]` is
    the RNG state and `output_rng_state` the advanced state.

    The target is either the model `fn` or the log density `logpdf_fn`. With
    `fn`, `original_trace` is a trace of `all_addresses`, the sites in
    `selection` are inferred, and the other sites keep their values in
    `original_trace`. With `logpdf_fn`, the target is `logpdf_fn` applied to a
    flat position vector followed by `inputs[1:]`; the position has as many
    elements as `init_position`, or as one chain of it when `num_chains` > 1.

    HMC and NUTS run `num_warmup` warmup iterations, which adapt the step size
    and the inverse mass matrix as the config requests, then `num_samples`
    iterations, of which they keep every `thinning`-th. `trace` has one row
    per kept sample with the values of the sites in `selection`, in that
    order (with `logpdf_fn`, the position); `log_densities` holds the log
    density of each kept sample in unconstrained coordinates, and
    `diagnostics` its [accepted, diverging] flags. With `fn`, the chain starts
    from the values in `original_trace`; with `logpdf_fn`, at
    `init_position`. `init_step_size` is the initial step size, and
    `init_inverse_mass_matrix` the initial inverse mass matrix (a vector for a
    diagonal matrix, the identity when absent). `total_warmup`, when set, is
    the length of the warmup schedule of which this op runs `num_warmup`
    iterations, starting at the iteration in the `warmup_offset` state slot.
    `num_chains` > 1 runs independent chains: `trace`, `diagnostics`,
    `log_densities` and `state_out` get a leading chain dimension, and the
    `rng` state slot holds the RNG state of every chain.

    MH runs `num_samples` Metropolis-Hastings steps, each redrawing the sites
    in `selection` from their distributions, and Gibbs runs `num_samples`
    sweeps, each redrawing every site in `selection` from its distribution
    given the other sites. `trace` is the resulting trace, of the type of
    `original_trace`, `diagnostics` the acceptance flag of the last step
    (always true for Gibbs), and `log_densities` is zero. `state_in` and
    `state_out` hold one slot, `weight`, the joint log density of the trace.
    `constraint` is a trace of `constrained_addresses` with observed values.

    VI runs `num_samples` optimizer steps with learning rate `init_step_size`.
    `trace` holds the location of the guide in unconstrained coordinates after
    each step, `diagnostics` the ELBO estimate of each step, and
    `log_densities` is zero. `init_position` is the initial location of the
    guide, and `init_scale` the initial scale of an AUTO_NORMAL guide.

    `state_out` is the state of the algorithm after the run, one tensor per
    name in `state_schema`; passing it as `state_in` continues the run. A
    non-empty `state_in` takes the place of the initial values that
    `init_inverse_mass_matrix`, `init_step_size` (except as the learning rate
    of VI), `init_position` and `init_scale` give. `autodiff_attrs` are added
    to the attributes of the differentiation of the log density.
  }];

  let arguments = args;
  let results = outputs;

  let assemblyFormat = [{
    `` custom<OptionalCallee>($fn)
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
  let summary = "Region form of impulse.sample";
  let description = [{
    `impulse.sample` with its sampler and its log density in regions.
    `sampling` takes `inputs` as block arguments and yields `outputs`.
    `logpdf` takes the non-RNG results followed by the non-RNG inputs as block
    arguments and yields their log density as a scalar. `logpdf` is empty when
    the site runs a model, whose sites are then nested in `sampling`, and in
    an `impulse.infer_region` whose own `logpdf` region is not empty, since
    that region then holds the density of every site.

    `address` is the full address of the site, as listed in the `selection`
    and `all_addresses` of the enclosing `impulse.infer_region`. `fn`,
    `support`, `dist` and `name` mean what they mean on `impulse.sample`, and
    `logpdf_fn` what `logpdf` means there.
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

  let regions = (region MaxSizedRegion<1>:$sampling,
                        MaxSizedRegion<1>:$logpdf);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $inputs `)` `sampling` $sampling `logpdf` $logpdf prop-dict attr-dict
    `:` functional-type($inputs, $outputs)
  }];
}

def InferRegionOp : Impulse_Op<"infer_region",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AutomaticAllocationScope,
     AttrSizedOperandSegments]>,
    Impulse_InferSignatureArgs, Impulse_InferSignatureResults {
  let summary = "Region form of impulse.infer";
  let description = [{
    `impulse.infer` with the model or the log density in regions. The
    operands, the results and the attributes shared with `impulse.infer` mean
    what they mean there.

    `sampling` holds the body of `fn`: its block arguments are `inputs`, and
    its sample sites are `impulse.sample_region` ops, each with its full
    address. It is empty with `logpdf_fn`.

    `logpdf` computes the log density of the position given by its block
    arguments. With `fn`, these are the values of the sites in `selection`,
    one argument per non-RNG result of each site, in that order; the other
    sites keep their values from `original_trace` or `constraint`. It then
    yields the joint log density followed by the log density of each site of
    `sampling`, in the order the sites appear, with zero for a site in an
    `impulse.if` arm that does not run. With `logpdf_fn`, its one argument is
    the position and it yields the value of `logpdf_fn`. Every other value
    that `logpdf` uses is defined outside the op. An empty `logpdf` stands for
    the density that `sampling` defines.

    `num_position_args` is the number of leading `logpdf` block arguments
    that form the position, and `position_size` the number of their elements.
  }];

  let arguments = !con(args, (ins
    DefaultValuedAttr<I64Attr, "0">:$num_position_args,
    DefaultValuedAttr<I64Attr, "0">:$position_size
  ));

  let results = outputs;

  let regions = (region MaxSizedRegion<1>:$sampling,
                        MaxSizedRegion<1>:$logpdf);

  let assemblyFormat = [{
    `` custom<OptionalCallee>($fn)
    `(` $inputs `)`
    (`given` $original_trace^)?
    (`constrained_by` $constraint^)?
    (`init_inverse_mass_matrix` `=` $init_inverse_mass_matrix^)?
    (`init_step_size` `=` $init_step_size^)?
    (`logpdf_fn` `=` $logpdf_fn^)?
    (`init_position` `=` $init_position^)?
    (`init_scale` `=` $init_scale^)?
    (`state` `=` `[` $state_in^ `]`)?
    `sampling` $sampling `logpdf` $logpdf
    prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RNG ops
//===----------------------------------------------------------------------===//

def RandomOp : Impulse_Op<"random"> {
  let summary = "Generate random numbers using specified distribution";
  let description = [{
    Draws `result` from `rng_distribution` with the parameters `a` and `b`,
    and returns the advanced RNG state as `output_rng_state`. `result` is a
    deterministic function of `rng_state`, `a` and `b`; the op fixes its
    distribution, not the value that a given `rng_state` yields.

    - UNIFORM: uniform on [a, b). The behavior is undefined if a >= b.
    - NORMAL: normal with mean a and standard deviation b. The behavior is
      undefined if b < 0.
    - MULTINORMAL: multivariate normal with mean a (a scalar or a vector) and
      covariance matrix b, which must be positive definite.
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

def RandomSplitOp : Impulse_Op<"random_split"> {
  let summary = "Split RNG state into multiple independent states";
  let description = [{
    Derives one RNG state per result from `rng_state` with the Threefry2x32
    block cipher keyed by `rng_state`. The results are statistically
    independent of each other and of `rng_state`, and a given `rng_state`
    always yields the same results.
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
    Iterates while the condition holds. Both regions take the current values
    as block arguments: `conditionRegion` yields a rank-0 boolean tensor, and
    `bodyRegion` yields the values of the next iteration. The results are the
    values for which the condition first evaluates to false.
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

  let hasRegionVerifier = 1;
}

def YieldOp : Impulse_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["ForOp", "WhileOp", "IfOp", "InferRegionOp", "SampleRegionOp"]>]> {
  let summary = "Terminator of the regions of impulse ops";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

//===----------------------------------------------------------------------===//
// Linear algebra ops
//===----------------------------------------------------------------------===//

def DotOp : Impulse_Op<"dot", [Pure]> {
  let summary = "Tensor contraction";
  let description = [{
    Tensor contraction of `lhs` and `rhs`: sums the elementwise products over
    the paired `lhs_contracting_dimensions`/`rhs_contracting_dimensions`, which
    must have equal size in each pair. The result shape is the remaining `lhs`
    dimensions followed by the remaining `rhs` dimensions, each in order. With
    no contracting dimensions the result is the outer product. A dimension
    index may appear at most once in the list of an operand.
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
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

def SelectOp : Impulse_Op<"select",
    [Pure, AllTypesMatch<["true_value", "false_value", "result"]>]> {
  let summary = "Select between two values";
  let description = [{
    Elementwise selection between `true_value` and `false_value`: each result
    element is the corresponding `true_value` element where `condition` is true,
    otherwise the `false_value` element. As an extension over `arith.select`, a
    rank-0 `tensor<i1>` `condition` selects between the two operands as whole
    tensors, so the operands may have a shape that differs from a per-element
    condition. When `condition` matches the operand shape, selection is per
    element.
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

def LogisticOp : Impulse_Op<"logistic", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultShape]> {
  let summary = "Computes logistic (sigmoid) function: 1 / (1 + exp(-x))";
  let description = [{
    Elementwise logistic (sigmoid) function `1 / (1 + exp(-x))` over `operand`.
    Result and operand share shape and element type.
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
    Elementwise log|Gamma(x)| over `operand`. Result and operand share shape
    and element type.
  }];

  let arguments = (ins AnyRankedTensor:$operand);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

#endif // IMPULSE_OPS
