//===- ImpulseOps.td - Impulse dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPULSE_OPS
#define IMPULSE_OPS

include "Impulse/ImpulseEnums.td"
include "Impulse/Dialect.td"

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
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    OptionalAttr<FlatSymbolRefAttr>:$logpdf,
    OptionalAttr<SymbolAttr>:$symbol,
    OptionalAttr<SupportAttr>:$support,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
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
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
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
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
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
    $fn `(` $inputs `)` `given` $constraint attr-dict `:` functional-type(operands, results)
  }];
}

def RegenerateOp : Impulse_Op<"regenerate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Regenerate selected addresses in a trace";
  let description = [{
    Regenerates selected addresses while keeping others fixed.
    Used internally by MH.

    Takes explicit old_trace and returns new trace with weight.

    Returns: (new_trace, weight, retvals...)
    - new_trace: tensor<1 x position_size x f64> - flattened samples
    - weight: tensor<f64> - accumulated log probability
    - retvals: original function return values
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    AnyRankedTensor:$original_trace,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$regenerate_addresses,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$new_trace,
    AnyRankedTensor:$weight,
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace attr-dict `:` functional-type(operands, results)
  }];
}

def MHOp : Impulse_Op<"mh", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Metropolis-Hastings step for probabilistic inference";
  let description = [{
    Performs one MH step: regenerates selected addresses and accepts/rejects
    based on weight ratio.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    AnyRankedTensor:$original_trace,
    AnyRankedTensor:$original_weight,
    Variadic<AnyType>:$inputs,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$regenerate_addresses,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$new_trace,
    AnyRankedTensor:$new_weight,
    AnyRankedTensor:$accepted,
    AnyType:$output_rng
  );

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace `weight` $original_weight attr-dict `:` functional-type(operands, results)
  }];
}

def InferOp : Impulse_Op<"infer", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]> {
  let summary = "MCMC inference for probabilistic programs";
  let description = [{
    Runs MCMC inference on selected addresses.

    Two modes of operation:
    1. Trace-based mode: `fn` and `original_trace` are provided. The model
       function with `impulse.sample` ops defines the density.
    2. Custom logpdf mode: `logpdf_fn` and `initial_position` are provided.
       The logpdf function maps position -> scalar log-density directly.

    The `selection` attribute determines which addresses to sample via HMC/NUTS.
    All sample addresses are included in the trace tensor for consistency.

    Returns: (trace, diagnostics, rng, final_position, final_gradient,
              final_potential_energy, final_step_size, final_inverse_mass_matrix)
    - trace: tensor<num_samples x position_size x f64>
    - diagnostics: tensor<num_samples x i1> - placeholder for future expansion
    - rng: updated RNG state
    - final_position: tensor<1 x position_size x f64> - position after last sample
    - final_gradient: tensor<1 x position_size x f64> - gradient at final position
    - final_potential_energy: tensor<f64> - potential energy at final position
    - final_step_size: tensor<f64> - adapted step size (after warmup)
    - final_inverse_mass_matrix: tensor - adapted inverse mass matrix (after warmup)
  }];

  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$fn,
    Variadic<AnyType>:$inputs,
    Optional<AnyRankedTensor>:$original_trace,
    AddressArrayAttr:$selection,
    AddressArrayAttr:$all_addresses,

    DefaultValuedAttr<I64Attr, "0">:$num_warmup,
    DefaultValuedAttr<I64Attr, "1">:$num_samples,
    DefaultValuedAttr<I64Attr, "1">:$thinning,

    // Algorithm-specific MCMC parameters
    Optional<AnyRankedTensor>:$inverse_mass_matrix,
    Optional<AnyRankedTensor>:$step_size,

    // Algorithm-specific configurations
    OptionalAttr<HMCConfigAttr>:$hmc_config,
    OptionalAttr<NUTSConfigAttr>:$nuts_config,

    // Custom logpdf mode
    OptionalAttr<FlatSymbolRefAttr>:$logpdf_fn,
    Optional<AnyRankedTensor>:$initial_position,

    Optional<AnyRankedTensor>:$initial_gradient,
    Optional<AnyRankedTensor>:$initial_potential_energy,

    OptionalAttr<DictionaryAttr>:$autodiff_attrs,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$trace,
    AnyRankedTensor:$diagnostics,
    AnyType:$output_rng_state,
    AnyRankedTensor:$final_position,
    AnyRankedTensor:$final_gradient,
    AnyRankedTensor:$final_potential_energy,
    AnyRankedTensor:$final_step_size,
    AnyRankedTensor:$final_inverse_mass_matrix
  );

  let assemblyFormat = [{
    ($fn^)?
    `(` $inputs `)`
    (`given` $original_trace^)?
    (`inverse_mass_matrix` `=` $inverse_mass_matrix^)?
    (`step_size` `=` $step_size^)?
    (`logpdf_fn` `=` $logpdf_fn^)?
    (`initial_position` `=` $initial_position^)?
    (`initial_gradient` `=` $initial_gradient^)?
    (`initial_potential_energy` `=` $initial_potential_energy^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RNG ops
//===----------------------------------------------------------------------===//

def RandomOp : Impulse_Op<"random"> {
  let summary = "Generate random numbers using specified distribution";
  let description = [{
  Generates random numbers using the rng_distribution algorithm and produces
  a result tensor.

  If rng_distribution = UNIFORM, then the random numbers are generated following
  the uniform distribution over the interval [a, b). If a >= b, the behavior is
  undefined.

  If rng_distribution = NORMAL, then the random numbers are generated following
  the normal distribution with mean = a and standard deviation = b. If b < 0,
  the behavior is undefined.

  If rng_distribution = MULTINORMAL, then the random numbers are generated
  following the multivariate normal distribution with mean = a (scalar or vector)
  and covariance matrix = b. The parameter b should be a positive definite matrix.

  By convention, the 0th operand in inputs is the initial RNG state and the
  0th operand in results is the updated RNG state.
  }];

  let arguments = (ins
    AnyType:$rng_state,
    AnyType:$a,
    AnyType:$b,
    RngDistributionAttr:$rng_distribution
  );

  let results = (outs AnyType:$output_rng_state, AnyType:$result);

  let assemblyFormat = [{
    $rng_state `,` $a `,` $b  attr-dict `:` functional-type(operands, results)
  }];
}

def RandomSplitOp : Impulse_Op<"randomSplit"> {
  let summary = "Split RNG state into multiple independent states";
  let description = [{
    Splits an RNG state into multiple independent RNG states.
    Reference: https://github.com/jax-ml/jax/blob/c25e095fcec9678a4ce5f723afce0c6a3c48a5e7/jax/_src/random.py#L281-L294
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
    ParentOneOf<["ForOp", "WhileOp", "IfOp"]>]> {
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
    Computes a general dot product operation. To be lowered to `stablehlo.dot_general`.
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
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

def CholeskyOp : Impulse_Op<"cholesky", [Pure]> {
  let summary = "Compute Cholesky decomposition of a symmetric positive definite matrix";
  let description = [{
    Computes the Cholesky decomposition of a symmetric positive definite matrix A.
    Returns L such that A = L @ L^T (if lower=true) or A = U^T @ U (if lower=false).
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$lower
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` functional-type(operands, results)
  }];
}

def TriangularSolveOp : Impulse_Op<"triangular_solve", [Pure]> {
  let summary = "Solve a triangular linear system";
  let description = [{
    Solves a system of linear equations with a triangular coefficient matrix.
    If left_side=true, solves op(A) @ X = B for X.
    If left_side=false, solves X @ op(A) = B for X.
    op(A) is determined by transpose_a: NO_TRANSPOSE, TRANSPOSE, or ADJOINT.
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
    $a `,` $b attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Tensor manipulation ops
//===----------------------------------------------------------------------===//

def SelectOp : Impulse_Op<"select", [Pure]> {
  let summary = "Exteneded select operation";
  let description = [{
    Extended select operation that supports:
    - `tensor<i1>` conditions with differently-sized operands
    - standard cases supported by `arith.select`
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
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];
}

def SliceOp : Impulse_Op<"slice", [Pure]> {
  let summary = "Extract a static slice from a tensor";
  let description = [{
    Extract a static slice from a tensor.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type($operand, $result)
  }];
}

def DynamicSliceOp : Impulse_Op<"dynamic_slice", [Pure]> {
  let summary = "Extract a slice from a tensor at dynamic start indices";
  let description = [{
    Extract a slice from a tensor at dynamic start indices.
  }];

  let arguments = (ins
    AnyRankedTensor:$operand,
    Variadic<AnyRankedTensor>:$start_indices,
    DenseI64ArrayAttr:$slice_sizes
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand `,` $start_indices attr-dict `:` functional-type(operands, results)
  }];
}

def DynamicUpdateSliceOp : Impulse_Op<"dynamic_update_slice", [Pure]> {
  let summary = "Update a slice in a tensor at dynamic start indices";
  let description = [{
    Update a slice in a tensor at dynamic start indices.
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
    Computes log(exp(x) + exp(y)).
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
    Computes the logistic (sigmoid) function: 1 / (1 + exp(-x)).
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
    Returns the number of 1-bits elementwise.
  }];

  let arguments = (ins AnyRankedTensor:$operand);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $operand attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// Debug ops
//===----------------------------------------------------------------------===//

def DumpOp : Impulse_Op<"dump"> {
  let summary = "Debug operation to dump a tensor value at runtime";
  let description = [{
    Debug operation that dumps a tensor value with a label.
  }];

  let arguments = (ins
    AnyType:$value,
    StrAttr:$label
  );

  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $value attr-dict `:` functional-type($value, results)
  }];
}

#endif // IMPULSE_OPS
