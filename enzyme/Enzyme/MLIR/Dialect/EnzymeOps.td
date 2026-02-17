//===- EnzymeOps.td - Enzyme dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_OPS
#define ENZYME_OPS

include "EnzymeEnums.td"

include "Dialect.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/IR/AttrTypeBase.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"


def PseudoAliasClassAttr : Enzyme_Attr<"PseudoAliasClass", "pseudoclass"> {
  let summary = "A pseudo alias class represents a memory allocation passed into a function.";

  let parameters = (ins
    "FlatSymbolRefAttr":$function,
    "unsigned":$argNumber,
    "unsigned":$depth
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$function,
                                        "unsigned":$argNumber,
                                        "unsigned":$depth), [{
      return $_get(function.getContext(), function, argNumber, depth);
    }]>
  ];

  let assemblyFormat = "`<` $function `(` $argNumber `,` $depth `)` `>`";
}

def OriginAttrInterface : AttrInterface<"OriginAttr"> {
  let cppNamespace = "::mlir::enzyme";
}

def ArgumentOriginAttr : Enzyme_Attr<"ArgumentOrigin", "argorigin", [OriginAttrInterface]> {
  let summary = "An argument origin is a unique way of referring to a function argument.";

  let parameters = (ins
    "FlatSymbolRefAttr":$function,
    "unsigned":$argNumber
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$function,
                                        "unsigned":$argNumber), [{
      return $_get(function.getContext(), function, argNumber);
    }]>
  ];

  let assemblyFormat = "`<` $function `(` $argNumber `)` `>`";
}

def ReturnOriginAttr : Enzyme_Attr<"ReturnOrigin", "retorigin", [OriginAttrInterface]> {
  let summary = "A return origin is a unique way of referring to a return value.";

  let parameters = (ins
    "FlatSymbolRefAttr":$function,
    "unsigned":$returnNumber
  );

  let builders = [
    AttrBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$function,
                                        "unsigned":$returnNumber), [{
      return $_get(function.getContext(), function, returnNumber);
    }]>
  ];

  let assemblyFormat = "`<` $function `(` $returnNumber `)` `>`";
}

def ActivityArrayAttr : TypedArrayAttrBase<
    ActivityAttr, "Array of argument activity states">;

def PlaceholderOp : Enzyme_Op<"placeholder",
    [Pure]> {
  let results = (outs AnyType:$output);
}

def ForwardDiffOp : Enzyme_Op<"fwddiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform forward mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero); 
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{

    /// Collect all primal input values
    ::llvm::SmallVector<::mlir::Value> getPrimalInputs() {
      return ::mlir::enzyme::detail::filterGradInputs<ForwardDiffOp, false>(
          *this);
    }

    /// Collect all input shadow values(for primals with activity marked as
    /// `enzyme_dup`/`enzyme_dupnoneed`
    ::llvm::SmallVector<::mlir::Value> getShadows() {
      return ::mlir::enzyme::detail::filterGradInputs<ForwardDiffOp, true, true,
                                                      false>(*this);
    }
  }];
}

def JVPOp : Enzyme_Op<"jvp",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Apply Jacobian-vector product for a function";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def AutoDiffOp : Enzyme_Op<"autodiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform reverse mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{

    /// Collect all primal input values
    ::llvm::SmallVector<::mlir::Value> getPrimalInputs() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffOp, false>(*this);
    }

    /// Collect all input shadow values(for primals with activity marked as
    /// `enzyme_dup`/`enzyme_dupnoneed`
    ::llvm::SmallVector<::mlir::Value> getShadows() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffOp, true, true,
                                                      false>(*this);
    }

    /// Collect all output co-tangent values(for outputs with ret_activity
    /// marked as `enzyme_active`/`enzyme_activenoneed`
    ::llvm::SmallVector<::mlir::Value> getDifferentialReturns() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffOp, true, false,
                                                      true>(*this);
    }
  }];
}

def VJPOp : Enzyme_Op<"vjp",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Apply vector-Jacobian product for a function";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def JacobianOp : Enzyme_Op<"jacobian",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Compute Jacobian for a function";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def AutoDiffRegionOp : Enzyme_Op<"autodiff_region", [AutomaticAllocationScope]> {
  let summary = "Perform reverse mode AD on a child region";
  let arguments = (ins Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero, OptionalAttr<StrAttr>:$fn);
  let regions = (region AnyRegion:$body);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `(` $inputs `)` $body attr-dict-with-keyword `:` functional-type($inputs, results)
  }];

  let hasCanonicalizer=1;

  let extraClassDeclaration = [{

    /// Collect all primal input values
    ::llvm::SmallVector<::mlir::Value> getPrimalInputs() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffRegionOp, false>(
          *this);
    }

    /// Collect all input shadow values(for primals with activity marked as
    /// `enzyme_dup`/`enzyme_dupnoneed`
    ::llvm::SmallVector<::mlir::Value> getShadows() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffRegionOp, true,
                                                      true, false>(*this);
    }

    /// Collect all output co-tangent values(for outputs with ret_activity
    /// marked as `enzyme_active`/`enzyme_activenoneed`
    ::llvm::SmallVector<::mlir::Value> getDifferentialReturns() {
      return ::mlir::enzyme::detail::filterGradInputs<AutoDiffRegionOp, true,
                                                      false, true>(*this);
    }

  }];
}

def ForwardDiffRegionOp : Enzyme_Op<"fwddiff_region", [AutomaticAllocationScope]> {
  let summary = "Perform forward mode AD on a child region";
  let arguments = (ins Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity, DefaultValuedAttr<I64Attr, "1">:$width, DefaultValuedAttr<BoolAttr, "false">:$strong_zero, OptionalAttr<StrAttr>:$fn);
  let regions = (region AnyRegion:$body);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    `(` $inputs `)` $body attr-dict-with-keyword `:` functional-type($inputs, results)
  }];
  
  let hasCanonicalizer = 1;
  let extraClassDeclaration = [{

    /// Collect all primal input values
    ::llvm::SmallVector<::mlir::Value> getPrimalInputs() {
      return ::mlir::enzyme::detail::filterGradInputs<ForwardDiffRegionOp, false>(
          *this);
    }

    /// Collect all input shadow values(for primals with activity marked as
    /// `enzyme_dup`/`enzyme_dupnoneed`
    ::llvm::SmallVector<::mlir::Value> getShadows() {
      return ::mlir::enzyme::detail::filterGradInputs<ForwardDiffRegionOp, true,
                                                      true, false>(*this);
    }

  }];
}

def YieldOp : Enzyme_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["AutoDiffRegionOp", "ForwardDiffRegionOp", "ForLoopOp", "WhileLoopOp", "IfOp"]>]> {
  let summary = "Yield values at the end of an autodiff_region, loop, or if ops";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];
}

def ForLoopOp : Enzyme_Op<"for_loop", [AutomaticAllocationScope]> {
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

def BatchOp : Enzyme_Op<"batch",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform reverse mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, DenseI64ArrayAttr:$batch_shape);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def PushOp : Enzyme_Op<"push", [
    TypesMatchWith<"type of 'value' matches element type of 'cache'",
                   "cache", "value",
                   "::llvm::cast<enzyme::CacheType>($_self).getType()">]> {
  let summary = "Push value to cache";
  let arguments = (ins AnyType : $cache, AnyType : $value);
}

def PopOp : Enzyme_Op<"pop", [
    TypesMatchWith<"type of 'output' matches element type of 'cache'",
                   "cache", "output",
                   "::llvm::cast<enzyme::CacheType>($_self).getType()">]> {
  let summary = "Retrieve information for the reverse mode pass.";
  let arguments = (ins AnyType : $cache);
  let results = (outs AnyType:$output);
}

def InitOp : Enzyme_Op<"init",
    [DeclareOpInterfaceMethods<PromotableAllocationOpInterface>]> {
  let summary = "Create enzyme.gradient and enzyme.cache";
  let arguments = (ins );
  let results = (outs AnyType);
}

def Cache : Enzyme_Type<"Cache"> {
  let summary = "Cache for reverse pass";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$type);
  let mnemonic = "Cache";
  let assemblyFormat = "`<` $type `>`";
}

def Gradient : Enzyme_Type<"Gradient"> {
  let summary = "Mutable storage for accumulating gradients";
  let description = [{
    Mutable storage for accumulating derivatives of immutable types (e.g. adding all the partial derivatives from users of a float64)
  }];
  let parameters = (ins "Type":$basetype);
  let mnemonic = "Gradient";
  let assemblyFormat = "`<` $basetype `>`";
}

def LoadOp : Enzyme_Op<"load",
    [TypesMatchWith<"type of 'result' matches element type of 'cache'",
                    "cache", "result",
                    "::llvm::cast<enzyme::CacheType>($_self).getType()">]> {
  let summary = "Load a value from a cache at the specified indices";
  let arguments = (ins Cache:$cache, Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$cache `[` $indices `]` attr-dict `:` type($cache)";
}

def StoreOp : Enzyme_Op<"store",
    [TypesMatchWith<"type of 'value' matches element type of 'cache'",
                     "cache", "value",
                     "::llvm::cast<enzyme::CacheType>($_self).getType()">]> {
  let summary = "Store a value to a cache at the specific indices";
  let arguments = (ins AnyType:$value, Cache:$cache, Variadic<Index>:$indices);
  let assemblyFormat = [{
    $value `,` $cache `[` $indices `]` attr-dict `:` type($cache)
  }];
}

def SetOp : Enzyme_Op<"set",
    [DeclareOpInterfaceMethods<PromotableMemOpInterface>,
     DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
     TypesMatchWith<"type of 'value' matches element type of 'gradient'",
                    "gradient", "value",
                    "::llvm::cast<enzyme::GradientType>($_self).getBasetype()">]> {
  let summary = "Store the current value of the gradient";
  let arguments = (ins Arg<AnyType, "the reference to store to",
                           [MemWrite]>:$gradient, AnyType : $value);
  let results = (outs );
}

def GetOp : Enzyme_Op<"get",
    [DeclareOpInterfaceMethods<PromotableMemOpInterface>,
     DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
     TypesMatchWith<"type of 'result' matches element type of 'gradient'",
                    "gradient", "result",
                    "::llvm::cast<enzyme::GradientType>($_self).getBasetype()">]> {
  let summary = "Load current value of gradient";
  let arguments = (ins Arg<AnyType, "the reference to load from",
                           [MemRead]>:$gradient);
  let results = (outs AnyType:$result);
}

def AddToOp : Enzyme_Op<"addTo", [Pure, Terminator, ReturnLike]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Linalg add to operation";
  let description = [{
    TODO
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

def GenericAdjointOp : Enzyme_Op<"genericAdjoint", [AttrSizedOperandSegments]> {
  let description = [{ }];

  let arguments = (ins Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       AffineMapArrayAttr:$indexing_maps,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<StrAttr>:$doc,
                       OptionalAttr<StrAttr>:$library_call);
  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);
  let regions = (region AnyRegion:$region);
  
}

def BroadcastOp : Enzyme_Op<"broadcast"> {
  let description = [{
  Broadcast the operand by adding extra dimensions with sizes provided by the `shape` attribute to the front.
  For scalar operands, ranked tensor is created.

  NOTE: Only works for scalar and *ranked* tensor operands for now.
  }];

  let arguments = (ins AnyType:$input, DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$output);

  let builders = [
    OpBuilder<(ins "Value":$input, "ArrayRef<int64_t>":$shape)>
  ];
}

def IgnoreDerivativesOp : Enzyme_Op<"ignore_derivatives",
    [Pure, SameOperandsAndResultElementType, SameOperandsAndResultShape]> {
  let summary = "Prevents the flow of gradients (and higher-order derivatives) by creating a new value that is detached from the original value. This is an identity operation on the primal.";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
}

def ConcatOp : Enzyme_Op<"concat"> {
  let summary = "Concatenate list of input arguments into a single batched value";
  let description = [{
    Concat list of input arguments into a generic value
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
    `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def ExtractOp : Enzyme_Op<"extract"> {
  let summary = "Extract value from batched operand at index";
  let description = [{
    Extract value from batched operand at index
  }];

  let arguments = (ins AnyType:$input, I64Attr:$index);
  let results = (outs AnyType:$output);
  
  let assemblyFormat = [{
     $input `[` $index `]` attr-dict `:` functional-type($input, results)
  }];
}

// Probabilistic programming
def SymbolAttr : Enzyme_Attr<"Symbol", "symbol"> {
  let summary = "Symbol associated with a Sample op";
  let description = [{
  Symbol associated with a Sample op.
  }];
  let parameters = (ins "uint64_t":$ptr);
  let assemblyFormat = "`<` $ptr `>`";
}

def AddressAttr : TypedArrayAttrBase<SymbolAttr, "Address as an array of symbols">;
def AddressArrayAttr : TypedArrayAttrBase<AddressAttr, "Array of addresses">;

def SampleOp : Enzyme_Op<"sample",
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

def UntracedCallOp : Enzyme_Op<"untracedCall"> {
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

def SimulateOp : Enzyme_Op<"simulate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

def GenerateOp : Enzyme_Op<"generate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

def RandomOp : Enzyme_Op<"random"> {
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

def RandomSplitOp : Enzyme_Op<"randomSplit"> {
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

def RegenerateOp : Enzyme_Op<"regenerate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

def MHOp : Enzyme_Op<"mh", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
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

def MCMCOp : Enzyme_Op<"mcmc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]> {
  let summary = "MCMC inference for probabilistic programs";
  let description = [{
    Runs MCMC inference on selected addresses.

    Two modes of operation:
    1. Trace-based mode: `fn` and `original_trace` are provided. The model
       function with `enzyme.sample` ops defines the density.
    2. Custom logpdf mode: `logpdf_fn` and `initial_position` are provided.
       The logpdf function maps position → scalar log-density directly.

    The `selection` attribute determines which addresses to sample via HMC/NUTS.
    All sample addresses are included in the trace tensor for consistency.

    Returns: (trace, diagnostics, rng)
    - trace: tensor<num_samples x position_size x f64>
    - diagnostics: tensor<num_samples x i1> - placeholder for future expansion
    - rng: updated RNG state
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

    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs
    AnyRankedTensor:$trace,
    AnyRankedTensor:$diagnostics,
    AnyType:$output_rng_state
  );

  let assemblyFormat = [{
    ($fn^)?
    `(` $inputs `)`
    (`given` $original_trace^)?
    (`inverse_mass_matrix` `=` $inverse_mass_matrix^)?
    (`step_size` `=` $step_size^)?
    (`logpdf_fn` `=` $logpdf_fn^)?
    (`initial_position` `=` $initial_position^)?
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def DotOp : Enzyme_Op<"dot", [Pure]> {
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

def CholeskyOp : Enzyme_Op<"cholesky", [Pure]> {
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

def TriangularSolveOp : Enzyme_Op<"triangular_solve", [Pure]> {
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

def SelectOp : Enzyme_Op<"select", [Pure]> {
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

def ReshapeOp : Enzyme_Op<"reshape", [Pure]> {
  let summary = "Reshape a tensor to a new static shape";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];
}

def SliceOp : Enzyme_Op<"slice", [Pure]> {
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

def DynamicSliceOp : Enzyme_Op<"dynamic_slice", [Pure]> {
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

def DynamicUpdateSliceOp : Enzyme_Op<"dynamic_update_slice", [Pure]> {
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

def DumpOp : Enzyme_Op<"dump"> {
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

def AffineAtomicRMWOp : Enzyme_Op<"affine_atomic_rmw"> {
  let summary = "affine atomic rmw operation";
  let description = [{
  }];

  let results = (outs AnyType:$result);

  let arguments = (ins
      AtomicRMWKindAttr:$kind,
      AnyType:$value,
      Arg<AnyMemRef, "the reference to rmw", [MemRead, MemWrite]>:$memref,
      Variadic<Index>:$indices,
      AffineMapAttr:$map);

  let assemblyFormat = [{
    $kind $value `,` $memref `,` `(` $map `)` `[` $indices `]` attr-dict `:` `(` type($value) `,`
    type($memref) `)` `->` type($result)
  }];
}

def WhileLoopOp : Enzyme_Op<"while_loop", [AutomaticAllocationScope]> {
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

def IfOp : Enzyme_Op<"if", [AutomaticAllocationScope, RecursiveMemoryEffects]> {
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

def LogAddExpOp : Enzyme_Op<"log_add_exp", [Pure]> {
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

def LogisticOp : Enzyme_Op<"logistic", [Pure]> {
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

def PopcountOp : Enzyme_Op<"popcount", [Pure, SameOperandsAndResultElementType, SameOperandsAndResultShape]> {
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

#endif // ENZYME_OPS
