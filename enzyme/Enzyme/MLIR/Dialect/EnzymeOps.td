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
    ParentOneOf<["AutoDiffRegionOp", "ForwardDiffRegionOp", "ForLoopOp", "WhileLoopOp"]>]> {
  let summary = "Yield values at the end of an autodiff_region or loop ops";
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
def Trace : Enzyme_Type<"Trace"> {
  let summary = "Execution trace for probabilistic programs";
  let description = [{
    Reference: https://www.gen.dev/docs/stable/tutorials/basics/gfi/#Traces
    Mutable storage for mapping all sampled symbols to their values, log-likelihoods,
    and various other information from executing a traced probabilistic function.
  }];
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "0">:$num_samples
  );
  let mnemonic = "Trace";
  let assemblyFormat = "(`<` $num_samples^ `>`)?";

  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, 0);
    }]>
  ];
}

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

def Constraint : Enzyme_Type<"Constraint"> {
  let summary = "A mutable storage mapping symbols to value constraints";
  let description = [{
  A mutable storage mapping symbols to value constraints.
  }];
  let mnemonic = "Constraint";
}

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
  let summary = "Simulate a probabilistic function to generate execution trace";
  let description = [{
  Simulate a probabilistic function to generate execution trace
  by replacing all SampleOps with distribution calls and recording
  all sampled values into the trace. This op returns the trace, the weight
  (accumulated log-probability), and the other outputs. By convention,
  the 0th operand in `inputs` or `outputs` is the initial RNG state (seed).
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$trace, AnyRankedTensor:$weight, Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def GenerateOp : Enzyme_Op<"generate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Generate an execution trace and weight from a probabilistic function";
  let description = [{
  Generate an execution trace and weight from a probabilistic function.
  If a `constraint` dict is provided AND the sample op's `symbol` is in the
  `constrained_symbols` array, we will use the corresponding constraint value
  instead of generating new samples from the probabilistic function.
  By convention, the 0th operand in `inputs` or `outputs` is the initial RNG
  state (seed).
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    AddressArrayAttr:$constrained_addresses,
    Constraint:$constraint,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$trace, AnyRankedTensor:$weight, Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $constraint attr-dict `:` functional-type($inputs, results)
  }];
}

def InitTraceOp : Enzyme_Op<"initTrace", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Initialize an execution trace for a probabilistic function";
  let description = [{
    Initialize an execution trace for a probabilistic function.
  }];
  let arguments = (ins );
  let results = (outs Trace:$trace);
  let assemblyFormat = "attr-dict `:` type($trace)";
  let hasCanonicalizer = 1;
}

def AddSampleToTraceOp : Enzyme_Op<"addSampleToTrace", [Pure]> {
  let summary = "Add a sampled value into the execution trace";
  let description = [{
  Add a sampled value into the execution trace.
  }];

  let arguments = (ins
    Trace:$trace,
    SymbolAttr:$symbol,
    Variadic<AnyType>:$sample
  );

  let results = (outs Trace:$updated_trace);

  let assemblyFormat = [{
    $sample `into` $trace attr-dict `:` functional-type(operands, results)
  }];
}

def AddSubtraceOp : Enzyme_Op<"addSubtrace", [Pure]> {
  let summary = "Insert a subtrace into a parent trace";
  let description = [{
  Insert a subtrace into a parent trace.
  }];

  let arguments = (ins
    Trace:$subtrace,
    SymbolAttr:$symbol,
    Trace:$trace
  );

  let results = (outs Trace:$updated_trace);

  let assemblyFormat = [{
    $subtrace `into` $trace attr-dict `:` functional-type(operands, results)
  }];
}

def AddWeightToTraceOp : Enzyme_Op<"addWeightToTrace", [Pure]> {
  let summary = "Add the aggregated weight into the execution trace";
  let description = [{
  Add the aggregated log-probability weight to the execution trace.
  }];

  let arguments = (ins
    Trace:$trace,
    AnyType:$weight
  );

  let results = (outs Trace:$updated_trace);

  let assemblyFormat = [{
    $weight `into` $trace attr-dict `:` functional-type(operands, results)
  }];
}

def AddRetvalToTraceOp : Enzyme_Op<"addRetvalToTrace", [Pure]> {
  let summary = "Add the function's return value(s) into the execution trace";
  let description = [{
  Add the function's return value(s) into the execution trace.
  }];

  let arguments = (ins
    Trace:$trace,
    Variadic<AnyType>:$retval
  );

  let results = (outs Trace:$updated_trace);

  let assemblyFormat = [{
    $retval `into` $trace attr-dict `:` functional-type(operands, results)
  }];
}

def GetSampleFromConstraintOp : Enzyme_Op<"getSampleFromConstraint"> {
  let summary = "Get sampled values from a constraint for a given symbol";
  let description = [{
  Get sampled values from a constraint for a given symbol.
  }];
  let arguments = (ins Constraint:$constraint, SymbolAttr:$symbol);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $constraint attr-dict `:` type(results)
  }];
}

def GetSubconstraintOp : Enzyme_Op<"getSubconstraint"> {
  let summary = "Get a subconstraint from a constraint for a given symbol";
  let description = [{
  Get a subconstraint from a constraint for a given symbol.
  }];
  let arguments = (ins Constraint:$constraint, SymbolAttr:$symbol);

  let results = (outs Constraint:$subconstraint);

  let assemblyFormat = [{
    $constraint attr-dict
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

def GetSubtraceOp : Enzyme_Op<"getSubtrace", [Pure]> {
  let summary = "Get a subtrace from a trace for a given symbol";
  let description = [{
    Get a subtrace from a trace for a given symbol.
  }];
  let arguments = (ins Trace:$trace, SymbolAttr:$symbol);

  let results = (outs Trace:$subtrace);

  let assemblyFormat = [{
    $trace attr-dict `:` functional-type(operands, results)
  }];
}

def GetSampleFromTraceOp : Enzyme_Op<"getSampleFromTrace", [Pure]> {
  let summary = "Get the sampled value for a given symbol from an execution trace";
  let description = [{
    Get the sampled value for a given symbol from an execution trace.
  }];
  let arguments = (ins Trace:$trace, SymbolAttr:$symbol);

  let results = (outs Variadic<AnyType>:$sample);

  let assemblyFormat = [{
    $trace attr-dict `:` functional-type(operands, results)
  }];
}

def GetWeightFromTraceOp : Enzyme_Op<"getWeightFromTrace", [Pure]> {
  let summary = "Get the accumulated weight from an execution trace";
  let description = [{
    Get the accumulated log-probability weight from an execution trace.
  }];
  let arguments = (ins Trace:$trace);

  let results = (outs AnyRankedTensor:$weight);

  let assemblyFormat = [{
    $trace attr-dict `:` functional-type(operands, results)
  }];
}

def GetFlattenedSamplesFromTraceOp : Enzyme_Op<"getFlattenedSamplesFromTrace", [Pure]> {
  let summary = "Get multiple sampled values from a trace and flatten to position vector";
  let description = [{
    Get sampled values for multiple addresses from an execution trace and
    flatten them into a single position vector for HMC.
  }];
  let arguments = (ins Trace:$trace, AddressArrayAttr:$selection);

  let results = (outs AnyType:$position);

  let assemblyFormat = [{
    $trace attr-dict `:` functional-type(operands, results)
  }];
}

def RecoverSampleOp : Enzyme_Op<"recover_sample", [Pure]> {
  let summary = "Recover a sample from a flattened position vector";
  let description = [{
    Recover a sample from a flattened position vector.
  }];

  let arguments = (ins
    AnyRankedTensor:$position,
    I64Attr:$offset
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $position `[` $offset `]` attr-dict `:` type($position) `->` type($result)
  }];
}

def UpdateOp : Enzyme_Op<"update", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Update trace with new position values and recompute weight";
  let description = [{
    Update selected addresses in a trace with new values from a position vector,
    re-evaluate the probabilistic function, and return the updated trace with
    the new weight (log probability) and updated RNG state.
    By convention, the 0th operand in inputs is the initial RNG state.
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    Trace:$original_trace,
    AnyRankedTensor:$position,
    AddressArrayAttr:$selection,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$updated_trace, AnyRankedTensor:$weight, AnyType:$output_rng_state);

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace `at` $position attr-dict `:` functional-type(operands, results)
  }];
}

def RegenerateOp : Enzyme_Op<"regenerate", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Regenerate selected addresses in a probabilistic function";
  let description = [{
    Regenerate selected addresses in a probabilistic function while keeping
    other addresses fixed to their values in the given trace.
    By convention, the 0th operand in inputs is the initial RNG state
    and the 0th operand in results is the updated RNG state.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    Trace:$original_trace,
    AddressArrayAttr:$selection,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$trace, AnyRankedTensor:$weight, AnyType:$output_rng_state);

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace attr-dict `:` functional-type(operands, results)
  }];
}

def MHOp : Enzyme_Op<"mh", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Metropolis-Hastings step for probabilistic inference";
  let description = [{
    Perform a Metropolis-Hastings step on a probabilistic function.
    This operation proposes a new trace by regenerating selected addresses,
    computes the acceptance probability, and returns the updated trace.
    By convention, the 0th operand in inputs is the initial RNG state
    and the 0th operand in results is the updated RNG state.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    Trace:$original_trace,
    AddressArrayAttr:$selection,
    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$new_trace, AnyRankedTensor:$accepted, AnyType:$output_rng_state);

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace attr-dict `:` functional-type(operands, results)
  }];
}

def MCMCOp : Enzyme_Op<"mcmc", [DeclareOpInterfaceMethods<SymbolUserOpInterface>, AttrSizedOperandSegments]> {
  let summary = "MCMC inference for probabilistic programs";
  let description = [{
    Performs MCMC inference on a probabilistic function.

    By default (num_warmup=0, num_samples=1, thinning=1), performs a single MCMC step
    and returns the resulting trace - backwards compatible with previous behavior.

    When num_samples > 1, runs a full MCMC chain and returns a trace where
    sampled values have an additional batch dimension (first dim = num_samples / thinning).

    Thinning controls the fraction of samples retained: when thinning=k, every k-th
    sample is kept, resulting in num_samples/k output samples.

    Warmup iterations are discarded and used only for adaptation.

    By convention, the 0th operand in inputs is the initial RNG state
    and the 0th operand in results is the updated RNG state.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    Variadic<AnyType>:$inputs,
    Trace:$original_trace,
    AddressArrayAttr:$selection,

    DefaultValuedAttr<I64Attr, "0">:$num_warmup,
    DefaultValuedAttr<I64Attr, "1">:$num_samples,
    DefaultValuedAttr<I64Attr, "1">:$thinning,

    // Algorithm-specific MCMC parameters
    Optional<AnyRankedTensor>:$inverse_mass_matrix,
    Optional<AnyRankedTensor>:$step_size,

    // Algorithm-specific configurations
    OptionalAttr<HMCConfigAttr>:$hmc_config,
    OptionalAttr<NUTSConfigAttr>:$nuts_config,

    DefaultValuedStrAttr<StrAttr, "">:$name
  );

  let results = (outs Trace:$new_trace, AnyRankedTensor:$accepted, AnyType:$output_rng_state);

  let assemblyFormat = [{
    $fn `(` $inputs `)` `given` $original_trace
    (`inverse_mass_matrix` `=` $inverse_mass_matrix^)?
    (`step_size` `=` $step_size^)?
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
    - `!enzyme.Trace` operands
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

def DynamicExtractOp : Enzyme_Op<"dynamic_extract", [Pure]> {
  let summary = "Extract slice from tensor at dynamic index along first dimension";
  let description = [{
    Extracts a slice at the specified dynamic index along the first dimension.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$index
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $index attr-dict `:` functional-type(operands, results)
  }];
}

def DynamicUpdateOp : Enzyme_Op<"dynamic_update", [Pure]> {
  let summary = "Update slice in tensor at dynamic index along first dimension";
  let description = [{
    Returns a new tensor with the slice at the specified dynamic index replaced.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$index,
    AnyRankedTensor:$value
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `,` $index `,` $value attr-dict `:` functional-type(operands, results)
  }];
}

#endif // ENZYME_OPS
