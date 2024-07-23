//===- EnzymeOps.td - Enzyme dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//include "mlir/Dialect/Linalg/IR/LinalgBase.td"

#ifndef ENZYME_OPS
#define ENZYME_OPS

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
include "mlir/Interfaces/SideEffectInterfaces.td"


class Enzyme_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Enzyme_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

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

def Activity : I32EnumAttr<"Activity",
    "Possible activity states for variables",
    [
      I32EnumAttrCase<"enzyme_active",  0>,
      I32EnumAttrCase<"enzyme_dup",  1>,
      I32EnumAttrCase<"enzyme_const",2>,
      I32EnumAttrCase<"enzyme_dupnoneed", 3>,
      I32EnumAttrCase<"enzyme_activenoneed",  4>,
      I32EnumAttrCase<"enzyme_constnoneed",  5>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzyme";
}

def ActivityAttr : EnumAttr<Enzyme_Dialect, Activity, "activity">;

def ActivityArrayAttr : TypedArrayAttrBase<
    ActivityAttr, "Array of argument activity states">;

def PlaceholderOp : Enzyme_Op<"placeholder",
    [Pure]> {
  let results = (outs AnyType:$output);
}

def ForwardDiffOp : Enzyme_Op<"fwddiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform forward mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def AutoDiffOp : Enzyme_Op<"autodiff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform reverse mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity, ActivityArrayAttr:$ret_activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
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

def PushOp : Enzyme_Op<"push"> {
  let summary = "Push value to cache or ShadowedGradient";
  let arguments = (ins AnyType : $cache, AnyType : $value);
  let results = (outs );
}

def PopOp : Enzyme_Op<"pop"> {
  let summary = "Retrieve information for the reverse mode pass.";
  let arguments = (ins AnyType : $cache);
  let results = (outs AnyType:$output);
}

def InitOp : Enzyme_Op<"init"> {
  let summary = "Creat enzyme.gradient and enzyme.cache";
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

def SetOp : Enzyme_Op<"set"> {
  let summary = "Store the current value of the gradient";
  let arguments = (ins Arg<AnyType, "the reference to store to",
                           [MemWrite]>:$gradient, AnyType : $value);
  let results = (outs );
}

def GetOp : Enzyme_Op<"get"> {
  let summary = "Load current value of gradient";
  let arguments = (ins Arg<AnyType, "the reference to load from",
                           [MemRead]>:$gradient);
  let results = (outs AnyType);
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

#endif // ENZYME_OPS
