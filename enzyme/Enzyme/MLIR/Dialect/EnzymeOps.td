//===- EnzymeOps.td - Enzyme dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_OPS
#define ENZYME_OPS

include "Dialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/AttrTypeBase.td"

def Activity : I32EnumAttr<"Activity",
    "Possible activity states for variables",
    [
      I32EnumAttrCase<"enzyme_out",  0>,
      I32EnumAttrCase<"enzyme_dup",  1>,
      I32EnumAttrCase<"enzyme_const",2>,
      I32EnumAttrCase<"enzyme_dupnoneed", 3>,
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
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def DiffOp : Enzyme_Op<"diff",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Perform reverse mode AD on a funcop";
  let arguments = (ins FlatSymbolRefAttr:$fn, Variadic<AnyType>:$inputs, ActivityArrayAttr:$activity);
  let results = (outs Variadic<AnyType>:$outputs);

  let assemblyFormat = [{
    $fn `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def PushCacheOp : Enzyme_Op<"push"> {
  let summary = "Store information for the reverse mode pass.";
  let arguments = (ins AnyType : $cache, AnyType : $value);
  let results = (outs );
}

def PopCacheOp : Enzyme_Op<"pop"> {
  let summary = "Retrieve information for the reverse mode pass.";
  let arguments = (ins AnyType : $cache);
  let results = (outs AnyType:$output);
}

def InitOp : Enzyme_Op<"init"> {
  let summary = "Creat enzyme.gradient and enzyme.cache";
  let arguments = (ins );
  let results = (outs AnyType);
}

def ReverseCache : Enzyme_Type<"Cache"> {
  let summary = "Cache for reverse pass";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$type);
  let mnemonic = "Cache";
  let assemblyFormat = "`<` $type `>`";
}

def SetGradientOp : Enzyme_Op<"set"> {
  let summary = "Write to gradient";
  let arguments = (ins AnyType : $gradient, AnyType : $value);
  let results = (outs );
}

def GetGradientOp : Enzyme_Op<"get"> {
  let summary = "Load value of gradient";
  let arguments = (ins AnyType : $gradient);
  let results = (outs AnyType);
}

def Gradient : Enzyme_Type<"Gradient"> {
  let summary = "Gradient storage";
  let description = [{
    "Cache for reverse pass"
  }];
  let parameters = (ins "Type":$basetype);
  let mnemonic = "Gradient";
  let assemblyFormat = "`<` $basetype `>`";
}

#endif // ENZYME_OPS
