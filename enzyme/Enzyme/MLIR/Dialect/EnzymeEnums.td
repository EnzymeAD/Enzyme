//===- EnzymeOps.td - Enzyme dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_ENUMS
#define ENZYME_ENUMS

include "mlir/IR/EnumAttr.td"
include "Dialect.td"

class Enzyme_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Enzyme_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
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

def ActivityAttr : EnumAttr<Enzyme_Dialect, Activity, "activity">{
  let extraClassDeclaration = [{
    using ValueType = ::mlir::enzyme::Activity;
  }];
}

def AtomicOrdering : I32EnumAttr<"Ordering",
    "Atomic ordering for LLVM's memory model",
    [
      I32EnumAttrCase<"not_atomic", 0>,
      I32EnumAttrCase<"unordered", 1>,
      I32EnumAttrCase<"monotonic", 2>,
      I32EnumAttrCase<"acquire", 4>,
      I32EnumAttrCase<"release", 5>,
      I32EnumAttrCase<"acq_rel", 6>,
      I32EnumAttrCase<"seq_cst", 7>,
    ]> {
  let cppNamespace = "::mlir::enzyme";
}

def AtomicOrdering : I32EnumAttr<"Ordering",
    "Atomic ordering for LLVM's memory model",
    [
      I32EnumAttrCase<"not_atomic", 0>,
      I32EnumAttrCase<"unordered", 1>,
      I32EnumAttrCase<"monotonic", 2>,
      I32EnumAttrCase<"acquire", 4>,
      I32EnumAttrCase<"release", 5>,
      I32EnumAttrCase<"acq_rel", 6>,
      I32EnumAttrCase<"seq_cst", 7>,
    ]> {
  let cppNamespace = "::mlir::enzyme";
}

#endif // ENZYME_ENUMS
