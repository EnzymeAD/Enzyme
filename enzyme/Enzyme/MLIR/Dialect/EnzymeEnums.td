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

def ActivityAttr : EnumAttr<Enzyme_Dialect, Activity, "activity">;

#endif // ENZYME_ENUMS
