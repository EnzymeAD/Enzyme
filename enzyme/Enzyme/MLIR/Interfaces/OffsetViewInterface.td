//===- OffsetViewInterface.td -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_OFFSETVIEWINTERFACE
#define ENZYME_MLIR_INTERFACES_OFFSETVIEWINTERFACE

include "mlir/Interfaces/ViewLikeInterface.td"

def OffsetViewInterface
    : OpInterface<"OffsetViewInterface", [ViewLikeOpInterface]> {
  let cppNamespace = "::mlir::enzyme";
  let description = [{
    Queries whether a view has the same starting address as its source,
    preserving any offset that the source already has. The shape and strides
    can differ. Implementing this interface alone does not guarantee that the
    address is preserved; isZeroOffset must return true for that instance.
    Use the inherited getViewSource and getViewDest methods to get the source
    and the view.
  }];

  let methods = [
    InterfaceMethod<
      [{
        Returns true if the view is proven to have the same starting address
        as its source, including any existing source offset. Returns false if
        the address changes or cannot be proven to be preserved.

        The generated model returns true for operations that always preserve
        the address. Operations that can change it must provide a custom model.
      }],
      "bool", "isZeroOffset", (ins),
      /*methodBody=*/[{ return true; }]
    >
  ];
}

#endif // ENZYME_MLIR_INTERFACES_OFFSETVIEWINTERFACE
