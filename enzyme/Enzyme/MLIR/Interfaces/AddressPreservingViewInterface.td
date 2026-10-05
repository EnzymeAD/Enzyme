//===- AddressPreservingViewInterface.td ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_ADDRESSPRESERVINGVIEWINTERFACE
#define ENZYME_MLIR_INTERFACES_ADDRESSPRESERVINGVIEWINTERFACE

include "mlir/Interfaces/ViewLikeInterface.td"

def AddressPreservingViewInterface
    : OpInterface<"AddressPreservingViewInterface", [ViewLikeOpInterface]> {
  let cppNamespace = "::mlir::enzyme";
  let description = [{
    The view has the same starting address as its source. It preserves any
    offset that the source already has. The shape and strides can differ.

    This guarantee applies to every instance of the operation. An operation
    that can change the starting address must not implement this interface.
    Use the inherited getViewSource and getViewDest methods to get the source
    and the view.
  }];
}

#endif // ENZYME_MLIR_INTERFACES_ADDRESSPRESERVINGVIEWINTERFACE
