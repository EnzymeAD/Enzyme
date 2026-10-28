//===- MemTransferOpInterface.td --------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_MEMTRANSFEROPINTERFACE
#define ENZYME_MLIR_INTERFACES_MEMTRANSFEROPINTERFACE

include "mlir/IR/OpBase.td"

def MemTransferOpInterface : OpInterface<"MemTransferOpInterface"> {
  let cppNamespace = "::mlir::enzyme";
  let description = [{
    An operation that copies memory from a source pointer into a destination
    pointer, such as memcpy or memmove.
  }];

  let methods = [
    InterfaceMethod<
      "::mlir::Value", "getTransferDest", (ins)
    >,
    InterfaceMethod<
      "::mlir::Value", "getTransferSource", (ins)
    >
  ];
}

#endif // ENZYME_MLIR_INTERFACES_MEMTRANSFEROPINTERFACE
