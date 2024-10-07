//===-- AutoDiffOpInterface.td - Op Interface for auto diff-*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces necessary to implement scalable automatic
// differentiation across an unbounded number of MLIR IR constructs.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES
#define ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES

include "mlir/IR/OpBase.td"

def AutoDiffOpInterface : OpInterface<"AutoDiffOpInterface"> {
  let description = [{
    A differentiable MLIR operation that is able to emit forward mode adjoints for itself.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
    /*desc=*/[{
      Emits a forward-mode tangent of the given function. All IR manipulation
      must go through the supplied arguments: `builder` is preset to insert new
      IR in the correct location and should be used to construct any new IR;
      `gutils` provides the mapping between main and derivative computation for
      values and operations and should be updated to keep track of newly created
      IR entities as well as of deleted entities. Returns success if generation
      was successful.
    }],
    /*retTy=*/"::mlir::LogicalResult",
    /*methodName=*/"createForwardModeTangent",
    /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::enzyme::MGradientUtils *":$gutils)
    >
  ];
}

def ControlFlowAutoDiffOpInterface 
    : OpInterface<"ControlFlowAutoDiffOpInterface"> {
  let description = [{
    A differentiable MLIR operation whose forward differentiation rules are
    driven by how control flows through the operation.

    There are two general assumptions:
      - the operation can communicate additional values along the control flow
        edges, which is used to put shadow values immediately after the primal
        values;
      - all values returned by the operation are yielded by all region-exiting
        terminators.
  }];
  let cppNamespace = "::mlir::enzyme";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Creates a copy of this operation additionally carrying required shadow
        values along control flow edges using the given builder. The `original`
        is the operation in the original primal code prior to differentiation,
        and this method is supposed to be called on the operation in the cloned
        function being constructed. Remapped operands contains a flat list of
        operands usable in the cloned function and can be fed to the Adaptor
        constructor.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"createWithShadows",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
                    "::mlir::enzyme::MGradientUtils *":$gutils,
                    "::mlir::Operation *":$original, 
                    "::mlir::ValueRange":$remappedOperands,
                    "::mlir::TypeRange":$returnTypes)
    >
  ];
}

def ReverseAutoDiffOpInterface : OpInterface<"ReverseAutoDiffOpInterface"> {
  let description = [{
    A differentiable MLIR operation that is able to emit reverse mode adjoints for itself.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
    /*desc=*/[{
      Emits a reverse-mode adjoint of the given function.
    }],
    /*retTy=*/"::mlir::LogicalResult",
    /*methodName=*/"createReverseModeAdjoint",
    /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::enzyme::MGradientUtilsReverse *":$gutils, "SmallVector<Value>":$caches)
    >,

    InterfaceMethod<
    /*desc=*/[{
      Creates or alters shadow values.
    }],
    /*retTy=*/"void",
    /*methodName=*/"createShadowValues",
    /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::enzyme::MGradientUtilsReverse *":$gutils)
    >,

    InterfaceMethod<
    /*desc=*/[{
      Caches values to create the adjoint.
    }],
    /*retTy=*/"SmallVector<Value>",
    /*methodName=*/"cacheValues",
    /*args=*/(ins "::mlir::enzyme::MGradientUtilsReverse *":$gutils)
    >
  ];
}

def ActivityOpInterface
    : OpInterface<"ActivityOpInterface"> {
  let cppNamespace = "::mlir::enzyme";
  
  let methods = [
    InterfaceMethod<
    /*desc=*/[{
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isInactive"
    >,
    InterfaceMethod<
    /*desc=*/[{
    }],
    /*retTy=*/"bool",
    /*methodName=*/"isArgInactive",
    /*args=*/(ins "size_t":$opidx)
    >
  ];
}

def ADDataFlowOpInterface
    : OpInterface<"ADDataFlowOpInterface"> {
  let cppNamespace = "::mlir::enzyme";
  
  let methods = [
    InterfaceMethod<
    /*desc=*/[{
    }],
    /*retTy=*/"SmallVector<Value>",
    /*methodName=*/"getPotentialIncomingValuesRes",
    /*args=*/(ins "::mlir::OpResult":$v)
    >,
    InterfaceMethod<
    /*desc=*/[{
    }],
    /*retTy=*/"SmallVector<Value>",
    /*methodName=*/"getPotentialIncomingValuesArg",
    /*args=*/(ins "::mlir::BlockArgument":$v)
    >,
    InterfaceMethod<
    /*desc=*/[{
    }],
    /*retTy=*/"SmallVector<Value>",
    /*methodName=*/"getPotentialTerminatorUsers",
    /*args=*/(ins "::mlir::Operation*":$terminator, "::mlir::Value":$val)
    >
  ];
}

def BatchOpInterface : OpInterface<"BatchOpInterface"> {
  let description = [{
    A batchable MLIR operation.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
    /*desc=*/[{
      Emits a batched version of a given operation.
    }],
    /*retTy=*/"::mlir::Operation*",
    /*methodName=*/"createBatch",
    /*args=*/(ins "::mlir::IRMapping &":$mapper, "::mlir::Operation::CloneOptions":$options, "::std::map<mlir::Operation*, mlir::Operation*>&":$opMap,  "::llvm::ArrayRef<int64_t>":$batchSizes)
    >
  ];
}

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFOPINTERFACES
