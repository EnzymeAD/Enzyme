//===-- AutoDiffOpInterface.td - Type interface ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type interfaces necessary to implement scalable
// automatic differentiation across an unbounded number of MLIR IR constructs.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE
#define ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE

include "mlir/IR/OpBase.td"

def AutoDiffTypeInterface : TypeInterface<"AutoDiffTypeInterface"> {
  let description = [{
    An MLIR type values of which can be processed by automatic differentiation.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Constructs the canonical null (zero) value for the given type. The
        provided builder should be used to construct the const-like operation.
        Null can be emitted by the differentiation pass.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"createNullValue",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Construct addition operation for the given type.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"createAddOp",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc, "::mlir::Value":$a, "::mlir::Value":$b)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Zero the operation in place
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"zeroInPlace",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc, "::mlir::Value":$val)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Check if the value with the given type is 0
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"isZero",
      /*args=*/(ins  "::mlir::Value":$val)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Check if the mlir Attribute with the given type is 0.       
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"isZeroAttr",
      /*args=*/(ins  "::mlir::Attribute":$attr)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns the type that can contain the adjoint value for this type. If
        `width` is provided, returns the type that can store `width` instances
        of the adjoint.
      }],
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getShadowType",
      /*args=*/(ins "int64_t":$width)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Construct complex conjugate for the given type.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"createConjOp",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc, "::mlir::Value":$a)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns whether the type is mutable in place or not.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isMutable",
      /*args=*/(ins )
    >,
    InterfaceMethod<
      /*desc=*/[{
        Returns an estimate of the size of a type.

        `INT64_MAX` if not known.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getApproxSize",
      /*args=*/(ins ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return INT64_MAX;
      }]
    >
  ];

  let extraSharedClassDeclaration = [{
    ::mlir::Type getShadowType() const {
      return $_type.getShadowType(/*width=*/1);
    }
  }];
}

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE
