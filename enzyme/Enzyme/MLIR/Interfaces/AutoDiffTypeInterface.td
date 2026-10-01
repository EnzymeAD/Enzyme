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
        Constructs the canonical null (zero) attr for the given type. The
        provided builder should be used to construct the const-like operation.
        Null can be emitted by the differentiation pass.
      }],
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"createNullAttr",
      /*args=*/(ins)
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
      /*retTy=*/"bool",
      /*methodName=*/"isZero",
      /*args=*/(ins  "::mlir::Value":$val)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Check if the mlir Attribute with the given type is 0.
      }],
      /*retTy=*/"bool",
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

def ClonableTypeInterface : TypeInterface<"ClonableTypeInterface"> {
  let description = [{
    An MLIR type values of which can be cloned.
    Used for mutable types which needs to allocate and copy their content, specifically
    for things like checkpointing where the underlying memory needs to be preserved.
  }];
  let cppNamespace = "::mlir::enzyme";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Clone a value.

        The clone does not have to have the type of what it clones: an
        `!llvm.ptr` whose size hint annotates a memory space is cloned into
        that space, and so comes back as a pointer in a different address
        space. Buffers that hold clones have to be typed from one, not from
        the value it was made of.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"cloneValue",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$value)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Does the type implement batch allocation and subelement derivation.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"implementsBatchAllocation",
      /*args=*/(ins "::mlir::Value":$base, "::mlir::OpFoldResult":$numel),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return false;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Tries to allocate multiple values.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"batchAllocate",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$base, "::mlir::OpFoldResult":$numel),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("multiple values allocation not implemented");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Given the bulk allocation return a subview to the element located at
        index `index`.

        The type returned by `deriveSubElement` does not have to match self
        completely but should be usable in place of it.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"deriveSubElement",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Location":$loc,
                    "::mlir::Value":$base, "::mlir::Value":$multiel, "::mlir::Value":$index),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        llvm_unreachable("bulk allocation is not implemented for type");
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Copy the contents of `src` into the already-allocated `dst`.

        Allocates nothing, so one clone can be reused across several
        checkpoints instead of being reallocated each time. `dst` and `src`
        must have this type and must not alias.

        At least one of `dst`/`src` must be a value whose allocation extent
        this type can recover -- for pointer types, one that traces to an
        `llvm_ext.alloc` or carries an `llvm_ext.ptr_size_hint`. A handle
        freshly loaded out of a buffer of clones has no recoverable extent, so
        the other side has to be a direct clone.
      }],
      /*retTy=*/"void",
      /*methodName=*/"copyValue",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$dst,
                    "::mlir::Value":$src)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Free the clone or batch allocated value.
      }],
      /*retTy=*/"void",
      /*methodName=*/"freeClonedValue",
      /*args=*/(ins "::mlir::OpBuilder &":$builder, "::mlir::Value":$clone)
    >
  ];

  let extraSharedClassDeclaration = [{}];
}

#endif // ENZYME_MLIR_INTERFACES_AUTODIFFTYPEINTERFACE
