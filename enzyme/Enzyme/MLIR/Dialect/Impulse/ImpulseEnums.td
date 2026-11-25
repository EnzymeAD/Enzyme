//===- ImpulseEnums.td - Impulse dialect enums ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPULSE_ENUMS
#define IMPULSE_ENUMS

include "mlir/IR/EnumAttr.td"

def RngDistribution : I32EnumAttr<"RngDistribution",
    "Random number distribution type",
    [
      I32EnumAttrCase<"UNIFORM", 0>,
      I32EnumAttrCase<"NORMAL", 1>,
      I32EnumAttrCase<"MULTINORMAL", 2>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def Transpose : I32EnumAttr<"Transpose",
    "Transpose mode for matrix operations",
    [
      I32EnumAttrCase<"NO_TRANSPOSE", 0>,
      I32EnumAttrCase<"TRANSPOSE", 1>,
      I32EnumAttrCase<"ADJOINT", 2>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def SupportKind : I32EnumAttr<"SupportKind",
    "Domain where distribution density is non-zero",
    [
      I32EnumAttrCase<"REAL", 0>,
      I32EnumAttrCase<"POSITIVE", 1>,
      I32EnumAttrCase<"UNIT_INTERVAL", 2>,
      I32EnumAttrCase<"INTERVAL", 3>,
      I32EnumAttrCase<"GREATER_THAN", 4>,
      I32EnumAttrCase<"LESS_THAN", 5>,
      I32EnumAttrCase<"SIMPLEX", 6>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def XformKind : I32EnumAttr<"XformKind",
    "Bijector applied to a base distribution or constraint",
    [
      I32EnumAttrCase<"IDENTITY", 0>,
      I32EnumAttrCase<"EXP", 1>,
      I32EnumAttrCase<"AFFINE", 2>,
      I32EnumAttrCase<"COMPOSE", 3>,
      I32EnumAttrCase<"SIGMOID", 4>,
      I32EnumAttrCase<"INTERVAL", 5>,
      I32EnumAttrCase<"GREATER_THAN", 6>,
      I32EnumAttrCase<"LESS_THAN", 7>,
      I32EnumAttrCase<"STICK_BREAKING", 8>,
      I32EnumAttrCase<"SOFTPLUS", 9>,
      I32EnumAttrCase<"SCALED_UNIT_LOWER_CHOLESKY", 10>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def DistBaseKind : I32EnumAttr<"DistBaseKind",
    "Primitive base distribution of a DistAttr",
    [
      I32EnumAttrCase<"STD_NORMAL", 0>,
      I32EnumAttrCase<"STD_UNIFORM", 1>,
      I32EnumAttrCase<"EXPONENTIAL", 2>,
      I32EnumAttrCase<"GAMMA", 3>,
      I32EnumAttrCase<"STUDENT_T", 4>,
      I32EnumAttrCase<"BERNOULLI", 5>,
      I32EnumAttrCase<"POISSON", 6>,
      I32EnumAttrCase<"DISCRETE_UNIFORM", 7>,
      I32EnumAttrCase<"HALF_NORMAL", 8>,
      I32EnumAttrCase<"HALF_CAUCHY", 9>,
      I32EnumAttrCase<"DIRICHLET", 10>,
      I32EnumAttrCase<"MVN", 11>,
      I32EnumAttrCase<"GUMBEL", 12>,
      I32EnumAttrCase<"HALF_STUDENT_T", 13>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def GuideKind : I32EnumAttr<"GuideKind",
    "Variational family used by VI",
    [
      I32EnumAttrCase<"AUTO_NORMAL", 0>,
      I32EnumAttrCase<"AUTO_MULTIVARIATE_NORMAL", 1>,
      I32EnumAttrCase<"AUTO_DELTA", 2>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def OptimizerKind : I32EnumAttr<"OptimizerKind",
    "Optimizer used for variational parameter updates",
    [
      I32EnumAttrCase<"ADAM", 0>,
      I32EnumAttrCase<"SGD", 1>,
      I32EnumAttrCase<"CLIPPED_ADAM", 2>,
      I32EnumAttrCase<"MOMENTUM", 3>,
      I32EnumAttrCase<"RMSPROP", 4>,
      I32EnumAttrCase<"ADAGRAD", 5>,
      I32EnumAttrCase<"RMSPROP_MOMENTUM", 6>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

#endif // IMPULSE_ENUMS
