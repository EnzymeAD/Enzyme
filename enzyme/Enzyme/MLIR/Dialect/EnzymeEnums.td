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

def RngDistribution : I32EnumAttr<"RngDistribution",
    "Random number distribution type",
    [
      I32EnumAttrCase<"UNIFORM", 0>,
      I32EnumAttrCase<"NORMAL", 1>,
      I32EnumAttrCase<"MULTINORMAL", 2>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzyme";
}

def RngDistributionAttr : EnumAttr<Enzyme_Dialect, RngDistribution, "rng_distribution">;

def Transpose : I32EnumAttr<"Transpose",
    "Transpose mode for matrix operations",
    [
      I32EnumAttrCase<"NO_TRANSPOSE", 0>,
      I32EnumAttrCase<"TRANSPOSE", 1>,
      I32EnumAttrCase<"ADJOINT", 2>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzyme";
}

def TransposeAttr : EnumAttr<Enzyme_Dialect, Transpose, "transpose">;

def HMCConfigAttr : Enzyme_Attr<"HMCConfig", "hmc_config"> {
  let summary = "Configuration for HMC inference";
  let description = [{
    HMC-specific parameters for fixed-trajectory Hamiltonian Monte Carlo.
    - trajectory_length: Total trajectory length for leapfrog integration.
    - adapt_step_size (default: true): Whether to adapt step size during warmup using dual averaging.
    - adapt_mass_matrix (default: true): Whether to adapt the mass matrix during warmup using Welford covariance.
  }];
  let parameters = (ins
    "FloatAttr":$trajectory_length,
    DefaultValuedParameter<"bool", "true">:$adapt_step_size,
    DefaultValuedParameter<"bool", "true">:$adapt_mass_matrix
  );
  let assemblyFormat = "`<` `trajectory_length` `=` $trajectory_length (`,` `adapt_step_size` `=` $adapt_step_size^)? (`,` `adapt_mass_matrix` `=` $adapt_mass_matrix^)? `>`";
}

def NUTSConfigAttr : Enzyme_Attr<"NUTSConfig", "nuts_config"> {
  let summary = "Configuration for NUTS inference";
  let description = [{
    NUTS-specific parameters for No-U-Turn Sampler with adaptive path length.
    - max_tree_depth: Maximum tree depth, controls max leapfrog steps (2^max_tree_depth).
    - max_delta_energy: Threshold for trajectory divergence (rejected if energy change exceeds this).
                        Default: 1000.0
    - adapt_step_size (default: true): Whether to adapt step size during warmup using dual averaging.
    - adapt_mass_matrix (default: true): Whether to adapt the mass matrix during warmup using Welford covariance.
  }];
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "10">:$max_tree_depth,
    OptionalParameter<"FloatAttr">:$max_delta_energy,
    DefaultValuedParameter<"bool", "true">:$adapt_step_size,
    DefaultValuedParameter<"bool", "true">:$adapt_mass_matrix
  );
  let assemblyFormat = "`<` `max_tree_depth` `=` $max_tree_depth (`,` `max_delta_energy` `=` $max_delta_energy^)? (`,` `adapt_step_size` `=` $adapt_step_size^)? (`,` `adapt_mass_matrix` `=` $adapt_mass_matrix^)? `>`";
}

#endif // ENZYME_ENUMS
