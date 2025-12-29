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

def HMCConfigAttr : Enzyme_Attr<"HMCConfig", "hmc_config"> {
  let summary = "Configuration for HMC inference";
  let description = [{
    HMC-specific parameters for fixed-trajectory Hamiltonian Monte Carlo.
    - num_steps: Number of leapfrog integration steps per sample
  }];
  let parameters = (ins
    "int64_t":$num_steps
  );
  let assemblyFormat = "`<` `num_steps` `=` $num_steps `>`";
}

def NUTSConfigAttr : Enzyme_Attr<"NUTSConfig", "nuts_config"> {
  let summary = "Configuration for NUTS inference";
  let description = [{
    NUTS-specific parameters for No-U-Turn Sampler with adaptive path length.
    - max_tree_depth: Maximum tree depth, controls max leapfrog steps (2^max_tree_depth).
    - max_delta_energy: Threshold for trajectory divergence (rejected if energy change exceeds this).
                        Default: 1000.0
  }];
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "10">:$max_tree_depth,
    OptionalParameter<"FloatAttr">:$max_delta_energy
  );
  let assemblyFormat = "`<` `max_tree_depth` `=` $max_tree_depth (`,` `max_delta_energy` `=` $max_delta_energy^)? `>`";
}

#endif // ENZYME_ENUMS
