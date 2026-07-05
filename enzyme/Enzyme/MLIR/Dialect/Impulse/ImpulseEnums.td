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
include "Dialect.td"

class Impulse_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Impulse_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

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

def RngDistributionAttr : EnumAttr<Impulse_Dialect, RngDistribution, "rng_distribution">;

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

def TransposeAttr : EnumAttr<Impulse_Dialect, Transpose, "transpose">;

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

def SupportKindAttr : EnumAttr<Impulse_Dialect, SupportKind, "support_kind">;

def SupportAttr : Impulse_Attr<"Support", "support"> {
  let summary = "Distribution support specification for constraint transforms";
  let description = [{
    Distribution support specification for constraint transforms.

    `event_size` is the constrained length K of one simplex block for the
    SIMPLEX support (the stick-breaking bijector maps K constrained values to
    K-1 unconstrained ones); a slice of length m*K is treated as m independent
    K-blocks. It is unused (null) for the dimension-preserving supports.
  }];
  let parameters = (ins
    "SupportKind":$kind,
    OptionalParameter<"FloatAttr">:$lower_bound,
    OptionalParameter<"FloatAttr">:$upper_bound,
    OptionalParameter<"IntegerAttr">:$event_size
  );
  let assemblyFormat = "`<` $kind (`lower` `=` $lower_bound^)? (`upper` `=` $upper_bound^)? (`event_size` `=` $event_size^)? `>`";
}

def HMCConfigAttr : Impulse_Attr<"HMCConfig", "hmc_config"> {
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

def NUTSConfigAttr : Impulse_Attr<"NUTSConfig", "nuts_config"> {
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
  let assemblyFormat = "`<` (`max_tree_depth` `=` $max_tree_depth^)? (`,` `max_delta_energy` `=` $max_delta_energy^)? (`,` `adapt_step_size` `=` $adapt_step_size^)? (`,` `adapt_mass_matrix` `=` $adapt_mass_matrix^)? `>`";
}

def GuideKind : I32EnumAttr<"GuideKind",
    "Variational family used by VI",
    [
      I32EnumAttrCase<"AUTO_NORMAL", 0>,
      I32EnumAttrCase<"AUTO_MULTIVARIATE_NORMAL", 1>,
      I32EnumAttrCase<"CUSTOM", 2>,
      I32EnumAttrCase<"AUTO_DELTA", 3>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def GuideKindAttr : EnumAttr<Impulse_Dialect, GuideKind, "guide_kind">;

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

def OptimizerKindAttr : EnumAttr<Impulse_Dialect, OptimizerKind, "optimizer_kind">;

def VIConfigAttr : Impulse_Attr<"VIConfig", "vi_config"> {
  let summary = "Configuration for variational inference";
  let description = [{
    VI-specific parameters for stochastic variational inference.
    Matches NumPyro's `SVI(model, guide, optim, loss=Trace_ELBO(num_particles=K))`
    semantics with pointwise-identical RNG splitting.

    - guide_kind: Variational family (AutoNormal, AutoMultivariateNormal, Custom).
    - optimizer_kind: Parameter update rule (Adam, SGD).
    - num_particles (default: 1): Number of Monte Carlo particles used to
      estimate the ELBO at each update step. When K > 1, one `split(key, K)`
      is emitted and each particle receives an independent key.
    - init_scale: Initial scale for auto guides (default: 0.1, matches NumPyro).
    - adam_b1, adam_b2, adam_eps: Adam hyperparameters. Defaults match
      `jax.example_libraries.optimizers.adam`: b1=0.9, b2=0.999, eps=1e-8.
      Only meaningful when optimizer_kind = ADAM.
  }];
  let parameters = (ins
    "GuideKindAttr":$guide_kind,
    "OptimizerKindAttr":$optimizer_kind,
    DefaultValuedParameter<"int64_t", "1">:$num_particles,
    OptionalParameter<"FloatAttr">:$init_scale,
    OptionalParameter<"FloatAttr">:$adam_b1,
    OptionalParameter<"FloatAttr">:$adam_b2,
    OptionalParameter<"FloatAttr">:$adam_eps,
    OptionalParameter<"FloatAttr">:$clip_norm,
    // Used only when no `init_position` operand is given.
    OptionalParameter<"FloatAttr">:$init_radius,
    OptionalParameter<"FloatAttr">:$momentum_mass,
    OptionalParameter<"FloatAttr">:$rmsprop_gamma,
    OptionalParameter<"FloatAttr">:$rmsprop_eps,
    // Skips non-finite updates, as `numpyro.SVI.stable_update`.
    DefaultValuedParameter<"bool", "false">:$stable_update
  );
  let assemblyFormat = "`<` `guide_kind` `=` $guide_kind `,` `optimizer_kind` `=` $optimizer_kind (`,` `num_particles` `=` $num_particles^)? (`,` `init_scale` `=` $init_scale^)? (`,` `adam_b1` `=` $adam_b1^)? (`,` `adam_b2` `=` $adam_b2^)? (`,` `adam_eps` `=` $adam_eps^)? (`,` `clip_norm` `=` $clip_norm^)? (`,` `init_radius` `=` $init_radius^)? (`,` `momentum_mass` `=` $momentum_mass^)? (`,` `rmsprop_gamma` `=` $rmsprop_gamma^)? (`,` `rmsprop_eps` `=` $rmsprop_eps^)? (`,` `stable_update` `=` $stable_update^)? `>`";
}

#endif // IMPULSE_ENUMS
