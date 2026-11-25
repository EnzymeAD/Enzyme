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

def XformKindAttr : EnumAttr<Impulse_Dialect, XformKind, "xform_kind">;

def XformAttr : Impulse_Attr<"Xform", "xform"> {
  let summary = "A bijection (transform) in the shared bijector library";
  let description = [{
    One bijector: `forward` maps base/unconstrained space to the observed
    /constrained space, `inverse` maps back, and the change of variables adds
    `logabsdetjac`. AFFINE reads its loc/scale from the sample site's operands
    (positional, after the RNG operand); it carries no static parameters. A
    COMPOSE carries a `components` array applied inner-to-outer (component 0 is
    applied first). The constraint bijectors SIGMOID, INTERVAL, GREATER_THAN,
    LESS_THAN, and STICK_BREAKING read their bound / event-size parameters from
    the site's SupportAttr; SOFTPLUS and SCALED_UNIT_LOWER_CHOLESKY are the VI
    guide transforms.
  }];
  let parameters = (ins
    "XformKind":$kind,
    OptionalParameter<"ArrayAttr">:$components
  );
  let assemblyFormat = "`<` $kind (`,` $components^)? `>`";
}

def DistBaseKind : I32EnumAttr<"DistBaseKind",
    "Primitive base distribution of a DistAttr",
    [
      I32EnumAttrCase<"STD_NORMAL", 0>,
      I32EnumAttrCase<"CUSTOM", 1>,
      I32EnumAttrCase<"STD_UNIFORM", 2>,
      I32EnumAttrCase<"EXPONENTIAL", 3>,
      I32EnumAttrCase<"GAMMA", 4>,
      I32EnumAttrCase<"STUDENT_T", 5>,
      I32EnumAttrCase<"BERNOULLI", 6>,
      I32EnumAttrCase<"POISSON", 7>,
      I32EnumAttrCase<"DISCRETE_UNIFORM", 8>,
      I32EnumAttrCase<"HALF_NORMAL", 9>,
      I32EnumAttrCase<"HALF_CAUCHY", 10>,
      I32EnumAttrCase<"DIRICHLET", 11>,
      I32EnumAttrCase<"MVN", 12>,
      I32EnumAttrCase<"GUMBEL", 13>,
      I32EnumAttrCase<"HALF_STUDENT_T", 14>,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::impulse";
}

def DistBaseKindAttr : EnumAttr<Impulse_Dialect, DistBaseKind, "dist_base_kind">;

def DistAttr : Impulse_Attr<"Dist", "dist"> {
  let summary = "Distribution as a primitive base plus a transform chain";
  let description = [{
    The distribution of a draw from `base` mapped through the bijections of
    `transforms`, applied in order. For example, `Normal(mu, sigma)` is
    `#impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>]>` and
    `LogNormal(mu, sigma)` is
    `#impulse.dist<STD_NORMAL, [#impulse.xform<AFFINE>, #impulse.xform<EXP>]>`.

    The parameters are the non-RNG inputs of the sample site. Only the
    parameterless bases STD_NORMAL and STD_UNIFORM take a non-empty
    `transforms`, whose bijections read the parameters (AFFINE reads loc and
    scale, the first two). The other bases read them in this order:
    EXPONENTIAL (rate), GAMMA (concentration, rate), STUDENT_T (df, loc,
    scale), BERNOULLI (logits), POISSON (rate), DISCRETE_UNIFORM (low, high),
    HALF_NORMAL (scale), HALF_CAUCHY (scale), DIRICHLET (concentration), MVN
    (mean, covariance), GUMBEL (loc, scale) and HALF_STUDENT_T (df, scale).
    The value of an MVN site has the type of its mean.

    The support of the distribution is the support of `base` mapped through
    each bijection. `event_dim` (default 0) is the rank of one event; it does
    not change the distribution.
  }];
  let parameters = (ins
    "DistBaseKind":$base,
    OptionalParameter<"ArrayAttr">:$transforms,
    DefaultValuedParameter<"int64_t", "0">:$event_dim
  );
  // Comma-led optional groups cannot parse `<base, event_dim = k>`.
  let assemblyFormat = "`<` $base `` custom<DistTail>($transforms, $event_dim) `>`";
}

def SupportAttr : Impulse_Attr<"Support", "support"> {
  let summary = "Support of a distribution";
  let description = [{
    The set of values on which a density is nonzero. INTERVAL reads
    `lower_bound` and `upper_bound`, GREATER_THAN reads `lower_bound`, and
    LESS_THAN reads `upper_bound`. For SIMPLEX, `event_size` is the length K
    of one simplex, and a value of m*K elements holds m simplices. The other
    kinds take no parameters.
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
    Parameters of Hamiltonian Monte Carlo with a fixed trajectory.
    - trajectory_length: the integration time of one transition.
    - adapt_step_size (default true): adapt the step size during warmup by
      dual averaging.
    - adapt_mass_matrix (default true): adapt the inverse mass matrix during
      warmup to the covariance of the warmup samples.
    - num_steps (default 0): when positive, each transition takes exactly
      `num_steps` leapfrog steps of the current step size and
      `trajectory_length` is ignored; when 0, it takes
      ceil(trajectory_length / step_size) steps of size
      trajectory_length / ceil(trajectory_length / step_size).
  }];
  let parameters = (ins
    "FloatAttr":$trajectory_length,
    DefaultValuedParameter<"bool", "true">:$adapt_step_size,
    DefaultValuedParameter<"bool", "true">:$adapt_mass_matrix,
    DefaultValuedParameter<"int64_t", "0">:$num_steps
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def NUTSConfigAttr : Impulse_Attr<"NUTSConfig", "nuts_config"> {
  let summary = "Configuration for NUTS inference";
  let description = [{
    Parameters of the No-U-Turn Sampler.
    - max_tree_depth (default 10): the maximum depth of the trajectory tree; a
      transition takes at most 2^max_tree_depth - 1 leapfrog steps.
    - max_delta_energy (default 1000): a leapfrog step whose energy exceeds
      the initial energy of the transition by more than this diverges.
    - adapt_step_size (default true): adapt the step size during warmup by
      dual averaging.
    - adapt_mass_matrix (default true): adapt the inverse mass matrix during
      warmup to the covariance of the warmup samples.
  }];
  let parameters = (ins
    DefaultValuedParameter<"int64_t", "10">:$max_tree_depth,
    OptionalParameter<"FloatAttr">:$max_delta_energy,
    DefaultValuedParameter<"bool", "true">:$adapt_step_size,
    DefaultValuedParameter<"bool", "true">:$adapt_mass_matrix
  );
  let assemblyFormat = "`<` struct(params) `>`";
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
    Stochastic variational inference: fits the family `guide_kind` over the
    unconstrained position by maximizing a Monte Carlo estimate of the ELBO
    with the update rule `optimizer_kind`.
    - num_particles (default 1): draws per ELBO estimate.
    - init_scale (default 0.1): the initial scale of the AUTO_NORMAL and
      AUTO_MULTIVARIATE_NORMAL guides.
    - adam_b1, adam_b2, adam_eps (defaults 0.9, 0.999, 1e-8): parameters of
      ADAM and CLIPPED_ADAM.
    - clip_norm (default 10): CLIPPED_ADAM clamps each gradient element to
      [-clip_norm, clip_norm].
    - momentum_mass (default 0.9): parameter of MOMENTUM, ADAGRAD and
      RMSPROP_MOMENTUM.
    - rmsprop_gamma, rmsprop_eps (defaults 0.9, 1e-8): parameters of RMSPROP
      and RMSPROP_MOMENTUM.
    - init_radius (default 2): without `init_position`, each element of the
      initial location is drawn uniformly from [-init_radius, init_radius).
    - stable_update (default false): skip an update whose ELBO estimate or
      gradient is not finite.
  }];
  let parameters = (ins
    EnumAttrParameter<GuideKindAttr>:$guide_kind,
    EnumAttrParameter<OptimizerKindAttr>:$optimizer_kind,
    DefaultValuedParameter<"int64_t", "1">:$num_particles,
    OptionalParameter<"FloatAttr">:$init_scale,
    OptionalParameter<"FloatAttr">:$adam_b1,
    OptionalParameter<"FloatAttr">:$adam_b2,
    OptionalParameter<"FloatAttr">:$adam_eps,
    OptionalParameter<"FloatAttr">:$clip_norm,
    OptionalParameter<"FloatAttr">:$init_radius,
    OptionalParameter<"FloatAttr">:$momentum_mass,
    OptionalParameter<"FloatAttr">:$rmsprop_gamma,
    OptionalParameter<"FloatAttr">:$rmsprop_eps,
    DefaultValuedParameter<"bool", "false">:$stable_update
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def MHConfigAttr : Impulse_Attr<"MHConfig", "mh_config"> {
  let summary = "Configuration for Metropolis-Hastings inference";
  let description = [{
    Marker config selecting the Metropolis-Hastings algorithm on `impulse.infer`
    / `impulse.infer_region`. MH carries no algorithm parameters of its own: the
    proposal selection lives in the shared `selection` attribute (the addresses
    to regenerate), the full trace layout in `all_addresses`, and the internal
    step count in `num_samples`.
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

def GibbsConfigAttr : Impulse_Attr<"GibbsConfig", "gibbs_config"> {
  let summary = "Configuration for enumerative Gibbs sampling";
  let description = [{
    Marker config selecting enumerative Gibbs on `impulse.infer` /
    `impulse.infer_region`: each selected site, in `selection` order, is
    redrawn from its exact full conditional, computed by scoring every value
    of its finite support (BERNOULLI: {0, 1}). Only the density terms that
    depend on the site are scored. `num_samples` is the number of sweeps.
    A `constraint` operand (with `constrained_addresses`) supplies the
    observed values of constrained sites, so a flip that makes a branch-local
    observation live scores it from the observation and writes it into the
    trace.
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

#endif // IMPULSE_ENUMS
