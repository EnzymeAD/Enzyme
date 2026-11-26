//===- ImpulseAttributes.td - Impulse dialect attributes ---*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IMPULSE_ATTRIBUTES
#define IMPULSE_ATTRIBUTES

include "ImpulseEnums.td"
include "Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class Impulse_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Impulse_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def RngDistributionAttr : EnumAttr<Impulse_Dialect, RngDistribution, "rng_distribution">;

def TransposeAttr : EnumAttr<Impulse_Dialect, Transpose, "transpose">;

def XformAttr : Impulse_Attr<"Xform", "xform"> {
  let summary = "A bijection of a distribution transform chain";
  let description = [{
    One bijection y = f(x) of the `transforms` of a `#impulse.dist`:
    IDENTITY x; EXP exp(x); AFFINE loc + scale * x, with loc and scale the
    first two parameters of the site and scale > 0; SIGMOID 1 / (1 + exp(-x));
    SOFTPLUS log(1 + exp(x)).
  }];
  let parameters = (ins "XformKind":$kind);
  let assemblyFormat = "`<` $kind `>`";
}

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
    HALF_NORMAL (scale), HALF_CAUCHY (scale), DIRICHLET (concentration),
    MULTIVARIATE_NORMAL (mean, covariance), GUMBEL (loc, scale) and
    HALF_STUDENT_T (df, scale). The value of a MULTIVARIATE_NORMAL site has
    the type of its mean.

    The support of the distribution has the kind of the support of `base`
    mapped through each bijection, and no bounds: EXP and SOFTPLUS give
    POSITIVE, SIGMOID gives UNIT_INTERVAL, AFFINE turns UNIT_INTERVAL into
    INTERVAL and keeps every other kind, and IDENTITY keeps it. The support of
    the discrete bases BERNOULLI, POISSON and DISCRETE_UNIFORM has the kind
    REAL.
  }];
  let parameters = (ins
    "DistBaseKind":$base,
    OptionalParameter<"ArrayAttr">:$transforms
  );
  let assemblyFormat = "`<` $base (`,` $transforms^)? `>`";
  let extraClassDeclaration = [{
    /// The support of `base` mapped through each transform, without bounds
    /// or an event size.
    SupportAttr getSupport() const;
    /// Whether a sample is a fixed function of one uniform or normal draw, so
    /// that its sampler needs no loop. The other bases are sampled by
    /// rejection, with a number of draws that depends on the values drawn.
    bool hasLoopFreeSampler() const;
    /// The type of a sample when the parameter types `paramTypes` fix it,
    /// else null.
    RankedTensorType getSampleType(TypeRange paramTypes) const;
    /// The number of parameters that `base` and `transforms` read.
    unsigned getNumParameters() const;
  }];
}

def SupportAttr : Impulse_Attr<"Support", "support"> {
  let summary = "Support of a distribution";
  let description = [{
    The set of values on which a density is nonzero. INTERVAL reads
    `lower_bound` and `upper_bound`, GREATER_THAN reads `lower_bound`, and
    LESS_THAN reads `upper_bound`; the bounds are written `lower = a` and
    `upper = b`. For SIMPLEX, `event_size` is the length K of one simplex, and
    a value of m*K elements holds m simplices. The other kinds take no
    parameters.

    Each kind is the image of a bijection from unconstrained values: REAL x;
    POSITIVE exp(x); UNIT_INTERVAL 1 / (1 + exp(-x)); INTERVAL lower_bound +
    (upper_bound - lower_bound) / (1 + exp(-x)); GREATER_THAN lower_bound +
    exp(x); LESS_THAN upper_bound - exp(x), each elementwise; SIMPLEX the
    stick-breaking map from m*(K-1) values to m simplices. The unconstrained
    coordinates of a value are its preimage.
  }];
  let parameters = (ins
    "SupportKind":$kind,
    OptionalParameter<"FloatAttr">:$lower_bound,
    OptionalParameter<"FloatAttr">:$upper_bound,
    OptionalParameter<"IntegerAttr">:$event_size
  );
  let extraClassDeclaration = [{
    /// The number of unconstrained values that `constrainedSize` values on
    /// this support map to: K - 1 per simplex of K for SIMPLEX, else
    /// `constrainedSize`.
    int64_t getUnconstrainedSize(int64_t constrainedSize) const;
  }];
  // Comma-led optional groups cannot parse `<LESS_THAN, upper = y>`.
  let assemblyFormat = [{
    `<` $kind `` custom<SupportParams>($lower_bound, $upper_bound, $event_size)
    `>`
  }];
}

def HMCConfigAttr : Impulse_Attr<"HMCConfig", "hmc_config"> {
  let summary = "Configuration for HMC inference";
  let description = [{
    Parameters of Hamiltonian Monte Carlo with a fixed trajectory.
    - trajectory_length: the integration time of one transition.
    - adapt_step_size (default true): adapt the step size during warmup by
      dual averaging toward a mean acceptance probability of 0.8.
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
      dual averaging toward a mean acceptance probability of 0.8.
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

def VIConfigAttr : Impulse_Attr<"VIConfig", "vi_config"> {
  let summary = "Configuration for variational inference";
  let description = [{
    Stochastic variational inference: fits the family `guide_kind` to the
    target by maximizing a Monte Carlo estimate of the ELBO with the update
    rule `optimizer_kind`.
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
    "GuideKind":$guide_kind,
    "OptimizerKind":$optimizer_kind,
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
    Selects Metropolis-Hastings: each of `num_samples` steps redraws the sites
    in `selection` from their distributions and accepts the move with the
    Metropolis-Hastings probability. The config has no parameters.
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

def GibbsConfigAttr : Impulse_Attr<"GibbsConfig", "gibbs_config"> {
  let summary = "Configuration for enumerative Gibbs sampling";
  let description = [{
    Selects enumerative Gibbs sampling: each of `num_samples` sweeps redraws
    the sites in `selection`, in that order, from their exact full
    conditionals, each computed by scoring both values of the site. Every
    site in `selection` is a scalar BERNOULLI site at the top level of the
    model. A site in `constrained_addresses` that runs in a branch of an
    `impulse.if` takes its value from `constraint` whenever it runs, and the
    trace holds that value. The config has no parameters.
  }];
  let parameters = (ins);
  let assemblyFormat = "";
}

def SymbolAttr : Impulse_Attr<"Symbol", "symbol"> {
  let summary = "Name of a sample site within its model";
  let description = [{
    An opaque integer that names a sample site within its model; the sites of
    one model must have distinct symbols.
  }];
  let parameters = (ins "uint64_t":$id);
  let assemblyFormat = "`<` $id `>`";
}

def AddressAttr : TypedArrayAttrBase<SymbolAttr, "Address as an array of symbols">;
def AddressArrayAttr : TypedArrayAttrBase<AddressAttr, "Array of addresses">;

#endif // IMPULSE_ATTRIBUTES
