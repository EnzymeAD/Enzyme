//===- VIUtils.h - Variational Inference lowering utilities ----* C++ -*-===//

#ifndef ENZYME_MLIR_INTERFACES_VI_UTILS_H
#define ENZYME_MLIR_INTERFACES_VI_UTILS_H

#include "Dialect/Impulse/Impulse.h"
#include "HMCUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace impulse {
namespace VI {

struct VIContext {
  FlatSymbolRefAttr fn;
  ArrayRef<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr selection;
  ArrayAttr allAddresses;
  Value stepSize; // learning rate
  int64_t positionSize;
  int64_t numSamples; // SVI.update iterations
  SmallVector<SupportInfo> supports;
  FlatSymbolRefAttr logpdfFn;
  DictionaryAttr autodiffAttrs;

  GuideKind guideKind;
  OptimizerKind optimizerKind;
  int64_t numParticles;
  double initScale;
  double adamB1;
  double adamB2;
  double adamEps;
  // Per-element gradient clip, as numpyro.optim.ClippedAdam.
  double clipNorm = 10.0;
  // init_to_uniform radius, used without `initial_position`.
  double initRadius = 2.0;
  double momentumMass = 0.9;
  double rmspropGamma = 0.9;
  double rmspropEps = 1.0e-8;
  // numpyro SVI.stable_update: skip non-finite steps; the RNG still advances.
  bool stableUpdate = false;
  // Warm-restart optimizer state from `state_in`; null means a cold start.
  bool isResume = false;
  Value initMLoc;
  Value initVLoc;
  Value initMScale;
  Value initVScale;
  Value initStepCount;
  bool debugDump = false;

  // When set, the joint density is cloned from here, not `@fn.generate`.
  // Block argument k binds to the (offset, size) slice logpdfArgSlices[k] of z.
  Region *logpdfRegion = nullptr;
  SmallVector<std::pair<int64_t, int64_t>> logpdfArgSlices;
  bool hasLogpdfRegion() const { return logpdfRegion != nullptr; }

  VIContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
            ArrayRef<Type> fnResultTypes, Value originalTrace,
            ArrayAttr selection, ArrayAttr allAddresses, Value stepSize,
            int64_t positionSize, int64_t numSamples,
            ArrayRef<SupportInfo> supports, GuideKind guideKind,
            OptimizerKind optimizerKind, int64_t numParticles, double initScale,
            double adamB1, double adamB2, double adamEps)
      : fn(fn), fnInputs(fnInputs),
        fnResultTypes(fnResultTypes.begin(), fnResultTypes.end()),
        originalTrace(originalTrace), selection(selection),
        allAddresses(allAddresses), stepSize(stepSize),
        positionSize(positionSize), numSamples(numSamples),
        supports(supports.begin(), supports.end()), guideKind(guideKind),
        optimizerKind(optimizerKind), numParticles(numParticles),
        initScale(initScale), adamB1(adamB1), adamB2(adamB2), adamEps(adamEps) {
  }

  bool hasCustomLogpdf() const { return logpdfFn != nullptr; }

  Type getElementType() const {
    return cast<RankedTensorType>(stepSize.getType()).getElementType();
  }

  RankedTensorType getPositionType() const {
    return RankedTensorType::get({1, positionSize}, getElementType());
  }

  RankedTensorType getScalarType() const {
    return RankedTensorType::get({}, getElementType());
  }

  int64_t getFullTraceSize() const {
    auto traceType = cast<RankedTensorType>(originalTrace.getType());
    return traceType.getShape()[1];
  }
};

/// One particle of NumPyro's Trace_ELBO with an AutoNormal guide.
struct SingleParticleELBOResult {
  Value elbo;
  Value gradLoc;
  Value gradScale;
  Value rng;
};

SingleParticleELBOResult emitSingleParticleELBO(OpBuilder &builder,
                                                Location loc, Value locParam,
                                                Value scaleParam, Value pkey,
                                                const VIContext &ctx);

/// NumPyro Trace_ELBO: K == 1 uses `rngStep` directly; K > 1 splits it K ways.
struct MultiParticleELBOResult {
  Value elbo;
  Value gradLoc;
  Value gradScale;
};

MultiParticleELBOResult emitELBOStep(OpBuilder &builder, Location loc,
                                     Value locParam, Value scaleParam,
                                     Value rngStep, const VIContext &ctx);

/// `gradElbo` is d(elbo)/dp; the minimized loss is -elbo.
Value applySGDUpdate(OpBuilder &builder, Location loc, Value param,
                     Value gradElbo, Value lr);

/// jax.example_libraries.optimizers.adam; `stepIdx` counts from 0.
struct AdamStepResult {
  Value paramNew;
  Value mNew;
  Value vNew;
};

AdamStepResult applyAdamUpdate(OpBuilder &builder, Location loc, Value param,
                               Value gradElbo, Value m, Value v, Value stepIdx,
                               Value lr, double b1, double b2, double eps,
                               // > 0 enables ClippedAdam's per-element clip.
                               double clipNorm = 0.0);

struct GenericOptimStepResult {
  Value paramNew;
  Value slot1New;
  Value slot2New;
};

/// jax.example_libraries.optimizers.momentum.
GenericOptimStepResult applyMomentumUpdate(OpBuilder &builder, Location loc,
                                           Value param, Value gradElbo,
                                           Value velocity, Value lr,
                                           double mass);

/// jax.example_libraries.optimizers.rmsprop.
GenericOptimStepResult applyRMSPropUpdate(OpBuilder &builder, Location loc,
                                          Value param, Value gradElbo,
                                          Value avgSqGrad, Value lr,
                                          double gamma, double eps);

/// jax.example_libraries.optimizers.adagrad; slot1 = g_sq, slot2 = m.
GenericOptimStepResult applyAdagradUpdate(OpBuilder &builder, Location loc,
                                          Value param, Value gradElbo,
                                          Value gSq, Value m, Value lr,
                                          double momentum);

/// jax.example_libraries.optimizers.rmsprop_momentum; slot1 = avg, slot2 = mom.
GenericOptimStepResult
applyRMSPropMomentumUpdate(OpBuilder &builder, Location loc, Value param,
                           Value gradElbo, Value avgSqGrad, Value mom, Value lr,
                           double gamma, double eps, double momentumMass);

/// `ctx.numSamples` iterations of NumPyro's SVI.update.
struct SVILoopResult {
  Value traceBuf;
  Value elboBuf;
  Value rngFinal;
  Value locFinal; // unconstrained
  // Unconstrained; flat scale_tril for AutoMVN, null for AutoDelta.
  Value scaleFinal;
  Value elboFinal;
  // Final optimizer state; the scale moments are null for AutoDelta.
  Value mLocFinal;
  Value vLocFinal;
  Value mScaleFinal;
  Value vScaleFinal;
  Value stepCountFinal;
};

SVILoopResult emitSVILoop(OpBuilder &builder, Location loc, Value loc0,
                          Value scale0, Value rng0, const VIContext &ctx);

/// AutoMultivariateNormal variant; `xScaleTril` is unconstrained and flat.
struct SingleParticleMVNResult {
  Value elbo;
  Value gradLoc;
  Value gradScaleTril;
  Value rng;
};

SingleParticleMVNResult emitSingleParticleMVNELBO(OpBuilder &builder,
                                                  Location loc, Value locParam,
                                                  Value xScaleTril, Value pkey,
                                                  const VIContext &ctx);

/// `xScaleTril0` is the unconstrained flat vector of length d*(d+1)/2.
SVILoopResult emitSVILoopMVN(OpBuilder &builder, Location loc, Value loc0,
                             Value xScaleTril0, Value rng0,
                             const VIContext &ctx);

/// log q = 0, but guide_seed is still split per site as NumPyro does.
struct SingleParticleAutoDeltaResult {
  Value elbo;
  Value gradLoc;
  Value rng;
};

SingleParticleAutoDeltaResult
emitSingleParticleAutoDelta(OpBuilder &builder, Location loc, Value locParam,
                            Value pkey, const VIContext &ctx);

SVILoopResult emitSVILoopAutoDelta(OpBuilder &builder, Location loc, Value loc0,
                                   Value rng0, const VIContext &ctx);

/// `ctx` is by value: the warm-restart slots are filled from `state_in`.
template <typename OpT>
LogicalResult lowerVI(PatternRewriter &rewriter, OpT op, VIContext ctx,
                      bool debugDump = false);

} // namespace VI
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_VI_UTILS_H
