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

/// `ctx` is by value: the warm-restart slots are filled from `state_in`.
template <typename OpT>
LogicalResult lowerVI(PatternRewriter &rewriter, OpT op, VIContext ctx,
                      bool debugDump = false);

} // namespace VI
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_VI_UTILS_H
