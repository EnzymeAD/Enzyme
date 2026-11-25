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

/// The model and configuration for lowering one VI inference op.
struct VIContext {
  FlatSymbolRefAttr fn;
  ArrayRef<Value> fnInputs;
  SmallVector<Type> fnResultTypes;
  Value originalTrace;
  ArrayAttr selection;
  ArrayAttr allAddresses;
  Value stepSize; // learning rate
  int64_t positionSize;
  int64_t numSamples; // optimizer steps
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
  // Per-element gradient clip of CLIPPED_ADAM.
  double clipNorm = 10.0;
  // Half-width of the uniform initial location used without `init_position`.
  double initRadius = 2.0;
  double momentumMass = 0.9;
  double rmspropGamma = 0.9;
  double rmspropEps = 1.0e-8;
  // Skip steps whose ELBO or gradient is not finite; the key still advances.
  bool stableUpdate = false;
  // Set when the op resumes from `state_in`.
  bool isResume = false;
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

/// Replaces the VI inference `op` with an SVI loop that starts from the op's
/// `state_in` when it has one. Fails with a diagnostic on guide kinds and
/// modes the lowering does not support.
template <typename OpT>
LogicalResult lowerVI(PatternRewriter &rewriter, OpT op, VIContext ctx,
                      bool debugDump = false);

} // namespace VI
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_VI_UTILS_H
