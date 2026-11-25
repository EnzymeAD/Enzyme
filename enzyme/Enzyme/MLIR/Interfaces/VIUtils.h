//===- VIUtils.h - Variational inference lowering ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

/// The model and configuration for lowering one VI inference op. `stepSize`
/// is the learning rate.
struct VIContext : public ModelContext {
  int64_t numSamples; // optimizer steps

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

  // Block argument k of `logpdfRegion` binds to the (offset, size) slice
  // logpdfArgSlices[k] of the position.
  SmallVector<std::pair<int64_t, int64_t>> logpdfArgSlices;

  VIContext(FlatSymbolRefAttr fn, ArrayRef<Value> fnInputs,
            ArrayRef<Type> fnResultTypes, Value originalTrace,
            ArrayAttr allAddresses, Value stepSize, int64_t positionSize,
            int64_t numSamples, ArrayRef<SupportInfo> supports,
            GuideKind guideKind, OptimizerKind optimizerKind,
            int64_t numParticles, double initScale, double adamB1,
            double adamB2, double adamEps)
      : ModelContext{fn,
                     fnInputs,
                     SmallVector<Type>(fnResultTypes),
                     originalTrace,
                     allAddresses,
                     stepSize,
                     positionSize,
                     SmallVector<SupportInfo>(supports),
                     /*logpdfFn=*/{},
                     /*autodiffAttrs=*/{}},
        numSamples(numSamples), guideKind(guideKind),
        optimizerKind(optimizerKind), numParticles(numParticles),
        initScale(initScale), adamB1(adamB1), adamB2(adamB2), adamEps(adamEps) {
  }
};

/// Replaces the VI inference `op` with an SVI loop that starts from the op's
/// `state_in` when it has one. Fails with a diagnostic on guide kinds and
/// modes the lowering does not support.
template <typename OpT>
LogicalResult lowerVI(PatternRewriter &rewriter, OpT op, VIContext ctx);

} // namespace VI
} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_INTERFACES_VI_UTILS_H
