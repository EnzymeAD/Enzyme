#include "Analysis/SampleDependenceAnalysis.h"
#include "Dialect/Ops.h"
#include "Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_SAMPLEINVARIANTCODEMOTIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct SampleInvariantCodeMotionPass
    : public ::mlir::enzyme::impl::SampleInvariantCodeMotionPassBase<
          SampleInvariantCodeMotionPass> {

  void runOnOperation() override {
    SmallVector<MCMCRegionOp> regions;
    getOperation()->walk([&](MCMCRegionOp op) { regions.push_back(op); });

    for (MCMCRegionOp regionOp : regions) {
      hoistSampleInvariantOps(regionOp);
    }
  }
};

} // namespace
