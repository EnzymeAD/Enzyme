#include "Analysis/SampleDependenceAnalysis.h"
#include "Dialect/Ops.h"
#include "Passes/Passes.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_PRINTSAMPLEDEPENDENCEPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

struct PrintSampleDependencePass
    : public ::mlir::enzyme::impl::PrintSampleDependencePassBase<
          PrintSampleDependencePass> {

  void runOnOperation() override {
    raw_ostream &os = llvm::outs();

    getOperation()->walk([&](MCMCRegionOp regionOp) {
      os << "=== SampleDependenceAnalysis for mcmc_region ===\n";

      SampleDependenceAnalysis analysis(regionOp);

      os << "Sample regions: " << analysis.getSampleOps().size() << "\n";

      regionOp.getSampler().walk([&](Operation *op) {
        bool dependent = analysis.isSampleDependent(op);
        bool hoistable = analysis.canHoist(op);

        os << (dependent ? "[DEP] " : "[INV] ");
        os << (hoistable ? "[HOIST] " : "[KEEP]  ");
        os << op->getName();

        if (op->getNumResults() > 0) {
          os << " -> ";
          llvm::interleaveComma(op->getResultTypes(), os);
        }

        os << "\n";
      });

      os << "=== End SampleDependenceAnalysis ===\n\n";
    });
  }
};

} // namespace
