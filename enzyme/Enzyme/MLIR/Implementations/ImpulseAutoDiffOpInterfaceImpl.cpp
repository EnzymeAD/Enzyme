#include "Implementations/CoreDialectsAutoDiffImplementations.h"

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::impulse;

namespace {
#include "Implementations/ImpulseDerivatives.inc"
} // namespace

void mlir::enzyme::registerImpulseDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, impulse::ImpulseDialect *) {
    registerInterfaces(context);
  });
}
