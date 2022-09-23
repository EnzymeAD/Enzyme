#ifndef ENZYME_PASSES
#define ENZYME_PASSES

include "mlir/Pass/PassBase.td"

def DifferentiatePass : Pass<"enzyme"> {
  let summary = "Differentiate Passes";
  let constructor = "mlir::enzyme::createDifferentiatePass()";
}

#endif // ENZYME_PASSES
