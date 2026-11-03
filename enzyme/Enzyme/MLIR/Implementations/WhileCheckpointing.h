//===- WhileCheckpointing.h - Shared helpers for while-loop AD --*- C++ -*-===//
//
// Attribute-reading helpers shared by SCFAutoDiffOpInterfaceImpl.cpp
// (scf::WhileOp) and StablehloAutoDiffOpInterfaceImpl.cpp
// (stablehlo::WhileOp).  Both ops carry enzyme.online_checkpointing and
// enzyme.checkpoint_period in the same way; only the region structure differs.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYMEMLIR_WHILE_CHECKPOINTING_H_
#define ENZYMEMLIR_WHILE_CHECKPOINTING_H_

#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
namespace enzyme {

/// Return true when `op` has been annotated for Online_r2 checkpointing.
inline bool whileNeedsOnlineCheckpointing(Operation *op) {
  auto enabled = op->getAttrOfType<BoolAttr>("enzyme.enable_checkpointing");
  return enabled && enabled.getValue() &&
         op->hasAttr("enzyme.online_checkpointing");
}

/// Return the checkpoint budget from enzyme.checkpoint_period, or nullopt.
inline std::optional<int64_t> whileGetCheckpointBudget(Operation *op) {
  if (auto a = op->getAttrOfType<IntegerAttr>("enzyme.checkpoint_period"))
    return a.getInt();
  return std::nullopt;
}

} // namespace enzyme
} // namespace mlir

#endif // ENZYMEMLIR_WHILE_CHECKPOINTING_H_
