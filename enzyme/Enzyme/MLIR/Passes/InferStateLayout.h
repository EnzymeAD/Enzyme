//===- InferStateLayout.h - Canonical impulse.infer state schema ---------===//

#ifndef ENZYME_MLIR_PASSES_INFER_STATE_LAYOUT_H
#define ENZYME_MLIR_PASSES_INFER_STATE_LAYOUT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir {
namespace impulse {

struct StateSlot {
  llvm::StringRef name;
  mlir::RankedTensorType type;
};

/// Sizes and types are per chain; numChains > 1 prepends a chain dimension.
inline llvm::SmallVector<StateSlot>
buildStateSchema(bool adaptive, int64_t positionSize, mlir::Type elemType,
                 mlir::RankedTensorType invMassType, int64_t numChains = 1,
                 mlir::RankedTensorType rngType = {}) {
  auto *ctx = elemType.getContext();
  auto i64Scalar =
      mlir::RankedTensorType::get({}, mlir::IntegerType::get(ctx, 64));
  auto f64 = mlir::Float64Type::get(ctx);
  auto positionType = mlir::RankedTensorType::get({1, positionSize}, elemType);
  auto scalarType = mlir::RankedTensorType::get({}, elemType);

  bool diagonal =
      !invMassType || invMassType.getRank() == 1 ||
      (invMassType.getRank() == 2 && invMassType.getShape()[0] == 1);
  mlir::RankedTensorType invType = invMassType ? invMassType : positionType;
  auto welfordVec = mlir::RankedTensorType::get({positionSize}, f64);
  mlir::RankedTensorType welfordM2 =
      diagonal ? welfordVec
               : mlir::RankedTensorType::get({positionSize, positionSize}, f64);

  llvm::SmallVector<StateSlot> slots = {
      {"position", positionType},       {"gradient", positionType},
      {"potential_energy", scalarType}, {"step_size", scalarType},
      {"inverse_mass_matrix", invType},
  };
  if (adaptive) {
    slots.append({
        {"da_log_step_size", scalarType},
        {"da_log_step_size_avg", scalarType},
        {"da_gradient_avg", scalarType},
        {"da_step_count", i64Scalar},
        {"da_prox_center", scalarType},
        {"welford_mean", welfordVec},
        {"welford_m2", welfordM2},
        {"welford_n", i64Scalar},
        {"window_idx", i64Scalar},
        {"warmup_offset", i64Scalar},
    });
  }
  if (numChains > 1) {
    for (StateSlot &s : slots) {
      llvm::SmallVector<int64_t> shape;
      shape.push_back(numChains);
      shape.append(s.type.getShape().begin(), s.type.getShape().end());
      s.type = mlir::RankedTensorType::get(shape, s.type.getElementType());
    }
    auto ui64 = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    mlir::RankedTensorType perChainRng =
        rngType ? rngType : mlir::RankedTensorType::get({2}, ui64);
    llvm::SmallVector<int64_t> rngShape;
    rngShape.push_back(numChains);
    rngShape.append(perChainRng.getShape().begin(),
                    perChainRng.getShape().end());
    slots.push_back({"rng", mlir::RankedTensorType::get(
                                rngShape, perChainRng.getElementType())});
  }
  return slots;
}

enum class VIGuide { None, AutoNormal, AutoDelta, AutoMVN };

inline llvm::SmallVector<StateSlot>
buildVIStateSchema(VIGuide guide, int64_t positionSize, mlir::Type elemType) {
  auto *ctx = elemType.getContext();
  auto i64Scalar =
      mlir::RankedTensorType::get({}, mlir::IntegerType::get(ctx, 64));
  auto vecType = mlir::RankedTensorType::get({1, positionSize}, elemType);
  int64_t flat = positionSize * (positionSize + 1) / 2;
  auto flatType = mlir::RankedTensorType::get({flat}, elemType);

  llvm::SmallVector<StateSlot> slots;
  switch (guide) {
  case VIGuide::AutoNormal:
    slots = {
        {"loc", vecType},
        {"scale_unconstrained", vecType},
        {"opt_m_loc", vecType},
        {"opt_v_loc", vecType},
        {"opt_m_scale", vecType},
        {"opt_v_scale", vecType},
        {"opt_step_count", i64Scalar},
    };
    break;
  case VIGuide::AutoDelta:
    slots = {
        {"loc", vecType},
        {"opt_m_loc", vecType},
        {"opt_v_loc", vecType},
        {"opt_step_count", i64Scalar},
    };
    break;
  case VIGuide::AutoMVN:
    slots = {
        {"loc", vecType},
        {"scale_tril_unconstrained", flatType},
        {"opt_m_loc", vecType},
        {"opt_v_loc", vecType},
        {"opt_m_scale_tril", flatType},
        {"opt_v_scale_tril", flatType},
        {"opt_step_count", i64Scalar},
    };
    break;
  case VIGuide::None:
    break;
  }
  return slots;
}

/// Only `weight`: trace and rng travel as the op's universal results.
inline llvm::SmallVector<StateSlot> buildMHStateSchema(mlir::Type elemType) {
  auto scalarType = mlir::RankedTensorType::get({}, elemType);
  return {{"weight", scalarType}};
}

inline mlir::ArrayAttr schemaNamesAttr(mlir::Builder &b,
                                       llvm::ArrayRef<StateSlot> slots) {
  llvm::SmallVector<mlir::Attribute> names;
  names.reserve(slots.size());
  for (const StateSlot &s : slots)
    names.push_back(b.getStringAttr(s.name));
  return b.getArrayAttr(names);
}

inline std::optional<unsigned> slotIndex(mlir::ArrayAttr schema,
                                         llvm::StringRef name) {
  if (!schema)
    return std::nullopt;
  for (auto [i, attr] : llvm::enumerate(schema))
    if (auto s = mlir::dyn_cast<mlir::StringAttr>(attr))
      if (s.getValue() == name)
        return static_cast<unsigned>(i);
  return std::nullopt;
}

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_PASSES_INFER_STATE_LAYOUT_H
