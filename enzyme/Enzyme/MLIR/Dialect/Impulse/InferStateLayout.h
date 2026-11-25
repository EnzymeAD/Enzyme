//===- InferStateLayout.h - Canonical impulse.infer state schema ---------===//

#ifndef ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H
#define ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

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

/// Names of the HMC/NUTS `state_schema` slots.
namespace state_slot {
inline constexpr llvm::StringLiteral kPosition = "position";
inline constexpr llvm::StringLiteral kGradient = "gradient";
inline constexpr llvm::StringLiteral kPotentialEnergy = "potential_energy";
inline constexpr llvm::StringLiteral kStepSize = "step_size";
inline constexpr llvm::StringLiteral kInverseMassMatrix = "inverse_mass_matrix";
inline constexpr llvm::StringLiteral kDaLogStepSize = "da_log_step_size";
inline constexpr llvm::StringLiteral kDaLogStepSizeAvg = "da_log_step_size_avg";
inline constexpr llvm::StringLiteral kDaGradientAvg = "da_gradient_avg";
inline constexpr llvm::StringLiteral kDaStepCount = "da_step_count";
inline constexpr llvm::StringLiteral kDaProxCenter = "da_prox_center";
inline constexpr llvm::StringLiteral kWelfordMean = "welford_mean";
inline constexpr llvm::StringLiteral kWelfordM2 = "welford_m2";
inline constexpr llvm::StringLiteral kWelfordN = "welford_n";
inline constexpr llvm::StringLiteral kWindowIdx = "window_idx";
inline constexpr llvm::StringLiteral kWarmupOffset = "warmup_offset";
inline constexpr llvm::StringLiteral kRng = "rng";
} // namespace state_slot

/// Whether an inverse mass matrix of type `type` is diagonal: `tensor<N>` and
/// `tensor<1xN>` hold the diagonal, any other shape a dense matrix.
inline bool isDiagonalInverseMass(mlir::RankedTensorType type) {
  return type.getRank() == 1 ||
         (type.getRank() == 2 && type.getShape()[0] == 1);
}

/// Sizes and types are per chain; numChains > 1 prepends a chain dimension.
inline llvm::SmallVector<StateSlot>
buildStateSchema(bool adaptive, int64_t positionSize, mlir::Type elemType,
                 mlir::RankedTensorType invMassType, int64_t numChains = 1,
                 mlir::RankedTensorType rngType = {}) {
  mlir::MLIRContext *ctx = elemType.getContext();
  auto i64TensorType =
      mlir::RankedTensorType::get({}, mlir::IntegerType::get(ctx, 64));
  auto f64 = mlir::Float64Type::get(ctx);
  auto positionType = mlir::RankedTensorType::get({1, positionSize}, elemType);
  auto scalarType = mlir::RankedTensorType::get({}, elemType);

  bool diagonal = !invMassType || isDiagonalInverseMass(invMassType);
  mlir::RankedTensorType invType = invMassType ? invMassType : positionType;
  auto welfordVec = mlir::RankedTensorType::get({positionSize}, f64);
  mlir::RankedTensorType welfordM2 =
      diagonal ? welfordVec
               : mlir::RankedTensorType::get({positionSize, positionSize}, f64);

  llvm::SmallVector<StateSlot> slots = {
      {state_slot::kPosition, positionType},
      {state_slot::kGradient, positionType},
      {state_slot::kPotentialEnergy, scalarType},
      {state_slot::kStepSize, scalarType},
      {state_slot::kInverseMassMatrix, invType},
  };
  if (adaptive) {
    slots.append({
        {state_slot::kDaLogStepSize, scalarType},
        {state_slot::kDaLogStepSizeAvg, scalarType},
        {state_slot::kDaGradientAvg, scalarType},
        {state_slot::kDaStepCount, i64TensorType},
        {state_slot::kDaProxCenter, scalarType},
        {state_slot::kWelfordMean, welfordVec},
        {state_slot::kWelfordM2, welfordM2},
        {state_slot::kWelfordN, i64TensorType},
        {state_slot::kWindowIdx, i64TensorType},
        {state_slot::kWarmupOffset, i64TensorType},
    });
  }
  if (numChains > 1) {
    for (StateSlot &slot : slots) {
      llvm::SmallVector<int64_t> shape;
      shape.push_back(numChains);
      shape.append(slot.type.getShape().begin(), slot.type.getShape().end());
      slot.type =
          mlir::RankedTensorType::get(shape, slot.type.getElementType());
    }
    auto ui64 = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Unsigned);
    mlir::RankedTensorType perChainRng =
        rngType ? rngType : mlir::RankedTensorType::get({2}, ui64);
    llvm::SmallVector<int64_t> rngShape;
    rngShape.push_back(numChains);
    rngShape.append(perChainRng.getShape().begin(),
                    perChainRng.getShape().end());
    slots.push_back(
        {state_slot::kRng,
         mlir::RankedTensorType::get(rngShape, perChainRng.getElementType())});
  }
  return slots;
}

enum class VIGuide { None, AutoNormal, AutoDelta, AutoMVN };

inline llvm::SmallVector<StateSlot>
buildVIStateSchema(VIGuide guide, int64_t positionSize, mlir::Type elemType) {
  mlir::MLIRContext *ctx = elemType.getContext();
  auto i64TensorType =
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
        {"opt_step_count", i64TensorType},
    };
    break;
  case VIGuide::AutoDelta:
    slots = {
        {"loc", vecType},
        {"opt_m_loc", vecType},
        {"opt_v_loc", vecType},
        {"opt_step_count", i64TensorType},
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
        {"opt_step_count", i64TensorType},
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

/// The `stateIn` value of slot `name` under `schema`, or null when the schema
/// has no such slot or `stateIn` is empty.
inline mlir::Value stateInSlot(mlir::ArrayAttr schema, mlir::ValueRange stateIn,
                               llvm::StringRef name) {
  std::optional<unsigned> idx = slotIndex(schema, name);
  if (!idx || *idx >= stateIn.size())
    return mlir::Value();
  return stateIn[*idx];
}

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H
