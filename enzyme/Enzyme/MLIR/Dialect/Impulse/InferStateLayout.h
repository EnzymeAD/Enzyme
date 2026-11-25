//===- InferStateLayout.h - Canonical infer state schema --------*- C++ -*-===//

#ifndef ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H
#define ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H

#include "Dialect/Impulse/Impulse.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace mlir {
namespace impulse {

/// One slot of a `state_schema`: its name and the type of its value.
struct StateSlot {
  StringRef name;
  RankedTensorType type;
};

/// Names of the `state_schema` slots.
namespace state_slot {
inline constexpr StringLiteral kPosition = "position";
inline constexpr StringLiteral kGradient = "gradient";
inline constexpr StringLiteral kPotentialEnergy = "potential_energy";
inline constexpr StringLiteral kStepSize = "step_size";
inline constexpr StringLiteral kInverseMassMatrix = "inverse_mass_matrix";
inline constexpr StringLiteral kDaLogStepSize = "da_log_step_size";
inline constexpr StringLiteral kDaLogStepSizeAvg = "da_log_step_size_avg";
inline constexpr StringLiteral kDaGradientAvg = "da_gradient_avg";
inline constexpr StringLiteral kDaStepCount = "da_step_count";
inline constexpr StringLiteral kDaProxCenter = "da_prox_center";
inline constexpr StringLiteral kWelfordMean = "welford_mean";
inline constexpr StringLiteral kWelfordM2 = "welford_m2";
inline constexpr StringLiteral kWelfordN = "welford_n";
inline constexpr StringLiteral kWindowIdx = "window_idx";
inline constexpr StringLiteral kWarmupOffset = "warmup_offset";
inline constexpr StringLiteral kRng = "rng";
inline constexpr StringLiteral kLoc = "loc";
inline constexpr StringLiteral kScaleUnconstrained = "scale_unconstrained";
inline constexpr StringLiteral kScaleTrilUnconstrained =
    "scale_tril_unconstrained";
inline constexpr StringLiteral kOptMLoc = "opt_m_loc";
inline constexpr StringLiteral kOptVLoc = "opt_v_loc";
inline constexpr StringLiteral kOptMScale = "opt_m_scale";
inline constexpr StringLiteral kOptVScale = "opt_v_scale";
inline constexpr StringLiteral kOptMScaleTril = "opt_m_scale_tril";
inline constexpr StringLiteral kOptVScaleTril = "opt_v_scale_tril";
inline constexpr StringLiteral kOptStepCount = "opt_step_count";
inline constexpr StringLiteral kWeight = "weight";
} // namespace state_slot

/// Whether the state schema and the warmup adaptation treat an inverse mass
/// matrix of type `type` as a diagonal: `tensor<N>` and `tensor<1xN>` hold the
/// diagonal for every N, `tensor<1x1>` included; any other shape is a dense
/// matrix.
inline bool isDiagonalInverseMass(RankedTensorType type) {
  return type.getRank() == 1 ||
         (type.getRank() == 2 && type.getShape()[0] == 1);
}

/// The HMC/NUTS slots for `positionSize` positions of `elemType`, with the
/// adaptation state when `adaptive`. A null `invMassType` stands for a
/// diagonal of position type. When `numChains` > 1, every slot gets a leading
/// chain dimension and an `rng` slot holds one key of type `rngType`
/// (`tensor<2xui64>` when null) per chain.
inline SmallVector<StateSlot>
buildStateSchema(bool adaptive, int64_t positionSize, Type elemType,
                 RankedTensorType invMassType, int64_t numChains = 1,
                 RankedTensorType rngType = {}) {
  MLIRContext *ctx = elemType.getContext();
  auto i64TensorType = RankedTensorType::get({}, IntegerType::get(ctx, 64));
  auto f64 = Float64Type::get(ctx);
  auto positionType = RankedTensorType::get({1, positionSize}, elemType);
  auto scalarType = RankedTensorType::get({}, elemType);

  bool diagonal = !invMassType || isDiagonalInverseMass(invMassType);
  RankedTensorType invType = invMassType ? invMassType : positionType;
  auto welfordVec = RankedTensorType::get({positionSize}, f64);
  RankedTensorType welfordM2 =
      diagonal ? welfordVec
               : RankedTensorType::get({positionSize, positionSize}, f64);

  SmallVector<StateSlot> slots = {
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
      SmallVector<int64_t> shape;
      shape.push_back(numChains);
      shape.append(slot.type.getShape().begin(), slot.type.getShape().end());
      slot.type = RankedTensorType::get(shape, slot.type.getElementType());
    }
    auto ui64 = IntegerType::get(ctx, 64, IntegerType::Unsigned);
    RankedTensorType perChainRng =
        rngType ? rngType : RankedTensorType::get({2}, ui64);
    SmallVector<int64_t> rngShape;
    rngShape.push_back(numChains);
    rngShape.append(perChainRng.getShape().begin(),
                    perChainRng.getShape().end());
    slots.push_back(
        {state_slot::kRng,
         RankedTensorType::get(rngShape, perChainRng.getElementType())});
  }
  return slots;
}

/// The VI slots of the guide `guide`, which is not CUSTOM.
inline SmallVector<StateSlot>
buildVIStateSchema(GuideKind guide, int64_t positionSize, Type elemType) {
  MLIRContext *ctx = elemType.getContext();
  auto i64TensorType = RankedTensorType::get({}, IntegerType::get(ctx, 64));
  auto vecType = RankedTensorType::get({1, positionSize}, elemType);
  int64_t flat = positionSize * (positionSize + 1) / 2;
  auto flatType = RankedTensorType::get({flat}, elemType);

  switch (guide) {
  case GuideKind::AUTO_NORMAL:
    return {
        {state_slot::kLoc, vecType},
        {state_slot::kScaleUnconstrained, vecType},
        {state_slot::kOptMLoc, vecType},
        {state_slot::kOptVLoc, vecType},
        {state_slot::kOptMScale, vecType},
        {state_slot::kOptVScale, vecType},
        {state_slot::kOptStepCount, i64TensorType},
    };
  case GuideKind::AUTO_DELTA:
    return {
        {state_slot::kLoc, vecType},
        {state_slot::kOptMLoc, vecType},
        {state_slot::kOptVLoc, vecType},
        {state_slot::kOptStepCount, i64TensorType},
    };
  case GuideKind::AUTO_MULTIVARIATE_NORMAL:
    return {
        {state_slot::kLoc, vecType},
        {state_slot::kScaleTrilUnconstrained, flatType},
        {state_slot::kOptMLoc, vecType},
        {state_slot::kOptVLoc, vecType},
        {state_slot::kOptMScaleTril, flatType},
        {state_slot::kOptVScaleTril, flatType},
        {state_slot::kOptStepCount, i64TensorType},
    };
  case GuideKind::CUSTOM:
    break;
  }
  llvm_unreachable("a CUSTOM guide has no state schema");
}

/// Only `weight`: trace and rng travel as the op's universal results.
inline SmallVector<StateSlot> buildMHStateSchema(Type elemType) {
  auto scalarType = RankedTensorType::get({}, elemType);
  return {{state_slot::kWeight, scalarType}};
}

/// The index of slot `name` in `schema`, if `schema` has one.
inline std::optional<unsigned> slotIndex(ArrayAttr schema, StringRef name) {
  if (!schema)
    return std::nullopt;
  for (auto [i, attr] : llvm::enumerate(schema))
    if (auto s = dyn_cast<StringAttr>(attr))
      if (s.getValue() == name)
        return static_cast<unsigned>(i);
  return std::nullopt;
}

/// The `stateIn` value of slot `name` under `schema`, or null when the schema
/// has no such slot or `stateIn` is empty.
inline Value stateInSlot(ArrayAttr schema, ValueRange stateIn, StringRef name) {
  std::optional<unsigned> idx = slotIndex(schema, name);
  if (!idx || *idx >= stateIn.size())
    return Value();
  return stateIn[*idx];
}

} // namespace impulse
} // namespace mlir

#endif // ENZYME_MLIR_DIALECT_IMPULSE_INFER_STATE_LAYOUT_H
