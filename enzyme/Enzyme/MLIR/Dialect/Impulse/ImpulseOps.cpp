//===- ImpulseOps.cpp - Impulse dialect ops ----------------------*- C++
//-*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Passes/InferStateLayout.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::impulse;

//===----------------------------------------------------------------------===//
// SampleOp
//===----------------------------------------------------------------------===//

LogicalResult SampleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  if (getLogpdfAttr()) {
    auto global = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        *this, getLogpdfAttr());
    if (!global)
      return emitOpError("'")
             << getLogpdf().value() << "' does not reference a valid global "
             << "funcOp";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// GenerateOp
//===----------------------------------------------------------------------===//

LogicalResult GenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// SimulateOp
//===----------------------------------------------------------------------===//

LogicalResult SimulateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// RegenerateOp
//===----------------------------------------------------------------------===//

LogicalResult
RegenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// MHOp
//===----------------------------------------------------------------------===//

LogicalResult MHOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto global =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, getFnAttr());
  if (!global)
    return emitOpError("'")
           << getFn() << "' does not reference a valid global funcOp";

  return success();
}

//===----------------------------------------------------------------------===//
// InferOp
//===----------------------------------------------------------------------===//

LogicalResult InferOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (auto fnAttr = getFnAttr()) {
    auto global =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
    if (!global)
      return emitOpError("'")
             << getFn().value() << "' does not reference a valid global funcOp";
  }

  if (auto logpdfAttr = getLogpdfFnAttr()) {
    auto global =
        symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, logpdfAttr);
    if (!global)
      return emitOpError("'") << logpdfAttr.getValue()
                              << "' does not reference a valid global funcOp";
  }

  return success();
}

LogicalResult InferOp::verify() {
  bool hasHMC = getHmcConfig().has_value();
  bool hasNUTS = getNutsConfig().has_value();
  bool hasVI = getViConfig().has_value();

  if (hasHMC + hasNUTS + hasVI != 1) {
    return emitOpError("exactly one of hmc_config, nuts_config, or vi_config "
                       "must be specified");
  }

  int64_t numChains = getNumChains();
  if (numChains < 1)
    return emitOpError("`num_chains` must be >= 1");
  if (numChains > 1 && !hasHMC && !hasNUTS)
    return emitOpError("`num_chains` > 1 requires a sampling config "
                       "(hmc_config or nuts_config)");

  if (!getFnAttr() && !getLogpdfFnAttr()) {
    return emitOpError("one of `fn` or `logpdf_fn` must be specified");
  }

  if (getFnAttr() && getLogpdfFnAttr()) {
    return emitOpError("specifying both `fn` and `logpdf_fn` is unsupported");
  }

  if (getLogpdfFnAttr() && !getInitPosition()) {
    return emitOpError(
        "custom logpdf mode requires `init_position` to be provided");
  }

  unsigned numStateOut = getStateOut().size();
  ArrayAttr schema = getStateSchemaAttr();
  if (numStateOut != 0) {
    if (!schema)
      return emitOpError(
          "`state_schema` is required when `state_out` is non-empty");
    if (schema.size() != numStateOut)
      return emitOpError("`state_schema` size (")
             << schema.size() << ") must match `state_out` size ("
             << numStateOut << ")";
  } else if (schema && schema.size() != 0) {
    return emitOpError(
        "`state_schema` must be empty when `state_out` is empty");
  }

  unsigned numStateIn = getStateIn().size();
  if (numStateIn != 0 && numStateIn != numStateOut) {
    return emitOpError("`state_in` must be empty or match `state_out` size (")
           << numStateOut << "), got " << numStateIn;
  }

  if (numStateOut != 0) {
    SmallVector<impulse::StateSlot> expected;
    if (hasVI) {
      auto locIdx = impulse::slotIndex(schema, "loc");
      if (!locIdx)
        return emitOpError("VI `state_schema` must contain a `loc` slot");
      auto locType =
          dyn_cast<RankedTensorType>(getStateOut()[*locIdx].getType());
      if (!locType || locType.getRank() != 2)
        return emitOpError("VI `loc` state slot must be a rank-2 `1xd` tensor");
      int64_t positionSize = locType.getShape().back();
      impulse::VIGuide guide;
      switch (getViConfig()->getGuideKind().getValue()) {
      case impulse::GuideKind::AUTO_NORMAL:
        guide = impulse::VIGuide::AutoNormal;
        break;
      case impulse::GuideKind::AUTO_DELTA:
        guide = impulse::VIGuide::AutoDelta;
        break;
      case impulse::GuideKind::AUTO_MULTIVARIATE_NORMAL:
        guide = impulse::VIGuide::AutoMVN;
        break;
      default:
        return emitOpError("VI `state_schema` is unsupported for this guide "
                           "kind");
      }
      expected = impulse::buildVIStateSchema(guide, positionSize,
                                             locType.getElementType());
      if (expected.size() != numStateOut)
        return emitOpError("`state_schema` size (")
               << numStateOut << ") does not match the canonical VI layout ("
               << expected.size() << ")";
    } else {
      auto posIdx = impulse::slotIndex(schema, "position");
      auto immIdx = impulse::slotIndex(schema, "inverse_mass_matrix");
      if (!posIdx || !immIdx)
        return emitOpError("`state_schema` must contain `position` and "
                           "`inverse_mass_matrix` slots");
      auto posType =
          dyn_cast<RankedTensorType>(getStateOut()[*posIdx].getType());
      auto immType =
          dyn_cast<RankedTensorType>(getStateOut()[*immIdx].getType());
      int64_t expectedPosRank = numChains > 1 ? 3 : 2;
      if (!posType || posType.getRank() != expectedPosRank || !immType)
        return emitOpError("`position`/`inverse_mass_matrix` state slots must "
                           "be ranked tensors of the expected rank");
      if (numChains > 1 && posType.getShape()[0] != numChains)
        return emitOpError("chain-major `position` slot leading dim (")
               << posType.getShape()[0] << ") must equal num_chains ("
               << numChains << ")";
      int64_t positionSize = posType.getShape().back();
      RankedTensorType singleImmType =
          numChains > 1 ? RankedTensorType::get(immType.getShape().drop_front(),
                                                immType.getElementType())
                        : immType;

      bool adaptive = impulse::slotIndex(schema, "window_idx").has_value();
      expected = impulse::buildStateSchema(adaptive, positionSize,
                                           posType.getElementType(),
                                           singleImmType, numChains);
      if (expected.size() != numStateOut)
        return emitOpError("`state_schema` size (")
               << numStateOut << ") does not match the canonical "
               << (adaptive ? "adaptive" : "plain") << " layout ("
               << expected.size() << ")";
    }
    for (unsigned i = 0; i < numStateOut; ++i) {
      StringRef name = cast<StringAttr>(schema[i]).getValue();
      if (name != expected[i].name)
        return emitOpError("`state_schema` slot ")
               << i << " is '" << name << "', expected '" << expected[i].name
               << "'";
      if (getStateOut()[i].getType() != expected[i].type)
        return emitOpError("`state_out` slot '")
               << name << "' has type " << getStateOut()[i].getType()
               << ", expected " << expected[i].type;
      if (numStateIn != 0 && getStateIn()[i].getType() != expected[i].type)
        return emitOpError("`state_in` slot '")
               << name << "' has type " << getStateIn()[i].getType()
               << ", expected " << expected[i].type;
    }
  }

  return success();
}
