//===- ImpulseOps.cpp - Impulse dialect ops ----------------------*- C++
//-*-===//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Impulse/InferStateLayout.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::impulse;

static LogicalResult verifyDistSupport(Operation *op, DistAttr dist,
                                       SupportAttr support) {
  // Only parameterless bases: the site's dist operands belong to the chain.
  if (dist && dist.getTransforms() && !dist.getTransforms().empty() &&
      dist.getBase() != DistBaseKind::STD_NORMAL &&
      dist.getBase() != DistBaseKind::STD_UNIFORM)
    return op->emitOpError("a transform chain is only supported on the "
                           "STD_NORMAL and STD_UNIFORM bases");
  if (!dist || !support)
    return success();
  if (dist.getSupport().getKind() != support.getKind())
    return op->emitOpError(
        "explicit support contradicts the support derived from its dist "
        "attribute");
  return success();
}

//===----------------------------------------------------------------------===//
// Symbol uses
//===----------------------------------------------------------------------===//

/// Checks that `callee`, the value of the attribute `attrName` of `op`,
/// references a func.func. A null `callee` passes.
static LogicalResult verifyCallee(Operation *op,
                                  SymbolTableCollection &symbolTable,
                                  FlatSymbolRefAttr callee,
                                  StringAttr attrName) {
  if (!callee || symbolTable.lookupNearestSymbolFrom<func::FuncOp>(op, callee))
    return success();
  return op->emitOpError("`") << attrName.getValue() << "` references '"
                              << callee << "', which is not a func.func";
}

LogicalResult SampleOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName())))
    return failure();
  return verifyCallee(*this, symbolTable, getLogpdfAttr(), getLogpdfAttrName());
}

LogicalResult GenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName());
}

LogicalResult SimulateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName());
}

LogicalResult
RegenerateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName());
}

LogicalResult MHOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName());
}

LogicalResult InferOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyCallee(*this, symbolTable, getFnAttr(), getFnAttrName())))
    return failure();
  return verifyCallee(*this, symbolTable, getLogpdfFnAttr(),
                      getLogpdfFnAttrName());
}

//===----------------------------------------------------------------------===//
// SampleOp
//===----------------------------------------------------------------------===//

LogicalResult SampleOp::verify() {
  if (!getFnAttr() && !getDistAttr())
    return emitOpError(
        "requires either a sampler symbol `fn` or a `dist` attribute");
  if (DistAttr dist = getDistAttr()) {
    SmallVector<Type> distArgTypes;
    for (unsigned i = 1; i < getInputs().size(); ++i)
      distArgTypes.push_back(getInputs()[i].getType());
    RankedTensorType derived = dist.getSampleType(distArgTypes);
    if (derived && getNumResults() >= 2 && getResult(1).getType() != derived)
      return emitOpError("distribution sample value type ")
             << getResult(1).getType()
             << " disagrees with the type derived from the dist attribute "
             << derived;
  }
  return verifyDistSupport(*this, getDistAttr(), getSupportAttr());
}

LogicalResult SampleRegionOp::verify() {
  return verifyDistSupport(*this, getDistAttr(), getSupportAttr());
}

//===----------------------------------------------------------------------===//
// InferOp
//===----------------------------------------------------------------------===//

namespace {
/// A canonical `state_schema` and the name diagnostics give its layout.
struct CanonicalStateLayout {
  SmallVector<StateSlot> slots;
  StringRef name;
};
} // namespace

/// The canonical `state_schema` of `op`, which has a non-empty `state_out`
/// and a `state_schema` of the same size. Fails after emitting when the slots
/// the layout depends on are missing or ill-typed.
static FailureOr<CanonicalStateLayout> getCanonicalStateLayout(InferOp op) {
  ArrayAttr schema = op.getStateSchemaAttr();
  if (op.getMhConfig() || op.getGibbsConfig()) {
    auto weightType = dyn_cast<RankedTensorType>(op.getStateOut()[0].getType());
    if (!weightType || weightType.getRank() != 0)
      return op.emitOpError("MH `weight` state slot must be a scalar tensor");
    return CanonicalStateLayout{buildMHStateSchema(weightType.getElementType()),
                                "MH"};
  }
  if (op.getViConfig()) {
    std::optional<unsigned> locIdx = slotIndex(schema, state_slot::kLoc);
    if (!locIdx)
      return op.emitOpError("VI `state_schema` must contain a `loc` slot");
    auto locType =
        dyn_cast<RankedTensorType>(op.getStateOut()[*locIdx].getType());
    if (!locType || locType.getRank() != 2)
      return op.emitOpError(
          "VI `loc` state slot must be a rank-2 `1xd` tensor");
    GuideKind guideKind = op.getViConfig()->getGuideKind().getValue();
    if (guideKind == GuideKind::CUSTOM)
      return op.emitOpError(
          "VI `state_schema` is unsupported for this guide kind");
    return CanonicalStateLayout{buildVIStateSchema(guideKind,
                                                   locType.getShape().back(),
                                                   locType.getElementType()),
                                "VI"};
  }

  std::optional<unsigned> posIdx = slotIndex(schema, state_slot::kPosition);
  std::optional<unsigned> immIdx =
      slotIndex(schema, state_slot::kInverseMassMatrix);
  if (!posIdx || !immIdx)
    return op.emitOpError("`state_schema` must contain `position` and "
                          "`inverse_mass_matrix` slots");
  auto posType =
      dyn_cast<RankedTensorType>(op.getStateOut()[*posIdx].getType());
  auto immType =
      dyn_cast<RankedTensorType>(op.getStateOut()[*immIdx].getType());
  int64_t numChains = op.getNumChains();
  int64_t expectedPosRank = numChains > 1 ? 3 : 2;
  if (!posType || posType.getRank() != expectedPosRank || !immType)
    return op.emitOpError("`position`/`inverse_mass_matrix` state slots must "
                          "be ranked tensors of the expected rank");
  if (numChains > 1 && posType.getShape()[0] != numChains)
    return op.emitOpError("chain-major `position` slot leading dim (")
           << posType.getShape()[0] << ") must equal num_chains (" << numChains
           << ")";
  RankedTensorType singleImmType =
      numChains > 1 ? RankedTensorType::get(immType.getShape().drop_front(),
                                            immType.getElementType())
                    : immType;
  bool adaptive = slotIndex(schema, state_slot::kWindowIdx).has_value();
  // output_rng_state stays a single chain's key when num_chains > 1.
  auto rngType = dyn_cast<RankedTensorType>(op.getOutputRngState().getType());
  return CanonicalStateLayout{
      buildStateSchema(adaptive, posType.getShape().back(),
                       posType.getElementType(), singleImmType, numChains,
                       rngType),
      adaptive ? "adaptive" : "plain"};
}

LogicalResult InferOp::verify() {
  bool hasHMC = getHmcConfig().has_value();
  bool hasNUTS = getNutsConfig().has_value();
  bool hasVI = getViConfig().has_value();
  bool hasMH = getMhConfig().has_value();
  bool hasGibbs = getGibbsConfig().has_value();

  if (hasHMC + hasNUTS + hasVI + hasMH + hasGibbs != 1) {
    return emitOpError("exactly one of hmc_config, nuts_config, vi_config, "
                       "mh_config, or gibbs_config must be specified");
  }

  if ((getConstraint() || getConstrainedAddressesAttr()) && !hasMH && !hasGibbs)
    return emitOpError("`constraint`/`constrained_addresses` are only valid "
                       "with mh_config or gibbs_config");

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

  if (numStateOut == 0)
    return success();
  FailureOr<CanonicalStateLayout> layout = getCanonicalStateLayout(*this);
  if (failed(layout))
    return failure();
  ArrayRef<StateSlot> expected = layout->slots;
  if (expected.size() != numStateOut)
    return emitOpError("`state_schema` size (")
           << numStateOut << ") does not match the canonical " << layout->name
           << " layout (" << expected.size() << ")";
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
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyBranchYield(IfOp op, Region &branch,
                                       StringRef name) {
  TypeRange yielded = branch.front().getTerminator()->getOperandTypes();
  if (yielded != op.getResultTypes())
    return op.emitOpError()
           << name << " branch yields (" << yielded << ") but results are ("
           << op.getResultTypes() << ")";
  return success();
}

LogicalResult IfOp::verifyRegions() {
  if (failed(verifyBranchYield(*this, getTrueBranch(), "true")))
    return failure();
  return verifyBranchYield(*this, getFalseBranch(), "false");
}
