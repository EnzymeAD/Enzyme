//===- HLFIREnzymeCallLowering.cpp - Fortran hook calls -> enzyme ops -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Rewrites Fortran Enzyme differentiation-hook calls into first-class enzyme
// ops, at the HLFIR stage (while hlfir.* intrinsics are still present):
//
//   %f  = fir.address_of(@_QPmm) : ...
//   %bp = fir.emboxproc %f : ... -> !fir.boxproc<() -> ()>
//   %r  = fir.call @_QPf__enzyme_fwddiff(%bp, enzyme_dup, %x, %dx, ...) : ...
//     ==>
//   %r  = enzyme.fwddiff @_QPmm(%x, %dx, ...)
//           {activity = [#enzyme<activity enzyme_dup>, ...],
//            ret_activity = [#enzyme<activity enzyme_dupnoneed>]} : (...) -> T
//
// The callee is recovered by tracing the boxproc operand back to its
// fir.address_of. Activity markers (enzyme_const / enzyme_dup / enzyme_dupnoneed
// / enzyme_out) are the C-bound globals from enzyme/Fortran/enzyme.f90; an
// operand that addresses one sets the activity of the *following* argument,
// exactly as HandleAutoDiff/getMetadataName do for the LLVM __enzyme_* path.
// enzyme_dup / enzyme_dupnoneed consume a (primal, shadow) pair.
//
//===----------------------------------------------------------------------===//

#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/StringRef.h"

#include <optional>

using namespace mlir;

namespace {

// Trace a value back to the name of the global symbol it ultimately addresses,
// peeling the ops Flang puts between the fir.address_of and the use
// (hlfir.declare, fir.emboxproc, fir.convert, fir.load). Mirrors
// getMetadataName in Enzyme.cpp.
static llvm::StringRef traceSymbolName(Value v) {
  while (v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return {};
    if (auto d = dyn_cast<hlfir::DeclareOp>(def)) {
      v = d.getMemref();
      continue;
    }
    if (auto e = dyn_cast<fir::EmboxProcOp>(def)) {
      v = e.getFunc();
      continue;
    }
    if (auto c = dyn_cast<fir::ConvertOp>(def)) {
      v = c.getValue();
      continue;
    }
    if (auto l = dyn_cast<fir::LoadOp>(def)) {
      v = l.getMemref();
      continue;
    }
    if (auto a = dyn_cast<fir::AddrOfOp>(def))
      return a.getSymbolAttr().getLeafReference().getValue();
    return {};
  }
  return {};
}

// If `name` is an activity marker global, return the activity it denotes.
static std::optional<enzyme::Activity> markerActivity(llvm::StringRef name) {
  if (name == "enzyme_const")
    return enzyme::Activity::enzyme_const;
  if (name == "enzyme_dup")
    return enzyme::Activity::enzyme_dup;
  if (name == "enzyme_dupnoneed")
    return enzyme::Activity::enzyme_dupnoneed;
  if (name == "enzyme_out")
    return enzyme::Activity::enzyme_active;
  return std::nullopt;
}

static bool activityHasShadow(enzyme::Activity a) {
  return a == enzyme::Activity::enzyme_dup ||
         a == enzyme::Activity::enzyme_dupnoneed;
}

// Rewrite one fir.call to a differentiation hook. Returns success if it was (or
// need not be) handled; failure on a malformed call.
static LogicalResult lowerEnzymeCall(fir::CallOp call) {
  auto callee = call.getCallee();
  if (!callee)
    return success(); // indirect call, not a hook
  llvm::StringRef cn = callee->getLeafReference().getValue();
  bool fwd = cn.contains("enzyme_fwddiff");
  bool rev = cn.contains("enzyme_autodiff");
  if (!fwd && !rev)
    return success();

  auto args = call.getArgs();
  if (args.empty())
    return call.emitError("enzyme differentiation hook call has no callee "
                          "operand");

  llvm::StringRef target = traceSymbolName(args[0]);
  if (target.empty())
    return call.emitError(
        "could not resolve the function being differentiated (expected a "
        "fir.emboxproc of a fir.address_of)");

  MLIRContext *ctx = call.getContext();
  OpBuilder b(call);

  // Walk the differentiation arguments, honoring activity markers. Default
  // activity is enzyme_dup, matching the LLVM path.
  SmallVector<Value> inputs;
  SmallVector<Attribute> activity;
  for (size_t i = 1, n = args.size(); i < n;) {
    enzyme::Activity ty = enzyme::Activity::enzyme_dup;
    if (auto m = markerActivity(traceSymbolName(args[i]))) {
      ty = *m;
      if (++i >= n)
        return call.emitError("activity marker is not followed by an argument");
    }
    inputs.push_back(args[i++]); // primal
    if (activityHasShadow(ty)) {
      if (i >= n)
        return call.emitError("enzyme_dup argument is missing its shadow");
      inputs.push_back(args[i++]); // shadow
    }
    activity.push_back(enzyme::ActivityAttr::get(ctx, ty));
  }

  // Result activity: forward returns the tangent (dupnoneed); reverse seeds an
  // active result.
  SmallVector<Attribute> retActivity;
  for ([[maybe_unused]] Type rt : call.getResultTypes())
    retActivity.push_back(enzyme::ActivityAttr::get(
        ctx, fwd ? enzyme::Activity::enzyme_dupnoneed
                 : enzyme::Activity::enzyme_active));

  Operation *newOp;
  if (fwd)
    newOp = enzyme::ForwardDiffOp::create(
        b, call.getLoc(), call.getResultTypes(), target, inputs,
        b.getArrayAttr(activity), b.getArrayAttr(retActivity));
  else
    newOp = enzyme::AutoDiffOp::create(
        b, call.getLoc(), call.getResultTypes(), target, inputs,
        b.getArrayAttr(activity), b.getArrayAttr(retActivity));

  call.replaceAllUsesWith(newOp->getResults());
  call.erase();
  return success();
}

struct HLFIRLowerEnzymeCallsPass
    : public PassWrapper<HLFIRLowerEnzymeCallsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HLFIRLowerEnzymeCallsPass)

  StringRef getArgument() const final { return "enzyme-lower-fortran-calls"; }
  StringRef getDescription() const final {
    return "Rewrite Fortran f__enzyme_fwddiff/f__enzyme_autodiff calls into "
           "enzyme.fwddiff/enzyme.autodiff ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<enzyme::EnzymeDialect>();
  }

  void runOnOperation() override {
    SmallVector<fir::CallOp> calls;
    getOperation().walk([&](fir::CallOp call) { calls.push_back(call); });
    for (fir::CallOp call : calls)
      if (failed(lowerEnzymeCall(call)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::enzyme::createHLFIRLowerEnzymeCallsPass() {
  return std::make_unique<HLFIRLowerEnzymeCallsPass>();
}

void mlir::enzyme::registerHLFIRLowerEnzymeCallsPass() {
  PassRegistration<HLFIRLowerEnzymeCallsPass>();
}
