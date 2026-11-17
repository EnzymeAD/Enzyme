//===- CIRRaiseEnzymeCalls.cpp - Raise __enzyme_* calls in ClangIR -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise the `__enzyme_autodiff` and
// `__enzyme_fwddiff` calls of a C or C++ program, as clang emits them in
// ClangIR, to enzyme.autodiff and enzyme.fwddiff ops, so that the program can
// be differentiated without writing those ops by hand. The marker arguments
// are read with the table in EnzymeCallMarkers.h, the same one the LLVM pass
// that lowers these calls reads, and an argument without a marker gets the
// activity that pass would give it.
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#include "../../EnzymeCallMarkers.h"

#ifdef ENZYME_CLANG_HAS_CIR
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CIRRAISEENZYMECALLSPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

#ifdef ENZYME_CLANG_HAS_CIR

// The activity an argument of this type has when no marker names one in
// conrrespond to  floats are active, pointers are duplicated, and integers are
// constant.
std::optional<DIFFE_TYPE> defaultActivity(Type ty, bool forward) {
  if (isa<cir::FPTypeInterface>(ty))
    return forward ? DIFFE_TYPE::DUP_ARG : DIFFE_TYPE::OUT_DIFF;
  if (isa<cir::PointerType>(ty))
    return DIFFE_TYPE::DUP_ARG;
  if (isa<cir::IntType, cir::BoolType>(ty))
    return DIFFE_TYPE::CONSTANT;
  if (auto complex = dyn_cast<cir::ComplexType>(ty))
    return defaultActivity(complex.getElementType(), forward);
  if (auto array = dyn_cast<cir::ArrayType>(ty))
    return defaultActivity(array.getElementType(), forward);
  if (auto vector = dyn_cast<cir::VectorType>(ty))
    return defaultActivity(vector.getElementType(), forward);
  if (auto record = dyn_cast<cir::StructType>(ty)) {
    // Duplicated if any member is, active if any member is, and constant
    // otherwise.
    DIFFE_TYPE result = DIFFE_TYPE::CONSTANT;
    for (Type member : record.getMembers()) {
      std::optional<DIFFE_TYPE> act = defaultActivity(member, forward);
      if (!act)
        return std::nullopt;
      if (*act == DIFFE_TYPE::DUP_ARG)
        return DIFFE_TYPE::DUP_ARG;
      if (*act != DIFFE_TYPE::CONSTANT)
        result = *act;
    }
    return result;
  }
  return std::nullopt;
}

// The activity attribute for a DIFFE_TYPE. A return can leave its primal out.
enzyme::Activity toActivity(DIFFE_TYPE ty, bool primalNeeded) {
  switch (ty) {
  case DIFFE_TYPE::OUT_DIFF:
    return primalNeeded ? enzyme::Activity::enzyme_active
                        : enzyme::Activity::enzyme_activenoneed;
  case DIFFE_TYPE::DUP_ARG:
    return primalNeeded ? enzyme::Activity::enzyme_dup
                        : enzyme::Activity::enzyme_dupnoneed;
  case DIFFE_TYPE::DUP_NONEED:
    return enzyme::Activity::enzyme_dupnoneed;
  case DIFFE_TYPE::CONSTANT:
    return primalNeeded ? enzyme::Activity::enzyme_const
                        : enzyme::Activity::enzyme_constnoneed;
  }
  llvm_unreachable("unknown DIFFE_TYPE");
}

// The global behind an argument, if it is a marker: clang passes `enzyme_dup`
// as a load of the global, and `&enzyme_dup` or an array marker as its
// address.
cir::GetGlobalOp markerGlobal(Value v) {
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  if (auto load = v.getDefiningOp<cir::LoadOp>())
    v = load.getAddr();
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  auto global = v.getDefiningOp<cir::GetGlobalOp>();
  if (global && global.getName().starts_with("enzyme_"))
    return global;
  return nullptr;
}

std::optional<StringRef> markerName(Value v) {
  if (cir::GetGlobalOp global = markerGlobal(v))
    return global.getName();
  return std::nullopt;
}

// Converts a scalar between what the call passes or is declared to return and
// what the function takes or returns. The call is variadic, so C has promoted
// a float to a double and a bool to an int.
FailureOr<Value> coerceScalar(OpBuilder &b, Location loc, Value v, Type to) {
  Type from = v.getType();
  if (from == to)
    return v;
  if (isa<cir::FPTypeInterface>(from) && isa<cir::FPTypeInterface>(to))
    return cir::CastOp::create(b, loc, to, cir::CastKind::floating, v)
        .getResult();
  if (isa<cir::IntType>(from) && isa<cir::IntType>(to))
    return cir::CastOp::create(b, loc, to, cir::CastKind::integral, v)
        .getResult();
  if (isa<cir::IntType>(from) && isa<cir::BoolType>(to))
    return cir::CastOp::create(b, loc, to, cir::CastKind::int_to_bool, v)
        .getResult();
  if (isa<cir::PointerType>(from) && isa<cir::PointerType>(to))
    return cir::CastOp::create(b, loc, to, cir::CastKind::bitcast, v)
        .getResult();
  return failure();
}

// Converts what the derivative returns to what the call was declared to
// return.
// A value that cannot be coerced will create a stack slot and do bitcast and
// store to it.
FailureOr<Value> coerceResult(OpBuilder &b, Location loc, Value v, Type to,
                              cir::FuncOp caller) {
  if (FailureOr<Value> scalar = coerceScalar(b, loc, v, to); succeeded(scalar))
    return scalar;
  Type from = v.getType();
  if (!isa<cir::StructType, cir::ArrayType>(from) ||
      !isa<cir::StructType, cir::ArrayType>(to))
    return failure();
  DataLayout dl = DataLayout::closest(caller);
  if (dl.getTypeSize(from) != dl.getTypeSize(to))
    return failure();
  OpBuilder entry = OpBuilder::atBlockBegin(&caller.getBody().front());
  Value slot = cir::AllocaOp::create(
      entry, loc, cir::PointerType::get(to), "enzyme.result",
      entry.getI64IntegerAttr(
          std::max(dl.getTypeABIAlignment(from), dl.getTypeABIAlignment(to))));
  Value asFrom = cir::CastOp::create(b, loc, cir::PointerType::get(from),
                                     cir::CastKind::bitcast, slot);
  cir::StoreOp::create(b, loc, v, asFrom);
  return cir::LoadOp::create(b, loc, to, slot).getResult();
}

// The function an `__enzyme_*` call differentiates: its first argument, a
// function address cast to a void pointer.
cir::FuncOp calledFunction(cir::CallOp call, SymbolTableCollection &symbols) {
  if (call.getArgOperands().empty())
    return nullptr;
  Value v = call.getArgOperands().front();
  while (auto cast = v.getDefiningOp<cir::CastOp>())
    v = cast.getSrc();
  auto global = v.getDefiningOp<cir::GetGlobalOp>();
  if (!global)
    return nullptr;
  return symbols.lookupNearestSymbolFrom<cir::FuncOp>(call,
                                                      global.getNameAttr());
}

LogicalResult raiseCall(cir::CallOp call, bool forward,
                        SymbolTableCollection &symbols) {
  Location loc = call.getLoc();
  StringRef callee = *call.getCallee();
  cir::FuncOp fn = calledFunction(call, symbols);
  if (!fn)
    return call.emitError()
           << "the first argument of " << callee
           << " must be the address of a function defined in this module";
  if (fn.isDeclaration())
    return call.emitError() << "cannot differentiate @" << fn.getName()
                            << ": it has no body in this module";

  ArrayRef<Type> paramTypes = fn.getFunctionType().getInputs();
  ArrayRef<Type> resultTypes = fn.getFunctionType().getReturnTypes();
  OperandRange operands = call.getArgOperands();

  OpBuilder b(call);
  SmallVector<Value> inputs;
  SmallVector<DIFFE_TYPE> activities;
  std::optional<DIFFE_TYPE> retOverride;
  bool primalReturn = false, strongZero = false;

  unsigned param = 0;
  for (unsigned i = 1; i < operands.size(); ++i) {
    std::optional<DIFFE_TYPE> activity;
    bool isFlag = false;
    while (std::optional<StringRef> name = markerName(operands[i])) {
      std::optional<enzyme_markers::MarkerInfo> marker =
          enzyme_markers::lookupEnzymeMarker(*name);
      if (!marker)
        return call.emitError() << "unknown Enzyme marker " << *name;
      switch (marker->role) {
      case enzyme_markers::MarkerRole::Activity:
        if (marker->extraOperands)
          return call.emitError()
                 << *name << " is not supported on ClangIR yet";
        activity = marker->activity;
        break;
      case enzyme_markers::MarkerRole::Modifier:
        // enzyme_not_overwritten: a combined derivative never assumes an
        // argument is overwritten.
        break;
      case enzyme_markers::MarkerRole::Interleave:
        return call.emitError() << *name << " is not supported on ClangIR yet";
      case enzyme_markers::MarkerRole::CallFlag:
        if (*name == "enzyme_primal_return")
          primalReturn = true;
        else if (*name == "enzyme_strong_zero")
          strongZero = true;
        else if (*name == "enzyme_const_return")
          retOverride = DIFFE_TYPE::CONSTANT;
        else if (*name == "enzyme_active_return")
          retOverride = DIFFE_TYPE::OUT_DIFF;
        else if (*name == "enzyme_dup_return")
          retOverride = DIFFE_TYPE::DUP_ARG;
        else
          return call.emitError()
                 << *name << " is not supported on ClangIR yet";
        isFlag = true;
        break;
      }
      // Only reverse mode has gradients to return, of an argument or to seed
      // the return with.
      if (forward && (activity == DIFFE_TYPE::OUT_DIFF ||
                      retOverride == DIFFE_TYPE::OUT_DIFF))
        return call.emitError() << *name << " is not valid in forward mode";
      if (isFlag)
        break;
      if (++i == operands.size())
        return call.emitError() << *name << " is not followed by an argument";
    }
    if (isFlag)
      continue;

    if (param == paramTypes.size())
      return call.emitError() << "too many arguments for @" << fn.getName()
                              << ", which takes " << paramTypes.size();
    Type paramTy = paramTypes[param];
    if (!activity)
      activity = defaultActivity(paramTy, forward);
    if (!activity)
      return call.emitError()
             << "cannot infer the activity of argument " << param << " of type "
             << paramTy << "; name it with a marker";

    auto pass = [&](StringRef what) -> LogicalResult {
      FailureOr<Value> v = coerceScalar(b, loc, operands[i], paramTy);
      if (failed(v))
        return call.emitError()
               << "cannot pass " << operands[i].getType() << " as " << what
               << "argument " << param << " of type " << paramTy;
      inputs.push_back(*v);
      return success();
    };
    if (failed(pass("")))
      return failure();
    activities.push_back(*activity);
    // Value in shadow space, should next to the primal value.
    if (enzyme_markers::markerActivityTakesShadow(*activity)) {
      if (++i == operands.size() || markerName(operands[i]))
        return call.emitError()
               << "argument " << param << " needs a shadow after it";
      if (failed(pass("the shadow of ")))
        return failure();
    }
    ++param;
  }
  if (param != paramTypes.size())
    return call.emitError() << "too few arguments for @" << fn.getName()
                            << ", which takes " << paramTypes.size();

  // The return: a float is active and seeded with 1.0 in reverse mode, and
  // has its tangent returned in forward mode. Only the derivative is returned
  // unless enzyme_primal_return asks for the primal too.
  SmallVector<DIFFE_TYPE> retActivities;
  SmallVector<Type> derivativeResults;
  for (Type ty : resultTypes) {
    std::optional<DIFFE_TYPE> ret =
        retOverride ? retOverride : defaultActivity(ty, forward);
    if (!ret)
      return call.emitError() << "cannot infer the activity of the return "
                              << "type " << ty;
    if (primalReturn)
      derivativeResults.push_back(ty);
    if (enzyme_markers::markerActivityTakesShadow(*ret)) {
      // The tangent, which only forward mode returns.
      if (!forward)
        return call.emitError() << "cannot differentiate a returned " << ty
                                << " in reverse mode";
      derivativeResults.push_back(ty);
    } else if (*ret == DIFFE_TYPE::OUT_DIFF) {
      // The seed, which only reverse mode takes.
      auto fpTy = dyn_cast<cir::FPTypeInterface>(ty);
      if (!fpTy)
        return call.emitError() << "cannot seed a return of type " << ty;
      llvm::APFloat one(fpTy.getFloatSemantics(), 1);
      inputs.push_back(
          cir::ConstantOp::create(b, loc, cir::FPAttr::get(ty, one)));
    }
    retActivities.push_back(*ret);
  }
  if (!forward)
    for (auto [ty, act] : llvm::zip_equal(paramTypes, activities))
      if (act == DIFFE_TYPE::OUT_DIFF)
        derivativeResults.push_back(ty);

  // The derivative's results as the function interface packs them.
  cast<enzyme::AutoDiffFunctionInterface>(fn.getOperation())
      .transformResultTypes(derivativeResults);

  MLIRContext *ctx = call.getContext();
  auto activityArray = [&](ArrayRef<DIFFE_TYPE> tys, bool primalNeeded) {
    return b.getArrayAttr(
        llvm::map_to_vector(tys, [&](DIFFE_TYPE ty) -> Attribute {
          return enzyme::ActivityAttr::get(ctx, toActivity(ty, primalNeeded));
        }));
  };
  ArrayAttr activityAttr = activityArray(activities, /*primalNeeded=*/true);
  ArrayAttr retActivityAttr = activityArray(retActivities, primalReturn);
  Operation *diff;
  if (forward)
    diff = enzyme::ForwardDiffOp::create(
        b, loc, derivativeResults, fn.getName(), inputs, activityAttr,
        retActivityAttr, /*width=*/uint64_t(1), strongZero);
  else
    diff = enzyme::AutoDiffOp::create(b, loc, derivativeResults, fn.getName(),
                                      inputs, activityAttr, retActivityAttr,
                                      /*width=*/uint64_t(1), strongZero,
                                      /*atomic_add=*/false);

  if (call.getNumResults() && !call.getResult().use_empty()) {
    if (diff->getNumResults() == 0)
      return call.emitError() << "the result of " << callee
                              << " is used, but the derivative returns nothing";
    FailureOr<Value> result =
        coerceResult(b, loc, diff->getResult(0), call.getResult().getType(),
                     call->getParentOfType<cir::FuncOp>());
    if (failed(result))
      return call.emitError()
             << callee << " is declared to return "
             << call.getResult().getType() << ", but the derivative returns "
             << diff->getResult(0).getType();
    call.getResult().replaceAllUsesWith(*result);
  }

  // Erase the orignal call and operands in caller
  SmallVector<Value> spelling(operands);
  call.erase();

  SmallVector<Operation *> worklist;
  auto queueIfDead = [&](Value v) {
    Operation *op = v.getDefiningOp();
    if (op && op->use_empty() &&
        isa<cir::CastOp, cir::LoadOp, cir::GetGlobalOp>(op) &&
        !llvm::is_contained(worklist, op))
      worklist.push_back(op);
  };
  for (Value v : spelling)
    queueIfDead(v);
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    SmallVector<Value> uses(op->getOperands());
    op->erase();
    for (Value v : uses)
      queueIfDead(v);
  }
  return success();
}

#endif // ENZYME_CLANG_HAS_CIR

struct CIRRaiseEnzymeCallsPass
    : public enzyme::impl::CIRRaiseEnzymeCallsPassBase<
          CIRRaiseEnzymeCallsPass> {
  void runOnOperation() override {
#ifndef ENZYME_CLANG_HAS_CIR
    getOperation()->emitError()
        << "cir-raise-enzyme-calls needs Enzyme built against a Clang with "
           "ClangIR";
    signalPassFailure();
#else
    ModuleOp module = getOperation();
    SymbolTableCollection symbols;
    SmallVector<std::pair<cir::CallOp, bool>> calls;
    module.walk([&](cir::CallOp call) {
      std::optional<StringRef> callee = call.getCallee();
      if (!callee)
        return;
      if (callee->starts_with("__enzyme_autodiff"))
        calls.push_back({call, /*forward=*/false});
      else if (callee->starts_with("__enzyme_fwddiff"))
        calls.push_back({call, /*forward=*/true});
    });
    for (auto [call, forward] : calls)
      if (failed(raiseCall(call, forward, symbols)))
        return signalPassFailure();
#endif
  }
};

} // end anonymous namespace
