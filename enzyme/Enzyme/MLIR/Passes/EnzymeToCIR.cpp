//===- EnzymeToCIR.cpp - Lower custom Enzyme operations to ClangIR -------- //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower the caches and gradients the Enzyme AD
// procedure leaves in ClangIR functions to ClangIR itself, so that the
// derivative lowers to LLVM along with the rest of the module. The MemRef
// lowering cannot take them: a memref cannot hold ClangIR types.
//===----------------------------------------------------------------------===//

#include "Dialect/Dialect.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"

#ifdef ENZYME_CLANG_HAS_CIR
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_ENZYMEOPSTOCIRPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;

namespace {

#ifdef ENZYME_CLANG_HAS_CIR

cir::IntType sizeType(MLIRContext *ctx) {
  return cir::IntType::get(ctx, 64, /*isSigned=*/false);
}

Value constSize(OpBuilder &b, Location loc, uint64_t v) {
  return cir::ConstantOp::create(
      b, loc, cir::IntAttr::get(sizeType(b.getContext()), v));
}

Value load(OpBuilder &b, Location loc, Value ptr) {
  return cir::LoadOp::create(
      b, loc, cast<cir::PointerType>(ptr.getType()).getPointee(), ptr);
}

Value alloca(OpBuilder &b, Location loc, Type ty, StringRef name,
             const DataLayout &dl) {
  return cir::AllocaOp::create(b, loc, cir::PointerType::get(ty), name,
                               b.getI64IntegerAttr(dl.getTypeABIAlignment(ty)));
}

Value bitcast(OpBuilder &b, Location loc, Value v, Type ty) {
  return cir::CastOp::create(b, loc, ty, cir::CastKind::bitcast, v);
}

class CacheRuntime {
public:
  CacheRuntime(ModuleOp module) : module(module), symbols(module) {}

  FailureOr<cir::FuncOp> libraryFunction(Location loc, StringRef name,
                                         cir::FuncType type) {
    if (auto fn = symbols.lookup<cir::FuncOp>(name)) {
      if (fn.getFunctionType() == type)
        return fn;
      return emitError(loc) << "cannot lower enzyme caches: @" << name
                            << " is declared with type " << fn.getFunctionType()
                            << ", not " << type;
    }
    OpBuilder b(module.getContext());
    auto fn = cir::FuncOp::create(b, loc, name, type);
    fn.setPrivate();
    symbols.insert(fn, module.getBody()->begin());
    return fn;
  }

  // push(data, size, capacity, value)
  FailureOr<cir::FuncOp> pushFunction(Location loc, Type elemTy) {
    if (auto fn = pushFns.lookup(elemTy))
      return fn;
    MLIRContext *ctx = module.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    FailureOr<cir::FuncOp> realloc = libraryFunction(
        loc, "realloc", cir::FuncType::get({voidPtr, sizeType(ctx)}, voidPtr));
    if (failed(realloc))
      return failure();

    auto fn = buildFunctionInModule(
        loc, "__enzyme_cir_push",
        cir::FuncType::get({slotTypes(elemTy)[0], slotTypes(elemTy)[1],
                            slotTypes(elemTy)[2], elemTy},
                           cir::VoidType::get(ctx)));
    Block *entry = fn.addEntryBlock();
    Block *grow = new Block(), *store = new Block();
    fn.getBody().push_back(grow);
    fn.getBody().push_back(store);
    Value data = entry->getArgument(0), size = entry->getArgument(1),
          capacity = entry->getArgument(2), value = entry->getArgument(3);

    OpBuilder b = OpBuilder::atBlockBegin(entry);

    // if size == capacity
    // goto grow;
    // else
    // goto store;
    Value sizeVal = load(b, loc, size);
    Value capacityVal = load(b, loc, capacity);
    Value full =
        cir::CmpOp::create(b, loc, cir::CmpOpKind::eq, sizeVal, capacityVal);
    cir::BrCondOp::create(b, loc, full, grow, store);

    // grow:
    // newCapacity = capacity == 0 ? kInitialCapacity ? capacity * 2
    // bytes = sizeof(Ty) * newCapacity
    // *data = realloc(bytes)
    // capacity = cap
    // goto store;
    b.setInsertionPointToStart(grow);
    Value empty = cir::CmpOp::create(b, loc, cir::CmpOpKind::eq, capacityVal,
                                     constSize(b, loc, 0));
    Value newCapacity = cir::SelectOp::create(
        b, loc, empty, constSize(b, loc, kInitialCapacity),
        cir::MulOp::create(b, loc, capacityVal, constSize(b, loc, 2)));
    uint64_t elemSize =
        DataLayout::closest(fn).getTypeSize(elemTy).getFixedValue();
    Value bytes =
        cir::MulOp::create(b, loc, newCapacity, constSize(b, loc, elemSize));
    // TODO: DO null checking
    Value grown =
        cir::CallOp::create(
            b, loc, SymbolRefAttr::get(*realloc), voidPtr,
            ValueRange{bitcast(b, loc, load(b, loc, data), voidPtr), bytes})
            .getResult();
    cir::StoreOp::create(
        b, loc, bitcast(b, loc, grown, cir::PointerType::get(elemTy)), data);
    cir::StoreOp::create(b, loc, newCapacity, capacity);
    cir::BrOp::create(b, loc, store);

    // store:
    // slot = *data + size
    // *slot = val
    // size = (size + 1)
    // return
    b.setInsertionPointToStart(store);
    Value slot = cir::PtrStrideOp::create(b, loc, cir::PointerType::get(elemTy),
                                          load(b, loc, data), sizeVal);
    cir::StoreOp::create(b, loc, value, slot);
    cir::StoreOp::create(
        b, loc, cir::AddOp::create(b, loc, sizeVal, constSize(b, loc, 1)),
        size);
    cir::ReturnOp::create(b, loc, ValueRange());

    pushFns[elemTy] = fn;
    return fn;
  }

  // pop(data, size, capacity) removes and returns the last value; get returns
  // it and leaves it.
  cir::FuncOp popFunction(Location loc, Type elemTy, bool remove) {
    auto &fns = remove ? popFns : getFns;
    if (auto fn = fns.lookup(elemTy))
      return fn;
    SmallVector<Type, 3> slots = slotTypes(elemTy);
    auto fn = buildFunctionInModule(
        loc, remove ? "__enzyme_cir_pop" : "__enzyme_cir_get",
        cir::FuncType::get(slots, elemTy));
    Block *entry = fn.addEntryBlock();
    OpBuilder b = OpBuilder::atBlockBegin(entry);
    Value size = entry->getArgument(1);
    Value last =
        cir::SubOp::create(b, loc, load(b, loc, size), constSize(b, loc, 1));
    if (remove)
      cir::StoreOp::create(b, loc, last, size);
    Value slot =
        cir::PtrStrideOp::create(b, loc, cir::PointerType::get(elemTy),
                                 load(b, loc, entry->getArgument(0)), last);
    cir::ReturnOp::create(b, loc, ValueRange{load(b, loc, slot)});
    fns[elemTy] = fn;
    return fn;
  }

  FailureOr<cir::FuncOp> freeFunction(Location loc) {
    MLIRContext *ctx = module.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    return libraryFunction(
        loc, "free", cir::FuncType::get({voidPtr}, cir::VoidType::get(ctx)));
  }

  // The stack slots of a cache: the buffer, its size, and its capacity.
  static SmallVector<Type, 3> slotTypes(Type elemTy) {
    MLIRContext *ctx = elemTy.getContext();
    return {cir::PointerType::get(cir::PointerType::get(elemTy)),
            cir::PointerType::get(sizeType(ctx)),
            cir::PointerType::get(sizeType(ctx))};
  }

private:
  static constexpr uint64_t kInitialCapacity = 8;

  cir::FuncOp buildFunctionInModule(Location loc, StringRef name,
                                    cir::FuncType type) {
    OpBuilder b(module.getContext());
    auto fn = cir::FuncOp::create(b, loc, name, type,
                                  cir::GlobalLinkageKind::InternalLinkage);
    fn.setPrivate();
    // Renames it if the module already has a symbol by that name.
    symbols.insert(fn, module.getBody()->begin());
    return fn;
  }

  ModuleOp module;
  SymbolTable symbols;
  DenseMap<Type, cir::FuncOp> pushFns, popFns, getFns;
};

LogicalResult lowerGradient(enzyme::InitOp init, Block *entry,
                            const DataLayout &dl) {
  Type ty = cast<enzyme::GradientType>(init.getType()).getBasetype();
  OpBuilder b = OpBuilder::atBlockBegin(entry);
  Value slot = alloca(b, init.getLoc(), ty, "enzyme.gradient", dl);
  for (Operation *user : llvm::make_early_inc_range(init->getUsers())) {
    b.setInsertionPoint(user);
    if (auto set = dyn_cast<enzyme::SetOp>(user)) {
      cir::StoreOp::create(b, set.getLoc(), set.getValue(), slot);
    } else if (auto get = dyn_cast<enzyme::GetOp>(user)) {
      get.replaceAllUsesWith(load(b, get.getLoc(), slot));
    } else {
      return user->emitError() << "cannot lower this use of an enzyme gradient";
    }
    user->erase();
  }
  init.erase();
  return success();
}

// Enumlate the enzyme.cache type with stack push and pop
LogicalResult lowerCache(enzyme::InitOp init, cir::FuncOp fn,
                         const DataLayout &dl, CacheRuntime &runtime) {
  Location loc = init.getLoc();
  Type elemTy = cast<enzyme::CacheType>(init.getType()).getType();
  SmallVector<Type, 3> slotTypes = CacheRuntime::slotTypes(elemTy);

  static constexpr StringLiteral slotNames[] = {
      "enzyme.cache.data", "enzyme.cache.size", "enzyme.cache.capacity"};
  OpBuilder b = OpBuilder::atBlockBegin(&fn.getBody().front());
  SmallVector<Value, 3> slots;
  for (auto &&[ty, name] : llvm::zip_equal(slotTypes, slotNames))
    slots.push_back(
        alloca(b, loc, cast<cir::PointerType>(ty).getPointee(), name, dl));

  // An empty cache: a null buffer, which realloc allocates on the first push.
  b.setInsertionPoint(init);
  cir::StoreOp::create(b, loc,
                       cir::ConstantOp::create(
                           b, loc,
                           cir::ConstPtrAttr::get(cir::PointerType::get(elemTy),
                                                  b.getI64IntegerAttr(0))),
                       slots[0]);
  cir::StoreOp::create(b, loc, constSize(b, loc, 0), slots[1]);
  cir::StoreOp::create(b, loc, constSize(b, loc, 0), slots[2]);

  for (Operation *user : llvm::make_early_inc_range(init->getUsers())) {
    b.setInsertionPoint(user);
    if (auto push = dyn_cast<enzyme::PushOp>(user)) {
      FailureOr<cir::FuncOp> pushFn = runtime.pushFunction(loc, elemTy);
      if (failed(pushFn))
        return failure();
      SmallVector<Value, 4> args(slots.begin(), slots.end());
      args.push_back(push.getValue());
      cir::CallOp::create(b, push.getLoc(), SymbolRefAttr::get(*pushFn), Type(),
                          args);
    } else if (isa<enzyme::PopOp, enzyme::GetOp>(user)) {
      cir::FuncOp popFn =
          runtime.popFunction(loc, elemTy, /*remove=*/isa<enzyme::PopOp>(user));
      Value v = cir::CallOp::create(b, user->getLoc(),
                                    SymbolRefAttr::get(popFn), elemTy, slots)
                    .getResult();
      user->getResult(0).replaceAllUsesWith(v);
    } else {
      return user->emitError() << "cannot lower this use of an enzyme cache";
    }
    user->erase();
  }
  init.erase();

  FailureOr<cir::FuncOp> freeFn = runtime.freeFunction(loc);
  if (failed(freeFn))
    return failure();
  auto voidPtr = cir::PointerType::get(cir::VoidType::get(fn.getContext()));
  fn.walk([&](cir::ReturnOp ret) {
    OpBuilder rb(ret);
    cir::CallOp::create(
        rb, loc, SymbolRefAttr::get(*freeFn), Type(),
        ValueRange{bitcast(rb, loc, load(rb, loc, slots[0]), voidPtr)});
  });
  return success();
}

#endif // ENZYME_CLANG_HAS_CIR

struct EnzymeToCIRPass
    : public enzyme::impl::EnzymeOpsToCIRPassBase<EnzymeToCIRPass> {
  void runOnOperation() override {
#ifndef ENZYME_CLANG_HAS_CIR
    getOperation()->emitError()
        << "convert-enzyme-to-cir needs Enzyme built against a Clang with "
           "ClangIR";
    signalPassFailure();
#else
    ModuleOp module = getOperation();
    CacheRuntime runtime(module);
    SmallVector<cir::FuncOp> fns(module.getOps<cir::FuncOp>());
    for (cir::FuncOp fn : fns) {
      if (fn.isDeclaration())
        continue;
      DataLayout dl = DataLayout::closest(fn);
      SmallVector<enzyme::InitOp> inits;
      fn.walk([&](enzyme::InitOp init) { inits.push_back(init); });
      for (enzyme::InitOp init : inits) {
        LogicalResult lowered = failure();
        if (isa<enzyme::GradientType>(init.getType()))
          lowered = lowerGradient(init, &fn.getBody().front(), dl);
        else if (isa<enzyme::CacheType>(init.getType()))
          lowered = lowerCache(init, fn, dl, runtime);
        else
          init.emitError() << "cannot lower an enzyme.init of "
                           << init.getType();
        if (failed(lowered))
          return signalPassFailure();
      }
    }
#endif
  }
};

} // end anonymous namespace
