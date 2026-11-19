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

  // push(data, size, capacity, inline, value)
  FailureOr<cir::FuncOp> pushFunction(Location loc, Type elemTy) {
    if (auto fn = pushFns.lookup(elemTy))
      return fn;
    MLIRContext *ctx = module.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    FailureOr<cir::FuncOp> malloc = libraryFunction(
        loc, "malloc", cir::FuncType::get({sizeType(ctx)}, voidPtr));
    FailureOr<cir::FuncOp> realloc = libraryFunction(
        loc, "realloc", cir::FuncType::get({voidPtr, sizeType(ctx)}, voidPtr));
    if (failed(malloc) || failed(realloc))
      return failure();

    SmallVector<Type, 5> params = slotTypes(elemTy);
    params.push_back(elemTy);
    auto fn = buildFunctionInModule(
        loc, "__enzyme_cir_push",
        cir::FuncType::get(params, cir::VoidType::get(ctx)));
    Block *entry = fn.addEntryBlock();
    Block *grow = new Block(), *leaveInline = new Block(),
          *regrow = new Block(), *store = new Block();
    fn.getBody().push_back(grow);
    fn.getBody().push_back(leaveInline);
    fn.getBody().push_back(regrow);
    fn.getBody().push_back(store);
    Value data = entry->getArgument(0), size = entry->getArgument(1),
          capacity = entry->getArgument(2), inlineBuf = entry->getArgument(3),
          value = entry->getArgument(4);

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
    // newCapacity = capacity == 0 ? kInlineCapacity : capacity * 2
    // bytes = sizeof(Ty) * newCapacity
    // capacity = newCapacity
    // if (*data == inline) goto leaveInline; else goto regrow;
    b.setInsertionPointToStart(grow);
    Value empty = cir::CmpOp::create(b, loc, cir::CmpOpKind::eq, capacityVal,
                                     constSize(b, loc, 0));
    Value newCapacity = cir::SelectOp::create(
        b, loc, empty, constSize(b, loc, kInlineCapacity),
        cir::MulOp::create(b, loc, capacityVal, constSize(b, loc, 2)));
    uint64_t elemSize =
        DataLayout::closest(fn).getTypeSize(elemTy).getFixedValue();
    Value bytes =
        cir::MulOp::create(b, loc, newCapacity, constSize(b, loc, elemSize));
    cir::StoreOp::create(b, loc, newCapacity, capacity);
    Value old = load(b, loc, data);
    Value inInline =
        cir::CmpOp::create(b, loc, cir::CmpOpKind::eq, old, inlineBuf);
    cir::BrCondOp::create(b, loc, inInline, leaveInline, regrow);

    // leaveInline:
    // *data = malloc(bytes), with what the inline buffer holds
    // goto store;
    // TODO: DO null checking
    b.setInsertionPointToStart(leaveInline);
    Value buffer = cir::CallOp::create(b, loc, SymbolRefAttr::get(*malloc),
                                       voidPtr, ValueRange{bytes})
                       .getResult();
    cir::MemCpyOp::create(
        b, loc, buffer, bitcast(b, loc, old, voidPtr),
        cir::MulOp::create(b, loc, sizeVal, constSize(b, loc, elemSize)),
        /*dst_alignment=*/nullptr, /*src_alignment=*/nullptr);
    cir::StoreOp::create(
        b, loc, bitcast(b, loc, buffer, cir::PointerType::get(elemTy)), data);
    cir::BrOp::create(b, loc, store);

    // regrow:
    // *data = realloc(*data, bytes)
    // goto store;
    b.setInsertionPointToStart(regrow);
    Value grown =
        cir::CallOp::create(b, loc, SymbolRefAttr::get(*realloc), voidPtr,
                            ValueRange{bitcast(b, loc, old, voidPtr), bytes})
            .getResult();
    cir::StoreOp::create(
        b, loc, bitcast(b, loc, grown, cir::PointerType::get(elemTy)), data);
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
    SmallVector<Type, 4> slots = slotTypes(elemTy);
    slots.pop_back();
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

  // free_each(data, size, capacity) frees each value in a cache of pointers
  // as `dealloc` says: by calling the function it names, and passing it the
  // alignment after a `:`, for an aligned operator delete.
  FailureOr<cir::FuncOp> freeEachFunction(Location loc, Type elemTy,
                                          StringRef dealloc) {
    if (auto fn = freeEachFns.lookup(dealloc))
      return fn;
    MLIRContext *ctx = module.getContext();
    auto [name, alignText] = dealloc.split(':');
    uint64_t alignment = 0;
    if (!alignText.empty() && alignText.getAsInteger(10, alignment))
      return emitError(loc)
             << "cannot lower enzyme caches: bad alignment in " << dealloc;
    SmallVector<Type, 2> params{elemTy};
    if (!alignText.empty())
      params.push_back(sizeType(ctx));
    // An aligned operator delete the module declares already takes the
    // alignment as whatever type it declares.
    auto declared = symbols.lookup<cir::FuncOp>(name);
    FailureOr<cir::FuncOp> deallocFn =
        declared && !alignText.empty()
            ? FailureOr<cir::FuncOp>(declared)
            : libraryFunction(
                  loc, name,
                  cir::FuncType::get(params, cir::VoidType::get(ctx)));
    if (failed(deallocFn))
      return failure();
    SmallVector<Type, 4> slots = slotTypes(elemTy);
    slots.pop_back();
    auto fn = buildFunctionInModule(
        loc, "__enzyme_cir_free_each",
        cir::FuncType::get(slots, cir::VoidType::get(ctx)));
    Block *entry = fn.addEntryBlock();
    Block *cond = new Block(), *body = new Block(), *done = new Block();
    fn.getBody().push_back(cond);
    fn.getBody().push_back(body);
    fn.getBody().push_back(done);
    Value data = entry->getArgument(0), size = entry->getArgument(1);

    // for (i = 0; i < size; i++) dealloc(data[i]);
    OpBuilder b = OpBuilder::atBlockBegin(entry);
    Value i = alloca(b, loc, sizeType(ctx), "i", DataLayout::closest(fn));
    cir::StoreOp::create(b, loc, constSize(b, loc, 0), i);
    cir::BrOp::create(b, loc, cond);

    b.setInsertionPointToStart(cond);
    Value iVal = load(b, loc, i);
    Value more = cir::CmpOp::create(b, loc, cir::CmpOpKind::lt, iVal,
                                    load(b, loc, size));
    cir::BrCondOp::create(b, loc, more, body, done);

    b.setInsertionPointToStart(body);
    Value slot = cir::PtrStrideOp::create(b, loc, cir::PointerType::get(elemTy),
                                          load(b, loc, data), iVal);
    SmallVector<Value, 2> args{load(b, loc, slot)};
    if (!alignText.empty()) {
      auto alignTy = dyn_cast<cir::IntType>(
          deallocFn->getFunctionType().getInputs().back());
      if (!alignTy)
        return emitError(loc) << "cannot lower enzyme caches: @" << name
                              << " does not take its alignment as an integer";
      args.push_back(cir::ConstantOp::create(
          b, loc, cir::IntAttr::get(alignTy, alignment)));
    }
    cir::CallOp::create(b, loc, SymbolRefAttr::get(*deallocFn), Type(), args);
    cir::StoreOp::create(
        b, loc, cir::AddOp::create(b, loc, iVal, constSize(b, loc, 1)), i);
    cir::BrOp::create(b, loc, cond);

    b.setInsertionPointToStart(done);
    cir::ReturnOp::create(b, loc, ValueRange());

    freeEachFns[dealloc] = fn;
    return fn;
  }

  FailureOr<cir::FuncOp> freeFunction(Location loc) {
    MLIRContext *ctx = module.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    return libraryFunction(
        loc, "free", cir::FuncType::get({voidPtr}, cir::VoidType::get(ctx)));
  }

  // release(data, inline) frees the buffer, unless it is the inline one.
  FailureOr<cir::FuncOp> releaseFunction(Location loc, Type elemTy) {
    if (auto fn = releaseFns.lookup(elemTy))
      return fn;
    MLIRContext *ctx = module.getContext();
    auto voidPtr = cir::PointerType::get(cir::VoidType::get(ctx));
    FailureOr<cir::FuncOp> free = freeFunction(loc);
    if (failed(free))
      return failure();
    SmallVector<Type, 4> slots = slotTypes(elemTy);
    auto fn = buildFunctionInModule(
        loc, "__enzyme_cir_release",
        cir::FuncType::get({slots[0], slots[3]}, cir::VoidType::get(ctx)));
    Block *entry = fn.addEntryBlock();
    Block *release = new Block(), *done = new Block();
    fn.getBody().push_back(release);
    fn.getBody().push_back(done);
    OpBuilder b = OpBuilder::atBlockBegin(entry);
    Value data = load(b, loc, entry->getArgument(0));
    Value onHeap = cir::CmpOp::create(b, loc, cir::CmpOpKind::ne, data,
                                      entry->getArgument(1));
    cir::BrCondOp::create(b, loc, onHeap, release, done);
    b.setInsertionPointToStart(release);
    cir::CallOp::create(b, loc, SymbolRefAttr::get(*free), Type(),
                        ValueRange{bitcast(b, loc, data, voidPtr)});
    cir::BrOp::create(b, loc, done);
    b.setInsertionPointToStart(done);
    cir::ReturnOp::create(b, loc, ValueRange());
    releaseFns[elemTy] = fn;
    return fn;
  }

  // What a cache is passed around as: where its buffer is, its size, its
  // capacity, and its inline buffer, which it starts in.
  static SmallVector<Type, 4> slotTypes(Type elemTy) {
    MLIRContext *ctx = elemTy.getContext();
    return {cir::PointerType::get(cir::PointerType::get(elemTy)),
            cir::PointerType::get(sizeType(ctx)),
            cir::PointerType::get(sizeType(ctx)),
            cir::PointerType::get(elemTy)};
  }

  // Maximum stack size for a cache.
  static constexpr uint64_t kInlineCapacity = 8;

private:

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
  DenseMap<Type, cir::FuncOp> pushFns, popFns, getFns, releaseFns;
  llvm::StringMap<cir::FuncOp> freeEachFns;
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

static constexpr StringLiteral cacheSlotNames[] = {
    "enzyme.cache.data", "enzyme.cache.size", "enzyme.cache.capacity",
    "enzyme.cache.inline"};

cir::ArrayType inlineBufferType(Type elemTy) {
  return cir::ArrayType::get(elemTy, CacheRuntime::kInlineCapacity);
}

// The first element of the inline buffer at `buffer`.
Value inlineBufferStart(OpBuilder &b, Location loc, Value buffer, Type elemTy) {
  return cir::CastOp::create(b, loc, cir::PointerType::get(elemTy),
                             cir::CastKind::array_to_ptrdecay, buffer);
}

cir::StructType cacheRecordType(Type elemTy) {
  MLIRContext *ctx = elemTy.getContext();
  SmallVector<Type, 4> slots = CacheRuntime::slotTypes(elemTy);
  SmallVector<Type, 4> members;
  for (Type slot : ArrayRef(slots).take_front(3))
    members.push_back(cast<cir::PointerType>(slot).getPointee());
  members.push_back(inlineBufferType(elemTy));
  SmallVector<cir::RecordMemberKind, 4> kinds(members.size(),
                                              cir::RecordMemberKind::Data);
  return cir::StructType::get(ctx, members, /*packed=*/false,
                              /*is_class=*/false, kinds);
}

// The address of each field of the cache record at `record`, and the start
// of its inline buffer.
SmallVector<Value, 4> recordSlots(OpBuilder &b, Location loc, Value record,
                                  Type elemTy) {
  SmallVector<Value, 4> slots;
  SmallVector<Type, 4> types = CacheRuntime::slotTypes(elemTy);
  for (unsigned i = 0; i < 3; ++i)
    slots.push_back(cir::GetMemberOp::create(b, loc, types[i], record,
                                             cacheSlotNames[i], i));
  Value buffer = cir::GetMemberOp::create(
      b, loc, cir::PointerType::get(inlineBufferType(elemTy)), record,
      cacheSlotNames[3], 3);
  slots.push_back(inlineBufferStart(b, loc, buffer, elemTy));
  return slots;
}

// Lowers what is done with `cache`, whose fields are at `slots`: its pushes
// and pops, and its conversions to `handle`, the pointer a call is passed.
LogicalResult lowerCacheUses(Value cache, Type elemTy, ArrayRef<Value> slots,
                             Value handle, CacheRuntime &runtime) {
  OpBuilder b(cache.getContext());
  for (Operation *user : llvm::make_early_inc_range(cache.getUsers())) {
    b.setInsertionPoint(user);
    Location loc = user->getLoc();
    if (auto push = dyn_cast<enzyme::PushOp>(user)) {
      FailureOr<cir::FuncOp> pushFn = runtime.pushFunction(loc, elemTy);
      if (failed(pushFn))
        return failure();
      SmallVector<Value, 4> args(slots.begin(), slots.end());
      args.push_back(push.getValue());
      cir::CallOp::create(b, loc, SymbolRefAttr::get(*pushFn), Type(), args);
    } else if (isa<enzyme::PopOp, enzyme::GetOp>(user)) {
      cir::FuncOp popFn =
          runtime.popFunction(loc, elemTy, /*remove=*/isa<enzyme::PopOp>(user));
      Value v = cir::CallOp::create(b, loc, SymbolRefAttr::get(popFn), elemTy,
                                    slots.take_front(3))
                    .getResult();
      user->getResult(0).replaceAllUsesWith(v);
    } else if (isa<UnrealizedConversionCastOp>(user) && handle) {
      user->getResult(0).replaceAllUsesWith(handle);
    } else {
      return user->emitError() << "cannot lower this use of an enzyme cache";
    }
    user->erase();
  }
  return success();
}

// Enumlate the enzyme.cache type with stack push and pop
LogicalResult lowerCache(enzyme::InitOp init, cir::FuncOp fn,
                         const DataLayout &dl, CacheRuntime &runtime) {
  Location loc = init.getLoc();
  Type elemTy = cast<enzyme::CacheType>(init.getType()).getType();
  SmallVector<Type, 4> slotTypes = CacheRuntime::slotTypes(elemTy);
  bool passed = llvm::any_of(init->getUsers(), [](Operation *user) {
    return isa<UnrealizedConversionCastOp>(user);
  });

  OpBuilder b = OpBuilder::atBlockBegin(&fn.getBody().front());
  SmallVector<Value, 4> slots;
  Value handle;
  if (passed) {
    Value record = alloca(b, loc, cacheRecordType(elemTy), "enzyme.cache", dl);
    slots = recordSlots(b, loc, record, elemTy);
    handle =
        bitcast(b, loc, record,
                cir::PointerType::get(cir::VoidType::get(fn.getContext())));
  } else {
    for (unsigned i = 0; i < 3; ++i)
      slots.push_back(alloca(b, loc,
                             cast<cir::PointerType>(slotTypes[i]).getPointee(),
                             cacheSlotNames[i], dl));
    Value buffer =
        alloca(b, loc, inlineBufferType(elemTy), cacheSlotNames[3], dl);
    slots.push_back(inlineBufferStart(b, loc, buffer, elemTy));
  }

  // An empty cache, in its inline buffer.
  b.setInsertionPoint(init);
  cir::StoreOp::create(b, loc, slots[3], slots[0]);
  cir::StoreOp::create(b, loc, constSize(b, loc, 0), slots[1]);
  cir::StoreOp::create(b, loc, constSize(b, loc, CacheRuntime::kInlineCapacity),
                       slots[2]);

  // A list of deferred frees: its owner frees what is on it, then the list.
  cir::FuncOp freeEach;
  if (auto dealloc = init->getAttrOfType<StringAttr>("enzyme.deferred_free")) {
    FailureOr<cir::FuncOp> fe =
        runtime.freeEachFunction(loc, elemTy, dealloc.getValue());
    if (failed(fe))
      return failure();
    freeEach = *fe;
  }

  if (failed(lowerCacheUses(init.getResult(), elemTy, slots, handle, runtime)))
    return failure();
  init.erase();

  FailureOr<cir::FuncOp> releaseFn = runtime.releaseFunction(loc, elemTy);
  if (failed(releaseFn))
    return failure();
  fn.walk([&](cir::ReturnOp ret) {
    OpBuilder rb(ret);
    if (freeEach)
      cir::CallOp::create(rb, loc, SymbolRefAttr::get(freeEach), Type(),
                          ArrayRef(slots).take_front(3));
    cir::CallOp::create(rb, loc, SymbolRefAttr::get(*releaseFn), Type(),
                        ValueRange{slots[0], slots[3]});
  });
  return success();
}

// A cache the function was passed, as a pointer to its record; its owner
// frees it.
LogicalResult lowerPassedCache(UnrealizedConversionCastOp cast,
                               CacheRuntime &runtime) {
  Location loc = cast.getLoc();
  Value handle = cast.getOperand(0);
  Type elemTy =
      llvm::cast<enzyme::CacheType>(cast.getResult(0).getType()).getType();
  OpBuilder b(cast);
  Value record =
      bitcast(b, loc, handle, cir::PointerType::get(cacheRecordType(elemTy)));
  SmallVector<Value, 4> slots = recordSlots(b, loc, record, elemTy);
  if (failed(lowerCacheUses(cast.getResult(0), elemTy, slots, handle, runtime)))
    return failure();
  cast.erase();
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
      SmallVector<UnrealizedConversionCastOp> passed;
      fn.walk([&](UnrealizedConversionCastOp cast) {
        if (cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
            isa<cir::PointerType>(cast.getOperand(0).getType()) &&
            isa<enzyme::CacheType>(cast.getResult(0).getType()))
          passed.push_back(cast);
      });
      for (UnrealizedConversionCastOp cast : passed)
        if (failed(lowerPassedCache(cast, runtime)))
          return signalPassFailure();
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
