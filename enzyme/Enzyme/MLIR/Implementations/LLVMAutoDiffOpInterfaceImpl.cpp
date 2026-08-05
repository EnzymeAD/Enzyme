//===- LLVMAutoDiffOpInterfaceImpl.cpp - Interface external model  --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the external model implementation of the automatic
// differentiation op interfaces for the upstream LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "Dialect/LLVMExt/LLVMExt.h"
#include "Implementations/CoreDialectsAutoDiffImplementations.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {
#include "Implementations/LLVMDerivatives.inc"

// Lets activity analysis treat llvm.store generically via StoreLikeInterface.
struct LLVMStoreLike
    : public StoreLikeInterface::ExternalModel<LLVMStoreLike, LLVM::StoreOp> {
  Value getStoredValue(Operation *op) const {
    return cast<LLVM::StoreOp>(op).getValue();
  }
  Value getStoredPointer(Operation *op) const {
    return cast<LLVM::StoreOp>(op).getAddr();
  }
};

struct InlineAsmActivityInterface
    : public ActivityOpInterface::ExternalModel<InlineAsmActivityInterface,
                                                LLVM::InlineAsmOp> {
  bool isInactive(Operation *op) const {
    auto asmOp = cast<LLVM::InlineAsmOp>(op);
    auto str = asmOp.getAsmString();
    return str.contains("cpuid") || str.contains("exit");
  }
  bool isArgInactive(Operation *op, size_t) const { return isInactive(op); }
};

struct SelectActivityInterface
    : public ActivityOpInterface::ExternalModel<SelectActivityInterface,
                                                LLVM::SelectOp> {
  bool isInactive(Operation *op) const { return false; }
  bool isArgInactive(Operation *op, size_t idx) const {
    // llvm.select is not inactive in general, but the condition is always
    // inactive.
    return idx == 0;
  }
};

class ArrayTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<ArrayTypeInterface,
                                                  LLVM::LLVMArrayType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm::errs() << " unsupported: createNullAttribute of LLVMArrayType\n";
    return nullptr;
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    auto arrayTy = cast<LLVM::LLVMArrayType>(self);
    Type elemTy = arrayTy.getElementType();
    auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
    if (!elemIface)
      return LLVM::ZeroOp::create(builder, loc, arrayTy);

    Value nullElem = elemIface.createNullValue(builder, loc);
    Value result = LLVM::PoisonOp::create(builder, loc, arrayTy);
    for (uint64_t i = 0; i < arrayTy.getNumElements(); ++i) {
      result = LLVM::InsertValueOp::create(builder, loc, result, nullElem, i);
    }
    return result;
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto arrayTy = cast<LLVM::LLVMArrayType>(self);
    Type elemTy = arrayTy.getElementType();
    Value result = LLVM::PoisonOp::create(builder, loc, arrayTy);
    auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
    for (uint64_t i = 0; i < arrayTy.getNumElements(); ++i) {
      Value aElem = LLVM::ExtractValueOp::create(builder, loc, a, i);
      Value bElem = LLVM::ExtractValueOp::create(builder, loc, b, i);
      Value sum;
      if (elemIface) {
        sum = elemIface.createAddOp(builder, loc, aElem, bElem);
      } else {
        sum = aElem;
      }
      result = LLVM::InsertValueOp::create(builder, loc, result, sum, i);
    }
    return result;
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }

  int64_t getApproxSize(Type self) const {
    auto arrayType = cast<LLVM::LLVMArrayType>(self);
    if (auto elemType =
            dyn_cast<AutoDiffTypeInterface>(arrayType.getElementType())) {
      int64_t elSize = elemType.getApproxSize();
      if (elSize != INT64_MAX)
        return arrayType.getNumElements() * elSize;
    }
    return INT64_MAX;
  }
};

class PointerTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<PointerTypeInterface,
                                                  LLVM::LLVMPointerType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm::errs() << " unsupported: createNullAttribute of pointertype\n";
    return nullptr;
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    return LLVM::ZeroOp::create(builder, loc, self);
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    llvm_unreachable("TODO");
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return true; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    if (auto allocaOp = val.getDefiningOp<LLVM::AllocaOp>()) {
      Value zero =
          LLVM::ConstantOp::create(builder, loc, builder.getI8IntegerAttr(0));
      auto elemType = cast<AutoDiffTypeInterface>(allocaOp.getElemType());
      unsigned byteSize = elemType.getApproxSize() / 8;
      Value byteValue = LLVM::ConstantOp::create(
          builder, loc, builder.getI64IntegerAttr(byteSize));
      Value arraySize = LLVM::SExtOp::create(builder, loc, byteValue.getType(),
                                             allocaOp.getArraySize());
      Value size = LLVM::MulOp::create(builder, loc, arraySize, byteValue);
      LLVM::MemsetOp::create(builder, loc, val, zero, size,
                             /*isVolatile=*/false);
      return success();
    }
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

struct GEPOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<GEPOpInterfaceReverse,
                                                       LLVM::GEPOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto gep = cast<LLVM::GEPOp>(op);
    auto newGep = cast<LLVM::GEPOp>(gutils->getNewFromOriginal(op));
    auto base = gep.getBase();
    if (!gutils->isConstantValue(base)) {
      auto baseShadow = gutils->invertPointerM(base, builder);
      auto shadowGep = cast<LLVM::GEPOp>(builder.clone(*newGep));
      shadowGep.getBaseMutable().assign(baseShadow);
      gutils->setInvertedPointer(gep.getRes(), shadowGep->getResult(0));
    }
  }
};

// Renaming a pointer into another address space moves nothing, so there is no
// adjoint to accumulate; the shadow is the same rename of the shadow pointer,
// as for llvm.getelementptr above. Needed because these sit in front of active
// memory: raise-llvm-ext emits one for a size hint that annotates the memory
// space of a pointer whose type cannot name it.
struct AddrSpaceCastOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          AddrSpaceCastOpInterfaceReverse, LLVM::AddrSpaceCastOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    auto asCast = cast<LLVM::AddrSpaceCastOp>(op);
    auto arg = asCast.getArg();
    if (gutils->isConstantValue(arg))
      return;

    auto newCast = cast<LLVM::AddrSpaceCastOp>(gutils->getNewFromOriginal(op));
    auto shadowCast = cast<LLVM::AddrSpaceCastOp>(builder.clone(*newCast));
    shadowCast.getArgMutable().assign(gutils->invertPointerM(arg, builder));
    gutils->setInvertedPointer(asCast.getRes(), shadowCast.getRes());
  }
};

struct LoadOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<LoadOpInterfaceReverse,
                                                       LLVM::LoadOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    Value addr = loadOp.getAddr();

    if (auto iface = dyn_cast<AutoDiffTypeInterface>(loadOp.getType())) {
      // A mutable type's derivative is a shadow, not a value to add into: there
      // is nothing to accumulate here, the same way the store adjoint below has
      // nothing to take back out. Loading a pointer is the case in hand.
      if (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr) &&
          !iface.isMutable()) {
        Value gradient = gutils->diffe(loadOp, builder);
        Value addrGradient = gutils->popCache(caches.front(), builder);

        if (!gutils->AtomicAdd) {
          Value loadedGradient = LLVM::LoadOp::create(builder, loadOp.getLoc(),
                                                      iface, addrGradient);
          Value addedGradient = iface.createAddOp(builder, loadOp.getLoc(),
                                                  loadedGradient, gradient);

          LLVM::StoreOp::create(builder, loadOp.getLoc(), addedGradient,
                                addrGradient);
        } else {
          LLVM::AtomicRMWOp::create(builder, loadOp.getLoc(),
                                    LLVM::AtomicBinOp::fadd, addrGradient,
                                    gradient, LLVM::AtomicOrdering::monotonic);
        }
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto loadOp = cast<LLVM::LoadOp>(op);
    auto addr = loadOp.getAddr();
    if (!(isa<AutoDiffTypeInterface>(loadOp.getType()) &&
          (!gutils->isConstantValue(loadOp) && !gutils->isConstantValue(addr))))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {
    // auto loadOp = cast<LLVM::LoadOp>(op);
    // Value ptr = loadOp.getAddr();
  }
};

struct StoreOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<StoreOpInterfaceReverse,
                                                       LLVM::StoreOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    Value val = storeOp.getValue();
    Value addr = storeOp.getAddr();

    auto iface = cast<AutoDiffTypeInterface>(val.getType());

    if (!gutils->isConstantValue(addr)) {
      Value addrGradient = gutils->popCache(caches.front(), builder);

      if (!iface.isMutable()) {
        if (!gutils->isConstantValue(val)) {
          Value loadedGradient = LLVM::LoadOp::create(
              builder, storeOp.getLoc(), val.getType(), addrGradient);
          gutils->addToDiffe(val, loadedGradient, builder);
        }

        auto zero =
            cast<AutoDiffTypeInterface>(gutils->getShadowType(val.getType()))
                .createNullValue(builder, op->getLoc());

        LLVM::StoreOp::create(builder, storeOp.getLoc(), zero, addrGradient);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    auto storeOp = cast<LLVM::StoreOp>(op);
    auto addr = storeOp.getAddr();
    if (gutils->isConstantValue(addr))
      return {};
    OpBuilder cacheBuilder(gutils->getNewFromOriginal(op));
    return {gutils->initAndPushCache(gutils->invertPointerM(addr, cacheBuilder),
                                     cacheBuilder)};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct ExtractValueOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          ExtractValueOpInterfaceReverse, LLVM::ExtractValueOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto evOp = cast<LLVM::ExtractValueOp>(op);
    Value container = evOp.getContainer();

    auto containerIface = dyn_cast<AutoDiffTypeInterface>(container.getType());
    if (!containerIface)
      return failure();

    if (!gutils->isConstantValue(evOp)) {
      Value gradient = gutils->diffe(evOp, builder);
      gutils->zeroDiffe(evOp, builder);
      // Create a zero aggregate matching the container type, then insert the
      // gradient at the extracted position.
      if (!gutils->isConstantValue(container)) {
        Value zero = containerIface.createNullValue(builder, op->getLoc());
        Value grad = LLVM::InsertValueOp::create(builder, op->getLoc(), zero,
                                                 gradient, evOp.getPosition());
        gutils->addToDiffe(container, grad, builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

struct InsertValueOpInterfaceReverse
    : public ReverseAutoDiffOpInterface::ExternalModel<
          InsertValueOpInterfaceReverse, LLVM::InsertValueOp> {
  LogicalResult createReverseModeAdjoint(Operation *op, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto ivOp = cast<LLVM::InsertValueOp>(op);
    Value container = ivOp.getContainer();
    Value value = ivOp.getValue();

    auto resultIface = dyn_cast<AutoDiffTypeInterface>(ivOp.getType());
    if (!resultIface)
      return failure();

    if (!gutils->isConstantValue(ivOp)) {
      Value gradient = gutils->diffe(ivOp, builder);
      gutils->zeroDiffe(ivOp, builder);

      // Propagate gradient to the inserted value: extract from the result
      // gradient at the insertion position.
      auto valIface = dyn_cast<AutoDiffTypeInterface>(value.getType());
      if (valIface && !gutils->isConstantValue(value)) {
        Value valGrad = LLVM::ExtractValueOp::create(
            builder, op->getLoc(), gradient, ivOp.getPosition());
        gutils->addToDiffe(value, valGrad, builder);
      }

      // Propagate gradient to the container: zero out the inserted position
      // in the result gradient, then add to the container gradient.
      if (!gutils->isConstantValue(container)) {
        Value zeroVal =
            valIface
                ? valIface.createNullValue(builder, op->getLoc())
                : LLVM::ZeroOp::create(builder, op->getLoc(), value.getType());
        Value containerGrad = LLVM::InsertValueOp::create(
            builder, op->getLoc(), gradient, zeroVal, ivOp.getPosition());
        gutils->addToDiffe(container, containerGrad, builder);
      }
    }

    return success();
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    return {};
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}
};

// What is recorded about the allocation a pointer names: how many bytes are
// reachable from it, and the pointer that says so. The second is not redundant.
// A clone is allocated and copied in the address space of the pointer it is
// made from -- which is what decides between host malloc/memcpy and the device
// runtime -- and the pointer carrying the extent is not always the one asked
// about: a size hint annotated with a memory space is attached to an
// llvm.addrspacecast of it (see raise-llvm-ext), which is how memory a frontend
// could only type as a host pointer still gets cloned on the device.
struct PtrExtent {
  Value size; // null when nothing records the extent
  Value ptr;  // pointer the extent was found on; the queried one by default
};

static PtrExtent findPtrExtent(Value ptr) {
  if (auto allocOp = ptr.getDefiningOp<llvm_ext::AllocOp>())
    return {allocOp.getSize(), ptr};

  for (auto user : ptr.getUsers())
    if (auto psh = dyn_cast<llvm_ext::PtrSizeHintOp>(user))
      return {psh.getSize(), ptr};

  // A hint on a cast of `ptr` describes the same bytes, in the space the cast
  // names. Looked for after the direct hints, so an unannotated one still wins.
  for (auto user : ptr.getUsers()) {
    auto cast = dyn_cast<LLVM::AddrSpaceCastOp>(user);
    if (!cast)
      continue;
    for (auto castUser : cast.getRes().getUsers())
      if (auto psh = dyn_cast<llvm_ext::PtrSizeHintOp>(castUser))
        return {psh.getSize(), cast.getRes()};
  }

  return {nullptr, ptr};
}

// Names `ptr` in the address space `inSpaceOf` is in, so that both ends of a
// copy agree on the space it is made in. Emits nothing when they already do.
static Value castLikeSpaceOf(OpBuilder &builder, Value ptr, Value inSpaceOf) {
  if (ptr.getType() == inSpaceOf.getType())
    return ptr;
  return LLVM::AddrSpaceCastOp::create(builder, ptr.getLoc(),
                                       inSpaceOf.getType(), ptr);
}

// llvm_ext.ptr_size_hint accepts AnyInteger, but llvm_ext.alloc and
// llvm_ext.memcpy require an i64 size, so a narrower (or wider) hint has to be
// converted rather than forwarded -- otherwise we build invalid IR that only
// fails later, in the verifier. Returns null if the size is not an integer.
static Value normalizeSizeToI64(OpBuilder &builder, Location loc, Value size) {
  auto i64Ty = builder.getIntegerType(64);
  if (size.getType() == i64Ty)
    return size;

  auto srcTy = dyn_cast<IntegerType>(size.getType());
  if (!srcTy) {
    llvm::errs() << "ptr size hint is not an integer: " << size << "\n";
    return nullptr;
  }
  // Sizes are non-negative, so zero-extend when widening.
  if (srcTy.getWidth() < 64)
    return LLVM::ZExtOp::create(builder, loc, i64Ty, size);
  return LLVM::TruncOp::create(builder, loc, i64Ty, size);
}

struct PointerClonableTypeInterface
    : public ClonableTypeInterface::ExternalModel<PointerClonableTypeInterface,
                                                  LLVM::LLVMPointerType> {
  mlir::Value cloneValue(Type self, OpBuilder &builder, Value value) const {
    PtrExtent extent = findPtrExtent(value);
    if (!extent.size) {
      llvm::errs() << "cannot find size of ptr: " << value << "\n";
      return nullptr;
    }

    Value size = normalizeSizeToI64(builder, value.getLoc(), extent.size);
    if (!size)
      return nullptr;

    // Clone from the pointer that carries the extent, so the allocation and the
    // copy land in the memory space that pointer names, and picking the
    // allocator and the copy is left to lower-llvm-ext: emitting host
    // malloc/memcpy here would silently produce a host read of device memory (a
    // segfault at runtime) whenever what is being checkpointed is device
    // memory.
    Value clone = llvm_ext::AllocOp::create(builder, value.getLoc(),
                                            extent.ptr.getType(), size);
    llvm_ext::MemcpyOp::create(builder, value.getLoc(), clone, extent.ptr,
                               size);

    return clone;
  }

  void copyValue(Type self, OpBuilder &builder, Value dst, Value src) const {
    // Prefer `src`: taking a checkpoint copies from the live ref, whose alloc
    // (or size hint) is visible. Restoring one copies *out of* a clone buffer,
    // where only `dst` -- a direct clone -- is traceable.
    PtrExtent extent = findPtrExtent(src);
    if (!extent.size)
      extent = findPtrExtent(dst);
    if (!extent.size) {
      llvm::errs() << "cannot find size of ptr to copy: " << src << "\n";
      return;
    }

    Value size = normalizeSizeToI64(builder, src.getLoc(), extent.size);
    if (!size)
      return;

    Value dstInSpace = castLikeSpaceOf(builder, dst, extent.ptr);
    Value srcInSpace = castLikeSpaceOf(builder, src, extent.ptr);
    llvm_ext::MemcpyOp::create(builder, src.getLoc(), dstInSpace, srcInSpace,
                               size);
  }

  void freeClonedValue(Type self, OpBuilder &builder, Value value) const {
    llvm_ext::FreeOp::create(builder, value.getLoc(), value);
  }
};

class StructTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<StructTypeInterface,
                                                  LLVM::LLVMStructType> {
public:
  mlir::Attribute createNullAttr(mlir::Type self) const {
    llvm::errs() << " unsupported: createNullAttribute of LLVMStructType\n";
    return nullptr;
  }

  mlir::Value createNullValue(mlir::Type self, OpBuilder &builder,
                              Location loc) const {
    auto structTy = cast<LLVM::LLVMStructType>(self);
    Value result = LLVM::PoisonOp::create(builder, loc, structTy);
    for (auto &&[i, elemTy] : llvm::enumerate(structTy.getBody())) {
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      if (!elemIface) {
        Value zero = LLVM::ZeroOp::create(builder, loc, elemTy);
        result = LLVM::InsertValueOp::create(builder, loc, result, zero, i);
        continue;
      }
      Value nullElem = elemIface.createNullValue(builder, loc);
      result = LLVM::InsertValueOp::create(builder, loc, result, nullElem, i);
    }
    return result;
  }

  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto structTy = cast<LLVM::LLVMStructType>(self);
    Value result = LLVM::PoisonOp::create(builder, loc, structTy);
    for (auto &&[i, elemTy] : llvm::enumerate(structTy.getBody())) {
      Value aElem = LLVM::ExtractValueOp::create(builder, loc, a, i);
      Value bElem = LLVM::ExtractValueOp::create(builder, loc, b, i);
      auto elemIface = dyn_cast<AutoDiffTypeInterface>(elemTy);
      Value sum;
      if (elemIface) {
        sum = elemIface.createAddOp(builder, loc, aElem, bElem);
      } else {
        sum = aElem;
      }
      result = LLVM::InsertValueOp::create(builder, loc, result, sum, i);
    }
    return result;
  }

  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    llvm_unreachable("TODO");
  }

  Type getShadowType(Type self, unsigned width) const {
    assert(width == 1 && "unsupported width != 1");
    return self;
  }

  bool isMutable(Type self) const { return false; }

  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }

  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

static Value packIntoStruct(ValueRange values, OpBuilder &builder,
                            Location loc) {
  SmallVector<Type> resultTypes =
      llvm::map_to_vector(values, [](Value v) { return v.getType(); });
  auto structType =
      LLVM::LLVMStructType::getLiteral(builder.getContext(), resultTypes);
  Value result = LLVM::PoisonOp::create(builder, loc, structType);
  for (auto &&[i, v] : llvm::enumerate(values))
    result = LLVM::InsertValueOp::create(builder, loc, result, v, i);

  return result;
}

class AutoDiffLLVMFuncOpFunctionInterface
    : public AutoDiffFunctionInterface::ExternalModel<
          AutoDiffLLVMFuncOpFunctionInterface, LLVM::LLVMFuncOp> {
public:
  void transformResultTypes(Operation *self,
                            SmallVectorImpl<Type> &resultTypes) const {
    auto fn = cast<mlir::FunctionOpInterface>(self);
    auto FTy = fn.getFunctionType();
    if (resultTypes.empty()) {
      // llvm.func ops that return no results need to explicitly return
      // LLVMVoidType
      resultTypes.push_back(LLVM::LLVMVoidType::get(FTy.getContext()));
    } else if (resultTypes.size() > 1) {
      auto structType =
          LLVM::LLVMStructType::getLiteral(FTy.getContext(), resultTypes);
      resultTypes.clear();
      resultTypes.push_back(structType);
    }
  }

  Operation *createCall(Operation *self, OpBuilder &builder, Location loc,
                        ValueRange args) const {
    return LLVM::CallOp::create(builder, loc, cast<LLVM::LLVMFuncOp>(self),
                                args);
  }

  Operation *createReturn(Operation *self, OpBuilder &builder, Location loc,
                          ValueRange retargs) const {
    if (retargs.size() > 1) {
      Value packedReturns = packIntoStruct(retargs, builder, loc);
      return LLVM::ReturnOp::create(builder, loc, packedReturns);
    }

    return LLVM::ReturnOp::create(builder, loc, retargs);
  }
};

} // namespace

void mlir::enzyme::registerLLVMDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, LLVM::LLVMDialect *) {
    registerInterfaces(context);
    LLVM::LLVMPointerType::attachInterface<PointerTypeInterface>(*context);
    LLVM::LLVMPointerType::attachInterface<PointerClonableTypeInterface>(
        *context);
    LLVM::LLVMStructType::attachInterface<StructTypeInterface>(*context);
    LLVM::LLVMArrayType::attachInterface<ArrayTypeInterface>(*context);

    LLVM::SelectOp::attachInterface<SelectActivityInterface>(*context);
    LLVM::StoreOp::attachInterface<LLVMStoreLike>(*context);
    LLVM::LoadOp::attachInterface<LoadOpInterfaceReverse>(*context);
    LLVM::StoreOp::attachInterface<StoreOpInterfaceReverse>(*context);
    LLVM::GEPOp::attachInterface<GEPOpInterfaceReverse>(*context);
    LLVM::AddrSpaceCastOp::attachInterface<AddrSpaceCastOpInterfaceReverse>(
        *context);
    LLVM::ExtractValueOp::attachInterface<ExtractValueOpInterfaceReverse>(
        *context);
    LLVM::InsertValueOp::attachInterface<InsertValueOpInterfaceReverse>(
        *context);
    LLVM::UnreachableOp::template attachInterface<
        detail::NoopRevAutoDiffInterface<LLVM::UnreachableOp>>(*context);
    LLVM::LLVMFuncOp::attachInterface<AutoDiffLLVMFuncOpFunctionInterface>(
        *context);
  });
}
