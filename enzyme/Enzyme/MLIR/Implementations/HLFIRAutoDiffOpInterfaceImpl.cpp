//===- HLFIRAutoDiffOpInterfaceImpl.cpp - HLFIR external models -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Enzyme autodiff interface external models for Flang's HLFIR array intrinsics,
// differentiated while they are still first-class hlfir.* ops (Tier 1 of
// PLAN_flang_enzyme_mlir.md). This first slice implements:
//
//   * AutoDiffTypeInterface for !hlfir.expr (shadow type, elementwise add,
//     conjugate) so hlfir-valued tangents/adjoints can be carried and summed.
//   * Forward-mode tangent for hlfir.matmul:
//       d(A*B) = dA*B + A*dB
//
// Reverse mode and the remaining intrinsics (sum, dot_product, ...) follow.
//
//===----------------------------------------------------------------------===//

#include "Implementations/HLFIRAutoDiffOpInterfaceImpl.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"

#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::enzyme;

namespace {

// Wrap a plain OpBuilder as a fir::FirOpBuilder rooted at the enclosing module
// (needed for the HLFIR element-access / elemental helpers).
static fir::FirOpBuilder getFirBuilder(OpBuilder &builder) {
  Operation *scope = builder.getInsertionBlock()->getParentOp();
  return fir::FirOpBuilder(builder, scope);
}

//===----------------------------------------------------------------------===//
// AutoDiffTypeInterface for !hlfir.expr
//===----------------------------------------------------------------------===//

// Build an hlfir.expr with the same shape as `sample` by evaluating `elementFn`
// (which returns the scalar element value) at every index.
static Value genExprElementwise(
    OpBuilder &builder, Location loc, hlfir::ExprType exprTy, Value sample,
    llvm::function_ref<Value(Location, fir::FirOpBuilder &, ValueRange)>
        elementFn) {
  fir::FirOpBuilder fbuilder = getFirBuilder(builder);
  Value shape = hlfir::genShape(loc, fbuilder, hlfir::Entity{sample});
  Type eleTy = exprTy.getElementType();
  auto kernel = [&](Location l, fir::FirOpBuilder &b,
                    ValueRange idx) -> hlfir::Entity {
    return hlfir::Entity{elementFn(l, b, idx)};
  };
  hlfir::ElementalOp elemental = hlfir::genElementalOp(
      loc, fbuilder, eleTy, shape, /*typeParams=*/{}, kernel,
      /*isUnordered=*/true, /*polymorphicMold=*/{}, /*exprType=*/exprTy);
  return elemental.getResult();
}

struct ExprTypeInterface
    : public AutoDiffTypeInterface::ExternalModel<ExprTypeInterface,
                                                  hlfir::ExprType> {
  // The adjoint of an hlfir.expr is another hlfir.expr of the same type (with a
  // leading batch dimension when width > 1).
  Type getShadowType(Type self, int64_t width) const {
    auto e = cast<hlfir::ExprType>(self);
    if (width == 1)
      return self;
    llvm::SmallVector<int64_t> shape;
    shape.push_back(width);
    shape.append(e.getShape().begin(), e.getShape().end());
    return hlfir::ExprType::get(e.getContext(), shape, e.getElementType(),
                                e.isPolymorphic());
  }

  // Elementwise a + b, delegating the scalar add to the element type's own
  // AutoDiffTypeInterface (so float/complex are handled correctly).
  Value createAddOp(Type self, OpBuilder &builder, Location loc, Value a,
                    Value b) const {
    auto exprTy = cast<hlfir::ExprType>(self);
    auto eIface = cast<AutoDiffTypeInterface>(exprTy.getElementType());
    return genExprElementwise(
        builder, loc, exprTy, a,
        [&](Location l, fir::FirOpBuilder &fb, ValueRange idx) -> Value {
          Value ea = hlfir::loadElementAt(l, fb, hlfir::Entity{a}, idx);
          Value eb = hlfir::loadElementAt(l, fb, hlfir::Entity{b}, idx);
          return eIface.createAddOp(fb, l, ea, eb);
        });
  }

  // Real element types are self-conjugate; complex conjugates elementwise.
  Value createConjOp(Type self, OpBuilder &builder, Location loc,
                     Value a) const {
    auto exprTy = cast<hlfir::ExprType>(self);
    auto eIface = cast<AutoDiffTypeInterface>(exprTy.getElementType());
    if (!isa<ComplexType>(exprTy.getElementType()))
      return a;
    return genExprElementwise(
        builder, loc, exprTy, a,
        [&](Location l, fir::FirOpBuilder &fb, ValueRange idx) -> Value {
          Value ea = hlfir::loadElementAt(l, fb, hlfir::Entity{a}, idx);
          return eIface.createConjOp(fb, l, ea);
        });
  }

  bool isMutable(Type self) const { return false; }

  // A zero hlfir.expr: an elemental yielding the element type's null value. The
  // extents must be static, since the Type alone carries no runtime shape.
  Value createNullValue(Type self, OpBuilder &builder, Location loc) const {
    auto exprTy = cast<hlfir::ExprType>(self);
    auto eIface = cast<AutoDiffTypeInterface>(exprTy.getElementType());
    for (int64_t e : exprTy.getShape())
      if (e == hlfir::ExprType::getUnknownExtent()) {
        llvm::errs() << "hlfir.expr createNullValue needs static extents: "
                     << self << "\n";
        return {};
      }
    fir::FirOpBuilder fbuilder = getFirBuilder(builder);
    llvm::SmallVector<Value> extents;
    for (int64_t e : exprTy.getShape())
      extents.push_back(arith::ConstantIndexOp::create(fbuilder, loc, e));
    Value shape = fir::ShapeOp::create(fbuilder, loc, extents);
    auto kernel = [&](Location l, fir::FirOpBuilder &b,
                      ValueRange idx) -> hlfir::Entity {
      return hlfir::Entity{eIface.createNullValue(b, l)};
    };
    hlfir::ElementalOp elemental = hlfir::genElementalOp(
        loc, fbuilder, exprTy.getElementType(), shape, /*typeParams=*/{},
        kernel, /*isUnordered=*/true, /*polymorphicMold=*/{},
        /*exprType=*/exprTy);
    return elemental.getResult();
  }
  Attribute createNullAttr(Type self) const { return {}; }
  LogicalResult zeroInPlace(Type self, OpBuilder &builder, Location loc,
                            Value val) const {
    return failure();
  }
  bool isZero(Type self, Value val) const { return false; }
  bool isZeroAttr(Type self, Attribute attr) const { return false; }
};

//===----------------------------------------------------------------------===//
// Forward-mode tangent for hlfir.matmul
//===----------------------------------------------------------------------===//

struct MatmulOpFwd
    : public AutoDiffOpInterface::ExternalModel<MatmulOpFwd, hlfir::MatmulOp> {
  LogicalResult createForwardModeTangent(Operation *op0, OpBuilder &builder,
                                         MGradientUtils *gutils) const {
    auto op = cast<hlfir::MatmulOp>(op0);
    gutils->eraseIfUnused(op);
    if (gutils->isConstantInstruction(op))
      return success();

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Type resTy = op.getResult().getType();
    auto fastmath = op.getFastmathAttr();

    // d(A*B) = dA*B + A*dB, summing only the active terms.
    Value res = nullptr;
    if (!gutils->isConstantValue(lhs)) {
      Value dlhs = gutils->invertPointerM(lhs, builder);
      Value nrhs = gutils->getNewFromOriginal(rhs);
      res = hlfir::MatmulOp::create(builder, loc, resTy, dlhs, nrhs, fastmath);
    }
    if (!gutils->isConstantValue(rhs)) {
      Value nlhs = gutils->getNewFromOriginal(lhs);
      Value drhs = gutils->invertPointerM(rhs, builder);
      Value term =
          hlfir::MatmulOp::create(builder, loc, resTy, nlhs, drhs, fastmath);
      if (!res)
        res = term;
      else
        res = cast<AutoDiffTypeInterface>(resTy).createAddOp(builder, loc, res,
                                                             term);
    }

    if (res)
      gutils->setDiffe(op.getResult(), res, builder);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reverse-mode adjoint for hlfir.matmul
//===----------------------------------------------------------------------===//
//
//   C = A*B,  with incoming result adjoint Ḡ:
//     Ā += matmul(Ḡ, transpose(B))         (Ḡ·Bᵀ)
//     B̄ += matmul_transpose(A, Ḡ)          (Aᵀ·Ḡ)
//
// so the reverse pass needs the primal B (for Ā) and A (for B̄); these are
// cached in the forward pass.

// Transpose an [r,c] hlfir.expr type to [c,r].
static hlfir::ExprType transposedExprType(hlfir::ExprType t) {
  auto shape = t.getShape();
  assert(shape.size() == 2 && "matmul operands are rank-2");
  llvm::SmallVector<int64_t> tshape{shape[1], shape[0]};
  return hlfir::ExprType::get(t.getContext(), tshape, t.getElementType(),
                              t.isPolymorphic());
}

struct MatmulOpRev
    : public ReverseAutoDiffOpInterface::ExternalModel<MatmulOpRev,
                                                       hlfir::MatmulOp> {
  // A (operand 0) is needed iff B (operand 1) is active; B is needed iff A is.
  static llvm::SmallVector<bool> neededPrimals(Operation *op,
                                               MGradientUtilsReverse *gutils) {
    auto mm = cast<hlfir::MatmulOp>(op);
    llvm::SmallVector<bool> toret(op->getNumOperands(), false);
    if (gutils->isConstantValue(mm.getResult()))
      return toret;
    if (!gutils->isConstantValue(mm.getLhs()))
      toret[1] = true; // Ā = Ḡ·Bᵀ needs B (operand 1)
    if (!gutils->isConstantValue(mm.getRhs()))
      toret[0] = true; // B̄ = Aᵀ·Ḡ needs A (operand 0)
    return toret;
  }

  SmallVector<Value> cacheValues(Operation *op,
                                 MGradientUtilsReverse *gutils) const {
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op->getResult(0)))
      return {};
    auto needed = neededPrimals(op, gutils);
    SmallVector<Value> caches;
    OpBuilder builder(gutils->getNewFromOriginal(op));
    for (auto en : llvm::enumerate(needed))
      if (en.value())
        caches.push_back(gutils->initAndPushCache(
            gutils->getNewFromOriginal(op->getOperand(en.index())), builder));
    return caches;
  }

  void createShadowValues(Operation *op, OpBuilder &builder,
                          MGradientUtilsReverse *gutils) const {}

  LogicalResult createReverseModeAdjoint(Operation *op0, OpBuilder &builder,
                                         MGradientUtilsReverse *gutils,
                                         SmallVector<Value> caches) const {
    auto op = cast<hlfir::MatmulOp>(op0);
    if (gutils->isConstantInstruction(op) ||
        gutils->isConstantValue(op.getResult()))
      return success();

    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto fastmath = op.getFastmathAttr();

    // Incoming adjoint of the result, then reset it.
    Value gc = gutils->diffe(op.getResult(), builder);
    gutils->zeroDiffe(op.getResult(), builder);

    // Retrieve the cached primal operands.
    auto needed = neededPrimals(op, gutils);
    SmallVector<Value> primal(op->getNumOperands(), nullptr);
    size_t count = 0;
    for (auto en : llvm::enumerate(needed))
      if (en.value()) {
        primal[en.index()] = gutils->popCache(caches[count], builder);
        count++;
      }

    // Ā += matmul(Ḡ, transpose(B))
    if (!gutils->isConstantValue(lhs)) {
      auto rhsExpr = dyn_cast<hlfir::ExprType>(rhs.getType());
      if (!rhsExpr) {
        op.emitError("hlfir.matmul reverse expects !hlfir.expr operands");
        return failure();
      }
      Value bt = hlfir::TransposeOp::create(
          builder, loc, transposedExprType(rhsExpr), primal[1]);
      Value contrib =
          hlfir::MatmulOp::create(builder, loc, lhs.getType(), gc, bt, fastmath);
      gutils->addToDiffe(lhs, contrib, builder);
    }

    // B̄ += matmul_transpose(A, Ḡ)  ==  Aᵀ·Ḡ
    if (!gutils->isConstantValue(rhs)) {
      Value contrib = hlfir::MatmulTransposeOp::create(
          builder, loc, rhs.getType(), primal[0], gc, fastmath);
      gutils->addToDiffe(rhs, contrib, builder);
    }

    return success();
  }
};

} // namespace

void mlir::enzyme::registerHLFIRDialectAutoDiffInterface(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, hlfir::hlfirDialect *) {
    hlfir::ExprType::attachInterface<ExprTypeInterface>(*context);
    hlfir::MatmulOp::attachInterface<MatmulOpFwd>(*context);
    hlfir::MatmulOp::attachInterface<MatmulOpRev>(*context);
  });
}
