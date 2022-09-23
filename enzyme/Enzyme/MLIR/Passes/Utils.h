#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"

static inline mlir::scf::IfOp
cloneWithResults(mlir::scf::IfOp op, mlir::OpBuilder &rewriter,
                 mlir::BlockAndValueMapping mapping = {}) {
  using namespace mlir;
  return rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                    mapping.lookupOrDefault(op.getCondition()),
                                    true);
}
static inline mlir::AffineIfOp
cloneWithResults(mlir::AffineIfOp op, mlir::OpBuilder &rewriter,
                 mlir::BlockAndValueMapping mapping = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<AffineIfOp>(op.getLoc(), op.getResultTypes(),
                                     op.getIntegerSet(), lower, true);
}

static inline mlir::scf::IfOp
cloneWithoutResults(mlir::scf::IfOp op, mlir::OpBuilder &rewriter,
                    mlir::BlockAndValueMapping mapping = {},
                    mlir::TypeRange types = {}) {
  using namespace mlir;
  return rewriter.create<scf::IfOp>(
      op.getLoc(), types, mapping.lookupOrDefault(op.getCondition()), true);
}
static inline mlir::AffineIfOp
cloneWithoutResults(mlir::AffineIfOp op, mlir::OpBuilder &rewriter,
                    mlir::BlockAndValueMapping mapping = {},
                    mlir::TypeRange types = {}) {
  using namespace mlir;
  SmallVector<mlir::Value> lower;
  for (auto o : op.getOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<AffineIfOp>(op.getLoc(), types, op.getIntegerSet(),
                                     lower, true);
}

static inline mlir::scf::ForOp
cloneWithoutResults(mlir::scf::ForOp op, mlir::PatternRewriter &rewriter,
                    mlir::BlockAndValueMapping mapping = {}) {
  using namespace mlir;
  return rewriter.create<scf::ForOp>(
      op.getLoc(), mapping.lookupOrDefault(op.getLowerBound()),
      mapping.lookupOrDefault(op.getUpperBound()),
      mapping.lookupOrDefault(op.getStep()));
}
static inline mlir::AffineForOp
cloneWithoutResults(mlir::AffineForOp op, mlir::PatternRewriter &rewriter,
                    mlir::BlockAndValueMapping mapping = {}) {
  using namespace mlir;
  SmallVector<Value> lower;
  for (auto o : op.getLowerBoundOperands())
    lower.push_back(mapping.lookupOrDefault(o));
  SmallVector<Value> upper;
  for (auto o : op.getUpperBoundOperands())
    upper.push_back(mapping.lookupOrDefault(o));
  return rewriter.create<AffineForOp>(op.getLoc(), lower, op.getLowerBoundMap(),
                                      upper, op.getUpperBoundMap(),
                                      op.getStep());
}

static inline void clearBlock(mlir::Block *block,
                              mlir::PatternRewriter &rewriter) {
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
    assert(op.use_empty() && "expected 'op' to have no uses");
    rewriter.eraseOp(&op);
  }
}

static inline mlir::Block *getThenBlock(mlir::scf::IfOp op) {
  return op.thenBlock();
}
static inline mlir::Block *getThenBlock(mlir::AffineIfOp op) {
  return op.getThenBlock();
}
static inline mlir::Block *getElseBlock(mlir::scf::IfOp op) {
  return op.elseBlock();
}
static inline mlir::Block *getElseBlock(mlir::AffineIfOp op) {
  if (op.hasElse())
    return op.getElseBlock();
  else
    return nullptr;
}

static inline mlir::Region &getThenRegion(mlir::scf::IfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getThenRegion(mlir::AffineIfOp op) {
  return op.getThenRegion();
}
static inline mlir::Region &getElseRegion(mlir::scf::IfOp op) {
  return op.getElseRegion();
}
static inline mlir::Region &getElseRegion(mlir::AffineIfOp op) {
  return op.getElseRegion();
}

static inline mlir::scf::YieldOp getThenYield(mlir::scf::IfOp op) {
  return op.thenYield();
}
static inline mlir::AffineYieldOp getThenYield(mlir::AffineIfOp op) {
  return llvm::cast<mlir::AffineYieldOp>(op.getThenBlock()->getTerminator());
}
static inline mlir::scf::YieldOp getElseYield(mlir::scf::IfOp op) {
  return op.elseYield();
}
static inline mlir::AffineYieldOp getElseYield(mlir::AffineIfOp op) {
  return llvm::cast<mlir::AffineYieldOp>(op.getElseBlock()->getTerminator());
}

static inline bool inBound(mlir::scf::IfOp op, mlir::Value v) {
  return op.getCondition() == v;
}
static inline bool inBound(mlir::AffineIfOp op, mlir::Value v) {
  return llvm::any_of(op.getOperands(), [&](mlir::Value e) { return e == v; });
}
static inline bool inBound(mlir::scf::ForOp op, mlir::Value v) {
  return op.getUpperBound() == v;
}
static inline bool inBound(mlir::AffineForOp op, mlir::Value v) {
  return llvm::any_of(op.getUpperBoundOperands(),
                      [&](mlir::Value e) { return e == v; });
}
static inline bool hasElse(mlir::scf::IfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}
static inline bool hasElse(mlir::AffineIfOp op) {
  return op.getElseRegion().getBlocks().size() > 0;
}
