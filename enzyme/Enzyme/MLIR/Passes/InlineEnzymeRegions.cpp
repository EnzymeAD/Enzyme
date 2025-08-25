//===- InlineEnzymeRegions.cpp - Inline/outline enzyme.autodiff  ------------ //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements passes to inlining and outlining to convert
// between enzyme.autodiff and enzyme.autodiff_region ops.
//
//===----------------------------------------------------------------------===//
#include "Dialect/Ops.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINEENZYMEINTOREGIONPASS
#define GEN_PASS_DEF_OUTLINEENZYMEFROMREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {
constexpr static llvm::StringLiteral kFnAttrsName = "fn_attrs";

static StringRef getFunctionTypeAttrName(Operation *operation) {
  return llvm::TypeSwitch<Operation *, StringRef>(operation)
      .Case<func::FuncOp, LLVM::LLVMFuncOp>(
          [](auto op) { return op.getFunctionTypeAttrName(); })
      .Default([](Operation *) {
        llvm_unreachable("expected op with a function type");
        return "";
      });
}

static StringRef getArgAttrsAttrName(Operation *operation) {
  return llvm::TypeSwitch<Operation *, StringRef>(operation)
      .Case<func::FuncOp, LLVM::LLVMFuncOp>(
          [](auto op) { return op.getArgAttrsAttrName(); })
      .Default([](Operation *) {
        llvm_unreachable("expected op with arg attrs");
        return "";
      });
}

static void serializeFunctionAttributes(Operation *fn,
                                        enzyme::AutoDiffRegionOp regionOp) {
  SmallVector<NamedAttribute> fnAttrs;
  fnAttrs.reserve(fn->getAttrDictionary().size());
  for (auto attr : fn->getAttrs()) {
    // Don't store the function type because it may change when outlining
    if (attr.getName() == getFunctionTypeAttrName(fn))
      continue;
    fnAttrs.push_back(attr);
  }

  regionOp->setAttr(kFnAttrsName,
                    DictionaryAttr::getWithSorted(fn->getContext(), fnAttrs));
}

static void deserializeFunctionAttributes(enzyme::AutoDiffRegionOp op,
                                          Operation *outlinedFunc,
                                          unsigned addedArgCount) {
  if (!op->hasAttrOfType<DictionaryAttr>(kFnAttrsName))
    return;

  MLIRContext *ctx = op->getContext();
  SmallVector<NamedAttribute> fnAttrs;
  for (auto attr : op->getAttrOfType<DictionaryAttr>(kFnAttrsName)) {
    // New arguments are potentially added when outlining due to references to
    // values outside the region. Insert an empty arg attr for each newly
    // added argument.
    if (attr.getName() == getArgAttrsAttrName(outlinedFunc)) {
      SmallVector<Attribute> argAttrs(
          cast<ArrayAttr>(attr.getValue()).getAsRange<DictionaryAttr>());
      for (unsigned i = 0; i < addedArgCount; ++i)
        argAttrs.push_back(DictionaryAttr::getWithSorted(ctx, {}));
      fnAttrs.push_back(
          NamedAttribute(attr.getName(), ArrayAttr::get(ctx, argAttrs)));
    } else
      fnAttrs.push_back(attr);
  }
  outlinedFunc->setAttrs(fnAttrs);
}

struct InlineEnzymeAutoDiff : public OpRewritePattern<enzyme::AutoDiffOp> {
  using OpRewritePattern<enzyme::AutoDiffOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(enzyme::AutoDiffOp op,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    auto *symbol = symbolTable.lookupNearestSymbolFrom(op, op.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbol);
    // Use a StringAttr rather than a SymbolRefAttr so the function can get
    // symbol-DCE'd
    auto fnAttr = StringAttr::get(op.getContext(), op.getFn());
    auto regionOp = rewriter.replaceOpWithNewOp<enzyme::AutoDiffRegionOp>(
        op, op.getResultTypes(), op.getInputs(), op.getActivity(),
        op.getRetActivity(), op.getWidth(), op.getStrongZero(), fnAttr);
    serializeFunctionAttributes(fn, regionOp);
    rewriter.cloneRegionBefore(fn.getFunctionBody(), regionOp.getBody(),
                               regionOp.getBody().begin());
    SmallVector<Operation *> toErase;
    for (Operation &bodyOp : regionOp.getBody().getOps()) {
      if (bodyOp.hasTrait<OpTrait::ReturnLike>()) {
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        rewriter.setInsertionPoint(&bodyOp);
        enzyme::YieldOp::create(rewriter, bodyOp.getLoc(),
                                bodyOp.getOperands());
        toErase.push_back(&bodyOp);
      }
    }

    for (Operation *opToErase : toErase)
      rewriter.eraseOp(opToErase);

    return success();
  }
};

// Based on
// https://github.com/llvm/llvm-project/blob/665da0a1649814471739c41a702e0e9447316b20/mlir/lib/Dialect/GPU/Transforms/KernelOutlining.cpp
static FailureOr<func::FuncOp>
outlineAutoDiffFunc(enzyme::AutoDiffRegionOp op, StringRef funcName,
                    SmallVectorImpl<Value> &inputs,
                    SmallVectorImpl<enzyme::Activity> &argActivities,
                    OpBuilder &builder) {
  Region &autodiffRegion = op.getBody();
  SmallVector<Type> argTypes(autodiffRegion.getArgumentTypes()), resultTypes;
  SmallVector<Location> argLocs(autodiffRegion.getNumArguments(), op.getLoc());
  // Infer the result types from an enzyme.yield op
  bool found = false;
  autodiffRegion.walk([&](enzyme::YieldOp yieldOp) {
    found = true;
    llvm::append_range(resultTypes, yieldOp.getOperandTypes());
    return WalkResult::interrupt();
  });
  if (!found)
    return op.emitError()
           << "enzyme.yield was not found in enzyme.autodiff_region";

  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(autodiffRegion, freeValues);

  for (Value value : freeValues) {
    inputs.push_back(value);
    argTypes.push_back(value.getType());
    argLocs.push_back(value.getLoc());
    argActivities.push_back(enzyme::Activity::enzyme_const);
  }
  auto fnType = builder.getFunctionType(argTypes, resultTypes);

  // FIXME: making this location the location of the
  // enzyme.autodiff_region op causes translation to LLVM IR to fail due
  // to some issue with the dbg info.
  Location loc = UnknownLoc::get(op.getContext());
  auto outlinedFunc = func::FuncOp::create(builder, loc, funcName, fnType);
  Region &outlinedBody = outlinedFunc.getBody();
  deserializeFunctionAttributes(op, outlinedFunc, freeValues.size());

  // Copy over the function body.
  IRMapping map;
  Block *entryBlock = builder.createBlock(&outlinedBody, outlinedBody.begin(),
                                          argTypes, argLocs);
  unsigned originalArgCount = autodiffRegion.getNumArguments();
  for (const auto &arg : autodiffRegion.getArguments())
    map.map(arg, entryBlock->getArgument(arg.getArgNumber()));
  for (const auto &operand : enumerate(freeValues))
    map.map(operand.value(),
            entryBlock->getArgument(originalArgCount + operand.index()));
  autodiffRegion.cloneInto(&outlinedBody, map);

  // Replace the terminators with returns
  for (Block &block : autodiffRegion) {
    Block *clonedBlock = map.lookup(&block);
    auto terminator = dyn_cast<enzyme::YieldOp>(clonedBlock->getTerminator());
    if (!terminator)
      continue;
    OpBuilder replacer(terminator);
    func::ReturnOp::create(replacer, terminator->getLoc(),
                           terminator->getOperands());
    terminator->erase();
  }

  // cloneInto results in two blocks, the actual outlined entry block and the
  // cloned autodiff_region entry block. Splice the cloned entry block into
  // the actual entry block, then erase the cloned autodiff_region entry.
  Block *clonedEntry = map.lookup(&autodiffRegion.front());
  entryBlock->getOperations().splice(entryBlock->getOperations().end(),
                                     clonedEntry->getOperations());
  clonedEntry->erase();
  return outlinedFunc;
}

LogicalResult outlineEnzymeAutoDiffRegion(enzyme::AutoDiffRegionOp op,
                                          StringRef defaultFuncName,
                                          OpBuilder &builder) {
  StringRef funcName = op.getFn().value_or(defaultFuncName);
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointAfter(op->getParentOfType<SymbolOpInterface>());

  SmallVector<enzyme::Activity> argActivities =
      llvm::map_to_vector(op.getActivity().getAsRange<enzyme::ActivityAttr>(),
                          [](auto attr) { return attr.getValue(); });
  SmallVector<Value> inputs(op.getInputs());

  FailureOr<func::FuncOp> outlinedFunc =
      outlineAutoDiffFunc(op, funcName, inputs, argActivities, builder);
  if (failed(outlinedFunc))
    return failure();
  builder.setInsertionPoint(op);
  ArrayAttr argActivityAttr = builder.getArrayAttr(llvm::map_to_vector(
      argActivities, [&op](enzyme::Activity actv) -> Attribute {
        return enzyme::ActivityAttr::get(op.getContext(), actv);
      }));
  auto newOp = enzyme::AutoDiffOp::create(
      builder, op.getLoc(), op.getResultTypes(), outlinedFunc->getName(),
      inputs, argActivityAttr, op.getRetActivity(), op.getWidth(),
      op.getStrongZero());
  op.replaceAllUsesWith(newOp.getResults());
  op.erase();
  return success();
}

struct InlineEnzymeIntoRegion
    : public enzyme::impl::InlineEnzymeIntoRegionPassBase<
          InlineEnzymeIntoRegion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<InlineEnzymeAutoDiff>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

struct OutlineEnzymeFromRegion
    : public enzyme::impl::OutlineEnzymeFromRegionPassBase<
          OutlineEnzymeFromRegion> {
  void runOnOperation() override {
    SmallVector<enzyme::AutoDiffRegionOp> toOutline;
    getOperation()->walk(
        [&](enzyme::AutoDiffRegionOp op) { toOutline.push_back(op); });

    OpBuilder builder(getOperation());
    unsigned increment = 0;
    for (auto regionOp : toOutline) {
      auto symbol = regionOp->getParentOfType<SymbolOpInterface>();
      std::string defaultName =
          (Twine(symbol.getName(), "_to_diff") + Twine(increment)).str();
      if (failed(outlineEnzymeAutoDiffRegion(regionOp, defaultName, builder)))
        return signalPassFailure();

      ++increment;
    }
  }
};

} // namespace
