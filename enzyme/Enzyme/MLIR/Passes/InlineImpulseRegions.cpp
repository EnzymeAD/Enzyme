//===- InlineImpulseRegions.cpp - Inline impulse.infer into region form --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINEINFERINTOREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

static bool fnBodyHasSamples(FunctionOpInterface fn) {
  return fn.getFunctionBody()
      .walk([](impulse::SampleOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// pipeline supports: single-block bodies, all sites at the top level, leaf
/// sites with logpdf functions following the (<non-RNG outputs>...,
/// <non-RNG inputs>...) convention, sub-model sites recursively well-formed.
static bool collectReachableAddresses(Operation *root, FunctionOpInterface fn,
                                      ArrayRef<Attribute> prefix,
                                      SymbolTableCollection &symbolTable,
                                      SmallVectorImpl<Attribute> &addresses) {
  Region &body = fn.getFunctionBody();
  if (!llvm::hasSingleElement(body))
    return false;
  Block &entry = body.front();

  bool nestedInControlFlow = false;
  fn.walk([&](impulse::SampleOp sampleOp) {
    if (sampleOp->getBlock() != &entry)
      nestedInControlFlow = true;
  });
  if (nestedInControlFlow)
    return false;

  for (Operation &op : entry) {
    auto sampleOp = dyn_cast<impulse::SampleOp>(&op);
    if (!sampleOp)
      continue;
    auto symbol = sampleOp.getSymbolAttr();
    if (!symbol || sampleOp.getNumResults() < 2)
      return false;

    auto callee = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(root, sampleOp.getFnAttr()));
    if (!callee || callee.getFunctionBody().empty() ||
        !llvm::hasSingleElement(callee.getFunctionBody()))
      return false;

    SmallVector<Attribute> address(prefix.begin(), prefix.end());
    address.push_back(symbol);

    if (sampleOp.getLogpdfAttr()) {
      auto logpdfFn = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(root, sampleOp.getLogpdfAttr()));
      if (!logpdfFn || logpdfFn.getFunctionBody().empty() ||
          !llvm::hasSingleElement(logpdfFn.getFunctionBody()))
        return false;
      if (fnBodyHasSamples(callee))
        return false;
      unsigned numOutputs = sampleOp.getNumResults() - 1;
      unsigned numParams =
          sampleOp.getInputs().empty() ? 0 : sampleOp.getInputs().size() - 1;
      if (logpdfFn.getNumArguments() != numOutputs + numParams)
        return false;
      addresses.push_back(ArrayAttr::get(root->getContext(), address));
    } else {
      // Sub-model: must contain sites and recursively qualify.
      if (!fnBodyHasSamples(callee))
        return false;
      if (!collectReachableAddresses(root, callee, address, symbolTable,
                                     addresses))
        return false;
    }
  }
  return true;
}

static void inlineFunctionIntoRegion(OpBuilder &builder, FunctionOpInterface fn,
                                     Region &targetRegion) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block *block = builder.createBlock(&targetRegion);
  Block &fnEntry = fn.getFunctionBody().front();

  for (auto arg : fnEntry.getArguments())
    block->addArgument(arg.getType(), arg.getLoc());

  builder.setInsertionPointToStart(block);

  IRMapping fnMapper;
  for (auto [fnArg, blockArg] :
       llvm::zip(fnEntry.getArguments(), block->getArguments()))
    fnMapper.map(fnArg, blockArg);

  for (Operation &op : fnEntry.getOperations()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      SmallVector<Value> yieldOperands;
      for (Value operand : op.getOperands())
        yieldOperands.push_back(fnMapper.lookupOrDefault(operand));
      impulse::YieldOp::create(builder, op.getLoc(), yieldOperands);
      continue;
    }
    builder.clone(op, fnMapper);
  }
}

/// `impulse.sample_region` at the builder's insertion point, inlining the
/// sampler and logpdf function bodies as sub-regions. The site's address is
/// `addressPrefix ++ [symbol]`. Result mappings are registered in `mapper`.
static void convertSampleToSampleRegion(OpBuilder &builder,
                                        impulse::SampleOp sampleOp,
                                        IRMapping &mapper,
                                        SymbolTableCollection &symbolTable,
                                        ArrayAttr addressPrefix = ArrayAttr()) {
  Location loc = sampleOp.getLoc();
  MLIRContext *ctx = sampleOp.getContext();

  SmallVector<Value> mappedInputs;
  for (Value input : sampleOp.getInputs())
    mappedInputs.push_back(mapper.lookupOrDefault(input));

  SmallVector<Attribute> address;
  if (addressPrefix)
    address.assign(addressPrefix.begin(), addressPrefix.end());
  address.push_back(sampleOp.getSymbolAttr());

  auto sampleRegionOp = impulse::SampleRegionOp::create(
      builder, loc, sampleOp.getResultTypes(), mappedInputs,
      sampleOp.getFnAttr(), sampleOp.getLogpdfAttr(),
      ArrayAttr::get(ctx, address), sampleOp.getSupportAttr(),
      sampleOp.getNameAttr());

  auto samplerFn = dyn_cast_or_null<FunctionOpInterface>(
      symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
  if (samplerFn && !samplerFn.getFunctionBody().empty())
    inlineFunctionIntoRegion(builder, samplerFn, sampleRegionOp.getSampler());

  if (sampleOp.getLogpdfAttr()) {
    auto logpdfFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp,
                                            sampleOp.getLogpdfAttr()));
    if (logpdfFn && !logpdfFn.getFunctionBody().empty())
      inlineFunctionIntoRegion(builder, logpdfFn, sampleRegionOp.getLogpdf());
  }

  for (auto [oldResult, newResult] :
       llvm::zip(sampleOp.getResults(), sampleRegionOp.getResults()))
    mapper.map(oldResult, newResult);
}

static bool dissolveSubmodelSites(impulse::InferRegionOp regionOp,
                                  SymbolTableCollection &symbolTable) {
  if (regionOp.getSampler().empty())
    return false;
  Block &samplerEntry = regionOp.getSampler().front();

  SmallVector<impulse::SampleRegionOp> submodels;
  for (Operation &op : samplerEntry) {
    auto site = dyn_cast<impulse::SampleRegionOp>(&op);
    if (!site)
      continue;
    if (site.getLogpdf().empty() && !site.getLogpdfFnAttr() &&
        !site.getSampler().empty() && site.getAddressAttr())
      submodels.push_back(site);
  }

  for (impulse::SampleRegionOp site : submodels) {
    ArrayAttr prefix = site.getAddressAttr();
    Block &body = site.getSampler().front();

    OpBuilder builder(site);
    IRMapping mapper;
    for (auto [arg, input] : llvm::zip(body.getArguments(), site.getInputs()))
      mapper.map(arg, input);

    for (Operation &op : body.without_terminator()) {
      if (auto innerSample = dyn_cast<impulse::SampleOp>(&op)) {
        convertSampleToSampleRegion(builder, innerSample, mapper, symbolTable,
                                    prefix);
        continue;
      }
      builder.clone(op, mapper);
    }

    auto *yield = body.getTerminator();
    for (auto [result, operand] :
         llvm::zip(site.getResults(), yield->getOperands()))
      result.replaceAllUsesWith(mapper.lookupOrDefault(operand));

    site.erase();
  }

  return !submodels.empty();
}

struct InlineInferOp : public OpRewritePattern<impulse::InferOp> {
  using OpRewritePattern<impulse::InferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::InferOp inferOp,
                                PatternRewriter &rewriter) const override {
    // Region form covers trace-based HMC/NUTS models only; everything else
    // stays flat for the regular expand-impulse lowering.
    if (inferOp.getLogpdfFnAttr())
      return failure();
    if (!inferOp.getOriginalTrace())
      return failure();
    if (!inferOp.getHmcConfig().has_value() &&
        !inferOp.getNutsConfig().has_value())
      return failure();
    auto fnAttr = inferOp.getFnAttr();
    if (!fnAttr)
      return failure();
    ArrayAttr selection = inferOp.getSelectionAttr();
    ArrayAttr allAddresses = inferOp.getAllAddressesAttr();
    if (!selection || selection.empty() || !allAddresses ||
        allAddresses.empty())
      return failure();

    SymbolTableCollection symbolTable;
    auto fn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(inferOp, fnAttr));
    if (!fn || fn.getFunctionBody().empty())
      return failure();

    SmallVector<Attribute> reachable;
    if (!collectReachableAddresses(inferOp, fn, {}, symbolTable, reachable))
      return failure();
    DenseSet<Attribute> reachableSet(reachable.begin(), reachable.end());
    DenseSet<Attribute> allSet(allAddresses.begin(), allAddresses.end());
    for (Attribute addr : selection)
      if (!reachableSet.contains(addr))
        return failure();
    for (Attribute addr : allAddresses)
      if (!reachableSet.contains(addr))
        return failure();
    for (Attribute addr : reachable)
      if (!allSet.contains(addr))
        return failure();

    // infer_region mirrors infer's operands and attributes one-to-one.
    OperationState state(inferOp.getLoc(),
                         impulse::InferRegionOp::getOperationName());
    state.addOperands(inferOp->getOperands());
    state.addTypes(inferOp->getResultTypes());
    for (NamedAttribute attr : inferOp->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    state.addRegion();
    state.addRegion();
    auto regionOp = cast<impulse::InferRegionOp>(rewriter.create(state));

    Block &fnEntry = fn.getFunctionBody().front();
    Block *bodyBlock = rewriter.createBlock(&regionOp.getSampler());
    for (auto arg : fnEntry.getArguments())
      bodyBlock->addArgument(arg.getType(), arg.getLoc());

    rewriter.setInsertionPointToStart(bodyBlock);

    IRMapping mapper;
    for (auto [fnArg, blockArg] :
         llvm::zip(fnEntry.getArguments(), bodyBlock->getArguments()))
      mapper.map(fnArg, blockArg);

    for (Operation &op : fnEntry.getOperations()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        SmallVector<Value> yieldOperands;
        for (Value operand : op.getOperands())
          yieldOperands.push_back(mapper.lookupOrDefault(operand));
        impulse::YieldOp::create(rewriter, op.getLoc(), yieldOperands);
        continue;
      }
      if (auto sampleOp = dyn_cast<impulse::SampleOp>(&op)) {
        convertSampleToSampleRegion(rewriter, sampleOp, mapper, symbolTable);
        continue;
      }
      rewriter.clone(op, mapper);
    }

    rewriter.replaceOp(inferOp, regionOp.getResults());
    return success();
  }
};

struct InlineLogpdfInferOp : public OpRewritePattern<impulse::InferOp> {
  using OpRewritePattern<impulse::InferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::InferOp inferOp,
                                PatternRewriter &rewriter) const override {
    auto logpdfFnAttr = inferOp.getLogpdfFnAttr();
    if (!logpdfFnAttr)
      return failure();
    if (!inferOp.getHmcConfig().has_value() &&
        !inferOp.getNutsConfig().has_value())
      return failure();
    auto initialPos = inferOp.getInitPosition();
    if (!initialPos)
      return failure();
    auto initialPosType = dyn_cast<RankedTensorType>(initialPos.getType());
    if (!initialPosType || !initialPosType.hasStaticShape())
      return failure();

    SymbolTableCollection symbolTable;
    auto logpdfFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(inferOp, logpdfFnAttr));
    if (!logpdfFn || logpdfFn.getFunctionBody().empty() ||
        !llvm::hasSingleElement(logpdfFn.getFunctionBody()))
      return failure();
    // logpdf_fn takes (position, inputs[1:]...): the position replaces the rng.
    if (logpdfFn.getNumArguments() != inferOp.getInputs().size())
      return failure();

    // infer_region mirrors infer's operands and attributes one-to-one.
    OperationState state(inferOp.getLoc(),
                         impulse::InferRegionOp::getOperationName());
    state.addOperands(inferOp->getOperands());
    state.addTypes(inferOp->getResultTypes());
    for (NamedAttribute attr : inferOp->getAttrs())
      state.addAttribute(attr.getName(), attr.getValue());
    state.addRegion();
    state.addRegion();
    auto regionOp = cast<impulse::InferRegionOp>(rewriter.create(state));

    inlineFunctionIntoRegion(rewriter, logpdfFn, regionOp.getLogpdf());

    Block &logpdfBlock = regionOp.getLogpdf().front();
    auto inputs = inferOp.getInputs();
    for (int64_t i = logpdfBlock.getNumArguments() - 1; i >= 1; --i) {
      logpdfBlock.getArgument(i).replaceAllUsesWith(inputs[i]);
      logpdfBlock.eraseArgument(i);
    }

    regionOp.setNumPositionArgsAttr(rewriter.getI64IntegerAttr(1));
    regionOp.setPositionSizeAttr(
        rewriter.getI64IntegerAttr(initialPosType.getNumElements()));

    // The sampler region stays empty: the density is the user logpdf.
    rewriter.replaceOp(inferOp, regionOp.getResults());
    return success();
  }
};

struct InlineInferIntoRegion
    : public enzyme::impl::InlineInferIntoRegionPassBase<
          InlineInferIntoRegion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineInferOp, InlineLogpdfInferOp>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);

    SymbolTableCollection symbolTable;
    getOperation()->walk([&](impulse::InferRegionOp regionOp) {
      while (dissolveSubmodelSites(regionOp, symbolTable)) {
      }
    });
  }
};

} // namespace
