//===- InlineImpulseRegions.cpp - Inline impulse.infer into region form --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/Impulse/Impulse.h"
#include "Dialect/Ops.h"
#include "Interfaces/DistBuilder.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetOperations.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINEINFERINTOREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

/// Returns whether `fn` contains an `impulse.sample`.
static bool fnBodyHasSamples(FunctionOpInterface fn) {
  return fn.getFunctionBody()
      .walk([](impulse::SampleOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// Returns the function `symbol` names from `from` when its body is a single
/// block, else null.
static FunctionOpInterface
lookupSingleBlockFn(SymbolTableCollection &symbolTable, Operation *from,
                    SymbolRefAttr symbol) {
  auto fn =
      symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(from, symbol);
  return llvm::hasSingleElement(fn.getFunctionBody()) ? fn : nullptr;
}

namespace {
/// The sites of a model that region form reaches.
struct ReachableSites {
  /// The address of every site, the sites of sub-models included.
  DenseSet<Attribute> addresses;
  /// The addresses of the sites in the entry block of the model.
  DenseSet<Attribute> topLevel;
  /// Whether a site of the model sits in an `impulse.if` arm.
  bool hasBranchSites = false;
};
} // namespace

/// Adds the sites `fn` reaches to `sites`, each address prefixed by `prefix`,
/// and recurses into sub-models. Fails when `fn` is not a single block, a site
/// has no symbol or no value, a sampler or `logpdf_fn` is not a single block, a
/// `logpdf_fn` site's sampler has sites of its own or its `logpdf_fn` does not
/// take the site's values and parameters, or a sub-model has no sites or sits
/// in a branch. A site may sit in `impulse.if` arms, nested in nothing else,
/// only with `allowBranchSites`; sub-models never allow it.
static LogicalResult collectReachableSites(Operation *root,
                                           FunctionOpInterface fn,
                                           ArrayRef<Attribute> prefix,
                                           bool allowBranchSites,
                                           SymbolTableCollection &symbolTable,
                                           ReachableSites &sites) {
  Region &body = fn.getFunctionBody();
  if (!llvm::hasSingleElement(body))
    return failure();
  Block &entry = body.front();
  MLIRContext *ctx = root->getContext();

  WalkResult result =
      fn.walk<WalkOrder::PreOrder>([&](impulse::SampleOp sampleOp) {
        bool nested = sampleOp->getBlock() != &entry;
        if (nested) {
          if (!allowBranchSites)
            return WalkResult::interrupt();
          for (Operation *p = sampleOp->getParentOp(); p != fn.getOperation();
               p = p->getParentOp())
            if (!isa<impulse::IfOp>(p))
              return WalkResult::interrupt();
          sites.hasBranchSites = true;
        }
        auto symbol = sampleOp.getSymbolAttr();
        if (!symbol || sampleOp.getNumResults() < 2)
          return WalkResult::interrupt();

        SmallVector<Attribute> address(prefix.begin(), prefix.end());
        address.push_back(symbol);
        auto addressAttr = ArrayAttr::get(ctx, address);
        if (prefix.empty() && !nested)
          sites.topLevel.insert(addressAttr);

        if (sampleOp.getDistAttr()) {
          // Dist sites may carry no `fn` symbol.
          sites.addresses.insert(addressAttr);
          return WalkResult::advance();
        }

        FunctionOpInterface callee =
            lookupSingleBlockFn(symbolTable, root, sampleOp.getFnAttr());
        if (!callee)
          return WalkResult::interrupt();

        if (sampleOp.getLogpdfFnAttr()) {
          FunctionOpInterface logpdfFn = lookupSingleBlockFn(
              symbolTable, root, sampleOp.getLogpdfFnAttr());
          if (!logpdfFn || fnBodyHasSamples(callee))
            return WalkResult::interrupt();
          unsigned numOutputs = sampleOp.getNumResults() - 1;
          unsigned numParams = sampleOp.getInputs().empty()
                                   ? 0
                                   : sampleOp.getInputs().size() - 1;
          if (logpdfFn.getNumArguments() != numOutputs + numParams)
            return WalkResult::interrupt();
          sites.addresses.insert(addressAttr);
          return WalkResult::advance();
        }
        if (nested || !fnBodyHasSamples(callee) ||
            failed(collectReachableSites(root, callee, address,
                                         /*allowBranchSites=*/false,
                                         symbolTable, sites)))
          return WalkResult::interrupt();
        return WalkResult::advance();
      });
  return failure(result.wasInterrupted());
}

/// Copies the entry block of `fn` into a new block of `targetRegion`, with
/// the function's return turned into an `impulse.yield`. The trailing
/// arguments of `fn` read `boundTrailingArgs`; the others become the
/// arguments of the new block. Each other op is handed to `cloneOp`, which by
/// default clones it.
static void inlineFunctionIntoRegion(
    OpBuilder &builder, FunctionOpInterface fn, Region &targetRegion,
    ValueRange boundTrailingArgs = {},
    function_ref<void(Operation &, IRMapping &)> cloneOp = nullptr) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Block &fnEntry = fn.getFunctionBody().front();
  ArrayRef<BlockArgument> blockArgs =
      fnEntry.getArguments().drop_back(boundTrailingArgs.size());
  SmallVector<Type> argTypes = llvm::map_to_vector(
      blockArgs, [](BlockArgument arg) { return arg.getType(); });
  SmallVector<Location> argLocs = llvm::map_to_vector(
      blockArgs, [](BlockArgument arg) { return arg.getLoc(); });
  Block *block = builder.createBlock(&targetRegion, {}, argTypes, argLocs);

  IRMapping fnMapper;
  fnMapper.map(blockArgs, block->getArguments());
  fnMapper.map(fnEntry.getArguments().take_back(boundTrailingArgs.size()),
               boundTrailingArgs);

  for (Operation &op : fnEntry.getOperations()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      SmallVector<Value> yieldOperands;
      for (Value operand : op.getOperands())
        yieldOperands.push_back(fnMapper.lookupOrDefault(operand));
      impulse::YieldOp::create(builder, op.getLoc(), yieldOperands);
      continue;
    }
    if (cloneOp)
      cloneOp(op, fnMapper);
    else
      builder.clone(op, fnMapper);
  }
}

/// Fills `region` with a block that takes the inputs of `sampleOp` and yields
/// a draw from `dist`, the rng first.
static void emitDistSamplerIntoRegion(OpBuilder &builder,
                                      impulse::SampleOp sampleOp,
                                      impulse::DistAttr dist, Region &region) {
  OpBuilder::InsertionGuard guard(builder);
  ValueRange inputs = sampleOp.getInputs();
  SmallVector<Location> locs =
      llvm::map_to_vector(inputs, [](Value input) { return input.getLoc(); });
  Block *block = builder.createBlock(&region, {}, inputs.getTypes(), locs);

  Value rng = block->getArgument(0);
  SmallVector<Value> distArgs(std::next(block->args_begin()),
                              block->args_end());
  auto sampleType = cast<RankedTensorType>(sampleOp.getResult(1).getType());
  SmallVector<Value> out = impulse::emitDistSampler(
      builder, sampleOp.getLoc(), dist, rng, distArgs, sampleType);
  impulse::YieldOp::create(builder, sampleOp.getLoc(), out);
}

/// Fills `region` with a block that takes a value of `sampleOp` followed by
/// its parameters and yields the log density of `dist` at that value.
static void emitDistLogpdfIntoRegion(OpBuilder &builder,
                                     impulse::SampleOp sampleOp,
                                     impulse::DistAttr dist, Region &region) {
  OpBuilder::InsertionGuard guard(builder);
  auto sampleType = cast<RankedTensorType>(sampleOp.getResult(1).getType());
  SmallVector<Type> types{sampleType};
  SmallVector<Location> locs{sampleOp.getLoc()};
  for (Value input : llvm::drop_begin(sampleOp.getInputs())) {
    types.push_back(input.getType());
    locs.push_back(input.getLoc());
  }
  Block *block = builder.createBlock(&region, {}, types, locs);

  Value observed = block->getArgument(0);
  SmallVector<Value> distArgs(std::next(block->args_begin()),
                              block->args_end());
  Value lp = impulse::emitDistLogpdf(builder, sampleOp.getLoc(), dist, observed,
                                     distArgs);
  impulse::YieldOp::create(builder, sampleOp.getLoc(), lp);
}

/// Emits an `impulse.sample_region` for `sampleOp` at its symbol appended to
/// `addressPrefix`, with operands mapped through `mapper`, and maps the
/// results of `sampleOp` to its results. Its regions hold the sampler and log
/// density of the distribution, or the inlined `fn` and `logpdf_fn`.
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

  impulse::DistAttr dist = sampleOp.getDistAttr();
  impulse::SupportAttr support = sampleOp.getSupportAttr();
  if (dist && !support)
    support = dist.getSupport();

  auto sampleRegionOp = impulse::SampleRegionOp::create(
      builder, loc, sampleOp.getResultTypes(), mappedInputs,
      sampleOp.getLogpdfFnAttr(), ArrayAttr::get(ctx, address), support, dist);

  auto inlineCallee = [&](SymbolRefAttr callee, Region &region) {
    auto calleeFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, callee));
    if (calleeFn && !calleeFn.getFunctionBody().empty())
      inlineFunctionIntoRegion(builder, calleeFn, region);
  };
  if (dist)
    emitDistLogpdfIntoRegion(builder, sampleOp, dist,
                             sampleRegionOp.getLogpdf());
  if (dist && impulse::distSamplerEmittable(dist))
    emitDistSamplerIntoRegion(builder, sampleOp, dist,
                              sampleRegionOp.getSampling());
  else
    inlineCallee(sampleOp.getFnAttr(), sampleRegionOp.getSampling());
  if (!dist && sampleOp.getLogpdfFnAttr())
    inlineCallee(sampleOp.getLogpdfFnAttr(), sampleRegionOp.getLogpdf());

  for (auto [oldResult, newResult] :
       llvm::zip(sampleOp.getResults(), sampleRegionOp.getResults()))
    mapper.map(oldResult, newResult);
}

/// Clones `body`, the body of the sub-model at `prefix` with its arguments
/// bound to `inputs`, at the insertion point of `builder`, and returns the
/// values it yields. Each site becomes an `impulse.sample_region` at its
/// symbol appended to `prefix`, and the body of each nested sub-model is
/// inlined the same way.
static SmallVector<Value>
inlineSubmodelBody(OpBuilder &builder, Block &body, ValueRange inputs,
                   ArrayAttr prefix, SymbolTableCollection &symbolTable) {
  IRMapping mapper;
  mapper.map(body.getArguments(), inputs);
  for (Operation &op : body.without_terminator()) {
    auto site = dyn_cast<impulse::SampleOp>(&op);
    if (!site) {
      builder.clone(op, mapper);
      continue;
    }
    if (site.getDistAttr() || site.getLogpdfFnAttr()) {
      convertSampleToSampleRegion(builder, site, mapper, symbolTable, prefix);
      continue;
    }
    SmallVector<Attribute> address(prefix.begin(), prefix.end());
    address.push_back(site.getSymbolAttr());
    auto callee = symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
        site, site.getFnAttr());
    SmallVector<Value> siteInputs = llvm::map_to_vector(
        site.getInputs(), [&](Value v) { return mapper.lookupOrDefault(v); });
    mapper.map(site.getResults(),
               inlineSubmodelBody(builder, callee.getFunctionBody().front(),
                                  siteInputs, builder.getArrayAttr(address),
                                  symbolTable));
  }
  return llvm::map_to_vector(
      body.getTerminator()->getOperands(),
      [&](Value operand) { return mapper.lookupOrDefault(operand); });
}

/// Replaces each sub-model site at the top level of the sampler of
/// `regionOp`, an `impulse.sample_region` whose sampler is a whole generative
/// function, with that function's body; see `inlineSubmodelBody`.
static void dissolveSubmodelSites(impulse::InferRegionOp regionOp,
                                  SymbolTableCollection &symbolTable) {
  if (regionOp.getSampling().empty())
    return;
  SmallVector<impulse::SampleRegionOp> submodels;
  for (auto site :
       regionOp.getSampling().front().getOps<impulse::SampleRegionOp>())
    if (site.getLogpdf().empty() && !site.getLogpdfFnAttr() &&
        !site.getSampling().empty() && site.getAddressAttr())
      submodels.push_back(site);

  for (impulse::SampleRegionOp site : submodels) {
    OpBuilder builder(site);
    site->replaceAllUsesWith(inlineSubmodelBody(
        builder, site.getSampling().front(), site.getInputs(),
        site.getAddressAttr(), symbolTable));
    site.erase();
  }
}

/// Creates an `impulse.infer_region` with the operands, result types and
/// attributes other than `name` of `inferOp` and two empty regions.
static impulse::InferRegionOp createInferRegion(PatternRewriter &rewriter,
                                                impulse::InferOp inferOp) {
  OperationState state(inferOp.getLoc(),
                       impulse::InferRegionOp::getOperationName());
  state.addOperands(inferOp->getOperands());
  state.addTypes(inferOp->getResultTypes());
  for (NamedAttribute attr : inferOp->getAttrs())
    if (attr.getName() != inferOp.getNameAttrName())
      state.addAttribute(attr.getName(), attr.getValue());
  state.addRegion();
  state.addRegion();
  return cast<impulse::InferRegionOp>(rewriter.create(state));
}

namespace {

/// Rewrites a model-mode `impulse.infer` into an `impulse.infer_region` whose
/// sampler is the model with an `impulse.sample_region` per site, when every
/// site is reachable in region form and listed in `all_addresses`.
struct InlineInferOp : public OpRewritePattern<impulse::InferOp> {
  using OpRewritePattern<impulse::InferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(impulse::InferOp inferOp,
                                PatternRewriter &rewriter) const override {
    if (inferOp.getLogpdfFnAttr())
      return failure();
    if (!inferOp.getOriginalTrace())
      return failure();
    if (!inferOp.getHmcConfig().has_value() &&
        !inferOp.getNutsConfig().has_value() &&
        !inferOp.getViConfig().has_value() &&
        !inferOp.getMhConfig().has_value() &&
        !inferOp.getGibbsConfig().has_value())
      return failure();
    // Constrained MH needs regenerate's observation refresh; keep it flat.
    if (inferOp.getMhConfig().has_value() && inferOp.getConstraint())
      return failure();
    // Only these region lowerings score sites inside `impulse.if` arms.
    bool allowBranchSites = inferOp.getHmcConfig().has_value() ||
                            inferOp.getNutsConfig().has_value() ||
                            inferOp.getGibbsConfig().has_value();
    auto fnAttr = inferOp.getFnAttr();
    if (!fnAttr)
      return failure();
    ArrayAttr selection = inferOp.getSelectionAttr();
    ArrayAttr allAddresses = inferOp.getAllAddressesAttr();
    if (!selection || selection.empty() || !allAddresses ||
        allAddresses.empty())
      return failure();

    SymbolTableCollection symbolTable;
    FunctionOpInterface fn = lookupSingleBlockFn(symbolTable, inferOp, fnAttr);
    if (!fn)
      return failure();

    ReachableSites reachable;
    if (failed(collectReachableSites(inferOp, fn, /*prefix=*/{},
                                     allowBranchSites, symbolTable, reachable)))
      return failure();
    DenseSet<Attribute> selected(selection.begin(), selection.end());
    DenseSet<Attribute> declared(allAddresses.begin(), allAddresses.end());
    // Selected sites become logpdf-region position args, so must be top-level.
    if (reachable.hasBranchSites &&
        !llvm::set_is_subset(selected, reachable.topLevel))
      return failure();
    if (!llvm::set_is_subset(selected, reachable.addresses) ||
        declared != reachable.addresses)
      return failure();

    impulse::InferRegionOp regionOp = createInferRegion(rewriter, inferOp);
    inlineFunctionIntoRegion(
        rewriter, fn, regionOp.getSampling(), /*boundTrailingArgs=*/{},
        [&](Operation &op, IRMapping &mapper) {
          if (auto sampleOp = dyn_cast<impulse::SampleOp>(&op)) {
            convertSampleToSampleRegion(rewriter, sampleOp, mapper,
                                        symbolTable);
            return;
          }
          Operation *cloned = rewriter.clone(op, mapper);
          if (!reachable.hasBranchSites || cloned->getNumRegions() == 0)
            return;
          SmallVector<impulse::SampleOp> nestedSites;
          cloned->walk([&](impulse::SampleOp nestedSite) {
            nestedSites.push_back(nestedSite);
          });
          for (impulse::SampleOp nestedSite : nestedSites) {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(nestedSite);
            IRMapping nestedMapper;
            convertSampleToSampleRegion(rewriter, nestedSite, nestedMapper,
                                        symbolTable);
            SmallVector<Value> replacements;
            for (Value result : nestedSite.getResults())
              replacements.push_back(nestedMapper.lookup(result));
            rewriter.replaceOp(nestedSite, replacements);
          }
        });

    rewriter.replaceOp(inferOp, regionOp.getResults());
    return success();
  }
};

/// Rewrites an HMC or NUTS `impulse.infer` with a `logpdf_fn` into an
/// `impulse.infer_region` whose logpdf region is that function inlined.
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
    FunctionOpInterface logpdfFn =
        lookupSingleBlockFn(symbolTable, inferOp, logpdfFnAttr);
    if (!logpdfFn)
      return failure();
    // logpdf_fn takes (position, inputs[1:]...): the position replaces the rng.
    if (logpdfFn.getNumArguments() != inferOp.getInputs().size())
      return failure();

    impulse::InferRegionOp regionOp = createInferRegion(rewriter, inferOp);
    inlineFunctionIntoRegion(rewriter, logpdfFn, regionOp.getLogpdf(),
                             inferOp.getInputs().drop_front());

    // The sampler region stays empty: the density is the user logpdf.
    rewriter.replaceOp(inferOp, regionOp.getResults());
    return success();
  }
};

/// Rewrites `impulse.infer` into `impulse.infer_region` where the model allows
/// it.
struct InlineInferIntoRegionPass
    : public enzyme::impl::InlineInferIntoRegionPassBase<
          InlineInferIntoRegionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineInferOp, InlineLogpdfInferOp>(&getContext());

    (void)applyPatternsGreedily(getOperation(), std::move(patterns));

    SymbolTableCollection symbolTable;
    getOperation()->walk([&](impulse::InferRegionOp regionOp) {
      dissolveSubmodelSites(regionOp, symbolTable);
    });
  }
};

} // namespace
