#include "Dialect/Ops.h"
#include "Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_INLINEMCMCINTOREGIONPASS
#define GEN_PASS_DEF_OUTLINEMCMCFROMREGIONPASS
#include "Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

namespace {

static void inlineFunctionIntoRegion(OpBuilder &builder, FunctionOpInterface fn,
                                     Region &targetRegion, Location loc) {
  Block *block = builder.createBlock(&targetRegion);
  Block &fnEntry = fn.getFunctionBody().front();

  for (auto arg : fnEntry.getArguments()) {
    block->addArgument(arg.getType(), arg.getLoc());
  }

  builder.setInsertionPointToStart(block);

  IRMapping fnMapper;
  for (auto [fnArg, blockArg] :
       llvm::zip(fnEntry.getArguments(), block->getArguments())) {
    fnMapper.map(fnArg, blockArg);
  }

  for (Operation &op : fnEntry.getOperations()) {
    if (op.hasTrait<OpTrait::ReturnLike>()) {
      SmallVector<Value> yieldOperands;
      for (Value operand : op.getOperands()) {
        yieldOperands.push_back(fnMapper.lookupOrDefault(operand));
      }
      enzyme::YieldOp::create(builder, op.getLoc(), yieldOperands);
      continue;
    }
    builder.clone(op, fnMapper);
  }
}

static LogicalResult
convertSampleToSampleRegion(OpBuilder &builder, enzyme::SampleOp sampleOp,
                            IRMapping &mapper,
                            SymbolTableCollection &symbolTable) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Location loc = sampleOp.getLoc();

  SmallVector<Value> mappedInputs;
  for (Value input : sampleOp.getInputs()) {
    mappedInputs.push_back(mapper.lookupOrDefault(input));
  }

  StringAttr fnStrAttr =
      StringAttr::get(sampleOp.getContext(), sampleOp.getFn());
  StringAttr logpdfStrAttr =
      sampleOp.getLogpdfAttr()
          ? StringAttr::get(sampleOp.getContext(), *sampleOp.getLogpdf())
          : StringAttr{};

  auto sampleRegionOp = enzyme::SampleRegionOp::create(
      builder, loc, sampleOp.getResultTypes(), mappedInputs, fnStrAttr,
      logpdfStrAttr, sampleOp.getSymbolAttr(), sampleOp.getSupportAttr(),
      sampleOp.getNameAttr());

  if (sampleOp.getFnAttr()) {
    auto samplerFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp, sampleOp.getFnAttr()));
    if (samplerFn && !samplerFn.getFunctionBody().empty()) {
      inlineFunctionIntoRegion(builder, samplerFn, sampleRegionOp.getSampler(),
                               loc);
    }
  }

  if (sampleOp.getLogpdfAttr()) {
    auto logpdfFn = dyn_cast_or_null<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(sampleOp,
                                            sampleOp.getLogpdfAttr()));
    if (logpdfFn && !logpdfFn.getFunctionBody().empty()) {
      inlineFunctionIntoRegion(builder, logpdfFn, sampleRegionOp.getLogpdf(),
                               loc);
    }
  }

  for (auto [oldResult, newResult] :
       llvm::zip(sampleOp.getResults(), sampleRegionOp.getResults())) {
    mapper.map(oldResult, newResult);
  }

  return success();
}

struct InlineMCMCOp : public OpRewritePattern<enzyme::MCMCOp> {
  using OpRewritePattern<enzyme::MCMCOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(enzyme::MCMCOp mcmcOp,
                                PatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTable;
    Location loc = mcmcOp.getLoc();

    bool isLogpdfMode = static_cast<bool>(mcmcOp.getLogpdfFnAttr());

    FunctionOpInterface targetFn;
    if (isLogpdfMode) {
      targetFn = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp,
                                              mcmcOp.getLogpdfFnAttr()));
    } else {
      targetFn = dyn_cast_or_null<FunctionOpInterface>(
          symbolTable.lookupNearestSymbolFrom(mcmcOp, mcmcOp.getFnAttr()));
    }

    if (!targetFn || targetFn.getFunctionBody().empty()) {
      return mcmcOp.emitError("Cannot inline empty model/logpdf function");
    }

    auto fnStrAttr = isLogpdfMode
                         ? StringAttr{}
                         : StringAttr::get(mcmcOp.getContext(),
                                           mcmcOp.getFnAttr().getValue());

    auto mcmcRegionOp = enzyme::MCMCRegionOp::create(
        rewriter, loc, mcmcOp.getResultTypes(), mcmcOp.getInputs(),
        mcmcOp.getOriginalTrace(), mcmcOp.getSelectionAttr(),
        mcmcOp.getAllAddressesAttr(), mcmcOp.getNumWarmupAttr(),
        mcmcOp.getNumSamplesAttr(), mcmcOp.getThinningAttr(),
        mcmcOp.getInverseMassMatrix(), mcmcOp.getStepSize(),
        mcmcOp.getHmcConfigAttr(), mcmcOp.getNutsConfigAttr(),
        mcmcOp.getLogpdfFnAttr(), mcmcOp.getInitialPosition(), fnStrAttr,
        mcmcOp.getNameAttr());

    Block *bodyBlock = rewriter.createBlock(&mcmcRegionOp.getBody());

    Block &fnEntry = targetFn.getFunctionBody().front();
    for (auto arg : fnEntry.getArguments()) {
      bodyBlock->addArgument(arg.getType(), arg.getLoc());
    }

    rewriter.setInsertionPointToStart(bodyBlock);

    IRMapping mapper;
    for (auto [fnArg, blockArg] :
         llvm::zip(fnEntry.getArguments(), bodyBlock->getArguments())) {
      mapper.map(fnArg, blockArg);
    }

    for (Operation &op : fnEntry.getOperations()) {
      if (op.hasTrait<OpTrait::ReturnLike>()) {
        SmallVector<Value> yieldOperands;
        for (Value operand : op.getOperands()) {
          yieldOperands.push_back(mapper.lookupOrDefault(operand));
        }
        enzyme::YieldOp::create(rewriter, op.getLoc(), yieldOperands);
        continue;
      }

      if (!isLogpdfMode) {
        if (auto sampleOp = dyn_cast<enzyme::SampleOp>(&op)) {
          if (failed(convertSampleToSampleRegion(rewriter, sampleOp, mapper,
                                                 symbolTable))) {
            return failure();
          }
          continue;
        }
      }

      rewriter.clone(op, mapper);
    }

    rewriter.replaceOp(mcmcOp, mcmcRegionOp.getResults());
    return success();
  }
};

static void convertSampleRegionToSample(func::FuncOp outlinedFunc) {
  SmallVector<enzyme::SampleRegionOp> toConvert;
  outlinedFunc.walk(
      [&](enzyme::SampleRegionOp op) { toConvert.push_back(op); });

  for (auto sampleRegionOp : toConvert) {
    OpBuilder builder(sampleRegionOp);
    auto *ctx = builder.getContext();

    FlatSymbolRefAttr fnAttr;
    if (auto fnStrAttr = sampleRegionOp.getFnAttr()) {
      fnAttr = FlatSymbolRefAttr::get(ctx, fnStrAttr);
    }

    FlatSymbolRefAttr logpdfAttr;
    if (auto logpdfStrAttr = sampleRegionOp.getLogpdfFnAttr()) {
      logpdfAttr = FlatSymbolRefAttr::get(ctx, logpdfStrAttr);
    }

    auto sampleOp = enzyme::SampleOp::create(
        builder, sampleRegionOp.getLoc(), sampleRegionOp.getResultTypes(),
        fnAttr, sampleRegionOp.getInputs(), logpdfAttr,
        sampleRegionOp.getSymbolAttr(), sampleRegionOp.getSupportAttr(),
        sampleRegionOp.getNameAttr());

    sampleRegionOp.replaceAllUsesWith(sampleOp.getResults());
    sampleRegionOp.erase();
  }
}

static FailureOr<func::FuncOp> outlineRegionToFunction(Region &region,
                                                       StringRef funcName,
                                                       OpBuilder &builder) {
  if (region.empty())
    return failure();

  Block &entryBlock = region.front();

  auto *terminator = entryBlock.getTerminator();
  auto yieldOp = cast<enzyme::YieldOp>(terminator);
  SmallVector<Type> resultTypes(yieldOp.getOperandTypes());

  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(region, freeValues);

  SmallVector<Type> argTypes(entryBlock.getArgumentTypes());
  SmallVector<Location> argLocs(entryBlock.getNumArguments(), region.getLoc());

  for (Value freeVal : freeValues) {
    argTypes.push_back(freeVal.getType());
    argLocs.push_back(freeVal.getLoc());
  }

  auto fnType = builder.getFunctionType(argTypes, resultTypes);
  auto outlinedFunc =
      func::FuncOp::create(builder, region.getLoc(), funcName, fnType);
  outlinedFunc.setPrivate();
  Region &outlinedBody = outlinedFunc.getBody();

  IRMapping map;
  Block *newEntryBlock = builder.createBlock(
      &outlinedBody, outlinedBody.begin(), argTypes, argLocs);

  unsigned originalArgCount = entryBlock.getNumArguments();
  for (auto arg : entryBlock.getArguments())
    map.map(arg, newEntryBlock->getArgument(arg.getArgNumber()));
  for (auto [idx, freeVal] : llvm::enumerate(freeValues))
    map.map(freeVal, newEntryBlock->getArgument(originalArgCount + idx));

  region.cloneInto(&outlinedBody, map);

  for (Block &block : outlinedBody) {
    if (!block.mightHaveTerminator())
      continue;
    auto terminator = dyn_cast<enzyme::YieldOp>(block.getTerminator());
    if (!terminator)
      continue;
    OpBuilder replacer(terminator);
    func::ReturnOp::create(replacer, terminator->getLoc(),
                           terminator->getOperands());
    terminator->erase();
  }

  Block *clonedEntry = map.lookup(&entryBlock);
  newEntryBlock->getOperations().splice(newEntryBlock->getOperations().end(),
                                        clonedEntry->getOperations());
  clonedEntry->erase();

  // TODO: Ponder over this.
  for (auto [idx, freeVal] : llvm::enumerate(freeValues)) {
    Operation *defOp = freeVal.getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
      continue;

    unsigned argIdx = originalArgCount + idx;
    Value funcArg = newEntryBlock->getArgument(argIdx);

    for (OpOperand &use : llvm::make_early_inc_range(funcArg.getUses())) {
      Operation *user = use.getOwner();
      if (user->getParentRegion() == &outlinedBody)
        continue;
      OpBuilder bodyBuilder(user);
      Operation *clonedConst = bodyBuilder.clone(*defOp);
      use.set(
          clonedConst->getResult(cast<OpResult>(freeVal).getResultNumber()));
    }
  }

  convertSampleRegionToSample(outlinedFunc);

  return outlinedFunc;
}

LogicalResult outlineMCMCRegion(enzyme::MCMCRegionOp regionOp,
                                StringRef funcName, OpBuilder &builder) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointAfter(
      regionOp->getParentOfType<SymbolOpInterface>());

  llvm::SetVector<Value> freeValues;
  getUsedValuesDefinedAbove(regionOp.getBody(), freeValues);

  FailureOr<func::FuncOp> outlinedFunc =
      outlineRegionToFunction(regionOp.getBody(), funcName, builder);
  if (failed(outlinedFunc))
    return failure();

  SmallVector<Value> allInputs(regionOp.getInputs());
  allInputs.append(freeValues.begin(), freeValues.end());

  builder.setInsertionPoint(regionOp);
  auto outlinedSymRef =
      FlatSymbolRefAttr::get(builder.getContext(), outlinedFunc->getName());

  bool isLogpdfMode = static_cast<bool>(regionOp.getLogpdfFnAttr());
  FlatSymbolRefAttr fnAttr =
      isLogpdfMode ? FlatSymbolRefAttr{} : outlinedSymRef;
  FlatSymbolRefAttr logpdfAttr =
      isLogpdfMode ? outlinedSymRef : regionOp.getLogpdfFnAttr();

  auto newOp = enzyme::MCMCOp::create(
      builder, regionOp.getLoc(), regionOp.getResultTypes(), fnAttr, allInputs,
      regionOp.getOriginalTrace(), regionOp.getSelectionAttr(),
      regionOp.getAllAddressesAttr(), regionOp.getNumWarmupAttr(),
      regionOp.getNumSamplesAttr(), regionOp.getThinningAttr(),
      regionOp.getInverseMassMatrix(), regionOp.getStepSize(),
      regionOp.getHmcConfigAttr(), regionOp.getNutsConfigAttr(), logpdfAttr,
      regionOp.getInitialPosition(), regionOp.getNameAttr());

  regionOp.replaceAllUsesWith(newOp.getResults());
  regionOp.erase();
  return success();
}

struct InlineMCMCIntoRegion
    : public enzyme::impl::InlineMCMCIntoRegionPassBase<InlineMCMCIntoRegion> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineMCMCOp>(&getContext());

    GreedyRewriteConfig config;
    (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }
};

struct OutlineMCMCFromRegion
    : public enzyme::impl::OutlineMCMCFromRegionPassBase<
          OutlineMCMCFromRegion> {
  void runOnOperation() override {
    SmallVector<enzyme::MCMCRegionOp> toOutline;
    getOperation()->walk(
        [&](enzyme::MCMCRegionOp op) { toOutline.push_back(op); });

    OpBuilder builder(getOperation());
    unsigned increment = 0;
    for (auto regionOp : toOutline) {
      auto symbol = regionOp->getParentOfType<SymbolOpInterface>();
      std::string defaultName =
          (Twine(symbol.getName()) + "_mcmc_model" + Twine(increment)).str();
      if (failed(outlineMCMCRegion(regionOp, defaultName, builder)))
        return signalPassFailure();
      ++increment;
    }
  }
};

} // namespace
