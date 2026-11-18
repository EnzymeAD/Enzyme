#include "Analysis/DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffControlFlowInterface.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

// TODO: this shouldn't depend on specific dialects except Enzyme.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "EnzymeLogic.h"
#include "Interfaces/GradientUtils.h"
#include "Interfaces/GradientUtilsReverse.h"
#include "Interfaces/Utils.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::enzyme;

// Have the derivative's dialect spells the control flow.
static const AutoDiffControlFlowInterface &
controlFlow(MGradientUtilsReverse *gutils) {
  return getAutoDiffControlFlow(gutils->newFunc.getOperation());
}

void handleReturns(Block *oBB, Block *newBB, Block *reverseBB,
                   MGradientUtilsReverse *gutils) {
  if (oBB->getNumSuccessors() == 0) {
    Operation *returnStatement = newBB->getTerminator();
    gutils->erase(returnStatement);

    OpBuilder forwardToBackwardBuilder(newBB, newBB->end());

    Operation *newBranchOp = controlFlow(gutils).createBranch(
        forwardToBackwardBuilder, oBB->getTerminator()->getLoc(), reverseBB);

    gutils->originalToNewFnOps[oBB->getTerminator()] = newBranchOp;
  }
}

// Returns true iff the operation:
//    1. Produces no active data nor active pointers
//    2. Does not propagate active data nor pointers (via side effects)
static bool isFullyInactive(Operation *op, MGradientUtils *gutils) {
  return llvm::all_of(
             op->getResults(),
             [gutils](Value v) { return gutils->isConstantValue(v); }) &&
         gutils->isConstantInstruction(op) && oputils::isReadOnly(op);
}

/*
Create reverse mode adjoint for an operation.
*/
LogicalResult MEnzymeLogic::visitChild(Operation *op, OpBuilder &builder,
                                       MGradientUtilsReverse *gutils) {
  if ((op->getBlock()->getTerminator() != op) && isFullyInactive(op, gutils)) {
    return success();
  }
  if (auto ifaceOp = dyn_cast<ReverseAutoDiffOpInterface>(op)) {
    SmallVector<Value> caches = ifaceOp.cacheValues(gutils);
    OpBuilder augmentBuilder(gutils->getNewFromOriginal(op));
    if (failed(ifaceOp.createShadowValues(augmentBuilder, gutils)))
      return failure();
    return ifaceOp.createReverseModeAdjoint(builder, gutils, caches);
  }
  op->emitError() << "could not compute the adjoint for this operation " << *op;
  return failure();
}

LogicalResult MEnzymeLogic::visitChildren(Block *oBB, Block *reverseBB,
                                          MGradientUtilsReverse *gutils) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  bool valid = true;
  if (!oBB->empty()) {
    auto first = oBB->rbegin();
    auto last = oBB->rend();
    for (auto it = first; it != last; ++it) {
      Operation *op = &*it;
      valid &= visitChild(op, revBuilder, gutils).succeeded();
    }
  }
  return success(valid);
}

void MEnzymeLogic::handlePredecessors(
    Block *oBB, Block *newBB, Block *reverseBB, MGradientUtilsReverse *gutils,
    llvm::function_ref<buildReturnFunction> buildReturnOp,
    SmallVectorImpl<EdgeIndexCache> &edgeCaches) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  const AutoDiffControlFlowInterface &flow = controlFlow(gutils);

  if (oBB->isEntryBlock()) {
    buildReturnOp(revBuilder, oBB);
    return;
  }

  SmallVector<std::pair<Block *, unsigned>> edges;
  for (auto it = oBB->pred_begin(), e = oBB->pred_end(); it != e; ++it)
    if ((*it)->isEntryBlock() || !(*it)->hasNoPredecessors())
      edges.push_back({*it, it.getSuccessorIndex()});

  if (edges.empty()) {
    // An unreachable block never runs, so neither does its reverse. It still
    // needs a terminator; the entry's reverse block ends in the return.
    Block *entry = &oBB->getParent()->front();
    flow.createBranch(revBuilder, oBB->getParent()->getLoc(),
                      gutils->mapReverseModeBlocks.lookupOrNull(entry));
    return;
  }

  Location loc = oBB->rbegin()->getLoc();

  bool single = edges.size() == 1;
  Value cache, flag;
  if (!single) {
    cache = gutils->insertInit(gutils->getIndexCacheType());
    flag =
        enzyme::PopOp::create(revBuilder, loc, gutils->getIndexType(), cache);
    // If all of the edge only have 1 sucessors. Directly push and pop.
    // Otherwise, record the branches and cache value here for later build push.
    if (llvm::all_of(edges, [](auto edge) {
          return edge.first->getTerminator()->getNumSuccessors() == 1;
        })) {
      for (auto [idx, edge] : llvm::enumerate(edges)) {
        OpBuilder predBuilder(
            gutils->getNewFromOriginal(edge.first)->getTerminator());
        Value index =
            flow.createBranchIndex(predBuilder, loc, int64_t(idx) - 1);
        enzyme::PushOp::create(predBuilder, loc, cache, index);
      }
    } else {
      edgeCaches.push_back({oBB, cache, edges});
    }
  }

  SmallVector<Value, 1> diffes;
  for (auto arg : oBB->getArguments()) {
    if (!gutils->isConstantValue(arg) &&
        !cast<AutoDiffTypeInterface>(arg.getType()).isMutable()) {
      diffes.push_back(gutils->diffe(arg, revBuilder));
      gutils->zeroDiffe(arg, revBuilder);
      continue;
    }
    diffes.push_back(nullptr);
  }

  Block *defaultBlock = nullptr;
  SmallVector<Block *> blocks;
  SmallVector<int64_t> indices;
  for (auto [idx, edge] : llvm::enumerate(edges)) {
    auto [pred, succ] = edge;
    Block *reversePred = gutils->mapReverseModeBlocks.lookupOrNull(pred);
    int64_t predIndex = static_cast<int64_t>(idx) - 1;
    if (idx == 0) {
      defaultBlock = reversePred;
    } else {
      indices.push_back(predIndex);
      blocks.push_back(reversePred);
    }

    auto iface = dyn_cast<BranchOpInterface>(pred->getTerminator());
    assert(iface && "predecessor did not implement branch op interface");
    // Unpack the parameter from predecessor based on the poped value
    SuccessorOperands succOperands = iface.getSuccessorOperands(succ);
    unsigned produced = succOperands.getProducedOperandCount();
    for (auto [i, operand] :
         llvm::enumerate(succOperands.getForwardedOperands())) {
      Value diffe = diffes[produced + i];
      if (!diffe || gutils->isConstantValue(operand))
        continue;
      // The argument's adjoint flows back along this edge only when control
      // came in along it.
      Value to_prop = single ? diffe
                             : flow.createSelectOnBranchIndex(
                                   revBuilder, loc, flag, predIndex, diffe,
                                   cast<AutoDiffTypeInterface>(diffe.getType())
                                       .createNullValue(revBuilder, loc));
      gutils->addToDiffe(operand, to_prop, revBuilder);
    }
  }

  // Attach reverse mode terminator
  if (single)
    flow.createBranch(revBuilder, loc, defaultBlock);
  else
    flow.createSwitch(revBuilder, loc, flag, defaultBlock, indices, blocks);
}

void MEnzymeLogic::pushEdgeIndices(MGradientUtilsReverse *gutils,
                                   ArrayRef<EdgeIndexCache> edgeCaches) {
  const AutoDiffControlFlowInterface &flow = controlFlow(gutils);

  // For those branch has multiple distances, we append a parameter to indicate
  // its source. And push in the destinations. The reverse mode destination is
  // rewritten into brining the index it has.
  for (const EdgeIndexCache &edgeCache : edgeCaches) {
    Block *newBB = gutils->getNewFromOriginal(edgeCache.block);
    Location loc = edgeCache.block->getParent()->getLoc();
    BlockArgument index = newBB->addArgument(
        flow.getBranchIndexType(newBB->getParentOp()->getContext()), loc);
    OpBuilder blockBuilder = OpBuilder::atBlockBegin(newBB);
    enzyme::PushOp::create(blockBuilder, loc, edgeCache.cache, index);
    for (auto [idx, edge] : llvm::enumerate(edgeCache.edges)) {
      Operation *term = gutils->getNewFromOriginal(edge.first)->getTerminator();
      OpBuilder termBuilder(term);
      cast<BranchOpInterface>(term)
          .getSuccessorOperands(edge.second)
          .append(flow.createBranchIndex(termBuilder, loc, int64_t(idx) - 1));
    }
  }
}

LogicalResult MEnzymeLogic::differentiate(
    MGradientUtilsReverse *gutils, Region &oldRegion, Region &newRegion,
    llvm::function_ref<buildReturnFunction> buildFuncReturnOp,
    std::function<std::pair<Value, Value>(Type)> cacheCreator) {
  gutils->registerCacheCreatorHook(cacheCreator);

  gutils->createReverseModeBlocks(oldRegion, newRegion);

  bool valid = true;
  SmallVector<EdgeIndexCache> edgeCaches;
  for (auto &oBB : oldRegion) {
    Block *newBB = gutils->getNewFromOriginal(&oBB);
    Block *reverseBB = gutils->mapReverseModeBlocks.lookupOrNull(&oBB);
    handleReturns(&oBB, newBB, reverseBB, gutils);
    valid &= visitChildren(&oBB, reverseBB, gutils).succeeded();
    handlePredecessors(&oBB, newBB, reverseBB, gutils, buildFuncReturnOp,
                       edgeCaches);
  }
  // Every branch has its final operands now.
  pushEdgeIndices(gutils, edgeCaches);

  gutils->deregisterCacheCreatorHook(cacheCreator);
  return success(valid);
}

FunctionOpInterface MEnzymeLogic::CreateReverseDiff(
    FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
    std::vector<bool> returnPrimals, std::vector<bool> returnShadows,
    DerivativeMode mode, bool freeMemory, bool atomicAdd, size_t width,
    mlir::Type addedType, MFnTypeInfo type_args,
    std::vector<bool> overwritten_args, void *augmented, bool omp,
    llvm::StringRef postpasses, bool verifyPostPasses, bool strongZero,
    bool markReadonly) {

  if (fn.getFunctionBody().empty()) {
    fn.emitError() << "cannot differentiate a function without a body: "
                   << fn.getNameAttr() << "\n";
    return nullptr;
  }

  MReverseCacheKey tup = {fn,
                          retType,
                          constants,
                          returnPrimals,
                          returnShadows,
                          mode,
                          freeMemory,
                          atomicAdd,
                          static_cast<unsigned>(width),
                          addedType,
                          type_args,
                          overwritten_args,
                          omp};

  {
    auto cachedFn = ReverseCachedFunctions.find(tup);
    if (cachedFn != ReverseCachedFunctions.end())
      return cachedFn->second;
  }

  SmallVector<bool> returnPrimalsP(returnPrimals.begin(), returnPrimals.end());
  SmallVector<bool> returnShadowsP(returnShadows.begin(), returnShadows.end());

  MGradientUtilsReverse *gutils = MGradientUtilsReverse::CreateFromClone(
      *this, mode, width, fn, TA, type_args, returnPrimalsP, returnShadowsP,
      retType, constants, addedType, omp, postpasses, verifyPostPasses,
      strongZero);
  if (markReadonly) {
    markReadOnlyLoads(gutils->oldFunc, [&](Operation *origOp) {
      gutils->getNewFromOriginal(origOp)->setAttr(
          "enzyme.readonly", UnitAttr::get(origOp->getContext()));
    });
  }
  gutils->AtomicAdd = atomicAdd;

  ReverseCachedFunctions[tup] = gutils->newFunc;

  Region &oldRegion = gutils->oldFunc.getFunctionBody();
  Region &newRegion = gutils->newFunc.getFunctionBody();

  auto buildFuncReturnOp = [&](OpBuilder &builder, Block *oBB) {
    SmallVector<mlir::Value> retargs;
    for (auto [arg, returnPrimal] :
         llvm::zip(oBB->getTerminator()->getOperands(), returnPrimals)) {
      if (returnPrimal) {
        retargs.push_back(gutils->getNewFromOriginal(arg));
      }
    }
    for (auto [arg, cv] : llvm::zip(oBB->getArguments(), constants)) {
      if (cv == DIFFE_TYPE::OUT_DIFF) {
        retargs.push_back(gutils->diffe(arg, builder));
      }
    }

    Location loc = oBB->rbegin()->getLoc();
    if (auto iface = dyn_cast<enzyme::AutoDiffFunctionInterface>(*fn))
      iface.createReturn(builder, loc, retargs);
    else
      fn->emitError() << "this function operation does not implement "
                         "AutoDiffFunctionInterface";
    return;
  };

  gutils->forceAugmentedReturns();

  auto res =
      differentiate(gutils, oldRegion, newRegion, buildFuncReturnOp, nullptr);

  auto nf = gutils->newFunc;

  delete gutils;

  if (!res.succeeded())
    return nullptr;

  if (postpasses != "") {
    mlir::PassManager pm(nf->getContext());
    pm.enableVerifier(verifyPostPasses);
    std::string error_message;
    // llvm::raw_string_ostream error_stream(error_message);
    mlir::LogicalResult result = mlir::parsePassPipeline(postpasses, pm);
    if (mlir::failed(result)) {
      return nullptr;
    }

    if (!mlir::succeeded(pm.run(nf))) {
      return nullptr;
    }
  }

  return nf;
}
