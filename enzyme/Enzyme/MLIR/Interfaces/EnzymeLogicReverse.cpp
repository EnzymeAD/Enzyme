#include "Analysis/DataFlowAliasAnalysis.h"
#include "Dialect/Ops.h"
#include "Interfaces/AutoDiffControlFlowInterface.h"
#include "Interfaces/AutoDiffOpInterface.h"
#include "Interfaces/AutoDiffTypeInterface.h"
#include "mlir/IR/Dominance.h"
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

#ifdef ENZYME_CLANG_HAS_CIR
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#endif

using namespace mlir;
using namespace mlir::enzyme;

// Have the derivative's dialect spells the control flow.
static const AutoDiffControlFlowInterface &
controlFlow(MGradientUtilsReverse *gutils) {
  return getAutoDiffControlFlow(gutils->newFunc.getOperation());
}

void handleReturns(Block *oBB, Block *newBB, Block *reverseBB,
                   MGradientUtilsReverse *gutils) {
  if (isFunctionReturnLike(oBB->getTerminator())) {
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
//    3. Caller doesn't pass active and inactive value.
static bool isFullyInactive(Operation *op, MGradientUtils *gutils) {
  // A call whose memory effects are unknown may free or write through an
  // active pointer it is passed, which its shadow has to follow, even when
  // nothing about it is active itself.
  if (isa<CallOpInterface>(op) && !isa<MemoryEffectOpInterface>(op) &&
      llvm::any_of(op->getOperands(), [gutils](Value v) {
        auto iface = dyn_cast<AutoDiffTypeInterface>(v.getType());
        return iface && iface.isMutable() && !gutils->isConstantValue(v);
      }))
    return false;
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
      gutils->recordReverseDefinitions(op->getResults(), reverseBB);
    }
  }
  return success(valid);
}

void MEnzymeLogic::handlePredecessors(
    Block *oBB, Block *newBB, Block *reverseBB, MGradientUtilsReverse *gutils,
    llvm::function_ref<buildReturnFunction> buildReturnOp,
    SmallVectorImpl<EdgeIndexCache> &edgeCaches) {
  gutils->recordReverseDefinitions(oBB->getArguments(), reverseBB);

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
    bool markReadonly, ReverseSweepEntries *entries) {

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

  // A derivative made to be split is taken apart, so it is neither shared
  // nor kept.
  if (!entries) {
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
  gutils->splitting = entries != nullptr;

  if (!entries)
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
  gutils->finishReverseValues();

  // Record all places in forward pass splitted
  if (entries && res.succeeded())
    for (Block &oBB : oldRegion)
      if (isFunctionReturnLike(oBB.getTerminator())) {
        Operation *ret = oBB.getTerminator();
        Operation *exit = gutils->originalToNewFnOps[ret];
        SmallVector<Value> returned;
        for (Value v : ret->getOperands())
          returned.push_back(gutils->getNewFromOriginal(v));
        OpBuilder builder(exit);
        // A pointer the caller wants the shadow of but that is inactive here
        // has itself for its shadow, as one stored into differentiated
        // memory does (see oputils::inactiveStoredValueShadow): not null,
        // which the caller would read through.
        for (auto [v, act] : llvm::zip(ret->getOperands(), retType))
          if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED)
            returned.push_back(gutils->isConstantValue(v)
                                   ? gutils->getNewFromOriginal(v)
                                   : gutils->invertPointerM(v, builder));
        entries->exits.push_back({exit, std::move(returned)});
      }

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

// Gives `f` the arguments `inputs` and results `results`, keeping the
// attributes of the first `keptArgs` arguments.
static void setSignature(FunctionOpInterface f, TypeRange inputs,
                         TypeRange results, unsigned keptArgs) {
  SmallVector<DictionaryAttr> argAttrs;
  f.getAllArgAttrs(argAttrs);
  argAttrs.resize(keptArgs);
  argAttrs.resize(inputs.size(), DictionaryAttr::get(f.getContext()));
  Type type;
#ifdef ENZYME_CLANG_HAS_CIR
  // cir.func has one optional result, null for void, which cloneTypeWith cannot
  // express.
  if (auto cirTy = dyn_cast<cir::FuncType>(f.getFunctionType())) {
    assert(results.size() <= 1 && "cir.func returns at most one value");
    type = cir::FuncType::get(f.getContext(), llvm::to_vector(inputs),
                              results.empty() ? Type() : results.front(),
                              cirTy.isVarArg());
  } else
#endif
    type = f.cloneTypeWith(inputs, results);
  f.setType(type);
  f.setAllArgAttrs(argAttrs);
  f.setAllResultAttrs(SmallVector<DictionaryAttr>(
      results.size(), DictionaryAttr::get(f.getContext())));
}

// Erases `blocks` of a region, which may refer to each other; nothing outside
// them may still use what they define.
static void eraseBlocks(ArrayRef<Block *> blocks) {
  for (Block *block : blocks)
    block->dropAllReferences();
  for (Block *block : blocks) {
    assert(llvm::all_of(block->getOperations(),
                        [](Operation &op) { return op.use_empty(); }));
    block->dropAllDefinedValueUses();
    block->erase();
  }
}

// The block of `region` that holds `op`, perhaps in a nested region.
static Block *topLevelBlock(Region &region, Operation *op) {
  return region.findAncestorOpInRegion(*op)->getBlock();
}

static Block *topLevelBlock(Region &region, Value v) {
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Operation *owner = arg.getOwner()->getParentOp();
    if (arg.getOwner()->getParent() == &region)
      return arg.getOwner();
    return topLevelBlock(region, owner);
  }
  return topLevelBlock(region, v.getDefiningOp());
}

FailureOr<MEnzymeLogic::SplitReverseDiff> MEnzymeLogic::CreateSplitReverseDiff(
    FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA, size_t width,
    MFnTypeInfo type_args, bool omp, llvm::StringRef postpasses,
    bool verifyPostPasses, bool strongZero) {
  MReverseCacheKey key = {fn,
                          retType,
                          constants,
                          {},
                          {},
                          DerivativeMode::ReverseModePrimal,
                          /*freeMemory*/ true,
                          /*atomicAdd*/ false,
                          static_cast<unsigned>(width),
                          Type(),
                          type_args,
                          {},
                          omp,
                          strongZero};
  if (auto found = SplitCachedFunctions.find(key);
      found != SplitCachedFunctions.end())
    return found->second;
  if (!splitting.insert(fn).second)
    return fn.emitError()
           << "cannot differentiate a recursive call to a function that "
              "touches memory in reverse mode";
  llvm::scope_exit unmark([&] { splitting.erase(fn); });

  // The combined derivative, taken apart below.
  ReverseSweepEntries entries;
  std::vector<bool> returnPrimals(fn.getNumResults(), false);
  std::vector<bool> returnShadows(fn.getNumResults(), false);
  FunctionOpInterface combined = CreateReverseDiff(
      fn, retType, constants, TA, returnPrimals, returnShadows,
      DerivativeMode::ReverseModeGradient, /*freeMemory*/ true,
      /*atomicAdd*/ false, width, Type(), type_args,
      std::vector<bool>(constants.size(), true), /*augmented*/ nullptr, omp,
      /*postpasses*/ "", verifyPostPasses, strongZero,
      /*markReadonly*/ false, &entries);
  if (!combined)
    return failure();
  llvm::scope_exit eraseCombined([&] { combined->erase(); });

  Region &body = combined.getFunctionBody();
  Block *entry = &body.front();
  MLIRContext *ctx = fn.getContext();
  const AutoDiffControlFlowInterface &flow =
      getAutoDiffControlFlow(combined.getOperation());

  // The reverse sweep is what the forward sweep's exits reach.
  SmallPtrSet<Block *, 16> reverse;
  SmallVector<Block *> todo;
  // The start of each reverse sweep region.
  for (auto &[exit, returned] : entries.exits)
    todo.push_back(exit->getSuccessor(0));
  while (!todo.empty()) {
    Block *block = todo.pop_back_val();
    if (reverse.insert(block).second)
      llvm::append_range(todo, block->getSuccessors());
  }
  auto inReverse = [&](Block *block) { return reverse.contains(block); };

  // The reverse of a path that never returns, as one ending in cir.trap, is
  // reached from neither sweep; it would otherwise be left in the forward
  // one.
  SmallPtrSet<Block *, 16> live(reverse.begin(), reverse.end());
  todo.push_back(entry);
  while (!todo.empty()) {
    Block *block = todo.pop_back_val();
    if (live.insert(block).second)
      llvm::append_range(todo, block->getSuccessors());
  }
  SmallVector<Block *> dead;
  for (Block &block : body)
    if (!live.contains(&block))
      dead.push_back(&block);
  eraseBlocks(dead);

  // The gradients belong to the reverse sweep and the caches to the tape; both
  // are made at the top of the entry block.
  SmallVector<enzyme::InitOp> gradients, caches;
  for (auto init : entry->getOps<enzyme::InitOp>()) {
    if (isa<enzyme::GradientType>(init.getType()))
      gradients.push_back(init);
    else if (isa<enzyme::CacheType>(init.getType()))
      caches.push_back(init);
  }
  for (enzyme::InitOp gradient : gradients)
    for (Operation *user : gradient->getUsers())
      if (!inReverse(topLevelBlock(body, user)) &&
          !(isa<enzyme::SetOp>(user) && user->getBlock() == entry))
        return user->emitError()
               << "cannot split a derivative whose forward sweep uses a "
                  "gradient";

  // Collect tapes cross sweep.
  llvm::SetVector<Value> crossing;
  for (Block *block : reverse)
    block->walk([&](Operation *op) {
      for (Value v : op->getOperands()) {
        if (auto arg = dyn_cast<BlockArgument>(v);
            arg && arg.getOwner() == entry)
          continue;
        if (auto init = v.getDefiningOp<enzyme::InitOp>();
            init && init->getBlock() == entry)
          continue;
        if (!inReverse(topLevelBlock(body, v)))
          crossing.insert(v);
      }
    });

  auto rematerializable = [&](Operation *def) {
    if (!def)
      return false;
    if (def->hasTrait<OpTrait::ConstantLike>() && def->getNumOperands() == 0)
      return true;
    // A stack slot only the reverse sweep uses, as one the derivative makes
    // for a loop of its own, is made again in the reverse function's frame:
    // the augmented function's is gone by then.
    if (isa<AllocaLikeInterface>(def) && def->getNumOperands() == 0 &&
        llvm::all_of(def->getUsers(), [&](Operation *user) {
          return inReverse(topLevelBlock(body, user));
        }))
      return true;
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def))
      if (cast->getNumOperands() == 1)
        if (auto init = cast->getOperand(0).getDefiningOp<enzyme::InitOp>())
          return init->getBlock() == entry &&
                 isa<enzyme::CacheType>(init.getType());
    return false;
  };
  SmallVector<Value> rematerialized, taped;
  for (Value v : crossing) {
    if (rematerializable(v.getDefiningOp()))
      rematerialized.push_back(v);
    else
      taped.push_back(v);
  }

  SmallVector<Type> tapeTypes;
  SmallVector<std::pair<unsigned, std::string>> deferredFrees;
  for (enzyme::InitOp cache : caches) {
    if (auto dealloc = cache->getAttrOfType<StringAttr>(
            MGradientUtilsReverse::deferredFreeAttrName))
      deferredFrees.push_back({tapeTypes.size(), dealloc.str()});
    tapeTypes.push_back(cache.getType());
  }
  for (Value v : taped)
    tapeTypes.push_back(enzyme::CacheType::get(ctx, v.getType()));
  bool manyExits = entries.exits.size() > 1;
  Type indexType = flow.getBranchIndexType(ctx);
  if (manyExits)
    tapeTypes.push_back(enzyme::CacheType::get(ctx, indexType));

  unsigned numResultAdjoints = llvm::count(retType, DIFFE_TYPE::OUT_DIFF);
  unsigned numArgs = combined.getNumArguments();
  auto fnIface = cast<AutoDiffFunctionInterface>(combined.getOperation());
  SmallVector<Type> tapeArgTypes;
  for (Type t : tapeTypes)
    tapeArgTypes.push_back(fnIface.getTapeType(t));
  // Appended to the entry block; the signature is set once it is final.
  auto addTapeArguments = [&](FunctionOpInterface f) {
    Block &fEntry = f.getFunctionBody().front();
    unsigned at = fEntry.getNumArguments();
    for (Type t : tapeArgTypes)
      fEntry.addArgument(t, f.getLoc());
    OpBuilder builder = OpBuilder::atBlockBegin(&fEntry);
    SmallVector<Value> tape;
    for (auto [arg, t] :
         llvm::zip(fEntry.getArguments().drop_front(at), tapeTypes))
      tape.push_back(arg.getType() == t
                         ? Value(arg)
                         : UnrealizedConversionCastOp::create(
                               builder, f.getLoc(), t, ValueRange{arg})
                               .getResult(0));
    return tape;
  };
  auto symbols = SymbolTable(SymbolTable::getNearestSymbolTable(fn));

  // The augmented function returns the primal results and the shadows of
  // the pointers among them, packed as its kind of function packs results.
  SmallVector<Type> augResults(fn.getResultTypes());
  for (auto [t, act] : llvm::zip(fn.getResultTypes(), retType))
    if (act == DIFFE_TYPE::DUP_ARG || act == DIFFE_TYPE::DUP_NONEED)
      augResults.push_back(cast<AutoDiffTypeInterface>(t).getShadowType(width));
  fnIface.transformResultTypes(augResults);

  IRMapping toAug;
  auto augmented = cast<FunctionOpInterface>(combined->clone(toAug));
  {
    SmallVector<Value> tape = addTapeArguments(augmented);
    for (auto [cache, arg] : llvm::zip(caches, tape))
      toAug.lookup(cache.getResult()).replaceAllUsesWith(arg);
    ArrayRef<Value> tapedCaches =
        ArrayRef(tape).slice(caches.size(), taped.size());
    DominanceInfo dom(combined);
    for (auto [index, exitAndReturned] : llvm::enumerate(entries.exits)) {
      auto &[exit, returned] = exitAndReturned;
      Operation *augExit = toAug.lookup(exit);
      OpBuilder builder(augExit);
      Location loc = augExit->getLoc();
      for (auto [v, cache] : llvm::zip(taped, tapedCaches)) {
        Value value = toAug.lookup(v);
        // On a way out it does not reach, nothing reads it.
        if (!dom.properlyDominates(v, exit)) {
          auto iface = dyn_cast<AutoDiffTypeInterface>(v.getType());
          if (!iface)
            return exit->emitError()
                   << "cannot split a derivative that needs a value of "
                   << v.getType() << " on one of its ways out only";
          value = iface.createNullValue(builder, loc);
        }
        enzyme::PushOp::create(builder, loc, cache, value);
      }
      if (manyExits)
        enzyme::PushOp::create(builder, loc, tape.back(),
                               flow.createBranchIndex(builder, loc, index));
      SmallVector<Value> results;
      for (Value v : returned)
        results.push_back(toAug.lookup(v));
      cast<AutoDiffFunctionInterface>(augmented.getOperation())
          .createReturn(builder, loc, results);
      augExit->erase();
    }
    SmallVector<Block *> dead;
    for (Block *block : reverse)
      dead.push_back(toAug.lookup(block));
    eraseBlocks(dead);
    for (enzyme::InitOp init :
         llvm::concat<enzyme::InitOp>(gradients, caches)) {
      Operation *augInit = toAug.lookup(init.getOperation());
      for (Operation *user : llvm::make_early_inc_range(augInit->getUsers()))
        user->erase();
      augInit->erase();
    }

    // Remove adjoints there
    Block &augEntry = augmented.getFunctionBody().front();
    llvm::BitVector adjoints(augEntry.getNumArguments());
    adjoints.set(numArgs - numResultAdjoints, numArgs);
    augEntry.eraseArguments(adjoints);
    setSignature(augmented, augEntry.getArgumentTypes(), augResults,
                 numArgs - numResultAdjoints);
    augmented.setName(("augmented_" + fn.getName()).str());
    symbols.insert(augmented);
  }

  IRMapping toRev;
  auto reverseFn = cast<FunctionOpInterface>(combined->clone(toRev));
  {
    Region &revBody = reverseFn.getFunctionBody();
    Block *revEntry = &revBody.front();
    Block *forwardEntry = new Block();
    revBody.getBlocks().insertAfter(Region::iterator(revEntry), forwardEntry);
    forwardEntry->getOperations().splice(forwardEntry->end(),
                                         revEntry->getOperations());
    SmallVector<Value> tape = addTapeArguments(reverseFn);
    OpBuilder builder = OpBuilder::atBlockEnd(revEntry);
    Location loc = reverseFn.getLoc();
    // Value belongs to forward only is removed later.
    for (enzyme::InitOp gradient : gradients) {
      Operation *init = toRev.lookup(gradient.getOperation());
      init->moveBefore(revEntry, revEntry->end());
      for (Operation *user : llvm::make_early_inc_range(init->getUsers()))
        if (user->getBlock() == forwardEntry) {
          Operation *zero = user->getOperand(1).getDefiningOp();
          if (zero && zero->getBlock() == forwardEntry)
            user->setOperand(1, builder.clone(*zero)->getResult(0));
          user->moveBefore(revEntry, revEntry->end());
        }
    }
    for (auto [cache, arg] : llvm::zip(caches, tape))
      toRev.lookup(cache.getResult()).replaceAllUsesWith(arg);
    SmallPtrSet<Block *, 16> revReverse;
    for (Block *block : reverse)
      revReverse.insert(toRev.lookup(block));
    auto usedInReverse = [&](OpOperand &use) {
      return revReverse.contains(topLevelBlock(revBody, use.getOwner()));
    };
    for (Value v : rematerialized) {
      Value copy =
          builder.clone(*toRev.lookup(v).getDefiningOp())->getResult(0);
      toRev.lookup(v).replaceUsesWithIf(copy, usedInReverse);
    }
    for (auto [v, cache] :
         llvm::zip(taped, ArrayRef(tape).slice(caches.size(), taped.size()))) {
      Value popped = enzyme::PopOp::create(builder, loc, v.getType(), cache);
      toRev.lookup(v).replaceUsesWithIf(popped, usedInReverse);
    }

    // Make those new entries as a switch function from different source.
    SmallVector<Block *> exitTargets;
    for (auto &[exit, returned] : entries.exits)
      exitTargets.push_back(toRev.lookup(exit->getSuccessor(0)));
    if (manyExits) {
      Value flag = enzyme::PopOp::create(builder, loc, indexType, tape.back());
      SmallVector<int64_t> indices;
      for (int64_t i = 1, e = exitTargets.size(); i < e; ++i)
        indices.push_back(i);
      flow.createSwitch(builder, loc, flag, exitTargets.front(), indices,
                        ArrayRef(exitTargets).drop_front());
    } else {
      flow.createBranch(builder, loc, exitTargets.front());
    }

    SmallVector<Block *> dead;
    for (Block &block : revBody)
      if (&block != revEntry && !revReverse.contains(&block))
        dead.push_back(&block);
    eraseBlocks(dead);
    setSignature(reverseFn, revEntry->getArgumentTypes(),
                 combined.getResultTypes(), numArgs);
    symbols.insert(reverseFn);
  }

  // The combined derivative goes; the reverse function takes its name.
  StringAttr name = combined.getNameAttr();
  eraseCombined.release();
  symbols.erase(combined);
  if (failed(symbols.rename(reverseFn, name)))
    return failure();

  if (postpasses != "")
    for (FunctionOpInterface f : {augmented, reverseFn}) {
      PassManager pm(ctx);
      pm.enableVerifier(verifyPostPasses);
      if (failed(parsePassPipeline(postpasses, pm)) ||
          failed(pm.run(f.getOperation())))
        return failure();
    }

  SplitReverseDiff split{augmented, reverseFn, tapeTypes, deferredFrees};
  SplitCachedFunctions[key] = split;
  return split;
}
