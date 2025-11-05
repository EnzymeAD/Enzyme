#include "CloneFunction.h"
#include "Dialect/Ops.h"
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
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;
using namespace mlir::enzyme;

void handleReturns(Block *oBB, Block *newBB, Block *reverseBB,
                   MGradientUtilsReverse *gutils) {
  if (oBB->getNumSuccessors() == 0) {
    Operation *returnStatement = newBB->getTerminator();
    gutils->erase(returnStatement);

    OpBuilder forwardToBackwardBuilder(newBB, newBB->end());

    Operation *newBranchOp = cf::BranchOp::create(
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
         gutils->isConstantInstruction(op);
}

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
    ifaceOp.createShadowValues(augmentBuilder, gutils);
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
    llvm::function_ref<buildReturnFunction> buildReturnOp) {
  OpBuilder revBuilder(reverseBB, reverseBB->end());
  if (oBB->hasNoPredecessors()) {
    buildReturnOp(revBuilder, oBB);
  } else {
    Location loc = oBB->rbegin()->getLoc();
    // TODO remove dependency on CF dialect

    Value cache = gutils->insertInit(gutils->getIndexCacheType());

    Value flag =
        enzyme::PopOp::create(revBuilder, loc, gutils->getIndexType(), cache);

    Block *defaultBlock = nullptr;

    SmallVector<Block *> blocks;
    SmallVector<APInt> indices;

    OpBuilder newBuilder(newBB, newBB->begin());

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

    for (auto [idx, pred] : llvm::enumerate(oBB->getPredecessors())) {
      auto reversePred = gutils->mapReverseModeBlocks.lookupOrNull(pred);

      Block *newPred = gutils->getNewFromOriginal(pred);

      OpBuilder predecessorBuilder(newPred->getTerminator());

      Value pred_idx_c =
          arith::ConstantIntOp::create(predecessorBuilder, loc, idx - 1, 32);
      enzyme::PushOp::create(predecessorBuilder, loc, cache, pred_idx_c);

      if (idx == 0) {
        defaultBlock = reversePred;

      } else {
        indices.push_back(APInt(32, idx - 1));
        blocks.push_back(reversePred);
      }

      auto term = pred->getTerminator();
      if (auto iface = dyn_cast<BranchOpInterface>(term)) {
        for (auto &op : term->getOpOperands())
          if (auto blk_idx =
                  iface.getSuccessorBlockArgument(op.getOperandNumber()))
            if (!gutils->isConstantValue(op.get()) &&
                (*blk_idx).getOwner() == oBB) {
              auto idx = (*blk_idx).getArgNumber();
              if (diffes[idx]) {

                Value rev_idx_c =
                    arith::ConstantIntOp::create(revBuilder, loc, idx - 1, 32);

                auto to_prop = arith::SelectOp::create(
                    revBuilder, loc,
                    arith::CmpIOp::create(revBuilder, loc,
                                          arith::CmpIPredicate::eq, flag,
                                          rev_idx_c),
                    diffes[idx],
                    cast<AutoDiffTypeInterface>(diffes[idx].getType())
                        .createNullValue(revBuilder, loc));

                gutils->addToDiffe(op.get(), to_prop, revBuilder);
              }
            }
      } else {
        assert(0 && "predecessor did not implement branch op interface");
      }
    }

    cf::SwitchOp::create(revBuilder, loc, flag, defaultBlock, ArrayRef<Value>(),
                         ArrayRef<APInt>(indices), ArrayRef<Block *>(blocks),
                         SmallVector<ValueRange>(indices.size(), ValueRange()));
  }
}

LogicalResult MEnzymeLogic::differentiate(
    MGradientUtilsReverse *gutils, Region &oldRegion, Region &newRegion,
    llvm::function_ref<buildReturnFunction> buildFuncReturnOp,
    std::function<std::pair<Value, Value>(Type)> cacheCreator) {
  gutils->registerCacheCreatorHook(cacheCreator);
  auto scope = llvm::make_scope_exit(
      [&]() { gutils->deregisterCacheCreatorHook(cacheCreator); });

  gutils->createReverseModeBlocks(oldRegion, newRegion);

  bool valid = true;
  for (auto &oBB : oldRegion) {
    Block *newBB = gutils->getNewFromOriginal(&oBB);
    Block *reverseBB = gutils->mapReverseModeBlocks.lookupOrNull(&oBB);
    handleReturns(&oBB, newBB, reverseBB, gutils);
    valid &= visitChildren(&oBB, reverseBB, gutils).succeeded();
    handlePredecessors(&oBB, newBB, reverseBB, gutils, buildFuncReturnOp);
  }
  return success(valid);
}

FunctionOpInterface MEnzymeLogic::CreateReverseDiff(
    FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
    std::vector<bool> returnPrimals, std::vector<bool> returnShadows,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented,
    bool omp, llvm::StringRef postpasses, bool verifyPostPasses,
    bool strongZero) {

  if (fn.getFunctionBody().empty()) {
    llvm::errs() << fn << "\n";
    llvm_unreachable("Differentiating empty function");
  }

  MReverseCacheKey tup = {fn,
                          retType,
                          constants,
                          returnPrimals,
                          returnShadows,
                          mode,
                          freeMemory,
                          static_cast<unsigned>(width),
                          addedType,
                          type_args,
                          volatile_args,
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
    if (isa<LLVM::LLVMFuncOp>(fn)) {
      if (retargs.size() > 1) {
        Value packedReturns = packIntoStruct(retargs, builder, loc);
        LLVM::ReturnOp::create(builder, loc, packedReturns);
      } else {
        LLVM::ReturnOp::create(builder, loc, retargs);
      }
    } else {
      func::ReturnOp::create(builder, loc, retargs);
    }
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

static mlir::enzyme::ActivityAttr activityFromDiffeType(mlir::MLIRContext *ctx,
                                                        DIFFE_TYPE ty) {
  auto activity = mlir::enzyme::Activity::enzyme_active;
  switch (ty) {
  case DIFFE_TYPE::DUP_ARG:
    activity = mlir::enzyme::Activity::enzyme_dup;
  default:
    break;
  };
  return mlir::enzyme::ActivityAttr::get(ctx, activity);
}

FlatSymbolRefAttr MEnzymeLogic::CreateSplitModeDiff(
    FunctionOpInterface fn, std::vector<DIFFE_TYPE> retType,
    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
    std::vector<bool> returnPrimals, std::vector<bool> returnShadows,
    DerivativeMode mode, bool freeMemory, size_t width, mlir::Type addedType,
    MFnTypeInfo type_args, std::vector<bool> volatile_args, void *augmented,
    bool omp, llvm::StringRef postpasses, bool verifyPostPasses,
    bool strongZero) {

  SymbolTable symbolTable(SymbolTable::getNearestSymbolTable(fn));

  IRMapping originalToNew;
  std::map<Operation *, Operation *> originalToNewOps;

  SmallPtrSet<mlir::Value, 1> returnvals;
  SmallPtrSet<mlir::Value, 1> constant_values;
  SmallPtrSet<mlir::Value, 1> nonconstant_values;
  for (auto &&[arg, act] :
       llvm::zip(fn.getFunctionBody().getArguments(), constants)) {
    if (act == DIFFE_TYPE::CONSTANT)
      constant_values.insert(arg);
    else
      nonconstant_values.insert(arg);
  }

  SmallVector<bool> returnPrimalsP(returnPrimals.begin(), returnPrimals.end());
  SmallVector<bool> returnShadowsP(returnShadows.begin(), returnShadows.end());

  auto name = fn.getName();

  SmallVector<Attribute> argActivityAttrs;
  for (auto act : constants)
    argActivityAttrs.push_back(activityFromDiffeType(fn.getContext(), act));

  SmallVector<Attribute> retActivityAttrs;
  for (auto act : retType)
    retActivityAttrs.push_back(activityFromDiffeType(fn.getContext(), act));

  auto argActivityAttr = ArrayAttr::get(fn.getContext(), argActivityAttrs);
  auto retActivityAttr = ArrayAttr::get(fn.getContext(), retActivityAttrs);

  auto customRuleName = name + "_reverse_rule";
  SmallVector<char> nameBuf;

  auto ruleNameAttr =
      StringAttr::get(fn.getContext(), customRuleName.toStringRef(nameBuf));

  SmallVector<Type> argTys(
      cast<FunctionType>(fn.getFunctionType()).getInputs().begin(),
      cast<FunctionType>(fn.getFunctionType()).getInputs().end());

  OpBuilder builder(fn);
  auto customRule = enzyme::CustomReverseRuleOp::create(
      builder, fn.getLoc(), ruleNameAttr, TypeAttr::get(fn.getFunctionType()),
      argActivityAttr, retActivityAttr);

  Block *ruleBody = new Block();
  customRule.getBody().push_back(ruleBody);

  OpBuilder ruleBuilder(ruleBody, ruleBody->begin());

  SmallVector<Type> revInputTypes, revOutputTypes, primalInputTypes,
      primalOutputTypes;

  auto revFuncType =
      FunctionType::get(fn.getContext(), revInputTypes, revOutputTypes);
  auto primalFuncType =
      FunctionType::get(fn.getContext(), primalInputTypes, primalOutputTypes);

  auto reverse = enzyme::CustomReverseRuleReverseOp::create(
      ruleBuilder, fn.getLoc(), revFuncType);
  enzyme::YieldOp::create(ruleBuilder, fn.getLoc(), ValueRange{});

  ruleBuilder.setInsertionPoint(reverse);

  auto newFunc = cast<FunctionOpInterface>(fn->cloneWithoutRegions());
  cloneInto(&fn.getFunctionBody(), &newFunc.getFunctionBody(), originalToNew,
            originalToNewOps);

  Block *fnEntry = &newFunc.getFunctionBody().front();
  IRMapping invertedPointers;

  SmallVector<Type> newArgTys;

  int numDup = 0;
  for (auto [act, arg] : llvm::zip_equal(
           constants, fn.getFunctionBody().front().getArguments())) {
    newArgTys.push_back(arg.getType());
    if (act == DIFFE_TYPE::DUP_ARG) {
      numDup++;
      auto shadowType =
          cast<AutoDiffTypeInterface>(arg.getType()).getShadowType(width);
      auto shadow = fnEntry->insertArgument(arg.getArgNumber() + numDup,
                                            shadowType, arg.getLoc());
      newArgTys.push_back(shadowType);
      invertedPointers.map(arg, shadow);
    }
  }

  auto newFuncType =
      FunctionType::get(newFunc.getContext(), newArgTys,
                        cast<FunctionType>(fn.getFunctionType()).getResults());
  newFunc.setFunctionTypeAttr(TypeAttr::get(newFuncType));

  MGradientUtilsReverse *gutils = new MGradientUtilsReverse(
      *this, newFunc, fn, TA, invertedPointers, returnPrimalsP, returnShadowsP,
      constant_values, nonconstant_values, retType, constants, originalToNew,
      originalToNewOps, mode, width, omp, postpasses, verifyPostPasses,
      strongZero);

  gutils->createReverseModeBlocks(fn.getFunctionBody(), reverse.getBody());
  gutils->registerCacheCreatorHook([&](Type ty) -> std::pair<Value, Value> {
    Value cache = enzyme::InitOp::create(ruleBuilder, fn.getLoc(), ty);
    return {cache, cache};
  });
  gutils->registerGradientCreatorHook([&](Location loc, Type ty) -> Value {
    auto reverseEntry = &reverse.getBody().front();
    OpBuilder gBuilder(reverseEntry, reverseEntry->begin());
    return enzyme::InitOp::create(gBuilder, loc, ty);
  });

  bool valid = true;
  for (auto &oBB : fn.getFunctionBody()) {
    Block *newBB = gutils->getNewFromOriginal(&oBB);
    Block *reverseBB = gutils->mapReverseModeBlocks.lookupOrNull(&oBB);
    if (oBB.getNumSuccessors() == 0) {
      Operation *oTerm = oBB.getTerminator();
      for (auto [res, act] : llvm::zip_equal(oTerm->getOperands(), retType)) {
        if (act == DIFFE_TYPE::OUT_DIFF) {
          OpBuilder diffeBuilder(reverseBB, reverseBB->begin());
          auto diffe = reverseBB->addArgument(res.getType(), res.getLoc());
          gutils->setDiffe(res, diffe, diffeBuilder);
        }
      }
    }

    OpBuilder revBuilder(reverseBB, reverseBB->end());

    auto first = oBB.rbegin();
    first++;
    auto last = oBB.rend();
    for (auto it = first; it != last; ++it) {
      Operation *op = &*it;
      valid &= visitChild(op, revBuilder, gutils).succeeded();
    }

    if (oBB.isEntryBlock()) {
      SmallVector<Value> toYield;
      OpBuilder rBuilder(reverseBB, reverseBB->end());
      for (auto [act, arg] : llvm::zip_equal(
               constants, fn.getFunctionBody().front().getArguments())) {
        if (act == DIFFE_TYPE::OUT_DIFF) {
          toYield.push_back(gutils->diffe(arg, rBuilder));
        }
      }
      enzyme::YieldOp::create(rBuilder, fn.getLoc(), toYield);
    }
  }

  ruleBuilder.setInsertionPoint(reverse);
  auto augmentedPrimal = enzyme::CustomReverseRuleAugmentedPrimalOp::create(
      ruleBuilder, fn.getLoc(), primalFuncType);
  augmentedPrimal.getBody().takeBody(newFunc.getFunctionBody());
  for (Block &b : augmentedPrimal.getBody()) {
    if (b.getNumSuccessors() == 0) {
      Operation *term = b.getTerminator();
      OpBuilder builder(term);
      enzyme::YieldOp::create(builder, term->getLoc(), term->getOperands());
      term->erase();
    }
  }

  delete gutils;

  newFunc->erase();

  return FlatSymbolRefAttr::get(ruleNameAttr);
}
