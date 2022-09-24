//===- EnzymeMLIRPass.cpp - //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//
//#include "PassDetails.h"

#include "../../EnzymeLogic.h"
#include "Dialect/Ops.h"
#include "PassDetails.h"
#include "Passes/Passes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "enzyme"

using namespace mlir;
using namespace enzyme;

inline bool operator<(const mlir::Type lhs, mlir::Type rhs) {
  return lhs.getImpl() < rhs.getImpl();
}

class MFnTypeInfo {
public:
  inline bool operator<(const MFnTypeInfo &rhs) const { return false; }
};

class MTypeAnalysis {
public:
  MFnTypeInfo getAnalyzedTypeInfo(FunctionOpInterface op) const {}
};

Type getShadowType(Type T, unsigned width) { return T; }

mlir::FunctionType getFunctionTypeForClone(
    mlir::FunctionType FTy, DerivativeMode mode, unsigned width,
    mlir::Type additionalArg, llvm::ArrayRef<DIFFE_TYPE> constant_args,
    bool diffeReturnArg, ReturnType returnValue, DIFFE_TYPE returnType) {
  SmallVector<mlir::Type, 4> RetTypes;
  if (returnValue == ReturnType::ArgsWithReturn ||
      returnValue == ReturnType::Return) {
    assert(FTy.getNumResults() == 1);
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  } else if (returnValue == ReturnType::ArgsWithTwoReturns ||
             returnValue == ReturnType::TwoReturns) {
    assert(FTy.getNumResults() == 1);
    RetTypes.push_back(FTy.getResult(0));
    if (returnType != DIFFE_TYPE::CONSTANT &&
        returnType != DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(FTy.getResult(0), width));
    } else {
      RetTypes.push_back(FTy.getResult(0));
    }
  }

  SmallVector<mlir::Type, 4> ArgTypes;

  // The user might be deleting arguments to the function by specifying them in
  // the VMap.  If so, we need to not add the arguments to the arg ty vector
  unsigned argno = 0;

  for (auto I : FTy.getInputs()) {
    ArgTypes.push_back(I);
    if (constant_args[argno] == DIFFE_TYPE::DUP_ARG ||
        constant_args[argno] == DIFFE_TYPE::DUP_NONEED) {
      ArgTypes.push_back(getShadowType(I, width));
    } else if (constant_args[argno] == DIFFE_TYPE::OUT_DIFF) {
      RetTypes.push_back(getShadowType(I, width));
    }
    ++argno;
  }

  if (diffeReturnArg) {
    ArgTypes.push_back(getShadowType(FTy.getResult(0), width));
  }
  if (additionalArg) {
    ArgTypes.push_back(additionalArg);
  }

  OpBuilder builder(FTy.getContext());
  if (returnValue == ReturnType::TapeAndTwoReturns ||
      returnValue == ReturnType::TapeAndReturn ||
      returnValue == ReturnType::Tape) {
    RetTypes.insert(RetTypes.begin(),
                    LLVM::LLVMPointerType::get(builder.getIntegerType(8)));
  }

  // Create a new function type...
  return builder.getFunctionType(ArgTypes, RetTypes);
}

void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap);
void cloneInto(Region *src, Region *dest, BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap) {
  cloneInto(src, dest, dest->end(), mapper, opMap);
}
Operation *clone(Operation *src, BlockAndValueMapping &mapper,
                 Operation::CloneOptions options,
                 std::map<Operation *, Operation *> &opMap) {
  SmallVector<Value, 8> operands;
  SmallVector<Block *, 2> successors;

  // Remap the operands.
  if (options.shouldCloneOperands()) {
    operands.reserve(src->getNumOperands());
    for (auto opValue : src->getOperands())
      operands.push_back(mapper.lookupOrDefault(opValue));
  }

  // Remap the successors.
  successors.reserve(src->getNumSuccessors());
  for (Block *successor : src->getSuccessors())
    successors.push_back(mapper.lookupOrDefault(successor));

  // Create the new operation.
  auto *newOp =
      src->create(src->getLoc(), src->getName(), src->getResultTypes(),
                  operands, src->getAttrs(), successors, src->getNumRegions());

  // Clone the regions.
  if (options.shouldCloneRegions()) {
    for (unsigned i = 0; i != src->getNumRegions(); ++i)
      cloneInto(&src->getRegion(i), &newOp->getRegion(i), mapper, opMap);
  }

  // Remember the mapping of any results.
  for (unsigned i = 0, e = src->getNumResults(); i != e; ++i)
    mapper.map(src->getResult(i), newOp->getResult(i));

  opMap[src] = newOp;
  return newOp;
}
/// Clone this region into 'dest' before the given position in 'dest'.
void cloneInto(Region *src, Region *dest, Region::iterator destPos,
               BlockAndValueMapping &mapper,
               std::map<Operation *, Operation *> &opMap) {
  assert(src);
  assert(dest && "expected valid region to clone into");
  assert(src != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (src->empty())
    return;

  // The below clone implementation takes special care to be read only for the
  // sake of multi threading. That essentially means not adding any uses to any
  // of the blocks or operation results contained within this region as that
  // would lead to a write in their use-def list. This is unavoidable for
  // 'Value's from outside the region however, in which case it is not read
  // only. Using the BlockAndValueMapper it is possible to remap such 'Value's
  // to ones owned by the calling thread however, making it read only once
  // again.

  // First clone all the blocks and block arguments and map them, but don't yet
  // clone the operations, as they may otherwise add a use to a block that has
  // not yet been mapped
  for (Block &block : *src) {
    Block *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest->getBlocks().insert(destPos, newBlock);
  }

  auto newBlocksRange =
      llvm::make_range(Region::iterator(mapper.lookup(&src->front())), destPos);

  // Now follow up with creating the operations, but don't yet clone their
  // regions, nor set their operands. Setting the successors is safe as all have
  // already been mapped. We are essentially just creating the operation results
  // to be able to map them.
  // Cloning the operands and region as well would lead to uses of operations
  // not yet mapped.
  auto cloneOptions =
      Operation::CloneOptions::all().cloneRegions(false).cloneOperands(false);
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    Block &sourceBlock = std::get<0>(zippedBlocks);
    Block &clonedBlock = std::get<1>(zippedBlocks);
    // Clone and remap the operations within this block.
    for (Operation &op : sourceBlock) {
      clonedBlock.push_back(clone(&op, mapper, cloneOptions, opMap));
    }
  }

  // Finally now that all operation results have been mapped, set the operands
  // and clone the regions.
  SmallVector<Value> operands;
  for (auto zippedBlocks : llvm::zip(*src, newBlocksRange)) {
    for (auto ops :
         llvm::zip(std::get<0>(zippedBlocks), std::get<1>(zippedBlocks))) {
      Operation &source = std::get<0>(ops);
      Operation &clone = std::get<1>(ops);

      operands.resize(source.getNumOperands());
      llvm::transform(
          source.getOperands(), operands.begin(),
          [&](Value operand) { return mapper.lookupOrDefault(operand); });
      clone.setOperands(operands);

      for (auto regions : llvm::zip(source.getRegions(), clone.getRegions()))
        cloneInto(&std::get<0>(regions), &std::get<1>(regions), mapper, opMap);
    }
  }
}

mlir::func::FuncOp CloneFunctionWithReturns(
    DerivativeMode mode, unsigned width, FunctionOpInterface F,
    BlockAndValueMapping &ptrInputs, ArrayRef<DIFFE_TYPE> constant_args,
    SmallPtrSetImpl<mlir::Value> &constants,
    SmallPtrSetImpl<mlir::Value> &nonconstants,
    SmallPtrSetImpl<mlir::Value> &returnvals, ReturnType returnValue,
    DIFFE_TYPE returnType, Twine name, BlockAndValueMapping &VMap,
    std::map<Operation *, Operation *> &OpMap, bool diffeReturnArg,
    mlir::Type additionalArg) {
  assert(!F.getBody().empty());
  // F = preprocessForClone(F, mode);
  // llvm::ValueToValueMapTy VMap;
  auto FTy = getFunctionTypeForClone(
      F.getFunctionType().cast<mlir::FunctionType>(), mode, width,
      additionalArg, constant_args, diffeReturnArg, returnValue, returnType);

  /*
  for (Block &BB : F.getBody().getBlocks()) {
    if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (auto rv = ri->getReturnValue()) {
        returnvals.insert(rv);
      }
    }
  }
  */

  // Create the new function...
  auto NewF = mlir::func::FuncOp::create(F.getLoc(), name.str(), FTy);
  ((Operation *)F)->getParentOfType<ModuleOp>().push_back(NewF);
  SymbolTable::setSymbolVisibility(NewF, SymbolTable::Visibility::Private);

  cloneInto(&F.getBody(), &NewF.getBody(), VMap, OpMap);

  {
    auto &blk = NewF.getBody().front();
    for (ssize_t i = constant_args.size() - 1; i >= 0; i--) {
      mlir::Value oval = F.getBody().front().getArgument(i);
      if (constant_args[i] == DIFFE_TYPE::CONSTANT)
        constants.insert(oval);
      else
        nonconstants.insert(oval);
      if (constant_args[i] == DIFFE_TYPE::DUP_ARG ||
          constant_args[i] == DIFFE_TYPE::DUP_NONEED) {
        mlir::Value val = blk.getArgument(i);
        mlir::Value dval;
        if (i == constant_args.size() - 1)
          dval = blk.addArgument(val.getType(), val.getLoc());
        else
          dval = blk.insertArgument(blk.args_begin() + i + 1, val.getType(),
                                    val.getLoc());
        ptrInputs.map(oval, dval);
      }
    }
  }

  return NewF;
}

class MTypeResults {
public:
  // TODO
  TypeTree getReturnAnalysis() { return TypeTree(); }
};

class MEnzymeLogic;

class MGradientUtils {
public:
  // From CacheUtility
  mlir::func::FuncOp newFunc;

  MEnzymeLogic &Logic;
  bool AtomicAdd;
  DerivativeMode mode;
  FunctionOpInterface oldFunc;
  BlockAndValueMapping invertedPointers;
  BlockAndValueMapping originalToNewFn;
  std::map<Operation *, Operation *> originalToNewFnOps;

  MTypeAnalysis &TA;
  MTypeResults TR;
  bool omp;

  unsigned width;
  ArrayRef<DIFFE_TYPE> ArgDiffeTypes;

  mlir::Value getNewFromOriginal(const mlir::Value originst) const {
    if (!originalToNewFn.contains(originst)) {
      llvm::errs() << oldFunc << "\n";
      llvm::errs() << newFunc << "\n";
      llvm::errs() << originst << "\n";
      llvm_unreachable("Could not get new val from original");
    }
    return originalToNewFn.lookupOrNull(originst);
  }
  mlir::Block *getNewFromOriginal(mlir::Block *originst) const {
    if (!originalToNewFn.contains(originst)) {
      llvm::errs() << oldFunc << "\n";
      llvm::errs() << newFunc << "\n";
      llvm::errs() << originst << "\n";
      llvm_unreachable("Could not get new blk from original");
    }
    return originalToNewFn.lookupOrNull(originst);
  }
  Operation *getNewFromOriginal(Operation *originst) const {
    auto found = originalToNewFnOps.find(originst);
    if (found == originalToNewFnOps.end()) {
      llvm::errs() << oldFunc << "\n";
      llvm::errs() << newFunc << "\n";
      for (auto &pair : originalToNewFnOps) {
        llvm::errs() << " map[" << pair.first << "] = " << pair.second << "\n";
        // llvm::errs() << " map[" << pair.first << "] = " << pair.second << "
        // -- " << *pair.first << " " << *pair.second << "\n";
      }
      llvm::errs() << originst << " - " << *originst << "\n";
      llvm_unreachable("Could not get new op from original");
    }
    return found->second;
  }
  mlir::Type getShadowType(mlir::Type T) const {
    return ::getShadowType(T, width);
  }
  MGradientUtils(MEnzymeLogic &Logic, mlir::func::FuncOp newFunc_,
                 FunctionOpInterface oldFunc_, MTypeAnalysis &TA_,
                 MTypeResults TR_, BlockAndValueMapping &invertedPointers_,
                 const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                 const SmallPtrSetImpl<mlir::Value> &activevals_,
                 DIFFE_TYPE ReturnActivity, ArrayRef<DIFFE_TYPE> ArgDiffeTypes_,
                 BlockAndValueMapping &originalToNewFn_,
                 std::map<Operation *, Operation *> &originalToNewFnOps_,
                 DerivativeMode mode, unsigned width, bool omp)
      : newFunc(newFunc_), Logic(Logic), mode(mode), oldFunc(oldFunc_), TA(TA_),
        TR(TR_), omp(omp), width(width), ArgDiffeTypes(ArgDiffeTypes_),
        originalToNewFn(originalToNewFn_),
        originalToNewFnOps(originalToNewFnOps_),
        invertedPointers(invertedPointers_) {

    /*
    for (BasicBlock &BB : *oldFunc) {
      for (Instruction &I : BB) {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          originalCalls.push_back(CI);
        }
      }
    }
    */

    /*
    for (BasicBlock &oBB : *oldFunc) {
      for (Instruction &oI : oBB) {
        newToOriginalFn[originalToNewFn[&oI]] = &oI;
      }
      newToOriginalFn[originalToNewFn[&oBB]] = &oBB;
    }
    for (Argument &oArg : oldFunc->args()) {
      newToOriginalFn[originalToNewFn[&oArg]] = &oArg;
    }
    */
    /*
    for (BasicBlock &BB : *newFunc) {
      originalBlocks.emplace_back(&BB);
    }
    tape = nullptr;
    tapeidx = 0;
    assert(originalBlocks.size() > 0);

    SmallVector<BasicBlock *, 4> ReturningBlocks;
    for (BasicBlock &BB : *oldFunc) {
      if (isa<ReturnInst>(BB.getTerminator()))
        ReturningBlocks.push_back(&BB);
    }
    for (BasicBlock &BB : *oldFunc) {
      bool legal = true;
      for (auto BRet : ReturningBlocks) {
        if (!(BRet == &BB || OrigDT.dominates(&BB, BRet))) {
          legal = false;
          break;
        }
      }
      if (legal)
        BlocksDominatingAllReturns.insert(&BB);
    }
    */
  }
  void erase(Operation *op) { op->erase(); }
  bool isConstantValue(mlir::Value v) const {
    if (isa<mlir::IntegerType>(v.getType()))
      return true;
    if (isa<mlir::IndexType>(v.getType()))
      return true;

    if (v.getDefiningOp<arith::ConstantOp>())
      return true;

    // TODO
    return false;
  }
  mlir::Value invertPointerM(mlir::Value v, OpBuilder &Builder2) {
    // TODO
    if (invertedPointers.contains(v))
      return invertedPointers.lookupOrNull(v);

    if (isConstantValue(v)) {
      if (auto FT = dyn_cast<mlir::FloatType>(v.getType())) {
        OpBuilder Builder2(getNewFromOriginal(v.getDefiningOp()));
        Value dv = Builder2.create<arith::ConstantFloatOp>(
            v.getLoc(), APFloat(FT.getFloatSemantics(), "0"), FT);
        invertedPointers.map(v, dv);
        return dv;
      }
      return getNewFromOriginal(v);
    }
    llvm::errs() << " could not invert pointer v " << v << "\n";
    llvm_unreachable("could not invert pointer");
  }
  void setDiffe(mlir::Value val, mlir::Value toset, OpBuilder &BuilderM) {
    /*
  if (auto arg = dyn_cast<Argument>(val))
    assert(arg->getParent() == oldFunc);
  if (auto inst = dyn_cast<Instruction>(val))
    assert(inst->getParent()->getParent() == oldFunc);
    */
    if (isConstantValue(val)) {
      llvm::errs() << newFunc << "\n";
      llvm::errs() << val << "\n";
    }
    assert(!isConstantValue(val));
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      assert(getShadowType(val.getType()) == toset.getType());
      auto found = invertedPointers.lookupOrNull(val);
      assert(found != nullptr);
      auto placeholder = found.getDefiningOp<enzyme::PlaceholderOp>();
      invertedPointers.erase(val);
      // replaceAWithB(placeholder, toset);
      placeholder.replaceAllUsesWith(toset);
      erase(placeholder);
      invertedPointers.map(val, toset);
      return;
    }
    /*
    Value *tostore = getDifferential(val);
    if (toset->getType() != tostore->getType()->getPointerElementType()) {
      llvm::errs() << "toset:" << *toset << "\n";
      llvm::errs() << "tostore:" << *tostore << "\n";
    }
    assert(toset->getType() == tostore->getType()->getPointerElementType());
    BuilderM.CreateStore(toset, tostore);
    */
  }
  void forceAugmentedReturns() {
    // TODO also block arguments
    // assert(TR.getFunction() == oldFunc);

    // Don't create derivatives for code that results in termination
    // if (notForAnalysis.find(&oBB) != notForAnalysis.end())
    //  continue;

    // LoopContext loopContext;
    // getContext(cast<BasicBlock>(getNewFromOriginal(&oBB)), loopContext);

    oldFunc.walk([&](Block *blk) {
      if (blk == &oldFunc.getBody().getBlocks().front())
        return;
      auto nblk = getNewFromOriginal(blk);
      for (auto val : llvm::reverse(blk->getArguments())) {
        if (isConstantValue(val))
          continue;
        auto i = val.getArgNumber();
        mlir::Value dval;
        if (i == blk->getArguments().size() - 1)
          dval = nblk->addArgument(getShadowType(val.getType()), val.getLoc());
        else
          dval =
              nblk->insertArgument(nblk->args_begin() + i + 1,
                                   getShadowType(val.getType()), val.getLoc());

        invertedPointers.map(val, dval);
      }
    });

    oldFunc.walk([&](Operation *inst) {
      if (inst == oldFunc)
        return;
      if (mode == DerivativeMode::ForwardMode ||
          mode == DerivativeMode::ForwardModeSplit) {
        OpBuilder BuilderZ(getNewFromOriginal(inst));
        for (auto res : inst->getResults()) {
          if (!isConstantValue(res)) {
            mlir::Type antiTy = getShadowType(res.getType());
            auto anti = BuilderZ.create<enzyme::PlaceholderOp>(res.getLoc(),
                                                               res.getType());
            invertedPointers.map(res, anti);
          }
        }
        return;
      }
      /*

      if (inst->getType()->isFPOrFPVectorTy())
        continue; //! op->getType()->isPointerTy() &&
                  //! !op->getType()->isIntegerTy()) {

      if (!TR.query(inst)[{-1}].isPossiblePointer())
        continue;

      if (isa<LoadInst>(inst)) {
        IRBuilder<> BuilderZ(inst);
        getForwardBuilder(BuilderZ);
        Type *antiTy = getShadowType(inst->getType());
        PHINode *anti =
            BuilderZ.CreatePHI(antiTy, 1, inst->getName() + "'il_phi");
        invertedPointers.insert(std::make_pair(
            (const Value *)inst, InvertedPointerVH(this, anti)));
        continue;
      }

      if (!isa<CallInst>(inst)) {
        continue;
      }

      if (isa<IntrinsicInst>(inst)) {
        continue;
      }

      if (isConstantValue(inst)) {
        continue;
      }

      CallInst *op = cast<CallInst>(inst);
      Function *called = op->getCalledFunction();

      IRBuilder<> BuilderZ(inst);
      getForwardBuilder(BuilderZ);
      Type *antiTy = getShadowType(inst->getType());

      PHINode *anti =
          BuilderZ.CreatePHI(antiTy, 1, op->getName() + "'ip_phi");
      invertedPointers.insert(
          std::make_pair((const Value *)inst, InvertedPointerVH(this, anti)));

      if (called && isAllocationFunction(called->getName(), TLI)) {
        anti->setName(op->getName() + "'mi");
      }
      */
    });
  }
};
class MDiffeGradientUtils : public MGradientUtils {
public:
  MDiffeGradientUtils(MEnzymeLogic &Logic, mlir::func::FuncOp newFunc_,
                      FunctionOpInterface oldFunc_, MTypeAnalysis &TA,
                      MTypeResults TR, BlockAndValueMapping &invertedPointers_,
                      const SmallPtrSetImpl<mlir::Value> &constantvalues_,
                      const SmallPtrSetImpl<mlir::Value> &returnvals_,
                      DIFFE_TYPE ActiveReturn,
                      ArrayRef<DIFFE_TYPE> constant_values,
                      BlockAndValueMapping &origToNew_,
                      std::map<Operation *, Operation *> &origToNewOps_,
                      DerivativeMode mode, unsigned width, bool omp)
      : MGradientUtils(Logic, newFunc_, oldFunc_, TA, TR, invertedPointers_,
                       constantvalues_, returnvals_, ActiveReturn,
                       constant_values, origToNew_, origToNewOps_, mode, width,
                       omp) {
    /* TODO
    assert(reverseBlocks.size() == 0);
    if (mode == DerivativeMode::ForwardMode ||
        mode == DerivativeMode::ForwardModeSplit) {
      return;
    }
    for (BasicBlock *BB : originalBlocks) {
      if (BB == inversionAllocs)
        continue;
      BasicBlock *RBB = BasicBlock::Create(BB->getContext(),
                                           "invert" + BB->getName(), newFunc);
      reverseBlocks[BB].push_back(RBB);
      reverseBlockToPrimal[RBB] = BB;
    }
    assert(reverseBlocks.size() != 0);
    */
  }

  // Technically diffe constructor
  static MDiffeGradientUtils *
  CreateFromClone(MEnzymeLogic &Logic, DerivativeMode mode, unsigned width,
                  FunctionOpInterface todiff, MTypeAnalysis &TA,
                  MFnTypeInfo &oldTypeInfo, DIFFE_TYPE retType,
                  bool diffeReturnArg, ArrayRef<DIFFE_TYPE> constant_args,
                  ReturnType returnValue, mlir::Type additionalArg, bool omp) {
    std::string prefix;

    switch (mode) {
    case DerivativeMode::ForwardMode:
    case DerivativeMode::ForwardModeSplit:
      prefix = "fwddiffe";
      break;
    case DerivativeMode::ReverseModeCombined:
    case DerivativeMode::ReverseModeGradient:
      prefix = "diffe";
      break;
    case DerivativeMode::ReverseModePrimal:
      llvm_unreachable("invalid DerivativeMode: ReverseModePrimal\n");
    }

    if (width > 1)
      prefix += std::to_string(width);

    BlockAndValueMapping originalToNew;
    std::map<Operation *, Operation *> originalToNewOps;

    SmallPtrSet<mlir::Value, 1> returnvals;
    SmallPtrSet<mlir::Value, 1> constant_values;
    SmallPtrSet<mlir::Value, 1> nonconstant_values;
    BlockAndValueMapping invertedPointers;
    auto newFunc = CloneFunctionWithReturns(
        mode, width, todiff, invertedPointers, constant_args, constant_values,
        nonconstant_values, returnvals, returnValue, retType,
        prefix + todiff.getName(), originalToNew, originalToNewOps,
        diffeReturnArg, additionalArg);
    MTypeResults TR; // TODO
    return new MDiffeGradientUtils(
        Logic, newFunc, todiff, TA, TR, invertedPointers, constant_values,
        nonconstant_values, retType, constant_args, originalToNew,
        originalToNewOps, mode, width, omp);
  }
};

class MAdjointGenerator {
public:
  MGradientUtils *gutils;
  MAdjointGenerator(MGradientUtils *gutils) : gutils(gutils) {}

  void eraseIfUnused(Operation &op, bool erase = true, bool check = true) {
    // TODO
  }
  void visit(Operation *op) {
    if (gutils->mode == DerivativeMode::ForwardMode) {
      if (auto mulOp = dyn_cast<arith::MulFOp>(op)) {
        // Derivative of r = a * b -> dr = a * db + da * b
        if (!gutils->isConstantValue(mulOp)) {
          mlir::Value res = nullptr;
          OpBuilder Builder2(gutils->getNewFromOriginal(op));
          for (int i = 0; i < 2; i++) {
            if (!gutils->isConstantValue(mulOp.getOperand(i))) {
              Value tmp = Builder2.create<arith::MulFOp>(
                  mulOp.getLoc(),
                  gutils->invertPointerM(mulOp.getOperand(i), Builder2),
                  gutils->getNewFromOriginal(mulOp.getOperand(1 - i)));
              if (res == nullptr)
                res = tmp;
              else
                res = Builder2.create<arith::AddFOp>(mulOp.getLoc(), res, tmp);
            }
          }
          gutils->setDiffe(mulOp, res, Builder2);
        }
        eraseIfUnused(*op);
        return;
      }
      if (auto addOp = dyn_cast<arith::AddFOp>(op)) {
        // Derivative of r = a + b -> dr = da + db
        if (!gutils->isConstantValue(addOp)) {
          mlir::Value res = nullptr;
          OpBuilder Builder2(gutils->getNewFromOriginal(op));
          for (int i = 0; i < 2; i++) {
            if (!gutils->isConstantValue(addOp.getOperand(i))) {
              Value tmp = gutils->invertPointerM(addOp.getOperand(i), Builder2);
              if (res == nullptr)
                res = tmp;
              else
                res = Builder2.create<arith::AddFOp>(addOp.getLoc(), res, tmp);
            }
          }
          gutils->setDiffe(addOp, res, Builder2);
        }
        eraseIfUnused(*op);
        return;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto nFor = cast<scf::ForOp>(gutils->getNewFromOriginal(op));
        OpBuilder Builder2(nFor);
        SmallVector<Type> nTypes;
        for (auto r : forOp->getResults()) {
          // TODO only if used
          nTypes.push_back(r.getType());
          if (!gutils->isConstantValue(r))
            nTypes.push_back(gutils->getShadowType(r.getType()));
        }
        SmallVector<Value> nArgs;
        for (auto r :
             llvm::zip(forOp.getIterOperands(), forOp.getRegionIterArgs())) {
          // TODO only if used
          nArgs.push_back(gutils->getNewFromOriginal(std::get<0>(r)));
          if (!gutils->isConstantValue(std::get<1>(r)))
            nArgs.push_back(gutils->invertPointerM(std::get<0>(r), Builder2));
        }
        auto repFor = Builder2.create<scf::ForOp>(
            forOp.getLoc(), gutils->getNewFromOriginal(forOp.getLowerBound()),
            gutils->getNewFromOriginal(forOp.getUpperBound()),
            gutils->getNewFromOriginal(forOp.getStep()), nArgs);
        repFor.getRegion().takeBody(nFor.getRegion());

        SmallVector<Value> reps;
        size_t idx = 0;
        for (auto r : forOp.getResults()) {
          // TODO only if used
          reps.push_back(repFor.getResult(idx));
          idx++;
          if (!gutils->isConstantValue(r)) {
            auto inverted = gutils->invertedPointers.lookupOrNull(r);
            assert(inverted);
            gutils->invertedPointers.map(r, repFor.getResult(idx));
            inverted.replaceAllUsesWith(repFor.getResult(idx));
            gutils->erase(inverted.getDefiningOp());
            idx++;
          }
        }
        nFor.replaceAllUsesWith(reps);
        gutils->erase(nFor);
        for (auto &o : llvm::make_early_inc_range(
                 forOp.getBody()->without_terminator())) {
          visit(&o);
        }
        auto oldYield = repFor.getBody()->getTerminator();
        Builder2.setInsertionPointToEnd(repFor.getBody());
        SmallVector<Value> nYields;
        for (auto r :
             llvm::zip(forOp.getResults(),
                       forOp.getBody()->getTerminator()->getOperands())) {
          // TODO only if used
          nYields.push_back(gutils->getNewFromOriginal(std::get<1>(r)));
          if (!gutils->isConstantValue(std::get<0>(r)))
            nYields.push_back(gutils->invertPointerM(std::get<1>(r), Builder2));
        }
        repFor.getBody()->push_back(oldYield->create(
            oldYield->getLoc(), oldYield->getName(), TypeRange(), nYields,
            oldYield->getAttrs(), oldYield->getSuccessors(),
            oldYield->getNumRegions()));
        gutils->erase(oldYield);
        return;
      }
    }
    // TODO
  }
};

void createTerminator(MDiffeGradientUtils *gutils, mlir::Block *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  MTypeResults &TR = gutils->TR;
  auto inst = oBB->getTerminator();

  mlir::Block *nBB = gutils->getNewFromOriginal(inst->getBlock());
  assert(nBB);
  auto newInst = nBB->getTerminator();

  OpBuilder nBuilder(inst);
  nBuilder.setInsertionPointToEnd(nBB);

  if (auto binst = dyn_cast<BranchOpInterface>(inst)) {
    // TODO generalize to cloneWithNewBlockArgs interface
    SmallVector<Value> newVals;

    SmallVector<int32_t> segSizes;
    for (size_t i = 0, len = binst.getSuccessorOperands(0)
                                 .getForwardedOperands()
                                 .getBeginOperandIndex();
         i < len; i++)
      newVals.push_back(gutils->getNewFromOriginal(binst->getOperand(i)));
    segSizes.push_back(newVals.size());
    for (size_t i = 0; i < newInst->getNumSuccessors(); i++) {
      size_t cur = newVals.size();
      for (auto op : binst.getSuccessorOperands(i).getForwardedOperands()) {
        newVals.push_back(gutils->getNewFromOriginal(op));
        if (!gutils->isConstantValue(op)) {
          newVals.push_back(gutils->invertPointerM(op, nBuilder));
        }
      }
      cur = newVals.size() - cur;
      segSizes.push_back(cur);
    }

    SmallVector<NamedAttribute> attrs(newInst->getAttrs());
    for (auto &attr : attrs) {
      if (attr.getName() == "operand_segment_sizes")
        attr.setValue(nBuilder.getDenseI32ArrayAttr(segSizes));
    }

    nBB->push_back(newInst->create(
        newInst->getLoc(), newInst->getName(), TypeRange(), newVals, attrs,
        newInst->getSuccessors(), newInst->getNumRegions()));
    gutils->erase(newInst);
    return;
  }

  // In forward mode we only need to update the return value
  if (!inst->hasTrait<OpTrait::ReturnLike>())
    return;

  SmallVector<mlir::Value, 2> retargs;

  switch (retVal) {
  case ReturnType::Return: {
    auto ret = inst->getOperand(0);

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      auto retTy = gutils->getShadowType(ret.getType()).cast<mlir::FloatType>();
      toret = nBuilder.create<arith::ConstantFloatOp>(
          ret.getLoc(), APFloat(retTy.getFloatSemantics(), "0"), retTy);
    }
    retargs.push_back(toret);

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    auto ret = inst->getOperand(0);

    retargs.push_back(gutils->getNewFromOriginal(ret));

    mlir::Value toret;
    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!isa<mlir::FloatType>(ret.getType()) &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      toret = gutils->invertPointerM(ret, nBuilder);
    } else {
      auto retTy = gutils->getShadowType(ret.getType()).cast<mlir::FloatType>();
      toret = nBuilder.create<arith::ConstantFloatOp>(
          ret.getLoc(), APFloat(retTy.getFloatSemantics(), "0"), retTy);
    }
    retargs.push_back(toret);
    break;
  }
  case ReturnType::Void: {
    break;
  }
  default: {
    llvm::errs() << "Invalid return type: " << to_string(retVal)
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  nBB->push_back(newInst->create(
      newInst->getLoc(), newInst->getName(), TypeRange(), retargs,
      newInst->getAttrs(), newInst->getSuccessors(), newInst->getNumRegions()));
  gutils->erase(newInst);
  return;
}

class MEnzymeLogic {
public:
  struct MForwardCacheKey {
    FunctionOpInterface todiff;
    DIFFE_TYPE retType;
    const std::vector<DIFFE_TYPE> constant_args;
    // std::map<llvm::Argument *, bool> uncacheable_args;
    bool returnUsed;
    DerivativeMode mode;
    unsigned width;
    mlir::Type additionalType;
    const MFnTypeInfo typeInfo;

    inline bool operator<(const MForwardCacheKey &rhs) const {
      if (todiff < rhs.todiff)
        return true;
      if (rhs.todiff < todiff)
        return false;

      if (retType < rhs.retType)
        return true;
      if (rhs.retType < retType)
        return false;

      if (std::lexicographical_compare(
              constant_args.begin(), constant_args.end(),
              rhs.constant_args.begin(), rhs.constant_args.end()))
        return true;
      if (std::lexicographical_compare(
              rhs.constant_args.begin(), rhs.constant_args.end(),
              constant_args.begin(), constant_args.end()))
        return false;

      /*
      for (auto &arg : todiff->args()) {
        auto foundLHS = uncacheable_args.find(&arg);
        auto foundRHS = rhs.uncacheable_args.find(&arg);
        if (foundLHS->second < foundRHS->second)
          return true;
        if (foundRHS->second < foundLHS->second)
          return false;
      }
      */

      if (returnUsed < rhs.returnUsed)
        return true;
      if (rhs.returnUsed < returnUsed)
        return false;

      if (mode < rhs.mode)
        return true;
      if (rhs.mode < mode)
        return false;

      if (width < rhs.width)
        return true;
      if (rhs.width < width)
        return false;

      if (additionalType < rhs.additionalType)
        return true;
      if (rhs.additionalType < additionalType)
        return false;

      if (typeInfo < rhs.typeInfo)
        return true;
      if (rhs.typeInfo < typeInfo)
        return false;
      // equal
      return false;
    }
  };

  std::map<MForwardCacheKey, mlir::func::FuncOp> ForwardCachedFunctions;
  mlir::func::FuncOp
  CreateForwardDiff(FunctionOpInterface fn, DIFFE_TYPE retType,
                    std::vector<DIFFE_TYPE> constants, MTypeAnalysis &TA,
                    bool returnUsed, DerivativeMode mode, bool freeMemory,
                    size_t width, mlir::Type addedType, MFnTypeInfo type_args,
                    std::vector<bool> volatile_args, void *augmented) {
    if (fn.getBody().empty()) {
      llvm::errs() << fn << "\n";
      llvm_unreachable("Differentiating empty function");
    }

    MForwardCacheKey tup = {
        fn, retType, constants,
        // std::map<Argument *, bool>(_uncacheable_args.begin(),
        //                           _uncacheable_args.end()),
        returnUsed, mode, width, addedType, type_args};

    if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
      return ForwardCachedFunctions.find(tup)->second;
    }
    bool retActive = retType != DIFFE_TYPE::CONSTANT;
    ReturnType returnValue =
        returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                   : (retActive ? ReturnType::Return : ReturnType::Void);
    auto gutils = MDiffeGradientUtils::CreateFromClone(
        *this, mode, width, fn, TA, type_args, retType,
        /*diffeReturnArg*/ false, constants, returnValue, addedType,
        /*omp*/ false);
    ForwardCachedFunctions[tup] = gutils->newFunc;

    insert_or_assign2<MForwardCacheKey, func::FuncOp>(ForwardCachedFunctions,
                                                      tup, gutils->newFunc);

    // gutils->FreeMemory = freeMemory;

    const SmallPtrSet<mlir::Block *, 4> guaranteedUnreachable;
    // = getGuaranteedUnreachable(gutils->oldFunc);

    // gutils->forceActiveDetection();
    gutils->forceAugmentedReturns();
    /*

    // TODO populate with actual unnecessaryInstructions once the dependency
    // cycle with activity analysis is removed
    SmallPtrSet<const Instruction *, 4> unnecessaryInstructionsTmp;
    for (auto BB : guaranteedUnreachable) {
      for (auto &I : *BB)
        unnecessaryInstructionsTmp.insert(&I);
    }
    if (mode == DerivativeMode::ForwardModeSplit)
      gutils->computeGuaranteedFrees();

    SmallPtrSet<const Value *, 4> unnecessaryValues;
    SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
    calculateUnusedValuesInFunction(
        *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
    returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
    gutils->unnecessaryValuesP = &unnecessaryValues;

    SmallPtrSet<const Instruction *, 4> unnecessaryStores;
    calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                    unnecessaryInstructions, gutils, TLI);
                                    */

    MAdjointGenerator *maker;

    // TODO split
    maker = new MAdjointGenerator(gutils);
    //, constant_args, retType, nullptr, {},
    //    /*returnuses*/ nullptr, nullptr, nullptr, unnecessaryValues,
    //    unnecessaryInstructions, unnecessaryStores, guaranteedUnreachable,
    //    nullptr);

    for (Block &oBB : gutils->oldFunc.getBody().getBlocks()) {
      // Don't create derivatives for code that results in termination
      if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
        auto newBB = gutils->getNewFromOriginal(&oBB);

        SmallVector<Operation *, 4> toerase;
        for (auto &I : oBB) {
          toerase.push_back(&I);
        }
        for (auto I : llvm::reverse(toerase)) {
          maker->eraseIfUnused(*I, /*erase*/ true, /*check*/ false);
        }
        OpBuilder builder(gutils->oldFunc.getContext());
        builder.setInsertionPointToEnd(newBB);
        builder.create<LLVM::UnreachableOp>(gutils->oldFunc.getLoc());
        continue;
      }

      auto term = oBB.getTerminator();
      assert(term);

      auto first = oBB.begin();
      auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
      for (auto it = first; it != last; ++it) {
        maker->visit(&*it);
      }

      createTerminator(gutils, &oBB, retType, returnValue);
    }

    // if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
    //  restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

    // gutils->eraseFictiousPHIs();

    mlir::Block *entry = &gutils->newFunc.getBody().front();

    // cleanupInversionAllocs(gutils, entry);
    // clearFunctionAttributes(gutils->newFunc);

    /*
    if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
      llvm::errs() << *gutils->oldFunc << "\n";
      llvm::errs() << *gutils->newFunc << "\n";
      report_fatal_error("function failed verification (4)");
    }
    */

    auto nf = gutils->newFunc;
    delete gutils;
    delete maker;

    // if (PostOpt)
    //  PPC.optimizeIntermediate(nf);
    // if (EnzymePrint) {
    //  llvm::errs() << nf << "\n";
    //}
    return nf;
  }
};

namespace {
struct DifferentiatePass : public DifferentiatePassBase<DifferentiatePass> {
  MEnzymeLogic Logic;

  void runOnOperation() override;

  template <typename T>
  void HandleAutoDiff(SymbolTableCollection &symbolTable, T CI) {
    std::vector<DIFFE_TYPE> constants;
    SmallVector<mlir::Value, 2> args;

    size_t truei = 0;
    auto activityAttr = CI.getActivity();

    for (unsigned i = 0; i < CI.getInputs().size(); ++i) {
      mlir::Value res = CI.getInputs()[i];

      auto mop = activityAttr[truei];
      auto iattr = cast<mlir::enzyme::ActivityAttr>(mop);
      DIFFE_TYPE ty = (DIFFE_TYPE)(iattr.getValue());

      constants.push_back(ty);
      args.push_back(res);
      if (ty == DIFFE_TYPE::DUP_ARG || ty == DIFFE_TYPE::DUP_NONEED) {
        ++i;
        res = CI.getInputs()[i];
        args.push_back(res);
      }

      truei++;
    }

    auto *symbolOp = symbolTable.lookupNearestSymbolFrom(CI, CI.getFnAttr());
    auto fn = cast<FunctionOpInterface>(symbolOp);

    DIFFE_TYPE retType =
        fn.getNumResults() == 0 ? DIFFE_TYPE::CONSTANT : DIFFE_TYPE::DUP_ARG;

    MTypeAnalysis TA;
    auto type_args = TA.getAnalyzedTypeInfo(fn);
    auto mode = DerivativeMode::ForwardMode;
    bool freeMemory = true;
    size_t width = 1;

    std::vector<bool> volatile_args;
    for (auto &a : fn.getBody().getArguments()) {
      volatile_args.push_back(!(mode == DerivativeMode::ReverseModeCombined));
    }

    auto newFunc = Logic.CreateForwardDiff(
        fn, retType, constants, TA,
        /*should return*/ false, mode, freeMemory, width,
        /*addedType*/ nullptr, type_args, volatile_args,
        /*augmented*/ nullptr);

    OpBuilder builder(CI);
    auto dCI = builder.create<func::CallOp>(CI.getLoc(), newFunc, args);
    CI.replaceAllUsesWith(dCI);
    CI->erase();
  }

  void lowerEnzymeCalls(SymbolTableCollection &symbolTable,
                        FunctionOpInterface op) {
    SmallVector<Operation *> toLower;
    op->walk([&](enzyme::ForwardDiffOp dop) {
      auto *symbolOp =
          symbolTable.lookupNearestSymbolFrom(dop, dop.getFnAttr());
      auto callableOp = cast<FunctionOpInterface>(symbolOp);

      lowerEnzymeCalls(symbolTable, callableOp);
      toLower.push_back(dop);
    });

    for (auto T : toLower) {
      if (auto F = dyn_cast<enzyme::ForwardDiffOp>(T)) {
        HandleAutoDiff(symbolTable, F);
      } else {
        llvm_unreachable("Illegal type");
      }
    }
  }
};

} // end anonymous namespace

namespace mlir {
namespace enzyme {
std::unique_ptr<Pass> createDifferentiatePass() {
  new DifferentiatePass();
  return std::make_unique<DifferentiatePass>();
}
} // namespace enzyme
} // namespace mlir

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

void DifferentiatePass::runOnOperation() {
  SymbolTableCollection symbolTable;
  symbolTable.getSymbolTable(getOperation());
  ConversionPatternRewriter B(getOperation()->getContext());
  getOperation()->walk(
      [&](FunctionOpInterface op) { lowerEnzymeCalls(symbolTable, op); });
}
