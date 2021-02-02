//===- GradientUtils.cpp - Helper class and utilities for AD     ---------===//
//
//                             Enzyme Project
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// If using this code in an academic setting, please cite the following:
// @incollection{enzymeNeurips,
// title = {Instead of Rewriting Foreign Code for Machine Learning,
//          Automatically Synthesize Fast Gradients},
// author = {Moses, William S. and Churavy, Valentin},
// booktitle = {Advances in Neural Information Processing Systems 33},
// year = {2020},
// note = {To appear in},
// }
//
//===----------------------------------------------------------------------===//
//
// This file define two helper classes GradientUtils and subclass
// DiffeGradientUtils. These classes contain utilities for managing the cache,
// recomputing statements, and in the case of DiffeGradientUtils, managing
// adjoint values and shadow pointers.
//
//===----------------------------------------------------------------------===//

#include "GradientUtils.h"

#include <llvm/Config/llvm-config.h>

#include "EnzymeLogic.h"

#include "FunctionUtils.h"

#include "LibraryFuncs.h"

#include "llvm/IR/GlobalValue.h"

#include "llvm/IR/Constants.h"

#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Transforms/Utils/SimplifyIndVar.h"

#include <algorithm>

Value *GradientUtils::unwrapM(Value *const val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &available,
                              UnwrapMode mode) {
  assert(val);
  assert(val->getName() != "<badref>");
  assert(val->getType());

  // llvm::errs() << " attempting unwrap of: " << *val << "\n";

  for (auto pair : available) {
    assert(pair.first);
    assert(pair.second);
    assert(pair.first->getType());
    assert(pair.second->getType());
    assert(pair.first->getType() == pair.second->getType());
  }

  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_mustcache")) {
    return val;
  }

  // assert(!val->getName().startswith("$tapeload"));

  auto cidx = std::make_pair(val, BuilderM.GetInsertBlock());
  if (unwrap_cache.find(cidx) != unwrap_cache.end()) {
    if (unwrap_cache[cidx]->getType() != val->getType()) {
      llvm::errs() << "val: " << *val << "\n";
      llvm::errs() << "unwrap_cache[cidx]: " << *unwrap_cache[cidx] << "\n";
    }
    assert(unwrap_cache[cidx]->getType() == val->getType());
    return unwrap_cache[cidx];
  }

  if (available.count(val)) {
    auto avail = available.lookup(val);
    assert(avail->getType());
    if (avail->getType() != val->getType()) {
      llvm::errs() << "val: " << *val << "\n";
      llvm::errs() << "available[val]: " << *available.lookup(val) << "\n";
    }
    assert(available.lookup(val)->getType() == val->getType());
    return available.lookup(val);
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    // if (inst->getParent() == &newFunc->getEntryBlock()) {
    //  return inst;
    //}
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
          // llvm::errs() << "allowed " << *inst << "from block domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
  }

// llvm::errs() << "uwval: " << *val << "\n";
#define getOp(v)                                                               \
  ({                                                                           \
    Value *___res;                                                             \
    if (mode == UnwrapMode::LegalFullUnwrap ||                                 \
        mode == UnwrapMode::AttemptFullUnwrap ||                               \
        mode == UnwrapMode::AttemptFullUnwrapWithLookup) {                     \
      ___res = unwrapM(v, BuilderM, available, mode);                          \
    } else {                                                                   \
      assert(mode == UnwrapMode::AttemptSingleUnwrap);                         \
      ___res = lookupM(v, BuilderM, available);                                \
    }                                                                          \
    ___res;                                                                    \
  })

  if (isa<Argument>(val) || isa<Constant>(val)) {
    unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
    return val;
  } else if (isa<AllocaInst>(val)) {
    unwrap_cache[std::make_pair(val, BuilderM.GetInsertBlock())] = val;
    return val;
  } else if (auto op = dyn_cast<CastInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateCast(op->getOpcode(), op0, op->getDestTy(),
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ExtractValueInst>(val)) {
    auto op0 = getOp(op->getAggregateOperand());
    if (op0 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateExtractValue(op0, op->getIndices(),
                                                op->getName() + "_unwrap");
    unwrap_cache[cidx] = toreturn;
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<BinaryOperator>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateBinOp(op->getOpcode(), op0, op1,
                                         op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<ICmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateICmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<FCmpInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto toreturn = BuilderM.CreateFCmp(op->getPredicate(), op0, op1,
                                        op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<SelectInst>(val)) {
    auto op0 = getOp(op->getOperand(0));
    if (op0 == nullptr)
      goto endCheck;
    auto op1 = getOp(op->getOperand(1));
    if (op1 == nullptr)
      goto endCheck;
    auto op2 = getOp(op->getOperand(2));
    if (op2 == nullptr)
      goto endCheck;
    auto toreturn =
        BuilderM.CreateSelect(op0, op1, op2, op->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(op);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto inst = dyn_cast<GetElementPtrInst>(val)) {
    auto ptr = getOp(inst->getPointerOperand());
    if (ptr == nullptr)
      goto endCheck;
    SmallVector<Value *, 4> ind;
    // llvm::errs() << "inst: " << *inst << "\n";
    for (unsigned i = 0; i < inst->getNumIndices(); ++i) {
      Value *a = inst->getOperand(1 + i);
      assert(a->getName() != "<badref>");
      auto op = getOp(a);
      if (op == nullptr)
        goto endCheck;
      ind.push_back(op);
    }
    auto toreturn = BuilderM.CreateGEP(ptr, ind, inst->getName() + "_unwrap");
    if (isa<GetElementPtrInst>(toreturn))
      cast<GetElementPtrInst>(toreturn)->setIsInBounds(inst->isInBounds());
    else {
      // llvm::errs() << "gep tr: " << *toreturn << " inst: " << *inst << "
      // ptr: " << *ptr << "\n"; llvm::errs() << "safe: " << *SAFE(inst,
      // getPointerOperand()) << "\n"; assert(0 && "illegal");
    }
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(inst);
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto load = dyn_cast<LoadInst>(val)) {
    if (load->getMetadata("enzyme_noneedunwrap"))
      return load;

    bool legalMove = mode == UnwrapMode::LegalFullUnwrap;
    if (mode != UnwrapMode::LegalFullUnwrap) {
      // TODO actually consider whether this is legal to move to the new
      // location, rather than recomputable anywhere
      legalMove = legalRecompute(load, available);
    }
    if (!legalMove)
      return nullptr;

    Value *idx = getOp(load->getOperand(0));
    if (idx == nullptr)
      goto endCheck;

    if (idx->getType() != load->getOperand(0)->getType()) {
      llvm::errs() << "load: " << *load << "\n";
      llvm::errs() << "load->getOperand(0): " << *load->getOperand(0) << "\n";
      llvm::errs() << "idx: " << *idx << "\n";
    }
    assert(idx->getType() == load->getOperand(0)->getType());
    auto toreturn = BuilderM.CreateLoad(idx, load->getName() + "_unwrap");
    if (auto newi = dyn_cast<Instruction>(toreturn))
      newi->copyIRFlags(load);
#if LLVM_VERSION_MAJOR >= 10
    toreturn->setAlignment(load->getAlign());
#else
    toreturn->setAlignment(load->getAlignment());
#endif
    toreturn->setVolatile(load->isVolatile());
    toreturn->setOrdering(load->getOrdering());
    toreturn->setSyncScopeID(load->getSyncScopeID());
    toreturn->setDebugLoc(getNewFromOriginal(load->getDebugLoc()));
    toreturn->setMetadata(LLVMContext::MD_tbaa,
                          load->getMetadata(LLVMContext::MD_tbaa));
    toreturn->setMetadata("enzyme_unwrapped",
                          MDNode::get(toreturn->getContext(), {}));
    // toreturn->setMetadata(LLVMContext::MD_invariant,
    // load->getMetadata(LLVMContext::MD_invariant));
    toreturn->setMetadata(LLVMContext::MD_invariant_group,
                          load->getMetadata(LLVMContext::MD_invariant_group));
    // TODO adding to cache only legal if no alias of any future writes
    unwrap_cache[cidx] = toreturn;
    assert(val->getType() == toreturn->getType());
    return toreturn;
  } else if (auto op = dyn_cast<CallInst>(val)) {

    bool legalMove = mode == UnwrapMode::LegalFullUnwrap;
    if (mode != UnwrapMode::LegalFullUnwrap) {
      // TODO actually consider whether this is legal to move to the new
      // location, rather than recomputable anywhere
      legalMove = legalRecompute(op, available);
    }
    if (!legalMove)
      return nullptr;

    std::vector<Value *> args;
    for (unsigned i = 0; i < op->getNumArgOperands(); ++i) {
      args.emplace_back(getOp(op->getArgOperand(i)));
      if (args[i] == nullptr)
        return nullptr;
    }

#if LLVM_VERSION_MAJOR >= 11
    Value *fn = getOp(op->getCalledOperand());
#else
    Value *fn = getOp(op->getCalledValue());
#endif
    if (fn == nullptr)
      return nullptr;

    auto toreturn =
        cast<CallInst>(BuilderM.CreateCall(op->getFunctionType(), fn, args));
    toreturn->copyIRFlags(op);
    toreturn->setAttributes(op->getAttributes());
    toreturn->setCallingConv(op->getCallingConv());
    toreturn->setTailCallKind(op->getTailCallKind());
    toreturn->setDebugLoc(getNewFromOriginal(op->getDebugLoc()));
    return toreturn;
  } else if (auto phi = dyn_cast<PHINode>(val)) {
    if (phi->getNumIncomingValues() == 1) {
      assert(phi->getIncomingValue(0) != phi);
      auto toreturn = getOp(phi->getIncomingValue(0));
      if (toreturn == nullptr)
        goto endCheck;
      assert(val->getType() == toreturn->getType());
      return toreturn;
    }
  }

endCheck:
  assert(val);
  if (mode == UnwrapMode::LegalFullUnwrap ||
      mode == UnwrapMode::AttemptFullUnwrapWithLookup) {
    assert(val->getName() != "<badref>");
    auto toreturn = lookupM(val, BuilderM);
    assert(val->getType() == toreturn->getType());
    return toreturn;
  }

  // llvm::errs() << "cannot unwrap following " << *val << "\n";

  if (auto inst = dyn_cast<Instruction>(val)) {
    // LoopContext lc;
    // if (BuilderM.GetInsertBlock() != inversionAllocs && !(
    // (reverseBlocks.find(BuilderM.GetInsertBlock()) != reverseBlocks.end())
    // && /*inLoop*/getContext(inst->getParent(), lc)) ) {
    if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
      if (BuilderM.GetInsertBlock()->size() &&
          BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
        if (DT.dominates(inst, &*BuilderM.GetInsertPoint())) {
          // llvm::errs() << "allowed " << *inst << "from domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      } else {
        if (DT.dominates(inst, BuilderM.GetInsertBlock())) {
          // llvm::errs() << "allowed " << *inst << "from block domination\n";
          assert(inst->getType() == val->getType());
          return inst;
        }
      }
    }
  }
  return nullptr;
}

Value *GradientUtils::cacheForReverse(IRBuilder<> &BuilderQ, Value *malloc,
                                      int idx) {
  assert(malloc);
  assert(BuilderQ.GetInsertBlock()->getParent() == newFunc);

  if (tape) {
    if (idx >= 0 && !tape->getType()->isStructTy()) {
      llvm::errs() << "cacheForReverse incorrect tape type: " << *tape
                   << " idx: " << idx << "\n";
    }
    assert(idx < 0 || tape->getType()->isStructTy());
    if (idx >= 0 &&
        (unsigned)idx >= cast<StructType>(tape->getType())->getNumElements()) {
      llvm::errs() << "oldFunc: " << *oldFunc << "\n";
      llvm::errs() << "newFunc: " << *newFunc << "\n";
      if (malloc)
        llvm::errs() << "malloc: " << *malloc << "\n";
      llvm::errs() << "tape: " << *tape << "\n";
      llvm::errs() << "idx: " << idx << "\n";
    }
    assert(idx < 0 ||
           (unsigned)idx < cast<StructType>(tape->getType())->getNumElements());
    Value *ret = (idx < 0) ? tape
                           : cast<Instruction>(BuilderQ.CreateExtractValue(
                                 tape, {(unsigned)idx}));

    if (ret->getType()->isEmptyTy()) {
      if (auto inst = dyn_cast_or_null<Instruction>(malloc)) {
        if (inst->getType() != ret->getType()) {
          llvm::errs() << "oldFunc: " << *oldFunc << "\n";
          llvm::errs() << "newFunc: " << *newFunc << "\n";
          llvm::errs() << "inst==malloc: " << *inst << "\n";
          llvm::errs() << "ret: " << *ret << "\n";
        }
        assert(inst->getType() == ret->getType());
        inst->replaceAllUsesWith(UndefValue::get(ret->getType()));
        erase(inst);
      }
      Type *retType = ret->getType();
      if (auto ri = dyn_cast<Instruction>(ret))
        erase(ri);
      return UndefValue::get(retType);
    }

    LimitContext ctx(BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }

    bool inLoop;
    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      if (malloc)
        ret->setName(malloc->getName() + "_fromtape");
    } else {
      if (auto ri = dyn_cast<Instruction>(ret))
        erase(ri);
      IRBuilder<> entryBuilder(inversionAllocs);
      entryBuilder.setFastMathFlags(getFast());
      ret = (idx < 0) ? tape
                      : cast<Instruction>(entryBuilder.CreateExtractValue(
                            tape, {(unsigned)idx}));

      Type *innerType = ret->getType();
      for (size_t i = 0, limit = getSubLimits(BuilderQ.GetInsertBlock()).size();
           i < limit; ++i) {
        if (!isa<PointerType>(innerType)) {
          llvm::errs() << "fn: " << *BuilderQ.GetInsertBlock()->getParent()
                       << "\n";
          llvm::errs() << "bq insertblock: " << *BuilderQ.GetInsertBlock()
                       << "\n";
          llvm::errs() << "ret: " << *ret << " type: " << *ret->getType()
                       << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          if (malloc)
            llvm::errs() << " malloc: " << *malloc << "\n";
        }
        assert(isa<PointerType>(innerType));
        innerType = cast<PointerType>(innerType)->getElementType();
      }

      assert(malloc);
      if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
          cast<IntegerType>(malloc->getType())->getBitWidth() == 1 &&
          innerType != ret->getType()) {
        assert(innerType == Type::getInt8Ty(malloc->getContext()));
      } else {
        if (innerType != malloc->getType()) {
          llvm::errs() << *cast<Instruction>(malloc)->getParent()->getParent()
                       << "\n";
          llvm::errs() << "innerType: " << *innerType << "\n";
          llvm::errs() << "malloc->getType(): " << *malloc->getType() << "\n";
          llvm::errs() << "ret: " << *ret << "\n";
          llvm::errs() << "malloc: " << *malloc << "\n";
        }
      }

      AllocaInst *cache =
          createCacheForScope(BuilderQ.GetInsertBlock(), innerType,
                              "mdyncache_fromtape", true, false);
      assert(malloc);
      bool isi1 = malloc->getType()->isIntegerTy() &&
                  cast<IntegerType>(malloc->getType())->getBitWidth() == 1;
      entryBuilder.CreateStore(ret, cache);

      auto v = lookupValueFromCache(/*forwardPass*/ true, BuilderQ,
                                    BuilderQ.GetInsertBlock(), cache, isi1);
      if (malloc) {
        assert(v->getType() == malloc->getType());
      }
      insert_or_assign(scopeMap, v, std::make_pair(cache, ctx));
      ret = cast<Instruction>(v);
    }

    if (malloc && !isa<UndefValue>(malloc)) {
      if (malloc->getType() != ret->getType()) {
        llvm::errs() << *oldFunc << "\n";
        llvm::errs() << *newFunc << "\n";
        llvm::errs() << *malloc << "\n";
        llvm::errs() << *ret << "\n";
      }
      assert(malloc->getType() == ret->getType());

      if (auto orig = isOriginal(malloc))
        originalToNewFn[orig] = ret;

      if (auto found = findInMap(scopeMap, malloc)) {
        // There already exists an alloaction for this, we should fully remove
        // it
        if (!inLoop) {

          // Remove stores into
          auto stores = scopeInstructions[found->first];
          scopeInstructions.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          std::vector<User *> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              IRBuilder<> lb(li);
              ValueToValueMapTy empty;
              li->replaceAllUsesWith(
                  unwrapM(ret, lb, empty, UnwrapMode::LegalFullUnwrap));
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap");
            }
          }

          {
            AllocaInst *preerase = found->first;
            scopeMap.erase(malloc);
            erase(preerase);
          }
        } else {
          // Remove stores into
          auto stores = scopeInstructions[found->first];
          scopeInstructions.erase(found->first);
          for (int i = stores.size() - 1; i >= 0; i--) {
            erase(stores[i]);
          }

          // Remove allocations for scopealloc since it is already allocated
          // by the augmented forward pass
          auto allocs = scopeAllocs[found->first];
          scopeAllocs.erase(found->first);
          for (auto allocinst : allocs) {
            CastInst *cast = nullptr;
            StoreInst *store = nullptr;
            for (auto use : allocinst->users()) {
              if (auto ci = dyn_cast<CastInst>(use)) {
                assert(cast == nullptr);
                cast = ci;
              }
              if (auto si = dyn_cast<StoreInst>(use)) {
                if (si->getValueOperand() == allocinst) {
                  assert(store == nullptr);
                  store = si;
                }
              }
            }
            if (cast) {
              assert(store == nullptr);
              for (auto use : cast->users()) {
                if (auto si = dyn_cast<StoreInst>(use)) {
                  if (si->getValueOperand() == cast) {
                    assert(store == nullptr);
                    store = si;
                  }
                }
              }
            }
            /*
            if (!store) {
                allocinst->getParent()->getParent()->dump();
                allocinst->dump();
            }
            assert(store);
            erase(store);
            */

            Instruction *storedinto =
                cast ? (Instruction *)cast : (Instruction *)allocinst;
            for (auto use : storedinto->users()) {
              // llvm::errs() << " found use of " << *storedinto << " of " <<
              // use << "\n";
              if (auto si = dyn_cast<StoreInst>(use))
                erase(si);
            }

            if (cast)
              erase(cast);
            // llvm::errs() << "considering inner loop for malloc: " <<
            // *malloc << " allocinst " << *allocinst << "\n";
            erase(allocinst);
          }

          // Remove frees
          auto tofree = scopeFrees[found->first];
          scopeFrees.erase(found->first);
          for (auto freeinst : tofree) {
            std::deque<Value *> ops = {freeinst->getArgOperand(0)};
            erase(freeinst);

            while (ops.size()) {
              auto z = dyn_cast<Instruction>(ops[0]);
              ops.pop_front();
              if (z && z->getNumUses() == 0) {
                for (unsigned i = 0; i < z->getNumOperands(); ++i) {
                  ops.push_back(z->getOperand(i));
                }
                erase(z);
              }
            }
          }

          // uses of the alloc
          std::vector<User *> users;
          for (auto u : found->first->users()) {
            users.push_back(u);
          }
          for (auto u : users) {
            if (auto li = dyn_cast<LoadInst>(u)) {
              IRBuilder<> lb(li);
              // llvm::errs() << "fixing li: " << *li << "\n";
              auto replacewith =
                  (idx < 0) ? tape
                            : lb.CreateExtractValue(tape, {(unsigned)idx});
              // llvm::errs() << "fixing with rw: " << *replacewith << "\n";
              li->replaceAllUsesWith(replacewith);
              erase(li);
            } else {
              llvm::errs() << "newFunc: " << *newFunc << "\n";
              llvm::errs() << "malloc: " << *malloc << "\n";
              llvm::errs() << "scopeMap[malloc]: " << *found->first << "\n";
              llvm::errs() << "u: " << *u << "\n";
              assert(0 && "illegal use for out of loop scopeMap");
            }
          }

          // cast<Instruction>(scopeMap[malloc])->getParent()->getParent()->dump();

          // llvm::errs() << "did erase for malloc: " << *malloc << " " <<
          // *scopeMap[malloc] << "\n";

          AllocaInst *preerase = found->first;
          scopeMap.erase(malloc);
          erase(preerase);
        }
      }
      // llvm::errs() << "replacing " << *malloc << " with " << *ret << "\n";
      cast<Instruction>(malloc)->replaceAllUsesWith(ret);
      std::string n = malloc->getName().str();
      erase(cast<Instruction>(malloc));
      ret->setName(n);
    }
    return ret;
  } else {
    assert(malloc);
    // assert(!isa<PHINode>(malloc));

    assert(idx >= 0 && (unsigned)idx == addedTapeVals.size());

    if (isa<UndefValue>(malloc)) {
      addedTapeVals.push_back(malloc);
      return malloc;
    }

    LimitContext ctx(BuilderQ.GetInsertBlock());
    if (auto inst = dyn_cast<Instruction>(malloc))
      ctx = LimitContext(inst->getParent());
    if (auto found = findInMap(scopeMap, malloc)) {
      ctx = found->second;
    }

    bool inLoop;

    if (ctx.ForceSingleIteration) {
      inLoop = true;
      ctx.ForceSingleIteration = false;
    } else {
      LoopContext lc;
      inLoop = getContext(ctx.Block, lc);
    }

    if (!inLoop) {
      addedTapeVals.push_back(malloc);
      return malloc;
    }

    ensureLookupCached(cast<Instruction>(malloc),
                       /*shouldFree=*/reverseBlocks.size() > 0);
    auto found2 = scopeMap.find(malloc);
    assert(found2 != scopeMap.end());
    assert(found2->second.first);

    Value *toadd;
    // if (ompOffset) {
    //  toadd = UndefValue::get(found2->second.first->getAllocatedType());
    //} else {
    toadd = scopeAllocs[found2->second.first][0];
    for (auto u : toadd->users()) {
      if (auto ci = dyn_cast<CastInst>(u)) {
        toadd = ci;
      }
    }
    //}

    // llvm::errs() << " malloc: " << *malloc << "\n";
    // llvm::errs() << " toadd: " << *toadd << "\n";
    Type *innerType = toadd->getType();
    for (size_t
             i = 0,
             limit =
                 getSubLimits(LimitContext(BuilderQ.GetInsertBlock())).size();
         i < limit; ++i) {
      innerType = cast<PointerType>(innerType)->getElementType();
    }

    if (EfficientBoolCache && malloc->getType()->isIntegerTy() &&
        toadd->getType() != innerType &&
        cast<IntegerType>(malloc->getType())->getBitWidth() == 1) {
      assert(innerType == Type::getInt8Ty(toadd->getContext()));
    } else {
      if (innerType != malloc->getType()) {
        llvm::errs() << "oldFunc:" << *oldFunc << "\n";
        llvm::errs() << "newFunc: " << *newFunc << "\n";
        llvm::errs() << " toadd: " << *toadd << "\n";
        llvm::errs() << "innerType: " << *innerType << "\n";
        llvm::errs() << "malloc: " << *malloc << "\n";
      }
      assert(innerType == malloc->getType());
    }
    addedTapeVals.push_back(toadd);
    return malloc;
  }
  llvm::errs()
      << "Fell through on cacheForReverse. This should never happen.\n";
  assert(false);
}

/// Given an edge from BB to branchingBlock get the corresponding block to
/// branch to in the reverse pass
BasicBlock *GradientUtils::getReverseOrLatchMerge(BasicBlock *BB,
                                                  BasicBlock *branchingBlock) {
  assert(BB);
  // BB should be a forward pass block, assert that
  if (reverseBlocks.find(BB) == reverseBlocks.end()) {
    llvm::errs() << *oldFunc << "\n";
    llvm::errs() << *newFunc << "\n";
    llvm::errs() << "BB: " << *BB << "\n";
    llvm::errs() << "branchingBlock: " << *branchingBlock << "\n";
  }
  assert(reverseBlocks.find(BB) != reverseBlocks.end());
  LoopContext lc;
  bool inLoop = getContext(BB, lc);

  LoopContext branchingContext;
  bool inLoopContext = getContext(branchingBlock, branchingContext);

  if (!inLoop)
    return reverseBlocks[BB];

  auto tup = std::make_tuple(BB, branchingBlock);
  if (newBlocksForLoop_cache.find(tup) != newBlocksForLoop_cache.end())
    return newBlocksForLoop_cache[tup];

  if (inLoop && inLoopContext && branchingBlock == lc.header &&
      lc.header == branchingContext.header) {
    BasicBlock *incB = BasicBlock::Create(
        BB->getContext(), "inc" + reverseBlocks[lc.header]->getName(),
        BB->getParent());
    incB->moveAfter(reverseBlocks[lc.header]);

    IRBuilder<> tbuild(incB);

    Value *av = tbuild.CreateLoad(lc.antivaralloc);
    Value *sub = tbuild.CreateAdd(av, ConstantInt::get(av->getType(), -1), "",
                                  /*NUW*/ false, /*NSW*/ true);
    tbuild.CreateStore(sub, lc.antivaralloc);
    tbuild.CreateBr(reverseBlocks[BB]);
    return newBlocksForLoop_cache[tup] = incB;
  }

  if (inLoop) {
    auto latches = getLatches(LI.getLoopFor(BB), lc.exitBlocks);

    if (std::find(latches.begin(), latches.end(), BB) != latches.end() &&
        std::find(lc.exitBlocks.begin(), lc.exitBlocks.end(), branchingBlock) !=
            lc.exitBlocks.end()) {
      BasicBlock *incB =
          BasicBlock::Create(BB->getContext(),
                             "merge" + reverseBlocks[lc.header]->getName() +
                                 "_" + branchingBlock->getName(),
                             BB->getParent());
      incB->moveAfter(reverseBlocks[branchingBlock]);

      IRBuilder<> tbuild(reverseBlocks[branchingBlock]);

      Value *lim = nullptr;
      if (lc.dynamic) {
        lim = lookupValueFromCache(/*forwardPass*/ false, tbuild, lc.preheader,
                                   cast<AllocaInst>(lc.limit), /*isi1*/ false);
      } else {
        lim = lookupM(lc.limit, tbuild);
      }

      tbuild.SetInsertPoint(incB);
      tbuild.CreateStore(lim, lc.antivaralloc);
      tbuild.CreateBr(reverseBlocks[BB]);

      return newBlocksForLoop_cache[tup] = incB;
    }
  }

  return newBlocksForLoop_cache[tup] = reverseBlocks[BB];
}

void GradientUtils::forceContexts() {
  for (auto BB : originalBlocks) {
    LoopContext lc;
    getContext(BB, lc);
  }
}

bool GradientUtils::legalRecompute(const Value *val,
                                   const ValueToValueMapTy &available) const {
  if (available.count(val)) {
    return true;
  }

  if (isa<PHINode>(val)) {
    if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(val))) {
      return legalRecompute(
          dli, available); // TODO ADD && !TR.intType(getOriginal(dli),
                           // /*mustfind*/false).isPossibleFloat();
    }
    // if (SE.isSCEVable(phi->getType())) {
    // auto scev =
    // const_cast<GradientUtils*>(this)->SE.getSCEV(const_cast<Value*>(val));
    // llvm::errs() << "phi: " << *val << " scev: " << *scev << "\n";
    //}
    // llvm::errs() << "illegal recompute: " << *val << "\n";
    return false;
  }

  if (isa<Instruction>(val) &&
      cast<Instruction>(val)->getMetadata("enzyme_mustcache")) {
    return false;
  }

  // If this is a load from cache already, dont force a cache of this
  if (isa<LoadInst>(val) && CacheLookups.count(cast<LoadInst>(val)))
    return true;

  // TODO consider callinst here

  if (auto li = dyn_cast<LoadInst>(val)) {

    // If this is an already unwrapped value, legal to recompute again.
    if (li->getMetadata("enzyme_unwrapped"))
      return true;

    const Instruction *orig = nullptr;
    if (li->getParent()->getParent() == oldFunc) {
      orig = li;
    } else {
      orig = isOriginal(li);
      // todo consider when we pass non original queries
    }

    if (orig) {
      auto found = can_modref_map->find(const_cast<Instruction *>(orig));
      if (found == can_modref_map->end()) {
        llvm::errs() << "can_modref_map:\n";
        for (auto &pair : *can_modref_map) {
          llvm::errs() << " + " << *pair.first << ": " << pair.second
                       << " of func "
                       << pair.first->getParent()->getParent()->getName()
                       << "\n";
        }
        llvm::errs() << "couldn't find in can_modref_map: " << *li
                     << " in fn: " << orig->getParent()->getParent()->getName();
      }
      assert(found != can_modref_map->end());
      return !found->second;
    } else {
      if (auto dli = dyn_cast_or_null<LoadInst>(hasUninverted(li))) {
        return legalRecompute(dli, available);
      }

      // TODO mark all the explicitly legal nodes (caches, etc)
      return true;
      llvm::errs() << *li
                   << " parent: " << li->getParent()->getParent()->getName()
                   << "\n";
      llvm_unreachable("unknown load to redo!");
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" ||
          n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r" ||
          n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" ||
          n == "__lgammal_r_finite" || isMemFreeLibMFunction(n)) {
        return true;
      }
    }
  }

  if (auto inst = dyn_cast<Instruction>(val)) {
    if (inst->mayReadOrWriteMemory()) {
      return false;
    }
  }

  return true;
}

//! Given the option to recompute a value or re-use an old one, return true if
//! it is faster to recompute this value from scratch
bool GradientUtils::shouldRecompute(const Value *val,
                                    const ValueToValueMapTy &available) {
  if (available.count(val))
    return true;
  // TODO: remake such that this returns whether a load to a cache is more
  // expensive than redoing the computation.

  // If this is a load from cache already, just reload this
  if (isa<LoadInst>(val) &&
      cast<LoadInst>(val)->getMetadata("enzyme_fromcache"))
    return true;

  if (isa<CastInst>(val) || isa<GetElementPtrInst>(val))
    return true;

  if (!isa<Instruction>(val))
    return true;

  // llvm::errs() << " considering recompute of " << *val << "\n";
  const Instruction *inst = cast<Instruction>(val);

  // if this has operands that need to be loaded and haven't already been loaded
  // TODO, just cache this
  for (auto &op : inst->operands()) {
    if (!legalRecompute(op, available)) {

      // If this is a load from cache already, dont force a cache of this
      if (isa<LoadInst>(op) && CacheLookups.count(cast<LoadInst>(op)))
        continue;

      // If a previously cached this operand, don't let it trigger the
      // heuristic for caching this value instead.
      if (scopeMap.find(op) != scopeMap.end())
        continue;

      // If the actually uncacheable operand is in a different loop scope
      // don't cache this value instead as it may require more memory
      LoopContext lc1;
      LoopContext lc2;
      bool inLoop1 =
          getContext(const_cast<Instruction *>(inst)->getParent(), lc1);
      bool inLoop2 = getContext(cast<Instruction>(op)->getParent(), lc2);
      if (inLoop1 != inLoop2 || (inLoop1 && (lc1.header != lc2.header))) {
        continue;
      }

      // If a placeholder phi for inversion (and we know from above not
      // recomputable)
      if (!isa<PHINode>(op) && dyn_cast_or_null<LoadInst>(hasUninverted(op))) {
        goto forceCache;
      }

      // Even if cannot recompute (say a phi node), don't force a reload if it
      // is possible to just use this instruction from forward pass without
      // issue
      if (auto i2 = dyn_cast<Instruction>(op)) {
        if (!i2->mayReadOrWriteMemory()) {
          LoopContext lc;
          bool inLoop = const_cast<GradientUtils *>(this)->getContext(
              i2->getParent(), lc);
          if (!inLoop) {
            if (i2->getParent() == &newFunc->getEntryBlock()) {
              continue;
            }
            // TODO upgrade this to be all returns that this could enter from
            bool legal = true;
            for (auto &BB : *oldFunc) {
              if (isa<ReturnInst>(BB.getTerminator())) {
                BasicBlock *returningBlock =
                    cast<BasicBlock>(getNewFromOriginal(&BB));
                if (i2->getParent() == returningBlock)
                  continue;
                if (!DT.dominates(i2, returningBlock)) {
                  legal = false;
                  break;
                }
              }
            }
            if (legal) {
              continue;
            }
          }
        }
      }
    forceCache:;
      // llvm::errs() << "shouldn't recompute " << *inst << "because of illegal
      // redo op: " << *op << "\n";
      return false;
    }
  }

  if (auto op = dyn_cast<IntrinsicInst>(val)) {
    if (!op->mayReadOrWriteMemory())
      return true;
    switch (op->getIntrinsicID()) {
    case Intrinsic::sin:
    case Intrinsic::cos:
    case Intrinsic::exp:
    case Intrinsic::log:
      return true;
    default:
      return false;
    }
  }

  if (auto ci = dyn_cast<CallInst>(val)) {
    if (auto called = ci->getCalledFunction()) {
      auto n = called->getName();
      if (n == "lgamma" || n == "lgammaf" || n == "lgammal" ||
          n == "lgamma_r" || n == "lgammaf_r" || n == "lgammal_r" ||
          n == "__lgamma_r_finite" || n == "__lgammaf_r_finite" ||
          n == "__lgammal_r_finite" || n == "tanh" || n == "tanhf" ||
          n == "__pow_finite" || isMemFreeLibMFunction(n)) {
        return true;
      }
    }
  }

  // cache a call, assuming its longer to run that
  if (isa<CallInst>(val)) {
    llvm::errs() << " caching call: " << *val << "\n";
    // cast<CallInst>(val)->getCalledFunction()->dump();
    return false;
  }

  return true;
}

GradientUtils *GradientUtils::CreateFromClone(
    Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA, AAResults &AA,
    DIFFE_TYPE retType, const std::vector<DIFFE_TYPE> &constant_args,
    bool returnUsed, std::map<AugmentedStruct, int> &returnMapping) {
  assert(!todiff->empty());

  // Since this is forward pass this should always return the tape (at index 0)
  returnMapping[AugmentedStruct::Tape] = 0;

  int returnCount = 0;

  if (returnUsed) {
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    returnMapping[AugmentedStruct::Return] = returnCount + 1;
    ++returnCount;
  }

  // We don't need to differentially return something that we know is not a
  // pointer (or somehow needed for shadow analysis)
  if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
    assert(!todiff->getReturnType()->isEmptyTy());
    assert(!todiff->getReturnType()->isVoidTy());
    assert(!todiff->getReturnType()->isFPOrFPVectorTy());
    returnMapping[AugmentedStruct::DifferentialReturn] = returnCount + 1;
    ++returnCount;
  }

  ReturnType returnValue;
  if (returnCount == 0)
    returnValue = ReturnType::Tape;
  else if (returnCount == 1)
    returnValue = ReturnType::TapeAndReturn;
  else if (returnCount == 2)
    returnValue = ReturnType::TapeAndTwoReturns;
  else
    llvm_unreachable("illegal number of elements in augmented return struct");

  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  ValueToValueMapTy originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  auto newFunc = CloneFunctionWithReturns(
      /*topLevel*/ false, todiff, AA, TLI, invertedPointers, constant_args,
      constant_values, nonconstant_values, returnvals,
      /*returnValue*/ returnValue, "fakeaugmented_" + todiff->getName(),
      &originalToNew,
      /*diffeReturnArg*/ false, /*additionalArg*/ nullptr);

  auto res = new GradientUtils(newFunc, todiff, TLI, TA, AA, invertedPointers,
                               constant_values, nonconstant_values,
                               /*ActiveValues*/ retType != DIFFE_TYPE::CONSTANT,
                               originalToNew, DerivativeMode::Forward);
  return res;
}

DiffeGradientUtils *DiffeGradientUtils::CreateFromClone(
    bool topLevel, Function *todiff, TargetLibraryInfo &TLI, TypeAnalysis &TA,
    AAResults &AA, DIFFE_TYPE retType,
    const std::vector<DIFFE_TYPE> &constant_args, ReturnType returnValue,
    Type *additionalArg) {
  assert(!todiff->empty());
  ValueToValueMapTy invertedPointers;
  SmallPtrSet<Instruction *, 4> constants;
  SmallPtrSet<Instruction *, 20> nonconstant;
  SmallPtrSet<Value *, 2> returnvals;
  ValueToValueMapTy originalToNew;

  SmallPtrSet<Value *, 4> constant_values;
  SmallPtrSet<Value *, 4> nonconstant_values;

  bool diffeReturnArg = retType == DIFFE_TYPE::OUT_DIFF;
  auto newFunc = CloneFunctionWithReturns(
      topLevel, todiff, AA, TLI, invertedPointers, constant_args,
      constant_values, nonconstant_values, returnvals, returnValue,
      "diffe" + todiff->getName(), &originalToNew,
      /*diffeReturnArg*/ diffeReturnArg, additionalArg);
  auto res = new DiffeGradientUtils(
      newFunc, todiff, TLI, TA, AA, invertedPointers, constant_values,
      nonconstant_values, /*ActiveValues*/ retType != DIFFE_TYPE::CONSTANT,
      originalToNew, topLevel ? DerivativeMode::Both : DerivativeMode::Reverse);
  return res;
}

Value *GradientUtils::invertPointerM(Value *oval, IRBuilder<> &BuilderM) {
  assert(oval);
  if (auto inst = dyn_cast<Instruction>(oval)) {
    assert(inst->getParent()->getParent() == oldFunc);
  }
  if (auto arg = dyn_cast<Argument>(oval)) {
    assert(arg->getParent() == oldFunc);
  }

  if (isa<ConstantPointerNull>(oval)) {
    return oval;
  } else if (isa<UndefValue>(oval)) {
    return oval;
  } else if (auto cint = dyn_cast<ConstantInt>(oval)) {
    return cint;
  }

  if (isConstantValue(oval)) {
    // NOTE, this is legal and the correct resolution, however, our activity
    // analysis honeypot no longer exists
    return lookupM(getNewFromOriginal(oval), BuilderM);
  }
  assert(!isConstantValue(oval));

  auto M = oldFunc->getParent();
  assert(oval);

  if (invertedPointers.find(oval) != invertedPointers.end()) {
    return lookupM(invertedPointers[oval], BuilderM);
  }

  if (auto arg = dyn_cast<GlobalVariable>(oval)) {
    if (!hasMetadata(arg, "enzyme_shadow")) {

      if (mode == DerivativeMode::Both && arg->getType()->getPointerAddressSpace() == 0) {
        bool seen = false;
        MemoryLocation
#if LLVM_VERSION_MAJOR >= 12
        Loc = MemoryLocation(oval, LocationSize::beforeOrAfterPointer());
#elif LLVM_VERSION_MAJOR >= 9
        Loc = MemoryLocation(oval, LocationSize::unknown());
#else
        Loc = MemoryLocation(oval, MemoryLocation::UnknownSize);
#endif
        for (CallInst* CI : originalCalls) {
          if (isa<IntrinsicInst>(CI)) continue;
          if (!isConstantInstruction(CI)) {
            Function* F = CI->getCalledFunction();
            #if LLVM_VERSION_MAJOR >= 11
              if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledOperand()))
            #else
              if (auto castinst = dyn_cast<ConstantExpr>(CI->getCalledValue()))
            #endif
              {
                if (castinst->isCast())
                  if (auto fn = dyn_cast<Function>(castinst->getOperand(0))) {
                      F = fn;
                  }
              }
            if (F && (isMemFreeLibMFunction(F->getName()) || F->getName() == "__fd_sincos_1")) {
              continue;
            }
            if (llvm::isModOrRefSet(AA.getModRefInfo(CI, Loc))) {
              seen = true;
              llvm::errs() << " cannot handle global " << *oval << " due to " << *CI << "\n";
              goto endCheck;
            }
          }
        }
        endCheck:;
        if (!seen) {
          IRBuilder<> bb(inversionAllocs);
          AllocaInst *antialloca = bb.CreateAlloca(arg->getValueType(),
            arg->getType()->getPointerAddressSpace(), nullptr, arg->getName() + "'ipa");
          invertedPointers[arg] = antialloca;

          if (arg->getAlignment()) {
      #if LLVM_VERSION_MAJOR >= 10
            antialloca->setAlignment(Align(arg->getAlignment()));
      #else
            antialloca->setAlignment(arg->getAlignment());
      #endif
          }

          auto st = bb.CreateStore(
              Constant::getNullValue(arg->getValueType()), antialloca);
          if (arg->getAlignment()) {
  #if LLVM_VERSION_MAJOR >= 10
            st->setAlignment(Align(arg->getAlignment()));
  #else
            st->setAlignment(arg->getAlignment());
  #endif
          }
          assert(invertedPointers[arg]->getType() == arg->getType());
          return lookupM(invertedPointers[arg], BuilderM);
        }
      }

      if ((llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch() ==
               Triple::nvptx ||
           llvm::Triple(newFunc->getParent()->getTargetTriple()).getArch() ==
               Triple::nvptx64) &&
          cast<PointerType>(arg->getType())->getAddressSpace() == 3) {
        llvm::errs() << "warning found shared memory\n";
        //#if LLVM_VERSION_MAJOR >= 11
        Type *type = cast<PointerType>(arg->getType())->getElementType();
        // TODO this needs initialization by entry
        auto shadow = new GlobalVariable(
            *arg->getParent(), type, arg->isConstant(), arg->getLinkage(),
            UndefValue::get(type), arg->getName() + "_shadow", arg,
            arg->getThreadLocalMode(), arg->getType()->getAddressSpace(),
            arg->isExternallyInitialized());
        arg->setMetadata("enzyme_shadow",
                         MDTuple::get(shadow->getContext(),
                                      {ConstantAsMetadata::get(shadow)}));
        shadow->setMetadata("enzyme_internalshadowglobal",
                            MDTuple::get(shadow->getContext(), {}));
        return invertedPointers[oval] = shadow;
        //#endif
      }

      llvm::errs() << *oldFunc->getParent() << "\n";
      llvm::errs() << *oldFunc << "\n";
      llvm::errs() << *newFunc << "\n";
      llvm::errs() << *arg << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global");
    }
    auto md = arg->getMetadata("enzyme_shadow");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *arg << "\n";
      llvm::errs() << *md << "\n";
      assert(0 && "cannot compute with global variable that doesn't have "
                  "marked shadow global");
      report_fatal_error("cannot compute with global variable that doesn't "
                         "have marked shadow global (metadata incorrect type)");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto cs = gvemd->getValue();
    return invertedPointers[oval] = cs;
  } else if (auto fn = dyn_cast<Function>(oval)) {
    //! Todo allow tape propagation
    //  Note that specifically this should _not_ be called with topLevel=true
    //  (since it may not be valid to always assume we can recompute the
    //  augmented primal) However, in the absence of a way to pass tape data
    //  from an indirect augmented (and also since we dont presently allow
    //  indirect augmented calls), topLevel MUST be true otherwise subcalls will
    //  not be able to lookup the augmenteddata/subdata (triggering an assertion
    //  failure, among much worse)
    std::map<Argument *, bool> uncacheable_args;
    FnTypeInfo type_args(fn);

    // conservatively assume that we can only cache existing floating types
    // (i.e. that all args are uncacheable)
    std::vector<DIFFE_TYPE> types;
    for (auto &a : fn->args()) {
      uncacheable_args[&a] = !a.getType()->isFPOrFPVectorTy();
      type_args.Arguments.insert(std::pair<Argument *, TypeTree>(&a, {}));
      type_args.KnownValues.insert(
          std::pair<Argument *, std::set<int64_t>>(&a, {}));
      DIFFE_TYPE typ;
      if (a.getType()->isFPOrFPVectorTy()) {
        typ = DIFFE_TYPE::OUT_DIFF;
      } else if (a.getType()->isIntegerTy() &&
                 cast<IntegerType>(a.getType())->getBitWidth() < 16) {
        typ = DIFFE_TYPE::CONSTANT;
      } else if (a.getType()->isVoidTy() || a.getType()->isEmptyTy()) {
        typ = DIFFE_TYPE::CONSTANT;
      } else {
        typ = DIFFE_TYPE::DUP_ARG;
      }
      types.push_back(typ);
    }

    DIFFE_TYPE retType = fn->getReturnType()->isFPOrFPVectorTy()
                             ? DIFFE_TYPE::OUT_DIFF
                             : DIFFE_TYPE::DUP_ARG;
    if (fn->getReturnType()->isVoidTy() || fn->getReturnType()->isEmptyTy() ||
        (fn->getReturnType()->isIntegerTy() &&
         cast<IntegerType>(fn->getReturnType())->getBitWidth() < 16))
      retType = DIFFE_TYPE::CONSTANT;

    // TODO re atomic add consider forcing it to be atomic always as fallback if
    // used in a parallel context
    auto &augdata = CreateAugmentedPrimal(
        fn, retType, /*constant_args*/ types, TLI, TA, AA,
        /*returnUsed*/ !fn->getReturnType()->isEmptyTy() &&
            !fn->getReturnType()->isVoidTy(),
        type_args, uncacheable_args, /*forceAnonymousTape*/ true, AtomicAdd,
        /*PostOpt*/ false);
    Constant *newf = CreatePrimalAndGradient(
        fn, retType, /*constant_args*/ types, TLI, TA, AA,
        /*returnValue*/ false, /*dretPtr*/ false, /*topLevel*/ false,
        /*additionalArg*/ Type::getInt8PtrTy(fn->getContext()), type_args,
        uncacheable_args,
        /*map*/ &augdata, AtomicAdd);
    if (!newf)
      newf = UndefValue::get(fn->getType());
    auto cdata = ConstantStruct::get(
        StructType::get(newf->getContext(),
                        {augdata.fn->getType(), newf->getType()}),
        {augdata.fn, newf});
    std::string globalname = ("_enzyme_" + fn->getName() + "'").str();
    auto GV = fn->getParent()->getNamedValue(globalname);

    if (GV == nullptr) {
      GV = new GlobalVariable(*fn->getParent(), cdata->getType(), true,
                              GlobalValue::LinkageTypes::InternalLinkage, cdata,
                              globalname);
    }

    return BuilderM.CreatePointerCast(GV, fn->getType());
  } else if (auto arg = dyn_cast<CastInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    invertedPointers[arg] =
        bb.CreateCast(arg->getOpcode(), invertPointerM(arg->getOperand(0), bb),
                      arg->getDestTy(), arg->getName() + "'ipc");
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<ConstantExpr>(oval)) {
    IRBuilder<> bb(inversionAllocs);
    auto ip = invertPointerM(arg->getOperand(0), bb);
    if (arg->isCast()) {
      if (auto C = dyn_cast<Constant>(ip))
        return ConstantExpr::getCast(
          arg->getOpcode(), C,
          arg->getType());
      else {
        invertedPointers[arg] =
            bb.CreateCast((Instruction::CastOps)arg->getOpcode(), ip,
                          arg->getType(), arg->getName() + "'ipc");
        return lookupM(invertedPointers[arg], BuilderM);
      }
    } else if (arg->getOpcode() == Instruction::GetElementPtr) {
      if (auto C = dyn_cast<Constant>(ip))
        return arg->getWithOperandReplaced(0, C);
      else {
        SmallVector<Value *, 4> invertargs;
        for (unsigned i = 0; i < arg->getNumOperands() - 1; ++i) {
          Value *b = getNewFromOriginal(arg->getOperand(1 + i));
          invertargs.push_back(b);
        }
        auto result = bb.CreateGEP(ip,
                                  invertargs, arg->getName() + "'ipg");
        //if (auto gep = dyn_cast<GetElementPtrInst>(result))
        //  gep->setIsInBounds(arg->isInBounds());
        invertedPointers[arg] = result;
        return lookupM(invertedPointers[arg], BuilderM);
      }
    } else {
      llvm::errs() << *arg << "\n";
      assert(0 && "unhandled");
    }
    goto end;
  } else if (auto arg = dyn_cast<ExtractValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto result =
        bb.CreateExtractValue(invertPointerM(arg->getOperand(0), bb),
                              arg->getIndices(), arg->getName() + "'ipev");
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<InsertValueInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto result =
        bb.CreateInsertValue(invertPointerM(arg->getOperand(0), bb),
                             invertPointerM(arg->getOperand(1), bb),
                             arg->getIndices(), arg->getName() + "'ipiv");
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<ExtractElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto result = bb.CreateExtractElement(
        invertPointerM(arg->getVectorOperand(), bb),
        getNewFromOriginal(arg->getIndexOperand()), arg->getName() + "'ipee");
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<InsertElementInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
    Value *op2 = arg->getOperand(2);
    auto result = bb.CreateInsertElement(
        invertPointerM(op0, bb), invertPointerM(op1, bb),
        getNewFromOriginal(op2), arg->getName() + "'ipie");
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<ShuffleVectorInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    Value *op1 = arg->getOperand(1);
#if LLVM_VERSION_MAJOR >= 11
    auto result = bb.CreateShuffleVector(
        invertPointerM(op0, bb), invertPointerM(op1, bb),
        arg->getShuffleMaskForBitcode(), arg->getName() + "'ipsv");
#else
    auto result =
        bb.CreateShuffleVector(invertPointerM(op0, bb), invertPointerM(op1, bb),
                               arg->getOperand(2), arg->getName() + "'ipsv");
#endif
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<SelectInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    auto result = bb.CreateSelect(getNewFromOriginal(arg->getCondition()),
                                  invertPointerM(arg->getTrueValue(), bb),
                                  invertPointerM(arg->getFalseValue(), bb),
                                  arg->getName() + "'ipse");
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<LoadInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *op0 = arg->getOperand(0);
    auto li = bb.CreateLoad(invertPointerM(op0, bb), arg->getName() + "'ipl");
#if LLVM_VERSION_MAJOR >= 10
    li->setAlignment(arg->getAlign());
#else
    li->setAlignment(arg->getAlignment());
#endif
    li->setVolatile(arg->isVolatile());
    li->setOrdering(arg->getOrdering());
    li->setSyncScopeID(arg->getSyncScopeID());
    invertedPointers[arg] = li;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<BinaryOperator>(oval)) {
    assert(arg->getType()->isIntOrIntVectorTy());
    IRBuilder<> bb(getNewFromOriginal(arg));
    Value *val0 = nullptr;
    Value *val1 = nullptr;

    val0 = invertPointerM(arg->getOperand(0), bb);
    val1 = invertPointerM(arg->getOperand(1), bb);

    auto li = bb.CreateBinOp(arg->getOpcode(), val0, val1, arg->getName());
    if (auto BI = dyn_cast<BinaryOperator>(li))
      BI->copyIRFlags(arg);
    invertedPointers[arg] = li;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto arg = dyn_cast<GetElementPtrInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(arg));
    SmallVector<Value *, 4> invertargs;
    for (unsigned i = 0; i < arg->getNumIndices(); ++i) {
      Value *b = getNewFromOriginal(arg->getOperand(1 + i));
      invertargs.push_back(b);
    }
    auto result = bb.CreateGEP(invertPointerM(arg->getPointerOperand(), bb),
                               invertargs, arg->getName() + "'ipg");
    if (auto gep = dyn_cast<GetElementPtrInst>(result))
      gep->setIsInBounds(arg->isInBounds());
    invertedPointers[arg] = result;
    return lookupM(invertedPointers[arg], BuilderM);
  } else if (auto inst = dyn_cast<AllocaInst>(oval)) {
    IRBuilder<> bb(getNewFromOriginal(inst));
    Value *asize = getNewFromOriginal(inst->getArraySize());
    AllocaInst *antialloca = bb.CreateAlloca(
        inst->getAllocatedType(), inst->getType()->getPointerAddressSpace(),
        asize, inst->getName() + "'ipa");
    invertedPointers[inst] = antialloca;

    if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
      antialloca->setAlignment(Align(inst->getAlignment()));
#else
      antialloca->setAlignment(inst->getAlignment());
#endif
    }

    if (auto ci = dyn_cast<ConstantInt>(asize)) {
      if (ci->isOne()) {
        auto st = bb.CreateStore(
            Constant::getNullValue(inst->getAllocatedType()), antialloca);
        if (inst->getAlignment()) {
#if LLVM_VERSION_MAJOR >= 10
          st->setAlignment(Align(inst->getAlignment()));
#else
          st->setAlignment(inst->getAlignment());
#endif
        }
        return lookupM(invertedPointers[inst], BuilderM);
      } else {
        // TODO handle alloca of size > 1
      }
    }

    auto dst_arg =
        bb.CreateBitCast(antialloca, Type::getInt8PtrTy(oval->getContext()));
    auto val_arg = ConstantInt::get(Type::getInt8Ty(oval->getContext()), 0);
    auto len_arg = bb.CreateMul(
        bb.CreateZExtOrTrunc(asize, Type::getInt64Ty(oval->getContext())),
        ConstantInt::get(Type::getInt64Ty(oval->getContext()),
                         M->getDataLayout().getTypeAllocSizeInBits(
                             inst->getAllocatedType()) /
                             8),
        "", true, true);
    auto volatile_arg = ConstantInt::getFalse(oval->getContext());

#if LLVM_VERSION_MAJOR == 6
    auto align_arg = ConstantInt::get(Type::getInt32Ty(oval->getContext()),
                                      antialloca->getAlignment());
    Value *args[] = {dst_arg, val_arg, len_arg, align_arg, volatile_arg};
#else
    Value *args[] = {dst_arg, val_arg, len_arg, volatile_arg};
#endif
    Type *tys[] = {dst_arg->getType(), len_arg->getType()};
    auto memset = cast<CallInst>(bb.CreateCall(
        Intrinsic::getDeclaration(M, Intrinsic::memset, tys), args));
#if LLVM_VERSION_MAJOR >= 10
    if (inst->getAlignment()) {
      memset->addParamAttr(
          0, Attribute::getWithAlignment(inst->getContext(),
                                         Align(inst->getAlignment())));
    }
#else
    if (inst->getAlignment() != 0) {
      memset->addParamAttr(0, Attribute::getWithAlignment(
                                  inst->getContext(), inst->getAlignment()));
    }
#endif
    memset->addParamAttr(0, Attribute::NonNull);
    return lookupM(invertedPointers[inst], BuilderM);
  } else if (auto phi = dyn_cast<PHINode>(oval)) {

    if (phi->getNumIncomingValues() == 0) {
      dumpMap(invertedPointers);
      assert(0 && "illegal iv of phi");
    }
    std::map<Value *, std::set<BasicBlock *>> mapped;
    for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
      mapped[phi->getIncomingValue(i)].insert(phi->getIncomingBlock(i));
    }

    if (false && mapped.size() == 1) {
      return invertPointerM(phi->getIncomingValue(0), BuilderM);
    }
#if 0
     else if (false && mapped.size() == 2) {
         IRBuilder <> bb(phi);
         auto which = bb.CreatePHI(Type::getInt1Ty(phi->getContext()), phi->getNumIncomingValues());
         //TODO this is not recursive

         int cnt = 0;
         Value* vals[2];
         for(auto v : mapped) {
            assert( cnt <= 1 );
            vals[cnt] = v.first;
            for (auto b : v.second) {
                which->addIncoming(ConstantInt::get(which->getType(), cnt), b);
            }
            ++cnt;
         }

         auto which2 = lookupM(which, BuilderM);
         auto result = BuilderM.CreateSelect(which2, invertPointerM(vals[1], BuilderM), invertPointerM(vals[0], BuilderM));
         return result;
     }
#endif

    else {
      auto NewV = getNewFromOriginal(phi);
      IRBuilder<> bb(NewV);
      // Note if the original phi node get's scev'd in NewF, it may
      // no longer be a phi and we need a new place to insert this phi
      // Note that if scev'd this can still be a phi with 0 incoming indicating
      // an unnecessary value to be replaced
      // TODO consider allowing the inverted pointer to become a scev
      if (!isa<PHINode>(NewV) ||
          cast<PHINode>(NewV)->getNumIncomingValues() == 0) {
        bb.SetInsertPoint(bb.GetInsertBlock()->getFirstNonPHI());
      }
      auto which = bb.CreatePHI(phi->getType(), phi->getNumIncomingValues());
      invertedPointers[phi] = which;
      for (unsigned int i = 0; i < phi->getNumIncomingValues(); ++i) {
        IRBuilder<> pre(
            cast<BasicBlock>(getNewFromOriginal(phi->getIncomingBlock(i)))
                ->getTerminator());
        Value *val = invertPointerM(phi->getIncomingValue(i), pre);
        which->addIncoming(val, cast<BasicBlock>(getNewFromOriginal(
                                    phi->getIncomingBlock(i))));
      }

      return lookupM(which, BuilderM);
    }
  }

end:;
  assert(BuilderM.GetInsertBlock());
  assert(BuilderM.GetInsertBlock()->getParent());
  assert(oval);

  llvm::errs() << *BuilderM.GetInsertBlock()->getParent()->getParent() << "\n";
  llvm::errs() << "fn:" << *BuilderM.GetInsertBlock()->getParent()
               << "\noval=" << *oval << " icv=" << isConstantValue(oval)
               << "\n";
  for (auto z : invertedPointers) {
    llvm::errs() << "available inversion for " << *z.first << " of "
                 << *z.second << "\n";
  }
  assert(0 && "cannot find deal with ptr that isnt arg");
  report_fatal_error("cannot find deal with ptr that isnt arg");
}

Value *GradientUtils::lookupM(Value *val, IRBuilder<> &BuilderM,
                              const ValueToValueMapTy &incoming_available,
                              bool tryLegalRecomputeCheck) {
  assert(val->getName() != "<badref>");
  if (isa<Constant>(val)) {
    return val;
  }
  if (isa<BasicBlock>(val)) {
    return val;
  }
  if (isa<Function>(val)) {
    return val;
  }
  if (isa<UndefValue>(val)) {
    return val;
  }
  if (isa<Argument>(val)) {
    return val;
  }
  if (isa<MetadataAsValue>(val)) {
    return val;
  }
  if (isa<InlineAsm>(val)) {
    return val;
  }

  if (!isa<Instruction>(val)) {
    llvm::errs() << *val << "\n";
  }

  auto inst = cast<Instruction>(val);
  assert(inst->getName() != "<badref>");
  if (inversionAllocs && inst->getParent() == inversionAllocs) {
    return val;
  }

  if (isOriginalBlock(*BuilderM.GetInsertBlock())) {
    if (BuilderM.GetInsertBlock()->size() &&
        BuilderM.GetInsertPoint() != BuilderM.GetInsertBlock()->end()) {
      Instruction *use = &*BuilderM.GetInsertPoint();
      while (isa<PHINode>(use))
        use = use->getNextNode();
      if (DT.dominates(inst, use)) {
        return inst;
      } else {
        llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
        llvm::errs() << "didnt dominate inst: " << *inst
                     << "  point: " << *BuilderM.GetInsertPoint()
                     << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
      }
    } else {
      if (inst->getParent() == BuilderM.GetInsertBlock() ||
          DT.dominates(inst, BuilderM.GetInsertBlock())) {
        // allowed from block domination
        return inst;
      } else {
        llvm::errs() << *BuilderM.GetInsertBlock()->getParent() << "\n";
        llvm::errs() << "didnt dominate inst: " << *inst
                     << "\nbb: " << *BuilderM.GetInsertBlock() << "\n";
      }
    }
    // This is a reverse block
  } else if (BuilderM.GetInsertBlock() != inversionAllocs) {
    // Something in the entry (or anything that dominates all returns, doesn't
    // need caching)

    BasicBlock *forwardBlock =
        originalForReverseBlock(*BuilderM.GetInsertBlock());

    // Don't allow this if we're not definitely using the last iteration of this
    // value
    //   + either because the value isn't in a loop
    //   + or because the forward of the block usage location isn't in a loop
    //   (thus last iteration)
    //   + or because the loop nests share no ancestry

    bool loopLegal = true;
    for (Loop *idx = LI.getLoopFor(inst->getParent()); idx != nullptr;
         idx = idx->getParentLoop()) {
      for (Loop *fdx = LI.getLoopFor(forwardBlock); fdx != nullptr;
           fdx = fdx->getParentLoop()) {
        if (idx == fdx) {
          loopLegal = false;
          break;
        }
      }
    }

    if (loopLegal) {
      if (inst->getParent() == &newFunc->getEntryBlock()) {
        return inst;
      }
      // TODO upgrade this to be all returns that this could enter from
      bool legal = true;
      for (auto &BB : *oldFunc) {
        if (isa<ReturnInst>(BB.getTerminator())) {
          BasicBlock *returningBlock =
              cast<BasicBlock>(getNewFromOriginal(&BB));
          if (inst->getParent() == returningBlock)
            continue;
          if (!DT.dominates(inst, returningBlock)) {
            legal = false;
            break;
          }
        }
      }
      if (legal) {
        return inst;
      }
    }
  }

  Instruction *prelcssaInst = inst;

  assert(inst->getName() != "<badref>");
  val = fixLCSSA(inst, BuilderM.GetInsertBlock());
  inst = cast<Instruction>(val);

  assert(!this->isOriginalBlock(*BuilderM.GetInsertBlock()));

  auto idx = std::make_pair(val, BuilderM.GetInsertBlock());
  if (lookup_cache.find(idx) != lookup_cache.end()) {
    auto result = lookup_cache[idx];
    assert(result);
    assert(result->getType());
    return result;
  }

  ValueToValueMapTy available;
  for (auto pair : incoming_available) {
    available[pair.first] = pair.second;
  }

  LoopContext lc;
  bool inLoop = getContext(inst->getParent(), lc);

  if (inLoop) {
    bool first = true;
    for (LoopContext idx = lc;; getContext(idx.parent->getHeader(), idx)) {
      if (!isOriginalBlock(*BuilderM.GetInsertBlock())) {
        available[idx.var] = BuilderM.CreateLoad(idx.antivaralloc);
      } else {
        available[idx.var] = idx.var;
      }
      if (!first && idx.var == inst)
        return available[idx.var];
      if (first) {
        first = false;
      }
      if (idx.parent == nullptr)
        break;
    }
  }

  if (available.count(inst))
    return available[inst];

  // TODO consider call as part of
  bool lrc = false, src = false;
  if (tryLegalRecomputeCheck &&
      (lrc = legalRecompute(prelcssaInst, available))) {
    if ((src = shouldRecompute(prelcssaInst, available))) {
      auto op = unwrapM(prelcssaInst, BuilderM, available,
                        UnwrapMode::AttemptSingleUnwrap);
      if (op) {
        assert(op);
        assert(op->getType());
        if (auto load_op = dyn_cast<LoadInst>(prelcssaInst)) {
          if (auto new_op = dyn_cast<LoadInst>(op)) {
            MDNode *invgroup =
                load_op->getMetadata(LLVMContext::MD_invariant_group);
            if (invgroup == nullptr) {
              invgroup = MDNode::getDistinct(load_op->getContext(), {});
              load_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
            }
            new_op->setMetadata(LLVMContext::MD_invariant_group, invgroup);
          }
        }
        lookup_cache[idx] = op;
        return op;
      }
    } else {
      if (isa<LoadInst>(prelcssaInst)) {
      }
    }
  }

  // llvm::errs() << "forcing cache of " << *inst << "lrc: " << lrc << " src: "
  // << src << "\n";
  if (auto origInst = isOriginal(inst))
    if (auto li = dyn_cast<LoadInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 12
      auto liobj = getUnderlyingObject(li->getPointerOperand(), 100);
#else
      auto liobj = GetUnderlyingObject(
          li->getPointerOperand(), oldFunc->getParent()->getDataLayout(), 100);
#endif

      if (scopeMap.find(inst) == scopeMap.end()) {
        for (auto pair : scopeMap) {
          if (auto li2 = dyn_cast<LoadInst>(const_cast<Value *>(pair.first))) {

#if LLVM_VERSION_MAJOR >= 12
            auto li2obj = getUnderlyingObject(li2->getPointerOperand(), 100);
#else
            auto li2obj =
                GetUnderlyingObject(li2->getPointerOperand(),
                                    oldFunc->getParent()->getDataLayout(), 100);
#endif

            if (liobj == li2obj && DT.dominates(li2, li)) {
              auto orig2 = isOriginal(li2);
              if (!orig2)
                continue;

              bool failed = false;

              // llvm::errs() << "found potential candidate loads: oli:"
              //             << *origInst << " oli2: " << *orig2 << "\n";

              auto scev1 = SE.getSCEV(li->getPointerOperand());
              auto scev2 = SE.getSCEV(li2->getPointerOperand());
              // llvm::errs() << " scev1: " << *scev1 << " scev2: " << *scev2
              //             << "\n";

              allInstructionsBetween(
                  OrigLI, orig2, origInst, [&](Instruction *I) -> bool {
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(AA, /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      // llvm::errs() << "FAILED: " << *I << "\n";
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                continue;

              if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
                if (auto ar2 = dyn_cast<SCEVAddRecExpr>(scev2)) {
                  if (ar1->getStart() != SE.getCouldNotCompute() &&
                      ar1->getStart() == ar2->getStart() &&
                      ar1->getStepRecurrence(SE) != SE.getCouldNotCompute() &&
                      ar1->getStepRecurrence(SE) ==
                          ar2->getStepRecurrence(SE)) {

                    LoopContext l1;
                    getContext(ar1->getLoop()->getHeader(), l1);
                    LoopContext l2;
                    getContext(ar2->getLoop()->getHeader(), l2);

                    // TODO IF len(ar2) >= len(ar1) then we can replace li with
                    // li2
                    if (SE.getSCEV(l1.limit) != SE.getCouldNotCompute() &&
                        SE.getSCEV(l1.limit) == SE.getSCEV(l2.limit)) {
                      // llvm::errs()
                      //    << " step1: " << *ar1->getStepRecurrence(SE)
                      //    << " step2: " << *ar2->getStepRecurrence(SE) <<
                      //    "\n";

                      inst = li2;
                      idx = std::make_pair(val, BuilderM.GetInsertBlock());
                      break;
                    }
                  }
                }
              }
            }
          }
        }
      }

      auto loadSize = (li->getParent()
                           ->getParent()
                           ->getParent()
                           ->getDataLayout()
                           .getTypeAllocSizeInBits(li->getType()) +
                       7) /
                      8;

      // this is guarded because havent told cacheForReverse how to move
      if (mode == DerivativeMode::Both && false)
        if (!li->isVolatile()) {
          auto scev1 = SE.getSCEV(li->getPointerOperand());
          llvm::errs() << "scev1: " << *scev1 << "\n";
          // Store in memcpy opt
          if (auto ar1 = dyn_cast<SCEVAddRecExpr>(scev1)) {
            if (auto step =
                    dyn_cast<SCEVConstant>(ar1->getStepRecurrence(SE))) {
              if (step->getAPInt() != loadSize)
                goto noSpeedCache;

              LoopContext l1;
              getContext(ar1->getLoop()->getHeader(), l1);

              if (l1.dynamic)
                goto noSpeedCache;

              BasicBlock *ctx = l1.preheader;

              IRBuilder<> v(ctx->getTerminator());

              auto origPH = cast_or_null<BasicBlock>(isOriginal(ctx));
              assert(origPH);
              if (OrigPDT.dominates(origPH, origInst->getParent())) {
                goto noSpeedCache;
              }

              Value *lim = unwrapM(l1.limit, v,
                                   /*available*/ ValueToValueMapTy(),
                                   UnwrapMode::AttemptFullUnwrapWithLookup);
              if (!lim) {
                goto noSpeedCache;
              }
              lim = v.CreateAdd(lim, ConstantInt::get(lim->getType(), 1), "",
                                true, true);
              llvm::errs() << "l1: " << *li << "\n";
              llvm::errs() << "lim: " << *lim << "\n";

              Value *start = nullptr;

              std::vector<Instruction *> toErase;
              {
#if LLVM_VERSION_MAJOR >= 12
                SCEVExpander Exp(SE,
                                 ctx->getParent()->getParent()->getDataLayout(),
                                 "enzyme");
#else
                fake::SCEVExpander Exp(
                    SE, ctx->getParent()->getParent()->getDataLayout(),
                    "enzyme");
#endif
                Exp.setInsertPoint(l1.header->getTerminator());
                Value *start0 = Exp.expandCodeFor(
                    ar1->getStart(), li->getPointerOperand()->getType());
                start = unwrapM(start0, v,
                                /*available*/ ValueToValueMapTy(),
                                UnwrapMode::AttemptFullUnwrapWithLookup);
                llvm::errs() << *l1.header << "\n";
                std::set<Value *> todo = {start0};
                while (todo.size()) {
                  Value *now = *todo.begin();
                  todo.erase(now);
                  if (Instruction *inst = dyn_cast<Instruction>(now)) {
                    if (inst != start && inst->getNumUses() == 0 &&
                        Exp.isInsertedInstruction(inst)) {
                      for (auto &op : inst->operands()) {
                        todo.insert(op);
                      }
                      toErase.push_back(inst);
                    }
                  }
                }
              }
              for (auto a : toErase)
                erase(a);

              if (!start)
                goto noSpeedCache;

              llvm::errs() << " getStart: " << *ar1->getStart() << "\n";
              llvm::errs() << " starT: " << *start << "\n";

              bool failed = false;
              allInstructionsBetween(
                  OrigLI, &*v.GetInsertPoint(), origInst,
                  [&](Instruction *I) -> bool {
                    // llvm::errs() << "examining instruction: " << *I << "
                    // between: " << *li2 << " and " << *li << "\n";
                    if (I->mayWriteToMemory() &&
                        writesToMemoryReadBy(AA, /*maybeReader*/ origInst,
                                             /*maybeWriter*/ I)) {
                      failed = true;
                      llvm::errs() << "memcpy FAILED: " << *I << "\n";
                      return /*earlyBreak*/ true;
                    }
                    return /*earlyBreak*/ false;
                  });
              if (failed)
                goto noSpeedCache;

              bool isi1 = val->getType()->isIntegerTy() &&
                          cast<IntegerType>(li->getType())->getBitWidth() == 1;

              AllocaInst *cache = nullptr;

              LoopContext tmp;
              if (!getContext(ctx, tmp)) {
                ctx = (BasicBlock *)((size_t)ctx | 1);
              }

              if (auto found = findInMap(scopeMap, (Value *)inst)) {
                cache = found->first;
              } else {
                // if freeing reverseblocks must exist
                assert(reverseBlocks.size());
                cache = createCacheForScope(
                    ctx, li->getType(), li->getName(), /*shouldFree*/ true,
                    /*allocate*/ true, /*extraSize*/ lim);
                assert(cache);
                scopeMap.insert(
                    std::make_pair(inst, std::make_pair(cache, ctx)));

                v.setFastMathFlags(getFast());
                assert(isOriginalBlock(*v.GetInsertBlock()));
                Value *outer =
                    getCachePointer(/*inForwardPass*/ true, v, ctx, cache, isi1,
                                    /*storeinstorecache*/ true,
                                    /*extraSize*/ lim);

                auto dst_arg = v.CreateBitCast(
                    outer, Type::getInt8PtrTy(inst->getContext()));
                scopeInstructions[cache].push_back(cast<Instruction>(dst_arg));
                auto src_arg = v.CreateBitCast(
                    start, Type::getInt8PtrTy(inst->getContext()));
                auto len_arg = v.CreateMul(
                    ConstantInt::get(lim->getType(), step->getAPInt()), lim, "",
                    true, true);
                if (Instruction *I = dyn_cast<Instruction>(len_arg))
                  scopeInstructions[cache].push_back(I);
                auto volatile_arg = ConstantInt::getFalse(inst->getContext());

                Value *nargs[] = {dst_arg, src_arg, len_arg, volatile_arg};

                Type *tys[] = {dst_arg->getType(), src_arg->getType(),
                               len_arg->getType()};

                auto memcpyF = Intrinsic::getDeclaration(
                    newFunc->getParent(), Intrinsic::memcpy, tys);
                llvm::errs() << *memcpyF << "\n";
                auto mem = cast<CallInst>(v.CreateCall(memcpyF, nargs));
                // memset->addParamAttr(0, Attribute::getWithAlignment(Context,
                // inst->getAlignment()));
                mem->addParamAttr(0, Attribute::NonNull);
                mem->addParamAttr(1, Attribute::NonNull);

                auto bsize = newFunc->getParent()
                                 ->getDataLayout()
                                 .getTypeAllocSizeInBits(li->getType()) /
                             8;
                if ((bsize & (bsize - 1)) == 0) {
#if LLVM_VERSION_MAJOR >= 10
                  mem->addParamAttr(
                      0, Attribute::getWithAlignment(memcpyF->getContext(),
                                                     Align(bsize)));
#else
                  mem->addParamAttr(0, Attribute::getWithAlignment(
                                           memcpyF->getContext(), bsize));
#endif
                }

#if LLVM_VERSION_MAJOR >= 11
                mem->addParamAttr(
                    1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                   li->getAlign()));
#elif LLVM_VERSION_MAJOR >= 10
                if (li->getAlign())
                  mem->addParamAttr(
                      1, Attribute::getWithAlignment(
                             memcpyF->getContext(), li->getAlign().getValue()));
#else
                if (li->getAlignment())
                  mem->addParamAttr(
                      1, Attribute::getWithAlignment(memcpyF->getContext(),
                                                     li->getAlignment()));
#endif

                // TODO alignment

                scopeInstructions[cache].push_back(mem);
              }

              assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
              Value *result = lookupValueFromCache(
                  /*isForwardPass*/ false, BuilderM, ctx, cache, isi1,
                  /*extraSize*/ lim, available[l1.var]);
              lookup_cache[idx] = result;
              return result;
            }
          }
        }

    noSpeedCache:;
    }

  ensureLookupCached(inst);
  bool isi1 = inst->getType()->isIntegerTy() &&
              cast<IntegerType>(inst->getType())->getBitWidth() == 1;
  assert(!isOriginalBlock(*BuilderM.GetInsertBlock()));
  auto found = findInMap(scopeMap, (Value *)inst);
  Value *result = lookupValueFromCache(/*isForwardPass*/ false, BuilderM,
                                       found->second, found->first, isi1);
  assert(result->getType() == inst->getType());
  lookup_cache[idx] = result;
  assert(result);
  assert(result->getType());
  return result;
}

//! Given a map of edges we could have taken to desired target, compute a value
//! that determines which target should be branched to
//  This function attempts to determine an equivalent condition from earlier in
//  the code and use that if possible, falling back to creating a phi node of
//  which edge was taken if necessary This function can be used in two ways:
//   * If replacePHIs is null (usual case), this function does the branch
//   * If replacePHIs isn't null, do not perform the branch and instead replace
//   the PHI's with the derived condition as to whether we should branch to a
//   particular target
void GradientUtils::branchToCorrespondingTarget(
    BasicBlock *ctx, IRBuilder<> &BuilderM,
    const std::map<BasicBlock *,
                   std::vector<std::pair</*pred*/ BasicBlock *,
                                         /*successor*/ BasicBlock *>>>
        &targetToPreds,
    const std::map<BasicBlock *, PHINode *> *replacePHIs) {
  assert(targetToPreds.size() > 0);
  if (replacePHIs) {
    if (replacePHIs->size() == 0)
      return;

    for (auto x : *replacePHIs) {
      assert(targetToPreds.find(x.first) != targetToPreds.end());
    }
  }

  if (targetToPreds.size() == 1) {
    if (replacePHIs == nullptr) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateBr(targetToPreds.begin()->first);
    } else {
      for (auto pair : *replacePHIs) {
        pair.second->replaceAllUsesWith(
            ConstantInt::getTrue(pair.second->getContext()));
        pair.second->eraseFromParent();
      }
    }
    return;
  }

  // Map of function edges to list of targets this can branch to we have
  std::map<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
           std::set<BasicBlock *>>
      done;
  {
    std::deque<
        std::tuple<std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
                   BasicBlock *>>
        Q; // newblock, target

    for (auto pair : targetToPreds) {
      for (auto pred_edge : pair.second) {
        Q.push_back(std::make_pair(pred_edge, pair.first));
      }
    }

    for (std::tuple<
             std::pair</*pred*/ BasicBlock *, /*successor*/ BasicBlock *>,
             BasicBlock *>
             trace;
         Q.size() > 0;) {
      trace = Q.front();
      Q.pop_front();
      auto edge = std::get<0>(trace);
      auto block = edge.first;
      auto target = std::get<1>(trace);

      if (done[edge].count(target))
        continue;
      done[edge].insert(target);

      Loop *blockLoop = LI.getLoopFor(block);

      for (BasicBlock *Pred : predecessors(block)) {
        // Don't go up the backedge as we can use the last value if desired via
        // lcssa
        if (blockLoop && blockLoop->getHeader() == block &&
            blockLoop == LI.getLoopFor(Pred))
          continue;

        Q.push_back(
            std::tuple<std::pair<BasicBlock *, BasicBlock *>, BasicBlock *>(
                std::make_pair(Pred, block), target));
      }
    }
  }

  IntegerType *T = (targetToPreds.size() == 2)
                       ? Type::getInt1Ty(BuilderM.getContext())
                       : Type::getInt8Ty(BuilderM.getContext());

  Instruction *equivalentTerminator = nullptr;

  std::set<BasicBlock *> blocks;
  for (auto pair : done) {
    const auto &edge = pair.first;
    blocks.insert(edge.first);
  }

  if (targetToPreds.size() == 3) {
    for (auto block : blocks) {
      std::set<BasicBlock *> foundtargets;
      std::set<BasicBlock *> uniqueTargets;
      for (BasicBlock *succ : successors(block)) {
        auto edge = std::make_pair(block, succ);
        for (BasicBlock *target : done[edge]) {
          if (foundtargets.find(target) != foundtargets.end()) {
            goto rnextpair;
          }
          foundtargets.insert(target);
          if (done[edge].size() == 1)
            uniqueTargets.insert(target);
        }
      }
      if (foundtargets.size() != 3)
        goto rnextpair;
      if (uniqueTargets.size() != 1)
        goto rnextpair;

      {
        BasicBlock *subblock = nullptr;
        for (auto block2 : blocks) {
          std::set<BasicBlock *> seen2;
          for (BasicBlock *succ : successors(block2)) {
            auto edge = std::make_pair(block2, succ);
            if (done[edge].size() != 1) {
              // llvm::errs() << " -- failed from noonesize\n";
              goto nextblock;
            }
            for (BasicBlock *target : done[edge]) {
              if (seen2.find(target) != seen2.end()) {
                // llvm::errs() << " -- failed from not uniqueTargets\n";
                goto nextblock;
              }
              seen2.insert(target);
              if (foundtargets.find(target) == foundtargets.end()) {
                // llvm::errs() << " -- failed from not unknown target\n";
                goto nextblock;
              }
              if (uniqueTargets.find(target) != uniqueTargets.end()) {
                // llvm::errs() << " -- failed from not same target\n";
                goto nextblock;
              }
            }
          }
          if (seen2.size() != 2) {
            // llvm::errs() << " -- failed from not 2 seen\n";
            goto nextblock;
          }
          subblock = block2;
          break;
        nextblock:;
        }

        if (subblock == nullptr)
          goto rnextpair;

        {
          auto bi1 = cast<BranchInst>(block->getTerminator());

          auto cond1 = lookupM(bi1->getCondition(), BuilderM);
          auto bi2 = cast<BranchInst>(subblock->getTerminator());
          auto cond2 = lookupM(bi2->getCondition(), BuilderM);

          if (replacePHIs == nullptr) {
            BasicBlock *staging =
                BasicBlock::Create(oldFunc->getContext(), "staging", newFunc);
            auto stagingIfNeeded = [&](BasicBlock *B) {
              auto edge = std::make_pair(block, B);
              if (done[edge].size() == 1) {
                return *done[edge].begin();
              } else {
                return staging;
              }
            };
            BuilderM.CreateCondBr(cond1, stagingIfNeeded(bi1->getSuccessor(0)),
                                  stagingIfNeeded(bi1->getSuccessor(1)));
            BuilderM.SetInsertPoint(staging);
            BuilderM.CreateCondBr(
                cond2,
                *done[std::make_pair(subblock, bi2->getSuccessor(0))].begin(),
                *done[std::make_pair(subblock, bi2->getSuccessor(1))].begin());
          } else {
            Value *otherBranch = nullptr;
            for (unsigned i = 0; i < 2; ++i) {
              Value *val = cond1;
              if (i == 1)
                val = BuilderM.CreateNot(val, "anot1_");
              auto edge = std::make_pair(block, bi1->getSuccessor(i));
              if (done[edge].size() == 1) {
                auto found = replacePHIs->find(*done[edge].begin());
                if (found == replacePHIs->end())
                  continue;
                if (&*BuilderM.GetInsertPoint() == found->second) {
                  if (found->second->getNextNode())
                    BuilderM.SetInsertPoint(found->second->getNextNode());
                  else
                    BuilderM.SetInsertPoint(found->second->getParent());
                }
                found->second->replaceAllUsesWith(val);
                found->second->eraseFromParent();
              } else {
                otherBranch = val;
              }
            }

            for (unsigned i = 0; i < 2; ++i) {
              auto edge = std::make_pair(subblock, bi2->getSuccessor(i));
              auto found = replacePHIs->find(*done[edge].begin());
              if (found == replacePHIs->end())
                continue;

              Value *val = cond2;
              if (i == 1)
                val = BuilderM.CreateNot(val, "bnot1_");
              val = BuilderM.CreateAnd(val, otherBranch,
                                       "andVal" + std::to_string(i));
              if (&*BuilderM.GetInsertPoint() == found->second) {
                if (found->second->getNextNode())
                  BuilderM.SetInsertPoint(found->second->getNextNode());
                else
                  BuilderM.SetInsertPoint(found->second->getParent());
              }
              found->second->replaceAllUsesWith(val);
              found->second->eraseFromParent();
            }
          }

          return;
        }
      }
    rnextpair:;
    }
  }

  BasicBlock *forwardBlock = BuilderM.GetInsertBlock();

  if (!isOriginalBlock(*forwardBlock)) {
    forwardBlock = originalForReverseBlock(*forwardBlock);
  }

  for (auto block : blocks) {
    std::set<BasicBlock *> foundtargets;
    for (BasicBlock *succ : successors(block)) {
      auto edge = std::make_pair(block, succ);
      if (done[edge].size() != 1) {
        goto nextpair;
      }
      BasicBlock *target = *done[edge].begin();
      if (foundtargets.find(target) != foundtargets.end()) {
        goto nextpair;
      }
      foundtargets.insert(target);
    }
    if (foundtargets.size() != targetToPreds.size()) {
      goto nextpair;
    }

    if (forwardBlock == block || DT.dominates(block, forwardBlock)) {
      equivalentTerminator = block->getTerminator();
      goto fast;
    }

  nextpair:;
  }
  goto nofast;

fast:;
  assert(equivalentTerminator);

  if (auto branch = dyn_cast<BranchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();
    assert(branch->getCondition());

    assert(branch->getCondition()->getType() == T);

    if (replacePHIs == nullptr) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(
          lookupM(branch->getCondition(), BuilderM),
          *done[std::make_pair(block, branch->getSuccessor(0))].begin(),
          *done[std::make_pair(block, branch->getSuccessor(1))].begin());
    } else {
      for (auto pair : *replacePHIs) {
        Value *phi = lookupM(branch->getCondition(), BuilderM);
        Value *val = nullptr;
        if (pair.first ==
            *done[std::make_pair(block, branch->getSuccessor(0))].begin()) {
          val = phi;
        } else if (pair.first ==
                   *done[std::make_pair(block, branch->getSuccessor(1))]
                        .begin()) {
          val = BuilderM.CreateNot(phi);
        } else {
          llvm::errs() << *pair.first->getParent() << "\n";
          llvm::errs() << *pair.first << "\n";
          llvm::errs() << *branch << "\n";
          llvm_unreachable("unknown successor for replacephi");
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else if (auto si = dyn_cast<SwitchInst>(equivalentTerminator)) {
    BasicBlock *block = equivalentTerminator->getParent();

    IRBuilder<> pbuilder(equivalentTerminator);
    pbuilder.setFastMathFlags(getFast());

    if (replacePHIs == nullptr) {
      SwitchInst *swtch = BuilderM.CreateSwitch(
          lookupM(si->getCondition(), BuilderM),
          *done[std::make_pair(block, si->getDefaultDest())].begin());
      for (auto switchcase : si->cases()) {
        swtch->addCase(
            switchcase.getCaseValue(),
            *done[std::make_pair(block, switchcase.getCaseSuccessor())]
                 .begin());
      }
    } else {
      for (auto pair : *replacePHIs) {
        Value *cas = si->findCaseDest(pair.first);
        Value *val = nullptr;
        Value *phi = lookupM(si->getCondition(), BuilderM);
        if (cas) {
          val = BuilderM.CreateICmpEQ(cas, phi);
        } else {
          // default case
          val = ConstantInt::getFalse(pair.second->getContext());
          for (auto switchcase : si->cases()) {
            val = BuilderM.CreateOr(
                val, BuilderM.CreateICmpEQ(switchcase.getCaseValue(), phi));
          }
          val = BuilderM.CreateNot(val);
        }
        if (&*BuilderM.GetInsertPoint() == pair.second) {
          if (pair.second->getNextNode())
            BuilderM.SetInsertPoint(pair.second->getNextNode());
          else
            BuilderM.SetInsertPoint(pair.second->getParent());
        }
        pair.second->replaceAllUsesWith(val);
        pair.second->eraseFromParent();
      }
    }
  } else {
    llvm::errs() << "unknown equivalent terminator\n";
    llvm::errs() << *equivalentTerminator << "\n";
    llvm_unreachable("unknown equivalent terminator");
  }
  return;

nofast:;

  // if freeing reverseblocks must exist
  assert(reverseBlocks.size());
  AllocaInst *cache = createCacheForScope(ctx, T, "", /*shouldFree*/ true);
  std::vector<BasicBlock *> targets;
  {
    size_t idx = 0;
    std::map<BasicBlock * /*storingblock*/,
             std::map<ConstantInt * /*target*/,
                      std::vector<BasicBlock *> /*predecessors*/>>
        storing;
    for (const auto &pair : targetToPreds) {
      for (auto pred : pair.second) {
        storing[pred.first][ConstantInt::get(T, idx)].push_back(pred.second);
      }
      targets.push_back(pair.first);
      ++idx;
    }
    assert(targets.size() > 0);

    for (const auto &pair : storing) {
      IRBuilder<> pbuilder(pair.first);

      if (pair.first->getTerminator())
        pbuilder.SetInsertPoint(pair.first->getTerminator());

      pbuilder.setFastMathFlags(getFast());

      Value *tostore = ConstantInt::get(T, 0);

      if (pair.second.size() == 1) {
        tostore = pair.second.begin()->first;
      } else {
        assert(0 && "multi exit edges not supported");
        exit(1);
        // for(auto targpair : pair.second) {
        //     tostore = pbuilder.CreateOr(tostore, pred);
        //}
      }
      storeInstructionInCache(ctx, pbuilder, tostore, cache);
    }
  }

  bool isi1 = T->isIntegerTy() && cast<IntegerType>(T)->getBitWidth() == 1;
  Value *which = lookupValueFromCache(
      /*forwardPass*/ isOriginalBlock(*BuilderM.GetInsertBlock()), BuilderM,
      ctx, cache, isi1);
  assert(which);
  assert(which->getType() == T);

  if (replacePHIs == nullptr) {
    if (targetToPreds.size() == 2) {
      assert(BuilderM.GetInsertBlock()->size() == 0 ||
             !isa<BranchInst>(BuilderM.GetInsertBlock()->back()));
      BuilderM.CreateCondBr(which, /*true*/ targets[1], /*false*/ targets[0]);
    } else {
      assert(targets.size() > 0);
      auto swit =
          BuilderM.CreateSwitch(which, targets.back(), targets.size() - 1);
      for (unsigned i = 0; i < targets.size() - 1; ++i) {
        swit->addCase(ConstantInt::get(T, i), targets[i]);
      }
    }
  } else {
    for (unsigned i = 0; i < targets.size(); ++i) {
      auto found = replacePHIs->find(targets[i]);
      if (found == replacePHIs->end())
        continue;

      Value *val = nullptr;
      if (targets.size() == 2 && i == 0) {
        val = BuilderM.CreateNot(which);
      } else if (targets.size() == 2 && i == 1) {
        val = which;
      } else {
        val = BuilderM.CreateICmpEQ(ConstantInt::get(T, i), which);
      }
      if (&*BuilderM.GetInsertPoint() == found->second) {
        if (found->second->getNextNode())
          BuilderM.SetInsertPoint(found->second->getNextNode());
        else
          BuilderM.SetInsertPoint(found->second->getParent());
      }
      found->second->replaceAllUsesWith(val);
      found->second->eraseFromParent();
    }
  }
  return;
}
