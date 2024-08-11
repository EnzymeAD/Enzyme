//===- EnzymeLogic.cpp - Implementation of forward and reverse pass generation//
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
// This file defines two functions CreatePrimalAndGradient and
// CreateAugmentedPrimal. CreatePrimalAndGradient takes a function, known
// TypeResults of the calling context, known activity analysis of the
// arguments. It creates a corresponding gradient
// function, computing the primal as well if requested.
// CreateAugmentedPrimal takes similar arguments and creates an augmented
// primal pass.
//
//===----------------------------------------------------------------------===//
#include "EnzymeLogic.h"
#include "ActivityAnalysis.h"
#include "AdjointGenerator.h"
#include "EnzymeLogic.h"
#include "TypeAnalysis/TypeAnalysis.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/ErrorHandling.h"
#include <cmath>

#if LLVM_VERSION_MAJOR >= 16
#define private public
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#undef private
#else
#include "SCEV/ScalarEvolution.h"
#include "SCEV/ScalarEvolutionExpander.h"
#endif

#include "llvm/Analysis/DependenceAnalysis.h"
#include <deque>

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"

#include "llvm/Demangle/Demangle.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"

#include "llvm/Support/AMDGPUMetadata.h"
#include "llvm/Support/TimeProfiler.h"

#include "llvm/ADT/StringSet.h"

#include "DiffeGradientUtils.h"
#include "FunctionUtils.h"
#include "GradientUtils.h"
#include "InstructionBatcher.h"
#include "LibraryFuncs.h"
#include "TraceGenerator.h"
#include "Utils.h"

#if LLVM_VERSION_MAJOR >= 14
#define addAttribute addAttributeAtIndex
#define getAttribute getAttributeAtIndex
#define removeAttribute removeAttributeAtIndex
#endif

using namespace llvm;

extern "C" {
llvm::cl::opt<bool>
    EnzymePrint("enzyme-print", cl::init(false), cl::Hidden,
                cl::desc("Print before and after fns for autodiff"));

llvm::cl::opt<bool>
    EnzymePrintUnnecessary("enzyme-print-unnecessary", cl::init(false),
                           cl::Hidden,
                           cl::desc("Print unnecessary values in function"));

cl::opt<bool> looseTypeAnalysis("enzyme-loose-types", cl::init(false),
                                cl::Hidden,
                                cl::desc("Allow looser use of types"));

cl::opt<bool> nonmarkedglobals_inactiveloads(
    "enzyme_nonmarkedglobals_inactiveloads", cl::init(true), cl::Hidden,
    cl::desc("Consider loads of nonmarked globals to be inactive"));

cl::opt<bool> EnzymeJuliaAddrLoad(
    "enzyme-julia-addr-load", cl::init(false), cl::Hidden,
    cl::desc("Mark all loads resulting in an addr(13)* to be legal to redo"));

cl::opt<bool> EnzymeAssumeUnknownNoFree(
    "enzyme-assume-unknown-nofree", cl::init(false), cl::Hidden,
    cl::desc("Assume unknown instructions are nofree as needed"));

LLVMValueRef (*EnzymeFixupReturn)(LLVMBuilderRef, LLVMValueRef) = nullptr;
}

struct CacheAnalysis {

  const ValueMap<const CallInst *, SmallPtrSet<const CallInst *, 1>>
      &allocationsWithGuaranteedFree;
  const ValueMap<Value *, GradientUtils::Rematerializer>
      &rematerializableAllocations;
  TypeResults &TR;
  AAResults &AA;
  Function *oldFunc;
  ScalarEvolution &SE;
  LoopInfo &OrigLI;
  DominatorTree &OrigDT;
  TargetLibraryInfo &TLI;
  const SmallPtrSetImpl<BasicBlock *> &unnecessaryBlocks;
  const std::vector<bool> &overwritten_args;
  DerivativeMode mode;
  std::map<Value *, bool> seen;
  bool omp;
  CacheAnalysis(
      const ValueMap<const CallInst *, SmallPtrSet<const CallInst *, 1>>
          &allocationsWithGuaranteedFree,
      const ValueMap<Value *, GradientUtils::Rematerializer>
          &rematerializableAllocations,
      TypeResults &TR, AAResults &AA, Function *oldFunc, ScalarEvolution &SE,
      LoopInfo &OrigLI, DominatorTree &OrigDT, TargetLibraryInfo &TLI,
      const SmallPtrSetImpl<BasicBlock *> &unnecessaryBlocks,
      const std::vector<bool> &overwritten_args, DerivativeMode mode, bool omp)
      : allocationsWithGuaranteedFree(allocationsWithGuaranteedFree),
        rematerializableAllocations(rematerializableAllocations), TR(TR),
        AA(AA), oldFunc(oldFunc), SE(SE), OrigLI(OrigLI), OrigDT(OrigDT),
        TLI(TLI), unnecessaryBlocks(unnecessaryBlocks),
        overwritten_args(overwritten_args), mode(mode), omp(omp) {}

  bool is_value_mustcache_from_origin(Value *obj) {
    if (seen.find(obj) != seen.end())
      return seen[obj];

    bool mustcache = false;

    // If the pointer operand is from an argument to the function, we need to
    // check if the argument
    //   received from the caller is uncacheable.
    if (rematerializableAllocations.count(obj)) {
      return false;
    } else if (isa<UndefValue>(obj) || isa<ConstantPointerNull>(obj)) {
      return false;
    } else if (auto arg = dyn_cast<Argument>(obj)) {
      if (arg->getArgNo() >= overwritten_args.size()) {
        llvm::errs() << "overwritten_args:\n";
        for (auto pair : overwritten_args) {
          llvm::errs() << " + " << pair << "\n";
        }
        llvm::errs() << "could not find " << *arg << " of func "
                     << arg->getParent()->getName() << " in args_map\n";
        llvm_unreachable("could not find arg in args_map");
      }
      if (overwritten_args[arg->getArgNo()]) {
        mustcache = true;
        // EmitWarning("UncacheableOrigin", *arg,
        //            "origin arg may need caching ", *arg);
      }
    } else if (auto pn = dyn_cast<PHINode>(obj)) {
      seen[pn] = false;
      for (auto &val : pn->incoming_values()) {
        if (is_value_mustcache_from_origin(val)) {
          mustcache = true;
          EmitWarning("UncacheableOrigin", *pn, "origin pn may need caching ",
                      *pn);
          break;
        }
      }
    } else if (auto ci = dyn_cast<CastInst>(obj)) {
      mustcache = is_value_mustcache_from_origin(ci->getOperand(0));
      if (mustcache) {
        EmitWarning("UncacheableOrigin", *ci, "origin ci may need caching ",
                    *ci);
      }
    } else if (auto gep = dyn_cast<GetElementPtrInst>(obj)) {
      mustcache = is_value_mustcache_from_origin(gep->getPointerOperand());
      if (mustcache) {
        EmitWarning("UncacheableOrigin", *gep, "origin gep may need caching ",
                    *gep);
      }
    } else if (auto II = dyn_cast<IntrinsicInst>(obj);
               II && isIntelSubscriptIntrinsic(*II)) {
      mustcache = is_value_mustcache_from_origin(II->getOperand(3));
      if (mustcache) {
        EmitWarning("UncacheableOrigin", *II,
                    "origin llvm.intel.subscript may need caching ", *II);
      }
    } else {

      // Pointer operands originating from call instructions that are not
      // malloc/free are conservatively considered uncacheable.
      if (auto obj_op = dyn_cast<CallInst>(obj)) {
        auto n = getFuncNameFromCall(obj_op);
        // If this is a known allocation which is not captured or returned,
        // a caller function cannot overwrite this (since it cannot access).
        // Since we don't currently perform this check, we can instead check
        // if the pointer has a guaranteed free (which is a weaker form of
        // the required property).
        if (allocationsWithGuaranteedFree.find(obj_op) !=
            allocationsWithGuaranteedFree.end()) {

        } else if (n == "julia.get_pgcstack" || n == "julia.ptls_states" ||
                   n == "jl_get_ptls_states") {

        } else {
          // OP is a non malloc/free call so we need to cache
          mustcache = true;
          EmitWarning("UncacheableOrigin", *obj_op,
                      "origin call may need caching ", *obj_op);
        }
      } else if (isa<AllocaInst>(obj)) {
        // No change to modref if alloca since the memory only exists in
        // this function.
      } else if (auto GV = dyn_cast<GlobalVariable>(obj)) {
        // In the absense of more fine-grained global info, assume object is
        // written to in a subseqent call unless this is known to be constant
        if (!GV->isConstant()) {
          mustcache = true;
        }
      } else {
        // In absence of more information, assume that the underlying object for
        // pointer operand is uncacheable in caller.
        mustcache = true;
        if (auto I = dyn_cast<Instruction>(obj))
          EmitWarning("UncacheableOrigin", *I,
                      "unknown origin may need caching ", *obj);
      }
    }

    return seen[obj] = mustcache;
  }

  bool is_load_uncacheable(Instruction &li) {
    assert(li.getParent()->getParent() == oldFunc);

    auto Arch = llvm::Triple(oldFunc->getParent()->getTargetTriple()).getArch();
    if (Arch == Triple::amdgcn &&
        cast<PointerType>(li.getOperand(0)->getType())->getAddressSpace() ==
            4) {
      return false;
    }

    if (EnzymeJuliaAddrLoad)
      if (auto PT = dyn_cast<PointerType>(li.getType()))
        if (PT->getAddressSpace() == 13)
          return false;

    // Only use invariant load data if either, we are not using Julia
    // or we are in combined mode. The reason for this is that Julia
    // incorrectly has invariant load info for a function, which specifies
    // the load value won't change over the course of a function, but
    // may change from a caller.
    bool checkFunction = true;
    if (li.hasMetadata(LLVMContext::MD_invariant_load)) {
      if (!EnzymeJuliaAddrLoad || mode == DerivativeMode::ReverseModeCombined)
        return false;
      else
        checkFunction = false;
    }

    // Find the underlying object for the pointer operand of the load
    // instruction.
    auto obj = getBaseObject(li.getOperand(0));

    if (auto obj_op = dyn_cast<CallInst>(obj)) {
      auto n = getFuncNameFromCall(obj_op);
      if (n == "julia.get_pgcstack" || n == "julia.ptls_states" ||
          n == "jl_get_ptls_states")
        return false;
    }

    // Openmp bound and local thread id are unchanging
    // definitionally cacheable.
    if (omp)
      if (auto arg = dyn_cast<Argument>(obj)) {
        if (arg->getArgNo() < 2) {
          return false;
        }
      }

    // Any load from a rematerializable allocation is definitionally
    // reloadable. Notably we don't need to perform the allFollowers
    // of check as the loop scope caching should allow us to ignore
    // such stores.
    if (rematerializableAllocations.count(obj))
      return false;

    // If not running combined, check if pointer operand is overwritten
    // by a subsequent call (i.e. not this function).
    bool can_modref = false;
    if (mode != DerivativeMode::ReverseModeCombined)
      can_modref = is_value_mustcache_from_origin(obj);

    if (!can_modref && checkFunction) {
      allFollowersOf(&li, [&](Instruction *inst2) {
        if (!inst2->mayWriteToMemory())
          return false;

#if LLVM_VERSION_MAJOR >= 12
        if (isa<FenceInst>(inst2))
          return false;
#endif

        if (unnecessaryBlocks.count(inst2->getParent())) {
          return false;
        }
        if (auto CI = dyn_cast<CallInst>(inst2)) {
          if (auto F = CI->getCalledFunction()) {
            if (F->getName() == "__kmpc_for_static_fini") {
              return false;
            }
          }
        }

        if (!overwritesToMemoryReadBy(AA, TLI, SE, OrigLI, OrigDT, &li,
                                      inst2)) {
          return false;
        }

        if (auto II = dyn_cast<IntrinsicInst>(inst2)) {
          if (II->getIntrinsicID() == Intrinsic::nvvm_barrier0 ||
              II->getIntrinsicID() == Intrinsic::amdgcn_s_barrier) {
            allUnsyncdPredecessorsOf(
                II,
                [&](Instruction *mid) {
                  if (!mid->mayWriteToMemory())
                    return false;

#if LLVM_VERSION_MAJOR >= 12
                  if (isa<FenceInst>(mid))
                    return false;
#endif

                  if (unnecessaryBlocks.count(mid->getParent())) {
                    return false;
                  }

                  if (!writesToMemoryReadBy(AA, TLI, &li, mid)) {
                    return false;
                  }

                  can_modref = true;
                  EmitWarning("Uncacheable", li, "Load may need caching ", li,
                              " due to ", *mid, " via ", *II);
                  return true;
                },
                [&]() {
                  // if gone past entry
                  if (mode != DerivativeMode::ReverseModeCombined) {
                    EmitWarning("Uncacheable", li, "Load may need caching ", li,
                                " due to entry via ", *II);
                    can_modref = true;
                  }
                });
            if (can_modref)
              return true;
            else
              return false;
          }
        }
        can_modref = true;
        EmitWarning("Uncacheable", li, "Load may need caching ", li, " due to ",
                    *inst2);
        // Early exit
        return true;
      });
    } else {

      EmitWarning("Uncacheable", li, "Load may need caching ", li,
                  " due to origin ", *obj);
    }

    return can_modref;
  }

  // Computes a map of LoadInst -> boolean for a function indicating whether
  // that load is "uncacheable".
  //   A load is considered "uncacheable" if the data at the loaded memory
  //   location can be modified after the load instruction.
  std::map<Instruction *, bool> compute_uncacheable_load_map() {
    std::map<Instruction *, bool> can_modref_map;
    for (auto &B : *oldFunc) {
      if (unnecessaryBlocks.count(&B))
        continue;
      for (auto &inst : B) {
        // For each load instruction, determine if it is uncacheable.
        if (auto op = dyn_cast<LoadInst>(&inst)) {
          can_modref_map[op] = is_load_uncacheable(*op);
        }
        if (auto II = dyn_cast<IntrinsicInst>(&inst)) {
          switch (II->getIntrinsicID()) {
          case Intrinsic::nvvm_ldu_global_i:
          case Intrinsic::nvvm_ldu_global_p:
          case Intrinsic::nvvm_ldu_global_f:
          case Intrinsic::nvvm_ldg_global_i:
          case Intrinsic::nvvm_ldg_global_p:
          case Intrinsic::nvvm_ldg_global_f:
            can_modref_map[II] = false;
            break;
          case Intrinsic::masked_load:
            can_modref_map[II] = is_load_uncacheable(*II);
            break;
          default:
            break;
          }
        }
      }
    }
    return can_modref_map;
  }

  std::vector<bool>
  compute_overwritten_args_for_one_callsite(CallInst *callsite_op) {
    auto Fn = getFunctionFromCall(callsite_op);
    if (!Fn)
      return {};

    StringRef funcName = getFuncNameFromCall(callsite_op);

    if (funcName == "llvm.julia.gc_preserve_begin")
      return {};

    if (funcName == "llvm.julia.gc_preserve_end")
      return {};

    if (funcName == "julia.pointer_from_objref")
      return {};

    if (funcName == "julia.write_barrier")
      return {};

    if (funcName == "julia.write_barrier_binding")
      return {};

    if (funcName == "enzyme_zerotype")
      return {};

    if (isMemFreeLibMFunction(funcName)) {
      return {};
    }

    if (isDebugFunction(callsite_op->getCalledFunction()))
      return {};

    if (isCertainPrint(funcName) || isAllocationFunction(funcName, TLI) ||
        isDeallocationFunction(funcName, TLI)) {
      return {};
    }

    if (startsWith(funcName, "MPI_") ||
        startsWith(funcName, "enzyme_wrapmpi$$"))
      return {};

    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      return {};
    }

    SmallVector<Value *, 4> args;
    SmallVector<Value *, 4> objs;
    SmallVector<bool, 4> args_safe;

    // First, we need to propagate the uncacheable status from the parent
    // function to the callee.
    //   because memory location x modified after parent returns => x modified
    //   after callee returns.
#if LLVM_VERSION_MAJOR >= 14
    for (unsigned i = 0; i < callsite_op->arg_size(); ++i)
#else
    for (unsigned i = 0; i < callsite_op->getNumArgOperands(); ++i)
#endif
    {
      args.push_back(callsite_op->getArgOperand(i));

      // If the UnderlyingObject is from one of this function's arguments, then
      // we need to propagate the volatility.
      Value *obj = getBaseObject(callsite_op->getArgOperand(i));

      objs.push_back(obj);

      bool init_safe = !is_value_mustcache_from_origin(obj);
      if (!init_safe) {
        auto CD = TR.query(obj)[{-1}];
        if (CD == BaseType::Integer || CD.isFloat())
          init_safe = true;
      }
      if (!init_safe && !isa<UndefValue>(obj) && !isa<ConstantInt>(obj) &&
          !isa<Function>(obj)) {
        EmitWarning("UncacheableOrigin", *callsite_op, "Callsite ",
                    *callsite_op, " arg ", i, " ",
                    *callsite_op->getArgOperand(i), " uncacheable from origin ",
                    *obj);
      }
      args_safe.push_back(init_safe);
    }

    // Second, we check for memory modifications that can occur in the
    // continuation of the
    //   callee inside the parent function.
    allFollowersOf(callsite_op, [&](Instruction *inst2) {
      // Don't consider modref from malloc/free as a need to cache
      if (auto obj_op = dyn_cast<CallInst>(inst2)) {
        StringRef sfuncName = getFuncNameFromCall(obj_op);

        if (isMemFreeLibMFunction(sfuncName)) {
          return false;
        }

        if (isDebugFunction(obj_op->getCalledFunction()))
          return false;

        if (isCertainPrint(sfuncName) || isAllocationFunction(sfuncName, TLI) ||
            isDeallocationFunction(sfuncName, TLI)) {
          return false;
        }

        if (sfuncName == "__kmpc_for_static_fini") {
          return false;
        }

        if (auto iasm = dyn_cast<InlineAsm>(obj_op->getCalledOperand())) {
          if (StringRef(iasm->getAsmString()).contains("exit"))
            return false;
        }
      }

      if (unnecessaryBlocks.count(inst2->getParent()))
        return false;

      if (!inst2->mayWriteToMemory())
        return false;

      for (unsigned i = 0; i < args.size(); ++i) {
        if (!args_safe[i])
          continue;

        // Any use of an arg from a rematerializable allocation
        // is definitionally reloadable in sub.
        if (rematerializableAllocations.count(objs[i]))
          continue;

        auto CD = TR.query(args[i])[{-1}];
        if (CD == BaseType::Integer || CD.isFloat())
          continue;

        if (llvm::isModSet(AA.getModRefInfo(
                inst2, MemoryLocation::getForArgument(callsite_op, i, TLI)))) {
          if (!isa<ConstantInt>(callsite_op->getArgOperand(i)) &&
              !isa<UndefValue>(callsite_op->getArgOperand(i)))
            EmitWarning("UncacheableArg", *callsite_op, "Callsite ",
                        *callsite_op, " arg ", i, " ",
                        *callsite_op->getArgOperand(i), " uncacheable due to ",
                        *inst2);
          args_safe[i] = false;
        }
      }
      return false;
    });

    std::vector<bool> overwritten_args;

    if (funcName == "__kmpc_fork_call") {
      Value *op = callsite_op->getArgOperand(2);
      Function *task = nullptr;
      while (!(task = dyn_cast<Function>(op))) {
        if (auto castinst = dyn_cast<ConstantExpr>(op))
          if (castinst->isCast()) {
            op = castinst->getOperand(0);
            continue;
          }
        if (auto CI = dyn_cast<CastInst>(op)) {
          op = CI->getOperand(0);
          continue;
        }
        llvm::errs() << "op: " << *op << "\n";
        assert(0 && "unknown fork call arg");
      }

      // Global.tid is cacheable
      overwritten_args.push_back(false);

      // Bound.tid is cacheable
      overwritten_args.push_back(false);

      // Ignore first three arguments of fork call
      for (unsigned i = 3; i < args.size(); ++i) {
        overwritten_args.push_back(!args_safe[i]);
      }
    } else {
      for (unsigned i = 0; i < args.size(); ++i) {
        overwritten_args.push_back(!args_safe[i]);
      }
    }

    return overwritten_args;
  }

  // Given a function and the arguments passed to it by its caller that are
  // uncacheable (_overwritten_args) compute
  //   the set of uncacheable arguments for each callsite inside the function. A
  //   pointer argument is uncacheable at a callsite if the memory pointed to
  //   might be modified after that callsite.
  std::map<CallInst *, const std::vector<bool>>
  compute_overwritten_args_for_callsites() {
    std::map<CallInst *, const std::vector<bool>> overwritten_args_map;

    for (auto &B : *oldFunc) {
      if (unnecessaryBlocks.count(&B))
        continue;
      for (auto &inst : B) {
        if (auto op = dyn_cast<CallInst>(&inst)) {

          // We do not need uncacheable args for intrinsic functions. So skip
          // such callsites.
          if (auto II = dyn_cast<IntrinsicInst>(&inst)) {
            if (!startsWith(II->getCalledFunction()->getName(), "llvm.julia"))
              continue;
          }

          // For all other calls, we compute the uncacheable args for this
          // callsite.
          overwritten_args_map.insert(
              std::pair<CallInst *, const std::vector<bool>>(
                  op, compute_overwritten_args_for_one_callsite(op)));
        }
      }
    }
    return overwritten_args_map;
  }
};

void calculateUnusedValuesInFunction(
    Function &func, llvm::SmallPtrSetImpl<const Value *> &unnecessaryValues,
    llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    bool returnValue, DerivativeMode mode, GradientUtils *gutils,
    TargetLibraryInfo &TLI, ArrayRef<DIFFE_TYPE> constant_args,
    const llvm::SmallPtrSetImpl<BasicBlock *> &oldUnreachable) {
  std::map<UsageKey, bool> CacheResults;
  for (auto pair : gutils->knownRecomputeHeuristic) {
    if (!pair.second ||
        gutils->unnecessaryIntermediates.count(cast<Instruction>(pair.first))) {
      CacheResults[UsageKey(pair.first, QueryType::Primal)] = false;
    }
  }
  std::map<UsageKey, bool> PrimalSeen;
  if (mode == DerivativeMode::ReverseModeGradient) {
    PrimalSeen = CacheResults;
  }

  for (const auto &pair : gutils->allocationsWithGuaranteedFree) {
    if (pair.second.size() == 0)
      continue;

    bool primalNeededInReverse =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Primal>(
            gutils, pair.first, mode, CacheResults, oldUnreachable);

    // If rematerializing a split or loop-level allocation, the primal
    // allocation is not needed in the reverse.
    if (gutils->needsCacheWholeAllocation(pair.first)) {
      primalNeededInReverse = true;
    } else if (primalNeededInReverse) {
      auto found = gutils->rematerializableAllocations.find(
          const_cast<CallInst *>(pair.first));
      if (found != gutils->rematerializableAllocations.end()) {
        if (mode != DerivativeMode::ReverseModeCombined)
          primalNeededInReverse = false;
        else if (auto inst = dyn_cast<Instruction>(pair.first))
          if (found->second.LI &&
              found->second.LI->contains(inst->getParent())) {
            primalNeededInReverse = false;
          }
      }
    }

    for (auto freeCall : pair.second) {
      if (!primalNeededInReverse)
        gutils->forwardDeallocations.insert(freeCall);
      else
        gutils->postDominatingFrees.insert(freeCall);
    }
  }
  // Consider allocations which are being rematerialized, but do not
  // have a guaranteed free.
  for (const auto &rmat : gutils->rematerializableAllocations) {
    if (isa<CallInst>(rmat.first) &&
        gutils->allocationsWithGuaranteedFree.count(cast<CallInst>(rmat.first)))
      continue;
    if (rmat.second.frees.size() == 0)
      continue;

    bool primalNeededInReverse =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Primal>(
            gutils, rmat.first, mode, CacheResults, oldUnreachable);
    // If rematerializing a split or loop-level allocation, the primal
    // allocation is not needed in the reverse.
    if (gutils->needsCacheWholeAllocation(rmat.first)) {
      primalNeededInReverse = true;
    } else if (primalNeededInReverse) {
      if (mode != DerivativeMode::ReverseModeCombined)
        primalNeededInReverse = false;
      else if (auto inst = dyn_cast<Instruction>(rmat.first))
        if (rmat.second.LI && rmat.second.LI->contains(inst->getParent())) {
          primalNeededInReverse = false;
        }
    }

    for (auto freeCall : rmat.second.frees) {
      if (!primalNeededInReverse)
        gutils->forwardDeallocations.insert(cast<CallInst>(freeCall));
    }
  }

  std::function<bool(const llvm::Value *)> isNoNeed = [&](const llvm::Value
                                                              *v) {
    auto Obj = getBaseObject(v);
    if (Obj != v)
      return isNoNeed(Obj);
    if (auto C = dyn_cast<LoadInst>(v))
      return isNoNeed(C->getOperand(0));
    else if (auto arg = dyn_cast<Argument>(v)) {
      auto act = constant_args[arg->getArgNo()];
      if (act == DIFFE_TYPE::DUP_NONEED) {
        return true;
      }
    } else if (isa<AllocaInst>(v) || isAllocationCall(v, TLI)) {
      if (!gutils->isConstantValue(const_cast<Value *>(v))) {
        std::set<const Value *> done;
        std::deque<const Value *> todo = {v};
        bool legal = true;
        while (todo.size()) {
          const Value *cur = todo.back();
          todo.pop_back();
          if (done.count(cur))
            continue;
          done.insert(cur);

          if (unnecessaryValues.count(cur))
            continue;

          for (auto u : cur->users()) {
            if (auto SI = dyn_cast<StoreInst>(u)) {
              if (SI->getValueOperand() != cur) {
                continue;
              }
            }
            if (auto I = dyn_cast<Instruction>(u)) {
              if (unnecessaryInstructions.count(I)) {
                if (!DifferentialUseAnalysis::is_use_directly_needed_in_reverse(
                        gutils, cur, mode, I, oldUnreachable,
                        QueryType::Primal)) {
                  continue;
                }
              }
              if (isDeallocationCall(I, TLI)) {
                continue;
              }
            }
            if (auto II = dyn_cast<IntrinsicInst>(u);
                II && isIntelSubscriptIntrinsic(*II)) {
              todo.push_back(&*u);
              continue;
            } else if (auto CI = dyn_cast<CallInst>(u)) {
              if (getFuncNameFromCall(CI) == "julia.write_barrier") {
                continue;
              }
              if (getFuncNameFromCall(CI) == "julia.write_barrier_binding") {
                continue;
              }
              bool writeOnlyNoCapture = true;
              if (shouldDisableNoWrite(CI)) {
                writeOnlyNoCapture = false;
              }
#if LLVM_VERSION_MAJOR >= 14
              for (size_t i = 0; i < CI->arg_size(); i++)
#else
              for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
              {
                if (cur == CI->getArgOperand(i)) {
                  if (!isNoCapture(CI, i)) {
                    writeOnlyNoCapture = false;
                    break;
                  }
                  if (!isWriteOnly(CI, i)) {
                    writeOnlyNoCapture = false;
                    break;
                  }
                }
              }
              // Don't need the primal argument if it is write only and
              // not captured
              if (writeOnlyNoCapture) {
                continue;
              }
            }
            if (isa<CastInst>(u) || isa<GetElementPtrInst>(u) ||
                isa<PHINode>(u)) {
              todo.push_back(&*u);
              continue;
            } else {
              legal = false;
              break;
            }
          }
        }
        if (legal) {
          return true;
        }
      }
    } else if (auto II = dyn_cast<IntrinsicInst>(v);
               II && isIntelSubscriptIntrinsic(*II)) {
      unsigned int ptrArgIdx = 3;
      return isNoNeed(II->getOperand(ptrArgIdx));
    }
    return false;
  };

  calculateUnusedValues(
      func, unnecessaryValues, unnecessaryInstructions, returnValue,
      [&](const Value *val) {
        bool ivn = DifferentialUseAnalysis::is_value_needed_in_reverse<
            QueryType::Primal>(gutils, val, mode, PrimalSeen, oldUnreachable);
        return ivn;
      },
      [&](const Instruction *inst) {
        if (auto II = dyn_cast<IntrinsicInst>(inst)) {
          if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
              II->getIntrinsicID() == Intrinsic::lifetime_end ||
              II->getIntrinsicID() == Intrinsic::stacksave ||
              II->getIntrinsicID() == Intrinsic::stackrestore) {
            return UseReq::Cached;
          }
        }

        if (mode == DerivativeMode::ReverseModeGradient &&
            gutils->knownRecomputeHeuristic.find(inst) !=
                gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[inst]) {
            return UseReq::Cached;
          }
        }

        if (llvm::isa<llvm::ReturnInst>(inst) && returnValue) {
          return UseReq::Need;
        }
        if (llvm::isa<llvm::BranchInst>(inst) ||
            llvm::isa<llvm::SwitchInst>(inst)) {
          size_t num = 0;
          for (auto suc : successors(inst->getParent())) {
            if (!oldUnreachable.count(suc)) {
              num++;
            }
          }
          if (num > 1 || mode != DerivativeMode::ReverseModeGradient) {
            return UseReq::Need;
          }
        }

        // We still need this value if used as increment/induction variable for
        // a loop
        // TODO this really should be more simply replaced by doing the loop
        // normalization on the original code as preprocessing

        // Below we specifically check if the instructions or any of its
        // newly-generated (e.g. not in original function) uses are used in the
        // loop calculation
        auto NewI = gutils->getNewFromOriginal(inst);
        std::set<Instruction *> todo = {NewI};
        {
          std::deque<Instruction *> toAnalyze;
          // Here we get the uses of inst from the original function
          std::set<Instruction *> UsesFromOrig;
          for (auto u : inst->users()) {
            if (auto I = dyn_cast<Instruction>(u)) {
              UsesFromOrig.insert(gutils->getNewFromOriginal(I));
            }
          }
          // We only analyze uses that were not available in the original
          // function
          for (auto u : NewI->users()) {
            if (auto I = dyn_cast<Instruction>(u)) {
              if (UsesFromOrig.count(I) == 0)
                toAnalyze.push_back(I);
            }
          }

          while (toAnalyze.size()) {
            auto Next = toAnalyze.front();
            toAnalyze.pop_front();
            if (todo.count(Next))
              continue;
            todo.insert(Next);
            for (auto u : Next->users()) {
              if (auto I = dyn_cast<Instruction>(u)) {
                toAnalyze.push_back(I);
              }
            }
          }
        }

        for (auto I : todo) {
          if (gutils->isInstructionUsedInLoopInduction(*I)) {
            return UseReq::Need;
          }
        }

        bool mayWriteToMemory = inst->mayWriteToMemory();
        if (unnecessaryValues.count(inst) && isAllocationCall(inst, TLI))
          return UseReq::Recur;

        if (auto obj_op = dyn_cast<CallInst>(inst)) {
          StringRef funcName = getFuncNameFromCall(obj_op);
          if (isDeallocationFunction(funcName, TLI)) {
            if (unnecessaryValues.count(obj_op->getArgOperand(0))) {
              return UseReq::Recur;
            }

            if (mode == DerivativeMode::ForwardMode ||
                mode == DerivativeMode::ForwardModeError ||
                mode == DerivativeMode::ForwardModeSplit ||
                ((mode == DerivativeMode::ReverseModePrimal ||
                  mode == DerivativeMode::ReverseModeCombined) &&
                 gutils->forwardDeallocations.count(obj_op)))
              return UseReq::Need;
            return UseReq::Recur;
          }
          if (hasMetadata(obj_op, "enzyme_zerostack")) {
            if (unnecessaryValues.count(
                    getBaseObject(obj_op->getArgOperand(0)))) {
              return UseReq::Recur;
            }
          }
          Intrinsic::ID ID = Intrinsic::not_intrinsic;
          if (isMemFreeLibMFunction(funcName, &ID) || isReadOnly(obj_op)) {
            mayWriteToMemory = false;
          }
          if (funcName == "memset" || funcName == "memset_pattern16" ||
              funcName == "memcpy" || funcName == "memmove") {
            if (isNoNeed(obj_op->getArgOperand(0)))
              return UseReq::Recur;
          }
        }

        if (auto si = dyn_cast<StoreInst>(inst)) {
          bool nnop = isNoNeed(si->getPointerOperand());
          if (isa<UndefValue>(si->getValueOperand()))
            return UseReq::Recur;
          if (nnop)
            return UseReq::Recur;
        }

        if (auto msi = dyn_cast<MemSetInst>(inst)) {
          if (isNoNeed(msi->getArgOperand(0)))
            return UseReq::Recur;
        }

        if (auto mti = dyn_cast<MemTransferInst>(inst)) {
          if (isNoNeed(mti->getArgOperand(0)))
            return UseReq::Recur;

          auto at = getBaseObject(mti->getArgOperand(1));

          bool newMemory = false;
          if (isa<AllocaInst>(at))
            newMemory = true;
          else if (isAllocationCall(at, TLI))
            newMemory = true;
          if (newMemory) {
            bool foundStore = false;
            allInstructionsBetween(
                *gutils->OrigLI, cast<Instruction>(at),
                const_cast<MemTransferInst *>(mti),
                [&](Instruction *I) -> bool {
                  if (!I->mayWriteToMemory())
                    return /*earlyBreak*/ false;
                  if (unnecessaryInstructions.count(I))
                    return /*earlyBreak*/ false;
                  if (auto CI = dyn_cast<CallInst>(I)) {
                    if (isReadOnly(CI))
                      return /*earlyBreak*/ false;
                  }

                  if (writesToMemoryReadBy(
                          *gutils->OrigAA, TLI,
                          /*maybeReader*/ const_cast<MemTransferInst *>(mti),
                          /*maybeWriter*/ I)) {
                    foundStore = true;
                    return /*earlyBreak*/ true;
                  }
                  return /*earlyBreak*/ false;
                });
            if (!foundStore) {
              return UseReq::Recur;
            }
          }
        }
        if ((mode == DerivativeMode::ReverseModePrimal ||
             mode == DerivativeMode::ReverseModeCombined ||
             mode == DerivativeMode::ForwardMode ||
             mode == DerivativeMode::ForwardModeError) &&
            mayWriteToMemory) {
          return UseReq::Need;
        }
        // Don't erase any store that needs to be preserved for a
        // rematerialization. However, if not used in a rematerialization, the
        // store should be eliminated in the reverse pass
        if (mode == DerivativeMode::ReverseModeGradient ||
            mode == DerivativeMode::ForwardModeSplit) {
          auto CI = dyn_cast<CallInst>(const_cast<Instruction *>(inst));
          const Function *CF = CI ? getFunctionFromCall(CI) : nullptr;
          StringRef funcName = CF ? CF->getName() : "";
          if (isa<MemTransferInst>(inst) || isa<StoreInst>(inst) ||
              isa<MemSetInst>(inst) || funcName == "julia.write_barrier" ||
              funcName == "julia.write_barrier_binding") {
            for (auto pair : gutils->rematerializableAllocations) {
              if (pair.second.stores.count(inst)) {
                if (DifferentialUseAnalysis::is_value_needed_in_reverse<
                        QueryType::Primal>(gutils, pair.first, mode, PrimalSeen,
                                           oldUnreachable)) {
                  return UseReq::Need;
                }
              }
            }
            return UseReq::Recur;
          }
        }

        bool ivn = DifferentialUseAnalysis::is_value_needed_in_reverse<
            QueryType::Primal>(gutils, inst, mode, PrimalSeen, oldUnreachable);
        if (ivn) {
          return UseReq::Need;
        }
        return UseReq::Recur;
      },
      [&](const Instruction *inst, const Value *val) {
        if (isNoNeed(val)) {
          if (auto SI = dyn_cast<StoreInst>(inst))
            if (SI->getPointerOperand() == val)
              return false;

          if (auto CI = dyn_cast<CallInst>(inst)) {
            if (isDeallocationCall(CI, TLI)) {
              if (CI->getArgOperand(0) == val)
                return false;
            }

            bool writeOnlyNoCapture = true;
            if (shouldDisableNoWrite(CI)) {
              writeOnlyNoCapture = false;
            }
#if LLVM_VERSION_MAJOR >= 14
            for (size_t i = 0; i < CI->arg_size(); i++)
#else
            for (size_t i = 0; i < CI->getNumArgOperands(); i++)
#endif
            {
              if (val == CI->getArgOperand(i)) {
                if (!isNoCapture(CI, i)) {
                  writeOnlyNoCapture = false;
                  break;
                }
                if (!isWriteOnly(CI, i)) {
                  writeOnlyNoCapture = false;
                  break;
                }
              }
            }
            // Don't need the primal argument if it is write only and not
            // captured
            if (writeOnlyNoCapture) {
              return false;
            }
          }
        }
        return true;
      });
  if (EnzymePrintUnnecessary) {
    llvm::errs() << " val use analysis of " << func.getName()
                 << ": mode=" << to_string(mode) << "\n";
    for (auto &BB : func)
      for (auto &I : BB) {
        bool ivn = DifferentialUseAnalysis::is_value_needed_in_reverse<
            QueryType::Primal>(gutils, &I, mode, PrimalSeen, oldUnreachable);
        bool isn = DifferentialUseAnalysis::is_value_needed_in_reverse<
            QueryType::Shadow>(gutils, &I, mode, PrimalSeen, oldUnreachable);
        llvm::errs() << I << " ivn=" << (int)ivn << " isn: " << (int)isn;
        auto found = gutils->knownRecomputeHeuristic.find(&I);
        if (found != gutils->knownRecomputeHeuristic.end()) {
          llvm::errs() << " krc=" << (int)found->second;
        }
        llvm::errs() << "\n";
      }
    llvm::errs() << "unnecessaryValues of " << func.getName()
                 << ": mode=" << to_string(mode) << "\n";
    for (auto a : unnecessaryValues) {
      bool ivn = DifferentialUseAnalysis::is_value_needed_in_reverse<
          QueryType::Primal>(gutils, a, mode, PrimalSeen, oldUnreachable);
      bool isn = DifferentialUseAnalysis::is_value_needed_in_reverse<
          QueryType::Shadow>(gutils, a, mode, PrimalSeen, oldUnreachable);
      llvm::errs() << *a << " ivn=" << (int)ivn << " isn: " << (int)isn;
      auto found = gutils->knownRecomputeHeuristic.find(a);
      if (found != gutils->knownRecomputeHeuristic.end()) {
        llvm::errs() << " krc=" << (int)found->second;
      }
      llvm::errs() << "\n";
    }
    llvm::errs() << "unnecessaryInstructions " << func.getName() << ":\n";
    for (auto a : unnecessaryInstructions) {
      llvm::errs() << *a << "\n";
    }
  }
}

void calculateUnusedStoresInFunction(
    Function &func,
    llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryStores,
    const llvm::SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    GradientUtils *gutils, TargetLibraryInfo &TLI) {
  calculateUnusedStores(func, unnecessaryStores, [&](const Instruction *inst) {
    if (auto si = dyn_cast<StoreInst>(inst)) {
      if (isa<UndefValue>(si->getValueOperand()))
        return false;
    }

    if (auto mti = dyn_cast<MemTransferInst>(inst)) {
      auto at = getBaseObject(mti->getArgOperand(1));
      bool newMemory = false;
      if (isa<AllocaInst>(at))
        newMemory = true;
      else if (isAllocationCall(at, TLI))
        newMemory = true;
      if (newMemory) {
        bool foundStore = false;
        allInstructionsBetween(
            *gutils->OrigLI, cast<Instruction>(at),
            const_cast<MemTransferInst *>(mti), [&](Instruction *I) -> bool {
              if (!I->mayWriteToMemory())
                return /*earlyBreak*/ false;
              if (unnecessaryStores.count(I))
                return /*earlyBreak*/ false;

              // if (I == &MTI) return;
              if (writesToMemoryReadBy(
                      *gutils->OrigAA, TLI,
                      /*maybeReader*/ const_cast<MemTransferInst *>(mti),
                      /*maybeWriter*/ I)) {
                foundStore = true;
                return /*earlyBreak*/ true;
              }
              return /*earlyBreak*/ false;
            });
        if (!foundStore) {
          // performing a memcpy out of unitialized memory
          return false;
        }
      }
    }

    return true;
  });
}

std::string to_string(Function &F, const std::vector<bool> &us) {
  std::string s = "{";
  auto arg = F.arg_begin();
  for (auto y : us) {
    s += arg->getName().str() + "@" + F.getName().str() + ":" +
         std::to_string(y) + ",";
    arg++;
  }
  return s + "}";
}

//! assuming not top level
std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>
getDefaultFunctionTypeForAugmentation(FunctionType *called, bool returnUsed,
                                      DIFFE_TYPE retType) {
  SmallVector<Type *, 4> args;
  SmallVector<Type *, 4> outs;
  for (auto &argType : called->params()) {
    args.push_back(argType);

    if (!argType->isFPOrFPVectorTy()) {
      args.push_back(argType);
    }
  }

  auto ret = called->getReturnType();
  // TODO CONSIDER a.getType()->isIntegerTy() &&
  // cast<IntegerType>(a.getType())->getBitWidth() < 16
  outs.push_back(getInt8PtrTy(called->getContext()));
  if (!ret->isVoidTy() && !ret->isEmptyTy()) {
    if (returnUsed) {
      outs.push_back(ret);
    }
    if (retType == DIFFE_TYPE::DUP_ARG || retType == DIFFE_TYPE::DUP_NONEED) {
      outs.push_back(ret);
    }
  }

  return std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>(args, outs);
}

//! assuming not top level
std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>
getDefaultFunctionTypeForGradient(FunctionType *called, DIFFE_TYPE retType,
                                  ArrayRef<DIFFE_TYPE> tys) {
  SmallVector<Type *, 4> args;
  SmallVector<Type *, 4> outs;

  size_t i = 0;
  for (auto &argType : called->params()) {
    args.push_back(argType);

    switch (tys[i]) {
    case DIFFE_TYPE::CONSTANT:
      break;
    case DIFFE_TYPE::OUT_DIFF:
      outs.push_back(argType);
      break;
    case DIFFE_TYPE::DUP_ARG:
    case DIFFE_TYPE::DUP_NONEED:
      args.push_back(argType);
      break;
    }
    i++;
  }

  auto ret = called->getReturnType();

  if (retType == DIFFE_TYPE::OUT_DIFF) {
    args.push_back(ret);
  }

  return std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>(args, outs);
}

//! assuming not top level
std::pair<SmallVector<Type *, 4>, SmallVector<Type *, 4>>
getDefaultFunctionTypeForGradient(FunctionType *called, DIFFE_TYPE retType) {
  SmallVector<DIFFE_TYPE, 4> act;
  for (auto &argType : called->params()) {

    if (argType->isFPOrFPVectorTy()) {
      act.push_back(DIFFE_TYPE::OUT_DIFF);
    } else {
      act.push_back(DIFFE_TYPE::DUP_ARG);
    }
  }
  return getDefaultFunctionTypeForGradient(called, retType, act);
}

bool shouldAugmentCall(CallInst *op, const GradientUtils *gutils) {
  assert(op->getParent()->getParent() == gutils->oldFunc);

  Function *called = op->getCalledFunction();

  bool modifyPrimal = !called || !isReadNone(op);

  if (modifyPrimal) {
#ifdef PRINT_AUGCALL
    if (called)
      llvm::errs() << "primal modified " << called->getName()
                   << " modified via reading from memory"
                   << "\n";
    else
      llvm::errs() << "primal modified " << *op->getCalledValue()
                   << " modified via reading from memory"
                   << "\n";
#endif
  }

  if (!op->getType()->isFPOrFPVectorTy() && !gutils->isConstantValue(op) &&
      gutils->TR.anyPointer(op)) {
    modifyPrimal = true;

#ifdef PRINT_AUGCALL
    if (called)
      llvm::errs() << "primal modified " << called->getName()
                   << " modified via return"
                   << "\n";
    else
      llvm::errs() << "primal modified " << *op->getCalledValue()
                   << " modified via return"
                   << "\n";
#endif
  }

  if (!called || called->empty())
    modifyPrimal = true;

#if LLVM_VERSION_MAJOR >= 14
  for (unsigned i = 0; i < op->arg_size(); ++i)
#else
  for (unsigned i = 0; i < op->getNumArgOperands(); ++i)
#endif
  {
    if (gutils->isConstantValue(op->getArgOperand(i)) && called &&
        !called->empty()) {
      continue;
    }

    auto argType = op->getArgOperand(i)->getType();

    if (!argType->isFPOrFPVectorTy() &&
        !gutils->isConstantValue(op->getArgOperand(i)) &&
        gutils->TR.anyPointer(op->getArgOperand(i))) {
      if (!isReadOnly(op, i)) {
        modifyPrimal = true;
#ifdef PRINT_AUGCALL
        if (called)
          llvm::errs() << "primal modified " << called->getName()
                       << " modified via arg " << i << "\n";
        else
          llvm::errs() << "primal modified " << *op->getCalledValue()
                       << " modified via arg " << i << "\n";
#endif
      }
    }
  }

  // Don't need to augment calls that are certain to not hit return
  if (isa<UnreachableInst>(op->getParent()->getTerminator())) {
    modifyPrimal = false;
  }

#ifdef PRINT_AUGCALL
  llvm::errs() << "PM: " << *op << " modifyPrimal: " << modifyPrimal
               << " cv: " << gutils->isConstantValue(op) << "\n";
#endif
  return modifyPrimal;
}

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            ModRefInfo mri) {
  if (mri == ModRefInfo::NoModRef)
    return os << "nomodref";
  else if (mri == ModRefInfo::ModRef)
    return os << "modref";
  else if (mri == ModRefInfo::Mod)
    return os << "mod";
  else if (mri == ModRefInfo::Ref)
    return os << "ref";
#if LLVM_VERSION_MAJOR <= 14
  else if (mri == ModRefInfo::MustModRef)
    return os << "mustmodref";
  else if (mri == ModRefInfo::MustMod)
    return os << "mustmod";
  else if (mri == ModRefInfo::MustRef)
    return os << "mustref";
#endif
  else
    llvm_unreachable("unknown modref");
  return os;
}

bool legalCombinedForwardReverse(
    CallInst *origop,
    const std::map<ReturnInst *, StoreInst *> &replacedReturns,
    SmallVectorImpl<Instruction *> &postCreate,
    SmallVectorImpl<Instruction *> &userReplace, const GradientUtils *gutils,
    const SmallPtrSetImpl<const Instruction *> &unnecessaryInstructions,
    const SmallPtrSetImpl<BasicBlock *> &oldUnreachable,
    const bool subretused) {
  Function *called = origop->getCalledFunction();
  Value *calledValue = origop->getCalledOperand();

  if (isa<PointerType>(origop->getType())) {
    bool sret = subretused;
    if (!sret && !gutils->isConstantValue(origop)) {
      sret = DifferentialUseAnalysis::is_value_needed_in_reverse<
          QueryType::Shadow>(gutils, origop, gutils->mode, oldUnreachable);
    }

    if (sret) {
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [not implemented] pointer return for combined "
                          "forward/reverse "
                       << called->getName() << "\n";
        else
          llvm::errs() << " [not implemented] pointer return for combined "
                          "forward/reverse "
                       << *calledValue << "\n";
      }
      return false;
    }
  }

  // Check any users of the returned value and determine all values that would
  // be needed to be moved to reverse pass
  //  to ensure the forward pass would remain correct and everything computable
  SmallPtrSet<Instruction *, 4> usetree;
  std::deque<Instruction *> todo{origop};

  bool legal = true;

  // Given a function I we know must be moved to the reverse for legality
  // reasons
  auto propagate = [&](Instruction *I) {
    // if only used in unneeded return, don't need to move this to reverse
    // (unless this is the original function)
    if (usetree.count(I))
      return;
    if (gutils->notForAnalysis.count(I->getParent()))
      return;
    if (auto ri = dyn_cast<ReturnInst>(I)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        usetree.insert(ri);
      }
      return;
    }

    if (isa<BranchInst>(I) || isa<SwitchInst>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [bi] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [bi] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }

    // Even though the value `I` depends on (perhaps indirectly) the call being
    // checked for, if neither `I` nor its pointer-valued shadow are used in the
    // reverse pass, we can ignore the dependency as long as `I` is not going to
    // have a combined forward and reverse pass.
    if (I != origop && unnecessaryInstructions.count(I)) {
      bool needShadow = false;
      if (!gutils->isConstantValue(I)) {
        needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
            QueryType::Shadow>(gutils, I, DerivativeMode::ReverseModeCombined,
                               oldUnreachable);
      }
      if (!needShadow) {
        if (gutils->isConstantInstruction(I) || !isa<CallInst>(I)) {
          userReplace.push_back(I);
          return;
        }
      }
    }

    if (isAllocationCall(I, gutils->TLI) ||
        isDeallocationCall(I, gutils->TLI)) {
      return;
    }

    if (isa<BranchInst>(I)) {
      legal = false;

      return;
    }
    if (isa<PHINode>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [phi] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [phi] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    if (!I->getType()->isVoidTy() &&
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Primal>(
            gutils, I, DerivativeMode::ReverseModeCombined, oldUnreachable)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [nv] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [nv] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    if (!I->getType()->isVoidTy() &&
        gutils->TR.query(I)[{-1}].isPossiblePointer() &&
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, I, DerivativeMode::ReverseModeCombined, oldUnreachable)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [ns] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [ns] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    if (I != origop && !isa<IntrinsicInst>(I) && isa<CallInst>(I)) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [ci] failed to replace function "
                       << (called->getName()) << " due to " << *I << "\n";
        else
          llvm::errs() << " [ci] failed to replace function " << (*calledValue)
                       << " due to " << *I << "\n";
      }
      return;
    }
    // Do not try moving an instruction that modifies memory, if we already
    // moved it. We need the originalToNew check because we may have deleted
    // the instruction, which wont require the failed to move.
    if (!isa<StoreInst>(I) || unnecessaryInstructions.count(I) == 0)
      if (I->mayReadOrWriteMemory() &&
          gutils->originalToNewFn.find(I) != gutils->originalToNewFn.end() &&
          gutils->getNewFromOriginal(I)->getParent() !=
              gutils->getNewFromOriginal(I->getParent())) {
        legal = false;
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [am] failed to replace function "
                         << (called->getName()) << " due to " << *I << "\n";
          else
            llvm::errs() << " [am] failed to replace function "
                         << (*calledValue) << " due to " << *I << "\n";
        }
        return;
      }

    usetree.insert(I);
    for (auto use : I->users()) {
      todo.push_back(cast<Instruction>(use));
    }
  };

  while (!todo.empty()) {
    auto inst = todo.front();
    todo.pop_front();

    if (inst->mayWriteToMemory()) {
      auto consider = [&](Instruction *user) {
        if (!user->mayReadFromMemory())
          return false;
        if (writesToMemoryReadBy(*gutils->OrigAA, gutils->TLI,
                                 /*maybeReader*/ user,
                                 /*maybeWriter*/ inst)) {

          propagate(user);
          // Fast return if not legal
          if (!legal)
            return true;
        }
        return false;
      };
      allFollowersOf(inst, consider);
      if (!legal)
        return false;
    }

    propagate(inst);
    if (!legal)
      return false;
  }

  // Check if any of the unmoved operations will make it illegal to move the
  // instruction

  for (auto inst : usetree) {
    if (!inst->mayReadFromMemory())
      continue;
    allFollowersOf(inst, [&](Instruction *post) {
      if (unnecessaryInstructions.count(post))
        return false;
      if (!post->mayWriteToMemory())
        return false;

      if (writesToMemoryReadBy(*gutils->OrigAA, gutils->TLI,
                               /*maybeReader*/ inst,
                               /*maybeWriter*/ post)) {
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [mem] failed to replace function "
                         << (called->getName()) << " due to " << *post
                         << " usetree: " << *inst << "\n";
          else
            llvm::errs() << " [mem] failed to replace function "
                         << (*calledValue) << " due to " << *post
                         << " usetree: " << *inst << "\n";
        }
        legal = false;
        return true;
      }
      return false;
    });
    if (!legal)
      break;
  }

  allFollowersOf(origop, [&](Instruction *post) {
    if (unnecessaryInstructions.count(post))
      return false;
    if (!origop->mayWriteToMemory() && !origop->mayReadFromMemory())
      return false;
    if (auto CI = dyn_cast<CallInst>(post)) {
      bool noFree = false;
      noFree |= CI->hasFnAttr(Attribute::NoFree);
      auto called = getFunctionFromCall(CI);
      StringRef funcName = getFuncNameFromCall(CI);
      if (funcName == "llvm.trap")
        noFree = true;
      if (!noFree && called) {
        noFree |= called->hasFnAttribute(Attribute::NoFree);
      }
      if (!noFree) {
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [freeing] failed to replace function "
                         << (called->getName()) << " due to freeing " << *post
                         << " usetree: " << *origop << "\n";
          else
            llvm::errs() << " [freeing] failed to replace function "
                         << (*calledValue) << " due to freeing " << *post
                         << " usetree: " << *origop << "\n";
        }
        legal = false;
        return true;
      }
    }
    return false;
  });

  if (!legal)
    return false;

  allFollowersOf(origop, [&](Instruction *inst) {
    if (auto ri = dyn_cast<ReturnInst>(inst)) {
      auto find = replacedReturns.find(ri);
      if (find != replacedReturns.end()) {
        postCreate.push_back(find->second);
        return false;
      }
    }

    if (usetree.count(inst) == 0)
      return false;
    if (inst->getParent() != origop->getParent()) {
      // Don't move a writing instruction (may change speculatable/etc things)
      if (inst->mayWriteToMemory()) {
        if (EnzymePrintPerf) {
          if (called)
            llvm::errs() << " [nonspec] failed to replace function "
                         << (called->getName()) << " due to " << *inst << "\n";
          else
            llvm::errs() << " [nonspec] failed to replace function "
                         << (*calledValue) << " due to " << *inst << "\n";
        }
        legal = false;
        // Early exit
        return true;
      }
    }
    if (isa<CallInst>(inst) &&
        gutils->originalToNewFn.find(inst) == gutils->originalToNewFn.end()) {
      legal = false;
      if (EnzymePrintPerf) {
        if (called)
          llvm::errs() << " [premove] failed to replace function "
                       << (called->getName()) << " due to " << *inst << "\n";
        else
          llvm::errs() << " [premove] failed to replace function "
                       << (*calledValue) << " due to " << *inst << "\n";
      }
      // Early exit
      return true;
    }
    postCreate.push_back(gutils->getNewFromOriginal(inst));
    return false;
  });

  if (!legal)
    return false;

  if (EnzymePrintPerf) {
    if (called)
      llvm::errs() << " choosing to replace function " << (called->getName())
                   << " and do both forward/reverse\n";
    else
      llvm::errs() << " choosing to replace function " << (*calledValue)
                   << " and do both forward/reverse\n";
  }

  return true;
}

void clearFunctionAttributes(Function *f) {
  for (Argument &Arg : f->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }
  if (f->hasFnAttribute(Attribute::OptimizeNone))
    f->removeFnAttr(Attribute::OptimizeNone);

#if LLVM_VERSION_MAJOR >= 14
  if (f->getAttributes().getRetDereferenceableBytes())
#else
  if (f->getDereferenceableBytes(llvm::AttributeList::ReturnIndex))
#endif
  {
#if LLVM_VERSION_MAJOR >= 14
    f->removeRetAttr(Attribute::Dereferenceable);
#else
    f->removeAttribute(llvm::AttributeList::ReturnIndex,
                       Attribute::Dereferenceable);
#endif
  }

  if (f->getAttributes().getRetAlignment()) {
#if LLVM_VERSION_MAJOR >= 14
    f->removeRetAttr(Attribute::Alignment);
#else
    f->removeAttribute(llvm::AttributeList::ReturnIndex, Attribute::Alignment);
#endif
  }
  Attribute::AttrKind attrs[] = {
#if LLVM_VERSION_MAJOR >= 17
    Attribute::NoFPClass,
#endif
    Attribute::NoUndef,
    Attribute::NonNull,
    Attribute::ZExt,
    Attribute::SExt,
    Attribute::NoAlias
  };
  for (auto attr : attrs) {
#if LLVM_VERSION_MAJOR >= 14
    if (f->hasRetAttribute(attr)) {
      f->removeRetAttr(attr);
    }
#else
    if (f->hasAttribute(llvm::AttributeList::ReturnIndex, attr)) {
      f->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }
  for (auto attr : {"enzyme_inactive", "enzyme_type"}) {
#if LLVM_VERSION_MAJOR >= 14
    if (f->getAttributes().hasRetAttr(attr)) {
      f->removeRetAttr(attr);
    }
#else
    if (f->getAttributes().hasAttribute(llvm::AttributeList::ReturnIndex,
                                        attr)) {
      f->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }
}

void cleanupInversionAllocs(DiffeGradientUtils *gutils, BasicBlock *entry) {
  while (gutils->inversionAllocs->size() > 0) {
    Instruction *inst = &gutils->inversionAllocs->back();
    if (isa<AllocaInst>(inst))
      inst->moveBefore(&gutils->newFunc->getEntryBlock().front());
    else
      inst->moveBefore(entry->getFirstNonPHIOrDbgOrLifetime());
  }

  (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);
  for (auto BBs : gutils->reverseBlocks) {
    if (pred_begin(BBs.second.front()) == pred_end(BBs.second.front())) {
      (IRBuilder<>(BBs.second.front())).CreateUnreachable();
      DeleteDeadBlock(BBs.second.front());
    }
  }
}

void restoreCache(
    DiffeGradientUtils *gutils,
    const std::map<std::pair<Instruction *, CacheType>, int> &mapping,
    const SmallPtrSetImpl<BasicBlock *> &guaranteedUnreachable) {
  // One must use this temporary map to first create all the replacements
  // prior to actually replacing to ensure that getSubLimits has the same
  // behavior and unwrap behavior for all replacements.
  SmallVector<std::pair<Value *, Value *>, 4> newIToNextI;

  for (const auto &m : mapping) {
    if (m.first.second == CacheType::Self &&
        gutils->knownRecomputeHeuristic.count(m.first.first)) {
      assert(gutils->knownRecomputeHeuristic.count(m.first.first));
      if (!isa<CallInst>(m.first.first)) {
        auto newi = gutils->getNewFromOriginal(m.first.first);
        if (auto PN = dyn_cast<PHINode>(newi))
          if (gutils->fictiousPHIs.count(PN)) {
            assert(gutils->fictiousPHIs[PN] == m.first.first);
            gutils->fictiousPHIs.erase(PN);
          }
        IRBuilder<> BuilderZ(newi->getNextNode());
        if (isa<PHINode>(m.first.first)) {
          BuilderZ.SetInsertPoint(
              cast<Instruction>(newi)->getParent()->getFirstNonPHI());
        }
        Value *nexti = gutils->cacheForReverse(BuilderZ, newi, m.second,
                                               /*replace*/ false);
        newIToNextI.emplace_back(newi, nexti);
      } else {
        auto newi = gutils->getNewFromOriginal((Value *)m.first.first);
        newIToNextI.emplace_back(newi, newi);
      }
    }
  }

  std::map<Value *, SmallVector<Instruction *, 4>> unwrapToOrig;
  for (auto pair : gutils->unwrappedLoads)
    unwrapToOrig[pair.second].push_back(const_cast<Instruction *>(pair.first));
  gutils->unwrappedLoads.clear();

  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    if (newi != nexti) {
      gutils->replaceAWithB(newi, nexti);
    }
  }

  // This most occur after all the replacements have been made
  // in the previous loop, lest a loop bound being unwrapped use
  // a value being replaced.
  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    for (auto V : unwrapToOrig[newi]) {
      ValueToValueMapTy available;
      if (auto MD = hasMetadata(V, "enzyme_available")) {
        for (auto &pair : MD->operands()) {
          auto tup = cast<MDNode>(pair);
          auto val = cast<ValueAsMetadata>(tup->getOperand(1))->getValue();
          assert(val);
          available[cast<ValueAsMetadata>(tup->getOperand(0))->getValue()] =
              val;
        }
      }
      IRBuilder<> lb(V);
      // This must disallow caching here as otherwise performing the loop in
      // the wrong order may result in first replacing the later unwrapped
      // value, caching it, then attempting to reuse it for an earlier
      // replacement.
      Value *nval = gutils->unwrapM(nexti, lb, available,
                                    UnwrapMode::LegalFullUnwrapNoTapeReplace,
                                    /*scope*/ nullptr, /*permitCache*/ false);
      assert(nval);
      V->replaceAllUsesWith(nval);
      V->eraseFromParent();
    }
  }

  // Erasure happens after to not erase the key of unwrapToOrig
  for (auto pair : newIToNextI) {
    auto newi = pair.first;
    auto nexti = pair.second;
    if (newi != nexti) {
      if (auto inst = dyn_cast<Instruction>(newi))
        gutils->erase(inst);
    }
  }

  // TODO also can consider switch instance as well
  // TODO can also insert to topLevel as well [note this requires putting the
  // intrinsic at the correct location]
  for (auto &BB : *gutils->oldFunc) {
    SmallVector<BasicBlock *, 4> unreachables;
    SmallVector<BasicBlock *, 4> reachables;
    for (auto Succ : successors(&BB)) {
      if (guaranteedUnreachable.find(Succ) != guaranteedUnreachable.end()) {
        unreachables.push_back(Succ);
      } else {
        reachables.push_back(Succ);
      }
    }

    if (unreachables.size() == 0 || reachables.size() == 0)
      continue;

    if (auto bi = dyn_cast<BranchInst>(BB.getTerminator())) {

      Value *condition = gutils->getNewFromOriginal(bi->getCondition());

      Constant *repVal = (bi->getSuccessor(0) == unreachables[0])
                             ? ConstantInt::getFalse(condition->getContext())
                             : ConstantInt::getTrue(condition->getContext());

      for (auto UI = condition->use_begin(), E = condition->use_end();
           UI != E;) {
        Use &U = *UI;
        ++UI;
        U.set(repVal);
      }
    }
    if (reachables.size() == 1)
      if (auto si = dyn_cast<SwitchInst>(BB.getTerminator())) {
        Value *condition = gutils->getNewFromOriginal(si->getCondition());

        Constant *repVal = nullptr;
        if (si->getDefaultDest() == reachables[0]) {
          std::set<int64_t> cases;
          for (auto c : si->cases()) {
            // TODO this doesnt work with unsigned 64 bit ints or higher
            // integer widths
            cases.insert(cast<ConstantInt>(c.getCaseValue())->getSExtValue());
          }
          int64_t legalNot = 0;
          while (cases.count(legalNot))
            legalNot++;
          repVal = ConstantInt::getSigned(condition->getType(), legalNot);
          cast<SwitchInst>(gutils->getNewFromOriginal(si))
              ->setCondition(repVal);
          // knowing which input was provided for the default dest is not
          // possible at compile time, give up on other use replacement
          continue;
        } else {
          for (auto c : si->cases()) {
            if (c.getCaseSuccessor() == reachables[0]) {
              repVal = c.getCaseValue();
            }
          }
        }
        assert(repVal);
        for (auto UI = condition->use_begin(), E = condition->use_end();
             UI != E;) {
          Use &U = *UI;
          ++UI;
          U.set(repVal);
        }
      }
  }
}

//! return structtype if recursive function
const AugmentedReturn &EnzymeLogic::CreateAugmentedPrimal(
    RequestContext context, Function *todiff, DIFFE_TYPE retType,
    ArrayRef<DIFFE_TYPE> constant_args, TypeAnalysis &TA, bool returnUsed,
    bool shadowReturnUsed, const FnTypeInfo &oldTypeInfo_,
    const std::vector<bool> _overwritten_args, bool forceAnonymousTape,
    unsigned width, bool AtomicAdd, bool omp) {

  TimeTraceScope timeScope("CreateAugmentedPrimal", todiff->getName());

  if (returnUsed)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());
  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isEmptyTy() &&
           !todiff->getReturnType()->isVoidTy());

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(oldTypeInfo_, todiff);
  AugmentedCacheKey tup = {todiff,        retType,
                           constant_args, _overwritten_args,
                           returnUsed,    shadowReturnUsed,
                           oldTypeInfo,   forceAnonymousTape,
                           AtomicAdd,     omp,
                           width};

  if (_overwritten_args.size() != todiff->arg_size()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << " overwritten_args.size() [" << _overwritten_args.size()
       << "] != todiff->arg_size()\n";
    ss << "todiff: " << *todiff << "\n";
    if (context.req) {
      ss << " at context: " << *context.req;
    } else {
      ss << *todiff << "\n";
    }
    if (EmitNoDerivativeError(ss.str(), todiff, context)) {
      auto newFunc = todiff;
      std::map<AugmentedStruct, int> returnMapping;
      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(newFunc, nullptr, {}, returnMapping, {}, {},
                                 constant_args, shadowReturnUsed))
          ->second;
    }
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
    llvm_unreachable(
        "attempting to differentiate function with wrong overwritten count");
  }

  assert(_overwritten_args.size() == todiff->arg_size());
  assert(constant_args.size() == todiff->getFunctionType()->getNumParams());

  auto found = AugmentedCachedFunctions.find(tup);
  if (found != AugmentedCachedFunctions.end()) {
    return found->second;
  }
  TargetLibraryInfo &TLI = PPC.FAM.getResult<TargetLibraryAnalysis>(*todiff);

  // TODO make default typing (not just constant)

  if (auto md = hasMetadata(todiff, "enzyme_augment")) {
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown augment for noninvertible function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    bool hasconstant = false;
    for (auto v : constant_args) {
      if (v == DIFFE_TYPE::CONSTANT) {
        hasconstant = true;
        break;
      }
    }

    if (hasconstant) {
      EmitWarning("NoCustom", *todiff,
                  "Massaging provided custom augmented forward pass to handle "
                  "constant argumented");
      SmallVector<Type *, 3> dupargs;
      std::vector<DIFFE_TYPE> next_constant_args(constant_args.begin(),
                                                 constant_args.end());
      {
        auto OFT = todiff->getFunctionType();
        for (size_t act_idx = 0; act_idx < constant_args.size(); act_idx++) {
          dupargs.push_back(OFT->getParamType(act_idx));
          switch (constant_args[act_idx]) {
          case DIFFE_TYPE::OUT_DIFF:
            break;
          case DIFFE_TYPE::DUP_ARG:
          case DIFFE_TYPE::DUP_NONEED:
            dupargs.push_back(OFT->getParamType(act_idx));
            break;
          case DIFFE_TYPE::CONSTANT:
            if (!OFT->getParamType(act_idx)->isFPOrFPVectorTy()) {
              next_constant_args[act_idx] = DIFFE_TYPE::DUP_ARG;
            } else {
              next_constant_args[act_idx] = DIFFE_TYPE::OUT_DIFF;
            }
            break;
          }
        }
      }

      auto &aug = CreateAugmentedPrimal(
          context, todiff, retType, next_constant_args, TA, returnUsed,
          shadowReturnUsed, oldTypeInfo_, _overwritten_args, forceAnonymousTape,
          width, AtomicAdd, omp);

      FunctionType *FTy =
          FunctionType::get(aug.fn->getReturnType(), dupargs,
                            todiff->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixaugment_" + todiff->getName(), todiff->getParent());

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto arg = NewF->arg_begin();
      SmallVector<Value *, 3> fwdargs;
      int act_idx = 0;
      while (arg != NewF->arg_end()) {
        arg->setName("arg" + Twine(act_idx));
        fwdargs.push_back(arg);
        switch (constant_args[act_idx]) {
        case DIFFE_TYPE::OUT_DIFF:
          break;
        case DIFFE_TYPE::DUP_ARG:
        case DIFFE_TYPE::DUP_NONEED:
          arg++;
          arg->setName("arg" + Twine(act_idx) + "'");
          fwdargs.push_back(arg);
          break;
        case DIFFE_TYPE::CONSTANT:
          if (next_constant_args[act_idx] != DIFFE_TYPE::OUT_DIFF) {
            fwdargs.push_back(arg);
          }
          break;
        }
        arg++;
        act_idx++;
      }
      auto cal = bb.CreateCall(aug.fn, fwdargs);
      cal->setCallingConv(aug.fn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy())
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      else if (NewF->getReturnType()->isVoidTy())
        bb.CreateRetVoid();
      else
        bb.CreateRet(cal);

      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(NewF, aug.tapeType, aug.tapeIndices,
                                 aug.returns, aug.overwritten_args_map,
                                 aug.can_modref_map, next_constant_args,
                                 shadowReturnUsed))
          ->second;
    }

    if (foundcalled->hasStructRetAttr() && !todiff->hasStructRetAttr()) {
      SmallVector<Type *, 3> args;
      Type *sretTy = nullptr;
      {
        size_t i = 0;
        for (auto &arg : foundcalled->args()) {
          if (!foundcalled->hasParamAttribute(i, Attribute::StructRet))
            args.push_back(arg.getType());
          else {
#if LLVM_VERSION_MAJOR >= 12
            sretTy = foundcalled->getParamAttribute(0, Attribute::StructRet)
                         .getValueAsType();
#else
            sretTy = arg.getType()->getPointerElementType();
#endif
          }
          i++;
        }
      }
      assert(foundcalled->getReturnType()->isVoidTy());
      FunctionType *FTy = FunctionType::get(
          sretTy, args, foundcalled->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixaugment_" + foundcalled->getName(), foundcalled->getParent());

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto AI = bb.CreateAlloca(sretTy);
      SmallVector<Value *, 3> argVs;
      auto arg = NewF->arg_begin();
      size_t realidx = 0;
      for (size_t i = 0; i < foundcalled->arg_size(); i++) {
        if (!foundcalled->hasParamAttribute(i, Attribute::StructRet)) {
          arg->setName("arg" + Twine(realidx));
          realidx++;
          argVs.push_back(arg);
          ++arg;
        } else
          argVs.push_back(AI);
      }
      auto cal = bb.CreateCall(foundcalled, argVs);
      cal->setCallingConv(foundcalled->getCallingConv());

      Value *res = bb.CreateLoad(sretTy, AI);
      bb.CreateRet(res);

      todiff->setMetadata(
          "enzyme_augment",
          llvm::MDTuple::get(todiff->getContext(),
                             {llvm::ValueAsMetadata::get(NewF)}));
      foundcalled = NewF;
    }

    if (foundcalled->getReturnType() == todiff->getReturnType()) {
      std::map<AugmentedStruct, int> returnMapping;
      returnMapping[AugmentedStruct::Return] = -1;
      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                 {}, constant_args, shadowReturnUsed))
          ->second;
    }

    if (auto ST = dyn_cast<StructType>(foundcalled->getReturnType())) {
      if (ST->getNumElements() == 3) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Tape] = 0;
        returnMapping[AugmentedStruct::Return] = 1;
        returnMapping[AugmentedStruct::DifferentialReturn] = 2;
        if (ST->getTypeAtIndex(1) != todiff->getReturnType() ||
            ST->getTypeAtIndex(2) != todiff->getReturnType()) {
          Type *retTys[] = {ST->getTypeAtIndex((unsigned)0),
                            todiff->getReturnType(), todiff->getReturnType()};
          auto RT =
              StructType::get(ST->getContext(), retTys, /*isPacked*/ false);
          FunctionType *FTy =
              FunctionType::get(RT, foundcalled->getFunctionType()->params(),
                                foundcalled->getFunctionType()->isVarArg());
          Function *NewF = Function::Create(
              FTy, Function::LinkageTypes::InternalLinkage,
              "fixaugment_" + foundcalled->getName(), foundcalled->getParent());

          BasicBlock *BB =
              BasicBlock::Create(NewF->getContext(), "entry", NewF);
          IRBuilder<> bb(BB);
          SmallVector<Value *, 3> argVs;
          size_t realidx = 0;
          for (auto &a : NewF->args()) {
            a.setName("arg" + Twine(realidx));
            realidx++;
            argVs.push_back(&a);
          }
          auto cal = bb.CreateCall(foundcalled, argVs);
          cal->setCallingConv(foundcalled->getCallingConv());

          Value *res = UndefValue::get(RT);
          res = bb.CreateInsertValue(res, bb.CreateExtractValue(cal, {0}), {0});
          for (unsigned i = 1; i <= 2; i++) {
            auto AI = bb.CreateAlloca(todiff->getReturnType());
            bb.CreateStore(
                bb.CreateExtractValue(cal, {i}),
                bb.CreatePointerCast(
                    AI, PointerType::getUnqual(ST->getTypeAtIndex(i))));
            Value *vres = bb.CreateLoad(todiff->getReturnType(), AI);
            res = bb.CreateInsertValue(res, vres, {i});
          }
          bb.CreateRet(res);

          todiff->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(todiff->getContext(),
                                 {llvm::ValueAsMetadata::get(NewF)}));
          foundcalled = NewF;
        }
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}, constant_args, shadowReturnUsed))
            ->second;
      }
      if (ST->getNumElements() == 2 &&
          ST->getElementType(0) == ST->getElementType(1)) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Return] = 0;
        returnMapping[AugmentedStruct::DifferentialReturn] = 1;
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}, constant_args, shadowReturnUsed))
            ->second;
      }
      if (ST->getNumElements() == 2) {
        std::map<AugmentedStruct, int> returnMapping;
        returnMapping[AugmentedStruct::Tape] = 0;
        returnMapping[AugmentedStruct::Return] = 1;
        if (ST->getTypeAtIndex(1) != todiff->getReturnType()) {
          Type *retTys[] = {ST->getTypeAtIndex((unsigned)0),
                            todiff->getReturnType()};
          auto RT =
              StructType::get(ST->getContext(), retTys, /*isPacked*/ false);
          FunctionType *FTy =
              FunctionType::get(RT, foundcalled->getFunctionType()->params(),
                                foundcalled->getFunctionType()->isVarArg());
          Function *NewF = Function::Create(
              FTy, Function::LinkageTypes::InternalLinkage,
              "fixaugment_" + foundcalled->getName(), foundcalled->getParent());

          BasicBlock *BB =
              BasicBlock::Create(NewF->getContext(), "entry", NewF);
          IRBuilder<> bb(BB);
          SmallVector<Value *, 3> argVs;
          size_t realidx = 0;
          for (auto &a : NewF->args()) {
            a.setName("arg" + Twine(realidx));
            realidx++;
            argVs.push_back(&a);
          }
          auto cal = bb.CreateCall(foundcalled, argVs);
          cal->setCallingConv(foundcalled->getCallingConv());

          Value *res = UndefValue::get(RT);
          res = bb.CreateInsertValue(res, bb.CreateExtractValue(cal, {0}), {0});
          for (unsigned i = 1; i <= 1; i++) {
            auto AI = bb.CreateAlloca(todiff->getReturnType());
            bb.CreateStore(
                bb.CreateExtractValue(cal, {i}),
                bb.CreatePointerCast(
                    AI, PointerType::getUnqual(ST->getTypeAtIndex(i))));
            Value *vres = bb.CreateLoad(todiff->getReturnType(), AI);
            res = bb.CreateInsertValue(res, vres, {i});
          }
          bb.CreateRet(res);

          todiff->setMetadata(
              "enzyme_augment",
              llvm::MDTuple::get(todiff->getContext(),
                                 {llvm::ValueAsMetadata::get(NewF)}));
          foundcalled = NewF;
        }
        return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                   AugmentedCachedFunctions, tup,
                   AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {},
                                   {}, constant_args, shadowReturnUsed))
            ->second;
      }
    }

    std::map<AugmentedStruct, int> returnMapping;
    if (!foundcalled->getReturnType()->isVoidTy())
      returnMapping[AugmentedStruct::Tape] = -1;

    return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
               AugmentedCachedFunctions, tup,
               AugmentedReturn(foundcalled, nullptr, {}, returnMapping, {}, {},
                               constant_args, shadowReturnUsed))
        ->second; // dyn_cast<StructType>(st->getElementType(0)));
  }

  std::map<AugmentedStruct, int> returnMapping;

  GradientUtils *gutils = GradientUtils::CreateFromClone(
      *this, width, todiff, TLI, TA, oldTypeInfo, retType, constant_args,
      /*returnUsed*/ returnUsed, /*shadowReturnUsed*/ shadowReturnUsed,
      returnMapping, omp);

  if (todiff->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No augmented forward pass found for " + todiff->getName();
    {
      std::string demangledName = llvm::demangle(todiff->getName().str());
      // replace all '> >' with '>>'
      size_t start = 0;
      while ((start = demangledName.find("> >", start)) != std::string::npos) {
        demangledName.replace(start, 3, ">>");
      }
      if (demangledName != todiff->getName())
        ss << "(" << demangledName << ")";
    }
    ss << "\n";
    if (context.req) {
      ss << " at context: " << *context.req;
    } else {
      ss << *todiff << "\n";
    }
    (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
    DeleteDeadBlock(gutils->inversionAllocs);
    clearFunctionAttributes(gutils->newFunc);
    if (EmitNoDerivativeError(ss.str(), todiff, context)) {
      auto newFunc = gutils->newFunc;
      delete gutils;
      return insert_or_assign<AugmentedCacheKey, AugmentedReturn>(
                 AugmentedCachedFunctions, tup,
                 AugmentedReturn(newFunc, nullptr, {}, returnMapping, {}, {},
                                 constant_args, shadowReturnUsed))
          ->second;
    }
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
    llvm_unreachable("attempting to differentiate function without definition");
  }
  gutils->AtomicAdd = AtomicAdd;
  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  // Convert uncacheable args from the input function to the preprocessed
  // function
  std::vector<bool> _overwritten_argsPP = _overwritten_args;

  gutils->forceActiveDetection();
  gutils->computeGuaranteedFrees();

  CacheAnalysis CA(gutils->allocationsWithGuaranteedFree,
                   gutils->rematerializableAllocations, gutils->TR,
                   *gutils->OrigAA, gutils->oldFunc,
                   PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
                   *gutils->OrigLI, *gutils->OrigDT, TLI, guaranteedUnreachable,
                   _overwritten_argsPP, DerivativeMode::ReverseModePrimal, omp);
  const std::map<CallInst *, const std::vector<bool>> overwritten_args_map =
      CA.compute_overwritten_args_for_callsites();
  gutils->overwritten_args_map_ptr = &overwritten_args_map;

  const std::map<Instruction *, bool> can_modref_map =
      CA.compute_uncacheable_load_map();
  gutils->can_modref_map = &can_modref_map;

  gutils->forceAugmentedReturns();

  gutils->computeMinCache();

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues,
                                  unnecessaryInstructions, returnUsed,
                                  DerivativeMode::ReverseModePrimal, gutils,
                                  TLI, constant_args, guaranteedUnreachable);
  gutils->unnecessaryValuesP = &unnecessaryValues;

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);

  insert_or_assign(AugmentedCachedFunctions, tup,
                   AugmentedReturn(gutils->newFunc, nullptr, {}, returnMapping,
                                   overwritten_args_map, can_modref_map,
                                   constant_args, shadowReturnUsed));

  auto getIndex = [&](Instruction *I, CacheType u, IRBuilder<> &B) -> unsigned {
    return gutils->getIndex(
        std::make_pair(I, u),
        AugmentedCachedFunctions.find(tup)->second.tapeIndices, B);
  };

  //! Explicitly handle all returns first to ensure that all instructions know
  //! whether or not they are used
  SmallPtrSet<Instruction *, 4> returnuses;

  for (BasicBlock &BB : *gutils->oldFunc) {
    if (auto orig_ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
      auto ri = gutils->getNewFromOriginal(orig_ri);
      Value *orig_oldval = orig_ri->getReturnValue();
      Value *oldval =
          orig_oldval ? gutils->getNewFromOriginal(orig_oldval) : nullptr;
      IRBuilder<> ib(ri);
      Value *rt = UndefValue::get(gutils->newFunc->getReturnType());
      if (oldval && returnUsed) {
        assert(returnMapping.find(AugmentedStruct::Return) !=
               returnMapping.end());
        auto idx = returnMapping.find(AugmentedStruct::Return)->second;
        if (idx < 0)
          rt = oldval;
        else
          rt = ib.CreateInsertValue(rt, oldval, {(unsigned)idx});
        if (Instruction *inst = dyn_cast<Instruction>(rt)) {
          returnuses.insert(inst);
        }
      }

      auto newri = ib.CreateRet(rt);
      gutils->originalToNewFn[orig_ri] = newri;
      gutils->newToOriginalFn.erase(ri);
      gutils->newToOriginalFn[newri] = orig_ri;
      gutils->erase(ri);
    }
  }

  AdjointGenerator maker(DerivativeMode::ReverseModePrimal, gutils,
                         constant_args, retType, getIndex, overwritten_args_map,
                         &returnuses,
                         &AugmentedCachedFunctions.find(tup)->second, nullptr,
                         unnecessaryValues, unnecessaryInstructions,
                         unnecessaryStores, guaranteedUnreachable, nullptr);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    auto term = oBB.getTerminator();
    assert(term);

    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      SmallVector<Instruction *, 4> toerase;

      // For having the prints still exist on bugs, check if indeed unused
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : toerase) {
        maker.eraseIfUnused(*I, /*erase*/ true, /*check*/ true);
      }
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      if (!newBB->getTerminator()) {
        for (auto next : successors(&oBB)) {
          auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
          sucBB->removePredecessor(newBB, /*KeepOneInputPHIs*/ true);
        }
        IRBuilder<> builder(newBB);
        builder.CreateUnreachable();
      }
      continue;
    }

    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
      llvm_unreachable("unknown terminator inst");
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    ++I;
    for (; I != E; ++I) {
      maker.visit(&*I);
      assert(oBB.rend() == E);
    }
  }

  if (gutils->knownRecomputeHeuristic.size()) {
    // Even though we could simply iterate through the heuristic map,
    // we explicity iterate in order of the instructions to maintain
    // a deterministic cache ordering.
    for (auto &BB : *gutils->oldFunc)
      for (auto &I : BB) {
        auto found = gutils->knownRecomputeHeuristic.find(&I);
        if (found != gutils->knownRecomputeHeuristic.end()) {
          if (!found->second && !isa<CallInst>(&I)) {
            auto newi = gutils->getNewFromOriginal(&I);
            IRBuilder<> BuilderZ(cast<Instruction>(newi)->getNextNode());
            if (isa<PHINode>(newi)) {
              BuilderZ.SetInsertPoint(
                  cast<Instruction>(newi)->getParent()->getFirstNonPHI());
            }
            gutils->cacheForReverse(BuilderZ, newi,
                                    getIndex(&I, CacheType::Self, BuilderZ));
          }
        }
      }
  }

  auto nf = gutils->newFunc;

  while (gutils->inversionAllocs->size() > 0) {
    gutils->inversionAllocs->back().moveBefore(
        gutils->newFunc->getEntryBlock().getFirstNonPHIOrDbgOrLifetime());
  }

  //! Keep track of inverted pointers we may need to return
  ValueToValueMapTy invertedRetPs;
  if (shadowReturnUsed) {
    for (BasicBlock &BB : *gutils->oldFunc) {
      if (auto ri = dyn_cast<ReturnInst>(BB.getTerminator())) {
        if (Value *orig_oldval = ri->getReturnValue()) {
          auto newri = gutils->getNewFromOriginal(ri);
          IRBuilder<> BuilderZ(newri);
          Value *invertri = nullptr;
          if (gutils->isConstantValue(orig_oldval)) {
            if (!EnzymeRuntimeActivityCheck &&
                gutils->TR.query(orig_oldval)[{-1}].isPossiblePointer()) {
              if (!isa<UndefValue>(orig_oldval) &&
                  !isa<ConstantPointerNull>(orig_oldval)) {
                std::string str;
                raw_string_ostream ss(str);
                ss << "Mismatched activity for: " << *ri
                   << " const val: " << *orig_oldval;
                if (CustomErrorHandler)
                  invertri = unwrap(CustomErrorHandler(
                      str.c_str(), wrap(ri), ErrorType::MixedActivityError,
                      gutils, wrap(orig_oldval), wrap(&BuilderZ)));
                else
                  EmitWarning("MixedActivityError", *ri, ss.str());
              }
            }
          }
          if (!invertri)
            invertri = gutils->invertPointerM(orig_oldval, BuilderZ,
                                              /*nullShadow*/ true);
          invertedRetPs[newri] = invertri;
        }
      }
    }
  }

  (IRBuilder<>(gutils->inversionAllocs)).CreateUnreachable();
  DeleteDeadBlock(gutils->inversionAllocs);

  for (Argument &Arg : gutils->newFunc->args()) {
    if (Arg.hasAttribute(Attribute::Returned))
      Arg.removeAttr(Attribute::Returned);
    if (Arg.hasAttribute(Attribute::StructRet))
      Arg.removeAttr(Attribute::StructRet);
  }

  if (gutils->newFunc->hasFnAttribute(Attribute::OptimizeNone))
    gutils->newFunc->removeFnAttr(Attribute::OptimizeNone);

#if LLVM_VERSION_MAJOR >= 14
  if (gutils->newFunc->getAttributes().getRetDereferenceableBytes())
#else
  if (gutils->newFunc->getDereferenceableBytes(
          llvm::AttributeList::ReturnIndex))
#endif
  {
#if LLVM_VERSION_MAJOR >= 14
    gutils->newFunc->removeRetAttr(Attribute::Dereferenceable);
#else
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     Attribute::Dereferenceable);
#endif
  }

  // TODO could keep nonnull if returning value -1
  if (gutils->newFunc->getAttributes().getRetAlignment()) {
#if LLVM_VERSION_MAJOR >= 14
    gutils->newFunc->removeRetAttr(Attribute::Alignment);
#else
    gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex,
                                     Attribute::Alignment);
#endif
  }

  llvm::Attribute::AttrKind attrs[] = {
#if LLVM_VERSION_MAJOR >= 17
    llvm::Attribute::NoFPClass,
#endif
    llvm::Attribute::NoAlias,
    llvm::Attribute::NoUndef,
    llvm::Attribute::NonNull,
    llvm::Attribute::ZExt,
    llvm::Attribute::SExt,
  };
  for (auto attr : attrs) {
#if LLVM_VERSION_MAJOR >= 14
    if (gutils->newFunc->hasRetAttribute(attr)) {
      gutils->newFunc->removeRetAttr(attr);
    }
#else
    if (gutils->newFunc->hasAttribute(llvm::AttributeList::ReturnIndex, attr)) {
      gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }
  for (auto attr : {"enzyme_inactive", "enzyme_type"}) {
#if LLVM_VERSION_MAJOR >= 14
    if (gutils->newFunc->getAttributes().hasRetAttr(attr)) {
      gutils->newFunc->removeRetAttr(attr);
    }
#else
    if (gutils->newFunc->getAttributes().hasAttribute(
            llvm::AttributeList::ReturnIndex, attr)) {
      gutils->newFunc->removeAttribute(llvm::AttributeList::ReturnIndex, attr);
    }
#endif
  }

  gutils->eraseFictiousPHIs();

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (2)");
  }

  SmallVector<Type *, 4> MallocTypes;

  bool nonRecursiveUse = false;

  for (auto a : gutils->getTapeValues()) {
    MallocTypes.push_back(a->getType());
    if (auto ei = dyn_cast<ExtractValueInst>(a)) {
      auto tidx = returnMapping.find(AugmentedStruct::Tape)->second;
      if (ei->getIndices().size() == 1 && ei->getIndices()[0] == (unsigned)tidx)
        if (auto cb = dyn_cast<CallBase>(ei->getOperand(0)))
          if (gutils->newFunc == cb->getCalledFunction())
            continue;
    }
    nonRecursiveUse = true;
  }
  if (MallocTypes.size() == 0)
    nonRecursiveUse = true;
  if (!nonRecursiveUse)
    MallocTypes.clear();

  Type *tapeType = StructType::get(nf->getContext(), MallocTypes);

  bool removeTapeStruct = MallocTypes.size() == 1;
  if (removeTapeStruct) {
    tapeType = MallocTypes[0];

    for (auto &a : AugmentedCachedFunctions.find(tup)->second.tapeIndices) {
      a.second = -1;
    }
  }

  bool recursive =
      AugmentedCachedFunctions.find(tup)->second.fn->getNumUses() > 0 ||
      forceAnonymousTape;
  bool noTape = MallocTypes.size() == 0 && !forceAnonymousTape;

  StructType *sty = cast<StructType>(gutils->newFunc->getReturnType());
  SmallVector<Type *, 4> RetTypes(sty->elements().begin(),
                                  sty->elements().end());
  if (!noTape) {
    if (recursive && !omp) {
      auto size =
          gutils->newFunc->getParent()->getDataLayout().getTypeAllocSizeInBits(
              tapeType);
      if (size != 0) {
        RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] =
            getDefaultAnonymousTapeType(gutils->newFunc->getContext());
      }
    }
  }

  int oldretIdx = -1;
  if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
    oldretIdx = returnMapping[AugmentedStruct::Return];
  }

  if (noTape || omp) {
    auto tidx = returnMapping.find(AugmentedStruct::Tape)->second;
    if (noTape)
      returnMapping.erase(AugmentedStruct::Tape);
    if (noTape)
      AugmentedCachedFunctions.find(tup)->second.returns.erase(
          AugmentedStruct::Tape);
    if (returnMapping.find(AugmentedStruct::Return) != returnMapping.end()) {
      AugmentedCachedFunctions.find(tup)
          ->second.returns[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
      returnMapping[AugmentedStruct::Return] -=
          (returnMapping[AugmentedStruct::Return] > tidx) ? 1 : 0;
    }
    if (returnMapping.find(AugmentedStruct::DifferentialReturn) !=
        returnMapping.end()) {
      AugmentedCachedFunctions.find(tup)
          ->second.returns[AugmentedStruct::DifferentialReturn] -=
          (returnMapping[AugmentedStruct::DifferentialReturn] > tidx) ? 1 : 0;
      returnMapping[AugmentedStruct::DifferentialReturn] -=
          (returnMapping[AugmentedStruct::DifferentialReturn] > tidx) ? 1 : 0;
    }
    RetTypes.erase(RetTypes.begin() + tidx);
  } else if (recursive) {
  } else {
    RetTypes[returnMapping.find(AugmentedStruct::Tape)->second] = tapeType;
  }

  bool noReturn = RetTypes.size() == 0;
  Type *RetType = StructType::get(nf->getContext(), RetTypes);
  if (noReturn)
    RetType = Type::getVoidTy(RetType->getContext());
  if (noReturn)
    assert(noTape || omp);

  bool removeStruct = RetTypes.size() == 1;

  if (removeStruct) {
    RetType = RetTypes[0];
    for (auto &a : returnMapping) {
      a.second = -1;
    }
    for (auto &a : AugmentedCachedFunctions.find(tup)->second.returns) {
      a.second = -1;
    }
  }

  ValueToValueMapTy VMap;
  SmallVector<Type *, 4> ArgTypes;
  for (const Argument &I : nf->args()) {
    ArgTypes.push_back(I.getType());
  }

  if (omp && !noTape) {
    // see lack of struct type for openmp
    ArgTypes.push_back(PointerType::getUnqual(tapeType));
    // ArgTypes.push_back(tapeType);
  }

  // Create a new function type...
  FunctionType *FTy =
      FunctionType::get(RetType, ArgTypes, nf->getFunctionType()->isVarArg());

  // Create the new function...
  Function *NewF = Function::Create(
      FTy, nf->getLinkage(), "augmented_" + todiff->getName(), nf->getParent());

  unsigned attrIndex = 0;
  auto i = nf->arg_begin(), j = NewF->arg_begin();
  while (i != nf->arg_end()) {
    VMap[i] = j;
    if (nf->hasParamAttribute(attrIndex, Attribute::NoCapture)) {
      NewF->addParamAttr(attrIndex, Attribute::NoCapture);
    }
    if (nf->hasParamAttribute(attrIndex, Attribute::NoAlias)) {
      NewF->addParamAttr(attrIndex, Attribute::NoAlias);
    }
    for (auto name : {"enzyme_sret", "enzyme_sret_v", "enzymejl_returnRoots",
                      "enzymejl_returnRoots_v", "enzymejl_parmtype",
                      "enzymejl_parmtype_ref", "enzyme_type"})
      if (nf->getAttributes().hasParamAttr(attrIndex, name)) {
        NewF->addParamAttr(attrIndex,
                           nf->getAttributes().getParamAttr(attrIndex, name));
      }

    j->setName(i->getName());
    ++j;
    ++i;
    ++attrIndex;
  }

#if LLVM_VERSION_MAJOR >= 14
  for (auto attr : {"enzyme_ta_norecur"})
    if (nf->getAttributes().hasAttributeAtIndex(AttributeList::FunctionIndex,
                                                attr)) {
      NewF->addFnAttr(
          nf->getAttributes().getAttribute(AttributeList::FunctionIndex, attr));
    }

  for (auto attr :
       {"enzyme_type", "enzymejl_parmtype", "enzymejl_parmtype_ref"})
    if (nf->getAttributes().hasAttributeAtIndex(AttributeList::ReturnIndex,
                                                attr)) {
      NewF->addAttribute(
          AttributeList::ReturnIndex,
          nf->getAttributes().getAttribute(AttributeList::ReturnIndex, attr));
    }
#else
  for (auto attr : {"enzyme_ta_norecur"})
    if (nf->getAttributes().hasAttribute(AttributeList::FunctionIndex, attr)) {
      NewF->addFnAttr(
          nf->getAttributes().getAttribute(AttributeList::FunctionIndex, attr));
    }

  for (auto attr :
       {"enzyme_type", "enzymejl_parmtype", "enzymejl_parmtype_ref"})
    if (nf->getAttributes().hasAttribute(AttributeList::ReturnIndex, attr)) {
      NewF->addAttribute(
          AttributeList::ReturnIndex,
          nf->getAttributes().getAttribute(AttributeList::ReturnIndex, attr));
    }
#endif

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, nf, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, nf, VMap, nf->getSubprogram() != nullptr, Returns, "",
                    nullptr);
#endif

  IRBuilder<> ib(NewF->getEntryBlock().getFirstNonPHI());

  AllocaInst *ret = noReturn ? nullptr : ib.CreateAlloca(RetType);

  if (!noTape) {
    Value *tapeMemory;
    if (recursive && !omp) {
      auto i64 = Type::getInt64Ty(NewF->getContext());
      auto size =
          NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(tapeType);
      Value *memory;
      if (size != 0) {
        CallInst *malloccall = nullptr;
        Instruction *zero = nullptr;
        tapeMemory = CreateAllocation(
            ib, tapeType, ConstantInt::get(i64, 1), "tapemem", &malloccall,
            EnzymeZeroCache ? &zero : nullptr, /*isDefault*/ true);
        memory = malloccall;
      } else {
        memory = ConstantPointerNull::get(
            getDefaultAnonymousTapeType(NewF->getContext()));
      }
      Value *Idxs[] = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      assert(memory);
      assert(ret);
      Value *gep = ret;
      if (!removeStruct) {
        gep = ib.CreateGEP(RetType, ret, Idxs, "");
        cast<GetElementPtrInst>(gep)->setIsInBounds(true);
      }
      auto storeinst = ib.CreateStore(memory, gep);
      PostCacheStore(storeinst, ib);
    } else if (omp) {
      j->setName("tape");
      tapeMemory = j;
      // if structs were supported by openmp we could do this, but alas, no
      // IRBuilder<> B(NewF->getEntryBlock().getFirstNonPHI());
      // tapeMemory = B.CreateAlloca(j->getType());
      // B.CreateStore(j, tapeMemory);
    } else {
      Value *Idxs[] = {
          ib.getInt32(0),
          ib.getInt32(returnMapping.find(AugmentedStruct::Tape)->second),
      };
      tapeMemory = ret;
      if (!removeStruct) {
        tapeMemory = ib.CreateGEP(RetType, ret, Idxs, "");
        cast<GetElementPtrInst>(tapeMemory)->setIsInBounds(true);
      }
      if (EnzymeZeroCache) {
        ZeroMemory(ib, tapeType, tapeMemory,
                   /*isTape*/ true);
      }
    }

    unsigned i = 0;
    for (auto v : gutils->getTapeValues()) {
      if (!isa<UndefValue>(v)) {
        if (!isa<Instruction>(VMap[v])) {
          llvm::errs() << " non constant for vmap[v=" << *v
                       << " ]= " << *VMap[v] << "\n";
        }
        auto inst = cast<Instruction>(VMap[v]);
        IRBuilder<> ib(inst->getNextNode());
        if (isa<PHINode>(inst))
          ib.SetInsertPoint(inst->getParent()->getFirstNonPHI());
        Value *Idxs[] = {ib.getInt32(0), ib.getInt32(i)};
        Value *gep = tapeMemory;
        if (!removeTapeStruct) {
          gep = ib.CreateGEP(tapeType, tapeMemory, Idxs, "");
          cast<GetElementPtrInst>(gep)->setIsInBounds(true);
        }
        auto storeinst = ib.CreateStore(VMap[v], gep);
        PostCacheStore(storeinst, ib);
      }
      ++i;
    }
  } else if (!nonRecursiveUse) {
    for (auto v : gutils->getTapeValues()) {
      if (isa<UndefValue>(v))
        continue;
      auto EV = cast<ExtractValueInst>(v);
      auto EV2 = cast<ExtractValueInst>(VMap[v]);
      assert(EV->use_empty());
      EV->eraseFromParent();
      assert(EV2->use_empty());
      EV2->eraseFromParent();
    }
  }

  for (BasicBlock &BB : *nf) {
    auto ri = dyn_cast<ReturnInst>(BB.getTerminator());
    if (ri == nullptr)
      continue;
    ReturnInst *rim = cast<ReturnInst>(VMap[ri]);
    IRBuilder<> ib(rim);
    if (returnUsed) {
      Value *rv = rim->getReturnValue();
      assert(rv);
      Value *actualrv = nullptr;
      if (auto iv = dyn_cast<InsertValueInst>(rv)) {
        if (iv->getNumIndices() == 1 && (int)iv->getIndices()[0] == oldretIdx) {
          actualrv = iv->getInsertedValueOperand();
        }
      }
      if (actualrv == nullptr) {
        if (oldretIdx < 0)
          actualrv = rv;
        else
          actualrv = ib.CreateExtractValue(rv, {(unsigned)oldretIdx});
      }
      Value *gep =
          removeStruct
              ? ret
              : ib.CreateConstGEP2_32(
                    RetType, ret, 0,
                    returnMapping.find(AugmentedStruct::Return)->second, "");
      if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
        ggep->setIsInBounds(true);
      }
      if (EnzymeFixupReturn)
        actualrv = unwrap(EnzymeFixupReturn(wrap(&ib), wrap(actualrv)));
      auto storeinst = ib.CreateStore(actualrv, gep);
      PostCacheStore(storeinst, ib);
    }

    if (shadowReturnUsed) {
      assert(invertedRetPs[ri]);
      Value *shadowRV = invertedRetPs[ri];

      if (!isa<UndefValue>(shadowRV)) {
        Value *gep =
            removeStruct
                ? ret
                : ib.CreateConstGEP2_32(
                      RetType, ret, 0,
                      returnMapping.find(AugmentedStruct::DifferentialReturn)
                          ->second,
                      "");
        if (auto ggep = dyn_cast<GetElementPtrInst>(gep)) {
          ggep->setIsInBounds(true);
        }
        if (!(isa<ConstantExpr>(shadowRV) || isa<ConstantData>(shadowRV) ||
              isa<ConstantAggregate>(shadowRV) ||
              isa<GlobalVariable>(shadowRV))) {
          auto found = VMap.find(shadowRV);
          assert(found != VMap.end());
          shadowRV = found->second;
        }
        if (EnzymeFixupReturn)
          shadowRV = unwrap(EnzymeFixupReturn(wrap(&ib), wrap(shadowRV)));
        auto storeinst = ib.CreateStore(shadowRV, gep);
        PostCacheStore(storeinst, ib);
      }
    }
    if (noReturn)
      ib.CreateRetVoid();
    else {
      ib.CreateRet(ib.CreateLoad(RetType, ret));
    }
    cast<Instruction>(VMap[ri])->eraseFromParent();
  }

  clearFunctionAttributes(NewF);
  PPC.LowerAllocAddr(NewF);

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("augmented function failed verification (3)");
  }
  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*NewF, PA);
  }

  SmallVector<CallInst *, 4> fnusers;
  SmallVector<std::pair<GlobalVariable *, DerivativeMode>, 1> gfnusers;
  for (auto user : AugmentedCachedFunctions.find(tup)->second.fn->users()) {
    if (auto CI = dyn_cast<CallInst>(user)) {
      fnusers.push_back(CI);
    } else {
      if (auto CS = dyn_cast<ConstantStruct>(user)) {
        for (auto cuser : CS->users()) {
          if (auto G = dyn_cast<GlobalVariable>(cuser)) {
            if (("_enzyme_reverse_" + todiff->getName() + "'").str() ==
                G->getName()) {
              gfnusers.emplace_back(G, DerivativeMode::ReverseModeGradient);
              continue;
            }
            if (("_enzyme_forwardsplit_" + todiff->getName() + "'").str() ==
                G->getName()) {
              gfnusers.emplace_back(G, DerivativeMode::ForwardModeSplit);
              continue;
            }
          }
          llvm::errs() << *gutils->newFunc->getParent() << "\n";
          llvm::errs() << *cuser << "\n";
          llvm::errs() << *user << "\n";
          llvm_unreachable("Bad cuser of staging augmented forward fn");
        }
        continue;
      }
      llvm::errs() << *gutils->newFunc->getParent() << "\n";
      llvm::errs() << *user << "\n";
      llvm_unreachable("Bad user of staging augmented forward fn");
    }
  }
  for (auto user : fnusers) {
    if (removeStruct || !nonRecursiveUse) {
      IRBuilder<> B(user);
      SmallVector<Value *, 4> args(user->arg_begin(), user->arg_end());
      auto rep = B.CreateCall(NewF, args);
      if (!rep->getType()->isVoidTy())
        rep->takeName(user);
      rep->copyIRFlags(user);
      rep->setAttributes(user->getAttributes());
      rep->setCallingConv(user->getCallingConv());
      rep->setTailCallKind(user->getTailCallKind());
      rep->setDebugLoc(gutils->getNewFromOriginal(user->getDebugLoc()));
      assert(user);
      SmallVector<ExtractValueInst *, 4> torep;
      for (auto u : user->users()) {
        assert(u);
        if (auto ei = dyn_cast<ExtractValueInst>(u)) {
          torep.push_back(ei);
        }
      }
      for (auto ei : torep) {
        ei->replaceAllUsesWith(rep);
        ei->eraseFromParent();
      }
      if (user->getParent()->getParent() == gutils->newFunc)
        gutils->erase(user);
      else
        user->eraseFromParent();
    } else {
      user->setCalledFunction(NewF);
    }
  }
  PPC.AlwaysInline(NewF);
  auto Arch = llvm::Triple(NewF->getParent()->getTargetTriple()).getArch();
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(NewF, /*mem2reg*/ true);

  AugmentedCachedFunctions.find(tup)->second.fn = NewF;
  if ((recursive && nonRecursiveUse) || (omp && !noTape))
    AugmentedCachedFunctions.find(tup)->second.tapeType = tapeType;
  AugmentedCachedFunctions.find(tup)->second.isComplete = true;

  for (auto pair : gfnusers) {
    auto GV = pair.first;
    GV->setName("_tmp");
    auto R = gutils->GetOrCreateShadowFunction(
        context, *this, TLI, TA, todiff, pair.second, width, gutils->AtomicAdd);
    SmallVector<std::pair<ConstantExpr *, bool>, 1> users;
    GV->replaceAllUsesWith(ConstantExpr::getPointerCast(R, GV->getType()));
    GV->eraseFromParent();
  }

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*gutils->newFunc, PA);
  }

  Function *tempFunc = gutils->newFunc;
  delete gutils;
  tempFunc->eraseFromParent();

  // Do not run post processing optimizations if the body of an openmp
  // parallel so the adjointgenerator can successfully extract the allocation
  // and frees and hoist them into the parent. Optimizing before then may
  // make the IR different to traverse, and thus impossible to find the allocs.
  if (PostOpt && !omp)
    PPC.optimizeIntermediate(NewF);
  if (EnzymePrint)
    llvm::errs() << *NewF << "\n";
  return AugmentedCachedFunctions.find(tup)->second;
}

void createTerminator(DiffeGradientUtils *gutils, BasicBlock *oBB,
                      DIFFE_TYPE retType, ReturnType retVal) {
  TypeResults &TR = gutils->TR;
  ReturnInst *inst = dyn_cast<ReturnInst>(oBB->getTerminator());
  // In forward mode we only need to update the return value
  if (inst == nullptr)
    return;

  ReturnInst *newInst = cast<ReturnInst>(gutils->getNewFromOriginal(inst));
  BasicBlock *nBB = newInst->getParent();
  assert(nBB);
  IRBuilder<> nBuilder(nBB);
  nBuilder.setFastMathFlags(getFast());

  SmallVector<Value *, 2> retargs;

  Value *toret = UndefValue::get(gutils->newFunc->getReturnType());

  Value *invertedPtr = nullptr;

  if (retType != DIFFE_TYPE::CONSTANT) {
    auto ret = inst->getOperand(0);
    Type *rt = ret->getType();
    while (auto AT = dyn_cast<ArrayType>(rt))
      rt = AT->getElementType();
    bool floatLike = rt->isFPOrFPVectorTy();
    if (!floatLike && TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      if (gutils->isConstantValue(ret)) {
        if (!EnzymeRuntimeActivityCheck &&
            TR.query(ret)[{-1}].isPossiblePointer()) {
          if (!isa<UndefValue>(ret) && !isa<ConstantPointerNull>(ret)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << *inst
               << " const val: " << *ret;
            if (CustomErrorHandler)
              invertedPtr = unwrap(CustomErrorHandler(
                  str.c_str(), wrap(inst), ErrorType::MixedActivityError,
                  gutils, wrap(ret), wrap(&nBuilder)));
            else
              EmitWarning("MixedActivityError", *inst, ss.str());
          }
        }
      }
    }
  }

  switch (retVal) {
  case ReturnType::Return: {
    auto ret = inst->getOperand(0);

    Type *rt = ret->getType();
    while (auto AT = dyn_cast<ArrayType>(rt))
      rt = AT->getElementType();
    bool floatLike = rt->isFPOrFPVectorTy();

    if (retType == DIFFE_TYPE::CONSTANT) {
      toret = gutils->getNewFromOriginal(ret);
    } else if (!floatLike &&
               TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = invertedPtr ? invertedPtr : gutils->invertPointerM(ret, nBuilder);
    } else if (!gutils->isConstantValue(ret)) {
      assert(!invertedPtr);
      toret = gutils->diffe(ret, nBuilder);
    } else {
      toret = invertedPtr
                  ? invertedPtr
                  : gutils->invertPointerM(ret, nBuilder, /*nullInit*/ true);
    }

    break;
  }
  case ReturnType::TwoReturns: {
    if (retType == DIFFE_TYPE::CONSTANT)
      assert(false && "Invalid return type");
    auto ret = inst->getOperand(0);

    Type *rt = ret->getType();
    while (auto AT = dyn_cast<ArrayType>(rt))
      rt = AT->getElementType();
    bool floatLike = rt->isFPOrFPVectorTy();

    toret =
        nBuilder.CreateInsertValue(toret, gutils->getNewFromOriginal(ret), 0);

    if (!floatLike && TR.getReturnAnalysis().Inner0().isPossiblePointer()) {
      toret = nBuilder.CreateInsertValue(
          toret,
          invertedPtr ? invertedPtr : gutils->invertPointerM(ret, nBuilder), 1);
    } else if (!gutils->isConstantValue(ret)) {
      assert(!invertedPtr);
      toret =
          nBuilder.CreateInsertValue(toret, gutils->diffe(ret, nBuilder), 1);
    } else {
      toret = nBuilder.CreateInsertValue(
          toret,
          invertedPtr
              ? invertedPtr
              : gutils->invertPointerM(ret, nBuilder, /*nullInit*/ true),
          1);
    }
    break;
  }
  case ReturnType::Void: {
    gutils->erase(gutils->getNewFromOriginal(inst));
    nBuilder.CreateRetVoid();
    return;
  }
  default: {
    llvm::errs() << "Invalid return type: " << to_string(retVal)
                 << "for function: \n"
                 << gutils->newFunc << "\n";
    assert(false && "Invalid return type for function");
    return;
  }
  }

  gutils->erase(newInst);
  nBuilder.CreateRet(toret);
  return;
}

Value *selectByWidth(IRBuilder<> &B, DiffeGradientUtils *gutils, Value *cond,
                     Value *tval, Value *fval) {
  auto width = gutils->getWidth();
  if (width == 1) {
    return B.CreateSelect(cond, tval, fval);
  }
  Value *res = UndefValue::get(tval->getType());

  for (unsigned int i = 0; i < width; ++i) {
    auto ntval = GradientUtils::extractMeta(B, tval, i);
    auto nfval = GradientUtils::extractMeta(B, fval, i);
    res = B.CreateInsertValue(res, B.CreateSelect(cond, ntval, nfval), {i});
  }
  return res;
}

void createInvertedTerminator(DiffeGradientUtils *gutils,
                              ArrayRef<DIFFE_TYPE> argTypes, BasicBlock *oBB,
                              AllocaInst *retAlloca, AllocaInst *dretAlloca,
                              unsigned extraArgs, DIFFE_TYPE retType) {
  LoopContext loopContext;
  BasicBlock *BB = cast<BasicBlock>(gutils->getNewFromOriginal(oBB));
  bool inLoop = gutils->getContext(BB, loopContext);
  BasicBlock *BB2 = gutils->reverseBlocks[BB].back();
  assert(BB2);
  IRBuilder<> Builder(BB2);
  Builder.setFastMathFlags(getFast());

  std::map<BasicBlock *, SmallVector<BasicBlock *, 4>> targetToPreds;
  for (auto pred : predecessors(BB)) {
    targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].push_back(pred);
  }

  if (targetToPreds.size() == 0) {
    SmallVector<Value *, 4> retargs;

    if (retAlloca) {
      auto result = Builder.CreateLoad(retAlloca->getAllocatedType(), retAlloca,
                                       "retreload");
      // TODO reintroduce invariant load/group
      // result->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(retAlloca->getContext(), {}));
      retargs.push_back(result);
    }

    if (dretAlloca) {
      auto result = Builder.CreateLoad(dretAlloca->getAllocatedType(),
                                       dretAlloca, "dretreload");
      // TODO reintroduce invariant load/group
      // result->setMetadata(LLVMContext::MD_invariant_load,
      // MDNode::get(dretAlloca->getContext(), {}));
      retargs.push_back(result);
    }

    for (auto &I : gutils->oldFunc->args()) {
      if (!gutils->isConstantValue(&I) &&
          argTypes[I.getArgNo()] == DIFFE_TYPE::OUT_DIFF) {
        retargs.push_back(gutils->diffe(&I, Builder));
      }
    }

    if (gutils->newFunc->getReturnType()->isVoidTy()) {
      assert(retargs.size() == 0);
      Builder.CreateRetVoid();
      return;
    }

    Value *toret = UndefValue::get(gutils->newFunc->getReturnType());
    for (unsigned i = 0; i < retargs.size(); ++i) {
      unsigned idx[] = {i};
      toret = Builder.CreateInsertValue(toret, retargs[i], idx);
    }
    Builder.CreateRet(toret);
    return;
  }

  // PHINodes to replace that will contain true iff the predecessor was given
  // basicblock
  std::map<BasicBlock *, PHINode *> replacePHIs;
  SmallVector<SelectInst *, 4> selects;

  IRBuilder<> phibuilder(BB2);
  bool setphi = false;

  // Ensure phi values have their derivatives propagated
  for (auto I = oBB->begin(), E = oBB->end(); I != E; ++I) {
    PHINode *orig = dyn_cast<PHINode>(&*I);
    if (orig == nullptr)
      break;
    if (gutils->isConstantInstruction(orig))
      continue;

    size_t size = 1;
    if (orig->getType()->isSized())
      size = (gutils->newFunc->getParent()->getDataLayout().getTypeSizeInBits(
                  orig->getType()) +
              7) /
             8;

    auto PNtypeT = gutils->TR.query(orig);
    auto PNtype = PNtypeT[{-1}];

    // TODO remove explicit type check and only use PNtype
    if (!gutils->TR.anyFloat(orig, /*anythingIsFloat*/ false) ||
        orig->getType()->isPointerTy())
      continue;

    Type *PNfloatType = PNtype.isFloat();
    if (!PNfloatType) {
      // Try to use the 0-th elem for all elems
      PNtype = PNtypeT[{0}];
      bool legal = true;
      for (size_t i = 1; i < size; i++) {
        if (!PNtypeT[{(int)i}].isFloat())
          continue;
        PNtype.checkedOrIn(PNtypeT[{(int)i}], /*pointerIntSame*/ true, legal);
        if (!legal) {
          break;
        }
      }
      if (legal) {
        PNfloatType = PNtype.isFloat();
        if (!PNfloatType) {
          if (looseTypeAnalysis) {
            if (orig->getType()->isFPOrFPVectorTy())
              PNfloatType = orig->getType()->getScalarType();
            if (orig->getType()->isIntOrIntVectorTy())
              continue;
          }
        }
      }
    }
    if (!PNfloatType) {
      std::string str;
      raw_string_ostream ss(str);
      ss << "Cannot deduce type of phi " << *orig << PNtypeT.str()
         << " sz: " << size << "\n";
      EmitNoTypeError(ss.str(), *orig, gutils, Builder);
      continue;
    }

    auto prediff = gutils->diffe(orig, Builder);

    bool handled = false;

    SmallVector<Instruction *, 4> activeUses;
    for (auto u : orig->users()) {
      if (!gutils->isConstantInstruction(cast<Instruction>(u)))
        activeUses.push_back(cast<Instruction>(u));
      else if (retType == DIFFE_TYPE::OUT_DIFF && isa<ReturnInst>(u))
        activeUses.push_back(cast<Instruction>(u));
    }
    if (activeUses.size() == 1 && inLoop &&
        gutils->getNewFromOriginal(orig->getParent()) == loopContext.header &&
        loopContext.exitBlocks.size() == 1) {
      SmallVector<BasicBlock *, 1> Latches;
      gutils->OrigLI->getLoopFor(orig->getParent())->getLoopLatches(Latches);
      bool allIncoming = true;
      for (auto Latch : Latches) {
        if (activeUses[0] != orig->getIncomingValueForBlock(Latch)) {
          allIncoming = false;
          break;
        }
      }
      if (allIncoming) {
        if (auto SI = dyn_cast<SelectInst>(activeUses[0])) {
          for (int i = 0; i < 2; i++) {
            if (SI->getOperand(i + 1) == orig) {
              auto oval = orig->getIncomingValueForBlock(
                  gutils->getOriginalFromNew(loopContext.preheader));
              BasicBlock *pred = loopContext.preheader;
              if (replacePHIs.find(pred) == replacePHIs.end()) {
                replacePHIs[pred] = Builder.CreatePHI(
                    Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
                if (!setphi) {
                  phibuilder.SetInsertPoint(replacePHIs[pred]);
                  setphi = true;
                }
              }

              auto ddiff = gutils->diffe(SI, Builder);
              gutils->setDiffe(
                  SI,
                  selectByWidth(Builder, gutils, replacePHIs[pred],
                                Constant::getNullValue(prediff->getType()),
                                ddiff),
                  Builder);
              handled = true;
              if (!gutils->isConstantValue(oval)) {
                BasicBlock *REB =
                    gutils->reverseBlocks[*loopContext.exitBlocks.begin()]
                        .back();
                IRBuilder<> EB(REB);
                if (REB->getTerminator())
                  EB.SetInsertPoint(REB->getTerminator());

                auto index = gutils->getOrInsertConditionalIndex(
                    gutils->getNewFromOriginal(SI->getOperand(0)), loopContext,
                    i == 1);
                Value *sdif = selectByWidth(
                    Builder, gutils,
                    Builder.CreateICmpEQ(
                        gutils->lookupM(index, EB),
                        Constant::getNullValue(index->getType())),
                    ddiff, Constant::getNullValue(ddiff->getType()));

                auto dif =
                    selectByWidth(Builder, gutils, replacePHIs[pred], sdif,
                                  Constant::getNullValue(prediff->getType()));
                auto addedSelects =
                    gutils->addToDiffe(oval, dif, Builder, PNfloatType);

                for (auto select : addedSelects)
                  selects.push_back(select);
              }
              break;
            }
          }
        }
        if (auto BO = dyn_cast<BinaryOperator>(activeUses[0])) {

          if (BO->getOpcode() == Instruction::FDiv &&
              BO->getOperand(0) == orig) {

            auto oval = orig->getIncomingValueForBlock(
                gutils->getOriginalFromNew(loopContext.preheader));
            BasicBlock *pred = loopContext.preheader;
            if (replacePHIs.find(pred) == replacePHIs.end()) {
              replacePHIs[pred] = Builder.CreatePHI(
                  Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
              if (!setphi) {
                phibuilder.SetInsertPoint(replacePHIs[pred]);
                setphi = true;
              }
            }

            auto ddiff = gutils->diffe(BO, Builder);
            gutils->setDiffe(
                BO,
                selectByWidth(Builder, gutils, replacePHIs[pred],
                              Constant::getNullValue(prediff->getType()),
                              ddiff),
                Builder);
            handled = true;

            if (!gutils->isConstantValue(oval)) {

              BasicBlock *REB =
                  gutils->reverseBlocks[*loopContext.exitBlocks.begin()].back();
              IRBuilder<> EB(REB);
              if (REB->getTerminator())
                EB.SetInsertPoint(REB->getTerminator());

              auto product = gutils->getOrInsertTotalMultiplicativeProduct(
                  gutils->getNewFromOriginal(BO->getOperand(1)), loopContext);

              auto dif = selectByWidth(
                  Builder, gutils, replacePHIs[pred],
                  Builder.CreateFDiv(ddiff, gutils->lookupM(product, EB)),
                  Constant::getNullValue(prediff->getType()));
              auto addedSelects =
                  gutils->addToDiffe(oval, dif, Builder, PNfloatType);

              for (auto select : addedSelects)
                selects.push_back(select);
            }
          }
        }
      }
    }

    if (!handled) {
      gutils->setDiffe(
          orig, Constant::getNullValue(gutils->getShadowType(orig->getType())),
          Builder);

      for (BasicBlock *opred : predecessors(oBB)) {
        auto oval = orig->getIncomingValueForBlock(opred);
        if (gutils->isConstantValue(oval)) {
          continue;
        }

        if (orig->getNumIncomingValues() == 1) {
          gutils->addToDiffe(oval, prediff, Builder, PNfloatType);
        } else {
          BasicBlock *pred =
              cast<BasicBlock>(gutils->getNewFromOriginal(opred));
          if (replacePHIs.find(pred) == replacePHIs.end()) {
            replacePHIs[pred] = Builder.CreatePHI(
                Type::getInt1Ty(pred->getContext()), 1, "replacePHI");
            if (!setphi) {
              phibuilder.SetInsertPoint(replacePHIs[pred]);
              setphi = true;
            }
          }
          auto dif = selectByWidth(Builder, gutils, replacePHIs[pred], prediff,
                                   Constant::getNullValue(prediff->getType()));
          auto addedSelects =
              gutils->addToDiffe(oval, dif, Builder, PNfloatType);

          for (auto select : addedSelects)
            selects.push_back(select);
        }
      }
    }
  }
  if (!setphi) {
    phibuilder.SetInsertPoint(Builder.GetInsertBlock(),
                              Builder.GetInsertPoint());
  }

  if (inLoop && BB == loopContext.header) {
    std::map<BasicBlock *, SmallVector<BasicBlock *, 4>> targetToPreds;
    for (auto pred : predecessors(BB)) {
      if (pred == loopContext.preheader)
        continue;
      targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].push_back(pred);
    }

    assert(targetToPreds.size() &&
           "only loops with one backedge are presently supported");

    Value *av = phibuilder.CreateLoad(loopContext.var->getType(),
                                      loopContext.antivaralloc);
    Value *phi =
        phibuilder.CreateICmpEQ(av, Constant::getNullValue(av->getType()));
    Value *nphi = phibuilder.CreateNot(phi);

    for (auto pair : replacePHIs) {
      Value *replaceWith = nullptr;

      if (pair.first == loopContext.preheader) {
        replaceWith = phi;
      } else {
        replaceWith = nphi;
      }

      pair.second->replaceAllUsesWith(replaceWith);
      pair.second->eraseFromParent();
    }
    BB2 = gutils->reverseBlocks[BB].back();
    Builder.SetInsertPoint(BB2);

    Builder.CreateCondBr(
        phi, gutils->getReverseOrLatchMerge(loopContext.preheader, BB),
        targetToPreds.begin()->first);

  } else {
    std::map<BasicBlock *, std::vector<std::pair<BasicBlock *, BasicBlock *>>>
        phiTargetToPreds;
    for (auto pair : replacePHIs) {
      phiTargetToPreds[pair.first].emplace_back(pair.first, BB);
    }
    BasicBlock *fakeTarget = nullptr;
    for (auto pred : predecessors(BB)) {
      if (phiTargetToPreds.find(pred) != phiTargetToPreds.end())
        continue;
      if (fakeTarget == nullptr)
        fakeTarget = pred;
      phiTargetToPreds[fakeTarget].emplace_back(pred, BB);
    }
    gutils->branchToCorrespondingTarget(BB, phibuilder, phiTargetToPreds,
                                        &replacePHIs);

    std::map<BasicBlock *, std::vector<std::pair<BasicBlock *, BasicBlock *>>>
        targetToPreds;
    for (auto pred : predecessors(BB)) {
      targetToPreds[gutils->getReverseOrLatchMerge(pred, BB)].emplace_back(pred,
                                                                           BB);
    }
    BB2 = gutils->reverseBlocks[BB].back();
    Builder.SetInsertPoint(BB2);
    gutils->branchToCorrespondingTarget(BB, Builder, targetToPreds);
  }

  // Optimize select of not to just be a select with operands switched
  for (SelectInst *select : selects) {
    if (BinaryOperator *bo = dyn_cast<BinaryOperator>(select->getCondition())) {
      if (bo->getOpcode() == BinaryOperator::Xor) {
        if (isa<ConstantInt>(bo->getOperand(0)) &&
            cast<ConstantInt>(bo->getOperand(0))->isOne()) {
          select->setCondition(bo->getOperand(1));
          auto tmp = select->getTrueValue();
          select->setTrueValue(select->getFalseValue());
          select->setFalseValue(tmp);
          if (bo->getNumUses() == 0)
            bo->eraseFromParent();
        } else if (isa<ConstantInt>(bo->getOperand(1)) &&
                   cast<ConstantInt>(bo->getOperand(1))->isOne()) {
          select->setCondition(bo->getOperand(0));
          auto tmp = select->getTrueValue();
          select->setTrueValue(select->getFalseValue());
          select->setFalseValue(tmp);
          if (bo->getNumUses() == 0)
            bo->eraseFromParent();
        }
      }
    }
  }
}

Function *EnzymeLogic::CreatePrimalAndGradient(
    RequestContext context, const ReverseCacheKey &&key, TypeAnalysis &TA,
    const AugmentedReturn *augmenteddata, bool omp) {

  TimeTraceScope timeScope("CreatePrimalAndGradient", key.todiff->getName());

  assert(key.mode == DerivativeMode::ReverseModeCombined ||
         key.mode == DerivativeMode::ReverseModeGradient);

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(key.typeInfo, key.todiff);

  if (key.retType != DIFFE_TYPE::CONSTANT)
    assert(!key.todiff->getReturnType()->isVoidTy());

  if (!isMemFreeLibMFunction(getFuncName(key.todiff)))
    assert(key.overwritten_args.size() == key.todiff->arg_size());

  Function *prevFunction = nullptr;
  if (ReverseCachedFunctions.find(key) != ReverseCachedFunctions.end()) {
    prevFunction = ReverseCachedFunctions.find(key)->second;
    if (!hasMetadata(prevFunction, "enzyme_placeholder"))
      return prevFunction;
    if (augmenteddata && !augmenteddata->isComplete)
      return prevFunction;
  }

  if (key.returnUsed)
    assert(key.mode == DerivativeMode::ReverseModeCombined);

  TargetLibraryInfo &TLI =
      PPC.FAM.getResult<TargetLibraryAnalysis>(*key.todiff);

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for (auto v : key.constant_args) {
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (hasMetadata(key.todiff, "enzyme_gradient")) {
    std::set<llvm::Type *> seen;
#ifndef NDEBUG
    DIFFE_TYPE subretType = whatType(key.todiff->getReturnType(),
                                     DerivativeMode::ReverseModeGradient,
                                     /*intAreConstant*/ false, seen);
    if (key.todiff->getReturnType()->isVoidTy() ||
        key.todiff->getReturnType()->isEmptyTy())
      subretType = DIFFE_TYPE::CONSTANT;
    assert(subretType == key.retType);
#endif

    if (key.mode == DerivativeMode::ReverseModeCombined) {
      auto res = getDefaultFunctionTypeForGradient(
          key.todiff->getFunctionType(),
          /*retType*/ key.retType, key.constant_args);

      Type *FRetTy =
          res.second.empty()
              ? Type::getVoidTy(key.todiff->getContext())
              : StructType::get(key.todiff->getContext(), {res.second});

      FunctionType *FTy = FunctionType::get(
          FRetTy, res.first, key.todiff->getFunctionType()->isVarArg());

      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixgradient_" + key.todiff->getName(), key.todiff->getParent());

      size_t argnum = 0;
      for (Argument &Arg : NewF->args()) {
        Arg.setName("arg" + Twine(argnum));
        ++argnum;
      }

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);

      auto &aug = CreateAugmentedPrimal(
          context, key.todiff, key.retType, key.constant_args, TA,
          key.returnUsed, key.shadowReturnUsed, key.typeInfo,
          key.overwritten_args,
          /*forceAnonymousTape*/ false, key.width, key.AtomicAdd, omp);

      SmallVector<Value *, 4> fwdargs;
      for (auto &a : NewF->args())
        fwdargs.push_back(&a);
      if (key.retType == DIFFE_TYPE::OUT_DIFF)
        fwdargs.pop_back();
      auto cal = bb.CreateCall(aug.fn, fwdargs);
      cal->setCallingConv(aug.fn->getCallingConv());

      llvm::Value *tape = nullptr;

      if (aug.returns.find(AugmentedStruct::Tape) != aug.returns.end()) {
        auto tapeIdx = aug.returns.find(AugmentedStruct::Tape)->second;
        tape = (tapeIdx == -1) ? cal : bb.CreateExtractValue(cal, tapeIdx);
        if (tape->getType()->isEmptyTy())
          tape = UndefValue::get(tape->getType());
      }

      if (aug.tapeType) {
        assert(tape);
        auto tapep = bb.CreatePointerCast(
            tape, PointerType::get(
                      aug.tapeType,
                      cast<PointerType>(tape->getType())->getAddressSpace()));
        auto truetape = bb.CreateLoad(aug.tapeType, tapep, "tapeld");
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        if (key.freeMemory) {
          auto size = NewF->getParent()->getDataLayout().getTypeAllocSizeInBits(
              aug.tapeType);
          if (size != 0) {
            CreateDealloc(bb, tape);
          }
        }
        tape = truetape;
      }

      auto revfn = CreatePrimalAndGradient(
          context,
          (ReverseCacheKey){.todiff = key.todiff,
                            .retType = key.retType,
                            .constant_args = key.constant_args,
                            .overwritten_args = key.overwritten_args,
                            .returnUsed = false,
                            .shadowReturnUsed = false,
                            .mode = DerivativeMode::ReverseModeGradient,
                            .width = key.width,
                            .freeMemory = key.freeMemory,
                            .AtomicAdd = key.AtomicAdd,
                            .additionalType = tape ? tape->getType() : nullptr,
                            .forceAnonymousTape = key.forceAnonymousTape,
                            .typeInfo = key.typeInfo},
          TA, &aug, omp);

      SmallVector<Value *, 4> revargs;
      for (auto &a : NewF->args()) {
        revargs.push_back(&a);
      }
      if (tape) {
        revargs.push_back(tape);
      }
      auto revcal = bb.CreateCall(revfn, revargs);
      revcal->setCallingConv(revfn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy()) {
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      } else if (NewF->getReturnType()->isVoidTy()) {
        bb.CreateRetVoid();
      } else {
        bb.CreateRet(revcal);
      }
      assert(!key.returnUsed);

      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, NewF)
          ->second;
    }

    auto md = key.todiff->getMetadata("enzyme_gradient");
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *key.todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown gradient for noninvertible function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2->getNumOperands() == 1);
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    if (hasconstant) {
      EmitWarning("NoCustom", *key.todiff,
                  "Massaging provided custom reverse pass");
      SmallVector<Type *, 3> dupargs;
      std::vector<DIFFE_TYPE> next_constant_args(key.constant_args);
      {
        auto OFT = key.todiff->getFunctionType();
        for (size_t act_idx = 0; act_idx < key.constant_args.size();
             act_idx++) {
          dupargs.push_back(OFT->getParamType(act_idx));
          switch (key.constant_args[act_idx]) {
          case DIFFE_TYPE::OUT_DIFF:
            break;
          case DIFFE_TYPE::DUP_ARG:
          case DIFFE_TYPE::DUP_NONEED:
            dupargs.push_back(OFT->getParamType(act_idx));
            break;
          case DIFFE_TYPE::CONSTANT:
            if (!OFT->getParamType(act_idx)->isFPOrFPVectorTy()) {
              next_constant_args[act_idx] = DIFFE_TYPE::DUP_ARG;
            } else {
              next_constant_args[act_idx] = DIFFE_TYPE::OUT_DIFF;
            }
            break;
          }
        }
      }

      auto revfn = CreatePrimalAndGradient(
          context,
          (ReverseCacheKey){.todiff = key.todiff,
                            .retType = key.retType,
                            .constant_args = next_constant_args,
                            .overwritten_args = key.overwritten_args,
                            .returnUsed = key.returnUsed,
                            .shadowReturnUsed = false,
                            .mode = DerivativeMode::ReverseModeGradient,
                            .width = key.width,
                            .freeMemory = key.freeMemory,
                            .AtomicAdd = key.AtomicAdd,
                            .additionalType = nullptr,
                            .forceAnonymousTape = key.forceAnonymousTape,
                            .typeInfo = key.typeInfo},
          TA, augmenteddata, omp);

      {
        auto arg = revfn->arg_begin();
        for (auto cidx : next_constant_args) {
          arg++;
          if (cidx == DIFFE_TYPE::DUP_ARG || cidx == DIFFE_TYPE::DUP_NONEED)
            arg++;
        }
        while (arg != revfn->arg_end()) {
          dupargs.push_back(arg->getType());
          arg++;
        }
      }

      FunctionType *FTy =
          FunctionType::get(revfn->getReturnType(), dupargs,
                            revfn->getFunctionType()->isVarArg());
      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixgradient_" + key.todiff->getName(), key.todiff->getParent());

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto arg = NewF->arg_begin();
      SmallVector<Value *, 3> revargs;
      size_t act_idx = 0;
      while (act_idx != key.constant_args.size()) {
        arg->setName("arg" + Twine(act_idx));
        revargs.push_back(arg);
        switch (key.constant_args[act_idx]) {
        case DIFFE_TYPE::OUT_DIFF:
          break;
        case DIFFE_TYPE::DUP_ARG:
        case DIFFE_TYPE::DUP_NONEED:
          arg++;
          arg->setName("arg" + Twine(act_idx) + "'");
          revargs.push_back(arg);
          break;
        case DIFFE_TYPE::CONSTANT:
          if (next_constant_args[act_idx] != DIFFE_TYPE::OUT_DIFF) {
            revargs.push_back(arg);
          }
          break;
        }
        arg++;
        act_idx++;
      }
      size_t pa = 0;
      while (arg != NewF->arg_end()) {
        revargs.push_back(arg);
        arg->setName("postarg" + Twine(pa));
        pa++;
        arg++;
      }
      auto cal = bb.CreateCall(revfn, revargs);
      cal->setCallingConv(revfn->getCallingConv());

      if (NewF->getReturnType()->isEmptyTy())
        bb.CreateRet(UndefValue::get(NewF->getReturnType()));
      else if (NewF->getReturnType()->isVoidTy())
        bb.CreateRetVoid();
      else
        bb.CreateRet(cal);

      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, NewF)
          ->second;
    }

    if (!key.returnUsed && key.freeMemory) {
      auto res =
          getDefaultFunctionTypeForGradient(key.todiff->getFunctionType(),
                                            /*retType*/ key.retType);
      assert(augmenteddata);
      bool badDiffRet = false;
      bool hasTape = true;
      if (foundcalled->arg_size() == res.first.size() + 1 /*tape*/) {
        auto lastarg = foundcalled->arg_end();
        lastarg--;
        res.first.push_back(lastarg->getType());
        if (key.retType == DIFFE_TYPE::OUT_DIFF) {
          lastarg--;
          if (lastarg->getType() != key.todiff->getReturnType())
            badDiffRet = true;
        }
      } else if (foundcalled->arg_size() == res.first.size()) {
        if (key.retType == DIFFE_TYPE::OUT_DIFF) {
          auto lastarg = foundcalled->arg_end();
          lastarg--;
          if (lastarg->getType() != key.todiff->getReturnType())
            badDiffRet = true;
        }
        hasTape = false;
        // res.first.push_back(StructType::get(todiff->getContext(), {}));
      } else {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << "Bad function type of custom reverse pass for function "
           << key.todiff->getName() << " of type "
           << *key.todiff->getFunctionType() << "\n";
        ss << "  expected gradient function to have argument types [";
        bool seen = false;
        for (auto a : res.first) {
          if (seen)
            ss << ", ";
          seen = true;
          ss << *a;
        }
        ss << "]\n";
        ss << "  Instead found " << foundcalled->getName() << " of type "
           << *foundcalled->getFunctionType() << "\n";
        if (context.req) {
          ss << " at context: " << *context.req;
        } else {
          ss << *key.todiff << "\n";
        }
        if (!EmitNoDerivativeError(ss.str(), key.todiff, context)) {
          assert(0 && "bad type for custom gradient");
          llvm_unreachable("bad type for custom gradient");
        }
      }

      auto st = dyn_cast<StructType>(foundcalled->getReturnType());
      bool wrongRet =
          st == nullptr && !foundcalled->getReturnType()->isVoidTy();
      if (wrongRet || badDiffRet) {
        // if (wrongRet || !hasTape) {
        Type *FRetTy =
            res.second.empty()
                ? Type::getVoidTy(key.todiff->getContext())
                : StructType::get(key.todiff->getContext(), {res.second});

        FunctionType *FTy = FunctionType::get(
            FRetTy, res.first, key.todiff->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(
            FTy, Function::LinkageTypes::InternalLinkage,
            "fixgradient_" + key.todiff->getName(), key.todiff->getParent());
        NewF->setAttributes(foundcalled->getAttributes());
        if (NewF->hasFnAttribute(Attribute::NoInline)) {
          NewF->removeFnAttr(Attribute::NoInline);
        }
        if (NewF->hasFnAttribute(Attribute::OptimizeNone)) {
          NewF->removeFnAttr(Attribute::OptimizeNone);
        }
        size_t argnum = 0;
        for (Argument &Arg : NewF->args()) {
          if (Arg.hasAttribute(Attribute::Returned))
            Arg.removeAttr(Attribute::Returned);
          if (Arg.hasAttribute(Attribute::StructRet))
            Arg.removeAttr(Attribute::StructRet);
          Arg.setName("arg" + Twine(argnum));
          ++argnum;
        }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder<> bb(BB);
        SmallVector<Value *, 4> args;
        for (auto &a : NewF->args())
          args.push_back(&a);
        if (badDiffRet) {
          auto idx = hasTape ? (args.size() - 2) : (args.size() - 1);
          Type *T = (foundcalled->arg_begin() + idx)->getType();

          auto AI = bb.CreateAlloca(T);
          bb.CreateStore(args[idx],
                         bb.CreatePointerCast(
                             AI, PointerType::getUnqual(args[idx]->getType())));
          Value *vres = bb.CreateLoad(T, AI);
          args[idx] = vres;
        }
        // if (!hasTape) {
        //  args.pop_back();
        //}
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());
        Value *val = cal;
        if (wrongRet) {
          auto ut = UndefValue::get(NewF->getReturnType());
          if (val->getType()->isEmptyTy() && res.second.size() == 0) {
            val = ut;
          } else if (res.second.size() == 1 &&
                     res.second[0] == val->getType()) {
            val = bb.CreateInsertValue(ut, cal, {0u});
          } else {
            llvm::errs() << *foundcalled << "\n";
            assert(0 && "illegal type for reverse");
            llvm_unreachable("illegal type for reverse");
          }
        }
        if (val->getType()->isVoidTy())
          bb.CreateRetVoid();
        else
          bb.CreateRet(val);
        foundcalled = NewF;
      }
      return insert_or_assign2<ReverseCacheKey, Function *>(
                 ReverseCachedFunctions, key, foundcalled)
          ->second;
    }

    EmitWarning("NoCustom", *key.todiff,
                "Not using provided custom reverse pass as require either "
                "return or non-constant");
  }

  if (augmenteddata && augmenteddata->constant_args != key.constant_args) {
    llvm::errs() << " sz: " << augmenteddata->constant_args.size() << "  "
                 << key.constant_args.size() << "\n";
    for (size_t i = 0; i < key.constant_args.size(); ++i) {
      llvm::errs() << " i: " << i << " "
                   << to_string(augmenteddata->constant_args[i]) << "  "
                   << to_string(key.constant_args[i]) << "\n";
    }
    assert(augmenteddata->constant_args.size() == key.constant_args.size());
    assert(augmenteddata->constant_args == key.constant_args);
  }

  ReturnType retVal =
      key.returnUsed ? (key.shadowReturnUsed ? ReturnType::ArgsWithTwoReturns
                                             : ReturnType::ArgsWithReturn)
                     : (key.shadowReturnUsed ? ReturnType::ArgsWithReturn
                                             : ReturnType::Args);

  bool diffeReturnArg = key.retType == DIFFE_TYPE::OUT_DIFF;

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(
      *this, key.mode, key.width, key.todiff, TLI, TA, oldTypeInfo, key.retType,
      augmenteddata ? augmenteddata->shadowReturnUsed : key.shadowReturnUsed,
      diffeReturnArg, key.constant_args, retVal, key.additionalType, omp);

  gutils->AtomicAdd = key.AtomicAdd;
  gutils->FreeMemory = key.freeMemory;
  insert_or_assign2<ReverseCacheKey, Function *>(ReverseCachedFunctions, key,
                                                 gutils->newFunc);

  if (key.todiff->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No reverse pass found for " + key.todiff->getName() << "\n";
    if (context.req) {
      ss << " at context: " << *context.req;
    } else {
      ss << *key.todiff << "\n";
    }
    BasicBlock *entry = &gutils->newFunc->getEntryBlock();
    cleanupInversionAllocs(gutils, entry);
    clearFunctionAttributes(gutils->newFunc);
    if (EmitNoDerivativeError(ss.str(), key.todiff, context)) {
      auto newFunc = gutils->newFunc;
      delete gutils;
      return newFunc;
    }
    llvm::errs() << "mod: " << *key.todiff->getParent() << "\n";
    llvm::errs() << *key.todiff << "\n";
    llvm_unreachable("attempting to differentiate function without definition");
  }

  if (augmenteddata && !augmenteddata->isComplete) {
    auto nf = gutils->newFunc;
    delete gutils;
    assert(!prevFunction);
    nf->setMetadata("enzyme_placeholder", MDTuple::get(nf->getContext(), {}));
    return nf;
  }

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  // Convert uncacheable args from the input function to the preprocessed
  // function
  const std::vector<bool> &_overwritten_argsPP = key.overwritten_args;

  gutils->forceActiveDetection();

  // requires is_value_needed_in_reverse, that needs unnecessaryValues
  // sets backwardsOnlyShadows, rematerializableAllocations, and
  // allocationsWithGuaranteedFrees
  gutils->computeGuaranteedFrees();
  CacheAnalysis CA(gutils->allocationsWithGuaranteedFree,
                   gutils->rematerializableAllocations, gutils->TR,
                   *gutils->OrigAA, gutils->oldFunc,
                   PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
                   *gutils->OrigLI, *gutils->OrigDT, TLI, guaranteedUnreachable,
                   _overwritten_argsPP, key.mode, omp);
  const std::map<CallInst *, const std::vector<bool>> overwritten_args_map =
      (augmenteddata) ? augmenteddata->overwritten_args_map
                      : CA.compute_overwritten_args_for_callsites();
  gutils->overwritten_args_map_ptr = &overwritten_args_map;

  const std::map<Instruction *, bool> can_modref_map =
      augmenteddata ? augmenteddata->can_modref_map
                    : CA.compute_uncacheable_load_map();
  gutils->can_modref_map = &can_modref_map;

  gutils->forceAugmentedReturns();

  std::map<std::pair<Instruction *, CacheType>, int> mapping;
  if (augmenteddata)
    mapping = augmenteddata->tapeIndices;

  auto getIndex = [&](Instruction *I, CacheType u, IRBuilder<> &B) -> unsigned {
    return gutils->getIndex(std::make_pair(I, u), mapping, B);
  };

  // requires is_value_needed_in_reverse, that needs unnecessaryValues
  // sets knownRecomputeHeuristic
  gutils->computeMinCache();

  SmallPtrSet<const Value *, 4> unnecessaryValues;
  SmallPtrSet<const Instruction *, 4> unnecessaryInstructions;
  calculateUnusedValuesInFunction(*gutils->oldFunc, unnecessaryValues,
                                  unnecessaryInstructions, key.returnUsed,
                                  key.mode, gutils, TLI, key.constant_args,
                                  guaranteedUnreachable);
  gutils->unnecessaryValuesP = &unnecessaryValues;

  SmallPtrSet<const Instruction *, 4> unnecessaryStores;
  calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                  unnecessaryInstructions, gutils, TLI);

  Value *additionalValue = nullptr;
  if (key.additionalType) {
    auto v = gutils->newFunc->arg_end();
    v--;
    additionalValue = v;
    assert(key.mode != DerivativeMode::ReverseModeCombined);
    assert(augmenteddata);

    // TODO VERIFY THIS
    if (augmenteddata->tapeType && (omp || key.forceAnonymousTape)) {
      IRBuilder<> BuilderZ(gutils->inversionAllocs);
      if (!augmenteddata->tapeType->isEmptyTy()) {
        auto tapep = BuilderZ.CreatePointerCast(
            additionalValue,
            PointerType::get(augmenteddata->tapeType,
                             cast<PointerType>(additionalValue->getType())
                                 ->getAddressSpace()));
        LoadInst *truetape =
            BuilderZ.CreateLoad(augmenteddata->tapeType, tapep, "truetape");
        truetape->setMetadata("enzyme_mustcache",
                              MDNode::get(truetape->getContext(), {}));

        if (!omp && gutils->FreeMemory) {
          CreateDealloc(BuilderZ, additionalValue);
        }
        additionalValue = truetape;
      } else {
        if (gutils->FreeMemory) {
          CreateDealloc(BuilderZ, additionalValue);
        }
        additionalValue = UndefValue::get(augmenteddata->tapeType);
      }
    }

    // TODO here finish up making recursive structs simply pass in i8*
    gutils->setTape(additionalValue);
  }

  Argument *differetval = nullptr;
  if (key.retType == DIFFE_TYPE::OUT_DIFF) {
    auto endarg = gutils->newFunc->arg_end();
    endarg--;
    if (key.additionalType)
      endarg--;
    differetval = endarg;

    if (!key.todiff->getReturnType()->isVoidTy()) {
      if (!(differetval->getType() ==
            gutils->getShadowType(key.todiff->getReturnType()))) {
        llvm::errs() << *gutils->oldFunc << "\n";
        llvm::errs() << *gutils->newFunc << "\n";
      }
      assert(differetval->getType() ==
             gutils->getShadowType(key.todiff->getReturnType()));
    }
  }

  // Explicitly handle all returns first to ensure that return instructions know
  // if they are used or not before
  //   processessing instructions
  std::map<ReturnInst *, StoreInst *> replacedReturns;
  llvm::AllocaInst *retAlloca = nullptr;
  llvm::AllocaInst *dretAlloca = nullptr;
  if (key.returnUsed) {
    retAlloca =
        IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
            .CreateAlloca(key.todiff->getReturnType(), nullptr, "toreturn");
  }
  if (key.shadowReturnUsed) {
    assert(key.retType == DIFFE_TYPE::DUP_ARG ||
           key.retType == DIFFE_TYPE::DUP_NONEED);
    assert(key.mode == DerivativeMode::ReverseModeCombined);
    dretAlloca =
        IRBuilder<>(&gutils->newFunc->getEntryBlock().front())
            .CreateAlloca(key.todiff->getReturnType(), nullptr, "dtoreturn");
  }
  if (key.mode == DerivativeMode::ReverseModeCombined ||
      key.mode == DerivativeMode::ReverseModeGradient) {
    for (BasicBlock &oBB : *gutils->oldFunc) {
      if (ReturnInst *orig = dyn_cast<ReturnInst>(oBB.getTerminator())) {
        ReturnInst *op = cast<ReturnInst>(gutils->getNewFromOriginal(orig));
        BasicBlock *BB = op->getParent();
        IRBuilder<> rb(op);
        rb.setFastMathFlags(getFast());

        if (retAlloca) {
          StoreInst *si = rb.CreateStore(
              gutils->getNewFromOriginal(orig->getReturnValue()), retAlloca);
          replacedReturns[orig] = si;
        }

        if (key.retType == DIFFE_TYPE::DUP_ARG ||
            key.retType == DIFFE_TYPE::DUP_NONEED) {
          if (dretAlloca) {
            rb.CreateStore(gutils->invertPointerM(orig->getReturnValue(), rb),
                           dretAlloca);
          }
        } else if (key.retType == DIFFE_TYPE::OUT_DIFF) {
          assert(orig->getReturnValue());
          assert(differetval);
          if (!gutils->isConstantValue(orig->getReturnValue())) {
            IRBuilder<> reverseB(gutils->reverseBlocks[BB].back());
            gutils->setDiffe(orig->getReturnValue(), differetval, reverseB);
          }
        } else {
          assert(dretAlloca == nullptr);
        }

        rb.CreateBr(gutils->reverseBlocks[BB].front());
        gutils->erase(op);
      }
    }
  }

  AdjointGenerator maker(key.mode, gutils, key.constant_args, key.retType,
                         getIndex, overwritten_args_map,
                         /*returnuses*/ nullptr, augmenteddata,
                         &replacedReturns, unnecessaryValues,
                         unnecessaryInstructions, unnecessaryStores,
                         guaranteedUnreachable, dretAlloca);

  for (BasicBlock &oBB : *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      auto newBB = cast<BasicBlock>(gutils->getNewFromOriginal(&oBB));
      SmallVector<BasicBlock *, 4> toRemove;
      if (key.mode != DerivativeMode::ReverseModeCombined) {
        if (auto II = dyn_cast<InvokeInst>(oBB.getTerminator())) {
          toRemove.push_back(cast<BasicBlock>(
              gutils->getNewFromOriginal(II->getNormalDest())));
        } else {
          for (auto next : successors(&oBB)) {
            auto sucBB = cast<BasicBlock>(gutils->getNewFromOriginal(next));
            toRemove.push_back(sucBB);
          }
        }
      }

      for (auto sucBB : toRemove) {
        if (sucBB->empty() || !isa<PHINode>(sucBB->begin()))
          continue;

        SmallVector<PHINode *, 2> phis;
        for (PHINode &Phi : sucBB->phis()) {
          phis.push_back(&Phi);
        }
        for (PHINode *Phi : phis) {
          unsigned NumPreds = Phi->getNumIncomingValues();
          if (NumPreds == 0)
            continue;
          Phi->removeIncomingValue(newBB);
        }
      }

      SmallVector<Instruction *, 2> toerase;
      for (auto &I : oBB) {
        toerase.push_back(&I);
      }
      for (auto I : llvm::reverse(toerase)) {
        maker.eraseIfUnused(*I, /*erase*/ true,
                            /*check*/ key.mode ==
                                DerivativeMode::ReverseModeCombined);
      }

      if (key.mode != DerivativeMode::ReverseModeCombined) {
        if (newBB->getTerminator())
          gutils->erase(newBB->getTerminator());
        IRBuilder<> builder(newBB);
        builder.CreateUnreachable();
      }
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);
    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
    }

    BasicBlock::reverse_iterator I = oBB.rbegin(), E = oBB.rend();
    ++I;
    for (; I != E; ++I) {
      maker.visit(&*I);
      assert(oBB.rend() == E);
    }

    createInvertedTerminator(gutils, key.constant_args, &oBB, retAlloca,
                             dretAlloca,
                             0 + (key.additionalType ? 1 : 0) +
                                 ((key.retType == DIFFE_TYPE::DUP_ARG ||
                                   key.retType == DIFFE_TYPE::DUP_NONEED)
                                      ? 1
                                      : 0),
                             key.retType);
  }

  if (key.mode == DerivativeMode::ReverseModeGradient)
    restoreCache(gutils, mapping, guaranteedUnreachable);

  gutils->eraseFictiousPHIs();

  BasicBlock *entry = &gutils->newFunc->getEntryBlock();

  auto Arch =
      llvm::Triple(gutils->newFunc->getParent()->getTargetTriple()).getArch();
  unsigned int SharedAddrSpace =
      Arch == Triple::amdgcn ? (int)AMDGPU::HSAMD::AddressSpaceQualifier::Local
                             : 3;

  if (key.mode == DerivativeMode::ReverseModeCombined) {
    BasicBlock *sharedBlock = nullptr;
    for (auto &g : gutils->newFunc->getParent()->globals()) {
      if (hasMetadata(&g, "enzyme_internalshadowglobal")) {
        IRBuilder<> entryBuilder(gutils->inversionAllocs,
                                 gutils->inversionAllocs->begin());

        if ((Arch == Triple::nvptx || Arch == Triple::nvptx64 ||
             Arch == Triple::amdgcn) &&
            g.getType()->getAddressSpace() == SharedAddrSpace) {
          if (sharedBlock == nullptr)
            sharedBlock = BasicBlock::Create(entry->getContext(), "shblock",
                                             gutils->newFunc);
          entryBuilder.SetInsertPoint(sharedBlock);
        }
        auto store = entryBuilder.CreateStore(
            Constant::getNullValue(g.getValueType()), &g);
        if (g.getAlign())
          store->setAlignment(*g.getAlign());
      }
    }
    if (sharedBlock) {
      BasicBlock *OldEntryInsts = entry->splitBasicBlock(entry->begin());
      entry->getTerminator()->eraseFromParent();
      IRBuilder<> ebuilder(entry);

      Value *tx, *ty, *tz;
      if (Arch == Triple::nvptx || Arch == Triple::nvptx64) {
        tx = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_x));
        ty = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_y));
        tz = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::nvvm_read_ptx_sreg_tid_z));
      } else if (Arch == Triple::amdgcn) {
        tx = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_x));
        ty = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_y));
        tz = ebuilder.CreateCall(Intrinsic::getDeclaration(
            gutils->newFunc->getParent(), Intrinsic::amdgcn_workitem_id_z));
      } else {
        llvm_unreachable("unknown gpu architecture");
      }
      Value *OrVal = ebuilder.CreateOr(ebuilder.CreateOr(tx, ty), tz);

      ebuilder.CreateCondBr(
          ebuilder.CreateICmpEQ(OrVal, ConstantInt::get(OrVal->getType(), 0)),
          sharedBlock, OldEntryInsts);

      IRBuilder<> instbuilder(OldEntryInsts, OldEntryInsts->begin());

      auto BarrierInst = Arch == Triple::amdgcn
                             ? (llvm::Intrinsic::ID)Intrinsic::amdgcn_s_barrier
                             : (llvm::Intrinsic::ID)Intrinsic::nvvm_barrier0;
      instbuilder.CreateCall(
          Intrinsic::getDeclaration(gutils->newFunc->getParent(), BarrierInst),
          {});
      OldEntryInsts->moveAfter(entry);
      sharedBlock->moveAfter(entry);
      IRBuilder<> sbuilder(sharedBlock);
      sbuilder.CreateBr(OldEntryInsts);
      SmallVector<AllocaInst *, 3> AIs;
      for (auto &I : *OldEntryInsts) {
        if (auto AI = dyn_cast<AllocaInst>(&I))
          AIs.push_back(AI);
      }
      for (auto AI : AIs)
        AI->moveBefore(entry->getFirstNonPHIOrDbgOrLifetime());
      entry = OldEntryInsts;
    }
  }

  cleanupInversionAllocs(gutils, entry);
  clearFunctionAttributes(gutils->newFunc);

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  auto nf = gutils->newFunc;
  delete gutils;

  PPC.LowerAllocAddr(nf);

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*nf, PA);
  }
  PPC.AlwaysInline(nf);
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(nf, /*mem2reg*/ true);

  if (prevFunction) {
    prevFunction->replaceAllUsesWith(nf);
    prevFunction->eraseFromParent();
  }

  // Do not run post processing optimizations if the body of an openmp
  // parallel so the adjointgenerator can successfully extract the allocation
  // and frees and hoist them into the parent. Optimizing before then may
  // make the IR different to traverse, and thus impossible to find the allocs.
  if (PostOpt && !omp)
    PPC.optimizeIntermediate(nf);
  if (EnzymePrint) {
    llvm::errs() << *nf << "\n";
  }
  return nf;
}

Function *EnzymeLogic::CreateForwardDiff(
    RequestContext context, Function *todiff, DIFFE_TYPE retType,
    ArrayRef<DIFFE_TYPE> constant_args, TypeAnalysis &TA, bool returnUsed,
    DerivativeMode mode, bool freeMemory, unsigned width,
    llvm::Type *additionalArg, const FnTypeInfo &oldTypeInfo_,
    const std::vector<bool> _overwritten_args,
    const AugmentedReturn *augmenteddata, bool omp) {

  TimeTraceScope timeScope("CreateForwardDiff", todiff->getName());

  assert(retType != DIFFE_TYPE::OUT_DIFF);

  assert(mode == DerivativeMode::ForwardMode ||
         mode == DerivativeMode::ForwardModeSplit ||
         mode == DerivativeMode::ForwardModeError);

  FnTypeInfo oldTypeInfo = preventTypeAnalysisLoops(oldTypeInfo_, todiff);

  if (retType != DIFFE_TYPE::CONSTANT)
    assert(!todiff->getReturnType()->isVoidTy());

  if (returnUsed)
    assert(!todiff->getReturnType()->isVoidTy());

  if (mode != DerivativeMode::ForwardMode &&
      mode != DerivativeMode::ForwardModeError)
    assert(_overwritten_args.size() == todiff->arg_size());

  ForwardCacheKey tup = {todiff,     retType, constant_args, _overwritten_args,
                         returnUsed, mode,    width,         additionalArg,
                         oldTypeInfo};

  if (ForwardCachedFunctions.find(tup) != ForwardCachedFunctions.end()) {
    return ForwardCachedFunctions.find(tup)->second;
  }

  TargetLibraryInfo &TLI = PPC.FAM.getResult<TargetLibraryAnalysis>(*todiff);

  // TODO change this to go by default function type assumptions
  bool hasconstant = false;
  for (auto v : constant_args) {
    assert(v != DIFFE_TYPE::OUT_DIFF);
    if (v == DIFFE_TYPE::CONSTANT) {
      hasconstant = true;
      break;
    }
  }

  if (auto md = hasMetadata(todiff, (mode == DerivativeMode::ForwardMode ||
                                     mode == DerivativeMode::ForwardModeError)
                                        ? "enzyme_derivative"
                                        : "enzyme_splitderivative")) {
    if (!isa<MDTuple>(md)) {
      llvm::errs() << *todiff << "\n";
      llvm::errs() << *md << "\n";
      report_fatal_error(
          "unknown derivative for function -- metadata incorrect");
    }
    auto md2 = cast<MDTuple>(md);
    assert(md2);
    assert(md2->getNumOperands() == 1);
    if (!md2->getOperand(0)) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "Failed to use custom forward mode derivative for "
         << todiff->getName() << "\n";
      ss << " found metadata (but null op0) " << *md2 << "\n";
      EmitFailure("NoDerivative", context.req->getDebugLoc(), context.req,
                  ss.str());
      return ForwardCachedFunctions[tup] = nullptr;
    }
    if (!isa<ConstantAsMetadata>(md2->getOperand(0))) {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << "Failed to use custom forward mode derivative for "
         << todiff->getName() << "\n";
      ss << " found metadata (but not constantasmetadata) "
         << *md2->getOperand(0) << "\n";
      EmitFailure("NoDerivative", context.req->getDebugLoc(), context.req,
                  ss.str());
      return ForwardCachedFunctions[tup] = nullptr;
    }
    auto gvemd = cast<ConstantAsMetadata>(md2->getOperand(0));
    auto foundcalled = cast<Function>(gvemd->getValue());

    if ((foundcalled->getReturnType()->isVoidTy() ||
         retType != DIFFE_TYPE::CONSTANT) &&
        !hasconstant && returnUsed)
      return foundcalled;

    if (!foundcalled->getReturnType()->isVoidTy() && !hasconstant) {
      if (returnUsed && retType == DIFFE_TYPE::CONSTANT) {
      }
      if (!returnUsed && retType != DIFFE_TYPE::CONSTANT && !hasconstant) {
        FunctionType *FTy = FunctionType::get(
            todiff->getReturnType(), foundcalled->getFunctionType()->params(),
            foundcalled->getFunctionType()->isVarArg());
        Function *NewF = Function::Create(
            FTy, Function::LinkageTypes::InternalLinkage,
            "fixderivative_" + todiff->getName(), todiff->getParent());
        for (auto pair : llvm::zip(NewF->args(), foundcalled->args())) {
          std::get<0>(pair).setName(std::get<1>(pair).getName());
        }

        BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
        IRBuilder<> bb(BB);
        SmallVector<Value *, 2> args;
        for (auto &a : NewF->args())
          args.push_back(&a);
        auto cal = bb.CreateCall(foundcalled, args);
        cal->setCallingConv(foundcalled->getCallingConv());

        bb.CreateRet(bb.CreateExtractValue(cal, 1));
        return ForwardCachedFunctions[tup] = NewF;
      }
      assert(returnUsed);
    }

    SmallVector<Type *, 2> curTypes;
    bool legal = true;
    SmallVector<DIFFE_TYPE, 4> nextConstantArgs;
    for (auto tup : llvm::zip(todiff->args(), constant_args)) {
      auto &arg = std::get<0>(tup);
      curTypes.push_back(arg.getType());
      if (std::get<1>(tup) != DIFFE_TYPE::CONSTANT) {
        curTypes.push_back(arg.getType());
        nextConstantArgs.push_back(std::get<1>(tup));
        continue;
      }
      auto TT = oldTypeInfo.Arguments.find(&arg)->second[{-1}];
      if (TT.isFloat()) {
        nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
        continue;
      } else if (TT == BaseType::Integer) {
        nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
        continue;
      } else {
        legal = false;
        break;
      }
    }
    if (augmenteddata && augmenteddata->returns.find(AugmentedStruct::Tape) !=
                             augmenteddata->returns.end()) {
      assert(additionalArg);
      curTypes.push_back(additionalArg);
    }
    if (legal) {
      Type *RT = todiff->getReturnType();
      if (returnUsed && retType != DIFFE_TYPE::CONSTANT) {
        RT = StructType::get(RT->getContext(), {RT, RT});
      }
      if (!returnUsed && retType == DIFFE_TYPE::CONSTANT) {
        RT = Type::getVoidTy(RT->getContext());
      }

      FunctionType *FTy = FunctionType::get(
          RT, curTypes, todiff->getFunctionType()->isVarArg());

      Function *NewF = Function::Create(
          FTy, Function::LinkageTypes::InternalLinkage,
          "fixderivative_" + todiff->getName(), todiff->getParent());

      auto foundArg = NewF->arg_begin();
      SmallVector<Value *, 2> nextArgs;
      for (auto tup : llvm::zip(todiff->args(), constant_args)) {
        nextArgs.push_back(foundArg);
        auto &arg = std::get<0>(tup);
        foundArg->setName(arg.getName());
        foundArg++;
        if (std::get<1>(tup) != DIFFE_TYPE::CONSTANT) {
          foundArg->setName(arg.getName() + "'");
          nextConstantArgs.push_back(std::get<1>(tup));
          nextArgs.push_back(foundArg);
          foundArg++;
          continue;
        }
        auto TT = oldTypeInfo.Arguments.find(&arg)->second[{-1}];
        if (TT.isFloat()) {
          nextArgs.push_back(Constant::getNullValue(arg.getType()));
          nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
          continue;
        } else if (TT == BaseType::Integer) {
          nextArgs.push_back(nextArgs.back());
          nextConstantArgs.push_back(DIFFE_TYPE::DUP_ARG);
          continue;
        } else {
          legal = false;
          break;
        }
      }
      if (augmenteddata && augmenteddata->returns.find(AugmentedStruct::Tape) !=
                               augmenteddata->returns.end()) {
        foundArg->setName("tapeArg");
        nextArgs.push_back(foundArg);
        foundArg++;
      }

      BasicBlock *BB = BasicBlock::Create(NewF->getContext(), "entry", NewF);
      IRBuilder<> bb(BB);
      auto cal = bb.CreateCall(foundcalled, nextArgs);
      cal->setCallingConv(foundcalled->getCallingConv());

      if (returnUsed && retType != DIFFE_TYPE::CONSTANT) {
        bb.CreateRet(cal);
      } else if (returnUsed) {
        bb.CreateRet(bb.CreateExtractValue(cal, 0));
      } else if (retType != DIFFE_TYPE::CONSTANT) {
        bb.CreateRet(bb.CreateExtractValue(cal, 1));
      } else {
        bb.CreateRetVoid();
      }

      return ForwardCachedFunctions[tup] = NewF;
    }

    EmitWarning("NoCustom", *todiff,
                "Cannot use provided custom derivative pass");
  }

  bool retActive = retType != DIFFE_TYPE::CONSTANT;

  ReturnType retVal =
      returnUsed ? (retActive ? ReturnType::TwoReturns : ReturnType::Return)
                 : (retActive ? ReturnType::Return : ReturnType::Void);

  bool diffeReturnArg = false;

  DiffeGradientUtils *gutils = DiffeGradientUtils::CreateFromClone(
      *this, mode, width, todiff, TLI, TA, oldTypeInfo, retType,
      /*shadowReturn*/ retActive, diffeReturnArg, constant_args, retVal,
      additionalArg, omp);

  insert_or_assign2<ForwardCacheKey, Function *>(ForwardCachedFunctions, tup,
                                                 gutils->newFunc);

  if (todiff->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    if (mode == DerivativeMode::ForwardModeError) {
      ss << "No forward mode error function found for " + todiff->getName()
         << "\n";
    } else {
      ss << "No forward mode derivative found for " + todiff->getName() << "\n";
    }
    if (context.req) {
      ss << " at context: " << *context.req;
    } else {
      ss << *todiff << "\n";
    }
    BasicBlock *entry = &gutils->newFunc->getEntryBlock();
    cleanupInversionAllocs(gutils, entry);
    clearFunctionAttributes(gutils->newFunc);
    if (EmitNoDerivativeError(ss.str(), todiff, context)) {
      auto newFunc = gutils->newFunc;
      delete gutils;
      return newFunc;
    }
    llvm::errs() << "mod: " << *todiff->getParent() << "\n";
    llvm::errs() << *todiff << "\n";
    llvm_unreachable("attempting to differentiate function without definition");
  }
  gutils->FreeMemory = freeMemory;

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(gutils->oldFunc);

  gutils->forceActiveDetection();

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
  SmallPtrSet<const Instruction *, 4> unnecessaryStores;

  AdjointGenerator *maker;

  std::unique_ptr<const std::map<Instruction *, bool>> can_modref_map;
  if (mode == DerivativeMode::ForwardModeSplit) {
    std::vector<bool> _overwritten_argsPP = _overwritten_args;

    gutils->computeGuaranteedFrees();
    CacheAnalysis CA(
        gutils->allocationsWithGuaranteedFree,
        gutils->rematerializableAllocations, gutils->TR, *gutils->OrigAA,
        gutils->oldFunc,
        PPC.FAM.getResult<ScalarEvolutionAnalysis>(*gutils->oldFunc),
        *gutils->OrigLI, *gutils->OrigDT, TLI, guaranteedUnreachable,
        _overwritten_argsPP, mode, omp);
    const std::map<CallInst *, const std::vector<bool>> overwritten_args_map =
        CA.compute_overwritten_args_for_callsites();
    gutils->overwritten_args_map_ptr = &overwritten_args_map;
    can_modref_map = std::make_unique<const std::map<Instruction *, bool>>(
        CA.compute_uncacheable_load_map());
    gutils->can_modref_map = can_modref_map.get();

    gutils->forceAugmentedReturns();

    gutils->computeMinCache();

    auto getIndex = [&](Instruction *I, CacheType u,
                        IRBuilder<> &B) -> unsigned {
      assert(augmenteddata);
      return gutils->getIndex(std::make_pair(I, u), augmenteddata->tapeIndices,
                              B);
    };

    calculateUnusedValuesInFunction(
        *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
        returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
    gutils->unnecessaryValuesP = &unnecessaryValues;

    calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                    unnecessaryInstructions, gutils, TLI);

    maker = new AdjointGenerator(
        mode, gutils, constant_args, retType, getIndex, overwritten_args_map,
        /*returnuses*/ nullptr, augmenteddata, nullptr, unnecessaryValues,
        unnecessaryInstructions, unnecessaryStores, guaranteedUnreachable,
        nullptr);

    if (additionalArg) {
      auto v = gutils->newFunc->arg_end();
      v--;
      Value *additionalValue = v;
      assert(augmenteddata);

      // TODO VERIFY THIS
      if (augmenteddata->tapeType &&
          augmenteddata->tapeType != additionalValue->getType()) {
        IRBuilder<> BuilderZ(gutils->inversionAllocs);
        if (!augmenteddata->tapeType->isEmptyTy()) {
          auto tapep = BuilderZ.CreatePointerCast(
              additionalValue, PointerType::getUnqual(augmenteddata->tapeType));
          LoadInst *truetape =
              BuilderZ.CreateLoad(augmenteddata->tapeType, tapep, "truetape");
          truetape->setMetadata("enzyme_mustcache",
                                MDNode::get(truetape->getContext(), {}));

          if (!omp && gutils->FreeMemory) {
            CreateDealloc(BuilderZ, additionalValue);
          }
          additionalValue = truetape;
        } else {
          if (gutils->FreeMemory) {
            auto size = gutils->newFunc->getParent()
                            ->getDataLayout()
                            .getTypeAllocSizeInBits(augmenteddata->tapeType);
            if (size != 0) {
              CreateDealloc(BuilderZ, additionalValue);
            }
          }
          additionalValue = UndefValue::get(augmenteddata->tapeType);
        }
      }

      // TODO here finish up making recursive structs simply pass in i8*
      gutils->setTape(additionalValue);
    }
  } else {
    gutils->forceAugmentedReturns();
    calculateUnusedValuesInFunction(
        *gutils->oldFunc, unnecessaryValues, unnecessaryInstructions,
        returnUsed, mode, gutils, TLI, constant_args, guaranteedUnreachable);
    gutils->unnecessaryValuesP = &unnecessaryValues;

    calculateUnusedStoresInFunction(*gutils->oldFunc, unnecessaryStores,
                                    unnecessaryInstructions, gutils, TLI);
    maker =
        new AdjointGenerator(mode, gutils, constant_args, retType, nullptr, {},
                             /*returnuses*/ nullptr, nullptr, nullptr,
                             unnecessaryValues, unnecessaryInstructions,
                             unnecessaryStores, guaranteedUnreachable, nullptr);
  }

  for (BasicBlock &oBB : *gutils->oldFunc) {
    // Don't create derivatives for code that results in termination
    if (guaranteedUnreachable.find(&oBB) != guaranteedUnreachable.end()) {
      for (auto &I : oBB) {
        maker->eraseIfUnused(I, /*erase*/ true, /*check*/ true);
      }
      continue;
    }

    auto term = oBB.getTerminator();
    assert(term);
    if (!isa<ReturnInst>(term) && !isa<BranchInst>(term) &&
        !isa<SwitchInst>(term)) {
      llvm::errs() << *oBB.getParent() << "\n";
      llvm::errs() << "unknown terminator instance " << *term << "\n";
      assert(0 && "unknown terminator inst");
    }

    auto first = oBB.begin();
    auto last = oBB.empty() ? oBB.end() : std::prev(oBB.end());
    for (auto it = first; it != last; ++it) {
      maker->visit(&*it);
    }

    createTerminator(gutils, &oBB, retType, retVal);
  }

  if (mode == DerivativeMode::ForwardModeSplit && augmenteddata)
    restoreCache(gutils, augmenteddata->tapeIndices, guaranteedUnreachable);

  gutils->eraseFictiousPHIs();

  BasicBlock *entry = &gutils->newFunc->getEntryBlock();

  auto Arch =
      llvm::Triple(gutils->newFunc->getParent()->getTargetTriple()).getArch();

  cleanupInversionAllocs(gutils, entry);
  clearFunctionAttributes(gutils->newFunc);

  if (llvm::verifyFunction(*gutils->newFunc, &llvm::errs())) {
    llvm::errs() << *gutils->oldFunc << "\n";
    llvm::errs() << *gutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  auto nf = gutils->newFunc;
  delete gutils;
  delete maker;

  PPC.LowerAllocAddr(nf);

  {
    PreservedAnalyses PA;
    PPC.FAM.invalidate(*nf, PA);
  }
  PPC.AlwaysInline(nf);
  if (Arch == Triple::nvptx || Arch == Triple::nvptx64)
    PPC.ReplaceReallocs(nf, /*mem2reg*/ true);

  if (PostOpt)
    PPC.optimizeIntermediate(nf);
  if (EnzymePrint) {
    llvm::errs() << *nf << "\n";
  }
  return nf;
}

static Value *floatValTruncate(IRBuilderBase &B, Value *v,
                               FloatTruncation truncation) {
  if (truncation.isToFPRT())
    return v;

  Type *toTy = truncation.getToType(B.getContext());
  if (auto vty = dyn_cast<VectorType>(v->getType()))
    toTy = VectorType::get(toTy, vty->getElementCount());
  return B.CreateFPTrunc(v, toTy, "enzyme_trunc");
}

static Value *floatValExpand(IRBuilderBase &B, Value *v,
                             FloatTruncation truncation) {
  if (truncation.isToFPRT())
    return v;

  Type *fromTy = truncation.getFromType(B.getContext());
  if (auto vty = dyn_cast<VectorType>(v->getType()))
    fromTy = VectorType::get(fromTy, vty->getElementCount());
  return B.CreateFPExt(v, fromTy, "enzyme_exp");
}

static Value *floatMemTruncate(IRBuilderBase &B, Value *v,
                               FloatTruncation truncation) {
  if (isa<VectorType>(v->getType()))
    report_fatal_error("vector operations not allowed in mem trunc mode");

  Type *toTy = truncation.getToType(B.getContext());
  return B.CreateBitCast(v, toTy);
}

static Value *floatMemExpand(IRBuilderBase &B, Value *v,
                             FloatTruncation truncation) {
  if (isa<VectorType>(v->getType()))
    report_fatal_error("vector operations not allowed in mem trunc mode");

  Type *fromTy = truncation.getFromType(B.getContext());
  return B.CreateBitCast(v, fromTy);
}

class TruncateUtils {
protected:
  FloatTruncation truncation;
  llvm::Module *M;
  Type *fromType;
  Type *toType;
  LLVMContext &ctx;

private:
  std::string getOriginalFPRTName(std::string Name) {
    return std::string(EnzymeFPRTOriginalPrefix) + truncation.mangleFrom() +
           "_" + Name;
  }
  std::string getFPRTName(std::string Name) {
    return std::string(EnzymeFPRTPrefix) + truncation.mangleFrom() + "_" + Name;
  }

  // Creates a function which contains the original floating point operation.
  // The user can use this to compare results against.
  void createOriginalFPRTFunc(Instruction &I, std::string Name,
                              SmallVectorImpl<Value *> &Args,
                              llvm::Type *RetTy) {
    auto MangledName = getOriginalFPRTName(Name);
    auto F = M->getFunction(MangledName);
    if (!F) {
      SmallVector<Type *, 4> ArgTypes;
      for (auto Arg : Args)
        ArgTypes.push_back(Arg->getType());
      FunctionType *FnTy =
          FunctionType::get(RetTy, ArgTypes, /*is_vararg*/ false);
      F = Function::Create(FnTy, Function::ExternalLinkage, MangledName, M);
    }
    if (F->isDeclaration()) {
      BasicBlock *Entry = BasicBlock::Create(F->getContext(), "entry", F);
      auto ClonedI = I.clone();
      for (unsigned It = 0; It < Args.size(); It++)
        ClonedI->setOperand(It, F->getArg(It));
      auto Return = ReturnInst::Create(F->getContext(), ClonedI, Entry);
      ClonedI->insertBefore(Return);
    }
  }

  Function *getFPRTFunc(std::string Name, SmallVectorImpl<Value *> &Args,
                        llvm::Type *RetTy) {
    auto MangledName = getFPRTName(Name);
    auto F = M->getFunction(MangledName);
    if (!F) {
      SmallVector<Type *, 4> ArgTypes;
      for (auto Arg : Args)
        ArgTypes.push_back(Arg->getType());
      FunctionType *FnTy =
          FunctionType::get(RetTy, ArgTypes, /*is_vararg*/ false);
      F = Function::Create(FnTy, Function::ExternalLinkage, MangledName, M);
    }
    return F;
  }

  CallInst *createFPRTGeneric(llvm::IRBuilderBase &B, std::string Name,
                              const SmallVectorImpl<Value *> &ArgsIn,
                              llvm::Type *RetTy) {
    SmallVector<Value *, 5> Args(ArgsIn.begin(), ArgsIn.end());
    Args.push_back(B.getInt64(truncation.getTo().exponentWidth));
    Args.push_back(B.getInt64(truncation.getTo().significandWidth));
    Args.push_back(B.getInt64(truncation.getMode()));
    auto FprtFunc = getFPRTFunc(Name, Args, RetTy);
    return cast<CallInst>(B.CreateCall(FprtFunc, Args));
  }

public:
  TruncateUtils(FloatTruncation truncation, Module *M)
      : truncation(truncation), M(M), ctx(M->getContext()) {
    fromType = truncation.getFromType(ctx);
    toType = truncation.getToType(ctx);
    if (fromType == toType)
      assert(truncation.isToFPRT());
  }

  Type *getFromType() { return fromType; }

  Type *getToType() { return toType; }

  CallInst *createFPRTConstCall(llvm::IRBuilderBase &B, Value *V) {
    assert(V->getType() == getFromType());
    SmallVector<Value *, 1> Args;
    Args.push_back(V);
    return createFPRTGeneric(B, "const", Args, getToType());
  }
  CallInst *createFPRTNewCall(llvm::IRBuilderBase &B, Value *V) {
    assert(V->getType() == getFromType());
    SmallVector<Value *, 1> Args;
    Args.push_back(V);
    return createFPRTGeneric(B, "new", Args, getToType());
  }
  CallInst *createFPRTGetCall(llvm::IRBuilderBase &B, Value *V) {
    SmallVector<Value *, 1> Args;
    Args.push_back(V);
    return createFPRTGeneric(B, "get", Args, getToType());
  }
  CallInst *createFPRTDeleteCall(llvm::IRBuilderBase &B, Value *V) {
    SmallVector<Value *, 1> Args;
    Args.push_back(V);
    return createFPRTGeneric(B, "delete", Args, B.getVoidTy());
  }
  CallInst *createFPRTOpCall(llvm::IRBuilderBase &B, llvm::Instruction &I,
                             llvm::Type *RetTy,
                             SmallVectorImpl<Value *> &ArgsIn) {
    std::string Name;
    if (auto BO = dyn_cast<BinaryOperator>(&I)) {
      Name = "binop_" + std::string(BO->getOpcodeName());
    } else if (auto II = dyn_cast<IntrinsicInst>(&I)) {
      auto FOp = II->getCalledFunction();
      assert(FOp);
      Name = "intr_" + std::string(FOp->getName());
      for (auto &C : Name)
        if (C == '.')
          C = '_';
    } else if (auto CI = dyn_cast<CallInst>(&I)) {
      if (auto F = CI->getCalledFunction())
        Name = "func_" + std::string(F->getName());
      else
        llvm_unreachable(
            "Unexpected indirect call inst for conversion to FPRT");
    } else if (auto CI = dyn_cast<FCmpInst>(&I)) {
      Name = "fcmp_" + std::string(CI->getPredicateName(CI->getPredicate()));
    } else {
      llvm_unreachable("Unexpected instruction for conversion to FPRT");
    }
    createOriginalFPRTFunc(I, Name, ArgsIn, RetTy);
    return createFPRTGeneric(B, Name, ArgsIn, RetTy);
  }
};

class TruncateGenerator : public llvm::InstVisitor<TruncateGenerator>,
                          public TruncateUtils {
private:
  ValueToValueMapTy &originalToNewFn;
  FloatTruncation truncation;
  Function *oldFunc;
  Function *newFunc;
  TruncateMode mode;
  EnzymeLogic &Logic;
  LLVMContext &ctx;

public:
  TruncateGenerator(ValueToValueMapTy &originalToNewFn,
                    FloatTruncation truncation, Function *oldFunc,
                    Function *newFunc, EnzymeLogic &Logic)
      : TruncateUtils(truncation, newFunc->getParent()),
        originalToNewFn(originalToNewFn), truncation(truncation),
        oldFunc(oldFunc), newFunc(newFunc), mode(truncation.getMode()),
        Logic(Logic), ctx(newFunc->getContext()) {}

  void checkHandled(llvm::Instruction &inst) {
    // TODO
    // if (all_of(inst.getOperandList(),
    //            [&](Use *use) { return use->get()->getType() == fromType; }))
    //   todo(inst);
  }

  // TODO
  void handleTrunc();
  void hendleIntToFloat();
  void handleFloatToInt();

  void visitInstruction(llvm::Instruction &inst) {
    using namespace llvm;

    // TODO explicitly handle all instructions rather than using the catch all
    // below

    switch (inst.getOpcode()) {
      // #include "InstructionDerivatives.inc"
    default:
      break;
    }

    checkHandled(inst);
  }

  Value *truncate(IRBuilder<> &B, Value *v) {
    switch (mode) {
    case TruncMemMode:
      if (isa<ConstantFP>(v))
        return createFPRTConstCall(B, v);
      return floatMemTruncate(B, v, truncation);
    case TruncOpMode:
    case TruncOpFullModuleMode:
      return floatValTruncate(B, v, truncation);
    }
    llvm_unreachable("Unknown trunc mode");
  }

  Value *expand(IRBuilder<> &B, Value *v) {
    switch (mode) {
    case TruncMemMode:
      return floatMemExpand(B, v, truncation);
    case TruncOpMode:
    case TruncOpFullModuleMode:
      return floatValExpand(B, v, truncation);
    }
    llvm_unreachable("Unknown trunc mode");
  }

  void todo(llvm::Instruction &I) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "cannot handle unknown instruction\n" << I;
    if (CustomErrorHandler) {
      IRBuilder<> Builder2(getNewFromOriginal(&I));
      CustomErrorHandler(ss.str().c_str(), wrap(&I), ErrorType::NoTruncate,
                         this, nullptr, wrap(&Builder2));
      return;
    } else {
      EmitFailure("NoTruncate", I.getDebugLoc(), &I, ss.str());
      return;
    }
  }

  void visitAllocaInst(llvm::AllocaInst &I) { return; }
  void visitICmpInst(llvm::ICmpInst &I) { return; }
  void visitFCmpInst(llvm::FCmpInst &CI) {
    switch (mode) {
    case TruncMemMode: {
      auto LHS = getNewFromOriginal(CI.getOperand(0));
      auto RHS = getNewFromOriginal(CI.getOperand(1));
      if (LHS->getType() != getFromType())
        return;

      auto newI = getNewFromOriginal(&CI);
      IRBuilder<> B(newI);
      auto truncLHS = truncate(B, LHS);
      auto truncRHS = truncate(B, RHS);

      SmallVector<Value *, 2> Args;
      Args.push_back(LHS);
      Args.push_back(RHS);
      Instruction *nres;
      if (truncation.isToFPRT())
        nres = createFPRTOpCall(B, CI, B.getInt1Ty(), Args);
      else
        nres =
            cast<FCmpInst>(B.CreateFCmp(CI.getPredicate(), truncLHS, truncRHS));
      nres->takeName(newI);
      nres->copyIRFlags(newI);
      newI->replaceAllUsesWith(nres);
      newI->eraseFromParent();
      return;
    }
    case TruncOpMode:
    case TruncOpFullModuleMode:
      return;
    }
  }
  void visitLoadInst(llvm::LoadInst &LI) {
    auto alignment = LI.getAlign();
    visitLoadLike(LI, alignment);
  }
  void visitStoreInst(llvm::StoreInst &SI) {
    auto align = SI.getAlign();
    visitCommonStore(SI, SI.getPointerOperand(), SI.getValueOperand(), align,
                     SI.isVolatile(), SI.getOrdering(), SI.getSyncScopeID(),
                     /*mask=*/nullptr);
  }
  void visitGetElementPtrInst(llvm::GetElementPtrInst &gep) { return; }
  void visitPHINode(llvm::PHINode &phi) { return; }
  void visitCastInst(llvm::CastInst &CI) {
    switch (mode) {
    case TruncMemMode: {
      if (CI.getSrcTy() == getFromType() || CI.getDestTy() == getFromType())
        todo(CI);
      return;
    }
    case TruncOpMode:
    case TruncOpFullModuleMode:
      return;
    }
  }
  void visitSelectInst(llvm::SelectInst &SI) {
    switch (mode) {
    case TruncMemMode: {
      auto newI = getNewFromOriginal(&SI);
      IRBuilder<> B(newI);
      auto newT = truncate(B, getNewFromOriginal(SI.getTrueValue()));
      auto newF = truncate(B, getNewFromOriginal(SI.getFalseValue()));
      auto nres = cast<SelectInst>(
          B.CreateSelect(getNewFromOriginal(SI.getCondition()), newT, newF));
      nres->takeName(newI);
      nres->copyIRFlags(newI);
      newI->replaceAllUsesWith(expand(B, nres));
      newI->eraseFromParent();
      return;
    }
    case TruncOpMode:
    case TruncOpFullModuleMode:
      return;
    }
    llvm_unreachable("");
  }
  void visitExtractElementInst(llvm::ExtractElementInst &EEI) { return; }
  void visitInsertElementInst(llvm::InsertElementInst &EEI) { return; }
  void visitShuffleVectorInst(llvm::ShuffleVectorInst &EEI) { return; }
  void visitExtractValueInst(llvm::ExtractValueInst &EEI) { return; }
  void visitInsertValueInst(llvm::InsertValueInst &EEI) { return; }
  void visitBinaryOperator(llvm::BinaryOperator &BO) {
    auto oldLHS = BO.getOperand(0);
    auto oldRHS = BO.getOperand(1);

    if (oldLHS->getType() != getFromType() &&
        oldRHS->getType() != getFromType())
      return;

    switch (BO.getOpcode()) {
    default:
      break;
    case BinaryOperator::Add:
    case BinaryOperator::Sub:
    case BinaryOperator::Mul:
    case BinaryOperator::UDiv:
    case BinaryOperator::SDiv:
    case BinaryOperator::URem:
    case BinaryOperator::SRem:
    case BinaryOperator::AShr:
    case BinaryOperator::LShr:
    case BinaryOperator::Shl:
    case BinaryOperator::And:
    case BinaryOperator::Or:
    case BinaryOperator::Xor:
      assert(0 && "Invalid binop opcode for float arg");
      return;
    }

    auto newI = getNewFromOriginal(&BO);
    IRBuilder<> B(newI);
    auto newLHS = truncate(B, getNewFromOriginal(oldLHS));
    auto newRHS = truncate(B, getNewFromOriginal(oldRHS));
    Instruction *nres = nullptr;
    if (truncation.isToFPRT()) {
      SmallVector<Value *, 2> Args({newLHS, newRHS});
      nres = createFPRTOpCall(B, BO, truncation.getToType(ctx), Args);
    } else {
      nres = cast<Instruction>(B.CreateBinOp(BO.getOpcode(), newLHS, newRHS));
    }
    nres->takeName(newI);
    nres->copyIRFlags(newI);
    newI->replaceAllUsesWith(expand(B, nres));
    newI->eraseFromParent();
    return;
  }
  void visitMemSetInst(llvm::MemSetInst &MS) { visitMemSetCommon(MS); }
  void visitMemSetCommon(llvm::CallInst &MS) { return; }
  void visitMemTransferInst(llvm::MemTransferInst &MTI) {
    using namespace llvm;
    Value *isVolatile = getNewFromOriginal(MTI.getOperand(3));
    auto srcAlign = MTI.getSourceAlign();
    auto dstAlign = MTI.getDestAlign();
    visitMemTransferCommon(MTI.getIntrinsicID(), srcAlign, dstAlign, MTI,
                           MTI.getOperand(0), MTI.getOperand(1),
                           getNewFromOriginal(MTI.getOperand(2)), isVolatile);
  }
  void visitMemTransferCommon(llvm::Intrinsic::ID ID, llvm::MaybeAlign srcAlign,
                              llvm::MaybeAlign dstAlign, llvm::CallInst &MTI,
                              llvm::Value *orig_dst, llvm::Value *orig_src,
                              llvm::Value *new_size, llvm::Value *isVolatile) {
    return;
  }
  void visitFenceInst(llvm::FenceInst &FI) { return; }

  bool handleIntrinsic(llvm::CallInst &CI, Intrinsic::ID ID) {
    if (isDbgInfoIntrinsic(ID))
      return true;

    auto newI = cast<llvm::CallInst>(getNewFromOriginal(&CI));
    IRBuilder<> B(newI);

    SmallVector<Value *, 2> orig_ops(CI.arg_size());
    for (unsigned i = 0; i < CI.arg_size(); ++i)
      orig_ops[i] = CI.getOperand(i);

    bool hasFromType = false;
    SmallVector<Value *, 2> new_ops(CI.arg_size());
    for (unsigned i = 0; i < CI.arg_size(); ++i) {
      if (orig_ops[i]->getType() == getFromType()) {
        new_ops[i] = truncate(B, getNewFromOriginal(orig_ops[i]));
        hasFromType = true;
      } else {
        new_ops[i] = getNewFromOriginal(orig_ops[i]);
      }
    }
    Type *retTy = CI.getType();
    if (CI.getType() == getFromType()) {
      hasFromType = true;
      retTy = getToType();
    }

    if (!hasFromType)
      return false;

    Instruction *intr = nullptr;
    Value *nres = nullptr;
    if (truncation.isToFPRT()) {
      nres = intr = createFPRTOpCall(B, CI, retTy, new_ops);
    } else {
      // TODO check that the intrinsic is overloaded
      nres = intr =
          createIntrinsicCall(B, ID, retTy, new_ops, &CI, CI.getName());
    }
    if (newI->getType() == getFromType())
      nres = expand(B, nres);
    intr->copyIRFlags(newI);
    newI->replaceAllUsesWith(nres);
    newI->eraseFromParent();
    return true;
  }
  void visitIntrinsicInst(llvm::IntrinsicInst &II) {
    handleIntrinsic(II, II.getIntrinsicID());
  }

  void visitReturnInst(llvm::ReturnInst &I) { return; }

  void visitBranchInst(llvm::BranchInst &I) { return; }
  void visitSwitchInst(llvm::SwitchInst &I) { return; }
  void visitUnreachableInst(llvm::UnreachableInst &I) { return; }
  void visitLoadLike(llvm::Instruction &I, llvm::MaybeAlign alignment,
                     llvm::Value *mask = nullptr,
                     llvm::Value *orig_maskInit = nullptr) {
    return;
  }

  void visitCommonStore(llvm::Instruction &I, llvm::Value *orig_ptr,
                        llvm::Value *orig_val, llvm::MaybeAlign prevalign,
                        bool isVolatile, llvm::AtomicOrdering ordering,
                        llvm::SyncScope::ID syncScope, llvm::Value *mask) {
    return;
  }

  bool
  handleAdjointForIntrinsic(llvm::Intrinsic::ID ID, llvm::Instruction &I,
                            llvm::SmallVectorImpl<llvm::Value *> &orig_ops) {
    using namespace llvm;

    switch (ID) {
    case Intrinsic::nvvm_ldu_global_i:
    case Intrinsic::nvvm_ldu_global_p:
    case Intrinsic::nvvm_ldu_global_f:
    case Intrinsic::nvvm_ldg_global_i:
    case Intrinsic::nvvm_ldg_global_p:
    case Intrinsic::nvvm_ldg_global_f: {
      auto CI = cast<ConstantInt>(I.getOperand(1));
      visitLoadLike(I, /*Align*/ MaybeAlign(CI->getZExtValue()));
      return true;
    }
    default:
      break;
    }

    if (ID == Intrinsic::masked_store) {
      auto align0 = cast<ConstantInt>(I.getOperand(2))->getZExtValue();
      auto align = MaybeAlign(align0);
      visitCommonStore(I, /*orig_ptr*/ I.getOperand(1),
                       /*orig_val*/ I.getOperand(0), align,
                       /*isVolatile*/ false, llvm::AtomicOrdering::NotAtomic,
                       SyncScope::SingleThread,
                       /*mask*/ getNewFromOriginal(I.getOperand(3)));
      return true;
    }
    if (ID == Intrinsic::masked_load) {
      auto align0 = cast<ConstantInt>(I.getOperand(1))->getZExtValue();
      auto align = MaybeAlign(align0);
      visitLoadLike(I, align,
                    /*mask*/ getNewFromOriginal(I.getOperand(2)),
                    /*orig_maskInit*/ I.getOperand(3));
      return true;
    }

    return false;
  }

  llvm::Value *getNewFromOriginal(llvm::Value *v) {
    auto found = originalToNewFn.find(v);
    assert(found != originalToNewFn.end());
    return found->second;
  }

  llvm::Instruction *getNewFromOriginal(llvm::Instruction *v) {
    return cast<Instruction>(getNewFromOriginal((llvm::Value *)v));
  }

  bool handleKnownCalls(llvm::CallInst &call, llvm::Function *called,
                        llvm::StringRef funcName,
                        llvm::CallInst *const newCall) {
    return false;
  }

  Value *GetShadow(RequestContext &ctx, Value *v) {
    if (auto F = dyn_cast<Function>(v))
      return Logic.CreateTruncateFunc(ctx, F, truncation, mode);
    llvm::errs() << " unknown get truncated func: " << *v << "\n";
    llvm_unreachable("unknown get truncated func");
    return v;
  }
  // Return
  void visitCallInst(llvm::CallInst &CI) {
    Intrinsic::ID ID;
    StringRef funcName = getFuncNameFromCall(const_cast<CallInst *>(&CI));
    if (isMemFreeLibMFunction(funcName, &ID))
      if (handleIntrinsic(CI, ID))
        return;

    using namespace llvm;

    CallInst *const newCall = cast<CallInst>(getNewFromOriginal(&CI));
    IRBuilder<> BuilderZ(newCall);

    if (auto called = CI.getCalledFunction())
      if (handleKnownCalls(CI, called, getFuncNameFromCall(&CI), newCall))
        return;

    if (mode != TruncOpFullModuleMode) {
      RequestContext ctx(&CI, &BuilderZ);
      auto val = GetShadow(ctx, getNewFromOriginal(CI.getCalledOperand()));
      newCall->setCalledOperand(val);
    }
    return;
  }
  void visitFPTruncInst(FPTruncInst &I) { return; }
  void visitFPExtInst(FPExtInst &I) { return; }
  void visitFPToUIInst(FPToUIInst &I) { return; }
  void visitFPToSIInst(FPToSIInst &I) { return; }
  void visitUIToFPInst(UIToFPInst &I) { return; }
  void visitSIToFPInst(SIToFPInst &I) { return; }
};

bool EnzymeLogic::CreateTruncateValue(RequestContext context, Value *v,
                                      FloatRepresentation from,
                                      FloatRepresentation to, bool isTruncate) {
  assert(context.req && context.ip);

  IRBuilderBase &B = *context.ip;

  Value *converted = nullptr;
  auto truncation = FloatTruncation(from, to, TruncMemMode);
  TruncateUtils TU(truncation, B.GetInsertBlock()->getParent()->getParent());
  if (isTruncate)
    converted = TU.createFPRTNewCall(B, v);
  else
    converted = TU.createFPRTGetCall(B, v);
  assert(converted);

  context.req->replaceAllUsesWith(converted);
  context.req->eraseFromParent();

  return true;
}

llvm::Function *EnzymeLogic::CreateTruncateFunc(RequestContext context,
                                                llvm::Function *totrunc,
                                                FloatTruncation truncation,
                                                TruncateMode mode) {
  TruncateCacheKey tup(totrunc, truncation, mode);
  if (TruncateCachedFunctions.find(tup) != TruncateCachedFunctions.end()) {
    return TruncateCachedFunctions.find(tup)->second;
  }

  FunctionType *orig_FTy = totrunc->getFunctionType();
  SmallVector<Type *, 4> params;

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    params.push_back(orig_FTy->getParamType(i));
  }

  Type *NewTy = totrunc->getReturnType();

  FunctionType *FTy = FunctionType::get(NewTy, params, totrunc->isVarArg());
  std::string truncName =
      std::string("__enzyme_done_truncate_") + truncateModeStr(mode) +
      "_func_" + truncation.mangleTruncation() + "_" + totrunc->getName().str();
  Function *NewF = Function::Create(FTy, totrunc->getLinkage(), truncName,
                                    totrunc->getParent());

  if (mode != TruncOpFullModuleMode)
    NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  TruncateCachedFunctions[tup] = NewF;

  if (totrunc->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No truncate mode found for " + totrunc->getName() << "\n";
    llvm::Value *toshow = totrunc;
    if (context.req) {
      toshow = context.req;
      ss << " at context: " << *context.req;
    } else {
      ss << *totrunc << "\n";
    }
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(toshow),
                         ErrorType::NoDerivative, nullptr, wrap(totrunc),
                         wrap(context.ip));
      return NewF;
    }
    if (context.req) {
      EmitFailure("NoTruncate", context.req->getDebugLoc(), context.req,
                  ss.str());
      return NewF;
    }
    llvm::errs() << "mod: " << *totrunc->getParent() << "\n";
    llvm::errs() << *totrunc << "\n";
    llvm_unreachable("attempting to truncate function without definition");
  }

  ValueToValueMapTy originalToNewFn;

  for (auto i = totrunc->arg_begin(), j = NewF->arg_begin();
       i != totrunc->arg_end();) {
    originalToNewFn[i] = j;
    j->setName(i->getName());
    ++j;
    ++i;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, totrunc, originalToNewFn,
                    CloneFunctionChangeType::LocalChangesOnly, Returns, "",
                    nullptr);
#else
  CloneFunctionInto(NewF, totrunc, originalToNewFn, true, Returns, "", nullptr);
#endif

  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  TruncateGenerator handle(originalToNewFn, truncation, totrunc, NewF, *this);
  for (auto &BB : *totrunc)
    for (auto &I : BB)
      handle.visit(&I);

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *totrunc << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (5)");
  }

  return NewF;
}

llvm::Function *EnzymeLogic::CreateBatch(RequestContext context,
                                         Function *tobatch, unsigned width,
                                         ArrayRef<BATCH_TYPE> arg_types,
                                         BATCH_TYPE ret_type) {

  BatchCacheKey tup = std::make_tuple(tobatch, width, arg_types, ret_type);
  if (BatchCachedFunctions.find(tup) != BatchCachedFunctions.end()) {
    return BatchCachedFunctions.find(tup)->second;
  }

  FunctionType *orig_FTy = tobatch->getFunctionType();
  SmallVector<Type *, 4> params;
  unsigned long numVecParams =
      std::count(arg_types.begin(), arg_types.end(), BATCH_TYPE::VECTOR);

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      Type *ty = GradientUtils::getShadowType(orig_FTy->getParamType(i), width);
      params.push_back(ty);
    } else {
      params.push_back(orig_FTy->getParamType(i));
    }
  }

  Type *NewTy = GradientUtils::getShadowType(tobatch->getReturnType(), width);

  FunctionType *FTy = FunctionType::get(NewTy, params, tobatch->isVarArg());
  Function *NewF =
      Function::Create(FTy, tobatch->getLinkage(),
                       "batch_" + tobatch->getName(), tobatch->getParent());

  if (tobatch->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No batch mode found for " + tobatch->getName() << "\n";
    llvm::Value *toshow = tobatch;
    if (context.req) {
      toshow = context.req;
      ss << " at context: " << *context.req;
    } else {
      ss << *tobatch << "\n";
    }
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(toshow),
                         ErrorType::NoDerivative, nullptr, wrap(tobatch),
                         wrap(context.ip));
      return NewF;
    }
    if (context.req) {
      EmitFailure("NoDerivative", context.req->getDebugLoc(), context.req,
                  ss.str());
      return NewF;
    }
    llvm::errs() << "mod: " << *tobatch->getParent() << "\n";
    llvm::errs() << *tobatch << "\n";
    llvm_unreachable("attempting to batch function without definition");
  }

  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  ValueToValueMapTy originalToNewFn;

  // Create placeholder for the old arguments
  BasicBlock *placeholderBB =
      BasicBlock::Create(NewF->getContext(), "placeholders", NewF);

  IRBuilder<> PlaceholderBuilder(placeholderBB);
#if LLVM_VERSION_MAJOR >= 18
  auto It = PlaceholderBuilder.GetInsertPoint();
  It.setHeadBit(true);
  PlaceholderBuilder.SetInsertPoint(It);
#endif
  PlaceholderBuilder.SetCurrentDebugLocation(DebugLoc());
  ValueToValueMapTy vmap;
  auto DestArg = NewF->arg_begin();
  auto SrcArg = tobatch->arg_begin();

  for (unsigned i = 0; i < orig_FTy->getNumParams(); ++i) {
    Argument *arg = SrcArg;
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      auto placeholder = PlaceholderBuilder.CreatePHI(
          arg->getType(), 0, "placeholder." + arg->getName());
      vmap[arg] = placeholder;
    } else {
      vmap[arg] = DestArg;
    }
    DestArg->setName(arg->getName());
    DestArg++;
    SrcArg++;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, tobatch, vmap,
                    CloneFunctionChangeType::LocalChangesOnly, Returns, "",
                    nullptr);
#else
  CloneFunctionInto(NewF, tobatch, vmap, true, Returns, "", nullptr);
#endif

  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  // find instructions to vectorize (going up / overestimation)
  SmallPtrSet<Value *, 32> toVectorize;
  SetVector<llvm::Value *, std::deque<llvm::Value *>> refinelist;

  for (unsigned i = 0; i < tobatch->getFunctionType()->getNumParams(); i++) {
    if (arg_types[i] == BATCH_TYPE::VECTOR) {
      Argument *arg = tobatch->arg_begin() + i;
      toVectorize.insert(arg);
    }
  }

  for (auto &BB : *tobatch)
    for (auto &Inst : BB) {
      toVectorize.insert(&Inst);
      refinelist.insert(&Inst);
    }

  // find scalar instructions
  while (!refinelist.empty()) {
    Value *todo = *refinelist.begin();
    refinelist.erase(refinelist.begin());

    if (isa<ReturnInst>(todo) && ret_type == BATCH_TYPE::VECTOR)
      continue;

    if (auto branch_inst = dyn_cast<BranchInst>(todo)) {
      if (!branch_inst->isConditional()) {
        toVectorize.erase(todo);
        continue;
      }
    }

    if (auto call_inst = dyn_cast<CallInst>(todo)) {
      if (call_inst->getFunctionType()->isVoidTy() &&
          call_inst->getFunctionType()->getNumParams() == 0)
        toVectorize.erase(todo);
      continue;
    }

    if (auto todo_inst = dyn_cast<Instruction>(todo)) {

      if (todo_inst->mayReadOrWriteMemory())
        continue;

      if (isa<AllocaInst>(todo_inst))
        continue;

      SetVector<llvm::Value *, std::deque<llvm::Value *>> toCheck;
      toCheck.insert(todo_inst->op_begin(), todo_inst->op_end());
      SmallPtrSet<Value *, 8> safe;
      bool legal = true;
      while (!toCheck.empty()) {
        Value *cur = *toCheck.begin();
        toCheck.erase(toCheck.begin());

        if (!std::get<1>(safe.insert(cur)))
          continue;

        if (toVectorize.count(cur) == 0)
          continue;

        if (Instruction *cur_inst = dyn_cast<Instruction>(cur)) {
          if (!isa<CallInst>(cur_inst) && !cur_inst->mayReadOrWriteMemory()) {
            for (auto &op : cur_inst->operands())
              toCheck.insert(op);
            continue;
          }
        }

        legal = false;
        break;
      }

      if (legal)
        if (toVectorize.erase(todo))
          for (auto user : todo_inst->users())
            refinelist.insert(user);
    }
  }

  // unwrap arguments
  ValueMap<const Value *, std::vector<Value *>> vectorizedValues;
  auto entry = std::next(NewF->begin());
  IRBuilder<> Builder2(entry->getFirstNonPHI());
  Builder2.SetCurrentDebugLocation(DebugLoc());
  for (unsigned i = 0; i < FTy->getNumParams(); ++i) {
    Argument *orig_arg = tobatch->arg_begin() + i;
    Argument *arg = NewF->arg_begin() + i;

    if (arg_types[i] == BATCH_TYPE::SCALAR) {
      originalToNewFn[tobatch->arg_begin() + i] = arg;
      continue;
    }

    Instruction *placeholder = cast<Instruction>(vmap[orig_arg]);

    for (unsigned j = 0; j < width; ++j) {
      ExtractValueInst *argVecElem =
          cast<ExtractValueInst>(Builder2.CreateExtractValue(
              arg, {j},
              "unwrap" + (orig_arg->hasName()
                              ? "." + orig_arg->getName() + Twine(j)
                              : "")));
      if (j == 0) {
        placeholder->replaceAllUsesWith(argVecElem);
        placeholder->eraseFromParent();
      }
      vectorizedValues[orig_arg].push_back(argVecElem);
    }
  }

  placeholderBB->eraseFromParent();

  // update mapping with cloned basic blocks
  for (auto i = tobatch->begin(), j = NewF->begin();
       i != tobatch->end() && j != NewF->end(); ++i, ++j) {
    originalToNewFn[&*i] = &*j;
  }

  // update mapping with cloned scalar values and the first vectorized values
  auto J = inst_begin(NewF);
  // skip the unwrapped vector params
  std::advance(J, width * numVecParams);
  for (auto I = inst_begin(tobatch);
       I != inst_end(tobatch) && J != inst_end(NewF); ++I) {
    if (toVectorize.count(&*I) != 0) {
      vectorizedValues[&*I].push_back(&*J);
      ++J;
    } else {
      originalToNewFn[&*I] = &*J;
      ++J;
    }
  }

  // create placeholders for vector instructions 1..<n
  for (BasicBlock &BB : *tobatch) {
    for (Instruction &I : BB) {
      if (I.getType()->isVoidTy())
        continue;

      auto found = vectorizedValues.find(&I);
      if (found != vectorizedValues.end()) {
        Instruction *new_val_1 = cast<Instruction>(found->second.front());
        if (I.hasName())
          new_val_1->setName(I.getName() + "0");
        Instruction *insertPoint =
            new_val_1->getNextNode() ? new_val_1->getNextNode() : new_val_1;
        IRBuilder<> Builder2(insertPoint);
        Builder2.SetCurrentDebugLocation(DebugLoc());
#if LLVM_VERSION_MAJOR >= 18
        auto It = Builder2.GetInsertPoint();
        It.setHeadBit(true);
        Builder2.SetInsertPoint(It);
#endif
        for (unsigned i = 1; i < width; ++i) {
          PHINode *placeholder = Builder2.CreatePHI(I.getType(), 0);
          vectorizedValues[&I].push_back(placeholder);
          if (I.hasName())
            placeholder->setName("placeholder." + I.getName() + Twine(i));
        }
      }
    }
  }

  InstructionBatcher *batcher =
      new InstructionBatcher(tobatch, NewF, width, vectorizedValues,
                             originalToNewFn, toVectorize, *this);

  for (auto val : toVectorize) {
    if (auto inst = dyn_cast<Instruction>(val)) {
      batcher->visit(inst);
      if (batcher->hasError)
        break;
    }
  }

  if (batcher->hasError) {
    delete batcher;
    NewF->eraseFromParent();
    return BatchCachedFunctions[tup] = nullptr;
  }

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *tobatch << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (4)");
  }

  delete batcher;

  return BatchCachedFunctions[tup] = NewF;
};

llvm::Function *
EnzymeLogic::CreateTrace(RequestContext context, llvm::Function *totrace,
                         const SmallPtrSetImpl<Function *> &sampleFunctions,
                         const SmallPtrSetImpl<Function *> &observeFunctions,
                         const StringSet<> &ActiveRandomVariables,
                         ProbProgMode mode, bool autodiff,
                         TraceInterface *interface) {
  TraceCacheKey tup(totrace, mode, autodiff, interface);
  if (TraceCachedFunctions.find(tup) != TraceCachedFunctions.end()) {
    return TraceCachedFunctions.find(tup)->second;
  }

  // Determine generative functions
  SmallPtrSet<Function *, 4> GenerativeFunctions;
  SetVector<Function *, std::deque<Function *>> workList;
  workList.insert(sampleFunctions.begin(), sampleFunctions.end());
  workList.insert(observeFunctions.begin(), observeFunctions.end());
  GenerativeFunctions.insert(sampleFunctions.begin(), sampleFunctions.end());
  GenerativeFunctions.insert(observeFunctions.begin(), observeFunctions.end());

  while (!workList.empty()) {
    auto todo = *workList.begin();
    workList.erase(workList.begin());

    for (auto &&U : todo->uses()) {
      if (auto &&call = dyn_cast<CallBase>(U.getUser())) {
        auto &&fun = call->getParent()->getParent();
        auto &&[it, inserted] = GenerativeFunctions.insert(fun);
        if (inserted)
          workList.insert(fun);
      }
    }
  }

  ValueToValueMapTy originalToNewFn;
  TraceUtils *tutils =
      TraceUtils::FromClone(mode, sampleFunctions, observeFunctions, interface,
                            totrace, originalToNewFn);
  TraceGenerator *tracer =
      new TraceGenerator(*this, tutils, autodiff, originalToNewFn,
                         GenerativeFunctions, ActiveRandomVariables);

  if (totrace->empty()) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No tracer found for " + totrace->getName() << "\n";
    llvm::Value *toshow = totrace;
    if (context.req) {
      toshow = context.req;
      ss << " at context: " << *context.req;
    } else {
      ss << *totrace << "\n";
    }
    if (CustomErrorHandler) {
      CustomErrorHandler(ss.str().c_str(), wrap(toshow),
                         ErrorType::NoDerivative, nullptr, wrap(totrace),
                         wrap(context.ip));
      auto newFunc = tutils->newFunc;
      delete tracer;
      delete tutils;
      return newFunc;
    }
    if (context.req) {
      EmitFailure("NoDerivative", context.req->getDebugLoc(), context.req,
                  ss.str());
      auto newFunc = tutils->newFunc;
      delete tracer;
      delete tutils;
      return newFunc;
    }
    llvm::errs() << "mod: " << *totrace->getParent() << "\n";
    llvm::errs() << *totrace << "\n";
    llvm_unreachable("attempting to trace function without definition");
  }

  tracer->visit(totrace);

  if (verifyFunction(*tutils->newFunc, &errs())) {
    errs() << *totrace << "\n";
    errs() << *tutils->newFunc << "\n";
    report_fatal_error("function failed verification (4)");
  }

  Function *NewF = tutils->newFunc;

  delete tracer;
  delete tutils;

  if (!autodiff) {
    PPC.AlwaysInline(NewF);

    if (PostOpt)
      PPC.optimizeIntermediate(NewF);
    if (EnzymePrint) {
      errs() << *NewF << "\n";
    }
  }

  return TraceCachedFunctions[tup] = NewF;
}

llvm::Value *EnzymeLogic::CreateNoFree(RequestContext context,
                                       llvm::Value *todiff) {
  if (isa<InlineAsm>(todiff))
    return todiff;
  else if (auto F = dyn_cast<Function>(todiff))
    return CreateNoFree(context, F);
  if (auto castinst = dyn_cast<ConstantExpr>(todiff))
    if (castinst->isCast()) {
      llvm::Constant *reps[] = {
          cast<llvm::Constant>(CreateNoFree(context, castinst->getOperand(0)))};
      return castinst->getWithOperands(reps);
    }

  // Alloca/allocations are unsafe here since one could store freeing functions
  // into them. For now we will be unsafe regarding indirect function call
  // frees.
  if (isa<AllocaInst>(todiff))
    return todiff;

  std::string demangledCall;
  if (auto CI = dyn_cast<CallInst>(todiff)) {
    TargetLibraryInfo &TLI =
        PPC.FAM.getResult<TargetLibraryAnalysis>(*CI->getParent()->getParent());
    if (isAllocationFunction(getFuncNameFromCall(CI), TLI))
      return CI;
    if (auto F = CI->getCalledFunction()) {

      // clang-format off
      const char* NoFreeDemanglesStartsWith[] = {
          "std::basic_ostream<char, std::char_traits<char>>& std::__ostream_insert<char, std::char_traits<char>>",
          "std::basic_ostream<char, std::char_traits<char>>::operator<<",
          "std::ostream::operator<<",
          "std::ostream& std::ostream::_M_insert",
      };
      // clang-format on

      demangledCall = llvm::demangle(F->getName().str());
      // replace all '> >' with '>>'
      size_t start = 0;
      while ((start = demangledCall.find("> >", start)) != std::string::npos) {
        demangledCall.replace(start, 3, ">>");
      }

      for (auto Name : NoFreeDemanglesStartsWith)
        if (startsWith(demangledCall, Name))
          return CI;
    }
  }

  if (auto GV = dyn_cast<GlobalVariable>(todiff)) {
    if (GV->getName() == "_ZSt4cerr")
      return GV;
    if (GV->getName() == "_ZSt4cout")
      return GV;
  }

  if (context.ip) {
    if (auto LI = dyn_cast<LoadInst>(todiff)) {
      if (auto smpl = simplifyLoad(LI))
        return CreateNoFree(context, smpl);
      auto prev = CreateNoFree(context, LI->getPointerOperand());
      if (prev == LI->getPointerOperand())
        return todiff;
      auto res = cast<LoadInst>(context.ip->CreateLoad(LI->getType(), prev));
      res->copyMetadata(*LI);
      return res;
    }
    if (auto CI = dyn_cast<CastInst>(todiff)) {
      auto prev = CreateNoFree(context, CI->getOperand(0));
      if (prev == CI->getOperand(0))
        return todiff;
      auto res = cast<CastInst>(
          context.ip->CreateCast(CI->getOpcode(), prev, CI->getType()));
      res->copyMetadata(*CI);
      return res;
    }
    if (auto gep = dyn_cast<GetElementPtrInst>(todiff)) {
      if (gep->hasAllConstantIndices() || gep->isInBounds()) {
        auto prev = CreateNoFree(context, gep->getPointerOperand());
        if (prev == gep->getPointerOperand())
          return todiff;
        SmallVector<Value *, 1> idxs;
        for (auto &ind : gep->indices())
          idxs.push_back(ind);
        auto res = cast<GetElementPtrInst>(
            context.ip->CreateGEP(gep->getSourceElementType(), prev, idxs));
        res->setIsInBounds(gep->isInBounds());
        res->copyMetadata(*gep);
        return res;
      }
    }
  }

  if (EnzymeAssumeUnknownNoFree) {
    return todiff;
  }

  std::string s;
  llvm::raw_string_ostream ss(s);
  ss << "No create nofree of unknown value\n";
  ss << *todiff << "\n";
  if (demangledCall.size()) {
    ss << " demangled (" << demangledCall << ")\n";
  }
  if (context.req) {
    ss << " at context: " << *context.req;
  }
  if (auto I = dyn_cast<Instruction>(todiff)) {
    auto fname = I->getParent()->getParent()->getName();
    if (startsWith(fname, "nofree_"))
      fname = fname.substr(7);
    std::string demangledName = llvm::demangle(fname.str());
    // replace all '> >' with '>>'
    size_t start = 0;
    while ((start = demangledName.find("> >", start)) != std::string::npos) {
      demangledName.replace(start, 3, ">>");
    }
    ss << " within func " << fname << " (" << demangledName << ")\n";
  }
  if (EmitNoDerivativeError(ss.str(), todiff, context)) {
    return todiff;
  }

  llvm::errs() << s;
  llvm_unreachable("unhandled, create no free");
}

llvm::Function *EnzymeLogic::CreateNoFree(RequestContext context, Function *F) {
  if (NoFreeCachedFunctions.find(F) != NoFreeCachedFunctions.end()) {
    return NoFreeCachedFunctions.find(F)->second;
  }
  bool hasNoFree = false;
  hasNoFree |= F->hasFnAttribute(Attribute::NoFree);
  if (hasNoFree)
    return F;

  TargetLibraryInfo &TLI = PPC.FAM.getResult<TargetLibraryAnalysis>(*F);

  if (isAllocationFunction(F->getName(), TLI))
    return F;

  // clang-format off
  StringSet<> NoFreeDemangles = {
      "std::__u::locale::~locale())",
      "std::__u::locale::use_facet(std::__u::locale::id&) const",
      "std::__u::ios_base::getloc() const",
      "std::__u::ios_base::clear(unsigned int)",

      "std::basic_ostream<char, std::char_traits<char>>::basic_ostream(std::basic_streambuf<char, std::char_traits<char>>*)",
      "std::basic_ostream<char, std::char_traits<char>>::flush()",
      "std::basic_ostream<char, std::char_traits<char>>& std::__ostream_insert<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&)",
      "std::basic_ostream<char, std::char_traits<char>>::put(char)",
      "std::basic_ostream<char, std::char_traits<char>>::~basic_ostream()",

      "std::basic_filebuf<char, std::char_traits<char>>::basic_filebuf()",
      "std::basic_filebuf<char, std::char_traits<char>>::open(char const*, std::_Ios_Openmode)",
      "std::basic_filebuf<char, std::char_traits<char>>::close()",
      "std::basic_filebuf<char, std::char_traits<char>>::~basic_filebuf()",

      "std::__detail::_Prime_rehash_policy::_M_need_rehash(unsigned long, unsigned long, unsigned long) const",

      "std::basic_streambuf<char, std::char_traits<char> >::xsputn(char const*, long)",

      "std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char>>::basic_ostringstream()",
      "std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char>>::str() const",
      "std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char>>::~basic_ostringstream()",

      "std::basic_ios<char, std::char_traits<char> >::init(std::basic_streambuf<char, std::char_traits<char> >*)",
      "std::basic_ios<char, std::char_traits<char>>::clear(std::_Ios_Iostate)",
      "std::basic_ios<char, std::char_traits<char>>::operator bool() const",
      "std::basic_ios<char, std::char_traits<char>>::operator!() const",
      "std::basic_ios<wchar_t, std::char_traits<wchar_t>>::imbue(std::locale const&)",

      "std::_Hash_bytes(void const*, unsigned long, unsigned long)",
      "unsigned long std::__1::__do_string_hash<char const*>(char const*, char const*)",
      "std::__1::hash<char const*>::operator()(char const*) const",

      "std::allocator<char>::allocator()",
      "std::allocator<char>::~allocator()",

      "std::basic_ifstream<char, std::char_traits<char>>::is_open()",
      
      "std::basic_ofstream<char, std::char_traits<char>>::basic_ofstream(char const*, std::_Ios_Openmode)",
      "std::basic_ofstream<char, std::char_traits<char>>::is_open()",
      "std::basic_ofstream<char, std::char_traits<char>>::close()",
      "std::basic_ofstream<char, std::char_traits<char>>::~basic_ofstream()",

      "std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>::basic_stringstream(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, std::_Ios_Openmode)",
      "std::__cxx11::basic_stringstream<char, std::char_traits<char>, std::allocator<char>>::~basic_stringstream()",
      "std::basic_ostream<wchar_t, std::char_traits<wchar_t>>::put(wchar_t)",

      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::basic_string(char const*, std::allocator<char> const&)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::basic_string(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>&&)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::_M_construct(unsigned long, char)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::_M_append(char const*, unsigned long)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::_M_assign(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::_M_replace(unsigned long, unsigned long, char const*, unsigned long)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::_M_replace_aux(unsigned long, unsigned long, unsigned long, char)",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::length() const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::data() const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::size() const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::c_str() const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::~basic_string()",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::compare(char const*) const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::compare(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&) const",
      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::reserve(unsigned long)",

      "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>>::~basic_string()",
      "std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char>>::overflow(int)",
      "std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char>>::pbackfail(int)",
      "std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char>>::underflow()",
      "std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char>>::_M_sync(char*, unsigned long, unsigned long)",
      "std::__cxx11::basic_stringbuf<char, std::char_traits<char>, std::allocator<char>>::basic_stringbuf(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char>> const&, std::_Ios_Openmode)",

      "std::basic_streambuf<char, std::char_traits<char>>::pubsync()",
      "std::basic_ifstream<char, std::char_traits<char>>::close()",
      "std::istream::ignore()",
      "std::basic_ifstream<char, std::char_traits<char>>::basic_ifstream()",
      "std::basic_ifstream<char, std::char_traits<char>>::basic_ifstream(char const*, std::_Ios_Openmode)",
      "std::basic_ifstream<char, std::char_traits<char>>::~basic_ifstream()",
      "std::basic_ifstream<char, std::char_traits<char>>::rdbuf() const",
      "std::__basic_file<char>::is_open() const",
      "std::__basic_file<char>::~__basic_file()",

      "std::ostream::flush()",
      "std::basic_streambuf<char, std::char_traits<char>>::xsgetn(char*, long)",

      "std::locale::locale(char const*)",
      "std::locale::global(std::locale const&)",
      "std::locale::~locale()",
      "std::ios_base::ios_base()",
      "std::ios_base::~ios_base()",

      // libc++
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::basic_string(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&)",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::~basic_string()",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init(char const*, unsigned long)",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::append(char const*, unsigned long)",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::data() const",
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::sentry::sentry(std::__1::basic_ostream<char, std::__1::char_traits<char>>&)",
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::sentry::~sentry()",
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::flush()",
      "std::__1::ios_base::__set_badbit_and_consider_rethrow()",
      "char* std::__1::addressof<char>(char&)",
      "char const* std::__1::addressof<char const>(char const&)",
      "std::__1::random_device::operator()()",

      "std::__1::locale::~locale()",
      "std::__1::locale::use_facet(std::__1::locale::id&) const",
      "std::__1::ios_base::ios_base()",
      "std::__1::ios_base::getloc() const",
      "std::__1::ios_base::clear(unsigned int)",
      "std::__1::basic_iostream<char, std::__1::char_traits<char>>::~basic_iostream()",
      "std::__1::basic_ios<char, std::__1::char_traits<char>>::~basic_ios()",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::basic_streambuf()",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::~basic_streambuf()",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::imbue(std::__1::locale const&)",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::setbuf(char*, long)",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::sync()",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::showmanyc()",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::xsgetn(char*, long)",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::uflow()",
      "std::__1::basic_filebuf<char, std::__1::char_traits<char>>::basic_filebuf()",
      "std::__1::basic_filebuf<char, std::__1::char_traits<char>>::~basic_filebuf()",
      "std::__1::basic_filebuf<char, std::__1::char_traits<char>>::open(char const*, unsigned int)",
      "std::__1::basic_filebuf<char, std::__1::char_traits<char>>::close()",
      "std::__1::basic_filebuf<char, std::__1::char_traits<char>>::sync()",
      "std::__1::basic_istream<char, std::__1::char_traits<char>>::~basic_istream()",
      "virtual thunk to std::__1::basic_istream<char, std::__1::char_traits<char>>::~basic_istream()",
      "virtual thunk to std::__1::basic_ostream<char, std::__1::char_traits<char>>::~basic_ostream()",
      "std::__1::basic_ifstream<char, std::__1::char_traits<char>>::~basic_ifstream()",
      "std::__1::ios_base::init(void*)",
      "std::__1::basic_istream<char, std::__1::char_traits<char>>::read(char*, long)",
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::~basic_ostream()",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init(unsigned long, char)",
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::write(char const*, long)",
  };
  const char* NoFreeDemanglesStartsWith[] = {
      "std::__1::basic_ostream<char, std::__1::char_traits<char>>::operator<<",
      "std::__1::ios_base::imbue",
      "std::__1::basic_streambuf<wchar_t, std::__1::char_traits<wchar_t>>::pubimbue",
      "std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::__init_buf_ptrs",
      "std::__1::basic_stringbuf<char, std::__1::char_traits<char>, std::__1::allocator<char>>::basic_stringbuf",
      "std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>::operator=",
      "std::__1::ctype<char>::widen",
      "std::__1::basic_streambuf<char, std::__1::char_traits<char>>::sputn",
      "std::basic_ostream<char, std::char_traits<char>>& std::flush",
      "std::basic_ostream<char, std::char_traits<char>>& std::operator<<",
      "std::basic_ostream<char, std::char_traits<char>>& std::basic_ostream<char, std::char_traits<char>>::_M_insert",
      "std::basic_ostream<char, std::char_traits<char>>& std::__ostream_insert<char, std::char_traits<char>>",
      "std::basic_ostream<wchar_t, std::char_traits<wchar_t>>& std::operator<<",
      "std::basic_ostream<wchar_t, std::char_traits<wchar_t>>::operator<<",
      "std::basic_ostream<wchar_t, std::char_traits<wchar_t>>& std::basic_ostream<wchar_t, std::char_traits<wchar_t>>::_M_insert",
      "std::istream::get",
      "std::ostream::put",
      "std::ostream::write",
      "std::ostream& std::ostream::_M_insert",
      "std::istream::read",
      "std::istream::operator>>",
      "std::basic_streambuf<char, std::char_traits<char>>::pubsetbuf",
      "std::basic_streambuf<char, std::char_traits<char>>::sputn",
      "std::istream& std::istream::_M_extract",
      "std::ctype<char>::widen",
      //Rust
      "std::io::stdio::_eprint",
  };

  StringSet<> NoFrees = {"mpfr_greater_p",
                        "fprintf",
                        "fputc",
                         "memchr",
                         "time",
                         "strlen",
                         "__cxa_begin_catch",
                         "__cxa_guard_acquire",
                         "__cxa_guard_release",
                         "__cxa_end_catch",
                         "compress2",
                         "malloc_usable_size",
                         "MPI_Allreduce",
                         "lgamma",
                         "lgamma_r",
                         "__kmpc_global_thread_num",
                         "nlopt_force_stop",
                         "cudaRuntimeGetVersion"
  };
  // clang-format on

  if (startsWith(F->getName(), "_ZNSolsE") || NoFrees.count(F->getName()))
    return F;

  std::string demangledName = llvm::demangle(F->getName().str());
  // replace all '> >' with '>>'
  size_t start = 0;
  while ((start = demangledName.find("> >", start)) != std::string::npos) {
    demangledName.replace(start, 3, ">>");
  }
  if (NoFreeDemangles.count(demangledName))
    return F;

  for (auto Name : NoFreeDemanglesStartsWith)
    if (startsWith(demangledName, Name))
      return F;

  switch (F->getIntrinsicID()) {
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
  case Intrinsic::cttz:
  case Intrinsic::ctlz:
    return F;
  default:;
  }

  {
    Intrinsic::ID ID = Intrinsic::not_intrinsic;
    if (isMemFreeLibMFunction(getFuncName(F), &ID))
      return F;
  }

  if (F->empty()) {
    if (EnzymeAssumeUnknownNoFree) {
      return F;
    }
    if (EnzymeEmptyFnInactive) {
      return F;
    }
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << "No create nofree of empty function (" << demangledName << ") "
       << F->getName() << ")\n";
    if (context.req) {
      ss << " at context: " << *context.req;
      if (auto CB = dyn_cast<CallBase>(context.req)) {
        if (auto F = CB->getCalledFunction()) {
          std::string demangleF = llvm::demangle(F->getName().str());
          // replace all '> >' with '>>'
          size_t start = 0;
          while ((start = demangleF.find("> >", start)) != std::string::npos) {
            demangleF.replace(start, 3, ">>");
          }
          ss << " (" << demangleF << ")";
        }
      }
    } else {
      ss << *F << "\n";
    }
    if (EmitNoDerivativeError(ss.str(), F, context)) {
      return F;
    }
    llvm::errs() << " unhandled, create no free of empty function: " << *F
                 << "\n";
    llvm_unreachable("unhandled, create no free");
  }

  Function *NewF = Function::Create(F->getFunctionType(), F->getLinkage(),
                                    "nofree_" + F->getName(), F->getParent());
  NewF->setAttributes(F->getAttributes());
  NewF->addAttribute(AttributeList::FunctionIndex,
                     Attribute::get(NewF->getContext(), Attribute::NoFree));

  NoFreeCachedFunctions[F] = NewF;

  ValueToValueMapTy VMap;

  for (auto i = F->arg_begin(), j = NewF->arg_begin(); i != F->arg_end();) {
    VMap[i] = j;
    j->setName(i->getName());
    ++j;
    ++i;
  }

  SmallVector<ReturnInst *, 4> Returns;
#if LLVM_VERSION_MAJOR >= 13
  CloneFunctionInto(NewF, F, VMap, CloneFunctionChangeType::LocalChangesOnly,
                    Returns, "", nullptr);
#else
  CloneFunctionInto(NewF, F, VMap, true, Returns, "", nullptr);
#endif

  NewF->setVisibility(llvm::GlobalValue::DefaultVisibility);
  NewF->setLinkage(llvm::GlobalValue::InternalLinkage);

  const SmallPtrSet<BasicBlock *, 4> guaranteedUnreachable =
      getGuaranteedUnreachable(NewF);

  SmallVector<Instruction *, 2> toErase;
  for (BasicBlock &BB : *NewF) {
    if (guaranteedUnreachable.count(&BB))
      continue;
    for (Instruction &I : BB) {
      StringRef funcName = "";
      if (auto CI = dyn_cast<CallInst>(&I)) {
        if (CI->hasFnAttr(Attribute::NoFree))
          continue;
        funcName = getFuncNameFromCall(CI);
      }
      if (auto CI = dyn_cast<InvokeInst>(&I)) {
        if (CI->hasFnAttr(Attribute::NoFree))
          continue;
        funcName = getFuncNameFromCall(CI);
      }
      if (isDeallocationFunction(funcName, TLI))
        toErase.push_back(&I);
      else {
        if (auto CI = dyn_cast<CallInst>(&I)) {
          auto callval = CI->getCalledOperand();
          CI->setCalledOperand(CreateNoFree(context, callval));
        }
        if (auto CI = dyn_cast<InvokeInst>(&I)) {
          auto callval = CI->getCalledOperand();
          CI->setCalledOperand(CreateNoFree(context, callval));
        }
      }
    }
  }
  NewF->setLinkage(Function::LinkageTypes::InternalLinkage);

  if (llvm::verifyFunction(*NewF, &llvm::errs())) {
    llvm::errs() << *F << "\n";
    llvm::errs() << *NewF << "\n";
    report_fatal_error("function failed verification (4)");
  }

  for (auto E : toErase) {
    E->eraseFromParent();
  }

  return NewF;
}

void EnzymeLogic::clear() {
  PPC.clear();
  AugmentedCachedFunctions.clear();
  ReverseCachedFunctions.clear();
  NoFreeCachedFunctions.clear();
  ForwardCachedFunctions.clear();
  BatchCachedFunctions.clear();
}
