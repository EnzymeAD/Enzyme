//===- ActivityAnalysis.cpp - Implementation of Activity Analysis ---------===//
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
// This file contains the implementation of Activity Analysis -- an AD-specific
// analysis that deduces if a given instruction or value can impact the
// calculation of a derivative. This file consists of two mutually recurive
// functions that compute this for values and instructions, respectively.
//
//===----------------------------------------------------------------------===//
#include <cstdint>
#include <deque>

#include <llvm/Config/llvm-config.h>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"

#include "llvm/IR/InstIterator.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/IR/InlineAsm.h"

#include "ActivityAnalysis.h"
#include "Utils.h"

#include "LibraryFuncs.h"
#include "TypeAnalysis/TBAA.h"

#include "llvm/Analysis/ValueTracking.h"

using namespace llvm;

cl::opt<bool> printconst("enzyme-print-activity", cl::init(false), cl::Hidden,
                         cl::desc("Print activity analysis algorithm"));

cl::opt<bool> nonmarkedglobals_inactive(
    "enzyme-globals-default-inactive", cl::init(false), cl::Hidden,
    cl::desc("Consider all nonmarked globals to be inactive"));

cl::opt<bool> emptyfnconst("enzyme-emptyfn-inactive", cl::init(false),
                           cl::Hidden,
                           cl::desc("Empty functions are considered inactive"));

#include "llvm/IR/InstIterator.h"
#include <map>
#include <set>
#include <unordered_map>

/// Is the use of value val as an argument of call CI potentially captured
static inline bool couldFunctionArgumentCapture(CallInst *CI, Value *val) {
  Function *F = CI->getCalledFunction();
  if (F == nullptr)
    return true;

  if (F->getIntrinsicID() == Intrinsic::memset)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memcpy)
    return false;
  if (F->getIntrinsicID() == Intrinsic::memmove)
    return false;

  if (F->empty())
    return false;

  auto arg = F->arg_begin();
  for (size_t i = 0, size = CI->getNumArgOperands(); i < size; i++) {
    if (val == CI->getArgOperand(i)) {
      // This is a vararg, assume captured
      if (arg == F->arg_end()) {
        return true;
      } else {
        if (!arg->hasNoCaptureAttr()) {
          return true;
        }
      }
    }
    if (arg != F->arg_end())
      arg++;
  }
  // No argument captured
  return false;
}

const char *KnownInactiveFunctionsStartingWith[] = {"_ZN4core3fmt",
                                                    "_ZN3std2io5stdio6_print",
                                                    "f90io"};

const char *KnownInactiveFunctions[] = {"__assert_fail",
                                        "__cxa_guard_acquire",
                                        "__cxa_guard_release",
                                        "__cxa_guard_abort",
                                        "posix_memalign",
                                        "printf",
                                        "vprintf",
                                        "puts",
                                        "__enzyme_float",
                                        "__enzyme_double",
                                        "__enzyme_integer",
                                        "__enzyme_pointer",
                                        "__kmpc_for_static_init_4",
                                        "__kmpc_for_static_init_4u",
                                        "__kmpc_for_static_init_8",
                                        "__kmpc_for_static_init_8u",
                                        "__kmpc_for_static_fini",
                                        "__kmpc_dispatch_init_4",
                                        "__kmpc_dispatch_init_4u",
                                        "__kmpc_dispatch_init_8",
                                        "__kmpc_dispatch_init_8u",
                                        "__kmpc_dispatch_next_4",
                                        "__kmpc_dispatch_next_4u",
                                        "__kmpc_dispatch_next_8",
                                        "__kmpc_dispatch_next_8u",
                                        "__kmpc_dispatch_fini_4",
                                        "__kmpc_dispatch_fini_4u",
                                        "__kmpc_dispatch_fini_8",
                                        "__kmpc_dispatch_fini_8u",
                                        "malloc_usable_size",
                                        "malloc_size",
                                        "MPI_Init",
                                        "MPI_Comm_size",
                                        "MPI_Comm_rank",
                                        "MPI_Get_processor_name",
                                        "MPI_Finalize",
                                        "_msize"};

/// Is the use of value val as an argument of call CI known to be inactive
/// This tool can only be used when in DOWN mode
bool ActivityAnalyzer::isFunctionArgumentConstant(CallInst *CI, Value *val) {
  assert(directions & DOWN);

  if (CI->hasFnAttr("enzyme_inactive")) {
    return true;
  }

  Function *F = CI->getCalledFunction();

  // Indirect function calls may actively use the argument
  if (F == nullptr)
    return false;

  auto Name = F->getName();

  // Allocations, deallocations, and c++ guards don't impact the activity
  // of arguments
  if (isAllocationFunction(*F, TLI) || isDeallocationFunction(*F, TLI))
    return true;

  for (auto FuncName : KnownInactiveFunctionsStartingWith) {
    if (Name.startswith(FuncName)) {
      return true;
    }
  }
  for (auto FuncName : KnownInactiveFunctions) {
    if (Name == FuncName)
      return true;
  }
  if (F->getIntrinsicID() == Intrinsic::trap)
    return true;

  /// Only the first argument (magnitude) of copysign is active
  if (F->getIntrinsicID() == Intrinsic::copysign &&
      CI->getArgOperand(0) != val) {
    return true;
  }

  /// Use of the value as a non-src/dst in memset/memcpy/memmove is an inactive
  /// use
  if (F->getIntrinsicID() == Intrinsic::memset && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memcpy && CI->getArgOperand(0) != val &&
      CI->getArgOperand(1) != val)
    return true;
  if (F->getIntrinsicID() == Intrinsic::memmove &&
      CI->getArgOperand(0) != val && CI->getArgOperand(1) != val)
    return true;

  // TODO interprocedural detection
  // Before potential introprocedural detection, any function without definition
  // may to be assumed to have an active use
  if (F->empty())
    return false;

  // With all other options exhausted we have to assume this function could
  // actively use the value
  return false;
}

/// Call the function propagateFromOperand on all operands of CI
/// that could impact the activity of the call instruction
static inline void propagateArgumentInformation(
    TargetLibraryInfo &TLI, CallInst &CI,
    std::function<bool(Value *)> propagateFromOperand) {

  if (auto F = CI.getCalledFunction()) {
    // These functions are known to only have the first argument impact
    // the activity of the call instruction
    auto Name = F->getName();
    if (Name == "lgamma" || Name == "lgammaf" || Name == "lgammal" ||
        Name == "lgamma_r" || Name == "lgammaf_r" || Name == "lgammal_r" ||
        Name == "__lgamma_r_finite" || Name == "__lgammaf_r_finite" ||
        Name == "__lgammal_r_finite") {

      propagateFromOperand(CI.getArgOperand(0));
      return;
    }

    // Allocations, deallocations, and c++ guards are fully inactive
    if (isAllocationFunction(*F, TLI) || isDeallocationFunction(*F, TLI) ||
        Name == "__cxa_guard_acquire" || Name == "__cxa_guard_release" ||
        Name == "__cxa_guard_abort")
      return;

    /// Only the src/dst in memset/memcpy/memmove impact the activity of the
    /// instruction
    if (auto F = CI.getCalledFunction()) {
      // memset cannot propagate activity as it sets
      // data to a given single byte which is inactive
      if (F->getIntrinsicID() == Intrinsic::memset) {
        return;
      }
      if (F->getIntrinsicID() == Intrinsic::memcpy ||
          F->getIntrinsicID() == Intrinsic::memmove) {
        propagateFromOperand(CI.getOperand(0));
        propagateFromOperand(CI.getOperand(1));
        return;
      }
    }
  }

  // For other calls, check all operands of the instruction
  // as conservatively they may impact the activity of the call
  for (auto &a : CI.arg_operands()) {
    if (propagateFromOperand(a))
      break;
  }
}

/// Return whether this instruction is known not to propagate adjoints
/// Note that instructions could return an active pointer, but
/// do not propagate adjoints themselves
bool ActivityAnalyzer::isConstantInstruction(TypeResults &TR, Instruction *I) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo
  assert(I);
  assert(TR.info.Function == I->getParent()->getParent());

  // The return instruction doesn't impact activity (handled specifically
  // during adjoint generation)
  if (isa<ReturnInst>(I))
    return true;

  // Branch, unreachable, and previously computed constants are inactive
  if (isa<UnreachableInst>(I) || isa<BranchInst>(I) ||
      (ConstantInstructions.find(I) != ConstantInstructions.end())) {
    return true;
  }

  /// Previously computed inactives remain inactive
  if ((ActiveInstructions.find(I) != ActiveInstructions.end())) {
    return false;
  }

  /// A store into all integral memory is inactive
  if (auto SI = dyn_cast<StoreInst>(I)) {
    auto StoreSize = SI->getParent()
                         ->getParent()
                         ->getParent()
                         ->getDataLayout()
                         .getTypeSizeInBits(SI->getValueOperand()->getType()) /
                     8;

    bool AllIntegral = true;
    bool SeenInteger = false;
    auto q = TR.query(SI->getPointerOperand()).Data0();
    for (int i = -1; i < (int)StoreSize; ++i) {
      auto dt = q[{i}];
      if (dt.isIntegral() || dt == BaseType::Anything) {
        SeenInteger = true;
      } else if (dt.isKnown()) {
        AllIntegral = false;
        break;
      }
    }

    if (AllIntegral && SeenInteger) {
      if (printconst)
        llvm::errs() << " constant instruction from TA " << *I << "\n";
      ConstantInstructions.insert(I);
      return true;
    }
  }

  if (isa<MemSetInst>(I)) {
    // memset's are definitionally inactive since
    // they copy a byte which cannot be active
    if (printconst)
      llvm::errs() << " constant instruction as memset " << *I << "\n";
    ConstantInstructions.insert(I);
    return true;
  }

  if (printconst)
    llvm::errs() << "checking if is constant[" << (int)directions << "] " << *I
                 << "\n";

  // Analyzer for inductive assumption where we attempt to prove this is
  // inactive from a lack of active users
  std::shared_ptr<ActivityAnalyzer> DownHypothesis;

  // If this instruction does not write to memory that outlives itself
  // (potentially propagating derivative information), the only way to propagate
  // derivative information is through the return value
  // TODO the "doesn't write to active memory" can be made more aggressive than
  // doesn't write to any memory
  if (!I->mayWriteToMemory() ||
      (isa<CallInst>(I) && AA.onlyReadsMemory(cast<CallInst>(I))) ||
      (isa<CallInst>(I) && cast<CallInst>(I)->getFunction() &&
       isMemFreeLibMFunction(cast<CallInst>(I)->getFunction()->getName()))) {

    // Even if returning a pointer, this instruction is considered inactive
    // since the instruction doesn't prop gradients. Thus, so long as we don't
    // return an object containing a float, this instruction is inactive
    if (!TR.intType(1, I, /*errifNotFound*/ false).isPossibleFloat()) {
      if (printconst)
        llvm::errs()
            << " constant instruction from known non-float non-writing "
               "instruction "
            << *I << "\n";
      ConstantInstructions.insert(I);
      return true;
    }

    // If the value returned is constant otherwise, the instruction is inactive
    if (isConstantValue(TR, I)) {
      if (printconst)
        llvm::errs() << " constant instruction from known constant non-writing "
                        "instruction "
                     << *I << "\n";
      ConstantInstructions.insert(I);
      return true;
    }

    // Even if the return is nonconstant, it's worth checking explicitly the
    // users since unlike isConstantValue, returning a pointer does not make the
    // instruction active
    if (directions & DOWN) {
      // We shall now induct on this instruction being inactive and try to prove
      // this fact from a lack of active users.

      // If we aren't a phi node (and thus potentially recursive on uses) and
      // already equal to the current direction, we don't need to induct,
      // reducing runtime.
      if (directions == DOWN && !isa<PHINode>(I)) {
        if (isValueInactiveFromUsers(TR, I)) {
          if (printconst)
            llvm::errs() << " constant instruction[" << directions
                         << "] from users instruction " << *I << "\n";
          ConstantInstructions.insert(I);
          return true;
        }
      } else {
        DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, DOWN));
        DownHypothesis->ConstantInstructions.insert(I);
        if (DownHypothesis->isValueInactiveFromUsers(TR, I)) {
          if (printconst)
            llvm::errs() << " constant instruction[" << directions
                         << "] from users instruction " << *I << "\n";
          ConstantInstructions.insert(I);
          insertConstantsFrom(*DownHypothesis);
          return true;
        }
      }
    }
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;
  if (directions & UP) {
    // If this instruction has no active operands, the instruction
    // is active.
    // TODO This isn't 100% accurate and will incorrectly mark a no-argument
    // function that reads from active memory as constant
    // Technically the additional constraint is that this does not read from
    // active memory, where we have assumed that the only active memory
    // we care about is accessible from arguments passed (and thus not globals)
    UpHypothesis =
        std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
    UpHypothesis->ConstantInstructions.insert(I);
    assert(directions & UP);
    if (UpHypothesis->isInstructionInactiveFromOrigin(TR, I)) {
      if (printconst)
        llvm::errs() << " constant instruction from origin "
                        "instruction "
                     << *I << "\n";
      ConstantInstructions.insert(I);
      insertConstantsFrom(*UpHypothesis);
      if (DownHypothesis)
        insertConstantsFrom(*DownHypothesis);
      return true;
    }
  }

  // Otherwise we must fall back and assume this instruction to be active.
  ActiveInstructions.insert(I);
  if (printconst)
    llvm::errs() << "couldnt decide fallback as nonconstant instruction("
                 << (int)directions << "):" << *I << "\n";
  return false;
}

bool ActivityAnalyzer::isConstantValue(TypeResults &TR, Value *Val) {
  // This analysis may only be called by instructions corresponding to
  // the function analyzed by TypeInfo -- however if the Value
  // was created outside a function (e.g. global, constant), that is allowed
  assert(Val);
  if (auto I = dyn_cast<Instruction>(Val)) {
    assert(TR.info.Function == I->getParent()->getParent());
  }
  if (auto Arg = dyn_cast<Argument>(Val)) {
    assert(TR.info.Function == Arg->getParent());
  }

  // Void values are definitionally inactive
  if (Val->getType()->isVoidTy())
    return true;

  // All function pointers are considered active in case an augmented primal
  // or reverse is needed
  if (isa<Function>(Val)) {
    return false;
  }

  // Undef, metadata, non-global constants, and blocks are inactive
  if (isa<UndefValue>(Val) || isa<MetadataAsValue>(Val) ||
      isa<ConstantData>(Val) || isa<ConstantAggregate>(Val) ||
      isa<BasicBlock>(Val)) {
    return true;
  }
  assert(!isa<InlineAsm>(Val));

  if (auto II = dyn_cast<IntrinsicInst>(Val)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_barrier0:
    case Intrinsic::nvvm_barrier0_popc:
    case Intrinsic::nvvm_barrier0_and:
    case Intrinsic::nvvm_barrier0_or:
    case Intrinsic::nvvm_membar_cta:
    case Intrinsic::nvvm_membar_gl:
    case Intrinsic::nvvm_membar_sys:
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
    case Intrinsic::prefetch:
#if LLVM_VERSION_MAJOR >= 8
    case Intrinsic::is_constant:
#endif
      return true;
    default:
      break;
    }
  }

  /// If we've already shown this value to be inactive
  if (ConstantValues.find(Val) != ConstantValues.end()) {
    return true;
  }

  /// If we've already shown this value to be active
  if (ActiveValues.find(Val) != ActiveValues.end()) {
    return false;
  }

  // All arguments must be marked constant/nonconstant ahead of time
  if (isa<Argument>(Val)) {
    llvm::errs() << *(cast<Argument>(Val)->getParent()) << "\n";
    llvm::errs() << *Val << "\n";
    assert(0 && "must've put arguments in constant/nonconstant");
  }

  // This value is certainly an integer (and only and integer, not a pointer or
  // float). Therefore its value is constant
  if (TR.intType(1, Val, /*errIfNotFound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral " << (int)directions << " "
                   << *Val << " "
                   << TR.intType(1, Val, /*errIfNotFound*/ false).str() << "\n";
    ConstantValues.insert(Val);
    return true;
  }

#if 0
  // This value is certainly a pointer to an integer (and only and integer, not
  // a pointer or float). Therefore its value is constant
  // TODO use typeInfo for more aggressive activity analysis
  if (val->getType()->isPointerTy() &&
      cast<PointerType>(val->getType())->isIntOrIntVectorTy() &&
      TR.firstPointer(1, val, /*errifnotfound*/ false).isIntegral()) {
    if (printconst)
      llvm::errs() << " Value const as integral pointer" << (int)directions
                   << " " << *val << "\n";
    ConstantValues.insert(val);
    return true;
  }
#endif

  if (auto GI = dyn_cast<GlobalVariable>(Val)) {
    // If operating under the assumption globals are inactive unless
    // explicitly marked as active, this is inactive
    if (!hasMetadata(GI, "enzyme_shadow") && nonmarkedglobals_inactive) {
      ConstantValues.insert(Val);
      return true;
    }

    // If this global is unchanging and the internal constant data
    // is inactive, the global is inactive
    if (GI->isConstant() && isConstantValue(TR, GI->getInitializer())) {
      ConstantValues.insert(Val);
      if (printconst)
        llvm::errs() << " VALUE const global " << *Val << "\n";
      return true;
    }

    // If this global is a pointer to an integer, it is inactive
    // TODO note this may need updating to consider the size
    // of the global
    auto res = TR.query(GI).Data0();
    auto dt = res[{-1}];
    dt |= res[{0}];
    if (dt.isIntegral()) {
      if (printconst)
        llvm::errs() << " VALUE const as global int pointer " << *Val
                     << " type - " << res.str() << "\n";
      return true;
    }

    // If this is a global local to this translation unit with inactive
    // initializer and no active uses, it is definitionally inactive
    bool usedJustInThisModule = GI->hasInternalLinkage() ||
                                GI->hasPrivateLinkage() ||
                                GI->hasPrivateLinkage();

    if (printconst)
      llvm::errs() << "pre attempting just used in module for: " << *GI
                   << " dir" << (char)directions
                   << " justusedin:" << usedJustInThisModule << "\n";

    if (directions == 3 && usedJustInThisModule) {
      // TODO this assumes global initializer cannot refer to itself (lest
      // infinite loop)
      if (!GI->hasInitializer() || isConstantValue(TR, GI->getInitializer())) {

        if (printconst)
          llvm::errs() << "attempting just used in module for: " << *GI << "\n";
        // Not looking at users to prove inactive (definition of down)
        // If all users are inactive, this is therefore inactive.
        // Since we won't look at origins to prove, we can inductively assume
        // this is inactive

        // As an optimization if we are going down already
        // and we won't use ourselves (done by PHI's), we
        // dont need to inductively assume we're true
        // and can instead use this object!
        if (directions == DOWN) {
          if (isValueInactiveFromUsers(TR, Val)) {
            ConstantValues.insert(Val);
            return true;
          }
        } else {
          auto DownHypothesis = std::shared_ptr<ActivityAnalyzer>(
              new ActivityAnalyzer(*this, DOWN));
          DownHypothesis->ConstantValues.insert(Val);
          if (DownHypothesis->isValueInactiveFromUsers(TR, Val)) {
            insertConstantsFrom(*DownHypothesis);
            ConstantValues.insert(Val);
            return true;
          }
        }
      }
    }

    // Otherwise we have to assume this global is active since it can
    // be arbitrarily used in an active way
    // TODO we can be more aggressive here in the future
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown global " << *Val << " type - "
                   << res.str() << "\n";
    return false;
  }

  // ConstantExpr's are inactive if their arguments are inactive
  // Note that since there can't be a recursive constant this shouldn't
  // infinite loop
  if (auto ce = dyn_cast<ConstantExpr>(Val)) {
    if (ce->isCast()) {
      if (isConstantValue(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from from operand " << *Val
                       << "\n";
        ConstantValues.insert(Val);
        return true;
      }
    }
    if (ce->isGEPWithNoNotionalOverIndexing()) {
      if (isConstantValue(TR, ce->getOperand(0))) {
        if (printconst)
          llvm::errs() << " VALUE const cast from gep operand " << *Val << "\n";
        ConstantValues.insert(Val);
        return true;
      }
    }
    if (printconst)
      llvm::errs() << " VALUE nonconst unknown expr " << *Val << "\n";
    return false;
  }

  std::shared_ptr<ActivityAnalyzer> UpHypothesis;

  // Handle types that could contain pointers
  //  Consider all types except
  //   * floating point types (since those are assumed not pointers)
  //   * integers that we know are not pointers
  bool containsPointer = true;
  if (Val->getType()->isFPOrFPVectorTy())
    containsPointer = false;
  if (!TR.intType(1, Val, /*errIfNotFound*/ false).isPossiblePointer())
    containsPointer = false;

  if (containsPointer) {

    auto TmpOrig =
#if LLVM_VERSION_MAJOR >= 12
        getUnderlyingObject(Val, 100);
#else
        GetUnderlyingObject(Val, TR.info.Function->getParent()->getDataLayout(),
                            100);
#endif

    // If we know that our origin is inactive from its arguments,
    // we are definitionally inactive
    if (directions & UP) {
      // If we are derived from an argument our activity is equal to the
      // activity of the argument by definition
      if (isa<Argument>(TmpOrig)) {
        bool res = isConstantValue(TR, TmpOrig);
        if (res) {
          ConstantValues.insert(Val);
        } else {
          ActiveValues.insert(Val);
        }
        return res;
      }

      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);

      // If our origin is a load of a known inactive (say inactive argument), we
      // are also inactive
      if (auto LI = dyn_cast<LoadInst>(TmpOrig)) {

        if (directions == UP) {
          if (isConstantValue(TR, LI->getPointerOperand())) {
            ConstantValues.insert(Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, LI->getPointerOperand())) {
            ConstantValues.insert(Val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }
        }
      } else if (auto op = dyn_cast<CallInst>(TmpOrig)) {
        if (op->hasFnAttr("enzyme_inactive")) {
          ConstantValues.insert(Val);
          insertConstantsFrom(*UpHypothesis);
          return true;
        }
        if (auto called = op->getCalledFunction()) {
          if (called->getName() == "free" || called->getName() == "_ZdlPv" ||
              called->getName() == "_ZdlPvm" || called->getName() == "munmap") {
            ConstantValues.insert(Val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }

          for (auto FuncName : KnownInactiveFunctionsStartingWith) {
            if (called->getName().startswith(FuncName)) {
              ConstantValues.insert(Val);
              insertConstantsFrom(*UpHypothesis);
              return true;
            }
          }
          for (auto FuncName : KnownInactiveFunctions) {
            if (called->getName() == FuncName) {
              ConstantValues.insert(Val);
              insertConstantsFrom(*UpHypothesis);
              return true;
            }
          }

          if (called->getIntrinsicID() == Intrinsic::trap) {
            ConstantValues.insert(Val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }

          // If requesting emptty unknown functions to be considered inactive, abide
          // by those rules
          if (!isCertainPrintMallocOrFree(called) && called->empty() &&
              !hasMetadata(called, "enzyme_gradient") && !isa<IntrinsicInst>(op) &&
              emptyfnconst) {
            ConstantValues.insert(Val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }
        }
      }

      // otherwise if the origin is a previously derived known inactive value
      // assess
      // TODO here we would need to potentially consider loading an active
      // global as we again assume that active memory is passed explicitly as an
      // argument
      if (TmpOrig != Val) {
        if (directions == UP && !isa<PHINode>(TmpOrig)) {
          if (isConstantValue(TR, TmpOrig)) {
            ConstantValues.insert(Val);
            return true;
          }
        } else {
          if (UpHypothesis->isConstantValue(TR, TmpOrig)) {
            ConstantValues.insert(Val);
            insertConstantsFrom(*UpHypothesis);
            return true;
          }
        }
      }
      if (auto inst = dyn_cast<Instruction>(Val)) {
        if (!inst->mayReadFromMemory() && !isa<AllocaInst>(Val)) {
          if (directions == UP && !isa<PHINode>(inst)) {
            if (isInstructionInactiveFromOrigin(TR, inst)) {
              ConstantValues.insert(Val);
              return true;
            }
          } else {
            if (UpHypothesis->isInstructionInactiveFromOrigin(TR, inst)) {
              ConstantValues.insert(Val);
              insertConstantsFrom(*UpHypothesis);
              return true;
            }
          }
        }
      }
    }

    // If not capable of looking at both users and uses, all the ways a pointer
    // can be loaded/stored cannot be assesed and therefore we default to assume
    // it to be active
    if (directions != 3) {
      if (printconst)
        llvm::errs() << " <Potential Pointer assumed active at "
                     << (int)directions << ">" << *Val << "\n";
      ActiveValues.insert(Val);
      return false;
    }

    if (printconst)
      llvm::errs() << " < MEMSEARCH" << (int)directions << ">" << *Val << "\n";
    // A pointer value is active if two things hold:
    //   an potentially active value is stored into the memory
    //   memory loaded from the value is used in an active way
    bool potentialStore = false;
    bool potentiallyActiveLoad = false;

    // Assume the value (not instruction) is itself active
    // In spite of that can we show that there are either no active stores
    // or no active loads
    std::shared_ptr<ActivityAnalyzer> Hypothesis =
        std::shared_ptr<ActivityAnalyzer>(
            new ActivityAnalyzer(*this, directions));
    Hypothesis->ActiveValues.insert(Val);

    if (isa<Instruction>(Val) || isa<Argument>(Val)) {
      // These are handled by iterating through all
    } else {
      llvm::errs() << "unknown pointer value type: " << *Val << "\n";
      assert(0 && "unknown pointer value type");
      llvm_unreachable("unknown pointer value type");
    }

    // Search through all the instructions in this function
    // for potential loads / stores of this value
    for (BasicBlock &BB : *TR.info.Function) {
      if (potentialStore && potentiallyActiveLoad)
        break;
      for (Instruction &I : BB) {
        if (potentialStore && potentiallyActiveLoad)
          break;

        // If this is a malloc or free, this doesn't impact the activity
        if (auto CI = dyn_cast<CallInst>(&I)) {
          if (auto F = CI->getCalledFunction()) {
            if (isAllocationFunction(*F, TLI) ||
                isDeallocationFunction(*F, TLI)) {
              continue;
            }
            if (F->getName() == "__cxa_guard_acquire" ||
                F->getName() == "__cxa_guard_release" ||
                F->getName() == "__cxa_guard_abort" ||
                F->getName() == "posix_memalign") {
              continue;
            }

            bool noUse = true;
            switch (F->getIntrinsicID()) {
            case Intrinsic::nvvm_barrier0:
            case Intrinsic::nvvm_barrier0_popc:
            case Intrinsic::nvvm_barrier0_and:
            case Intrinsic::nvvm_barrier0_or:
            case Intrinsic::nvvm_membar_cta:
            case Intrinsic::nvvm_membar_gl:
            case Intrinsic::nvvm_membar_sys:
            case Intrinsic::assume:
            case Intrinsic::stacksave:
            case Intrinsic::stackrestore:
            case Intrinsic::lifetime_start:
            case Intrinsic::lifetime_end:
            case Intrinsic::dbg_addr:
            case Intrinsic::dbg_declare:
            case Intrinsic::dbg_value:
            case Intrinsic::invariant_start:
            case Intrinsic::invariant_end:
            case Intrinsic::var_annotation:
            case Intrinsic::ptr_annotation:
            case Intrinsic::annotation:
            case Intrinsic::codeview_annotation:
            case Intrinsic::expect:
            case Intrinsic::type_test:
            case Intrinsic::donothing:
            case Intrinsic::prefetch:
#if LLVM_VERSION_MAJOR >= 8
            case Intrinsic::is_constant:
#endif
              noUse = true;
              break;
            default:
              noUse = false;
              break;
            }
            if (noUse)
              continue;
          }
        }

        Value *memval = Val;

        // BasicAA stupidy assumes that non-pointer's don't alias
        // if this is a nonpointer, use something else to force alias
        // consideration
        if (!memval->getType()->isPointerTy()) {
          if (auto ci = dyn_cast<CastInst>(Val)) {
            if (ci->getOperand(0)->getType()->isPointerTy()) {
              memval = ci->getOperand(0);
            }
          }
          for (auto user : Val->users()) {
            if (isa<CastInst>(user) && user->getType()->isPointerTy()) {
              memval = user;
              break;
            }
          }
        }

#if LLVM_VERSION_MAJOR >= 12
        auto AARes = AA.getModRefInfo(
            &I, MemoryLocation(memval, LocationSize::beforeOrAfterPointer()));
#elif LLVM_VERSION_MAJOR >= 9
        auto AARes = AA.getModRefInfo(
            &I, MemoryLocation(memval, LocationSize::unknown()));
#else
        auto AARes = AA.getModRefInfo(
            &I, MemoryLocation(memval, MemoryLocation::UnknownSize));
#endif

        // Still having failed to replace the location used by AA, fall back to
        // getModref against any location.
        if (!memval->getType()->isPointerTy()) {
          if (auto CB = dyn_cast<CallInst>(&I)) {
            AARes = createModRefInfo(AA.getModRefBehavior(CB));
          } else {
            bool mayRead = I.mayReadFromMemory();
            bool mayWrite = I.mayWriteToMemory();
            AARes = mayRead
                        ? (mayWrite ? ModRefInfo::ModRef : ModRefInfo::Ref)
                        : (mayWrite ? ModRefInfo::Mod : ModRefInfo::NoModRef);
          }
        }

        // TODO this aliasing information is too conservative, the question
        // isn't merely aliasing but whether there is a path for THIS value to
        // eventually be loaded by it not simply because there isnt aliasing

        // If we haven't already shown a potentially active load
        // check if this loads the given value and is active
        if (!potentiallyActiveLoad && isRefSet(AARes)) {
          if (printconst)
            llvm::errs() << "potential active load: " << I << "\n";
          if (auto LI = dyn_cast<LoadInst>(&I)) {
            // If the ref'ing value is a load check if the loaded value is
            // active
            potentiallyActiveLoad = !Hypothesis->isConstantValue(TR, LI);
          } else {
            // Otherwise fallback and check any part of the instruction is
            // active
            // TODO: note that this can be optimized (especially for function
            // calls)
            potentiallyActiveLoad = !Hypothesis->isConstantInstruction(TR, &I);
          }
        }
        if (!potentialStore && isModSet(AARes)) {
          if (printconst)
            llvm::errs() << "potential active store: " << I << "\n";
          if (auto SI = dyn_cast<StoreInst>(&I)) {
            potentialStore = !Hypothesis->isConstantValue(
                TR, SI->getValueOperand()); // true;
          } else {
            // Otherwise fallback and check if the instruction is active
            // TODO: note that this can be optimized (especially for function
            // calls)
            potentialStore = !Hypothesis->isConstantInstruction(TR, &I);
          }
        }
      }
    }

    if (printconst)
      llvm::errs() << " </MEMSEARCH" << (int)directions << ">" << *Val
                   << " potentiallyActiveLoad=" << potentiallyActiveLoad
                   << " potentialStore=" << potentialStore << "\n";
    if (potentiallyActiveLoad && potentialStore) {
      insertAllFrom(*Hypothesis);
      return false;
    } else {
      // We now know that there isn't a matching active load/store pair in this
      // function Now the only way that this memory can facilitate a transfer of
      // active information is if it is done outside of the function

      // This can happen if either:
      // a) the memory had an active load or store before this function was
      // called b) the memory had an active load or store after this function
      // was called

      // Case a) can occur if:
      //    1) this memory came from an active global
      //    2) this memory came from an active argument
      //    3) this memory came from a load from active memory
      // In other words, assuming this value is inactive, going up this
      // location's argument must be inactive

      assert(UpHypothesis);
      // UpHypothesis.ConstantValues.insert(val);
      UpHypothesis->insertConstantsFrom(*Hypothesis);
      assert(directions & UP);
      bool ActiveUp = !UpHypothesis->isInstructionInactiveFromOrigin(TR, Val);

      // Case b) can occur if:
      //    1) this memory is used as part of an active return
      //    2) this memory is stored somewhere

      // We never verify that an origin wasn't stored somewhere or returned.
      // to remedy correctness for now let's do something extremely simple
      std::shared_ptr<ActivityAnalyzer> DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      DownHypothesis->insertConstantsFrom(*Hypothesis);
      bool ActiveDown =
          DownHypothesis->isValueActivelyStoredOrReturned(TR, Val);
      // BEGIN TEMPORARY

      if (!ActiveDown && TmpOrig != Val) {

        if (isa<Argument>(TmpOrig) || isa<GlobalVariable>(TmpOrig) ||
            isa<AllocaInst>(TmpOrig) ||
            (isCalledFunction(TmpOrig) &&
             isAllocationFunction(*isCalledFunction(TmpOrig), TLI))) {
          std::shared_ptr<ActivityAnalyzer> DownHypothesis2 =
              std::shared_ptr<ActivityAnalyzer>(
                  new ActivityAnalyzer(*DownHypothesis, DOWN));
          DownHypothesis2->ConstantValues.insert(TmpOrig);
          if (DownHypothesis2->isValueActivelyStoredOrReturned(TR, TmpOrig)) {
            if (printconst)
              llvm::errs() << " active from ivasor: " << *TmpOrig << "\n";
            ActiveDown = true;
          }
        } else {
          // unknown origin that could've been stored/returned/etc
          if (printconst)
            llvm::errs() << " active from unknown origin: " << *TmpOrig << "\n";
          ActiveDown = true;
        }
      }

      // END TEMPORARY

      // We can now consider the three places derivative information can be
      // transferred
      //   Case A) From the origin
      //   Case B) Though the return
      //   Case C) Within the function (via either load or store)

      bool ActiveMemory = false;

      // If it is transferred via active origin and return, clearly this is
      // active
      ActiveMemory |= (ActiveUp && ActiveDown);

      // If we come from an active origin and load, memory is clearly active
      ActiveMemory |= (ActiveUp && potentiallyActiveLoad);

      // If we come from an active origin and only store into it, it changes
      // future state
      ActiveMemory |= (ActiveUp && potentialStore);

      // If we go to an active return and store active memory, this is active
      ActiveMemory |= (ActiveDown && potentialStore);
      // Actually more generally, if we are ActiveDown (returning memory that is
      // used) in active return, we must be active. This is necessary to ensure
      // mallocs have their differential shadows created when returned [TODO
      // investigate more]
      ActiveMemory |= ActiveDown;

      // If we go to an active return and only load it, however, that doesnt
      // transfer derivatives and we can say this memory is inactive

      if (printconst)
        llvm::errs() << " @@MEMSEARCH" << (int)directions << ">" << *Val
                     << " potentiallyActiveLoad=" << potentiallyActiveLoad
                     << " potentialStore=" << potentialStore
                     << " ActiveUp=" << ActiveUp << " ActiveDown=" << ActiveDown
                     << " ActiveMemory=" << ActiveMemory << "\n";

      if (ActiveMemory) {
        ActiveValues.insert(Val);
        assert(Hypothesis->directions == directions);
        assert(Hypothesis->ActiveValues.count(Val));
        insertAllFrom(*Hypothesis);
        return false;
      } else {
        ConstantValues.insert(Val);
        insertConstantsFrom(*Hypothesis);
        insertConstantsFrom(*UpHypothesis);
        insertConstantsFrom(*DownHypothesis);
        return true;
      }
    }
  }

  // For all non-pointers, it is now sufficient to simply prove that
  // either activity does not flow in, or activity does not flow out
  // This alone cuts off the flow (being unable to flow through memory)

  // Not looking at uses to prove inactive (definition of up), if the creator of
  // this value is inactive, we are inactive Since we won't look at uses to
  // prove, we can inductively assume this is inactive
  if (directions & UP) {
    if (directions == UP && !isa<PHINode>(Val)) {
      if (isInstructionInactiveFromOrigin(TR, Val)) {
        ConstantValues.insert(Val);
        return true;
      }
    } else {
      UpHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, UP));
      UpHypothesis->ConstantValues.insert(Val);
      if (UpHypothesis->isInstructionInactiveFromOrigin(TR, Val)) {
        insertConstantsFrom(*UpHypothesis);
        ConstantValues.insert(Val);
        return true;
      }
    }
  }

  if (directions & DOWN) {
    // Not looking at users to prove inactive (definition of down)
    // If all users are inactive, this is therefore inactive.
    // Since we won't look at origins to prove, we can inductively assume this
    // is inactive

    // As an optimization if we are going down already
    // and we won't use ourselves (done by PHI's), we
    // dont need to inductively assume we're true
    // and can instead use this object!
    if (directions == DOWN && !isa<PHINode>(Val)) {
      if (isValueInactiveFromUsers(TR, Val)) {
        if (UpHypothesis)
          insertConstantsFrom(*UpHypothesis);
        ConstantValues.insert(Val);
        return true;
      }
    } else {
      auto DownHypothesis =
          std::shared_ptr<ActivityAnalyzer>(new ActivityAnalyzer(*this, DOWN));
      DownHypothesis->ConstantValues.insert(Val);
      if (DownHypothesis->isValueInactiveFromUsers(TR, Val)) {
        insertConstantsFrom(*DownHypothesis);
        if (UpHypothesis)
          insertConstantsFrom(*UpHypothesis);
        ConstantValues.insert(Val);
        return true;
      }
    }
  }

  if (printconst)
    llvm::errs() << " Value nonconstant (couldn't disprove)[" << (int)directions
                 << "]" << *Val << "\n";
  ActiveValues.insert(Val);
  return false;
}

/// Is the instruction guaranteed to be inactive because of its operands
bool ActivityAnalyzer::isInstructionInactiveFromOrigin(TypeResults &TR,
                                                       llvm::Value *val) {
  // Must be an analyzer only searching up
  assert(directions == UP);
  assert(!isa<Argument>(val));
  assert(!isa<GlobalVariable>(val));

  // Not an instruction and thus not legal to search for activity via operands
  if (!isa<Instruction>(val)) {
    llvm::errs() << "unknown pointer source: " << *val << "\n";
    assert(0 && "unknown pointer source");
    llvm_unreachable("unknown pointer source");
    return false;
  }

  Instruction *inst = cast<Instruction>(val);
  if (printconst)
    llvm::errs() << " < UPSEARCH" << (int)directions << ">" << *inst << "\n";

  // cpuid is explicitly an inactive instruction
  if (auto call = dyn_cast<CallInst>(inst)) {
#if LLVM_VERSION_MAJOR >= 11
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledOperand())) {
#else
    if (auto iasm = dyn_cast<InlineAsm>(call->getCalledValue())) {
#endif
      if (StringRef(iasm->getAsmString()).contains("cpuid")) {
        if (printconst)
          llvm::errs() << " constant instruction from known cpuid instruction "
                       << *inst << "\n";
        return true;
      }
    }
  }

  if (isa<MemSetInst>(inst)) {
    // memset's are definitionally inactive since
    // they copy a byte which cannot be active
    if (printconst)
      llvm::errs() << " constant instruction as memset " << *inst << "\n";
    return true;
  }

  if (auto SI = dyn_cast<StoreInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, SI->getValueOperand()) ||
        isConstantValue(TR, SI->getPointerOperand())) {
      if (printconst)
        llvm::errs() << " constant instruction as memset " << *inst << "\n";
      return true;
    }
  }

  if (auto MTI = dyn_cast<MemTransferInst>(inst)) {
    // if either src or dst is inactive, there cannot be a transfer of active
    // values and thus the store is inactive
    if (isConstantValue(TR, MTI->getArgOperand(0)) ||
        isConstantValue(TR, MTI->getArgOperand(1))) {
      if (printconst)
        llvm::errs() << " constant instruction as memset " << *inst << "\n";
      return true;
    }
  }

  // Calls to print/assert/cxa guard are definitionally inactive
  if (auto op = dyn_cast<CallInst>(inst)) {
    if (op->hasFnAttr("enzyme_inactive")) {
      return true;
    }
    if (auto called = op->getCalledFunction()) {
      if (called->getName() == "free" || called->getName() == "_ZdlPv" ||
          called->getName() == "_ZdlPvm" || called->getName() == "munmap") {
        return true;
      }

      for (auto FuncName : KnownInactiveFunctionsStartingWith) {
        if (called->getName().startswith(FuncName)) {
          return true;
        }
      }
      for (auto FuncName : KnownInactiveFunctions) {
        if (called->getName() == FuncName)
          return true;
      }

      if (called->getIntrinsicID() == Intrinsic::trap)
        return true;

      // If requesting emptty unknown functions to be considered inactive, abide
      // by those rules
      if (!isCertainPrintMallocOrFree(called) && called->empty() &&
          !hasMetadata(called, "enzyme_gradient") && !isa<IntrinsicInst>(op) &&
          emptyfnconst) {
        return true;
      }
    }
  }

  // Intrinsics known always to be inactive
  if (auto II = dyn_cast<IntrinsicInst>(inst)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::nvvm_barrier0:
    case Intrinsic::nvvm_barrier0_popc:
    case Intrinsic::nvvm_barrier0_and:
    case Intrinsic::nvvm_barrier0_or:
    case Intrinsic::nvvm_membar_cta:
    case Intrinsic::nvvm_membar_gl:
    case Intrinsic::nvvm_membar_sys:
    case Intrinsic::assume:
    case Intrinsic::stacksave:
    case Intrinsic::stackrestore:
    case Intrinsic::lifetime_start:
    case Intrinsic::lifetime_end:
    case Intrinsic::dbg_addr:
    case Intrinsic::dbg_declare:
    case Intrinsic::dbg_value:
    case Intrinsic::invariant_start:
    case Intrinsic::invariant_end:
    case Intrinsic::var_annotation:
    case Intrinsic::ptr_annotation:
    case Intrinsic::annotation:
    case Intrinsic::codeview_annotation:
    case Intrinsic::expect:
    case Intrinsic::type_test:
    case Intrinsic::donothing:
    case Intrinsic::prefetch:
#if LLVM_VERSION_MAJOR >= 8
    case Intrinsic::is_constant:
#endif
      return true;
    default:
      break;
    }
  }

  if (auto gep = dyn_cast<GetElementPtrInst>(inst)) {
    // A gep's only args that could make it active is the pointer operand
    if (isConstantValue(TR, gep->getPointerOperand())) {
      if (printconst)
        llvm::errs() << "constant(" << (int)directions << ") up-gep " << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (auto ci = dyn_cast<CallInst>(inst)) {
    bool seenuse = false;

    propagateArgumentInformation(TLI, *ci, [&](Value *a) {
      if (!isConstantValue(TR, a)) {
        seenuse = true;
        if (printconst)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-call "
                       << *inst << " op " << *a << "\n";
        return true;
      }
      return false;
    });

    // TODO consider calling interprocedural here
    // TODO: Really need an attribute that determines whether a function
    // can access a global (not even necessarily read)
    // if (ci->hasFnAttr(Attribute::ReadNone) ||
    // ci->hasFnAttr(Attribute::ArgMemOnly))
    if (!seenuse) {
      if (printconst)
        llvm::errs() << "constant(" << (int)directions << ")  up-call:" << *inst
                     << "\n";
      return true;
    }
    return !seenuse;
  } else if (auto si = dyn_cast<SelectInst>(inst)) {

    if (isConstantValue(TR, si->getTrueValue()) &&
        isConstantValue(TR, si->getFalseValue())) {

      if (printconst)
        llvm::errs() << "constant(" << (int)directions << ") up-sel:" << *inst
                     << "\n";
      return true;
    }
    return false;
  } else if (isa<SIToFPInst>(inst) || isa<UIToFPInst>(inst) ||
             isa<FPToSIInst>(inst) || isa<FPToUIInst>(inst)) {

    if (printconst)
      llvm::errs() << "constant(" << (int)directions << ") up-fpcst:" << *inst
                   << "\n";
    return true;
  } else {
    bool seenuse = false;
    //! TODO does not consider reading from global memory that is active and not
    //! an argument
    for (auto &a : inst->operands()) {
      bool hypval = isConstantValue(TR, a);
      if (!hypval) {
        if (printconst)
          llvm::errs() << "nonconstant(" << (int)directions << ")  up-inst "
                       << *inst << " op " << *a << "\n";
        seenuse = true;
        break;
      }
    }

    if (!seenuse) {
      if (printconst)
        llvm::errs() << "constant(" << (int)directions << ")  up-inst:" << *inst
                     << "\n";
      return true;
    }
    return false;
  }
}

/// Is the value free of any active uses
bool ActivityAnalyzer::isValueInactiveFromUsers(TypeResults &TR,
                                                llvm::Value *val) {
  assert(directions & DOWN);
  // Must be an analyzer only searching down, unless used outside
  // assert(directions == DOWN);

  // To ensure we can call down

  if (printconst)
    llvm::errs() << " <Value USESEARCH" << (int)directions << ">" << *val
                 << "\n";

  bool seenuse = false;
  // user, predecessor
  std::deque<std::pair<User *, Value *>> todo;
  for (const auto a : val->users()) {
    todo.push_back(std::make_pair(a, val));
  }
  std::set<std::pair<User *, Value *>> done = {};

  while (todo.size()) {
    auto pair = todo.front();
    todo.pop_front();
    if (done.count(pair))
      continue;
    done.insert(pair);
    User *a = pair.first;

    if (printconst)
      llvm::errs() << "      considering use of " << *val << " - " << *a
                   << "\n";

    if (!isa<Instruction>(a)) {
      if (isa<ConstantExpr>(a)) {
        if (!isValueInactiveFromUsers(TR, a)) {
          llvm::errs() << "   inactive user of " << *val << " in " << *a
                       << "\n";
          return false;
        } else
          continue;
      }
      if (isa<ConstantData>(a)) {
        continue;
      }

      if (printconst)
        llvm::errs() << "      unknown non instruction use of " << *val << " - "
                     << *a << "\n";
      return false;
    }

    if (isa<AllocaInst>(a)) {
      if (printconst)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  allocainst use:" << *val << " user " << *a << "\n";
      continue;
    }

    if (isa<SIToFPInst>(a) || isa<UIToFPInst>(a) || isa<FPToSIInst>(a) ||
        isa<FPToUIInst>(a)) {
      if (printconst)
        llvm::errs() << "found constant(" << (int)directions
                     << ")  si-fp use:" << *val << " user " << *a << "\n";
      continue;
    }

    // if this instruction is in a different function, conservatively assume
    // it is active
    if (cast<Instruction>(a)->getParent()->getParent() != TR.info.Function) {
      if (printconst)
        llvm::errs() << "found use in different function(" << (int)directions
                     << ")  val:" << *val << " user " << *a << "\n";
      return false;
    }

    // This use is only active if specified
    if (isa<ReturnInst>(a)) {
      return !ActiveReturns;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      bool ConstantArg = isFunctionArgumentConstant(call, pair.second);
      if (ConstantArg) {
        if (printconst) {
          llvm::errs() << "Value found constant callinst use:" << *val
                       << " user " << *call << "\n";
        }
        continue;
      }
    }

    // If this doesn't write to memory this can only be an active use
    // if its return is used in an active way, therefore add this to
    // the list of users to analyze
    if (auto I = dyn_cast<Instruction>(a)) {
      if (!I->mayWriteToMemory()) {
        if (TR.intType(1, I, /*errIfNotFound*/ false).isIntegral()) {
          continue;
        }
        for (auto u : I->users()) {
          todo.push_back(std::make_pair(u, (Value *)I));
        }
        continue;
      }
    }

    if (printconst)
      llvm::errs() << "Value nonconstant inst (uses):" << *val << " user " << *a
                   << "\n";
    seenuse = true;
    break;
  }

  if (printconst)
    llvm::errs() << " </Value USESEARCH" << (int)directions
                 << " const=" << (!seenuse) << ">" << *val << "\n";
  return !seenuse;
}

/// Is the value potentially actively returned or stored
bool ActivityAnalyzer::isValueActivelyStoredOrReturned(TypeResults &TR,
                                                       llvm::Value *val) {
  // Must be an analyzer only searching down
  assert(directions == DOWN);

  if (StoredOrReturnedCache.find(val) != StoredOrReturnedCache.end()) {
    return StoredOrReturnedCache[val];
  }

  if (printconst)
    llvm::errs() << " <ASOR" << (int)directions << ">" << *val << "\n";

  StoredOrReturnedCache[val] = false;

  for (const auto a : val->users()) {
    if (isa<AllocaInst>(a)) {
      continue;
    }
    // Loading a value prevents its pointer from being captured
    if (isa<LoadInst>(a)) {
      continue;
    }

    if (isa<ReturnInst>(a)) {
      if (!ActiveReturns)
        continue;

      if (printconst)
        llvm::errs() << " </ASOR" << (int)directions << " active from-ret>"
                     << *val << "\n";
      StoredOrReturnedCache[val] = true;
      return true;
    }

    if (auto call = dyn_cast<CallInst>(a)) {
      if (!couldFunctionArgumentCapture(call, val)) {
        continue;
      }
      bool ConstantArg = isFunctionArgumentConstant(call, val);
      if (ConstantArg) {
        continue;
      }
    }

    if (auto SI = dyn_cast<StoreInst>(a)) {
      // If we are being stored into, not storing this value
      // this case can be skipped
      if (SI->getValueOperand() != val) {
        continue;
      }
      // Storing into active memory, return true
      if (!isConstantValue(TR, SI->getPointerOperand())) {
        StoredOrReturnedCache[val] = true;
        if (printconst)
          llvm::errs() << " </ASOR" << (int)directions << " active from-store>"
                       << *val << " store=" << *SI << "\n";
        return true;
      }
    }

    if (auto inst = dyn_cast<Instruction>(a)) {
      if (!inst->mayWriteToMemory() ||
          (isa<CallInst>(inst) && AA.onlyReadsMemory(cast<CallInst>(inst)))) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (isConstantValue(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a)) {
          continue;
        }
      }
    }

    if (auto F = isCalledFunction(a)) {
      if (isAllocationFunction(*F, TLI)) {
        // if not written to memory and returning a known constant, this
        // cannot be actively returned/stored
        if (isConstantValue(TR, a)) {
          continue;
        }
        // if not written to memory and returning a value itself
        // not actively stored or returned, this is not actively
        // stored or returned
        if (!isValueActivelyStoredOrReturned(TR, a)) {
          continue;
        }
      } else if (isDeallocationFunction(*F, TLI)) {
        // freeing memory never counts
        continue;
      }
    }
    // fallback and conservatively assume that if the value is written to
    // it is written to active memory
    // TODO handle more memory instructions above to be less conservative

    if (printconst)
      llvm::errs() << " </ASOR" << (int)directions << " active from-unknown>"
                   << *val << " - use=" << *a << "\n";
    return StoredOrReturnedCache[val] = true;
  }

  if (printconst)
    llvm::errs() << " </ASOR" << (int)directions << " inactive>" << *val
                 << "\n";
  return false;
}
