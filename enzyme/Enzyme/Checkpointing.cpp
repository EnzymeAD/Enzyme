//===- Checkpointing.cpp - Scheme-driven checkpointing of time loops -----===//
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
// See Checkpointing.h. The generated code has three layers:
//
//  - The loop function `enzyme.ckpt.for.<step>(start, n, vt, data,
//    [region, bytes]..., args...)`, which replaces the marker and is the
//    primal.
//  - Per loop and activity: trampolines that run step i of the primal, the
//    augmented forward pass of step i (returning its tape), and the reverse
//    pass of step i, all reading the step's arguments from an environment
//    struct. The augmented forward and reverse passes of the loop pack that
//    environment and call the driver.
//  - Per module: the driver `__enzyme_ckpt_fwd` / `__enzyme_ckpt_rev`, which
//    runs the scheme's action loop.
//
//===----------------------------------------------------------------------===//

#include "Checkpointing.h"
#include "enzyme/checkpoint_schedule.h"

#include <set>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "Utils.h"

using namespace llvm;

static cl::opt<bool> EnzymeCheckpointSplitSteps(
    "enzyme-checkpoint-split-steps", cl::init(false), cl::Hidden,
    cl::desc("Differentiate each checkpointed step as its augmented forward "
             "pass followed by its reverse pass, instead of in combined mode"));

static cl::opt<int> EnzymeCheckpointLoopVerbose(
    "enzyme-checkpoint-loop-verbose", cl::init(0), cl::Hidden,
    cl::desc("The verbosity of the reference schemes of loops annotated for "
             "checkpointing: 1 prints a summary, 2 every action"));

static cl::opt<bool> EnzymePrintCheckpointRegions(
    "enzyme-print-checkpoint-regions", cl::init(false), cl::Hidden,
    cl::desc("Print the memory each checkpointed loop snapshots"));

static constexpr const char *CheckpointAttr = "enzyme_checkpoint";
static constexpr const char *CheckpointRegionsAttr =
    "enzyme_checkpoint_nregions";
static constexpr const char *CheckpointStepMD = "enzyme_checkpoint_step";
static constexpr const char *CheckpointRegionSpacesAttr =
    "enzyme_checkpoint_region_spaces";

// Slots the driver itself uses: the state the reverse sweep starts from, and
// the state before the last step.
static constexpr int64_t EntrySlot = -1;
static constexpr int64_t LastSlot = -2;

// Values of the protocol in include/enzyme/checkpoint.h.
enum : int32_t {
  CKPT_STORE = 1,
  CKPT_RESTORE = 2,
  CKPT_FORWARD = 3,
  CKPT_FIRSTUTURN = 4,
  CKPT_UTURN = 5,
  CKPT_DONE = 7
};

// Fields of EnzymeCheckpointScheme after `version`.
enum VTableField : unsigned {
  VT_Init = 1,
  VT_NextAction,
  VT_Store,
  VT_Restore,
  VT_SetNSteps,
  VT_Finalize,
  VT_SaveState,
  VT_LoadState,
  VT_SetPaths,
};

// Fields of the driver's handle. It holds the schedule from the forward to
// the reverse pass, but neither the step's arguments nor the regions: each
// pass gets those from its own arguments, which Enzyme keeps for the reverse
// pass (and a garbage-collected frontend keeps alive with them).
enum HandleField : unsigned {
  H_VT = 0,
  H_Data,
  H_State,
  H_Start,
  H_N,
  H_LastJ,
  H_Empty,
};

// A region whose snapshot callbacks take (ENZYME_CKPT_REGION_CALLBACK): the
// space of its marker, which gives the callbacks as its size.
static constexpr unsigned CallbackRegionSpace = ~0u;
static constexpr unsigned RegionCallbackFlag = 1;
// Fields of EnzymeCkptCallbacks.
enum CallbacksField : unsigned {
  CB_Enter = 0,
  CB_Save,
  CB_Restore,
  CB_Sync,
  CB_Leave,
};

// Parameters of the loop function before the region pairs.
static constexpr unsigned LoopFixedParams = 4;

bool isCheckpointLoop(const Function *F) {
  // Derivatives cloned from a loop function carry its attributes and
  // metadata, but not its signature: only the loop function itself counts.
  return F && F->hasFnAttribute(CheckpointAttr) &&
         F->getName().starts_with("enzyme.ckpt.");
}

/// A loop run until its step returns false, rather than a given number of
/// times.
static bool isWhileLoop(const Function *F) {
  return F->getFnAttribute(CheckpointAttr).getValueAsString() == "while";
}

static unsigned getNumRegions(const Function *F) {
  unsigned n = 0;
  F->getFnAttribute(CheckpointRegionsAttr)
      .getValueAsString()
      .getAsInteger(10, n);
  return n;
}

static unsigned getFirstStepArg(const Function *F) {
  return LoopFixedParams + 2 * getNumRegions(F);
}

static Function *getStep(const Function *F) {
  auto *MD = cast<MDTuple>(F->getMetadata(CheckpointStepMD));
  return cast<Function>(
      cast<ConstantAsMetadata>(MD->getOperand(0))->getValue());
}

//===----------------------------------------------------------------------===//
// Lowering of the marker
//===----------------------------------------------------------------------===//

/// The name of the marker `V` is, and whether it is passed by reference: a
/// C caller passes the marker's value, a Fortran caller (implicit interface)
/// its address, and passes every argument after it by reference too.
static std::optional<StringRef> markerName(Value *V, bool &byRef) {
  V = V->stripPointerCasts();
  byRef = true;
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    V = LI->getPointerOperand()->stripPointerCasts();
    byRef = false;
  }
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    return GV->getName();
  return {};
}

/// Load the integer a by-reference argument points to, as an i64. The width
/// is that of the variable, where it can be seen, and 64 bits otherwise.
static Value *loadInteger(IRBuilder<> &B, Value *ptr) {
  Type *I64 = B.getInt64Ty();
  Type *T = nullptr;
  Value *base = getBaseObject(ptr);
  if (base == ptr->stripPointerCasts()) {
    if (auto *AI = dyn_cast<AllocaInst>(base))
      T = AI->getAllocatedType();
    else if (auto *GV = dyn_cast<GlobalVariable>(base))
      T = GV->getValueType();
  }
  if (!T || !T->isIntegerTy())
    T = I64;
  Value *V = B.CreateLoad(T, B.CreatePointerCast(ptr, getUnqual(T)));
  return B.CreateSExtOrTrunc(V, I64);
}

static Value *castArg(IRBuilder<> &B, Value *V, Type *T) {
  if (V->getType() == T)
    return V;
  // A scalar passed by reference to a step that takes it by value.
  if (V->getType()->isPointerTy() &&
      (T->isIntegerTy() || T->isFloatingPointTy()))
    return B.CreateLoad(T, B.CreatePointerCast(V, getUnqual(T)));
  if (V->getType()->isIntegerTy() && T->isIntegerTy())
    return B.CreateSExtOrTrunc(V, T);
  if (V->getType()->isFloatingPointTy() && T->isFloatingPointTy())
    return B.CreateFPCast(V, T);
  if (V->getType()->isPointerTy() && T->isPointerTy())
    return B.CreatePointerBitCastOrAddrSpaceCast(V, T);
  return nullptr;
}

static Function *createLoopFunction(Module &M, Function *step,
                                    ArrayRef<Type *> regionTypes, Type *vtTy,
                                    Type *dataTy, bool isWhile,
                                    ArrayRef<bool> callbackRegions = {}) {
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  SmallVector<Type *, 8> params = {I64, I64, vtTy, dataTy};
  for (Type *T : regionTypes) {
    params.push_back(T);
    params.push_back(I64);
  }
  auto *stepFT = step->getFunctionType();
  for (unsigned i = 1; i < stepFT->getNumParams(); i++)
    params.push_back(stepFT->getParamType(i));

  auto *FT = FunctionType::get(Type::getVoidTy(Ctx), params, false);
  auto *F = Function::Create(
      FT, GlobalValue::InternalLinkage,
      (isWhile ? "enzyme.ckpt.while." : "enzyme.ckpt.for.") + step->getName(),
      &M);
  F->addFnAttr(CheckpointAttr, isWhile ? "while" : "for");
  F->addFnAttr(CheckpointRegionsAttr, std::to_string(regionTypes.size()));
  F->addFnAttr(Attribute::NoInline);
  F->setMetadata(CheckpointStepMD,
                 MDTuple::get(Ctx, {ConstantAsMetadata::get(step)}));
  // The schedule's own arguments carry no derivative, but for the root of a
  // callback region, whose shadow its callbacks are given.
  for (unsigned i = 0; i < LoopFixedParams + 2 * regionTypes.size(); i++) {
    unsigned r = (i - LoopFixedParams) / 2;
    if (i >= LoopFixedParams && (i - LoopFixedParams) % 2 == 0 &&
        r < callbackRegions.size() && callbackRegions[r])
      continue;
    F->addParamAttr(i, Attribute::get(Ctx, "enzyme_inactive"));
  }

  auto *entry = BasicBlock::Create(Ctx, "entry", F);
  auto *body = BasicBlock::Create(Ctx, "body", F);
  auto *exit = BasicBlock::Create(Ctx, "exit", F);
  IRBuilder<> B(entry);
  Value *start = F->getArg(0);
  Value *end = B.CreateAdd(start, F->getArg(1), "end");
  // A while loop runs its step at least once, and as long as it returns true.
  if (isWhile)
    B.CreateBr(body);
  else
    B.CreateCondBr(B.CreateICmpSLT(start, end), body, exit);

  B.SetInsertPoint(body);
  auto *iv = B.CreatePHI(I64, 2, "i");
  iv->addIncoming(start, entry);
  SmallVector<Value *, 8> args = {
      B.CreateSExtOrTrunc(iv, stepFT->getParamType(0))};
  for (unsigned i = getFirstStepArg(F); i < F->arg_size(); i++)
    args.push_back(F->getArg(i));
  auto *call = B.CreateCall(step, args);
  call->setCallingConv(step->getCallingConv());
  auto *next = B.CreateAdd(iv, ConstantInt::get(I64, 1), "i.next");
  iv->addIncoming(next, body);
  if (isWhile)
    B.CreateCondBr(
        B.CreateICmpNE(call, Constant::getNullValue(call->getType())), body,
        exit);
  else
    B.CreateCondBr(B.CreateICmpSLT(next, end), body, exit);

  B.SetInsertPoint(exit);
  B.CreateRetVoid();
  return F;
}

static bool lowerMarker(CallInst *CI, bool isWhile) {
  Module &M = *CI->getModule();
  LLVMContext &Ctx = M.getContext();
  const char *marker =
      isWhile ? "__enzyme_checkpoint_while" : "__enzyme_checkpoint_for";
  auto fail = [&](const Twine &msg) {
    std::string str = msg.str();
    EmitFailure("CheckpointMarker", CI->getDebugLoc(), CI, str);
    return false;
  };

  if (CI->arg_size() < (isWhile ? 1u : 3u))
    return fail(Twine(marker) + " needs a step function" +
                (isWhile ? "" : ", a start and a number of steps"));
  Value *stepV = CI->getArgOperand(0)->stripPointerCasts();
  if (auto *GA = dyn_cast<GlobalAlias>(stepV))
    stepV = GA->getAliaseeObject();
  auto *step = dyn_cast<Function>(stepV);
  if (!step)
    return fail(Twine(marker) + " needs a known step function");
  auto *stepFT = step->getFunctionType();
  if (stepFT->isVarArg() || stepFT->getNumParams() == 0 ||
      !stepFT->getParamType(0)->isIntegerTy())
    return fail(Twine("the step of ") + marker +
                " must take the step index as an integer first argument, by "
                "value");
  if (isWhile && !stepFT->getReturnType()->isIntegerTy())
    return fail("the step of __enzyme_checkpoint_while must return whether "
                "to go on, as an integer or bool");

  Value *vt = nullptr, *data = nullptr;
  SmallVector<std::pair<Value *, Value *>, 2> regions;
  unsigned idx = isWhile ? 1 : 3;
  IRBuilder<> B(CI);
  Type *I64 = Type::getInt64Ty(Ctx);
  while (idx < CI->arg_size()) {
    bool byRef;
    auto name = markerName(CI->getArgOperand(idx), byRef);
    if (name && *name == "enzyme_scheme") {
      if (idx + 2 >= CI->arg_size())
        return fail("enzyme_scheme needs a scheme and its data");
      // By reference, the scheme is a variable holding its address, and the
      // data is the object itself.
      vt = CI->getArgOperand(idx + 1);
      if (byRef)
        vt = B.CreateLoad(getInt8PtrTy(Ctx),
                          B.CreatePointerCast(vt, getUnqual(getInt8PtrTy(Ctx))));
      data = CI->getArgOperand(idx + 2);
      idx += 3;
      continue;
    }
    if (name && *name == "enzyme_checkpoint_region") {
      if (idx + 2 >= CI->arg_size())
        return fail("enzyme_checkpoint_region needs a pointer and a size");
      Value *bytes = CI->getArgOperand(idx + 2);
      if (byRef && bytes->getType()->isPointerTy())
        bytes = loadInteger(B, bytes);
      regions.emplace_back(CI->getArgOperand(idx + 1), bytes);
      idx += 3;
      continue;
    }
    break;
  }
  if (!vt)
    return fail(Twine(marker) +
                " needs enzyme_scheme, followed by the scheme and its data");
  unsigned nargs = CI->arg_size() - idx;
  if (nargs + 1 != stepFT->getNumParams())
    return fail(Twine(marker) + " passes " + Twine(nargs) +
                " arguments to a step that takes " +
                Twine(stepFT->getNumParams() - 1) + " after the index");

  SmallVector<Value *, 8> args;
  if (isWhile) {
    // Steps from 0, as many as it takes.
    args.push_back(ConstantInt::get(I64, 0));
    args.push_back(ConstantInt::getSigned(I64, -1));
  } else {
    for (unsigned i = 1; i <= 2; i++) {
      Value *V = CI->getArgOperand(i);
      args.push_back(V->getType()->isPointerTy() ? loadInteger(B, V)
                                                 : castArg(B, V, I64));
    }
  }
  if (!args[0] || !args[1])
    return fail("the start and number of steps must be integers");
  args.push_back(vt);
  args.push_back(data);
  SmallVector<Type *, 2> regionTypes;
  for (auto &R : regions) {
    if (!R.first->getType()->isPointerTy())
      return fail("enzyme_checkpoint_region needs a pointer");
    Value *bytes = castArg(B, R.second, I64);
    if (!bytes)
      return fail("the size of an enzyme_checkpoint_region must be an integer");
    regionTypes.push_back(R.first->getType());
    args.push_back(R.first);
    args.push_back(bytes);
  }
  for (unsigned i = 0; i < nargs; i++) {
    Value *a =
        castArg(B, CI->getArgOperand(idx + i), stepFT->getParamType(i + 1));
    if (!a)
      return fail("argument " + Twine(i) + " of " + marker +
                  " does not match the step");
    args.push_back(a);
  }

  Function *loop = createLoopFunction(M, step, regionTypes, vt->getType(),
                                      data->getType(), isWhile);
  auto *call = B.CreateCall(loop, args);
  call->setDebugLoc(CI->getDebugLoc());
  if (!CI->getType()->isVoidTy())
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  CI->eraseFromParent();
  return true;
}

//===----------------------------------------------------------------------===//
// Loop annotations
//===----------------------------------------------------------------------===//
//
// `[[enzyme::checkpointing_enable("binomial", count)]]` on a for statement
// (the Clang plugin, as in Reactant) calls `__enzyme_set_checkpointing(mode,
// count)` at the top of the loop body: mode is a schedule of
// enzyme/checkpoint_schedule.h (1 periodic, 2 Revolve, 3 store all, 4
// binomial) and count its budget, all ones if it is not given, for the
// schedule's default. Periodic and binomial are Enzyme-MLIR's
// enzyme.enable_checkpointing, enzyme.binomial_checkpointing and
// enzyme.checkpoint_period, which Enzyme-JAX raises the call into, and take
// the same steps.
//
// Here the loop is outlined, one iteration a step, into a checkpointed loop
// run by the reference Revolve or Periodic scheme of enzyme/checkpoint.h:
// induction variables are recomputed from the step index, other values
// carried from one iteration to the next and values used after the loop go
// through the stack, and the snapshot holds the globals the step writes (as
// for any checkpointed loop), those stack slots, and the heap blocks the loop
// writes whose size is known before it. A loop that writes memory of unknown
// extent is an error: give the regions with __enzyme_checkpoint_for.

/// The schedule of enzyme/checkpoint_schedule.h a loop annotation names, or
/// -1: "binomial" is Enzyme-MLIR's binomial schedule (its
/// enzyme.binomial_checkpointing), "revolve" Revolve, "regular" an older
/// name of "periodic".
static int64_t getScheduleTag(StringRef name) {
  return StringSwitch<int64_t>(name)
      .Case("none", ENZYME_CKPT_SCHEDULE_NONE)
      .Case("periodic", ENZYME_CKPT_SCHEDULE_PERIODIC)
      .Case("regular", ENZYME_CKPT_SCHEDULE_PERIODIC)
      .Case("revolve", ENZYME_CKPT_SCHEDULE_REVOLVE)
      .Case("store_all", ENZYME_CKPT_SCHEDULE_STORE_ALL)
      .Case("binomial", ENZYME_CKPT_SCHEDULE_BINOMIAL)
      .Default(-1);
}

static bool isLoopAnnotation(const Function *F) {
  return F && F->getName().contains("__enzyme_set_checkpointing");
}

/// The object `V` points into and its size in bytes, computed in front of
/// `IP`, if both are known there.
static std::optional<std::pair<Value *, Value *>>
getKnownAllocation(Value *V, Instruction *IP, DominatorTree &DT) {
  const DataLayout &DL = IP->getModule()->getDataLayout();
  Value *base = getBaseObject(V);
  IRBuilder<> B(IP);
  Type *I64 = B.getInt64Ty();
  auto available = [&](Value *X) {
    if (isa<Constant>(X) || isa<Argument>(X))
      return true;
    auto *I = dyn_cast<Instruction>(X);
    return I && DT.dominates(I, IP);
  };
  if (!available(base))
    return {};
  if (auto *AI = dyn_cast<AllocaInst>(base)) {
    auto size = AI->getAllocationSize(DL);
    if (!size || size->isScalable())
      return {};
    return std::make_pair(base, (Value *)ConstantInt::get(
                                    I64, size->getFixedValue()));
  }
  auto *CB = dyn_cast<CallBase>(base);
  Function *callee = CB ? getFunctionFromCall(CB) : nullptr;
  if (!callee)
    return {};
  StringRef name = callee->getName();
  if (name == "malloc" || name == "_Znwm" || name == "_Znam") {
    Value *n = CB->getArgOperand(0);
    if (!available(n))
      return {};
    return std::make_pair(base, B.CreateZExtOrTrunc(n, I64));
  }
  if (name == "calloc") {
    Value *n = CB->getArgOperand(0), *m = CB->getArgOperand(1);
    if (!available(n) || !available(m))
      return {};
    return std::make_pair(base, B.CreateMul(B.CreateZExtOrTrunc(n, I64),
                                            B.CreateZExtOrTrunc(m, I64)));
  }
  return {};
}

/// The object `P` points into, looking through julia.gc_loaded, which makes
/// a pointer into a Julia object the object roots.
static Value *getHintBase(Value *P) {
  Value *B = getBaseObject(P);
  if (auto *CI = dyn_cast<CallInst>(B))
    if (Function *F = getFunctionFromCall(CI))
      if (F->getName() == "julia.gc_loaded")
        return getBaseObject(CI->getArgOperand(1));
  return B;
}

/// Whether `V` points into the Julia runtime's state of the task (its
/// thread's allocator, the current task), which the step's allocations and
/// safepoints use and which is no part of what the loop computes.
static bool isJuliaTaskState(Value *V) {
  Value *B = getBaseObject(V);
  while (auto *LI = dyn_cast<LoadInst>(B))
    B = getBaseObject(LI->getPointerOperand());
  auto *CI = dyn_cast<CallInst>(B);
  Function *F = CI ? getFunctionFromCall(CI) : nullptr;
  return F && (F->getName() == "julia.get_pgcstack" ||
               F->getName() == "julia.get_pgcstack_or_new" ||
               F->getName() == "julia.ptls_states");
}

/// Whether `B` is memory allocated where it is, which a step that allocates
/// it does not need to have snapshotted: the copy Julia makes of an array
/// that may alias another, the exception it throws.
static bool isFreshAllocation(Value *B) {
  auto *CB = dyn_cast<CallBase>(B);
  if (!CB)
    return false;
  if (CB->returnDoesNotAlias())
    return true;
  Function *F = getFunctionFromCall(CB);
  if (!F)
    return false;
  StringRef name = F->getName();
  return name == "julia.gc_alloc_obj" || name == "malloc" || name == "calloc" ||
         name == "_Znwm" || name == "_Znam" ||
         name.contains("jl_alloc_genericmemory") ||
         name.contains("jl_gc_alloc");
}

/// The arguments that `V`, a pointer loaded from memory (through phis and
/// selects), is ultimately loaded from, if they are all it is loaded from:
/// memory the step allocated, and in Julia code its stack slots, which hold
/// references to the state, may be on the way.
static bool getLoadRoots(Value *V, SmallPtrSetImpl<Argument *> &roots,
                         bool julia, SmallPtrSetImpl<Value *> *seen = nullptr) {
  SmallPtrSet<Value *, 8> local;
  if (!seen)
    seen = &local;
  V = getHintBase(V);
  if (!seen->insert(V).second)
    return true;
  if (isFreshAllocation(V) || isa<ConstantPointerNull>(V))
    return true;
  if (auto *A = dyn_cast<Argument>(V)) {
    roots.insert(A);
    return true;
  }
  if (auto *LI = dyn_cast<LoadInst>(V)) {
    Value *from = getHintBase(LI->getPointerOperand());
    if (julia && isa<AllocaInst>(from))
      return true;
    return getLoadRoots(from, roots, julia, seen);
  }
  if (auto *Phi = dyn_cast<PHINode>(V)) {
    for (Value *In : Phi->incoming_values())
      if (!getLoadRoots(In, roots, julia, seen))
        return false;
    return true;
  }
  if (auto *Sel = dyn_cast<SelectInst>(V))
    return getLoadRoots(Sel->getTrueValue(), roots, julia, seen) &&
           getLoadRoots(Sel->getFalseValue(), roots, julia, seen);
  return false;
}

/// The arguments of `step` it may write through, and a store through a
/// pointer that is none of its arguments, globals or stack slots, if any.
/// Pointers of unknown origin passed to a call are not counted: they are
/// as often opaque handles (a stream, a file, a communicator), and what the
/// callee writes through memory it is not visibly given cannot be seen here
/// in any case; that is what __enzyme_ptr_size_hint is for.
///
/// A store through a pointer the step loads from an argument (the data of a
/// Julia array, which the step loads from the array each time) is a write to
/// the memory that argument points to, and the argument is in `indirect`.
///
/// In Julia code (`julia`) a callee may write what any object it is passed
/// points to, as Julia marks an argument readonly when the callee does not
/// write the object itself: those arguments are in `indirect` too.
static Instruction *getWrittenArgs(Function *step,
                                   SmallPtrSetImpl<Argument *> &written,
                                   SmallPtrSetImpl<Argument *> &indirect,
                                   bool julia) {
  Instruction *unknown = nullptr;
  // `readonly`: a call does not write through `ptr` itself.
  auto note = [&](Value *ptr, Instruction *I, bool direct = true,
                  bool readonly = false) {
    Value *base = getHintBase(ptr);
    if (isFreshAllocation(base))
      return;
    // Through pointers loaded from an argument (an array held by a struct,
    // the element of an array of arrays, whichever one a phi picks).
    if (direct &&
        (isa<LoadInst>(base) || isa<PHINode>(base) || isa<SelectInst>(base))) {
      SmallPtrSet<Argument *, 2> roots;
      if (getLoadRoots(base, roots, julia)) {
        indirect.insert(roots.begin(), roots.end());
        return;
      }
    }
    if (auto *A = dyn_cast<Argument>(base)) {
      // A callee may write what a Julia object points to (an array's data)
      // even when it does not write the object, which is all `readonly`
      // says of it.
      if (!direct && julia)
        indirect.insert(A);
      if (!readonly)
        written.insert(A);
    }
    // A function (a kernel launched, a callback) is not memory written.
    else if (direct && !isa<GlobalVariable>(base) && !isa<AllocaInst>(base) &&
             !isa<Function>(base) && !isa<ConstantPointerNull>(base) &&
             !unknown)
      unknown = I;
  };
  for (Instruction &I : instructions(step)) {
    if (auto *SI = dyn_cast<StoreInst>(&I))
      note(SI->getPointerOperand(), &I);
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      note(RMW->getPointerOperand(), &I);
    else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
      note(CX->getPointerOperand(), &I);
    else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
      note(MI->getDest(), &I);
    else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(CB) || CB->onlyReadsMemory())
        continue;
      for (unsigned i = 0; i < CB->arg_size(); i++) {
        Value *A = CB->getArgOperand(i);
        if (A->getType()->isPointerTy() && (!CB->onlyReadsMemory(i) || julia))
          note(A, &I, /*direct*/ false, CB->onlyReadsMemory(i));
      }
    }
  }
  return unknown;
}

static bool isPtrSizeHint(const Function *F) {
  return F && F->getName().contains("__enzyme_ptr_size_hint");
}

/// Where the pointer `B` was loaded from, if it was.
static Value *getLoadedFrom(Value *B) {
  auto *LI = dyn_cast<LoadInst>(B);
  return LI ? LI->getPointerOperand()->stripPointerCasts() : nullptr;
}

/// Whether the loop may write the object `Obj` points into.
static bool mayWriteInLoop(Value *Obj, ArrayRef<BasicBlock *> blocks) {
  Value *base = getBaseObject(Obj);
  for (BasicBlock *BB : blocks)
    for (Instruction &I : *BB) {
      Value *ptr = nullptr;
      if (auto *SI = dyn_cast<StoreInst>(&I))
        ptr = SI->getPointerOperand();
      else if (auto *MI = dyn_cast<MemIntrinsic>(&I))
        ptr = MI->getDest();
      else if (auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->onlyReadsMemory())
          continue;
        for (unsigned i = 0; i < CB->arg_size(); i++) {
          Value *A = CB->getArgOperand(i);
          if (A->getType()->isPointerTy() && !CB->onlyReadsMemory(i) &&
              getBaseObject(A) == base)
            return true;
        }
        continue;
      }
      if (ptr && getBaseObject(ptr) == base)
        return true;
    }
  return false;
}

/// The object `V` points into, its size and its memory space from a call
/// `__enzyme_ptr_size_hint(ptr, bytes[, space])` in front of `IP`, as
/// Enzyme-MLIR reads it: the extent of an allocation Enzyme did not see made,
/// and the memory space it really is in (a cudaMalloc'ed buffer is a plain
/// pointer).
///
/// A Julia array's data pointer is loaded from the array where it is used, so
/// the hint and the loop may hold two loads of it: they are the same pointer
/// when they load from the same place and the loop does not write it.
///
/// With `indirect`, the hint is for the memory `V` points to: its pointer is
/// loaded from `V`, as the data of a Julia array is from the array.
static std::optional<std::tuple<Value *, Value *, unsigned>>
getSizeHint(Value *V, Instruction *IP, DominatorTree &DT,
            ArrayRef<BasicBlock *> loop, bool indirect = false) {
  Value *base = getHintBase(V);
  auto matches = [&](Value *P) {
    Value *B = getHintBase(P);
    Value *from = getLoadedFrom(B);
    if (indirect)
      return from && from == V->stripPointerCasts() &&
             !mayWriteInLoop(from, loop);
    if (B == base)
      return true;
    return from && from == getLoadedFrom(base) && !mayWriteInLoop(from, loop);
  };
  for (Instruction &I : instructions(*IP->getFunction())) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI || !isPtrSizeHint(getFunctionFromCall(CI)) || CI->arg_size() < 2 ||
        !DT.dominates(CI, IP) || !matches(CI->getArgOperand(0)))
      continue;
    unsigned space = 0;
    if (CI->arg_size() > 2) {
      auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(2));
      if (!C)
        continue;
      space = C->getZExtValue();
    }
    IRBuilder<> B(IP);
    return std::make_tuple(
        CI->getArgOperand(0),
        B.CreateZExtOrTrunc(CI->getArgOperand(1), B.getInt64Ty()), space);
  }
  return {};
}

/// Where a region starts, as the loads and constant offsets that reach it
/// from a value: two regions with the same address are the same memory, as
/// the data of an array a struct holds, loaded for the struct and for the
/// array, both in front of the loop.
static SmallVector<const void *, 4> getRegionAddress(Value *V) {
  SmallVector<const void *, 4> key;
  const DataLayout &DL = cast<Instruction>(V)->getModule()->getDataLayout();
  while (true) {
    V = V->stripPointerCasts();
    if (auto *LI = dyn_cast<LoadInst>(V)) {
      key.push_back(nullptr);
      V = LI->getPointerOperand();
      continue;
    }
    if (auto *GEP = dyn_cast<GEPOperator>(V)) {
      APInt off(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
      if (GEP->accumulateConstantOffset(DL, off)) {
        key.push_back((const void *)(uintptr_t)(off.getZExtValue() + 1));
        V = GEP->getPointerOperand();
        continue;
      }
    }
    if (auto *CI = dyn_cast<CallInst>(V))
      if (Function *F = getFunctionFromCall(CI))
        if (F->getName() == "julia.pointer_from_objref") {
          V = CI->getArgOperand(0);
          continue;
        }
    key.push_back(V);
    return key;
  }
}

static bool isJuliaState(const Function *F) {
  return F && (F->getName() == "__enzyme_julia_state" ||
               F->getName() == "__enzyme_julia_dynamic_state" ||
               F->getName() == "__enzyme_julia_ref_slots");
}

/// The callbacks of `__enzyme_julia_ref_slots(refs, pointers)` in front of
/// `IP`: a stack slot holding a Julia object reference (`refs`), or a pointer
/// into one's data (`pointers`), a value the loop carries or leaves, is a
/// callback region with them, so that the shadow slot follows the primal one
/// as the shadows of the state do.
static Value *getJuliaRefSlotCallbacks(Instruction *IP, DominatorTree &DT,
                                       bool pointers) {
  for (Instruction &I : instructions(*IP->getFunction())) {
    auto *CI = dyn_cast<CallInst>(&I);
    Function *F = CI ? getFunctionFromCall(CI) : nullptr;
    if (F && F->getName() == "__enzyme_julia_ref_slots" &&
        CI->arg_size() == 2 && DT.dominates(CI, IP))
      return CI->getArgOperand(pointers ? 1 : 0);
  }
  return nullptr;
}

/// The memory that can be written through the Julia value `V` points into,
/// or whose data `V` is, from a call `__enzyme_julia_state(obj, ptr, bytes,
/// ...)` in front of `IP`, which Enzyme.jl makes from the value's type: the
/// data of the arrays it holds, and the mutable objects among it. `self` is
/// whether `V` points into the value rather than being data loaded from it.
static std::optional<SmallVector<std::tuple<Value *, Value *, unsigned>, 2>>
getJuliaState(Value *V, Instruction *IP, DominatorTree &DT, bool &self) {
  SmallVector<Value *, 2> bases = {getBaseObject(V)};
  if (auto *LI = dyn_cast<LoadInst>(bases[0]))
    bases.push_back(getBaseObject(LI->getPointerOperand()));
  for (Instruction &I : instructions(*IP->getFunction())) {
    auto *CI = dyn_cast<CallInst>(&I);
    if (!CI || !isJuliaState(getFunctionFromCall(CI)) || CI->arg_size() < 1 ||
        !DT.dominates(CI, IP) ||
        !is_contained(bases, getBaseObject(CI->getArgOperand(0))))
      continue;
    self = getBaseObject(CI->getArgOperand(0)) == bases[0];
    SmallVector<std::tuple<Value *, Value *, unsigned>, 2> regions;
    IRBuilder<> B(IP);
    // __enzyme_julia_dynamic_state(value, root, callbacks, ...): state the
    // frontend snapshots itself, from each root, at run time.
    if (getFunctionFromCall(CI)->getName() == "__enzyme_julia_dynamic_state") {
      if (CI->arg_size() % 2 != 1)
        continue;
      for (unsigned i = 1; i + 1 < CI->arg_size(); i += 2)
        regions.push_back(
            {CI->getArgOperand(i),
             B.CreatePtrToInt(CI->getArgOperand(i + 1), B.getInt64Ty()),
             CallbackRegionSpace});
      return regions;
    }
    for (unsigned i = 1; i + 1 < CI->arg_size(); i += 2)
      regions.push_back(
          {CI->getArgOperand(i),
           B.CreateZExtOrTrunc(CI->getArgOperand(i + 1), B.getInt64Ty()), 0});
    return regions;
  }
  return {};
}

/// Outline the loop around the annotation `marker` into a checkpointed loop.
static bool outlineAnnotatedLoop(CallInst *marker) {
  Function &F = *marker->getFunction();
  Module &M = *F.getParent();
  LLVMContext &Ctx = M.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  Type *I32 = Type::getInt32Ty(Ctx);
  auto *Ptr = PointerType::getUnqual(Ctx);
  DebugLoc loc = marker->getDebugLoc();
  Instruction *anchor = marker;
  auto fail = [&](const Twine &msg) {
    std::string str = msg.str();
    EmitFailure("CheckpointLoop", loc, anchor, str);
    return false;
  };
  // Before the loop is found, a failure drops the annotation itself.
  auto failEarly = [&](const Twine &msg) {
    fail(msg);
    marker->eraseFromParent();
    return false;
  };

  auto *modeC = dyn_cast<ConstantInt>(marker->getArgOperand(0));
  if (!modeC)
    return failEarly("the checkpointing mode of a loop must be a constant");
  int64_t mode = modeC->getSExtValue();
  std::optional<uint64_t> count;
  if (marker->arg_size() > 1)
    if (auto *C = dyn_cast<ConstantInt>(marker->getArgOperand(1)))
      if (!C->isMinusOne())
        count = C->getZExtValue();
  if (mode > ENZYME_CKPT_SCHEDULE_BINOMIAL)
    return failEarly("unknown checkpointing mode " + Twine(mode));

  // The loop's variables as values, where they can be.
  {
    DominatorTree DT(F);
    SmallVector<AllocaInst *, 8> allocas;
    for (Instruction &I : F.getEntryBlock())
      if (auto *AI = dyn_cast<AllocaInst>(&I))
        if (isAllocaPromotable(AI))
          allocas.push_back(AI);
    if (!allocas.empty()) {
      AssumptionCache AC(F);
      PromoteMemToReg(allocas, DT, &AC);
    }
  }
  DominatorTree DT(F);
  LoopInfo LI(DT);
  AssumptionCache AC(F);
  TargetLibraryInfoImpl TLII{Triple(M.getTargetTriple())};
  TargetLibraryInfo TLI(TLII, &F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);

  Loop *L = LI.getLoopFor(marker->getParent());
  if (!L)
    return failEarly("__enzyme_set_checkpointing is not inside a loop");
  // The annotation may have been copied, by unrolling say.
  SmallVector<CallInst *, 2> markers;
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : *BB)
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (LI.getLoopFor(BB) == L &&
            isLoopAnnotation(getFunctionFromCall(CI)))
          markers.push_back(CI);
  for (CallInst *CI : markers)
    CI->eraseFromParent();
  anchor = &*L->getHeader()->getFirstNonPHIIt();
  if (mode < 1)
    return true;

  simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false);

  // Exits that can only throw (Julia's checks of an array's dimensions, an
  // `unreachable` after a failed assumption): the blocks that follow them,
  // up to the `unreachable`, become part of the step, which throws as the
  // loop did. They must not be reached otherwise.
  SmallPtrSet<BasicBlock *, 8> throwing;
  SmallVector<BasicBlock *, 4> exits;
  L->getUniqueExitBlocks(exits);
  BasicBlock *E = nullptr;
  for (BasicBlock *EB : exits) {
    SmallVector<BasicBlock *, 8> todo = {EB};
    SmallPtrSet<BasicBlock *, 8> seen = {EB};
    bool throws = true;
    while (!todo.empty() && throws) {
      BasicBlock *BB = todo.pop_back_val();
      Instruction *T = BB->getTerminator();
      if (isa<ReturnInst>(T) || isa<ResumeInst>(T) ||
          (T->getNumSuccessors() && !isAnyBranch(T) && !isa<SwitchInst>(T)))
        throws = false;
      for (BasicBlock *S : successors(BB)) {
        if (L->contains(S))
          throws = false;
        else if (seen.insert(S).second)
          todo.push_back(S);
      }
    }
    if (throws)
      throwing.insert(seen.begin(), seen.end());
    else if (E)
      return fail("a checkpointed loop must leave to a single block, "
                  "except to throw");
    else
      E = EB;
  }
  BasicBlock *P = L->getLoopPreheader(), *H = L->getHeader(),
             *latch = L->getLoopLatch(), *X = nullptr;
  if (E)
    for (BasicBlock *Pred : predecessors(E))
      if (L->contains(Pred))
        X = X && X != Pred ? (BasicBlock *)E : Pred;
  if (!P || !latch || !E || !X || X == E || (X != latch && X != H))
    return fail("a checkpointed loop must have a single latch and leave "
                "from its header or its latch, to a single block");
  // Blocks where it throws that are reached from outside the loop too (a
  // shared error path): the step gets a copy, and they stay.
  SmallPtrSet<BasicBlock *, 8> shared;
  {
    SmallVector<BasicBlock *, 8> todo;
    for (BasicBlock *BB : throwing)
      for (BasicBlock *Pred : predecessors(BB))
        if (!L->contains(Pred) && !throwing.count(Pred))
          if (shared.insert(BB).second)
            todo.push_back(BB);
    while (!todo.empty())
      for (BasicBlock *S : successors(todo.pop_back_val()))
        if (throwing.count(S) && shared.insert(S).second)
          todo.push_back(S);
  }
  // What the step runs: the loop and where it throws.
  auto inStepBlock = [&](BasicBlock *BB) {
    return L->contains(BB) || throwing.count(BB);
  };
  auto inStep = [&](Instruction *I) { return inStepBlock(I->getParent()); };
  // The steps that do not throw.
  const SCEV *BTC =
      throwing.empty() ? SE.getBackedgeTakenCount(L) : SE.getExitCount(L, X);
  if (isa<SCEVCouldNotCompute>(BTC))
    return fail("the number of iterations of a checkpointed loop must be "
                "known when it starts; for other loops use "
                "__enzyme_checkpoint_while");
  // A step runs the header and what follows it up to the backedge or the
  // exit, so there is a step for every time the header runs: one more than
  // the backedge is taken. A loop that leaves from its header does so in its
  // last step, having run only the header.
  const SCEV *N =
      SE.getAddExpr(SE.getTruncateOrZeroExtend(BTC, I64), SE.getOne(I64));

  Instruction *IP = P->getTerminator();
#if LLVM_VERSION_MAJOR >= 19
  SCEVExpander Exp(SE, "ckpt");
#else
  SCEVExpander Exp(SE, M.getDataLayout(), "ckpt");
#endif
  Value *nsteps = Exp.expandCodeFor(N, I64, IP);

  // Induction variables, recomputed in each step from its index.
  struct IV {
    PHINode *phi;
    Value *start, *step;
  };
  SmallVector<IV, 4> ivs;
  SmallVector<PHINode *, 4> carried;
  for (PHINode &phi : H->phis()) {
    bool usedAfter = any_of(
        phi.users(), [&](User *U) { return !inStep(cast<Instruction>(U)); });
    auto *AR = SE.isSCEVable(phi.getType())
                   ? dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&phi))
                   : nullptr;
    if (!usedAfter && AR && AR->getLoop() == L && AR->isAffine() &&
        (phi.getType()->isIntegerTy() || phi.getType()->isPointerTy())) {
      const SCEV *stepS = AR->getStepRecurrence(SE);
      ivs.push_back({&phi, Exp.expandCodeFor(AR->getStart(), phi.getType(), IP),
                     Exp.expandCodeFor(stepS, stepS->getType(), IP)});
    } else
      carried.push_back(&phi);
  }

  // Everything else carried from one iteration to the next, or used after
  // the loop, through the stack.
#if LLVM_VERSION_MAJOR >= 19
  auto allocaIP = F.getEntryBlock().getFirstInsertionPt();
#else
  Instruction *allocaIP = &*F.getEntryBlock().getFirstInsertionPt();
#endif
  FoldSingleEntryPHINodes(E);
  for (PHINode *phi : carried)
    DemotePHIToStack(phi, allocaIP);
  for (BasicBlock *BB : L->blocks())
    for (Instruction &I : make_early_inc_range(*BB)) {
      // The header's phis are carried or recomputed, above; another block's
      // phi (a value picked in the latch) may be used after the loop too.
      if (isa<PHINode>(I) && BB == H)
        continue;
      if (any_of(I.users(),
                 [&](User *U) { return !inStep(cast<Instruction>(U)); }))
        DemoteRegToStack(I, false, allocaIP);
    }

  // Floating-point values the loop uses but does not compute go to the
  // step by reference, through a stack slot: the driver passes the step's
  // arguments by reference to take their derivatives, which accumulate in
  // the slot's shadow.
  {
    const DataLayout &DL = M.getDataLayout();
    SmallMapVector<Value *, AllocaInst *, 4> slots;
    SmallVector<BasicBlock *, 8> stepBlocks(L->blocks().begin(),
                                            L->blocks().end());
    stepBlocks.append(throwing.begin(), throwing.end());
    for (BasicBlock *BB : stepBlocks)
      for (Instruction &I : *BB)
        for (Use &U : I.operands()) {
          Value *V = U.get();
          if (!V->getType()->isFPOrFPVectorTy())
            continue;
          if (auto *VI = dyn_cast<Instruction>(V)) {
            if (inStep(VI))
              continue;
          } else if (!isa<Argument>(V))
            continue;
          auto *Phi = dyn_cast<PHINode>(&I);
          if (Phi && !inStepBlock(Phi->getIncomingBlock(U)))
            continue;
          AllocaInst *&slot = slots[V];
          if (!slot) {
            slot = new AllocaInst(V->getType(), DL.getAllocaAddrSpace(),
                                  V->getName() + ".byref", allocaIP);
            IRBuilder<>(IP).CreateStore(V, slot);
          }
          Instruction *at =
              Phi ? Phi->getIncomingBlock(U)->getTerminator() : &I;
          U.set(IRBuilder<>(at).CreateLoad(V->getType(), slot,
                                           V->getName() + ".ld"));
        }
  }

  // The step: one iteration, from the header to the backedge or the exit.
  SmallVector<BasicBlock *, 8> blocks(L->blocks().begin(),
                                      L->blocks().end());
  blocks.append(throwing.begin(), throwing.end());
  SmallPtrSet<Value *, 4> ivPhis;
  for (auto &iv : ivs)
    ivPhis.insert(iv.phi);
  SetVector<Value *> liveins;
  auto noteLivein = [&](Value *V) {
    if (auto *I = dyn_cast<Instruction>(V)) {
      if (!inStep(I))
        liveins.insert(V);
    } else if (isa<Argument>(V))
      liveins.insert(V);
  };
  for (BasicBlock *BB : blocks)
    for (Instruction &I : *BB) {
      if (ivPhis.count(&I))
        continue;
      auto *Phi = dyn_cast<PHINode>(&I);
      for (Use &U : I.operands())
        // Not what a shared error path gets from outside the loop.
        if (!Phi || inStepBlock(Phi->getIncomingBlock(U)))
          noteLivein(U.get());
    }
  for (auto &iv : ivs) {
    noteLivein(iv.start);
    noteLivein(iv.step);
  }
  // Julia's derived pointers (into an object, 11, or loaded from it, 13)
  // may not be stored to memory, where the driver keeps the step's
  // arguments, nor live across its calls: the step makes them again from
  // the objects they derive from.
  SmallVector<Instruction *, 4> remat;
  if (M.getFunction("julia.get_pgcstack") || M.getFunction("julia.gc_loaded")) {
    auto isDerived = [](Value *V) {
      auto *PT = dyn_cast<PointerType>(V->getType());
      if (!PT || PT->getAddressSpace() < 11 || PT->getAddressSpace() > 13)
        return false;
      if (isa<AddrSpaceCastInst>(V) || isa<GetElementPtrInst>(V) ||
          isa<BitCastInst>(V))
        return true;
      auto *CI = dyn_cast<CallInst>(V);
      Function *callee = CI ? getFunctionFromCall(CI) : nullptr;
      return callee && callee->getName() == "julia.gc_loaded";
    };
    bool again = true;
    while (again) {
      again = false;
      for (Value *V : liveins.getArrayRef()) {
        if (!isDerived(V))
          continue;
        auto *I = cast<Instruction>(V);
        liveins.remove(V);
        remat.push_back(I);
        for (Value *Op : I->operands())
          if (!isa<Function>(Op))
            noteLivein(Op);
        again = true;
        break;
      }
    }
    // Each before what is made from it.
    llvm::stable_sort(remat, [&](Instruction *A, Instruction *B) {
      return A != B && DT.dominates(A, B);
    });
  }

  SmallVector<Type *, 8> params = {I64};
  for (Value *V : liveins)
    params.push_back(V->getType());
  auto *step = Function::Create(FunctionType::get(Type::getVoidTy(Ctx),
                                                  params, false),
                                GlobalValue::InternalLinkage,
                                F.getName() + ".ckpt.step", &M);
  ValueToValueMapTy VMap;
  for (auto [i, V] : enumerate(liveins)) {
    step->getArg(i + 1)->setName(V->getName());
    VMap[V] = step->getArg(i + 1);
  }
  auto mapped = [&](Value *V) -> Value * {
    return isa<Constant>(V) ? V : (Value *)VMap[V];
  };
  auto *entry = BasicBlock::Create(Ctx, "entry", step);
  IRBuilder<> SB(entry);
  for (Instruction *I : remat) {
    Instruction *C = I->clone();
    SB.Insert(C, I->getName());
    RemapInstruction(C, VMap, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    C->setDebugLoc(DebugLoc());
    VMap[I] = C;
  }
  Value *k = step->getArg(0);
  k->setName("k");
  SmallVector<std::pair<PHINode *, Value *>, 4> ivValues;
  for (auto &iv : ivs) {
    Value *start = mapped(iv.start), *stride = mapped(iv.step);
    Value *off = SB.CreateMul(SB.CreateSExtOrTrunc(k, stride->getType()),
                              stride);
    Value *v = iv.phi->getType()->isPointerTy()
                   ? SB.CreateGEP(SB.getInt8Ty(), start, off)
                   : SB.CreateAdd(start,
                                  SB.CreateSExtOrTrunc(off, start->getType()));
    ivValues.push_back({iv.phi, v});
  }
  SmallVector<BasicBlock *, 8> cloned;
  for (BasicBlock *BB : blocks) {
    auto *NB = CloneBasicBlock(BB, VMap, "", step);
    VMap[BB] = NB;
    cloned.push_back(NB);
  }
  auto *next = BasicBlock::Create(Ctx, "next", step);
  ReturnInst::Create(Ctx, next);
  VMap[E] = next;
  SmallVector<Instruction *, 4> deadPhis;
  for (auto [phi, v] : ivValues) {
    deadPhis.push_back(cast<Instruction>(VMap[phi]));
    VMap[phi] = v;
  }
  // The copies of shared error paths are entered from the step only.
  for (BasicBlock *NB : cloned)
    for (PHINode &Phi : NB->phis())
      for (unsigned i = Phi.getNumIncomingValues(); i-- > 0;)
        if (!inStepBlock(Phi.getIncomingBlock(i)))
          Phi.removeIncomingValue(i, /*DeletePHIIfEmpty*/ false);
  remapInstructionsInBlocks(cloned, VMap);
  for (Instruction *I : deadPhis) {
    I->dropAllReferences();
    I->eraseFromParent();
  }
  auto *newH = cast<BasicBlock>(VMap[H]);
  SB.CreateBr(newH);
  cast<BasicBlock>(VMap[latch])->getTerminator()->replaceSuccessorWith(newH,
                                                                        next);
  // The step is a function of its own: the loop's debug locations are not,
  // and the loop's stack slots are arguments, which lifetime markers cannot
  // apply to.
  for (BasicBlock &BB : *step)
    for (Instruction &I : make_early_inc_range(BB)) {
      I.setDebugLoc(DebugLoc());
#if LLVM_VERSION_MAJOR >= 19
      I.dropDbgRecords();
#endif
      if (auto *II = dyn_cast<IntrinsicInst>(&I))
        if (isa<DbgInfoIntrinsic>(II) || II->isLifetimeStartOrEnd())
          II->eraseFromParent();
    }

  // What a snapshot holds besides the globals.
  SmallPtrSet<Argument *, 8> written, indirect;
  bool julia =
      M.getFunction("julia.get_pgcstack") || M.getFunction("julia.gc_loaded");
  if (Instruction *I = getWrittenArgs(step, written, indirect, julia)) {
    std::string inst;
    raw_string_ostream ss(inst);
    ss << *I;
    step->eraseFromParent();
    return fail("a checkpointed loop writes through a pointer it loads from "
                "memory, whose extent is not known (" + ss.str() +
                "); give the regions with __enzyme_checkpoint_for");
  }
  SmallVector<std::tuple<Value *, Value *, unsigned>, 4> regions;
  std::set<SmallVector<const void *, 4>> seen;
  auto unseen = [&](Value *ptr) {
    return isa<Instruction>(ptr) ? seen.insert(getRegionAddress(ptr)).second
                                 : seen.insert({ptr}).second;
  };
  for (auto [i, V] : enumerate(liveins)) {
    if (!V->getType()->isPointerTy() || isJuliaTaskState(V))
      continue;
    bool w = written.count(step->getArg(i + 1));
    bool ind = indirect.count(step->getArg(i + 1));
    if (julia)
      if (auto *AI = dyn_cast<AllocaInst>(V))
        if (auto *PT = dyn_cast<PointerType>(AI->getAllocatedType()))
          if (PT->getAddressSpace() == 10 || PT->getAddressSpace() == 0)
            if (Value *cb = getJuliaRefSlotCallbacks(
                    IP, DT, PT->getAddressSpace() == 0)) {
              if (unseen(V)) {
                IRBuilder<> B(IP);
                regions.push_back({V, B.CreatePtrToInt(cb, B.getInt64Ty()),
                                   CallbackRegionSpace});
              }
              continue;
            }
    // What a Julia value holds, as its type says.
    if (julia && (w || ind)) {
      bool self = false;
      if (auto state = getJuliaState(V, IP, DT, self)) {
        Value *base = getBaseObject(V);
        bool slot = isa<AllocaInst>(base);
        if (w && self && !slot && none_of(*state, [&](auto &r) {
              return getBaseObject(std::get<0>(r)) == base;
            })) {
          std::string name;
          raw_string_ostream ss(name);
          V->printAsOperand(ss, false);
          step->eraseFromParent();
          return fail("a checkpointed loop writes the Julia object " +
                      ss.str() +
                      " itself, of which only what it holds can be in a "
                      "snapshot (does it resize an array?)");
        }
        for (auto &r : *state)
          if (unseen(std::get<0>(r)))
            regions.push_back(r);
        if (!slot)
          continue;
        ind = false;
      }
    }
    // The memory V points to, which the loop writes through a pointer it
    // loads from V; V itself, which holds that pointer, is not written, so
    // the pointer loaded before the loop is the one the loop writes through.
    if (ind) {
      auto hint = getSizeHint(V, IP, DT, blocks, /*indirect*/ true);
      if (!hint) {
        std::string name;
        raw_string_ostream ss(name);
        V->printAsOperand(ss, false);
        step->eraseFromParent();
        return fail("a checkpointed loop writes through a pointer it loads "
                    "from " +
                    ss.str() +
                    ", whose extent is not known before "
                    "it; give it with __enzyme_ptr_size_hint on that pointer");
      }
      if (unseen(std::get<0>(*hint)))
        regions.push_back(*hint);
      if (!w)
        continue;
    }
    if (auto hint = getSizeHint(V, IP, DT, blocks)) {
      if (unseen(std::get<0>(*hint)))
        regions.push_back(*hint);
      continue;
    }
    auto alloc = getKnownAllocation(V, IP, DT);
    if (!alloc) {
      if (w) {
        std::string name;
        raw_string_ostream ss(name);
        V->printAsOperand(ss, false);
        step->eraseFromParent();
        return fail("a checkpointed loop writes through " + ss.str() +
                    ", whose extent is not known before it; give it with "
                    "__enzyme_ptr_size_hint, or the regions with "
                    "__enzyme_checkpoint_for");
      }
      continue;
    }
    // Stack slots always: they are the loop's own state.
    if ((w || isa<AllocaInst>(alloc->first)) && unseen(alloc->first))
      regions.push_back({alloc->first, alloc->second, 0});
  }

  // The scheme: the reference one of enzyme/checkpoint.h if it is here, or
  // __enzyme_checkpoint_builtin(mode) from the runtime.
  IRBuilder<> B(IP);
  // (In C++ the header's tables have internal names, mangled.)
  StringRef table =
      mode == ENZYME_CKPT_SCHEDULE_REVOLVE     ? "EnzymeCkptRevolve"
      : mode == ENZYME_CKPT_SCHEDULE_STORE_ALL ? "EnzymeCkptStoreAll"
      : mode == ENZYME_CKPT_SCHEDULE_BINOMIAL  ? "EnzymeCkptBinomial"
                                               : "EnzymeCkptPeriodic";
  Value *vt = nullptr;
  for (GlobalVariable &G : M.globals())
    if (G.getName() == table ||
        (G.hasLocalLinkage() && G.getName().starts_with("_ZL") &&
         G.getName().ends_with(table)))
      vt = &G;
  if (!vt) {
    FunctionCallee builtin = M.getOrInsertFunction(
        "__enzyme_checkpoint_builtin", FunctionType::get(Ptr, {I64}, false));
    // The scheme carries no derivative, wherever it is defined, and the call
    // only returns the address of a constant table.
    if (auto *F = dyn_cast<Function>(builtin.getCallee())) {
      F->addFnAttr(Attribute::get(Ctx, "enzyme_inactive"));
      F->addFnAttr(Attribute::get(Ctx, "enzyme_no_escaping_allocation"));
      F->setDoesNotAccessMemory();
      F->setDoesNotThrow();
      F->setWillReturn();
    }
    auto *call = B.CreateCall(builtin, {ConstantInt::get(I64, mode)});
    call->addFnAttr(Attribute::get(Ctx, "enzyme_inactive"));
    call->setDoesNotAccessMemory();
    call->setMetadata("enzyme_inactive", MDNode::get(Ctx, {}));
    vt = call;
  }
  // EnzymeCkptConfig: the budget, or 0 for the schedule's default
  // (enzyme/checkpoint_schedule.h), which Enzyme-MLIR's takes too.
  auto *CfgTy = StructType::get(Ctx, {I64, I32, Ptr, I64, Ptr});
  IRBuilder<> EB(&*allocaIP);
  auto *cfg = EB.CreateAlloca(CfgTy, nullptr, "ckpt.config");
  Value *budget = ConstantInt::get(I64, count ? *count : 0);
  B.CreateStore(budget, B.CreateStructGEP(CfgTy, cfg, 0));
  B.CreateStore(ConstantInt::get(I32, EnzymeCheckpointLoopVerbose),
                B.CreateStructGEP(CfgTy, cfg, 1));
  B.CreateStore(ConstantPointerNull::get(Ptr),
                B.CreateStructGEP(CfgTy, cfg, 2));
  B.CreateStore(ConstantInt::get(I64, 0), B.CreateStructGEP(CfgTy, cfg, 3));
  B.CreateStore(ConstantPointerNull::get(Ptr),
                B.CreateStructGEP(CfgTy, cfg, 4));

  SmallVector<Type *, 4> regionTypes;
  SmallVector<Value *, 8> args = {ConstantInt::get(I64, 0), nsteps, vt, cfg};
  std::string spaces;
  bool device = false;
  SmallVector<bool, 4> callbackRegions;
  for (auto &[ptr, bytes, space] : regions) {
    regionTypes.push_back(ptr->getType());
    args.push_back(ptr);
    args.push_back(bytes);
    spaces += (spaces.empty() ? "" : ",") + std::to_string(space);
    device |= space != 0;
    callbackRegions.push_back(space == CallbackRegionSpace);
  }
  for (Value *V : liveins)
    args.push_back(V);
  Function *loop =
      createLoopFunction(M, step, regionTypes, vt->getType(), cfg->getType(),
                         /*isWhile*/ false, callbackRegions);
  if (device)
    loop->addFnAttr(CheckpointRegionSpacesAttr, spaces);
  B.CreateCall(loop, args)->setDebugLoc(loc);

  // The loop is now the call.
  IP->eraseFromParent();
  IRBuilder<>(P).CreateBr(E);
  SmallVector<BasicBlock *, 8> dead;
  for (BasicBlock *BB : blocks)
    if (!shared.count(BB))
      dead.push_back(BB);
  for (BasicBlock *BB : shared)
    for (BasicBlock *Pred : dead)
      if (is_contained(predecessors(BB), Pred))
        BB->removePredecessor(Pred);
  for (BasicBlock *BB : dead)
    BB->dropAllReferences();
  for (BasicBlock *BB : dead)
    BB->eraseFromParent();
  return true;
}

// A loop can also be annotated with its loop metadata, the way Julia spells
// loop annotations (`Expr(:loopinfo, (Symbol("enzyme.checkpoint"), "revolve",
// 4))` at the end of the loop body) and other front ends without a call to
// emit can:
//
//   br i1 %c, label %exit, label %header, !llvm.loop !0
//   !0 = distinct !{!0, !1}
//   !1 = !{!"enzyme.checkpoint", !"revolve", i64 4}
//
// The mode is "revolve" or "binomial" (binomial), "periodic" or "regular"
// (periodic), "none", or the integer of __enzyme_set_checkpointing; the count
// is optional. Each such loop gets the equivalent __enzyme_set_checkpointing
// call at the top of its header, and the entry is removed from its metadata.

static constexpr const char *CheckpointLoopMD = "enzyme.checkpoint";

static MDNode *getCheckpointLoopMD(Loop *L) {
  MDNode *LoopID = L->getLoopID();
  if (!LoopID)
    return nullptr;
  for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; i++)
    if (auto *N = dyn_cast_or_null<MDNode>(LoopID->getOperand(i)))
      if (N->getNumOperands() > 0)
        if (auto *S = dyn_cast_or_null<MDString>(N->getOperand(0)))
          if (S->getString() == CheckpointLoopMD)
            return N;
  return nullptr;
}

static bool markersFromLoopMetadata(Function &F) {
  bool hasLoopMD = false;
  for (BasicBlock &BB : F)
    if (BB.getTerminator() &&
        BB.getTerminator()->getMetadata(LLVMContext::MD_loop))
      hasLoopMD = true;
  if (!hasLoopMD)
    return false;
  LLVMContext &Ctx = F.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  DominatorTree DT(F);
  LoopInfo LI(DT);
  bool changed = false;
  for (Loop *L : LI.getLoopsInPreorder()) {
    MDNode *N = getCheckpointLoopMD(L);
    if (!N)
      continue;
    Instruction *IP = &*L->getHeader()->getFirstInsertionPt();
    DebugLoc loc = L->getStartLoc();
    int64_t mode = 1, count = -1;
    bool ok = true;
    auto fail = [&](const Twine &msg) {
      std::string str = msg.str();
      EmitFailure("CheckpointLoop", loc, IP, str);
      ok = false;
    };
    if (N->getNumOperands() > 3)
      fail("enzyme.checkpoint loop metadata takes at most a mode and a "
           "count");
    if (ok && N->getNumOperands() > 1) {
      const MDOperand &op = N->getOperand(1);
      if (auto *S = dyn_cast_or_null<MDString>(op)) {
        StringRef name = S->getString();
        mode = getScheduleTag(name);
        if (mode < 0)
          fail("unknown checkpointing mode '" + name +
               "', expected \"revolve\", \"binomial\", \"periodic\", "
               "\"store_all\" or \"none\"");
      } else if (auto *C = mdconst::dyn_extract_or_null<ConstantInt>(op)) {
        mode = C->getSExtValue();
      } else {
        fail("the checkpointing mode of a loop must be a string or an "
             "integer");
      }
    }
    if (ok && N->getNumOperands() > 2) {
      auto *C = mdconst::dyn_extract_or_null<ConstantInt>(N->getOperand(2));
      if (!C)
        fail("the number of checkpoints of a loop must be an integer");
      else
        count = C->getSExtValue();
    }

    // The entry has been read; drop it so the loop is annotated only once.
    MDNode *LoopID = L->getLoopID();
    SmallVector<Metadata *, 4> MDs(1);
    for (unsigned i = 1, e = LoopID->getNumOperands(); i < e; i++)
      if (LoopID->getOperand(i) != N)
        MDs.push_back(LoopID->getOperand(i));
    if (MDs.size() == 1) {
      SmallVector<BasicBlock *, 2> latches;
      L->getLoopLatches(latches);
      for (BasicBlock *BB : latches)
        BB->getTerminator()->setMetadata(LLVMContext::MD_loop, nullptr);
    } else {
      MDNode *NewID = MDNode::getDistinct(Ctx, MDs);
      NewID->replaceOperandWith(0, NewID);
      L->setLoopID(NewID);
    }
    changed = true;
    if (!ok)
      continue;

    FunctionCallee marker = F.getParent()->getOrInsertFunction(
        "__enzyme_set_checkpointing",
        FunctionType::get(Type::getVoidTy(Ctx), {I64, I64}, false));
    IRBuilder<> B(IP);
    auto *CI = B.CreateCall(
        marker, {ConstantInt::get(I64, mode), ConstantInt::get(I64, count)});
    CI->setDebugLoc(loc);
  }
  return changed;
}

bool keepCheckpointLoops(Module &M) {
  bool changed = false;
  LLVMContext &Ctx = M.getContext();
  auto hint = [&](StringRef name, Metadata *V = nullptr) -> Metadata * {
    SmallVector<Metadata *, 2> ops = {MDString::get(Ctx, name)};
    if (V)
      ops.push_back(V);
    return MDNode::get(Ctx, ops);
  };
  auto *I1 = Type::getInt1Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  SmallVector<Metadata *, 4> hints = {
      hint("llvm.loop.unroll.disable"),
      hint("llvm.loop.vectorize.enable",
           ConstantAsMetadata::get(ConstantInt::get(I1, 0))),
      hint("llvm.loop.interleave.count",
           ConstantAsMetadata::get(ConstantInt::get(I32, 1))),
      hint("llvm.loop.distribute.enable",
           ConstantAsMetadata::get(ConstantInt::get(I1, 0)))};
  DenseMap<MDNode *, MDNode *> done;
  for (Function &F : M)
    for (BasicBlock &BB : F) {
      Instruction *T = BB.getTerminator();
      MDNode *LoopID = T ? T->getMetadata(LLVMContext::MD_loop) : nullptr;
      if (!LoopID)
        continue;
      MDNode *&NewID = done[LoopID];
      if (!NewID) {
        bool annotated = false;
        SmallVector<Metadata *, 8> ops = {nullptr};
        for (unsigned i = 1; i < LoopID->getNumOperands(); i++) {
          Metadata *Op = LoopID->getOperand(i);
          // The hints replace whatever the loop said about these.
          if (auto *N = dyn_cast_or_null<MDNode>(Op))
            if (N->getNumOperands())
              if (auto *S = dyn_cast_or_null<MDString>(N->getOperand(0))) {
                if (S->getString() == CheckpointLoopMD)
                  annotated = true;
                if (S->getString().starts_with("llvm.loop.unroll.") ||
                    S->getString().starts_with("llvm.loop.vectorize.") ||
                    S->getString() == "llvm.loop.interleave.count" ||
                    S->getString() == "llvm.loop.distribute.enable")
                  continue;
              }
          ops.push_back(Op);
        }
        if (!annotated) {
          NewID = LoopID;
          continue;
        }
        ops.append(hints.begin(), hints.end());
        NewID = MDNode::getDistinct(Ctx, ops);
        NewID->replaceOperandWith(0, NewID);
      }
      if (NewID != LoopID) {
        T->setMetadata(LLVMContext::MD_loop, NewID);
        changed = true;
      }
    }
  return changed;
}

static bool outlineAnnotatedLoops(Module &M) {
  bool changed = false;
  for (Function &F : M)
    if (!F.isDeclaration())
      changed |= markersFromLoopMetadata(F);
  while (true) {
    CallInst *marker = nullptr;
    for (Function &F : M) {
      for (Instruction &I : instructions(F))
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (isLoopAnnotation(getFunctionFromCall(CI))) {
            marker = CI;
            break;
          }
      if (marker)
        break;
    }
    if (!marker)
      return changed;
    // A failure is reported, and the annotation is gone either way.
    outlineAnnotatedLoop(marker);
    changed = true;
  }
}

bool lowerCheckpointMarkers(Module &M) {
  bool annotated = outlineAnnotatedLoops(M);
  // The size hints have been read; they have no run-time effect.
  for (Function &F : M)
    for (Instruction &I : make_early_inc_range(instructions(F)))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (isPtrSizeHint(getFunctionFromCall(CI)) ||
            isJuliaState(getFunctionFromCall(CI))) {
          CI->eraseFromParent();
          annotated = true;
        }
  SmallVector<std::pair<CallInst *, bool>, 4> calls;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto *callee =
            dyn_cast<Function>(CI->getCalledOperand()->stripPointerCasts());
        if (!callee)
          continue;
        // Fortran callers use an implicit interface to f__enzyme_...
        if (callee->getName().contains("__enzyme_checkpoint_for"))
          calls.push_back({CI, false});
        else if (callee->getName().contains("__enzyme_checkpoint_while"))
          calls.push_back({CI, true});
      }
  bool changed = annotated;
  for (auto [CI, isWhile] : calls)
    changed |= lowerMarker(CI, isWhile);
  return changed;
}

//===----------------------------------------------------------------------===//
// What a snapshot holds
//===----------------------------------------------------------------------===//

namespace {
struct GlobalAccesses {
  SmallSetVector<GlobalVariable *, 8> written, read;
};
} // namespace

static void noteAccess(Value *ptr, bool write, GlobalAccesses &acc) {
  if (auto *GV = dyn_cast<GlobalVariable>(getBaseObject(ptr))) {
    if (write)
      acc.written.insert(GV);
    else
      acc.read.insert(GV);
  }
}

static void scanFunction(Function &F, GlobalAccesses &acc,
                         SmallVectorImpl<Function *> *callees) {
  for (Instruction &I : instructions(F)) {
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      noteAccess(SI->getPointerOperand(), true, acc);
    } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
      noteAccess(LI->getPointerOperand(), false, acc);
    } else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
      noteAccess(RMW->getPointerOperand(), true, acc);
      noteAccess(RMW->getPointerOperand(), false, acc);
    } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
      noteAccess(CX->getPointerOperand(), true, acc);
      noteAccess(CX->getPointerOperand(), false, acc);
    } else if (auto *MT = dyn_cast<MemTransferInst>(&I)) {
      noteAccess(MT->getDest(), true, acc);
      noteAccess(MT->getSource(), false, acc);
    } else if (auto *MS = dyn_cast<MemSetInst>(&I)) {
      noteAccess(MS->getDest(), true, acc);
    } else if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (isa<IntrinsicInst>(CB))
        continue;
      Function *callee = getFunctionFromCall(CB);
      if (callee && !callee->empty()) {
        if (callees)
          callees->push_back(callee);
        continue;
      }
      // A body-less callee may read or write any global passed to it.
      for (unsigned i = 0; i < CB->arg_size(); i++) {
        if (!CB->getArgOperand(i)->getType()->isPointerTy())
          continue;
        noteAccess(CB->getArgOperand(i), false, acc);
        if (!CB->onlyReadsMemory(i))
          noteAccess(CB->getArgOperand(i), true, acc);
      }
    }
  }
}

/// The globals a snapshot before a step of `step` must hold: those the step
/// writes, and those it reads that other code may write.
static SmallVector<GlobalVariable *, 8> getGlobalRegions(Function *step) {
  Module &M = *step->getParent();

  GlobalAccesses inStep;
  SmallPtrSet<Function *, 16> closure;
  SmallVector<Function *, 16> todo = {step};
  while (!todo.empty()) {
    Function *F = todo.pop_back_val();
    if (!closure.insert(F).second)
      continue;
    scanFunction(*F, inStep, &todo);
  }

  GlobalAccesses elsewhere;
  for (Function &F : M)
    if (!F.empty() && !closure.count(&F) && !isCheckpointLoop(&F))
      scanFunction(F, elsewhere, nullptr);

  SmallPtrSet<GlobalVariable *, 8> shadows;
  for (GlobalVariable &GV : M.globals())
    if (auto *MD = GV.getMetadata("enzyme_shadow"))
      for (auto &op : MD->operands())
        if (auto *CAM = dyn_cast_or_null<ConstantAsMetadata>(op))
          if (auto *S = dyn_cast<GlobalVariable>(
                  CAM->getValue()->stripPointerCasts()))
            shadows.insert(S);

  SmallVector<GlobalVariable *, 8> result;
  for (GlobalVariable &GV : M.globals()) {
    if (GV.isConstant() || shadows.count(&GV) ||
        GV.getMetadata("enzyme_internalshadowglobal") ||
        hasMetadata(&GV, "enzyme_inactive") ||
        GV.getName().starts_with("enzyme_") ||
        GV.getName().starts_with("__enzyme") || !GV.getValueType()->isSized())
      continue;
    if (inStep.written.count(&GV) ||
        (inStep.read.count(&GV) && elsewhere.written.count(&GV)))
      result.push_back(&GV);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// What a snapshot of an object holds: access paths
//===----------------------------------------------------------------------===//
//
// For a step argument that points to an object graph (a Julia closure, a C
// struct of pointers), the accesses of the step, as paths from the argument:
// the byte offsets of the pointer fields followed to reach an object, then the
// byte offset of the access in it, or "the whole object" where the offset is
// not a constant. A scheme that copies the state itself can copy just what
// these reach (see set_paths in enzyme/checkpoint.h).

namespace {
struct AccessPath {
  SmallVector<int64_t, 4> path;
  // Offset of the access in the object reached, or -1 for the whole object.
  int64_t offset;
  bool read = false, write = false;
};

struct PathAnalysis {
  static constexpr unsigned MaxDepth = 8;
  std::map<std::pair<SmallVector<int64_t, 4>, int64_t>, AccessPath> accesses;
  std::set<std::tuple<const Function *, unsigned, SmallVector<int64_t, 4>>>
      seenArgs;

  void note(ArrayRef<int64_t> path, int64_t offset, bool read, bool write) {
    SmallVector<int64_t, 4> P(path.begin(), path.end());
    auto &A = accesses[{P, offset}];
    A.path = P;
    A.offset = offset;
    A.read |= read;
    A.write |= write;
  }
  void whole(ArrayRef<int64_t> path, bool read, bool write) {
    note(path, -1, read, write);
  }

  /// Follow the uses of `V`, which points `offset` bytes (-1: unknown) into
  /// the object at `path`.
  void visit(Value *V, SmallVector<int64_t, 4> path, int64_t offset) {
    SmallVector<std::pair<Value *, int64_t>, 8> todo = {{V, offset}};
    SmallPtrSet<Value *, 16> seen;
    while (!todo.empty()) {
      auto [cur, off] = todo.pop_back_val();
      if (!seen.insert(cur).second)
        continue;
      for (User *U : cur->users()) {
        auto *I = dyn_cast<Instruction>(U);
        if (!I) {
          whole(path, true, true);
          continue;
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
          const DataLayout &DL = GEP->getModule()->getDataLayout();
          APInt c(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
          int64_t next = -1;
          if (off >= 0 && GEP->accumulateConstantOffset(DL, c))
            next = off + c.getSExtValue();
          todo.push_back({GEP, next});
        } else if (isa<CastInst>(I) && I->getType()->isPointerTy()) {
          todo.push_back({I, off});
        } else if (isa<PHINode>(I) || isa<SelectInst>(I)) {
          todo.push_back({I, off});
        } else if (auto *LI = dyn_cast<LoadInst>(I)) {
          if (off < 0) {
            whole(path, true, false);
          } else if (LI->getType()->isPointerTy() && path.size() < MaxDepth) {
            // A pointer field: follow it to the object it points to.
            note(path, off, true, false);
            auto next = path;
            next.push_back(off);
            visit(LI, next, 0);
          } else {
            note(path, off, true, false);
          }
        } else if (auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->getValueOperand() == cur) {
            // The address escapes into memory.
            whole(path, true, true);
          } else if (off < 0 || SI->getValueOperand()->getType()->isPointerTy()) {
            // A pointer field that is reassigned: the object it pointed to is
            // no longer reached through it.
            whole(path, true, true);
          } else {
            note(path, off, false, true);
          }
        } else if (auto *AI = dyn_cast<AtomicRMWInst>(I)) {
          (void)AI;
          if (off < 0)
            whole(path, true, true);
          else
            note(path, off, true, true);
        } else if (auto *MI = dyn_cast<MemIntrinsic>(I)) {
          if (MI->getRawDest() == cur)
            whole(path, false, true);
          else
            whole(path, true, false);
        } else if (auto *CB = dyn_cast<CallBase>(I)) {
          visitCall(CB, cur, path, off);
        } else if (isa<ICmpInst>(I)) {
          continue;
        } else {
          whole(path, true, true);
        }
      }
    }
  }

  void visitCall(CallBase *CB, Value *cur, ArrayRef<int64_t> path,
                 int64_t off) {
    SmallVector<int64_t, 4> P(path.begin(), path.end());
    Function *F = getFunctionFromCall(CB);
    StringRef name = F ? F->getName() : "";
    // Julia's GC bookkeeping neither reads nor writes the object's data.
    if (name == "julia.write_barrier" || name == "julia.write_barrier_binding" ||
        name == "julia.gc_preserve_begin" || name == "julia.gc_preserve_end" ||
        name.starts_with("llvm.lifetime") || name.starts_with("llvm.assume"))
      return;
    // A pointer into the object, derived from its base.
    if (name == "julia.gc_loaded") {
      if (CB->getArgOperand(1) == cur)
        visit(CB, P, off);
      return;
    }
    if (isa<IntrinsicInst>(CB) && !isa<MemIntrinsic>(CB)) {
      whole(P, true, true);
      return;
    }
    if (!F || F->empty() || CB->getCalledOperand() == cur) {
      whole(P, !CB->onlyWritesMemory(), !CB->onlyReadsMemory());
      return;
    }
    for (unsigned i = 0; i < CB->arg_size(); i++) {
      if (CB->getArgOperand(i) != cur)
        continue;
      if (off != 0) {
        // A pointer into the middle of the object.
        whole(P, true, true);
        continue;
      }
      if (seenArgs.insert({F, i, P}).second)
        visit(F->getArg(i), P, 0);
    }
  }
};
} // namespace

/// The accesses of `step` through its argument `argNo`, and whether each is
/// kept: writes, and reads too when `reads`.
static SmallVector<AccessPath, 8> getAccessPaths(Function *step, unsigned argNo,
                                                 bool reads) {
  PathAnalysis PA;
  PA.visit(step->getArg(argNo), {}, 0);
  SmallVector<AccessPath, 8> result;
  for (auto &[key, A] : PA.accesses)
    if (A.write || (reads && A.read))
      result.push_back(A);
  return result;
}

/// The encoding set_paths takes: for each path, its length n, its n offsets,
/// the offset of the access (-1: the whole object), and flags (1 read, 2
/// write).
static SmallVector<int64_t, 32> encodePaths(ArrayRef<AccessPath> paths) {
  SmallVector<int64_t, 32> out;
  for (auto &A : paths) {
    out.push_back(A.path.size());
    out.append(A.path.begin(), A.path.end());
    out.push_back(A.offset);
    out.push_back((A.read ? 1 : 0) | (A.write ? 2 : 0));
  }
  return out;
}

//===----------------------------------------------------------------------===//
// The driver
//===----------------------------------------------------------------------===//

namespace {
struct DriverTypes {
  LLVMContext &Ctx;
  Type *Void;
  IntegerType *I32, *I64;
  PointerType *I8P;
  StructType *Action, *Region, *VTable, *Handle;
  FunctionType *InitFT, *NextFT, *StoreFT, *FinalizeFT, *StateFT, *StepFT,
      *FwdFT, *RevDriverFT, *PathsFT, *WhileFT, *NStepsFT, *CallbackFT;

  DriverTypes(LLVMContext &Ctx) : Ctx(Ctx) {
    Void = Type::getVoidTy(Ctx);
    I32 = Type::getInt32Ty(Ctx);
    I64 = Type::getInt64Ty(Ctx);
    I8P = getInt8PtrTy(Ctx);
    Action = StructType::get(Ctx, {I32, I64, I64, I64});
    Region = StructType::get(Ctx, {I8P, I64, I32, I32, I8P, I8P});
    VTable = StructType::get(
        Ctx, {I32, I8P, I8P, I8P, I8P, I8P, I8P, I8P, I8P, I8P});
    Handle = StructType::get(Ctx, {I8P, I8P, I8P, I64, I64, I64, I32});
    InitFT = FunctionType::get(I8P, {I8P, I64, I64}, false);
    NextFT = FunctionType::get(Void, {I8P, getUnqual(Action)}, false);
    StoreFT = FunctionType::get(Void, {I8P, I64, I64, I8P, I64}, false);
    FinalizeFT = FunctionType::get(Void, {I8P}, false);
    StateFT = FunctionType::get(Void, {I8P, I64, I64, I8P}, false);
    // The primal of step i, or its derivative.
    StepFT = FunctionType::get(Void, {I8P, I64}, false);
    // Step i of a while loop: whether to go on.
    WhileFT = FunctionType::get(I32, {I8P, I64}, false);
    NStepsFT = FunctionType::get(Void, {I8P, I64}, false);
    // vt, data, start, n, regions, nregions, bytes, env, primal, paths,
    // npaths, primal_while (null for a for loop)
    FwdFT = FunctionType::get(
        I8P, {I8P, I8P, I64, I64, I8P, I64, I64, I8P, I8P, I8P, I64, I8P},
        false);
    PathsFT = FunctionType::get(Void, {I8P, I8P, I64}, false);
    // handle, regions, nregions, env, primal, turn
    RevDriverFT =
        FunctionType::get(Void, {I8P, I8P, I64, I8P, I8P, I8P}, false);
    CallbackFT = FunctionType::get(Void, {I8P, I8P, I8P}, false);
  }
};

/// Builds the control flow of the driver.
struct DriverBuilder {
  DriverTypes &T;
  Function *F;
  IRBuilder<> B;
  FunctionCallee Malloc, Free;

  DriverBuilder(DriverTypes &T, Function *F)
      : T(T), F(F), B(BasicBlock::Create(T.Ctx, "entry", F)) {
    Module &M = *F->getParent();
    Malloc = M.getOrInsertFunction("malloc", T.I8P, T.I64);
    Free = M.getOrInsertFunction("free", T.Void, T.I8P);
  }

  BasicBlock *block(const Twine &name) {
    return BasicBlock::Create(T.Ctx, name, F);
  }

  Value *field(StructType *ST, Value *ptr, unsigned idx) {
    ptr = B.CreatePointerCast(ptr, getUnqual(ST));
    return B.CreateStructGEP(ST, ptr, idx);
  }
  Value *load(StructType *ST, Value *ptr, unsigned idx, const Twine &name) {
    return B.CreateLoad(ST->getElementType(idx), field(ST, ptr, idx), name);
  }
  void store(StructType *ST, Value *ptr, unsigned idx, Value *V) {
    B.CreateStore(V, field(ST, ptr, idx));
  }
  Value *vtFn(Value *vt, VTableField idx, const Twine &name) {
    return load(T.VTable, vt, idx, name);
  }
  void callFn(Value *fp, FunctionType *FT, ArrayRef<Value *> args) {
    B.CreateCall(FT, B.CreatePointerCast(fp, getUnqual(FT)), args);
  }
  /// Call `fp` if it is not null, then continue after.
  void callIfSet(Value *fp, FunctionType *FT, ArrayRef<Value *> args,
                 Value *extraCond = nullptr) {
    Value *cond = B.CreateICmpNE(fp, ConstantPointerNull::get(T.I8P));
    if (extraCond)
      cond = B.CreateAnd(cond, extraCond);
    auto *callBB = block("call");
    auto *after = block("after");
    B.CreateCondBr(cond, callBB, after);
    B.SetInsertPoint(callBB);
    callFn(fp, FT, args);
    B.CreateBr(after);
    B.SetInsertPoint(after);
  }
  /// Run the primal of steps [from, to).
  void forwardSteps(Value *primal, Value *env, Value *start, Value *from,
                    Value *to) {
    auto *pre = B.GetInsertBlock();
    auto *body = block("fwd.body");
    auto *after = block("fwd.after");
    B.CreateCondBr(B.CreateICmpSLT(from, to), body, after);
    B.SetInsertPoint(body);
    auto *j = B.CreatePHI(T.I64, 2, "j");
    j->addIncoming(from, pre);
    callFn(primal, T.StepFT, {env, B.CreateAdd(start, j)});
    auto *next = B.CreateAdd(j, ConstantInt::get(T.I64, 1));
    j->addIncoming(next, B.GetInsertBlock());
    B.CreateCondBr(B.CreateICmpSLT(next, to), body, after);
    B.SetInsertPoint(after);
  }
  /// Save or restore the state for an action.
  void snapshot(bool save, Value *vt, Value *state, Value *slot, Value *step,
                Value *regions, Value *nregions, Value *env) {
    callIfSet(vtFn(vt, save ? VT_SaveState : VT_LoadState, "state_fn"),
              T.StateFT, {state, slot, step, env});
    callIfSet(vtFn(vt, save ? VT_Store : VT_Restore, "store_fn"), T.StoreFT,
              {state, slot, step, regions, nregions},
              B.CreateICmpNE(nregions, ConstantInt::get(T.I64, 0)));
  }
  /// Call callback `idx` of each callback region.
  void regionCallbacks(Value *regions, Value *nregions, CallbacksField idx) {
    auto *pre = B.GetInsertBlock();
    auto *body = block("cb.body");
    auto *call = block("cb.call");
    auto *next = block("cb.next");
    auto *after = block("cb.after");
    B.CreateCondBr(B.CreateICmpSGT(nregions, ConstantInt::get(T.I64, 0)), body,
                   after);
    B.SetInsertPoint(body);
    auto *r = B.CreatePHI(T.I64, 2, "r");
    r->addIncoming(ConstantInt::get(T.I64, 0), pre);
    Value *region = B.CreateGEP(
        T.Region, B.CreatePointerCast(regions, getUnqual(T.Region)), r);
    Value *flags = B.CreateLoad(T.I32, B.CreateStructGEP(T.Region, region, 3));
    B.CreateCondBr(B.CreateICmpNE(B.CreateAnd(flags, RegionCallbackFlag),
                                  ConstantInt::get(T.I32, 0)),
                   call, next);
    B.SetInsertPoint(call);
    Value *ptr = B.CreateLoad(T.I8P, B.CreateStructGEP(T.Region, region, 0));
    Value *shadow = B.CreateLoad(T.I8P, B.CreateStructGEP(T.Region, region, 4));
    Value *cb = B.CreateLoad(T.I8P, B.CreateStructGEP(T.Region, region, 5));
    Value *fp = B.CreateLoad(
        T.I8P, B.CreateConstInBoundsGEP1_64(T.I8P, cb, (unsigned)idx));
    callFn(fp, T.CallbackFT, {cb, ptr, shadow});
    B.CreateBr(next);
    B.SetInsertPoint(next);
    auto *r1 = B.CreateAdd(r, ConstantInt::get(T.I64, 1));
    r->addIncoming(r1, next);
    B.CreateCondBr(B.CreateICmpSLT(r1, nregions), body, after);
    B.SetInsertPoint(after);
  }
  void trap() {
    B.CreateCall(getIntrinsicDeclaration(F->getParent(), Intrinsic::trap), {});
    B.CreateUnreachable();
  }
  Value *slot(int64_t s) { return ConstantInt::getSigned(T.I64, s); }
};
} // namespace

/// The forward sweep: the schedule up to its first turn. The state before the
/// last step goes to slot -2, and the last step runs without taping: the
/// reverse sweep starts by restoring it and running its derivative.
static Function *getOrCreateFwdDriver(Module &M, DriverTypes &T) {
  if (auto *F = M.getFunction("__enzyme_ckpt_fwd"))
    return F;
  auto *F = Function::Create(T.FwdFT, GlobalValue::InternalLinkage,
                             "__enzyme_ckpt_fwd", &M);
  F->addFnAttr(Attribute::NoInline);
  DriverBuilder D(T, F);
  auto &B = D.B;
  auto *A = F->arg_begin();
  Value *vt = A++, *data = A++, *start = A++, *n = A++, *regions = A++,
        *nregions = A++, *bytes = A++, *env = A++, *primal = A++,
        *paths = A++, *npaths = A++, *primalWhile = A++;
  const DataLayout &DL = M.getDataLayout();

  auto *action = B.CreateAlloca(T.Action, nullptr, "action");
  Value *h = B.CreateCall(
      D.Malloc, {ConstantInt::get(T.I64, DL.getTypeAllocSize(T.Handle))}, "h");
  D.store(T.Handle, h, H_VT, vt);
  D.store(T.Handle, h, H_Data, data);
  D.store(T.Handle, h, H_Start, start);
  D.store(T.Handle, h, H_N, n);
  D.store(T.Handle, h, H_LastJ, ConstantInt::get(T.I64, 0));
  D.store(T.Handle, h, H_Empty, ConstantInt::get(T.I32, 0));
  Value *state = B.CreateCall(
      T.InitFT,
      B.CreatePointerCast(D.vtFn(vt, VT_Init, "init"), getUnqual(T.InitFT)),
      {data, n, bytes}, "state");
  D.store(T.Handle, h, H_State, state);
  D.callIfSet(D.vtFn(vt, VT_SetPaths, "set_paths"), T.PathsFT,
              {state, paths, npaths},
              B.CreateICmpNE(npaths, ConstantInt::get(T.I64, 0)));
  D.regionCallbacks(regions, nregions, CB_Enter);

  auto *loop = D.block("loop");
  B.CreateBr(loop);
  B.SetInsertPoint(loop);
  D.callFn(D.vtFn(vt, VT_NextAction, "next"), T.NextFT, {state, action});
  Value *flag = D.load(T.Action, action, 0, "flag");
  Value *it = D.load(T.Action, action, 1, "iteration");
  Value *sit = D.load(T.Action, action, 2, "startiteration");
  Value *cp = D.load(T.Action, action, 3, "cpnum");

  auto *bad = D.block("bad");
  auto *storeBB = D.block("store");
  auto *restoreBB = D.block("restore");
  auto *fwdBB = D.block("forward");
  auto *turnBB = D.block("firstuturn");
  auto *doneBB = D.block("done");
  auto *SW = B.CreateSwitch(flag, bad, 5);
  SW->addCase(ConstantInt::get(T.I32, CKPT_STORE), storeBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_RESTORE), restoreBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_FORWARD), fwdBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_FIRSTUTURN), turnBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_DONE), doneBB);

  B.SetInsertPoint(bad);
  D.trap();

  B.SetInsertPoint(storeBB);
  D.snapshot(true, vt, state, cp, it, regions, nregions, env);
  B.CreateBr(loop);

  // Only a while loop's schedule goes back before its first turn, to reach
  // the state before the last step once the loop has ended.
  B.SetInsertPoint(restoreBB);
  D.snapshot(false, vt, state, cp, it, regions, nregions, env);
  B.CreateBr(loop);

  B.SetInsertPoint(fwdBB);
  auto *forLoop = D.block("forward.for");
  auto *whileLoop = D.block("forward.while");
  B.CreateCondBr(B.CreateICmpEQ(primalWhile, ConstantPointerNull::get(T.I8P)),
                 forLoop, whileLoop);
  B.SetInsertPoint(forLoop);
  D.forwardSteps(primal, env, start, sit, it);
  B.CreateBr(loop);

  // A while loop may end on the way: the scheme learns how many steps it had,
  // and plans the rest of the schedule.
  B.SetInsertPoint(whileLoop);
  {
    auto *body = D.block("while.body");
    auto *next = D.block("while.next");
    auto *ended = D.block("while.ended");
    B.CreateCondBr(B.CreateICmpSLT(sit, it), body, loop);
    B.SetInsertPoint(body);
    auto *j = B.CreatePHI(T.I64, 2, "j");
    j->addIncoming(sit, whileLoop);
    Value *go = B.CreateCall(
        T.WhileFT, B.CreatePointerCast(primalWhile, getUnqual(T.WhileFT)),
        {env, B.CreateAdd(start, j)}, "go");
    Value *j1 = B.CreateAdd(j, ConstantInt::get(T.I64, 1));
    B.CreateCondBr(B.CreateICmpEQ(go, ConstantInt::get(T.I32, 0)), ended,
                   next);
    B.SetInsertPoint(next);
    j->addIncoming(j1, next);
    B.CreateCondBr(B.CreateICmpSLT(j1, it), body, loop);
    B.SetInsertPoint(ended);
    D.store(T.Handle, h, H_N, j1);
    D.callIfSet(D.vtFn(vt, VT_SetNSteps, "set_nsteps"), T.NStepsFT,
                {state, j1});
    B.CreateBr(loop);
  }

  B.SetInsertPoint(turnBB);
  Value *lastj = B.CreateSub(it, ConstantInt::get(T.I64, 1));
  D.store(T.Handle, h, H_LastJ, lastj);
  D.snapshot(true, vt, state, D.slot(LastSlot), lastj, regions, nregions, env);
  D.callFn(primal, T.StepFT, {env, B.CreateAdd(start, lastj)});
  // The steps ran without their derivatives: the shadow's references are
  // made to follow the primal's for what comes after the loop.
  D.regionCallbacks(regions, nregions, CB_Sync);
  B.CreateRet(h);

  // Nothing to reverse (n == 0).
  B.SetInsertPoint(doneBB);
  D.store(T.Handle, h, H_Empty, ConstantInt::get(T.I32, 1));
  B.CreateRet(h);
  return F;
}

/// The reverse sweep. It keeps the state it starts from in slot -1, and puts
/// it back once the schedule is done: the primal state is left as the forward
/// pass left it.
static Function *getOrCreateRevDriver(Module &M, DriverTypes &T) {
  if (auto *F = M.getFunction("__enzyme_ckpt_rev"))
    return F;
  auto *F = Function::Create(T.RevDriverFT, GlobalValue::InternalLinkage,
                             "__enzyme_ckpt_rev", &M);
  F->addFnAttr(Attribute::NoInline);
  DriverBuilder D(T, F);
  auto &B = D.B;
  auto *A = F->arg_begin();
  Value *h = A++, *regions = A++, *nregions = A++, *env = A++, *primal = A++,
        *turn = A++;

  auto *action = B.CreateAlloca(T.Action, nullptr, "action");
  Value *vt = D.load(T.Handle, h, H_VT, "vt");
  Value *state = D.load(T.Handle, h, H_State, "state");
  Value *start = D.load(T.Handle, h, H_Start, "start");
  Value *n = D.load(T.Handle, h, H_N, "n");
  Value *lastj = D.load(T.Handle, h, H_LastJ, "lastj");
  Value *empty = D.load(T.Handle, h, H_Empty, "empty");
  Value *isEmpty = B.CreateICmpNE(empty, ConstantInt::get(T.I32, 0));

  auto *first = D.block("firstuturn");
  auto *loop = D.block("loop");
  auto *finish = D.block("finish");
  auto *free = D.block("free");
  B.CreateCondBr(isEmpty, free, first);

  B.SetInsertPoint(first);
  D.snapshot(true, vt, state, D.slot(EntrySlot), n, regions, nregions, env);
  D.snapshot(false, vt, state, D.slot(LastSlot), lastj, regions, nregions, env);
  D.regionCallbacks(regions, nregions, CB_Sync);
  D.callFn(turn, T.StepFT, {env, B.CreateAdd(start, lastj)});
  B.CreateBr(loop);

  B.SetInsertPoint(loop);
  D.callFn(D.vtFn(vt, VT_NextAction, "next"), T.NextFT, {state, action});
  Value *flag = D.load(T.Action, action, 0, "flag");
  Value *it = D.load(T.Action, action, 1, "iteration");
  Value *sit = D.load(T.Action, action, 2, "startiteration");
  Value *cp = D.load(T.Action, action, 3, "cpnum");

  auto *bad = D.block("bad");
  auto *storeBB = D.block("store");
  auto *restoreBB = D.block("restore");
  auto *fwdBB = D.block("forward");
  auto *turnBB = D.block("uturn");
  auto *SW = B.CreateSwitch(flag, bad, 5);
  SW->addCase(ConstantInt::get(T.I32, CKPT_STORE), storeBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_RESTORE), restoreBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_FORWARD), fwdBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_UTURN), turnBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_DONE), finish);

  B.SetInsertPoint(bad);
  D.trap();

  B.SetInsertPoint(storeBB);
  D.snapshot(true, vt, state, cp, it, regions, nregions, env);
  B.CreateBr(loop);

  B.SetInsertPoint(restoreBB);
  D.snapshot(false, vt, state, cp, it, regions, nregions, env);
  B.CreateBr(loop);

  B.SetInsertPoint(fwdBB);
  D.forwardSteps(primal, env, start, sit, it);
  B.CreateBr(loop);

  B.SetInsertPoint(turnBB);
  Value *i = B.CreateAdd(start, B.CreateSub(it, ConstantInt::get(T.I64, 1)));
  D.regionCallbacks(regions, nregions, CB_Sync);
  D.callFn(turn, T.StepFT, {env, i});
  B.CreateBr(loop);

  B.SetInsertPoint(finish);
  D.snapshot(false, vt, state, D.slot(EntrySlot), n, regions, nregions, env);
  D.regionCallbacks(regions, nregions, CB_Sync);
  B.CreateBr(free);

  B.SetInsertPoint(free);
  D.regionCallbacks(regions, nregions, CB_Leave);
  D.callIfSet(D.vtFn(vt, VT_Finalize, "finalize"), T.FinalizeFT, {state});
  B.CreateCall(D.Free, {B.CreatePointerCast(h, T.I8P)});
  B.CreateRetVoid();
  return F;
}

//===----------------------------------------------------------------------===//
// The step's derivatives and the trampolines
//===----------------------------------------------------------------------===//

namespace {
struct StepInfo {
  Function *loop;
  Function *step;
  unsigned firstArg;
  /// Activity of each step parameter, the index first.
  std::vector<DIFFE_TYPE> stepActivity;
  FnTypeInfo stepTypeInfo;
  /// The environment: the loop's step arguments, each followed by its shadow
  /// (its `width` shadows) if it has one.
  StructType *env;
  std::string suffix;
  unsigned width = 1;

  StepInfo(Function *loop)
      : loop(loop), step(getStep(loop)), firstArg(getFirstStepArg(loop)),
        stepTypeInfo(step), env(nullptr) {}
};
} // namespace

/// The type of the shadows of a value of type `T` in vector mode of `width`.
static Type *getShadowType(Type *T, unsigned width) {
  return width == 1 ? T : ArrayType::get(T, width);
}

static bool getStepInfo(StepInfo &S, ArrayRef<DIFFE_TYPE> constant_args,
                        const FnTypeInfo &typeInfo, unsigned width,
                        RequestContext &context) {
  // In vector mode the shadows come in `width`s, and the step's derivative
  // takes them so: the schedule and the snapshots, of primal state only, are
  // the same for all of them.
  LLVMContext &Ctx = S.loop->getContext();
  S.width = width;
  S.stepActivity.push_back(DIFFE_TYPE::CONSTANT);
  SmallVector<Type *, 8> envTys;
  S.suffix = width == 1 ? "" : ("w" + Twine(width) + ".").str();
  for (unsigned k = S.firstArg; k < S.loop->arg_size(); k++) {
    DIFFE_TYPE act = constant_args[k];
    Type *T = S.loop->getArg(k)->getType();
    envTys.push_back(T);
    switch (act) {
    case DIFFE_TYPE::CONSTANT:
      S.suffix += "c";
      break;
    case DIFFE_TYPE::DUP_ARG:
    case DIFFE_TYPE::DUP_NONEED:
      // The step reads its state, so it needs the primal too.
      act = DIFFE_TYPE::DUP_ARG;
      envTys.push_back(getShadowType(T, width));
      S.suffix += "d";
      break;
    case DIFFE_TYPE::OUT_DIFF:
      EmitNoDerivativeError(
          "active arguments passed by value to a checkpointed loop are not "
          "supported; pass them by reference",
          S.loop, context);
      return false;
    }
    S.stepActivity.push_back(act);
  }
  S.env = StructType::get(Ctx, envTys);

  unsigned p = 0;
  for (auto &a : S.step->args()) {
    TypeTree dt;
    if (p == 0) {
      dt = TypeTree(BaseType::Integer).Only(-1, nullptr);
    } else {
      auto found = typeInfo.Arguments.find(S.loop->getArg(S.firstArg + p - 1));
      if (found != typeInfo.Arguments.end())
        dt = found->second;
      else if (a.getType()->isFPOrFPVectorTy())
        dt = TypeTree(ConcreteType(a.getType()->getScalarType()))
                 .Only(-1, nullptr);
      else if (a.getType()->isIntOrIntVectorTy())
        dt = TypeTree(BaseType::Integer).Only(-1, nullptr);
      else if (a.getType()->isPointerTy())
        dt = TypeTree(BaseType::Pointer).Only(-1, nullptr);
    }
    S.stepTypeInfo.Arguments.insert(std::make_pair(&a, dt));
    S.stepTypeInfo.KnownValues.insert(std::make_pair(&a, std::set<int64_t>()));
    p++;
  }
  return true;
}

/// The derivative of one step, its forward and reverse passes combined: a
/// step is reversed right after it is rerun, so it needs no tape.
static Function *getStepGradient(EnzymeLogic &Logic, RequestContext context,
                                 StepInfo &S, TypeAnalysis &TA,
                                 bool runtimeActivity, bool strongZero,
                                 bool AtomicAdd) {
  std::vector<bool> overwritten(S.step->arg_size(), false);
  return Logic.CreatePrimalAndGradient(
      context,
      (ReverseCacheKey){.todiff = S.step,
                        .retType = DIFFE_TYPE::CONSTANT,
                        .constant_args = S.stepActivity,
                        .subsequent_calls_may_write = false,
                        .overwritten_args = overwritten,
                        .returnUsed = false,
                        .shadowReturnUsed = false,
                        .mode = DerivativeMode::ReverseModeCombined,
                        .width = S.width,
                        .freeMemory = true,
                        .AtomicAdd = AtomicAdd,
                        .additionalType = nullptr,
                        .forceAnonymousTape = false,
                        .typeInfo = S.stepTypeInfo,
                        .runtimeActivity = runtimeActivity,
                        .strongZero = strongZero},
      TA, /*augmented*/ nullptr);
}

/// Load the step's arguments (and shadows, if `shadows`) from the env.
static void loadStepArgs(IRBuilder<> &B, StepInfo &S, Value *env, Value *i,
                         bool shadows, SmallVectorImpl<Value *> &args) {
  env = B.CreatePointerCast(env, getUnqual(S.env));
  args.push_back(
      B.CreateSExtOrTrunc(i, S.step->getFunctionType()->getParamType(0)));
  unsigned field = 0;
  for (unsigned p = 1; p < S.stepActivity.size(); p++) {
    Type *T = S.env->getElementType(field);
    args.push_back(B.CreateLoad(T, B.CreateStructGEP(S.env, env, field)));
    field++;
    if (S.stepActivity[p] == DIFFE_TYPE::DUP_ARG) {
      // In vector mode, the array of the shadows.
      if (shadows)
        args.push_back(B.CreateLoad(S.env->getElementType(field),
                                    B.CreateStructGEP(S.env, env, field)));
      field++;
    }
  }
}

/// `void (env, i)`: step i, run by `callee` with the step's arguments (and
/// shadows) from the env.
static Function *getTrampoline(DriverTypes &T, StepInfo &S, StringRef kind,
                               Function *callee, bool shadows) {
  Module &M = *S.loop->getParent();
  std::string name = ("enzyme.ckpt." + kind + "." + S.loop->getName() + "." +
                      S.suffix)
                         .str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = Function::Create(T.StepFT, GlobalValue::InternalLinkage, name, &M);
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), shadows, args);
  B.CreateCall(callee, args)->setCallingConv(callee->getCallingConv());
  B.CreateRetVoid();
  return F;
}

/// The augmented forward pass of the step, for turns in split mode: the
/// step is taken as in a loop whose later iterations may overwrite what it
/// reads, so it caches rather than recomputes, as when the loop is not
/// checkpointed.
static const AugmentedReturn *
getStepAugmented(EnzymeLogic &Logic, RequestContext context, StepInfo &S,
                 TypeAnalysis &TA, bool runtimeActivity, bool strongZero,
                 bool AtomicAdd) {
  std::vector<bool> overwritten(S.step->arg_size(), true);
  std::vector<bool> nowrite(S.step->arg_size(), false);
  return &Logic.CreateAugmentedPrimal(
      context, S.step, DIFFE_TYPE::CONSTANT, S.stepActivity, TA,
      /*returnUsed*/ false, /*shadowReturnUsed*/ false, S.stepTypeInfo,
      /*subsequent_calls_may_write*/ true, overwritten, nowrite,
      /*forceAnonymousTape*/ false, runtimeActivity, strongZero, S.width,
      AtomicAdd);
}

/// The tape type of `aug`, or null if it has none.
static Type *getTapeType(const AugmentedReturn &aug) {
  auto found = aug.returns.find(AugmentedStruct::Tape);
  if (found == aug.returns.end())
    return nullptr;
  Type *RT = aug.fn->getReturnType();
  return found->second == -1
             ? RT
             : cast<StructType>(RT)->getElementType(found->second);
}

/// The reverse pass of the step, from the tape of `aug`.
static Function *getStepReverse(EnzymeLogic &Logic, RequestContext context,
                                StepInfo &S, TypeAnalysis &TA,
                                const AugmentedReturn &aug,
                                bool runtimeActivity, bool strongZero,
                                bool AtomicAdd) {
  std::vector<bool> overwritten(S.step->arg_size(), true);
  return Logic.CreatePrimalAndGradient(
      context,
      (ReverseCacheKey){.todiff = S.step,
                        .retType = DIFFE_TYPE::CONSTANT,
                        .constant_args = S.stepActivity,
                        .subsequent_calls_may_write = true,
                        .overwritten_args = overwritten,
                        .returnUsed = false,
                        .shadowReturnUsed = false,
                        .mode = DerivativeMode::ReverseModeGradient,
                        .width = S.width,
                        .freeMemory = true,
                        .AtomicAdd = AtomicAdd,
                        .additionalType = getTapeType(aug),
                        .forceAnonymousTape = false,
                        .typeInfo = S.stepTypeInfo,
                        .runtimeActivity = runtimeActivity,
                        .strongZero = strongZero},
      TA, &aug);
}

/// `void (env, i)`: a turn in split mode, the augmented forward pass of step
/// i and right after it its reverse pass. The tape does not leave the
/// trampoline.
static Function *getSplitTurnTrampoline(DriverTypes &T, StepInfo &S,
                                        const AugmentedReturn &aug,
                                        Function *rev) {
  Module &M = *S.loop->getParent();
  std::string name =
      ("enzyme.ckpt.splitturn." + S.loop->getName() + "." + S.suffix).str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = Function::Create(T.StepFT, GlobalValue::InternalLinkage, name, &M);
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));
  // In a Julia module the tape may hold Julia objects, which the frame built
  // from the task's GC stack roots across the reverse pass.
  if (auto *GCStack = M.getFunction("julia.get_pgcstack"))
    B.CreateCall(GCStack->getFunctionType(), GCStack, {});
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), /*shadows*/ true, args);
  auto *call = B.CreateCall(aug.fn, args);
  call->setCallingConv(aug.fn->getCallingConv());
  auto found = aug.returns.find(AugmentedStruct::Tape);
  if (found != aug.returns.end())
    args.push_back(found->second == -1
                       ? (Value *)call
                       : B.CreateExtractValue(call, (unsigned)found->second));
  B.CreateCall(rev, args)->setCallingConv(rev->getCallingConv());
  B.CreateRetVoid();
  return F;
}

/// `i32 (env, i)`: step i of a while loop, returning whether to go on.
static Function *getWhileTrampoline(DriverTypes &T, StepInfo &S) {
  Module &M = *S.loop->getParent();
  std::string name =
      ("enzyme.ckpt.primal_while." + S.loop->getName() + "." + S.suffix).str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = Function::Create(T.WhileFT, GlobalValue::InternalLinkage, name, &M);
  IRBuilder<> B(BasicBlock::Create(M.getContext(), "entry", F));
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), /*shadows*/ false, args);
  auto *call = B.CreateCall(S.step, args);
  call->setCallingConv(S.step->getCallingConv());
  B.CreateRet(B.CreateZExt(
      B.CreateICmpNE(call, Constant::getNullValue(call->getType())), T.I32));
  return F;
}

//===----------------------------------------------------------------------===//
// The augmented forward and reverse passes of the loop
//===----------------------------------------------------------------------===//

/// The loop's parameter types, each followed by its shadow (or `width` of
/// them) if duplicated.
static SmallVector<Type *, 8>
getInterleavedParams(Function *loop, ArrayRef<DIFFE_TYPE> constant_args,
                     unsigned width) {
  SmallVector<Type *, 8> params;
  for (unsigned k = 0; k < loop->arg_size(); k++) {
    params.push_back(loop->getArg(k)->getType());
    if (constant_args[k] == DIFFE_TYPE::DUP_ARG ||
        constant_args[k] == DIFFE_TYPE::DUP_NONEED)
      params.push_back(getShadowType(loop->getArg(k)->getType(), width));
  }
  return params;
}

namespace {
/// What one pass of a checkpointed loop hands the driver, built from that
/// pass's own arguments.
struct PassFrame {
  Value *env;
  Value *regions;
  Value *nregions;
  Value *bytes;
  SmallVector<Value *, 8> primals;
};
} // namespace

/// Map the arguments of `F` (the loop's arguments, each followed by its
/// shadow if duplicated) to the step's environment and the snapshot regions.
static PassFrame buildFrame(IRBuilder<> &B, Function *F, StepInfo &S,
                            ArrayRef<DIFFE_TYPE> constant_args,
                            DriverTypes &T) {
  Function *loop = S.loop;
  Module &M = *loop->getParent();
  const DataLayout &DL = M.getDataLayout();
  PassFrame frame;

  SmallVector<Value *, 8> shadows;
  {
    auto *A = F->arg_begin();
    for (unsigned k = 0; k < loop->arg_size(); k++) {
      frame.primals.push_back(A++);
      if (constant_args[k] == DIFFE_TYPE::DUP_ARG ||
          constant_args[k] == DIFFE_TYPE::DUP_NONEED)
        shadows.push_back(A++);
      else
        shadows.push_back(nullptr);
    }
  }
  auto &primals = frame.primals;

  // The environment.
  auto *env = B.CreateAlloca(S.env, nullptr, "env");
  {
    unsigned field = 0;
    for (unsigned k = S.firstArg; k < loop->arg_size(); k++) {
      B.CreateStore(primals[k], B.CreateStructGEP(S.env, env, field++));
      if (S.stepActivity[k - S.firstArg + 1] == DIFFE_TYPE::DUP_ARG)
        B.CreateStore(shadows[k], B.CreateStructGEP(S.env, env, field++));
    }
  }
  frame.env = B.CreatePointerCast(env, T.I8P);

  // The regions: those marked at the call, then the globals.
  auto globals = getGlobalRegions(S.step);
  unsigned nmarked = getNumRegions(loop);
  unsigned nregions = nmarked + globals.size();
  auto *regionArr = ArrayType::get(T.Region, std::max(nregions, 1u));
  auto *regions = B.CreateAlloca(regionArr, nullptr, "regions");
  Value *bytes = ConstantInt::get(T.I64, 0);
  // The globals' entries come from a constant table: a global's address
  // stored by an instruction outside the functions that use it trips up
  // activity analysis of those functions.
  if (!globals.empty()) {
    SmallVector<Constant *, 8> entries;
    uint64_t globalBytes = 0;
    for (auto *GV : globals) {
      uint64_t size = DL.getTypeAllocSize(GV->getValueType());
      globalBytes += size;
      entries.push_back(ConstantStruct::get(
          T.Region,
          {ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, T.I8P),
           ConstantInt::get(T.I64, size),
           ConstantInt::get(T.I32, GV->getType()->getPointerAddressSpace()),
           ConstantInt::get(T.I32, 0), ConstantPointerNull::get(T.I8P),
           ConstantPointerNull::get(T.I8P)}));
    }
    auto *tableTy = ArrayType::get(T.Region, entries.size());
    std::string name = ("enzyme.ckpt.regions." + S.step->getName()).str();
    auto *table = M.getGlobalVariable(name, /*AllowInternal*/ true);
    if (!table || table->getValueType() != tableTy)
      table = new GlobalVariable(M, tableTy, /*isConstant*/ true,
                                 GlobalValue::PrivateLinkage,
                                 ConstantArray::get(tableTy, entries), name);
    B.CreateMemCpy(B.CreateConstInBoundsGEP2_32(regionArr, regions, 0, nmarked),
                   MaybeAlign(1), table, MaybeAlign(1),
                   DL.getTypeAllocSize(tableTy));
    bytes = ConstantInt::get(T.I64, globalBytes);
  }
  // The memory space of a marked region is its pointer's, unless the loop
  // says otherwise (a device buffer behind a plain host pointer, from
  // __enzyme_ptr_size_hint).
  SmallVector<unsigned, 4> spaces;
  if (loop->hasFnAttribute(CheckpointRegionSpacesAttr)) {
    SmallVector<StringRef, 4> parts;
    loop->getFnAttribute(CheckpointRegionSpacesAttr)
        .getValueAsString()
        .split(parts, ',');
    for (StringRef part : parts)
      spaces.push_back(std::stoul(part.str()));
  }
  for (unsigned r = 0; r < nmarked; r++) {
    unsigned k = LoopFixedParams + 2 * r;
    Value *ptr = primals[k];
    Value *size = primals[k + 1];
    unsigned AS = r < spaces.size()
                      ? spaces[r]
                      : cast<PointerType>(ptr->getType())->getAddressSpace();
    // A callback region: its "size" is its callbacks.
    bool callback = AS == CallbackRegionSpace;
    Value *callbacks = ConstantPointerNull::get(T.I8P);
    if (callback) {
      callbacks = B.CreateIntToPtr(size, T.I8P);
      size = ConstantInt::get(T.I64, 0);
      AS = 0;
    }
    Value *shadow =
        shadows[k] && !shadows[k]->getType()->isArrayTy()
            ? B.CreatePointerBitCastOrAddrSpaceCast(shadows[k], T.I8P)
            : (Value *)ConstantPointerNull::get(T.I8P);
    Value *slot = B.CreateConstInBoundsGEP2_32(regionArr, regions, 0, r);
    B.CreateStore(B.CreatePointerBitCastOrAddrSpaceCast(ptr, T.I8P),
                  B.CreateStructGEP(T.Region, slot, 0));
    B.CreateStore(size, B.CreateStructGEP(T.Region, slot, 1));
    B.CreateStore(ConstantInt::get(T.I32, AS),
                  B.CreateStructGEP(T.Region, slot, 2));
    B.CreateStore(ConstantInt::get(T.I32, callback ? RegionCallbackFlag : 0),
                  B.CreateStructGEP(T.Region, slot, 3));
    B.CreateStore(shadow, B.CreateStructGEP(T.Region, slot, 4));
    B.CreateStore(callbacks, B.CreateStructGEP(T.Region, slot, 5));
    bytes = B.CreateAdd(bytes, size);
  }
  frame.regions = B.CreatePointerCast(regions, T.I8P);
  frame.nregions = ConstantInt::get(T.I64, nregions);
  frame.bytes = bytes;
  return frame;
}

static void printRegions(StepInfo &S) {
  if (!EnzymePrintCheckpointRegions)
    return;
  const DataLayout &DL = S.loop->getParent()->getDataLayout();
  llvm::errs() << "checkpoint regions of " << S.step->getName() << ":\n";
  for (unsigned r = 0; r < getNumRegions(S.loop); r++)
    llvm::errs() << "  marked region " << r << "\n";
  for (auto *GV : getGlobalRegions(S.step))
    llvm::errs() << "  global " << GV->getName() << " ("
                 << DL.getTypeAllocSize(GV->getValueType()) << " bytes)\n";
}

Function *createCheckpointAugmented(EnzymeLogic &Logic, RequestContext context,
                                    Function *loop,
                                    ArrayRef<DIFFE_TYPE> constant_args,
                                    TypeAnalysis &TA,
                                    const FnTypeInfo &typeInfo,
                                    bool runtimeActivity, bool strongZero,
                                    unsigned width, bool AtomicAdd) {
  Module &M = *loop->getParent();
  LLVMContext &Ctx = M.getContext();
  DriverTypes T(Ctx);

  StepInfo S(loop);
  if (!getStepInfo(S, constant_args, typeInfo, width, context))
    return nullptr;

  auto *FT = FunctionType::get(
      T.I8P, getInterleavedParams(loop, constant_args, width), false);
  auto *F = Function::Create(FT, GlobalValue::InternalLinkage,
                             "augmented_" + loop->getName(), &M);
  F->addFnAttr("enzyme_checkpoint_pass");
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", F));
  PassFrame frame = buildFrame(B, F, S, constant_args, T);
  printRegions(S);

  // The accesses through the first argument after the index, for schemes
  // that copy the state themselves.
  Value *paths = ConstantPointerNull::get(T.I8P);
  uint64_t npaths = 0;
  if (S.step->arg_size() > 1 &&
      S.step->getArg(1)->getType()->isPointerTy()) {
    auto encoded = encodePaths(getAccessPaths(S.step, 1, /*reads*/ true));
    npaths = encoded.size();
    if (npaths) {
      auto *Ty = ArrayType::get(T.I64, npaths);
      std::string name = ("enzyme.ckpt.paths." + S.step->getName()).str();
      auto *G = M.getGlobalVariable(name, /*AllowInternal*/ true);
      if (!G)
        G = new GlobalVariable(M, Ty, /*isConstant*/ true,
                               GlobalValue::PrivateLinkage,
                               ConstantDataArray::get(Ctx, encoded), name);
      paths = B.CreatePointerCast(G, T.I8P);
    }
  }

  auto &primals = frame.primals;
  Value *h = B.CreateCall(
      getOrCreateFwdDriver(M, T),
      {B.CreatePointerCast(primals[2], T.I8P),
       B.CreatePointerCast(primals[3], T.I8P), primals[0], primals[1],
       frame.regions, frame.nregions, frame.bytes, frame.env,
       B.CreatePointerCast(
           getTrampoline(T, S, "primal", S.step, /*shadows*/ false), T.I8P),
       paths, ConstantInt::get(T.I64, npaths),
       isWhileLoop(loop)
           ? B.CreatePointerCast(getWhileTrampoline(T, S), T.I8P)
           : (Value *)ConstantPointerNull::get(T.I8P)},
      "handle");
  B.CreateRet(h);
  return F;
}

Function *createCheckpointGradient(EnzymeLogic &Logic, RequestContext context,
                                   const ReverseCacheKey &key,
                                   TypeAnalysis &TA) {
  Function *loop = key.todiff;
  Module &M = *loop->getParent();
  LLVMContext &Ctx = M.getContext();
  DriverTypes T(Ctx);

  StepInfo S(loop);
  if (!getStepInfo(S, key.constant_args, key.typeInfo, key.width, context))
    return nullptr;
  // What a turn runs: the step's combined derivative, or with
  // -enzyme-checkpoint-split-steps its augmented and reverse passes.
  const AugmentedReturn *aug = nullptr;
  Function *grad = nullptr;
  if (EnzymeCheckpointSplitSteps) {
    aug = getStepAugmented(Logic, context, S, TA, key.runtimeActivity,
                           key.strongZero, key.AtomicAdd);
    grad = getStepReverse(Logic, context, S, TA, *aug, key.runtimeActivity,
                          key.strongZero, key.AtomicAdd);
  } else {
    grad = getStepGradient(Logic, context, S, TA, key.runtimeActivity,
                           key.strongZero, key.AtomicAdd);
  }
  if (!grad)
    return nullptr;

  auto params = getInterleavedParams(loop, key.constant_args, key.width);
  bool combined = key.mode == DerivativeMode::ReverseModeCombined;
  Function *augF = nullptr;
  if (combined) {
    augF = createCheckpointAugmented(Logic, context, loop, key.constant_args,
                                     TA, key.typeInfo, key.runtimeActivity,
                                     key.strongZero, key.width, key.AtomicAdd);
    if (!augF)
      return nullptr;
  } else if (key.additionalType) {
    params.push_back(key.additionalType);
  }
  auto *FT = FunctionType::get(T.Void, params, false);
  auto *F = Function::Create(
      FT, GlobalValue::InternalLinkage,
      (combined ? "diffe" : "diffe_rev_") + loop->getName(), &M);
  F->addFnAttr("enzyme_checkpoint_pass");
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", F));
  Value *h;
  if (combined) {
    SmallVector<Value *, 8> args;
    for (auto &a : F->args())
      args.push_back(&a);
    h = B.CreateCall(augF, args, "handle");
  } else {
    h = B.CreatePointerCast(F->getArg(F->arg_size() - 1), T.I8P);
  }
  PassFrame frame = buildFrame(B, F, S, key.constant_args, T);
  B.CreateCall(
      getOrCreateRevDriver(M, T),
      {h, frame.regions, frame.nregions, frame.env,
       B.CreatePointerCast(
           getTrampoline(T, S, "primal", S.step, /*shadows*/ false), T.I8P),
       B.CreatePointerCast(
           aug ? getSplitTurnTrampoline(T, S, *aug, grad)
               : getTrampoline(T, S, "turn", grad, /*shadows*/ true),
           T.I8P)});
  B.CreateRetVoid();
  return F;
}
