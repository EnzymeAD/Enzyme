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

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "Utils.h"

using namespace llvm;

static cl::opt<bool> EnzymeCheckpointSplitSteps(
    "enzyme-checkpoint-split-steps", cl::init(false), cl::Hidden,
    cl::desc("Differentiate each checkpointed step as its augmented forward "
             "pass followed by its reverse pass, instead of in combined mode"));

static cl::opt<bool> EnzymePrintCheckpointRegions(
    "enzyme-print-checkpoint-regions", cl::init(false), cl::Hidden,
    cl::desc("Print the memory each checkpointed loop snapshots"));

static constexpr const char *CheckpointAttr = "enzyme_checkpoint";
static constexpr const char *CheckpointRegionsAttr =
    "enzyme_checkpoint_nregions";
static constexpr const char *CheckpointStepMD = "enzyme_checkpoint_step";

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
                                    Type *dataTy, bool isWhile) {
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
  // The schedule's own arguments carry no derivative.
  for (unsigned i = 0; i < LoopFixedParams + 2 * regionTypes.size(); i++)
    F->addParamAttr(i, Attribute::get(Ctx, "enzyme_inactive"));

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

bool lowerCheckpointMarkers(Module &M) {
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
  bool changed = false;
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
      *FwdFT, *RevDriverFT, *PathsFT, *WhileFT, *NStepsFT;

  DriverTypes(LLVMContext &Ctx) : Ctx(Ctx) {
    Void = Type::getVoidTy(Ctx);
    I32 = Type::getInt32Ty(Ctx);
    I64 = Type::getInt64Ty(Ctx);
    I8P = getInt8PtrTy(Ctx);
    Action = StructType::get(Ctx, {I32, I64, I64, I64});
    Region = StructType::get(Ctx, {I8P, I64, I32, I32});
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
  D.callFn(turn, T.StepFT, {env, i});
  B.CreateBr(loop);

  B.SetInsertPoint(finish);
  D.snapshot(false, vt, state, D.slot(EntrySlot), n, regions, nregions, env);
  B.CreateBr(free);

  B.SetInsertPoint(free);
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
           ConstantInt::get(T.I32, 0)}));
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
  for (unsigned r = 0; r < nmarked; r++) {
    Value *ptr = primals[LoopFixedParams + 2 * r];
    Value *size = primals[LoopFixedParams + 2 * r + 1];
    unsigned AS = cast<PointerType>(ptr->getType())->getAddressSpace();
    Value *slot = B.CreateConstInBoundsGEP2_32(regionArr, regions, 0, r);
    B.CreateStore(B.CreatePointerBitCastOrAddrSpaceCast(ptr, T.I8P),
                  B.CreateStructGEP(T.Region, slot, 0));
    B.CreateStore(size, B.CreateStructGEP(T.Region, slot, 1));
    B.CreateStore(ConstantInt::get(T.I32, AS),
                  B.CreateStructGEP(T.Region, slot, 2));
    B.CreateStore(ConstantInt::get(T.I32, 0),
                  B.CreateStructGEP(T.Region, slot, 3));
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
