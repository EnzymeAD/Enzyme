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

static cl::opt<bool> EnzymePrintCheckpointRegions(
    "enzyme-print-checkpoint-regions", cl::init(false), cl::Hidden,
    cl::desc("Print the memory each checkpointed loop snapshots"));

static constexpr const char *CheckpointAttr = "enzyme_checkpoint";
static constexpr const char *CheckpointRegionsAttr =
    "enzyme_checkpoint_nregions";
static constexpr const char *CheckpointStepMD = "enzyme_checkpoint_step";

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
};

// Fields of the driver's handle.
enum HandleField : unsigned {
  H_VT = 0,
  H_Data,
  H_State,
  H_Start,
  H_N,
  H_Regions,
  H_NRegions,
  H_Env,
  H_LastTape,
  H_LastJ,
  H_Empty,
};

// Parameters of the loop function before the region pairs.
static constexpr unsigned LoopFixedParams = 4;

bool isCheckpointLoop(const Function *F) {
  return F && F->hasFnAttribute(CheckpointAttr);
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
                                    Type *dataTy) {
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
  auto *F = Function::Create(FT, GlobalValue::InternalLinkage,
                             "enzyme.ckpt.for." + step->getName(), &M);
  F->addFnAttr(CheckpointAttr, "for");
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
  B.CreateCondBr(B.CreateICmpSLT(next, end), body, exit);

  B.SetInsertPoint(exit);
  B.CreateRetVoid();
  return F;
}

static bool lowerMarker(CallInst *CI) {
  Module &M = *CI->getModule();
  LLVMContext &Ctx = M.getContext();
  auto fail = [&](const Twine &msg) {
    std::string str = msg.str();
    EmitFailure("CheckpointMarker", CI->getDebugLoc(), CI, str);
    return false;
  };

  if (CI->arg_size() < 3)
    return fail("__enzyme_checkpoint_for needs a step function, a start and "
                "a number of steps");
  Value *stepV = CI->getArgOperand(0)->stripPointerCasts();
  if (auto *GA = dyn_cast<GlobalAlias>(stepV))
    stepV = GA->getAliaseeObject();
  auto *step = dyn_cast<Function>(stepV);
  if (!step)
    return fail("__enzyme_checkpoint_for needs a known step function");
  auto *stepFT = step->getFunctionType();
  if (stepFT->isVarArg() || stepFT->getNumParams() == 0 ||
      !stepFT->getParamType(0)->isIntegerTy())
    return fail("the step of __enzyme_checkpoint_for must take the step "
                "index as an integer first argument, by value");

  Value *vt = nullptr, *data = nullptr;
  SmallVector<std::pair<Value *, Value *>, 2> regions;
  unsigned idx = 3;
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
    return fail("__enzyme_checkpoint_for needs enzyme_scheme, followed by "
                "the scheme and its data");
  unsigned nargs = CI->arg_size() - idx;
  if (nargs + 1 != stepFT->getNumParams())
    return fail("__enzyme_checkpoint_for passes " + Twine(nargs) +
                " arguments to a step that takes " +
                Twine(stepFT->getNumParams() - 1) + " after the index");

  SmallVector<Value *, 8> args;
  for (unsigned i = 1; i <= 2; i++) {
    Value *V = CI->getArgOperand(i);
    args.push_back(V->getType()->isPointerTy() ? loadInteger(B, V)
                                               : castArg(B, V, I64));
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
      return fail("argument " + Twine(i) +
                  " of __enzyme_checkpoint_for does not match the step");
    args.push_back(a);
  }

  Function *loop =
      createLoopFunction(M, step, regionTypes, vt->getType(), data->getType());
  auto *call = B.CreateCall(loop, args);
  call->setDebugLoc(CI->getDebugLoc());
  if (!CI->getType()->isVoidTy())
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
  CI->eraseFromParent();
  return true;
}

bool lowerCheckpointMarkers(Module &M) {
  SmallVector<CallInst *, 4> calls;
  for (Function &F : M)
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I)) {
        auto *callee =
            dyn_cast<Function>(CI->getCalledOperand()->stripPointerCasts());
        // Fortran callers use an implicit interface to f__enzyme_...
        if (callee && callee->getName().contains("__enzyme_checkpoint_for"))
          calls.push_back(CI);
      }
  bool changed = false;
  for (auto *CI : calls)
    changed |= lowerMarker(CI);
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
// The driver
//===----------------------------------------------------------------------===//

namespace {
struct DriverTypes {
  LLVMContext &Ctx;
  Type *Void;
  IntegerType *I32, *I64;
  PointerType *I8P;
  StructType *Action, *Region, *VTable, *Handle;
  FunctionType *InitFT, *NextFT, *StoreFT, *FinalizeFT, *StateFT, *PrimalFT,
      *AugFT, *RevFT, *FwdFT, *RevDriverFT;

  DriverTypes(LLVMContext &Ctx) : Ctx(Ctx) {
    Void = Type::getVoidTy(Ctx);
    I32 = Type::getInt32Ty(Ctx);
    I64 = Type::getInt64Ty(Ctx);
    I8P = getInt8PtrTy(Ctx);
    Action = StructType::get(Ctx, {I32, I64, I64, I64});
    Region = StructType::get(Ctx, {I8P, I64, I32, I32});
    VTable =
        StructType::get(Ctx, {I32, I8P, I8P, I8P, I8P, I8P, I8P, I8P, I8P});
    Handle = StructType::get(
        Ctx, {I8P, I8P, I8P, I64, I64, I8P, I64, I8P, I8P, I64, I32});
    InitFT = FunctionType::get(I8P, {I8P, I64, I64}, false);
    NextFT = FunctionType::get(Void, {I8P, getUnqual(Action)}, false);
    StoreFT = FunctionType::get(Void, {I8P, I64, I64, I8P, I64}, false);
    FinalizeFT = FunctionType::get(Void, {I8P}, false);
    StateFT = FunctionType::get(Void, {I8P, I64, I64, I8P}, false);
    PrimalFT = FunctionType::get(Void, {I8P, I64}, false);
    AugFT = FunctionType::get(I8P, {I8P, I64}, false);
    RevFT = FunctionType::get(Void, {I8P, I64, I8P}, false);
    // vt, data, start, n, regions, nregions, bytes, env, envsize, primal, aug
    FwdFT = FunctionType::get(
        I8P, {I8P, I8P, I64, I64, I8P, I64, I64, I8P, I64, I8P, I8P}, false);
    // handle, primal, aug, rev
    RevDriverFT = FunctionType::get(Void, {I8P, I8P, I8P, I8P}, false);
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
    callFn(primal, T.PrimalFT, {env, B.CreateAdd(start, j)});
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
};
} // namespace

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
        *nregions = A++, *bytes = A++, *env = A++, *envsize = A++,
        *primal = A++, *aug = A++;
  const DataLayout &DL = M.getDataLayout();

  auto *action = B.CreateAlloca(T.Action, nullptr, "action");
  Value *h = B.CreateCall(
      D.Malloc, {ConstantInt::get(T.I64, DL.getTypeAllocSize(T.Handle))}, "h");
  D.store(T.Handle, h, H_VT, vt);
  D.store(T.Handle, h, H_Data, data);
  D.store(T.Handle, h, H_Start, start);
  D.store(T.Handle, h, H_N, n);
  D.store(T.Handle, h, H_NRegions, nregions);
  // The step's arguments and the regions outlive this call.
  Value *regionBytes = B.CreateMul(
      nregions, ConstantInt::get(T.I64, DL.getTypeAllocSize(T.Region)));
  Value *regionCopy = B.CreateCall(
      D.Malloc, {B.CreateAdd(regionBytes, ConstantInt::get(T.I64, 1))});
  B.CreateMemCpy(regionCopy, MaybeAlign(1), regions, MaybeAlign(1),
                 regionBytes);
  D.store(T.Handle, h, H_Regions, regionCopy);
  Value *envCopy = B.CreateCall(
      D.Malloc, {B.CreateAdd(envsize, ConstantInt::get(T.I64, 1))});
  B.CreateMemCpy(envCopy, MaybeAlign(1), env, MaybeAlign(1), envsize);
  D.store(T.Handle, h, H_Env, envCopy);
  D.store(T.Handle, h, H_LastTape, ConstantPointerNull::get(T.I8P));
  D.store(T.Handle, h, H_LastJ, ConstantInt::get(T.I64, 0));
  D.store(T.Handle, h, H_Empty, ConstantInt::get(T.I32, 0));
  Value *state = B.CreateCall(
      T.InitFT,
      B.CreatePointerCast(D.vtFn(vt, VT_Init, "init"), getUnqual(T.InitFT)),
      {data, n, bytes}, "state");
  D.store(T.Handle, h, H_State, state);

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
  auto *fwdBB = D.block("forward");
  auto *turnBB = D.block("firstuturn");
  auto *doneBB = D.block("done");
  auto *SW = B.CreateSwitch(flag, bad, 4);
  SW->addCase(ConstantInt::get(T.I32, CKPT_STORE), storeBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_FORWARD), fwdBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_FIRSTUTURN), turnBB);
  SW->addCase(ConstantInt::get(T.I32, CKPT_DONE), doneBB);

  B.SetInsertPoint(bad);
  D.trap();

  B.SetInsertPoint(storeBB);
  D.snapshot(true, vt, state, cp, it, regionCopy, nregions, envCopy);
  B.CreateBr(loop);

  B.SetInsertPoint(fwdBB);
  D.forwardSteps(primal, envCopy, start, sit, it);
  B.CreateBr(loop);

  // The last step runs with taping; its tape is the first to be reversed.
  B.SetInsertPoint(turnBB);
  Value *lastj = B.CreateSub(it, ConstantInt::get(T.I64, 1));
  Value *tape =
      B.CreateCall(T.AugFT, B.CreatePointerCast(aug, getUnqual(T.AugFT)),
                   {envCopy, B.CreateAdd(start, lastj)}, "tape");
  D.store(T.Handle, h, H_LastTape, tape);
  D.store(T.Handle, h, H_LastJ, lastj);
  B.CreateRet(h);

  // Nothing to reverse (n == 0).
  B.SetInsertPoint(doneBB);
  D.store(T.Handle, h, H_Empty, ConstantInt::get(T.I32, 1));
  B.CreateRet(h);
  return F;
}

static Function *getOrCreateRevDriver(Module &M, DriverTypes &T) {
  if (auto *F = M.getFunction("__enzyme_ckpt_rev"))
    return F;
  auto *F = Function::Create(T.RevDriverFT, GlobalValue::InternalLinkage,
                             "__enzyme_ckpt_rev", &M);
  F->addFnAttr(Attribute::NoInline);
  DriverBuilder D(T, F);
  auto &B = D.B;
  auto *A = F->arg_begin();
  Value *h = A++, *primal = A++, *aug = A++, *rev = A++;

  auto *action = B.CreateAlloca(T.Action, nullptr, "action");
  Value *vt = D.load(T.Handle, h, H_VT, "vt");
  Value *state = D.load(T.Handle, h, H_State, "state");
  Value *start = D.load(T.Handle, h, H_Start, "start");
  Value *regions = D.load(T.Handle, h, H_Regions, "regions");
  Value *nregions = D.load(T.Handle, h, H_NRegions, "nregions");
  Value *env = D.load(T.Handle, h, H_Env, "env");
  Value *empty = D.load(T.Handle, h, H_Empty, "empty");

  auto *first = D.block("firstuturn");
  auto *loop = D.block("loop");
  auto *finish = D.block("finish");
  B.CreateCondBr(B.CreateICmpNE(empty, ConstantInt::get(T.I32, 0)), finish,
                 first);

  B.SetInsertPoint(first);
  Value *lastj = D.load(T.Handle, h, H_LastJ, "lastj");
  Value *tape = D.load(T.Handle, h, H_LastTape, "lasttape");
  D.callFn(rev, T.RevFT, {env, B.CreateAdd(start, lastj), tape});
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
  Value *t = B.CreateCall(T.AugFT, B.CreatePointerCast(aug, getUnqual(T.AugFT)),
                          {env, i}, "tape");
  D.callFn(rev, T.RevFT, {env, i, t});
  B.CreateBr(loop);

  B.SetInsertPoint(finish);
  D.callIfSet(D.vtFn(vt, VT_Finalize, "finalize"), T.FinalizeFT, {state});
  B.CreateCall(D.Free, {env});
  B.CreateCall(D.Free, {regions});
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
  /// if it has one.
  StructType *env;
  std::string suffix;

  StepInfo(Function *loop)
      : loop(loop), step(getStep(loop)), firstArg(getFirstStepArg(loop)),
        stepTypeInfo(step), env(nullptr) {}
};
} // namespace

static bool getStepInfo(StepInfo &S, ArrayRef<DIFFE_TYPE> constant_args,
                        const FnTypeInfo &typeInfo, unsigned width,
                        RequestContext &context) {
  if (width != 1) {
    EmitNoDerivativeError("checkpointed loops do not support vector mode yet",
                          S.loop, context);
    return false;
  }
  LLVMContext &Ctx = S.loop->getContext();
  S.stepActivity.push_back(DIFFE_TYPE::CONSTANT);
  SmallVector<Type *, 8> envTys;
  S.suffix = "";
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
      envTys.push_back(T);
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

static const AugmentedReturn &
getStepAugmented(EnzymeLogic &Logic, RequestContext context, StepInfo &S,
                 TypeAnalysis &TA, bool runtimeActivity, bool strongZero,
                 bool AtomicAdd) {
  // Later steps overwrite what a step reads: nothing is left uncached.
  std::vector<bool> overwritten(S.step->arg_size(), true);
  std::vector<bool> nowrite(S.step->arg_size(), false);
  return Logic.CreateAugmentedPrimal(
      context, S.step, DIFFE_TYPE::CONSTANT, S.stepActivity, TA,
      /*returnUsed*/ false, /*shadowReturnUsed*/ false, S.stepTypeInfo,
      /*subsequent_calls_may_write*/ true, overwritten, nowrite,
      /*forceAnonymousTape*/ true, runtimeActivity, strongZero, /*width*/ 1,
      AtomicAdd);
}

static Type *getTapeType(const AugmentedReturn &aug) {
  auto found = aug.returns.find(AugmentedStruct::Tape);
  if (found == aug.returns.end())
    return nullptr;
  Type *RT = aug.fn->getReturnType();
  return found->second == -1
             ? RT
             : cast<StructType>(RT)->getElementType(found->second);
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
      if (shadows)
        args.push_back(B.CreateLoad(T, B.CreateStructGEP(S.env, env, field)));
      field++;
    }
  }
}

static Function *createTrampoline(Module &M, FunctionType *FT,
                                  const Twine &name) {
  auto *F = Function::Create(FT, GlobalValue::InternalLinkage, name, &M);
  BasicBlock::Create(M.getContext(), "entry", F);
  return F;
}

static Function *getPrimalTrampoline(DriverTypes &T, StepInfo &S) {
  Module &M = *S.loop->getParent();
  std::string name =
      ("enzyme.ckpt.primal." + S.loop->getName() + "." + S.suffix).str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = createTrampoline(M, T.PrimalFT, name);
  IRBuilder<> B(&F->getEntryBlock());
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), /*shadows*/ false, args);
  B.CreateCall(S.step, args)->setCallingConv(S.step->getCallingConv());
  B.CreateRetVoid();
  return F;
}

static Function *getAugTrampoline(DriverTypes &T, StepInfo &S,
                                  const AugmentedReturn &aug) {
  Module &M = *S.loop->getParent();
  std::string name =
      ("enzyme.ckpt.aug." + S.loop->getName() + "." + S.suffix).str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = createTrampoline(M, T.AugFT, name);
  IRBuilder<> B(&F->getEntryBlock());
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), /*shadows*/ true, args);
  auto *call = B.CreateCall(aug.fn, args);
  call->setCallingConv(aug.fn->getCallingConv());
  Value *tape = ConstantPointerNull::get(T.I8P);
  auto found = aug.returns.find(AugmentedStruct::Tape);
  if (found != aug.returns.end()) {
    tape = found->second == -1
               ? (Value *)call
               : B.CreateExtractValue(call, (unsigned)found->second);
    tape = B.CreatePointerCast(tape, T.I8P);
  }
  B.CreateRet(tape);
  return F;
}

static Function *getRevTrampoline(DriverTypes &T, StepInfo &S, Function *rev,
                                  Type *tapeType) {
  Module &M = *S.loop->getParent();
  std::string name =
      ("enzyme.ckpt.rev." + S.loop->getName() + "." + S.suffix).str();
  if (auto *F = M.getFunction(name))
    return F;
  auto *F = createTrampoline(M, T.RevFT, name);
  IRBuilder<> B(&F->getEntryBlock());
  SmallVector<Value *, 8> args;
  loadStepArgs(B, S, F->getArg(0), F->getArg(1), /*shadows*/ true, args);
  if (tapeType)
    args.push_back(B.CreatePointerCast(F->getArg(2), tapeType));
  auto *call = B.CreateCall(rev, args);
  call->setCallingConv(rev->getCallingConv());
  B.CreateRetVoid();
  return F;
}

//===----------------------------------------------------------------------===//
// The augmented forward and reverse passes of the loop
//===----------------------------------------------------------------------===//

/// The loop's parameter types, each followed by its shadow if duplicated.
static SmallVector<Type *, 8>
getInterleavedParams(Function *loop, ArrayRef<DIFFE_TYPE> constant_args) {
  SmallVector<Type *, 8> params;
  for (unsigned k = 0; k < loop->arg_size(); k++) {
    params.push_back(loop->getArg(k)->getType());
    if (constant_args[k] == DIFFE_TYPE::DUP_ARG ||
        constant_args[k] == DIFFE_TYPE::DUP_NONEED)
      params.push_back(loop->getArg(k)->getType());
  }
  return params;
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
  const DataLayout &DL = M.getDataLayout();
  DriverTypes T(Ctx);

  StepInfo S(loop);
  if (!getStepInfo(S, constant_args, typeInfo, width, context))
    return nullptr;
  auto &aug = getStepAugmented(Logic, context, S, TA, runtimeActivity,
                               strongZero, AtomicAdd);

  auto *FT = FunctionType::get(T.I8P, getInterleavedParams(loop, constant_args),
                               false);
  auto *F = Function::Create(FT, GlobalValue::InternalLinkage,
                             "augmented_" + loop->getName(), &M);
  IRBuilder<> B(BasicBlock::Create(Ctx, "entry", F));

  // Map each loop parameter to the new arguments.
  SmallVector<Value *, 8> primals, shadows;
  {
    auto *A = F->arg_begin();
    for (unsigned k = 0; k < loop->arg_size(); k++) {
      primals.push_back(A++);
      if (constant_args[k] == DIFFE_TYPE::DUP_ARG ||
          constant_args[k] == DIFFE_TYPE::DUP_NONEED)
        shadows.push_back(A++);
      else
        shadows.push_back(nullptr);
    }
  }

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

  // The regions: those marked at the call, then the globals.
  auto globals = getGlobalRegions(S.step);
  unsigned nmarked = getNumRegions(loop);
  unsigned nregions = nmarked + globals.size();
  auto *regionArr = ArrayType::get(T.Region, std::max(nregions, 1u));
  auto *regions = B.CreateAlloca(regionArr, nullptr, "regions");
  Value *bytes = ConstantInt::get(T.I64, 0);
  auto setRegion = [&](unsigned r, Value *ptr, Value *size) {
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
  };
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
    auto *table = new GlobalVariable(
        M, tableTy, /*isConstant*/ true, GlobalValue::PrivateLinkage,
        ConstantArray::get(tableTy, entries),
        "enzyme.ckpt.regions." + S.step->getName());
    B.CreateMemCpy(B.CreateConstInBoundsGEP2_32(regionArr, regions, 0, nmarked),
                   MaybeAlign(1), table, MaybeAlign(1),
                   DL.getTypeAllocSize(tableTy));
    bytes = ConstantInt::get(T.I64, globalBytes);
  }
  for (unsigned r = 0; r < nmarked; r++)
    setRegion(r, primals[LoopFixedParams + 2 * r],
              primals[LoopFixedParams + 2 * r + 1]);

  if (EnzymePrintCheckpointRegions) {
    llvm::errs() << "checkpoint regions of " << S.step->getName() << ":\n";
    for (unsigned r = 0; r < nmarked; r++)
      llvm::errs() << "  marked region " << r << "\n";
    for (auto *GV : globals)
      llvm::errs() << "  global " << GV->getName() << " ("
                   << DL.getTypeAllocSize(GV->getValueType()) << " bytes)\n";
  }

  Function *fwd = getOrCreateFwdDriver(M, T);
  Value *h = B.CreateCall(
      fwd,
      {B.CreatePointerCast(primals[2], T.I8P),
       B.CreatePointerCast(primals[3], T.I8P), primals[0], primals[1],
       B.CreatePointerCast(regions, T.I8P), ConstantInt::get(T.I64, nregions),
       bytes, B.CreatePointerCast(env, T.I8P),
       ConstantInt::get(T.I64, DL.getTypeAllocSize(S.env)),
       B.CreatePointerCast(getPrimalTrampoline(T, S), T.I8P),
       B.CreatePointerCast(getAugTrampoline(T, S, aug), T.I8P)},
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
  auto &aug = getStepAugmented(Logic, context, S, TA, key.runtimeActivity,
                               key.strongZero, key.AtomicAdd);
  Type *tapeType = getTapeType(aug);

  std::vector<bool> overwritten(S.step->arg_size(), true);
  Function *rev = Logic.CreatePrimalAndGradient(
      context,
      (ReverseCacheKey){.todiff = S.step,
                        .retType = DIFFE_TYPE::CONSTANT,
                        .constant_args = S.stepActivity,
                        .subsequent_calls_may_write = true,
                        .overwritten_args = overwritten,
                        .returnUsed = false,
                        .shadowReturnUsed = false,
                        .mode = DerivativeMode::ReverseModeGradient,
                        .width = 1,
                        .freeMemory = true,
                        .AtomicAdd = key.AtomicAdd,
                        .additionalType = tapeType,
                        .forceAnonymousTape = true,
                        .typeInfo = S.stepTypeInfo,
                        .runtimeActivity = key.runtimeActivity,
                        .strongZero = key.strongZero},
      TA, &aug);
  if (!rev)
    return nullptr;

  auto params = getInterleavedParams(loop, key.constant_args);
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
  B.CreateCall(
      getOrCreateRevDriver(M, T),
      {h, B.CreatePointerCast(getPrimalTrampoline(T, S), T.I8P),
       B.CreatePointerCast(getAugTrampoline(T, S, aug), T.I8P),
       B.CreatePointerCast(getRevTrampoline(T, S, rev, tapeType), T.I8P)});
  B.CreateRetVoid();
  return F;
}
