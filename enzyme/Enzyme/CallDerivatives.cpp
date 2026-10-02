//===- CallDerivatives.cpp - Implementation of known call derivatives --===//
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
// This file contains the implementation of functions in instruction visitor
// AdjointGenerator that generate corresponding augmented forward pass code,
// and adjoints for certain known functions.
//
//===----------------------------------------------------------------------===//

#include "AdjointGenerator.h"
#include "FlangRuntime.h"

using namespace llvm;

static cl::opt<bool> EnzymeFortranMPIRuntimeAccumulate(
    "enzyme-fortran-mpi-runtime-accumulate", cl::init(false), cl::Hidden,
    cl::desc("Add received adjoints of Fortran MPI sends by the size of the "
             "MPI datatype even when the buffer type is known (for testing)"));

extern "C" {
void (*EnzymeShadowAllocRewrite)(LLVMValueRef, void *, LLVMValueRef, uint64_t,
                                 LLVMValueRef, uint8_t) = nullptr;
}

/// Whether \p Begin has exactly one llvm.julia.gc_preserve_end outside the
/// abort-only blocks the reverse pass drops, as reversing the region requires;
/// CanonicalizeGCPreserveEnds merges them where it can.
static bool hasSingleGCPreserveEnd(CallInst *Begin, GradientUtils *gutils) {
  unsigned Ends = 0;
  for (auto U : Begin->users())
    if (auto CI = dyn_cast<CallInst>(U))
      if (getFuncNameFromCall(CI) == "llvm.julia.gc_preserve_end" &&
          !gutils->notForAnalysis.count(CI->getParent()))
        ++Ends;
  return Ends == 1;
}

/// Forward mode of an MPI call whose derivative is the same call with the
/// buffers \p ShadowArgs replaced by their shadows (for the reductions, of a
/// sum). At vector width > 1 the call is replayed once per lane. All other
/// arguments, including the trailing `ierr` of the Fortran ABI, are passed as
/// in the original call. An inactive constant buffer, such as MPI_IN_PLACE or
/// MPI_STATUS_IGNORE, is passed unchanged.
static void createMPIForwardCall(CallInst &call, ArrayRef<unsigned> ShadowArgs,
                                 GradientUtils *gutils, IRBuilder<> &Builder2) {
  std::vector<ValueType> BundleTypes(call.arg_size(), ValueType::Primal);
  SmallVector<Value *, 2> shadows;
  for (unsigned i : ShadowArgs) {
    Value *arg = call.getArgOperand(i);
    if (isa<Constant>(arg) && gutils->isConstantValue(arg)) {
      shadows.push_back(nullptr);
      continue;
    }
    BundleTypes[i] = ValueType::Shadow;
    shadows.push_back(gutils->invertPointerM(arg, Builder2));
  }
  auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                         /*lookup*/ false);

  for (unsigned l = 0; l < gutils->getWidth(); l++) {
    SmallVector<Value *, 8> args;
    for (auto &op : call.args())
      args.push_back(gutils->getNewFromOriginal(op));
    for (auto en : llvm::enumerate(ShadowArgs)) {
      Value *sh = shadows[en.index()];
      if (!sh)
        continue;
      if (gutils->getWidth() > 1)
        sh = gutils->extractMeta(Builder2, sh, l);
      Type *T = args[en.value()]->getType();
      if (sh->getType()->isIntegerTy())
        sh = Builder2.CreateIntToPtr(sh, T);
      else if (sh->getType() != T)
        sh = Builder2.CreatePointerCast(sh, T);
      args[en.value()] = sh;
    }
    auto dcall = Builder2.CreateCall(call.getFunctionType(),
                                     call.getCalledOperand(), args, Defs);
    dcall->setCallingConv(call.getCallingConv());
    dcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
  }
}

// Point-to-point communication of the Fortran MPI ABI ("mpi_isend_",
// "mpi_wait_", ...). Requests there are INTEGER handles, which activity
// cannot hold a pointer, so the shadow of a request cannot carry Enzyme's
// bookkeeping as in the C ABI. Instead, every differentiated call gets a heap
// record (FortranMPIField) that the reverse pass reaches through the tape.
// Requests are active (see ActivityAnalysis), and the shadow of the request
// of a nonblocking call holds the index of its record in a table of slots,
// from which mpi_wait / mpi_waitall take it before the request is freed.
//
// Reverse mode of isend/irecv + wait:
//   reverse of the wait:          start the adjoint communication: irecv of
//                                 the adjoint of an isend's buffer into a
//                                 temporary, isend of the adjoint of an
//                                 irecv's buffer
//   reverse of the isend/irecv:   wait for it, then add the temporary to the
//                                 isend's buffer adjoint, or zero the irecv's
// Forward mode is the call on the shadows, as in the C ABI: the shadow of a
// request holds the request of the tangent communication.
namespace {
enum class FortranMPIField {
  Slot = 0,     // i32, index in the table of slots (nonblocking calls)
  Handle = 1,   // i32, primal request handle
  DBuf = 2,     // ptr, shadow of the buffer
  Tmp = 3,      // ptr, received adjoint of an isend's buffer
  Count = 4,    // i32
  DataType = 5, // i32
  Peer = 6,     // i32, destination or source
  Tag = 7,      // i32
  Comm = 8,     // i32
  Kind = 9,     // i8, FortranMPIKind
  State = 10,   // i8, FortranMPIState
  AdjReq = 11,  // i32, request of the adjoint (or tangent) communication
};
enum class FortranMPIKind { Isend = 1, Irecv = 2 };
enum class FortranMPIState { Posted = 0, Waited = 1, Started = 2 };
} // namespace

static StructType *getFortranMPIRecord(LLVMContext &C) {
  auto i32 = Type::getInt32Ty(C);
  auto i8 = Type::getInt8Ty(C);
  auto P = getInt8PtrTy(C);
  Type *types[] = {i32, i32, P, P, i32, i32, i32, i32, i32, i8, i8, i32};
  return StructType::get(C, types, false);
}

static Value *getFortranMPIField(IRBuilder<> &B, Value *R, FortranMPIField F) {
  return B.CreateStructGEP(getFortranMPIRecord(B.getContext()), R,
                           (unsigned)F);
}

/// The Fortran MPI routine \p name ("MPI_Irecv", ...) in the mangling of
/// \p caller; every argument, including the trailing ierr, is a pointer.
static FunctionCallee getFortranMPIFunction(Module &M, StringRef caller,
                                            StringRef name, unsigned nargs) {
  SmallVector<Type *, 8> tys(nargs, getInt8PtrTy(M.getContext()));
  return M.getOrInsertFunction(
      getRenamedPerCallingConv(caller, name),
      FunctionType::get(Type::getVoidTy(M.getContext()), tys, false));
}

/// The table of slots of records of pending nonblocking requests (slot 0 is
/// none): its storage and its capacity.
static GlobalVariable *getFortranMPISlotsGlobal(Module &M, bool capacity) {
  StringRef name = capacity ? "__enzyme_fortran_mpi_nslots"
                            : "__enzyme_fortran_mpi_slots";
  if (auto GV = M.getNamedGlobal(name))
    return GV;
  Type *T = capacity ? (Type *)Type::getInt32Ty(M.getContext())
                     : (Type *)getInt8PtrTy(M.getContext());
  return new GlobalVariable(M, T, /*isConstant*/ false,
                            GlobalValue::LinkOnceODRLinkage,
                            Constant::getNullValue(T), name);
}

static Function *createFortranMPIHelper(Module &M, StringRef name,
                                        FunctionType *FT, bool &created) {
  Function *F = cast<Function>(M.getOrInsertFunction(name, FT).getCallee());
  created = F->empty();
  if (created) {
    F->setLinkage(Function::LinkageTypes::InternalLinkage);
    F->addFnAttr(Attribute::NoUnwind);
  }
  return F;
}

/// i32 slot_alloc(ptr R): put the record R in a free slot of the table,
/// growing it if needed, and return the slot.
static Function *getFortranMPISlotAlloc(Module &M) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(M, "__enzyme_fortran_mpi_slot_alloc",
                                       FunctionType::get(i32, {P}, false),
                                       created);
  if (!created)
    return F;
  auto slotsGV = getFortranMPISlotsGlobal(M, false);
  auto capGV = getFortranMPISlotsGlobal(M, true);
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *loop = BasicBlock::Create(C, "loop", F);
  BasicBlock *check = BasicBlock::Create(C, "check", F);
  BasicBlock *grow = BasicBlock::Create(C, "grow", F);
  BasicBlock *found = BasicBlock::Create(C, "found", F);
  IRBuilder<> B(entry);
  Value *slots = B.CreateLoad(P, slotsGV, "slots");
  Value *cap = B.CreateLoad(i32, capGV, "cap");
  B.CreateBr(loop);

  B.SetInsertPoint(loop);
  PHINode *i = B.CreatePHI(i32, 2, "i");
  i->addIncoming(ConstantInt::get(i32, 1), entry);
  B.CreateCondBr(B.CreateICmpSLT(i, cap), check, grow);

  B.SetInsertPoint(check);
  Value *inc = B.CreateAdd(i, ConstantInt::get(i32, 1));
  Value *isFree = B.CreateIsNull(
      B.CreateLoad(P, B.CreateInBoundsGEP(P, slots, {B.CreateZExt(i, i64)})));
  i->addIncoming(inc, check);
  BasicBlock *foundOld = BasicBlock::Create(C, "found.old", F);
  B.CreateCondBr(isFree, foundOld, loop);
  B.SetInsertPoint(foundOld);
  B.CreateBr(found);

  // Double the table (at least 64 slots); the first new slot is free
  B.SetInsertPoint(grow);
  Value *newcap = B.CreateSelect(
      B.CreateICmpSLT(cap, ConstantInt::get(i32, 32)), ConstantInt::get(i32, 64),
      B.CreateMul(cap, ConstantInt::get(i32, 2)));
  auto ReallocFT = FunctionType::get(P, {P, i64}, false);
  Value *bytes = B.CreateMul(B.CreateZExt(newcap, i64), ConstantInt::get(i64, 8));
  Value *newslots =
      B.CreateCall(M.getOrInsertFunction("realloc", ReallocFT), {slots, bytes});
  Value *oldbytes = B.CreateMul(B.CreateZExt(cap, i64), ConstantInt::get(i64, 8));
  Type *memsetTys[] = {P, i64};
  B.CreateCall(getIntrinsicDeclaration(&M, Intrinsic::memset, memsetTys),
               {B.CreateInBoundsGEP(Type::getInt8Ty(C), newslots, {oldbytes}),
                ConstantInt::get(Type::getInt8Ty(C), 0),
                B.CreateSub(bytes, oldbytes), ConstantInt::getFalse(C)});
  B.CreateStore(newslots, slotsGV);
  B.CreateStore(newcap, capGV);
  Value *first = B.CreateSelect(B.CreateICmpSLT(cap, ConstantInt::get(i32, 1)),
                                ConstantInt::get(i32, 1), cap);
  B.CreateBr(found);

  B.SetInsertPoint(found);
  PHINode *slot = B.CreatePHI(i32, 2, "slot");
  slot->addIncoming(i, foundOld);
  slot->addIncoming(first, grow);
  PHINode *table = B.CreatePHI(P, 2, "table");
  table->addIncoming(slots, foundOld);
  table->addIncoming(newslots, grow);
  B.CreateStore(F->getArg(0),
                B.CreateInBoundsGEP(P, table, {B.CreateZExt(slot, i64)}));
  B.CreateRet(slot);
  return F;
}

/// ptr slot_take(i32 slot, i32 handle): take the record of the request
/// `handle` out of slot `slot` of the table and return it, or null if the
/// slot does not hold the record of that request (e.g. a request of an
/// inactive buffer, or MPI_REQUEST_NULL).
static Function *getFortranMPISlotTake(Module &M) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(M, "__enzyme_fortran_mpi_slot_take",
                                       FunctionType::get(P, {i32, i32}, false),
                                       created);
  if (!created)
    return F;
  auto RecTy = getFortranMPIRecord(C);
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *inrange = BasicBlock::Create(C, "inrange", F);
  BasicBlock *nonnull = BasicBlock::Create(C, "nonnull", F);
  BasicBlock *found = BasicBlock::Create(C, "found", F);
  BasicBlock *none = BasicBlock::Create(C, "none", F);
  Value *slot = F->getArg(0), *handle = F->getArg(1);
  IRBuilder<> B(entry);
  Value *cap = B.CreateLoad(i32, getFortranMPISlotsGlobal(M, true));
  B.CreateCondBr(B.CreateAnd(B.CreateICmpSGT(slot, ConstantInt::get(i32, 0)),
                             B.CreateICmpSLT(slot, cap)),
                 inrange, none);
  B.SetInsertPoint(inrange);
  Value *ptr = B.CreateInBoundsGEP(
      P, B.CreateLoad(P, getFortranMPISlotsGlobal(M, false)),
      {B.CreateZExt(slot, i64)});
  Value *rec = B.CreateLoad(P, ptr, "rec");
  B.CreateCondBr(B.CreateIsNull(rec), none, nonnull);
  B.SetInsertPoint(nonnull);
  Value *h = B.CreateLoad(
      i32, B.CreateStructGEP(RecTy, rec, (unsigned)FortranMPIField::Handle));
  B.CreateCondBr(B.CreateICmpEQ(h, handle), found, none);
  B.SetInsertPoint(found);
  B.CreateStore(ConstantPointerNull::get(cast<PointerType>(P)), ptr);
  B.CreateStore(
      ConstantInt::get(Type::getInt8Ty(C), (int)FortranMPIState::Waited),
      B.CreateStructGEP(RecTy, rec, (unsigned)FortranMPIField::State));
  B.CreateRet(rec);
  B.SetInsertPoint(none);
  B.CreateRet(ConstantPointerNull::get(cast<PointerType>(P)));
  return F;
}

/// Put the record \p R of a nonblocking call, whose request handle is stored
/// at \p request, in a slot, and store the slot in the shadow \p drequest
/// of the request.
static void assignFortranMPISlot(IRBuilder<> &B, Value *R, Value *request,
                                 Value *drequest) {
  auto &M = *B.GetInsertBlock()->getModule();
  auto i32 = Type::getInt32Ty(B.getContext());
  B.CreateStore(B.CreateLoad(i32, request),
                getFortranMPIField(B, R, FortranMPIField::Handle));
  Value *slot = B.CreateCall(getFortranMPISlotAlloc(M), {R});
  B.CreateStore(slot, getFortranMPIField(B, R, FortranMPIField::Slot));
  B.CreateStore(slot, drequest);
}

/// void start(ptr R): start the adjoint communication of the record of an
/// isend or irecv (if any and not yet started) in reverse mode.
static Function *getFortranMPIStart(Module &M, StringRef caller) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i8 = Type::getInt8Ty(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(
      M, ("__enzyme_fortran_mpi_start_" + caller).str(),
      FunctionType::get(Type::getVoidTy(C), {P}, false), created);
  if (!created)
    return F;
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *notnull = BasicBlock::Create(C, "notnull", F);
  BasicBlock *nonnull = BasicBlock::Create(C, "start", F);
  BasicBlock *isend = BasicBlock::Create(C, "isend", F);
  BasicBlock *irecv = BasicBlock::Create(C, "irecv", F);
  BasicBlock *end = BasicBlock::Create(C, "end", F);

  Value *R = F->getArg(0);
  IRBuilder<> B(entry);
  Value *ierr = B.CreateAlloca(i32, nullptr, "ierr");
  Value *tysize = B.CreateAlloca(i32, nullptr, "tysize");
  B.CreateCondBr(B.CreateIsNull(R), end, notnull);

  B.SetInsertPoint(notnull);
  Value *state = B.CreateLoad(i8, getFortranMPIField(B, R, FortranMPIField::State));
  B.CreateCondBr(
      B.CreateICmpEQ(state, ConstantInt::get(i8, (int)FortranMPIState::Started)),
      end, nonnull);

  B.SetInsertPoint(nonnull);
  B.CreateStore(ConstantInt::get(i8, (int)FortranMPIState::Started),
                getFortranMPIField(B, R, FortranMPIField::State));
  Value *kind = B.CreateLoad(i8, getFortranMPIField(B, R, FortranMPIField::Kind));
  B.CreateCondBr(B.CreateICmpEQ(kind, ConstantInt::get(i8, (int)FortranMPIKind::Isend)),
                 isend, irecv);

  auto args = [&](IRBuilder<> &B, Value *buf) {
    return SmallVector<Value *, 8>{
        buf,
        getFortranMPIField(B, R, FortranMPIField::Count),
        getFortranMPIField(B, R, FortranMPIField::DataType),
        getFortranMPIField(B, R, FortranMPIField::Peer),
        getFortranMPIField(B, R, FortranMPIField::Tag),
        getFortranMPIField(B, R, FortranMPIField::Comm),
        getFortranMPIField(B, R, FortranMPIField::AdjReq),
        ierr};
  };

  // The adjoint of an isend's buffer comes back from the receiver
  B.SetInsertPoint(isend);
  B.CreateCall(getFortranMPIFunction(M, caller, "MPI_Type_size", 3),
               {getFortranMPIField(B, R, FortranMPIField::DataType), tysize, ierr});
  Value *len = B.CreateMul(
      B.CreateSExt(B.CreateLoad(i32, getFortranMPIField(B, R, FortranMPIField::Count)), i64),
      B.CreateSExt(B.CreateLoad(i32, tysize), i64));
  Value *tmp = CreateAllocation(B, i8, len, "mpi_adjoint_recv");
  B.CreateStore(tmp, getFortranMPIField(B, R, FortranMPIField::Tmp));
  B.CreateCall(getFortranMPIFunction(M, caller, "MPI_Irecv", 8), args(B, tmp));
  B.CreateBr(end);

  // The adjoint of an irecv's buffer goes back to the sender
  B.SetInsertPoint(irecv);
  B.CreateCall(getFortranMPIFunction(M, caller, "MPI_Isend", 8),
               args(B, B.CreateLoad(P, getFortranMPIField(B, R, FortranMPIField::DBuf))));
  B.CreateBr(end);

  B.SetInsertPoint(end);
  B.CreateRetVoid();
  return F;
}

/// void finish(ptr R): complete the adjoint communication of the record of
/// an isend or irecv; one whose wait the primal did not pass is started here.
static Function *getFortranMPIFinish(Module &M, StringRef caller) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i8 = Type::getInt8Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(
      M, ("__enzyme_fortran_mpi_finish_" + caller).str(),
      FunctionType::get(Type::getVoidTy(C), {P}, false), created);
  if (!created)
    return F;
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *nonnull = BasicBlock::Create(C, "nonnull", F);
  BasicBlock *end = BasicBlock::Create(C, "end", F);

  Value *R = F->getArg(0);
  IRBuilder<> B(entry);
  Value *ierr = B.CreateAlloca(i32, nullptr, "ierr");
  // Large enough for MPI_STATUS_SIZE of the common MPI libraries
  Value *status = B.CreateAlloca(ArrayType::get(i32, 32), nullptr, "status");
  B.CreateCondBr(B.CreateIsNull(R), end, nonnull);

  B.SetInsertPoint(nonnull);
  {
    BasicBlock *unlink = BasicBlock::Create(C, "unlink", F);
    BasicBlock *wait = BasicBlock::Create(C, "wait", F);
    Value *state = B.CreateLoad(i8, getFortranMPIField(B, R, FortranMPIField::State));
    B.CreateCondBr(
        B.CreateICmpEQ(state, ConstantInt::get(i8, (int)FortranMPIState::Posted)),
        unlink, wait);
    B.SetInsertPoint(unlink);
    B.CreateCall(
        getFortranMPISlotTake(M),
        {B.CreateLoad(i32, getFortranMPIField(B, R, FortranMPIField::Slot)),
         B.CreateLoad(i32, getFortranMPIField(B, R, FortranMPIField::Handle))});
    B.CreateBr(wait);
    B.SetInsertPoint(wait);
    B.CreateCall(getFortranMPIStart(M, caller), {R});
  }
  B.CreateCall(getFortranMPIFunction(M, caller, "MPI_Wait", 3),
               {getFortranMPIField(B, R, FortranMPIField::AdjReq), status, ierr});
  B.CreateBr(end);

  B.SetInsertPoint(end);
  B.CreateRetVoid();
  return F;
}

/// ptr waitall_save(ptr count, ptr requests, ptr drequests): take the
/// records of the requests of an mpi_waitall out of their slots (and clear
/// the shadows of the requests): {i64 count, ptr records[count]}.
static Function *getFortranMPIWaitallSave(Module &M) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(M, "__enzyme_fortran_mpi_waitall_save",
                                       FunctionType::get(P, {P, P, P}, false),
                                       created);
  if (!created)
    return F;
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *loop = BasicBlock::Create(C, "loop", F);
  BasicBlock *end = BasicBlock::Create(C, "end", F);
  IRBuilder<> B(entry);
  Value *n = B.CreateSExt(B.CreateLoad(i32, F->getArg(0)), i64, "n");
  Value *arr = CreateAllocation(B, i64, B.CreateAdd(n, ConstantInt::get(i64, 1)),
                                "mpi_waitall_records");
  B.CreateStore(n, arr);
  B.CreateCondBr(B.CreateICmpSGT(n, ConstantInt::get(i64, 0)), loop, end);

  B.SetInsertPoint(loop);
  PHINode *i = B.CreatePHI(i64, 2, "i");
  i->addIncoming(ConstantInt::get(i64, 0), entry);
  Value *req = B.CreateInBoundsGEP(i32, F->getArg(1), {i});
  Value *dreq = B.CreateInBoundsGEP(i32, F->getArg(2), {i});
  Value *rec = B.CreateCall(getFortranMPISlotTake(M),
                            {B.CreateLoad(i32, dreq), B.CreateLoad(i32, req)});
  B.CreateStore(ConstantInt::get(i32, 0), dreq);
  Value *inc = B.CreateAdd(i, ConstantInt::get(i64, 1));
  B.CreateStore(rec, B.CreateInBoundsGEP(P, arr, {inc}));
  i->addIncoming(inc, loop);
  B.CreateCondBr(B.CreateICmpEQ(inc, n), end, loop);

  B.SetInsertPoint(end);
  B.CreateRet(arr);
  return F;
}

/// void waitall_each(ptr records): apply \p Each to the records saved by
/// waitall_save, then free them.
static Function *getFortranMPIWaitallEach(Module &M, Function *Each) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(
      M, ("__enzyme_fortran_mpi_waitall" + Each->getName()).str(),
      FunctionType::get(Type::getVoidTy(C), {P}, false), created);
  if (!created)
    return F;
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *loop = BasicBlock::Create(C, "loop", F);
  BasicBlock *end = BasicBlock::Create(C, "end", F);
  Value *arr = F->getArg(0);
  IRBuilder<> B(entry);
  Value *n = B.CreateLoad(i64, arr, "n");
  B.CreateCondBr(B.CreateICmpSGT(n, ConstantInt::get(i64, 0)), loop, end);

  B.SetInsertPoint(loop);
  PHINode *i = B.CreatePHI(i64, 2, "i");
  i->addIncoming(ConstantInt::get(i64, 0), entry);
  Value *inc = B.CreateAdd(i, ConstantInt::get(i64, 1));
  B.CreateCall(Each, {B.CreateLoad(P, B.CreateInBoundsGEP(P, arr, {inc}))});
  i->addIncoming(inc, loop);
  B.CreateCondBr(B.CreateICmpEQ(inc, n), end, loop);

  B.SetInsertPoint(end);
  CreateDealloc(B, arr);
  B.CreateRetVoid();
  return F;
}

/// void accumulate(ptr dbuf, ptr tmp, i64 len, i32 tysize): dbuf += tmp as
/// doubles (MPI datatype of 8 or 16 bytes) or floats (4 bytes).
static Function *getFortranMPIAccumulate(Module &M) {
  auto &C = M.getContext();
  auto P = getInt8PtrTy(C);
  auto i32 = Type::getInt32Ty(C);
  auto i64 = Type::getInt64Ty(C);
  bool created;
  Function *F = createFortranMPIHelper(
      M, "__enzyme_fortran_mpi_accumulate",
      FunctionType::get(Type::getVoidTy(C), {P, P, i64, i32}, false), created);
  if (!created)
    return F;
  BasicBlock *entry = BasicBlock::Create(C, "entry", F);
  BasicBlock *end = BasicBlock::Create(C, "end", F);
  BasicBlock *notdouble = BasicBlock::Create(C, "notdouble", F);
  IRBuilder<> B(entry);
  Value *dbuf = F->getArg(0), *tmp = F->getArg(1), *len = F->getArg(2),
        *tysize = F->getArg(3);
  auto addLoop = [&](Type *T, BasicBlock *pred) {
    BasicBlock *loop = BasicBlock::Create(C, "loop", F, end);
    BasicBlock *body = BasicBlock::Create(C, "body", F, end);
    auto n = B.CreateUDiv(len, ConstantInt::get(i64, T->getPrimitiveSizeInBits() / 8));
    B.CreateBr(loop);
    B.SetInsertPoint(loop);
    PHINode *i = B.CreatePHI(i64, 2, "i");
    i->addIncoming(ConstantInt::get(i64, 0), pred);
    B.CreateCondBr(B.CreateICmpULT(i, n), body, end);
    B.SetInsertPoint(body);
    Value *dp = B.CreateInBoundsGEP(T, dbuf, {i});
    Value *tp = B.CreateInBoundsGEP(T, tmp, {i});
    B.CreateStore(B.CreateFAdd(B.CreateLoad(T, dp), B.CreateLoad(T, tp)), dp);
    i->addIncoming(B.CreateAdd(i, ConstantInt::get(i64, 1)), body);
    B.CreateBr(loop);
  };
  BasicBlock *isdouble = BasicBlock::Create(C, "double", F, notdouble);
  B.CreateCondBr(
      B.CreateOr(B.CreateICmpEQ(tysize, ConstantInt::get(i32, 8)),
                 B.CreateICmpEQ(tysize, ConstantInt::get(i32, 16))),
      isdouble, notdouble);
  B.SetInsertPoint(isdouble);
  addLoop(Type::getDoubleTy(C), isdouble);
  B.SetInsertPoint(notdouble);
  BasicBlock *isfloat = BasicBlock::Create(C, "float", F, end);
  B.CreateCondBr(B.CreateICmpEQ(tysize, ConstantInt::get(i32, 4)), isfloat,
                 end);
  B.SetInsertPoint(isfloat);
  addLoop(Type::getFloatTy(C), isfloat);
  B.SetInsertPoint(end);
  B.CreateRetVoid();
  return F;
}

bool AdjointGenerator::handleFortranMPIPointToPoint(CallInst &call,
                                                    Function *called,
                                                    StringRef funcName) {
  bool isIsend = funcName == "MPI_Isend";
  bool isIrecv = funcName == "MPI_Irecv";
  bool isWait = funcName == "MPI_Wait";
  bool isWaitall = funcName == "MPI_Waitall";
  bool isSend = funcName == "MPI_Send" || funcName == "MPI_Ssend";
  bool isRecv = funcName == "MPI_Recv";
  bool isBarrier = funcName == "MPI_Barrier";
  if (!(isIsend || isIrecv || isWait || isWaitall || isSend || isRecv ||
        isBarrier))
    return false;

  bool reverseMode = Mode == DerivativeMode::ReverseModePrimal ||
                     Mode == DerivativeMode::ReverseModeCombined ||
                     Mode == DerivativeMode::ReverseModeGradient;
  // Forward mode is the call on the shadows (and a barrier has none), as in
  // the C ABI (handleMPI).
  if (!reverseMode)
    return false;

  IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
  BuilderZ.setFastMathFlags(getFast());
  if (gutils->getWidth() > 1) {
    std::string s;
    raw_string_ostream ss(s);
    ss << funcName << " of the Fortran MPI ABI is not supported in vector "
       << "reverse mode: " << call;
    EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
    return true;
  }

  Module &M = *gutils->newFunc->getParent();
  LLVMContext &C = call.getContext();
  StringRef caller = called->getName();
  auto RecTy = getFortranMPIRecord(C);
  auto P = getInt8PtrTy(C);
  auto i8 = Type::getInt8Ty(C);
  auto i32 = Type::getInt32Ty(C);
  auto i64 = Type::getInt64Ty(C);
  auto newCall = cast<CallInst>(gutils->getNewFromOriginal(&call));
  bool primal = Mode == DerivativeMode::ReverseModePrimal ||
                Mode == DerivativeMode::ReverseModeCombined;
  bool gradient = Mode == DerivativeMode::ReverseModeGradient ||
                  Mode == DerivativeMode::ReverseModeCombined;

  // A record holding the arguments of an isend/irecv/send/recv and the shadow
  // of its buffer.
  auto newRecord = [&](IRBuilder<> &B, int kind) {
    Value *R = CreateAllocation(B, RecTy, ConstantInt::get(i64, 1),
                                "mpi_record");
    Value *dbuf = gutils->invertPointerM(call.getOperand(0), B);
    if (dbuf->getType()->isIntegerTy())
      dbuf = B.CreateIntToPtr(dbuf, P);
    B.CreateStore(dbuf, getFortranMPIField(B, R, FortranMPIField::DBuf));
    B.CreateStore(ConstantPointerNull::get(cast<PointerType>(P)),
                  getFortranMPIField(B, R, FortranMPIField::Tmp));
    FortranMPIField fields[] = {FortranMPIField::Count,
                                FortranMPIField::DataType,
                                FortranMPIField::Peer, FortranMPIField::Tag,
                                FortranMPIField::Comm};
    for (unsigned i = 0; i < 5; i++)
      B.CreateStore(
          B.CreateLoad(i32, gutils->getNewFromOriginal(call.getOperand(i + 1))),
          getFortranMPIField(B, R, fields[i]));
    B.CreateStore(ConstantInt::get(i8, kind),
                  getFortranMPIField(B, R, FortranMPIField::Kind));
    B.CreateStore(ConstantInt::get(i8, (int)FortranMPIState::Posted),
                  getFortranMPIField(B, R, FortranMPIField::State));
    return R;
  };

  // The record (or records of an mpi_waitall) the reverse pass needs, cached
  // on the tape. \p make builds it in the primal, at the builder given.
  auto cacheRecord = [&](IRBuilder<> &B, Value *R) -> Value * {
    if (!primal) {
      R = BuilderZ.CreatePHI(P, 0);
      return gutils->cacheForReverse(BuilderZ, R,
                                     getIndex(&call, CacheType::Tape, BuilderZ));
    }
    return gutils->cacheForReverse(B, R,
                                   getIndex(&call, CacheType::Tape, B));
  };

  IRBuilder<> After(newCall->getNextNode());
  After.SetCurrentDebugLocation(newCall->getDebugLoc());

  // Reverse mode
  Value *R = nullptr;
  if (isIsend || isIrecv) {
    if (gutils->isConstantInstruction(&call))
      return false;
    if (primal) {
      R = newRecord(After, (int)(isIsend ? FortranMPIKind::Isend
                                         : FortranMPIKind::Irecv));
      Value *dreq = gutils->invertPointerM(call.getOperand(6), After);
      assignFortranMPISlot(After, R,
                           gutils->getNewFromOriginal(call.getOperand(6)), dreq);
    }
    R = cacheRecord(After, R);
  } else if (isSend || isRecv) {
    if (primal)
      R = newRecord(BuilderZ, 0);
    R = cacheRecord(BuilderZ, R);
  } else if (isBarrier) {
    // The communicator, for the barrier of the reverse pass
    if (primal) {
      R = CreateAllocation(BuilderZ, RecTy, ConstantInt::get(i64, 1),
                           "mpi_record");
      BuilderZ.CreateStore(
          BuilderZ.CreateLoad(i32,
                              gutils->getNewFromOriginal(call.getOperand(0))),
          getFortranMPIField(BuilderZ, R, FortranMPIField::Comm));
    }
    R = cacheRecord(BuilderZ, R);
  } else if (isWait) {
    if (primal) {
      // Before the wait frees the request
      Value *dreq = gutils->invertPointerM(call.getOperand(0), BuilderZ);
      R = BuilderZ.CreateCall(
          getFortranMPISlotTake(M),
          {BuilderZ.CreateLoad(i32, dreq),
           BuilderZ.CreateLoad(i32,
                               gutils->getNewFromOriginal(call.getOperand(0)))});
      BuilderZ.CreateStore(ConstantInt::get(i32, 0), dreq);
    }
    R = cacheRecord(BuilderZ, R);
  } else {
    if (primal)
      R = BuilderZ.CreateCall(
          getFortranMPIWaitallSave(M),
          {gutils->getNewFromOriginal(call.getOperand(0)),
           gutils->getNewFromOriginal(call.getOperand(1)),
           gutils->invertPointerM(call.getOperand(1), BuilderZ)});
    R = cacheRecord(BuilderZ, R);
  }

  if (gradient) {
    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);
    R = lookup(R, Builder2);
    if (isBarrier) {
      Value *ierr = IRBuilder<>(gutils->inversionAllocs)
                        .CreateAlloca(i32, nullptr, "enzyme_mpi_ierr");
      Builder2.CreateCall(
          getFortranMPIFunction(M, caller, "MPI_Barrier", 2),
          {getFortranMPIField(Builder2, R, FortranMPIField::Comm), ierr});
      CreateDealloc(Builder2, R);
    } else if (isWait) {
      Builder2.CreateCall(getFortranMPIStart(M, caller), {R});
    } else if (isWaitall) {
      Builder2.CreateCall(
          getFortranMPIWaitallEach(M, getFortranMPIStart(M, caller)), {R});
    } else {
      Value *dbuf =
          Builder2.CreateLoad(P, getFortranMPIField(Builder2, R, FortranMPIField::DBuf));
      Value *tysize = MPI_TYPE_SIZE(
          getFortranMPIField(Builder2, R, FortranMPIField::DataType), Builder2,
          i32, called);
      Value *len = Builder2.CreateMul(
          Builder2.CreateSExt(
              Builder2.CreateLoad(i32, getFortranMPIField(Builder2, R, FortranMPIField::Count)),
              i64),
          Builder2.CreateSExt(tysize, i64));
      Value *ierr = IRBuilder<>(gutils->inversionAllocs)
                        .CreateAlloca(i32, nullptr, "enzyme_mpi_ierr");
      Value *tmp = nullptr;
      if (isIsend || isIrecv) {
        Builder2.CreateCall(getFortranMPIFinish(M, caller),
                            {R});
        if (isIsend)
          tmp = Builder2.CreateLoad(P, getFortranMPIField(Builder2, R, FortranMPIField::Tmp));
      } else {
        SmallVector<Value *, 8> args = {
            nullptr,
            getFortranMPIField(Builder2, R, FortranMPIField::Count),
            getFortranMPIField(Builder2, R, FortranMPIField::DataType),
            getFortranMPIField(Builder2, R, FortranMPIField::Peer),
            getFortranMPIField(Builder2, R, FortranMPIField::Tag),
            getFortranMPIField(Builder2, R, FortranMPIField::Comm)};
        if (isSend) {
          // The adjoint of the sent buffer comes back from the receiver
          tmp = CreateAllocation(Builder2, i8, len, "mpi_adjoint_recv");
          args[0] = tmp;
          args.push_back(IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(ArrayType::get(i32, 32), nullptr,
                                           "enzyme_mpi_status"));
          args.push_back(ierr);
          Builder2.CreateCall(getFortranMPIFunction(M, caller, "MPI_Recv", 8),
                              args);
        } else {
          // The adjoint of the received buffer goes back to the sender
          args[0] = dbuf;
          args.push_back(ierr);
          Builder2.CreateCall(getFortranMPIFunction(M, caller, "MPI_Send", 7),
                              args);
        }
      }
      if (tmp) {
        // adjoint(buf) += received adjoint
        auto &DL = M.getDataLayout();
        if (!EnzymeFortranMPIRuntimeAccumulate &&
            TR.query(call.getOperand(0))
                .Data0()
                .ShiftIndices(DL, 0, 1, 0)[{0}]
                .isFloat())
          DifferentiableMemCopyFloats(call, call.getOperand(0), tmp, dbuf, len,
                                      Builder2, {});
        else
          // The buffer type is unknown to type analysis (e.g. a buffer only
          // filled by MPI): add by the size of the MPI datatype.
          Builder2.CreateCall(getFortranMPIAccumulate(M),
                              {dbuf, tmp, len, tysize});
        CreateDealloc(Builder2, tmp);
      } else {
        // The received values were overwritten: their adjoint is zero
        Type *memsetTys[] = {P, i64};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(&M, Intrinsic::memset, memsetTys),
            {dbuf, ConstantInt::get(i8, 0), len,
             ConstantInt::getFalse(C)}));
        memset->addParamAttr(0, Attribute::NonNull);
      }
      CreateDealloc(Builder2, R);
    }
  }
  if (Mode == DerivativeMode::ReverseModeGradient)
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
  return true;
}

void AdjointGenerator::handleMPI(llvm::CallInst &call, llvm::Function *called,
                                 llvm::StringRef funcName) {
  using namespace llvm;

  assert(called);

  if (isFortranMPICall(called->getName()) &&
      handleFortranMPIPointToPoint(call, called, funcName))
    return;

  IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
  BuilderZ.setFastMathFlags(getFast());

  // In forward mode, the derivative of an MPI call is the call on the shadows,
  // which is replayed per lane at vector width > 1 (createMPIForwardCall).
  if (gutils->getWidth() > 1 && Mode != DerivativeMode::ForwardMode &&
      Mode != DerivativeMode::ForwardModeError) {
    std::string s;
    raw_string_ostream ss(s);
    ss << funcName << " is not supported in vector reverse mode: " << call;
    EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
    return;
  }

  // MPI send / recv can only send float/integers
  if (funcName == "PMPI_Isend" || funcName == "MPI_Isend" ||
      funcName == "PMPI_Irecv" || funcName == "MPI_Irecv") {
    if (!gutils->isConstantInstruction(&call)) {
      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined) {
        assert(!gutils->isConstantValue(call.getOperand(0)));
        assert(!gutils->isConstantValue(call.getOperand(6)));
        Value *d_req = gutils->invertPointerM(call.getOperand(6), BuilderZ);
        if (d_req->getType()->isIntegerTy()) {
          d_req = BuilderZ.CreateIntToPtr(
              d_req, getUnqual(getInt8PtrTy(call.getContext())));
        }

        auto i64 = Type::getInt64Ty(call.getContext());
        auto impi = getMPIHelper(call.getContext());

        Value *impialloc =
            CreateAllocation(BuilderZ, impi, ConstantInt::get(i64, 1));
        BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));

        d_req = BuilderZ.CreateBitCast(d_req, getUnqual(impialloc->getType()));
        Value *d_req_prev = BuilderZ.CreateLoad(impialloc->getType(), d_req);
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(d_req_prev,
                                       getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::Old>(BuilderZ, impialloc, impi));
        BuilderZ.CreateStore(impialloc, d_req);

        if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          Value *tysize =
              MPI_TYPE_SIZE(gutils->getNewFromOriginal(call.getOperand(2)),
                            BuilderZ, call.getType(), called);

          auto len_arg = BuilderZ.CreateZExtOrTrunc(
              gutils->getNewFromOriginal(call.getOperand(1)),
              Type::getInt64Ty(call.getContext()));
          len_arg = BuilderZ.CreateMul(
              len_arg,
              BuilderZ.CreateZExtOrTrunc(tysize,
                                         Type::getInt64Ty(call.getContext())),
              "", true, true);

          Value *firstallocation =
              CreateAllocation(BuilderZ, Type::getInt8Ty(call.getContext()),
                               len_arg, "mpirecv_malloccache");
          BuilderZ.CreateStore(firstallocation, getMPIMemberPtr<MPI_Elem::Buf>(
                                                    BuilderZ, impialloc, impi));
          BuilderZ.SetInsertPoint(gutils->getNewFromOriginal(&call));
        } else {
          Value *ibuf = gutils->invertPointerM(call.getOperand(0), BuilderZ);
          if (ibuf->getType()->isIntegerTy())
            ibuf =
                BuilderZ.CreateIntToPtr(ibuf, getInt8PtrTy(call.getContext()));
          BuilderZ.CreateStore(
              ibuf, getMPIMemberPtr<MPI_Elem::Buf>(BuilderZ, impialloc, impi));
        }

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(1)), i64),
            getMPIMemberPtr<MPI_Elem::Count>(BuilderZ, impialloc, impi));

        Value *dataType = gutils->getNewFromOriginal(call.getOperand(2));
        if (dataType->getType()->isIntegerTy())
          dataType = BuilderZ.CreateIntToPtr(
              dataType, getInt8PtrTy(dataType->getContext()));
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(dataType,
                                       getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::DataType>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(3)), i64),
            getMPIMemberPtr<MPI_Elem::Src>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            BuilderZ.CreateZExtOrTrunc(
                gutils->getNewFromOriginal(call.getOperand(4)), i64),
            getMPIMemberPtr<MPI_Elem::Tag>(BuilderZ, impialloc, impi));

        Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
        if (comm->getType()->isIntegerTy())
          comm = BuilderZ.CreateIntToPtr(comm,
                                         getInt8PtrTy(dataType->getContext()));
        BuilderZ.CreateStore(
            BuilderZ.CreatePointerCast(comm, getInt8PtrTy(call.getContext())),
            getMPIMemberPtr<MPI_Elem::Comm>(BuilderZ, impialloc, impi));

        BuilderZ.CreateStore(
            ConstantInt::get(
                Type::getInt8Ty(impialloc->getContext()),
                (funcName == "MPI_Isend" || funcName == "PMPI_Isend")
                    ? (int)MPI_CallType::ISEND
                    : (int)MPI_CallType::IRECV),
            getMPIMemberPtr<MPI_Elem::Call>(BuilderZ, impialloc, impi));
        // TODO old
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 17
        if (Function *recvfn = called->getParent()->getFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Wait"))) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
#endif
        if (statusType == nullptr) {
          statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
          llvm::errs() << " warning could not automatically determine mpi "
                          "status type, assuming [24 x i8]\n";
        }
        Value *req =
            lookup(gutils->getNewFromOriginal(call.getOperand(6)), Builder2);
        Value *d_req = lookup(
            gutils->invertPointerM(call.getOperand(6), Builder2), Builder2);
        if (d_req->getType()->isIntegerTy()) {
          d_req =
              Builder2.CreateIntToPtr(d_req, getInt8PtrTy(call.getContext()));
        }
        auto impi = getMPIHelper(call.getContext());
        Type *helperTy = getUnqual(impi);
        Value *helper = Builder2.CreatePointerCast(d_req, getUnqual(helperTy));
        helper = Builder2.CreateLoad(helperTy, helper);

        auto i64 = Type::getInt64Ty(call.getContext());

        Value *firstallocation;
        firstallocation = Builder2.CreateLoad(
            getInt8PtrTy(call.getContext()),
            getMPIMemberPtr<MPI_Elem::Buf>(Builder2, helper, impi));
        Value *len_arg = nullptr;
        if (auto C = dyn_cast<Constant>(
                gutils->getNewFromOriginal(call.getOperand(1)))) {
          len_arg = Builder2.CreateZExtOrTrunc(C, i64);
        } else {
          len_arg = Builder2.CreateLoad(
              i64, getMPIMemberPtr<MPI_Elem::Count>(Builder2, helper, impi));
        }
        Value *tysize = nullptr;
        if (auto C = dyn_cast<Constant>(
                gutils->getNewFromOriginal(call.getOperand(2)))) {
          tysize = C;
        } else {
          tysize = Builder2.CreateLoad(
              getInt8PtrTy(call.getContext()),
              getMPIMemberPtr<MPI_Elem::DataType>(Builder2, helper, impi));
        }

        Value *prev;
        prev = Builder2.CreateLoad(
            getInt8PtrTy(call.getContext()),
            getMPIMemberPtr<MPI_Elem::Old>(Builder2, helper, impi));

        Builder2.CreateStore(prev, Builder2.CreatePointerCast(
                                       d_req, getUnqual(prev->getType())));

        assert(shouldFree());

        assert(tysize);
        tysize = MPI_TYPE_SIZE(tysize, Builder2, call.getType(), called);

        Value *args[] = {/*req*/ req,
                         /*status*/ IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(statusType)};
        FunctionCallee waitFunc = nullptr;
        for (auto name : {
                 "MPI_Wait",
             })
          if (Function *recvfn = called->getParent()->getFunction(
                  getRenamedPerCallingConv(called->getName(), name))) {
            auto statusArg = recvfn->arg_end();
            statusArg--;
            if (statusArg->getType()->isIntegerTy())
              args[1] = Builder2.CreatePtrToInt(args[1], statusArg->getType());
            else
              args[1] = Builder2.CreateBitCast(args[1], statusArg->getType());
            waitFunc = recvfn;
            break;
          }
        if (!waitFunc) {
          Type *types[sizeof(args) / sizeof(*args)];
          for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
            types[i] = args[i]->getType();
          FunctionType *FT = FunctionType::get(call.getType(), types, false);
          waitFunc = called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Wait"), FT);
        }
        assert(waitFunc);

        // Need to preserve the shadow Request (operand 6 in isend/irecv),
        // which becomes operand 0 for iwait.
        auto ReqDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::None, ValueType::None, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::Shadow},
            Builder2, /*lookup*/ true);

        auto BufferDefs = gutils->getInvertedBundles(
            &call,
            {ValueType::Shadow, ValueType::None, ValueType::None,
             ValueType::None, ValueType::None, ValueType::None,
             ValueType::None},
            Builder2, /*lookup*/ true);

        auto fcall = Builder2.CreateCall(waitFunc, args, ReqDefs);
        fcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
        if (auto F = dyn_cast<Function>(waitFunc.getCallee()))
          fcall->setCallingConv(F->getCallingConv());
        len_arg = Builder2.CreateMul(
            len_arg,
            Builder2.CreateZExtOrTrunc(tysize,
                                       Type::getInt64Ty(Builder2.getContext())),
            "", true, true);
        if (funcName == "MPI_Irecv" || funcName == "PMPI_Irecv") {
          auto val_arg =
              ConstantInt::get(Type::getInt8Ty(Builder2.getContext()), 0);
          auto volatile_arg = ConstantInt::getFalse(Builder2.getContext());
          assert(!gutils->isConstantValue(call.getOperand(0)));
          auto dbuf = firstallocation;
          Value *nargs[] = {dbuf, val_arg, len_arg, volatile_arg};
          Type *tys[] = {dbuf->getType(), len_arg->getType()};

          auto memset = cast<CallInst>(Builder2.CreateCall(
              getIntrinsicDeclaration(called->getParent(), Intrinsic::memset,
                                      tys),
              nargs, BufferDefs));
          memset->addParamAttr(0, Attribute::NonNull);
        } else if (funcName == "MPI_Isend" || funcName == "PMPI_Isend") {
          assert(!gutils->isConstantValue(call.getOperand(0)));
          Value *shadow = lookup(
              gutils->invertPointerM(call.getOperand(0), Builder2), Builder2);

          // TODO add operand bundle (unless force inlined?)
          DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                      shadow, len_arg, Builder2, BufferDefs);

          if (shouldFree()) {
            CreateDealloc(Builder2, firstallocation);
          }
        } else
          assert(0 && "illegal mpi");

        CreateDealloc(Builder2, helper);
      }
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        assert(!gutils->isConstantValue(call.getOperand(0)));
        assert(!gutils->isConstantValue(call.getOperand(6)));

        createMPIForwardCall(call, {/*buf*/ 0, /*request*/ 6}, gutils,
                             Builder2);
        return;
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Wait" || funcName == "PMPI_Wait") {
    Value *d_reqp = nullptr;
    auto impi = getMPIHelper(call.getContext());
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      Value *req = gutils->getNewFromOriginal(call.getOperand(0));
      Value *d_req = gutils->invertPointerM(call.getOperand(0), BuilderZ);

      if (req->getType()->isIntegerTy()) {
        req = BuilderZ.CreateIntToPtr(
            req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      Value *isNull = nullptr;
      if (auto GV = gutils->newFunc->getParent()->getNamedValue(
              "ompi_request_null")) {
        Value *reql = BuilderZ.CreatePointerCast(req, getUnqual(GV->getType()));
        reql = BuilderZ.CreateLoad(GV->getType(), reql);
        isNull = BuilderZ.CreateICmpEQ(reql, GV);
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      d_reqp = BuilderZ.CreateLoad(
          getUnqual(impi),
          BuilderZ.CreatePointerCast(d_req, getUnqual(getUnqual(impi))));
      if (isNull)
        d_reqp =
            CreateSelect(BuilderZ, isNull,
                         Constant::getNullValue(d_reqp->getType()), d_reqp);
      if (auto I = dyn_cast<Instruction>(d_reqp))
        gutils->TapesToPreventRecomputation.insert(I);
      d_reqp = gutils->cacheForReverse(
          BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(0)));
      Value *req =
          lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);

      if (Mode != DerivativeMode::ReverseModeCombined) {
        d_reqp = BuilderZ.CreatePHI(getUnqual(impi), 0);
        d_reqp = gutils->cacheForReverse(
            BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
      } else
        assert(d_reqp);
      d_reqp = lookup(d_reqp, Builder2);

      Value *isNull = Builder2.CreateICmpEQ(
          d_reqp, Constant::getNullValue(d_reqp->getType()));

      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *nonnullBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_nonnull");
      BasicBlock *endBlock = gutils->addReverseBlock(
          nonnullBlock, currentBlock->getName() + "_end",
          /*fork*/ true, /*push*/ false);

      Builder2.CreateCondBr(isNull, endBlock, nonnullBlock);
      Builder2.SetInsertPoint(nonnullBlock);

      Value *cache = Builder2.CreateLoad(impi, d_reqp);

      Value *args[] = {
          getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
          req};
      Type *types[sizeof(args) / sizeof(*args) - 1];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
        types[i] = args[i]->getType();
      Function *dwait = getOrInsertDifferentialMPI_Wait(
          *called->getParent(), types, call.getOperand(0)->getType(),
          called->getName());

      // Need to preserve the shadow Request (operand 0 in wait).
      // However, this doesn't end up preserving
      // the underlying buffers for the adjoint. To rememdy, force inline.
      auto cal =
          Builder2.CreateCall(dwait, args,
                              gutils->getInvertedBundles(
                                  &call, {ValueType::Shadow, ValueType::None},
                                  Builder2, /*lookup*/ true));
      cal->setCallingConv(dwait->getCallingConv());
      cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
      cal->addFnAttr(Attribute::AlwaysInline);
      Builder2.CreateBr(endBlock);
      {
        auto found = gutils->reverseBlockToPrimal.find(endBlock);
        assert(found != gutils->reverseBlockToPrimal.end());
        SmallVector<BasicBlock *, 4> &vec =
            gutils->reverseBlocks[found->second];
        assert(vec.size());
        vec.push_back(endBlock);
      }
      Builder2.SetInsertPoint(endBlock);
    } else if (Mode == DerivativeMode::ForwardMode ||
               Mode == DerivativeMode::ForwardModeError) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(0)));

      createMPIForwardCall(call, {/*request*/ 0, /*status*/ 1}, gutils,
                           Builder2);
      return;
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Waitall" || funcName == "PMPI_Waitall") {
    Value *d_reqp = nullptr;
    auto impi = getMPIHelper(call.getContext());
    PointerType *reqType = getUnqual(impi);
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      Value *count = gutils->getNewFromOriginal(call.getOperand(0));
      Value *req = gutils->getNewFromOriginal(call.getOperand(1));
      Value *d_req = gutils->invertPointerM(call.getOperand(1), BuilderZ);

      if (req->getType()->isIntegerTy()) {
        req = BuilderZ.CreateIntToPtr(
            req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      if (d_req->getType()->isIntegerTy()) {
        d_req = BuilderZ.CreateIntToPtr(
            d_req, getUnqual(getInt8PtrTy(call.getContext())));
      }

      Function *dsave = getOrInsertDifferentialWaitallSave(
          *gutils->oldFunc->getParent(),
          {count->getType(), req->getType(), d_req->getType()}, reqType);

      d_reqp = BuilderZ.CreateCall(dsave, {count, req, d_req});
      cast<CallInst>(d_reqp)->setCallingConv(dsave->getCallingConv());
      cast<CallInst>(d_reqp)->setDebugLoc(
          gutils->getNewFromOriginal(call.getDebugLoc()));
      d_reqp = gutils->cacheForReverse(
          BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(1)));
      Value *count =
          lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2);
      Value *req_orig =
          lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2);

      if (Mode != DerivativeMode::ReverseModeCombined) {
        d_reqp = BuilderZ.CreatePHI(getUnqual(reqType), 0);
        d_reqp = gutils->cacheForReverse(
            BuilderZ, d_reqp, getIndex(&call, CacheType::Tape, BuilderZ));
      }

      d_reqp = lookup(d_reqp, Builder2);

      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *loopBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_loop");
      BasicBlock *nonnullBlock = gutils->addReverseBlock(
          loopBlock, currentBlock->getName() + "_nonnull");
      BasicBlock *eloopBlock = gutils->addReverseBlock(
          nonnullBlock, currentBlock->getName() + "_eloop");
      BasicBlock *endBlock =
          gutils->addReverseBlock(eloopBlock, currentBlock->getName() + "_end",
                                  /*fork*/ true, /*push*/ false);

      Builder2.CreateCondBr(
          Builder2.CreateICmpNE(count,
                                ConstantInt::get(count->getType(), 0, false)),
          loopBlock, endBlock);

      Builder2.SetInsertPoint(loopBlock);
      auto idx = Builder2.CreatePHI(count->getType(), 2);
      idx->addIncoming(ConstantInt::get(count->getType(), 0, false),
                       currentBlock);
      Value *inc = Builder2.CreateAdd(
          idx, ConstantInt::get(count->getType(), 1, false), "", true, true);
      idx->addIncoming(inc, eloopBlock);

      Value *idxs[] = {idx};
      Value *req = Builder2.CreateInBoundsGEP(reqType, req_orig, idxs);
      Value *d_req = Builder2.CreateInBoundsGEP(reqType, d_reqp, idxs);

      d_req = Builder2.CreateLoad(
          getUnqual(impi),
          Builder2.CreatePointerCast(d_req, getUnqual(getUnqual(impi))));

      Value *isNull = Builder2.CreateICmpEQ(
          d_req, Constant::getNullValue(d_req->getType()));

      Builder2.CreateCondBr(isNull, eloopBlock, nonnullBlock);
      Builder2.SetInsertPoint(nonnullBlock);

      Value *cache = Builder2.CreateLoad(impi, d_req);

      Value *args[] = {
          getMPIMemberPtr<MPI_Elem::Buf, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Count, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::DataType, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Src, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Tag, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Comm, false>(Builder2, cache, impi),
          getMPIMemberPtr<MPI_Elem::Call, false>(Builder2, cache, impi),
          req};
      Type *types[sizeof(args) / sizeof(*args) - 1];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args) - 1; i++)
        types[i] = args[i]->getType();
      Function *dwait = getOrInsertDifferentialMPI_Wait(
          *called->getParent(), types, req->getType(), called->getName());
      // Need to preserve the shadow Request (operand 6 in isend/irecv), which
      // becomes operand 0 for iwait. However, this doesn't end up preserving
      // the underlying buffers for the adjoint. To remedy, force inline the
      // function.
      auto cal = Builder2.CreateCall(
          dwait, args,
          gutils->getInvertedBundles(&call,
                                     {ValueType::None, ValueType::None,
                                      ValueType::None, ValueType::None,
                                      ValueType::None, ValueType::None,
                                      ValueType::Shadow},
                                     Builder2, /*lookup*/ true));
      cal->setCallingConv(dwait->getCallingConv());
      cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
      cal->addFnAttr(Attribute::AlwaysInline);
      Builder2.CreateBr(eloopBlock);

      Builder2.SetInsertPoint(eloopBlock);
      Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, count), endBlock,
                            loopBlock);
      {
        auto found = gutils->reverseBlockToPrimal.find(endBlock);
        assert(found != gutils->reverseBlockToPrimal.end());
        SmallVector<BasicBlock *, 4> &vec =
            gutils->reverseBlocks[found->second];
        assert(vec.size());
        vec.push_back(endBlock);
      }
      Builder2.SetInsertPoint(endBlock);
      if (shouldFree()) {
        CreateDealloc(Builder2, d_reqp);
      }
    } else if (Mode == DerivativeMode::ForwardMode ||
               Mode == DerivativeMode::ForwardModeError) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);

      assert(!gutils->isConstantValue(call.getOperand(1)));

      createMPIForwardCall(call, {/*array_of_requests*/ 1}, gutils, Builder2);
      return;
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Send" || funcName == "MPI_Ssend" ||
      funcName == "PMPI_Send" || funcName == "PMPI_Ssend") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);
      if (shadow->getType()->isIntegerTy())
        shadow =
            Builder2.CreateIntToPtr(shadow, getInt8PtrTy(call.getContext()));

      Type *statusType = nullptr;
#if LLVM_VERSION_MAJOR < 17
      if (called->getContext().supportsTypedPointers()) {
        if (Function *recvfn = called->getParent()->getFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Recv"))) {
          auto statusArg = recvfn->arg_end();
          statusArg--;
          if (auto PT = dyn_cast<PointerType>(statusArg->getType()))
            statusType = PT->getPointerElementType();
        }
      }
#endif
      if (statusType == nullptr) {
        statusType = ArrayType::get(Type::getInt8Ty(call.getContext()), 24);
        llvm::errs() << " warning could not automatically determine mpi "
                        "status type, assuming [24 x i8]\n";
      }

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *src = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        src = lookup(src, Builder2);

      Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        tag = lookup(tag, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buf*/ 0}, gutils, Builder2);
        return;
      }

      Value *args[] = {
          /*buf*/ NULL,
          /*count*/ count,
          /*datatype*/ datatype,
          /*src*/ src,
          /*tag*/ tag,
          /*comm*/ comm,
          /*status*/
          IRBuilder<>(gutils->inversionAllocs).CreateAlloca(statusType)};

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      auto len_arg = Builder2.CreateZExtOrTrunc(
          args[1], Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      Value *firstallocation =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpirecv_malloccache");
      args[0] = firstallocation;

      Type *types[sizeof(args) / sizeof(*args)];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
        types[i] = args[i]->getType();
      FunctionType *FT = FunctionType::get(call.getType(), types, false);

      Builder2.SetInsertPoint(Builder2.GetInsertBlock());

      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None,
           ValueType::None, ValueType::None, ValueType::None},
          Builder2, /*lookup*/ true);

      auto fcall = Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Recv"), FT),
          args);
      fcall->setCallingConv(call.getCallingConv());

      DifferentiableMemCopyFloats(call, call.getOperand(0), firstallocation,
                                  shadow, len_arg, Builder2, BufferDefs);

      if (shouldFree()) {
        CreateDealloc(Builder2, firstallocation);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  if (funcName == "MPI_Recv" || funcName == "PMPI_Recv") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *source = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        source = lookup(source, Builder2);

      Value *tag = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        tag = lookup(tag, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(5));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buf*/ 0}, gutils, Builder2);
        return;
      }

      Value *args[] = {shadow, count, datatype, source, tag, comm};

      auto Defs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal, ValueType::Primal,
           ValueType::None},
          Builder2, /*lookup*/ !forwardMode);

      Type *types[sizeof(args) / sizeof(*args)];
      for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
        types[i] = args[i]->getType();
      FunctionType *FT = FunctionType::get(call.getType(), types, false);

      auto fcall = Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Send"), FT),
          args, Defs);
      fcall->setCallingConv(call.getCallingConv());

      auto dst_arg =
          Builder2.CreateBitCast(args[0], getInt8PtrTy(call.getContext()));
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto len_arg = Builder2.CreateZExtOrTrunc(
          args[1], Type::getInt64Ty(call.getContext()));
      auto tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());

      Value *nargs[] = {dst_arg, val_arg, len_arg, volatile_arg};
      Type *tys[] = {dst_arg->getType(), len_arg->getType()};

      auto MemsetDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::None, ValueType::None, ValueType::None,
           ValueType::None, ValueType::None, ValueType::None},
          Builder2, /*lookup*/ true);
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          nargs));
      memset->addParamAttr(0, Attribute::NonNull);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int root,
  //           MPI_Comm comm )
  // 1. if root, malloc intermediate buffer
  // 2. reduce sum diff(buffer) into intermediate
  // 3. if root, set shadow(buffer) = intermediate [memcpy] then free
  // 3-e. else, set shadow(buffer) = 0 [memset]
  if (funcName == "MPI_Bcast" || funcName == "PMPI_Bcast") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *shadow = gutils->invertPointerM(call.getOperand(0), Builder2);
      if (!forwardMode)
        shadow = lookup(shadow, Builder2);
      if (shadow->getType()->isIntegerTy())
        shadow =
            Builder2.CreateIntToPtr(shadow, getInt8PtrTy(call.getContext()));

      ConcreteType CT =
          TR.firstPointer(1, call.getOperand(0), &call, gutils, &Builder2);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = getUnqual(MPI_OP_type);

      Value *count = gutils->getNewFromOriginal(call.getOperand(1));
      if (!forwardMode)
        count = lookup(count, Builder2);
      Value *datatype = gutils->getNewFromOriginal(call.getOperand(2));
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);
      Value *root = gutils->getNewFromOriginal(call.getOperand(3));
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(call.getOperand(4));
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      if (forwardMode) {
        createMPIForwardCall(call, {/*buffer*/ 0}, gutils, Builder2);
        return;
      }

      Value *rank = MPI_COMM_RANK(comm, Builder2, root->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      auto len_arg = Builder2.CreateZExtOrTrunc(
          count, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. if root, malloc intermediate buffer, else undef
      PHINode *buf;

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        Value *rootbuf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             len_arg, "mpireduce_malloccache");
        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);

        buf = Builder2.CreatePHI(rootbuf->getType(), 2);
        buf->addIncoming(rootbuf, rootBlock);
        buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);
      }

      // Need to preserve the shadow buffer.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal, ValueType::Primal},
          Builder2, /*lookup*/ true);

      // 2. reduce sum diff(buffer) into intermediate
      {
        // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
        // MPI_Datatype datatype,
        //     MPI_Op op, int root, MPI_Comm comm)
        Value *args[] = {
            /*sendbuf*/ shadow,
            /*recvbuf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op (MPI_SUM)*/
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(), called,
                                  MPI_OP_Ptr_type, MPI_OP_type, CT,
                                  root->getType(), Builder2),
            /*int root*/ root,
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);

        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Reduce"), FT),
            args, BufferDefs);
      }

      // 3. if root, set shadow(buffer) = intermediate [memcpy]
      BasicBlock *currentBlock = Builder2.GetInsertBlock();
      BasicBlock *rootBlock = gutils->addReverseBlock(
          currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
      BasicBlock *nonrootBlock = gutils->addReverseBlock(
          rootBlock, currentBlock->getName() + "_nonroot", gutils->newFunc);
      BasicBlock *mergeBlock = gutils->addReverseBlock(
          nonrootBlock, currentBlock->getName() + "_post", gutils->newFunc);

      Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, root), rootBlock,
                            nonrootBlock);

      Builder2.SetInsertPoint(rootBlock);

      {
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *nargs[] = {shadow, buf, len_arg, volatile_arg};

        Type *tys[] = {shadow->getType(), buf->getType(), len_arg->getType()};

        auto memcpyF = getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                               Intrinsic::memcpy, tys);

        auto mem =
            cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
        mem->setCallingConv(memcpyF->getCallingConv());

        // Free up the memory of the buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }
      }

      Builder2.CreateBr(mergeBlock);

      Builder2.SetInsertPoint(nonrootBlock);

      // 3-e. else, set shadow(buffer) = 0 [memset]
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());
      Value *args[] = {shadow, val_arg, len_arg, volatile_arg};
      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          args, BufferDefs));
      memset->addParamAttr(0, Attribute::NonNull);
      Builder2.CreateBr(mergeBlock);

      Builder2.SetInsertPoint(mergeBlock);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 1.5 if root, set intermediate = diff(recvbuffer)
  // 2. MPI_Bcast intermediate to all
  // 3. if root, Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
  // MPI_Datatype datatype,
  //                      MPI_Op op, int root, MPI_Comm comm)

  llvm::StringRef canonMPIName = canonicalizeMPIName(funcName);

  if (canonMPIName == "MPI_Reduce") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      // TODO insert a check for sum

      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      // Get the operations from MPI_Reduce
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_root = call.getOperand(5);
      Value *orig_comm = call.getOperand(6);

      // The Fortran MPI ABI ("mpi_reduce_", "mpi_reduce__", ...) passes all
      // arguments by reference and appends an `ierr` argument.
      bool fortranABI = isFortranMPICall(called->getName());

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          } else if (fortranABI && GV->isConstant() &&
                     GV->hasDefinitiveInitializer()) {
            // The Fortran ABI passes the operator as a reference to an
            // integer handle
            if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer())) {
              // MPICH native ABI (also covers the MPICH ABI Compatibility
              // Initiative: Intel MPI, MVAPICH, Cray MPICH)
              if (CI->getValue() == 1476395011) {
                isSum = true;
              }
              // MPI 5.0 standard ABI (Chapter 20), where predefined op
              // handles are fixed compile-time constants, identical in C
              // and Fortran: MPI_SUM == 33.
              if (CI->getValue() == 33) {
                isSum = true;
              }
            }
          }
        }
        // MPICH native ABI
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
        // MPI 5.0 standard ABI: predefined op handles are small-integer
        // pointer constants (inttoptr), with MPI_SUM == 33.
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 33) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        if (fortranABI) {
          // Integer MPI operator handles of a native Fortran ABI are
          // implementation-defined and cannot be mapped portably at compile
          // time; warn and assume the common case of MPI_SUM. This remains
          // true with MPI 5.0: although its standard ABI (Chapter 20) does
          // fix handle values as portable compile-time constants (MPI_SUM
          // == 33, recognized above), supporting that ABI is optional and
          // neither Open MPI nor MPICH use it by default, so code compiled
          // against the default/native ABIs still carries
          // implementation-defined handle values.
          llvm::errs() << "warning: cannot determine MPI op used in `" << call
                       << "`, assuming MPI_SUM\n";
        } else {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_reduce op: " << *orig_op << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
          return;
        }
      }

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so build the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[1] = ValueType::Shadow;
      auto BufferDefs =
          gutils->getInvertedBundles(&call, BufferBundleTypes, Builder2,
                                     /*lookup*/ !forwardMode);

      Value *count = gutils->getNewFromOriginal(orig_count);
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(orig_datatype);
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *op = gutils->getNewFromOriginal(orig_op);
      if (!forwardMode)
        op = lookup(op, Builder2);

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      // The Fortran ABI passes the count and root arguments by reference;
      // load them where their value is needed.
      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *countVal = fortranABI ? Builder2.CreateLoad(i32Ty, count) : count;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 1}, gutils,
                             Builder2);
        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          countVal, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpireduce_malloccache");

      // 1.5 if root, set intermediate = diff(recvbuffer)
      {

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        {
          auto volatile_arg = ConstantInt::getFalse(call.getContext());
          Value *nargs[] = {buf, shadow_recvbuf, len_arg, volatile_arg};

          Type *tys[] = {nargs[0]->getType(), nargs[1]->getType(),
                         len_arg->getType()};

          auto memcpyF = getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                                 Intrinsic::memcpy, tys);

          auto mem =
              cast<CallInst>(Builder2.CreateCall(memcpyF, nargs, BufferDefs));
          mem->setCallingConv(memcpyF->getCallingConv());
        }

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 2. MPI_Bcast intermediate to all
      {
        // int MPI_Bcast( void *buffer, int count, MPI_Datatype datatype, int
        // root,
        //     MPI_Comm comm )
        // The Fortran ABI passes count/datatype/root/comm by reference (as
        // held here) and appends an `ierr` argument.
        SmallVector<Value *, 8> args = {
            /*buf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*int root*/ root,
            /*comm*/ comm,
        };
        if (fortranABI) {
          args.push_back(IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(i32Ty, nullptr, "enzyme_mpi_ierr"));
        }
        SmallVector<Type *, 8> types;
        for (auto *arg : args) {
          types.push_back(arg->getType());
        }

        FunctionType *FT = FunctionType::get(
            fortranABI ? Type::getVoidTy(call.getContext()) : call.getType(),
            types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Bcast"), FT),
            args, BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  len_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffers
  // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
  //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)

  if (canonMPIName == "MPI_Allreduce") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      // TODO insert a check for sum

      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      // Get the operations from MPI_Allreduce
      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_recvbuf = call.getOperand(1);
      Value *orig_count = call.getOperand(2);
      Value *orig_datatype = call.getOperand(3);
      Value *orig_op = call.getOperand(4);
      Value *orig_comm = call.getOperand(5);

      // The Fortran MPI ABI ("mpi_allreduce_", "mpi_allreduce__", ...) passes
      // all arguments by reference and appends an `ierr` argument.
      bool fortranABI = isFortranMPICall(called->getName());

      bool isSum = false;
      if (Constant *C = dyn_cast<Constant>(orig_op)) {
        while (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
          C = CE->getOperand(0);
        }
        if (auto GV = dyn_cast<GlobalVariable>(C)) {
          if (GV->getName() == "ompi_mpi_op_sum") {
            isSum = true;
          } else if (fortranABI && GV->isConstant() &&
                     GV->hasDefinitiveInitializer()) {
            // The Fortran ABI passes the operator as a reference to an
            // integer handle
            if (auto *CI = dyn_cast<ConstantInt>(GV->getInitializer())) {
              // MPICH native ABI (also covers the MPICH ABI Compatibility
              // Initiative: Intel MPI, MVAPICH, Cray MPICH)
              if (CI->getValue() == 1476395011) {
                isSum = true;
              }
              // MPI 5.0 standard ABI (Chapter 20), where predefined op
              // handles are fixed compile-time constants, identical in C
              // and Fortran: MPI_SUM == 33.
              if (CI->getValue() == 33) {
                isSum = true;
              }
            }
          }
        }
        // MPICH native ABI
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 1476395011) {
            isSum = true;
          }
        }
        // MPI 5.0 standard ABI: predefined op handles are small-integer
        // pointer constants (inttoptr), with MPI_SUM == 33.
        if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
          if (CI->getValue() == 33) {
            isSum = true;
          }
        }
      }
      if (!isSum) {
        if (fortranABI) {
          // Integer MPI operator handles of a native Fortran ABI are
          // implementation-defined and cannot be mapped portably at compile
          // time; warn and assume the common case of MPI_SUM. This remains
          // true with MPI 5.0: although its standard ABI (Chapter 20) does
          // fix handle values as portable compile-time constants (MPI_SUM
          // == 33, recognized above), supporting that ABI is optional and
          // neither Open MPI nor MPICH use it by default, so code compiled
          // against the default/native ABIs still carries
          // implementation-defined handle values.
          llvm::errs() << "warning: cannot determine MPI op used in `" << call
                       << "`, assuming MPI_SUM\n";
        } else {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << " call: " << call << "\n";
          ss << " unhandled mpi_allreduce op: " << *orig_op << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
          return;
        }
      }

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so build the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[1] = ValueType::Shadow;
      auto BufferDefs =
          gutils->getInvertedBundles(&call, BufferBundleTypes, Builder2,
                                     /*lookup*/ !forwardMode);

      Value *count = gutils->getNewFromOriginal(orig_count);
      if (!forwardMode)
        count = lookup(count, Builder2);

      Value *datatype = gutils->getNewFromOriginal(orig_datatype);
      if (!forwardMode)
        datatype = lookup(datatype, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Value *op = gutils->getNewFromOriginal(orig_op);
      if (!forwardMode)
        op = lookup(op, Builder2);

      // The Fortran ABI passes the count argument by reference; load it
      // where its value is needed.
      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *countVal = fortranABI ? Builder2.CreateLoad(i32Ty, count) : count;

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 1}, gutils,
                             Builder2);
        return;
      }

      Value *tysize = MPI_TYPE_SIZE(datatype, Builder2, call.getType(), called);

      // Get the length for the allocation of the intermediate buffer
      auto len_arg = Builder2.CreateZExtOrTrunc(
          countVal, Type::getInt64Ty(call.getContext()));
      len_arg =
          Builder2.CreateMul(len_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           len_arg, "mpireduce_malloccache");

      // 2. MPI_Allreduce (sum) of diff(recvbuffer) to intermediate
      {
        // int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
        //              MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
        // The Fortran ABI passes count/datatype/op/comm by reference (as
        // held here) and appends an `ierr` argument.
        SmallVector<Value *, 8> args = {
            /*sendbuf*/ shadow_recvbuf,
            /*recvbuf*/ buf,
            /*count*/ count,
            /*datatype*/ datatype,
            /*op*/ op,
            /*comm*/ comm,
        };
        if (fortranABI) {
          args.push_back(IRBuilder<>(gutils->inversionAllocs)
                             .CreateAlloca(i32Ty, nullptr, "enzyme_mpi_ierr"));
        }
        SmallVector<Type *, 8> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(
            fortranABI ? Type::getVoidTy(call.getContext()) : call.getType(),
            types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Allreduce"),
                FT),
            args, BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
      auto volatile_arg = ConstantInt::getFalse(call.getContext());
      Value *args[] = {shadow_recvbuf, val_arg, len_arg, volatile_arg};
      Type *tys[] = {args[0]->getType(), args[2]->getType()};
      auto memset = cast<CallInst>(Builder2.CreateCall(
          getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                  Intrinsic::memset, tys),
          args, BufferDefs));
      memset->addParamAttr(0, Attribute::NonNull);

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  len_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 2. Scatter diff(recvbuffer) to intermediate buffer
  // 3. if root, Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
  //           int root, MPI_Comm comm)

  if (funcName == "MPI_Gather" || funcName == "PMPI_Gather") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_root = call.getOperand(6);
      Value *orig_comm = call.getOperand(7);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!sendcount)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      bool fortranABI = isFortranMPICall(called->getName());

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *sendcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, sendcount) : sendcount;
      Value *recvcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, recvcount) : recvcount;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }

      // Get the length for the allocation of the intermediate buffer
      auto sendlen_arg = Builder2.CreateZExtOrTrunc(
          sendcountVal, Type::getInt64Ty(call.getContext()));
      sendlen_arg =
          Builder2.CreateMul(sendlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so size the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[3] = ValueType::Shadow;
      auto BufferDefs = gutils->getInvertedBundles(&call, BufferBundleTypes,
                                                   Builder2, /*lookup*/ true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           sendlen_arg, "mpireduce_malloccache");

      // 2. Scatter diff(recvbuffer) to intermediate buffer
      {
        // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
        // sendtype,
        //     void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
        //     MPI_Comm comm)
        //
        // The Fortran MPI ABI passes all arguments by reference and appends
        // an `ierr` argument, so the generated call must match the convention
        // of the caller.
        SmallVector<Value *, 10> args;
        if (fortranABI) {
          args = {shadow_recvbuf, recvcount, recvtype, buf,
                  sendcount,      sendtype,  root,     comm};
          for (size_t i = 8, e = call.arg_size(); i < e; i++)
            args.push_back(lookup(
                gutils->getNewFromOriginal(call.getArgOperand(i)), Builder2));
        } else {
          args = {shadow_recvbuf, recvcountVal, recvtype, buf,
                  sendcountVal,   sendtype,     rootVal,  comm};
        }
        SmallVector<Type *, 10> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Scatter"), FT),
            args, BufferDefs);
      }

      // 3. if root, Zero diff(recvbuffer) [memset to 0]
      {

        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcountVal, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        recvlen_arg = Builder2.CreateMul(
            recvlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, rootVal->getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);

        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  sendlen_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. if root, malloc intermediate buffer, else undef
  // 2. Gather diff(recvbuffer) to intermediate buffer
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. if root, free intermediate buffer

  // int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype
  // sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
  //           MPI_Comm comm)
  if (canonMPIName == "MPI_Scatter") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_root = call.getOperand(6);
      Value *orig_comm = call.getOperand(7);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);
      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);
      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!forwardMode)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      Value *root = gutils->getNewFromOriginal(orig_root);
      if (!forwardMode)
        root = lookup(root, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      bool fortranABI = isFortranMPICall(called->getName());

      Type *i32Ty = Type::getInt32Ty(call.getContext());
      Value *sendcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, sendcount) : sendcount;
      Value *recvcountVal =
          fortranABI ? Builder2.CreateLoad(i32Ty, recvcount) : recvcount;
      Value *rootVal = fortranABI ? Builder2.CreateLoad(i32Ty, root) : root;

      Value *rank = MPI_COMM_RANK(comm, Builder2, rootVal->getType(), called);
      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }
      // Get the length for the allocation of the intermediate buffer
      auto recvlen_arg = Builder2.CreateZExtOrTrunc(
          recvcountVal, Type::getInt64Ty(call.getContext()));
      recvlen_arg =
          Builder2.CreateMul(recvlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers. The Fortran ABI call
      // has an extra `ierr` argument, so size the bundle to match the actual
      // call arity: shadow send/recv buffers, primal everything else.
      std::vector<ValueType> BufferBundleTypes(call.arg_size(),
                                               ValueType::Primal);
      BufferBundleTypes[0] = ValueType::Shadow;
      BufferBundleTypes[3] = ValueType::Shadow;
      auto BufferDefs = gutils->getInvertedBundles(&call, BufferBundleTypes,
                                                   Builder2, /*lookup*/ true);

      // 1. if root, malloc intermediate buffer, else undef
      PHINode *buf;
      PHINode *sendlen_phi;

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        auto sendlen_arg = Builder2.CreateZExtOrTrunc(
            sendcountVal, Type::getInt64Ty(call.getContext()));
        sendlen_arg =
            Builder2.CreateMul(sendlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        sendlen_arg = Builder2.CreateMul(
            sendlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, rootVal->getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);

        Value *rootbuf =
            CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                             sendlen_arg, "mpireduce_malloccache");

        Builder2.CreateBr(mergeBlock);

        Builder2.SetInsertPoint(mergeBlock);

        buf = Builder2.CreatePHI(rootbuf->getType(), 2);
        buf->addIncoming(rootbuf, rootBlock);
        buf->addIncoming(UndefValue::get(buf->getType()), currentBlock);

        sendlen_phi = Builder2.CreatePHI(sendlen_arg->getType(), 2);
        sendlen_phi->addIncoming(sendlen_arg, rootBlock);
        sendlen_phi->addIncoming(UndefValue::get(sendlen_arg->getType()),
                                 currentBlock);
      }

      // 2. Gather diff(recvbuffer) to intermediate buffer
      {
        // int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype
        // sendtype,
        //     void *recvbuf, int recvcount, MPI_Datatype recvtype,
        //     int root, MPI_Comm comm)
        //
        // The Fortran MPI ABI passes all arguments by reference and appends
        // an `ierr` argument, so the generated call must match the convention
        // of the caller.
        SmallVector<Value *, 10> args;
        if (fortranABI) {
          args = {shadow_recvbuf, recvcount, recvtype, buf,
                  sendcount,      sendtype,  root,     comm};
          for (size_t i = 8, e = call.arg_size(); i < e; i++)
            args.push_back(lookup(
                gutils->getNewFromOriginal(call.getArgOperand(i)), Builder2));
        } else {
          args = {shadow_recvbuf, recvcountVal, recvtype, buf,
                  sendcountVal,   sendtype,     rootVal,  comm};
        }
        SmallVector<Type *, 10> types;
        for (auto *arg : args)
          types.push_back(arg->getType());

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(), "MPI_Gather"), FT),
            args, BufferDefs);
      }

      // 3. Zero diff(recvbuffer) [memset to 0]
      {
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
      }

      // 4. if root, diff(sendbuffer) += intermediate buffer (diffmemcopy)
      // 5. if root, free intermediate buffer

      {
        BasicBlock *currentBlock = Builder2.GetInsertBlock();
        BasicBlock *rootBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_root", gutils->newFunc);
        BasicBlock *mergeBlock = gutils->addReverseBlock(
            rootBlock, currentBlock->getName() + "_post", gutils->newFunc);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(rank, rootVal), rootBlock,
                              mergeBlock);

        Builder2.SetInsertPoint(rootBlock);

        // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
        DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                    sendlen_phi, Builder2, BufferDefs);

        // Free up intermediate buffer
        if (shouldFree()) {
          CreateDealloc(Builder2, buf);
        }

        Builder2.CreateBr(mergeBlock);
        Builder2.SetInsertPoint(mergeBlock);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Approximate algo (for sum):  -> if statement yet to be
  // 1. malloc intermediate buffer
  // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
  // intermediate buffer
  // 3. Zero diff(recvbuffer) [memset to 0]
  // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
  // 5. free intermediate buffer

  // int MPI_Allgather(const void *sendbuf, int sendcount, MPI_Datatype
  // sendtype,
  //           void *recvbuf, int recvcount, MPI_Datatype recvtype,
  //           MPI_Comm comm)

  if (funcName == "MPI_Allgather" || funcName == "PMPI_Allgather") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      bool forwardMode = Mode == DerivativeMode::ForwardMode ||
                         Mode == DerivativeMode::ForwardModeError;

      IRBuilder<> Builder2 =
          forwardMode ? IRBuilder<>(&call) : IRBuilder<>(call.getParent());
      if (forwardMode) {
        getForwardBuilder(Builder2);
      } else {
        getReverseBuilder(Builder2);
      }

      Value *orig_sendbuf = call.getOperand(0);
      Value *orig_sendcount = call.getOperand(1);
      Value *orig_sendtype = call.getOperand(2);
      Value *orig_recvbuf = call.getOperand(3);
      Value *orig_recvcount = call.getOperand(4);
      Value *orig_recvtype = call.getOperand(5);
      Value *orig_comm = call.getOperand(6);

      Value *shadow_recvbuf = gutils->invertPointerM(orig_recvbuf, Builder2);
      if (!forwardMode)
        shadow_recvbuf = lookup(shadow_recvbuf, Builder2);

      if (shadow_recvbuf->getType()->isIntegerTy())
        shadow_recvbuf = Builder2.CreateIntToPtr(
            shadow_recvbuf, getInt8PtrTy(call.getContext()));

      Value *shadow_sendbuf = gutils->invertPointerM(orig_sendbuf, Builder2);
      if (!forwardMode)
        shadow_sendbuf = lookup(shadow_sendbuf, Builder2);

      if (shadow_sendbuf->getType()->isIntegerTy())
        shadow_sendbuf = Builder2.CreateIntToPtr(
            shadow_sendbuf, getInt8PtrTy(call.getContext()));

      Value *recvcount = gutils->getNewFromOriginal(orig_recvcount);
      if (!forwardMode)
        recvcount = lookup(recvcount, Builder2);

      Value *recvtype = gutils->getNewFromOriginal(orig_recvtype);
      if (!forwardMode)
        recvtype = lookup(recvtype, Builder2);

      Value *sendcount = gutils->getNewFromOriginal(orig_sendcount);
      if (!forwardMode)
        sendcount = lookup(sendcount, Builder2);

      Value *sendtype = gutils->getNewFromOriginal(orig_sendtype);
      if (!forwardMode)
        sendtype = lookup(sendtype, Builder2);

      Value *comm = gutils->getNewFromOriginal(orig_comm);
      if (!forwardMode)
        comm = lookup(comm, Builder2);

      Value *tysize = MPI_TYPE_SIZE(sendtype, Builder2, call.getType(), called);

      if (forwardMode) {
        createMPIForwardCall(call, {/*sendbuf*/ 0, /*recvbuf*/ 3}, gutils,
                             Builder2);
        return;
      }
      // Get the length for the allocation of the intermediate buffer
      auto sendlen_arg = Builder2.CreateZExtOrTrunc(
          sendcount, Type::getInt64Ty(call.getContext()));
      sendlen_arg =
          Builder2.CreateMul(sendlen_arg,
                             Builder2.CreateZExtOrTrunc(
                                 tysize, Type::getInt64Ty(call.getContext())),
                             "", true, true);

      // Need to preserve the shadow send/recv buffers.
      auto BufferDefs = gutils->getInvertedBundles(
          &call,
          {ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Shadow, ValueType::Primal, ValueType::Primal,
           ValueType::Primal},
          Builder2, /*lookup*/ true);

      // 1. Alloc intermediate buffer
      Value *buf =
          CreateAllocation(Builder2, Type::getInt8Ty(call.getContext()),
                           sendlen_arg, "mpireduce_malloccache");

      ConcreteType CT =
          TR.firstPointer(1, orig_sendbuf, &call, gutils, &Builder2);
      auto MPI_OP_type = getInt8PtrTy(call.getContext());
      Type *MPI_OP_Ptr_type = getUnqual(MPI_OP_type);

      // 2. reduce diff(recvbuffer) then scatter to corresponding input node's
      // intermediate buffer
      {
        // int MPI_Reduce_scatter_block(const void* send_buffer,
        //                    void* receive_buffer,
        //                    int count,
        //                    MPI_Datatype datatype,
        //                    MPI_Op operation,
        //                    MPI_Comm communicator);
        Value *args[] = {
            /*sendbuf*/ shadow_recvbuf,
            /*recvbuf*/ buf,
            /*recvcount*/ sendcount,
            /*recvtype*/ sendtype,
            /*op (MPI_SUM)*/
            getOrInsertOpFloatSum(*gutils->newFunc->getParent(), called,
                                  MPI_OP_Ptr_type, MPI_OP_type, CT,
                                  call.getType(), Builder2),
            /*comm*/ comm,
        };
        Type *types[sizeof(args) / sizeof(*args)];
        for (size_t i = 0; i < sizeof(args) / sizeof(*args); i++)
          types[i] = args[i]->getType();

        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                getRenamedPerCallingConv(called->getName(),
                                         "MPI_Reduce_scatter_block"),
                FT),
            args, BufferDefs);
      }

      // 3. zero diff(recvbuffer) [memset to 0]
      {
        auto recvlen_arg = Builder2.CreateZExtOrTrunc(
            recvcount, Type::getInt64Ty(call.getContext()));
        recvlen_arg =
            Builder2.CreateMul(recvlen_arg,
                               Builder2.CreateZExtOrTrunc(
                                   tysize, Type::getInt64Ty(call.getContext())),
                               "", true, true);
        recvlen_arg = Builder2.CreateMul(
            recvlen_arg,
            Builder2.CreateZExtOrTrunc(
                MPI_COMM_SIZE(comm, Builder2, call.getType(), called),
                Type::getInt64Ty(call.getContext())),
            "", true, true);
        auto val_arg = ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
        auto volatile_arg = ConstantInt::getFalse(call.getContext());
        Value *args[] = {shadow_recvbuf, val_arg, recvlen_arg, volatile_arg};
        Type *tys[] = {args[0]->getType(), args[2]->getType()};
        auto memset = cast<CallInst>(Builder2.CreateCall(
            getIntrinsicDeclaration(gutils->newFunc->getParent(),
                                    Intrinsic::memset, tys),
            args, BufferDefs));
        memset->addParamAttr(0, Attribute::NonNull);
      }

      // 4. diff(sendbuffer) += intermediate buffer (diffmemcopy)
      DifferentiableMemCopyFloats(call, orig_sendbuf, buf, shadow_sendbuf,
                                  sendlen_arg, Builder2, BufferDefs);

      // Free up intermediate buffer
      if (shouldFree()) {
        CreateDealloc(Builder2, buf);
      }
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Adjoint of barrier is to place a barrier at the corresponding
  // location in the reverse.
  if (funcName == "MPI_Barrier" || funcName == "PMPI_Barrier") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      auto callval = call.getCalledOperand();
      // Copy all arguments to match the call's arity: Fortran ABI manglings
      // of MPI_Barrier (e.g. "mpi_barrier_") take an extra `ierr` argument.
      SmallVector<Value *, 4> args;
      for (unsigned i = 0; i < call.arg_size(); i++)
        args.push_back(lookup(gutils->getNewFromOriginal(call.getArgOperand(i)),
                              Builder2));
      Builder2.CreateCall(call.getFunctionType(), callval, args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Remove free's in forward pass so the comm can be used in the reverse
  // pass
  if (funcName == "MPI_Comm_free" || funcName == "MPI_Comm_disconnect") {
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // Adjoint of MPI_Comm_split / MPI_Graph_create (which allocates a comm in a
  // pointer) is to free the created comm at the corresponding place in the
  // reverse pass
  auto commFound = MPIInactiveCommAllocators.find(funcName);
  if (commFound != MPIInactiveCommAllocators.end()) {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);

      Value *args[] = {lookup(call.getOperand(commFound->second), Builder2)};
      Type *types[] = {args[0]->getType()};

      FunctionType *FT = FunctionType::get(call.getType(), types, false);
      Builder2.CreateCall(
          called->getParent()->getOrInsertFunction(
              getRenamedPerCallingConv(called->getName(), "MPI_Comm_free"), FT),
          args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  // MPI_Init / MPI_Finalize don't participate in the computation being
  // differentiated - just duplicate them as-is in both passes.
  if (funcName == "MPI_Init" || funcName == "PMPI_Init" ||
      funcName == "MPI_Init_thread" || funcName == "PMPI_Init_thread" ||
      funcName == "MPI_Finalize" || funcName == "PMPI_Finalize" ||
      funcName == "MPI_Test" || funcName == "PMPI_Test" ||
      funcName == "MPI_Probe" || funcName == "PMPI_Probe") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined ||
        Mode == DerivativeMode::ReverseModePrimal) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      SmallVector<Value *, 8> args;
      for (unsigned i = 0; i < call.arg_size(); ++i) {
        args.push_back(lookup(gutils->getNewFromOriginal(call.getArgOperand(i)),
                              Builder2));
      }
      Builder2.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                          args);
    }
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      IRBuilder<> Builder2(&call);
      getForwardBuilder(Builder2);
      SmallVector<Value *, 8> args;
      for (unsigned i = 0; i < call.arg_size(); ++i) {
        args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      }
      Builder2.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                          args);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return;
  }

  llvm::errs() << *gutils->oldFunc->getParent() << "\n";
  llvm::errs() << *gutils->oldFunc << "\n";
  llvm::errs() << call << "\n";
  llvm::errs() << called << "\n";
  llvm_unreachable("Unhandled MPI FUNCTION");
}

// Classify the op callee of a julia.atomicmodify pseudo-intrinsic call
// (op has signature elty (elty oldval, args...), with elty an integer type
// holding the bits of the modified value) as an atomicrmw-style operation
// whose derivative Enzyme knows. On success sets opArgNo to the index of the
// op parameter providing the update value and FT to the floating point type
// the update is performed in (nullptr for Xchg).
static llvm::AtomicRMWInst::BinOp
classifyAtomicModifyOp(llvm::Function *op, unsigned &opArgNo, llvm::Type *&FT) {
  using namespace llvm;
  FT = nullptr;
  if (!op || op->empty() || op->isVarArg() ||
      op->getFunctionType()->getNumParams() == 0)
    return AtomicRMWInst::BAD_BINOP;
  ReturnInst *Ret = nullptr;
  for (auto &BB : *op)
    if (auto R = dyn_cast<ReturnInst>(BB.getTerminator())) {
      if (Ret)
        return AtomicRMWInst::BAD_BINOP;
      Ret = R;
    }
  if (!Ret || !Ret->getReturnValue())
    return AtomicRMWInst::BAD_BINOP;
  auto peel = [](Value *V) {
    while (auto BC = dyn_cast<BitCastInst>(V))
      V = BC->getOperand(0);
    return V;
  };
  Value *RV = peel(Ret->getReturnValue());
  Value *Old = op->getArg(0);
  if (auto A = dyn_cast<Argument>(RV)) {
    if (A == Old)
      return AtomicRMWInst::BAD_BINOP;
    opArgNo = A->getArgNo();
    return AtomicRMWInst::Xchg;
  }
  if (auto BO = dyn_cast<BinaryOperator>(RV)) {
    Value *L = peel(BO->getOperand(0));
    Value *R = peel(BO->getOperand(1));
    FT = BO->getType();
    if (!FT->isFPOrFPVectorTy())
      return AtomicRMWInst::BAD_BINOP;
    if (BO->getOpcode() == Instruction::FAdd) {
      if (L == Old && isa<Argument>(R) && R != Old) {
        opArgNo = cast<Argument>(R)->getArgNo();
        return AtomicRMWInst::FAdd;
      }
      if (R == Old && isa<Argument>(L) && L != Old) {
        opArgNo = cast<Argument>(L)->getArgNo();
        return AtomicRMWInst::FAdd;
      }
    } else if (BO->getOpcode() == Instruction::FSub) {
      if (L == Old && isa<Argument>(R) && R != Old) {
        opArgNo = cast<Argument>(R)->getArgNo();
        return AtomicRMWInst::FSub;
      }
    }
  }
  return AtomicRMWInst::BAD_BINOP;
}

bool AdjointGenerator::handleKnownCallDerivatives(
    CallInst &call, Function *called, StringRef funcName,
    bool subsequent_calls_may_write, const std::vector<bool> &overwritten_args,
    CallInst *const newCall) {
  bool subretused = false;
  bool shadowReturnUsed = false;
  DIFFE_TYPE subretType =
      gutils->getReturnDiffeType(&call, &subretused, &shadowReturnUsed);

  IRBuilder<> BuilderZ(newCall);
  BuilderZ.setFastMathFlags(getFast());

  // Julia's atomic modify pseudo-intrinsic (introduced in Julia 1.13):
  //   {old, new} = julia.atomicmodify.iN.pAS(ptr, op, ordering, syncscope,
  //                                          args...)
  // which atomically performs old = *ptr; new = op(old, args...);
  // *ptr = new, where op's parameter i (i >= 1) is forwarded from call
  // operand i + 3. The pseudo-intrinsic must be kept intact (including in
  // generated derivative code); it is only expanded to atomicrmw/cmpxchg by
  // Julia's ExpandAtomicModify pass after GC lowering.
  if (startsWith(funcName, "julia.atomicmodify.")) {
    // Retire the primal call replayed in the pass being generated. Like the
    // generic call path, tape its result in the augmented primal and read it
    // back from the tape in the derivative pass if the latter needs the
    // primal value: the modification must never be repeated by recomputing
    // the call there.
    auto finishPrimal = [&]() {
      bool primalNeededInReverse = false;
      {
        auto found = gutils->knownRecomputeHeuristic.find(&call);
        if (found != gutils->knownRecomputeHeuristic.end())
          primalNeededInReverse = !found->second;
      }
      if (!primalNeededInReverse &&
          Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode &&
          Mode != DerivativeMode::ForwardModeError && subretused &&
          !gutils->unnecessaryIntermediates.count(&call)) {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                              ? DerivativeMode::ReverseModeGradient
                              : Mode;
        primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, minCutMode, Seen,
                                   oldUnreachable);
      }
      if (primalNeededInReverse) {
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
        eraseIfUnused(call);
      } else if (Mode == DerivativeMode::ReverseModeGradient ||
                 Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
    };

    // The pseudo-intrinsic called with the operands of the primal call,
    // except for the location and (if non-null) the operand at valIdx.
    auto emitCall = [&](Value *ptr, unsigned valIdx, Value *val) -> Value * {
      SmallVector<Value *, 6> args;
      for (size_t i = 0; i < call.arg_size(); i++) {
        if (i == 0)
          args.push_back(ptr);
        else if (i == valIdx && val)
          args.push_back(val);
        else
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      }
      auto CI = BuilderZ.CreateCall(call.getFunctionType(), called, args);
      CI->setCallingConv(call.getCallingConv());
      CI->setAttributes(call.getAttributes());
      CI->copyMetadata(*newCall);
      return CI;
    };

    // Fully inactive calls only need the primal replayed. This must be
    // handled here rather than falling through to the generic call path,
    // as the latter may decide against the constant fallback (e.g. for a
    // nocapture, non-readonly pointer argument) and then reject the
    // variadic call with `Number of arg operands != function parameters`.
    if (gutils->isConstantInstruction(&call) &&
        gutils->isConstantValue(&call)) {
      finishPrimal();
      return true;
    }

    Type *elty = cast<StructType>(call.getType())->getElementType(0);
    unsigned opArgNo = 0;
    Type *FT = nullptr;
    auto opKind = classifyAtomicModifyOp(
        dyn_cast<Function>(call.getArgOperand(1)), opArgNo, FT);
    if (opKind != AtomicRMWInst::BAD_BINOP &&
        call.arg_size() - 3 != cast<Function>(call.getArgOperand(1))
                                   ->getFunctionType()
                                   ->getNumParams())
      opKind = AtomicRMWInst::BAD_BINOP;
    // Call operand forwarded to op's update value parameter.
    unsigned valIdx = opArgNo + 3;
    Value *valOp = opKind != AtomicRMWInst::BAD_BINOP
                       ? call.getArgOperand(valIdx)
                       : nullptr;

    bool constval = gutils->isConstantValue(&call);
    bool constptr = gutils->isConstantValue(call.getArgOperand(0));

    // No shadow memory is involved; replaying the primal suffices.
    if (constval && constptr) {
      finishPrimal();
      return true;
    }

    auto &DL = gutils->newFunc->getParent()->getDataLayout();
    auto storeSize = (DL.getTypeSizeInBits(elty) + 7) / 8;
    auto vd = TR.firstPointer(storeSize, call.getArgOperand(0), &call, gutils,
                              /*errifnotfound*/ nullptr,
                              /*pointerIntSame*/ true);

    bool constargs = true;
    for (size_t i = 4; i < call.arg_size(); i++)
      if (!gutils->isConstantValue(call.getArgOperand(i))) {
        constargs = false;
        break;
      }

    // Non-differentiable (integer/pointer) data modified within duplicated
    // memory: replicate the modification on the shadow location with the
    // primal arguments (like inactive stores into active memory), keeping
    // e.g. lock states and counters of shadow objects consistent. As for an
    // inactive store, only the modified data has to be inactive, not the
    // operands op derives it from; when the type of the data is unknown, an
    // active operand however suggests floating point data, which is left
    // to the rules for recognized ops below.
    if (constval &&
        (vd.isKnown() ? !vd.isFloat() : (looseTypeAnalysis && constargs))) {
      // Which pass emits the shadow modification is decided exactly as for
      // an inactive store into duplicated memory (visitCommonStore); in
      // particular the augmented primal and the split derivative pass must
      // not both replay it, and a shadow that is only materialized in the
      // reverse pass must be updated there.
      bool forwardsShadow, backwardsShadow;
      shadowStoreSchedule(call, call.getArgOperand(0), forwardsShadow,
                          backwardsShadow);

      if ((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined &&
           (forwardsShadow || backwardsShadow)) ||
          Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        Value *dptr = gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
        applyChainRule(
            BuilderZ, [&](Value *dptr) { emitCall(dptr, 0, nullptr); }, dptr);
      }
      finishPrimal();
      return true;
    }

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError ||
        Mode == DerivativeMode::ForwardModeSplit) {
      if (opKind != AtomicRMWInst::BAD_BINOP) {
        // new = op(old, v) is linear in (old, v) for FAdd/FSub/Xchg, so the
        // same op applied to the shadow location and the shadow of v
        // computes the tangent {dold, dnew}.
        //
        // The shadow of v is its tangent (zero if inactive), except for an
        // inactive value exchanged into duplicated memory: like an inactive
        // store, that writes the primal value for pointer/integer data (and
        // zero for floating point data) into the shadow location.
        Value *dval = nullptr;
        if (!gutils->isConstantValue(valOp)) {
          dval = gutils->invertPointerM(valOp, BuilderZ);
        } else if (opKind == AtomicRMWInst::Xchg && !constptr) {
          if (!gutils->runtimeActivity && vd == BaseType::Pointer &&
              !isa<UndefValue>(valOp) && !isa<ConstantPointerNull>(valOp)) {
            std::string str;
            raw_string_ostream ss(str);
            ss << "Mismatched activity for: " << call
               << " const val: " << *valOp;
            if (CustomErrorHandler) {
              dval = unwrap(CustomErrorHandler(
                  str.c_str(), wrap(&call), ErrorType::MixedActivityError,
                  gutils, wrap(valOp), wrap(&BuilderZ)));
            } else
              EmitWarningAlways("MixedActivityError", call, ss.str(),
                                MixedActivityHint);
          }
          if (!dval)
            dval = gutils->invertPointerM(valOp, BuilderZ,
                                          TypeTree(vd).Only(-1, nullptr));
        }
        if (!dval)
          dval =
              Constant::getNullValue(gutils->getShadowType(valOp->getType()));

        auto rule = [&](Value *dptr, Value *dval) -> Value * {
          if (!dptr) {
            // Inactive location: dold = 0 and dnew = dv (-dv for FSub), in
            // the bits of the modified value.
            Value *dnew = dval;
            if (opKind == AtomicRMWInst::FSub) {
              dnew = BuilderZ.CreateBitCast(dnew, FT);
              dnew = BuilderZ.CreateFNeg(dnew);
            }
            dnew = BuilderZ.CreateBitCast(dnew, elty);
            return BuilderZ.CreateInsertValue(
                Constant::getNullValue(call.getType()), dnew, {1});
          }
          return emitCall(dptr, valIdx, dval);
        };
        Value *diff = applyChainRule(
            call.getType(), BuilderZ, rule,
            constptr ? nullptr
                     : gutils->invertPointerM(call.getArgOperand(0), BuilderZ),
            dval);
        if (!constval)
          setDiffe(&call, diff, BuilderZ);
        finishPrimal();
        return true;
      }
    } else if (Mode == DerivativeMode::ReverseModePrimal) {
      // Nothing to do in the primal pass beyond replaying the call. Besides
      // the inactive case this also covers the augmented primal paired with
      // a ForwardModeSplit derivative, which handles any recognized op.
      if (constval || opKind != AtomicRMWInst::BAD_BINOP) {
        if (!constval)
          resolveShadowPlaceholder(call);
        finishPrimal();
        return true;
      }
    } else if ((Mode == DerivativeMode::ReverseModeCombined ||
                Mode == DerivativeMode::ReverseModeGradient) &&
               constval &&
               (opKind == AtomicRMWInst::FAdd ||
                opKind == AtomicRMWInst::FSub)) {
      if (!gutils->isConstantValue(valOp) && !constptr) {
        auto order = static_cast<AtomicOrdering>(
            cast<ConstantInt>(call.getArgOperand(2))->getZExtValue());
        auto ssid = static_cast<SyncScope::ID>(
            cast<ConstantInt>(call.getArgOperand(3))->getZExtValue());
        auto align = call.getParamAlign(0).value_or(DL.getABITypeAlign(elty));
        addAtomicRMWValueAdjoint(call, call.getArgOperand(0), valOp, elty, FT,
                                 /*negate*/ opKind == AtomicRMWInst::FSub,
                                 align, order, ssid, /*isVolatile*/ false);
      }
      finishPrimal();
      return true;
    }

    // Remaining cases (active result in reverse mode, or an op that is not
    // a recognized linear modification) are not supported.
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << *gutils->oldFunc << "\n" << call << "\n";
    ss << " Active atomic modify not yet handled";
    Value *rval = EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
    if (!constval) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError ||
          Mode == DerivativeMode::ForwardModeSplit) {
        if (!rval)
          rval = Constant::getNullValue(gutils->getShadowType(call.getType()));
        setDiffe(&call, rval, BuilderZ);
      } else {
        resolveShadowPlaceholder(call, rval);
      }
    }
    if (!call.getType()->isVoidTy()) {
      for (auto &U :
           make_early_inc_range(gutils->getNewFromOriginal(&call)->uses())) {
        U.set(UndefValue::get(call.getType()));
      }
    }
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  // The OpenMP worksharing and synchronization calls are mirrored in the
  // reverse pass; forward derivatives keep only the primal call.
  const bool isReverseMode = Mode == DerivativeMode::ReverseModeGradient ||
                             Mode == DerivativeMode::ReverseModeCombined;
  if (isReverseMode && called) {
    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
      assert(fini);
      Value *args[] = {
          lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
          lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2)};
      auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
      fcall->setCallingConv(fini->getCallingConv());
      return true;
    }
  }

  // Canonicalize MPI routine names across calling conventions: the C
  // convention ("MPI_Recv", "PMPI_Recv") as well as Fortran ABI manglings
  // ("mpi_recv_", "mpi_comm_rank__", ...) all map to the canonical C name
  // (without profiling prefix) used throughout handleMPI.
  llvm::StringRef canonMPIName = canonicalizeMPIName(funcName);
  if (!canonMPIName.empty() &&
      (!gutils->isConstantInstruction(&call) || canonMPIName == "MPI_Barrier" ||
       canonMPIName == "MPI_Comm_free" ||
       canonMPIName == "MPI_Comm_disconnect" || canonMPIName == "MPI_Init" ||
       canonMPIName == "MPI_Init_thread" || canonMPIName == "MPI_Finalize" ||
       canonMPIName == "MPI_Test" || canonMPIName == "MPI_Probe" ||
       MPIInactiveCommAllocators.find(canonMPIName) !=
           MPIInactiveCommAllocators.end())) {
    handleMPI(call, called, canonMPIName);
    return true;
  }

  if (auto blas = extractBLAS(funcName)) {
    if (handleBLAS(call, called, *blas, overwritten_args))
      return true;
  }

  if (funcName == "printf" || funcName == "puts" ||
      startsWith(funcName, "_ZN3std2io5stdio6_print") ||
      startsWith(funcName, "_ZN4core3fmt")) {
    if (Mode == DerivativeMode::ReverseModeGradient) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    }
    return true;
  }
  if (called && (called->getName().contains("__enzyme_float") ||
                 called->getName().contains("__enzyme_double") ||
                 called->getName().contains("__enzyme_integer") ||
                 called->getName().contains("__enzyme_pointer"))) {
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  // Handle lgamma, safe to recompute so no store/change to forward
  if (called) {
    if (funcName == "__kmpc_for_static_init_4" ||
        funcName == "__kmpc_for_static_init_4u" ||
        funcName == "__kmpc_for_static_init_8" ||
        funcName == "__kmpc_for_static_init_8u") {
      if (isReverseMode) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto fini = called->getParent()->getFunction("__kmpc_for_static_fini");
        assert(fini);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(fini->getFunctionType(), fini, args);
        fcall->setCallingConv(fini->getCallingConv());
      }
      return true;
    }
    if (funcName == "__kmpc_for_static_fini") {
      if (isReverseMode) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      }
      return true;
    }
    // TODO check
    // Adjoint of barrier is to place a barrier at the corresponding
    // location in the reverse.
    if (funcName == "__kmpc_barrier") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto callval = call.getCalledOperand();
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getOperand(1)), Builder2)};
        Builder2.CreateCall(call.getFunctionType(), callval, args);
      }
      return true;
    }
    if (funcName == "__kmpc_critical") {
      if (isReverseMode) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto crit2 = called->getParent()->getFunction("__kmpc_end_critical");
        assert(crit2);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
        fcall->setCallingConv(crit2->getCallingConv());
      }
      return true;
    }
    if (funcName == "__kmpc_end_critical") {
      if (isReverseMode) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto crit2 = called->getParent()->getFunction("__kmpc_critical");
        assert(crit2);
        Value *args[] = {
            lookup(gutils->getNewFromOriginal(call.getArgOperand(0)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(1)), Builder2),
            lookup(gutils->getNewFromOriginal(call.getArgOperand(2)),
                   Builder2)};
        auto fcall = Builder2.CreateCall(crit2->getFunctionType(), crit2, args);
        fcall->setCallingConv(crit2->getCallingConv());
      }
      return true;
    }

    // The calling thread's copy of a threadprivate variable. Only inactive
    // ones (e.g. ICON's timers) are supported: keep the primal call, and
    // tape its result if the reverse pass needs it.
    if (funcName == "__kmpc_threadprivate_cached") {
      if (!gutils->isConstantValue(&call)) {
        std::string s;
        llvm::raw_string_ostream ss(s);
        ss << " active threadprivate variables are not supported: " << call
           << "\n";
        EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
        return true;
      }
      bool primalNeededInReverse = false;
      if (Mode != DerivativeMode::ReverseModeCombined &&
          Mode != DerivativeMode::ForwardMode &&
          Mode != DerivativeMode::ForwardModeError && subretused) {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        auto minCutMode = (Mode == DerivativeMode::ReverseModePrimal)
                              ? DerivativeMode::ReverseModeGradient
                              : Mode;
        primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, minCutMode, Seen,
                                   oldUnreachable);
      }
      if (primalNeededInReverse) {
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
        eraseIfUnused(call);
      } else if (Mode == DerivativeMode::ReverseModeGradient ||
                 Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        eraseIfUnused(call);
      }
      return true;
    }

    if (startsWith(funcName, "__kmpc") &&
        funcName != "__kmpc_global_thread_num") {
      std::string s;
      llvm::raw_string_ostream ss(s);
      ss << " unhandled openmp function: " << call << "\n";
      EmitNoDerivativeError(ss.str(), call, gutils, BuilderZ);
      return true;
    }

    auto mod = call.getParent()->getParent()->getParent();
#include "CallDerivatives.inc"

    if (funcName == "llvm.julia.gc_preserve_end") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {

        auto begin_call = cast<CallInst>(call.getOperand(0));

        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        if (!hasSingleGCPreserveEnd(begin_call, gutils)) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          ss << "cannot reverse gc preserve region with several ends: "
             << *begin_call << "\n";
          EmitNoDerivativeError(ss.str(), call, gutils, Builder2);
          return true;
        }
        SmallVector<Value *, 1> args;
        for (auto &arg : begin_call->args()) {
          bool primalUsed = false;
          bool shadowUsed = false;
          gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

          if (primalUsed)
            args.push_back(
                gutils->lookupM(gutils->getNewFromOriginal(arg), Builder2));

          if (!gutils->isConstantValue(arg) && shadowUsed) {
            Value *ptrshadow = gutils->lookupM(
                gutils->invertPointerM(arg, BuilderZ), Builder2);
            if (gutils->getWidth() == 1)
              args.push_back(ptrshadow);
            else
              for (size_t i = 0; i < gutils->getWidth(); ++i)
                args.push_back(gutils->extractMeta(Builder2, ptrshadow, i));
          }
        }

        auto newp = Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                "llvm.julia.gc_preserve_begin",
                FunctionType::get(Type::getTokenTy(call.getContext()),
                                  ArrayRef<Type *>(), true)),
            args);
        auto ifound = gutils->invertedPointers.find(begin_call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<CallInst>(&*ifound->second);
        gutils->invertedPointers.erase(ifound);
        gutils->invertedPointers.insert(std::make_pair(
            (const Value *)begin_call, InvertedPointerVH(gutils, newp)));

        gutils->replaceAWithB(placeholder, newp);
        gutils->erase(placeholder);
      }
      return true;
    }
    if (funcName == "llvm.julia.gc_preserve_begin") {
      SmallVector<Value *, 1> args;
      for (auto &arg : call.args()) {
        bool primalUsed = false;
        bool shadowUsed = false;
        gutils->getReturnDiffeType(arg, &primalUsed, &shadowUsed);

        if (primalUsed)
          args.push_back(gutils->getNewFromOriginal(arg));

        if (!gutils->isConstantValue(arg) && shadowUsed) {
          Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
          if (gutils->getWidth() == 1)
            args.push_back(ptrshadow);
          else
            for (size_t i = 0; i < gutils->getWidth(); ++i)
              args.push_back(gutils->extractMeta(BuilderZ, ptrshadow, i));
        }
      }

      auto newp = BuilderZ.CreateCall(called, args);
      auto oldp = gutils->getNewFromOriginal(&call);
      gutils->replaceAWithB(oldp, newp);
      gutils->erase(oldp);

      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);

        auto ifound = gutils->invertedPointers.find(&call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<CallInst>(&*ifound->second);
        if (!hasSingleGCPreserveEnd(&call, gutils)) {
          gutils->invertedPointers.erase(ifound);
          gutils->erase(placeholder);
          return true;
        }
        Builder2.CreateCall(
            called->getParent()->getOrInsertFunction(
                "llvm.julia.gc_preserve_end",
                FunctionType::get(Builder2.getVoidTy(), call.getType(), false)),
            placeholder);
      }
      return true;
    }

    // void _FortranAAssign(Descriptor &to, const Descriptor &from,
    //                      const char *sourceFile, int sourceLine)
    // is LLVM flang's assignment between descriptors; its derivative is the
    // same assignment applied to the shadow descriptors.
    if (funcName == "_FortranAAssign" && call.arg_size() == 4) {
      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        if (!gutils->isConstantInstruction(&call)) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);

          Value *file = gutils->getNewFromOriginal(call.getArgOperand(2));
          Value *line = gutils->getNewFromOriginal(call.getArgOperand(3));
          Value *shadowTo =
              gutils->invertPointerM(call.getArgOperand(0), Builder2);
          Value *shadowFrom =
              gutils->invertPointerM(call.getArgOperand(1), Builder2);

          auto rule = [&](Value *sTo, Value *sFrom) {
            auto dcall = Builder2.CreateCall(called->getFunctionType(), called,
                                             {sTo, sFrom, file, line});
            dcall->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
          };
          applyChainRule(Builder2, rule, shadowTo, shadowFrom);

          eraseIfUnused(call);
          return true;
        }
      }
      // An assignment between inactive descriptors (e.g. of derived types
      // holding only integers) is just the primal call in reverse mode too.
      if ((Mode == DerivativeMode::ReverseModePrimal ||
           Mode == DerivativeMode::ReverseModeCombined ||
           Mode == DerivativeMode::ReverseModeGradient) &&
          gutils->isConstantValue(call.getArgOperand(0)) &&
          gutils->isConstantValue(call.getArgOperand(1))) {
        if (Mode == DerivativeMode::ReverseModeGradient)
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
    }

    // void _FortranAEtime(const Descriptor *values, const Descriptor *time,
    //                     const char *sourceFile, int line)
    // is LLVM flang's ETIME. Its results are inactive, but it overwrites the
    // real(4) elements values(1:2) and time, whose shadow must then be zero:
    // after the call in forward mode, and in the reverse pass.
    if (funcName == "_FortranAEtime" && call.arg_size() == 4 &&
        (!gutils->isConstantValue(call.getArgOperand(0)) ||
         !gutils->isConstantValue(call.getArgOperand(1))) &&
        (Mode == DerivativeMode::ForwardMode ||
         Mode == DerivativeMode::ReverseModePrimal ||
         Mode == DerivativeMode::ReverseModeGradient ||
         Mode == DerivativeMode::ReverseModeCombined)) {
      auto &DL = gutils->newFunc->getParent()->getDataLayout();
      Type *I8PtrTy = getInt8PtrTy(call.getContext());
      Type *IdxTy = DL.getIntPtrType(call.getContext());
      unsigned P = DL.getPointerSize();
      // Offsets in a CFI descriptor: base_addr, and extent and sm of dim[0],
      // which follows elem_len, version, rank, type, attribute and extra.
      unsigned DimOff = 2 * P + 8;

      // The shadow elements to zero, or null where the shadow is the primal
      // or the element does not exist.
      auto shadowElements = [&](IRBuilder<> &B) {
        auto field = [&](Value *desc, Type *T, unsigned off) {
          Value *p = B.CreatePointerCast(desc, I8PtrTy);
          p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), p, off);
          return B.CreateLoad(T, B.CreatePointerCast(p, getUnqual(T)));
        };
        SmallVector<Value *, 6> elems;
        for (unsigned i = 0; i < 2; ++i) {
          Value *orig = call.getArgOperand(i);
          if (gutils->isConstantValue(orig))
            continue;
          Value *desc = gutils->getNewFromOriginal(orig);
          Value *sdescs = gutils->invertPointerM(orig, B);
          Value *base = field(desc, I8PtrTy, 0);
          for (unsigned w = 0; w < gutils->getWidth(); ++w) {
            Value *sdesc = gutils->getWidth() == 1
                               ? sdescs
                               : gutils->extractMeta(B, sdescs, w);
            Value *sbase = field(sdesc, I8PtrTy, 0);
            Value *valid = B.CreateAnd(B.CreateICmpNE(sbase, base),
                                       B.CreateIsNotNull(sbase));
            auto add = [&](Value *cond, Value *addr) {
              elems.push_back(
                  B.CreateSelect(cond, addr, Constant::getNullValue(I8PtrTy)));
            };
            if (i == 1) {
              add(valid, sbase);
              continue;
            }
            Value *extent = field(desc, IdxTy, DimOff + P);
            Value *sm = field(desc, IdxTy, DimOff + 2 * P);
            add(B.CreateAnd(
                    valid, B.CreateICmpSGE(extent, ConstantInt::get(IdxTy, 1))),
                sbase);
            add(B.CreateAnd(
                    valid, B.CreateICmpSGE(extent, ConstantInt::get(IdxTy, 2))),
                B.CreateInBoundsGEP(B.getInt8Ty(), sbase, sm));
          }
        }
        return elems;
      };
      auto zero = [&](IRBuilder<> &B, Value *addr) {
        auto VT = FixedVectorType::get(B.getFloatTy(), 1);
        B.CreateMaskedStore(Constant::getNullValue(VT),
                            B.CreatePointerCast(addr, getUnqual(VT)), Align(4),
                            B.CreateVectorSplat(1, B.CreateIsNotNull(addr)));
      };

      if (Mode == DerivativeMode::ForwardMode) {
        IRBuilder<> B(newCall->getNextNode());
        for (auto addr : shadowElements(B))
          zero(B, addr);
        return true;
      }

      unsigned N = 0;
      for (unsigned i = 0; i < 2; ++i)
        if (!gutils->isConstantValue(call.getArgOperand(i)))
          N += (i == 0 ? 2 : 1) * gutils->getWidth();
      Type *TapeTy = ArrayType::get(I8PtrTy, N);
      Value *tape;
      if (Mode == DerivativeMode::ReverseModeGradient) {
        tape = BuilderZ.CreatePHI(TapeTy, 0);
      } else {
        tape = UndefValue::get(TapeTy);
        auto elems = shadowElements(BuilderZ);
        for (unsigned i = 0; i < N; ++i)
          tape = BuilderZ.CreateInsertValue(tape, elems[i], i);
        if (auto I = dyn_cast<Instruction>(tape))
          gutils->TapesToPreventRecomputation.insert(I);
      }
      tape = gutils->cacheForReverse(
          BuilderZ, tape, getIndex(&call, CacheType::Tape, BuilderZ));

      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModeCombined) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        tape = lookup(tape, Builder2);
        for (unsigned i = 0; i < N; ++i)
          zero(Builder2, Builder2.CreateExtractValue(tape, i));
      }
      if (Mode == DerivativeMode::ReverseModeGradient)
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    // Other LLVM flang runtime functions acting on descriptors (allocation,
    // pointer association, initialization, copies): the derivative is the
    // same call on the shadow descriptors. Memory that the call allocates is
    // zeroed in the shadow, and re-initialized for derived types, whose
    // components hold descriptors.
    //
    // In reverse mode the augmented pass replays the allocation group the
    // same way, and the reverse pass undoes what it did to the shadows:
    //  - Allocate: the shadow memory is deallocated again in the reverse pass
    //    (its base address is taped, since the shadow descriptor may have been
    //    reused in between).
    //  - Deallocate: the shadow memory is detached rather than freed (its base
    //    address taped, the shadow descriptor nulled), because the reverse pass
    //    still accumulates adjoints into it; the reverse pass attaches it
    //    again.
    //  - Structure changes (bounds, association, initialization): nothing.
    if (auto replay = getFlangShadowReplay(funcName)) {
      bool forward = Mode == DerivativeMode::ForwardMode ||
                     Mode == DerivativeMode::ForwardModeError;
      bool reverse = Mode == DerivativeMode::ReverseModePrimal ||
                     Mode == DerivativeMode::ReverseModeCombined ||
                     Mode == DerivativeMode::ReverseModeGradient;
      if ((forward || reverse) && !gutils->isConstantInstruction(&call)) {
        SmallVector<int, 2> shadowed;
        for (int i : replay->shadowArgs)
          if (i >= 0 && (unsigned)i < call.arg_size() &&
              !gutils->isConstantValue(call.getArgOperand(i)))
            shadowed.push_back(i);
        unsigned expected = 0;
        for (int i : replay->shadowArgs)
          if (i >= 0 && (unsigned)i < call.arg_size())
            expected++;
        // All descriptors inactive: nothing to replay, the primal call is all
        // there is (e.g. allocating an inactive variable next to an active
        // one in the same statement).
        if (shadowed.empty()) {
          if (Mode == DerivativeMode::ReverseModeGradient)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return true;
        }
        // A copy from an inactive into an active descriptor would need the
        // shadow to be zeroed rather than copied into: not handled.
        if (shadowed.size() == expected &&
            (forward || replay->kind != FlangReplayKind::ForwardOnly)) {
          auto &M = *called->getParent();
          auto &C = call.getContext();
          auto I8Ptr = PointerType::getUnqual(C);
          auto I32 = Type::getInt32Ty(C);
          auto nullp = ConstantPointerNull::get(I8Ptr);
          auto zero32 = ConstantInt::get(I32, 0);
          bool allocates = replay->kind == FlangReplayKind::Allocate;
          bool deallocates = replay->kind == FlangReplayKind::Deallocate;

          IRBuilder<> Builder2(&call);
          if (forward)
            getForwardBuilder(Builder2);
          else
            Builder2.SetInsertPoint(newCall->getNextNode());

          // The shadow descriptor of argument 0 in lane w.
          auto shadowDesc = [&](IRBuilder<> &B, Value *shadows, unsigned w) {
            return gutils->getWidth() > 1 ? gutils->extractMeta(B, shadows, w)
                                          : shadows;
          };

          // Base addresses of the shadow memory of argument 0, one per lane:
          // after the call for Allocate, before it for Deallocate.
          SmallVector<Value *, 1> bases;
          if (Mode != DerivativeMode::ReverseModeGradient) {
            SmallVector<Value *, 8> primalArgs;
            for (auto &op : call.args())
              primalArgs.push_back(gutils->getNewFromOriginal(op));
            SmallVector<Value *, 2> shadows;
            for (int i : shadowed)
              shadows.push_back(
                  gutils->invertPointerM(call.getArgOperand(i), Builder2));

            FunctionCallee sizeFn, initFn;
            if (allocates) {
              sizeFn = M.getOrInsertFunction(
                  "_FortranASize",
                  FunctionType::get(Type::getInt64Ty(C), {I8Ptr, I8Ptr, I32},
                                    false));
              initFn = M.getOrInsertFunction(
                  "_FortranAInitialize",
                  FunctionType::get(Type::getVoidTy(C), {I8Ptr, I8Ptr, I32},
                                    false));
            }

            for (unsigned w = 0; w < gutils->getWidth(); w++) {
              SmallVector<Value *, 8> args(primalArgs);
              for (unsigned k = 0; k < shadowed.size(); k++)
                args[shadowed[k]] = shadowDesc(Builder2, shadows[k], w);
              Value *desc = args[0];

              if (deallocates && reverse) {
                // base_addr leads every descriptor.
                bases.push_back(Builder2.CreateLoad(I8Ptr, desc));
                Builder2.CreateStore(nullp, desc);
                continue;
              }

              auto dcall =
                  Builder2.CreateCall(called->getFunctionType(), called, args);
              dcall->setDebugLoc(
                  gutils->getNewFromOriginal(call.getDebugLoc()));
              dcall->setCallingConv(call.getCallingConv());

              if (allocates) {
                // base_addr and elem_len lead every descriptor.
                Value *count =
                    Builder2.CreateCall(sizeFn, {desc, nullp, zero32});
                Value *elemLen = Builder2.CreateLoad(
                    Type::getInt64Ty(C), Builder2.CreateConstInBoundsGEP1_64(
                                             Type::getInt8Ty(C), desc, 8));
                Value *base = Builder2.CreateLoad(I8Ptr, desc);
                Builder2.CreateMemSet(base, Builder2.getInt8(0),
                                      Builder2.CreateMul(count, elemLen),
                                      MaybeAlign());
                Builder2.CreateCall(initFn, {desc, nullp, zero32});
                bases.push_back(base);
              }
            }
          }

          if (forward || !(allocates || deallocates)) {
            if (Mode == DerivativeMode::ReverseModeGradient)
              eraseIfUnused(call, /*erase*/ true, /*check*/ false);
            else
              eraseIfUnused(call);
            return true;
          }

          // Tape the shadow base addresses for the reverse pass.
          unsigned W = gutils->getWidth();
          Type *TapeTy = ArrayType::get(I8Ptr, W);
          Value *tape;
          if (Mode == DerivativeMode::ReverseModeGradient) {
            tape = BuilderZ.CreatePHI(TapeTy, 0);
          } else {
            tape = UndefValue::get(TapeTy);
            for (unsigned w = 0; w < W; ++w)
              tape = Builder2.CreateInsertValue(tape, bases[w], w);
            if (auto I = dyn_cast<Instruction>(tape))
              gutils->TapesToPreventRecomputation.insert(I);
          }
          tape = gutils->cacheForReverse(
              Mode == DerivativeMode::ReverseModeGradient ? BuilderZ : Builder2,
              tape, getIndex(&call, CacheType::Tape, BuilderZ));

          if (Mode == DerivativeMode::ReverseModeGradient ||
              Mode == DerivativeMode::ReverseModeCombined) {
            IRBuilder<> Builder3(&call);
            getReverseBuilder(Builder3);
            tape = lookup(tape, Builder3);
            Value *shadows =
                lookup(gutils->invertPointerM(call.getArgOperand(0), Builder3),
                       Builder3);
            FunctionCallee deallocFn;
            if (allocates)
              deallocFn = M.getOrInsertFunction(
                  startsWith(funcName, "_FortranAPointer")
                      ? "_FortranAPointerDeallocate"
                      : "_FortranAAllocatableDeallocate",
                  FunctionType::get(
                      I32, {I8Ptr, Type::getInt1Ty(C), I8Ptr, I8Ptr, I32},
                      false));
            for (unsigned w = 0; w < W; ++w) {
              Value *desc = shadowDesc(Builder3, shadows, w);
              Builder3.CreateStore(Builder3.CreateExtractValue(tape, w), desc);
              // Deallocate through the runtime, which also deallocates the
              // allocatable components of derived types; with STAT, so that
              // a reused shadow descriptor cannot abort the program.
              if (allocates)
                Builder3.CreateCall(deallocFn, {desc, Builder3.getTrue(), nullp,
                                                nullp, zero32});
            }
          }
          if (Mode == DerivativeMode::ReverseModeGradient)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return true;
        }
      }
    }

    /*
     * int gsl_sf_legendre_array_e(const gsl_sf_legendre_t norm,
                                   const size_t lmax,
                                   const double x,
                                   const double csphase,
                                   double result_array[]);
    */
    // d L(n, x) / dx = L(n,x) * x * (n-1) + 1
    if (funcName == "gsl_sf_legendre_array_e") {
      if (gutils->isConstantValue(call.getArgOperand(4))) {
        eraseIfUnused(call);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModePrimal) {
        eraseIfUnused(call);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        ValueType BundleTypes[5] = {ValueType::None, ValueType::None,
                                    ValueType::None, ValueType::None,
                                    ValueType::Shadow};
        auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                               /*lookup*/ true);

        Type *types[6] = {
            call.getOperand(0)->getType(), call.getOperand(1)->getType(),
            call.getOperand(2)->getType(), call.getOperand(3)->getType(),
            call.getOperand(4)->getType(), call.getOperand(4)->getType(),
        };
        FunctionType *FT = FunctionType::get(call.getType(), types, false);
        auto F = getOrInsertPerCallingConv(*called->getParent(), called,
                                           "gsl_sf_legendre_deriv_array_e", FT);

        llvm::Value *args[6] = {
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(0)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(1)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(2)),
                            Builder2),
            gutils->lookupM(gutils->getNewFromOriginal(call.getOperand(3)),
                            Builder2),
            nullptr,
            nullptr};

        Type *typesS[] = {args[1]->getType()};
        FunctionType *FTS =
            FunctionType::get(args[1]->getType(), typesS, false);
        auto FS = getOrInsertPerCallingConv(*called->getParent(), called,
                                            "gsl_sf_legendre_array_n", FTS);
        Value *alSize = Builder2.CreateCall(FS, args[1]);
        Value *tmp = CreateAllocation(Builder2, types[2], alSize);
        Value *dtmp = CreateAllocation(Builder2, types[2], alSize);
        Builder2.CreateLifetimeStart(tmp);
        Builder2.CreateLifetimeStart(dtmp);

        args[4] = Builder2.CreateBitCast(tmp, types[4]);
        args[5] = Builder2.CreateBitCast(dtmp, types[5]);

        Builder2.CreateCall(F, args, Defs);
        Builder2.CreateLifetimeEnd(tmp);
        CreateDealloc(Builder2, tmp);

        BasicBlock *currentBlock = Builder2.GetInsertBlock();

        BasicBlock *loopBlock = gutils->addReverseBlock(
            currentBlock, currentBlock->getName() + "_loop");
        BasicBlock *endBlock =
            gutils->addReverseBlock(loopBlock, currentBlock->getName() + "_end",
                                    /*fork*/ true, /*push*/ false);

        Builder2.CreateCondBr(
            Builder2.CreateICmpEQ(args[1], Constant::getNullValue(types[1])),
            endBlock, loopBlock);
        Builder2.SetInsertPoint(loopBlock);

        auto idx = Builder2.CreatePHI(types[1], 2);
        idx->addIncoming(ConstantInt::get(types[1], 0, false), currentBlock);

        auto acc_idx = Builder2.CreatePHI(types[2], 2);

        Value *inc = Builder2.CreateAdd(
            idx, ConstantInt::get(types[1], 1, false), "", true, true);
        idx->addIncoming(inc, loopBlock);
        acc_idx->addIncoming(Constant::getNullValue(types[2]), currentBlock);

        Value *idxs[] = {idx};
        Value *dtmp_idx = Builder2.CreateInBoundsGEP(types[2], dtmp, idxs);
        Value *d_req = Builder2.CreateInBoundsGEP(
            types[2],
            Builder2.CreatePointerCast(
                gutils->invertPointerM(call.getOperand(4), Builder2),
                getUnqual(types[2])),
            idxs);

        auto l0 = Builder2.CreateLoad(types[2], dtmp_idx);
        auto l1 = Builder2.CreateLoad(types[2], d_req);
        auto acc = Builder2.CreateFAdd(acc_idx, Builder2.CreateFMul(l0, l1));
        Builder2.CreateStore(Constant::getNullValue(types[2]), d_req);

        acc_idx->addIncoming(acc, loopBlock);

        Builder2.CreateCondBr(Builder2.CreateICmpEQ(inc, args[1]), endBlock,
                              loopBlock);

        Builder2.SetInsertPoint(endBlock);
        {
          auto found = gutils->reverseBlockToPrimal.find(endBlock);
          assert(found != gutils->reverseBlockToPrimal.end());
          SmallVector<BasicBlock *, 4> &vec =
              gutils->reverseBlocks[found->second];
          assert(vec.size());
          vec.push_back(endBlock);
        }

        auto fin_idx = Builder2.CreatePHI(types[2], 2);
        fin_idx->addIncoming(Constant::getNullValue(types[2]), currentBlock);
        fin_idx->addIncoming(acc, loopBlock);

        Builder2.CreateLifetimeEnd(dtmp);
        CreateDealloc(Builder2, dtmp);

        ((DiffeGradientUtils *)gutils)
            ->addToDiffe(call.getOperand(2), fin_idx, Builder2, types[2]);

        return true;
      }
    }

    // Functions that only modify pointers and don't allocate memory,
    // needs to be run on shadow in primal
    if (funcName == "_ZSt29_Rb_tree_insert_and_rebalancebPSt18_Rb_tree_"
                    "node_baseS0_RS_") {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(3)))
        return true;
      SmallVector<Value *, 2> args;
      for (auto &arg : call.args()) {
        if (gutils->isConstantValue(arg))
          args.push_back(gutils->getNewFromOriginal(arg));
        else
          args.push_back(gutils->invertPointerM(arg, BuilderZ));
      }
      BuilderZ.CreateCall(called, args);
      return true;
    }

    // Functions that initialize a shadow data structure (with no
    // other arguments) needs to be run on shadow in primal.
    if (funcName == "_ZNSt8ios_baseC2Ev" || funcName == "_ZNSt8ios_baseD2Ev" ||
        funcName == "_ZNSt6localeC1Ev" || funcName == "_ZNSt6localeD1Ev" ||
        funcName == "_ZNKSt5ctypeIcE13_M_widen_initEv") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(0)))
        return true;
      Value *args[] = {gutils->invertPointerM(call.getArgOperand(0), BuilderZ)};
      BuilderZ.CreateCall(called, args);
      return true;
    }

    if (funcName == "_ZNSt9basic_iosIcSt11char_traitsIcEE4initEPSt15basic_"
                    "streambufIcS1_E") {
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (gutils->isConstantValue(call.getArgOperand(0)))
        return true;
      Value *args[] = {gutils->invertPointerM(call.getArgOperand(0), BuilderZ),
                       gutils->invertPointerM(call.getArgOperand(1), BuilderZ)};
      BuilderZ.CreateCall(called, args);
      return true;
    }

    // if constant instruction and readonly (thus must be pointer return)
    // and shadow return recomputable from shadow arguments.
    if (funcName == "__dynamic_cast" ||
        funcName == "_ZSt18_Rb_tree_decrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPKSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_decrementPSt18_Rb_tree_node_base" ||
        funcName == "_ZSt18_Rb_tree_incrementPSt18_Rb_tree_node_base" ||
        funcName == "jl_ptr_to_array" || funcName == "jl_ptr_to_array_1d") {
      bool shouldCache = false;
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          shouldCache = true;
        }
      }
      ValueToValueMapTy empty;
      bool lrc = gutils->legalRecompute(&call, empty, nullptr);

      if (!gutils->isConstantValue(&call)) {
        auto ifound = gutils->invertedPointers.find(&call);
        assert(ifound != gutils->invertedPointers.end());
        auto placeholder = cast<PHINode>(&*ifound->second);

        if (subretType == DIFFE_TYPE::DUP_ARG) {
          Value *shadow = placeholder;
          if (lrc || Mode == DerivativeMode::ReverseModePrimal ||
              Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ForwardMode ||
              Mode == DerivativeMode::ForwardModeError) {
            if (gutils->isConstantValue(call.getArgOperand(0)))
              shadow = gutils->getNewFromOriginal(&call);
            else {
              SmallVector<Value *, 2> args;
              size_t i = 0;
              for (auto &arg : call.args()) {
                if (gutils->isConstantValue(arg) ||
                    (funcName == "__dynamic_cast" && i > 0) ||
                    (funcName == "jl_ptr_to_array_1d" && i != 1) ||
                    (funcName == "jl_ptr_to_array" && i != 1))
                  args.push_back(gutils->getNewFromOriginal(arg));
                else
                  args.push_back(gutils->invertPointerM(arg, BuilderZ));
                i++;
              }
              shadow = BuilderZ.CreateCall(called, args);
            }
          }

          bool needsReplacement = true;
          if (!lrc && (Mode == DerivativeMode::ReverseModePrimal ||
                       Mode == DerivativeMode::ReverseModeGradient)) {
            shadow = gutils->cacheForReverse(
                BuilderZ, shadow, getIndex(&call, CacheType::Shadow, BuilderZ));
            if (Mode == DerivativeMode::ReverseModeGradient)
              needsReplacement = false;
          }
          gutils->invertedPointers.erase((const Value *)&call);
          gutils->invertedPointers.insert(std::make_pair(
              (const Value *)&call, InvertedPointerVH(gutils, shadow)));
          if (needsReplacement) {
            assert(shadow != placeholder);
            gutils->replaceAWithB(placeholder, shadow);
            gutils->erase(placeholder);
          }
        } else {
          gutils->invertedPointers.erase((const Value *)&call);
          gutils->erase(placeholder);
        }
      }

      if (Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        eraseIfUnused(call);
        assert(gutils->isConstantInstruction(&call));
        return true;
      }

      if (!shouldCache && !lrc) {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        bool primalNeededInReverse =
            DifferentialUseAnalysis::is_value_needed_in_reverse<
                QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);
        {
          auto found = gutils->knownRecomputeHeuristic.find(&call);
          if (found != gutils->knownRecomputeHeuristic.end()) {
            if (!found->second) {
              primalNeededInReverse = true;
            }
          }
        }
        shouldCache = primalNeededInReverse;
      }

      if (shouldCache) {
        BuilderZ.SetInsertPoint(newCall->getNextNode());
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
      }
      eraseIfUnused(call);
      assert(gutils->isConstantInstruction(&call));
      return true;
    }

    if (called) {
      if (funcName == "julia.write_barrier" ||
          funcName == "julia.write_barrier_binding") {
        std::map<UsageKey, bool> Seen =
            gutils->populateSeenFromKnownRecompute();
        bool backwardsShadow = false;
        bool forwardsShadow = true;
        for (auto pair : gutils->backwardsOnlyShadows) {
          if (pair.second.stores.count(&call)) {
            backwardsShadow = true;
            forwardsShadow = pair.second.primalInitialize;
            if (auto inst = dyn_cast<Instruction>(pair.first))
              if (!forwardsShadow && pair.second.LI &&
                  pair.second.LI->contains(inst->getParent()))
                backwardsShadow = false;
            break;
          }
        }

        if (Mode == DerivativeMode::ForwardMode ||
            Mode == DerivativeMode::ForwardModeError ||
            (Mode == DerivativeMode::ReverseModeCombined &&
             (forwardsShadow || backwardsShadow)) ||
            (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
            (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow)) {
          IRBuilder<> BuilderZ(gutils->getNewFromOriginal(&call));
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            SmallVector<Value *, 1> iargs;
            bool first = true;
            for (auto &arg : call.args()) {
              if (!gutils->isConstantValue(arg)) {
                Value *ptrshadow = gutils->invertPointerM(arg, BuilderZ);
                if (gutils->getWidth() > 1) {
                  ptrshadow = gutils->extractMeta(BuilderZ, ptrshadow, i);
                }
                iargs.push_back(ptrshadow);
              } else {
                if (first)
                  break;
              }
              first = false;
            }
            if (iargs.size()) {
              BuilderZ.CreateCall(called, iargs);
            }
          }
        }

        bool forceErase = false;
        if (Mode == DerivativeMode::ReverseModeGradient) {

          // Since we won't redo the store in the reverse pass, do not
          // force the write barrier.
          forceErase = true;
          for (const auto &pair : gutils->rematerializableAllocations) {
            if (!pair.second.stores.count(&call))
              continue;
            if (gutils->allocationsToBeRematerialized.count(pair.first))
              // However, if we are rematerailizing the allocation and not
              // inside the loop level rematerialization, we do still need the
              // reverse passes ``fake primal'' store and therefore write
              // barrier
              if (!pair.second.LI || !pair.second.LI->contains(&call)) {
                forceErase = false;
              }
          }
        }
        if (forceErase)
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        else
          eraseIfUnused(call);

        return true;
      }
      Intrinsic::ID ID = Intrinsic::not_intrinsic;
      if (isMemFreeLibMFunction(funcName, &ID)) {
        if (Mode == DerivativeMode::ReverseModePrimal ||
            gutils->isConstantInstruction(&call)) {

          if (gutils->knownRecomputeHeuristic.find(&call) !=
              gutils->knownRecomputeHeuristic.end()) {
            if (!gutils->knownRecomputeHeuristic[&call]) {
              gutils->cacheForReverse(
                  BuilderZ, newCall,
                  getIndex(&call, CacheType::Self, BuilderZ));
            }
          }
          eraseIfUnused(call);
          return true;
        }

        if (ID != Intrinsic::not_intrinsic) {
          SmallVector<Value *, 2> orig_ops(call.getNumOperands());
          for (unsigned i = 0; i < call.getNumOperands(); ++i) {
            orig_ops[i] = call.getOperand(i);
          }
          bool cached = handleAdjointForIntrinsic(ID, call, orig_ops);
          if (!cached) {
            if (gutils->knownRecomputeHeuristic.find(&call) !=
                gutils->knownRecomputeHeuristic.end()) {
              if (!gutils->knownRecomputeHeuristic[&call]) {
                gutils->cacheForReverse(
                    BuilderZ, newCall,
                    getIndex(&call, CacheType::Self, BuilderZ));
              }
            }
          }
          eraseIfUnused(call);
          return true;
        }
      }
    }
  }
  if (auto assembly = dyn_cast<InlineAsm>(call.getCalledOperand())) {
    if (assembly->getAsmString() == "maxpd $1, $0") {
      if (Mode == DerivativeMode::ReverseModePrimal ||
          gutils->isConstantInstruction(&call)) {

        if (gutils->knownRecomputeHeuristic.find(&call) !=
            gutils->knownRecomputeHeuristic.end()) {
          if (!gutils->knownRecomputeHeuristic[&call]) {
            gutils->cacheForReverse(BuilderZ, newCall,
                                    getIndex(&call, CacheType::Self, BuilderZ));
          }
        }
        eraseIfUnused(call);
        return true;
      }

      SmallVector<Value *, 2> orig_ops(call.getNumOperands());
      for (unsigned i = 0; i < call.getNumOperands(); ++i) {
        orig_ops[i] = call.getOperand(i);
      }
      handleAdjointForIntrinsic(Intrinsic::maxnum, call, orig_ops);
      if (gutils->knownRecomputeHeuristic.find(&call) !=
          gutils->knownRecomputeHeuristic.end()) {
        if (!gutils->knownRecomputeHeuristic[&call]) {
          gutils->cacheForReverse(BuilderZ, newCall,
                                  getIndex(&call, CacheType::Self, BuilderZ));
        }
      }
      eraseIfUnused(call);
      return true;
    }
  }

  if (funcName == "realloc") {
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      if (!gutils->isConstantValue(&call)) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();

        auto rule = [&](Value *ip) {
          ValueType BundleTypes[2] = {ValueType::Shadow, ValueType::Primal};

          auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                 /*lookup*/ false);

          llvm::Value *args[2] = {
              ip, gutils->getNewFromOriginal(call.getOperand(1))};
          CallInst *CI = Builder2.CreateCall(
              call.getFunctionType(), call.getCalledFunction(), args, Defs);
          CI->setAttributes(call.getAttributes());
          CI->setCallingConv(call.getCallingConv());
          CI->setTailCallKind(call.getTailCallKind());
          CI->setDebugLoc(dbgLoc);
          return CI;
        };

        Value *CI = applyChainRule(
            call.getType(), Builder2, rule,
            gutils->invertPointerM(call.getOperand(0), Builder2));

        auto found = gutils->invertedPointers.find(&call);
        PHINode *placeholder = cast<PHINode>(&*found->second);

        gutils->invertedPointers.erase(found);
        gutils->replaceAWithB(placeholder, CI);
        gutils->erase(placeholder);
        gutils->invertedPointers.insert(
            std::make_pair(&call, InvertedPointerVH(gutils, CI)));
      }
      eraseIfUnused(call);
      return true;
    }
  }

  if (isAllocationFunction(funcName, gutils->TLI)) {

    bool constval = gutils->isConstantValue(&call);

    if (!constval) {
      auto dbgLoc = gutils->getNewFromOriginal(&call)->getDebugLoc();
      auto found = gutils->invertedPointers.find(&call);
      PHINode *placeholder = cast<PHINode>(&*found->second);
      IRBuilder<> bb(placeholder);

      SmallVector<Value *, 8> args;
      for (auto &arg : call.args()) {
        args.push_back(gutils->getNewFromOriginal(arg));
      }

      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ForwardModeSplit) {

        Value *anti = placeholder;
        // If rematerializable allocations and split mode, we can
        // simply elect to build the entire piece in the reverse
        // since it should be possible to perform any shadow stores
        // of pointers (from rematerializable property) and it does
        // not escape the function scope (lest it not be
        // rematerializable) so all input derivatives remain zero.
        bool backwardsShadow = false;
        bool forwardsShadow = true;
        bool inLoop = false;
        bool isAlloca = isa<AllocaInst>(&call);
        {
          auto found = gutils->backwardsOnlyShadows.find(&call);
          if (found != gutils->backwardsOnlyShadows.end()) {
            backwardsShadow = true;
            forwardsShadow = found->second.primalInitialize;
            // If in a loop context, maintain the same free behavior.
            if (found->second.LI &&
                found->second.LI->contains(call.getParent()))
              inLoop = true;
          }
        }
        {

          if (!forwardsShadow) {
            if (Mode == DerivativeMode::ReverseModePrimal) {
              // Needs a stronger replacement check/assertion.
              Value *replacement = getUndefinedValueForType(
                  *gutils->oldFunc->getParent(), placeholder->getType());
              gutils->replaceAWithB(placeholder, replacement);
              gutils->invertedPointers.erase(found);
              gutils->invertedPointers.insert(std::make_pair(
                  &call, InvertedPointerVH(gutils, replacement)));
              gutils->erase(placeholder);
              anti = nullptr;
              goto endAnti;
            } else if (inLoop) {
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  placeholder);
              if (hasMetadata(&call, "enzyme_fromstack"))
                isAlloca = true;
              goto endAnti;
            }
          }
          placeholder->setName("");
          if (shadowHandlers.find(funcName) != shadowHandlers.end()) {
            bb.SetInsertPoint(placeholder);

            if (Mode == DerivativeMode::ReverseModeCombined ||
                (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
                (Mode == DerivativeMode::ReverseModeGradient &&
                 backwardsShadow)) {
              anti = applyChainRule(call.getType(), bb, [&]() {
                return shadowHandlers[funcName](bb, &call, args, gutils);
              });
              if (anti->getType() != placeholder->getType()) {
                llvm::errs() << "orig: " << call << "\n";
                llvm::errs() << "placeholder: " << *placeholder << "\n";
                llvm::errs() << "anti: " << *anti << "\n";
              }
              gutils->invertedPointers.erase(found);
              bb.SetInsertPoint(placeholder);

              gutils->replaceAWithB(placeholder, anti);
              gutils->erase(placeholder);
            }

            if (auto inst = dyn_cast<Instruction>(anti))
              bb.SetInsertPoint(inst);

            if (!backwardsShadow)
              anti = gutils->cacheForReverse(
                  bb, anti, getIndex(&call, CacheType::Shadow, BuilderZ));
          } else {
            bool zeroed = false;
            uint64_t idx = 0;
            Value *prev = nullptr;
            ;
            auto rule = [&]() {
              Value *anti =
                  bb.CreateCall(call.getFunctionType(), call.getCalledOperand(),
                                args, call.getName() + "'mi");
              cast<CallInst>(anti)->setAttributes(call.getAttributes());
              cast<CallInst>(anti)->setCallingConv(call.getCallingConv());
              cast<CallInst>(anti)->setTailCallKind(call.getTailCallKind());
              cast<CallInst>(anti)->setDebugLoc(dbgLoc);

              if (anti->getType()->isPointerTy()) {
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NoAlias);
                cast<CallInst>(anti)->addAttributeAtIndex(
                    AttributeList::ReturnIndex, Attribute::NonNull);

                if (funcName == "malloc" || funcName == "_Znwm" ||
                    funcName == "??2@YAPAXI@Z" ||
                    funcName == "??2@YAPEAX_K@Z") {
                  if (auto ci = dyn_cast<ConstantInt>(args[0])) {
                    unsigned derefBytes = ci->getLimitedValue();
                    CallInst *cal =
                        cast<CallInst>(gutils->getNewFromOriginal(&call));
                    cast<CallInst>(anti)->addDereferenceableRetAttr(derefBytes);
                    cal->addDereferenceableRetAttr(derefBytes);
#if !defined(FLANG) && !defined(ROCM)
                    AttrBuilder B(ci->getContext());
#else
                    AttrBuilder B;
#endif
                    B.addDereferenceableOrNullAttr(derefBytes);
                    cast<CallInst>(anti)->setAttributes(
                        cast<CallInst>(anti)->getAttributes().addRetAttributes(
                            call.getContext(), B));
                    cal->setAttributes(cal->getAttributes().addRetAttributes(
                        call.getContext(), B));
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NoAlias);
                    cal->addAttributeAtIndex(AttributeList::ReturnIndex,
                                             Attribute::NonNull);
                  }
                }
                if (funcName == "julia.gc_alloc_obj" ||
                    funcName == "jl_gc_alloc_typed" ||
                    funcName == "ijl_gc_alloc_typed") {
                  if (EnzymeShadowAllocRewrite) {
                    bool used = unnecessaryInstructions.find(&call) ==
                                unnecessaryInstructions.end();
                    EnzymeShadowAllocRewrite(wrap(anti), gutils, wrap(&call),
                                             idx, wrap(prev), used);
                  }
                }
              }
              if (Mode == DerivativeMode::ReverseModeCombined ||
                  (Mode == DerivativeMode::ReverseModePrimal &&
                   forwardsShadow) ||
                  (Mode == DerivativeMode::ReverseModeGradient &&
                   backwardsShadow) ||
                  (Mode == DerivativeMode::ForwardModeSplit &&
                   backwardsShadow)) {
                if (!inLoop) {
                  zeroKnownAllocation(bb, anti, args, funcName, gutils->TLI,
                                      &call);
                  zeroed = true;
                }
              }
              idx++;
              prev = anti;
              return anti;
            };

            anti = applyChainRule(call.getType(), bb, rule);

            gutils->invertedPointers.erase(found);
            if (&*bb.GetInsertPoint() == placeholder)
              bb.SetInsertPoint(placeholder->getNextNode());
            gutils->replaceAWithB(placeholder, anti);
            gutils->erase(placeholder);

            if (!backwardsShadow)
              anti = gutils->cacheForReverse(
                  bb, anti, getIndex(&call, CacheType::Shadow, BuilderZ));
            else {
              if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
                isAlloca = true;
                bb.SetInsertPoint(cast<Instruction>(anti));
                Value *Size;
                if (funcName == "malloc")
                  Size = args[0];
                else if (funcName == "julia.gc_alloc_obj" ||
                         funcName == "jl_gc_alloc_typed" ||
                         funcName == "ijl_gc_alloc_typed")
                  Size = args[1];
                else
                  llvm_unreachable("Unknown allocation to upgrade");

                Type *elTy = Type::getInt8Ty(call.getContext());
                if (MD->getNumOperands() == 2) {
                  elTy = (Type *)cast<ConstantInt>(
                             cast<ConstantAsMetadata>(MD->getOperand(1))
                                 ->getValue())
                             ->getLimitedValue();
                  Value *tsize = ConstantInt::get(
                      Size->getType(), (gutils->newFunc->getParent()
                                            ->getDataLayout()
                                            .getTypeAllocSizeInBits(elTy) +
                                        7) /
                                           8);

                  Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
                }
                std::string name = "";
#if LLVM_VERSION_MAJOR < 17
                if (call.getContext().supportsTypedPointers()) {
                  for (auto U : call.users()) {
                    if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
                      if (MD->getNumOperands() == 1) {
                        elTy = U->getType()->getPointerElementType();
                        Value *tsize = ConstantInt::get(
                            Size->getType(),
                            (gutils->newFunc->getParent()
                                 ->getDataLayout()
                                 .getTypeAllocSizeInBits(elTy) +
                             7) /
                                8);

                        Size = bb.CreateUDiv(Size, tsize, "", /*exact*/ true);
                      }
                      name = (U->getName() + "'ai").str();
                      break;
                    }
                  }
                }
#endif
                auto rule = [&](Value *anti) {
                  bb.SetInsertPoint(cast<Instruction>(anti));
                  Value *replacement = bb.CreateAlloca(elTy, Size, name);
                  if (name.size() == 0)
                    replacement->takeName(anti);
                  else
                    anti->setName("");
                  auto Alignment = cast<ConstantInt>(cast<ConstantAsMetadata>(
                                                         MD->getOperand(0))
                                                         ->getValue())
                                       ->getLimitedValue();
                  if (Alignment) {
                    cast<AllocaInst>(replacement)
                        ->setAlignment(Align(Alignment));
                  }
#if LLVM_VERSION_MAJOR < 17
                  if (call.getContext().supportsTypedPointers()) {
                    if (anti->getType()->getPointerElementType() != elTy)
                      replacement = bb.CreatePointerCast(
                          replacement,
                          getUnqual(anti->getType()->getPointerElementType()));
                  }
#endif
                  auto PT = cast<PointerType>(anti->getType());
                  if (PT->getAddressSpace()) {
                    replacement = bb.CreateAddrSpaceCast(replacement, PT);
                    cast<Instruction>(replacement)
                        ->setMetadata(
                            "enzyme_backstack",
                            MDNode::get(replacement->getContext(), {}));
                  }
                  gutils->replaceAWithB(cast<Instruction>(anti), replacement);
                  bb.SetInsertPoint(cast<Instruction>(anti)->getNextNode());
                  gutils->erase(cast<Instruction>(anti));
                  return replacement;
                };

                auto replacement =
                    applyChainRule(call.getType(), bb, rule, anti);
                anti = replacement;
              }
            }

            if (Mode == DerivativeMode::ReverseModeCombined ||
                (Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
                (Mode == DerivativeMode::ReverseModeGradient &&
                 backwardsShadow) ||
                (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow)) {
              if (!inLoop) {
                assert(zeroed);
              }
            }
          }
          gutils->invertedPointers.insert(
              std::make_pair(&call, InvertedPointerVH(gutils, anti)));
        }
      endAnti:;
        if (((Mode == DerivativeMode::ReverseModeCombined && shouldFree()) ||
             (Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
             (Mode == DerivativeMode::ForwardModeSplit && shouldFree())) &&
            !isAlloca) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          assert(anti);
          Value *tofree = lookup(anti, Builder2);
          assert(tofree);
          assert(tofree->getType());
          for (size_t i = 0; i < gutils->getWidth(); i++) {
            Value *tofree_i =
                gutils->getWidth() == 1
                    ? tofree
                    : GradientUtils::extractMeta(Builder2, tofree, i);

            auto CI = freeKnownAllocation(Builder2, tofree_i, funcName, dbgLoc,
                                          gutils->TLI, &call, gutils);
            if (CI) {
              CI->addAttributeAtIndex(AttributeList::FirstArgIndex,
                                      Attribute::NonNull);
              bool combined = Mode == DerivativeMode::ReverseModeCombined;
              auto ident = MDNode::getDistinct(
                  CI->getContext(),
                  {ConstantAsMetadata::get(
                      combined ? ConstantInt::getTrue(CI->getContext())
                               : ConstantInt::getFalse(CI->getContext()))});
              Value *anti_i =
                  gutils->getWidth() == 1
                      ? anti
                      : GradientUtils::extractMeta(Builder2, anti, i);
              cast<Instruction>(anti_i)->setMetadata(
                  "enzyme_cache_alloc", MDNode::get(CI->getContext(), {ident}));
              CI->setMetadata("enzyme_cache_free",
                              MDNode::get(CI->getContext(), {ident}));
            }
          }
        }
      } else if (Mode == DerivativeMode::ForwardMode ||
                 Mode == DerivativeMode::ForwardModeError) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);

        SmallVector<Value *, 2> args;
        for (unsigned i = 0; i < call.arg_size(); ++i) {
          auto arg = call.getArgOperand(i);
          args.push_back(gutils->getNewFromOriginal(arg));
        }

        uint64_t idx = 0;
        Value *prev = gutils->getNewFromOriginal(&call);
        auto rule = [&]() {
          SmallVector<ValueType, 2> BundleTypes(args.size(), ValueType::Primal);

          auto Defs = gutils->getInvertedBundles(&call, BundleTypes, Builder2,
                                                 /*lookup*/ false);

          CallInst *CI = Builder2.CreateCall(
              call.getFunctionType(), call.getCalledFunction(), args, Defs);
          CI->setAttributes(call.getAttributes());
          CI->setCallingConv(call.getCallingConv());
          CI->setTailCallKind(call.getTailCallKind());
          CI->setDebugLoc(dbgLoc);

          if (funcName == "julia.gc_alloc_obj" ||
              funcName == "jl_gc_alloc_typed" ||
              funcName == "ijl_gc_alloc_typed") {
            if (EnzymeShadowAllocRewrite) {
              bool used = unnecessaryInstructions.find(&call) ==
                          unnecessaryInstructions.end();
              EnzymeShadowAllocRewrite(wrap(CI), gutils, wrap(&call), idx,
                                       wrap(prev), used);
            }
          }
          idx++;
          prev = CI;
          return CI;
        };

        Value *CI = applyChainRule(call.getType(), Builder2, rule);

        auto found = gutils->invertedPointers.find(&call);
        PHINode *placeholder = cast<PHINode>(&*found->second);

        gutils->invertedPointers.erase(found);
        gutils->replaceAWithB(placeholder, CI);
        gutils->erase(placeholder);
        gutils->invertedPointers.insert(
            std::make_pair(&call, InvertedPointerVH(gutils, CI)));
      }
    }

    // Cache and rematerialization irrelevant for forward mode.
    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      eraseIfUnused(call);
      return true;
    }

    std::map<UsageKey, bool> Seen = gutils->populateSeenFromKnownRecompute();
    bool primalNeededInReverse =
        Mode == DerivativeMode::ForwardMode ||
                Mode == DerivativeMode::ForwardModeError
            ? false
            : DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, &call, Mode, Seen, oldUnreachable);

    // If we explicitly decided we need this in the reverse pass, mark it as
    // such.
    {
      auto found = gutils->knownRecomputeHeuristic.find(&call);
      if (found != gutils->knownRecomputeHeuristic.end() && !found->second) {
        primalNeededInReverse = true;
      }
    }
    bool cacheWholeAllocation = gutils->needsCacheWholeAllocation(&call);
    if (cacheWholeAllocation) {
      primalNeededInReverse = true;
    }

    auto restoreFromStack = [&](MDNode *MD) {
      IRBuilder<> B(newCall);
      Value *Size;
      if (funcName == "malloc")
        Size = call.getArgOperand(0);
      else if (funcName == "julia.gc_alloc_obj" ||
               funcName == "jl_gc_alloc_typed" ||
               funcName == "ijl_gc_alloc_typed")
        Size = call.getArgOperand(1);
      else
        llvm_unreachable("Unknown allocation to upgrade");
      Size = gutils->getNewFromOriginal(Size);

      if (isa<ConstantInt>(Size)) {
        B.SetInsertPoint(gutils->inversionAllocs);
      }
      Type *elTy = Type::getInt8Ty(call.getContext());
      if (MD->getNumOperands() == 2) {
        elTy = (Type *)cast<ConstantInt>(
                   cast<ConstantAsMetadata>(MD->getOperand(1))->getValue())
                   ->getLimitedValue();
        Value *tsize = ConstantInt::get(Size->getType(),
                                        (gutils->newFunc->getParent()
                                             ->getDataLayout()
                                             .getTypeAllocSizeInBits(elTy) +
                                         7) /
                                            8);
        Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
      }
      Instruction *I = nullptr;
#if LLVM_VERSION_MAJOR < 17
      if (call.getContext().supportsTypedPointers()) {
        for (auto U : call.users()) {
          if (hasMetadata(cast<Instruction>(U), "enzyme_caststack")) {
            if (MD->getNumOperands() == 1) {
              elTy = U->getType()->getPointerElementType();
              Value *tsize = ConstantInt::get(
                  Size->getType(), (gutils->newFunc->getParent()
                                        ->getDataLayout()
                                        .getTypeAllocSizeInBits(elTy) +
                                    7) /
                                       8);

              Size = B.CreateUDiv(Size, tsize, "", /*exact*/ true);
            }
            I = gutils->getNewFromOriginal(cast<Instruction>(U));
            break;
          }
        }
      }
#endif
      Value *replacement = B.CreateAlloca(elTy, Size);
      for (auto MD : {"enzyme_active", "enzyme_inactive", "enzyme_type",
                      "enzymejl_allocart", "enzymejl_allocart_name",
                      "enzymejl_gc_alloc_rt"})
        if (auto M = call.getMetadata(MD))
          cast<AllocaInst>(replacement)->setMetadata(MD, M);
      if (I)
        replacement->takeName(I);
      else
        replacement->takeName(newCall);
      auto Alignment =
          cast<ConstantInt>(
              cast<ConstantAsMetadata>(MD->getOperand(0))->getValue())
              ->getLimitedValue();
      // Don't set zero alignment
      if (Alignment) {
        cast<AllocaInst>(replacement)->setAlignment(Align(Alignment));
      }
#if LLVM_VERSION_MAJOR < 17
      if (call.getContext().supportsTypedPointers()) {
        if (call.getType()->getPointerElementType() != elTy)
          replacement = B.CreatePointerCast(
              replacement, getUnqual(call.getType()->getPointerElementType()));
      }
#endif
      auto PT = cast<PointerType>(call.getType());
      if (PT->getAddressSpace()) {
        replacement = B.CreateAddrSpaceCast(replacement, PT);
        cast<Instruction>(replacement)
            ->setMetadata("enzyme_backstack",
                          MDNode::get(replacement->getContext(), {}));
      }
      gutils->replaceAWithB(newCall, replacement);
      gutils->erase(newCall);
    };

    // Don't erase any allocation that is being rematerialized.
    {
      auto found = gutils->rematerializableAllocations.find(&call);
      if (found != gutils->rematerializableAllocations.end()) {
        // If rematerializing (e.g. needed in reverse, but not needing
        //  the whole allocation):
        if (gutils->allocationsToBeRematerialized.count(&call)) {
          assert(!unnecessaryValues.count(&call));
          // if rematerialize, don't ever cache and downgrade to stack
          // allocation where possible. Note that for allocations which are
          // within a loop, we will create the rematerialized allocation in the
          // rematerialied loop. Note that what matters here is whether the
          // actual call itself here is inside the loop, not whether the
          // rematerialization is loop level. This is because one can have a
          // loop level cache, but a function level allocation (e.g. for stack
          // allocas). If we deleted it here, we would have no allocation!
          auto AllocationLoop = gutils->OrigLI->getLoopFor(call.getParent());
          // An allocation within a loop, must definitionally be a loop level
          // allocation (but not always the other way around.
          if (AllocationLoop)
            assert(found->second.LI);
          if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop) {
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            } else {
              restoreFromStack(MD);
            }
            return true;
          }

          // No need to free GC.
          if (EnzymeJuliaAddrLoad && isa<PointerType>(call.getType()) &&
              cast<PointerType>(call.getType())->getAddressSpace() == 10) {
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop)
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            return true;
          }

          // Otherwise if in reverse pass, free the newly created allocation.
          if (Mode == DerivativeMode::ReverseModeGradient ||
              Mode == DerivativeMode::ReverseModeCombined ||
              Mode == DerivativeMode::ForwardModeSplit) {
            IRBuilder<> Builder2(&call);
            getReverseBuilder(Builder2);
            auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
            auto freecall = freeKnownAllocation(
                Builder2, lookup(newCall, Builder2), funcName, dbgLoc,
                gutils->TLI, &call, gutils);
            if (freecall) {
              auto ident = MDNode::getDistinct(
                  freecall->getContext(),
                  {ConstantAsMetadata::get(
                      ConstantInt::getTrue(freecall->getContext()))});
              newCall->setMetadata(
                  "enzyme_cache_alloc",
                  MDNode::get(freecall->getContext(), {ident}));
              freecall->setMetadata(
                  "enzyme_cache_free",
                  MDNode::get(freecall->getContext(), {ident}));
            }
            if (Mode == DerivativeMode::ReverseModeGradient && AllocationLoop)
              gutils->rematerializedPrimalOrShadowAllocations.push_back(
                  newCall);
            return true;
          }
          // If in primal, do nothing (keeping the original caching behavior)
          if (Mode == DerivativeMode::ReverseModePrimal)
            return true;
        } else if (!cacheWholeAllocation) {
          if (unnecessaryValues.count(&call)) {
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
            return true;
          }
          // If not caching allocation and not needed in the reverse, we can
          // use the original freeing behavior for the function. If in the
          // reverse pass we should not recreate this allocation.
          if (Mode == DerivativeMode::ReverseModeGradient)
            eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          else if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
            restoreFromStack(MD);
          }
          return true;
        }
      }
    }

    // If an allocation is not needed in the reverse, maintain the original
    // free behavior and do not rematerialize this for the reverse. However,
    // this is only safe to perform for allocations with a guaranteed free
    // as can we can only guarantee that we don't erase those frees.
    bool hasPDFree = gutils->allocationsWithGuaranteedFree.count(&call);
    if (!primalNeededInReverse && hasPDFree) {
      if (unnecessaryValues.count(&call)) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (Mode == DerivativeMode::ReverseModeGradient ||
          Mode == DerivativeMode::ForwardModeSplit) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else {
        if (auto MD = hasMetadata(&call, "enzyme_fromstack")) {
          restoreFromStack(MD);
        }
      }
      return true;
    }

    // If an object is managed by the GC do not preserve it for later free,
    // Thus it only needs caching if there is a need for it in the reverse.
    if (EnzymeJuliaAddrLoad && isa<PointerType>(call.getType()) &&
        cast<PointerType>(call.getType())->getAddressSpace() == 10) {
      if (!subretused) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
      if (!primalNeededInReverse) {
        if (Mode == DerivativeMode::ReverseModeGradient ||
            Mode == DerivativeMode::ForwardModeSplit) {
          auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                       call.getName() + "_replacementJ");
          gutils->fictiousPHIs[pn] = &call;
          gutils->replaceAWithB(newCall, pn);
          gutils->erase(newCall);
        }
      } else if (Mode != DerivativeMode::ReverseModeCombined) {
        gutils->cacheForReverse(BuilderZ, newCall,
                                getIndex(&call, CacheType::Self, BuilderZ));
      }
      return true;
    }

    if (EnzymeFreeInternalAllocations)
      hasPDFree = true;

    // TODO enable this if we need to free the memory
    // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
    // TO FREE'ing
    if ((primalNeededInReverse &&
         !gutils->unnecessaryIntermediates.count(&call)) ||
        hasPDFree) {
      if (hasPDFree && Mode == DerivativeMode::ReverseModePrimal) {
        auto ident =
            MDNode::getDistinct(newCall->getContext(),
                                {ConstantAsMetadata::get(ConstantInt::getFalse(
                                    newCall->getContext()))});
        newCall->setMetadata("enzyme_cache_alloc",
                             MDNode::get(newCall->getContext(), {ident}));
      }
      Value *nop = gutils->cacheForReverse(
          BuilderZ, newCall, getIndex(&call, CacheType::Self, BuilderZ));
      if (hasPDFree &&
          ((Mode == DerivativeMode::ReverseModeGradient && shouldFree()) ||
           Mode == DerivativeMode::ReverseModeCombined ||
           (Mode == DerivativeMode::ForwardModeSplit && shouldFree()))) {
        IRBuilder<> Builder2(&call);
        getReverseBuilder(Builder2);
        auto dbgLoc = gutils->getNewFromOriginal(call.getDebugLoc());
        auto freecall =
            freeKnownAllocation(Builder2, lookup(nop, Builder2), funcName,
                                dbgLoc, gutils->TLI, &call, gutils);
        if (freecall) {
          bool combined = Mode == DerivativeMode::ReverseModeCombined;
          auto ident = MDNode::getDistinct(
              freecall->getContext(),
              {ConstantAsMetadata::get(
                  combined ? ConstantInt::getTrue(freecall->getContext())
                           : ConstantInt::getFalse(freecall->getContext()))});
          if (combined)
            newCall->setMetadata("enzyme_cache_alloc",
                                 MDNode::get(freecall->getContext(), {ident}));
          freecall->setMetadata("enzyme_cache_free",
                                MDNode::get(freecall->getContext(), {ident}));
        }
      }
    } else if (Mode == DerivativeMode::ReverseModeGradient ||
               Mode == DerivativeMode::ReverseModeCombined ||
               Mode == DerivativeMode::ForwardModeSplit) {
      // Note that here we cannot simply replace with null as users who
      // try to find the shadow pointer will use the shadow of null rather
      // than the true shadow of this
      auto pn = BuilderZ.CreatePHI(call.getType(), 1,
                                   call.getName() + "_replacementB");
      gutils->fictiousPHIs[pn] = &call;
      gutils->replaceAWithB(newCall, pn);
      gutils->erase(newCall);
    }

    return true;
  }

  if (funcName == "julia.gc_loaded") {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }
    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    if (auto placeholder = dyn_cast<PHINode>(&*ifound->second)) {

      bool needShadow = DifferentialUseAnalysis::is_value_needed_in_reverse<
          QueryType::Shadow>(gutils, &call, Mode, oldUnreachable);
      if (!needShadow) {
        gutils->invertedPointers.erase(ifound);
        gutils->erase(placeholder);
        eraseIfUnused(call);
        return true;
      }

      gutils->invertedPointers.erase(ifound);
      auto res = gutils->invertPointerM(&call, BuilderZ);

      gutils->replaceAWithB(placeholder, res);
      gutils->erase(placeholder);
    }
    eraseIfUnused(call);

    return true;
  }

  if (funcName == "julia.pointer_from_objref") {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }

    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    auto placeholder = cast<PHINode>(&*ifound->second);

    bool needShadow =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, &call, Mode, oldUnreachable);
    if (!needShadow) {
      gutils->invertedPointers.erase(ifound);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return true;
    }

    Value *ptrshadow = gutils->invertPointerM(call.getArgOperand(0), BuilderZ);

    Value *val = applyChainRule(
        call.getType(), BuilderZ,
        [&](Value *v) -> Value * { return BuilderZ.CreateCall(called, {v}); },
        ptrshadow);

    gutils->replaceAWithB(placeholder, val);
    gutils->erase(placeholder);
    eraseIfUnused(call);
    return true;
  }
  if (funcName.contains("__enzyme_todense")) {
    if (gutils->isConstantValue(&call)) {
      eraseIfUnused(call);
      return true;
    }

    auto ifound = gutils->invertedPointers.find(&call);
    assert(ifound != gutils->invertedPointers.end());

    auto placeholder = cast<PHINode>(&*ifound->second);

    bool needShadow =
        DifferentialUseAnalysis::is_value_needed_in_reverse<QueryType::Shadow>(
            gutils, &call, Mode, oldUnreachable);
    if (!needShadow) {
      gutils->invertedPointers.erase(ifound);
      gutils->erase(placeholder);
      eraseIfUnused(call);
      return true;
    }

    // The shadow pointer is built from the shadows of the extra arguments.
    // An inactive argument (e.g. an index) is passed as is to every lane.
    SmallVector<Value *, 3> args;
    SmallVector<bool, 3> argIsShadow;
    for (size_t i = 0; i < 2; i++) {
      args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
      argIsShadow.push_back(false);
    }
    for (size_t i = 2; i < call.arg_size(); ++i) {
      auto arg = call.getArgOperand(i);
      if (gutils->isConstantValue(arg)) {
        args.push_back(gutils->getNewFromOriginal(arg));
        argIsShadow.push_back(false);
      } else {
        args.push_back(gutils->invertPointerM(arg, BuilderZ));
        argIsShadow.push_back(true);
      }
    }

    Value *res = UndefValue::get(gutils->getShadowType(call.getType()));
    if (gutils->getWidth() == 1) {
      res = BuilderZ.CreateCall(called, args);
    } else {
      for (size_t w = 0; w < gutils->getWidth(); ++w) {
        SmallVector<Value *, 3> targs = {args[0], args[1]};
        for (size_t i = 2; i < call.arg_size(); ++i)
          targs.push_back(argIsShadow[i]
                              ? GradientUtils::extractMeta(BuilderZ, args[i], w)
                              : args[i]);

        auto tres = BuilderZ.CreateCall(called, targs);
        res = BuilderZ.CreateInsertValue(res, tres, w);
      }
    }

    gutils->replaceAWithB(placeholder, res);
    gutils->erase(placeholder);
    eraseIfUnused(call);
    return true;
  }

  if (funcName == "memcpy" || funcName == "memmove") {
    auto ID = (funcName == "memcpy") ? Intrinsic::memcpy : Intrinsic::memmove;
    visitMemTransferCommon(ID, /*srcAlign*/ MaybeAlign(1),
                           /*dstAlign*/ MaybeAlign(1), call,
                           call.getArgOperand(0), call.getArgOperand(1),
                           gutils->getNewFromOriginal(call.getArgOperand(2)),
                           ConstantInt::getFalse(call.getContext()));
    return true;
  }
  if (funcName == "memset" || funcName == "memset_pattern16" ||
      funcName == "__memset_chk") {
    visitMemSetCommon(call);
    return true;
  }
  if (funcName == "enzyme_zerotype") {
    IRBuilder<> BuilderZ(&call);
    getForwardBuilder(BuilderZ);

    bool backwardsShadow = false;
    bool forwardsShadow = true;
    for (auto pair : gutils->backwardsOnlyShadows) {
      if (pair.second.stores.count(&call)) {
        backwardsShadow = true;
        forwardsShadow = pair.second.primalInitialize;
        if (auto inst = dyn_cast<Instruction>(pair.first))
          if (!forwardsShadow && pair.second.LI &&
              pair.second.LI->contains(inst->getParent()))
            backwardsShadow = false;
      }
    }

    bool forceErase =
        !((Mode == DerivativeMode::ReverseModePrimal && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeCombined && forwardsShadow) ||
          (Mode == DerivativeMode::ReverseModeGradient && backwardsShadow) ||
          (Mode == DerivativeMode::ForwardModeSplit && backwardsShadow));

    if (forceErase)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    else
      eraseIfUnused(call);

    Value *orig_op0 = call.getArgOperand(0);

    // If constant destination then no operation needs doing
    if (gutils->isConstantValue(orig_op0)) {
      return true;
    }

    if (!forceErase) {
      Value *op0 = gutils->invertPointerM(orig_op0, BuilderZ);
      Value *op1 = gutils->getNewFromOriginal(call.getArgOperand(1));
      Value *op2 = gutils->getNewFromOriginal(call.getArgOperand(2));
      auto Defs = gutils->getInvertedBundles(
          &call, {ValueType::Shadow, ValueType::Primal, ValueType::Primal},
          BuilderZ, /*lookup*/ false);

      applyChainRule(
          BuilderZ,
          [&](Value *op0) {
            SmallVector<Value *, 4> args = {op0, op1, op2};
            auto cal =
                BuilderZ.CreateCall(call.getCalledFunction(), args, Defs);
            llvm::SmallVector<unsigned int, 9> ToCopy2(MD_ToCopy);
            ToCopy2.push_back(LLVMContext::MD_noalias);
            cal->copyMetadata(call, ToCopy2);
            cal->setAttributes(call.getAttributes());
            if (auto m = hasMetadata(&call, "enzyme_zerostack"))
              cal->setMetadata("enzyme_zerostack", m);
            cal->setCallingConv(call.getCallingConv());
            cal->setTailCallKind(call.getTailCallKind());
            cal->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));
          },
          op0);
    }
    return true;
  }
  if (funcName == "cuStreamCreate") {
    Value *val = nullptr;
    llvm::Type *PT = getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 17
    if (call.getContext().supportsTypedPointers()) {
      if (isa<PointerType>(call.getArgOperand(0)->getType()))
        PT = call.getArgOperand(0)->getType()->getPointerElementType();
    }
#endif
    if (Mode == DerivativeMode::ReverseModePrimal ||
        Mode == DerivativeMode::ReverseModeCombined) {
      val = gutils->getNewFromOriginal(call.getOperand(0));
      if (!isa<PointerType>(val->getType()))
        val = BuilderZ.CreateIntToPtr(val, getUnqual(PT));
      val = BuilderZ.CreateLoad(PT, val);
      val = gutils->cacheForReverse(BuilderZ, val,
                                    getIndex(&call, CacheType::Tape, BuilderZ));

    } else if (Mode == DerivativeMode::ReverseModeGradient) {
      PHINode *toReplace =
          BuilderZ.CreatePHI(PT, 1, call.getName() + "_psxtmp");
      val = gutils->cacheForReverse(BuilderZ, toReplace,
                                    getIndex(&call, CacheType::Tape, BuilderZ));
    }
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      val = gutils->lookupM(val, Builder2);
      auto FreeFunc = getOrInsertPerCallingConv(
          *gutils->newFunc->getParent(), called, "cuStreamDestroy",
          FunctionType::get(call.getType(), {PT}, false));
      Value *nargs[] = {val};
      Builder2.CreateCall(FreeFunc, nargs);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "cuStreamDestroy") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "cuStreamSynchronize") {
    if (Mode == DerivativeMode::ReverseModeGradient ||
        Mode == DerivativeMode::ReverseModeCombined) {
      IRBuilder<> Builder2(&call);
      getReverseBuilder(Builder2);
      Value *nargs[] = {gutils->lookupM(
          gutils->getNewFromOriginal(call.getOperand(0)), Builder2)};
      auto callval = call.getCalledOperand();
      Builder2.CreateCall(call.getFunctionType(), callval, nargs);
    }
    if (Mode == DerivativeMode::ReverseModeGradient)
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }
  if (funcName == "posix_memalign" || funcName == "cuMemAllocAsync" ||
      funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2" ||
      funcName == "cudaMalloc" || funcName == "cudaMallocAsync" ||
      funcName == "cudaMallocHost" || funcName == "cudaMallocFromPoolAsync") {
    bool constval = gutils->isConstantInstruction(&call);

    Value *val;
    llvm::Type *PT = getInt8PtrTy(call.getContext());
#if LLVM_VERSION_MAJOR < 17
    if (call.getContext().supportsTypedPointers()) {
      if (isa<PointerType>(call.getArgOperand(0)->getType()))
        PT = call.getArgOperand(0)->getType()->getPointerElementType();
    }
#endif
    if (!constval) {
      Value *stream = nullptr;
      if (funcName == "cuMemAllocAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocFromPoolAsync")
        stream = gutils->getNewFromOriginal(call.getArgOperand(3));

      auto M = gutils->newFunc->getParent();

      if (Mode == DerivativeMode::ReverseModePrimal ||
          Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ForwardMode ||
          Mode == DerivativeMode::ForwardModeError) {
        Value *ptrshadow =
            gutils->invertPointerM(call.getArgOperand(0), BuilderZ);
        SmallVector<Value *, 1> args;
        SmallVector<ValueType, 1> valtys;
        args.push_back(ptrshadow);
        valtys.push_back(ValueType::Shadow);
        for (size_t i = 1; i < call.arg_size(); ++i) {
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
          valtys.push_back(ValueType::Primal);
        }

        auto Defs = gutils->getInvertedBundles(&call, valtys, BuilderZ,
                                               /*lookup*/ false);

        val = applyChainRule(
            PT, BuilderZ,
            [&](Value *ptrshadow) {
              args[0] = ptrshadow;

              BuilderZ.CreateCall(called, args, Defs);
              if (!isa<PointerType>(ptrshadow->getType()))
                ptrshadow = BuilderZ.CreateIntToPtr(ptrshadow, getUnqual(PT));
              Value *val = BuilderZ.CreateLoad(PT, ptrshadow);

              auto dst_arg =
                  BuilderZ.CreateBitCast(val, getInt8PtrTy(call.getContext()));

              auto val_arg =
                  ConstantInt::get(Type::getInt8Ty(call.getContext()), 0);
              auto len_arg = gutils->getNewFromOriginal(
                  call.getArgOperand((funcName == "posix_memalign") ? 2 : 1));

              if (funcName == "posix_memalign" ||
                  funcName == "cudaMallocHost") {
                BuilderZ.CreateMemSet(dst_arg, val_arg, len_arg, MaybeAlign());
              } else if (funcName == "cudaMalloc") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cudaMemset",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cudaMallocAsync" ||
                         funcName == "cudaMallocFromPoolAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cudaMemsetAsync",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAllocAsync") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType(),
                               stream->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called, "cuMemsetD8Async",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg, stream};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else if (funcName == "cuMemAlloc" ||
                         funcName == "cuMemAlloc_v2") {
                Type *tys[] = {PT, val_arg->getType(), len_arg->getType()};
                auto F = getOrInsertPerCallingConv(
                    *M, called,
                    funcName == "cuMemAlloc_v2" ? "cuMemsetD8_v2"
                                                : "cuMemsetD8",
                    FunctionType::get(call.getType(), tys, false));
                Value *nargs[] = {dst_arg, val_arg, len_arg};
                auto memset = cast<CallInst>(BuilderZ.CreateCall(F, nargs));
                memset->addParamAttr(0, Attribute::NonNull);
              } else {
                llvm_unreachable("unhandled allocation");
              }
              return val;
            },
            ptrshadow);

        if (Mode != DerivativeMode::ForwardMode &&
            Mode != DerivativeMode::ForwardModeError)
          val = gutils->cacheForReverse(
              BuilderZ, val, getIndex(&call, CacheType::Tape, BuilderZ));
      } else if (Mode == DerivativeMode::ReverseModeGradient) {
        PHINode *toReplace = BuilderZ.CreatePHI(gutils->getShadowType(PT), 1,
                                                call.getName() + "_psxtmp");
        val = gutils->cacheForReverse(
            BuilderZ, toReplace, getIndex(&call, CacheType::Tape, BuilderZ));
      }

      if (Mode == DerivativeMode::ReverseModeCombined ||
          Mode == DerivativeMode::ReverseModeGradient) {
        if (shouldFree()) {
          IRBuilder<> Builder2(&call);
          getReverseBuilder(Builder2);
          Value *tofree = gutils->lookupM(val, Builder2, ValueToValueMapTy(),
                                          /*tryLegalRecompute*/ false);

          Type *VoidTy = Type::getVoidTy(M->getContext());
          Type *IntPtrTy = getInt8PtrTy(M->getContext());

          Value *streamL = nullptr;
          if (stream)
            streamL = gutils->lookupM(stream, Builder2);

          applyChainRule(
              BuilderZ,
              [&](Value *tofree) {
                if (funcName == "posix_memalign") {
                  auto FreeFunc =
                      M->getOrInsertFunction("free", VoidTy, IntPtrTy);
                  Builder2.CreateCall(FreeFunc, tofree);
                } else if (funcName == "cuMemAllocAsync") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cuMemFreeAsync",
                      FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()},
                                        false));
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cuMemAlloc" ||
                           funcName == "cuMemAlloc_v2") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cuMemFree",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMalloc") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFree",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocAsync" ||
                           funcName == "cudaMallocFromPoolAsync") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFreeAsync",
                      FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()},
                                        false));
                  Value *nargs[] = {tofree, streamL};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else if (funcName == "cudaMallocHost") {
                  auto FreeFunc = getOrInsertPerCallingConv(
                      *M, called, "cudaFreeHost",
                      FunctionType::get(VoidTy, {IntPtrTy}, false));
                  Value *nargs[] = {tofree};
                  Builder2.CreateCall(FreeFunc, nargs);
                } else
                  llvm_unreachable("unknown function to free");
              },
              tofree);
        }
      }
    }

    // TODO enable this if we need to free the memory
    // NOTE THAT TOPLEVEL IS THERE SIMPLY BECAUSE THAT WAS PREVIOUS ATTITUTE
    // TO FREE'ing
    if (Mode == DerivativeMode::ReverseModeGradient) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    } else if (Mode == DerivativeMode::ReverseModePrimal) {
      // if (is_value_needed_in_reverse<Primal>(
      //        TR, gutils, orig, /*topLevel*/ Mode ==
      //        DerivativeMode::Both))
      //        {

      //  gutils->cacheForReverse(BuilderZ, newCall,
      //                          getIndex(orig, CacheType::Self, BuilderZ));
      //} else if (Mode != DerivativeMode::Forward) {
      // Note that here we cannot simply replace with null as users who try
      // to find the shadow pointer will use the shadow of null rather than
      // the true shadow of this
      //}
    } else if (Mode == DerivativeMode::ReverseModeCombined && shouldFree()) {
      IRBuilder<> Builder2(newCall->getNextNode());
      auto ptrv = gutils->getNewFromOriginal(call.getOperand(0));
      if (!isa<PointerType>(ptrv->getType()))
        ptrv = BuilderZ.CreateIntToPtr(ptrv, getUnqual(PT));
      auto load = Builder2.CreateLoad(PT, ptrv, "posix_preread");
      Builder2.SetInsertPoint(&call);
      getReverseBuilder(Builder2);
      auto tofree = gutils->lookupM(load, Builder2, ValueToValueMapTy(),
                                    /*tryLegal*/ false);
      Value *streamL = nullptr;
      if (funcName == "cuMemAllocAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(2));
      else if (funcName == "cudaMallocFromPoolAsync")
        streamL = gutils->getNewFromOriginal(call.getArgOperand(3));
      if (streamL)
        streamL = gutils->lookupM(streamL, Builder2);

      auto M = gutils->newFunc->getParent();
      Type *VoidTy = Type::getVoidTy(M->getContext());
      Type *IntPtrTy = getInt8PtrTy(M->getContext());

      if (funcName == "posix_memalign") {
        auto FreeFunc = M->getOrInsertFunction("free", VoidTy, IntPtrTy);
        Builder2.CreateCall(FreeFunc, tofree);
      } else if (funcName == "cuMemAllocAsync") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cuMemFreeAsync",
            FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()}, false));
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cuMemAlloc" || funcName == "cuMemAlloc_v2") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cuMemFree",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMalloc") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFree",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocAsync" ||
                 funcName == "cudaMallocFromPoolAsync") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFreeAsync",
            FunctionType::get(VoidTy, {IntPtrTy, streamL->getType()}, false));
        Value *nargs[] = {tofree, streamL};
        Builder2.CreateCall(FreeFunc, nargs);
      } else if (funcName == "cudaMallocHost") {
        auto FreeFunc = getOrInsertPerCallingConv(
            *M, called, "cudaFreeHost",
            FunctionType::get(VoidTy, {IntPtrTy}, false));
        Value *nargs[] = {tofree};
        Builder2.CreateCall(FreeFunc, nargs);
      } else
        llvm_unreachable("unknown function to free");
    }

    return true;
  }

  // Remove free's in forward pass so the memory can be used in the reverse
  // pass
  if (isDeallocationFunction(funcName, gutils->TLI)) {
    assert(gutils->invertedPointers.find(&call) ==
           gutils->invertedPointers.end());

    if (Mode == DerivativeMode::ForwardMode ||
        Mode == DerivativeMode::ForwardModeError) {
      if (!gutils->isConstantValue(call.getArgOperand(0))) {
        IRBuilder<> Builder2(&call);
        getForwardBuilder(Builder2);
        auto origfree = call.getArgOperand(0);
        auto newfree = gutils->getNewFromOriginal(call.getArgOperand(0));
        auto tofree = gutils->invertPointerM(origfree, Builder2);

        Function *free = getOrInsertCheckedFree(
            *call.getModule(), &call, newfree->getType(), gutils->getWidth());

        bool used = true;
        if (auto instArg = dyn_cast<Instruction>(call.getArgOperand(0)))
          used = unnecessaryInstructions.find(instArg) ==
                 unnecessaryInstructions.end();

        SmallVector<Value *, 3> args;
        if (used)
          args.push_back(newfree);
        else
          args.push_back(
              Constant::getNullValue(call.getArgOperand(0)->getType()));

        auto rule = [&args](Value *tofree) { args.push_back(tofree); };
        applyChainRule(Builder2, rule, tofree);

        for (size_t i = 1; i < call.arg_size(); i++) {
          args.push_back(gutils->getNewFromOriginal(call.getArgOperand(i)));
        }

        auto frees = Builder2.CreateCall(free->getFunctionType(), free, args);
        frees->setDebugLoc(gutils->getNewFromOriginal(call.getDebugLoc()));

        eraseIfUnused(call);
        return true;
      }
      eraseIfUnused(call);
    }
    auto callval = call.getCalledOperand();

    for (auto rmat : gutils->backwardsOnlyShadows) {
      if (gutils->allocationsToBeRematerialized.count(rmat.first) &&
          rmat.second.frees.count(&call)) {
        bool shouldFree = false;
        if (rmat.second.primalInitialize) {
          if (Mode == DerivativeMode::ReverseModePrimal)
            shouldFree = true;
        }

        if (shouldFree) {
          IRBuilder<> Builder2(&call);
          getForwardBuilder(Builder2);
          auto origfree = call.getArgOperand(0);
          auto tofree = gutils->invertPointerM(origfree, Builder2);
          if (tofree != origfree) {
            SmallVector<Value *, 2> args = {tofree};
            CallInst *CI =
                Builder2.CreateCall(call.getFunctionType(), callval, args);
            CI->setAttributes(call.getAttributes());
          }
        }
        break;
      }
    }

    // If a rematerializable allocation.
    for (auto rmat : gutils->rematerializableAllocations) {
      if (gutils->allocationsToBeRematerialized.count(rmat.first) &&
          rmat.second.frees.count(&call)) {
        // Leave the original free behavior since this won't be used
        // in the reverse pass in split mode
        if (Mode == DerivativeMode::ReverseModePrimal) {
          eraseIfUnused(call);
          return true;
        } else if (Mode == DerivativeMode::ReverseModeGradient) {
          eraseIfUnused(call, /*erase*/ true, /*check*/ false);
          return true;
        } else {
          assert(Mode == DerivativeMode::ReverseModeCombined);
          std::map<UsageKey, bool> Seen =
              gutils->populateSeenFromKnownRecompute();
          bool primalNeededInReverse =
              DifferentialUseAnalysis::is_value_needed_in_reverse<
                  QueryType::Primal>(gutils, rmat.first, Mode, Seen,
                                     oldUnreachable);
          {
            auto found = gutils->knownRecomputeHeuristic.find(rmat.first);
            if (found != gutils->knownRecomputeHeuristic.end()) {
              if (!found->second) {
                primalNeededInReverse = true;
              }
            }
          }
          bool cacheWholeAllocation =
              gutils->needsCacheWholeAllocation(rmat.first);
          if (cacheWholeAllocation) {
            primalNeededInReverse = true;
          }
          // If in a loop context, maintain the same free behavior, unless
          // caching whole allocation.
          if (!cacheWholeAllocation) {
            eraseIfUnused(call);
            return true;
          }
          assert(!unnecessaryValues.count(rmat.first));
          (void)primalNeededInReverse;
          assert(primalNeededInReverse);
        }
      }
    }

    if (gutils->forwardDeallocations.count(&call)) {
      if (Mode == DerivativeMode::ReverseModeGradient) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      } else
        eraseIfUnused(call);
      return true;
    }

    if (gutils->postDominatingFrees.count(&call)) {
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    if (call.getMetadata("enzyme_cache_free")) {
      bool hasGuaranteedFree = false;
      for (const auto &pair : gutils->allocationsWithGuaranteedFree) {
        if (pair.second.count(&call)) {
          hasGuaranteedFree = true;
          break;
        }
      }
      if (!hasGuaranteedFree) {
        eraseIfUnused(call, /*erase*/ true, /*check*/ false);
        return true;
      }
    }

    llvm::Value *val = getBaseObject(call.getArgOperand(0));
    if (isa<ConstantPointerNull>(val)) {
      llvm::errs() << "removing free of null pointer\n";
      eraseIfUnused(call, /*erase*/ true, /*check*/ false);
      return true;
    }

    // TODO HANDLE FREE
    llvm::errs() << "freeing without malloc " << *val << " in " << call << "\n";
    eraseIfUnused(call, /*erase*/ true, /*check*/ false);
    return true;
  }

  if (call.hasFnAttr("enzyme_sample")) {
    if (Mode != DerivativeMode::ReverseModeCombined &&
        Mode != DerivativeMode::ReverseModeGradient)
      return true;

    bool constval = gutils->isConstantInstruction(&call);

    if (constval)
      return true;

    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);

    auto trace = call.getArgOperand(call.arg_size() - 1);
    auto address = call.getArgOperand(0);

    auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
    auto daddress = lookup(gutils->getNewFromOriginal(address), Builder2);

    Value *dchoice;
    if (TR.query(&call)[{-1}].isPossiblePointer()) {
      dchoice = gutils->invertPointerM(&call, Builder2);
    } else {
      dchoice = diffe(&call, Builder2);
    }

    if (call.hasMetadata("enzyme_gradient_setter")) {
      auto gradient_setter = cast<Function>(
          cast<ValueAsMetadata>(
              call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
              ->getValue());

      TraceUtils::InsertChoiceGradient(
          Builder2, gradient_setter->getFunctionType(), gradient_setter,
          daddress, dchoice, dtrace);
    }

    return true;
  }

  if (call.hasFnAttr("enzyme_insert_argument")) {
    IRBuilder<> Builder2(&call);
    getReverseBuilder(Builder2);

    auto name = call.getArgOperand(0);
    auto arg = call.getArgOperand(1);
    auto trace = call.getArgOperand(2);

    auto gradient_setter = cast<Function>(
        cast<ValueAsMetadata>(
            call.getMetadata("enzyme_gradient_setter")->getOperand(0).get())
            ->getValue());

    auto dtrace = lookup(gutils->getNewFromOriginal(trace), Builder2);
    auto dname = lookup(gutils->getNewFromOriginal(name), Builder2);
    Value *darg;

    if (TR.query(arg)[{-1}].isPossiblePointer()) {
      darg = gutils->invertPointerM(arg, Builder2);
    } else {
      darg = diffe(arg, Builder2);
    }

    TraceUtils::InsertArgumentGradient(Builder2,
                                       gradient_setter->getFunctionType(),
                                       gradient_setter, dname, darg, dtrace);
    return true;
  }

  return false;
}
