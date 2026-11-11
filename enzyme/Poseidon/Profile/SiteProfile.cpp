#include "SiteProfile.h"

#include "Instrument.h"
#include "Pass/Session.h"
#include "ProfileRead.h"
#include "Sites/Annotations.h"
#include "Sites/Canonicalize.h"
#include "Sites/HostSites.h"
#include "Sites/KernelSites.h"
#include "Sites/LaunchDescriptors.h"
#include "Sites/Markers.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/RaiseWMMA.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <map>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {

// Canonicalize, slot-number and instrument one site body exactly as profile
// use canonicalizes and slot-numbers it, so the profile is recorded against
// the form the solve will later see. Null if the body holds nothing profilable.
// `staticOut`, if given, receives the header the profile will carry.
Function *instrumentedClone(Session &S, Function &F, unsigned siteId,
                            std::string *staticOut = nullptr) {
  Function *clone = canonicalize(F);
  S.SiteClones.push_back(clone);
  setSlotMetadata(*clone);
  // Before the probes: they break the reduction shape the trip count is read
  // from.
  SmallVector<std::pair<size_t, unsigned>, 4> trips;
  collectScalarLoopReductionTrips(S.Opts, *clone, trips);
  if (!instrumentForProfiling(*clone, siteId)) {
    clone->eraseFromParent();
    return nullptr;
  }
  std::string staticData;
  {
    raw_string_ostream os(staticData);
    os << "SiteId = " << siteId << "\n";
    for (const auto &kv : trips)
      os << "RedTrip = " << kv.first << " " << kv.second << "\n";
    os << "CanonicalHash = " << canonicalFormHash(*clone) << "\n";
    os << kProfileFormatKey << " = " << kProfileFormat << "\n";
  }
  emitProfileStaticData(*clone, staticData);
  if (staticOut)
    *staticOut = staticData;
  return clone;
}

CallInst *withShadows(CallInst *CI, Function &clone, const MarkerArgs &marker);

} // namespace

// Profile generation. The marker becomes an ordinary reverse-mode derivative
// request over the instrumented clone: the custom derivative on each probe is
// what puts the gradient record inside the reverse pass, so the host's AD needs
// no profiling mode of its own.
bool profileSite(Session &S, CallInst *CI,
                 DenseMap<Function *, Function *> &clones,
                 unsigned &nextSiteId) {
  Function *F = markerTarget(CI);
  if (!F)
    return false;

  MarkerArgs marker;
  if (!parseMarker(CI, F, marker))
    return false;

  // Two annotations on the same body share one instrumented clone, so their
  // executions accumulate into one profile record, as they did when the clone
  // came from the host's preprocessing cache.
  Function *&clone = clones[F];
  if (!clone) {
    // Two markers on the same body share one profile record and therefore one
    // site id; the id counts clones in marker order.
    clone = instrumentedClone(S, *F, nextSiteId++);
    if (!clone) {
      warn(CI, Twine("no profilable operation in ") + F->getName());
      return false;
    }
  }
  if (!marker.madeShadows.empty()) {
    CI = withShadows(CI, *clone, marker);
    marker = MarkerArgs();
    if (!CI || !parseMarker(CI, F, marker))
      return false;
  }

  Module &M = *CI->getModule();
  FunctionType *FT = CI->getFunctionType();
  Function *autodiff = Function::Create(FT, GlobalValue::ExternalLinkage,
                                        "__enzyme_autodiff_poseidon", &M);
  autodiff->setCallingConv(CI->getCallingConv());

  SmallVector<Value *, 16> args;
  SmallVector<size_t, 16> from;
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  for (size_t i = 0; i < CI->arg_size(); ++i) {
    if (!marker.forward[i])
      continue;
    args.push_back(i == (size_t)sret ? clone : CI->getArgOperand(i));
    from.push_back(i);
  }

  IRBuilder<> Builder(CI);
  CallInst *adCall = Builder.CreateCall(FT, autodiff, args);
  adCall->setCallingConv(CI->getCallingConv());
  adCall->setDebugLoc(CI->getDebugLoc());
  AttributeList old = CI->getAttributes();
  adCall->setAttributes(AttributeList::get(M.getContext(), old.getFnAttrs(),
                                           old.getRetAttrs(), {}));
  for (size_t j = 0; j < from.size(); ++j)
    adCall->setAttributes(adCall->getAttributes().addParamAttributes(
        M.getContext(), j,
        AttrBuilder(M.getContext(), old.getParamAttrs(from[j]))));

  CI->replaceAllUsesWith(adCall);
  CI->eraseFromParent();
  return true;
}

namespace {

// The width of the floating-point scalar a store writes, 0 if it writes none.
unsigned fpStoreWidth(Type *T) {
  if (auto *VT = dyn_cast<VectorType>(T))
    T = VT->getElementType();
  if (T->isFloatingPointTy())
    return T->getScalarSizeInBits() / 8;
  if (auto *ST = dyn_cast<StructType>(T))
    for (Type *E : ST->elements())
      if (unsigned w = fpStoreWidth(E))
        return w;
  if (auto *AT = dyn_cast<ArrayType>(T))
    return fpStoreWidth(AT->getElementType());
  return 0;
}

// Which parameters the site writes through, and in what element width. An
// application does not say which of its buffers are outputs, and reverse mode
// needs to know: the output shadows are the ones the seed goes into.
void collectOutputParams(Function &clone,
                         SmallVectorImpl<unsigned> &seedWidth) {
  seedWidth.assign(clone.arg_size(), 0);
  auto mark = [&](Value *ptr, unsigned w) {
    int idx = traceToArgIndex(ptr);
    if (w && idx >= 0 && (size_t)idx < seedWidth.size())
      seedWidth[idx] = std::max(seedWidth[idx], w);
  };
  for (Instruction &I : instructions(clone)) {
    if (auto *SI = dyn_cast<StoreInst>(&I))
      mark(SI->getPointerOperand(),
           fpStoreWidth(SI->getValueOperand()->getType()));
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      mark(RMW->getPointerOperand(),
           fpStoreWidth(RMW->getValOperand()->getType()));
  }
}

bool holdsPointer(Type *T) {
  if (T->isPointerTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(T))
    return llvm::any_of(ST->elements(), holdsPointer);
  if (auto *AT = dyn_cast<ArrayType>(T))
    return holdsPointer(AT->getElementType());
  return false;
}

// Where an aggregate of type `T` holds pointers: the byte offset of each from
// the aggregate's start and the insertvalue path that reaches it, in layout
// order.
struct PointerField {
  uint64_t offset;
  SmallVector<unsigned, 4> path;
};
void pointerFields(const DataLayout &DL, Type *T, uint64_t offset,
                   SmallVectorImpl<unsigned> &path,
                   SmallVectorImpl<PointerField> &out) {
  if (T->isPointerTy()) {
    out.push_back({offset, SmallVector<unsigned, 4>(path.begin(), path.end())});
    return;
  }
  if (!holdsPointer(T))
    return;
  if (auto *ST = dyn_cast<StructType>(T)) {
    const StructLayout *SL = DL.getStructLayout(ST);
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      path.push_back(i);
      pointerFields(DL, ST->getElementType(i), offset + SL->getElementOffset(i),
                    path, out);
      path.pop_back();
    }
  } else if (auto *AT = dyn_cast<ArrayType>(T)) {
    uint64_t size = DL.getTypeAllocSize(AT->getElementType());
    for (uint64_t i = 0; i < AT->getNumElements(); ++i) {
      path.push_back((unsigned)i);
      pointerFields(DL, AT->getElementType(), offset + i * size, path, out);
      path.pop_back();
    }
  }
}

// The pointer fields of by-value aggregate parameters the site writes through,
// in the element width it writes: collectOutputParams for the pointers a
// kernel's lambda captured.
void collectOutputFields(Function &clone,
                         std::map<LaunchRef, unsigned> &seedWidth) {
  auto mark = [&](Value *ptr, unsigned w) {
    LaunchRef r = tracePointerRoot(ptr);
    if (w && r.isField())
      seedWidth[r] = std::max(seedWidth[r], w);
  };
  for (Instruction &I : instructions(clone)) {
    if (auto *SI = dyn_cast<StoreInst>(&I))
      mark(SI->getPointerOperand(),
           fpStoreWidth(SI->getValueOperand()->getType()));
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      mark(RMW->getPointerOperand(),
           fpStoreWidth(RMW->getValOperand()->getType()));
  }
}

bool holdsFP(Type *T) {
  if (T->isFPOrFPVectorTy())
    return true;
  if (auto *ST = dyn_cast<StructType>(T))
    return llvm::any_of(ST->elements(), holdsFP);
  if (auto *AT = dyn_cast<ArrayType>(T))
    return holdsFP(AT->getElementType());
  return false;
}

// The static shared memory of `K` once profiled: the derivative it runs keeps
// a shadow next to every shared buffer that holds floating-point data.
uint64_t profiledSharedBytes(Function &K) {
  const Target &T = Target::of(K);
  const DataLayout &DL = K.getParent()->getDataLayout();
  SetVector<Function *> funcs;
  SetVector<GlobalVariable *> shared;
  funcs.insert(&K);
  for (size_t f = 0; f < funcs.size(); ++f)
    for (Instruction &I : instructions(*funcs[f])) {
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *callee = CB->getCalledFunction();
            callee && !callee->isDeclaration())
          funcs.insert(callee);
      SmallVector<Value *, 4> ops(I.operands());
      while (!ops.empty()) {
        Value *V = ops.pop_back_val();
        if (auto *GV = dyn_cast<GlobalVariable>(V)) {
          if (T.isShared(*GV))
            shared.insert(GV);
        } else if (auto *CE = dyn_cast<ConstantExpr>(V)) {
          ops.append(CE->op_begin(), CE->op_end());
        }
      }
    }
  uint64_t bytes = 0;
  for (GlobalVariable *GV : shared) {
    uint64_t size = DL.getTypeAllocSize(GV->getValueType());
    bytes += holdsFP(GV->getValueType()) ? 2 * size : size;
  }
  return bytes;
}

} // namespace

// Profile generation for an attribute site. The kernel keeps its name, its
// launch handle and its original parameters and gains one shadow parameter per
// pointer the launch passes: a pointer parameter, or a pointer field of an
// aggregate it takes by value (the captures of a lambda). Its body becomes the
// derivative request over the instrumented clone. The host half of the same
// compilation reads the descriptor recorded here and fills those shadows in at
// every launch.
bool profileAnnotatedKernel(Session &S, Function &K, unsigned &nextSiteId) {
  Module &M = *K.getParent();
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();

  const uint64_t maxShared = Target::of(M).maxStaticSharedBytes();
  if (uint64_t bytes = profiledSharedBytes(K); bytes > maxShared) {
    diagnoseSite(K,
                 displayName(K) + " would declare " + std::to_string(bytes) +
                     " bytes of static shared memory once profiled (a shadow "
                     "next to each floating-point shared buffer), above the " +
                     std::to_string(maxShared) +
                     " a kernel can declare. The kernel is not profiled and "
                     "stays as written.",
                 DS_Warning);
    return false;
  }

  Function *body = nullptr;
  CallInst *call = outlineKernelBody(K, body);
  if (!call)
    return false;
  const unsigned nOrigArgs = K.arg_size();

  std::string header;
  Function *clone = instrumentedClone(S, *body, nextSiteId, &header);
  if (!clone) {
    InlineFunctionInfo IFI;
    (void)InlineFunction(*call, IFI);
    body->eraseFromParent();
    return false;
  }
  unsigned siteId = nextSiteId++;
  const std::string profileName = clone->getName().str();

  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(*clone, seedWidth);
  std::map<LaunchRef, unsigned> fieldSeedWidth;
  collectOutputFields(*clone, fieldSeedWidth);

  // The aggregate a parameter passes by value or as a value, if it holds a
  // pointer.
  auto aggregateOf = [&](unsigned i) -> Type * {
    Type *T = K.getArg(i)->hasByValAttr()   ? K.getParamByValType(i)
              : K.getArg(i)->hasByRefAttr() ? K.getParamByRefType(i)
                                            : K.getArg(i)->getType();
    return T->isAggregateType() && holdsPointer(T) ? T : nullptr;
  };
  struct Shadowed {
    unsigned param;
    int64_t offset; // -1: the parameter is the pointer
    SmallVector<unsigned, 4> path;
    unsigned seed;
  };
  SmallVector<Shadowed, 8> shadowed;
  for (unsigned i = 0; i < K.arg_size(); ++i) {
    if (Type *T = aggregateOf(i)) {
      SmallVector<unsigned, 4> path;
      SmallVector<PointerField, 8> fields;
      pointerFields(DL, T, 0, path, fields);
      for (PointerField &f : fields) {
        LaunchRef r{(int)i, (int64_t)f.offset};
        auto it = fieldSeedWidth.find(r);
        shadowed.push_back(
            {i, r.offset, f.path, it == fieldSeedWidth.end() ? 0 : it->second});
      }
    } else if (K.getArg(i)->getType()->isPointerTy() &&
               !K.getArg(i)->hasByValAttr() && !K.getArg(i)->hasByRefAttr()) {
      shadowed.push_back({i, -1, {}, seedWidth[i]});
    }
  }

  SmallVector<Type *, 16> params(K.getFunctionType()->params());
  Type *PtrTy = PointerType::getUnqual(C);
  for (size_t k = 0; k < shadowed.size(); ++k)
    params.push_back(PtrTy);

  Function *P =
      Function::Create(FunctionType::get(Type::getVoidTy(C), params, false),
                       K.getLinkage(), K.getName() + ".poseidon.profile", &M);
  P->setCallingConv(K.getCallingConv());
  // Function attributes only: the parameter attributes describe the primal
  // kernel's accesses, which say nothing about the shadows or about what the
  // reverse pass touches. A by-value aggregate keeps the byval that makes it
  // one, which is how the launch passes it.
  P->setAttributes(AttributeList::get(C, K.getAttributes().getFnAttrs(),
                                      AttributeSet(), {}));
  for (unsigned i = 0; i < K.arg_size(); ++i)
    if (K.getArg(i)->hasByValAttr() || K.getArg(i)->hasByRefAttr()) {
      P->addParamAttr(
          i, K.getArg(i)->hasByValAttr()
                 ? Attribute::getWithByValType(C, K.getParamByValType(i))
                 : Attribute::getWithByRefType(C, K.getParamByRefType(i)));
      if (MaybeAlign align = K.getParamAlign(i))
        P->addParamAttr(i, Attribute::getWithAlignment(C, *align));
    }
  P->setMemoryEffects(MemoryEffects::unknown());
  // The profiled kernel is launched with the original's block shape. The
  // probes it calls are linked into this module and compiled under the same
  // bound.
  Target::of(M).boundProfiledLaunch(*P);

  auto activity = [&](StringRef name) {
    return MetadataAsValue::get(C, MDString::get(C, name));
  };
  IRBuilder<> B(BasicBlock::Create(C, "entry", P));
  SmallVector<Value *, 32> args{clone};
  unsigned shadow = K.arg_size();
  const Shadowed *next = shadowed.begin();
  for (unsigned i = 0; i < K.arg_size(); ++i) {
    Value *a = P->getArg(i);
    if (a->getType() != clone->getArg(i)->getType())
      a = B.CreateAddrSpaceCast(a, clone->getArg(i)->getType());
    const Shadowed *first = next;
    while (next != shadowed.end() && next->param == i)
      ++next;
    if (first == next) {
      args.push_back(activity("enzyme_const"));
      args.push_back(a);
      continue;
    }
    // Every pointer is duplicated, inputs included: with a constant input the
    // computation reading it has no active argument and Enzyme drops the
    // reverse pass the profile is recorded from. An aggregate's shadow is the
    // aggregate with each pointer replaced by its shadow.
    Value *sh = nullptr;
    if (first->offset < 0) {
      sh = P->getArg(shadow++);
    } else if (Type *T = K.getArg(i)->hasByValAttr()
                             ? K.getParamByValType(i)
                             : K.getArg(i)->getParamByRefType()) {
      Align align = K.getParamAlign(i).valueOrOne();
      AllocaInst *copy = B.CreateAlloca(T);
      copy->setAlignment(std::max(align, copy->getAlign()));
      B.CreateMemCpy(copy, copy->getAlign(), a, align, DL.getTypeAllocSize(T));
      for (const Shadowed *f = first; f != next; ++f)
        B.CreateStore(P->getArg(shadow++),
                      B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), copy,
                                                   (uint64_t)f->offset));
      sh = B.CreatePointerBitCastOrAddrSpaceCast(copy, a->getType());
    } else {
      sh = a;
      for (const Shadowed *f = first; f != next; ++f)
        sh = B.CreateInsertValue(sh, P->getArg(shadow++), f->path);
    }
    args.push_back(activity("enzyme_dup"));
    args.push_back(a);
    args.push_back(sh);
  }

  SmallVector<Type *, 32> argTys;
  for (Value *a : args)
    argTys.push_back(a->getType());
  Function *autodiff = Function::Create(
      FunctionType::get(Type::getVoidTy(C), argTys, false),
      GlobalValue::ExternalLinkage, "__enzyme_autodiff_poseidon", &M);

  B.CreateCall(autodiff, args);
  B.CreateRetVoid();

  std::string kernelName = K.getName().str();
  P->addFnAttr(kSiteLowered);
  K.replaceAllUsesWith(P);
  K.eraseFromParent();
  if (body->use_empty())
    body->eraseFromParent();
  P->setName(kernelName);

  bool anyField =
      llvm::any_of(shadowed, [](const Shadowed &s) { return s.offset >= 0; });
  std::string message;
  raw_string_ostream msg(message);
  msg << "[poseidon-profgen] " << kernelName << ": site " << siteId << ", "
      << shadowed.size() << " shadow buffer(s), output(s)";
  for (const Shadowed &s : shadowed)
    if (s.seed) {
      msg << ' ' << s.param;
      if (s.offset >= 0)
        msg << "[+" << s.offset << "]";
    }
  msg << "\n";
  recordDescriptor(
      S, LaunchKind::Profiling, kernelName,
      [&](raw_ostream &os) {
        os << ' ' << siteId << ' ' << nOrigArgs << ' ' << shadowed.size();
        for (const Shadowed &s : shadowed)
          os << ' ' << s.param;
        for (const Shadowed &s : shadowed)
          os << ' ' << s.seed;
        if (anyField) {
          os << " off";
          for (const Shadowed &s : shadowed)
            os << ' ' << std::max<int64_t>(s.offset, 0);
        }
      },
      std::move(message));
  // The host half registers it with the profiling runtime, so that a kernel
  // that never launches still gets a profile, one saying so.
  recordProfileHeader(S, kernelName, profileName, header);
  return true;
}

namespace {

Constant *cString(Module &M, StringRef text) {
  auto *init = ConstantDataArray::getString(M.getContext(), text);
  auto *GV =
      new GlobalVariable(M, init->getType(), true, GlobalValue::PrivateLinkage,
                         init, "poseidon.str");
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

Constant *constantArray(Module &M, Type *T, ArrayRef<Constant *> elts) {
  auto *init = ConstantArray::get(ArrayType::get(T, elts.size()), elts);
  auto *GV =
      new GlobalVariable(M, init->getType(), true, GlobalValue::PrivateLinkage,
                         init, "poseidon.site");
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

// The source name of a parameter, for the runtime's diagnostics.
std::string paramName(Function &B, Argument &A) {
  if (A.hasName())
    return A.getName().str();
  if (DISubprogram *SP = B.getSubprogram())
    for (auto *N : SP->getRetainedNodes())
      if (auto *V = dyn_cast<DILocalVariable>(N))
        if (V->getArg() == A.getArgNo() + 1)
          return V->getName().str();
  return "";
}

// Registers, for the duration of `CB`, the extent of every stack object a
// pointer argument points into: the runtime has no other way to size one.
// Returns how many it pushed.
unsigned pushStackArrays(CallBase &CB, IRBuilder<> &B) {
  Module &M = *CB.getModule();
  const DataLayout &DL = M.getDataLayout();
  FunctionCallee push = M.getOrInsertFunction(
      "__poseidon_stack_push",
      FunctionType::get(B.getVoidTy(), {B.getPtrTy(), B.getInt64Ty()}, false));
  SmallPtrSet<AllocaInst *, 4> seen;
  unsigned n = 0;
  for (Value *arg : CB.args()) {
    if (!arg->getType()->isPointerTy())
      continue;
    auto *AI = dyn_cast<AllocaInst>(getUnderlyingObject(arg));
    if (!AI || !seen.insert(AI).second)
      continue;
    Value *bytes =
        B.CreateMul(B.CreateZExtOrTrunc(AI->getArraySize(), B.getInt64Ty()),
                    B.getInt64(DL.getTypeAllocSize(AI->getAllocatedType())));
    B.CreateCall(push, {AI, bytes});
    ++n;
  }
  return n;
}

void popStackArrays(CallBase &CB, unsigned n) {
  if (!n)
    return;
  Module &M = *CB.getModule();
  Instruction *at = CB.getNextNode();
  if (auto *II = dyn_cast<InvokeInst>(&CB)) {
    BasicBlock *normal = II->getNormalDest();
    BasicBlock *mid = BasicBlock::Create(M.getContext(), "poseidon.stack.pop",
                                         CB.getFunction(), normal);
    at = IRBuilder<>(mid).CreateBr(normal);
    normal->replacePhiUsesWith(II->getParent(), mid);
    II->setNormalDest(mid);
  }
  IRBuilder<> B(at);
  FunctionCallee pop = M.getOrInsertFunction(
      "__poseidon_stack_pop",
      FunctionType::get(B.getVoidTy(), {B.getInt32Ty()}, false));
  B.CreateCall(pop, {B.getInt32(n)});
}

} // namespace

// Every call in this module that passes a stack object to someone else; a
// site may be reached through any of them.
void registerStackArrays(Session &S, Module &M) {
  SmallPtrSet<Function *, 8> clones;
  for (WeakVH &handle : S.SiteClones)
    if (auto *F = dyn_cast_or_null<Function>(handle))
      clones.insert(F);
  SmallVector<CallBase *, 16> todo;
  for (Function &F : M) {
    if (F.isDeclaration() || clones.count(&F))
      continue;
    for (Instruction &I : instructions(F)) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || isa<IntrinsicInst>(CB) || isa<CallBrInst>(CB) ||
          CB->isInlineAsm() || CB->isMustTailCall())
        continue;
      if (Function *callee = CB->getCalledFunction()) {
        StringRef name = callee->getName();
        if (name.starts_with("__poseidon") || name.starts_with("poseidon") ||
            name.starts_with("__enzyme"))
          continue;
      }
      todo.push_back(CB);
    }
  }
  for (CallBase *CB : todo) {
    IRBuilder<> B(CB);
    popStackArrays(*CB, pushStackArrays(*CB, B));
  }
}

// The extent of every global this module defines, registered before main.
void registerGlobals(Module &M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *PtrTy = PointerType::getUnqual(C);
  Type *I64 = Type::getInt64Ty(C);
  StructType *entryTy = StructType::get(C, {PtrTy, I64});
  SmallVector<Constant *, 32> entries;
  for (GlobalVariable &GV : M.globals()) {
    StringRef name = GV.getName();
    if (GV.isDeclaration() || GV.isThreadLocal() || GV.getAddressSpace() ||
        GV.hasAvailableExternallyLinkage() || name.starts_with("llvm.") ||
        GV.getSection() == "llvm.metadata" || name.starts_with("poseidon") ||
        name.starts_with("fpprofile_static_"))
      continue;
    uint64_t bytes = DL.getTypeAllocSize(GV.getValueType());
    if (bytes)
      entries.push_back(
          ConstantStruct::get(entryTy, {&GV, ConstantInt::get(I64, bytes)}));
  }
  if (entries.empty())
    return;
  Constant *table = constantArray(M, entryTy, entries);
  FunctionCallee reg = M.getOrInsertFunction(
      "__poseidon_register_globals",
      FunctionType::get(Type::getVoidTy(C), {PtrTy, I64}, false));
  Function *ctor =
      Function::Create(FunctionType::get(Type::getVoidTy(C), {}, false),
                       GlobalValue::InternalLinkage, "poseidon.globals", &M);
  ctor->addFnAttr(Attribute::NoUnwind);
  IRBuilder<> B(BasicBlock::Create(C, "entry", ctor));
  B.CreateCall(reg, {table, ConstantInt::get(I64, entries.size())});
  B.CreateRetVoid();
  appendToGlobalCtors(M, ctor, 0);
}

namespace {

struct HostSiteData {
  Function *clone;
  unsigned siteId;
  SmallVector<unsigned, 8> ptrParams;
  Constant *name, *param, *seed, *paramNames;
};

// One call of a host site's body becomes the reverse-mode request over the
// instrumented clone: the runtime finds the allocation each pointer argument
// points into and hands back its shadow, and a floating-point return is seeded
// by the request itself.
void emitProfiledHostCall(CallBase *CB, const HostSiteData &D) {
  auto *CI = dyn_cast<CallInst>(CB);
  if (!CI)
    CI = changeToCall(cast<InvokeInst>(CB));
  Module &M = *CI->getModule();
  LLVMContext &C = M.getContext();
  Type *PtrTy = PointerType::getUnqual(C);
  const unsigned nptr = D.ptrParams.size();
  ArrayType *arrTy = ArrayType::get(PtrTy, nptr);

  IRBuilder<> B(CI);
  unsigned pushed = pushStackArrays(*CI, B);
  Value *shadows = nullptr;
  if (nptr) {
    IRBuilder<> entry(
        &*CI->getFunction()->getEntryBlock().getFirstInsertionPt());
    Value *ptrs = entry.CreateAlloca(arrTy, nullptr, "poseidon.ptrs");
    shadows = entry.CreateAlloca(arrTy, nullptr, "poseidon.shadows");
    for (unsigned k = 0; k < nptr; ++k)
      B.CreateStore(CI->getArgOperand(D.ptrParams[k]),
                    B.CreateConstInBoundsGEP2_32(arrTy, ptrs, 0, k));
    FunctionCallee request = M.getOrInsertFunction(
        "__poseidon_host_shadows",
        FunctionType::get(B.getVoidTy(),
                          {PtrTy, B.getInt32Ty(), B.getInt32Ty(), PtrTy, PtrTy,
                           PtrTy, PtrTy, PtrTy},
                          false));
    B.CreateCall(request, {D.name, B.getInt32(D.siteId), B.getInt32(nptr), ptrs,
                           D.param, D.seed, D.paramNames, shadows});
  }

  auto activity = [&](StringRef name) {
    return MetadataAsValue::get(C, MDString::get(C, name));
  };
  SmallVector<Value *, 32> args{D.clone};
  if (!CI->getType()->isVoidTy())
    args.push_back(activity("enzyme_primal_return"));
  unsigned k = 0;
  for (Argument &A : D.clone->args()) {
    Value *a = CI->getArgOperand(A.getArgNo());
    if (A.getType()->isPointerTy()) {
      args.push_back(activity("enzyme_dup"));
      args.push_back(a);
      args.push_back(B.CreateLoad(
          PtrTy, B.CreateConstInBoundsGEP2_32(arrTy, shadows, 0, k++)));
    } else {
      args.push_back(activity(
          A.getType()->isFloatingPointTy() ? "enzyme_out" : "enzyme_const"));
      args.push_back(a);
    }
  }
  SmallVector<Type *, 32> argTys;
  for (Value *a : args)
    argTys.push_back(a->getType());
  Function *autodiff = Function::Create(
      FunctionType::get(CI->getType(), argTys, false),
      GlobalValue::ExternalLinkage, "__enzyme_autodiff_poseidon", &M);
  CallInst *adCall = B.CreateCall(autodiff, args);
  adCall->setDebugLoc(CI->getDebugLoc());
  popStackArrays(*adCall, pushed);
  CI->replaceAllUsesWith(adCall);
  CI->eraseFromParent();
}

} // namespace

bool profileHostSite(Session &S, Function &B, unsigned &nextSiteId) {
  Module &M = *B.getParent();
  LLVMContext &C = M.getContext();
  takeHostBody(B);
  SmallVector<CallBase *, 4> calls = callsOf(B);
  Function *clone = instrumentedClone(S, B, nextSiteId);
  if (!clone) {
    markOutlineTransparent(&B);
    return false;
  }
  HostSiteData D{clone, nextSiteId++, {}, nullptr, nullptr, nullptr, nullptr};

  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(*clone, seedWidth);
  SmallVector<Constant *, 8> param, seed, names;
  Type *I32 = Type::getInt32Ty(C);
  for (Argument &A : B.args()) {
    if (!A.getType()->isPointerTy())
      continue;
    D.ptrParams.push_back(A.getArgNo());
    param.push_back(ConstantInt::get(I32, A.getArgNo() + 1));
    seed.push_back(ConstantInt::get(I32, seedWidth[A.getArgNo()]));
    names.push_back(cString(M, paramName(B, A)));
  }
  StringRef symbol = B.getName();
  symbol.consume_back("_poseidon_body");
  auto [fnSymbol, where] = symbol.split(".poseidon.");
  std::string name = llvm::demangle(fnSymbol.str());
  if (!where.empty()) {
    auto [callerSymbol, k] = where.rsplit('.');
    name += " at call " + k.str() + " in " + llvm::demangle(callerSymbol.str());
  }
  if (!D.ptrParams.empty()) {
    D.name = cString(M, name);
    D.param = constantArray(M, I32, param);
    D.seed = constantArray(M, I32, seed);
    D.paramNames = constantArray(M, PointerType::getUnqual(C), names);
  }

  for (CallBase *CB : calls)
    emitProfiledHostCall(CB, D);

  errs() << "[poseidon-profgen] " << name << ": site " << D.siteId << ", "
         << D.ptrParams.size() << " shadow buffer(s), output(s)";
  for (unsigned p : D.ptrParams)
    if (seedWidth[p])
      errs() << ' ' << p;
  if (!B.getReturnType()->isVoidTy())
    errs() << " return";
  errs() << ", " << calls.size() << " call(s)\n";
  return true;
}

namespace {

// `p` with the object it points into replaced by `shadow`.
Value *intoShadow(Value *p, AllocaInst *obj, AllocaInst *shadow,
                  IRBuilder<> &B) {
  if (p == obj)
    return shadow;
  auto *I = cast<Instruction>(p);
  Value *base = intoShadow(I->getOperand(0), obj, shadow, B);
  Instruction *copy = I->clone();
  copy->setOperand(0, base);
  return B.Insert(copy, p->getName() + ".shadow");
}

// The parameters the site reads a pointer through. A shadow of such an object
// would have to hold shadows of those pointers, and one of zeros hands the
// reverse pass null ones, so the object is passed as a constant instead.
void collectPointerHolders(Function &clone, SmallVectorImpl<bool> &holds) {
  holds.assign(clone.arg_size(), false);
  for (Instruction &I : instructions(clone))
    if (auto *LI = dyn_cast<LoadInst>(&I))
      if (LI->getType()->isPtrOrPtrVectorTy())
        if (int idx = traceToArgIndex(LI->getPointerOperand()); idx >= 0)
          holds[idx] = true;
}

// Whether the program may read what `call` leaves in `obj`: an instruction the
// call reaches before the object's lifetime ends, the call itself when it runs
// again, or whatever the object's address escapes to. What the site writes and
// nothing reads reaches no result of the program (dFEM's JVP writes the
// q-function's primal outputs next to the tangents it is called for), so it is
// no output to seed.
bool readAfterCall(AllocaInst *obj, CallInst *call) {
  SmallPtrSet<const Instruction *, 16> readers{call}, ends;
  SmallPtrSet<const Value *, 16> seen{obj};
  SmallVector<const Value *, 8> work{obj};
  while (!work.empty()) {
    const Value *V = work.pop_back_val();
    for (const User *U : V->users()) {
      const auto *I = dyn_cast<Instruction>(U);
      if (!I)
        return true;
      if (isa<GetElementPtrInst, BitCastInst, AddrSpaceCastInst>(I)) {
        if (seen.insert(I).second)
          work.push_back(I);
      } else if (I->isLifetimeStartOrEnd()) {
        ends.insert(I);
      } else if (isa<LoadInst>(I)) {
        readers.insert(I);
      } else if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getValueOperand() == V)
          return true;
      } else if (auto *MI = dyn_cast<MemIntrinsic>(I)) {
        if (auto *MT = dyn_cast<MemTransferInst>(MI);
            MT && MT->getRawSource() == V)
          readers.insert(I);
      } else if (I != call) {
        return true;
      }
    }
  }

  SmallPtrSet<const BasicBlock *, 16> visited;
  SmallVector<const BasicBlock *, 16> blocks;
  auto reads = [&](BasicBlock::const_iterator it, const BasicBlock *BB) {
    for (; it != BB->end(); ++it) {
      if (readers.count(&*it))
        return true;
      if (ends.count(&*it))
        return false;
    }
    for (const BasicBlock *S : successors(BB))
      if (visited.insert(S).second)
        blocks.push_back(S);
    return false;
  };
  if (reads(std::next(call->getIterator()), call->getParent()))
    return true;
  while (!blocks.empty()) {
    const BasicBlock *BB = blocks.pop_back_val();
    if (reads(BB->begin(), BB))
      return true;
  }
  return false;
}

// The marker in the explicit form, with a shadow for every pointer marked
// poseidon_shadow: a local object like the one the pointer points into, zero,
// and seeded by the kernel path's rule when the site writes into it and the
// program reads it afterwards (readAfterCall). It is made at every execution of
// the call, because the object the call sees is a new one each time: its
// storage is reused, its value is not.
CallInst *withShadows(CallInst *CI, Function &clone, const MarkerArgs &marker) {
  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(clone, seedWidth);
  SmallVector<bool, 8> holdsPointers;
  collectPointerHolders(clone, holdsPointers);
  Function *body = markerTargetOrNull(CI);
  const DataLayout &DL = CI->getModule()->getDataLayout();
  LLVMContext &C = CI->getContext();

  MapVector<AllocaInst *, std::pair<AllocaInst *, unsigned>> objects;
  DenseMap<unsigned, AllocaInst *> objectOf;
  DenseSet<unsigned> constant;
  for (const MarkerArgs::MadeShadow &S : marker.madeShadows) {
    Value *p = CI->getArgOperand(S.arg);
    if (pointsNowhere(p) || holdsPointers[S.param]) {
      constant.insert(S.activity);
      continue;
    }
    AllocaInst *obj = localObject(p, CI->getFunction());
    if (!obj) {
      warn(CI, perCallSiteName(*body) + " no longer takes argument " +
                   Twine(S.param + 1) + " as a pointer into a local object; " +
                   "the call is not profiled");
      return nullptr;
    }
    unsigned &width = objects[obj].second;
    width = std::max(width, seedWidth[S.param]);
    objectOf[S.arg] = obj;
  }
  SmallPtrSet<AllocaInst *, 4> unread;
  for (auto &[obj, entry] : objects)
    if (entry.second && !readAfterCall(obj, CI)) {
      entry.second = 0;
      unread.insert(obj);
    }

  for (auto &[obj, entry] : objects) {
    auto &[shadow, width] = entry;
    Type *T = width == 8   ? Type::getDoubleTy(C)
              : width == 4 ? Type::getFloatTy(C)
                           : nullptr;
    if (width && !T) {
      fail(CI, "no seed rule for a " + Twine(width) + "-byte output element");
      return nullptr;
    }
    shadow = new AllocaInst(obj->getAllocatedType(), obj->getAddressSpace(),
                            obj->getArraySize(), obj->getAlign(),
                            obj->getName() + ".shadow",
                            std::next(obj->getIterator()));
    uint64_t bytes = obj->getAllocationSize(DL)->getFixedValue();
    IRBuilder<> B(CI);
    B.CreateMemSet(shadow, B.getInt8(0), bytes, obj->getAlign());
    if (uint64_t n = T ? bytes / width : 0) {
      auto [at, idx] =
          SplitBlockAndInsertSimpleForLoop(B.getInt64(n), CI->getIterator());
      IRBuilder<> L(at);
      L.CreateStore(ConstantFP::get(T, 1.0), L.CreateGEP(T, shadow, idx));
    }
  }

  IRBuilder<> B(CI);
  SmallVector<Value *, 32> args;
  for (unsigned i = 0; i < CI->arg_size(); ++i) {
    Value *a = CI->getArgOperand(i);
    if (llvm::any_of(marker.madeShadows,
                     [&](const auto &S) { return S.activity == i; }))
      a = MetadataAsValue::get(
          C,
          MDString::get(C, constant.count(i) ? "enzyme_const" : "enzyme_dup"));
    args.push_back(a);
    if (AllocaInst *obj = objectOf.lookup(i))
      args.push_back(intoShadow(a, obj, objects[obj].first, B));
  }
  CallInst *NC =
      B.CreateCall(CI->getFunctionType(), CI->getCalledOperand(), args);
  NC->setCallingConv(CI->getCallingConv());
  NC->setDebugLoc(CI->getDebugLoc());
  AttributeList old = CI->getAttributes();
  NC->setAttributes(
      AttributeList::get(C, old.getFnAttrs(), old.getRetAttrs(), {}));
  CI->replaceAllUsesWith(NC);
  CI->eraseFromParent();

  errs() << "[poseidon-profgen] " << perCallSiteName(*body) << ": "
         << objects.size() << " shadow object(s), output(s)";
  for (const MarkerArgs::MadeShadow &S : marker.madeShadows)
    if (!constant.count(S.activity) && seedWidth[S.param] &&
        !unread.count(objectOf.lookup(S.arg)))
      errs() << ' ' << S.param;
  if (!unread.empty()) {
    errs() << ", not read after the call:";
    for (const MarkerArgs::MadeShadow &S : marker.madeShadows)
      if (unread.count(objectOf.lookup(S.arg)))
        errs() << ' ' << S.param;
  }
  errs() << "\n";
  return NC;
}

} // namespace

} // namespace poseidon
