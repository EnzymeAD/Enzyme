//===- SiteProfile.cpp - profile generation for a site --------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SiteProfile.h"

#include "Enzyme/Utils.h"
#include "Instrument.h"
#include "Pass/Diagnostics.h"
#include "Pass/Session.h"
#include "ProfileReader.h"
#include "Sites/Annotations.h"
#include "Sites/CallSites.h"
#include "Sites/Canonicalize.h"
#include "Sites/LaunchManifest.h"
#include "Sites/Markers.h"
#include "Sites/Site.h"
#include "Sites/SiteIdentity.h"
#include "Support/Utils.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/Matmul.h"
#include "runtime/profile/ProfileFormat.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/ValueTracking.h"
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
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

/// Canonicalize, slot-number and instrument a clone of the site body `F`
/// exactly as profile use canonicalizes and slot-numbers it. Null if the body
/// holds nothing profilable. `staticOut`, if given, receives the header the
/// profile will carry.
static Function *instrumentedClone(Session &S, Function &F, unsigned siteId,
                                   std::string *staticOut = nullptr) {
  Function *clone = canonicalize(F);
  S.SiteClones.push_back(clone);
  setSlotMetadata(*clone);
  // Before the probes: they break the reduction shape the trip count is read
  // from.
  SmallVector<std::pair<size_t, unsigned>, 4> trips;
  collectScalarLoopReductionTrips(S.Opts, *clone, trips);
  const std::string profileName = bodyProfileStem(S.Sites, F);
  if (!instrumentForProfiling(*clone, siteId, profileName)) {
    clone->eraseFromParent();
    return nullptr;
  }
  std::string staticData;
  {
    raw_string_ostream os(staticData);
    os << kProfileKeySiteId << " = " << siteId << '\n';
    for (const auto &kv : trips)
      os << "ReductionTrip = " << kv.first << ' ' << kv.second << '\n';
    os << kProfileKeyCanonicalHash << " = " << canonicalFormHash(*clone)
       << '\n';
  }
  emitProfileStaticData(*clone, profileName, staticData);
  if (staticOut)
    *staticOut = staticData;
  return clone;
}

static Expected<CallInst *> withShadows(CallInst *CI, Function &clone,
                                        const MarkerArgs &marker, bool print);

/// The reverse-mode request over `args`, whose first is the instrumented clone,
/// of type `FT`, or taking exactly `args` and returning `ret` when FT is null.
static CallInst *emitAutodiffRequest(IRBuilder<> &B, Type *ret,
                                     ArrayRef<Value *> args,
                                     FunctionType *FT = nullptr) {
  if (!FT) {
    SmallVector<Type *, 32> argTys;
    for (Value *a : args)
      argTys.push_back(a->getType());
    FT = FunctionType::get(ret, argTys, false);
  }
  Function *autodiff =
      Function::Create(FT, GlobalValue::ExternalLinkage, kAutodiffRequest,
                       B.GetInsertBlock()->getModule());
  return B.CreateCall(FT, autodiff, args);
}

/// Profile generation. The marker becomes an ordinary reverse-mode derivative
/// request over the instrumented clone: the custom derivative on each probe is
/// what puts the gradient record inside the reverse pass, so the host's AD
/// needs no profiling mode of its own.
static Expected<bool> profileMarker(Session &S, CallInst *CI,
                                    DenseMap<Function *, Function *> &clones,
                                    unsigned &nextSiteId) {
  Expected<Function *> target = markerTarget(CI);
  if (!target)
    return target.takeError();
  Function *F = *target;
  Expected<MarkerArgs> marker = parseMarker(CI, F);
  if (!marker)
    return marker.takeError();

  // Two annotations on the same body share one instrumented clone, so their
  // executions accumulate into one profile record under one site id.
  Function *&clone = clones[F];
  if (!clone) {
    // The site id counts clones in marker order.
    clone = instrumentedClone(S, *F, nextSiteId++);
    if (!clone) {
      warn(CI, Twine("no profilable operation in ") + F->getName());
      return false;
    }
  }
  if (!marker->madeShadows.empty()) {
    Expected<CallInst *> shadowed =
        withShadows(CI, *clone, *marker, S.Opts.Diagnostics.Print);
    if (!shadowed)
      return shadowed.takeError();
    CI = *shadowed;
    marker = parseMarker(CI, F);
    if (!marker)
      return marker.takeError();
  }

  Module &M = *CI->getModule();
  // A request returns the gradient unless asked for the primal, and the
  // program goes on computing with the value of a marker call of F's type.
  const bool returnPrimal =
      !CI->use_empty() && CI->getType() == F->getReturnType() &&
      llvm::none_of(CI->args(), [](Value *A) {
        return isActivityMarker(A, "enzyme_primal_return");
      });
  LLVMContext &C = M.getContext();
  AttributeList old = CI->getAttributes();
  SmallVector<Value *, 16> args;
  SmallVector<AttributeSet, 16> argAttrs;
  bool sret =
      CI->hasStructRetAttr() || F->hasParamAttribute(0, Attribute::StructRet);
  for (size_t i = 0; i < CI->arg_size(); ++i) {
    if (!marker->forward[i])
      continue;
    args.push_back(i == (size_t)sret ? clone : CI->getArgOperand(i));
    argAttrs.push_back(old.getParamAttrs(i));
    if (i == (size_t)sret && returnPrimal) {
      args.push_back(activityArg(C, "enzyme_primal_return"));
      argAttrs.emplace_back();
    }
  }

  // The request takes the marker's type when its arguments are that type's
  // parameters. A marker with a fixed list of parameters (a C++
  // instantiation) has none for enzyme_primal_return, and has two for a
  // poseidon_tau and its target, which the request leaves out; the request
  // then takes a variadic type.
  FunctionType *FT = CI->getFunctionType();
  if (!FT->isVarArg() && args.size() != FT->getNumParams())
    FT = FunctionType::get(CI->getType(), {clone->getType()},
                           /*isVarArg=*/true);
  IRBuilder<> Builder(CI);
  CallInst *adCall = emitAutodiffRequest(Builder, CI->getType(), args, FT);
  adCall->getCalledFunction()->setCallingConv(CI->getCallingConv());
  adCall->setCallingConv(CI->getCallingConv());
  adCall->setDebugLoc(CI->getDebugLoc());
  adCall->setAttributes(
      AttributeList::get(C, old.getFnAttrs(), old.getRetAttrs(), argAttrs));

  CI->replaceAllUsesWith(adCall);
  CI->eraseFromParent();
  return true;
}

/// The first type `isLeaf` accepts among `T` and, through struct fields in
/// order and array elements, the types `T` is made of; null if there is none.
static Type *findLeafType(Type *T, function_ref<bool(Type *)> isLeaf) {
  if (isLeaf(T))
    return T;
  if (auto *ST = dyn_cast<StructType>(T))
    for (Type *E : ST->elements())
      if (Type *L = findLeafType(E, isLeaf))
        return L;
  if (auto *AT = dyn_cast<ArrayType>(T))
    return findLeafType(AT->getElementType(), isLeaf);
  return nullptr;
}

static bool isFPLeaf(Type *T) { return T->isFPOrFPVectorTy(); }

static bool isPointerLeaf(Type *T) { return T->isPointerTy(); }

/// The width of the floating-point scalar a store writes, 0 if it writes none.
static unsigned fpStoreWidth(Type *T) {
  Type *L = findLeafType(T, isFPLeaf);
  return L ? L->getScalarSizeInBits() / 8 : 0;
}

/// Every store and atomicrmw of `clone`: each object the pointer it writes
/// through may point into and the width of the floating-point scalar it writes
/// (fpStoreWidth). The pointer may choose between objects, or between an object
/// and null: a view that holds null when it is empty writes into the object it
/// was made from.
static void
forEachWrite(Function &clone,
             function_ref<void(const Value *obj, unsigned width)> mark) {
  auto write = [&](Value *ptr, Type *T) {
    SmallVector<const Value *, 4> objs;
    getUnderlyingObjects(ptr, objs, /*LI=*/nullptr, /*MaxLookup=*/0);
    for (const Value *obj : objs)
      if (!isa<ConstantPointerNull>(obj))
        mark(obj, fpStoreWidth(T));
  };
  for (Instruction &I : instructions(clone)) {
    if (auto *SI = dyn_cast<StoreInst>(&I))
      write(SI->getPointerOperand(), SI->getValueOperand()->getType());
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
      write(RMW->getPointerOperand(), RMW->getValOperand()->getType());
  }
}

/// Which parameters the site writes through, and in what element width. An
/// application does not say which of its buffers are outputs, and reverse mode
/// needs to know: the output shadows are the ones the seed goes into.
static void collectOutputParams(Function &clone,
                                SmallVectorImpl<unsigned> &seedWidth) {
  seedWidth.assign(clone.arg_size(), 0);
  auto mark = [&](const Value *obj, unsigned w) {
    if (const auto *A = dyn_cast<Argument>(obj); w && A)
      seedWidth[A->getArgNo()] = std::max(seedWidth[A->getArgNo()], w);
  };
  forEachWrite(clone, mark);
}

static bool holdsPointer(Type *T) { return findLeafType(T, isPointerLeaf); }

namespace {

/// Where an aggregate of type `T` holds pointers: the byte offset of each from
/// the aggregate's start and the insertvalue path that reaches it, in layout
/// order.
struct PointerField {
  uint64_t offset;
  SmallVector<unsigned, 4> path;
};

} // namespace

static void pointerFields(const DataLayout &DL, Type *T, uint64_t offset,
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

/// The pointer fields of by-value aggregate parameters the site writes through,
/// in the element width it writes: collectOutputParams for the pointers a
/// kernel's lambda captured.
static void collectOutputFields(Function &clone,
                                std::map<LaunchRef, unsigned> &seedWidth) {
  auto mark = [&](const Value *obj, unsigned w) {
    LaunchRef r = tracePointerRoot(obj);
    if (w && r.isField())
      seedWidth[r] = std::max(seedWidth[r], w);
  };
  forEachWrite(clone, mark);
}

namespace {
/// A pointer the profiled kernel takes a shadow of: parameter `param` itself
/// (offset -1), or the pointer field at byte `offset` and member `path` of
/// the aggregate it takes; `seed` is the width the shadow is seeded at, 0 for
/// an input.
struct Shadowed {
  unsigned param;
  int64_t offset;
  SmallVector<unsigned, 4> path;
  unsigned seed;
};
} // namespace

/// The pointers of kernel `K` whose shadows its profiled form takes, in
/// parameter order, seeded as the outputs of the instrumented `clone`.
static SmallVector<Shadowed, 8> shadowedPointers(Function &K, Function &clone) {
  const DataLayout &DL = K.getParent()->getDataLayout();
  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(clone, seedWidth);
  std::map<LaunchRef, unsigned> fieldSeedWidth;
  collectOutputFields(clone, fieldSeedWidth);

  // The aggregate a parameter passes by value or as a value, if it holds a
  // pointer.
  auto aggregateOf = [&](unsigned i) -> Type * {
    Type *T = K.getArg(i)->hasByValAttr()   ? K.getParamByValType(i)
              : K.getArg(i)->hasByRefAttr() ? K.getParamByRefType(i)
                                            : K.getArg(i)->getType();
    return T->isAggregateType() && holdsPointer(T) ? T : nullptr;
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
  return shadowed;
}

/// A kernel with the parameters of `K` and one pointer parameter per entry of
/// `shadowed`, launched as `K` is.
static Function *createProfiledKernel(Function &K,
                                      ArrayRef<Shadowed> shadowed) {
  LLVMContext &C = K.getContext();
  SmallVector<Type *, 16> params(K.getFunctionType()->params());
  Type *PtrTy = getInt8PtrTy(C);
  for (size_t k = 0; k < shadowed.size(); ++k)
    params.push_back(PtrTy);

  Function *P = Function::Create(
      FunctionType::get(Type::getVoidTy(C), params, false), K.getLinkage(),
      K.getName() + ".poseidon.profile", K.getParent());
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
  Target::of(*K.getParent()).boundProfiledLaunch(*P);
  return P;
}

/// The body of the profiled kernel `P` of `K`: a reverse-mode request over the
/// instrumented `clone` that passes each pointer of `shadowed` with the shadow
/// P takes for it.
static void emitProfiledBody(Function &P, Function &K, Function &clone,
                             ArrayRef<Shadowed> shadowed) {
  LLVMContext &C = K.getContext();
  const DataLayout &DL = K.getParent()->getDataLayout();
  IRBuilder<> B(BasicBlock::Create(C, "entry", &P));
  SmallVector<Value *, 32> args{&clone};
  unsigned shadow = K.arg_size();
  const Shadowed *next = shadowed.begin();
  for (unsigned i = 0; i < K.arg_size(); ++i) {
    Value *a = P.getArg(i);
    if (a->getType() != clone.getArg(i)->getType())
      a = B.CreateAddrSpaceCast(a, clone.getArg(i)->getType());
    const Shadowed *first = next;
    while (next != shadowed.end() && next->param == i)
      ++next;
    if (first == next) {
      args.push_back(activityArg(C, "enzyme_const"));
      args.push_back(a);
      continue;
    }
    // Every pointer is duplicated, inputs included: with a constant input the
    // computation reading it has no active argument and Enzyme drops the
    // reverse pass the profile is recorded from. An aggregate's shadow is the
    // aggregate with each pointer replaced by its shadow.
    Value *sh = nullptr;
    if (first->offset < 0) {
      sh = B.CreatePointerBitCastOrAddrSpaceCast(P.getArg(shadow++),
                                                 a->getType());
    } else if (Type *T = K.getArg(i)->hasByValAttr()
                             ? K.getParamByValType(i)
                             : K.getArg(i)->getParamByRefType()) {
      Align align = K.getParamAlign(i).valueOrOne();
      AllocaInst *copy = B.CreateAlloca(T);
      copy->setAlignment(std::max(align, copy->getAlign()));
      B.CreateMemCpy(copy, copy->getAlign(), a, align, DL.getTypeAllocSize(T));
      for (const Shadowed *f = first; f != next; ++f)
        B.CreateStore(P.getArg(shadow++),
                      B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), copy,
                                                   (uint64_t)f->offset));
      sh = B.CreatePointerBitCastOrAddrSpaceCast(copy, a->getType());
    } else {
      sh = a;
      for (const Shadowed *f = first; f != next; ++f)
        sh = B.CreateInsertValue(sh, P.getArg(shadow++), f->path);
    }
    args.push_back(activityArg(C, "enzyme_dup"));
    args.push_back(a);
    args.push_back(sh);
  }

  emitAutodiffRequest(B, Type::getVoidTy(C), args);
  B.CreateRetVoid();
}

/// Records for the host half how a launch of the profiled kernel
/// `kernelName`, of `numArgs` original arguments, fills in `shadowed`, and the
/// header of its profile `profileName`.
static void recordProfiledLaunch(Session &S, StringRef kernelName,
                                 unsigned siteId, unsigned numArgs,
                                 ArrayRef<Shadowed> shadowed,
                                 StringRef profileName, StringRef header) {
  bool anyField =
      llvm::any_of(shadowed, [](const Shadowed &s) { return s.offset >= 0; });
  std::string message;
  if (S.Opts.Diagnostics.Print) {
    raw_string_ostream msg(message);
    msg << "[poseidon] " << kernelName << ": site " << siteId << ", "
        << shadowed.size() << " shadow buffer(s), output(s)";
    for (const Shadowed &s : shadowed)
      if (s.seed) {
        msg << ' ' << s.param;
        if (s.offset >= 0)
          msg << "[+" << s.offset << ']';
      }
    msg << '\n';
  }
  recordDescriptor(
      S.Launches, LaunchKind::Profiling, kernelName,
      [&](raw_ostream &os) {
        os << ' ' << siteId << ' ' << numArgs << ' ' << shadowed.size();
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
  recordProfileHeader(S.Launches, kernelName, profileName, header);
}

/// Profile generation for an attribute site. The kernel keeps its name, its
/// launch handle and its original parameters and gains one shadow parameter per
/// pointer the launch passes: a pointer parameter, or a pointer field of an
/// aggregate it takes by value (the captures of a lambda). Its body becomes the
/// derivative request over the instrumented clone. The host half of the same
/// compilation reads the descriptor recorded here and fills those shadows in at
/// every launch.
static Expected<bool> profileKernel(Session &S, Function &K,
                                    unsigned &nextSiteId) {
  Function *body = nullptr;
  Expected<CallInst *> outlined = outlineKernelBody(S.Sites, K, body);
  if (!outlined)
    return outlined.takeError();
  CallInst *call = *outlined;

  std::string header;
  Function *clone = instrumentedClone(S, *body, nextSiteId, &header);
  if (!clone) {
    diagnose(K, body->getSubprogram(),
             displayName(K) +
                 " holds no floating-point operation the pass profiles; the "
                 "kernel is not profiled and stays as written",
             DS_Warning);
    inlineKernelBody(call, body);
    // Outlined and folded back, the kernel is new code to the analyses.
    return true;
  }
  const unsigned siteId = nextSiteId++;
  const std::string profileName = bodyProfileStem(S.Sites, *body);

  SmallVector<Shadowed, 8> shadowed = shadowedPointers(K, *clone);
  Function *P = createProfiledKernel(K, shadowed);
  emitProfiledBody(*P, K, *clone, shadowed);

  const std::string kernelName = K.getName().str();
  const unsigned numArgs = K.arg_size();
  P->addFnAttr(kSiteLowered);
  K.replaceAllUsesWith(P);
  K.eraseFromParent();
  if (body->use_empty())
    body->eraseFromParent();
  P->setName(kernelName);
  recordProfiledLaunch(S, kernelName, siteId, numArgs, shadowed, profileName,
                       header);
  return true;
}

static Constant *constantArray(Module &M, Type *T, ArrayRef<Constant *> elts) {
  Constant *init = ConstantArray::get(ArrayType::get(T, elts.size()), elts);
  auto *GV =
      new GlobalVariable(M, init->getType(), true, GlobalValue::PrivateLinkage,
                         init, "poseidon.site");
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

/// The source name of a parameter, for the runtime's diagnostics.
static std::string paramName(Function &B, Argument &A) {
  if (A.hasName())
    return A.getName().str();
  if (DISubprogram *SP = B.getSubprogram())
    for (auto *N : SP->getRetainedNodes())
      if (auto *V = dyn_cast<DILocalVariable>(N))
        if (V->getArg() == A.getArgNo() + 1)
          return V->getName().str();
  return "";
}

/// Registers, for the duration of `CB`, the extent of every stack object a
/// pointer argument points into: the runtime has no other way to size one.
/// Returns how many it pushed.
static unsigned pushStackArrays(CallBase &CB, IRBuilder<> &B) {
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

static void popStackArrays(CallBase &CB, unsigned n) {
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

/// Every call in this module that passes a stack object to someone else; a
/// site may be reached through any of them.
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

/// The extent of every global this module defines, registered before main.
void registerGlobals(Module &M) {
  LLVMContext &C = M.getContext();
  const DataLayout &DL = M.getDataLayout();
  Type *PtrTy = getInt8PtrTy(C);
  Type *I64 = Type::getInt64Ty(C);
  StructType *entryTy = StructType::get(C, {PtrTy, I64});
  SmallVector<Constant *, 32> entries;
  for (GlobalVariable &GV : M.globals()) {
    StringRef name = GV.getName();
    if (GV.isDeclaration() || GV.isThreadLocal() || GV.getAddressSpace() ||
        GV.hasAvailableExternallyLinkage() || name.starts_with("llvm.") ||
        GV.getSection() == "llvm.metadata" || name.starts_with("poseidon") ||
        name.starts_with("poseidon.static."))
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

} // namespace

/// One call of a host site's body becomes the reverse-mode request over the
/// instrumented clone: the runtime finds the allocation each pointer argument
/// points into and hands back its shadow, and a floating-point return is seeded
/// by the request itself.
static void emitProfiledHostCall(CallBase *CB, const HostSiteData &D) {
  auto *CI = dyn_cast<CallInst>(CB);
  if (!CI)
    CI = changeToCall(cast<InvokeInst>(CB));
  Module &M = *CI->getModule();
  LLVMContext &C = M.getContext();
  Type *PtrTy = getInt8PtrTy(C);
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

  SmallVector<Value *, 32> args{D.clone};
  if (!CI->getType()->isVoidTy())
    args.push_back(activityArg(C, "enzyme_primal_return"));
  unsigned k = 0;
  for (Argument &A : D.clone->args()) {
    Value *a = CI->getArgOperand(A.getArgNo());
    if (A.getType()->isPointerTy()) {
      args.push_back(activityArg(C, "enzyme_dup"));
      args.push_back(a);
      args.push_back(B.CreateLoad(
          PtrTy, B.CreateConstInBoundsGEP2_32(arrTy, shadows, 0, k++)));
    } else {
      args.push_back(activityArg(
          C, A.getType()->isFloatingPointTy() ? "enzyme_out" : "enzyme_const"));
      args.push_back(a);
    }
  }
  CallInst *adCall = emitAutodiffRequest(B, CI->getType(), args);
  adCall->setDebugLoc(CI->getDebugLoc());
  popStackArrays(*adCall, pushed);
  CI->replaceAllUsesWith(adCall);
  CI->eraseFromParent();
}

static bool profileHostBody(Session &S, Function &B, unsigned &nextSiteId) {
  Module &M = *B.getParent();
  LLVMContext &C = M.getContext();
  releaseHostBody(B);
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
    names.push_back(getStringPointer(M, "", paramName(B, A)));
  }
  std::string name = perCallSiteName(siteKeyOfBody(B));
  if (!D.ptrParams.empty()) {
    D.name = getStringPointer(M, "", name);
    D.param = constantArray(M, I32, param);
    D.seed = constantArray(M, I32, seed);
    D.paramNames = constantArray(M, getInt8PtrTy(C), names);
  }

  for (CallBase *CB : calls)
    emitProfiledHostCall(CB, D);

  if (!S.Opts.Diagnostics.Print)
    return true;
  errs() << "[poseidon] " << name << ": site " << D.siteId << ", "
         << D.ptrParams.size() << " shadow buffer(s), output(s)";
  for (unsigned p : D.ptrParams)
    if (seedWidth[p])
      errs() << ' ' << p;
  if (!B.getReturnType()->isVoidTy())
    errs() << " return";
  errs() << ", " << calls.size() << " call(s)\n";
  return true;
}

/// `p` with the object it points into replaced by `shadow`.
static Value *intoShadow(Value *p, Value *obj, AllocaInst *shadow,
                         IRBuilder<> &B) {
  if (p == obj)
    return shadow;
  auto *I = cast<Instruction>(p);
  Value *base = intoShadow(I->getOperand(0), obj, shadow, B);
  Instruction *copy = I->clone();
  copy->setOperand(0, base);
  return B.Insert(copy, p->getName() + ".shadow");
}

/// A local object like `obj` (localObject), for its shadow: next to an alloca,
/// in the entry block for an aggregate the function takes by value, which is
/// in the stack's address space as the parameter is.
static AllocaInst *shadowObject(Value *obj) {
  if (auto *AI = dyn_cast<AllocaInst>(obj))
    return new AllocaInst(AI->getAllocatedType(), AI->getAddressSpace(),
                          AI->getArraySize(), AI->getAlign(),
                          AI->getName() + ".shadow",
                          std::next(AI->getIterator()));
  auto *A = cast<Argument>(obj);
  Function &F = *A->getParent();
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *T = A->getPointeeInMemoryValueType();
  Align align =
      std::max(A->getParamAlign().valueOrOne(), DL.getABITypeAlign(T));
  return new AllocaInst(T, DL.getAllocaAddrSpace(), nullptr, align,
                        A->getName() + ".shadow",
                        F.getEntryBlock().getFirstInsertionPt());
}

static uint64_t objectBytes(Value *obj, const DataLayout &DL) {
  if (auto *AI = dyn_cast<AllocaInst>(obj))
    return AI->getAllocationSize(DL)->getFixedValue();
  return DL.getTypeAllocSize(
      cast<Argument>(obj)->getPointeeInMemoryValueType());
}

/// The parameters the site reads a pointer through. A shadow of such an object
/// would have to hold shadows of those pointers, and one of zeros hands the
/// reverse pass null ones, so the object is passed as a constant instead.
static void collectPointerHolders(Function &clone,
                                  SmallVectorImpl<bool> &holds) {
  holds.assign(clone.arg_size(), false);
  for (Instruction &I : instructions(clone))
    if (auto *LI = dyn_cast<LoadInst>(&I))
      if (LI->getType()->isPtrOrPtrVectorTy())
        if (int idx = traceToArgIndex(LI->getPointerOperand()); idx >= 0)
          holds[idx] = true;
}

/// Whether the program may read what `call` leaves in `obj`, a local object
/// (localObject): an instruction the call reaches before the object's lifetime
/// ends, the call itself when it runs again, or whatever the object's address
/// escapes to. A by-value parameter's lifetime ends with the function. What the
/// site writes and nothing reads reaches no result of the program (a JVP that
/// writes the primal outputs next to the tangents it is called for), so it is
/// no output to seed.
static bool readAfterCall(Value *obj, CallInst *call) {
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

namespace {

/// `p` as a root pointer, the variable index terms of its address arithmetic
/// (value and scale, in value order) and a constant byte offset, through every
/// GEP and pointer cast. Two pointers with the same root and terms lie a
/// constant distance apart, however the frontend ordered the casts and GEPs.
struct Address {
  const Value *root = nullptr;
  SmallVector<std::pair<const Value *, int64_t>, 2> terms;
  int64_t offset = 0;
};

} // namespace

static Address addressOf(const Value *p, const DataLayout &DL) {
  Address a;
  MapVector<const Value *, int64_t> terms;
  for (;;) {
    if (auto *GEP = dyn_cast<GEPOperator>(p)) {
      unsigned bits = DL.getIndexTypeSizeInBits(GEP->getType());
      SmallMapVector<Value *, APInt, 4> variable;
      APInt constant(bits, 0);
      if (!GEP->collectOffset(DL, bits, variable, constant))
        break;
      a.offset += constant.getSExtValue();
      for (auto &[v, scale] : variable)
        terms[v] += scale.getSExtValue();
      p = GEP->getPointerOperand();
    } else if (auto *O = dyn_cast<Operator>(p);
               O && (O->getOpcode() == Instruction::BitCast ||
                     O->getOpcode() == Instruction::AddrSpaceCast)) {
      p = O->getOperand(0);
    } else {
      break;
    }
  }
  a.root = p;
  for (auto &[v, scale] : terms)
    if (scale)
      a.terms.push_back({v, scale});
  llvm::sort(a.terms);
  return a;
}

namespace {

/// The memory a call reaches through pointers of no object the pass knows
/// (accessedExtent) that lie at constant distances from one another
/// (addressOf), in bytes from their common root and terms: one shadow covers
/// it, so that their shadows lie as far apart as they do and overlap where they
/// do. Its byte `start` mirrors the byte `lo`, chosen so that the shadow keeps
/// the strictest alignment an access claims.
struct Region {
  const Value *root = nullptr;
  SmallVector<std::pair<const Value *, int64_t>, 2> terms;
  Value *pointer = nullptr;
  unsigned param = 0;
  int64_t pointerAt = 0, lo = 0, hi = 0, alignedAt = 0, start = 0;
  uint64_t align = 1;
  unsigned width = 0;
  SmallVector<std::pair<int64_t, int64_t>, 2> writes;
  AllocaInst *shadow = nullptr;
};

} // namespace

/// The marker in the explicit form, with a shadow for every pointer marked
/// poseidon_shadow: a local object like the one the pointer points into, zero,
/// and seeded by the kernel path's rule when the site writes into it and the
/// program reads it afterwards (readAfterCall). A pointer into no object the
/// pass knows has a local object as its shadow that covers what the call
/// reaches through it, zero and seeded wherever the call writes, since what it
/// writes outlives the caller. Either is made at every execution of the call,
/// because the object the call sees is a new one each time: its storage is
/// reused, its value is not.
namespace {
/// What withShadows makes for the pointers marked poseidon_shadow: a local
/// object's shadow and seed width, per object; a shadow over an extent, per
/// region; which marker argument takes which; and the pointers that need
/// none (constant) or whose object the program does not read afterwards
/// (unread).
struct ShadowPlan {
  MapVector<Value *, std::pair<AllocaInst *, unsigned>> objects;
  DenseMap<unsigned, Value *> objectOf;
  DenseMap<Value *, unsigned> objectParam;
  SmallVector<Region, 4> regions;
  DenseMap<unsigned, std::pair<unsigned, int64_t>> regionOf;
  DenseSet<unsigned> constant;
  SmallPtrSet<Value *, 4> unread;
};

/// The memory a shadow stands for: bytes from a pointer, the first parameter
/// that reaches it, and whether the call writes floating-point data into it,
/// for the overlap check.
struct Reach {
  Value *pointer;
  int64_t lo, hi;
  unsigned param;
  bool written, fromExtent;
};
} // namespace

/// Sorts each pointer `marker` marks into `plan`: one the call needs no
/// shadow for, one into a local object of the caller, or one into the
/// constant range the call reaches through it. An error if a pointer is none
/// of these.
static Error planShadows(CallInst *CI, Function &clone,
                         const MarkerArgs &marker, ArrayRef<unsigned> seedWidth,
                         StringRef site, ShadowPlan &plan) {
  SmallVector<bool, 8> holdsPointers;
  collectPointerHolders(clone, holdsPointers);
  Function &caller = *CI->getFunction();
  const DataLayout &DL = CI->getModule()->getDataLayout();
  for (const MarkerArgs::MadeShadow &S : marker.madeShadows) {
    Value *p = CI->getArgOperand(S.arg);
    if (pointsNowhere(p) || holdsPointers[S.param]) {
      plan.constant.insert(S.activity);
      continue;
    }
    if (Value *obj = localObject(p, &caller)) {
      unsigned &width = plan.objects[obj].second;
      width = std::max(width, seedWidth[S.param]);
      plan.objectOf[S.arg] = obj;
      plan.objectParam.try_emplace(obj, S.param);
      continue;
    }
    Expected<PointerExtent> ext = accessedExtent(*clone.getArg(S.param));
    if (!ext)
      return userError(
          *CI, site +
                   " no longer reaches a constant range of memory through "
                   "argument " +
                   Twine(S.param + 1) + " (a pointer the call " +
                   toString(ext.takeError()) +
                   "), so it cannot be a Poseidon site. Copy the argument "
                   "into a local object, or pass its shadow in the call: "
                   "enzyme_dup before the pointer and the shadow after it.");
    if (!ext->accessed()) {
      plan.constant.insert(S.activity);
      continue;
    }
    Address addr = addressOf(p, DL);
    const int64_t at = addr.offset;
    SmallVector<Region, 4> &regions = plan.regions;
    unsigned k =
        llvm::find_if(regions,
                      [&](const Region &R) {
                        return R.root == addr.root && R.terms == addr.terms;
                      }) -
        regions.begin();
    if (k == regions.size()) {
      Region &R = regions.emplace_back();
      R.root = addr.root;
      R.terms = addr.terms;
      R.pointer = p;
      R.param = S.param;
      R.pointerAt = at;
      R.lo = at + ext->lo;
      R.hi = at + ext->hi;
    }
    Region &R = regions[k];
    R.lo = std::min(R.lo, at + ext->lo);
    R.hi = std::max(R.hi, at + ext->hi);
    if (ext->align > R.align) {
      R.align = ext->align;
      R.alignedAt = at + ext->alignedAt;
    }
    if (ext->written()) {
      R.writes.push_back({at + ext->writeLo, at + ext->writeHi});
      R.width = std::max(R.width, seedWidth[S.param]);
    }
    plan.regionOf[S.arg] = {k, at};
  }
  return Error::success();
}

/// The memory each shadow of `plan` stands for.
static SmallVector<Reach, 8> reachesOf(const ShadowPlan &plan,
                                       const DataLayout &DL) {
  SmallVector<Reach, 8> reaches;
  for (auto &[obj, entry] : plan.objects)
    reaches.push_back({obj, 0, (int64_t)objectBytes(obj, DL),
                       plan.objectParam.lookup(obj), entry.second != 0, false});
  for (const Region &R : plan.regions)
    reaches.push_back({R.pointer, R.lo - R.pointerAt, R.hi - R.pointerAt,
                       R.param, R.width != 0, true});
  return reaches;
}

/// The type a shadow of `width`-byte output elements is seeded as: null for
/// no output; an error at `CI` for a width without a rule.
static Expected<Type *> seedTypeOf(CallInst *CI, unsigned width) {
  LLVMContext &C = CI->getContext();
  if (width == 8)
    return Type::getDoubleTy(C);
  if (width == 4)
    return Type::getFloatTy(C);
  if (width)
    return userError(*CI, "no seed rule for a " + Twine(width) +
                              "-byte output element");
  return nullptr;
}

/// Sets `n` elements of type `T` from `at` to 1.0 before `CI`, at every
/// execution.
static void seedOnes(CallInst *CI, Value *at, Type *T, uint64_t n) {
  if (!n)
    return;
  auto [in, idx] = SplitBlockAndInsertSimpleForLoop(
      ConstantInt::get(Type::getInt64Ty(CI->getContext()), n),
      CI->getIterator());
  IRBuilder<> L(in);
  L.CreateStore(ConstantFP::get(T, 1.0), L.CreateGEP(T, at, idx));
}

/// Makes the shadows of `plan` before `CI`, zero and seeded where the call
/// writes; an error, before any change, for an output element without a seed
/// rule.
static Error emitShadows(CallInst *CI, ShadowPlan &plan) {
  for (auto &[obj, entry] : plan.objects)
    if (Error E = seedTypeOf(CI, entry.second).takeError())
      return E;
  for (const Region &R : plan.regions)
    if (Error E = seedTypeOf(CI, R.width).takeError())
      return E;

  const DataLayout &DL = CI->getModule()->getDataLayout();
  for (auto &[obj, entry] : plan.objects) {
    auto &[shadow, width] = entry;
    Type *T = cantFail(seedTypeOf(CI, width));
    shadow = shadowObject(obj);
    uint64_t bytes = objectBytes(obj, DL);
    IRBuilder<> B(CI);
    B.CreateMemSet(shadow, B.getInt8(0), bytes, shadow->getAlign());
    if (T)
      seedOnes(CI, shadow, T, bytes / width);
  }

  Function &caller = *CI->getFunction();
  IRBuilder<> entry(&*caller.getEntryBlock().getFirstInsertionPt());
  for (Region &R : plan.regions) {
    Type *T = cantFail(seedTypeOf(CI, R.width));
    const int64_t align = R.align;
    R.start = ((R.lo - R.alignedAt) % align + align) % align;
    const uint64_t bytes = R.start + (R.hi - R.lo);
    R.shadow =
        entry.CreateAlloca(ArrayType::get(entry.getInt8Ty(), bytes),
                           DL.getAllocaAddrSpace(), nullptr, "poseidon.shadow");
    R.shadow->setAlignment(Align(align));
    IRBuilder<> B(CI);
    B.CreateMemSet(R.shadow, B.getInt8(0), bytes, Align(align));
    if (T)
      for (auto [lo, hi] : R.writes)
        seedOnes(CI,
                 B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), R.shadow,
                                              R.start + (lo - R.lo)),
                 T, (hi - lo) / R.width);
  }
  return Error::success();
}

/// A shadow made from an extent is an object of its own, which stands for the
/// memory its pointers reach only where no other shadow's memory overlaps it:
/// the reverse pass would miss what the call writes through one pointer and
/// reads through the other. Stops the profiling run, before `CI`, where such
/// memory overlaps and the call writes into one of the two.
static void emitOverlapChecks(CallInst *CI, ArrayRef<Reach> reaches,
                              StringRef site) {
  Module &M = *CI->getModule();
  for (auto g = reaches.begin(); g != reaches.end(); ++g)
    for (auto h = std::next(g); h != reaches.end(); ++h) {
      if (!(g->fromExtent || h->fromExtent) || !(g->written || h->written))
        continue;
      IRBuilder<> B(CI);
      auto bounds = [&](const Reach &R) {
        Value *addr = B.CreatePtrToInt(
            B.CreatePointerBitCastOrAddrSpaceCast(R.pointer, B.getPtrTy()),
            B.getInt64Ty());
        Value *lo = B.CreateAdd(addr, B.getInt64(R.lo));
        Value *hi = B.CreateAdd(addr, B.getInt64(R.hi));
        return std::make_pair(lo, hi);
      };
      auto [glo, ghi] = bounds(*g);
      auto [hlo, hhi] = bounds(*h);
      Value *gBeforeH = B.CreateICmpULT(glo, hhi);
      Value *hBeforeG = B.CreateICmpULT(hlo, ghi);
      Value *overlap = B.CreateAnd(gBeforeH, hBeforeG);
      Instruction *stop = SplitBlockAndInsertIfThen(overlap, CI->getIterator(),
                                                    /*Unreachable=*/true);
      IRBuilder<> S(stop);
      FunctionCallee report = M.getOrInsertFunction(
          Target::of(M).probes().OverlappingArguments,
          FunctionType::get(S.getVoidTy(),
                            {S.getPtrTy(), S.getInt32Ty(), S.getInt32Ty()},
                            false));
      S.CreateCall(report, {getStringPointer(M, "", site),
                            S.getInt32(std::min(g->param, h->param) + 1),
                            S.getInt32(std::max(g->param, h->param) + 1)});
    }
}

/// `CI` again, with each pointer `marker` marks followed by its shadow from
/// `plan` and its activity marker made explicit; replaces `CI`.
static CallInst *emitShadowedCall(CallInst *CI, const MarkerArgs &marker,
                                  ShadowPlan &plan) {
  LLVMContext &C = CI->getContext();
  IRBuilder<> B(CI);
  SmallVector<Value *, 32> args;
  for (unsigned i = 0; i < CI->arg_size(); ++i) {
    Value *a = CI->getArgOperand(i);
    if (llvm::any_of(marker.madeShadows,
                     [&](const auto &S) { return S.activity == i; }))
      a = activityArg(C,
                      plan.constant.count(i) ? "enzyme_const" : "enzyme_dup");
    args.push_back(a);
    if (Value *obj = plan.objectOf.lookup(i))
      args.push_back(intoShadow(a, obj, plan.objects[obj].first, B));
    if (auto it = plan.regionOf.find(i); it != plan.regionOf.end()) {
      auto [k, at] = it->second;
      const Region &R = plan.regions[k];
      args.push_back(B.CreatePointerBitCastOrAddrSpaceCast(
          B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), R.shadow,
                                       R.start + (at - R.lo)),
          a->getType()));
    }
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
  return NC;
}

/// Prints, under -poseidon-print, the shadows `plan` made for the site `site`.
static void printShadows(StringRef site, const MarkerArgs &marker,
                         const ShadowPlan &plan, ArrayRef<unsigned> seedWidth) {
  errs() << "[poseidon] " << site << ": "
         << plan.objects.size() + plan.regions.size()
         << " shadow object(s), output(s)";
  for (const MarkerArgs::MadeShadow &S : marker.madeShadows)
    if (!plan.constant.count(S.activity) && seedWidth[S.param] &&
        !plan.unread.count(plan.objectOf.lookup(S.arg)))
      errs() << ' ' << S.param;
  if (!plan.unread.empty()) {
    errs() << ", not read after the call:";
    for (const MarkerArgs::MadeShadow &S : marker.madeShadows)
      if (plan.unread.count(plan.objectOf.lookup(S.arg)))
        errs() << ' ' << S.param;
  }
  errs() << '\n';
}

static Expected<CallInst *> withShadows(CallInst *CI, Function &clone,
                                        const MarkerArgs &marker, bool print) {
  SmallVector<unsigned, 8> seedWidth;
  collectOutputParams(clone, seedWidth);
  const std::string site = perCallSiteName(markerTargetOrNull(CI)->getName());

  ShadowPlan plan;
  if (Error E = planShadows(CI, clone, marker, seedWidth, site, plan))
    return std::move(E);
  SmallVector<Reach, 8> reaches =
      reachesOf(plan, CI->getModule()->getDataLayout());
  // An object the program does not read after the call is no output of it.
  for (auto &[obj, entry] : plan.objects)
    if (entry.second && !readAfterCall(obj, CI)) {
      entry.second = 0;
      plan.unread.insert(obj);
    }
  if (Error E = emitShadows(CI, plan))
    return std::move(E);
  emitOverlapChecks(CI, reaches, site);
  CallInst *NC = emitShadowedCall(CI, marker, plan);
  if (print)
    printShadows(site, marker, plan, seedWidth);
  return NC;
}

Expected<bool> profileSite(Session &S, const Site &site,
                           DenseMap<Function *, Function *> &clones,
                           unsigned &nextSiteId) {
  switch (site.form) {
  case Site::Form::Marker:
    return profileMarker(S, site.marker, clones, nextSiteId);
  case Site::Form::Kernel:
    return profileKernel(S, *site.fn, nextSiteId);
  case Site::Form::HostBody:
    return profileHostBody(S, *site.fn, nextSiteId);
  }
  llvm_unreachable("every site form is profiled");
}

} // namespace poseidon
