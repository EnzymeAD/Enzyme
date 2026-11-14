#include "KernelSites.h"

#include "Annotations.h"
#include "HostSites.h"
#include "LaunchDescriptors.h"
#include "Markers.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

namespace poseidon {

// The body a site's call targets is a device of this pass and must not reach
// codegen as a call: under -fcuda-rdc a call that is not folded back is a real
// ABI call, and the kernel entry then pays its register and frame budget
// (measured: 148 registers and one resident block per SM against 70 and three
// for the same drift kernel without the annotation, a 2x wall-clock gap on
// GH200; and 91 registers against 62 for the elasticity reduce, which at 648
// threads per block exceeds the 64K per-block register file and makes the
// launch fail outright). Marked on the solve path for both annotation forms, so
// that a site the solve leaves unchanged is inlined exactly like one it
// rewrote; the AlwaysInlinerPass registerPasses schedules right after this pass
// does the folding.
void markOutlineTransparent(Function *body) {
  body->removeFnAttr(Attribute::NoInline);
  body->removeFnAttr(Attribute::OptimizeNone);
  body->addFnAttr(Attribute::AlwaysInline);
}

namespace {

// Whether every use of `P`, a pointer to an aggregate, is one reroot can
// rebuild over a pointer in another address space.
bool reroots(const Value *P) {
  for (const Use &U : P->uses()) {
    const User *I = U.getUser();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      if (U.getOperandNo() != GEP->getPointerOperandIndex() || !reroots(GEP))
        return false;
    } else if (isa<MemTransferInst>(I)) {
      if (U.getOperandNo() != 1)
        return false;
    } else if (!isa<LoadInst>(I) && !isa<AddrSpaceCastInst>(I)) {
      return false;
    }
  }
  return true;
}

// The uses of `from` rebuilt over `to`, which points to the same aggregate
// through the generic address space.
void reroot(Value *from, Value *to) {
  for (Use &U : make_early_inc_range(from->uses())) {
    auto *I = cast<Instruction>(U.getUser());
    IRBuilder<> B(I);
    if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      SmallVector<Value *, 4> idx(GEP->indices());
      Value *N = B.CreateGEP(GEP->getSourceElementType(), to, idx, "",
                             GEP->getNoWrapFlags());
      N->takeName(GEP);
      reroot(GEP, N);
      GEP->eraseFromParent();
    } else if (auto *MT = dyn_cast<MemTransferInst>(I)) {
      CallInst *N = isa<MemMoveInst>(MT)
                        ? B.CreateMemMove(MT->getRawDest(), MT->getDestAlign(),
                                          to, MT->getSourceAlign(),
                                          MT->getLength(), MT->isVolatile())
                        : B.CreateMemCpy(MT->getRawDest(), MT->getDestAlign(),
                                         to, MT->getSourceAlign(),
                                         MT->getLength(), MT->isVolatile());
      N->copyMetadata(*MT);
      MT->eraseFromParent();
    } else if (isa<AddrSpaceCastInst>(I) && I->getType() == to->getType()) {
      I->replaceAllUsesWith(to);
      I->eraseFromParent();
    } else {
      U.set(to);
    }
  }
}

} // namespace

// Make the kernel a wrapper around its own body so that both phases see the
// structure they already handle: a function holding the annotated computation,
// reached through one call whose arguments are the site's inputs. Returns that
// call, or null if the kernel cannot be a site.
CallInst *outlineKernelBody(Function &K, Function *&bodyOut) {
  if (K.isVarArg() || !K.getReturnType()->isVoidTy()) {
    K.getContext().diagnose(DiagnosticInfoUnsupported(
        K,
        "Poseidon: an annotated kernel must be a void, non-variadic "
        "function",
        K.getSubprogram(), DS_Warning));
    return nullptr;
  }
  Module &M = *K.getParent();
  // An AMDGPU kernel takes an aggregate by reference into its constant
  // argument segment; the body takes it through a generic pointer, which the
  // profiled kernel can point at a copy that holds the shadows.
  SmallVector<Type *, 8> params(K.getFunctionType()->params());
  for (Argument &A : K.args())
    if (A.hasByRefAttr()) {
      if (!reroots(&A)) {
        K.getContext().diagnose(DiagnosticInfoUnsupported(
            K,
            "Poseidon: " + displayName(K) + " uses its aggregate parameter " +
                Twine(A.getArgNo() + 1) +
                " otherwise than by loading or copying from it; the kernel "
                "is not a site",
            K.getSubprogram(), DS_Warning));
        return nullptr;
      }
      params[A.getArgNo()] = PointerType::getUnqual(M.getContext());
    }
  // Named alike with and without -fgpu-rdc, which profile generation adds and
  // a solve need not: the profile is found by this name.
  Function *B = Function::Create(
      FunctionType::get(K.getReturnType(), params, false),
      GlobalValue::InternalLinkage,
      withoutExternalizedPostfix(K.getName()) + "_poseidon_body", &M);
  B->setAttributes(K.getAttributes());
  // The body reads the kernel's own copy of a by-value aggregate. Left byval,
  // folding the body back would copy the aggregate a second time.
  for (Argument &A : B->args())
    if (A.hasByValAttr() || A.hasByRefAttr()) {
      A.removeAttr(Attribute::ByVal);
      A.removeAttr(Attribute::ByRef);
      A.addAttr(Attribute::get(M.getContext(), kByValueParam));
    }
  // The body is an ordinary device function; only the kernel stays an entry.
  B->setCallingConv(CallingConv::C);
  B->setSubprogram(K.getSubprogram());
  K.setSubprogram(nullptr);
  B->splice(B->begin(), &K);
  for (auto i = K.arg_begin(), j = B->arg_begin(); i != K.arg_end(); ++i, ++j) {
    j->setName(i->getName());
    if (i->getType() != j->getType())
      reroot(&*i, &*j);
    else
      i->replaceAllUsesWith(&*j);
  }

  IRBuilder<> Builder(BasicBlock::Create(M.getContext(), "entry", &K));
  SmallVector<Value *, 8> args;
  for (Argument &A : K.args())
    args.push_back(A.hasByRefAttr()
                       ? Builder.CreateAddrSpaceCast(
                             &A, B->getArg(A.getArgNo())->getType())
                       : &A);
  CallInst *call = Builder.CreateCall(B, args);
  call->setCallingConv(B->getCallingConv());
  Builder.CreateRetVoid();
  K.addFnAttr(kSiteLowered);
  // Folded back, the body's code still names the subprogram it took along;
  // without it again the kernel compiles with no line table, and ptxas then
  // allocates it differently.
  K.addFnAttr(kHostWrapper);
  bodyOut = B;
  return call;
}

// --------------------------------------------------------------------------
// Device call sites: __poseidon_optimize in device code
// --------------------------------------------------------------------------

// The object of `caller` that `p` points into, through the address arithmetic
// its shadow can repeat, when the pass knows the object's whole extent: a
// static alloca, or an aggregate `caller` takes by value (byval). Null if there
// is none. (An AMDGPU kernel takes an aggregate by reference into its constant
// argument segment, and the frontend copies it into an alloca once its address
// is taken.)
Value *localObject(Value *p, const Function *caller) {
  for (;;) {
    if (auto *AI = dyn_cast<AllocaInst>(p))
      return AI->getFunction() == caller && AI->isStaticAlloca() ? AI : nullptr;
    if (auto *A = dyn_cast<Argument>(p))
      return A->getParent() == caller && A->hasByValAttr() ? A : nullptr;
    if (auto *GEP = dyn_cast<GetElementPtrInst>(p))
      p = GEP->getPointerOperand();
    else if (isa<BitCastInst, AddrSpaceCastInst>(p))
      p = cast<Instruction>(p)->getOperand(0);
    else
      return nullptr;
  }
}

// A pointer argument that points into no object: null, or undefined because the
// call's copy never reads it and the optimizer stopped passing the object (the
// address of an empty functor).
bool pointsNowhere(Value *p) {
  return isa<ConstantPointerNull>(p) || isa<UndefValue>(p);
}

namespace {

// accessedExtent: every use of the pointer, through constant address
// arithmetic and into the functions it is passed to.
struct ExtentWalk {
  const DataLayout &DL;
  PointerExtent ext;
  bool anyAccess = false, anyWrite = false;
  std::string why;
  SmallSet<std::pair<const Argument *, int64_t>, 4> entered;
  // The parameters the pointer is in on the way down.
  SmallPtrSet<const Argument *, 4> active;

  bool fail(const Twine &reason) {
    why = reason.str();
    return false;
  }

  void claim(int64_t off, MaybeAlign align) {
    if (align && align->value() > ext.align) {
      ext.align = align->value();
      ext.alignedAt = off;
    }
  }

  void access(int64_t off, uint64_t bytes, bool write) {
    if (!bytes)
      return;
    const int64_t end = off + (int64_t)bytes;
    ext.lo = anyAccess ? std::min(ext.lo, off) : off;
    ext.hi = anyAccess ? std::max(ext.hi, end) : end;
    anyAccess = true;
    if (!write)
      return;
    ext.writeLo = anyWrite ? std::min(ext.writeLo, off) : off;
    ext.writeHi = anyWrite ? std::max(ext.writeHi, end) : end;
    anyWrite = true;
  }

  bool memoryAccess(int64_t off, Type *T, Align align, bool write) {
    TypeSize bytes = DL.getTypeStoreSize(T);
    if (bytes.isScalable())
      return fail("accesses a scalable vector through");
    access(off, bytes.getFixedValue(), write);
    claim(off, align);
    return true;
  }

  // An optimizer may load anywhere a parameter's attributes call
  // dereferenceable, also after the extent is taken, so the extent covers it.
  // A function that receives the pointer again at another offset while it is
  // being walked reaches no constant range through it.
  bool enter(Argument &A, int64_t off) {
    if (!entered.insert({&A, off}).second)
      return true;
    if (!active.insert(&A).second)
      return fail("passes to " + displayName(*A.getParent()) +
                  ", which passes it on to itself at another offset");
    access(off,
           std::max(A.getDereferenceableBytes(),
                    A.getDereferenceableOrNullBytes()),
           false);
    claim(off, A.getParamAlign());
    const bool ok = walk(&A, off);
    active.erase(&A);
    return ok;
  }

  bool walk(Value *P, int64_t off) {
    for (Use &U : P->uses()) {
      auto *I = dyn_cast<Instruction>(U.getUser());
      if (!I)
        return fail("uses otherwise than to load and store through");
      if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
        APInt delta(DL.getIndexTypeSizeInBits(GEP->getType()), 0);
        if (!GEP->accumulateConstantOffset(DL, delta))
          return fail("indexes with a value that is not a constant");
        if (!walk(GEP, off + delta.getSExtValue()))
          return false;
      } else if (isa<BitCastInst, AddrSpaceCastInst>(I)) {
        if (!walk(I, off))
          return false;
      } else if (auto *LI = dyn_cast<LoadInst>(I)) {
        if (!memoryAccess(off, LI->getType(), LI->getAlign(), false))
          return false;
      } else if (auto *SI = dyn_cast<StoreInst>(I)) {
        if (U.getOperandNo() != SI->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, SI->getValueOperand()->getType(), SI->getAlign(),
                          true))
          return false;
      } else if (auto *RMW = dyn_cast<AtomicRMWInst>(I)) {
        if (U.getOperandNo() != RMW->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, RMW->getValOperand()->getType(), RMW->getAlign(),
                          true))
          return false;
      } else if (auto *CX = dyn_cast<AtomicCmpXchgInst>(I)) {
        if (U.getOperandNo() != CX->getPointerOperandIndex())
          return fail("stores to memory");
        if (!memoryAccess(off, CX->getNewValOperand()->getType(),
                          CX->getAlign(), true))
          return false;
      } else if (isa<MemTransferInst, MemSetInst>(I)) {
        auto *MI = cast<MemIntrinsic>(I);
        auto *len = dyn_cast<ConstantInt>(MI->getLength());
        if (!len)
          return fail("copies or sets through with a length that is not a "
                      "constant");
        const bool dest = U.getOperandNo() == 0;
        access(off, len->getZExtValue(), dest);
        claim(off, dest ? MI->getDestAlign()
                        : cast<MemTransferInst>(MI)->getSourceAlign());
      } else if (auto *II = dyn_cast<IntrinsicInst>(I);
                 II &&
                 (II->isLifetimeStartOrEnd() || II->isAssumeLikeIntrinsic())) {
        continue;
      } else if (isa<ICmpInst>(I)) {
        continue;
      } else if (auto *CB = dyn_cast<CallBase>(I)) {
        if (!CB->isArgOperand(&U))
          return fail("calls as a function");
        const unsigned no = CB->getArgOperandNo(&U);
        if (CB->doesNotAccessMemory(no) && CB->doesNotCapture(no))
          continue;
        if (Type *T = CB->getParamByValType(no)) {
          if (!memoryAccess(off, T, CB->getParamAlign(no).valueOrOne(), false))
            return false;
          continue;
        }
        Function *callee = CB->getCalledFunction();
        if (!callee || callee->isDeclaration() || no >= callee->arg_size())
          return fail("passes to " +
                      (callee ? displayName(*callee)
                              : std::string("an indirect call")) +
                      ", whose body the pass cannot see");
        if (!enter(*callee->getArg(no), off))
          return false;
      } else if (isa<PHINode, SelectInst>(I)) {
        return fail("merges with another pointer");
      } else if (isa<PtrToIntInst>(I)) {
        return fail("converts to an integer");
      } else if (isa<ReturnInst>(I)) {
        return fail("returns");
      } else {
        return fail("uses otherwise than to load and store through");
      }
    }
    return true;
  }
};

} // namespace

std::optional<PointerExtent> accessedExtent(Argument &A, std::string &why) {
  ExtentWalk W{A.getParent()->getParent()->getDataLayout(), {}};
  if (!W.enter(A, 0)) {
    why = W.why;
    return std::nullopt;
  }
  return W.ext;
}

std::string perCallSiteName(StringRef symbol) {
  auto [fnSymbol, where] = symbol.split(".poseidon.");
  std::string name = llvm::demangle(fnSymbol.str());
  if (!where.empty()) {
    auto [callerSymbol, k] = where.rsplit('.');
    name += " at call " + k.str() + " in " + llvm::demangle(callerSymbol.str());
  }
  return name;
}

// __poseidon_optimize in device code: every call of the copy lowerPerCallSites
// made becomes a marker over that copy, so that profiling, deferral and the
// solve are the marker's. Its pointers are marked poseidon_shadow for
// profileSite, which makes the shadow of a pointer into an object of the
// caller whose extent it knows (localObject; inlining has put the objects a
// call passes there by now) or else of the constant range the call reaches
// through it (accessedExtent).
bool lowerDeviceCallSites(Module &M) {
  if (!Target::of(M).isDevice())
    return false;
  SmallVector<Function *, 4> bodies = hostSiteBodies(M);
  if (bodies.empty())
    return false;
  LLVMContext &C = M.getContext();
  for (Function *B : bodies) {
    double tau = takeHostBody(*B);
    for (CallBase *CB : callsOf(*B)) {
      if (!isa<CallInst>(CB)) {
        warn(CB, perCallSiteName(B->getName()) +
                     " is reached through an invoke, so this call is not a "
                     "Poseidon site. The call runs as written.");
        markOutlineTransparent(B);
        continue;
      }
      std::string why;
      for (Use &U : CB->args()) {
        std::string reason;
        if (!U->getType()->isPointerTy() || pointsNowhere(U.get()) ||
            localObject(U.get(), CB->getFunction()) ||
            accessedExtent(*B->getArg(U.getOperandNo()), reason))
          continue;
        why +=
            (Twine(why.empty() ? "" : "; ") + "argument " +
             Twine(U.getOperandNo() + 1) + " is a pointer the call " + reason)
                .str();
      }
      if (!why.empty()) {
        warn(CB, perCallSiteName(B->getName()) +
                     " is not a Poseidon site: " + why +
                     ". The pass makes the shadow of a pointer into a local "
                     "object of the caller or an aggregate it takes by value, "
                     "or else of the constant range of memory the call "
                     "reaches through the pointer. Copy such an argument into "
                     "a local object, or write the call with "
                     "__poseidon_fp_optimize and shadows of your own. The "
                     "call runs as written.");
        markOutlineTransparent(B);
        continue;
      }
      SmallVector<Value *, 16> args{B};
      if (tau > 0.0) {
        args.push_back(activityArg(C, "poseidon_tau"));
        args.push_back(ConstantFP::get(Type::getDoubleTy(C), tau));
      }
      if (!CB->getType()->isVoidTy())
        args.push_back(activityArg(C, "enzyme_primal_return"));
      for (Value *a : CB->args()) {
        Type *T = a->getType();
        args.push_back(activityArg(C, T->isPointerTy() ? "poseidon_shadow"
                                      : T->isFloatingPointTy()
                                          ? "enzyme_out"
                                          : "enzyme_const"));
        args.push_back(a);
      }
      auto *FT = FunctionType::get(CB->getType(), {PointerType::getUnqual(C)},
                                   /*isVarArg=*/true);
      FunctionCallee fn = M.getOrInsertFunction("__poseidon_fp_optimize", FT);
      IRBuilder<> Builder(CB);
      CallInst *CI = Builder.CreateCall(FT, fn.getCallee(), args);
      CI->setDebugLoc(CB->getDebugLoc());
      AttributeList old = CB->getAttributes();
      CI->setAttributes(
          AttributeList::get(C, old.getFnAttrs(), old.getRetAttrs(), {}));
      CB->replaceAllUsesWith(CI);
      CB->eraseFromParent();
    }
  }
  return true;
}

// Undo outlineKernelBody: a filtered site has to leave the kernel exactly as it
// was found.
void inlineKernelBody(CallInst *call, Function *body) {
  InlineFunctionInfo IFI;
  (void)InlineFunction(*call, IFI);
  if (body->use_empty())
    body->eraseFromParent();
}

} // namespace poseidon
