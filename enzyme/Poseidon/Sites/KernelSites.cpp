#include "KernelSites.h"

#include "Annotations.h"
#include "HostSites.h"
#include "Markers.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"
#include "Techniques/Matmul/HostDispatch.h"

#include "llvm/ADT/STLExtras.h"
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

// The local object of `caller` that `p` points into, through the address
// arithmetic its shadow can repeat; null if there is none.
AllocaInst *localObject(Value *p, const Function *caller) {
  for (;;) {
    if (auto *AI = dyn_cast<AllocaInst>(p))
      return AI->getFunction() == caller && AI->isStaticAlloca() ? AI : nullptr;
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

std::string perCallSiteName(const Function &B) {
  auto [fnSymbol, where] = B.getName().split(".poseidon.");
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
// profileSite, which can only make shadows of local objects of the caller;
// inlining has put the objects a call passes there by now.
bool lowerDeviceCallSites(Module &M) {
  if (!Target::of(M).isDevice())
    return false;
  SmallVector<Function *, 4> bodies = hostSiteBodies(M);
  if (bodies.empty())
    return false;
  LLVMContext &C = M.getContext();
  auto activity = [&](StringRef name) {
    return MetadataAsValue::get(C, MDString::get(C, name));
  };
  for (Function *B : bodies) {
    double tau = takeHostBody(*B);
    for (CallBase *CB : callsOf(*B)) {
      Function *caller = CB->getFunction();
      std::string why;
      if (!isa<CallInst>(CB))
        why = "is reached through an invoke";
      for (Use &U : CB->args())
        if (why.empty() && U->getType()->isPointerTy() &&
            !pointsNowhere(U.get()) && !localObject(U.get(), caller))
          why = ("takes argument " + Twine(U.getOperandNo() + 1) +
                 " as a pointer into memory that is not a local object of " +
                 displayName(*caller))
                    .str();
      if (!why.empty()) {
        warn(CB, perCallSiteName(*B) + " " + why +
                     ", so this call is not a Poseidon site: the pass makes "
                     "shadows of local objects only, and "
                     "__poseidon_fp_optimize takes the caller's own. The call "
                     "runs as written.");
        markOutlineTransparent(B);
        continue;
      }
      SmallVector<Value *, 16> args{B};
      if (tau > 0.0) {
        args.push_back(activity("poseidon_tau"));
        args.push_back(ConstantFP::get(Type::getDoubleTy(C), tau));
      }
      if (!CB->getType()->isVoidTy())
        args.push_back(activity("enzyme_primal_return"));
      for (Value *a : CB->args()) {
        Type *T = a->getType();
        args.push_back(activity(T->isPointerTy()         ? "poseidon_shadow"
                                : T->isFloatingPointTy() ? "enzyme_out"
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
