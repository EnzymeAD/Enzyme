#include "DeviceMath.h"

#include "Utils.h"

#include "Flags.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

namespace poseidon {

namespace {

// Defined by poseidon.h in device code.
constexpr StringLiteral kKeepalive = "__poseidon_device_math";

// What keepDeviceMath changed, for releaseDeviceMath to undo.
constexpr StringLiteral kKeptCall = "poseidon-kept-call";
constexpr StringLiteral kKeptAlwaysInline = "poseidon-kept-always-inline";

constexpr StringLiteral kUndefinedCallee = "poseidon.undefined.callee";

StringRef mathLibrary(const Triple &T) {
  return T.isAMDGPU() ? "ocml" : "libdevice";
}

std::string includeAdvice(const Triple &T) {
  return (" poseidon/poseidon.h links the " + mathLibrary(T) +
          " functions a rewrite or a site's derivative may call: include it in "
          "this translation unit (with -poseidon-kernels, pass -include "
          "poseidon/poseidon.h).")
      .str();
}

// The math intrinsics the target selects at `Ty` only as a call of its math
// library's function.
bool needsMathLibrary(const Triple &T, Intrinsic::ID ID, Type *Ty) {
  switch (ID) {
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::exp10:
    return T.isNVPTX() && Ty->isDoubleTy();
  case Intrinsic::tan:
  case Intrinsic::asin:
  case Intrinsic::acos:
  case Intrinsic::atan:
  case Intrinsic::atan2:
  case Intrinsic::sinh:
  case Intrinsic::cosh:
  case Intrinsic::tanh:
    return Ty->isDoubleTy() || (T.isAMDGPU() && Ty->isFloatTy());
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::log:
  case Intrinsic::log2:
  case Intrinsic::log10:
  case Intrinsic::pow:
    return Ty->isDoubleTy();
  default:
    return false;
  }
}

void collectGlobals(Value *V, SmallVectorImpl<WeakVH> &out) {
  if (auto *G = dyn_cast<GlobalValue>(V)) {
    out.push_back(G);
    return;
  }
  if (auto *C = dyn_cast<Constant>(V))
    for (Value *op : C->operands())
      collectGlobals(op, out);
}

} // namespace

bool keepDeviceMath(Module &M) {
  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || deviceMathName(F.getName()).empty() ||
        F.hasFnAttribute(Attribute::NoInline))
      continue;
    if (F.hasFnAttribute(Attribute::AlwaysInline)) {
      F.removeFnAttr(Attribute::AlwaysInline);
      F.addFnAttr(kKeptAlwaysInline);
    }
    F.addFnAttr(Attribute::NoInline);
    F.addFnAttr(kKeptCall);
    changed = true;
  }
  return changed;
}

bool releaseDeviceMath(Module &M) {
  bool changed = false;
  for (Function &F : M) {
    if (!F.hasFnAttribute(kKeptCall))
      continue;
    F.removeFnAttr(kKeptCall);
    F.removeFnAttr(Attribute::NoInline);
    if (F.hasFnAttribute(kKeptAlwaysInline)) {
      F.removeFnAttr(kKeptAlwaysInline);
      F.addFnAttr(Attribute::AlwaysInline);
    }
    changed = true;
  }

  Function *keepalive = M.getFunction(kKeepalive);
  if (!keepalive)
    return changed;
  removeFromUsedLists(M, [&](Constant *C) { return C == keepalive; });
  SmallVector<WeakVH, 16> maybeDead;
  maybeDead.push_back(keepalive);
  while (!maybeDead.empty()) {
    auto *G = dyn_cast_or_null<GlobalValue>(maybeDead.pop_back_val());
    if (!G)
      continue;
    G->removeDeadConstantUsers();
    if (!G->use_empty() || (!G->isDeclaration() && !G->isDiscardableIfUnused()))
      continue;
    if (auto *F = dyn_cast<Function>(G)) {
      for (Instruction &I : instructions(*F))
        for (Value *op : I.operands())
          collectGlobals(op, maybeDead);
    } else if (auto *V = dyn_cast<GlobalVariable>(G);
               V && V->hasInitializer()) {
      collectGlobals(V->getInitializer(), maybeDead);
    }
    Comdat *C =
        isa<GlobalObject>(G) ? cast<GlobalObject>(G)->getComdat() : nullptr;
    G->eraseFromParent();
    if (C && C->getUsers().empty())
      M.getComdatSymbolTable().erase(C->getName());
  }
  return true;
}

void inlineDeviceMath(Function &F) {
  // ocml's functions are not always-inline, and a build without the plugin
  // inlines them all the same.
  const bool amdgpu = Triple(F.getParent()->getTargetTriple()).isAMDGPU();
  for (bool inlined = true; inlined;) {
    inlined = false;
    SmallVector<CallBase *, 8> calls;
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *callee = CB->getCalledFunction();
            callee && !callee->isDeclaration() &&
            !deviceMathName(callee->getName()).empty() &&
            (amdgpu || callee->hasFnAttribute(Attribute::AlwaysInline)))
          calls.push_back(CB);
    for (CallBase *CB : calls) {
      InlineFunctionInfo IFI;
      inlined |= InlineFunction(*CB, IFI).isSuccess();
    }
  }
}

CallInst *createMathCall(IRBuilderBase &B, FunctionCallee F,
                         ArrayRef<Value *> Args, const Twine &Name) {
  CallInst *CI = B.CreateCall(F, Args, Name);
  if (auto *Fn = dyn_cast<Function>(F.getCallee())) {
    CI->setCallingConv(Fn->getCallingConv());
    if (Fn->isDeclaration() && !Fn->isIntrinsic() &&
        isDeviceModule(*Fn->getParent()))
      CI->setMetadata(kUndefinedCallee, MDNode::get(CI->getContext(), {}));
  }
  return CI;
}

void requireDeviceCallees(Module &M) {
  if (!isDeviceModule(M))
    return;
  const Triple T(M.getTargetTriple());
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      Function *callee = CI ? CI->getCalledFunction() : nullptr;
      if (!callee || !callee->isDeclaration())
        continue;
      std::string msg;
      if (CI->getMetadata(kUndefinedCallee)) {
        msg = (Twine("Poseidon: a rewrite in ") + demangle(F.getName()) +
               " calls " + callee->getName() +
               ", which this device module does not define." + includeAdvice(T))
                  .str();
      } else if (flags::ProfileGenerate &&
                 needsMathLibrary(T, callee->getIntrinsicID(), CI->getType())) {
        msg = (Twine("Poseidon: the derivative profile generation takes in ") +
               demangle(F.getName()) + " calls " + callee->getName() +
               ", which " + (T.isAMDGPU() ? "AMDGPU" : "NVPTX") +
               " lowers only through " + mathLibrary(T) +
               ", and this module does not define " + mathLibrary(T) +
               "'s function for it." + includeAdvice(T))
                  .str();
      }
      if (!msg.empty())
        report_fatal_error(Twine(msg), /*gen_crash_diag=*/false);
    }
}

} // namespace poseidon
