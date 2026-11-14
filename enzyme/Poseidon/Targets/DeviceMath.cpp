#include "DeviceMath.h"
#include "MathFunctions.h"
#include "Peers.h"

#include "IR/Utils.h"

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
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <optional>
#include <unordered_set>

using namespace llvm;

namespace poseidon {

namespace {

// Defined by poseidon.h in device code.
constexpr StringLiteral kKeepalive = "__poseidon_device_math";

// What keepDeviceMath changed, for releaseDeviceMath to undo.
constexpr StringLiteral kKeptCall = "poseidon-kept-call";
constexpr StringLiteral kKeptAlwaysInline = "poseidon-kept-always-inline";

constexpr StringLiteral kUndefinedCallee = "poseidon.undefined.callee";

std::string includeAdvice(const Target &T) {
  return (" poseidon/poseidon.h links the " + T.mathLibrary() +
          " functions a rewrite or a site's derivative may call: include it in "
          "this translation unit (with -poseidon-kernels, pass -include "
          "poseidon/poseidon.h).")
      .str();
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

const std::unordered_set<std::string> &mathBases() {
  static const std::unordered_set<std::string> Bases = [] {
    std::unordered_set<std::string> bases;
    for (const MathFunction &F : mathFunctions())
      if (F.Device)
        bases.insert(F.Name.str());
    return bases;
  }();
  return Bases;
}

std::string retypeLibmFunction(StringRef name, Type *newType) {
  std::string baseName = name.str();
  if (!baseName.empty() && (baseName.back() == 'f' || baseName.back() == 'l'))
    baseName.pop_back();
  if (libmFuncs().count(baseName)) {
    if (newType->isHalfTy() || newType->isBFloatTy() || newType->isFloatTy())
      return baseName + "f";
    if (newType->isDoubleTy())
      return baseName;
    if (newType->isFP128Ty() || newType->isX86_FP80Ty())
      return baseName + "l";
  }
  return "";
}

bool deviceCallsMathLibrary(StringRef op, Type *Ty) {
  static const std::unordered_set<std::string> ops = {
      "sin",  "cos",  "tan",   "asin",  "acos",  "atan", "atan2", "sinh",
      "cosh", "tanh", "asinh", "acosh", "atanh", "exp",  "exp2",  "expm1",
      "log",  "log2", "log10", "log1p", "pow",   "cbrt"};
  return ops.count(op.str()) && (Ty->isFloatTy() || Ty->isDoubleTy());
}

StringRef deviceMathName(StringRef fnName) {
  for (const Target *T : {&amdgpuTarget(), &nvptxTarget()})
    if (std::optional<StringRef> base = T->mathBase(fnName))
      return *base;
  return "";
}

std::string getLibmFunctionForPrecision(StringRef funcName, Type *newType) {
  for (const Target *T : {&amdgpuTarget(), &nvptxTarget(), &hostTarget()})
    if (std::optional<std::string> name =
            T->retypeMathFunction(funcName, newType))
      return *name;
  llvm_unreachable("the host retypes every name");
}

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
  const Target &T = Target::of(F);
  for (bool inlined = true; inlined;) {
    inlined = false;
    SmallVector<CallBase *, 8> calls;
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        if (Function *callee = CB->getCalledFunction();
            callee && !callee->isDeclaration() &&
            !deviceMathName(callee->getName()).empty() &&
            T.inlinesMathCall(*callee))
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
        Target::of(*Fn->getParent()).isDevice())
      CI->setMetadata(kUndefinedCallee, MDNode::get(CI->getContext(), {}));
  }
  return CI;
}

void requireDeviceCallees(Module &M, bool profileGenerate) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return;
  for (Function &F : M)
    for (Instruction &I : instructions(F)) {
      auto *CI = dyn_cast<CallInst>(&I);
      Function *callee = CI ? CI->getCalledFunction() : nullptr;
      if (!callee || !callee->isDeclaration())
        continue;
      std::string msg;
      if (CI->getMetadata(kUndefinedCallee)) {
        msg = (Twine("a rewrite in ") + demangle(F.getName()) + " calls " +
               callee->getName() +
               ", which this device module does not define." + includeAdvice(T))
                  .str();
      } else if (profileGenerate &&
                 T.needsMathLibrary(callee->getIntrinsicID(), CI->getType())) {
        msg = (Twine("the derivative profile generation takes in ") +
               demangle(F.getName()) + " calls " + callee->getName() +
               ", which " + T.name() + " lowers only through " +
               T.mathLibrary() + ", and this module does not define " +
               T.mathLibrary() + "'s function for it." + includeAdvice(T))
                  .str();
      }
      if (!msg.empty())
        reportUserError(*CI, msg);
    }
}

} // namespace poseidon
