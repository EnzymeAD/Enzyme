//===- MathLibrary.cpp - the math functions the pass knows by name --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MathLibrary.h"
#include "Target.h"

#include "Support/Diagnostics.h"
#include "Support/Utils.h"

#include "llvm/ADT/STLExtras.h"
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
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <optional>

using namespace llvm;

namespace poseidon {

/// Defined by poseidon.h in device code under POSEIDON_DEVICE_MATH: the keeper
/// of the math functions, and the mark of the macro.
static constexpr StringLiteral kKeepalive = "__poseidon_device_math";
static constexpr StringLiteral kMacroMark = "__poseidon_device_math_defined";

/// What keepDeviceMath changed, for releaseDeviceMath to undo.
static constexpr StringLiteral kKeptCall = "poseidon-kept-call";
static constexpr StringLiteral kKeptAlwaysInline =
    "poseidon-kept-always-inline";

static constexpr StringLiteral kUndefinedCallee = "poseidon.undefined.callee";

static std::string includeAdvice(const Target &T) {
  return (" poseidon/poseidon.h links the " + T.mathLibrary() +
          " functions a rewrite or a site's derivative may call when "
          "POSEIDON_DEVICE_MATH is defined: include it in this translation "
          "unit (with -poseidon-kernels, pass -include poseidon/poseidon.h) "
          "and define the macro, as poseidon-clang and the Poseidon CMake "
          "targets do.")
      .str();
}

static bool hasRow(StringRef Name, bool MathFunction::*Column) {
  return any_of(mathFunctions(), [&](const MathFunction &F) {
    return F.*Column && F.Name == Name;
  });
}

StringRef mathFunctionOf(StringRef Name) {
  if (std::optional<StringRef> Device = deviceMathName(Name))
    return *Device;
  auto IsRow = [](StringRef N) {
    return any_of(mathFunctions(),
                  [&](const MathFunction &F) { return F.Name == N; });
  };
  if (IsRow(Name))
    return Name;
  if (Name.ends_with("f") && IsRow(Name.drop_back()))
    return Name.drop_back();
  return "";
}

bool isLibmFunction(StringRef Name) {
  return hasRow(Name, &MathFunction::Libm);
}

bool isOptimizableLibmCall(StringRef Name) {
  return hasRow(mathFunctionOf(Name), &MathFunction::Optimizable);
}

bool hasFloatSuffixInC(StringRef Name) {
  return hasRow(Name, &MathFunction::FloatSuffixInC);
}

bool isDeviceMathBase(StringRef Name) {
  return hasRow(Name, &MathFunction::Device);
}

std::optional<std::string> retypeLibmFunction(StringRef name, Type *newType) {
  StringRef baseName = mathFunctionOf(name);
  if (!isLibmFunction(baseName))
    return std::nullopt;
  if (newType->isHalfTy() || newType->isBFloatTy() || newType->isFloatTy())
    return (baseName + "f").str();
  if (newType->isDoubleTy())
    return baseName.str();
  if (newType->isFP128Ty() || newType->isX86_FP80Ty())
    return (baseName + "l").str();
  return std::nullopt;
}

bool deviceCallsMathLibrary(StringRef op, Type *Ty) {
  return hasRow(op, &MathFunction::DeviceCall) &&
         (Ty->isFloatTy() || Ty->isDoubleTy());
}

std::optional<StringRef> deviceMathName(StringRef fnName) {
  for (const Target *T : {&amdgpuTarget(), &nvptxTarget()})
    if (std::optional<StringRef> base = T->mathBase(fnName))
      return base;
  return std::nullopt;
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
    if (F.isDeclaration() || !deviceMathName(F.getName()) ||
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

Error requireDeviceMathMacro(Module &M) {
  const Target &T = Target::of(M);
  if (!T.isDevice() || M.getFunction(kMacroMark))
    return Error::success();
  return userError("this device compile has Poseidon sites and does not "
                   "include poseidon/poseidon.h under POSEIDON_DEVICE_MATH, "
                   "which keeps the " +
                   T.mathLibrary() +
                   " functions a rewrite or a site's derivative calls; "
                   "include the header and define POSEIDON_DEVICE_MATH");
}

bool releaseDeviceMath(Module &M) {
  bool changed = false;
  if (Function *mark = M.getFunction(kMacroMark)) {
    removeFromUsedLists(M, [&](Constant *C) { return C == mark; });
    mark->removeDeadConstantUsers();
    if (mark->use_empty()) {
      mark->eraseFromParent();
      changed = true;
    }
  }
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
  SmallVector<WeakVH, 16> maybeDead{keepalive};
  eraseDeadGlobals(maybeDead, /*declarations=*/true);
  return true;
}

void inlineDeviceMath(Function &F) {
  const Target &T = Target::of(F);
  inlineCallsWhile(F, [&](CallBase &CB) {
    Function *callee = CB.getCalledFunction();
    return callee && !callee->isDeclaration() &&
           deviceMathName(callee->getName()) && T.inlinesMathCall(*callee);
  });
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

Error requireDeviceCallees(Module &M, bool profileGenerate) {
  const Target &T = Target::of(M);
  if (!T.isDevice())
    return Error::success();
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
        return userError(*CI, msg);
    }
  return Error::success();
}

} // namespace poseidon
