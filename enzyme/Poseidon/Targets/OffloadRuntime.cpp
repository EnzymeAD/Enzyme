//===- OffloadRuntime.cpp - the host half of a CUDA or HIP compilation ----===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OffloadRuntime.h"

#include "Enzyme/Utils.h"
#include "Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/VersionTuple.h"

using namespace llvm;

namespace poseidon {

OffloadRuntime offloadRuntimeOf(const Module &M) {
  if (!M.getSDKVersion().empty() || M.getFunction("__cudaRegisterFatBinary"))
    return OffloadRuntime::CUDA;
  if (compilationUnitId(M))
    return OffloadRuntime::HIP;
  return OffloadRuntime::None;
}

static bool isLaunchStub(const Function &F) {
  return F.getName().contains(launchStubPrefix());
}

// Whether `M` launches kernels: a module constructor registers them, or a
// stub launches one.
static bool launchesKernels(const Module &M) {
  return M.getFunction("__cuda_module_ctor") ||
         M.getFunction("__hip_module_ctor") || llvm::any_of(M, isLaunchStub);
}

bool isOffloadHostHalf(const Module &M) {
  return !Target::of(M).isDevice() &&
         (offloadRuntimeOf(M) != OffloadRuntime::None || launchesKernels(M));
}

std::optional<StringRef> compilationUnitId(const Module &M) {
  constexpr StringLiteral cuidPrefix = "__hip_cuid_";
  for (const GlobalVariable &G : M.globals())
    if (G.getName().starts_with(cuidPrefix))
      return G.getName().drop_front(cuidPrefix.size());
  return std::nullopt;
}

StringRef launchStubPrefix() { return "__device_stub__"; }

StringRef withoutExternalizedPostfix(StringRef name) {
  for (StringRef postfix : {"__intern__", ".intern."}) {
    size_t at = name.rfind(postfix);
    if (at == StringRef::npos)
      continue;
    StringRef hash = name.drop_front(at + postfix.size());
    if (hash.empty() || !llvm::all_of(hash, llvm::isHexDigit))
      continue;
    return name.take_front(at);
  }
  return name;
}

// The part of a kernel's mangled name shared with its launch stub: kernel
// `_Z<L><id><params>` and stub `_Z<L+15>__device_stub__<id><params>`.
static StringRef mangledSuffix(StringRef mangled) {
  if (mangled.starts_with("_Z")) {
    mangled = mangled.drop_front(2);
    // An internal-linkage name (a static kernel template).
    mangled.consume_front("L");
    while (!mangled.empty() && mangled[0] >= '0' && mangled[0] <= '9')
      mangled = mangled.drop_front(1);
  }
  return mangled;
}

bool isLaunchStubFor(const Function &F, StringRef kernelName) {
  StringRef n = F.getName();
  if (n == kernelName)
    return true;
  const StringRef stubPrefix = launchStubPrefix();
  if (!n.contains(stubPrefix))
    return false;
  StringRef kernel = withoutExternalizedPostfix(kernelName);
  if (n.ends_with(mangledSuffix(kernel)))
    return true;
  // A kernel in a namespace or a class: its identifier, and with it the
  // stub's prefix, sits inside the nested name rather than at its front.
  std::string stub = demangle(n);
  size_t at = stub.find(stubPrefix);
  if (at == std::string::npos || stub == n)
    return false;
  stub.erase(at, stubPrefix.size());
  return stub == demangle(kernel);
}

bool isPopCallConfiguration(StringRef name) {
  return name == "__cudaPopCallConfiguration" ||
         name == "__hipPopCallConfiguration";
}

bool isLaunchKernel(StringRef name) {
  return name == "cudaLaunchKernel" || name == "cudaLaunchKernel_ptsz" ||
         name == "hipLaunchKernel" || name == "hipLaunchKernel_spt";
}

CallInst *findPopConfig(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *cal = CI->getCalledFunction())
        if (isPopCallConfiguration(cal->getName()) && CI->arg_size() >= 2)
          return CI;
  return nullptr;
}

CallInst *findLaunchCall(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *cal = CI->getCalledFunction())
        if (isLaunchKernel(cal->getName()) && CI->arg_size() >= 4)
          return CI;
  return nullptr;
}

FunctionCallee popCallConfiguration(Module &M) {
  LLVMContext &C = M.getContext();
  Type *ptr = getInt8PtrTy(C);
  FunctionType *FT =
      FunctionType::get(Type::getInt32Ty(C), {ptr, ptr, ptr, ptr}, false);
  return M.getOrInsertFunction(M.getFunction("__hipPopCallConfiguration")
                                   ? "__hipPopCallConfiguration"
                                   : "__cudaPopCallConfiguration",
                               FT);
}

Value *poppedGrid(CallBase &pop) { return pop.getArgOperand(0); }
Value *poppedBlock(CallBase &pop) { return pop.getArgOperand(1); }

Value *launchedKernel(CallBase &launch) { return launch.getArgOperand(0); }
Value *launchArguments(CallBase &launch) {
  return launch.getArgOperand(launch.arg_size() - 3);
}
Value *launchSharedBytes(CallBase &launch) {
  return launch.getArgOperand(launch.arg_size() - 2);
}
Value *launchStream(CallBase &launch) {
  return launch.getArgOperand(launch.arg_size() - 1);
}
Value *launchStreamHandle(IRBuilderBase &B, CallBase &launch) {
  Value *stream = launchStream(launch);
  const Function *F = launch.getCalledFunction();
  if (!F || (F->getName() != "cudaLaunchKernel_ptsz" &&
             F->getName() != "hipLaunchKernel_spt"))
    return stream;
  auto *ptr = cast<PointerType>(stream->getType());
  Value *perThread = ConstantExpr::getIntToPtr(B.getInt64(2), ptr);
  return B.CreateSelect(B.CreateIsNull(stream), perThread, stream);
}

std::optional<VariableRegistration> variableRegistration(Module &M) {
  const bool hip = M.getFunction("__hip_register_globals");
  Function *registerGlobals =
      M.getFunction(hip ? "__hip_register_globals" : "__cuda_register_globals");
  if (registerGlobals)
    return VariableRegistration{
        hip ? OffloadRuntime::HIP : OffloadRuntime::CUDA, registerGlobals};
#if LLVM_VERSION_MAJOR >= 23
  StringRef entries = offloading::getOffloadEntrySection(M);
#else
  StringRef entries = "llvm_offload_entries";
#endif
  for (const GlobalVariable &G : M.globals())
    if (G.getSection() == entries && G.hasInitializer())
      if (auto *K = dyn_cast_or_null<ConstantInt>(
              G.getInitializer()->getAggregateElement(2u)))
        return VariableRegistration{K->getZExtValue() ==
                                            object::OffloadKind::OFK_HIP
                                        ? OffloadRuntime::HIP
                                        : OffloadRuntime::CUDA,
                                    nullptr};
  return std::nullopt;
}

void registerDeviceVariable(Module &M, const VariableRegistration &R,
                            GlobalVariable *V, StringRef name, uint64_t size) {
  if (!R.RegisterGlobals) {
    offloading::emitOffloadingEntry(
        M,
        R.Runtime == OffloadRuntime::HIP ? object::OffloadKind::OFK_HIP
                                         : object::OffloadKind::OFK_Cuda,
        V, name, size,
        offloading::OffloadGlobalEntry | offloading::OffloadGlobalConstant,
        /*Data=*/0);
    return;
  }
  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy = getInt8PtrTy(C);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
  FunctionCallee registerVar = M.getOrInsertFunction(
      R.Runtime == OffloadRuntime::HIP ? "__hipRegisterVar"
                                       : "__cudaRegisterVar",
      FunctionType::get(
          VoidTy, {PtrTy, PtrTy, PtrTy, PtrTy, I32Ty, I64Ty, I32Ty, I32Ty},
          false));
  for (BasicBlock &BB : *R.RegisterGlobals) {
    auto *ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!ret)
      continue;
    IRBuilder<> B(ret);
    Constant *nameStr = B.CreateGlobalString(name);
    B.CreateCall(registerVar,
                 {R.RegisterGlobals->getArg(0), V, nameStr, nameStr,
                  B.getInt32(0), ConstantInt::get(I64Ty, size), B.getInt32(1),
                  B.getInt32(0)});
  }
}

StringRef deviceProfilerInit() { return "__poseidon_profile_init_cuda"; }

} // namespace poseidon
