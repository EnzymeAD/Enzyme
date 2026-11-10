#include "OffloadRuntime.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
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

bool launchesKernels(const Module &M) {
  return M.getFunction("__cuda_module_ctor") ||
         M.getFunction("__hip_module_ctor") ||
         llvm::any_of(M, [](const Function &F) { return isLaunchStub(F); });
}

std::optional<StringRef> compilationUnitId(const Module &M) {
  constexpr StringLiteral cuidPrefix = "__hip_cuid_";
  for (const GlobalVariable &G : M.globals())
    if (G.getName().starts_with(cuidPrefix))
      return G.getName().drop_front(cuidPrefix.size());
  return std::nullopt;
}

StringRef launchStubPrefix() { return "__device_stub__"; }

bool isLaunchStub(const Function &F) {
  return F.getName().contains(launchStubPrefix());
}

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

bool isPopCallConfiguration(StringRef name) {
  return name == "__cudaPopCallConfiguration" ||
         name == "__hipPopCallConfiguration";
}

bool isLaunchKernel(StringRef name) {
  return name == "cudaLaunchKernel" || name == "cudaLaunchKernel_ptsz" ||
         name == "hipLaunchKernel" || name == "hipLaunchKernel_spt";
}

FunctionCallee popCallConfiguration(Module &M) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
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
  Type *PtrTy = PointerType::getUnqual(C);
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

StringRef deviceProfilerInit(OffloadRuntime) {
  return "__poseidon_profile_init_cuda";
}

} // namespace poseidon
