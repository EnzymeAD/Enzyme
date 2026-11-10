#include "DeviceProfile.h"

#include "CostModel.h"
#include "FPProfilerCUDA.h"
#include "Utils.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <optional>

using namespace llvm;

namespace poseidon {

namespace {
#ifdef POSEIDON_DEVICE_PROFILER
// runtime/fpprofiler/FPProfilerDevice.cu, compiled by the build to NVPTX
// bitcode.
#include "FPProfilerDevice.inc"
#endif
#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
// The same, compiled as HIP to AMDGPU bitcode.
#include "FPProfilerDeviceAMDGPU.inc"
#endif
} // namespace

bool registerDeviceProfileTable(Module &M, bool profileGenerate) {
  if (!profileGenerate || isDeviceModule(M) ||
      M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    return false;
  // Without relocatable device code the unit registers its kernels and device
  // variables itself; with it, the linker wrapper registers the offloading
  // entries the unit leaves against the linked image. HIP names CUDA's
  // registration functions with hip for cuda.
  const bool hip = M.getFunction("__hip_register_globals");
  Function *registerGlobals =
      M.getFunction(hip ? "__hip_register_globals" : "__cuda_register_globals");
#if LLVM_VERSION_MAJOR >= 23
  StringRef entries = offloading::getOffloadEntrySection(M);
#else
  StringRef entries = "llvm_offload_entries";
#endif
  // The offload kind of the unit's own entries, CUDA's or HIP's.
  std::optional<uint64_t> kind;
  for (const GlobalVariable &G : M.globals())
    if (G.getSection() == entries && G.hasInitializer())
      if (auto *K = dyn_cast_or_null<ConstantInt>(
              G.getInitializer()->getAggregateElement(2u))) {
        kind = K->getZExtValue();
        break;
      }
  bool relocatable = !registerGlobals && kind;
  if (!registerGlobals && !relocatable)
    return false;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy = PointerType::getUnqual(C);
  Type *I32Ty = Type::getInt32Ty(C);
  Type *I64Ty = Type::getInt64Ty(C);
  const uint64_t size = sizeof(PoseidonCUDAProfileTable);
  auto *shadowTy = ArrayType::get(Type::getInt8Ty(C), size);
  auto *shadow = new GlobalVariable(
      M, shadowTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      Constant::getNullValue(shadowTy), POSEIDON_CUDA_PROFILE_TABLE);
  shadow->setAlignment(Align(8));

  if (relocatable) {
    offloading::emitOffloadingEntry(
        M,
        *kind == object::OffloadKind::OFK_HIP ? object::OffloadKind::OFK_HIP
                                              : object::OffloadKind::OFK_Cuda,
        shadow, POSEIDON_CUDA_PROFILE_TABLE, size,
        offloading::OffloadGlobalEntry | offloading::OffloadGlobalConstant,
        /*Data=*/0);
  } else {
    // What clang emits for a __constant__ variable of the unit.
    FunctionCallee registerVar = M.getOrInsertFunction(
        hip ? "__hipRegisterVar" : "__cudaRegisterVar",
        FunctionType::get(
            VoidTy, {PtrTy, PtrTy, PtrTy, PtrTy, I32Ty, I64Ty, I32Ty, I32Ty},
            false));
    for (BasicBlock &BB : *registerGlobals) {
      auto *ret = dyn_cast<ReturnInst>(BB.getTerminator());
      if (!ret)
        continue;
      IRBuilder<> B(ret);
      Constant *name = B.CreateGlobalString(POSEIDON_CUDA_PROFILE_TABLE);
      B.CreateCall(registerVar, {registerGlobals->getArg(0), shadow, name, name,
                                 B.getInt32(0), ConstantInt::get(I64Ty, size),
                                 B.getInt32(1), B.getInt32(0)});
    }
  }

  Function *ctor = Function::Create(
      FunctionType::get(VoidTy, false), GlobalValue::InternalLinkage,
      "__poseidon_profile_register_module_table", &M);
  ctor->addFnAttr(Attribute::NoUnwind);
  IRBuilder<> B(BasicBlock::Create(C, "entry", ctor));
  B.CreateCall(M.getOrInsertFunction("__poseidon_profile_register_module",
                                     FunctionType::get(VoidTy, {PtrTy}, false)),
               {shadow});
  B.CreateRetVoid();
  // After the CUDA module constructor, so that a unit loaded once the profiler
  // runs is registered with the CUDA runtime before it is bound.
  appendToGlobalCtors(M, ctor, 65535);
  return true;
}

bool linkDeviceProfiler(Module &M, bool profileGenerate) {
  if (!profileGenerate || !isDeviceModule(M))
    return false;
  if (GlobalVariable *T = M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    if (!T->isDeclaration())
      return false;
  // Built without CUDA (ROCm): no device profiler to link, and no host half
  // either, so the probes stay external and the program does not link.
  StringRef bitcode;
#ifdef POSEIDON_DEVICE_PROFILER
  if (Triple(M.getTargetTriple()).isNVPTX())
    bitcode = StringRef(reinterpret_cast<const char *>(kDeviceProfiler),
                        sizeof(kDeviceProfiler));
#endif
#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
  if (Triple(M.getTargetTriple()).isAMDGPU())
    bitcode = StringRef(reinterpret_cast<const char *>(kDeviceProfilerAMDGPU),
                        sizeof(kDeviceProfilerAMDGPU));
#endif
  if (bitcode.empty())
    return false;
  Expected<std::unique_ptr<Module>> parsed = parseBitcodeFile(
      MemoryBufferRef(bitcode, "FPProfilerDevice.bc"), M.getContext());
  if (!parsed)
    report_fatal_error(Twine("Poseidon: the device profiler bitcode does not "
                             "load: ") +
                       toString(parsed.takeError()));
  Module &L = **parsed;
  GlobalVariable *libTable = L.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE);
  if (!libTable ||
      L.getDataLayout().getTypeAllocSize(libTable->getValueType()) !=
          sizeof(PoseidonCUDAProfileTable))
    report_fatal_error("Poseidon: the device profiler bitcode does not define "
                       "the table the host runtime writes");
#ifdef POSEIDON_DEVICE_PROFILER_AMDGPU
  // The AMDGPU library carries the ROCm device libraries of one processor, so
  // a probe compiled for another cannot call into it.
  if (Triple(M.getTargetTriple()).isAMDGPU())
    for (const Function &F : L) {
      const Function *probe = F.isDeclaration() || F.hasLocalLinkage()
                                  ? nullptr
                                  : M.getFunction(F.getName());
      if (!probe || !probe->isDeclaration())
        continue;
      for (const User *U : probe->users())
        if (const auto *call = dyn_cast<CallBase>(U))
          if (targetCpuOf(*call->getFunction()) !=
              POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH)
            report_fatal_error(
                Twine("Poseidon: the AMDGPU device profiler was built for '") +
                POSEIDON_DEVICE_PROFILER_AMDGPU_ARCH
                "' (POSEIDON_AMDGPU_ARCH) but " +
                call->getFunction()->getName() + " compiles for target-cpu '" +
                targetCpuOf(*call->getFunction()) + "'");
    }
#endif

  L.setTargetTriple(M.getTargetTriple());
  L.setDataLayout(M.getDataLayout());
  // The library's module flags and keep-alive lists describe its own compile;
  // the module keeps its own, and takes only what its probes reach.
  if (NamedMDNode *moduleFlags = L.getModuleFlagsMetadata())
    L.eraseNamedMetadata(moduleFlags);
  for (StringRef used : {"llvm.used", "llvm.compiler.used"})
    if (GlobalVariable *G = L.getNamedGlobal(used))
      G->eraseFromParent();
  // Code generation follows the GPU the module is compiled for.
  for (Function &F : L) {
    F.removeFnAttr("target-cpu");
    F.removeFnAttr("target-features");
  }

  StringSet<> own;
  for (const GlobalValue &G : M.global_values())
    if (!G.isDeclaration())
      own.insert(G.getName());
  SmallVector<std::string, 16> linked;
  for (const GlobalValue &G : L.global_values())
    if (!G.isDeclaration() && !G.hasLocalLinkage() &&
        !own.contains(G.getName()))
      linked.push_back(G.getName().str());

  // The host half registers the table whether or not a probe reads it.
  GlobalVariable *table = M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE);
  if (!table)
    table = new GlobalVariable(
        M, libTable->getValueType(), /*isConstant=*/false,
        GlobalValue::ExternalLinkage, nullptr, POSEIDON_CUDA_PROFILE_TABLE,
        nullptr, GlobalValue::NotThreadLocal, libTable->getAddressSpace());
  appendToCompilerUsed(M, {table});

  if (Linker::linkModules(M, std::move(*parsed), Linker::Flags::LinkOnlyNeeded))
    report_fatal_error("Poseidon: linking the device profiler failed");

  // Local to the module, so relocatable device code links one copy per unit;
  // the table is merged across units instead, because the linker wrapper
  // registers it by name in the linked image.
  for (const std::string &name : linked)
    if (GlobalValue *G = M.getNamedValue(name); G && !G->isDeclaration())
      G->setLinkage(name == POSEIDON_CUDA_PROFILE_TABLE
                        ? GlobalValue::WeakODRLinkage
                        : GlobalValue::InternalLinkage);
  return true;
}

} // namespace poseidon
