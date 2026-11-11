#include "DeviceProfile.h"

#include "FPProfilerCUDA.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <optional>

using namespace llvm;

namespace poseidon {

bool registerDeviceProfileTable(Module &M, bool profileGenerate) {
  if (!profileGenerate || Target::of(M).isDevice() ||
      M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    return false;
  std::optional<VariableRegistration> registration = variableRegistration(M);
  if (!registration)
    return false;

  LLVMContext &C = M.getContext();
  Type *VoidTy = Type::getVoidTy(C);
  Type *PtrTy = PointerType::getUnqual(C);
  const uint64_t size = sizeof(PoseidonCUDAProfileTable);
  auto *shadowTy = ArrayType::get(Type::getInt8Ty(C), size);
  auto *shadow = new GlobalVariable(
      M, shadowTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
      Constant::getNullValue(shadowTy), POSEIDON_CUDA_PROFILE_TABLE);
  shadow->setAlignment(Align(8));
  registerDeviceVariable(M, *registration, shadow, POSEIDON_CUDA_PROFILE_TABLE,
                         size);

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
  const Target &target = Target::of(M);
  if (!profileGenerate || !target.isDevice())
    return false;
  if (GlobalVariable *T = M.getNamedGlobal(POSEIDON_CUDA_PROFILE_TABLE))
    if (!T->isDeclaration())
      return false;
  // A build without the target's device profiler links none, and has no host
  // half either, so the probes stay external and the program does not link.
  StringRef bitcode = target.profilerBitcode();
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
  // A library built for one processor serves only the probes compiled for it.
  for (const Function &F : L) {
    const Function *probe = F.isDeclaration() || F.hasLocalLinkage()
                                ? nullptr
                                : M.getFunction(F.getName());
    if (!probe || !probe->isDeclaration())
      continue;
    for (const User *U : probe->users())
      if (const auto *call = dyn_cast<CallBase>(U))
        if (std::string why = target.profilerMismatch(*call->getFunction());
            !why.empty()) {
          M.getContext().diagnose(DiagnosticInfoUnsupported(
              *call->getFunction(), "Poseidon: " + why, call->getDebugLoc(),
              DS_Error));
          return false;
        }
  }

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
