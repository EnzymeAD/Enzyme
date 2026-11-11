//===- OffloadRuntime.h - the host half of a CUDA or HIP compilation -----===//
//
// What clang's offload lowering leaves in the host module of a CUDA or HIP
// translation unit and Poseidon reads or extends: kernel launches through
// stubs, device variable registration, and the names kernels are given. HIP
// names CUDA's runtime functions with hip for cuda.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_OFFLOAD_RUNTIME_H
#define POSEIDON_OFFLOAD_RUNTIME_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

#include <cstdint>
#include <optional>

namespace llvm {
class CallBase;
class CallInst;
class Function;
class GlobalVariable;
class IRBuilderBase;
class Module;
class Value;
} // namespace llvm

namespace poseidon {

enum class OffloadRuntime { None, CUDA, HIP };

// The runtime whose host half `M` is, from what clang records on every such
// module, kernels or not: the CUDA SDK version or __cudaRegisterFatBinary for
// CUDA, the compilation-unit id for HIP. The host half of a .cu and a plain
// .cpp share the triple.
OffloadRuntime offloadRuntimeOf(const llvm::Module &M);

// Whether `M` launches kernels: a module constructor registers them, or a
// stub launches one.
bool launchesKernels(const llvm::Module &M);

// Whether `M` is the host half of a CUDA or HIP compilation.
bool isOffloadHostHalf(const llvm::Module &M);

// The id clang gives both halves of a HIP compilation.
std::optional<llvm::StringRef> compilationUnitId(const llvm::Module &M);

// The launch stub of a kernel: `_Z<L+15>__device_stub__<id><params>` for the
// kernel `_Z<L><id><params>`.
llvm::StringRef launchStubPrefix();
bool isLaunchStub(const llvm::Function &F);

// `name` without the postfix clang appends under -fgpu-rdc to the device-side
// name of a kernel with internal linkage ("__intern__<hash>", in HIP
// ".intern.<hash>"). The host stub's name has none, and the hash follows the
// compile's macros, so neither the stub nor a later compile without -fgpu-rdc
// can be matched on it.
llvm::StringRef withoutExternalizedPostfix(llvm::StringRef name);

// A stub pops the configuration its <<<grid, block, shared, stream>>> site
// pushed into four out-parameters, in that order, and ends in a launch
// (cudaLaunchKernel, hipLaunchKernel or their per-thread-stream forms) whose
// first operand is the kernel's handle and, whatever the host ABI made of the
// two dim3 operands, whose last three are the argument array, the dynamic
// shared memory size and the stream.
bool isPopCallConfiguration(llvm::StringRef name);
bool isLaunchKernel(llvm::StringRef name);
// The pop of the runtime `M` declares one of, CUDA's otherwise.
llvm::FunctionCallee popCallConfiguration(llvm::Module &M);
llvm::Value *poppedGrid(llvm::CallBase &pop);
llvm::Value *poppedBlock(llvm::CallBase &pop);
llvm::Value *launchedKernel(llvm::CallBase &launch);
llvm::Value *launchArguments(llvm::CallBase &launch);
llvm::Value *launchSharedBytes(llvm::CallBase &launch);
llvm::Value *launchStream(llvm::CallBase &launch);

// How the host half registers a device variable with its runtime: without
// relocatable device code the unit registers its kernels and device variables
// itself (`RegisterGlobals`); with it, the linker wrapper registers the
// offloading entries the unit leaves against the linked image, and the runtime
// is the one of the unit's own entries. None if `M` does neither.
struct VariableRegistration {
  OffloadRuntime Runtime;
  llvm::Function *RegisterGlobals;
};
std::optional<VariableRegistration> variableRegistration(llvm::Module &M);
// What clang emits for a __constant__ variable `V` of the unit.
void registerDeviceVariable(llvm::Module &M, const VariableRegistration &R,
                            llvm::GlobalVariable *V, llvm::StringRef name,
                            uint64_t size);

// The device profiler's host library initialization a profile-generation build
// calls from the top of main.
llvm::StringRef deviceProfilerInit(OffloadRuntime R);

} // namespace poseidon

#endif // POSEIDON_OFFLOAD_RUNTIME_H
