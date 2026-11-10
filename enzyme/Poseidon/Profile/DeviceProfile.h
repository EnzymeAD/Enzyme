//===- DeviceProfile.h - profiling device code of any CUDA or HIP build --===//
//
// The probes the pass places in device code call the device half of the CUDA
// profiler, which the plugin carries as bitcode and links into each device
// module it compiles for profile generation. Each module holds its own copy of
// the profiler's table, and the host half of the same translation unit
// registers it, so a program is profiled from any number of translation units
// with or without relocatable device code.
//===---------------------------------------------------------------------===//
#ifndef POSEIDON_DEVICE_PROFILE_H
#define POSEIDON_DEVICE_PROFILE_H

namespace llvm {
class Module;
} // namespace llvm

namespace poseidon {

// Profile generation, host module of a CUDA or HIP translation unit, before
// its registration is inlined: register the host shadow of the device module's
// profiler table with the CUDA or HIP runtime and with the profiler runtime.
bool registerDeviceProfileTable(llvm::Module &M);

// Profile generation, device module, after the last probe is placed: link the
// device half of the profiler.
bool linkDeviceProfiler(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_DEVICE_PROFILE_H
