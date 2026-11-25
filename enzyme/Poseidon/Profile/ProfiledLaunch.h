//===- ProfiledLaunch.h - the launches of profiled kernels ------*- C++ -*-===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host half of profile generation for a CUDA or HIP unit: its launch stubs
// rewritten from the descriptors its device half recorded.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_PROFILED_LAUNCH_H
#define POSEIDON_PROFILE_PROFILED_LAUNCH_H

#include "llvm/Support/Error.h"

namespace llvm {
class Module;
} // namespace llvm

namespace poseidon {

struct Options;

/// Rewrite the launch in each launch stub that a profile generation descriptor
/// in M's manifest matches into a call of __poseidon_launch_profiled, which
/// allocates, zeroes and seeds the shadow buffers the profiling kernel takes
/// and launches it. Host module only. An error at a matched stub that does
/// not launch through the CUDA or HIP launch API, or for a descriptor that does
/// not parse.
llvm::Expected<bool> rewriteProfileStubBodies(const Options &O,
                                              llvm::Module &M);

/// Registers the header of every kernel the device half profiled with the
/// host profiler, as a host site's is: the device profiler writes a kernel's
/// profile only once the kernel has run, and the host profiler writes one with
/// no records for a registered site that nothing wrote.
bool registerProfiledKernels(const Options &O, llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_PROFILE_PROFILED_LAUNCH_H
