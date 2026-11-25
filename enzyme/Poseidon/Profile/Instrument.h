//===- Instrument.h - what profile generation adds to a module ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The profiling probes on a site clone, and what a profile-generate compile
// adds to a module besides the per-site requests. The probes the pass places
// in device code call the device half of the profiler, which the plugin
// carries as bitcode and links into each device module it compiles for profile
// generation. Each module holds its own copy of the profiler's table, and the
// host half of the same translation unit registers it, so a program is
// profiled from any number of translation units with or without relocatable
// device code.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_INSTRUMENT_H
#define POSEIDON_INSTRUMENT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"

#include <cstddef>

namespace llvm {
class Constant;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

// The profiler runtime function `name` of type `FT`, declared in `M` if absent,
// marked enzyme_inactive and nofree so that AD never differentiates it.
llvm::FunctionCallee getProfilerHook(llvm::Module &M, llvm::StringRef name,
                                     llvm::FunctionType *FT);

// A generic pointer to a private constant C string `contents` in the global
// address space of `M`: the global `gvName`, made if absent, or a new
// poseidon.str global when `gvName` is empty.
llvm::Constant *getStringPointer(llvm::Module &M, llvm::StringRef gvName,
                                 llvm::StringRef contents);

// Place the profiling probes on a canonicalized, slot-numbered site clone.
// Every optimizable instruction gets its value logged in the primal and its
// result routed through an identity probe whose Enzyme custom derivative logs
// the accumulated adjoint. That probe's augmented forward also carries the
// condition-number perturbation, armed at run time by `siteId`, so an armed run
// computes the whole site at the probed relative precision. Returns the number
// of instrumented instructions.
size_t instrumentForProfiling(llvm::Function &clone, unsigned siteId);

// Embed the site's compile-time static data (header lines the profile-use
// compile cannot recompute) in the instrumented module and register it with the
// profiler runtime, which copies it verbatim into the site's .fpprofile header.
// `text` is a sequence of newline-terminated "Key = value" lines.
void emitProfileStaticData(llvm::Function &clone, llvm::StringRef text);

// Register `text` before main with the host profiler runtime as the header of
// the site whose profile is written under `name`. False if host module M
// already does.
bool registerProfileStatic(llvm::Module &M, llvm::StringRef name,
                           llvm::StringRef text);

// Host module of a CUDA or HIP translation unit, before its registration is
// inlined: register the host shadow of the device module's profiler table
// with the CUDA or HIP runtime and with the profiler runtime.
bool registerDeviceProfileTable(llvm::Module &M, bool profileGenerate);

// Device module, after the last probe is placed: link the device half of the
// profiler.
bool linkDeviceProfiler(llvm::Module &M, bool profileGenerate);

// Device module of a profile-generate compile, once its derivatives are in
// place: an error at every kernel whose static shared memory (what it and the
// functions it calls use, the shadows Enzyme made among them) exceeds what a
// kernel can declare, naming its buffers, their shadows and the shadows of
// other kernels' buffers it reaches.
void requireSharedMemoryLimit(llvm::Module &M);

// One block-geometry probe at the entry of every function carrying value
// probes, so launch geometry is recorded per function.
void injectBlockDimProbes(llvm::Module &M, bool print);

// Host module: call the device profiler's initialization from the top of main;
// whether it placed the call.
bool injectHostProfilerInit(llvm::Module &M, bool print);

// Device module: declare the probes enzyme_inactive before anything calls them,
// so that AD never differentiates them.
void predeclareInactiveProfilerProbes(llvm::Module &M);

} // namespace poseidon

#endif // POSEIDON_INSTRUMENT_H
