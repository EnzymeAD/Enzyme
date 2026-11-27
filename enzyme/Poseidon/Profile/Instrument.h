//===- Instrument.h - what profile generation adds to a module ------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Each site's body instrumented with the profiling probes, and the call that
// reaches it made a reverse-mode request over the instrumented clone with the
// shadows it needs (InstrumentSite.cpp); what else a profile-generate compile
// adds to a module; and the host half of profile generation for a CUDA or HIP
// unit, its launch stubs rewritten from the descriptors its device half
// recorded.
//
// The probes the pass places
// in device code call the device half of the profiler, which the plugin
// carries as bitcode and links into each device module it compiles for profile
// generation. Each module holds its own copy of the profiler's table, and the
// host half of the same translation unit registers it, so a program is
// profiled from any number of translation units with or without relocatable
// device code.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_PROFILE_INSTRUMENT_H
#define POSEIDON_PROFILE_INSTRUMENT_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Error.h"

#include <cstddef>

namespace llvm {
class CallInst;
class Constant;
class Function;
class Module;
} // namespace llvm

namespace poseidon {

struct Options;
struct LaunchManifest;
class Session;
struct Site;

/// The profiler runtime function `name` of type `FT`, declared in `M` if
/// absent, marked enzyme_inactive and nofree so that AD never differentiates
/// it.
llvm::FunctionCallee getProfilerHook(llvm::Module &M, llvm::StringRef name,
                                     llvm::FunctionType *FT);

/// A generic pointer to a private constant C string `contents` in the global
/// address space of `M`: the global `gvName`, made if absent, or a new
/// poseidon.str global when `gvName` is empty.
llvm::Constant *getStringPointer(llvm::Module &M, llvm::StringRef gvName,
                                 llvm::StringRef contents);

/// Place the profiling probes on a canonicalized, slot-numbered site clone.
/// Every optimizable instruction gets its value logged in the primal and its
/// result routed through an identity probe whose Enzyme custom derivative logs
/// the accumulated adjoint. That probe's augmented forward also carries the
/// condition-number perturbation, armed at run time by `siteId`, so an armed
/// run computes the whole site at the kappa run's relative precision. The
/// profile is written as `profileName`. Returns the number of instrumented
/// instructions.
size_t instrumentForProfiling(llvm::Function &clone, unsigned siteId,
                              llvm::StringRef profileName);

/// Embed the site's compile-time static data (header lines the profile-use
/// compile cannot recompute) in the instrumented module and register it with
/// the profiler runtime, which copies it verbatim into the site's .fpprofile
/// header. `text` is a sequence of newline-terminated "Key = value" lines.
void emitProfileStaticData(llvm::Function &clone, llvm::StringRef profileName,
                           llvm::StringRef text);

/// Register `text` before main with the host profiler runtime as the header of
/// the site whose profile is written under `profileName`, through globals and a
/// constructor named after `key`. False if host module M already does.
bool registerProfileStatic(llvm::Module &M, llvm::StringRef key,
                           llvm::StringRef profileName, llvm::StringRef text);

/// Host module of a CUDA or HIP translation unit, before its registration is
/// inlined: register the host shadow of the device module's profiler table
/// with the CUDA or HIP runtime and with the profiler runtime, with the counts
/// the device half recorded with recordProfileTableSize. An error when what it
/// recorded cannot be read.
llvm::Expected<bool> registerDeviceProfileTable(const Options &O,
                                                llvm::Module &M);

/// Device module, once its probes are placed: records for the host half how
/// many functions it profiles and how many probes the largest holds, which the
/// device profiler's tables are sized by.
void recordProfileTableSize(LaunchManifest &L, llvm::Module &M);

/// Device module, after the last probe is placed: link the device half of the
/// profiler. An error at a function whose probes it cannot serve.
llvm::Expected<bool> linkDeviceProfiler(llvm::Module &M, bool profileGenerate);

/// Device module of a profile-generate compile, once its derivatives are in
/// place: an error at every kernel whose static shared memory (what it and the
/// functions it calls use, the shadows Enzyme made among them) exceeds what a
/// kernel can declare, naming its buffers, their shadows and the shadows of
/// other kernels' buffers it reaches.
llvm::Error requireSharedMemoryLimit(llvm::Module &M);

/// One block-geometry probe at the entry of every function carrying value
/// probes, so launch geometry is recorded per function.
void injectBlockDimProbes(llvm::Module &M, bool print);

/// Host module: call the device profiler's initialization from the top of main;
/// whether it placed the call.
bool injectHostProfilerInit(llvm::Module &M, bool print);

/// Host module of a -poseidon-profile-generate=<dir> compile: defines the
/// directory the program's profiling runs write to, as clang's
/// -fprofile-generate=<dir> does; POSEIDON_PROFILE_DIR overrides it at run
/// time. Every unit defines it, weakly, so the linker keeps one.
bool defineProfileDirectory(const Options &O, llvm::Module &M);

/// Device module: declare the probes enzyme_inactive before anything calls
/// them, so that AD never differentiates them.
void predeclareInactiveProfilerProbes(llvm::Module &M);

/// Instruments the site and makes the call that reaches it a derivative
/// request; whether it did, or the mistake that keeps the site from being
/// profiled. Markers of one body share `clones`' instrumented clone, and the
/// clones made take their ids from `nextSiteId`.
llvm::Expected<bool>
profileSite(Session &S, const Site &site,
            llvm::DenseMap<llvm::Function *, llvm::Function *> &clones,
            unsigned &nextSiteId);
/// Registers with the host profiler, around each call that passes a stack
/// object, that object's extent, so that a site reached through it can size
/// its shadow.
void registerStackArrays(Session &S, llvm::Module &M);
/// Registers before main the extent of every global the module defines.
void registerGlobals(llvm::Module &M);

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

#endif // POSEIDON_PROFILE_INSTRUMENT_H
