//===- DeviceProfiler.h - state the CUDA profiler's halves share ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The state the two halves of the CUDA profiler share: the device half
// (DeviceProfilerDevice.cu) updates it, the host half (DeviceProfilerHost.cu)
// allocates it and writes the profile from it.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_PROFILE_DEVICE_PROFILER_H
#define POSEIDON_RUNTIME_PROFILE_DEVICE_PROFILER_H

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifndef POSEIDON_HD
#if defined(__CUDACC__) || defined(__HIP__)
#define POSEIDON_HD __host__ __device__
#else
#define POSEIDON_HD
#endif
#endif

// Device heap for the reverse passes' caches; POSEIDON_PROFILE_HEAP_MB.
#define POSEIDON_DEFAULT_HEAP_MB 4096
#define POSEIDON_MAX_OPERANDS 8

// Length of the per-function name slot the device registry copies a profiled
// function's mangled name into. A truncated name would put the profile under a
// name the pass does not look for, so overflow aborts in poseidonRegisterIdx.
#define POSEIDON_FUNCNAME_MAXLEN 2048

// Length of the per-function buffer the compile-time static data (header
// lines the pass computes about the site and cannot recompute at profile use)
// is published through. Overflow aborts in __poseidon_device_publish_static.
#define POSEIDON_PROFILE_STATIC_MAXLEN 4096

// The device variable every profiled device module defines and the host half
// of the same translation unit registers.
#define POSEIDON_DEVICE_PROFILE_TABLE "__poseidon_profile_table"

POSEIDON_HD inline uint64_t doubleToOrderedUint64(double val) {
  uint64_t bits;
  memcpy(&bits, &val, sizeof(bits));
  if (bits & 0x8000000000000000ULL)
    return ~bits;
  else
    return bits | 0x8000000000000000ULL;
}

POSEIDON_HD inline double orderedUint64ToDouble(uint64_t bits) {
  if (bits & 0x8000000000000000ULL)
    bits = bits & ~0x8000000000000000ULL;
  else
    bits = ~bits;
  double val;
  memcpy(&val, &bits, sizeof(val));
  return val;
}

// Every minimum in the profiler is stored complemented and folded with
// atomicMax, so that 0 is the identity for every field and the whole table can
// be initialized with a plain cudaMemset instead of a kernel launch from a
// static initializer, which in a multi-TU executable can run before the other
// translation units' __cudaRegisterFatBinary constructors. It also makes two
// records of one function merge field by field with max and +.
//
// 0 is a strict identity: ~ordered(v) == 0 only for a positive NaN, which every
// writer rejects first. An unset field (stored 0) decodes to the sentinel the
// .fpprofile readers expect.
POSEIDON_HD inline uint64_t doubleToOrderedUint64Min(double val) {
  return ~doubleToOrderedUint64(val);
}

POSEIDON_HD inline double orderedUint64ToDoubleMin(uint64_t stored) {
  return orderedUint64ToDouble(~stored);
}

// `v` computed at the relative precision `eps`: a kappa run's
// perturbation of the measured site's values, on the host and the device alike.
// The sign is a hash of the value's bits: a uniform (1+eps) factor commutes
// through linear operations and cancellations ((1+eps)x - (1+eps)y =
// (1+eps)(x-y)), hiding exactly the cancellation amplification the kappa run
// must expose, while a deterministic +/-eps keyed on the value's bits models
// independent per-operation rounding and stays reproducible across runs.
POSEIDON_HD inline double poseidonPerturbed(double v, double eps) {
  uint64_t b;
  memcpy(&b, &v, sizeof(b));
  b ^= b >> 33;
  b *= 0xff51afd7ed558ccdULL;
  b ^= b >> 33;
  return v * (1.0 + ((b & 1ULL) ? eps : -eps));
}

struct GPUProfileSlot {
  uint64_t minRes;
  uint64_t maxRes;
  uint64_t minOperands[POSEIDON_MAX_OPERANDS];
  uint64_t maxOperands[POSEIDON_MAX_OPERANDS];
  uint64_t minMagOperands[POSEIDON_MAX_OPERANDS]; // smallest nonzero |operand|
  double sumGrad;
  // Sum of |gradient|: the signed sum cancels for a value whose adjoint
  // alternates in sign across executions, the magnitude sum cannot.
  double sumAbsGrad;
  double sumSens;
  uint64_t exec;
  uint32_t numOperands;
};

// The device allocations every profiled device module records into. Without
// relocatable device code each module has its own copy of the device globals,
// so the handles are one constant per module, written by the host half into
// every module that registered one.
struct DeviceProfileTable {
  GPUProfileSlot *slots;
  size_t slotCount;
  // Each profiled function gets its own slot block and output file. A probe's
  // funcName is a distinct device-global string constant, so its pointer is a
  // race-free key (atomicCAS on the 64-bit pointer). Slot block =
  // slots[funcIdx*slotCount + idx]. The capacity is the number of functions
  // the registered modules profile; overflow aborts in poseidonRegisterIdx
  // rather than fold one function's atomics into another's slot block.
  int maxFuncs;
  const char **funcReg; // [maxFuncs]
  char *funcNames;      // [maxFuncs][POSEIDON_FUNCNAME_MAXLEN]
  // Per-function launch geometry and block count, recorded by
  // __poseidon_device_log_block_dims once per block. They are per function
  // because clones of one function run under different block shapes. 0 =
  // unknown.
  uint32_t *gridDimByFunc;               // [maxFuncs][3]
  uint32_t *blockDimByFunc;              // [maxFuncs][3]
  unsigned long long *launchCountByFunc; // [maxFuncs]
  // Per-function compile-time static data, published by the site's own kernel
  // through the same pointer-keyed registry the names use and copied verbatim
  // into the .fpprofile header. [maxFuncs][POSEIDON_PROFILE_STATIC_MAXLEN]
  char *staticByFunc;
  // Kappa run: one site id per run; eps == 0 disables it.
  double kappaEps;
  int kappaSite;
  // Nonzero in a run of the condition-number measurement, which reads only
  // the metric: the probes record nothing and the tables above are null. Zero
  // in an unbound table, so a probe there still reports it.
  int metricOnly;
};

#endif // POSEIDON_RUNTIME_PROFILE_DEVICE_PROFILER_H
