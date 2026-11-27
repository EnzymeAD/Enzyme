//===- DeviceProfilerDevice.cu - device half of the GPU profiler ----------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The device half of the GPU profiler: the probes the pass places in a site's
// instrumented clone. It is linked into every profiled device module, so a
// program needs no relocatable device code to be profiled and each module holds
// its own copy of what is below.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>

#include "../DeviceApi.h"

#ifdef __HIP__
// The derivative a probed site runs keeps its caches in memory it mallocs and
// frees, and HIP defines its device malloc and free only where a source calls
// them: referenced here, they are defined for every module linking this one.
__device__ void *(*poseidonMalloc)(size_t) = malloc;
__device__ void (*poseidonFree)(void *) = free;
#endif

#include "DeviceProfiler.h"

// Written by the host half of the same translation unit; null until then.
__constant__ DeviceProfileTable __poseidon_profile_table;

// One-shot gate for device-side fatal messages: every thread of a failing
// launch would otherwise print the same 200-character diagnostic.
__device__ int d_fatalReported = 0;
__device__ __attribute__((noinline)) bool poseidonFirstFatal() {
  return atomicCAS(&d_fatalReported, 0, 1) == 0;
}

// Open-addressed pointer-keyed registry for the value table. A full registry is
// a hard error: reusing slot 0 would silently mix unrelated kernels'
// statistics.
__device__ __attribute__((noinline)) int
poseidonRegisterIdx(const char **reg, char *names, int cap, const char *fn,
                    const char *which) {
  if (!reg) {
    if (poseidonFirstFatal())
      printf("[poseidon] the profiler table of the device module "
             "holding '%s' was never written. The host runtime binds it from "
             "__poseidon_profile_init_device, which a profile-generate compile "
             "calls at the top of main; link libposeidon_profile_%s and "
             "compile main with -poseidon-profile-generate.\n",
             fn ? fn : "?", POSEIDON_CUDA_OR_HIP("cuda", "hip"));
    gpuTrap();
  }
  if (!fn) {
    if (poseidonFirstFatal())
      printf("[poseidon] null %s function-name key (probe emitted "
             "without its poseidon.site.<name> global).\n",
             which);
    gpuTrap();
  }
  unsigned long long key = (unsigned long long)fn;
  int home = (int)((key >> 4) % (unsigned long long)cap);
  for (int p = 0; p < cap; ++p) {
    int i = home + p;
    if (i >= cap)
      i -= cap;
    const char *cur = *(const char *volatile *)&reg[i];
    if (cur == fn)
      return i;
    if (cur == nullptr) {
      unsigned long long old =
          atomicCAS((unsigned long long *)&reg[i], 0ull, key);
      if (old == 0ull) { // claimed: copy the name for host readback
        char *dst = names + (size_t)i * POSEIDON_FUNCNAME_MAXLEN;
        int k = 0;
        for (; k < POSEIDON_FUNCNAME_MAXLEN - 1 && fn[k]; ++k)
          dst[k] = fn[k];
        dst[k] = '\0';
        // A clipped name would put the profile where the pass never looks.
        if (fn[k]) {
          if (poseidonFirstFatal())
            printf("[poseidon] %s function name exceeds "
                   "POSEIDON_FUNCNAME_MAXLEN (%d) and would be TRUNCATED. The "
                   "profile would be written under a name the compiler can "
                   "never look up. Raise POSEIDON_FUNCNAME_MAXLEN and rebuild. "
                   "Name starts: '%.128s'\n",
                   which, (int)POSEIDON_FUNCNAME_MAXLEN, fn);
          gpuTrap();
        }
        __threadfence();
        return i;
      }
      if (old == key)
        return i;
    }
  }
  if (poseidonFirstFatal())
    printf("[poseidon] %s function registry is FULL (%d entries, "
           "the functions the registered device modules profile) while "
           "registering '%s', whose module registered fewer. Aborting rather "
           "than folding this function's statistics into another function's "
           "slots.\n",
           which, cap, fn);
  gpuTrap();
  return -1;
}

__device__ inline int poseidonFuncIdx(const char *fn) {
  const DeviceProfileTable &T = __poseidon_profile_table;
  return poseidonRegisterIdx(T.funcReg, T.funcNames, T.maxFuncs, fn, "value");
}

// The slot of probe idx of the function funcName. A probe index past the
// table is a hard error.
__device__ inline GPUProfileSlot *poseidonSlot(const char *funcName,
                                               size_t idx) {
  const DeviceProfileTable &T = __poseidon_profile_table;
  if (T.funcReg && idx >= T.slotCount) {
    if (poseidonFirstFatal())
      printf("[poseidon] probe %llu of '%s' is past the %llu "
             "probes the registered device modules declared.\n",
             (unsigned long long)idx, funcName,
             (unsigned long long)T.slotCount);
    gpuTrap();
  }
  return &T.slots[(size_t)poseidonFuncIdx(funcName) * T.slotCount + idx];
}

// A device call site whose arguments reach overlapping memory, one of them
// written, that the profiling build gives shadows that are separate objects:
// the gradients the run would record miss what the call writes through one
// pointer and reads through the other.
extern "C" __device__ __attribute__((noinline)) void
__poseidon_device_overlapping_arguments(const char *site, int a, int b) {
  if (poseidonFirstFatal())
    printf("[poseidon] %s takes arguments %d and %d as pointers "
           "into overlapping memory, which the profiling build gives shadows "
           "that do not overlap; its profile would be wrong. Pass the call "
           "memory that does not overlap, or copy one of the two into a local "
           "object.\n",
           site, a, b);
  gpuTrap();
}

// Probe entry points are inactive and never inlined: they are pure
// side-effect sinks Enzyme must not differentiate.
#if defined(__has_attribute)
#if __has_attribute(enzyme_inactive)
#define POSEIDON_PROBE_ATTRS                                                   \
  __attribute__((enzyme_inactive, enzyme_nofree, noinline))
#endif
#endif
#ifndef POSEIDON_PROBE_ATTRS
#define POSEIDON_PROBE_ATTRS __attribute__((noinline))
#endif

// Kappa-run perturbation of the value of a store a site makes through its own
// pointer arguments: perturbs v by a relative eps when site is the kappa run's
// site (POSEIDON_KAPPA_SITE, POSEIDON_KAPPA_EPS). With eps == 0 it returns v
// unchanged, so the baseline is bit-identical to an uninstrumented run.
extern "C" __device__ __attribute__((noinline)) double
__poseidon_device_perturb(int site, double v) {
  double eps = __poseidon_profile_table.kappaEps;
  if (eps == 0.0 || site != __poseidon_profile_table.kappaSite)
    return v;
  return poseidonPerturbed(v, eps);
}

extern "C" __device__ __attribute__((noinline)) float
__poseidon_device_perturbf(int site, float v) {
  return (float)__poseidon_device_perturb(site, (double)v);
}

// Complemented minimum (see doubleToOrderedUint64Min): folds with atomicMax so
// the zero-initialized state is the identity.
__device__ inline void atomicMinDouble(uint64_t *addr, double val) {
  if (isnan(val))
    return;
  atomicMax((unsigned long long *)addr,
            (unsigned long long)doubleToOrderedUint64Min(val));
}

__device__ inline void atomicMaxDouble(uint64_t *addr, double val) {
  if (isnan(val))
    return;
  atomicMax((unsigned long long *)addr,
            (unsigned long long)doubleToOrderedUint64(val));
}

__device__ inline void atomicAddFinite(double *addr, double val) {
  if (isnan(val) || isinf(val))
    return;
  atomicAdd(addr, val);
}

// Block-geometry probe at the entry of a profiled function, keyed by the
// funcName global its value probes use, so the geometry lands in that
// function's own record.
extern "C" __device__ POSEIDON_PROBE_ATTRS void
__poseidon_device_log_block_dims(const char *funcName) {
  if (__poseidon_profile_table.metricOnly)
    return;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const DeviceProfileTable &T = __poseidon_profile_table;
    int fi = poseidonFuncIdx(funcName);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 0], blockDim.x);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 1], blockDim.y);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 2], blockDim.z);
    atomicAdd(&T.launchCountByFunc[fi], 1ULL);
    atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 0], gridDim.x);
    atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 1], gridDim.y);
    atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 2], gridDim.z);
  }
}

// Compile-time static data probe at the entry of a site clone, keyed by its
// funcName global. The text is a constant of the module, so every launch writes
// the same bytes and the first writer wins.
extern "C" __device__ POSEIDON_PROBE_ATTRS void
__poseidon_device_publish_static(const char *funcName, const char *text) {
  if (__poseidon_profile_table.metricOnly)
    return;
  if (threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0)
    return;
  if (!text || !text[0])
    return;
  int fi = poseidonFuncIdx(funcName);
  char *dst = __poseidon_profile_table.staticByFunc +
              (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN;
  if (dst[0] != '\0')
    return;
  int k = 0;
  for (; k < POSEIDON_PROFILE_STATIC_MAXLEN - 1 && text[k]; ++k)
    dst[k] = text[k];
  if (text[k]) {
    if (poseidonFirstFatal())
      printf("[poseidon] the static profile data of '%s' exceeds "
             "POSEIDON_PROFILE_STATIC_MAXLEN (%d) and would be TRUNCATED into "
             "the profile header. Raise it and rebuild.\n",
             funcName, (int)POSEIDON_PROFILE_STATIC_MAXLEN);
    gpuTrap();
  }
  dst[k] = '\0';
  __threadfence();
}

extern "C" __device__ POSEIDON_PROBE_ATTRS void
__poseidon_device_log_value(const char *funcName, size_t idx, double res,
                            size_t numOperands, double *operands) {
  const DeviceProfileTable &T = __poseidon_profile_table;
  if (T.metricOnly)
    return;
  GPUProfileSlot *slot = poseidonSlot(funcName, idx);

  atomicMinDouble(&slot->minRes, res);
  atomicMaxDouble(&slot->maxRes, res);
  atomicAdd((unsigned long long *)&slot->exec, 1ULL);

  slot->numOperands = (uint32_t)numOperands;

  for (size_t i = 0; i < numOperands && i < POSEIDON_MAX_OPERANDS; i++) {
    atomicMinDouble(&slot->minOperands[i], operands[i]);
    atomicMaxDouble(&slot->maxOperands[i], operands[i]);
    if (operands[i] != 0.0)
      atomicMinDouble(&slot->minMagOperands[i], fabs(operands[i]));
  }
}

extern "C" __device__ POSEIDON_PROBE_ATTRS void
__poseidon_device_log_grad(const char *funcName, size_t idx, double value,
                           double grad) {
  const DeviceProfileTable &T = __poseidon_profile_table;
  if (T.metricOnly)
    return;
  GPUProfileSlot *slot = poseidonSlot(funcName, idx);

  if (!isnan(grad) && !isinf(grad) && !isnan(value) && !isinf(value)) {
    atomicAddFinite(&slot->sumGrad, grad);
    atomicAddFinite(&slot->sumAbsGrad, fabs(grad));
    atomicAddFinite(&slot->sumSens, fabs(grad * value));
  }
}
