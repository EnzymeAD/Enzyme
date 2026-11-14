// The device half of the CUDA profiler: the probes the pass places in a site's
// instrumented clone. The plugin carries it as NVPTX bitcode and links it into
// every device module it profiles (linkDeviceProfiler), so a program needs no
// relocatable device code to be profiled and each module holds its own copy of
// what is below.
#include <cstdint>
#include <cstdio>

#ifdef __HIP__
#include <hip/hip_runtime.h>
#define __trap __builtin_trap
// The derivative a probed site runs keeps its caches in memory it mallocs and
// frees, and HIP defines its device malloc and free only where a source calls
// them: referenced here, they are defined for every module linking this one.
__device__ void *(*poseidonMalloc)(size_t) = malloc;
__device__ void (*poseidonFree)(void *) = free;
#endif

#include "FPProfilerCUDA.h"

// Written by the host half of the same translation unit; null until then.
__constant__ PoseidonCUDAProfileTable __poseidon_profile_table;

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
                    const char *which, const char *envVar) {
  if (!reg) {
    if (poseidonFirstFatal())
      printf("Poseidon FPProfiler: the profiler table of the device module "
             "holding '%s' was never written. The host runtime binds it from "
             "__poseidon_profile_init_cuda, which a profile-generate compile "
             "calls at the top of main; link libposeidon_profile_cuda and "
             "compile main with -poseidon-profile-generate.\n",
             fn ? fn : "?");
    __trap();
  }
  if (!fn) {
    if (poseidonFirstFatal())
      printf("Poseidon FPProfiler: null %s function-name key (probe emitted "
             "without its poseidon_site_<name> global).\n",
             which);
    __trap();
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
        // A clipped name is unrecoverable, not degraded: the dump lands under
        // the prefix while the pass looks under the stem of the FULL name, so
        // the site is silently left unoptimized. Refuse instead.
        if (fn[k]) {
          if (poseidonFirstFatal())
            printf("Poseidon FPProfiler: %s function name exceeds "
                   "POSEIDON_FUNCNAME_MAXLEN (%d) and would be TRUNCATED. The "
                   "profile would be written under a name the compiler can "
                   "never look up. Raise POSEIDON_FUNCNAME_MAXLEN and rebuild. "
                   "Name starts: '%.128s'\n",
                   which, (int)POSEIDON_FUNCNAME_MAXLEN, fn);
          __trap();
        }
        __threadfence();
        return i;
      }
      if (old == key)
        return i;
    }
  }
  if (poseidonFirstFatal())
    printf("Poseidon FPProfiler: %s function registry is FULL (%d entries) "
           "while registering '%s'. Raise %s. Aborting rather than folding "
           "this function's statistics into another function's slots.\n",
           which, cap, fn, envVar);
  __trap();
  return -1;
}

__device__ inline int poseidonFuncIdx(const char *fn) {
  const PoseidonCUDAProfileTable &T = __poseidon_profile_table;
  return poseidonRegisterIdx(T.funcReg, T.funcNames, T.maxFuncs, fn, "value",
                             "POSEIDON_PROFILE_MAX_FUNCS");
}

// A device call site whose arguments reach overlapping memory, one of them
// written, that the profiling build gives shadows that are separate objects:
// the gradients the run would record miss what the call writes through one
// pointer and reads through the other.
extern "C" __device__ __attribute__((noinline)) void
poseidonOverlappingArgumentsCUDA(const char *site, int a, int b) {
  if (poseidonFirstFatal())
    printf("Poseidon FPProfiler: %s takes arguments %d and %d as pointers "
           "into overlapping memory, which the profiling build gives shadows "
           "that do not overlap; its profile would be wrong. Pass the call "
           "memory that does not overlap, or copy one of the two into a local "
           "object.\n",
           site, a, b);
  __trap();
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

// Condition-number probe (profile-gen only). The compiler wraps the value of
// every store a site makes through its own pointer arguments with
// poseidonProbePerturbCUDA and passes the site's id as an immediate; the driver
// re-runs the profiling workload once per (site, eps) with
// POSEIDON_PROBE_SITE=<id> and POSEIDON_PROBE_EPS=<eps> and reads the declared
// quantity of interest back from metric.txt. With
// no env set (eps == 0) the hook returns its input unchanged, so the baseline
// is bit-identical to an uninstrumented run.
extern "C" __device__ __attribute__((noinline)) double
poseidonProbePerturbCUDA(int site, double v) {
  double eps = __poseidon_profile_table.probeEps;
  if (eps == 0.0 || site != __poseidon_profile_table.probeSite)
    return v;
  // Value-dependent sign (bit hash): a uniform (1+eps) factor commutes through
  // linear operations and cancellations ((1+eps)x - (1+eps)y = (1+eps)(x-y)),
  // hiding exactly the cancellation amplification the probe must expose. A
  // deterministic +/-eps keyed on the value's bits models independent
  // per-operation rounding while staying reproducible across runs.
  unsigned long long b = __double_as_longlong(v);
  b ^= b >> 33;
  b *= 0xff51afd7ed558ccdULL;
  b ^= b >> 33;
  return v * (1.0 + ((b & 1ULL) ? eps : -eps));
}

extern "C" __device__ __attribute__((noinline)) float
poseidonProbePerturbCUDAf(int site, float v) {
  return (float)poseidonProbePerturbCUDA(site, (double)v);
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

// Block-geometry probe, injected at the entry of every function that carries
// value probes (injectBlockDimProbes) and keyed by the SAME funcName global the
// value probes use, so the geometry lands in the profiled function's own record
// rather than in a module-global bucket shared with every other kernel.
extern "C" __device__ POSEIDON_PROBE_ATTRS void poseidonProfileBlockDimsCUDA(const char *funcName) {
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const PoseidonCUDAProfileTable &T = __poseidon_profile_table;
    int fi = poseidonFuncIdx(funcName);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 0], blockDim.x);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 1], blockDim.y);
    atomicMax(&T.blockDimByFunc[(size_t)fi * 3 + 2], blockDim.z);
    atomicAdd(&T.launchCountByFunc[fi], 1ULL);
  }
}

// Compile-time static data probe, injected at the entry of the site clone
// alongside the geometry probe and keyed by the same funcName global. The text
// is a constant of the module, so every launch writes the same bytes and the
// first writer wins.
extern "C" __device__ POSEIDON_PROBE_ATTRS void poseidonProfileStaticCUDA(const char *funcName,
                                                 const char *text) {
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
      printf("Poseidon FPProfiler: the static profile data of '%s' exceeds "
             "POSEIDON_PROFILE_STATIC_MAXLEN (%d) and would be TRUNCATED into "
             "the profile header. Raise it and rebuild.\n",
             funcName, (int)POSEIDON_PROFILE_STATIC_MAXLEN);
    __trap();
  }
  dst[k] = '\0';
  __threadfence();
}

extern "C" __device__ POSEIDON_PROBE_ATTRS void poseidonLogValueCUDA(const char *funcName, size_t idx,
                                              double res, size_t numOperands,
                                              double *operands) {
  const PoseidonCUDAProfileTable &T = __poseidon_profile_table;
  if (T.funcReg && idx >= T.slotCount) {
    printf("Poseidon FPProfiler: probe idx %llu >= "
           "POSEIDON_PROFILE_MAX_SLOTS=%llu. "
           "Consider increasing POSEIDON_PROFILE_MAX_SLOTS.\n",
           (unsigned long long)idx, (unsigned long long)T.slotCount);
    __trap();
  }

  int fi = poseidonFuncIdx(funcName);
  GPUProfileSlot *slot = &T.slots[(size_t)fi * T.slotCount + idx];

  // Record this function's launch grid (idempotent atomicMax; gridDim is fixed
  // per launch). Keyed by the same fi as the slots.
  atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 0], gridDim.x);
  atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 1], gridDim.y);
  atomicMax(&T.gridDimByFunc[(size_t)fi * 3 + 2], gridDim.z);

  atomicMinDouble(&slot->minRes, res);
  atomicMaxDouble(&slot->maxRes, res);
  atomicAddFinite(&slot->sumValue, res);
  atomicAdd((unsigned long long *)&slot->exec, 1ULL);

  slot->numOperands = (uint32_t)numOperands;

  for (size_t i = 0; i < numOperands && i < POSEIDON_MAX_OPERANDS; i++) {
    atomicMinDouble(&slot->minOperands[i], operands[i]);
    atomicMaxDouble(&slot->maxOperands[i], operands[i]);
    if (operands[i] != 0.0)
      atomicMinDouble(&slot->minMagOperands[i], fabs(operands[i]));
  }
}

extern "C" __device__ POSEIDON_PROBE_ATTRS void poseidonLogGradCUDA(const char *funcName,
                                             size_t idx, double value,
                                             double grad) {
  const PoseidonCUDAProfileTable &T = __poseidon_profile_table;
  if (T.funcReg && idx >= T.slotCount) {
    printf("Poseidon FPProfiler: probe idx %llu >= "
           "POSEIDON_PROFILE_MAX_SLOTS=%llu. "
           "Consider increasing POSEIDON_PROFILE_MAX_SLOTS.\n",
           (unsigned long long)idx, (unsigned long long)T.slotCount);
    __trap();
  }

  int fi = poseidonFuncIdx(funcName);
  GPUProfileSlot *slot = &T.slots[(size_t)fi * T.slotCount + idx];

  if (!isnan(grad) && !isinf(grad) && !isnan(value) && !isinf(value)) {
    atomicAddFinite(&slot->sumGrad, grad);
    atomicAddFinite(&slot->sumAbsGrad, fabs(grad));
    atomicAddFinite(&slot->sumSens, fabs(grad * value));
  }
}
