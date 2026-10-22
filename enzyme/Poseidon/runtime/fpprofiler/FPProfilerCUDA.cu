#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

// The one filename rule, shared verbatim with the host runtime and with the
// pass that reads what this file writes.
#include "FPProfileName.h"
#include "poseidon/poseidon.h"

// Per-function slot range: bounds the static optimizable-instruction count of
// a single site (one slot per instruction, setPoseidonMetadata), not the
// dynamic execution count. Overflow aborts in poseidonLogValueCUDA; raise
// POSEIDON_PROFILE_MAX_SLOTS if a site ever needs more.
#define POSEIDON_DEFAULT_MAX_SLOTS 4096
#define POSEIDON_DEFAULT_MAX_FUNCS 512
#define POSEIDON_MAX_OPERANDS 8

__device__ __host__ inline uint64_t doubleToOrderedUint64(double val) {
  uint64_t bits;
  memcpy(&bits, &val, sizeof(bits));
  if (bits & 0x8000000000000000ULL)
    return ~bits;
  else
    return bits | 0x8000000000000000ULL;
}

__device__ __host__ inline double orderedUint64ToDouble(uint64_t bits) {
  if (bits & 0x8000000000000000ULL)
    bits = bits & ~0x8000000000000000ULL;
  else
    bits = ~bits;
  double val;
  memcpy(&val, &bits, sizeof(val));
  return val;
}

// Every minimum in this file is stored complemented and folded with atomicMax,
// so that 0 is the identity for every field and the whole table can be
// initialized with a plain cudaMemset instead of a kernel launch from a static
// initializer, which in a multi-TU executable can run before the other
// translation units' __cudaRegisterFatBinary constructors.
//
// 0 is a strict identity: ~ordered(v) == 0 only for a positive NaN, which every
// writer rejects first. Readback is orderedUint64ToDoubleMin; an unset field
// (stored 0, so ~0 == UINT64_MAX) decodes to the same sentinel value the
// .fpprofile readers expect, so the emitted text is unchanged.
__device__ __host__ inline uint64_t doubleToOrderedUint64Min(double val) {
  return ~doubleToOrderedUint64(val);
}

__device__ __host__ inline double orderedUint64ToDoubleMin(uint64_t stored) {
  return orderedUint64ToDouble(~stored);
}

struct GPUProfileSlot {
  uint64_t minRes;
  uint64_t maxRes;
  uint64_t minOperands[POSEIDON_MAX_OPERANDS];
  uint64_t maxOperands[POSEIDON_MAX_OPERANDS];
  uint64_t minMagOperands[POSEIDON_MAX_OPERANDS]; // smallest nonzero |operand|
  double sumValue;
  double sumGrad;
  // Sum of |gradient|. The signed sum is what the accuracy model weights with,
  // and it cancels for any value whose adjoint alternates in sign across
  // executions -- a sum-factorization contraction read by a derivative operator
  // does exactly that. The magnitude sum cannot cancel, so it is the honest
  // answer to "how much does the output move when this value moves".
  double sumAbsGrad;
  double sumSens;
  uint64_t exec;
  uint32_t numOperands;
};

__constant__ GPUProfileSlot *d_slots;
__constant__ size_t d_slotCount;

// Length of the per-function name slot the device registry copies a profiled
// function's mangled name into. A truncated name is not a cosmetic loss: the
// host would write the profile under the truncated name while the pass looks
// for it under the full one. 2048 covers every name observed with an order of
// magnitude of slack; overflow aborts in poseidonRegisterIdx rather than
// clipping.
#define POSEIDON_FUNCNAME_MAXLEN 2048

// Length of the per-function buffer the compile-time static data (the header
// lines Poseidon computed about the site and cannot recompute at profile-use)
// is published through. Overflow aborts in poseidonProfileStaticCUDA rather than
// writing a truncated header line.
#define POSEIDON_PROFILE_STATIC_MAXLEN 4096

// Multi-function support: each wrapped kernel gets its own slot block and
// output file. A probe's funcName is a distinct device-global string constant,
// so its pointer is a race-free key (atomicCAS on the 64-bit pointer). Slot
// block = base[funcIdx*perFuncCount + idx]. The capacity is a runtime quantity
// (POSEIDON_PROFILE_MAX_FUNCS): one fused framework integrator can produce
// hundreds of profiled clones in one module. Overflow aborts
// (poseidonRegisterIdx); it must never fold one function's atomics into
// another function's slot block.
__constant__ int d_maxFuncs;
__constant__ const char **d_funcReg;  // [d_maxFuncs]
__constant__ char *d_funcNames;       // [d_maxFuncs][POSEIDON_FUNCNAME_MAXLEN]

// Per-function launch geometry: gridDim is captured in the value logger,
// blockDim and the launch/CTA count in the entry probe
// poseidonProfileBlockDimsCUDA. All three are per function: distinct clones of
// one operator run under different block shapes, and a module-global max would
// corrupt the matmul M/N reconstruction and the boundary-cast frequency scale
// that read them. 0 = unknown (neutral).
__constant__ uint32_t *d_gridDimByFunc;   // [d_maxFuncs][3]
__constant__ uint32_t *d_blockDimByFunc;  // [d_maxFuncs][3]
__constant__ unsigned long long *d_launchCountByFunc; // [d_maxFuncs]

// Per-function compile-time static data, published by the site's own kernel
// through the same pointer-keyed registry the names use and copied verbatim
// into the .fpprofile header. [d_maxFuncs][POSEIDON_PROFILE_STATIC_MAXLEN]
__constant__ char *d_staticByFunc;

// One-shot gate for device-side fatal messages: every thread of a failing
// launch would otherwise print the same 200-character diagnostic.
__device__ int d_fatalReported = 0;
__device__ __attribute__((noinline)) bool poseidonFirstFatal() {
  return atomicCAS(&d_fatalReported, 0, 1) == 0;
}

// Open-addressed pointer-keyed registry for the value table. A full registry is
// a hard error: reusing slot 0 would silently mix unrelated kernels'
// statistics.
// noinline: this is the only function holding the registry's atomicCAS, and a
// device-profiled single-TU build (-include + a large inline threshold) would
// otherwise replicate that cmpxchg into the function Enzyme is
// differentiating, where reverse mode cannot handle it.
__device__ __attribute__((noinline)) int
poseidonRegisterIdx(const char **reg, char *names, int cap, const char *fn,
                    const char *which, const char *envVar) {
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
  return poseidonRegisterIdx(d_funcReg, d_funcNames, d_maxFuncs, fn, "value",
                             "POSEIDON_PROFILE_MAX_FUNCS");
}

// Condition-number probe (profile-gen only). The compiler wraps the value of
// every store a site makes through its own pointer arguments with
// poseidonProbePerturbCUDA and passes the site's id as an immediate; the driver
// re-runs the profiling workload once per (site, eps) with
// POSEIDON_PROBE_SITE=<id> and POSEIDON_PROBE_EPS=<eps> and reads the declared
// quantity of interest back from metric.txt (scripts/poseidon_probe.py). With
// no env set (eps == 0) the hook returns its input unchanged, so the baseline
// is bit-identical to an uninstrumented run.
__device__ double d_probeEps = 0.0;
__device__ int d_probeSite = -1;

// Probe entry points are inactive and never inlined: they are pure
// side-effect sinks Enzyme must not differentiate, and device-side profiling
// of a single translation unit compiles this file into that unit (-include)
// with a very large inline threshold, so without noinline the registry's
// atomicCAS is inlined into the function Enzyme is differentiating and
// reverse mode dies on `cannot handle unknown instruction: cmpxchg`.
#if defined(__has_attribute)
#if __has_attribute(enzyme_inactive)
#define POSEIDON_PROBE_ATTRS                                                   \
  __attribute__((enzyme_inactive, enzyme_nofree, noinline))
#endif
#endif
#ifndef POSEIDON_PROBE_ATTRS
#define POSEIDON_PROBE_ATTRS __attribute__((noinline))
#endif

extern "C" __device__ __attribute__((noinline)) double
poseidonProbePerturbCUDA(int site, double v) {
  double eps = d_probeEps;
  if (eps == 0.0 || site != d_probeSite)
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
    int fi = poseidonFuncIdx(funcName);
    atomicMax(&d_blockDimByFunc[(size_t)fi * 3 + 0], blockDim.x);
    atomicMax(&d_blockDimByFunc[(size_t)fi * 3 + 1], blockDim.y);
    atomicMax(&d_blockDimByFunc[(size_t)fi * 3 + 2], blockDim.z);
    atomicAdd(&d_launchCountByFunc[fi], 1ULL);
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
  char *dst = d_staticByFunc + (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN;
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
  if (idx >= d_slotCount) {
    printf("Poseidon FPProfiler: probe idx %llu >= "
           "POSEIDON_PROFILE_MAX_SLOTS=%llu. "
           "Consider increasing POSEIDON_PROFILE_MAX_SLOTS.\n",
           (unsigned long long)idx, (unsigned long long)d_slotCount);
    __trap();
  }

  int fi = poseidonFuncIdx(funcName);
  GPUProfileSlot *slot = &d_slots[(size_t)fi * d_slotCount + idx];

  // Record this function's launch grid (idempotent atomicMax; gridDim is fixed
  // per launch). Keyed by the same fi as the slots.
  atomicMax(&d_gridDimByFunc[(size_t)fi * 3 + 0], gridDim.x);
  atomicMax(&d_gridDimByFunc[(size_t)fi * 3 + 1], gridDim.y);
  atomicMax(&d_gridDimByFunc[(size_t)fi * 3 + 2], gridDim.z);

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
  if (idx >= d_slotCount) {
    printf("Poseidon FPProfiler: probe idx %llu >= "
           "POSEIDON_PROFILE_MAX_SLOTS=%llu. "
           "Consider increasing POSEIDON_PROFILE_MAX_SLOTS.\n",
           (unsigned long long)idx, (unsigned long long)d_slotCount);
    __trap();
  }

  int fi = poseidonFuncIdx(funcName);
  GPUProfileSlot *slot = &d_slots[(size_t)fi * d_slotCount + idx];

  if (!isnan(grad) && !isinf(grad) && !isnan(value) && !isinf(value)) {
    atomicAddFinite(&slot->sumGrad, grad);
    atomicAddFinite(&slot->sumAbsGrad, fabs(grad));
    atomicAddFinite(&slot->sumSens, fabs(grad * value));
  }
}

static std::string profileDir = POSEIDON_DEFAULT_PROFILE_DIR;
static GPUProfileSlot *g_devSlots = nullptr;
static size_t g_slotCount = 0;
// Host-side handles on the device-allocated registry (the device side reaches
// them through __constant__ pointers; the host needs the raw pointers to read
// them back at exit).
static int g_maxFuncs = 0;
static const char **g_devFuncReg = nullptr;
static char *g_devFuncNames = nullptr;
static uint32_t *g_devGridByFunc = nullptr;
static uint32_t *g_devBlockByFunc = nullptr;
static unsigned long long *g_devLaunchByFunc = nullptr;
static char *g_devStaticByFunc = nullptr;

// A device-side profiler abort (__trap) poisons the CUDA context but does not
// by itself fail the process: the host would return 0 with a truncated
// profile. Both atexit writers therefore check the context first and take the
// process down with a nonzero status. Every CUDA call in the registration path
// is checked: an ignored failure leaves a sticky error on the context that the
// next component to call cudaGetLastError() reports as its own, on an
// unrelated kernel launch in another translation unit.
#define POSEIDON_CUDA_OK(call)                                                 \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr,                                                          \
              "Poseidon FPProfiler: %s failed during registration (%s).\n"     \
              "  The profiler registers from a static constructor, so this is  \
the\n"                                                                         \
              "  usual symptom of the device module not being registered yet " \
              "in a\n  multi-translation-unit binary.\n",                      \
              #call, cudaGetErrorString(_e));                                  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

static void poseidonProfileExitCheck() {
  cudaError_t st = cudaDeviceSynchronize();
  if (st == cudaSuccess)
    return;
  fflush(stdout);
  fprintf(stderr,
          "Poseidon FPProfiler: the profiled run ended with a CUDA error "
          "(%s). A device-side profiler abort leaves the context in this "
          "state (see the 'Poseidon FPProfiler:' message above); the profile "
          "is incomplete and is NOT written.\n",
          cudaGetErrorString(st));
  fflush(stderr);
  _exit(1);
}

// `mkdir -p`: a missing intermediate component must not surface as an ignored
// "could not open" warning.
static void poseidonMkdirP(const std::string &dir) {
  if (dir.empty())
    return;
  std::string acc;
  size_t i = 0;
  if (dir[0] == '/') {
    acc = "/";
    i = 1;
  }
  while (i <= dir.size()) {
    size_t j = dir.find('/', i);
    if (j == std::string::npos)
      j = dir.size();
    if (j > i) {
      acc += dir.substr(i, j - i);
      struct stat st = {0};
      if (stat(acc.c_str(), &st) == -1)
        mkdir(acc.c_str(), 0755);
      acc += "/";
    }
    i = j + 1;
  }
}

// A profile that cannot be written is a profile the solve will not find;
// refuse rather than warn.
static void poseidonProfileOpenFatal(const std::string &path,
                                     const char *funcName) {
  fflush(stdout);
  fprintf(stderr,
          "Poseidon FPProfiler: could not open profile file for writing:\n"
          "  %s\n"
          "  (%s; path component is %zu bytes)\n"
          "  function: %s\n"
          "The profile is LOST, and a solve run against this directory would "
          "silently skip the site. Aborting.\n",
          path.c_str(), strerror(errno),
          path.size() - path.find_last_of('/') - 1, funcName);
  fflush(stderr);
  _exit(1);
}

static void writeAllCUDAProfiles() {
  poseidonProfileExitCheck();
  if (!g_devSlots || !g_devFuncNames)
    return;
  const size_t nameBytes = (size_t)g_maxFuncs * POSEIDON_FUNCNAME_MAXLEN;
  char *h_names = new char[nameBytes];
  if (cudaMemcpy(h_names, g_devFuncNames, nameBytes, cudaMemcpyDeviceToHost) !=
      cudaSuccess) {
    delete[] h_names;
    return;
  }

  size_t total = (size_t)g_maxFuncs * g_slotCount;
  GPUProfileSlot *h_slots = new GPUProfileSlot[total];
  if (cudaMemcpy(h_slots, g_devSlots, total * sizeof(GPUProfileSlot),
                 cudaMemcpyDeviceToHost) != cudaSuccess) {
    delete[] h_slots;
    delete[] h_names;
    return;
  }

  const size_t staticBytes =
      (size_t)g_maxFuncs * POSEIDON_PROFILE_STATIC_MAXLEN;
  char *h_static = new char[staticBytes];
  if (cudaMemcpy(h_static, g_devStaticByFunc, staticBytes,
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    memset(h_static, 0, staticBytes);

  uint32_t *h_gridByFunc = new uint32_t[(size_t)g_maxFuncs * 3];
  uint32_t *h_blockByFunc = new uint32_t[(size_t)g_maxFuncs * 3];
  unsigned long long *h_launchByFunc = new unsigned long long[g_maxFuncs];
  cudaMemcpy(h_gridByFunc, g_devGridByFunc,
             (size_t)g_maxFuncs * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_blockByFunc, g_devBlockByFunc,
             (size_t)g_maxFuncs * 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_launchByFunc, g_devLaunchByFunc,
             (size_t)g_maxFuncs * sizeof(unsigned long long),
             cudaMemcpyDeviceToHost);

  poseidonMkdirP(profileDir);

  for (int fi = 0; fi < g_maxFuncs; ++fi) {
    const char *h_funcName = h_names + (size_t)fi * POSEIDON_FUNCNAME_MAXLEN;
    if (h_funcName[0] == '\0')
      continue;
    GPUProfileSlot *base = h_slots + (size_t)fi * g_slotCount;

    std::string path = profileDir + "/" +
                       poseidon::profileNameStem(
                           h_funcName, strlen(h_funcName)) +
                       ".fpprofile";
    errno = 0;
    std::ofstream out(path);
    if (!out.is_open())
      poseidonProfileOpenFatal(path, h_funcName);
    out << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10);

    out << "MaxBlockDims = " << h_blockByFunc[fi * 3 + 0] << " "
        << h_blockByFunc[fi * 3 + 1] << " " << h_blockByFunc[fi * 3 + 2]
        << "\n";
    out << "MaxGridDims = " << h_gridByFunc[fi * 3 + 0] << " "
        << h_gridByFunc[fi * 3 + 1] << " " << h_gridByFunc[fi * 3 + 2] << "\n";
    out << "LaunchCount = " << h_launchByFunc[fi] << "\n";
    // Compile-time static data, already formatted as newline-terminated
    // header lines by the pass that embedded it.
    out << (h_static + (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN);

    for (size_t i = 0; i < g_slotCount; i++) {
      GPUProfileSlot &slot = base[i];
      if (slot.exec == 0)
        continue;
      out << i << "\n";
      out << "\tMinRes = " << orderedUint64ToDoubleMin(slot.minRes) << "\n";
      out << "\tMaxRes = " << orderedUint64ToDouble(slot.maxRes) << "\n";
      out << "\tSumValue = " << slot.sumValue << "\n";
      out << "\tSumSens = " << slot.sumSens << "\n";
      out << "\tSumGrad = " << slot.sumGrad << "\n";
      out << "\tSumAbsGrad = " << slot.sumAbsGrad << "\n";
      out << "\tExec = " << slot.exec << "\n";
      out << "\tNumOperands = " << slot.numOperands << "\n";
      for (uint32_t j = 0; j < slot.numOperands && j < POSEIDON_MAX_OPERANDS;
           j++) {
        // stored 0 == never written (see doubleToOrderedUint64Min)
        double mmag = (slot.minMagOperands[j] == 0)
                          ? 0.0
                          : orderedUint64ToDoubleMin(slot.minMagOperands[j]);
        out << "\tOperand[" << j << "] = ["
            << orderedUint64ToDoubleMin(slot.minOperands[j]) << ", "
            << orderedUint64ToDouble(slot.maxOperands[j]) << ", " << mmag
            << "]\n";
      }
    }
    out.close();
  }
  delete[] h_slots;
  delete[] h_names;
  delete[] h_static;
  delete[] h_gridByFunc;
  delete[] h_blockByFunc;
  delete[] h_launchByFunc;
}

// Reads a positive integer environment override, aborting (never silently
// falling back) when the variable is set to something unparseable.
static size_t readPositiveEnv(const char *name, size_t dflt) {
  const char *v = getenv(name);
  if (!v || !*v)
    return dflt;
  char *end = nullptr;
  unsigned long long parsed = strtoull(v, &end, 10);
  if (end == v || *end != '\0' || parsed == 0) {
    fprintf(stderr,
            "Poseidon FPProfiler: %s='%s' is not a positive integer.\n", name,
            v);
    abort();
  }
  return (size_t)parsed;
}

// The CUDA half of registration cannot run from a static constructor: it
// writes the device-side __constant__ handles with cudaMemcpyToSymbol, and in
// a multi-translation-unit binary a dynamic initializer can run before the
// device module carrying those symbols is registered ("invalid device symbol",
// reported later as a sticky context error by an unrelated component). So the
// static initializer does only environment parsing and atexit registration,
// and the CUDA work happens here, called from the top of main by the compiler
// (injectHostProfilerInit). Idempotent and safe to call by hand as well.
static bool g_cudaInitDone = false;

extern "C" void __poseidon_profile_init_cuda() {
  if (g_cudaInitDone)
    return;
  g_cudaInitDone = true;

  if (const char *envPath = getenv("POSEIDON_PROFILE_DIR"))
    profileDir = envPath;

  size_t count = readPositiveEnv("POSEIDON_PROFILE_MAX_SLOTS",
                                 POSEIDON_DEFAULT_MAX_SLOTS);
  size_t nfuncs =
      readPositiveEnv("POSEIDON_PROFILE_MAX_FUNCS", POSEIDON_DEFAULT_MAX_FUNCS);

  // Function registry: pointer keys, names, and per-function launch geometry.
  // Allocated up front (device code cannot allocate) and zeroed, which is also
  // the "unclaimed" state for every field.
  const size_t regBytes = nfuncs * sizeof(const char *);
  const size_t nameBytes = nfuncs * POSEIDON_FUNCNAME_MAXLEN;
  const size_t dimBytes = nfuncs * 3 * sizeof(uint32_t);
  const size_t lcBytes = nfuncs * sizeof(unsigned long long);
  const size_t stBytes = nfuncs * POSEIDON_PROFILE_STATIC_MAXLEN;
  cudaError_t regErr = cudaSuccess;
  auto alloc = [&](void **p, size_t b) {
    cudaError_t e = cudaMalloc(p, b);
    if (e != cudaSuccess && regErr == cudaSuccess)
      regErr = e;
  };
  alloc((void **)&g_devFuncReg, regBytes);
  alloc((void **)&g_devFuncNames, nameBytes);
  alloc((void **)&g_devGridByFunc, dimBytes);
  alloc((void **)&g_devBlockByFunc, dimBytes);
  alloc((void **)&g_devLaunchByFunc, lcBytes);
  alloc((void **)&g_devStaticByFunc, stBytes);
  if (regErr != cudaSuccess) {
    fprintf(stderr,
            "Poseidon FPProfiler: cudaMalloc for the %zu-entry function "
            "registry failed (%s). Lower POSEIDON_PROFILE_MAX_FUNCS.\n",
            nfuncs, cudaGetErrorString(regErr));
    abort();
  }
  POSEIDON_CUDA_OK(cudaMemset(g_devFuncReg, 0, regBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devFuncNames, 0, nameBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devGridByFunc, 0, dimBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devBlockByFunc, 0, dimBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devLaunchByFunc, 0, lcBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devStaticByFunc, 0, stBytes));
  g_maxFuncs = (int)nfuncs;
  int maxFuncsI = (int)nfuncs;
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_maxFuncs, &maxFuncsI, sizeof(maxFuncsI)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_funcReg, &g_devFuncReg, sizeof(g_devFuncReg)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_funcNames, &g_devFuncNames, sizeof(g_devFuncNames)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_gridDimByFunc, &g_devGridByFunc,
                   sizeof(g_devGridByFunc)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_blockDimByFunc, &g_devBlockByFunc,
                   sizeof(g_devBlockByFunc)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_launchCountByFunc, &g_devLaunchByFunc,
                   sizeof(g_devLaunchByFunc)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_staticByFunc, &g_devStaticByFunc,
                   sizeof(g_devStaticByFunc)));

  size_t bytes = nfuncs * count * sizeof(GPUProfileSlot);
  cudaError_t err = cudaMalloc(&g_devSlots, bytes);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Poseidon FPProfiler: cudaMalloc(%zu bytes for %zu funcs x %zu "
            "slots) failed (%s). Lower POSEIDON_PROFILE_MAX_SLOTS or "
            "POSEIDON_PROFILE_MAX_FUNCS.\n",
            bytes, nfuncs, count, cudaGetErrorString(err));
    g_slotCount = 0;
    abort();
  }
  g_slotCount = count;
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_slots, &g_devSlots, sizeof(g_devSlots)));
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_slotCount, &count, sizeof(count)));

  // Condition-number probe (see poseidonProbePerturbCUDA): one site id per run;
  // eps == 0 (default) disables the hook.
  {
    int probeSite = -1;
    double probeEps = 0.0;
    if (const char *envSite = getenv("POSEIDON_PROBE_SITE")) {
      char *end = nullptr;
      long id = strtol(envSite, &end, 10);
      if (end == envSite) {
        fprintf(stderr,
                "Poseidon FPProfiler: POSEIDON_PROBE_SITE='%s' is not an "
                "integer site id.\n",
                envSite);
        abort();
      }
      probeSite = (int)id;
      probeEps = 1e-6;
      if (const char *envEps = getenv("POSEIDON_PROBE_EPS")) {
        char *e2 = nullptr;
        double v = strtod(envEps, &e2);
        if (e2 == envEps || v == 0.0) {
          fprintf(stderr,
                  "Poseidon FPProfiler: POSEIDON_PROBE_EPS='%s' is not a "
                  "nonzero number.\n",
                  envEps);
          abort();
        }
        probeEps = v;
      }
      fprintf(stderr, "Poseidon FPProfiler: perturbing site %d (eps=%g)\n",
              probeSite, probeEps);
    }
    POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_probeSite, &probeSite, sizeof(probeSite)));
    POSEIDON_CUDA_OK(cudaMemcpyToSymbol(d_probeEps, &probeEps, sizeof(probeEps)));
  }

  // Zero IS the initial state of every field (see doubleToOrderedUint64Min),
  // so no init kernel is launched from this static initializer.
  POSEIDON_CUDA_OK(cudaMemset(g_devSlots, 0, bytes));
  cudaDeviceSynchronize();
  // Registered HERE, not from the static constructor: atexit handlers run in
  // reverse registration order, and the CUDA runtime registers its own teardown
  // when it is first initialized -- which is the cudaMalloc above. Registering
  // after that guarantees this dump runs BEFORE the driver shuts down.
  std::atexit(writeAllCUDAProfiles);
}

// ---------------------------------------------------------------------------
// Profile-generation launches
//
// A kernel annotated with POSEIDON_OPTIMIZE is compiled into its own reverse
// pass, which takes one shadow buffer per pointer argument. The application
// never sees them: the compiler rewrites the launch stub into a call of the
// helper below, which sizes, allocates and seeds them.
// ---------------------------------------------------------------------------

__global__ void poseidonSeedShadowF64(double *p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = 1.0;
}
__global__ void poseidonSeedShadowF32(float *p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = 1.0f;
}

// The one seed rule. A zero adjoint propagates zero through the whole reverse
// pass, so every recorded gradient would be zero and every candidate's accuracy
// cost degenerate; the seed is one on the site's outputs.
static void poseidonSeedShadow(void *p, size_t bytes, int width) {
  if (width == 8) {
    size_t n = bytes / sizeof(double);
    if (n)
      poseidonSeedShadowF64<<<(unsigned)((n + 255) / 256), 256>>>((double *)p,
                                                                  n);
  } else if (width == 4) {
    size_t n = bytes / sizeof(float);
    if (n)
      poseidonSeedShadowF32<<<(unsigned)((n + 255) / 256), 256>>>((float *)p,
                                                                  n);
  } else {
    fprintf(stderr,
            "Poseidon FPProfiler: no seed rule for a %d-byte output element.\n",
            width);
    abort();
  }
}

// The allocation a device pointer points into. Taken through the driver entry
// point rather than by linking libcuda, so a profile-generation build needs no
// extra library on its link line.
static void poseidonAllocRange(void *p, void *&base, size_t &size) {
  typedef int (*PFN_MemGetAddressRange)(unsigned long long *, size_t *,
                                        unsigned long long);
  static PFN_MemGetAddressRange getRange = nullptr;
  if (!getRange) {
    void *fn = nullptr;
    cudaError_t e =
        cudaGetDriverEntryPoint("cuMemGetAddressRange", &fn, cudaEnableDefault);
    if (e != cudaSuccess || !fn) {
      fprintf(stderr,
              "Poseidon FPProfiler: cuMemGetAddressRange is not available "
              "(%s); the size of a profiled kernel's buffers cannot be "
              "determined.\n",
              cudaGetErrorString(e));
      abort();
    }
    getRange = (PFN_MemGetAddressRange)fn;
  }
  unsigned long long b = 0;
  size_t n = 0;
  int rc = getRange(&b, &n, (unsigned long long)(uintptr_t)p);
  if (rc != 0) {
    fprintf(stderr,
            "Poseidon FPProfiler: cuMemGetAddressRange failed (%d) for the "
            "device pointer %p passed to a profiled kernel. Only allocations "
            "the CUDA allocator owns can be shadowed.\n",
            rc, p);
    abort();
  }
  base = (void *)(uintptr_t)b;
  size = n;
}

namespace {
struct PoseidonShadow {
  void *p = nullptr;
  size_t bytes = 0;
  // The sites whose outputs this shadow already carries the seed for. A
  // profiled launch runs its own reverse pass immediately, in forward order, so
  // the first site that writes a buffer consumes its adjoint and leaves zero
  // behind for the sites that write it later; each site's outputs carry the
  // seed once.
  std::set<int> seeded;
};
} // namespace

// Keyed by the base address of the device allocation, not by (site, argument):
// the adjoint of a buffer is a property of the buffer, so every site and every
// launch that passes a pointer into an allocation must see the same shadow, at
// the same offset. Keying by argument gives a buffer that one site writes and
// another reads two unrelated adjoints and breaks the chain between them. The
// shadow is created zeroed on first sight and then accumulates for the rest of
// the run: a per-launch reseed would report a different gradient for a site
// that reads its own output.
static std::map<void *, PoseidonShadow> &poseidonShadows() {
  // Never destroyed: the atexit handler that frees the buffers is registered
  // before the map is first touched, so it runs after the map's own destructor
  // would have.
  static auto *m = new std::map<void *, PoseidonShadow>();
  return *m;
}

static void poseidonFreeShadows() {
  for (auto &kv : poseidonShadows())
    if (kv.second.p)
      cudaFree(kv.second.p);
  poseidonShadows().clear();
}

extern "C" void __poseidon_launch_profiled(const void *func, int site,
                                           const unsigned *grid,
                                           const unsigned *block, size_t shmem,
                                           void *stream, int nargs, void **args,
                                           int nptr, const int *ptrIdx,
                                           const int *seedWidth) {
  static bool freeRegistered = false;
  if (!freeRegistered) {
    freeRegistered = true;
    std::atexit(poseidonFreeShadows);
  }

  std::vector<void *> full(args, args + nargs);
  std::vector<void *> shadow((size_t)nptr, nullptr);
  std::vector<void *> base((size_t)nptr, nullptr);
  std::vector<size_t> off((size_t)nptr, 0), size((size_t)nptr, 0);
  // An allocation this launch writes through any of its arguments is an output
  // of the launch, whichever argument first reaches it below.
  std::map<void *, int> width;
  for (int k = 0; k < nptr; ++k) {
    void *p = *(void **)args[ptrIdx[k]];
    if (!p)
      continue;
    poseidonAllocRange(p, base[k], size[k]);
    off[k] = (uintptr_t)p - (uintptr_t)base[k];
    int &w = width[base[k]];
    w = std::max(w, seedWidth[k]);
  }
  for (int k = 0; k < nptr; ++k) {
    if (base[k]) {
      PoseidonShadow &sb = poseidonShadows()[base[k]];
      // A base seen with a different extent is a reused address, not the same
      // buffer: the previous allocation was freed and its shadow is stale.
      if (!sb.p || sb.bytes != size[k]) {
        if (sb.p)
          POSEIDON_CUDA_OK(cudaFree(sb.p));
        POSEIDON_CUDA_OK(cudaMalloc(&sb.p, size[k]));
        sb.bytes = size[k];
        POSEIDON_CUDA_OK(cudaMemset(sb.p, 0, size[k]));
        sb.seeded.clear();
      }
      if (int w = width[base[k]])
        if (sb.seeded.insert(site).second)
          poseidonSeedShadow(sb.p, size[k], w);
      shadow[k] = (char *)sb.p + off[k];
    }
    full.push_back(&shadow[k]);
  }

  dim3 g(grid[0], grid[1], grid[2]);
  dim3 b(block[0], block[1], block[2]);
  // The reverse pass needs a shadow of the kernel's dynamic shared memory too,
  // and Enzyme places it directly after the primal block.
  cudaError_t e = cudaLaunchKernel(func, g, b, full.data(), shmem * 2,
                                   (cudaStream_t)stream);
  if (e == cudaSuccess)
    e = cudaStreamSynchronize((cudaStream_t)stream);
  if (e != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon FPProfiler: the profiled launch of site %d failed (%s). "
            "The profile is incomplete and is NOT written.\n",
            site, cudaGetErrorString(e));
    fflush(stderr);
    _exit(1);
  }
}

// Static-constructor half: nothing to do; the symbol exists so that a
// translation unit referencing it pulls this runtime in.
static int RegisterFPProfileCUDARuntime() { return 0; }

extern "C" {
int POSEIDON_PROFILE_CUDA_RUNTIME_VAR = RegisterFPProfileCUDARuntime();
}
