//===- FPProfilerCUDA.cu - host half of the CUDA profiler -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host half of the CUDA profiler, a library a profile-generate build links
// (libposeidon_profile_cuda, and compiled as HIP libposeidon_profile_hip): it
// allocates the tables the probes of every profiled device module record into
// (FPProfilerDevice.cu), writes each module's handles to them, and writes the
// profile at exit.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __HIP__
#include "../hip_as_cuda.h"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "FPProfileName.h"
#include "FPProfilerCUDA.h"
#include "ShadowTable.h"
#include "poseidon/poseidon.h"

using namespace poseidon;

static std::string profileDir;
static GPUProfileSlot *g_devSlots = nullptr;
static size_t g_slotCount = 0;
// Host-side handles on the device-allocated registry, read back at exit.
static int g_maxFuncs = 0;
static const char **g_devFuncReg = nullptr;
static char *g_devFuncNames = nullptr;
static uint32_t *g_devGridByFunc = nullptr;
static uint32_t *g_devBlockByFunc = nullptr;
static unsigned long long *g_devLaunchByFunc = nullptr;
static char *g_devStaticByFunc = nullptr;

// What every profiled device module's table holds, and the host shadows of
// those tables: one per translation unit, registered by that unit's module
// constructor, so each names the copy in its own device module.
static PoseidonCUDAProfileTable g_table;
static std::vector<const void *> &poseidonModuleTables() {
  static auto *v = new std::vector<const void *>();
  return *v;
}

// Every GPU runtime call is checked: an ignored failure leaves a sticky error
// on the context that the next caller of cudaGetLastError() reports as its
// own, or a profile that is wrong.
#define POSEIDON_CUDA_OK(call)                                                 \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fflush(stdout);                                                          \
      fprintf(stderr, "Poseidon FPProfiler: %s failed (%s).\n", #call,         \
              cudaGetErrorString(_e));                                         \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// A device-side profiler abort (__trap) poisons the context without failing
// the process, so the exit writer checks the context first and exits nonzero.
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

// Tells the host profiler which kernels this writer wrote a profile for. Weak:
// a program without the host profiler registers no kernel.
extern "C" void poseidonProfileWritten(const char *funcName)
    __attribute__((weak));

// The canonical form a record was taken against, from its static header lines.
static std::string poseidonCanonicalHash(const char *text) {
  const char *p =
      strstr(text, (std::string(kProfileKeyCanonicalHash) + " = ").c_str());
  if (!p)
    return "";
  const char *e = strchr(p, '\n');
  return std::string(p, e ? (size_t)(e - p) : strlen(p));
}

// The n elements at the device address `from`. The tables live until exit,
// so a copy that finds them gone means the program reset the device first.
template <typename T>
static std::vector<T> poseidonReadBack(const T *from, size_t n) {
  std::vector<T> to(n);
  cudaError_t e =
      cudaMemcpy(to.data(), from, n * sizeof(T), cudaMemcpyDeviceToHost);
  if (e != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon FPProfiler: the profile tables could not be read back "
            "at exit (%s). A program that resets the device before returning "
            "from main (cudaDeviceReset or hipDeviceReset) frees them; the "
            "profile is written at exit, so do not reset the device in a "
            "profiling run. The profile is NOT written.\n",
            cudaGetErrorString(e));
    fflush(stderr);
    _exit(1);
  }
  return to;
}

// Whether the host profiler measures the condition numbers after this
// writer, by running the program again. Weak: a program without the host
// profiler does not.
extern "C" bool poseidonProbeWillMeasure() __attribute__((weak));

static void writeAllCUDAProfiles() {
  poseidonProfileExitCheck();
  // A probe run, or a program that profiled no kernel, has no tables.
  if (!g_devSlots)
    return;
  std::vector<char> names = poseidonReadBack(
      g_devFuncNames, (size_t)g_maxFuncs * POSEIDON_FUNCNAME_MAXLEN);
  std::vector<GPUProfileSlot> slots =
      poseidonReadBack(g_devSlots, (size_t)g_maxFuncs * g_slotCount);
  std::vector<char> statics = poseidonReadBack(
      g_devStaticByFunc, (size_t)g_maxFuncs * POSEIDON_PROFILE_STATIC_MAXLEN);
  std::vector<uint32_t> gridByFunc =
      poseidonReadBack(g_devGridByFunc, (size_t)g_maxFuncs * 3);
  std::vector<uint32_t> blockByFunc =
      poseidonReadBack(g_devBlockByFunc, (size_t)g_maxFuncs * 3);
  std::vector<unsigned long long> launchByFunc =
      poseidonReadBack(g_devLaunchByFunc, (size_t)g_maxFuncs);
  char *h_names = names.data(), *h_static = statics.data();
  GPUProfileSlot *h_slots = slots.data();
  uint32_t *h_gridByFunc = gridByFunc.data(),
           *h_blockByFunc = blockByFunc.data();
  unsigned long long *h_launchByFunc = launchByFunc.data();

  // A function compiled into several device modules (an inline device function
  // called from kernels of two translation units) has one entry per module
  // that ran it; its record is all of them, as with one device copy.
  std::map<std::string, int> firstOf;
  for (int fi = 0; fi < g_maxFuncs; ++fi) {
    char *name = h_names + (size_t)fi * POSEIDON_FUNCNAME_MAXLEN;
    if (name[0] == '\0')
      continue;
    auto [it, fresh] = firstOf.emplace(name, fi);
    if (fresh)
      continue;
    int to = it->second;
    // Two functions of one name (a static kernel, or a lambda of a static
    // function, in two units) cannot share one profile file.
    std::string ha = poseidonCanonicalHash(
        h_static + (size_t)to * POSEIDON_PROFILE_STATIC_MAXLEN);
    std::string hb = poseidonCanonicalHash(
        h_static + (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN);
    if (!ha.empty() && !hb.empty() && ha != hb) {
      fflush(stdout);
      fprintf(stderr,
              "Poseidon FPProfiler: two different functions were profiled "
              "under the name '%s' in two device modules (%s, %s), and one "
              "profile file cannot hold both. Give them distinct names. The "
              "profile is NOT written.\n",
              name, ha.c_str(), hb.c_str());
      fflush(stderr);
      _exit(1);
    }
    for (size_t i = 0; i < g_slotCount; i++) {
      GPUProfileSlot &a = h_slots[(size_t)to * g_slotCount + i];
      const GPUProfileSlot &b = h_slots[(size_t)fi * g_slotCount + i];
      a.minRes = std::max(a.minRes, b.minRes);
      a.maxRes = std::max(a.maxRes, b.maxRes);
      for (int j = 0; j < POSEIDON_MAX_OPERANDS; j++) {
        a.minOperands[j] = std::max(a.minOperands[j], b.minOperands[j]);
        a.maxOperands[j] = std::max(a.maxOperands[j], b.maxOperands[j]);
        a.minMagOperands[j] =
            std::max(a.minMagOperands[j], b.minMagOperands[j]);
      }
      a.sumGrad += b.sumGrad;
      a.sumAbsGrad += b.sumAbsGrad;
      a.sumSens += b.sumSens;
      a.exec += b.exec;
      a.numOperands = std::max(a.numOperands, b.numOperands);
    }
    for (int d = 0; d < 3; d++) {
      h_gridByFunc[to * 3 + d] =
          std::max(h_gridByFunc[to * 3 + d], h_gridByFunc[fi * 3 + d]);
      h_blockByFunc[to * 3 + d] =
          std::max(h_blockByFunc[to * 3 + d], h_blockByFunc[fi * 3 + d]);
    }
    h_launchByFunc[to] += h_launchByFunc[fi];
    char *st = h_static + (size_t)to * POSEIDON_PROFILE_STATIC_MAXLEN;
    if (st[0] == '\0')
      memcpy(st, h_static + (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN,
             POSEIDON_PROFILE_STATIC_MAXLEN);
    name[0] = '\0';
  }

  profileMkdirP(profileDir);

  for (int fi = 0; fi < g_maxFuncs; ++fi) {
    const char *h_funcName = h_names + (size_t)fi * POSEIDON_FUNCNAME_MAXLEN;
    if (h_funcName[0] == '\0')
      continue;
    GPUProfileSlot *base = h_slots + (size_t)fi * g_slotCount;

    std::ostringstream out;
    out << "MaxBlockDims = " << h_blockByFunc[fi * 3 + 0] << " "
        << h_blockByFunc[fi * 3 + 1] << " " << h_blockByFunc[fi * 3 + 2]
        << "\n";
    out << "MaxGridDims = " << h_gridByFunc[fi * 3 + 0] << " "
        << h_gridByFunc[fi * 3 + 1] << " " << h_gridByFunc[fi * 3 + 2] << "\n";
    out << "LaunchCount = " << h_launchByFunc[fi] << "\n";
    // Compile-time static data, already newline-terminated header lines.
    out << (h_static + (size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN);

    for (size_t i = 0; i < g_slotCount; i++) {
      const GPUProfileSlot &slot = base[i];
      if (slot.exec == 0)
        continue;
      ProfileRecord r{orderedUint64ToDoubleMin(slot.minRes),
                      orderedUint64ToDouble(slot.maxRes),
                      slot.sumSens,
                      slot.sumGrad,
                      slot.sumAbsGrad,
                      slot.exec,
                      {},
                      {},
                      {}};
      if (slot.numOperands > POSEIDON_MAX_OPERANDS) {
        fflush(stdout);
        fprintf(stderr,
                "Poseidon FPProfiler: slot %zu of %s records %u operands, "
                "more than the %d a device profile holds "
                "(POSEIDON_MAX_OPERANDS). The profile is NOT written.\n",
                i, h_funcName, slot.numOperands, POSEIDON_MAX_OPERANDS);
        fflush(stderr);
        _exit(1);
      }
      for (uint32_t j = 0; j < slot.numOperands; j++) {
        r.minOperands.push_back(orderedUint64ToDoubleMin(slot.minOperands[j]));
        r.maxOperands.push_back(orderedUint64ToDouble(slot.maxOperands[j]));
        // Stored 0 is never written (doubleToOrderedUint64Min).
        r.minMagOperands.push_back(
            slot.minMagOperands[j] == 0
                ? 0.0
                : orderedUint64ToDoubleMin(slot.minMagOperands[j]));
      }
      writeProfileRecord(out, i, r);
    }
    profileWriteFile(profileDir + "/" + profileFileName(h_funcName), out.str(),
                     h_funcName);
    if (poseidonProfileWritten)
      poseidonProfileWritten(h_funcName);
  }
}

static bool g_cudaInitDone = false;
static size_t g_heapBytes = 0;
// The device the tables were set up on.
static int g_device = -1;

static void poseidonBindModule(const void *table) {
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(table, &g_table, sizeof(g_table)));
}

// The function registry (pointer keys, names, per-function launch geometry)
// and the slots the probes record into. Allocated up front (device code cannot
// allocate) and zeroed, which is also the "unclaimed" state for every field.
static void allocateTables(size_t nfuncs, size_t count) {
  const size_t regBytes = nfuncs * sizeof(const char *);
  const size_t nameBytes = nfuncs * POSEIDON_FUNCNAME_MAXLEN;
  const size_t dimBytes = nfuncs * 3 * sizeof(uint32_t);
  const size_t lcBytes = nfuncs * sizeof(unsigned long long);
  const size_t stBytes = nfuncs * POSEIDON_PROFILE_STATIC_MAXLEN;
  const size_t slotBytes = nfuncs * count * sizeof(GPUProfileSlot);
  cudaError_t err = cudaSuccess;
  auto alloc = [&](void **p, size_t b) {
    cudaError_t e = cudaMalloc(p, b);
    if (e != cudaSuccess && err == cudaSuccess)
      err = e;
  };
  alloc((void **)&g_devFuncReg, regBytes);
  alloc((void **)&g_devFuncNames, nameBytes);
  alloc((void **)&g_devGridByFunc, dimBytes);
  alloc((void **)&g_devBlockByFunc, dimBytes);
  alloc((void **)&g_devLaunchByFunc, lcBytes);
  alloc((void **)&g_devStaticByFunc, stBytes);
  alloc((void **)&g_devSlots, slotBytes);
  if (err != cudaSuccess) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon FPProfiler: allocating the profile tables of %zu "
            "profiled functions of up to %zu probes each (%zu bytes) failed "
            "(%s).\n",
            nfuncs, count,
            regBytes + nameBytes + 2 * dimBytes + lcBytes + stBytes + slotBytes,
            cudaGetErrorString(err));
    abort();
  }
  POSEIDON_CUDA_OK(cudaMemset(g_devFuncReg, 0, regBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devFuncNames, 0, nameBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devGridByFunc, 0, dimBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devBlockByFunc, 0, dimBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devLaunchByFunc, 0, lcBytes));
  POSEIDON_CUDA_OK(cudaMemset(g_devStaticByFunc, 0, stBytes));
  // Zero is the initial state of every field (see doubleToOrderedUint64Min).
  POSEIDON_CUDA_OK(cudaMemset(g_devSlots, 0, slotBytes));
  g_maxFuncs = (int)nfuncs;
  g_slotCount = count;
  g_table.maxFuncs = (int)nfuncs;
  g_table.funcReg = g_devFuncReg;
  g_table.funcNames = g_devFuncNames;
  g_table.gridDimByFunc = g_devGridByFunc;
  g_table.blockDimByFunc = g_devBlockByFunc;
  g_table.launchCountByFunc = g_devLaunchByFunc;
  g_table.staticByFunc = g_devStaticByFunc;
  g_table.slots = g_devSlots;
  g_table.slotCount = count;
}

// What the registered device modules profile: the tables hold the sum of
// their profiled functions, each with as many slots as the largest needs.
static size_t g_registeredFunctions = 0;
static size_t g_registeredSlots = 0;

// Called by the module constructor of every translation unit whose device
// code was compiled for profile generation, with the number of functions that
// module profiles and the probes the largest of them holds. A unit loaded
// after main started is bound to the tables already allocated, which it must
// fit.
extern "C" void __poseidon_profile_register_module(const void *table,
                                                   unsigned functions,
                                                   unsigned slots) {
  poseidonModuleTables().push_back(table);
  g_registeredFunctions += functions;
  g_registeredSlots = std::max<size_t>(g_registeredSlots, slots);
  if (!g_cudaInitDone)
    return;
  if (!g_table.metricOnly && (g_registeredFunctions > (size_t)g_maxFuncs ||
                              g_registeredSlots > g_slotCount)) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon FPProfiler: a device module loaded after main started "
            "profiles %u function(s) of up to %u probes, which the profile "
            "tables sized for the modules registered before main do not "
            "hold. Link the module into the program instead of loading it.\n",
            functions, slots);
    fflush(stderr);
    _exit(1);
  }
  poseidonBindModule(table);
}

// Runs after the profile writer: the condition-number measurement runs the
// program again while this process exits, and those runs need the device
// memory this one holds.
static void poseidonReleaseForProbe() {
  if (poseidonProbeWillMeasure && poseidonProbeWillMeasure())
    (void)cudaDeviceReset();
}

// The CUDA half of registration, called from the top of main. It cannot run
// from a static constructor: in a multi-translation-unit binary a dynamic
// initializer can run before the device module carrying the __constant__
// handles is registered. Idempotent.
extern "C" void __poseidon_profile_init_cuda() {
  if (g_cudaInitDone)
    return;
  g_cudaInitDone = true;

  profileDir = profileDirectory();
  POSEIDON_CUDA_OK(cudaGetDevice(&g_device));

  // A profiled kernel's reverse pass caches what it cannot recompute in the
  // device heap, which CUDA sizes at 8 MiB and an application sizes for its
  // own kernels. It can only be resized before the first kernel that
  // allocates, so it is raised here, and never lowered.
  const size_t heapBytes =
      profileEnvPositive("POSEIDON_PROFILE_HEAP_MB", POSEIDON_DEFAULT_HEAP_MB)
      << 20;
  POSEIDON_CUDA_OK(cudaDeviceGetLimit(&g_heapBytes, cudaLimitMallocHeapSize));
  if (g_heapBytes < heapBytes) {
    cudaError_t e = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapBytes);
    if (e != cudaSuccess) {
      fflush(stdout);
      fprintf(stderr,
              "Poseidon FPProfiler: reserving %zu MiB of device heap for the "
              "profiled kernels' reverse passes failed (%s); set "
              "POSEIDON_PROFILE_HEAP_MB to what the device can spare.\n",
              heapBytes >> 20, cudaGetErrorString(e));
      fflush(stderr);
      _exit(1);
    }
    g_heapBytes = heapBytes;
  }

  // A run of the condition-number measurement reports only the metric, and
  // the tables would cost each of the runs sharing the device their memory.
  g_table.metricOnly = profileIsProbeChild();
  if (!g_table.metricOnly)
    allocateTables(std::max<size_t>(g_registeredFunctions, 1),
                   std::max<size_t>(g_registeredSlots, 1));

  g_table.probeSite = (int)profileEnvInteger("POSEIDON_PROBE_SITE", -1);
  g_table.probeEps = 0.0;
  if (g_table.probeSite >= 0) {
    g_table.probeEps = profileEnvPositiveReal("POSEIDON_PROBE_EPS", 1e-6);
    fprintf(stderr, "Poseidon FPProfiler: perturbing site %d (eps=%g)\n",
            g_table.probeSite, g_table.probeEps);
  }

  for (const void *table : poseidonModuleTables())
    poseidonBindModule(table);
  POSEIDON_CUDA_OK(cudaDeviceSynchronize());
  // atexit handlers run in reverse registration order, and the CUDA runtime
  // registers its own teardown when it is first initialized above, so the
  // writer runs before the driver shuts down and the release after the
  // writer.
  std::atexit(poseidonReleaseForProbe);
  std::atexit(writeAllCUDAProfiles);
}

// ---------------------------------------------------------------------------
// Profile-generation launches
//
// A profiled kernel's reverse pass takes one shadow buffer per pointer
// argument; __poseidon_launch_profiled sizes, allocates and seeds them.
// ---------------------------------------------------------------------------

template <typename T> __global__ void poseidonSeedShadow(T *p, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = T(1);
}

namespace {
// Shadows are zeroed and seeded on the stream of the launch that takes them,
// so that the launch sees them ready whatever stream it runs on.
struct DeviceShadowOps {
  static void *allocZeroed(size_t bytes, const char *, void *stream) {
    void *p = nullptr;
    POSEIDON_CUDA_OK(cudaMalloc(&p, bytes));
    POSEIDON_CUDA_OK(cudaMemsetAsync(p, 0, bytes, (cudaStream_t)stream));
    return p;
  }
  static void free(void *p) { POSEIDON_CUDA_OK(cudaFree(p)); }
  // A run that ended with a device error exits from the profile writer, which
  // runs after this, with the message that names it.
  static void freeAtExit(void *p) { (void)cudaFree(p); }
  // A zero adjoint propagates zero through the whole reverse pass, so the seed
  // is one on the site's outputs.
  template <typename T> static void fillOnes(T *p, size_t n, void *stream) {
    if (n)
      poseidonSeedShadow<<<(unsigned)((n + 255) / 256), 256, 0,
                           (cudaStream_t)stream>>>(p, n);
  }
};

// Profiled launches from several host threads share one shadow table.
static std::mutex g_shadowLock;
} // namespace

// The allocation a device pointer points into. Taken through the driver entry
// point rather than by linking libcuda, so a profile-generation build needs no
// extra library on its link line; HIP's runtime answers it itself.
static void poseidonAllocRange(void *p, void *&base, size_t &size) {
#ifdef __HIP__
  hipError_t e = hipMemGetAddressRange(&base, &size, p);
  if (e != hipSuccess) {
    fprintf(stderr,
            "Poseidon FPProfiler: hipMemGetAddressRange failed (%s) for the "
            "device pointer %p passed to a profiled kernel. Only allocations "
            "the HIP allocator owns can be shadowed.\n",
            hipGetErrorString(e), p);
    abort();
  }
#else
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
#endif
}

// A profiled kernel's launch with the shadows of its pointer arguments
// (ShadowTable).
extern "C" void
__poseidon_launch_profiled(const void *func, int site, const unsigned *grid,
                           const unsigned *block, size_t shmem, void *stream,
                           int nargs, void **args, int nptr, const int *ptrIdx,
                           const int *ptrOff, const int *seedWidth) {
  int device = -1;
  POSEIDON_CUDA_OK(cudaGetDevice(&device));
  if (device != g_device) {
    fflush(stdout);
    fprintf(stderr,
            "Poseidon FPProfiler: site %d is launched on device %d, and the "
            "profile tables were set up on device %d; profile generation "
            "supports one device per process, so profile on one device.\n",
            site, device, g_device);
    fflush(stderr);
    _exit(1);
  }
  std::vector<void *> full(args, args + nargs);
  std::vector<void *> shadow((size_t)nptr, nullptr);
  std::vector<void *> base((size_t)nptr, nullptr);
  std::vector<size_t> off((size_t)nptr, 0), size((size_t)nptr, 0);
  // An allocation this launch writes through any of its arguments is an output
  // of the launch, whichever argument first reaches it below.
  std::map<void *, int> width;
  for (int k = 0; k < nptr; ++k) {
    // A pointer argument, or a pointer field of an aggregate argument (the
    // captures of a lambda the kernel takes by value).
    void *p = *(void **)((char *)args[ptrIdx[k]] + ptrOff[k]);
    if (!p)
      continue;
    poseidonAllocRange(p, base[k], size[k]);
    off[k] = (uintptr_t)p - (uintptr_t)base[k];
    int &w = width[base[k]];
    w = std::max(w, seedWidth[k]);
  }
  {
    std::lock_guard<std::mutex> lock(g_shadowLock);
    for (int k = 0; k < nptr; ++k) {
      if (base[k])
        shadow[k] = (char *)ShadowTable<DeviceShadowOps>::get().shadowOf(
                        (uintptr_t)base[k], size[k], site, width[base[k]],
                        "a profiled kernel", stream) +
                    off[k];
      full.push_back(&shadow[k]);
    }
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
            "The profile is incomplete and is NOT written. A reverse pass that "
            "outgrows the device heap (%zu MiB, POSEIDON_PROFILE_HEAP_MB) "
            "faults this way too.\n",
            site, cudaGetErrorString(e), g_heapBytes >> 20);
    fflush(stderr);
    _exit(1);
  }
}
