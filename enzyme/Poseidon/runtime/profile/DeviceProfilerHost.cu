//===- DeviceProfilerHost.cu - host half of the GPU profiler --------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The host half of the GPU profiler, a library a profile-generate build links
// (libposeidon_profile_cuda, and compiled as HIP libposeidon_profile_hip): it
// allocates the tables the probes of every profiled device module record into
// (DeviceProfilerDevice.cu), writes each module's handles to them, and writes
// the profile at exit.
//
//===----------------------------------------------------------------------===//

#include "../DeviceApi.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "DeviceProfiler.h"
#include "ProfileFormat.h"
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
static GPUTrafficSlot *g_devTraffic = nullptr;
static int *g_devTrafficFunc = nullptr;

// What every profiled device module's table holds, and the host shadows of
// those tables: one per translation unit, registered by that unit's module
// constructor, so each names the copy in its own device module.
static DeviceProfileTable g_table;
static std::vector<const void *> &poseidonModuleTables() {
  static auto *v = new std::vector<const void *>();
  return *v;
}

// A device-side profiler abort (gpuTrap) poisons the context without failing
// the process, so the exit writer checks the context first and exits nonzero.
static void poseidonProfileExitCheck() {
  gpuError_t st = gpuDeviceSynchronize();
  if (st != gpuSuccess)
    profileFail("the profiled run ended with a %s error (%s). A device-side "
                "profiler abort leaves the context in this state (see the "
                "'[poseidon]' message above); the profile is "
                "incomplete and is NOT written.",
                POSEIDON_CUDA_OR_HIP("CUDA", "HIP"), gpuGetErrorString(st));
}

// Tells the host profiler which kernels this writer wrote a profile for. Weak:
// a program without the host profiler registers no kernel.
extern "C" void __poseidon_profile_written(const char *funcName)
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
  gpuError_t e =
      gpuMemcpy(to.data(), from, n * sizeof(T), gpuMemcpyDeviceToHost);
  if (e != gpuSuccess)
    profileFail("the profile tables could not be read back at exit (%s). A "
                "program that resets the device before returning from main "
                "(cudaDeviceReset or hipDeviceReset) frees them; the profile "
                "is written at exit, so do not reset the device in a "
                "profiling run. The profile is NOT written.",
                gpuGetErrorString(e));
  return to;
}

namespace {
// The traffic of one array over every launch of a kernel site: the extents
// each launch touched, summed.
struct ArrayTraffic {
  unsigned long long loads = 0, stores = 0;
  unsigned long long loadBytes = 0, storeBytes = 0;
  unsigned widths = 0;

  ArrayTraffic &operator+=(const ArrayTraffic &o) {
    loads += o.loads;
    stores += o.stores;
    loadBytes += o.loadBytes;
    storeBytes += o.storeBytes;
    widths |= o.widths;
    return *this;
  }
};
} // namespace

// The traffic of every record of one registry entry.
using EntryTraffic = std::array<ArrayTraffic, POSEIDON_MAX_TRAFFIC_RECORDS>;

// Per registry entry, per traffic record: what the launches so far moved.
static std::map<int, EntryTraffic> &launchTraffic() {
  static auto *m = new std::map<int, EntryTraffic>();
  return *m;
}

namespace {
// The tables read back at exit, a row per registry entry.
struct ProfileTables {
  std::vector<char> names, statics;
  std::vector<GPUProfileSlot> slots;
  std::vector<uint32_t> grid, block; // three per row
  std::vector<unsigned long long> launches;
  std::map<int, EntryTraffic> traffic;

  char *name(int fi) { return &names[(size_t)fi * POSEIDON_FUNCNAME_MAXLEN]; }
  char *header(int fi) {
    return &statics[(size_t)fi * POSEIDON_PROFILE_STATIC_MAXLEN];
  }
  GPUProfileSlot *slotsOf(int fi) { return &slots[(size_t)fi * g_slotCount]; }
};
} // namespace

static ProfileTables readBackTables() {
  const size_t n = (size_t)g_maxFuncs;
  ProfileTables t;
  t.names = poseidonReadBack(g_devFuncNames, n * POSEIDON_FUNCNAME_MAXLEN);
  t.slots = poseidonReadBack(g_devSlots, n * g_slotCount);
  t.statics =
      poseidonReadBack(g_devStaticByFunc, n * POSEIDON_PROFILE_STATIC_MAXLEN);
  t.grid = poseidonReadBack(g_devGridByFunc, n * 3);
  t.block = poseidonReadBack(g_devBlockByFunc, n * 3);
  t.launches = poseidonReadBack(g_devLaunchByFunc, n);
  t.traffic = launchTraffic();
  return t;
}

// Folds row `from` into row `to`: every field is a commutative reduction of
// what the probes record.
static void mergeRow(ProfileTables &t, int to, int from) {
  for (size_t i = 0; i < g_slotCount; i++) {
    GPUProfileSlot &a = t.slotsOf(to)[i];
    const GPUProfileSlot &b = t.slotsOf(from)[i];
    a.minRes = std::max(a.minRes, b.minRes);
    a.maxRes = std::max(a.maxRes, b.maxRes);
    for (int j = 0; j < POSEIDON_MAX_OPERANDS; j++) {
      a.minOperands[j] = std::max(a.minOperands[j], b.minOperands[j]);
      a.maxOperands[j] = std::max(a.maxOperands[j], b.maxOperands[j]);
      a.minMagOperands[j] = std::max(a.minMagOperands[j], b.minMagOperands[j]);
    }
    a.sumGrad += b.sumGrad;
    a.sumAbsGrad += b.sumAbsGrad;
    a.sumSens += b.sumSens;
    a.sumSqGrad += b.sumSqGrad;
    a.sumSqSens += b.sumSqSens;
    a.exec += b.exec;
    a.numOperands = std::max(a.numOperands, b.numOperands);
  }
  for (int d = 0; d < 3; d++) {
    t.grid[to * 3 + d] = std::max(t.grid[to * 3 + d], t.grid[from * 3 + d]);
    t.block[to * 3 + d] = std::max(t.block[to * 3 + d], t.block[from * 3 + d]);
  }
  t.launches[to] += t.launches[from];
  auto moved = t.traffic.find(from);
  if (moved != t.traffic.end()) {
    EntryTraffic &into = t.traffic[to];
    for (int r = 0; r < POSEIDON_MAX_TRAFFIC_RECORDS; ++r)
      into[r] += moved->second[r];
    t.traffic.erase(from);
  }
  if (t.header(to)[0] == '\0')
    memcpy(t.header(to), t.header(from), POSEIDON_PROFILE_STATIC_MAXLEN);
}

// A function compiled into several device modules (an inline device function
// called from kernels of two translation units) has one row per module that
// ran it; its record is all of them, as with one device copy. Merged rows lose
// their name.
static void mergeModuleCopies(ProfileTables &t) {
  std::map<std::string, int> firstOf;
  for (int fi = 0; fi < g_maxFuncs; ++fi) {
    char *name = t.name(fi);
    if (name[0] == '\0')
      continue;
    auto [it, fresh] = firstOf.emplace(name, fi);
    if (fresh)
      continue;
    const int to = it->second;
    // Two functions of one name (a static kernel, or a lambda of a static
    // function, in two units) cannot share one profile file.
    std::string ha = poseidonCanonicalHash(t.header(to));
    std::string hb = poseidonCanonicalHash(t.header(fi));
    if (!ha.empty() && !hb.empty() && ha != hb)
      profileFail("two different functions were profiled under the name '%s' "
                  "in two device modules (%s, %s), and one profile file "
                  "cannot hold both. Give them distinct names. The profile is "
                  "NOT written.",
                  name, ha.c_str(), hb.c_str());
    mergeRow(t, to, fi);
    name[0] = '\0';
  }
}

// The profile of row `fi`: its launch geometry, its static header lines and
// the record of every slot that executed.
static std::string profileOf(ProfileTables &t, int fi) {
  const char *funcName = t.name(fi);
  std::ostringstream out;
  out << "MaxBlockDims = " << t.block[fi * 3 + 0] << " " << t.block[fi * 3 + 1]
      << " " << t.block[fi * 3 + 2] << "\n";
  out << "MaxGridDims = " << t.grid[fi * 3 + 0] << " " << t.grid[fi * 3 + 1]
      << " " << t.grid[fi * 3 + 2] << "\n";
  out << "LaunchCount = " << t.launches[fi] << "\n";
  // Compile-time static data, already newline-terminated header lines.
  out << t.header(fi);
  auto traffic = t.traffic.find(fi);
  if (traffic != t.traffic.end())
    for (int r = 0; r < POSEIDON_MAX_TRAFFIC_RECORDS; ++r) {
      const ArrayTraffic &a = traffic->second[r];
      if (!a.loads && !a.stores)
        continue;
      unsigned width =
          __builtin_popcount(a.widths) == 1 ? 1u << __builtin_ctz(a.widths) : 0;
      out << kProfileKeyTraffic << " = " << r << " " << width << " " << a.loads
          << " " << a.stores << " " << a.loadBytes << " " << a.storeBytes
          << "\n";
    }
  for (size_t i = 0; i < g_slotCount; i++) {
    const GPUProfileSlot &slot = t.slotsOf(fi)[i];
    if (slot.exec == 0)
      continue;
    if (slot.numOperands > POSEIDON_MAX_OPERANDS)
      profileFail("slot %zu of %s records %u operands, more than the %d a "
                  "device profile holds (POSEIDON_MAX_OPERANDS). The profile "
                  "is NOT written.",
                  i, funcName, slot.numOperands, POSEIDON_MAX_OPERANDS);
    ProfileRecord r{orderedUint64ToDoubleMin(slot.minRes),
                    orderedUint64ToDouble(slot.maxRes),
                    slot.sumSens,
                    slot.sumGrad,
                    slot.sumAbsGrad,
                    slot.sumSqGrad,
                    slot.sumSqSens,
                    slot.exec,
                    {},
                    {},
                    {}};
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
  return out.str();
}

static void writeDeviceProfiles() {
  poseidonProfileExitCheck();
  // A kappa run, or a program that profiled no kernel, has no tables.
  if (!g_devSlots)
    return;
  ProfileTables t = readBackTables();
  mergeModuleCopies(t);
  profileMkdirP(profileDir);
  for (int fi = 0; fi < g_maxFuncs; ++fi) {
    const char *funcName = t.name(fi);
    if (funcName[0] == '\0')
      continue;
    profileWriteFile(profileDir + "/" + profileFileName(funcName),
                     profileOf(t, fi), funcName);
    if (__poseidon_profile_written)
      __poseidon_profile_written(funcName);
  }
}

static bool g_deviceInitDone = false;
static size_t g_heapBytes = 0;
// The device the tables were set up on.
static int g_device = -1;

static void poseidonBindModule(const void *table) {
  POSEIDON_GPU_CHECK(gpuMemcpyToSymbol(table, &g_table, sizeof(g_table)));
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
  const size_t trafficBytes =
      nfuncs * POSEIDON_MAX_TRAFFIC_RECORDS * sizeof(GPUTrafficSlot);
  gpuError_t err = gpuSuccess;
  auto alloc = [&](void **p, size_t b) {
    gpuError_t e = gpuMalloc(p, b);
    if (e != gpuSuccess && err == gpuSuccess)
      err = e;
  };
  alloc((void **)&g_devFuncReg, regBytes);
  alloc((void **)&g_devFuncNames, nameBytes);
  alloc((void **)&g_devGridByFunc, dimBytes);
  alloc((void **)&g_devBlockByFunc, dimBytes);
  alloc((void **)&g_devLaunchByFunc, lcBytes);
  alloc((void **)&g_devStaticByFunc, stBytes);
  alloc((void **)&g_devSlots, slotBytes);
  alloc((void **)&g_devTraffic, trafficBytes);
  alloc((void **)&g_devTrafficFunc, sizeof(int));
  if (err != gpuSuccess)
    fatal(
        "allocating the profile tables of %zu profiled functions of up to %zu "
        "probes each (%zu bytes) failed (%s).",
        nfuncs, count,
        regBytes + nameBytes + 2 * dimBytes + lcBytes + stBytes + slotBytes +
            trafficBytes + sizeof(int),
        gpuGetErrorString(err));
  POSEIDON_GPU_CHECK(gpuMemset(g_devFuncReg, 0, regBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devFuncNames, 0, nameBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devGridByFunc, 0, dimBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devBlockByFunc, 0, dimBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devLaunchByFunc, 0, lcBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devStaticByFunc, 0, stBytes));
  // Zero is the initial state of every field (see doubleToOrderedUint64Min).
  POSEIDON_GPU_CHECK(gpuMemset(g_devSlots, 0, slotBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devTraffic, 0, trafficBytes));
  POSEIDON_GPU_CHECK(gpuMemset(g_devTrafficFunc, 0, sizeof(int)));
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
  g_table.traffic = g_devTraffic;
  g_table.trafficFunc = g_devTrafficFunc;
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
extern "C" void POSEIDON_PROFILE_REGISTER_MODULE(const void *table,
                                                 unsigned functions,
                                                 unsigned slots) {
  poseidonModuleTables().push_back(table);
  g_registeredFunctions += functions;
  g_registeredSlots = std::max<size_t>(g_registeredSlots, slots);
  if (!g_deviceInitDone)
    return;
  if (!g_table.metricOnly && (g_registeredFunctions > (size_t)g_maxFuncs ||
                              g_registeredSlots > g_slotCount))
    profileFail("a device module loaded after main started profiles %u "
                "function(s) of up to %u probes, which the profile tables "
                "sized for the modules registered before main do not hold. "
                "Link the module into the program instead of loading it.",
                functions, slots);
  poseidonBindModule(table);
}

// The CUDA half of registration, called from the top of main. It cannot run
// from a static constructor: in a multi-translation-unit binary a dynamic
// initializer can run before the device module carrying the __constant__
// handles is registered. Idempotent.
extern "C" void __poseidon_profile_init_device() {
  if (g_deviceInitDone)
    return;
  g_deviceInitDone = true;

  profileDir = profileDirectory();
  POSEIDON_GPU_CHECK(gpuGetDevice(&g_device));

  // A profiled kernel's reverse pass caches what it cannot recompute in the
  // device heap, which CUDA sizes at 8 MiB and an application sizes for its
  // own kernels. It can only be resized before the first kernel that
  // allocates, so it is raised here, and never lowered.
  const size_t heapBytes =
      profileEnvPositive("POSEIDON_PROFILE_HEAP_MB", POSEIDON_DEFAULT_HEAP_MB)
      << 20;
  POSEIDON_GPU_CHECK(gpuDeviceGetLimit(&g_heapBytes, gpuLimitMallocHeapSize));
  if (g_heapBytes < heapBytes) {
    gpuError_t e = gpuDeviceSetLimit(gpuLimitMallocHeapSize, heapBytes);
    if (e != gpuSuccess)
      profileFail("reserving %zu MiB of device heap for the profiled kernels' "
                  "reverse passes failed (%s); set POSEIDON_PROFILE_HEAP_MB to "
                  "what the device can spare.",
                  heapBytes >> 20, gpuGetErrorString(e));
    g_heapBytes = heapBytes;
  }

  // A run of the condition-number measurement reports only the metric, and
  // the tables would cost each of the runs sharing the device their memory.
  g_table.metricOnly = profileIsKappaChild();
  if (!g_table.metricOnly)
    allocateTables(std::max<size_t>(g_registeredFunctions, 1),
                   std::max<size_t>(g_registeredSlots, 1));

  const KappaPerturbation kappa = profileKappaPerturbation();
  g_table.kappaSite = kappa.site;
  g_table.kappaEps = kappa.eps;
  if (kappa.site >= 0)
    report("perturbing site %d (eps=%g)", kappa.site, kappa.eps);

  for (const void *table : poseidonModuleTables())
    poseidonBindModule(table);
  POSEIDON_GPU_CHECK(gpuDeviceSynchronize());
  // atexit handlers run in reverse registration order, and the CUDA runtime
  // registers its own teardown when it is first initialized above, so the
  // writer runs before the driver shuts down.
  std::atexit(writeDeviceProfiles);
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
    p[i] = T(outputSeed(i));
}

namespace {
// Shadows are zeroed and seeded on the stream of the launch that takes them,
// so that the launch sees them ready whatever stream it runs on.
struct DeviceShadowOps {
  static void *allocZeroed(size_t bytes, const char *, void *stream) {
    void *p = nullptr;
    POSEIDON_GPU_CHECK(gpuMalloc(&p, bytes));
    POSEIDON_GPU_CHECK(gpuMemsetAsync(p, 0, bytes, (gpuStream_t)stream));
    return p;
  }
  static void free(void *p) { POSEIDON_GPU_CHECK(gpuFree(p)); }
  // A run that ended with a device error exits from the profile writer, which
  // runs after this, with the message that names it.
  static void freeAtExit(void *p) { (void)gpuFree(p); }
  // A zero adjoint propagates zero through the whole reverse pass, so the
  // site's outputs carry the seed.
  template <typename T> static void fillSeed(T *p, size_t n, void *stream) {
    if (n)
      poseidonSeedShadow<<<(unsigned)((n + 255) / 256), 256, 0,
                           (gpuStream_t)stream>>>(p, n);
  }
};
} // namespace

// Profiled launches from several host threads share one shadow table.
static std::mutex g_shadowLock;

// The allocation a device pointer points into. Taken through the driver entry
// point rather than by linking libcuda, so a profile-generation build needs no
// extra library on its link line; HIP's runtime answers it itself.
static void poseidonAllocRange(void *p, void *&base, size_t &size) {
#ifdef __HIP__
  hipError_t e = hipMemGetAddressRange(&base, &size, p);
  if (e != hipSuccess)
    fatal("hipMemGetAddressRange failed (%s) for the device pointer %p "
          "passed to a profiled kernel. Only allocations the HIP "
          "allocator owns can be shadowed.",
          hipGetErrorString(e), p);
#else
  typedef int (*PFN_MemGetAddressRange)(unsigned long long *, size_t *,
                                        unsigned long long);
  static PFN_MemGetAddressRange getRange = nullptr;
  if (!getRange) {
    void *fn = nullptr;
    cudaError_t e =
        cudaGetDriverEntryPoint("cuMemGetAddressRange", &fn, cudaEnableDefault);
    if (e != cudaSuccess || !fn)
      fatal("cuMemGetAddressRange is not available (%s); the size of a "
            "profiled kernel's buffers cannot be determined.",
            cudaGetErrorString(e));
    getRange = (PFN_MemGetAddressRange)fn;
  }
  unsigned long long b = 0;
  size_t n = 0;
  int rc = getRange(&b, &n, (unsigned long long)(uintptr_t)p);
  if (rc != 0)
    fatal("cuMemGetAddressRange failed (%d) for the device pointer %p "
          "passed to a profiled kernel. Only allocations the CUDA "
          "allocator owns can be shadowed.",
          rc, p);
  base = (void *)(uintptr_t)b;
  size = n;
#endif
}

// Adds the extents the launch that just finished touched to its function's
// traffic records, and clears them for the next launch.
static void foldLaunchTraffic() {
  if (!g_devTrafficFunc)
    return;
  int fi1 = 0;
  POSEIDON_GPU_CHECK(
      gpuMemcpy(&fi1, g_devTrafficFunc, sizeof(int), gpuMemcpyDeviceToHost));
  if (!fi1)
    return;
  const int fi = fi1 - 1;
  GPUTrafficSlot *dev =
      g_devTraffic + (size_t)fi * POSEIDON_MAX_TRAFFIC_RECORDS;
  GPUTrafficSlot launch[POSEIDON_MAX_TRAFFIC_RECORDS];
  POSEIDON_GPU_CHECK(
      gpuMemcpy(launch, dev, sizeof(launch), gpuMemcpyDeviceToHost));
  POSEIDON_GPU_CHECK(gpuMemset(dev, 0, sizeof(launch)));
  POSEIDON_GPU_CHECK(gpuMemset(g_devTrafficFunc, 0, sizeof(int)));
  EntryTraffic &total = launchTraffic()[fi];
  for (int r = 0; r < POSEIDON_MAX_TRAFFIC_RECORDS; ++r) {
    const GPUTrafficSlot &l = launch[r];
    ArrayTraffic a;
    a.loads = l.loads;
    a.stores = l.stores;
    if (l.loads)
      a.loadBytes = l.loadHigh - ~l.loadLow;
    if (l.stores)
      a.storeBytes = l.storeHigh - ~l.storeLow;
    a.widths = l.widths;
    total[r] += a;
  }
}

// A profiled kernel's launch with the shadows of its pointer arguments
// (ShadowTable).
extern "C" void
POSEIDON_LAUNCH_PROFILED(const void *func, int site, const unsigned *grid,
                         const unsigned *block, size_t shmem, void *stream,
                         int nargs, void **args, int nptr, const int *ptrIdx,
                         const int *ptrOff, const int *seedWidth) {
  int device = -1;
  POSEIDON_GPU_CHECK(gpuGetDevice(&device));
  if (device != g_device)
    profileFail("site %d is launched on device %d, and the profile tables "
                "were set up on device %d; profile generation supports one "
                "device per process, so profile on one device.",
                site, device, g_device);
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
  gpuError_t e =
      gpuLaunchKernel(func, g, b, full.data(), shmem * 2, (gpuStream_t)stream);
  if (e == gpuSuccess)
    e = gpuStreamSynchronize((gpuStream_t)stream);
  if (e != gpuSuccess)
    profileFail("the profiled launch of site %d failed (%s). The profile is "
                "incomplete and is NOT written. A reverse pass that outgrows "
                "the device heap (%zu MiB, POSEIDON_PROFILE_HEAP_MB) faults "
                "this way too.",
                site, gpuGetErrorString(e), g_heapBytes >> 20);
  foldLaunchTraffic();
}
