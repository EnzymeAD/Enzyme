// The host half of the CUDA profiler, a library a profile-generate build links
// (libposeidon_profile_cuda, and compiled as HIP libposeidon_profile_hip): it
// allocates the tables the probes of every profiled device module record into
// (FPProfilerDevice.cu), writes each module's handles to them, and writes the
// profile at exit.
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
#include "FPProfilerCUDA.h"
#include "poseidon/poseidon.h"

static std::string profileDir = POSEIDON_DEFAULT_PROFILE_DIR;
static GPUProfileSlot *g_devSlots = nullptr;
static size_t g_slotCount = 0;
// Host-side handles on the device-allocated registry (the device side reaches
// them through its module's PoseidonCUDAProfileTable; the host needs the raw
// pointers to read them back at exit).
static int g_maxFuncs = 0;
static const char **g_devFuncReg = nullptr;
static char *g_devFuncNames = nullptr;
static uint32_t *g_devGridByFunc = nullptr;
static uint32_t *g_devBlockByFunc = nullptr;
static unsigned long long *g_devLaunchByFunc = nullptr;
static char *g_devStaticByFunc = nullptr;

// What every profiled device module's table holds, and the host shadows of
// those tables: one per translation unit, registered with the CUDA runtime by
// that unit's own module constructor (the pass emits both), so each names the
// copy in its own device module.
static PoseidonCUDAProfileTable g_table;
static std::vector<const void *> &poseidonModuleTables() {
  static auto *v = new std::vector<const void *>();
  return *v;
}

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

// The canonical form a record was taken against, from its static header lines.
static std::string poseidonCanonicalHash(const char *text) {
  const char *p = strstr(text, "CanonicalHash = ");
  if (!p)
    return "";
  const char *e = strchr(p, '\n');
  return std::string(p, e ? (size_t)(e - p) : strlen(p));
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
      a.sumValue += b.sumValue;
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
static size_t g_heapBytes = 0;

static void poseidonBindModule(const void *table) {
  POSEIDON_CUDA_OK(cudaMemcpyToSymbol(table, &g_table, sizeof(g_table)));
}

// Called by the module constructor of every translation unit whose device
// code was compiled for profile generation; a unit loaded after main started
// is bound here, the others when main starts.
extern "C" void __poseidon_profile_register_module(const void *table) {
  poseidonModuleTables().push_back(table);
  if (g_cudaInitDone)
    poseidonBindModule(table);
}

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

  // A profiled kernel's reverse pass caches what it cannot recompute in the
  // device heap, which CUDA sizes at 8 MiB and an application sizes for its
  // own kernels. It can only be resized before the first kernel that
  // allocates, so it is raised here, and never lowered.
  size_t heapBytes =
      readPositiveEnv("POSEIDON_PROFILE_HEAP_MB", POSEIDON_DEFAULT_HEAP_MB)
      << 20;
  POSEIDON_CUDA_OK(cudaDeviceGetLimit(&g_heapBytes, cudaLimitMallocHeapSize));
  if (g_heapBytes < heapBytes) {
    POSEIDON_CUDA_OK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapBytes));
    g_heapBytes = heapBytes;
  }

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
  g_table.maxFuncs = (int)nfuncs;
  g_table.funcReg = g_devFuncReg;
  g_table.funcNames = g_devFuncNames;
  g_table.gridDimByFunc = g_devGridByFunc;
  g_table.blockDimByFunc = g_devBlockByFunc;
  g_table.launchCountByFunc = g_devLaunchByFunc;
  g_table.staticByFunc = g_devStaticByFunc;

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
  g_table.slots = g_devSlots;
  g_table.slotCount = count;

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
    g_table.probeSite = probeSite;
    g_table.probeEps = probeEps;
  }

  // Zero IS the initial state of every field (see doubleToOrderedUint64Min),
  // so no init kernel is launched from this static initializer.
  POSEIDON_CUDA_OK(cudaMemset(g_devSlots, 0, bytes));
  for (const void *table : poseidonModuleTables())
    poseidonBindModule(table);
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

extern "C" void
__poseidon_launch_profiled(const void *func, int site, const unsigned *grid,
                           const unsigned *block, size_t shmem, void *stream,
                           int nargs, void **args, int nptr, const int *ptrIdx,
                           const int *ptrOff, const int *seedWidth) {
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
            "The profile is incomplete and is NOT written. A reverse pass that "
            "outgrows the device heap (%zu MiB, POSEIDON_PROFILE_HEAP_MB) "
            "faults this way too.\n",
            site, cudaGetErrorString(e), g_heapBytes >> 20);
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
