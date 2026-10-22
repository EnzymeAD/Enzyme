// Profile generation of POSEIDON_OPTIMIZE functions on the host.
//
// A host site's reverse pass takes one shadow per pointer argument, sized to
// the allocation the argument points into, exactly as a profiled kernel launch
// (__poseidon_launch_profiled). The device has cuMemGetAddressRange for that;
// the host does not, so this runtime keeps the extents itself: every heap
// allocation of the process (malloc and its family are replaced here, which
// also covers operator new and every library), the globals and the stack
// objects the instrumented translation units register.

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <unistd.h>
#include <vector>

// Defined by every translation unit that has a host site: without one the
// hooks below pass straight through.
extern "C" __attribute__((weak)) const char __poseidon_host_sites_present;

namespace {

struct Extent {
  uintptr_t base;
  size_t bytes;

  bool contains(uintptr_t a) const {
    return a >= base && (a - base < bytes || (bytes == 0 && a == base));
  }
};

struct Registry {
  std::mutex lock;
  std::map<uintptr_t, size_t> heap;
  std::map<uintptr_t, size_t> globals;
};

// Never destroyed: allocations are freed during exit, after static destructors
// would have run.
Registry &registry() {
  alignas(Registry) static unsigned char storage[sizeof(Registry)];
  static Registry *r = new (storage) Registry();
  return *r;
}

// Set while the registry itself allocates, so that its own nodes are not
// recorded through the hooks below.
__thread bool inRegistry __attribute__((tls_model("initial-exec")));

// The stack objects the calls in flight on this thread were passed, innermost
// last.
__thread std::vector<Extent> *stackArrays
    __attribute__((tls_model("initial-exec")));

bool findIn(const std::map<uintptr_t, size_t> &m, uintptr_t a, Extent &out) {
  auto it = m.upper_bound(a);
  if (it == m.begin())
    return false;
  --it;
  Extent e{it->first, it->second};
  if (!e.contains(a))
    return false;
  out = e;
  return true;
}

bool findExtent(const void *p, Extent &out) {
  uintptr_t a = (uintptr_t)p;
  if (stackArrays)
    for (auto it = stackArrays->rbegin(); it != stackArrays->rend(); ++it)
      if (it->contains(a)) {
        out = *it;
        return true;
      }
  Registry &r = registry();
  std::lock_guard<std::mutex> g(r.lock);
  return findIn(r.globals, a, out) || findIn(r.heap, a, out);
}

void noteHeap(void *p, size_t bytes) {
  if (!p || inRegistry || !&__poseidon_host_sites_present)
    return;
  inRegistry = true;
  {
    Registry &r = registry();
    std::lock_guard<std::mutex> g(r.lock);
    r.heap[(uintptr_t)p] = bytes;
  }
  inRegistry = false;
}

void forgetHeap(void *p) {
  if (!p || inRegistry || !&__poseidon_host_sites_present)
    return;
  inRegistry = true;
  {
    Registry &r = registry();
    std::lock_guard<std::mutex> g(r.lock);
    r.heap.erase((uintptr_t)p);
  }
  inRegistry = false;
}

// A profiling run that runs out of memory says so once, before whatever the
// program does with the null pointer.
void noteFailure(size_t bytes) {
  static std::atomic<bool> said{false};
  if (!bytes || said.exchange(true))
    return;
  char msg[128];
  int n = snprintf(msg, sizeof msg,
                   "Poseidon FPProfiler: out of memory (an allocation of %zu "
                   "bytes failed)\n",
                   bytes);
  if (n > 0)
    (void)!write(2, msg, (size_t)n < sizeof msg ? (size_t)n : sizeof msg - 1);
}

} // namespace

#if defined(__GLIBC__)
extern "C" {
void *__libc_malloc(size_t);
void *__libc_calloc(size_t, size_t);
void *__libc_realloc(void *, size_t);
void *__libc_memalign(size_t, size_t);
void __libc_free(void *);

void *malloc(size_t bytes) {
  void *p = __libc_malloc(bytes);
  if (!p)
    noteFailure(bytes);
  noteHeap(p, bytes);
  return p;
}

void *calloc(size_t n, size_t size) {
  void *p = __libc_calloc(n, size);
  if (!p)
    noteFailure(n * size);
  noteHeap(p, n * size);
  return p;
}

void *realloc(void *old, size_t bytes) {
  void *p = __libc_realloc(old, bytes);
  if (p || bytes == 0)
    forgetHeap(old);
  else
    noteFailure(bytes);
  noteHeap(p, bytes);
  return p;
}

void free(void *p) {
  forgetHeap(p);
  __libc_free(p);
}

void *memalign(size_t align, size_t bytes) {
  void *p = __libc_memalign(align, bytes);
  if (!p)
    noteFailure(bytes);
  noteHeap(p, bytes);
  return p;
}

void *aligned_alloc(size_t align, size_t bytes) {
  return memalign(align, bytes);
}

int posix_memalign(void **out, size_t align, size_t bytes) {
  if (align % sizeof(void *) || (align & (align - 1)))
    return EINVAL;
  void *p = __libc_memalign(align, bytes);
  if (!p) {
    noteFailure(bytes);
    return ENOMEM;
  }
  noteHeap(p, bytes);
  *out = p;
  return 0;
}

void *valloc(size_t bytes) {
  return memalign((size_t)sysconf(_SC_PAGESIZE), bytes);
}
}
#endif

namespace {

struct Shadow {
  void *p = nullptr;
  size_t bytes = 0;
  // The sites whose outputs this shadow already carries the seed for (see
  // __poseidon_launch_profiled).
  std::set<int> seeded;
};

std::mutex shadowLock;

std::map<uintptr_t, Shadow> &shadows() {
  static auto *m = new std::map<uintptr_t, Shadow>();
  return *m;
}

void freeShadows() {
  for (auto &kv : shadows())
    free(kv.second.p);
  shadows().clear();
}

template <typename T> void fill(void *p, size_t bytes) {
  T *e = (T *)p;
  for (size_t i = 0, n = bytes / sizeof(T); i < n; ++i)
    e[i] = T(1);
}

// The seed rule of the kernel path: one on the site's outputs.
void seed(void *p, size_t bytes, int width) {
  if (width == 8)
    fill<double>(p, bytes);
  else if (width == 4)
    fill<float>(p, bytes);
  else {
    fprintf(stderr,
            "Poseidon FPProfiler: no seed rule for a %d-byte output element.\n",
            width);
    abort();
  }
}

} // namespace

extern "C" void __poseidon_stack_push(void *base, uint64_t bytes) {
  if (!stackArrays) {
    inRegistry = true;
    stackArrays = new std::vector<Extent>();
    inRegistry = false;
  }
  inRegistry = true;
  stackArrays->push_back({(uintptr_t)base, (size_t)bytes});
  inRegistry = false;
}

extern "C" void __poseidon_stack_pop(uint32_t n) {
  if (stackArrays)
    stackArrays->resize(stackArrays->size() > n ? stackArrays->size() - n : 0);
}

extern "C" void __poseidon_register_globals(const void *table, uint64_t n) {
  struct Entry {
    void *p;
    uint64_t bytes;
  };
  const Entry *e = (const Entry *)table;
  inRegistry = true;
  {
    Registry &r = registry();
    std::lock_guard<std::mutex> g(r.lock);
    for (uint64_t i = 0; i < n; ++i)
      r.globals[(uintptr_t)e[i].p] = e[i].bytes;
  }
  inRegistry = false;
}

// One call of a host site: the shadow of each pointer argument, by the rules
// of __poseidon_launch_profiled. Keyed by the base of the allocation, created
// zeroed on first sight and replaced when that base comes back with another
// extent; an allocation the call writes through any argument is seeded once
// per site.
extern "C" void __poseidon_host_shadows(const char *fn, int site, int nptr,
                                        void *const *ptrs, const int *param,
                                        const int *seedWidth,
                                        const char *const *paramName,
                                        void **out) {
  std::vector<Extent> ext((size_t)nptr, Extent{0, 0});
  std::map<uintptr_t, int> width;
  for (int k = 0; k < nptr; ++k) {
    if (!ptrs[k])
      continue;
    if (!findExtent(ptrs[k], ext[k])) {
      fflush(stdout);
      fprintf(stderr,
              "Poseidon FPProfiler: argument %d%s%s%s of %s points to %p, "
              "which the profiler did not see allocated. It sizes a site's "
              "shadows from heap allocations, and from the globals and the "
              "stack objects passed directly to a call in a translation unit "
              "that defines a POSEIDON_OPTIMIZE function. Allocate this array "
              "on the heap, or profile the call with __poseidon_fp_optimize "
              "and a shadow of your own.\n",
              param[k], paramName[k][0] ? " ('" : "", paramName[k],
              paramName[k][0] ? "')" : "", fn, ptrs[k]);
      fflush(stderr);
      abort();
    }
    int &w = width[ext[k].base];
    if (seedWidth[k] > w)
      w = seedWidth[k];
  }

  std::lock_guard<std::mutex> g(shadowLock);
  static bool freeRegistered = false;
  if (!freeRegistered) {
    freeRegistered = true;
    atexit(freeShadows);
  }
  for (int k = 0; k < nptr; ++k) {
    out[k] = nullptr;
    if (!ptrs[k])
      continue;
    Shadow &sb = shadows()[ext[k].base];
    if (!sb.p || sb.bytes != ext[k].bytes) {
      free(sb.p);
      sb.p = calloc(ext[k].bytes ? ext[k].bytes : 1, 1);
      if (!sb.p) {
        fprintf(stderr,
                "Poseidon FPProfiler: cannot allocate a %zu-byte shadow for "
                "%s.\n",
                ext[k].bytes, fn);
        abort();
      }
      sb.bytes = ext[k].bytes;
      sb.seeded.clear();
    }
    if (int w = width[ext[k].base])
      if (sb.seeded.insert(site).second)
        seed(sb.p, sb.bytes, w);
    out[k] = (char *)sb.p + ((uintptr_t)ptrs[k] - ext[k].base);
  }
}
