//===- ShadowTable.h - shadows of a profiled program's allocations --------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The shadow buffers a profiled site's reverse pass takes, one per allocation
// its pointer arguments reach, for host sites (Shadows.cpp) and
// kernels (DeviceProfilerHost.cu) alike.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_PROFILE_SHADOW_TABLE_H
#define POSEIDON_RUNTIME_PROFILE_SHADOW_TABLE_H

#include "DeviceProfiler.h"
#include "ProfileFormat.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>

namespace poseidon {

// The adjoint seed of element `i` of an output allocation: a sign and a
// magnitude in [0.5, 1.5) hashed from `i`. A value's profiled gradient is that
// of the seeded combination of the outputs, and a seed of ones asks for the
// derivative of their sum, which is zero for a value whose outputs sum to a
// constant: a flux added to one element and subtracted from another, or
// anything a derivative of a partition-of-unity basis distributes. No fixed
// combination cancels this seed, it is the same in every run, and its
// magnitude averages to one. splitmix64's finalizer: the sign is the low bit,
// the magnitude the top 53.
POSEIDON_HD inline double outputSeed(uint64_t i) {
  uint64_t h = i + 0x9e3779b97f4a7c15ULL;
  h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
  h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
  h ^= h >> 31;
  double mag = 0.5 + (double)(h >> 11) * 0x1.0p-53;
  return (h & 1) ? -mag : mag;
}

// Keyed by the base address of the allocation: the adjoint is a property of
// the buffer, so every site and call that passes a pointer into it must see
// the same shadow at the same offset. A shadow starts zeroed and accumulates
// for the rest of the run: a per-call reseed would report a different gradient
// for a site that reads its own output. Memory is Ops', each operation ordered
// on `stream`, an opaque handle of the launch that takes the shadow:
//   void *allocZeroed(size_t bytes, const char *site, void *stream);
//   void free(void *p);         // a shadow being replaced
//   void freeAtExit(void *p);
//   void fillSeed<T>(T *p, size_t n, void *stream); // p[i] = outputSeed(i)
template <class Ops> class ShadowTable {
public:
  // The table of the process, never destroyed: its buffers are freed at exit
  // by a handler registered on its first use, after which nothing reads it.
  static ShadowTable &get() {
    static ShadowTable *table = [] {
      std::atexit([] { get().freeAll(); });
      return new ShadowTable();
    }();
    return *table;
  }

  // A pointer of a call: the allocation [base, base + bytes) it points into,
  // base 0 for a null pointer, its offset there, and the element size of the
  // output the call writes through it, 0 for none.
  struct Pointer {
    uintptr_t base = 0;
    size_t bytes = 0, offset = 0;
    int seedWidth = 0;
  };

  // Writes to `out` the shadow of each of the `n` pointers `ptrs` of one call
  // of `site`, null for a null pointer. An allocation the call writes through
  // any of its pointers is an output of the call. Calls from several threads
  // share the table.
  void shadowsOf(const Pointer *ptrs, size_t n, int site, const char *siteName,
                 void *stream, void **out) {
    std::map<uintptr_t, int> width;
    for (size_t k = 0; k < n; ++k)
      if (ptrs[k].base) {
        int &w = width[ptrs[k].base];
        w = std::max(w, ptrs[k].seedWidth);
      }
    std::lock_guard<std::mutex> g(lock);
    for (size_t k = 0; k < n; ++k)
      out[k] = ptrs[k].base
                   ? (char *)shadowOf(ptrs[k].base, ptrs[k].bytes, site,
                                      width[ptrs[k].base], siteName, stream) +
                         ptrs[k].offset
                   : nullptr;
  }

private:
  // The shadow of the allocation [base, base + bytes). A base seen with a
  // different extent is a reused address, not the same buffer: the previous
  // allocation was freed and its shadow is stale. When `width` is the element
  // size of an output of `site`, the shadow carries the seed for that site
  // once: a site's reverse pass runs right after it, in program order,
  // so the first site that writes a buffer consumes its adjoint and leaves
  // zero behind for the sites that write it later.
  void *shadowOf(uintptr_t base, size_t bytes, int site, int width,
                 const char *siteName, void *stream) {
    Shadow &sb = shadows[base];
    if (!sb.p || sb.bytes != bytes) {
      if (sb.p)
        Ops::free(sb.p);
      sb.p = Ops::allocZeroed(bytes, siteName, stream);
      sb.bytes = bytes;
      sb.seeded.clear();
    }
    if (width && sb.seeded.insert(site).second)
      seed(sb.p, bytes, width, stream);
    return sb.p;
  }

  struct Shadow {
    void *p = nullptr;
    size_t bytes = 0;
    std::set<int> seeded;
  };

  static void seed(void *p, size_t bytes, int width, void *stream) {
    if (width == 8)
      Ops::fillSeed((double *)p, bytes / sizeof(double), stream);
    else if (width == 4)
      Ops::fillSeed((float *)p, bytes / sizeof(float), stream);
    else
      fatal("no seed rule for a %d-byte output element.", width);
  }

  void freeAll() {
    for (auto &kv : shadows)
      if (kv.second.p)
        Ops::freeAtExit(kv.second.p);
    shadows.clear();
  }

  std::map<uintptr_t, Shadow> shadows;
  std::mutex lock;
};

} // namespace poseidon

#endif // POSEIDON_RUNTIME_PROFILE_SHADOW_TABLE_H
