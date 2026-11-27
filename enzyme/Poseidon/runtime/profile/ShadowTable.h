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

#include "ProfileFormat.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <set>

namespace poseidon {

// Keyed by the base address of the allocation: the adjoint is a property of
// the buffer, so every site and call that passes a pointer into it must see
// the same shadow at the same offset. A shadow starts zeroed and accumulates
// for the rest of the run: a per-call reseed would report a different gradient
// for a site that reads its own output. Memory is Ops', each operation ordered
// on `stream`, an opaque handle of the launch that takes the shadow:
//   void *allocZeroed(size_t bytes, const char *site, void *stream);
//   void free(void *p);         // a shadow being replaced
//   void freeAtExit(void *p);
//   void fillOnes<T>(T *p, size_t n, void *stream);
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

  // The shadow of the allocation [base, base + bytes). A base seen with a
  // different extent is a reused address, not the same buffer: the previous
  // allocation was freed and its shadow is stale. When `width` is the element
  // size of an output of `site`, the shadow carries the seed one for that
  // site once: a site's reverse pass runs right after it, in program order,
  // so the first site that writes a buffer consumes its adjoint and leaves
  // zero behind for the sites that write it later.
  void *shadowOf(uintptr_t base, size_t bytes, int site, int width,
                 const char *siteName, void *stream = nullptr) {
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

private:
  struct Shadow {
    void *p = nullptr;
    size_t bytes = 0;
    std::set<int> seeded;
  };

  static void seed(void *p, size_t bytes, int width, void *stream) {
    if (width == 8)
      Ops::fillOnes((double *)p, bytes / sizeof(double), stream);
    else if (width == 4)
      Ops::fillOnes((float *)p, bytes / sizeof(float), stream);
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
};

} // namespace poseidon

#endif // POSEIDON_RUNTIME_PROFILE_SHADOW_TABLE_H
