//===- FPProfileName.h - one profile filename rule, writer and reader -----===//
//
// The FP profiler runtime (FPProfiler.cpp, FPProfilerCUDA.cu) writes and the
// Poseidon pass reads a file named after the profiled function; both derive
// that name by this rule, a pure function of the name:
//
//   stem(n) = n                                      if |n| <= 200
//           = n[0:160] + "~" + hex16(fnv1a64(n))     otherwise
//
// Identity at or below 200 characters keeps every cached profile filename
// where it is; above it the 177-character stem leaves room for the longest
// suffix appended (".wmma.fpprofile") under NAME_MAX = 255. The hash is over
// the full name so instantiations sharing a 160-character prefix stay apart.
//
// Dependency-free on purpose: included by the LLVM pass, a host C++ runtime,
// and a .cu that clang -include's into an application translation unit.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FPPROFILENAME_H
#define POSEIDON_FPPROFILENAME_H

#include <cstddef>
#include <cstdint>
#include <string>

namespace poseidon {

// Names at or below this length are used verbatim.
static const size_t kProfileNameStemThreshold = 200;
// Readable prefix retained when a name is stemmed.
static const size_t kProfileNameStemPrefix = 160;

inline uint64_t profileNameHash(const char *s, size_t n) {
  uint64_t h = 14695981039346656037ULL; // FNV-1a 64 offset basis
  for (size_t i = 0; i < n; ++i) {
    h ^= (uint64_t)(unsigned char)s[i];
    h *= 1099511628211ULL; // FNV-1a 64 prime
  }
  return h;
}

inline std::string profileNameHex16(uint64_t v) {
  static const char *digits = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[(size_t)i] = digits[v & 0xFULL];
    v >>= 4;
  }
  return out;
}

inline std::string profileNameStem(const char *name, size_t n) {
  if (n <= kProfileNameStemThreshold)
    return std::string(name, n);
  return std::string(name, kProfileNameStemPrefix) + "~" +
         profileNameHex16(profileNameHash(name, n));
}

inline std::string profileNameStem(const std::string &name) {
  return profileNameStem(name.data(), name.size());
}

} // namespace poseidon

#endif // POSEIDON_FPPROFILENAME_H
