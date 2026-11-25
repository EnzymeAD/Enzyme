//===- FPProfileName.h - the profile files the runtime writes -------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The FP profiler runtime (FPProfiler.cpp, FPProfilerCUDA.cu) writes and the
// Poseidon pass reads a file named after the profiled function; both derive
// that name by this rule, a pure function of the name:
//
//   stem(n) = n                                      if |n| <= 200
//           = n[0:160] + "~" + hex16(fnv1a64(n))     otherwise
//
// The 177-character stem leaves room for the longest name built from it,
// "launches_" + stem + "_" + 16 hex digits + ".json", under NAME_MAX = 255.
// The hash is over the full name so instantiations sharing a 160-character
// prefix stay apart.
//
// Also the header keys both sides spell, and the file handling both runtime
// halves share. Free of LLVM: included by the pass and the runtimes.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_FPPROFILENAME_H
#define POSEIDON_FPPROFILENAME_H

#include "poseidon/poseidon.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

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

inline constexpr char kProfileFileSuffix[] = ".fpprofile";

// The file the profile of the function `name` is written to.
inline std::string profileFileName(const std::string &name) {
  return profileNameStem(name) + kProfileFileSuffix;
}

// Header lines: "<key> = <value>".
inline constexpr char kProfileKeySiteId[] = "SiteId";
inline constexpr char kProfileKeyCanonicalHash[] = "CanonicalHash";
inline constexpr char kProfileKeyKappa[] = "Kappa";
inline constexpr char kProfileKeyKappaMark[] = "KappaMark";
// The KappaMark of a run that declared a metric and did not measure it.
inline constexpr char kProfileKappaUnmeasured[] = "unmeasured";
// The whole header of a site the run instrumented and never executed.
inline constexpr char kProfileNeverExecuted[] = "Executed = 0";

// The directory a profiling run writes to.
inline std::string profileDirectory() {
  const char *dir = getenv("POSEIDON_PROFILE_DIR");
  return dir ? dir : POSEIDON_DEFAULT_PROFILE_DIR;
}

// Whether this process is a run of the condition-number measurement, which
// records the metric and nothing else.
inline bool profileIsProbeChild() {
  const char *request = getenv("POSEIDON_PROBE");
  return request && !strcmp(request, "child");
}

// `mkdir -p`: a missing intermediate component must not surface as an ignored
// "could not open" warning.
inline void profileMkdirP(const std::string &dir) {
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
      struct stat st = {};
      if (stat(acc.c_str(), &st) == -1)
        mkdir(acc.c_str(), 0755);
      acc += "/";
    }
    i = j + 1;
  }
}

// A profile that cannot be written is a profile the solve will not find, so
// the run stops. Called with errno from the failed open.
[[noreturn]] inline void profileOpenFatal(const std::string &path,
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

} // namespace poseidon

#endif // POSEIDON_FPPROFILENAME_H
