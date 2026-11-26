//===- ProfileFormat.h - the profile files the runtime writes -------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The FP profiler runtime (Profiler.cpp, DeviceProfilerHost.cu) writes the
// profile of a site to <name>.fpprofile, where <name> is the string the
// compile gave the site's probes (SiteIdentity.h, bodyProfileStem).
//
// Also the format both sides spell, the file handling the runtime halves and
// poseidon-kappa share, and how every runtime reports an error. Free of LLVM:
// included by the pass, the runtimes and poseidon-kappa.
//
// A profile is a header, lines "<key> = <value>", then one record per profiled
// instruction executed: its slot index alone on a line, then, each on a line
// of its own and in this order, MinRes, MaxRes, SumSens, SumGrad, SumAbsGrad,
// SumSqGrad and SumSqSens (the sums of gradient^2 and of (gradient x value)^2,
// both or neither, in every record of a profile or in none), Exec,
// NumOperands and Operand[i] = [minimum, maximum, smallest nonzero magnitude
// or 0] for each operand.
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_RUNTIME_PROFILE_PROFILE_FORMAT_H
#define POSEIDON_RUNTIME_PROFILE_PROFILE_FORMAT_H

#include "poseidon/poseidon.h"

#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace poseidon {

//===----------------------------------------------------------------------===//
// Errors
//===----------------------------------------------------------------------===//

// The message after "[poseidon] ", on stderr after what the program wrote to
// stdout.
inline void vreport(const char *format, va_list args) {
  fflush(stdout);
  fputs("[poseidon] ", stderr);
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
  fflush(stderr);
}

__attribute__((format(printf, 1, 2))) inline void report(const char *format,
                                                         ...) {
  va_list args;
  va_start(args, format);
  vreport(format, args);
  va_end(args);
}

// An error that leaves the program's results wrong stops it with the message,
// by abort.
[[noreturn]] __attribute__((format(printf, 1, 2))) inline void
fatal(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vreport(format, args);
  va_end(args);
  abort();
}

// The call `call` returned `status`; the message names the function called,
// without its arguments.
[[noreturn]] inline void fatalCall(const char *call, const char *status) {
  fatal("%.*s failed: %s", (int)strcspn(call, "("), call, status);
}

// A run whose profile would be wrong or lost stops with the message and exit
// status 1, without running exit handlers: it may already be in one.
[[noreturn]] __attribute__((format(printf, 1, 2))) inline void
profileFail(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vreport(format, args);
  va_end(args);
  _exit(1);
}

//===----------------------------------------------------------------------===//
// Files
//===----------------------------------------------------------------------===//

inline constexpr char kProfileFileSuffix[] = ".fpprofile";

// The file the profile of the site the compile named `name` is written to.
inline std::string profileFileName(const std::string &name) {
  return name + kProfileFileSuffix;
}

// Header keys.
inline constexpr char kProfileKeySiteId[] = "SiteId";
inline constexpr char kProfileKeyCanonicalHash[] = "CanonicalHash";
inline constexpr char kProfileKeyKappa[] = "Kappa";
inline constexpr char kProfileKeyKappaMark[] = "KappaMark";
inline constexpr char kProfileKeyExecuted[] = "Executed";
// The words of a KappaMark line, which poseidon-kappa writes and the pass
// reads: the condition number was measured (ok), changed less than the noise
// of the runs (below-noise), held at no two neighbouring perturbations
// (nonlinear), or did not finish at the smallest one (diverged); or the run
// declared a metric and did not measure it (unmeasured).
inline constexpr char kProfileKappaOk[] = "ok";
inline constexpr char kProfileKappaBelowNoise[] = "below-noise";
inline constexpr char kProfileKappaNonlinear[] = "nonlinear";
inline constexpr char kProfileKappaDiverged[] = "diverged";
inline constexpr char kProfileKappaUnmeasured[] = "unmeasured";
// The whole header of a site the run instrumented and never executed.
inline constexpr char kProfileNeverExecuted[] = "Executed = 0";

// What a profile records about one instruction.
struct ProfileRecord {
  double minRes, maxRes, sumSens, sumGrad, sumAbsGrad, sumSqGrad, sumSqSens;
  uint64_t exec;
  std::vector<double> minOperands, maxOperands, minMagOperands;
};

// Whether `line` of a profile starts a record: the slot index alone.
inline bool profileIsSlotLine(const std::string &line) {
  return !line.empty() &&
         line.find_first_not_of("0123456789") == std::string::npos;
}

// Writes the record of the instruction at `slot`.
inline void writeProfileRecord(std::ostream &out, size_t slot,
                               const ProfileRecord &r) {
  out << std::scientific
      << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << slot << "\n";
  out << "\tMinRes = " << r.minRes << "\n";
  out << "\tMaxRes = " << r.maxRes << "\n";
  out << "\tSumSens = " << r.sumSens << "\n";
  out << "\tSumGrad = " << r.sumGrad << "\n";
  out << "\tSumAbsGrad = " << r.sumAbsGrad << "\n";
  out << "\tSumSqGrad = " << r.sumSqGrad << "\n";
  out << "\tSumSqSens = " << r.sumSqSens << "\n";
  out << "\tExec = " << r.exec << "\n";
  out << "\tNumOperands = " << r.minOperands.size() << "\n";
  for (size_t i = 0; i < r.minOperands.size(); ++i)
    out << "\tOperand[" << i << "] = [" << r.minOperands[i] << ", "
        << r.maxOperands[i] << ", " << r.minMagOperands[i] << "]\n";
}

// The directory -poseidon-profile-generate=<dir> named, defined by each unit
// it compiled; weak, as a program may have none.
extern "C" const char __poseidon_profile_dir[] __attribute__((weak));

// The directory a profiling run writes to: POSEIDON_PROFILE_DIR, else the one
// the program was built with, else the default.
inline std::string profileDirectory() {
  if (const char *dir = getenv("POSEIDON_PROFILE_DIR"))
    return dir;
  if (__poseidon_profile_dir)
    return __poseidon_profile_dir;
  return POSEIDON_DEFAULT_PROFILE_DIR;
}

// Whether this process is a run of the condition-number measurement, which
// records the metric and nothing else.
inline bool profileIsKappaChild() {
  const char *request = getenv("POSEIDON_KAPPA");
  return request && !strcmp(request, "child");
}

// Creates each missing component of `dir`; a component that cannot be
// created fails the open of a file in it.
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
  profileFail("could not open profile file for writing:\n"
              "  %s\n"
              "  (%s; path component is %zu bytes)\n"
              "  function: %s\n"
              "The profile is LOST, and a solve run against this directory "
              "would silently skip the site. Aborting.",
              path.c_str(), strerror(errno),
              path.size() - path.find_last_of('/') - 1, funcName);
}

// Writes `text` to `path` whole or not at all: through a file of this
// process that replaces `path` by rename, so that processes writing the same
// profile concurrently (the ranks of an MPI job) leave one of their files, not
// an interleaving of several. A file that cannot be written stops the run.
inline void profileWriteFile(const std::string &path, const std::string &text,
                             const char *what) {
  const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
  errno = 0;
  FILE *f = fopen(tmp.c_str(), "w");
  if (!f)
    profileOpenFatal(tmp, what);
  const bool written = fwrite(text.data(), 1, text.size(), f) == text.size();
  if (fclose(f) != 0 || !written || rename(tmp.c_str(), path.c_str()) != 0)
    profileOpenFatal(path, what);
}

// The environment variable `name` as a strict decimal integer; `dflt` when it
// is unset or empty. Anything else stops the run, naming the variable.
[[noreturn]] inline void profileBadEnv(const char *name, const char *value,
                                       const char *expected) {
  profileFail("%s='%s' is not %s.", name, value, expected);
}

inline long long profileEnvInteger(const char *name, long long dflt) {
  const char *v = getenv(name);
  if (!v || !*v)
    return dflt;
  char *end = nullptr;
  errno = 0;
  long long parsed = strtoll(v, &end, 10);
  if (end == v || *end != '\0' || errno == ERANGE)
    profileBadEnv(name, v, "an integer");
  return parsed;
}

// The environment variable `name` as a positive integer, as above.
inline unsigned long long profileEnvPositive(const char *name,
                                             unsigned long long dflt) {
  long long parsed = profileEnvInteger(name, (long long)dflt);
  if (parsed <= 0)
    profileBadEnv(name, getenv(name), "a positive integer");
  return (unsigned long long)parsed;
}

// Replaces the Kappa lines of the profile at `path` with `lines`, kept in the
// header block the profile-use compile reads before the first slot.
inline void profileReplaceKappa(const std::string &path,
                                const std::vector<std::string> &lines) {
  std::vector<std::string> kept;
  {
    std::ifstream in(path);
    std::string l;
    while (std::getline(in, l))
      if (l.compare(0, strlen(kProfileKeyKappa), kProfileKeyKappa) != 0)
        kept.push_back(l);
  }
  size_t at = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (profileIsSlotLine(kept[i]))
      break;
    at = i + 1;
    if (kept[i].compare(0, strlen(kProfileKeyCanonicalHash),
                        kProfileKeyCanonicalHash) == 0)
      break;
  }
  kept.insert(kept.begin() + at, lines.begin(), lines.end());
  std::string text;
  for (const std::string &l : kept)
    text += l + "\n";
  profileWriteFile(path, text, "the condition number");
}

// A positive finite number.
inline bool profileParsePositive(const char *text, const char *stop,
                                 double &out) {
  char *end = nullptr;
  out = strtod(text, &end);
  return end == stop && end != text && out > 0.0 && out < HUGE_VAL;
}

inline double profileEnvPositiveReal(const char *name, double dflt) {
  const char *v = getenv(name);
  if (!v || !*v)
    return dflt;
  double out;
  if (!profileParsePositive(v, v + strlen(v), out))
    profileBadEnv(name, v, "a positive finite number");
  return out;
}

// A comma-separated list of positive finite numbers.
inline std::vector<double> profileEnvPositiveList(const char *name,
                                                  std::vector<double> dflt) {
  const char *v = getenv(name);
  if (!v || !*v)
    return dflt;
  std::vector<double> out;
  for (const char *p = v;;) {
    const char *comma = strchr(p, ',');
    const char *stop = comma ? comma : p + strlen(p);
    double x;
    if (!profileParsePositive(p, stop, x))
      profileBadEnv(name, v, "a comma-separated list of positive numbers");
    out.push_back(x);
    if (!comma)
      return out;
    p = comma + 1;
  }
}

// The relative perturbations the condition-number measurement runs each site
// at, POSEIDON_KAPPA_LADDER.
inline std::vector<double> profileKappaLadder() {
  return profileEnvPositiveList("POSEIDON_KAPPA_LADDER", {1e-8, 1e-6, 1e-4});
}

// The site a run of the condition-number measurement perturbs,
// POSEIDON_KAPPA_SITE, and the relative perturbation, POSEIDON_KAPPA_EPS, which
// a site requires. Site -1 and eps 0 perturb nothing, as in every other run.
struct KappaPerturbation {
  int site = -1;
  double eps = 0.0;
};

inline KappaPerturbation profileKappaPerturbation() {
  if (!profileIsKappaChild())
    return {};
  KappaPerturbation p;
  p.site = (int)profileEnvInteger("POSEIDON_KAPPA_SITE", -1);
  if (p.site < 0)
    return {};
  const char *eps = getenv("POSEIDON_KAPPA_EPS");
  if (!eps || !*eps)
    profileFail("POSEIDON_KAPPA_SITE=%d is set without POSEIDON_KAPPA_EPS.",
                p.site);
  p.eps = profileEnvPositiveReal("POSEIDON_KAPPA_EPS", 0.0);
  return p;
}

// The declared metric a profiling run writes into its profile directory, one
// line "<name> <value>", and poseidon-kappa reads back. The value is the last
// field: the name may hold spaces.
inline constexpr char kProfileMetricFile[] = "metric.txt";

inline void profileWriteMetric(const std::string &dir, const std::string &name,
                               double value) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%.17g", value);
  profileWriteFile(dir + "/" + kProfileMetricFile, name + " " + buf + "\n",
                   "the declared quantity of interest");
}

// False when `dir` holds no metric or one that does not parse.
inline bool profileReadMetric(const std::string &dir, std::string &name,
                              double &value) {
  std::ifstream in(dir + "/" + kProfileMetricFile);
  std::string line;
  if (!in.is_open() || !std::getline(in, line))
    return false;
  size_t at = line.rfind(' ');
  if (at == std::string::npos)
    return false;
  const char *field = line.c_str() + at + 1;
  char *end = nullptr;
  value = strtod(field, &end);
  name = line.substr(0, at);
  return end != field && *end == '\0';
}

// The sites a profiling run lists for poseidon-kappa in the file
// POSEIDON_KAPPA_SITES names: the directory the run wrote its profiles and
// metric to on the first line, then "<site id> <profile path>" per site.
struct KappaSiteList {
  std::string dir;
  std::map<int, std::string> sites;
};

inline void profileWriteKappaSites(const std::string &path,
                                   const KappaSiteList &list) {
  std::string text = list.dir + "\n";
  for (const auto &[site, profile] : list.sites)
    text += std::to_string(site) + " " + profile + "\n";
  profileWriteFile(path, text, "the sites to measure");
}

// False when the run wrote no list to `path`.
inline bool profileReadKappaSites(const std::string &path,
                                  KappaSiteList &list) {
  std::ifstream in(path);
  if (!in.is_open() || !std::getline(in, list.dir))
    return false;
  int site;
  std::string profile;
  while (in >> site && std::getline(in >> std::ws, profile))
    list.sites.emplace(site, profile);
  return true;
}

} // namespace poseidon

#endif // POSEIDON_RUNTIME_PROFILE_PROFILE_FORMAT_H
