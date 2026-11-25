//===- FPProfiler.cpp - host half of the FP profiler runtime --------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FPProfileName.h"
#include "FPProfilerCUDA.h"
#include "poseidon/poseidon.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class ProfileInfo : public poseidon::ProfileRecord {
public:
  ProfileInfo()
      : ProfileRecord{std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::lowest(),
                      0.0,
                      0.0,
                      0.0,
                      0,
                      {},
                      {},
                      {}} {}

  void updateValue(double value, const double *operands, size_t numOperands) {
    ++exec;

    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
      minMagOperands.resize(numOperands, 0.0);
    }
    for (size_t i = 0; i < numOperands; ++i) {
      double op = operands[i];
      if (std::isnan(op))
        continue;
      minOperands[i] = std::min(minOperands[i], op);
      maxOperands[i] = std::max(maxOperands[i], op);
      if (op != 0.0)
        minMagOperands[i] = minMagOperands[i] == 0.0
                                ? std::fabs(op)
                                : std::min(minMagOperands[i], std::fabs(op));
    }

    if (!std::isnan(value)) {
      minRes = std::min(minRes, value);
      maxRes = std::max(maxRes, value);
    }
  }

  void updateGradient(double value, double grad) {
    if (!std::isnan(grad) && !std::isnan(value) && !std::isinf(grad) &&
        !std::isinf(value)) {
      sumGrad += grad;
      sumAbsGrad += std::fabs(grad);
      sumSens += std::fabs(grad * value);
    }
  }
};

using namespace poseidon;

// Compile-time static data of each site, registered before main. Built on
// first use, as a namespace-scope map could be constructed after the
// constructor that registers into it runs, and never destroyed, as the exit
// handler that reads it can run after a destructor registered later.
static std::unordered_map<std::string, std::string> &staticHeaders() {
  static auto *m = new std::unordered_map<std::string, std::string>();
  return *m;
}

// The declared quantity of interest, latest value wins. Never destroyed: it is
// first set in main, after the exit handler that reads it was registered.
static std::string &metricName() {
  static auto *n = new std::string();
  return *n;
}
static double g_metricValue = 0.0;
static bool g_metricSet = false;

// A run of the condition-number measurement: its parent reads back only the
// metric, so it records and writes nothing else.
static bool g_metricOnly = false;

// The site the condition-number probe perturbs in this run and by how much;
// eps 0 perturbs nothing.
static int g_probeSite = -1;
static double g_probeEps = 0.0;

// The file names of the profiles this process wrote, the sites of this run
// the condition-number measurement may perturb. Never destroyed: the exit
// handler that reads it is registered before it is first used.
static std::set<std::string> &writtenThisRun() {
  static auto *s = new std::set<std::string>();
  return *s;
}

class FPProfiler {
private:
  std::string functionName;
  // By slot index, the order the records are written in: the profile's bytes
  // key the site's cached DP table.
  std::map<size_t, ProfileInfo> profileInfo;
  static std::string dir;

public:
  FPProfiler(const std::string &funcName) : functionName(funcName) {}

  static void setOutputDir(const std::string &dir_) { dir = dir_; }

  static const std::string &outputDir() { return dir; }

  std::string getOutputPath() const {
    return dir + "/" + profileFileName(functionName);
  }

  void updateValue(size_t idx, double res, size_t numOperands,
                   const double *operands) {
    auto it = profileInfo.try_emplace(idx).first;
    it->second.updateValue(res, operands, numOperands);
  }

  void updateGradient(size_t idx, double value, double grad) {
    auto it = profileInfo.try_emplace(idx).first;
    it->second.updateGradient(value, grad);
  }

  void write() const {
    profileMkdirP(dir);
    std::ostringstream out;
    auto hdr = staticHeaders().find(functionName);
    if (hdr != staticHeaders().end())
      out << hdr->second;
    // Lets profile use tell a site this run never executed from one the
    // profiled program did not contain, which leaves no file at all.
    if (profileInfo.empty())
      out << kProfileNeverExecuted << "\n";

    for (const auto &[slot, info] : profileInfo)
      writeProfileRecord(out, slot, info);

    out << "\n";
    profileWriteFile(getOutputPath(), out.str(), functionName.c_str());
    writtenThisRun().insert(profileFileName(functionName));
  }
};

std::string FPProfiler::dir;

static std::unordered_map<std::string, std::unique_ptr<FPProfiler>>
    profilerRegistry;
static std::mutex registryMutex;

// The profiler of the function funcName, created on its first record. Called
// with registryMutex held.
static FPProfiler &profilerFor(const char *funcName) {
  std::unique_ptr<FPProfiler> &p = profilerRegistry[funcName];
  if (!p)
    p = std::make_unique<FPProfiler>(funcName);
  return *p;
}

// The registered sites whose profile the device profiler of this process
// wrote. Never destroyed: that writer first uses it during exit, and a
// destructor registered then would run before this file's exit handler reads
// it.
static std::set<std::string> &writtenElsewhere() {
  static auto *s = new std::set<std::string>();
  return *s;
}

static void writeMetric() {
  if (!g_metricSet)
    return;
  const std::string dir = FPProfiler::outputDir();
  profileMkdirP(dir);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", g_metricValue);
  profileWriteFile(dir + "/metric.txt", metricName() + " " + buf + "\n",
                   "the declared quantity of interest");
}

// ---------------------------------------------------------------------------
// The condition numbers of the declared quantity of interest
//
// poseidon-probe measures how far the metric moves when one site is computed
// at a relative precision eps, by running the program again (once unperturbed
// for the noise floor, once per site and eps). Each run does what its
// environment says, by POSEIDON_PROBE:
//   unset    a profiling run: marks the profiles of its sites unmeasured, so
//            that profile use stops on them, and says how to measure them;
//            under poseidon-probe (POSEIDON_PROBE_SITES names a file) it marks
//            them and lists them in that file instead;
//   0        declined: leaves the profiles as a run without a metric does;
//   child    a run of the measurement: perturbs site POSEIDON_PROBE_SITE by
//            POSEIDON_PROBE_EPS and records the metric alone.
// ---------------------------------------------------------------------------

static std::vector<std::string> &probeArgv() {
  static std::vector<std::string> a;
  return a;
}

// Captured before main so that a workload which rewrites its own argv (MPI
// launchers do) is named with the arguments it was given.
static void captureProbeArgv() {
  std::ifstream in("/proc/self/cmdline", std::ios::binary);
  if (!in.is_open())
    return;
  std::string all((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
  size_t at = 0;
  while (at < all.size()) {
    size_t end = all.find('\0', at);
    if (end == std::string::npos)
      end = all.size();
    probeArgv().push_back(all.substr(at, end - at));
    at = end + 1;
  }
}

// The command line, quoted for a POSIX shell, that the notice tells the user to
// run again.
static std::string probeCommand() {
  std::string cmd;
  for (const std::string &a : probeArgv()) {
    if (!cmd.empty())
      cmd += ' ';
    if (!a.empty() &&
        a.find_first_not_of("abcdefghijklmnopqrstuvwxyz"
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            "0123456789@%_+=:,./-") == std::string::npos) {
      cmd += a;
      continue;
    }
    cmd += '\'';
    for (char c : a)
      cmd += c == '\'' ? std::string("'\\''") : std::string(1, c);
    cmd += '\'';
  }
  return cmd;
}

// The sites this run wrote and executed, by site id, with their profile paths:
// a profile this run did not write is another program's, and a site this run
// never executed cannot move the metric (perturbing it would cost a run of the
// workload per ladder step).
static std::map<int, std::string> probeSites(const std::string &pdir) {
  // The device profiler's profiles are this run's too.
  for (const std::string &name : writtenElsewhere())
    writtenThisRun().insert(profileFileName(name));
  std::map<int, std::string> sites;
  DIR *d = opendir(pdir.c_str());
  if (!d)
    return sites;
  std::vector<std::string> names;
  while (struct dirent *e = readdir(d))
    names.push_back(e->d_name);
  closedir(d);
  std::sort(names.begin(), names.end());
  const std::string siteIdFormat =
      std::string(" ") + kProfileKeySiteId + " = %d";
  for (const std::string &n : names) {
    if (!writtenThisRun().count(n))
      continue;
    std::ifstream in(pdir + "/" + n);
    std::string line;
    int id = -1;
    bool executed = true;
    while (std::getline(in, line)) {
      if (!line.empty() &&
          line.find_first_not_of("0123456789") == std::string::npos)
        break;
      executed &= line != kProfileNeverExecuted;
      std::sscanf(line.c_str(), siteIdFormat.c_str(), &id);
    }
    if (id >= 0 && executed)
      sites.emplace(id, pdir + "/" + n);
  }
  return sites;
}

// At exit of a profiling run that declared a metric and executed two sites or
// more: what POSEIDON_PROBE asks of it (above).
static void poseidonProbeSites() {
  const char *request = getenv("POSEIDON_PROBE");
  if (g_probeSite >= 0 || profileIsProbeChild() || !g_metricSet)
    return;
  const std::map<int, std::string> sites = probeSites(FPProfiler::outputDir());
  if (sites.size() < 2)
    return;
  if (request && !strcmp(request, "0")) {
    fprintf(stderr,
            "[probe] metric \"%s\" was declared and its condition numbers at "
            "%zu sites are not measured (POSEIDON_PROBE=0): each site is "
            "solved against its own target\n",
            metricName().c_str(), sites.size());
    return;
  }
  // Until a measurement replaces it, so that profile use can tell a declared
  // metric left unmeasured from none.
  for (const auto &kv : sites)
    profileReplaceKappa(kv.second, {std::string(kProfileKeyKappaMark) + " = " +
                                    kProfileKappaUnmeasured});
  if (const char *list = getenv("POSEIDON_PROBE_SITES")) {
    std::string text;
    for (const auto &[site, path] : sites)
      text += std::to_string(site) + " " + path + "\n";
    profileWriteFile(list, text, "the sites to measure");
    return;
  }
  const std::vector<double> ladder = profileProbeLadder();
  fprintf(stderr,
          "[probe] metric \"%s\" was declared but its condition numbers at %zu "
          "sites were not measured, and profile use needs them "
          "(POSEIDON_PROBE=0 on this run solves each site against its own "
          "target instead); measuring takes %zu more runs of this program: "
          "poseidon-probe -- %s\n",
          metricName().c_str(), sites.size(), 1 + sites.size() * ladder.size(),
          probeCommand().c_str());
}

// Whether the profile at `path` holds a record, which only a run that executed
// its site writes: a line that is a slot index.
static bool holdsRecords(const std::string &path) {
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line))
    if (!line.empty() &&
        line.find_first_not_of("0123456789") == std::string::npos)
      return true;
  return false;
}

// A site that logged nothing has no profiler; it is written from its
// registered header alone, unless the directory already holds a profile of
// its executions, which another program sharing the site and the directory
// wrote. The device profiler's exit handler may run before or after this one:
// before, the kernels it wrote are skipped here; after, it replaces what was
// written here for them.
static void writeAllProfiles() {
  if (g_metricOnly) {
    writeMetric();
    return;
  }
  for (auto &pair : profilerRegistry)
    pair.second->write();
  for (const auto &pair : staticHeaders())
    if (!profilerRegistry.count(pair.first) &&
        !writtenElsewhere().count(pair.first)) {
      FPProfiler neverExecuted(pair.first);
      if (!holdsRecords(neverExecuted.getOutputPath()))
        neverExecuted.write();
    }
  writeMetric();
}

// Whether this program was built for profile generation: a site registered
// here, or the device profiler wrote a profile. A profile-use build links this
// runtime for poseidon_metric and has none.
static bool profilesSites() {
  return !staticHeaders().empty() || !profilerRegistry.empty() ||
         !writtenElsewhere().empty();
}

static void writeAllProfilesAtExit() {
  std::lock_guard<std::mutex> lock(registryMutex);
  if (!g_metricOnly && !profilesSites())
    return;
  writeAllProfiles();
  profilerRegistry.clear();
  poseidonProbeSites();
}

static int RegisterFPProfileRuntime() {
  g_metricOnly = profileIsProbeChild();
  g_probeSite = (int)profileEnvInteger("POSEIDON_PROBE_SITE", -1);
  if (g_probeSite >= 0)
    g_probeEps = profileEnvPositiveReal("POSEIDON_PROBE_EPS", 1e-6);
  FPProfiler::setOutputDir(profileDirectory());

  captureProbeArgv();
  std::atexit(writeAllProfilesAtExit);

  return 0;
}

extern "C" {
int POSEIDON_PROFILE_RUNTIME_VAR = RegisterFPProfileRuntime();
}

extern "C" {

void poseidon_metric(const char *name, double value) {
  if (!name)
    return;
  std::lock_guard<std::mutex> lock(registryMutex);
  metricName() = name;
  g_metricValue = value;
  g_metricSet = true;
}

void poseidonRegisterProfileStatic(const char *funcName, const char *text) {
  if (!funcName || !text)
    return;
  std::lock_guard<std::mutex> lock(registryMutex);
  staticHeaders()[funcName] = text;
}

// The condition-number probe's perturbation of a value of the host site
// `site` (FPProfilerCUDA.h poseidonPerturbed); the identity unless this run
// probes that site.
double poseidonProbePerturb(int site, double v) {
  return site == g_probeSite && g_probeEps != 0.0
             ? poseidonPerturbed(v, g_probeEps)
             : v;
}

float poseidonProbePerturbf(int site, float v) {
  return (float)poseidonProbePerturb(site, (double)v);
}

void poseidonProfileWritten(const char *funcName) {
  if (!funcName)
    return;
  std::lock_guard<std::mutex> lock(registryMutex);
  writtenElsewhere().insert(funcName);
}

void poseidonLogGrad(const char *funcName, size_t idx, double value,
                     double grad) {
  if (!funcName || g_metricOnly)
    return;

  std::lock_guard<std::mutex> lock(registryMutex);
  profilerFor(funcName).updateGradient(idx, value, grad);
}

void poseidonLogValue(const char *funcName, size_t idx, double res,
                      size_t numOperands, double *operands) {
  if (!funcName || g_metricOnly)
    return;

  std::lock_guard<std::mutex> lock(registryMutex);
  profilerFor(funcName).updateValue(idx, res, numOperands, operands);
}

} // extern "C"