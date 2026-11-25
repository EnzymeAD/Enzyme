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
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

extern char **environ;

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

// When the runtime was registered, before main: the start of the profile run.
static std::chrono::steady_clock::time_point g_runtimeStart;

// The file names of the profiles this process wrote, the sites of this run
// the condition-number probe may measure. Never destroyed: the exit handler
// that reads it is registered before it is first used.
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
// Per-site condition number of the declared quantity of interest
//
// How far the metric moves when one site is computed at a relative precision
// eps. With POSEIDON_PROBE=1 a profiling run re-executes itself once
// unperturbed for the noise floor and once per (site, eps), and writes the
// result into the profiles it just wrote; without, it marks them unmeasured;
// with POSEIDON_PROBE=0 it leaves them as a run without a metric writes them. A
// child has POSEIDON_PROBE=child, never re-executes and records nothing but the
// metric.
// ---------------------------------------------------------------------------

static std::vector<std::string> &probeArgv() {
  static std::vector<std::string> a;
  return a;
}

// Captured before main so that a workload which rewrites its own argv (MPI
// launchers do) is still re-run with the arguments it was given.
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

namespace {
struct ProbeRun {
  double eps;
  double kappa;
  bool diverged;
};
} // namespace

// The value is the last field of the line: the name may hold spaces.
static bool probeReadMetric(const std::string &dir, double &value) {
  std::ifstream in(dir + "/metric.txt");
  std::string line;
  if (!in.is_open() || !std::getline(in, line))
    return false;
  size_t at = line.rfind(' ');
  if (at == std::string::npos)
    return false;
  const char *field = line.c_str() + at + 1;
  char *end = nullptr;
  value = std::strtod(field, &end);
  return end != field && *end == '\0';
}

// Run the workload again with `extra` added to the environment, its output in
// <dir>/run.log. Returns the exit status, -1 if the child could not be started.
static int probeRun(const std::string &dir,
                    const std::vector<std::string> &extra) {
  std::vector<std::string> keys;
  for (const std::string &kv : extra)
    keys.push_back(kv.substr(0, kv.find('=') + 1));
  std::vector<std::string> env;
  for (char **e = environ; *e; ++e) {
    bool shadowed = false;
    for (const std::string &k : keys)
      shadowed |= strncmp(*e, k.c_str(), k.size()) == 0;
    if (!shadowed)
      env.push_back(*e);
  }
  for (const std::string &kv : extra)
    env.push_back(kv);
  std::vector<char *> envp;
  for (std::string &kv : env)
    envp.push_back(&kv[0]);
  envp.push_back(nullptr);

  std::vector<char *> argv;
  for (std::string &a : probeArgv())
    argv.push_back(&a[0]);
  argv.push_back(nullptr);

  const std::string log = dir + "/run.log";
  fflush(nullptr);
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd >= 0) {
      dup2(fd, 1);
      dup2(fd, 2);
      close(fd);
    }
    execve("/proc/self/exe", argv.data(), envp.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

// Two slopes within this factor of each other agree.
constexpr double kProbeSlopeAgreement = 2.0;

// A slope that holds across two neighbouring eps is a measured derivative; a
// ladder with no such pair is marked nonlinear; an eps that diverges above a
// clean one says where the site breaks without discarding the clean
// measurement.
static void probeDecide(const std::vector<ProbeRun> &ladder, double &kappa,
                        std::string &mark, bool &hasDiverge) {
  hasDiverge = false;
  for (const ProbeRun &r : ladder)
    hasDiverge |= r.diverged;
  if (ladder.empty() || ladder.front().diverged) {
    kappa = std::numeric_limits<double>::infinity();
    mark = "diverged";
    hasDiverge = false;
    return;
  }
  for (size_t i = 0; i + 1 < ladder.size(); ++i) {
    if (ladder[i].diverged || ladder[i + 1].diverged)
      continue;
    double lo = std::fabs(ladder[i].kappa), hi = std::fabs(ladder[i + 1].kappa);
    if (lo > hi)
      std::swap(lo, hi);
    if (hi == 0.0 || (lo > 0.0 && hi / lo <= kProbeSlopeAgreement)) {
      kappa = ladder[i].kappa;
      mark = "ok";
      return;
    }
  }
  for (const ProbeRun &r : ladder)
    if (!r.diverged) {
      kappa = r.kappa;
      break;
    }
  mark = hasDiverge ? "ok" : "nonlinear";
}

// Replace the site's Kappa lines with `hdr`, keeping them in the header block
// the profile-use compile reads before the first slot.
static void probeReplaceKappa(const std::string &path,
                              const std::vector<std::string> &hdr) {
  std::vector<std::string> lines;
  {
    std::ifstream in(path);
    std::string l;
    while (std::getline(in, l))
      lines.push_back(l);
  }
  std::vector<std::string> kept;
  for (const std::string &l : lines)
    if (l.compare(0, strlen(kProfileKeyKappa), kProfileKeyKappa) != 0)
      kept.push_back(l);
  size_t at = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (!kept[i].empty() &&
        kept[i].find_first_not_of("0123456789") == std::string::npos)
      break;
    at = i + 1;
    if (kept[i].compare(0, strlen(kProfileKeyCanonicalHash),
                        kProfileKeyCanonicalHash) == 0)
      break;
  }
  kept.insert(kept.begin() + at, hdr.begin(), hdr.end());
  std::string text;
  for (const std::string &l : kept)
    text += l + "\n";
  profileWriteFile(path, text, "the condition number");
}

static void probeWriteKappa(const std::string &path, double kappa,
                            const std::string &mark) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%s = %.17g", kProfileKeyKappa, kappa);
  probeReplaceKappa(path,
                    {buf, std::string(kProfileKeyKappaMark) + " = " + mark});
}

static void probeRemoveDir(const std::string &dir) {
  if (DIR *d = opendir(dir.c_str())) {
    while (struct dirent *e = readdir(d))
      if (strcmp(e->d_name, ".") && strcmp(e->d_name, ".."))
        unlink((dir + "/" + e->d_name).c_str());
    closedir(d);
  }
  rmdir(dir.c_str());
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

// The scratch directory of the run of `site` at `eps` under `tmp`.
static std::string probeRunDir(const char *tmp, int site, double eps) {
  char dir[64];
  std::snprintf(dir, sizeof(dir), "%s/site%d_eps%g", tmp, site, eps);
  return dir;
}

static std::vector<std::string> probeChildEnv(const std::string &dir) {
  return {"POSEIDON_PROBE=child", "POSEIDON_PROFILE_DIR=" + dir};
}

// A response this many times the noise floor is the site's, not noise.
constexpr double kProbeAboveNoise = 4.0;

// The ladder of runs of `site` at each eps, and whether any response rose
// above the noise floor; the metric of the unperturbed run is m0.
static std::vector<ProbeRun> probeSite(const char *tmp, int site,
                                       const std::vector<double> &ladder,
                                       double m0, double noise,
                                       bool &responded) {
  std::vector<ProbeRun> runs;
  responded = false;
  for (double eps : ladder) {
    const std::string d = probeRunDir(tmp, site, eps);
    profileMkdirP(d);
    std::vector<std::string> env = probeChildEnv(d);
    env.push_back("POSEIDON_PROBE_SITE=" + std::to_string(site));
    char epsVar[64];
    std::snprintf(epsVar, sizeof(epsVar), "POSEIDON_PROBE_EPS=%.17g", eps);
    env.push_back(epsVar);
    const int rc = probeRun(d, env);
    double mk = 0.0;
    if (rc != 0 || !probeReadMetric(d, mk) || !std::isfinite(mk)) {
      fprintf(stderr, "    eps=%-8g DIVERGED (exit %d)\n", eps, rc);
      runs.push_back({eps, 0.0, true});
      continue;
    }
    const double resp = std::fabs(mk - m0);
    const double kappa = m0 != 0.0 ? resp / (eps * std::fabs(m0)) : resp / eps;
    const bool over = resp > kProbeAboveNoise * noise;
    fprintf(stderr,
            "    eps=%-8g metric=%-24.17g response=%-12.4e kappa=%-12.4e %s\n",
            eps, mk, resp, kappa, over ? "above noise" : "within noise");
    runs.push_back({eps, kappa, false});
    responded |= over;
  }
  return runs;
}

// Whether this run goes on to measure the condition numbers of `sites`; a
// run that does not says why and how, and leaves them marked unmeasured
// unless POSEIDON_PROBE=0 declined them.
static bool probeRequested(const char *request,
                           const std::map<int, std::string> &sites,
                           size_t runs) {
  if (request && !strcmp(request, "0")) {
    fprintf(stderr,
            "[probe] metric \"%s\" was declared and its condition numbers at "
            "%zu sites are not measured (POSEIDON_PROBE=0): each site is "
            "solved against its own target\n",
            metricName().c_str(), sites.size());
    return false;
  }
  // Until a measurement replaces it, so that profile use can tell a declared
  // metric left unmeasured from none.
  for (const auto &kv : sites)
    probeReplaceKappa(kv.second, {std::string(kProfileKeyKappaMark) + " = " +
                                  kProfileKappaUnmeasured});
  if (!request || strcmp(request, "1")) {
    fprintf(stderr,
            "[probe] metric \"%s\" was declared but its condition numbers at "
            "%zu sites were not measured, and profile use needs them "
            "(POSEIDON_PROBE=0 on this run solves each site against its own "
            "target instead); measuring takes %zu more runs of this program: "
            "poseidon-probe -- %s\n",
            metricName().c_str(), sites.size(), runs, probeCommand().c_str());
    return false;
  }
  if (probeArgv().empty()) {
    fprintf(stderr, "[probe] cannot read this program's command line from "
                    "/proc/self/cmdline; not measuring condition numbers\n");
    return false;
  }
  return true;
}

// At exit of a profiling run that declared a metric: the condition number of
// the metric at each site this run executed, measured by running the program
// again (probeRequested says when), written into the site's profile.
static void poseidonProbeDriver() {
  const std::chrono::duration<double> profileRun =
      std::chrono::steady_clock::now() - g_runtimeStart;
  // A run the driver started, or one armed by hand, is a measurement.
  const char *request = getenv("POSEIDON_PROBE");
  if (g_probeSite >= 0 || profileIsProbeChild() || !g_metricSet)
    return;
  const std::map<int, std::string> sites = probeSites(FPProfiler::outputDir());
  if (sites.size() < 2)
    return;
  const std::vector<double> ladder =
      profileEnvPositiveList("POSEIDON_PROBE_LADDER", {1e-8, 1e-6, 1e-4});
  const size_t runs = 1 + sites.size() * ladder.size();
  if (!probeRequested(request, sites, runs))
    return;
  fprintf(
      stderr,
      "[probe] measuring the condition numbers of metric \"%s\" at %zu sites: "
      "%zu more runs of this program (run it without POSEIDON_PROBE=1 to "
      "skip)\n",
      metricName().c_str(), sites.size(), runs);

  char tmpl[] = "/tmp/poseidon_probe_XXXXXX";
  const char *tmp = mkdtemp(tmpl);
  if (!tmp) {
    fprintf(stderr, "[probe] cannot create a scratch directory; skipping\n");
    return;
  }
  const double m0 = g_metricValue;
  const auto measureStart = std::chrono::steady_clock::now();
  const std::string base2 = std::string(tmp) + "/baseline2";
  profileMkdirP(base2);
  fprintf(stderr, "[probe] baseline run 2 -> noise floor\n");
  double m2 = 0.0;
  if (probeRun(base2, probeChildEnv(base2)) != 0 ||
      !probeReadMetric(base2, m2)) {
    fprintf(stderr,
            "[probe] the second baseline run failed (see %s/run.log); not "
            "measuring condition numbers\n",
            base2.c_str());
    return;
  }
  const double noise = std::fabs(m2 - m0);
  fprintf(stderr, "[probe] metric %s = %.17g and %.17g; noise floor %.6e\n",
          metricName().c_str(), m0, m2, noise);

  std::map<int, std::pair<double, std::string>> result;
  for (const auto &[site, path] : sites) {
    fprintf(stderr, "[probe] site %d (%s)\n", site, path.c_str());
    bool responded = false;
    const std::vector<ProbeRun> siteRuns =
        probeSite(tmp, site, ladder, m0, noise, responded);
    double kappa = 0.0;
    std::string mark = "below-noise";
    bool hasDiverge = false;
    if (responded)
      probeDecide(siteRuns, kappa, mark, hasDiverge);
    result[site] = {kappa, mark};
    fprintf(stderr, "    -> Kappa = %.6e  (%s)%s\n", kappa, mark.c_str(),
            hasDiverge ? "  diverges above" : "");
  }

  fprintf(stderr, "[probe] site  kappa            mark          profile\n");
  for (const auto &[site, path] : sites) {
    const auto &[kappa, mark] = result[site];
    probeWriteKappa(path, kappa, mark);
    fprintf(stderr, "[probe] %-5d %-16.6e %-13s %s\n", site, kappa,
            mark.c_str(), path.c_str());
  }
  const std::chrono::duration<double> measured =
      std::chrono::steady_clock::now() - measureStart;
  fprintf(stderr, "[probe] profile run: %.2f s\n", profileRun.count());
  fprintf(stderr, "[probe] condition-number measurement: %zu runs, %.2f s\n",
          runs, measured.count());

  probeRemoveDir(base2);
  for (const auto &kv : sites)
    for (double eps : ladder)
      probeRemoveDir(probeRunDir(tmp, kv.first, eps));
  rmdir(tmp);
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
  poseidonProbeDriver();
}

static int RegisterFPProfileRuntime() {
  g_runtimeStart = std::chrono::steady_clock::now();
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

// Whether this run, at exit, measures the condition numbers by running the
// program again; the device profiler then releases the device for those runs.
bool poseidonProbeWillMeasure() {
  const char *request = getenv("POSEIDON_PROBE");
  std::lock_guard<std::mutex> lock(registryMutex);
  return !g_metricOnly && g_metricSet && g_probeSite < 0 && request &&
         !strcmp(request, "1") && profilesSites();
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