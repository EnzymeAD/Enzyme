//===- poseidon-kappa.cpp - measure a profiling run's condition numbers ---===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// poseidon-kappa -- <program> [args...]
//
// Runs a profile-generation build of a program that calls poseidon_metric,
// then measures the condition number of that metric at each site the run
// executed: how far the metric moves when the site is computed at a relative
// precision eps. It runs the program once more unperturbed, for the noise
// floor, and once per site and eps of POSEIDON_KAPPA_LADDER, each in a scratch
// directory of its own, and writes each site's condition number into its
// profile (Kappa and KappaMark).
//
// The profiling run is the program's: it gets this process's standard streams,
// and its exit status is this tool's, unless the measurement fails, which
// exits 1. The scratch directories are removed on every exit. The profiling
// run lists the sites to measure in the file POSEIDON_KAPPA_SITES names; a run
// of the measurement has POSEIDON_KAPPA=child and POSEIDON_KAPPA_SITE /
// POSEIDON_KAPPA_EPS, and writes its metric alone (FPProfiler.cpp).
//
//===----------------------------------------------------------------------===//

#include "runtime/profile/ProfileFormat.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <ftw.h>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern char **environ;

using namespace poseidon;

namespace {

// The program and its arguments.
std::vector<std::string> Program;

// Runs the program with `extra` added to the environment, its output in `log`
// when one is given. Returns its exit status, 128 + the signal that ended it,
// or -1 if it could not be started.
int runProgram(const std::vector<std::string> &extra, const std::string &log) {
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
  for (std::string &a : Program)
    argv.push_back(&a[0]);
  argv.push_back(nullptr);

  fflush(nullptr);
  pid_t pid = fork();
  if (pid < 0)
    return -1;
  if (pid == 0) {
    if (!log.empty()) {
      int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (fd >= 0) {
        dup2(fd, 1);
        dup2(fd, 2);
        close(fd);
      }
    }
    environ = envp.data();
    execvp(argv[0], argv.data());
    fprintf(stderr, "poseidon-kappa: cannot run %s: %s\n", argv[0],
            strerror(errno));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
    return -1;
  return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
}

// The metric a run wrote into `dir`, and its name. The value is the last field
// of the line: the name may hold spaces.
bool readMetric(const std::string &dir, double &value,
                std::string *name = nullptr) {
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
  if (name)
    *name = line.substr(0, at);
  return end != field && *end == '\0';
}

// The sites the profiling run listed, by site id, with their profiles.
std::map<int, std::string> readSites(const std::string &path) {
  std::map<int, std::string> sites;
  std::ifstream in(path);
  int site;
  std::string profile;
  while (in >> site && std::getline(in >> std::ws, profile))
    sites.emplace(site, profile);
  return sites;
}

// A scratch directory, removed with everything in it when it goes out of
// scope, as llvm::FileRemover removes a file.
class ScratchDir {
public:
  explicit ScratchDir(std::string path) : path(std::move(path)) {}
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;
  ~ScratchDir() { nftw(path.c_str(), removeEntry, 16, FTW_DEPTH | FTW_PHYS); }
  const std::string &get() const { return path; }

private:
  static int removeEntry(const char *entry, const struct stat *, int,
                         struct FTW *) {
    std::remove(entry);
    return 0;
  }
  std::string path;
};

struct Step {
  double eps;
  double kappa;
  bool diverged;
};

// Two slopes within this factor of each other agree.
constexpr double kSlopeAgreement = 2.0;

// A slope that holds across two neighbouring eps is a measured derivative; a
// ladder with no such pair is marked nonlinear; an eps that diverges above a
// clean one says where the site breaks without discarding the clean
// measurement.
void decide(const std::vector<Step> &ladder, double &kappa, std::string &mark,
            bool &hasDiverge) {
  hasDiverge = false;
  for (const Step &r : ladder)
    hasDiverge |= r.diverged;
  if (ladder.empty() || ladder.front().diverged) {
    kappa = std::numeric_limits<double>::infinity();
    mark = kProfileKappaDiverged;
    hasDiverge = false;
    return;
  }
  for (size_t i = 0; i + 1 < ladder.size(); ++i) {
    if (ladder[i].diverged || ladder[i + 1].diverged)
      continue;
    double lo = std::fabs(ladder[i].kappa), hi = std::fabs(ladder[i + 1].kappa);
    if (lo > hi)
      std::swap(lo, hi);
    if (hi == 0.0 || (lo > 0.0 && hi / lo <= kSlopeAgreement)) {
      kappa = ladder[i].kappa;
      mark = kProfileKappaOk;
      return;
    }
  }
  for (const Step &r : ladder)
    if (!r.diverged) {
      kappa = r.kappa;
      break;
    }
  mark = hasDiverge ? kProfileKappaOk : kProfileKappaNonlinear;
}

// The scratch directory of the run of `site` at `eps` under `tmp`.
std::string runDir(const std::string &tmp, int site, double eps) {
  char dir[64];
  std::snprintf(dir, sizeof(dir), "%s/site%d_eps%g", tmp.c_str(), site, eps);
  return dir;
}

std::vector<std::string> childEnv(const std::string &dir) {
  return {"POSEIDON_KAPPA=child", "POSEIDON_PROFILE_DIR=" + dir};
}

// A response this many times the noise floor is the site's, not noise.
constexpr double kAboveNoise = 4.0;

// The ladder of runs of `site` at each eps, and whether any response rose
// above the noise floor; the metric of the unperturbed run is m0.
std::vector<Step> measureSite(const std::string &tmp, int site,
                              const std::vector<double> &ladder, double m0,
                              double noise, bool &responded) {
  std::vector<Step> steps;
  responded = false;
  for (double eps : ladder) {
    const std::string d = runDir(tmp, site, eps);
    profileMkdirP(d);
    std::vector<std::string> env = childEnv(d);
    env.push_back("POSEIDON_KAPPA_SITE=" + std::to_string(site));
    char epsVar[64];
    std::snprintf(epsVar, sizeof(epsVar), "POSEIDON_KAPPA_EPS=%.17g", eps);
    env.push_back(epsVar);
    const int rc = runProgram(env, d + "/run.log");
    double mk = 0.0;
    if (rc != 0 || !readMetric(d, mk) || !std::isfinite(mk)) {
      fprintf(stderr, "    eps=%-8g DIVERGED (exit %d)\n", eps, rc);
      steps.push_back({eps, 0.0, true});
      continue;
    }
    const double resp = std::fabs(mk - m0);
    const double kappa = m0 != 0.0 ? resp / (eps * std::fabs(m0)) : resp / eps;
    const bool over = resp > kAboveNoise * noise;
    fprintf(stderr,
            "    eps=%-8g metric=%-24.17g response=%-12.4e kappa=%-12.4e %s\n",
            eps, mk, resp, kappa, over ? "above noise" : "within noise");
    steps.push_back({eps, kappa, false});
    responded |= over;
  }
  return steps;
}

void writeKappa(const std::string &path, double kappa,
                const std::string &mark) {
  char line[128];
  std::snprintf(line, sizeof(line), "%s = %.17g", kProfileKeyKappa, kappa);
  profileReplaceKappa(path,
                      {line, std::string(kProfileKeyKappaMark) + " = " + mark});
}

// The condition numbers of the metric m0 named `metric` at `sites`, written
// into their profiles; the scratch directories go under `tmp`. False, with the
// output of the run on stderr, if the second unperturbed run failed.
bool measure(const std::string &tmp, const std::map<int, std::string> &sites,
             const std::vector<double> &ladder, const std::string &metric,
             double m0) {
  const std::string base2 = tmp + "/baseline2";
  profileMkdirP(base2);
  fprintf(stderr, "[poseidon] baseline run 2 -> noise floor\n");
  double m2 = 0.0;
  const int rc = runProgram(childEnv(base2), base2 + "/run.log");
  if (rc != 0 || !readMetric(base2, m2)) {
    fprintf(stderr,
            "poseidon-kappa: error: the second unperturbed run (exit %d) wrote "
            "no metric; its output follows\n",
            rc);
    if (std::ifstream log(base2 + "/run.log"); log)
      std::cerr << log.rdbuf();
    return false;
  }
  const double noise = std::fabs(m2 - m0);
  fprintf(stderr, "[poseidon] metric %s = %.17g and %.17g; noise floor %.6e\n",
          metric.c_str(), m0, m2, noise);

  std::map<int, std::pair<double, std::string>> result;
  for (const auto &[site, path] : sites) {
    fprintf(stderr, "[poseidon] site %d (%s)\n", site, path.c_str());
    bool responded = false;
    const std::vector<Step> steps =
        measureSite(tmp, site, ladder, m0, noise, responded);
    double kappa = 0.0;
    std::string mark = kProfileKappaBelowNoise;
    bool hasDiverge = false;
    if (responded)
      decide(steps, kappa, mark, hasDiverge);
    result[site] = {kappa, mark};
    fprintf(stderr, "    -> Kappa = %.6e  (%s)%s\n", kappa, mark.c_str(),
            hasDiverge ? "  diverges above" : "");
  }

  fprintf(stderr, "[poseidon] site  kappa            mark          profile\n");
  for (const auto &[site, path] : sites) {
    const auto &[kappa, mark] = result[site];
    writeKappa(path, kappa, mark);
    fprintf(stderr, "[poseidon] %-5d %-16.6e %-13s %s\n", site, kappa,
            mark.c_str(), path.c_str());
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  int First = 1;
  if (First < argc && !strcmp(argv[First], "--"))
    ++First;
  if (First >= argc || argv[First][0] == '-') {
    fprintf(stderr,
            "usage: poseidon-kappa -- <program> [args...]\n"
            "Runs a profile-generation build, then runs it again to measure "
            "each site's\ncondition number against the metric it declares "
            "with poseidon_metric, and\nwrites them into its profiles. That "
            "takes 1 + <sites> x\n<steps> more runs, at the perturbation steps "
            "POSEIDON_KAPPA_LADDER lists\n(default 1e-8,1e-6,1e-4).\n");
    return 2;
  }
  Program.assign(argv + First, argv + argc);
  const std::vector<double> ladder = profileKappaLadder();

  const char *tmpdir = getenv("TMPDIR");
  std::string tmpl = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp") +
                     "/poseidon_kappa_XXXXXX";
  if (!mkdtemp(&tmpl[0])) {
    fprintf(stderr,
            "poseidon-kappa: cannot create a scratch directory in %s: "
            "%s\n",
            tmpl.c_str(), strerror(errno));
    return 1;
  }
  const ScratchDir scratch(tmpl);
  const std::string &tmp = scratch.get();
  const std::string siteList = tmp + "/sites";

  const auto profileStart = std::chrono::steady_clock::now();
  const int status = runProgram({"POSEIDON_KAPPA_SITES=" + siteList}, "");
  const std::chrono::duration<double> profileRun =
      std::chrono::steady_clock::now() - profileStart;
  // A program that failed, declared no metric or executed fewer than two
  // sites has nothing to measure.
  const std::map<int, std::string> sites = readSites(siteList);
  double m0 = 0.0;
  std::string metric;
  if (status != 0 || sites.size() < 2 ||
      !readMetric(profileDirectory(), m0, &metric))
    return status;

  const size_t runs = 1 + sites.size() * ladder.size();
  fprintf(stderr,
          "[poseidon] measuring the condition numbers of metric \"%s\" at %zu "
          "sites: %zu more runs of this program (run it without "
          "poseidon-kappa to skip)\n",
          metric.c_str(), sites.size(), runs);
  const auto measureStart = std::chrono::steady_clock::now();
  if (!measure(tmp, sites, ladder, metric, m0))
    return 1;
  const std::chrono::duration<double> measured =
      std::chrono::steady_clock::now() - measureStart;
  fprintf(stderr, "[poseidon] profile run: %.2f s\n", profileRun.count());
  fprintf(stderr, "[poseidon] condition-number measurement: %zu runs, %.2f s\n",
          runs, measured.count());
  return 0;
}
