#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

extern char **environ;

// The one filename rule, shared verbatim with the CUDA runtime and with the
// pass that reads what this file writes.
#include "FPProfileName.h"
#include "poseidon/poseidon.h"

#include <cstdio>

class ProfileInfo {
public:
  double minRes = std::numeric_limits<double>::max();
  double maxRes = std::numeric_limits<double>::lowest();
  std::vector<double> minOperands;
  std::vector<double> maxOperands;
  double sumValue = 0.0;
  double sumSens = 0.0;
  double sumGrad = 0.0;
  unsigned exec = 0;

  void updateValue(double value, const double *operands, size_t numOperands) {
    ++exec;

    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
    }
    for (size_t i = 0; i < numOperands; ++i) {
      if (!std::isnan(operands[i])) {
        minOperands[i] = std::min(minOperands[i], operands[i]);
        maxOperands[i] = std::max(maxOperands[i], operands[i]);
      }
    }

    if (!std::isnan(value)) {
      minRes = std::min(minRes, value);
      maxRes = std::max(maxRes, value);
      sumValue += value;
    }
  }

  void updateGradient(double value, double grad) {
    if (!std::isnan(grad) && !std::isnan(value) && !std::isinf(grad) &&
        !std::isinf(value)) {
      sumGrad += grad;
      sumSens += std::fabs(grad * value);
    }
  }
};

// `mkdir -p`: a missing intermediate component must not surface as an ignored
// "could not open" warning.
static void mkdirP(const std::string &dir) {
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

// Compile-time static data (the header lines the pass computed about a site
// and cannot recompute at profile-use), registered before main by a constructor
// the pass emits. A function-local static: a plain namespace-scope map could be
// constructed after that constructor runs.
static std::unordered_map<std::string, std::string> &staticHeaders() {
  static std::unordered_map<std::string, std::string> m;
  return m;
}

// The declared quantity of interest, latest value wins.
static std::string &metricName() {
  static std::string n;
  return n;
}
static double g_metricValue = 0.0;
static bool g_metricSet = false;

class FPProfiler {
private:
  std::string functionName;
  std::unordered_map<size_t, ProfileInfo> profileInfo;
  static std::string dir;

public:
  FPProfiler(const std::string &funcName) : functionName(funcName) {}

  static void setOutputDir(const std::string &dir_) { dir = dir_; }

  static const std::string &outputDir() { return dir; }

  std::string getOutputPath() const {
    return dir + "/" + poseidon::profileNameStem(functionName) + ".fpprofile";
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
    std::string outputPath = getOutputPath();

    mkdirP(dir);

    std::ofstream out(outputPath);
    // A profile that cannot be written is a profile the solve will not
    // find. Refuse rather than warn.
    if (!out.is_open()) {
      std::cerr << "Poseidon FPProfiler: could not open profile file for "
                   "writing:\n  "
                << outputPath << "\n  function: " << functionName
                << "\nThe profile is LOST, and a solve run against this "
                   "directory would silently skip the site. Aborting."
                << std::endl;
      std::abort();
    }

    out << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10);

    auto hdr = staticHeaders().find(functionName);
    if (hdr != staticHeaders().end())
      out << hdr->second;

    for (const auto &pair : profileInfo) {
      const auto i = pair.first;
      const auto &info = pair.second;
      out << i << "\n";
      out << "\tMinRes = " << info.minRes << "\n";
      out << "\tMaxRes = " << info.maxRes << "\n";
      out << "\tSumValue = " << info.sumValue << "\n";
      out << "\tSumSens = " << info.sumSens << "\n";
      out << "\tSumGrad = " << info.sumGrad << "\n";
      out << "\tExec = " << info.exec << "\n";
      out << "\tNumOperands = " << info.minOperands.size() << "\n";
      for (size_t i = 0; i < info.minOperands.size(); ++i) {
        out << "\tOperand[" << i << "] = [" << info.minOperands[i] << ", "
            << info.maxOperands[i] << "]\n";
      }
    }

    out << "\n";
    out.close();
  }
};

std::string FPProfiler::dir = POSEIDON_DEFAULT_PROFILE_DIR;

static std::unordered_map<std::string, std::unique_ptr<FPProfiler>>
    profilerRegistry;
static std::mutex registryMutex;

static void writeMetric() {
  if (!g_metricSet)
    return;
  const std::string dir = FPProfiler::outputDir();
  mkdirP(dir);
  const std::string path = dir + "/metric.txt";
  std::ofstream out(path);
  if (!out.is_open()) {
    std::cerr << "Poseidon FPProfiler: could not open " << path
              << " for writing; the declared quantity of interest is LOST and "
                 "the condition-number probe would score every site as "
                 "unresponsive. Aborting."
              << std::endl;
    std::abort();
  }
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", g_metricValue);
  out << metricName() << " " << buf << "\n";
  out.close();
}

// ---------------------------------------------------------------------------
// Per-site condition number of the declared quantity of interest
//
// The accuracy target bounds the number the application declares with
// poseidon_metric. How far that number moves when one site is computed at a
// relative precision eps is the site's condition number, and it is what puts
// every site's accuracy cost on one application-level scale. Measuring it takes
// the workload run again per (site, eps), so the profiling run does exactly
// that: it re-executes itself with the site armed through the environment, once
// more unperturbed for the noise floor, and folds the result into the profiles
// it just wrote. A child has POSEIDON_PROBE=0 and never re-executes.
// ---------------------------------------------------------------------------

// The children re-run the same GPU workload while this process is still inside
// its exit handlers, and on a shared device the memory this one still holds is
// what makes a child fail to allocate. Weak, so a profile-generation build with
// no CUDA in it still links.
extern "C" int cudaDeviceReset() __attribute__((weak));

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

namespace {
struct ProbeRun {
  double eps;
  double kappa;
  bool diverged;
};
} // namespace

static bool probeReadMetric(const std::string &dir, double &value) {
  std::ifstream in(dir + "/metric.txt");
  if (!in.is_open())
    return false;
  std::string name;
  if (!(in >> name >> value))
    return false;
  return true;
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

// The ladder rule. A slope that holds across two neighbouring eps is a measured
// derivative; a ladder with no such pair is a tolerance threshold rather than a
// slope and is marked nonlinear; and an eps that diverges above a clean one
// says where the site breaks without discarding the clean measurement.
static void probeDecide(const std::vector<ProbeRun> &ladder, double &kappa,
                        std::string &mark, double &divergesAbove,
                        bool &hasDiverge) {
  hasDiverge = false;
  divergesAbove = 0.0;
  for (const ProbeRun &r : ladder)
    if (r.diverged && !hasDiverge) {
      hasDiverge = true;
      divergesAbove = r.eps;
    }
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
    if (hi == 0.0 || (lo > 0.0 && hi / lo <= 2.0)) {
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

// Replace the site's Kappa lines, keeping them in the header block the
// profile-use compile reads before the first slot.
static void probeWriteKappa(const std::string &path, double kappa,
                            const std::string &mark, bool hasDiverge,
                            double divergesAbove) {
  std::vector<std::string> lines;
  {
    std::ifstream in(path);
    std::string l;
    while (std::getline(in, l))
      lines.push_back(l);
  }
  std::vector<std::string> kept;
  for (const std::string &l : lines)
    if (l.compare(0, 5, "Kappa") != 0)
      kept.push_back(l);
  size_t at = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    if (!kept[i].empty() &&
        kept[i].find_first_not_of("0123456789") == std::string::npos)
      break;
    at = i + 1;
    if (kept[i].compare(0, 13, "CanonicalHash") == 0)
      break;
  }
  char buf[128];
  std::vector<std::string> hdr;
  std::snprintf(buf, sizeof(buf), "Kappa = %.17g", kappa);
  hdr.push_back(buf);
  hdr.push_back("KappaMark = " + mark);
  if (hasDiverge) {
    std::snprintf(buf, sizeof(buf), "KappaDivergesAbove = %.17g",
                  divergesAbove);
    hdr.push_back(buf);
  }
  kept.insert(kept.begin() + at, hdr.begin(), hdr.end());
  std::ofstream out(path);
  for (const std::string &l : kept)
    out << l << "\n";
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

static void poseidonProbeDriver() {
  if (getenv("POSEIDON_PROBE_SITE"))
    return; // a child: it is the measurement, not the driver
  if (const char *off = getenv("POSEIDON_PROBE"); off && !strcmp(off, "0"))
    return;
  if (!g_metricSet)
    return;
  if (probeArgv().empty())
    return;

  const std::string pdir = FPProfiler::outputDir();
  std::map<int, std::string> sites;
  if (DIR *d = opendir(pdir.c_str())) {
    std::vector<std::string> names;
    while (struct dirent *e = readdir(d))
      names.push_back(e->d_name);
    closedir(d);
    std::sort(names.begin(), names.end());
    for (const std::string &n : names) {
      if (n.size() < 10 || n.compare(n.size() - 10, 10, ".fpprofile"))
        continue;
      std::ifstream in(pdir + "/" + n);
      std::string line;
      while (std::getline(in, line)) {
        int id = 0;
        if (std::sscanf(line.c_str(), " SiteId = %d", &id) == 1) {
          sites.emplace(id, pdir + "/" + n);
          break;
        }
      }
    }
  }
  if (sites.size() < 2)
    return;

  std::vector<double> ladder{1e-8, 1e-6, 1e-4};
  if (const char *l = getenv("POSEIDON_PROBE_LADDER")) {
    ladder.clear();
    for (const char *p = l; *p;) {
      char *end = nullptr;
      double v = strtod(p, &end);
      if (end == p)
        break;
      ladder.push_back(v);
      p = (*end == ',') ? end + 1 : end;
    }
  }

  char tmpl[] = "/tmp/poseidon_probe_XXXXXX";
  const char *tmp = mkdtemp(tmpl);
  if (!tmp) {
    std::cerr << "[probe] cannot create a scratch directory; skipping"
              << std::endl;
    return;
  }
  if (cudaDeviceReset)
    (void)cudaDeviceReset();

  const double m0 = g_metricValue;
  auto childEnv = [&](const std::string &dir) {
    return std::vector<std::string>{"POSEIDON_PROBE=0",
                                    "POSEIDON_PROFILE_DIR=" + dir};
  };

  const std::string base2 = std::string(tmp) + "/baseline2";
  mkdirP(base2);
  std::cerr << "[probe] baseline run 2 -> noise floor" << std::endl;
  double m2 = 0.0;
  int rc = probeRun(base2, childEnv(base2));
  if (rc != 0 || !probeReadMetric(base2, m2)) {
    std::cerr << "[probe] the second baseline run failed (see " << base2
              << "/run.log); not measuring condition numbers" << std::endl;
    return;
  }
  const double noise = std::fabs(m2 - m0);
  fprintf(stderr, "[probe] metric %s = %.17g and %.17g; noise floor %.6e\n",
          metricName().c_str(), m0, m2, noise);

  std::map<int, std::tuple<double, std::string, bool, double>> result;
  for (const auto &kv : sites) {
    const int sid = kv.first;
    fprintf(stderr, "[probe] site %d (%s)\n", sid, kv.second.c_str());
    std::vector<ProbeRun> runs;
    bool responded = false;
    for (double eps : ladder) {
      char dbuf[64];
      std::snprintf(dbuf, sizeof(dbuf), "%s/site%d_eps%g", tmp, sid, eps);
      const std::string d = dbuf;
      mkdirP(d);
      auto env = childEnv(d);
      env.push_back("POSEIDON_PROBE_SITE=" + std::to_string(sid));
      char ebuf[64];
      std::snprintf(ebuf, sizeof(ebuf), "POSEIDON_PROBE_EPS=%.17g", eps);
      env.push_back(ebuf);
      int crc = probeRun(d, env);
      double mk = 0.0;
      bool ok = crc == 0 && probeReadMetric(d, mk) && std::isfinite(mk);
      if (!ok) {
        fprintf(stderr, "    eps=%-8g DIVERGED (exit %d)\n", eps, crc);
        runs.push_back({eps, 0.0, true});
        continue;
      }
      double resp = std::fabs(mk - m0);
      double kappa = m0 != 0.0 ? resp / (eps * std::fabs(m0)) : resp / eps;
      bool over = resp > 4.0 * noise;
      fprintf(
          stderr,
          "    eps=%-8g metric=%-24.17g response=%-12.4e kappa=%-12.4e %s\n",
          eps, mk, resp, kappa, over ? "above noise" : "within noise");
      runs.push_back({eps, kappa, false});
      responded |= over;
    }
    double kappa = 0.0, above = 0.0;
    std::string mark = "below-noise";
    bool hasDiverge = false;
    if (responded)
      probeDecide(runs, kappa, mark, above, hasDiverge);
    result[sid] = {kappa, mark, hasDiverge, above};
    fprintf(stderr, "    -> Kappa = %.6e  (%s)%s\n", kappa, mark.c_str(),
            hasDiverge ? "  diverges above" : "");
  }

  std::cerr << "[probe] site  kappa            mark          profile"
            << std::endl;
  for (const auto &kv : sites) {
    auto [kappa, mark, hasDiverge, above] = result[kv.first];
    probeWriteKappa(kv.second, kappa, mark, hasDiverge, above);
    fprintf(stderr, "[probe] %-5d %-16.6e %-13s %s\n", kv.first, kappa,
            mark.c_str(), kv.second.c_str());
  }

  probeRemoveDir(base2);
  for (const auto &kv : sites)
    for (double eps : ladder) {
      char dbuf[64];
      std::snprintf(dbuf, sizeof(dbuf), "%s/site%d_eps%g", tmp, kv.first, eps);
      probeRemoveDir(dbuf);
    }
  rmdir(tmp);
}

static void writeAllProfilesAtExit() {
  std::lock_guard<std::mutex> lock(registryMutex);
  for (auto &pair : profilerRegistry) {
    pair.second->write();
  }
  profilerRegistry.clear();
  writeMetric();
  poseidonProbeDriver();
}

static int RegisterFPProfileRuntime() {
  if (const char *envPath = getenv("POSEIDON_PROFILE_DIR"))
    FPProfiler::setOutputDir(envPath);
  else
    FPProfiler::setOutputDir(POSEIDON_DEFAULT_PROFILE_DIR);

  captureProbeArgv();
  std::atexit(writeAllProfilesAtExit);

  return 0;
}

extern "C" int POSEIDON_PROFILE_RUNTIME_VAR = RegisterFPProfileRuntime();

extern "C" {

void ProfilerWrite() {
  std::lock_guard<std::mutex> lock(registryMutex);
  for (auto &pair : profilerRegistry) {
    pair.second->write();
  }
  writeMetric();
}

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

void poseidonLogGrad(const char *funcName, size_t idx, double value,
                     double grad) {
  if (!funcName)
    return;

  std::lock_guard<std::mutex> lock(registryMutex);

  auto it = profilerRegistry.find(funcName);
  if (it == profilerRegistry.end()) {
    profilerRegistry[funcName] = std::make_unique<FPProfiler>(funcName);
    it = profilerRegistry.find(funcName);
  }

  it->second->updateGradient(idx, value, grad);
}

void poseidonLogValue(const char *funcName, size_t idx, double res,
                      size_t numOperands, double *operands) {
  if (!funcName)
    return;

  std::lock_guard<std::mutex> lock(registryMutex);

  auto it = profilerRegistry.find(funcName);
  if (it == profilerRegistry.end()) {
    profilerRegistry[funcName] = std::make_unique<FPProfiler>(funcName);
    it = profilerRegistry.find(funcName);
  }

  it->second->updateValue(idx, res, numOperands, operands);
}

} // extern "C"