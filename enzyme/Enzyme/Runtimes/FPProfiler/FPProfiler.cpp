#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <utility>
#include <vector>

double EPS = 0.0;

class ValueInfo {
public:
  double minRes = std::numeric_limits<double>::max();
  double maxRes = std::numeric_limits<double>::lowest();
  std::vector<double> minOperands;
  std::vector<double> maxOperands;
  unsigned executions = 0;

  double runningSumLog = 0.0;
  unsigned runningCountNonZero = 0;
  double runningSumArith = 0.0;
  unsigned validCount = 0;

  void update(double res, const double *operands, unsigned numOperands) {
    ++executions;

    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
    }
    for (unsigned i = 0; i < numOperands; ++i) {
      if (!std::isnan(operands[i])) {
        minOperands[i] = std::min(minOperands[i], operands[i]);
        maxOperands[i] = std::max(maxOperands[i], operands[i]);
      }
    }

    if (!std::isnan(res)) {
      minRes = std::min(minRes, res);
      maxRes = std::max(maxRes, res);

      double absRes = std::fabs(res);
      runningSumArith += absRes;
      ++validCount;

      if (EPS != 0.0) {
        runningSumLog += std::log(absRes + EPS);
      } else {
        if (absRes != 0.0) {
          runningSumLog += std::log(absRes);
          ++runningCountNonZero;
        }
      }
    }
  }

  double getGeoMean() const {
    if (validCount == 0)
      return 0.0;

    if (EPS != 0.0) {
      return std::exp(runningSumLog / validCount) - EPS;
    } else {
      if (runningCountNonZero == 0) {
        return 0.0;
      }
      return std::exp(runningSumLog / runningCountNonZero);
    }
  }

  double getArithMean() const {
    if (validCount == 0)
      return 0.0;
    return runningSumArith / validCount;
  }

  double getMaxAbs() const {
    return std::max(std::abs(minRes), std::abs(maxRes));
  }
};

class GradInfo {
public:
  double runningSumLog = 0.0;
  unsigned runningCountNonZero = 0;
  double runningSumArith = 0.0;
  unsigned validCount = 0;

  double minGrad = std::numeric_limits<double>::max();
  double maxGrad = std::numeric_limits<double>::lowest();

  void update(double grad) {
    if (!std::isnan(grad)) {
      minGrad = std::min(minGrad, grad);
      maxGrad = std::max(maxGrad, grad);

      double absGrad = std::fabs(grad);

      runningSumArith += absGrad;
      ++validCount;

      if (EPS != 0.0) {
        runningSumLog += std::log(absGrad + EPS);
      } else {
        if (absGrad != 0.0) {
          runningSumLog += std::log(absGrad);
          ++runningCountNonZero;
        }
      }
    }
  }

  double getGeoMean() const {
    if (validCount == 0)
      return 0.0;

    if (EPS != 0.0) {
      return std::exp(runningSumLog / validCount) - EPS;
    } else {
      if (runningCountNonZero == 0) {
        return 0.0;
      }
      return std::exp(runningSumLog / runningCountNonZero);
    }
  }

  double getArithMean() const {
    if (validCount == 0)
      return 0.0;
    return runningSumArith / validCount;
  }

  double getMaxAbs() const {
    return std::max(std::abs(minGrad), std::abs(maxGrad));
  }
};

class FPProfiler {
private:
  std::string functionName;
  std::unordered_map<std::string, ValueInfo> valueInfo;
  std::unordered_map<std::string, GradInfo> gradInfo;
  static std::string profileOutputDir;

public:
  FPProfiler(const std::string &funcName) : functionName(funcName) {}

  static void setOutputDir(const std::string &dir) { profileOutputDir = dir; }

  std::string getOutputPath() const {
    std::string dir =
        profileOutputDir.empty() ? "./fpprofile" : profileOutputDir;
    return dir + "/" + functionName + ".fpprofile";
  }

  void updateValue(const std::string &id, double res, unsigned numOperands,
                   const double *operands) {
    auto &info = valueInfo.emplace(id, ValueInfo()).first->second;
    info.update(res, operands, numOperands);
  }

  void updateGrad(const std::string &id, double grad) {
    auto &info = gradInfo.emplace(id, GradInfo()).first->second;
    info.update(grad);
  }

  void write() const {
    std::string outputPath = getOutputPath();

    std::string dir =
        profileOutputDir.empty() ? "./fpprofile" : profileOutputDir;
    struct stat st = {0};
    if (stat(dir.c_str(), &st) == -1) {
      mkdir(dir.c_str(), 0755);
    }

    std::ofstream out(outputPath);
    if (!out.is_open()) {
      std::cerr << "Warning: Could not open profile file: " << outputPath
                << std::endl;
      return;
    }

    out << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (const auto &pair : valueInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      out << "Value:" << id << "\n";
      out << "\tMinRes = " << info.minRes << "\n";
      out << "\tMaxRes = " << info.maxRes << "\n";
      out << "\tExecutions = " << info.executions << "\n";
      out << "\tGeoMeanAbs = " << info.getGeoMean() << "\n";
      out << "\tArithMeanAbs = " << info.getArithMean() << "\n";
      out << "\tMaxAbs = " << info.getMaxAbs() << "\n";
      for (unsigned i = 0; i < info.minOperands.size(); ++i) {
        out << "\tOperand[" << i << "] = [" << info.minOperands[i] << ", "
            << info.maxOperands[i] << "]\n";
      }
    }

    for (const auto &pair : gradInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      out << "Grad:" << id << "\n";
      out << "\tGeoMeanAbs = " << info.getGeoMean() << "\n";
      out << "\tArithMeanAbs = " << info.getArithMean() << "\n";
      out << "\tMaxAbs = " << info.getMaxAbs() << "\n";
    }

    out << "\n";
    out.close();
  }
};

std::string FPProfiler::profileOutputDir;

static std::unordered_map<std::string, std::unique_ptr<FPProfiler>>
    profilerRegistry;
static std::mutex registryMutex;

static std::string getFunctionNameFromId(const char *id) {
  std::string idStr(id);
  size_t colonPos = idStr.find(':');
  if (colonPos != std::string::npos) {
    return idStr.substr(0, colonPos);
  }
  return idStr;
}

static void writeAllProfilesAtExit() {
  std::lock_guard<std::mutex> lock(registryMutex);
  for (auto &pair : profilerRegistry) {
    pair.second->write();
  }
  profilerRegistry.clear();
}

static int RegisterFPProfileRuntime() {
  const char *envPath = getenv("ENZYME_FPPROFILE_DIR");
  if (envPath) {
    FPProfiler::setOutputDir(envPath);
  } else {
    FPProfiler::setOutputDir("./fpprofile");
  }

  const char *envEps = getenv("ENZYME_FPPROFILE_EPS");
  if (envEps) {
    char *endptr;
    double epsValue = std::strtod(envEps, &endptr);
    if (*endptr == '\0' && epsValue >= 0.0) {
      EPS = epsValue;
    } else {
      std::cerr << "Warning: Invalid ENZYME_FPPROFILE_EPS value: " << envEps
                << ". Using default: " << EPS << std::endl;
    }
  }

  std::atexit(writeAllProfilesAtExit);

  return 0;
}

extern "C" int ENZYME_FPPROFILE_RUNTIME_VAR = RegisterFPProfileRuntime();

extern "C" {

void ProfilerWrite() {
  std::lock_guard<std::mutex> lock(registryMutex);
  for (auto &pair : profilerRegistry) {
    pair.second->write();
  }
}

void enzymeLogGrad(const char *id, double grad) {
  if (!id)
    return;

  std::string funcName = getFunctionNameFromId(id);
  std::lock_guard<std::mutex> lock(registryMutex);

  auto it = profilerRegistry.find(funcName);
  if (it == profilerRegistry.end()) {
    profilerRegistry[funcName] = std::make_unique<FPProfiler>(funcName);
    it = profilerRegistry.find(funcName);
  }

  it->second->updateGrad(id, grad);
}

void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands) {
  if (!id)
    return;

  std::string funcName = getFunctionNameFromId(id);
  std::lock_guard<std::mutex> lock(registryMutex);

  auto it = profilerRegistry.find(funcName);
  if (it == profilerRegistry.end()) {
    profilerRegistry[funcName] = std::make_unique<FPProfiler>(funcName);
    it = profilerRegistry.find(funcName);
  }

  it->second->updateValue(id, res, numOperands, operands);
}

} // extern "C"