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
    if (!std::isnan(grad) && !std::isnan(value)) {
      sumGrad += grad;
      sumSens += std::fabs(grad * value);
    }
  }
};

class FPProfiler {
private:
  std::string functionName;
  std::unordered_map<size_t, ProfileInfo> profileInfo;
  static std::string dir;

public:
  FPProfiler(const std::string &funcName) : functionName(funcName) {}

  static void setOutputDir(const std::string &dir_) { dir = dir_; }

  std::string getOutputPath() const {
    return dir + "/" + functionName + ".fpprofile";
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

std::string FPProfiler::dir = "./fpprofile";

static std::unordered_map<std::string, std::unique_ptr<FPProfiler>>
    profilerRegistry;
static std::mutex registryMutex;

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

void enzymeLogGrad(const char *funcName, size_t idx, double value,
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

void enzymeLogValue(const char *funcName, size_t idx, double res,
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