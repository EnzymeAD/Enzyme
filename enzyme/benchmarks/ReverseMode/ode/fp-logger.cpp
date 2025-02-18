#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "fp-logger.hpp"

class ValueInfo {
public:
  double minRes = std::numeric_limits<double>::max();
  double maxRes = std::numeric_limits<double>::lowest();
  std::vector<double> minOperands;
  std::vector<double> maxOperands;
  unsigned executions = 0;
  double logSum = 0.0;
  unsigned logCount = 0;

  void update(double res, const double *operands, unsigned numOperands) {
    minRes = std::min(minRes, res);
    maxRes = std::max(maxRes, res);
    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
    }
    for (unsigned i = 0; i < numOperands; ++i) {
      minOperands[i] = std::min(minOperands[i], operands[i]);
      maxOperands[i] = std::max(maxOperands[i], operands[i]);
    }
    ++executions;

    if (!std::isnan(res)) {
      logSum += std::log1p(std::fabs(res));
      ++logCount;
    }
  }

  double getGeometricAverage() const {
    if (logCount == 0) {
      return 0.;
    }
    return std::expm1(logSum / logCount);
  }
};

class ErrorInfo {
public:
  double minErr = std::numeric_limits<double>::max();
  double maxErr = std::numeric_limits<double>::lowest();

  void update(double err) {
    minErr = std::min(minErr, err);
    maxErr = std::max(maxErr, err);
  }
};

class GradInfo {
public:
  double logSum = 0.0;
  unsigned count = 0;

  void update(double grad) {
    if (!std::isnan(grad)) {
      logSum += std::log1p(std::fabs(grad));
      ++count;
    }
  }

  double getGeometricAverage() const {
    if (count == 0) {
      return 0.;
    }
    return std::expm1(logSum / count);
  }
};

class Logger {
private:
  std::unordered_map<std::string, ValueInfo> valueInfo;
  std::unordered_map<std::string, ErrorInfo> errorInfo;
  std::unordered_map<std::string, GradInfo> gradInfo;

public:
  void updateValue(const std::string &id, double res, unsigned numOperands,
                   const double *operands) {
    auto &info = valueInfo.emplace(id, ValueInfo()).first->second;
    info.update(res, operands, numOperands);
  }

  void updateError(const std::string &id, double err) {
    auto &info = errorInfo.emplace(id, ErrorInfo()).first->second;
    info.update(err);
  }

  void updateGrad(const std::string &id, double grad) {
    auto &info = gradInfo.emplace(id, GradInfo()).first->second;
    info.update(grad);
  }

  void print() const {
    std::cout << std::scientific
              << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (const auto &pair : valueInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      std::cout << "Value:" << id << "\n";
      std::cout << "\tMinRes = " << info.minRes << "\n";
      std::cout << "\tMaxRes = " << info.maxRes << "\n";
      std::cout << "\tExecutions = " << info.executions << "\n";
      std::cout << "\tGeometric Average = " << info.getGeometricAverage()
                << "\n";
      for (unsigned i = 0; i < info.minOperands.size(); ++i) {
        std::cout << "\tOperand[" << i << "] = [" << info.minOperands[i] << ", "
                  << info.maxOperands[i] << "]\n";
      }
    }

    for (const auto &pair : errorInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      std::cout << "Error:" << id << "\n";
      std::cout << "\tMinErr = " << info.minErr << "\n";
      std::cout << "\tMaxErr = " << info.maxErr << "\n";
    }

    for (const auto &pair : gradInfo) {
      const auto &id = pair.first;
      const auto &info = pair.second;
      std::cout << "Grad:" << id << "\n";
      std::cout << "\tGrad = " << info.getGeometricAverage() << "\n";
    }
  }
};

Logger *logger = nullptr;

void initializeLogger() { logger = new Logger(); }

void destroyLogger() {
  delete logger;
  logger = nullptr;
}

void printLogger() { logger->print(); }

void enzymeLogError(const char *id, double err) {
  assert(logger && "Logger is not initialized");
  logger->updateError(id, err);
}

void enzymeLogGrad(const char *id, double grad) {
  assert(logger && "Logger is not initialized");
  logger->updateGrad(id, grad);
}

void enzymeLogValue(const char *id, double res, unsigned numOperands,
                    double *operands) {
  assert(logger && "Logger is not initialized");
  logger->updateValue(id, res, numOperands, operands);
}