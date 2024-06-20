// RUN: %clang++ -O0 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O1 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O2 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -
// RUN: %clang++ -O3 %s -S -emit-llvm -o - | %opt - %OPloadEnzyme %enzyme -S | %lli -

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../test_utils.h"

extern double __enzyme_error_estimate(void *, ...);

struct InstructionIdentifier {
  std::string moduleName;
  std::string functionName;
  unsigned blockIdx;
  unsigned instIdx;

  bool operator<(const InstructionIdentifier &other) const {
    if (moduleName < other.moduleName)
      return true;
    if (moduleName > other.moduleName)
      return false;
    if (functionName < other.functionName)
      return true;
    if (functionName > other.functionName)
      return false;
    if (blockIdx < other.blockIdx)
      return true;
    if (blockIdx > other.blockIdx)
      return false;
    return instIdx < other.instIdx;
  }
};

class InstructionInfo {
public:
  double minRes = std::numeric_limits<double>::max();
  double maxRes = std::numeric_limits<double>::lowest();
  double minErr = std::numeric_limits<double>::max();
  double maxErr = std::numeric_limits<double>::lowest();
  std::vector<double> minOperands;
  std::vector<double> maxOperands;
  unsigned executions = 0;

  void update(double res, double err, const double *operands,
              unsigned numOperands) {
    minRes = std::min(minRes, res);
    maxRes = std::max(maxRes, res);
    minErr = std::min(minErr, err);
    maxErr = std::max(maxErr, err);
    if (minOperands.empty()) {
      minOperands.resize(numOperands, std::numeric_limits<double>::max());
      maxOperands.resize(numOperands, std::numeric_limits<double>::lowest());
    }
    for (unsigned i = 0; i < numOperands; ++i) {
      minOperands[i] = std::min(minOperands[i], operands[i]);
      maxOperands[i] = std::max(maxOperands[i], operands[i]);
    }
    ++executions;
  }
};

class DataManager {
private:
  std::map<InstructionIdentifier, InstructionInfo> instructionData;

public:
  void logExecution(const std::string &moduleName,
                    const std::string &functionName, unsigned blockIdx,
                    unsigned instIdx, double res, double err,
                    const double *operands, unsigned numOperands) {
    InstructionIdentifier id = {moduleName, functionName, blockIdx, instIdx};
    instructionData[id].update(res, err, operands, numOperands);
  }

  void printData() {
    for (auto &entry : instructionData) {
      auto &id = entry.first;
      auto &info = entry.second;
      std::cout << "Module: " << id.moduleName
                << ", Function: " << id.functionName
                << ", BlockIdx: " << id.blockIdx << ", InstIdx: " << id.instIdx
                << "\n"
                << "Min Res: " << info.minRes << ", Max Res: " << info.maxRes
                << ", Min Error: " << info.minErr
                << ", Max Error: " << info.maxErr
                << ", Executions: " << info.executions << "\n";
      for (size_t i = 0; i < info.minOperands.size(); ++i) {
        std::cout << "Operand[" << i << "] Range: [" << info.minOperands[i]
                  << ", " << info.maxOperands[i] << "]\n";
      }
      std::cout << "\n";
    }
  }
};

DataManager *logger = nullptr;

void initializeDataManager() { logger = new DataManager(); }

void destroyDataManager() {
  delete logger;
  logger = nullptr;
}

void enzymeLogError(double res, double err, const char *opcodeName,
                    const char *calleeName, const char *moduleName,
                    const char *functionName, unsigned blockIdx,
                    unsigned instIdx, unsigned numOperands, double *operands) {
  logger->logExecution(moduleName, functionName, blockIdx, instIdx, res,
                            err, operands, numOperands);
}

// An example from https://dl.acm.org/doi/10.1145/3371128
double fun(double x) {
  double v1 = cos(x);
  double v2 = 1 - v1;
  double v3 = x * x;
  double v4 = v2 / v3;

  printf("v1 = %.18e, v2 = %.18e, v3 = %.18e, v4 = %.18e\n", v1, v2, v3, v4);

  return v4;
}

int main() {
  initializeDataManager();
  double res = fun(1e-7);
  __enzyme_error_estimate((void *)fun, 2e-7, 0.0);
  __enzyme_error_estimate((void *)fun, 7e-7, 0.0);
  double error = __enzyme_error_estimate((void *)fun, 1e-7, 0.0);
  printf("res = %.18e, abs error = %.18e, rel error = %.18e\n", res, error,
         fabs(error / res));
  APPROX_EQ(error, 2.2222222222e-2, 1e-4);
  logger->printData();
  destroyDataManager();
}
