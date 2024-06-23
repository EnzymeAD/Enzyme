#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

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
  void update(const std::string &moduleName, const std::string &functionName,
              unsigned blockIdx, unsigned instIdx, double res, double err,
              const double *operands, unsigned numOperands) {
    InstructionIdentifier id = {moduleName, functionName, blockIdx, instIdx};
    instructionData[id].update(res, err, operands, numOperands);
  }

  void print() {
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

static DataManager *logger = nullptr;

void initializeLogger() { logger = new DataManager(); }

void destroyLogger() {
  delete logger;
  logger = nullptr;
}

void printLogger() { logger->print(); }

void enzymeLogError(double res, double err, const char *opcodeName,
                    const char *calleeName, const char *moduleName,
                    const char *functionName, unsigned blockIdx,
                    unsigned instIdx, unsigned numOperands, double *operands) {
  assert(logger && "Logger is not initialized");
  logger->update(moduleName, functionName, blockIdx, instIdx, res, err,
                 operands, numOperands);
}
