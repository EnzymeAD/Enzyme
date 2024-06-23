#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

struct InstructionIdentifier {
  std::string moduleName;
  std::string functionName;
  unsigned blockIdx;
  unsigned instIdx;

  bool operator==(const InstructionIdentifier &other) const {
    return moduleName == other.moduleName &&
           functionName == other.functionName && blockIdx == other.blockIdx &&
           instIdx == other.instIdx;
  }
};

namespace std {
template <> struct hash<InstructionIdentifier> {
  std::size_t operator()(const InstructionIdentifier &id) const noexcept {
    std::size_t h1 = std::hash<std::string>{}(id.moduleName);
    std::size_t h2 = std::hash<std::string>{}(id.functionName);
    std::size_t h3 = std::hash<unsigned>{}(id.blockIdx);
    std::size_t h4 = std::hash<unsigned>{}(id.instIdx);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
  }
};
} // namespace std

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
  std::unordered_map<InstructionIdentifier, InstructionInfo> instructionData;

public:
  void update(const std::string &moduleName, const std::string &functionName,
              unsigned blockIdx, unsigned instIdx, double res, double err,
              const double *operands, unsigned numOperands) {
    InstructionIdentifier id = {moduleName, functionName, blockIdx, instIdx};
    auto &info = instructionData.emplace(id, InstructionInfo()).first->second;
    info.update(res, err, operands, numOperands);
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
