//=- PoseidonLogger.cpp - Logging utilities for Poseidon ------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logging-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <regex>

#include "PoseidonLogger.h"
#include "PoseidonUtils.h"

using namespace llvm;

extern "C" {
cl::opt<std::string>
    FPOptLogPath("fpopt-log-path", cl::init(""), cl::Hidden,
                 cl::desc("Which log to use in the FPOpt pass"));
cl::opt<bool> FPOptLooseCoverage(
    "fpopt-loose-coverage", cl::init(false), cl::Hidden,
    cl::desc("Allow unexecuted FP instructions in subgraph indentification"));
cl::opt<double>
    FPOptWidenRange("fpopt-widen-range", cl::init(1), cl::Hidden,
                    cl::desc("Ablation study only: widen the range of input "
                             "hypercube by this factor"));
}

bool extractValueFromLog(const std::string &logPath,
                         const std::string &functionName, size_t blockIdx,
                         size_t instIdx, ValueInfo &data) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex valuePattern("^Value:" + functionName + ":" +
                          std::to_string(blockIdx) + ":" +
                          std::to_string(instIdx) + "$");

  std::regex newEntryPattern("^(Value|Grad|Error):");

  std::regex minResPattern(R"(^\s*MinRes\s*=\s*([\d\.eE+\-]+))");
  std::regex maxResPattern(R"(^\s*MaxRes\s*=\s*([\d\.eE+\-]+))");
  std::regex executionsPattern(R"(^\s*Executions\s*=\s*(\d+))");
  std::regex geoMeanPattern(R"(^\s*GeoMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex arithMeanPattern(R"(^\s*ArithMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex maxAbsPattern(R"(^\s*MaxAbs\s*=\s*([\d\.eE+\-]+))");

  std::regex operandPattern(
      R"(^\s*Operand\[(\d+)\]\s*=\s*\[([\d\.eE+\-]+),\s*([\d\.eE+\-]+)\])");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (std::regex_search(line, valuePattern)) {
      std::string minResLine, maxResLine, execLine;
      std::string geoMeanLine, arithMeanLine, maxAbsLine;

      if (std::getline(file, minResLine) && std::getline(file, maxResLine) &&
          std::getline(file, execLine) && std::getline(file, geoMeanLine) &&
          std::getline(file, arithMeanLine) && std::getline(file, maxAbsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r') {
            s.pop_back();
          }
        };
        stripCR(minResLine);
        stripCR(maxResLine);
        stripCR(execLine);
        stripCR(geoMeanLine);
        stripCR(arithMeanLine);
        stripCR(maxAbsLine);

        std::smatch mMinRes, mMaxRes, mExec, mGeo, mArith, mMaxAbs;
        if (std::regex_search(minResLine, mMinRes, minResPattern) &&
            std::regex_search(maxResLine, mMaxRes, maxResPattern) &&
            std::regex_search(execLine, mExec, executionsPattern) &&
            std::regex_search(geoMeanLine, mGeo, geoMeanPattern) &&
            std::regex_search(arithMeanLine, mArith, arithMeanPattern) &&
            std::regex_search(maxAbsLine, mMaxAbs, maxAbsPattern)) {

          data.minRes = stringToDouble(mMinRes[1]);
          data.maxRes = stringToDouble(mMaxRes[1]);
          data.executions = static_cast<unsigned>(std::stoul(mExec[1]));
          data.geoMean = stringToDouble(mGeo[1]);
          data.arithMean = stringToDouble(mArith[1]);
          data.maxAbs = stringToDouble(mMaxAbs[1]);
        } else {
          std::string error =
              "Failed to parse stats for: Function: " + functionName +
              ", BlockIdx: " + std::to_string(blockIdx) +
              ", InstIdx: " + std::to_string(instIdx);
          llvm_unreachable(error.c_str());
        }
      } else {
        std::string error =
            "Incomplete stats block for: Function: " + functionName +
            ", BlockIdx: " + std::to_string(blockIdx) +
            ", InstIdx: " + std::to_string(instIdx);
        llvm_unreachable(error.c_str());
      }

      while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
          line.pop_back();

        if (std::regex_search(line, newEntryPattern)) {
          if (FPOptWidenRange != 1.0) {
            double center = (data.minRes + data.maxRes) / 2.0;
            double half_range = (data.maxRes - data.minRes) / 2.0;
            double new_half_range = half_range * FPOptWidenRange;
            data.minRes = center - new_half_range;
            data.maxRes = center + new_half_range;

            for (size_t i = 0; i < data.minOperands.size(); ++i) {
              double op_center =
                  (data.minOperands[i] + data.maxOperands[i]) / 2.0;
              double op_half_range =
                  (data.maxOperands[i] - data.minOperands[i]) / 2.0;
              double op_new_half_range = op_half_range * FPOptWidenRange;
              data.minOperands[i] = op_center - op_new_half_range;
              data.maxOperands[i] = op_center + op_new_half_range;
            }
          }
          return true;
        }

        std::smatch operandMatch;
        if (std::regex_search(line, operandMatch, operandPattern)) {
          unsigned opIdx = static_cast<unsigned>(std::stoul(operandMatch[1]));
          double minVal = stringToDouble(operandMatch[2]);
          double maxVal = stringToDouble(operandMatch[3]);

          if (opIdx >= data.minOperands.size()) {
            data.minOperands.resize(opIdx + 1, 0.0);
            data.maxOperands.resize(opIdx + 1, 0.0);
          }
          data.minOperands[opIdx] = minVal;
          data.maxOperands[opIdx] = maxVal;
        }
      }
    }
  }

  llvm::errs() << "Failed to extract value info for: Function: " << functionName
               << ", BlockIdx: " << blockIdx << ", InstIdx: " << instIdx
               << "\n";
  return false;
}

bool extractGradFromLog(const std::string &logPath,
                        const std::string &functionName, size_t blockIdx,
                        size_t instIdx, GradInfo &data) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    llvm_unreachable("Failed to open log file");
  }

  std::string line;
  std::regex gradPattern("^Grad:" + functionName + ":" +
                         std::to_string(blockIdx) + ":" +
                         std::to_string(instIdx) + "$");

  std::regex geoMeanPattern(R"(^\s*GeoMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex arithMeanPattern(R"(^\s*ArithMeanAbs\s*=\s*([\d\.eE+\-]+))");
  std::regex maxAbsPattern(R"(^\s*MaxAbs\s*=\s*([\d\.eE+\-]+))");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    if (std::regex_search(line, gradPattern)) {
      std::string geoMeanLine, arithMeanLine, maxAbsLine;
      if (std::getline(file, geoMeanLine) &&
          std::getline(file, arithMeanLine) && std::getline(file, maxAbsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r') {
            s.pop_back();
          }
        };
        stripCR(geoMeanLine);
        stripCR(arithMeanLine);
        stripCR(maxAbsLine);

        std::smatch mGeo, mArith, mMax;
        if (std::regex_search(geoMeanLine, mGeo, geoMeanPattern) &&
            std::regex_search(arithMeanLine, mArith, arithMeanPattern) &&
            std::regex_search(maxAbsLine, mMax, maxAbsPattern)) {

          data.geoMean = stringToDouble(mGeo[1]);
          data.arithMean = stringToDouble(mArith[1]);
          data.maxAbs = stringToDouble(mMax[1]);
          return true;
        }
      }

      llvm::errs() << "Incomplete gradient block for: Function: "
                   << functionName << ", BlockIdx: " << blockIdx
                   << ", InstIdx: " << instIdx << "\n";
      return false;
    }
  }

  llvm::errs() << "Failed to extract gradient for: Function: " << functionName
               << ", BlockIdx: " << blockIdx << ", InstIdx: " << instIdx
               << "\n";
  return false;
}

bool isLogged(const std::string &logPath, const std::string &functionName) {
  std::ifstream file(logPath);
  if (!file.is_open()) {
    assert(0 && "Failed to open log file");
  }

  std::regex functionRegex("^Value:" + functionName);

  std::string line;
  while (std::getline(file, line)) {
    if (std::regex_search(line, functionRegex)) {
      return true;
    }
  }

  return false;
}