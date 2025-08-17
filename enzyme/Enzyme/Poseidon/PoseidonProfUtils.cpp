//=- PoseidonProfUtils.cpp - Profiling utilities for Poseidon
//------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements profiling-related utilities for the Poseidon
// optimization pass.
//
//===----------------------------------------------------------------------===//

#include <fstream>
#include <regex>

#include "PoseidonProfUtils.h"
#include "PoseidonUtils.h"

using namespace llvm;

extern "C" {
cl::opt<bool> FPOptLooseCoverage(
    "fpopt-loose-coverage", cl::init(false), cl::Hidden,
    cl::desc("Allow unexecuted FP instructions in subgraph indentification"));
cl::opt<double>
    FPOptWidenRange("fpopt-widen-range", cl::init(1), cl::Hidden,
                    cl::desc("Ablation study only: widen the range of input "
                             "hypercube by this factor"));
}

void parseProfileFile(const std::string &profilePath,
                      std::unordered_map<size_t, ProfileInfo> &profileMap) {
  profileMap.clear();
  std::ifstream file(profilePath);
  if (!file.is_open()) {
    llvm::errs() << "Warning: Could not open profile file: " << profilePath
                 << "\n";
    return;
  }

  std::string line;
  std::regex indexPattern(R"(^(\d+)$)");

  std::regex minResPattern(R"(^\s*MinRes\s*=\s*([\d\.eE+\-]+))");
  std::regex maxResPattern(R"(^\s*MaxRes\s*=\s*([\d\.eE+\-]+))");
  std::regex sumValuePattern(R"(^\s*SumValue\s*=\s*([\d\.eE+\-]+))");
  std::regex sumSensPattern(R"(^\s*SumSens\s*=\s*([\d\.eE+\-]+))");
  std::regex sumGradPattern(R"(^\s*SumGrad\s*=\s*([\d\.eE+\-]+))");
  std::regex execPattern(R"(^\s*Exec\s*=\s*(\d+))");
  std::regex numOperandsPattern(R"(^\s*NumOperands\s*=\s*(\d+))");
  std::regex operandPattern(
      R"(^\s*Operand\[(\d+)\]\s*=\s*\[([\d\.eE+\-]+),\s*([\d\.eE+\-]+)\])");

  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    std::smatch match;
    if (std::regex_match(line, match, indexPattern)) {
      size_t idx = std::stoull(match[1]);
      ProfileInfo info;

      std::string minResLine, maxResLine, sumValueLine, sumSensLine,
          sumGradLine, execLine, numOperandsLine;

      if (std::getline(file, minResLine) && std::getline(file, maxResLine) &&
          std::getline(file, sumValueLine) && std::getline(file, sumSensLine) &&
          std::getline(file, sumGradLine) && std::getline(file, execLine) &&
          std::getline(file, numOperandsLine)) {

        auto stripCR = [](std::string &s) {
          if (!s.empty() && s.back() == '\r')
            s.pop_back();
        };
        stripCR(minResLine);
        stripCR(maxResLine);
        stripCR(sumValueLine);
        stripCR(sumSensLine);
        stripCR(sumGradLine);
        stripCR(execLine);
        stripCR(numOperandsLine);

        std::smatch mMinRes, mMaxRes, mSumValue, mSumSens, mSumGrad, mExec,
            mNumOperands;
        if (std::regex_search(minResLine, mMinRes, minResPattern) &&
            std::regex_search(maxResLine, mMaxRes, maxResPattern) &&
            std::regex_search(sumValueLine, mSumValue, sumValuePattern) &&
            std::regex_search(sumSensLine, mSumSens, sumSensPattern) &&
            std::regex_search(sumGradLine, mSumGrad, sumGradPattern) &&
            std::regex_search(execLine, mExec, execPattern) &&
            std::regex_search(numOperandsLine, mNumOperands,
                              numOperandsPattern)) {

          info.minRes = stringToDouble(mMinRes[1]);
          info.maxRes = stringToDouble(mMaxRes[1]);
          info.sumValue = stringToDouble(mSumValue[1]);
          info.sumSens = stringToDouble(mSumSens[1]);
          info.sumGrad = stringToDouble(mSumGrad[1]);
          info.exec = static_cast<unsigned>(std::stoul(mExec[1]));
          unsigned numOperands =
              static_cast<unsigned>(std::stoul(mNumOperands[1]));

          info.minOperands.resize(numOperands, 0.0);
          info.maxOperands.resize(numOperands, 0.0);

          for (unsigned i = 0; i < numOperands; ++i) {
            if (std::getline(file, line)) {
              if (!line.empty() && line.back() == '\r')
                line.pop_back();

              std::smatch operandMatch;
              if (std::regex_search(line, operandMatch, operandPattern)) {
                unsigned opIdx =
                    static_cast<unsigned>(std::stoul(operandMatch[1]));
                double minVal = stringToDouble(operandMatch[2]);
                double maxVal = stringToDouble(operandMatch[3]);

                if (opIdx < numOperands) {
                  info.minOperands[opIdx] = minVal;
                  info.maxOperands[opIdx] = maxVal;
                }
              }
            }
          }

          if (FPOptWidenRange != 1.0) {
            double center = (info.minRes + info.maxRes) / 2.0;
            double half_range = (info.maxRes - info.minRes) / 2.0;
            double new_half_range = half_range * FPOptWidenRange;
            info.minRes = center - new_half_range;
            info.maxRes = center + new_half_range;

            for (size_t i = 0; i < info.minOperands.size(); ++i) {
              double op_center =
                  (info.minOperands[i] + info.maxOperands[i]) / 2.0;
              double op_half_range =
                  (info.maxOperands[i] - info.minOperands[i]) / 2.0;
              double op_new_half_range = op_half_range * FPOptWidenRange;
              info.minOperands[i] = op_center - op_new_half_range;
              info.maxOperands[i] = op_center + op_new_half_range;
            }
          }

          profileMap[idx] = info;
        } else {
          llvm::errs() << "Warning: Failed to parse profile fields for index "
                       << idx << "\n";
        }
      } else {
        llvm::errs() << "Warning: Incomplete profile entry for index " << idx
                     << "\n";
      }
    }
  }
}
