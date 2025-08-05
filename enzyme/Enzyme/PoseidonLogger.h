//=- PoseidonLogger.h - Logging utilities for Poseidon --------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares logging-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_LOGGER_H
#define ENZYME_POSEIDON_LOGGER_H

#include "llvm/ADT/SmallVector.h"

#include <limits>

using namespace llvm;

struct GradInfo {
  double geoMean;
  double arithMean;
  double maxAbs;

  GradInfo() : geoMean(0.0), arithMean(0.0), maxAbs(0.0) {}
};

struct ValueInfo {
  double minRes;
  double maxRes;
  unsigned executions;
  double geoMean;
  double arithMean;
  double maxAbs;

  SmallVector<double, 2> minOperands;
  SmallVector<double, 2> maxOperands;

  ValueInfo()
      : minRes(std::numeric_limits<double>::max()),
        maxRes(std::numeric_limits<double>::lowest()), executions(0),
        geoMean(0.0), arithMean(0.0), maxAbs(0.0) {}
};

bool extractValueFromLog(const std::string &logPath,
                         const std::string &functionName, size_t blockIdx,
                         size_t instIdx, ValueInfo &data);

bool extractGradFromLog(const std::string &logPath,
                        const std::string &functionName, size_t blockIdx,
                        size_t instIdx, GradInfo &data);

bool isLogged(const std::string &logPath, const std::string &functionName);

#endif // ENZYME_POSEIDON_LOGGER_H