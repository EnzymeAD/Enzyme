//=- PoseidonProfUtils.h - Profiling utilities for Poseidon
//--------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares profiling-related utilities for the Poseidon optimization
// pass.
//
//===----------------------------------------------------------------------===//

#ifndef ENZYME_POSEIDON_PROF_UTILS_H
#define ENZYME_POSEIDON_PROF_UTILS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

#include <limits>
#include <unordered_map>

using namespace llvm;

extern "C" {
extern llvm::cl::opt<bool> FPOptLooseCoverage;
extern llvm::cl::opt<double> FPOptWidenRange;
}

struct ProfileInfo {
  double minRes;
  double maxRes;
  double sumValue; // Sum of values (not abs)
  double sumSens;  // Sum of sensitivity scores = |grad * value|
  double sumGrad;  // Sum of gradients (not abs)
  unsigned exec;

  SmallVector<double, 2> minOperands;
  SmallVector<double, 2> maxOperands;

  ProfileInfo()
      : minRes(std::numeric_limits<double>::max()),
        maxRes(std::numeric_limits<double>::lowest()), sumValue(0.0),
        sumSens(0.0), sumGrad(0.0), exec(0) {}
};

void parseProfileFile(const std::string &profilePath,
                     std::unordered_map<size_t, ProfileInfo> &profileMap);

#endif // ENZYME_POSEIDON_PROF_UTILS_H