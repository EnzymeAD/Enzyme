//===- MatmulProfile.cpp - matmul profiles from scalar records ------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR/Utils.h"
#include "MatmulInternal.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <limits>
#include <vector>

using namespace llvm;

namespace poseidon {

static MatmulProfile loadProfileRaise(const AbstractMatmul &m,
                                      const ProfileMap &profile) {
  const ScalarLoopHandle *h = &m.scalarLoop;
  size_t fmaIdx = readProfIdxMetadata(h->fma);
  auto fmaIt = profile.find(fmaIdx);
  if (fmaIt == profile.end())
    report_fatal_error("no scalar profile record for fma instIdx " +
                       Twine(fmaIdx));
  const ProfileInfo &fmaInfo = fmaIt->second;
  if (fmaInfo.exec == 0)
    report_fatal_error("scalar profile record for fma instIdx " +
                       Twine(fmaIdx) + " has Exec=0");

  const ProfileInfo *abInfo = &fmaInfo;
  size_t abIdx = fmaIdx;
  if (h->fmul) {
    size_t fmulIdx = readProfIdxMetadata(h->fmul);
    auto it = profile.find(fmulIdx);
    if (it == profile.end())
      report_fatal_error("no scalar profile record for fmul instIdx " +
                         Twine(fmulIdx));
    abInfo = &it->second;
    abIdx = fmulIdx;
  }
  if (abInfo->minOperands.size() < 2)
    reportUserError(*(h->fmul ? h->fmul : h->fma),
                    "the profile records " +
                        Twine((unsigned)abInfo->minOperands.size()) +
                        " operand range(s) of instruction " + Twine(abIdx) +
                        ", the multiply of a matrix product, which has two; "
                        "re-run profile generation.");

  MatmulProfile out;

  size_t mkCells = (size_t)m.M * m.K;
  size_t knCells = (size_t)m.K * m.N;
  size_t mnCells = (size_t)m.M * m.N;

  double aMin = abInfo->minOperands[0];
  double aMax = abInfo->maxOperands[0];
  double aMinMag = abInfo->minMagOperands[0];
  out.a.assign(mkCells, CellStat{aMin, aMax, abInfo->exec, aMinMag});

  double bMin = abInfo->minOperands[1];
  double bMax = abInfo->maxOperands[1];
  double bMinMag = abInfo->minMagOperands[1];
  out.b.assign(knCells, CellStat{bMin, bMax, abInfo->exec, bMinMag});

  out.gradD.assign(mnCells, GradCellStat{fmaInfo.sumGrad});

  out.totalCalls = fmaInfo.exec;
  return out;
}

// Origin::HostGemmLoopNest: the contraction is spread over R fma sites (the
// unrolled outer contraction level), so the operand ranges are the union over
// the members and the execution counts their sum.
static MatmulProfile loadProfileHostGemm(const AbstractMatmul &m,
                                         const ProfileMap &profile) {
  const RuntimeGemmHandle &h = *m.hostGemm;
  double aMin = std::numeric_limits<double>::max();
  double aMax = std::numeric_limits<double>::lowest();
  double bMin = aMin, bMax = aMax;
  double aMinMag = 0.0, bMinMag = 0.0;
  double sumGrad = 0.0;
  uint64_t exec = 0;
  for (unsigned r = 0; r < h.fmas.size(); ++r) {
    Instruction *ab = h.fmuls[r] ? h.fmuls[r] : h.fmas[r];
    size_t abIdx = readProfIdxMetadata(ab);
    auto abIt = profile.find(abIdx);
    if (abIt == profile.end())
      report_fatal_error("no scalar profile record for host-GEMM member " +
                         Twine(r) + " (instIdx " + Twine(abIdx) + ")");
    const ProfileInfo &ai = abIt->second;
    if (ai.minOperands.size() < 2)
      reportUserError(
          *ab, "the profile records " + Twine((unsigned)ai.minOperands.size()) +
                   " operand range(s) of instruction " + Twine(abIdx) +
                   ", the multiply of a matrix product, which "
                   "has two; re-run profile generation.");
    size_t fmaIdx = readProfIdxMetadata(h.fmas[r]);
    auto fIt = profile.find(fmaIdx);
    if (fIt == profile.end() || fIt->second.exec == 0)
      report_fatal_error("host-GEMM member " + Twine(r) +
                         " has no profiled execution count (instIdx " +
                         Twine(fmaIdx) + ")");
    // Operand 0 of the fma is whichever load the source wrote first; the A/B
    // roles were settled by the address analysis, so map through it.
    bool aIsOp0 = (h.aLoads[r] == ab->getOperand(0));
    unsigned ao = aIsOp0 ? 0 : 1, bo = aIsOp0 ? 1 : 0;
    aMin = std::min(aMin, ai.minOperands[ao]);
    aMax = std::max(aMax, ai.maxOperands[ao]);
    bMin = std::min(bMin, ai.minOperands[bo]);
    bMax = std::max(bMax, ai.maxOperands[bo]);
    if (ai.minMagOperands[ao] > 0.0)
      aMinMag = aMinMag > 0.0 ? std::min(aMinMag, ai.minMagOperands[ao])
                              : ai.minMagOperands[ao];
    if (ai.minMagOperands[bo] > 0.0)
      bMinMag = bMinMag > 0.0 ? std::min(bMinMag, ai.minMagOperands[bo])
                              : ai.minMagOperands[bo];
    sumGrad += fIt->second.sumGrad;
    exec += fIt->second.exec;
  }

  MatmulProfile out;
  out.a.assign((size_t)m.M * m.K, CellStat{aMin, aMax, exec, aMinMag});
  out.b.assign((size_t)m.K * m.N, CellStat{bMin, bMax, exec, bMinMag});
  out.gradD.assign((size_t)m.M * m.N, GradCellStat{sumGrad});
  out.totalCalls = exec;
  return out;
}

MatmulProfile loadProfile(const AbstractMatmul &m,
                          const ProfileMap &scalarProfile) {
  switch (m.origin) {
  case AbstractMatmul::Origin::Invalid:
    report_fatal_error("Invalid AbstractMatmul origin");
  case AbstractMatmul::Origin::ScalarLoopReduction:
    return loadProfileRaise(m, scalarProfile);
  case AbstractMatmul::Origin::HostGemmLoopNest:
    return loadProfileHostGemm(m, scalarProfile);
  }
  llvm_unreachable("unknown AbstractMatmul origin");
}

} // namespace poseidon
