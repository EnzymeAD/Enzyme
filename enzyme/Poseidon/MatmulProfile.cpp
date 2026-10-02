// Matmul profile loading from the scalar .fpprofile records.
#include "Flags.h"
#include "Utils.h"
#include "MatmulInternal.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace llvm;

namespace poseidon {

static void resizeCells(MatmulProfile &p) {
  p.a.assign((size_t)p.M * p.K, CellStat{});
  p.b.assign((size_t)p.K * p.N, CellStat{});
  p.c.assign((size_t)p.M * p.N, CellStat{});
  p.d.assign((size_t)p.M * p.N, CellStat{});
  p.gradD.assign((size_t)p.M * p.N, GradCellStat{});
}

static MatmulProfile
loadProfileRaise(const AbstractMatmul &m,
                 const std::unordered_map<size_t, ProfileInfo> &profile) {
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
    report_fatal_error("expected >= 2 operand stats on fmul/fma instIdx " +
                       Twine(abIdx) + ", got " +
                       Twine((unsigned)abInfo->minOperands.size()));

  MatmulProfile out;
  out.M = m.M;
  out.N = m.N;
  out.K = m.K;
  out.aType = m.aType;
  out.bType = m.bType;
  out.cType = m.accType;
  out.dType = m.dType;
  resizeCells(out);

  size_t mkCells = (size_t)m.M * m.K;
  size_t knCells = (size_t)m.K * m.N;
  size_t mnCells = (size_t)m.M * m.N;

  double aMin = abInfo->minOperands[0];
  double aMax = abInfo->maxOperands[0];
  double aMag = std::max(std::fabs(aMin), std::fabs(aMax));
  double aMinMag =
      abInfo->minMagOperands.size() > 0 ? abInfo->minMagOperands[0] : 0.0;
  out.a.assign(mkCells, CellStat{aMin, aMax, aMag * (double)abInfo->exec,
                                 abInfo->exec, aMinMag});

  double bMin = abInfo->minOperands[1];
  double bMax = abInfo->maxOperands[1];
  double bMag = std::max(std::fabs(bMin), std::fabs(bMax));
  double bMinMag =
      abInfo->minMagOperands.size() > 1 ? abInfo->minMagOperands[1] : 0.0;
  out.b.assign(knCells, CellStat{bMin, bMax, bMag * (double)abInfo->exec,
                                 abInfo->exec, bMinMag});

  out.gradD.assign(mnCells, GradCellStat{fmaInfo.sumGrad, fmaInfo.exec});

  out.totalCalls = fmaInfo.exec;
  return out;
}

// Origin::HostGemmLoopNest. Same construction as loadProfileRaise, but the
// contraction is spread over R fma sites (the unrolled outer contraction
// level), so the operand ranges are the UNION over the members and the
// execution counts are their sum. Taking one member's record instead would
// under-report both the value range the accuracy model samples from and the
// MAC count the cost model scales by.
static MatmulProfile
loadProfileHostGemm(const AbstractMatmul &m,
                    const std::unordered_map<size_t, ProfileInfo> &profile) {
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
      report_fatal_error("expected >= 2 operand stats on host-GEMM member " +
                         Twine(r) + " (instIdx " + Twine(abIdx) + ")");
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
    if (ai.minMagOperands.size() > ao && ai.minMagOperands[ao] > 0.0)
      aMinMag = aMinMag > 0.0 ? std::min(aMinMag, ai.minMagOperands[ao])
                              : ai.minMagOperands[ao];
    if (ai.minMagOperands.size() > bo && ai.minMagOperands[bo] > 0.0)
      bMinMag = bMinMag > 0.0 ? std::min(bMinMag, ai.minMagOperands[bo])
                              : ai.minMagOperands[bo];
    sumGrad += fIt->second.sumGrad;
    exec += fIt->second.exec;
  }

  MatmulProfile out;
  out.M = m.M;
  out.N = m.N;
  out.K = m.K;
  out.aType = m.aType;
  out.bType = m.bType;
  out.cType = m.accType;
  out.dType = m.dType;
  resizeCells(out);
  double aMag = std::max(std::fabs(aMin), std::fabs(aMax));
  double bMag = std::max(std::fabs(bMin), std::fabs(bMax));
  out.a.assign((size_t)m.M * m.K,
               CellStat{aMin, aMax, aMag * (double)exec, exec, aMinMag});
  out.b.assign((size_t)m.K * m.N,
               CellStat{bMin, bMax, bMag * (double)exec, exec, bMinMag});
  out.gradD.assign((size_t)m.M * m.N, GradCellStat{sumGrad, exec});
  out.totalCalls = exec;
  return out;
}

MatmulProfile
loadProfile(const AbstractMatmul &m,
            const std::unordered_map<size_t, ProfileInfo> &scalarProfile) {
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
