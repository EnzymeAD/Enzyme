// Matmul accuracy model: profile-driven sampling, the memoized accuracy cost
// and the Ozaki-II quantization model.
#include "Flags.h"
#include "Optimize.h"
#include "Utils.h"
#include "MatmulInternal.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

static double sampleCell(const CellStat &c, std::mt19937_64 &rng,
                         unsigned sampleLogBits) {
  if (c.count == 0)
    report_fatal_error("sampleCell: unexpected unprofiled cell");
  assert(c.min <= c.max && "sampleCell: corrupted profile (min > max)");
  if (c.min == c.max)
    return c.min;
  if (sampleLogBits == 0) {
    std::uniform_real_distribution<double> dist(c.min, c.max);
    return dist(rng);
  }
  // Log-uniform magnitude over [maxMag*2^-B, maxMag] plus sign: uniform
  // [min,max] sampling under-represents the many small elements of real
  // matrices, to which fixed-point (Ozaki) error is acutely sensitive.
  double maxMag = std::max(std::fabs(c.min), std::fabs(c.max));
  if (maxMag == 0.0)
    return 0.0;
  double loMag = maxMag * std::ldexp(1.0, -(int)sampleLogBits);
  double lu = std::uniform_real_distribution<double>(std::log(loMag),
                                                     std::log(maxMag))(rng);
  double mag = std::exp(lu);
  bool neg = (c.min < 0.0) &&
             ((c.max <= 0.0) ||
              (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.5));
  return neg ? -mag : mag;
}

// Percentile of a sample vector: the confidence bound for the call-site
// domain-error estimate.
static double percentile(std::vector<double> &v, double q) {
  if (v.empty())
    return 0.0;
  std::sort(v.begin(), v.end());
  size_t idx = (size_t)std::ceil(q * (double)v.size());
  if (idx > 0)
    idx--;
  if (idx >= v.size())
    idx = v.size() - 1;
  return v[idx];
}

// How the confidence level is spelled in a diagnostic: 0.95 -> "p95".
static std::string percentileLabel(double confidence) {
  std::string s;
  raw_string_ostream os(s);
  os << format("p%g", confidence * 100.0);
  return s;
}

// Accuracy-model generation: bump whenever the body of getMatmulAccuracyCost
// or getOzakiIIAccuracyCost changes the number it returns for existing inputs,
// since the memo below is keyed on the model's inputs only.
static constexpr uint64_t kAccModelGeneration = 1;

// Matmul accuracy cache: the per-candidate cost is a deterministic function of
// the profile, dims, precision params, sample count, seed and sampling
// distribution, so it is memoized to flags::Cache keyed by a hash of exactly
// those (one file per key, race-free across parallel variant builds). The
// distribution term matters: fpOptimize samples log-uniformly over 40 bits
// under an error budget, and a shared memo would otherwise serve the other
// mode's model.
static uint64_t accHashMix(uint64_t h, uint64_t v) {
  h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  return h;
}
static uint64_t accHashD(uint64_t h, double d) {
  uint64_t b;
  std::memcpy(&b, &d, sizeof(b));
  return accHashMix(h, b);
}
static uint64_t accProfileHash(const AbstractMatmul &m,
                               const MatmulProfile &prof,
                               unsigned sampleLogBits) {
  uint64_t h = 1469598103934665603ULL;
  h = accHashMix(h, m.M);
  h = accHashMix(h, m.N);
  h = accHashMix(h, m.K);
  auto hashCells = [&](const std::vector<CellStat> &v) {
    h = accHashMix(h, v.size());
    for (const auto &c : v) {
      h = accHashD(h, c.min);
      h = accHashD(h, c.max);
      h = accHashD(h, c.sumValue);
      h = accHashMix(h, c.count);
      h = accHashD(h, c.minMag);
    }
  };
  hashCells(prof.a);
  hashCells(prof.b);
  h = accHashMix(h, prof.gradD.size());
  for (const auto &g : prof.gradD) {
    h = accHashD(h, g.sumGrad);
    h = accHashMix(h, g.count);
  }
  h = accHashMix(h, flags::NumSamples);
  h = accHashMix(h, (uint64_t)flags::RandomSeed);
  h = accHashMix(h, (uint64_t)sampleLogBits);
  h = accHashMix(h, kAccModelGeneration);
  return h;
}
static bool accCacheLookup(uint64_t key, double &acc, double &dom) {
  if (flags::Cache.empty())
    return false;
  SmallString<256> p(flags::Cache);
  sys::path::append(p, "matmulacc_" + utohexstr(key) + ".txt");
  auto buf = MemoryBuffer::getFile(p);
  if (!buf)
    return false;
  StringRef s = (*buf)->getBuffer().trim();
  auto [accS, domS] = s.split(' ');
  double a, d;
  if (accS.getAsDouble(a) || domS.getAsDouble(d)) // getAsDouble: true == error
    return false;
  acc = a;
  dom = d;
  return true;
}
static void accCacheStore(uint64_t key, double acc, double dom) {
  if (flags::Cache.empty())
    return;
  (void)sys::fs::create_directories(flags::Cache, true);
  SmallString<256> p(flags::Cache);
  sys::path::append(p, "matmulacc_" + utohexstr(key) + ".txt");
  std::error_code EC;
  raw_fd_ostream os(p, EC, sys::fs::OF_Text);
  if (!EC)
    os << format("%.17g %.17g\n", acc, dom);
}

// Round `x` to `bits` significand bits, keeping the F64 exponent range: an
// emulation scheme's captured precision (16 bits for two BF16 limbs, 22 for two
// FP16/TF32 limbs) is not a hardware FPKind, and rounding to the nearest kind
// is wrong in both directions. The exponent range is charged separately via
// `exponentPrec`.
static double roundToMantBits(double x, unsigned bits) {
  if (bits == 0 || bits >= 53 || !std::isfinite(x) || x == 0.0)
    return x;
  int e;
  std::frexp(x, &e); // |x| in [2^(e-1), 2^e)
  double scale = std::ldexp(1.0, (int)bits - e);
  if (!std::isfinite(scale) || scale == 0.0)
    return x; // subnormal/huge: below any tolerance this model expresses
  return std::nearbyint(x * scale) / scale;
}

// orderTileK (0 = off): simulate the raised loop's accumulation order (per
// k-tile partial dots folded into the accumulator). A same-precision DMMA raise
// is otherwise bit-transparent to the rounding model while the hardware's tile
// order is unspecified; modeling it keeps a zero-cost zero-error candidate out
// of the budget-0 solution. inputMantBits (0 = off): round the operands to that
// many significand bits instead of `inputPrec` (see roundToMantBits).
double getMatmulAccuracyCost(const AbstractMatmul &m, const MatmulProfile &prof,
                             double confidence, unsigned sampleLogBits,
                             FPKind inputPrec, FPKind accPrec,
                             double *domainErrOut, FPKind exponentPrec,
                             unsigned orderTileK, unsigned inputMantBits) {
  const unsigned numSamples =
      std::max<unsigned>(8u, std::min<unsigned>(flags::NumSamples, 256u));
  const size_t mk = (size_t)m.M * m.K;
  const size_t kn = (size_t)m.K * m.N;
  const size_t mn = (size_t)m.M * m.N;
  if (prof.a.size() != mk || prof.b.size() != kn || prof.gradD.size() != mn)
    report_fatal_error(
        "getMatmulAccuracyCost: unexpected profile cell-vector shape");

  uint64_t cacheKey = accProfileHash(m, prof, sampleLogBits);
  cacheKey = accHashMix(cacheKey, 0x6d6dULL); // "mm" discriminator
  cacheKey = accHashMix(cacheKey, (uint64_t)inputPrec);
  cacheKey = accHashMix(cacheKey, (uint64_t)accPrec);
  cacheKey = accHashMix(cacheKey, (uint64_t)exponentPrec);
  cacheKey = accHashD(cacheKey, flags::ExponentPenalty.getValue());
  // The percentile the domain error is read off decides the number stored
  // here, so a cached result is never reused across confidence levels. Mixed
  // only away from the default, so entries banked before the level was a
  // setting stay valid at the level they were computed at.
  if (confidence != kDefaultConfidence)
    cacheKey = accHashD(cacheKey, confidence);
  // Mixed only when active so pre-existing cache entries stay valid.
  if (orderTileK)
    cacheKey = accHashMix(cacheKey, 0x746b00ULL | orderTileK); // "tk" tag
  if (inputMantBits)
    cacheKey = accHashMix(cacheKey, 0x6d6200ULL | inputMantBits); // "mb" tag
  {
    double cAcc, cDom;
    if (accCacheLookup(cacheKey, cAcc, cDom)) {
      if (domainErrOut)
        *domainErrOut = cDom;
      return cAcc;
    }
  }

  std::mt19937_64 rng(flags::RandomSeed);
  unsigned overflowed = 0;
  std::vector<double> Ad(mk), Bd(kn);
  std::vector<double> Ar(mk), Br(kn);
  std::vector<double> gold(mn), cand(mn);

  double sumSampleErr = 0.0;
  unsigned sampleCount = 0;
  std::vector<double> relSamples; // per-sample relative domain error
  relSamples.reserve(numSamples);
  for (unsigned s = 0; s < numSamples; ++s) {
    for (size_t i = 0; i < mk; ++i) {
      Ad[i] = sampleCell(prof.a[i], rng, sampleLogBits);
      Ar[i] = inputMantBits ? roundToMantBits(Ad[i], inputMantBits)
                            : roundToPrec(Ad[i], inputPrec);
    }
    for (size_t i = 0; i < kn; ++i) {
      Bd[i] = sampleCell(prof.b[i], rng, sampleLogBits);
      Br[i] = inputMantBits ? roundToMantBits(Bd[i], inputMantBits)
                            : roundToPrec(Bd[i], inputPrec);
    }

    for (unsigned mi = 0; mi < m.M; ++mi) {
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accD = 0.0;
        double accC = 0.0;
        double tilePartial = 0.0; // orderTileK: per-k-tile partial dot
        for (unsigned ki = 0; ki < m.K; ++ki) {
          accD += Ad[mi * m.K + ki] * Bd[ki * m.N + ni];
          double prod = Ar[mi * m.K + ki] * Br[ki * m.N + ni];
          if (orderTileK) {
            // Raised order: dot within the k-tile, then fold the tile partial
            // into the running accumulator (mirrors per-tile mma into acc).
            tilePartial = roundToPrec(tilePartial + prod, accPrec);
            if ((ki + 1) % orderTileK == 0 || ki + 1 == m.K) {
              accC = roundToPrec(accC + tilePartial, accPrec);
              tilePartial = 0.0;
            }
          } else {
            accC = roundToPrec(accC + prod, accPrec);
          }
        }
        gold[mi * m.N + ni] = accD;
        cand[mi * m.N + ni] = accC;
      }
    }

    double sampleErr = 0.0, sampleDen = 0.0;
    for (size_t i = 0; i < mn; ++i) {
      double g = std::fabs(prof.gradD[i].sumGrad);
      sampleErr += g * std::fabs(gold[i] - cand[i]);
      sampleDen += g * std::fabs(gold[i]);
    }
    if (std::isfinite(sampleErr)) {
      sumSampleErr += sampleErr;
      ++sampleCount;
    } else {
      ++overflowed;
    }
    // Domain-error estimate (call-site budget): sensitivity-weighted relative
    // output error; the ratio cancels the size/exec scaling of the bare sum.
    if (sampleDen > 0.0 && std::isfinite(sampleErr / sampleDen))
      relSamples.push_back(sampleErr / sampleDen);
  }
  if (sampleCount == 0)
    report_fatal_error(
        "getMatmulAccuracyCost: unexpected all-non-finite samples");
  double acc = sumSampleErr / sampleCount;
  double dom = percentile(relSamples, confidence);
  if (flags::Print) {
    llvm::errs() << "DOMERR matmul in=" << fpKindName(inputPrec);
    if (inputMantBits)
      llvm::errs() << "(" << inputMantBits << "b)";
    llvm::errs() << " acc=" << fpKindName(accPrec) << " "
                 << percentileLabel(confidence) << "_relErr=" << dom
                 << " (sumcost=" << acc << ")\n";
  }

  // Exponent-range adequacy: the absolute error above is blind to a format that
  // flushes the smallest operands to zero, so if the exponent format cannot
  // represent the smallest nonzero operand magnitude the profiler observed,
  // charge flags::ExponentPenalty. Emulation schemes store values in a
  // narrow-exponent base type, so the check uses `exponentPrec` (Invalid =
  // inputPrec).
  FPKind expK = (exponentPrec == FPKind::Invalid) ? inputPrec : exponentPrec;
  double minSub = minSubnormalForKind(expK);
  if (minSub > 0.0) {
    double opMinMag = 0.0;
    for (const auto &c : prof.a)
      if (c.minMag > 0.0 && (opMinMag == 0.0 || c.minMag < opMinMag))
        opMinMag = c.minMag;
    for (const auto &c : prof.b)
      if (c.minMag > 0.0 && (opMinMag == 0.0 || c.minMag < opMinMag))
        opMinMag = c.minMag;
    if (opMinMag > 0.0 && opMinMag < minSub)
      overflowed = 1;
  }
  // A format that overflowed or flushed a profiled operand is refused on the
  // tolerance path as well, not only priced out of the budget path.
  if (overflowed) {
    acc += flags::ExponentPenalty.getValue();
    dom = std::numeric_limits<double>::infinity();
  }
  if (domainErrOut)
    *domainErrOut = dom;
  accCacheStore(cacheKey, acc, dom);
  return acc;
}

// Ozaki Scheme II: the integer K-sum is CRT-reconstructed exactly, so the only
// error is the input quantization, which is per row of A and per column of B to
// `capturedBits` bits, matching the computeSft+fillRes kernel; num_moduli sets
// the captured bits.
double getOzakiIIAccuracyCost(const AbstractMatmul &m,
                              const MatmulProfile &prof, double confidence,
                              unsigned sampleLogBits, unsigned capturedBits,
                              double *domainErrOut) {
  const unsigned numSamples =
      std::max<unsigned>(8u, std::min<unsigned>(flags::NumSamples, 256u));
  const size_t mk = (size_t)m.M * m.K;
  const size_t kn = (size_t)m.K * m.N;
  const size_t mn = (size_t)m.M * m.N;
  if (prof.a.size() != mk || prof.b.size() != kn || prof.gradD.size() != mn)
    report_fatal_error(
        "getOzakiIIAccuracyCost: unexpected profile cell-vector shape");

  uint64_t cacheKey = accProfileHash(m, prof, sampleLogBits);
  cacheKey = accHashMix(cacheKey, 0x6f7aULL); // "oz" discriminator
  cacheKey = accHashMix(cacheKey, capturedBits);
  if (confidence != kDefaultConfidence)
    cacheKey = accHashD(cacheKey, confidence);
  {
    double cAcc, cDom;
    if (accCacheLookup(cacheKey, cAcc, cDom)) {
      if (domainErrOut)
        *domainErrOut = cDom;
      return cAcc;
    }
  }

  std::mt19937_64 rng(flags::RandomSeed);
  unsigned overflowed = 0;
  std::vector<double> Ad(mk), Bd(kn), Ar(mk), Br(kn), gold(mn), cand(mn);
  double sumSampleErr = 0.0;
  unsigned sampleCount = 0;
  std::vector<double> relSamples;
  relSamples.reserve(numSamples);
  for (unsigned s = 0; s < numSamples; ++s) {
    for (size_t i = 0; i < mk; ++i)
      Ad[i] = sampleCell(prof.a[i], rng, sampleLogBits);
    for (size_t i = 0; i < kn; ++i)
      Bd[i] = sampleCell(prof.b[i], rng, sampleLogBits);
    // Quantize A per row and B per column as the real kernel does: each row or
    // column is scaled by 2^(beta-1-ilogb(max)) and rounded to the integer
    // grid, so a small element keeps fewer than beta bits.
    for (unsigned mi = 0; mi < m.M; ++mi) {
      double rowmax = 0.0;
      for (unsigned ki = 0; ki < m.K; ++ki)
        rowmax = std::max(rowmax, std::fabs(Ad[mi * m.K + ki]));
      double scl =
          rowmax > 0.0
              ? std::ldexp(1.0, (int)capturedBits - 1 - std::ilogb(rowmax))
              : 1.0;
      for (unsigned ki = 0; ki < m.K; ++ki)
        Ar[mi * m.K + ki] = std::round(Ad[mi * m.K + ki] * scl) / scl;
    }
    for (unsigned ni = 0; ni < m.N; ++ni) {
      double colmax = 0.0;
      for (unsigned ki = 0; ki < m.K; ++ki)
        colmax = std::max(colmax, std::fabs(Bd[ki * m.N + ni]));
      double scl =
          colmax > 0.0
              ? std::ldexp(1.0, (int)capturedBits - 1 - std::ilogb(colmax))
              : 1.0;
      for (unsigned ki = 0; ki < m.K; ++ki)
        Br[ki * m.N + ni] = std::round(Bd[ki * m.N + ni] * scl) / scl;
    }
    for (unsigned mi = 0; mi < m.M; ++mi)
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accD = 0.0, accC = 0.0;
        for (unsigned ki = 0; ki < m.K; ++ki) {
          accD += Ad[mi * m.K + ki] * Bd[ki * m.N + ni];
          accC += Ar[mi * m.K + ki] * Br[ki * m.N + ni]; // EXACT K-sum
        }
        gold[mi * m.N + ni] = accD;
        cand[mi * m.N + ni] = accC;
      }
    double sampleErr = 0.0, sampleDen = 0.0;
    for (size_t i = 0; i < mn; ++i) {
      double g = std::fabs(prof.gradD[i].sumGrad);
      sampleErr += g * std::fabs(gold[i] - cand[i]);
      sampleDen += g * std::fabs(gold[i]);
    }
    if (std::isfinite(sampleErr)) {
      sumSampleErr += sampleErr;
      ++sampleCount;
    } else {
      ++overflowed;
    }
    if (sampleDen > 0.0 && std::isfinite(sampleErr / sampleDen))
      relSamples.push_back(sampleErr / sampleDen);
  }
  if (sampleCount == 0)
    report_fatal_error(
        "getOzakiIIAccuracyCost: unexpected all-non-finite samples");
  double acc = sumSampleErr / sampleCount;
  double dom = percentile(relSamples, confidence);
  if (overflowed) {
    acc += flags::ExponentPenalty.getValue();
    dom = std::numeric_limits<double>::infinity();
  }
  if (domainErrOut)
    *domainErrOut = dom;
  if (flags::Print)
    llvm::errs() << "DOMERR ozakiII capturedBits=" << capturedBits << " "
                 << percentileLabel(confidence) << "_relErr=" << dom
                 << " (sumcost=" << acc << ")\n";
  accCacheStore(cacheKey, acc, dom);
  return acc;
}

} // namespace poseidon
