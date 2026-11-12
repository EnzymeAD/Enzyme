// Matmul accuracy model: profile-driven sampling and the arithmetic of each
// candidate on the samples. The model's identity is a digest of the files
// POSEIDON_ACCURACY_MODEL_FILES names (CMakeLists.txt), so they hold only what
// decides the numbers it returns; storing and finding those numbers is
// MatmulAccuracyMemo's.
#include "Core/UniformReal.h"
#include "MatmulAccuracyMemo.h"
#include "MatmulInternal.h"
#include "MatmulSimulation.h"
#include "Pass/Session.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
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
  if (sampleLogBits == 0)
    return uniformReal(rng, c.min, c.max);
  // Log-uniform magnitude over [maxMag*2^-B, maxMag] plus sign: uniform
  // [min,max] sampling under-represents the many small elements of real
  // matrices, to which fixed-point (Ozaki) error is acutely sensitive.
  double maxMag = std::max(std::fabs(c.min), std::fabs(c.max));
  if (maxMag == 0.0)
    return 0.0;
  double loMag = maxMag * std::ldexp(1.0, -(int)sampleLogBits);
  double lu = uniformReal(rng, std::log(loMag), std::log(maxMag));
  double mag = std::exp(lu);
  bool neg =
      (c.min < 0.0) && ((c.max <= 0.0) || (uniformReal(rng, 0.0, 1.0) < 0.5));
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

static unsigned sampleCount(const Options &O) {
  return std::max<unsigned>(8u, std::min<unsigned>(O.Model.NumSamples, 256u));
}

static void checkProfileShape(const AbstractMatmul &m,
                              const MatmulProfile &prof, StringRef model) {
  if (prof.a.size() != (size_t)m.M * m.K ||
      prof.b.size() != (size_t)m.K * m.N ||
      prof.gradD.size() != (size_t)m.M * m.N)
    report_fatal_error(model + ": unexpected profile cell-vector shape");
}

namespace {
// A candidate's product of sampled operands: A (M x K) and B (K x N),
// row-major, into C (M x N).
using SimulatedProduct =
    function_ref<void(const std::vector<double> &A,
                      const std::vector<double> &B, std::vector<double> &C)>;

struct SampledError {
  double accuracyCost = 0.0;
  double domainError = 0.0;
  // Some sample's error was not finite: the candidate overflowed.
  bool nonFinite = false;
};
} // namespace

// The candidate's product against the FP64 one on sampled operands. Per
// sample, the error is the gradient-weighted absolute error summed over the
// outputs; the accuracy cost is its mean over the samples where it is finite,
// and the domain error (call-site budget) the `confidence` percentile of its
// ratio to the weighted result, which cancels the size/exec scaling of the
// bare sum.
static SampledError sampleError(const Options &O, const AbstractMatmul &m,
                                const MatmulProfile &prof, double confidence,
                                unsigned sampleLogBits, unsigned numSamples,
                                SimulatedProduct simulate, StringRef model) {
  const size_t mk = (size_t)m.M * m.K;
  const size_t kn = (size_t)m.K * m.N;
  const size_t mn = (size_t)m.M * m.N;
  std::mt19937_64 rng(O.Model.RandomSeed);
  std::vector<double> Ad(mk), Bd(kn), gold(mn), cand(mn);
  SampledError result;
  double sumSampleErr = 0.0;
  unsigned finiteSamples = 0;
  std::vector<double> relSamples;
  relSamples.reserve(numSamples);
  for (unsigned s = 0; s < numSamples; ++s) {
    for (size_t i = 0; i < mk; ++i)
      Ad[i] = sampleCell(prof.a[i], rng, sampleLogBits);
    for (size_t i = 0; i < kn; ++i)
      Bd[i] = sampleCell(prof.b[i], rng, sampleLogBits);
    for (unsigned mi = 0; mi < m.M; ++mi)
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accD = 0.0;
        for (unsigned ki = 0; ki < m.K; ++ki)
          accD += Ad[mi * m.K + ki] * Bd[ki * m.N + ni];
        gold[mi * m.N + ni] = accD;
      }
    simulate(Ad, Bd, cand);

    double sampleErr = 0.0, sampleDen = 0.0;
    for (size_t i = 0; i < mn; ++i) {
      double g = std::fabs(prof.gradD[i].sumGrad);
      sampleErr += g * std::fabs(gold[i] - cand[i]);
      sampleDen += g * std::fabs(gold[i]);
    }
    if (std::isfinite(sampleErr)) {
      sumSampleErr += sampleErr;
      ++finiteSamples;
    } else {
      result.nonFinite = true;
    }
    if (sampleDen > 0.0 && std::isfinite(sampleErr / sampleDen))
      relSamples.push_back(sampleErr / sampleDen);
  }
  if (finiteSamples == 0)
    report_fatal_error(model + ": unexpected all-non-finite samples");
  result.accuracyCost = sumSampleErr / finiteSamples;
  result.domainError = percentile(relSamples, confidence);
  return result;
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
double getMatmulAccuracyCost(Session &S, const AbstractMatmul &m,
                             const MatmulProfile &prof, double confidence,
                             unsigned sampleLogBits, FPKind inputPrec,
                             FPKind accPrec, double *domainErrOut,
                             FPKind exponentPrec, unsigned orderTileK,
                             unsigned inputMantBits) {
  const Options &O = S.Opts;
  checkProfileShape(m, prof, "getMatmulAccuracyCost");
  const unsigned numSamples = sampleCount(O);
  const MatmulAccuracyMemo memo = MatmulAccuracyMemo::matmul(
      O, m, prof, confidence, sampleLogBits, numSamples, inputPrec, accPrec,
      exponentPrec, orderTileK, inputMantBits);
  {
    double cAcc, cDom;
    if (memo.lookup(S, cAcc, cDom)) {
      if (domainErrOut)
        *domainErrOut = cDom;
      return cAcc;
    }
  }

  std::vector<double> Ar, Br;
  auto product = [&](const std::vector<double> &Ad,
                     const std::vector<double> &Bd, std::vector<double> &C) {
    Ar.resize(Ad.size());
    Br.resize(Bd.size());
    for (size_t i = 0; i < Ad.size(); ++i)
      Ar[i] = inputMantBits ? roundToMantBits(Ad[i], inputMantBits)
                            : roundToPrec(Ad[i], inputPrec);
    for (size_t i = 0; i < Bd.size(); ++i)
      Br[i] = inputMantBits ? roundToMantBits(Bd[i], inputMantBits)
                            : roundToPrec(Bd[i], inputPrec);
    for (unsigned mi = 0; mi < m.M; ++mi) {
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accC = 0.0;
        double tilePartial = 0.0; // orderTileK: per-k-tile partial dot
        for (unsigned ki = 0; ki < m.K; ++ki) {
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
        C[mi * m.N + ni] = accC;
      }
    }
  };
  SampledError e = sampleError(O, m, prof, confidence, sampleLogBits,
                               numSamples, product, "getMatmulAccuracyCost");
  double acc = e.accuracyCost;
  double dom = e.domainError;
  if (O.Diagnostics.Print) {
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
  // charge -poseidon-exponent-penalty. The operands reach `exponentPrec`
  // (Invalid = inputPrec) unscaled: a scheme that scales them first has its
  // own simulation.
  bool outOfRange = e.nonFinite;
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
      outOfRange = true;
  }
  // A format that overflowed or flushed a profiled operand is refused on the
  // tolerance path as well, not only priced out of the budget path.
  if (outOfRange) {
    acc += O.Model.ExponentPenalty;
    dom = std::numeric_limits<double>::infinity();
  }
  if (domainErrOut)
    *domainErrOut = dom;
  memo.store(O, acc, dom);
  return acc;
}

double getTcecLibraryAccuracyCost(Session &S, const AbstractMatmul &m,
                                  const MatmulProfile &prof, double confidence,
                                  unsigned sampleLogBits, unsigned mode,
                                  double *domainErrOut) {
  if (mode != 1)
    report_fatal_error("the matrix accuracy model simulates "
                       "__poseidon_tcec_dgemm in mode 1 (FP16TCEC) only");
  const Options &O = S.Opts;
  checkProfileShape(m, prof, "getTcecLibraryAccuracyCost");
  const unsigned numSamples = sampleCount(O);
  const MatmulAccuracyMemo memo = MatmulAccuracyMemo::tcecLibrary(
      O, m, prof, confidence, sampleLogBits, numSamples);
  {
    double cAcc, cDom;
    if (memo.lookup(S, cAcc, cDom)) {
      if (domainErrOut)
        *domainErrOut = cDom;
      return cAcc;
    }
  }

  auto product = [&](const std::vector<double> &Ad,
                     const std::vector<double> &Bd, std::vector<double> &C) {
    simulateTcecDgemm(Ad.data(), Bd.data(), m.M, m.N, m.K, C.data());
  };
  SampledError e =
      sampleError(O, m, prof, confidence, sampleLogBits, numSamples, product,
                  "getTcecLibraryAccuracyCost");
  double acc = e.accuracyCost;
  double dom = e.domainError;
  if (e.nonFinite) {
    acc += O.Model.ExponentPenalty;
    dom = std::numeric_limits<double>::infinity();
  }
  if (domainErrOut)
    *domainErrOut = dom;
  if (O.Diagnostics.Print)
    llvm::errs() << "DOMERR tcec library " << percentileLabel(confidence)
                 << "_relErr=" << dom << " (sumcost=" << acc << ")\n";
  memo.store(O, acc, dom);
  return acc;
}

// Ozaki Scheme II with `moduli` moduli, as runtime/ozaki/ozaki_rt.cu computes
// it on row-major operands: the shifts of A's rows from sums of squares over
// contiguous vectors, those of B's columns over strided ones.
double getOzakiIIAccuracyCost(Session &S, const AbstractMatmul &m,
                              const MatmulProfile &prof, double confidence,
                              unsigned sampleLogBits, unsigned moduli,
                              double *domainErrOut) {
  const Options &O = S.Opts;
  checkProfileShape(m, prof, "getOzakiIIAccuracyCost");
  const unsigned numSamples = sampleCount(O);
  const MatmulAccuracyMemo memo = MatmulAccuracyMemo::ozakiII(
      O, m, prof, confidence, sampleLogBits, numSamples, moduli);
  {
    double cAcc, cDom;
    if (memo.lookup(S, cAcc, cDom)) {
      if (domainErrOut)
        *domainErrOut = cDom;
      return cAcc;
    }
  }

  auto product = [&](const std::vector<double> &Ad,
                     const std::vector<double> &Bd, std::vector<double> &C) {
    simulateOzakiIIDgemm(moduli, {Ad.data(), m.K, m.M, /*contiguous=*/true},
                         {Bd.data(), m.N, m.N, /*contiguous=*/false}, m.K,
                         C.data());
  };
  SampledError e = sampleError(O, m, prof, confidence, sampleLogBits,
                               numSamples, product, "getOzakiIIAccuracyCost");
  double acc = e.accuracyCost;
  double dom = e.domainError;
  if (e.nonFinite) {
    acc += O.Model.ExponentPenalty;
    dom = std::numeric_limits<double>::infinity();
  }
  if (domainErrOut)
    *domainErrOut = dom;
  if (O.Diagnostics.Print)
    llvm::errs() << "DOMERR ozakiII nm=" << moduli << " "
                 << percentileLabel(confidence) << "_relErr=" << dom
                 << " (sumcost=" << acc << ")\n";
  memo.store(O, acc, dom);
  return acc;
}

} // namespace poseidon
