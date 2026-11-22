//===- MatmulAccuracy.cpp - accuracy of matrix-product candidates ---------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Matmul accuracy model: profile-driven sampling and the arithmetic of each
// candidate on the samples. The model's identity is a digest of the files
// POSEIDON_ACCURACY_MODEL_FILES names (CMakeLists.txt), so they hold only what
// decides the numbers it returns; storing and finding those numbers is
// MatmulAccuracyMemo's.
//
//===----------------------------------------------------------------------===//

#include "Core/Parallel.h"
#include "Core/UniformReal.h"
#include "MatmulAccuracyMemo.h"
#include "MatmulInternal.h"
#include "MatmulSimulation.h"
#include "Pass/Session.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

namespace poseidon {

namespace {
// How one operand cell is drawn, fixed by its profile before the first draw.
struct CellSampler {
  enum class Kind { Fixed, Uniform, LogUniform };
  enum class Sign { Positive, Negative, Drawn };
  Kind kind = Kind::Fixed;
  Sign sign = Sign::Positive;
  // Fixed: lo. Uniform: [lo, hi]. LogUniform: the range of the log-magnitude.
  double lo = 0.0, hi = 0.0;

  unsigned draws() const {
    switch (kind) {
    case Kind::Fixed:
      return 0;
    case Kind::Uniform:
      return 1;
    case Kind::LogUniform:
      return sign == Sign::Drawn ? 2 : 1;
    }
    llvm_unreachable("unknown CellSampler::Kind");
  }

  double sample(std::mt19937_64 &rng) const {
    switch (kind) {
    case Kind::Fixed:
      return lo;
    case Kind::Uniform:
      return uniformReal(rng, lo, hi);
    case Kind::LogUniform: {
      double mag = std::exp(uniformReal(rng, lo, hi));
      bool neg = sign == Sign::Negative ||
                 (sign == Sign::Drawn && uniformReal(rng, 0.0, 1.0) < 0.5);
      return neg ? -mag : mag;
    }
    }
    llvm_unreachable("unknown CellSampler::Kind");
  }
};
} // namespace

static CellSampler cellSampler(const CellStat &c, unsigned sampleLogBits) {
  if (c.count == 0)
    report_fatal_error("sampleCell: unexpected unprofiled cell");
  assert(c.min <= c.max && "sampleCell: corrupted profile (min > max)");
  CellSampler s;
  if (c.min == c.max) {
    s.lo = c.min;
    return s;
  }
  if (sampleLogBits == 0) {
    s.kind = CellSampler::Kind::Uniform;
    s.lo = c.min;
    s.hi = c.max;
    return s;
  }
  // Log-uniform magnitude over [maxMag*2^-B, maxMag] plus sign: uniform
  // [min,max] sampling under-represents the many small elements of real
  // matrices, to which fixed-point (Ozaki) error is acutely sensitive.
  double maxMag = std::max(std::fabs(c.min), std::fabs(c.max));
  if (maxMag == 0.0)
    return s;
  double loMag = maxMag * std::ldexp(1.0, -(int)sampleLogBits);
  s.kind = CellSampler::Kind::LogUniform;
  s.lo = std::log(loMag);
  s.hi = std::log(maxMag);
  s.sign = !(c.min < 0.0) ? CellSampler::Sign::Positive
           : c.max <= 0.0 ? CellSampler::Sign::Negative
                          : CellSampler::Sign::Drawn;
  return s;
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

static StringRef modelName(const MatmulAccuracyQuery &q) {
  switch (q.model) {
  case MatmulAccuracyQuery::Model::Rounded:
    return "getMatmulAccuracyCost";
  case MatmulAccuracyQuery::Model::TcecLibrary:
    return "getTcecLibraryAccuracyCost";
  case MatmulAccuracyQuery::Model::OzakiII:
    return "getOzakiIIAccuracyCost";
  }
  llvm_unreachable("unknown MatmulAccuracyQuery::Model");
}

// The parts of a query the sampled products read: the exponent format is
// checked against the profile afterwards, and operands rounded to a bit count
// never see the input format.
static MatmulAccuracyQuery sampledPart(MatmulAccuracyQuery q) {
  q.exponentPrec = FPKind::Invalid;
  if (q.inputMantBits)
    q.inputPrec = FPKind::Invalid;
  return q;
}

static bool operator==(const MatmulAccuracyQuery &a,
                       const MatmulAccuracyQuery &b) {
  auto fields = [](const MatmulAccuracyQuery &q) {
    return std::make_tuple(q.model, q.inputPrec, q.accPrec, q.exponentPrec,
                           q.orderTileK, q.inputMantBits, q.mode, q.moduli);
  };
  return fields(a) == fields(b);
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

// A Rounded query's product: operands rounded once, then every
// multiply-accumulate rounded to the accumulator format. orderTileK (0 = off)
// simulates the raised loop's accumulation order (per k-tile partial dots
// folded into the accumulator). A same-precision DMMA raise is otherwise
// bit-transparent to the rounding model while the hardware's tile order is
// unspecified; modeling it keeps a zero-cost zero-error candidate out of the
// budget-0 solution.
static void roundedProduct(const AbstractMatmul &m,
                           const MatmulAccuracyQuery &q, const double *A,
                           const double *B, double *Ar, double *Br, double *C) {
  const size_t mk = (size_t)m.M * m.K, kn = (size_t)m.K * m.N;
  const Rounding toInput(q.inputPrec), toAcc(q.accPrec);
  for (size_t i = 0; i < mk; ++i)
    Ar[i] = q.inputMantBits ? roundToMantBits(A[i], q.inputMantBits)
                            : toInput(A[i]);
  for (size_t i = 0; i < kn; ++i)
    Br[i] = q.inputMantBits ? roundToMantBits(B[i], q.inputMantBits)
                            : toInput(B[i]);
  const unsigned tileK = q.orderTileK;
  for (unsigned mi = 0; mi < m.M; ++mi) {
    const double *row = Ar + (size_t)mi * m.K;
    for (unsigned ni = 0; ni < m.N; ++ni) {
      const double *col = Br + ni;
      double accC = 0.0;
      double tilePartial = 0.0;
      for (unsigned ki = 0; ki < m.K; ++ki) {
        double prod = row[ki] * col[(size_t)ki * m.N];
        if (tileK) {
          // Raised order: dot within the k-tile, then fold the tile partial
          // into the running accumulator (mirrors per-tile mma into acc).
          tilePartial = toAcc(tilePartial + prod);
          if ((ki + 1) % tileK == 0 || ki + 1 == m.K) {
            accC = toAcc(accC + tilePartial);
            tilePartial = 0.0;
          }
        } else {
          accC = toAcc(accC + prod);
        }
      }
      C[(size_t)mi * m.N + ni] = accC;
    }
  }
}

namespace {
// What one candidate's product returns on the sampled operands, against the
// FP64 product.
struct SampledError {
  double accuracyCost = 0.0;
  double domainError = 0.0;
  // Some sample's error was not finite: the candidate overflowed.
  bool nonFinite = false;
};

// The products of distinct sampled parts on the same samples. Every query of
// one product draws the same operands, from the same seed and profile, so the
// operands and the FP64 product are computed once per sample. A sample owns
// its random stream, its operands and its slot of every result, so the
// numbers do not depend on how the samples are spread over threads.
class SampledProducts {
public:
  SampledProducts(const Options &O, const AbstractMatmul &m,
                  const MatmulProfile &prof, unsigned sampleLogBits,
                  unsigned numSamples, ArrayRef<MatmulAccuracyQuery> parts)
      : m(m), prof(prof), parts(parts), numSamples(numSamples),
        err(parts.size() * numSamples), den(numSamples) {
    cells.reserve(prof.a.size() + prof.b.size());
    unsigned long long draws = 0;
    for (const std::vector<CellStat> *v : {&prof.a, &prof.b})
      for (const CellStat &c : *v) {
        cells.push_back(cellSampler(c, sampleLogBits));
        draws += cells.back().draws();
      }
    std::mt19937_64 rng(O.Model.RandomSeed);
    streams.reserve(numSamples);
    for (unsigned s = 0; s < numSamples; ++s) {
      streams.push_back(rng);
      rng.discard(draws);
    }
    forEachIndex(O.Model.Threads, numSamples,
                 [this](size_t s) { runSample(s); });
  }

  // The error of parts[j], reduced over the samples in their order.
  SampledError error(size_t j, double confidence) const {
    SampledError result;
    double sumSampleErr = 0.0;
    unsigned finiteSamples = 0;
    std::vector<double> relSamples;
    relSamples.reserve(numSamples);
    for (unsigned s = 0; s < numSamples; ++s) {
      double sampleErr = err[j * numSamples + s];
      double sampleDen = den[s];
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
      report_fatal_error(modelName(parts[j]) +
                         ": unexpected all-non-finite samples");
    result.accuracyCost = sumSampleErr / finiteSamples;
    result.domainError = percentile(relSamples, confidence);
    return result;
  }

private:
  // Per sample, the error is the gradient-weighted absolute error summed over
  // the outputs, and `den` the gradient-weighted FP64 result it is relative
  // to.
  void runSample(unsigned s) {
    const size_t mk = (size_t)m.M * m.K;
    const size_t kn = (size_t)m.K * m.N;
    const size_t mn = (size_t)m.M * m.N;
    std::mt19937_64 rng = streams[s];
    std::vector<double> A(mk), B(kn), gold(mn), cand(mn), Ar, Br;
    for (size_t i = 0; i < mk; ++i)
      A[i] = cells[i].sample(rng);
    for (size_t i = 0; i < kn; ++i)
      B[i] = cells[mk + i].sample(rng);
    for (unsigned mi = 0; mi < m.M; ++mi)
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accD = 0.0;
        for (unsigned ki = 0; ki < m.K; ++ki)
          accD += A[mi * m.K + ki] * B[ki * m.N + ni];
        gold[mi * m.N + ni] = accD;
      }
    double sampleDen = 0.0;
    for (size_t i = 0; i < mn; ++i)
      sampleDen += std::fabs(prof.gradD[i].sumGrad) * std::fabs(gold[i]);
    den[s] = sampleDen;

    for (size_t j = 0; j < parts.size(); ++j) {
      const MatmulAccuracyQuery &q = parts[j];
      switch (q.model) {
      case MatmulAccuracyQuery::Model::Rounded:
        Ar.resize(mk);
        Br.resize(kn);
        roundedProduct(m, q, A.data(), B.data(), Ar.data(), Br.data(),
                       cand.data());
        break;
      case MatmulAccuracyQuery::Model::TcecLibrary:
        simulateTcecDgemm(A.data(), B.data(), m.M, m.N, m.K, cand.data());
        break;
      case MatmulAccuracyQuery::Model::OzakiII:
        // As runtime/ozaki/ozaki_rt.cu computes it on row-major operands: the
        // shifts of A's rows from sums of squares over contiguous vectors,
        // those of B's columns over strided ones.
        simulateOzakiIIDgemm(
            q.moduli, {A.data(), m.K, m.M, /*contiguous=*/true},
            {B.data(), m.N, m.N, /*contiguous=*/false}, m.K, cand.data());
        break;
      }
      double sampleErr = 0.0;
      for (size_t i = 0; i < mn; ++i)
        sampleErr +=
            std::fabs(prof.gradD[i].sumGrad) * std::fabs(gold[i] - cand[i]);
      err[j * numSamples + s] = sampleErr;
    }
  }

  const AbstractMatmul &m;
  const MatmulProfile &prof;
  ArrayRef<MatmulAccuracyQuery> parts;
  const unsigned numSamples;
  std::vector<CellSampler> cells;
  std::vector<std::mt19937_64> streams;
  std::vector<double> err, den;
};
} // namespace

static MatmulAccuracyMemo memoOf(const Options &O, const AbstractMatmul &m,
                                 const MatmulProfile &prof, double confidence,
                                 unsigned sampleLogBits, unsigned numSamples,
                                 const MatmulAccuracyQuery &q) {
  switch (q.model) {
  case MatmulAccuracyQuery::Model::Rounded:
    return MatmulAccuracyMemo::matmul(
        O, m, prof, confidence, sampleLogBits, numSamples, q.inputPrec,
        q.accPrec, q.exponentPrec, q.orderTileK, q.inputMantBits);
  case MatmulAccuracyQuery::Model::TcecLibrary:
    return MatmulAccuracyMemo::tcecLibrary(O, m, prof, confidence,
                                           sampleLogBits, numSamples);
  case MatmulAccuracyQuery::Model::OzakiII:
    return MatmulAccuracyMemo::ozakiII(O, m, prof, confidence, sampleLogBits,
                                       numSamples, q.moduli);
  }
  llvm_unreachable("unknown MatmulAccuracyQuery::Model");
}

// A query's result from the error of its sampled part. Per sample, the error
// is the gradient-weighted absolute error summed over the outputs; the
// accuracy cost is its mean over the samples where it is finite, and the
// domain error (call-site budget) the `confidence` percentile of its ratio to
// the weighted result, which cancels the size/exec scaling of the bare sum.
static MatmulAccuracy finish(const Options &O, const MatmulProfile &prof,
                             double confidence, const MatmulAccuracyQuery &q,
                             const SampledError &e) {
  MatmulAccuracy r{e.accuracyCost, e.domainError};
  bool outOfRange = e.nonFinite;
  switch (q.model) {
  case MatmulAccuracyQuery::Model::Rounded: {
    if (O.Diagnostics.Print) {
      llvm::errs() << "DOMERR matmul in=" << fpKindName(q.inputPrec);
      if (q.inputMantBits)
        llvm::errs() << "(" << q.inputMantBits << "b)";
      llvm::errs() << " acc=" << fpKindName(q.accPrec) << " "
                   << percentileLabel(confidence) << "_relErr=" << r.domainError
                   << " (sumcost=" << r.accuracyCost << ")\n";
    }
    // Exponent-range adequacy: the absolute error above is blind to a format
    // that flushes the smallest operands to zero, so if the exponent format
    // cannot represent the smallest nonzero operand magnitude the profiler
    // observed, charge -poseidon-exponent-penalty. The operands reach
    // `exponentPrec` (Invalid = inputPrec) unscaled: a scheme that scales
    // them first has its own simulation.
    FPKind expK =
        (q.exponentPrec == FPKind::Invalid) ? q.inputPrec : q.exponentPrec;
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
    break;
  }
  case MatmulAccuracyQuery::Model::TcecLibrary:
  case MatmulAccuracyQuery::Model::OzakiII:
    break;
  }
  // A format that overflowed or flushed a profiled operand is refused on the
  // tolerance path as well, not only priced out of the budget path.
  if (outOfRange) {
    r.accuracyCost += O.Model.ExponentPenalty;
    r.domainError = std::numeric_limits<double>::infinity();
  }
  if (O.Diagnostics.Print) {
    if (q.model == MatmulAccuracyQuery::Model::TcecLibrary)
      llvm::errs() << "DOMERR tcec library " << percentileLabel(confidence)
                   << "_relErr=" << r.domainError
                   << " (sumcost=" << r.accuracyCost << ")\n";
    else if (q.model == MatmulAccuracyQuery::Model::OzakiII)
      llvm::errs() << "DOMERR ozakiII nm=" << q.moduli << " "
                   << percentileLabel(confidence) << "_relErr=" << r.domainError
                   << " (sumcost=" << r.accuracyCost << ")\n";
  }
  return r;
}

void evaluateMatmulAccuracy(Session &S, const AbstractMatmul &m,
                            const MatmulProfile &prof, double confidence,
                            unsigned sampleLogBits,
                            ArrayRef<MatmulAccuracyQuery> queries,
                            MutableArrayRef<MatmulAccuracy> results) {
  assert(queries.size() == results.size() && "one result per query");
  const Options &O = S.Opts;
  const unsigned numSamples = sampleCount(O);
  for (const MatmulAccuracyQuery &q : queries) {
    checkProfileShape(m, prof, modelName(q));
    if (q.model == MatmulAccuracyQuery::Model::TcecLibrary && q.mode != 1)
      report_fatal_error("the matrix accuracy model simulates "
                         "__poseidon_tcec_dgemm in mode 1 (FP16TCEC) only");
  }

  // Looked up in query order; a query whose entry an earlier query of this
  // product computes takes that result, as it would have read it back.
  std::vector<MatmulAccuracyMemo> memos;
  SmallVector<std::optional<size_t>> sameEntryAs(queries.size());
  SmallVector<size_t> computed;
  for (size_t i = 0; i < queries.size(); ++i) {
    memos.push_back(
        memoOf(O, m, prof, confidence, sampleLogBits, numSamples, queries[i]));
    if (!O.Herbie.Cache.empty())
      for (size_t c : computed)
        if (memos[c].file() == memos[i].file()) {
          sameEntryAs[i] = c;
          break;
        }
    if (sameEntryAs[i])
      continue;
    if (memos[i].lookup(S, results[i].accuracyCost, results[i].domainError))
      continue;
    computed.push_back(i);
  }
  if (computed.empty())
    return;

  SmallVector<MatmulAccuracyQuery> parts;
  SmallVector<size_t> partOf;
  for (size_t i : computed) {
    MatmulAccuracyQuery part = sampledPart(queries[i]);
    auto it = llvm::find(parts, part);
    partOf.push_back(it - parts.begin());
    if (it == parts.end())
      parts.push_back(part);
  }
  const SampledProducts products(O, m, prof, sampleLogBits, numSamples, parts);
  for (size_t c = 0; c < computed.size(); ++c) {
    const size_t i = computed[c];
    results[i] = finish(O, prof, confidence, queries[i],
                        products.error(partOf[c], confidence));
    memos[i].store(O, results[i].accuracyCost, results[i].domainError);
  }
  for (size_t i = 0; i < queries.size(); ++i)
    if (sameEntryAs[i])
      results[i] = results[*sameEntryAs[i]];
}

} // namespace poseidon
