//===- Accuracy.cpp - accuracy of matrix-product candidates ---------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Matmul accuracy model: the operand statistics of a product from its scalar
// profile records, profile-driven sampling, the arithmetic of each candidate
// on the samples, and the memo of the numbers it returns.
//
//===----------------------------------------------------------------------===//

#include "BuildIdentity.h"
#include "Core/Session.h"
#include "MatmulInternal.h"
#include "Simulation.h"
#include "Sites/Canonicalize.h"
#include "Support/CacheKey.h"
#include "Support/Diagnostics.h"
#include "Support/Parallel.h"
#include "Support/UniformReal.h"
#include "Support/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
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

static Expected<std::optional<MatmulProfile>>
loadConstantShapeProfile(const RecognizedMatmul &m, const ProfileMap &profile,
                         ErrorMoment moment) {
  const ScalarLoopHandle *h = &m.scalarLoop();
  size_t fmaIdx = requireSlot(h->fma);
  auto fmaIt = profile.find(fmaIdx);
  if (fmaIt == profile.end() || fmaIt->second.exec == 0)
    return std::optional<MatmulProfile>();
  const ProfileInfo &fmaInfo = fmaIt->second;

  const ProfileInfo *abInfo = &fmaInfo;
  size_t abIdx = fmaIdx;
  if (h->fmul) {
    size_t fmulIdx = requireSlot(h->fmul);
    auto it = profile.find(fmulIdx);
    if (it == profile.end())
      return userError(*h->fmul, "no scalar profile record for fmul instIdx " +
                                     Twine(fmulIdx));
    abInfo = &it->second;
    abIdx = fmulIdx;
    // The fma reads the multiply's result, so a run that executed the fma
    // executed the multiply.
    if (abInfo->exec == 0)
      return userError(*h->fmul, "the profile records no execution of "
                                 "instruction " +
                                     Twine(fmulIdx) +
                                     ", the multiply of a matrix product whose "
                                     "accumulation it records executed; "
                                     "re-run profile generation");
  }
  if (abInfo->minOperands.size() < 2)
    return userError(*(h->fmul ? h->fmul : h->fma),
                     "the profile records " +
                         Twine((unsigned)abInfo->minOperands.size()) +
                         " operand range(s) of instruction " + Twine(abIdx) +
                         ", the multiply of a matrix product, which has two; "
                         "re-run profile generation");

  MatmulProfile out;
  out.a = {abInfo->minOperands[0], abInfo->maxOperands[0],
           abInfo->minMagOperands[0]};
  out.b = {abInfo->minOperands[1], abInfo->maxOperands[1],
           abInfo->minMagOperands[1]};
  out.gradSum = fmaInfo.sumAbsGrad;
  out.sqGradSum = fmaInfo.sumSqGrad;
  out.moment = moment;
  out.macs = fmaInfo.exec;
  return out;
}

// A runtime-shape product: the contraction is spread over R fma sites (the
// unrolled outer contraction level), so the operand ranges are the union over
// the members and the execution counts their sum.
static Expected<std::optional<MatmulProfile>>
loadRuntimeShapeProfile(const RecognizedMatmul &m, const ProfileMap &profile,
                        ErrorMoment moment) {
  const RuntimeGemmHandle &h = m.runtimeGemm();
  for (Instruction *fma : h.fmas) {
    auto it = profile.find(requireSlot(fma));
    if (it == profile.end() || it->second.exec == 0)
      return std::optional<MatmulProfile>();
  }
  double aMin = std::numeric_limits<double>::max();
  double aMax = std::numeric_limits<double>::lowest();
  double bMin = aMin, bMax = aMax;
  double aMinMag = 0.0, bMinMag = 0.0;
  double gradSum = 0.0, sqGradSum = 0.0;
  uint64_t exec = 0;
  for (unsigned r = 0; r < h.fmas.size(); ++r) {
    Instruction *ab = h.fmuls[r] ? h.fmuls[r] : h.fmas[r];
    size_t abIdx = requireSlot(ab);
    auto abIt = profile.find(abIdx);
    if (abIt == profile.end())
      return userError(
          *ab, "no scalar profile record for runtime-shape product member " +
                   Twine(r) + " (instIdx " + Twine(abIdx) + ")");
    const ProfileInfo &ai = abIt->second;
    if (ai.minOperands.size() < 2)
      return userError(
          *ab, "the profile records " + Twine((unsigned)ai.minOperands.size()) +
                   " operand range(s) of instruction " + Twine(abIdx) +
                   ", the multiply of a matrix product, which "
                   "has two; re-run profile generation");
    auto fIt = profile.find(requireSlot(h.fmas[r]));
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
    gradSum += fIt->second.sumAbsGrad;
    sqGradSum += fIt->second.sumSqGrad;
    exec += fIt->second.exec;
  }

  MatmulProfile out;
  out.a = {aMin, aMax, aMinMag};
  out.b = {bMin, bMax, bMinMag};
  out.gradSum = gradSum;
  out.sqGradSum = sqGradSum;
  out.moment = moment;
  out.macs = exec;
  return out;
}

Expected<std::optional<MatmulProfile>>
loadProfile(const RecognizedMatmul &m, const ProfileMap &scalarProfile,
            ErrorMoment moment) {
  if (m.isConstantShape())
    return loadConstantShapeProfile(m, scalarProfile, moment);
  return loadRuntimeShapeProfile(m, scalarProfile, moment);
}

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
// domain-error estimate. No sample bounds nothing, so the error is unbounded.
static double percentile(std::vector<double> &v, double q) {
  if (v.empty())
    return std::numeric_limits<double>::infinity();
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

// The matrix model draws whole operand matrices per sample, so it takes
// -poseidon-num-samples within [8, 256], as the flag says.
constexpr unsigned kMinMatmulSamples = 8, kMaxMatmulSamples = 256;
static unsigned sampleCount(const Options &O) {
  return std::clamp(O.Model.NumSamples, kMinMatmulSamples, kMaxMatmulSamples);
}

// The query as diagnostics name it, e.g. "matmul in=f16 acc=f32", or
// "matmul in=f16 min=6.103516e-05 acc=f32" with a floor above the format's
// subnormals.
static std::string describe(const MatmulAccuracyQuery &q) {
  std::string floor;
  if (q.operandMin > 0.0)
    raw_string_ostream(floor) << " min=" << q.operandMin;
  switch (q.model) {
  case MatmulAccuracyQuery::Model::Rounded: {
    std::string s = "matmul in=" + std::string(dtypeName(q.inputPrec));
    if (q.inputMantBits)
      s += "(" + std::to_string(q.inputMantBits) + "b)";
    return s + floor + " acc=" + dtypeName(q.accPrec);
  }
  case MatmulAccuracyQuery::Model::TcecLibrary:
    return "tcec library" + floor;
  case MatmulAccuracyQuery::Model::OzakiII:
    return "ozakiII nm=" + std::to_string(q.moduli);
  }
  llvm_unreachable("unknown MatmulAccuracyQuery::Model");
}

// The parts of a query the sampled products read: the exponent format is
// checked against the profile afterwards, and operands rounded to a bit count
// never see the input format.
static MatmulAccuracyQuery sampledPart(MatmulAccuracyQuery q) {
  q.exponentPrec = NumberFormat::Invalid;
  if (q.inputMantBits)
    q.inputPrec = NumberFormat::Invalid;
  return q;
}

static bool operator==(const MatmulAccuracyQuery &a,
                       const MatmulAccuracyQuery &b) {
  auto fields = [](const MatmulAccuracyQuery &q) {
    return std::make_tuple(q.model, q.inputPrec, q.accPrec, q.exponentPrec,
                           q.orderTileK, q.inputMantBits, q.moduli,
                           q.operandMin);
  };
  return fields(a) == fields(b);
}

// Round `x` to `bits` significand bits, keeping the F64 exponent range: an
// emulation scheme's captured precision (16 bits for two BF16 limbs, 22 for two
// FP16/TF32 limbs) is not a hardware NumberFormat, and rounding to the nearest
// kind is wrong in both directions. The exponent range is charged separately
// via `exponentPrec`.
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

// A Rounded query's product: operands rounded once, those below operandMin
// read as zero, then every multiply-accumulate rounded to the accumulator
// format. orderTileK (0 = off)
// simulates the raised loop's accumulation order (per k-tile partial dots
// folded into the accumulator). A same-precision DMMA raise is otherwise
// bit-transparent to the rounding model while the hardware's tile order is
// unspecified; modeling it keeps a zero-cost zero-error candidate out of the
// budget-0 solution.
static void roundedProduct(const RecognizedMatmul &m,
                           const MatmulAccuracyQuery &q, const double *A,
                           const double *B, double *Ar, double *Br, double *C) {
  const size_t mk = (size_t)m.M * m.K, kn = (size_t)m.K * m.N;
  const Rounding toInput(q.inputPrec), toAcc(q.accPrec);
  auto narrow = [&](double x) {
    double r =
        q.inputMantBits ? roundToMantBits(x, q.inputMantBits) : toInput(x);
    return std::fabs(r) < q.operandMin ? std::copysign(0.0, r) : r;
  };
  for (size_t i = 0; i < mk; ++i)
    Ar[i] = narrow(A[i]);
  for (size_t i = 0; i < kn; ++i)
    Br[i] = narrow(B[i]);
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
  SampledProducts(const Options &O, const RecognizedMatmul &m,
                  const MatmulProfile &prof, unsigned sampleLogBits,
                  unsigned numSamples, ArrayRef<MatmulAccuracyQuery> parts)
      : m(m), prof(prof), parts(parts), numSamples(numSamples),
        aCell(cellSampler(prof.a, sampleLogBits)),
        bCell(cellSampler(prof.b, sampleLogBits)),
        err(parts.size() * numSamples), den(numSamples) {
    const unsigned long long draws =
        (unsigned long long)m.M * m.K * aCell.draws() +
        (unsigned long long)m.K * m.N * bCell.draws();
    std::mt19937_64 rng(O.Model.RandomSeed);
    streams.reserve(numSamples);
    for (unsigned s = 0; s < numSamples; ++s) {
      streams.push_back(rng);
      rng.discard(draws);
    }
    forEachIndex(O.Model.Threads, numSamples,
                 [this](size_t s) { runSample(s); });
  }

  // The error of parts[j], reduced over the samples in their order. No finite
  // sample is an error at the product: its profiled operands overflow every
  // candidate.
  Expected<SampledError> error(size_t j) const {
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
      double rel = relativeError(sampleErr, sampleDen, prof.moment);
      if (sampleDen > 0.0 && std::isfinite(rel))
        relSamples.push_back(rel);
    }
    if (finiteSamples == 0)
      return userError(anchorOf(m), Twine(describe(parts[j])) +
                                        ": unexpected all-non-finite samples");
    result.accuracyCost = sumSampleErr / finiteSamples;
    result.domainError = percentile(relSamples, kDefaultConfidence);
    return result;
  }

private:
  // Per sample, the error is the weighted error summed over the outputs, and
  // `den` the same weighting of the FP64 result it is relative to, both in the
  // profile's moment.
  void runSample(unsigned s) {
    const size_t mk = (size_t)m.M * m.K;
    const size_t kn = (size_t)m.K * m.N;
    const size_t mn = (size_t)m.M * m.N;
    std::mt19937_64 rng = streams[s];
    std::vector<double> A(mk), B(kn), gold(mn), cand(mn), Ar, Br;
    for (size_t i = 0; i < mk; ++i)
      A[i] = aCell.sample(rng);
    for (size_t i = 0; i < kn; ++i)
      B[i] = bCell.sample(rng);
    for (unsigned mi = 0; mi < m.M; ++mi)
      for (unsigned ni = 0; ni < m.N; ++ni) {
        double accD = 0.0;
        for (unsigned ki = 0; ki < m.K; ++ki)
          accD += A[mi * m.K + ki] * B[ki * m.N + ni];
        gold[mi * m.N + ni] = accD;
      }
    const double weight = prof.moment == ErrorMoment::First
                              ? std::fabs(prof.gradSum)
                              : prof.sqGradSum;
    double sampleDen = 0.0;
    for (size_t i = 0; i < mn; ++i)
      sampleDen += weight * momentError(gold[i], 0.0, prof.moment);
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
        simulateTcecDgemm(A.data(), B.data(), m.M, m.N, m.K, q.operandMin,
                          cand.data());
        break;
      case MatmulAccuracyQuery::Model::OzakiII:
        // As runtime/gemm/OzakiII.cu computes it on row-major operands: the
        // shifts of A's rows from sums of squares over contiguous vectors,
        // those of B's columns over strided ones.
        simulateOzakiIIDgemm(
            q.moduli, {A.data(), m.K, m.M, /*contiguous=*/true},
            {B.data(), m.N, m.N, /*contiguous=*/false}, m.K, cand.data());
        break;
      }
      double sampleErr = 0.0;
      for (size_t i = 0; i < mn; ++i)
        sampleErr += weight * momentError(gold[i], cand[i], prof.moment);
      err[j * numSamples + s] = sampleErr;
    }
  }

  const RecognizedMatmul &m;
  const MatmulProfile &prof;
  ArrayRef<MatmulAccuracyQuery> parts;
  const unsigned numSamples;
  // Every element of an operand is drawn the same way.
  const CellSampler aCell, bCell;
  std::vector<std::mt19937_64> streams;
  std::vector<double> err, den;
};
} // namespace

namespace {
// The memo entry of one evaluation of the matrix accuracy model. A result is a
// deterministic function of the model's inputs, so it is kept in
// -poseidon-cache, one entry file per key. The key holds the model's inputs
// and the compiler that computed it, never its results: hosts differ in the
// last digits of a result for equal inputs. An entry is filed under every part
// but the compiler, so one computed by another build is found and reported.
class Memo {
public:
  Memo(const Options &O, const RecognizedMatmul &m, const MatmulProfile &prof,
       unsigned sampleLogBits, unsigned numSamples,
       const MatmulAccuracyQuery &q);

  // The stored result; false when the model has to compute it.
  bool lookup(const Options &O, MatmulAccuracy &result) const;
  void store(const Options &O, const MatmulAccuracy &result) const;
  // The entry's file in -poseidon-cache: two memos of one file hold one
  // result.
  const std::string &file() const { return path; }

private:
  CacheKey key;
  std::string path;
};
} // namespace

template <typename T> static void appendBytes(std::string &bytes, T v) {
  bytes.append(reinterpret_cast<const char *>(&v), sizeof(v));
}

// A part only when the device's product reads as zero operands its format
// keeps: otherwise the model computes what it computes without the part.
static void addOperandMin(CacheKey &key, double operandMin) {
  if (operandMin > 0.0)
    key.add("operand min", exactDouble(operandMin));
}

Memo::Memo(const Options &O, const RecognizedMatmul &m,
           const MatmulProfile &prof, unsigned sampleLogBits,
           unsigned numSamples, const MatmulAccuracyQuery &q) {
  switch (q.model) {
  case MatmulAccuracyQuery::Model::Rounded:
    key.add("model", "rounded");
    key.add("input format", dtypeName(q.inputPrec));
    key.add("accumulator format", dtypeName(q.accPrec));
    key.add("exponent format", dtypeName(q.exponentPrec));
    key.add("order tile K", std::to_string(q.orderTileK));
    key.add("input mantissa bits", std::to_string(q.inputMantBits));
    addOperandMin(key, q.operandMin);
    break;
  case MatmulAccuracyQuery::Model::TcecLibrary:
    key.add("model", "TCEC library");
    // The library whose product the model simulates.
    key.add("TCEC backend", "cuMpSGEMM");
    addOperandMin(key, q.operandMin);
    break;
  case MatmulAccuracyQuery::Model::OzakiII:
    key.add("model", "Ozaki-II");
    key.add("moduli", std::to_string(q.moduli));
    break;
  }
  key.add("M", std::to_string(m.M));
  key.add("N", std::to_string(m.N));
  key.add("K", std::to_string(m.K));
  std::string operands;
  for (const CellStat *c : {&prof.a, &prof.b}) {
    appendBytes(operands, c->min);
    appendBytes(operands, c->max);
    appendBytes(operands, c->minMag);
  }
  key.addContent("operand statistics", operands);
  key.add("gradient sum", exactDouble(prof.gradSum));
  // Only a second-moment cost reads the squared sum.
  if (prof.moment == ErrorMoment::Second)
    key.add("squared gradient sum", exactDouble(prof.sqGradSum));
  key.add("samples", std::to_string(numSamples));
  key.add("seed", std::to_string(O.Model.RandomSeed));
  key.add("sampling log bits", std::to_string(sampleLogBits));
  key.add("confidence", exactDouble(kDefaultConfidence));
  SmallString<256> p(O.Cache.Dir);
  sys::path::append(p, "matmul_accuracy_" + key.digest() + ".json");
  path = p.str().str();
  key.addCompiler(kPassIdentity);
}

void Memo::store(const Options &O, const MatmulAccuracy &result) const {
  if (O.Cache.Dir.empty())
    return;
  (void)sys::fs::create_directories(O.Cache.Dir, true);
  // As text: a domain error can be infinite, which JSON cannot hold.
  json::Object value;
  value["accuracy cost"] = exactDouble(result.accuracyCost);
  value["domain error"] = exactDouble(result.domainError);
  value["out of range"] = result.outOfRange;
  if (Error E = writeCacheEntry(path, key, std::move(value))) {
    std::string why = toString(std::move(E));
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] could not write the accuracy memo " << path
                   << ": " << why << '\n';
  }
}

static bool parseAccNumber(const json::Object &value, StringRef name,
                           double &out) {
  std::optional<StringRef> text = value.getString(name);
  return text && !text->getAsDouble(out);
}

bool Memo::lookup(const Options &O, MatmulAccuracy &result) const {
  if (O.Cache.Dir.empty())
    return false;
  CacheEntryRead read = readCacheEntry(path, key);
  if (read.value) {
    const json::Object *value = read.value->getAsObject();
    std::optional<StringRef> outOfRange =
        value ? value->getString("out of range") : std::nullopt;
    if (outOfRange &&
        parseAccNumber(*value, "accuracy cost", result.accuracyCost) &&
        parseAccNumber(*value, "domain error", result.domainError)) {
      result.outOfRange = outOfRange->str();
      return true;
    }
    read.unreadable = "its value does not parse";
  }
  if (!O.Diagnostics.Print)
    return false;
  if (!read.unreadable.empty())
    llvm::errs() << "[poseidon] accuracy memo " << path << " is unreadable ("
                 << read.unreadable << "); recomputing it\n";
  else if (!read.differs.empty())
    llvm::errs() << "[poseidon] accuracy memo " << path
                 << (read.differs == "format"
                         ? " was written in another cache format"
                         : " was computed for a different " + read.differs)
                 << "; recomputing it\n";
  return false;
}

// A query's result from the error of its sampled part. Per sample, the error
// is the weighted error summed over the outputs in the profile's moment; the
// accuracy cost is its mean over the samples where it is finite, and the
// domain error (call-site budget) the kDefaultConfidence percentile of its
// ratio to the weighted result (the root of the ratio in the second moment),
// which cancels the size/exec scaling of the bare sum. A format that overflowed
// on a sample, or flushes the smallest operand magnitude the profile recorded,
// is out of range, as is a device whose product reads it as zero: the absolute
// error is blind to a flushed operand.
static MatmulAccuracy finish(const Options &O, const MatmulProfile &prof,
                             const MatmulAccuracyQuery &q,
                             const SampledError &e) {
  MatmulAccuracy r{e.accuracyCost, e.domainError, {}};
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] domain error " << describe(q) << ' '
                 << percentileLabel(kDefaultConfidence)
                 << "_relErr=" << r.domainError
                 << " (sumcost=" << r.accuracyCost << ")\n";
  if (e.nonFinite)
    r.outOfRange = "a sampled product overflows its format";
  // The operands reach `exponentPrec` (Invalid = inputPrec) unscaled: a
  // scheme that scales them first has its own simulation.
  if (q.model == MatmulAccuracyQuery::Model::Rounded) {
    NumberFormat expK = (q.exponentPrec == NumberFormat::Invalid)
                            ? q.inputPrec
                            : q.exponentPrec;
    double minSub = minSubnormalOf(expK);
    if (minSub > 0.0)
      for (const CellStat *c : {&prof.a, &prof.b}) {
        if (c->minMag <= 0.0)
          continue;
        if (c->minMag < minSub)
          r.outOfRange = (Twine(dtypeName(expK)) +
                          " flushes the smallest profiled operand "
                          "magnitude, " +
                          shortDouble(c->minMag) + ", to zero")
                             .str();
        else if (c->minMag < q.operandMin)
          r.outOfRange =
              ("this device's product keeps " + Twine(dtypeName(expK)) +
               " operands down to " + shortDouble(q.operandMin) +
               " and reads the smallest profiled operand "
               "magnitude, " +
               shortDouble(c->minMag) + ", as zero")
                  .str();
      }
  }
  return r;
}

Error evaluateMatmulAccuracy(Session &S, const RecognizedMatmul &m,
                             const MatmulProfile &prof, unsigned sampleLogBits,
                             ArrayRef<MatmulAccuracyQuery> queries,
                             MutableArrayRef<MatmulAccuracy> results) {
  assert(queries.size() == results.size() && "one result per query");
  const Options &O = S.Opts;
  const unsigned numSamples = sampleCount(O);

  // Looked up in query order; a query whose entry an earlier query of this
  // product computes takes that result, as it would have read it back.
  std::vector<Memo> memos;
  SmallVector<std::optional<size_t>> sameEntryAs(queries.size());
  SmallVector<size_t> computed;
  for (size_t i = 0; i < queries.size(); ++i) {
    memos.emplace_back(O, m, prof, sampleLogBits, numSamples, queries[i]);
    if (!O.Cache.Dir.empty())
      for (size_t c : computed)
        if (memos[c].file() == memos[i].file()) {
          sameEntryAs[i] = c;
          break;
        }
    if (sameEntryAs[i])
      continue;
    if (memos[i].lookup(O, results[i]))
      continue;
    computed.push_back(i);
  }
  if (computed.empty())
    return Error::success();

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
    Expected<SampledError> error = products.error(partOf[c]);
    if (!error)
      return error.takeError();
    results[i] = finish(O, prof, queries[i], *error);
    memos[i].store(O, results[i]);
  }
  for (size_t i = 0; i < queries.size(); ++i)
    if (sameEntryAs[i])
      results[i] = results[*sameEntryAs[i]];
  return Error::success();
}

} // namespace poseidon
