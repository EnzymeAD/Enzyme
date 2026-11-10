// Matmul accuracy model: profile-driven sampling, the memoized accuracy cost
// and the Ozaki-II quantization model.
#include "BuildIdentity.h"
#include "CacheKey.h"
#include "Flags.h"
#include "MatmulInternal.h"
#include "Optimize.h"
#include "Utils.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
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

// The accuracy memo: a model's result is a deterministic function of its
// inputs, so it is kept in flags::Cache, one entry file per key. The key holds
// the model's inputs and the model's identity, never its results: hosts differ
// in the last digits of a result for equal inputs. An entry is filed under
// every part but the identity, so one computed by another model is found and
// reported, and one computed by a model declared equivalent is reused.
namespace {
struct AccMemo {
  CacheKey key;
  std::string path;
  std::string legacyName;
};
} // namespace

// Exact, and readable back by StringRef::getAsDouble, infinity included.
static std::string accNumber(double v) {
  std::string s;
  raw_string_ostream(s) << format("%.17g", v);
  return s;
}

template <typename T> static void appendBytes(std::string &bytes, T v) {
  bytes.append(reinterpret_cast<const char *>(&v), sizeof(v));
}

static void addModelInputs(CacheKey &key, const AbstractMatmul &m,
                           const MatmulProfile &prof, double confidence,
                           unsigned sampleLogBits, unsigned numSamples) {
  key.add("M", std::to_string(m.M));
  key.add("N", std::to_string(m.N));
  key.add("K", std::to_string(m.K));
  std::string operands;
  for (const std::vector<CellStat> *cells : {&prof.a, &prof.b}) {
    appendBytes(operands, (uint64_t)cells->size());
    for (const CellStat &c : *cells) {
      appendBytes(operands, c.min);
      appendBytes(operands, c.max);
      appendBytes(operands, c.count);
      appendBytes(operands, c.minMag);
    }
  }
  key.addContent("operand statistics", operands);
  std::string gradients;
  appendBytes(gradients, (uint64_t)prof.gradD.size());
  for (const GradCellStat &g : prof.gradD)
    appendBytes(gradients, g.sumGrad);
  key.addContent("gradient statistics", gradients);
  key.add("samples", std::to_string(numSamples));
  key.add("seed", std::to_string(flags::RandomSeed));
  key.add("sampling log bits", std::to_string(sampleLogBits));
  key.add("exponent penalty", accNumber(flags::ExponentPenalty));
  key.add("confidence", accNumber(confidence));
}

static void fileAccMemo(AccMemo &memo) {
  SmallString<256> p(flags::Cache);
  sys::path::append(p, "matmulacc_" + memo.key.digest() + ".json");
  memo.path = p.str().str();
  memo.key.add("accuracy model", kAccuracyModelIdentity);
}

// The name the memo had while it was keyed on a hash of the inputs alone, kept
// so -poseidon-import-accuracy-memo can bring its entries over.
namespace {
struct LegacyAccHash {
  uint64_t h = 1469598103934665603ULL;
  void mix(uint64_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2); }
  void mixD(double d) {
    uint64_t b;
    std::memcpy(&b, &d, sizeof(b));
    mix(b);
  }
  LegacyAccHash(const AbstractMatmul &m, const MatmulProfile &prof,
                unsigned sampleLogBits) {
    mix(m.M);
    mix(m.N);
    mix(m.K);
    for (const std::vector<CellStat> *cells : {&prof.a, &prof.b}) {
      mix(cells->size());
      for (const auto &c : *cells) {
        mixD(c.min);
        mixD(c.max);
        mixD(c.sumValue);
        mix(c.count);
        mixD(c.minMag);
      }
    }
    mix(prof.gradD.size());
    for (const auto &g : prof.gradD) {
      mixD(g.sumGrad);
      mix(g.count);
    }
    mix(flags::NumSamples);
    mix((uint64_t)flags::RandomSeed);
    mix((uint64_t)sampleLogBits);
    mix(1); // the model generation every legacy entry was written under
  }
  std::string name() const { return "matmulacc_" + utohexstr(h) + ".txt"; }
};
} // namespace

static void storeAccMemo(const AccMemo &memo, double acc, double dom) {
  (void)sys::fs::create_directories(flags::Cache, true);
  // As text: a domain error can be infinite, which JSON cannot hold.
  json::Object value;
  value["accuracy cost"] = accNumber(acc);
  value["domain error"] = accNumber(dom);
  if (Error E = writeCacheEntry(memo.path, memo.key, std::move(value)))
    llvm::errs() << "[poseidon] could not write the accuracy memo " << memo.path
                 << ": " << toString(std::move(E)) << "\n";
}

static bool parseAccNumber(const json::Object &value, StringRef name,
                           double &out) {
  std::optional<StringRef> text = value.getString(name);
  return text && !text->getAsDouble(out);
}

static bool lookupAccMemo(const AccMemo &memo, double &acc, double &dom) {
  if (flags::Cache.empty())
    return false;
  CacheEntryRead read = readCacheEntry(
      memo.path, memo.key, {"accuracy model", kAccuracyModelEquivalents});
  if (read.value) {
    const json::Object *value = read.value->getAsObject();
    if (value && parseAccNumber(*value, "accuracy cost", acc) &&
        parseAccNumber(*value, "domain error", dom))
      return true;
    read.unreadable = "its value does not parse";
  }
  if (!read.unreadable.empty())
    llvm::errs() << "[poseidon] accuracy memo " << memo.path
                 << " is unreadable (" << read.unreadable
                 << "); recomputing it.\n";
  else if (!read.differs.empty())
    llvm::errs() << "[poseidon] accuracy memo " << memo.path
                 << (read.differs == "format"
                         ? " was written in another cache format"
                         : " was computed for a different " + read.differs)
                 << "; recomputing it.\n";

  if (flags::ImportAccuracyMemo.empty())
    return false;
  SmallString<256> legacy(flags::ImportAccuracyMemo);
  sys::path::append(legacy, memo.legacyName);
  auto buf = MemoryBuffer::getFile(legacy);
  if (!buf)
    return false;
  auto [accS, domS] = (*buf)->getBuffer().trim().split(' ');
  double a, d;
  if (accS.getAsDouble(a) || domS.getAsDouble(d))
    return false;
  acc = a;
  dom = d;
  storeAccMemo(memo, acc, dom);
  llvm::errs() << "[poseidon] imported accuracy memo " << legacy << " as "
               << memo.path << "\n";
  return true;
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

  AccMemo memo;
  memo.key.add("model", "getMatmulAccuracyCost");
  memo.key.add("input format", fpKindName(inputPrec));
  memo.key.add("accumulator format", fpKindName(accPrec));
  memo.key.add("exponent format", fpKindName(exponentPrec));
  memo.key.add("orderTileK", std::to_string(orderTileK));
  memo.key.add("inputMantBits", std::to_string(inputMantBits));
  addModelInputs(memo.key, m, prof, confidence, sampleLogBits, numSamples);
  fileAccMemo(memo);
  if (!flags::ImportAccuracyMemo.empty()) {
    LegacyAccHash legacy(m, prof, sampleLogBits);
    legacy.mix(0x6d6dULL);
    legacy.mix((uint64_t)inputPrec);
    legacy.mix((uint64_t)accPrec);
    legacy.mix((uint64_t)exponentPrec);
    legacy.mixD(flags::ExponentPenalty.getValue());
    if (confidence != kDefaultConfidence)
      legacy.mixD(confidence);
    if (orderTileK)
      legacy.mix(0x746b00ULL | orderTileK);
    if (inputMantBits)
      legacy.mix(0x6d6200ULL | inputMantBits);
    memo.legacyName = legacy.name();
  }
  {
    double cAcc, cDom;
    if (lookupAccMemo(memo, cAcc, cDom)) {
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
  if (!flags::Cache.empty())
    storeAccMemo(memo, acc, dom);
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

  AccMemo memo;
  memo.key.add("model", "getOzakiIIAccuracyCost");
  memo.key.add("capturedBits", std::to_string(capturedBits));
  addModelInputs(memo.key, m, prof, confidence, sampleLogBits, numSamples);
  fileAccMemo(memo);
  if (!flags::ImportAccuracyMemo.empty()) {
    LegacyAccHash legacy(m, prof, sampleLogBits);
    legacy.mix(0x6f7aULL);
    legacy.mix(capturedBits);
    if (confidence != kDefaultConfidence)
      legacy.mixD(confidence);
    memo.legacyName = legacy.name();
  }
  {
    double cAcc, cDom;
    if (lookupAccMemo(memo, cAcc, cDom)) {
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
  if (!flags::Cache.empty())
    storeAccMemo(memo, acc, dom);
  return acc;
}

} // namespace poseidon
