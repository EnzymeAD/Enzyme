// Matmul candidate proposal: the direct-dispatch proposer and
// generateMatmulCandidates.
#include "CostModel.h"
#include "Evaluators.h"
#include "Flags.h"
#include "HostDispatch.h"
#include "MatmulInternal.h"
#include "Optimize.h"
#include "OzakiII.h"
#include "Utils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>

using namespace llvm;

namespace poseidon {

// Direct reduced-precision GEMM host dispatch (direct_dispatch_rel): each FP64
// operand is rounded once to the library's input format and one library GEMM
// accumulates in FP32 (runtime/direct/direct_rt.cu). Same numerics as the
// in-kernel Direct raise up to accumulation order, but a distinct realized
// rate. Priced only from this device's measured direct_dispatch_rel,<class>
// row; absent the row it is not proposed.
//
// f32/f32 is the one variant that does not run on tensor cores: it is cuBLAS's
// SGEMM, the answer to "what does writing this product in single precision
// buy", and it belongs here because the runtime reaches it by changing the
// compute type alone. It has no MMA tile, so its tile triple is 0.
struct DirectDispatchVariant {
  FPKind inputPrec;
  FPKind accPrec;
  unsigned mode; // runtime compute mode: 1 = FP16, 2 = BF16, 3 = TF32, 4 = FP32
  unsigned tileM, tileN, tileK;
};
static constexpr DirectDispatchVariant kDirectDispatchVariants[] = {
    {FPKind::F16, FPKind::F32, 1, 16, 16, 16},
    {FPKind::BF16, FPKind::F32, 2, 16, 16, 16},
    {FPKind::TF32, FPKind::F32, 3, 16, 16, 8},
    {FPKind::F32, FPKind::F32, 4, 0, 0, 0},
};

// Cost-model row key, spelled "<in>_<acc>" like the in-kernel classes.
static std::string directDispatchClass(const DirectDispatchVariant &v) {
  return std::string(fpKindName(v.inputPrec)) + "_" + fpKindName(v.accPrec);
}

// Every proposer below prices from `baselinePerMac`, the site's own FP64
// per-MAC cost that each *_dispatch_rel row was measured against, times
// `padWaste`, the ratio that extrapolates a square-measured row to this shape.
static void proposeDirectDispatch(const AbstractMatmul &m,
                                  const MatmulProfile &prof, double confidence,
                                  unsigned sampleLogBits, double baselinePerMac,
                                  double padWaste, CandidateMatmul &cm) {
  for (const DirectDispatchVariant &v : kDirectDispatchVariants) {
    const std::string cls = directDispatchClass(v);
    double rel = queryCostModelOr("direct_dispatch_rel", cls, -1.0);
    if (rel <= 0.0) {
      static std::set<std::string> reported;
      if (reported.insert(cls).second)
        llvm::errs()
            << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << "x"
            << m.N << "x" << m.K << ": direct " << cls
            << " host dispatch NOT PROPOSED -- this device's cost model "
               "carries "
               "no measured 'direct_dispatch_rel,"
            << cls
            << "' row. Run poseidon-calibrate on this device; "
               "refusing to price a library dispatch on an unmeasured model.\n";
      if (flags::StrictMode)
        report_fatal_error(
            "Poseidon strict mode: direct host-dispatch class '" + Twine(cls) +
            "' has no measured direct_dispatch_rel row in " +
            Twine(costModelPath()));
      continue;
    }
    CandidateMatmul::Option opt;
    opt.tileM = v.tileM;
    opt.tileN = v.tileN;
    opt.tileK = v.tileK;
    opt.inputPrec = v.inputPrec;
    opt.accPrec = v.accPrec;
    opt.strategy = CandidateMatmul::Option::Strategy::DirectDispatch;
    opt.strategyParam = v.mode;
    opt.mChain = opt.nChain = opt.kChain = 1;
    opt.padM = opt.padN = opt.padK = 0;
    opt.compCost = baselinePerMac * rel * padWaste;
    // Same rounding as the in-kernel Direct candidate and no operand scaling in
    // the runtime, so it borrows that accuracy model verbatim: for f32/f32 that
    // is a 24-bit-significand operand product with an FP32 accumulator, not the
    // 22-bit two-limb capture the TCEC candidate is modelled with.
    opt.accuracyCost =
        getMatmulAccuracyCost(m, prof, confidence, sampleLogBits, v.inputPrec,
                              v.accPrec, &opt.domainError);
    cm.candidates.push_back(opt);
  }
}

constexpr unsigned kOzIITileM = 16, kOzIITileN = 16, kOzIITileK = 16;
constexpr FPKind kOzIIInputPrec = FPKind::S8;
constexpr FPKind kOzIIAccPrec = FPKind::S32;

// `rectM`/`rectK` key the shape-qualified measured row
// "ozaki_dispatch_rect_rel,nmNN_m<M>k<K>". Above padWaste 1.05 the runtime
// takes its RECTANGULAR arm, whose cost is linear in the column count, so
// charging S^3/(M*N*K) on top of a square-measured row over-prices the
// candidate by the whole padding ratio; the measured rectangular row supersedes
// it when present and no ratio is applied on top. The row is keyed by (M,K)
// only: with a rectangular dispatch the per-MAC cost no longer depends on the
// column count, and the column count is the one dimension that follows the
// mesh.
static void proposeOzakiIIDispatch(const AbstractMatmul &m,
                                   const MatmulProfile &prof, double confidence,
                                   unsigned sampleLogBits,
                                   double baselinePerMac, double padWaste,
                                   unsigned mChain, unsigned nChain,
                                   unsigned kChain, unsigned padM,
                                   unsigned padN, unsigned padK, unsigned rectM,
                                   unsigned rectK, CandidateMatmul &cm) {
  unsigned log2K = 0;
  for (unsigned kk = 1; kk < m.K; kk <<= 1)
    ++log2K;
  for (unsigned nm : kOzIIModuliCounts) {
    long capturedBits = ozakiIICapturedBits(nm, log2K);
    const std::string rectKey = "nm" + std::to_string(nm) + "_m" +
                                std::to_string(rectM) + "k" +
                                std::to_string(rectK);
    double relRect = queryCostModelOr("ozaki_dispatch_rect_rel", rectKey, -1.0);
    double rel = relRect > 0.0
                     ? relRect
                     : queryCostModelOr("ozaki_dispatch_rel",
                                        "nm" + std::to_string(nm), -1.0);
    if (rel <= 0.0) {
      if (flags::Print)
        llvm::errs()
            << "  Matmul[" << m.id << "] ozaki-ii nm=" << nm
            << " NOT PROPOSED -- neither 'ozaki_dispatch_rect_rel," << rectKey
            << "' nor 'ozaki_dispatch_rel,nm" << nm
            << "' is in the cost model; run poseidon-calibrate on this "
               "device to add the rung.\n";
      continue;
    }
    double shapeRatio = padWaste;
    if (relRect > 0.0) {
      shapeRatio = 1.0; // the measurement already is this shape
    } else if (padWaste >= 1.05) {
      static std::set<const void *> warned;
      if (warned.insert((const void *)&m).second)
        llvm::errs()
            << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << "x"
            << m.N << "x" << m.K << ": padWaste=" << padWaste
            << " >= 1.05, so __poseidon_ozaki_dgemm_ex will take its "
               "RECTANGULAR arm, whose cost is linear in the column count -- "
               "but the cost model has no measured 'ozaki_dispatch_rect_rel,"
            << rectKey
            << "' row, so this candidate is priced from the SQUARE row times "
               "the padding ratio and is over-priced by roughly that ratio. "
               "Measure the row before trusting a refusal at this site.\n";
    }
    CandidateMatmul::Option opt;
    opt.tileM = kOzIITileM;
    opt.tileN = kOzIITileN;
    opt.tileK = kOzIITileK;
    opt.inputPrec = kOzIIInputPrec;
    opt.accPrec = kOzIIAccPrec;
    opt.strategy = CandidateMatmul::Option::Strategy::OzakiII;
    opt.strategyParam = nm; // num_moduli
    opt.mChain = mChain;
    opt.nChain = nChain;
    opt.kChain = kChain;
    opt.padM = padM;
    opt.padN = padN;
    opt.padK = padK;
    opt.compCost = baselinePerMac * rel * shapeRatio;
    opt.accuracyCost =
        getOzakiIIAccuracyCost(m, prof, confidence, sampleLogBits,
                               (unsigned)capturedBits, &opt.domainError);
    cm.candidates.push_back(opt);
  }
}

// TCEC host dispatch: the in-kernel TCEC (OzakiI) scheme as a library call,
// which realizes a different rate and so is a distinct candidate. Priced from
// the measured tcec_dispatch_rel row; absent the row it is not proposed.
static void proposeTcecDispatch(const AbstractMatmul &m,
                                const MatmulProfile &prof, double confidence,
                                unsigned sampleLogBits, double baselinePerMac,
                                double padWaste, CandidateMatmul &cm) {
  double relTcec = queryCostModelOr("tcec_dispatch_rel", "fp16tcec", -1.0);
  if (relTcec <= 0.0)
    return;
  CandidateMatmul::Option opt;
  opt.tileM = 16;
  opt.tileN = 16;
  opt.tileK = 16;
  opt.inputPrec = FPKind::F16;
  opt.accPrec = FPKind::F32;
  opt.strategy = CandidateMatmul::Option::Strategy::TcecDispatch;
  opt.strategyParam = 1; // runtime compute mode: FP16TCEC
  opt.mChain = opt.nChain = opt.kChain = 1;
  opt.padM = opt.padN = opt.padK = 0;
  opt.compCost = baselinePerMac * relTcec * padWaste;
  // Same two-limb FP16 split with FP32 correction as the in-kernel TCEC n=2
  // candidate, so it borrows that accuracy model (22 captured bits).
  opt.accuracyCost = getMatmulAccuracyCost(
      m, prof, confidence, sampleLogBits, FPKind::F32, FPKind::F32,
      &opt.domainError,
      /*exponentPrec=*/FPKind::F16, /*orderTileK=*/0, /*inputMantBits=*/22);
  cm.candidates.push_back(opt);
}

// Native cuBLAS DGEMM through the same dispatch entry (nm=0): full FP64, priced
// from the measured ozaki_dispatch_rel,dgemm row times `padWaste`.
// The native cuBLAS DGEMM dispatch (num_moduli = 0). It takes NO padWaste: the
// runtime's nm == 0 arm maps every layout combination straight onto one
// cublasDgemm on the real M x Ncols x K, and the square zero-padding that
// ozaki_dispatch_rel is corrected for happens only on the emulated arms
// (runtime/ozaki/ozaki_rt.cu, "Native path needs no padding"). Charging the
// caller's square ratio here priced a rectangular launch at up to two orders of
// magnitude above what it runs at, which kept a bit-exact candidate out of
// every solution.
static void
proposeNativeDgemmDispatch(const AbstractMatmul &m, const MatmulProfile &prof,
                           double confidence, unsigned sampleLogBits,
                           double baselinePerMac, CandidateMatmul &cm) {
  double relDgemm = queryCostModelOr("ozaki_dispatch_rel", "dgemm", -1.0);
  if (relDgemm <= 0.0)
    report_fatal_error("Poseidon: the cost model has no "
                       "'ozaki_dispatch_rel,dgemm' row for the native DGEMM "
                       "candidate; run poseidon-calibrate.");
  CandidateMatmul::Option opt;
  opt.tileM = 16;
  opt.tileN = 16;
  opt.tileK = 16;
  opt.inputPrec = FPKind::F64;
  opt.accPrec = FPKind::F64;
  opt.strategy = CandidateMatmul::Option::Strategy::OzakiII;
  opt.strategyParam = 0; // nm=0 => native cuBLAS DGEMM in the runtime
  opt.mChain = opt.nChain = opt.kChain = 1;
  opt.padM = opt.padN = opt.padK = 0;
  opt.compCost = baselinePerMac * relDgemm;
  // cuBLAS DGEMM is full FP64 but not the scalar chain's accumulation order, so
  // the reorder is priced through orderTileK instead of claiming exact zero.
  opt.accuracyCost = getMatmulAccuracyCost(
      m, prof, confidence, sampleLogBits, FPKind::F64, FPKind::F64,
      &opt.domainError,
      /*exponentPrec=*/FPKind::Invalid, /*orderTileK=*/opt.tileK);
  cm.candidates.push_back(opt);
}

void generateMatmulCandidates(
    ArrayRef<AbstractMatmul> matmuls,
    const std::unordered_map<size_t, ProfileInfo> &scalarProfile,
    double confidence, unsigned sampleLogBits,
    SmallVectorImpl<CandidateMatmul> &out) {
  for (const AbstractMatmul &m : matmuls) {
    if (m.origin == AbstractMatmul::Origin::Invalid)
      report_fatal_error("unexpected Origin::Invalid");

    bool profiled = false;
    switch (m.origin) {
    case AbstractMatmul::Origin::ScalarLoopReduction: {
      size_t idx = readProfIdxMetadata(m.scalarLoop.fma);
      auto it = scalarProfile.find(idx);
      profiled = it != scalarProfile.end() && it->second.exec > 0;
      break;
    }
    case AbstractMatmul::Origin::HostGemmLoopNest: {
      // Recognition already required every member's count (the profile-scale
      // shape is reconstructed from them), so this can only re-confirm.
      profiled = true;
      for (Instruction *f : m.hostGemm->fmas) {
        size_t idx = readProfIdxMetadata(f);
        auto it = scalarProfile.find(idx);
        if (it == scalarProfile.end() || it->second.exec == 0) {
          profiled = false;
          break;
        }
      }
      break;
    }
    case AbstractMatmul::Origin::Invalid:
      llvm_unreachable("handled above");
    }
    if (!profiled) {
      if (!flags::LooseCoverage)
        report_fatal_error("matmul site has no profile coverage; set "
                           "-poseidon-loose-coverage to suppress");
      const Function *F = m.scalarLoop.fma ? m.scalarLoop.fma->getFunction()
                          : m.hostGemm && m.hostGemm->cStore
                              ? m.hostGemm->cStore->getFunction()
                              : nullptr;
      llvm::errs()
          << "[poseidon] matmul[" << m.id << "] in '"
          << (F ? F->getName() : "<unknown>")
          << "' uncovered — skipping under -poseidon-loose-coverage.\n";
      continue;
    }

    MatmulProfile prof = loadProfile(m, scalarProfile);

    CandidateMatmul cm;
    cm.matmul = const_cast<AbstractMatmul *>(&m);
    cm.executions = prof.totalCalls;

    switch (m.origin) {
    case AbstractMatmul::Origin::ScalarLoopReduction: {
      // Every candidate below is an in-kernel raise that allocates addrspace(3)
      // scratch in this body, so the enclosing kernel's static shared budget is
      // a precondition on all of them (computed once per matmul).
      Function *raiseF =
          m.scalarLoop.fma ? m.scalarLoop.fma->getFunction() : nullptr;
      uint64_t existingShmem = raiseF ? enclosingKernelSharedBytes(*raiseF) : 0;

      // The loop body is one MAC's worth of work and the cost-model rows are
      // reciprocal throughput per op, so this sum is the baseline's cost per
      // MAC (executions = profiled total MACs).
      double baselinePerMac = 0.0;
      for (Instruction *I : m.footprint)
        baselinePerMac += getInstructionCompCost(I);
      cm.initialCompCost = baselinePerMac;
      cm.initialAccCost = 0.0;

      for (const WmmaTarget &t : getAvailableWmmaTargets()) {
        // Integer kinds (S8/S32) have no FP accuracy model; they only appear
        // in the Ozaki-II dispatch candidates, never as Direct replacements.
        if (t.inputPrec == FPKind::S8 || t.inputPrec == FPKind::S32)
          continue;
        // ceildiv chain: boundary tiles are zero-padded at materialization.
        unsigned mChain = (m.M + t.M - 1) / t.M;
        unsigned nChain = (m.N + t.N - 1) / t.N;
        unsigned kChain = (m.K + t.K - 1) / t.K;
        unsigned padM = mChain * t.M - m.M;
        unsigned padN = nChain * t.N - m.N;
        unsigned padK = kChain * t.K - m.K;

        // Buildability before price: a shape whose tiles do not fit the static
        // shared cap is rejected by ptxas for the whole module.
        {
          CandidateMatmul::Option probe;
          probe.tileM = t.M;
          probe.tileN = t.N;
          probe.tileK = t.K;
          probe.inputPrec = t.inputPrec;
          probe.accPrec = t.accPrec;
          probe.strategy = CandidateMatmul::Option::Strategy::Direct;
          probe.mChain = mChain;
          probe.nChain = nChain;
          probe.kChain = kChain;
          if (!inKernelRaiseFits(raiseF, m, probe, existingShmem,
                                 "in-kernel Direct raise " +
                                     mmaShapeSuffix(t.M, t.N, t.K) + " " +
                                     fpKindName(t.inputPrec) + "/" +
                                     fpKindName(t.accPrec)))
            continue;
        }

        // Measured row for the class; an uncalibrated class is refused, not
        // estimated.
        double candCost = 0.0;
        if (!priceInKernelRaiseFromMeasuredRow(
                m, inKernelDirectClass(t.inputPrec, t.accPrec), t.M, t.N, t.K,
                cm.initialCompCost,
                "in-kernel Direct raise " + mmaShapeSuffix(t.M, t.N, t.K) +
                    " " + fpKindName(t.inputPrec) + "/" + fpKindName(t.accPrec),
                candCost))
          continue;

        CandidateMatmul::Option opt;
        opt.tileM = t.M;
        opt.tileN = t.N;
        opt.tileK = t.K;
        opt.inputPrec = t.inputPrec;
        opt.accPrec = t.accPrec;
        opt.strategy = CandidateMatmul::Option::Strategy::Direct;
        opt.strategyParam = 1;
        opt.mChain = mChain;
        opt.nChain = nChain;
        opt.kChain = kChain;
        opt.padM = padM;
        opt.padN = padN;
        opt.padK = padK;
        opt.compCost = candCost;
        // Identity-precision raises (f64->f64 DMMA) are invisible to the
        // rounding model, so charge the tensor tile's accumulation-order change
        // (orderTileK) instead; otherwise the candidate prices as exactly zero
        // error and can enter the budget-0 no-op solution.
        bool identityPrec = (t.inputPrec == m.aType && t.accPrec == m.accType);
        opt.accuracyCost =
            getMatmulAccuracyCost(m, prof, confidence, sampleLogBits,
                                  t.inputPrec, t.accPrec, &opt.domainError,
                                  /*exponentPrec=*/FPKind::Invalid,
                                  /*orderTileK=*/identityPrec ? t.K : 0);
        cm.candidates.push_back(opt);
      }

      // Ozaki Scheme I: operands are split into N narrow Veltkamp slices, an
      // N(N+1)/2-mma chain accumulates the cross products into F32 groups
      // d[i+j], and the readback combines them as sum_w d[w]/SCALE^w. The F32
      // accumulator rounds the K-sum, so the family is capped at F32-class
      // accuracy regardless of N.
      struct OzakiVariantSpec {
        unsigned tileM, tileN, tileK;
        FPKind inputPrec;
        FPKind accPrec;
        unsigned N;
      };
      static constexpr OzakiVariantSpec kOzakiVariants[] = {
          {16, 16, 16, FPKind::F16, FPKind::F32, 2},
          {16, 16, 16, FPKind::F16, FPKind::F32, 3},
          {16, 16, 16, FPKind::F16, FPKind::F32, 5},
          {16, 16, 16, FPKind::BF16, FPKind::F32, 2},
          {16, 16, 8, FPKind::TF32, FPKind::F32, 2},
          {16, 16, 8, FPKind::TF32, FPKind::F32, 3},
          {16, 16, 8, FPKind::TF32, FPKind::F32, 5},
          // Integer Ozaki-I (s8 slices) is intentionally not proposed:
          // in-kernel it has the same structural slowdown as in-kernel
          // Ozaki-II, and the host-dispatched Ozaki-II family covers its range.
      };
      for (const OzakiVariantSpec &v : kOzakiVariants) {
        bool tileAvailable = false;
        for (const WmmaTarget &wt : getAvailableWmmaTargets()) {
          if (wt.M == v.tileM && wt.N == v.tileN && wt.K == v.tileK &&
              wt.inputPrec == v.inputPrec && wt.accPrec == v.accPrec) {
            tileAvailable = true;
            break;
          }
        }
        if (!tileAvailable)
          continue;
        unsigned mChain = (m.M + v.tileM - 1) / v.tileM;
        unsigned nChain = (m.N + v.tileN - 1) / v.tileN;
        unsigned kChain = (m.K + v.tileK - 1) / v.tileK;
        unsigned padM = mChain * v.tileM - m.M;
        unsigned padN = nChain * v.tileN - m.N;
        unsigned padK = kChain * v.tileK - m.K;
        // No size gate: the padded fill zero-fills boundary tiles and every
        // slice outside the valid extent, so padding is numerically inert. The
        // one precondition is the static shared cap (N slice buffers per side
        // plus N F32 tiles live at once), a capability limit rather than a
        // price.
        {
          CandidateMatmul::Option probe;
          probe.tileM = v.tileM;
          probe.tileN = v.tileN;
          probe.tileK = v.tileK;
          probe.inputPrec = v.inputPrec;
          probe.accPrec = v.accPrec;
          probe.strategy = CandidateMatmul::Option::Strategy::OzakiI;
          probe.strategyParam = v.N;
          probe.mChain = mChain;
          probe.nChain = nChain;
          probe.kChain = kChain;
          if (!inKernelRaiseFits(
                  raiseF, m, probe, existingShmem,
                  "in-kernel TCEC n=" + std::to_string(v.N) + " " +
                      mmaShapeSuffix(v.tileM, v.tileN, v.tileK) + " " +
                      fpKindName(v.inputPrec) + "/" + fpKindName(v.accPrec)))
            continue;
        }
        double tcecCost = 0.0;
        if (!priceInKernelRaiseFromMeasuredRow(
                m, inKernelTcecClass(v.N, v.inputPrec, v.accPrec), v.tileM,
                v.tileN, v.tileK, cm.initialCompCost,
                "in-kernel TCEC n=" + std::to_string(v.N) + " " +
                    mmaShapeSuffix(v.tileM, v.tileN, v.tileK) + " " +
                    fpKindName(v.inputPrec) + "/" + fpKindName(v.accPrec),
                tcecCost))
          continue;
        CandidateMatmul::Option opt;
        opt.tileM = v.tileM;
        opt.tileN = v.tileN;
        opt.tileK = v.tileK;
        opt.inputPrec = v.inputPrec;
        opt.accPrec = v.accPrec;
        opt.strategy = CandidateMatmul::Option::Strategy::OzakiI;
        opt.strategyParam = v.N;
        opt.mChain = mChain;
        opt.nChain = nChain;
        opt.kChain = kChain;
        opt.padM = padM;
        opt.padN = padN;
        opt.padK = padK;
        opt.compCost = tcecCost;
        // Captured significand bits of the N-slice Veltkamp split
        // (emitNwayVeltkampSplit): slice 0 keeps p bits and each further slice
        // adds p, since the residual is rounded to p bits and rescaled by the
        // same factor; the SCALE only keeps the residual inside the slice
        // format's exponent range, charged via exponentPrec. The fill narrows
        // F64 to F32 and the accumulator is F32, so the family is capped at 24
        // bits. p = 8 for BF16, 11 for F16/TF32 (verified by a bit-level
        // simulation of the emitted split).
        unsigned sliceSigBits = (v.inputPrec == FPKind::BF16) ? 8u : 11u;
        unsigned capturedBits = std::min(sliceSigBits * v.N, 24u);
        opt.accuracyCost = getMatmulAccuracyCost(
            m, prof, confidence, sampleLogBits, FPKind::F32, FPKind::F32,
            &opt.domainError,
            /*exponentPrec=*/v.inputPrec, /*orderTileK=*/0,
            /*inputMantBits=*/capturedBits);
        cm.candidates.push_back(opt);
      }

      // Ozaki Scheme II (CRT/modular INT8, parameterized by num_moduli) is
      // dispatch-only, proposed under -poseidon-ozaki-host-dispatch.
      if (flags::OzakiHostDispatch) {
        bool s8TileAvailable = false;
        for (const WmmaTarget &wt : getAvailableWmmaTargets()) {
          if (wt.M == kOzIITileM && wt.N == kOzIITileN && wt.K == kOzIITileK &&
              wt.inputPrec == kOzIIInputPrec && wt.accPrec == kOzIIAccPrec) {
            s8TileAvailable = true;
            break;
          }
        }
        llvm::Function *ozF =
            m.scalarLoop.fma ? m.scalarLoop.fma->getFunction() : nullptr;
        GemmBodyNote ozProbe;
        bool ozDispatchable = ozF && computeGemmBodyNote(*ozF, m, ozProbe);
        if (flags::Print && s8TileAvailable && !ozDispatchable)
          llvm::errs() << "  Matmul[" << m.id << "] " << m.M << "x" << m.N
                       << "x" << m.K
                       << ": host-dispatched Ozaki-II NOT PROPOSED -- body is "
                          "not a host-dispatchable GEMM (computeGemmBodyNote "
                          "failed: operands do not trace to Function "
                          "arguments, or the body is not a pure GEMM).\n";

        // All three extents are profile-scale: mixing in the deploy-time m.K
        // gives a spurious r^2 waste under surrogate profiling.
        double redK = m.globalK ? (double)m.globalK : (double)m.K;
        double padWaste = 1.0;
        if (m.globalM && m.globalN && redK > 0.0) {
          double S = std::max({(double)m.globalM, (double)m.globalN, redK});
          double useful = (double)m.globalM * (double)m.globalN * redK;
          padWaste = (S * S * S) / useful; // padding waste, >= 1
        }

        if (s8TileAvailable && ozDispatchable) {
          unsigned mChain = (m.M + kOzIITileM - 1) / kOzIITileM;
          unsigned nChain = (m.N + kOzIITileN - 1) / kOzIITileN;
          unsigned kChain = (m.K + kOzIITileK - 1) / kOzIITileK;
          proposeOzakiIIDispatch(
              m, prof, confidence, sampleLogBits, cm.initialCompCost, padWaste,
              mChain, nChain, kChain, mChain * kOzIITileM - m.M,
              nChain * kOzIITileN - m.N, kChain * kOzIITileK - m.K,
              m.globalM ? m.globalM : m.M, (unsigned)redK, cm);
        }

        // The TCEC and Direct runtimes repack operands rather than padding to
        // a square, so no shape ratio is charged for them.
        if (ozDispatchable)
          proposeTcecDispatch(m, prof, confidence, sampleLogBits,
                              cm.initialCompCost,
                              /*padWaste=*/1.0, cm);

        if (ozDispatchable)
          proposeDirectDispatch(m, prof, confidence, sampleLogBits,
                                cm.initialCompCost,
                                /*padWaste=*/1.0, cm);

        if (ozDispatchable)
          proposeNativeDgemmDispatch(m, prof, confidence, sampleLogBits,
                                     cm.initialCompCost, cm);
      }
      break;
    }
    case AbstractMatmul::Origin::HostGemmLoopNest: {
      const RuntimeGemmHandle &g = *m.hostGemm;
      const unsigned R = g.fmas.size();

      // Baseline composed from the per-op rows, in the unit every
      // "*_dispatch_rel" row is measured in: one MAC of this loop nest is one
      // fma at the element type, plus the epilogue's fadd amortized over the
      // K-long reduction it closes.
      double perMac = 0.0;
      {
        double bodySum = 0.0, epi = 0.0;
        for (Instruction *I : m.footprint) {
          double c = getInstructionCompCost(I);
          if (I == g.epilogue)
            epi += c;
          else
            bodySum += c;
        }
        perMac = bodySum / (double)R + (m.K ? epi / (double)m.K : 0.0);
      }
      cm.initialCompCost = perMac;
      cm.initialAccCost = 0.0;

      // __poseidon_ozaki_dgemm_ex ZERO-PADS to S = roundup(max(M,Ncols,K),16),
      // so a rectangular product pays S^3 for M*N*K of useful work. Charged
      // here unless a measured rectangular row supersedes it.
      double padWaste = 1.0;
      {
        double S =
            std::max({(double)m.globalM, (double)m.globalN, (double)m.globalK});
        double useful =
            (double)m.globalM * (double)m.globalN * (double)m.globalK;
        if (useful > 0.0)
          padWaste = (S * S * S) / useful;
      }

      if (flags::OzakiHostDispatch) {
        proposeOzakiIIDispatch(
            m, prof, confidence, sampleLogBits, perMac, padWaste,
            /*mChain=*/1,
            /*nChain=*/1, /*kChain=*/1, /*padM=*/0,
            /*padN=*/0, /*padK=*/0, m.globalM, m.globalK, cm);
        // The TCEC and Direct runtimes repack operands rather than padding to a
        // square, so no shape ratio is charged for them.
        proposeTcecDispatch(m, prof, confidence, sampleLogBits, perMac,
                            /*padWaste=*/1.0, cm);
        proposeDirectDispatch(m, prof, confidence, sampleLogBits, perMac,
                              /*padWaste=*/1.0, cm);
        proposeNativeDgemmDispatch(m, prof, confidence, sampleLogBits, perMac,
                                   cm);
      }
      if (flags::Print)
        llvm::errs() << "[hostgemm] matmul[" << m.id
                     << "] NOTE: the square-padded dispatch price scales with "
                        "the COLUMN COUNT, which is a runtime value here, so "
                        "padWaste below is the ratio AT PROFILE SCALE only and "
                        "under-states the deploy cost. The rectangular "
                        "measured row 'ozaki_dispatch_rect_rel,nm<NM>_m"
                     << m.globalM << "k" << m.globalK
                     << "' is column-count independent and supersedes it when "
                        "present.\n"
                     << "[hostgemm] matmul[" << m.id
                     << "] baseline: composed/MAC=" << perMac
                     << "; padWaste(S^3/MNK)=" << padWaste << "\n";
      break;
    }
    case AbstractMatmul::Origin::Invalid:
      llvm_unreachable("handled above");
    }

    // compCost is per-MAC reciprocal throughput; rel is it divided by the naive
    // per-MAC baseline, the unit the measured *_dispatch_rel rows use.
    if (flags::Print) {
      llvm::errs() << "[poseidon] matmul[" << cm.matmul->id
                   << "] pricing: baseline/MAC=" << cm.initialCompCost
                   << " executions=" << cm.executions << "\n";
      for (size_t ci = 0; ci < cm.candidates.size(); ++ci) {
        const auto &o = cm.candidates[ci];
        llvm::errs() << "[poseidon]   #" << ci << " " << matmulOptionLabel(o)
                     << "  compCost/MAC=" << o.compCost << "  rel="
                     << (cm.initialCompCost > 0
                             ? o.compCost / cm.initialCompCost
                             : -1.0)
                     << "  domainError=" << o.domainError << "\n";
      }
    }

    out.push_back(std::move(cm));
  }
}

} // namespace poseidon
