// Matmul candidate proposal: one proposer per (scheme, realization), run in a
// fixed order by generateMatmulCandidates.
#include "HostDispatch.h"
#include "MatmulInternal.h"
#include "Model/CostModel.h"
#include "Pass/Session.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <string>

using namespace llvm;

namespace poseidon {

namespace {
struct Product {
  Session &S;
  const AbstractMatmul &m;
  const MatmulProfile &prof;
  double confidence;
  unsigned sampleLogBits;
  // The product's own per-MAC cost: the cost-model rows are ratios to it.
  double baselinePerMac;
};
} // namespace

// The accuracy model is called with what the scheme computes, whichever code
// realizes it; only the reorder charge reads the realized tile, and the split
// library call, which scales its operands before it splits them, has its own
// simulation.
static double schemeAccuracyCost(const Product &P, const SchemeVariant &v,
                                 Realization r, unsigned tileK,
                                 double *domainError) {
  const AbstractMatmul &m = P.m;
  switch (v.scheme) {
  case Scheme::Direct: {
    // Identity-format products (f64->f64 DMMA) are invisible to the rounding
    // model, so charge the tensor tile's accumulation-order change
    // (orderTileK) instead; otherwise the candidate prices as exactly zero
    // error and can enter the budget-0 no-op solution.
    bool identityPrec = (v.operand == m.aType && v.accumulator == m.accType);
    return getMatmulAccuracyCost(P.S, m, P.prof, P.confidence, P.sampleLogBits,
                                 v.operand, v.accumulator, domainError,
                                 /*exponentPrec=*/FPKind::Invalid,
                                 /*orderTileK=*/identityPrec ? tileK : 0);
  }
  case Scheme::Split:
    if (r == Realization::LibraryCall)
      return getTcecLibraryAccuracyCost(
          P.S, m, P.prof, P.confidence, P.sampleLogBits,
          splitLibraryComputeMode(v), domainError);
    return getMatmulAccuracyCost(P.S, m, P.prof, P.confidence, P.sampleLogBits,
                                 FPKind::F32, FPKind::F32, domainError,
                                 /*exponentPrec=*/v.operand, /*orderTileK=*/0,
                                 /*inputMantBits=*/splitCapturedBits(v));
  case Scheme::Modular:
    return getOzakiIIAccuracyCost(P.S, m, P.prof, P.confidence, P.sampleLogBits,
                                  modulusCount(v), domainError);
  case Scheme::Native:
    // Full FP64 but not the scalar chain's accumulation order, so the reorder
    // is priced through orderTileK instead of claiming exact zero.
    return getMatmulAccuracyCost(P.S, m, P.prof, P.confidence, P.sampleLogBits,
                                 v.operand, v.accumulator, domainError,
                                 /*exponentPrec=*/FPKind::Invalid,
                                 /*orderTileK=*/tileK);
  }
  llvm_unreachable("unknown Scheme");
}

static bool wmmaTileAvailable(Session &S, unsigned M, unsigned N, unsigned K,
                              FPKind operand, FPKind accumulator) {
  for (const WmmaTarget &wt : getAvailableWmmaTargets(S))
    if (wt.M == M && wt.N == N && wt.K == K && wt.inputPrec == operand &&
        wt.accPrec == accumulator)
      return true;
  return false;
}

//===----------------------------------------------------------------------===//
// In-kernel realization. Only a constant-trip product has one: the raise needs
// compile-time tile extents. It allocates addrspace(3) scratch in the body, so
// the enclosing kernel's static shared budget is a precondition, and it is
// priced from the measured wmma_inkernel_rel row of its class, the
// shape-qualified row first, with no shape ratio on top.
//===----------------------------------------------------------------------===//

namespace {
struct InKernelSite {
  Function *raiseF = nullptr;
  uint64_t existingShmem = 0;
};
} // namespace

static InKernelSite inKernelSite(const AbstractMatmul &m) {
  InKernelSite K;
  K.raiseF = m.scalarLoop.fma ? m.scalarLoop.fma->getFunction() : nullptr;
  K.existingShmem = K.raiseF ? enclosingKernelSharedBytes(*K.raiseF) : 0;
  return K;
}

static std::string inKernelDescription(const SchemeVariant &v, unsigned tileM,
                                       unsigned tileN, unsigned tileK) {
  const std::string tile =
      mmaShapeSuffix(tileM, tileN, tileK) + " " + schemeFormats(v, '/');
  switch (v.scheme) {
  case Scheme::Direct:
    return "in-kernel Direct raise " + tile;
  case Scheme::Split:
    return "in-kernel TCEC " + schemeParameter(v) + " " + tile;
  case Scheme::Modular:
  case Scheme::Native:
    break;
  }
  llvm_unreachable("no in-kernel realization of this scheme");
}

static void proposeInKernel(const Product &P, const InKernelSite &K,
                            const SchemeVariant &v, unsigned tileM,
                            unsigned tileN, unsigned tileK, MatmulUnit &cm) {
  const AbstractMatmul &m = P.m;
  MatmulUnit::Option opt;
  opt.tileM = tileM;
  opt.tileN = tileN;
  opt.tileK = tileK;
  opt.variant = v;
  opt.realization = Realization::InKernel;
  // ceildiv chain: boundary tiles are zero-padded at materialization.
  opt.mChain = (m.M + tileM - 1) / tileM;
  opt.nChain = (m.N + tileN - 1) / tileN;
  opt.kChain = (m.K + tileK - 1) / tileK;
  opt.padM = opt.mChain * tileM - m.M;
  opt.padN = opt.nChain * tileN - m.N;
  opt.padK = opt.kChain * tileK - m.K;
  const std::string what = inKernelDescription(v, tileM, tileN, tileK);

  // Buildability before price: a shape whose tiles do not fit the static
  // shared cap is rejected by ptxas for the whole module.
  if (!inKernelRaiseFits(P.S.Opts, K.raiseF, m, opt, K.existingShmem, what))
    return;
  // An uncalibrated class is refused, not estimated.
  if (!priceInKernelRaiseFromMeasuredRow(P.S, m, inKernelCostClass(v), tileM,
                                         tileN, tileK, P.baselinePerMac, what,
                                         opt.compCost))
    return;
  opt.accuracyCost =
      schemeAccuracyCost(P, v, opt.realization, tileK, &opt.domainError);
  cm.candidates.push_back(opt);
}

static void proposeInKernelDirect(const Product &P, const InKernelSite *K,
                                  MatmulUnit &cm) {
  if (!K)
    return;
  for (const WmmaTarget &t : getAvailableWmmaTargets(P.S)) {
    // Integer kinds (S8/S32) have no FP accuracy model; they only appear in
    // the modular library calls, never as Direct replacements.
    if (t.inputPrec == FPKind::S8 || t.inputPrec == FPKind::S32)
      continue;
    proposeInKernel(P, *K, SchemeVariant::direct(t.inputPrec, t.accPrec), t.M,
                    t.N, t.K, cm);
  }
}

// The split scheme in the kernel: operands are split into N narrow Veltkamp
// slices, an N(N+1)/2-mma chain accumulates the cross products into F32 groups
// d[i+j], and the readback combines them as sum_w d[w]/SCALE^w. The F32
// accumulator rounds the K-sum, so the family is capped at F32-class accuracy
// regardless of N.
struct InKernelSplitVariant {
  unsigned tileM, tileN, tileK;
  SchemeVariant variant;
};
static constexpr InKernelSplitVariant kInKernelSplitVariants[] = {
    {16, 16, 16, SchemeVariant::split(FPKind::F16, FPKind::F32, 2)},
    {16, 16, 16, SchemeVariant::split(FPKind::F16, FPKind::F32, 3)},
    {16, 16, 16, SchemeVariant::split(FPKind::F16, FPKind::F32, 5)},
    {16, 16, 16, SchemeVariant::split(FPKind::BF16, FPKind::F32, 2)},
    {16, 16, 8, SchemeVariant::split(FPKind::TF32, FPKind::F32, 2)},
    {16, 16, 8, SchemeVariant::split(FPKind::TF32, FPKind::F32, 3)},
    {16, 16, 8, SchemeVariant::split(FPKind::TF32, FPKind::F32, 5)},
    // Integer slices (s8) are intentionally not proposed: in-kernel they have
    // the same structural slowdown as an in-kernel modular product, and the
    // modular library calls cover their range.
};

static void proposeInKernelSplit(const Product &P, const InKernelSite *K,
                                 MatmulUnit &cm) {
  if (!K)
    return;
  for (const InKernelSplitVariant &s : kInKernelSplitVariants) {
    if (!wmmaTileAvailable(P.S, s.tileM, s.tileN, s.tileK, s.variant.operand,
                           s.variant.accumulator))
      continue;
    // No size gate: the padded fill zero-fills boundary tiles and every slice
    // outside the valid extent, so padding is numerically inert. The one
    // precondition is the static shared cap (N slice buffers per side plus N
    // F32 tiles live at once), a capability limit rather than a price.
    proposeInKernel(P, *K, s.variant, s.tileM, s.tileN, s.tileK, cm);
  }
}

//===----------------------------------------------------------------------===//
// Library-call realization, proposed under -poseidon-ozaki-host-dispatch. The
// body must be a GEMM the runtime can take over (recognition already
// established it for a runtime-shape product). Each scheme is priced from
// `baselinePerMac` times its measured *_dispatch_rel row, times the shape rule
// the scheme's runtime needs to extrapolate that row to this shape.
//===----------------------------------------------------------------------===//

constexpr unsigned kModularTile = 16;

namespace {
struct LibrarySite {
  bool dispatchable = false;
  bool modularOffered = false;
  // S^3 / (M*N*K), the zero padding __poseidon_ozaki_dgemm_ex pays to reach a
  // square of side S.
  double padWaste = 1.0;
  // Keys the rectangular modular row "ozaki_dispatch_rect_rel,nmNN_m<M>k<K>".
  unsigned rectM = 0, rectK = 0;
  // The extents the dispatch is compiled with, when all three are constants of
  // this compile (a scalar-loop product's descriptor); 0 when one is a
  // run-time value.
  unsigned staticM = 0, staticN = 0, staticK = 0;
  // The chains and padding the modular candidate carries.
  unsigned mChain = 1, nChain = 1, kChain = 1;
  unsigned padM = 0, padN = 0, padK = 0;
};
} // namespace

static double squarePaddingRatio(const AbstractMatmul &m) {
  switch (m.origin) {
  case AbstractMatmul::Origin::ScalarLoopReduction: {
    // All three extents are profile-scale: mixing in the deploy-time m.K
    // gives a spurious r^2 waste under surrogate profiling.
    double redK = m.globalK ? (double)m.globalK : (double)m.K;
    double padWaste = 1.0;
    if (m.globalM && m.globalN && redK > 0.0) {
      double S = std::max({(double)m.globalM, (double)m.globalN, redK});
      double useful = (double)m.globalM * (double)m.globalN * redK;
      padWaste = (S * S * S) / useful; // padding waste, >= 1
    }
    return padWaste;
  }
  case AbstractMatmul::Origin::HostGemmLoopNest: {
    double padWaste = 1.0;
    double S =
        std::max({(double)m.globalM, (double)m.globalN, (double)m.globalK});
    double useful = (double)m.globalM * (double)m.globalN * (double)m.globalK;
    if (useful > 0.0)
      padWaste = (S * S * S) / useful;
    return padWaste;
  }
  case AbstractMatmul::Origin::Invalid:
    break;
  }
  llvm_unreachable("unexpected Origin::Invalid");
}

static LibrarySite librarySite(Session &S, const AbstractMatmul &m) {
  LibrarySite L;
  L.padWaste = squarePaddingRatio(m);
  switch (m.origin) {
  case AbstractMatmul::Origin::ScalarLoopReduction: {
    bool s8TileAvailable = wmmaTileAvailable(
        S, kModularTile, kModularTile, kModularTile, FPKind::S8, FPKind::S32);
    Function *F = m.scalarLoop.fma ? m.scalarLoop.fma->getFunction() : nullptr;
    GemmBodyNote probe;
    L.dispatchable = F && computeGemmBodyNote(S.Opts, *F, m, probe);
    if (S.Opts.Diagnostics.Print && s8TileAvailable && !L.dispatchable)
      llvm::errs() << "  Matmul[" << m.id << "] " << m.M << "x" << m.N << "x"
                   << m.K
                   << ": host-dispatched Ozaki-II NOT PROPOSED -- body is "
                      "not a host-dispatchable GEMM (computeGemmBodyNote "
                      "failed: operands do not trace to Function "
                      "arguments, or the body is not a pure GEMM).\n";
    L.modularOffered = s8TileAvailable && L.dispatchable;
    if (L.dispatchable && !probe.runtimeDims) {
      L.staticM = probe.gM;
      L.staticN = probe.gNcols;
      L.staticK = probe.gK;
    }
    L.rectM = m.globalM ? m.globalM : m.M;
    L.rectK = m.globalK ? m.globalK : m.K;
    L.mChain = (m.M + kModularTile - 1) / kModularTile;
    L.nChain = (m.N + kModularTile - 1) / kModularTile;
    L.kChain = (m.K + kModularTile - 1) / kModularTile;
    L.padM = L.mChain * kModularTile - m.M;
    L.padN = L.nChain * kModularTile - m.N;
    L.padK = L.kChain * kModularTile - m.K;
    return L;
  }
  case AbstractMatmul::Origin::HostGemmLoopNest:
    L.dispatchable = true;
    L.modularOffered = true;
    L.rectM = m.globalM;
    L.rectK = m.globalK;
    return L;
  case AbstractMatmul::Origin::Invalid:
    break;
  }
  llvm_unreachable("unexpected Origin::Invalid");
}

static MatmulUnit::Option libraryOption(const SchemeVariant &v, unsigned tileM,
                                        unsigned tileN, unsigned tileK) {
  MatmulUnit::Option opt;
  opt.tileM = tileM;
  opt.tileN = tileN;
  opt.tileK = tileK;
  opt.variant = v;
  opt.realization = Realization::LibraryCall;
  opt.mChain = opt.nChain = opt.kChain = 1;
  opt.padM = opt.padN = opt.padK = 0;
  return opt;
}

static void pushLibraryCandidate(const Product &P, MatmulUnit::Option opt,
                                 MatmulUnit &cm) {
  opt.accuracyCost = schemeAccuracyCost(P, opt.variant, opt.realization,
                                        opt.tileK, &opt.domainError);
  cm.candidates.push_back(opt);
}

// The rectangular row supersedes the square one when present: above padWaste
// 1.05 the runtime takes its RECTANGULAR arm, whose cost is linear in the
// column count, so charging S^3/(M*N*K) on top of a square-measured row
// over-prices the candidate by the whole padding ratio; no ratio is applied on
// top of the rectangular row. It is keyed by (M,K) only: with a rectangular
// dispatch the per-MAC cost no longer depends on the column count, and the
// column count is the one dimension that follows the mesh.
static void proposeLibraryModular(const Product &P, const LibrarySite &L,
                                  MatmulUnit &cm) {
  if (!L.modularOffered)
    return;
  const AbstractMatmul &m = P.m;
  for (unsigned nm : kOzIIModuliCounts) {
    const SchemeVariant v = SchemeVariant::modular(nm);
    const std::string rectKey = "nm" + std::to_string(nm) + "_m" +
                                std::to_string(L.rectM) + "k" +
                                std::to_string(L.rectK);
    double relRect = queryCostModelOr(P.S.CostModel, "ozaki_dispatch_rect_rel",
                                      rectKey, -1.0);
    double rel = relRect > 0.0
                     ? relRect
                     : queryCostModelOr(P.S.CostModel, "ozaki_dispatch_rel",
                                        "nm" + std::to_string(nm), -1.0);
    if (rel <= 0.0) {
      if (P.S.Opts.Diagnostics.Print)
        llvm::errs()
            << "  Matmul[" << m.id << "] " << schemeTag(v) << " "
            << schemeParameter(v)
            << " NOT PROPOSED -- neither 'ozaki_dispatch_rect_rel," << rectKey
            << "' nor 'ozaki_dispatch_rel,nm" << nm
            << "' is in the cost model; run poseidon-calibrate on this "
               "device to add the rung.\n";
      continue;
    }
    double shapeRatio = L.padWaste;
    if (relRect > 0.0) {
      shapeRatio = 1.0; // the measurement already is this shape
    } else if (L.padWaste >= 1.05) {
      if (P.S.Matmul.WarnedRectangularRows.insert((const void *)&m).second)
        llvm::errs()
            << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << "x"
            << m.N << "x" << m.K << ": padWaste=" << L.padWaste
            << " >= 1.05, so __poseidon_ozaki_dgemm_ex will take its "
               "RECTANGULAR arm, whose cost is linear in the column count -- "
               "but the cost model has no measured 'ozaki_dispatch_rect_rel,"
            << rectKey
            << "' row, so this candidate is priced from the SQUARE row times "
               "the padding ratio and is over-priced by roughly that ratio. "
               "Measure the row before trusting a refusal at this site.\n";
    }
    MatmulUnit::Option opt =
        libraryOption(v, kModularTile, kModularTile, kModularTile);
    opt.mChain = L.mChain;
    opt.nChain = L.nChain;
    opt.kChain = L.kChain;
    opt.padM = L.padM;
    opt.padN = L.padN;
    opt.padK = L.padK;
    opt.compCost = P.baselinePerMac * rel * shapeRatio;
    pushLibraryCandidate(P, opt, cm);
  }
}

// The split scheme as __poseidon_tcec_dgemm, which realizes a different rate
// than the in-kernel raise and so is a distinct candidate. Absent its measured
// tcec_dispatch_rel row it is not proposed. The runtime repacks operands
// rather than padding to a square, so no shape ratio is charged.
static void proposeLibrarySplit(const Product &P, const LibrarySite &L,
                                MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  double rel =
      queryCostModelOr(P.S.CostModel, "tcec_dispatch_rel", "fp16tcec", -1.0);
  if (rel <= 0.0)
    return;
  MatmulUnit::Option opt = libraryOption(
      SchemeVariant::split(FPKind::F16, FPKind::F32, 2), 16, 16, 16);
  opt.compCost = P.baselinePerMac * rel;
  pushLibraryCandidate(P, opt, cm);
}

// The direct scheme as __poseidon_direct_dgemm: each FP64 operand is rounded
// once to the library's input format and one library GEMM accumulates in FP32
// (runtime/direct/direct_rt.cu). Same numerics as the in-kernel raise up to
// accumulation order, but a distinct realized rate. Priced only from this
// device's measured direct_dispatch_rel,<class> row; absent the row it is not
// proposed. The runtime repacks operands rather than padding to a square, so
// no shape ratio is charged.
//
// f32/f32 is the one variant that does not run on tensor cores: it is cuBLAS's
// SGEMM, the answer to "what does writing this product in single precision
// buy", and it belongs here because the runtime reaches it by changing the
// compute type alone. It has no MMA tile, so its tile triple is 0.
struct LibraryDirectVariant {
  SchemeVariant variant;
  unsigned tileM, tileN, tileK;
};
static constexpr LibraryDirectVariant kLibraryDirectVariants[] = {
    {SchemeVariant::direct(FPKind::F16, FPKind::F32), 16, 16, 16},
    {SchemeVariant::direct(FPKind::BF16, FPKind::F32), 16, 16, 16},
    {SchemeVariant::direct(FPKind::TF32, FPKind::F32), 16, 16, 8},
    {SchemeVariant::direct(FPKind::F32, FPKind::F32), 0, 0, 0},
};

static void proposeLibraryDirect(const Product &P, const LibrarySite &L,
                                 MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  const AbstractMatmul &m = P.m;
  for (const LibraryDirectVariant &d : kLibraryDirectVariants) {
    const std::string cls = schemeFormats(d.variant, '_');
    double rel =
        queryCostModelOr(P.S.CostModel, "direct_dispatch_rel", cls, -1.0);
    if (rel <= 0.0) {
      if (P.S.Matmul.ReportedDirectRows.insert(cls).second)
        llvm::errs()
            << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << "x"
            << m.N << "x" << m.K << ": " << schemeTag(d.variant) << " " << cls
            << " host dispatch NOT PROPOSED -- this device's cost model "
               "carries no measured 'direct_dispatch_rel,"
            << cls
            << "' row. Run poseidon-calibrate on this device; "
               "refusing to price a library dispatch on an unmeasured model.\n";
      if (P.S.Opts.Model.StrictMode)
        report_fatal_error(
            "Poseidon strict mode: direct host-dispatch class '" + Twine(cls) +
            "' has no measured direct_dispatch_rel row in " +
            Twine(costModelPath(P.S.CostModel)));
      continue;
    }
    MatmulUnit::Option opt =
        libraryOption(d.variant, d.tileM, d.tileN, d.tileK);
    opt.compCost = P.baselinePerMac * rel;
    pushLibraryCandidate(P, opt, cm);
  }
}

// The native product through the modular entry with num_moduli = 0, which the
// runtime maps onto one cublasDgemm on the real M x Ncols x K for every layout
// combination. It takes no padding ratio: the square zero-padding that
// ozaki_dispatch_rel is corrected for happens only on the emulated arms
// (runtime/ozaki/ozaki_rt.cu, "Native path needs no padding"). Charging the
// square ratio here priced a rectangular launch at up to two orders of
// magnitude above what it runs at, which kept a bit-exact candidate out of
// every solution. Its per-MAC rate still depends on the shape, a DGEMM with
// few columns running far below the square one, so a call whose extents are
// constants of this compile is priced from the row measured at those extents
// when the model has one. A run-time extent leaves only the square row: the
// surrogate's shape is not the one the call runs at.
static void proposeLibraryNative(const Product &P, const LibrarySite &L,
                                 MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  const AbstractMatmul &m = P.m;
  const bool isStatic = L.staticM && L.staticN && L.staticK;
  const std::string shape = std::to_string(L.staticM) + "x" +
                            std::to_string(L.staticN) + "x" +
                            std::to_string(L.staticK);
  const std::string key = "dgemm_m" + std::to_string(L.staticM) + "n" +
                          std::to_string(L.staticN) + "k" +
                          std::to_string(L.staticK);
  double rel = isStatic ? queryCostModelOr(P.S.CostModel,
                                           "ozaki_dispatch_rect_rel", key, -1.0)
                        : -1.0;
  if (rel <= 0.0) {
    rel = queryCostModelOr(P.S.CostModel, "ozaki_dispatch_rel", "dgemm", -1.0);
    if (rel <= 0.0)
      report_fatal_error("Poseidon: the cost model has no "
                         "'ozaki_dispatch_rel,dgemm' row for the native DGEMM "
                         "candidate; run poseidon-calibrate.");
    const double side = std::max({L.staticM, L.staticN, L.staticK});
    const double padWaste =
        isStatic
            ? side * side * side / ((double)L.staticM * L.staticN * L.staticK)
            : 1.0;
    if (padWaste >= 1.05 &&
        P.S.Matmul.WarnedSquareNativeRows.insert((const void *)&m).second)
      llvm::errs() << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M
                   << "x" << m.N << "x" << m.K << ": the native DGEMM runs at "
                   << shape << " (padWaste=" << padWaste
                   << "), but the cost model has no measured "
                      "'ozaki_dispatch_rect_rel,"
                   << key
                   << "' row, so it is priced from the SQUARE "
                      "'ozaki_dispatch_rel,dgemm' row, a rate a DGEMM of this "
                      "shape need not reach. Measure the row (poseidon-"
                      "calibrate --ozaki-rect "
                   << shape << ") before trusting a pick at this site.\n";
  }
  MatmulUnit::Option opt = libraryOption(SchemeVariant::native(), 16, 16, 16);
  opt.compCost = P.baselinePerMac * rel;
  pushLibraryCandidate(P, opt, cm);
}

void generateMatmulCandidates(Session &S, ArrayRef<AbstractMatmul> matmuls,
                              const ProfileMap &scalarProfile,
                              double confidence, unsigned sampleLogBits,
                              SmallVectorImpl<MatmulUnit> &out) {
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
      if (!S.Opts.Profiling.LooseCoverage)
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

    MatmulUnit cm;
    cm.matmul = const_cast<AbstractMatmul *>(&m);
    cm.executions = prof.totalCalls;

    std::optional<InKernelSite> kernel;
    double baselinePerMac = 0.0;
    switch (m.origin) {
    case AbstractMatmul::Origin::ScalarLoopReduction:
      kernel = inKernelSite(m);
      // The loop body is one MAC's worth of work and the cost-model rows are
      // reciprocal throughput per op, so this sum is the baseline's cost per
      // MAC (executions = profiled total MACs).
      for (Instruction *I : m.footprint)
        baselinePerMac += getInstructionCompCost(S.CostModel, I);
      break;
    case AbstractMatmul::Origin::HostGemmLoopNest: {
      const RuntimeGemmHandle &g = *m.hostGemm;
      const unsigned R = g.fmas.size();
      // Composed in the unit every "*_dispatch_rel" row is measured in: one
      // MAC of this loop nest is one fma at the element type, plus the
      // epilogue's fadd amortized over the K-long reduction it closes.
      double bodySum = 0.0, epi = 0.0;
      for (Instruction *I : m.footprint) {
        double c = getInstructionCompCost(S.CostModel, I);
        if (I == g.epilogue)
          epi += c;
        else
          bodySum += c;
      }
      baselinePerMac = bodySum / (double)R + (m.K ? epi / (double)m.K : 0.0);
      break;
    }
    case AbstractMatmul::Origin::Invalid:
      llvm_unreachable("handled above");
    }
    cm.initialCompCost = baselinePerMac;
    cm.initialAccCost = 0.0;
    const Product P{S, m, prof, confidence, sampleLogBits, baselinePerMac};

    // The candidate index is the position in this sequence.
    const InKernelSite *K = kernel ? &*kernel : nullptr;
    proposeInKernelDirect(P, K, cm);
    proposeInKernelSplit(P, K, cm);
    if (S.Opts.Matmul.OzakiHostDispatch) {
      const LibrarySite L = librarySite(S, m);
      proposeLibraryModular(P, L, cm);
      proposeLibrarySplit(P, L, cm);
      proposeLibraryDirect(P, L, cm);
      proposeLibraryNative(P, L, cm);
    }

    if (m.origin == AbstractMatmul::Origin::HostGemmLoopNest &&
        S.Opts.Diagnostics.Print)
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
                   << "] baseline: composed/MAC=" << baselinePerMac
                   << "; padWaste(S^3/MNK)=" << squarePaddingRatio(m) << "\n";

    // compCost is per-MAC reciprocal throughput; rel is it divided by the naive
    // per-MAC baseline, the unit the measured *_dispatch_rel rows use.
    if (S.Opts.Diagnostics.Print) {
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

std::string MatmulUnit::candidateName(size_t i) const {
  return matmulOptionLabel(candidates[i]);
}

std::string MatmulUnit::selectionText(size_t i) const {
  return "matmul[" + std::to_string(matmul->id) + "] -> " +
         matmulOptionLabel(candidates[i]);
}

std::string MatmulUnit::choiceText(size_t i) const {
  return "matmul[" + std::to_string(matmul->id) + "] -> " +
         matmulOptionLabel(candidates[i]) + " (#" + std::to_string(i) + ")";
}

json::Object MatmulUnit::stepJSON(size_t i) {
  json::Object stepObj;
  stepObj["type"] = "matmul_rewrite";
  stepObj["candidate_index"] = static_cast<int64_t>(i);
  stepObj["matmul_id"] = static_cast<int64_t>(matmul->id);
  stepObj["target"] = matmulOptionLabel(candidates[i]);
  stepObj["computation_cost_delta"] = getCompCostDelta(i);
  stepObj["accuracy_cost_delta"] = jsonFloat(getAccCostDelta(i));
  stepObj["executions"] = static_cast<int64_t>(executions);
  return stepObj;
}

void MatmulUnit::scaleErrors(double factor) {
  initialAccCost *= factor;
  for (auto &cand : candidates)
    cand.accuracyCost *= factor;
}

void MatmulUnit::printPriced(raw_ostream &OS) {
  OS << "Matmul[" << matmul->id
     << "] candidates (initial cost=" << initialCompCost
     << ", executions=" << executions << ", globalM=" << matmul->globalM
     << ", globalN=" << matmul->globalN << ", globalK=" << matmul->globalK
     << ", K=" << matmul->K << ", gridCTAs=" << matmul->gridCTAs << "):\n";
  if (candidates.empty()) {
    OS << "  (no candidates emitted)\n";
    return;
  }
  for (size_t i = 0; i < candidates.size(); ++i)
    OS << "  " << matmulOptionLabel(candidates[i])
       << ": Δcost=" << getCompCostDelta(i)
       << " ΔaccCost=" << getAccCostDelta(i) << "\n";
}

void MatmulUnit::writeReportStep(raw_ostream &OS, size_t i, StringRef pad) {
  OS << pad << "[Matmul] matmul[" << matmul->id << "] -> "
     << matmulOptionLabel(candidates[i]) << " (#" << i << ")\n";
}

// Dispatch and raise prices are measured wall-clock ratios that repeat to about
// 0.6% run to run, so a smaller modelled cost gap is a tie.
static constexpr double kCostTieBandRel = 0.006;

SmallVector<SolutionStep> errorBudgetSelector(const Options &O,
                                              SmallVector<MatmulUnit, 4> &CMs,
                                              double budget,
                                              double confidence) {
  SmallVector<SolutionStep> steps;
  for (auto &cm : CMs) {
    long best = -1;
    double bestComp = cm.initialCompCost;
    for (size_t i = 0; i < cm.candidates.size(); ++i) {
      const MatmulUnit::Option &c = cm.candidates[i];
      if (c.domainError <= budget && c.compCost < bestComp) {
        bestComp = c.compCost;
        best = (long)i;
      }
    }
    // Inside the calibration noise band two costs are a tie: among qualifying
    // candidates within band of the cheapest, the winner is the smallest
    // modelled domainError, then the smallest compCost, then the lowest index.
    // The guard never widens the qualifying set.
    if (best >= 0) {
      const double band = bestComp + kCostTieBandRel * std::fabs(bestComp);
      long tieBest = best;
      for (size_t i = 0; i < cm.candidates.size(); ++i) {
        const MatmulUnit::Option &c = cm.candidates[i];
        if (c.domainError > budget || c.compCost >= cm.initialCompCost)
          continue;
        if (c.compCost > band)
          continue;
        const MatmulUnit::Option &b = cm.candidates[tieBest];
        if (c.domainError < b.domainError ||
            (c.domainError == b.domainError && c.compCost < b.compCost))
          tieBest = (long)i;
      }
      if (tieBest != best && O.Diagnostics.Print)
        llvm::errs() << "Poseidon error-budget: cost tie inside "
                     << (kCostTieBandRel * 100.0) << "% calibration band ("
                     << matmulOptionLabel(cm.candidates[best]) << " cost "
                     << cm.candidates[best].compCost << " vs "
                     << matmulOptionLabel(cm.candidates[tieBest]) << " cost "
                     << cm.candidates[tieBest].compCost
                     << ") -> broken on modelled domain error ("
                     << cm.candidates[tieBest].domainError << " < "
                     << cm.candidates[best].domainError << ")\n";
      best = tieBest;
    }
    if (best < 0) {
      if (O.Diagnostics.Print)
        llvm::errs() << "Poseidon error-budget: matmul -> baseline F64 (no "
                        "rewrite clears "
                     << budget << " at confidence " << confidence
                     << " faster than F64)\n";
      continue;
    }
    if (O.Diagnostics.Print)
      llvm::errs() << "Poseidon error-budget: matmul -> "
                   << matmulOptionLabel(cm.candidates[best]) << " (domain err "
                   << cm.candidates[best].domainError << " <= " << budget
                   << " at confidence " << confidence << ")\n";
    steps.emplace_back(&cm, (size_t)best);
  }
  return steps;
}

} // namespace poseidon
