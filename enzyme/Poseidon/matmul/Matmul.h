#ifndef POSEIDON_MATMUL_H
#define POSEIDON_MATMUL_H

#include "Precision.h"
#include "ProfileRead.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/InstructionCost.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
class Function;
class Loop;
class SCEV;
class ScalarEvolution;
} // namespace llvm

namespace poseidon {

struct CellStat {
  double min = 0.0;
  double max = 0.0;
  double sumValue = 0.0;
  uint64_t count = 0;
  double minMag = 0.0; // smallest nonzero |value| observed (0 = unknown)
};

struct GradCellStat {
  double sumGrad = 0.0;
  uint64_t count = 0;
};

struct MatmulProfile {
  unsigned M = 0, N = 0, K = 0;
  FPKind aType = FPKind::Invalid;
  FPKind bType = FPKind::Invalid;
  FPKind cType = FPKind::Invalid;
  FPKind dType = FPKind::Invalid;

  std::vector<CellStat> a;         // size M*K
  std::vector<CellStat> b;         // size K*N
  std::vector<CellStat> c;         // size M*N
  std::vector<CellStat> d;         // size M*N
  std::vector<GradCellStat> gradD; // size M*N

  uint64_t aInfNanCount = 0;
  uint64_t bInfNanCount = 0;
  uint64_t cInfNanCount = 0;
  uint64_t dInfNanCount = 0;
  uint64_t totalCalls = 0;
};

enum class TidAxis : uint8_t {
  Unknown,
  TidX,
  TidY,
  TidZ,
  Other,
};

struct ScalarLoopHandle {
  // Topology snapshot: the materializer must not depend on the LoopInfo that
  // produced the detection (destroyed when detection's local scope unwinds).
  llvm::BasicBlock *preheader = nullptr;
  llvm::BasicBlock *exitBB = nullptr;
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> blocks;
  llvm::PHINode *accPhi = nullptr;
  llvm::Instruction *fma = nullptr;
  llvm::Instruction *fmul = nullptr;
  llvm::LoadInst *aLoad = nullptr;
  llvm::LoadInst *bLoad = nullptr;
  llvm::Value *aBase = nullptr;
  llvm::Value *bBase = nullptr;

  // Per-thread iter-0 addresses of A and B as SCEVs, including the blockIdx
  // contribution. The materializer must SCEV-expand these instead of
  // reconstructing `aBase + tid*ld`, which drops the blockIdx*blockDim term and
  // misaddresses multi-block 2D-grid kernels.
  const llvm::SCEV *aStartSCEV = nullptr;
  const llvm::SCEV *bStartSCEV = nullptr;
  llvm::ScalarEvolution *SE = nullptr;

  // Profile header confirms blockDim.z == 1 (2D block): the materializer can
  // skip tid.z / ntid.z reads and the (tz * ny) term. Only set when verified.
  bool is2DBlock = false;

  TidAxis aRowAxis = TidAxis::Unknown;
  TidAxis bColAxis = TidAxis::Unknown;

  // A matrix index may be fused from two thread axes,
  // idx = tid[fastAxis] + fuseMult * tid[slowAxis] (sum-factorized FEM
  // kernels);
  // `*SlowAxis == Unknown` means not fused. `*FuseUnitByte` is the byte stride
  // of one step of the fused index, which the materializer asserts against its
  // own derivation.
  TidAxis aRowSlowAxis = TidAxis::Unknown;
  TidAxis bColSlowAxis = TidAxis::Unknown;
  unsigned aRowFuseMult = 0;
  unsigned bColFuseMult = 0;
  int64_t aRowFuseUnitByte = 0;
  int64_t bColFuseUnitByte = 0;

  // Threads per CTA from the profile header (0 = unknown). A block whose size
  // is not a multiple of 32 has a PARTIALLY POPULATED last warp, in which the
  // warp-collective wmma ops are undefined; the materializer must then confine
  // the tensor chain to warp 0.
  unsigned blockThreads = 0;

  int64_t aStrideByte = 0;
  int64_t bStrideByte = 0;
  int64_t aLeadingDimByte = 0;
  int64_t bLeadingDimByte = 0;

  unsigned tripCount = 0;
};

// Runtime-dimension dense GEMM recognized from a thread-parallel reduction
// nest: extents and leading dimensions are runtime expressions, carried
// symbolically here and as launch-argument references in the host-dispatch
// descriptor. Cost numbers come from the profile at profile scale (profM/N/K).

// One GEMM dimension (extent or leading dimension) as an affine function of at
// most ONE parameter of the optimized body:
//     value = mul * param + add          (param < 0  =>  the constant `add`)
// That is the largest form the host-side stub rewrite can evaluate from launch
// arguments alone, and it covers a finite-element apply's geometry (M = nDofs,
// K = d*numPoints, Ncols = d*NE, lda = ldb = d*numPoints, ldc = nDofs).
struct RuntimeDim {
  int param = -1; // parameter index OF THE OPTIMIZED BODY (-1 = constant)
  int64_t mul = 1;
  int64_t add = 0;
  bool isConst() const { return param < 0; }
  bool operator==(const RuntimeDim &o) const {
    return param == o.param && mul == o.mul && add == o.add;
  }
};

std::string runtimeDimString(const RuntimeDim &d);

struct RuntimeGemmHandle {
  // Chain members: the R reduction loops the contraction was split across by
  // full unrolling of the outer contraction level (R = 1 when there was only
  // one level). fmas[r] is member r's accumulating fma; fmuls[r] is its
  // separate fmul, or null for an llvm.fmuladd.
  llvm::SmallVector<llvm::Instruction *, 8> fmas;
  llvm::SmallVector<llvm::Instruction *, 8> fmuls;
  llvm::SmallVector<llvm::LoadInst *, 8> aLoads;
  llvm::SmallVector<llvm::LoadInst *, 8> bLoads;

  // Epilogue: `C[..] (+)= acc`. cLoad/epilogue are null for a beta=0 store.
  llvm::Instruction *epilogue = nullptr;
  llvm::LoadInst *cLoad = nullptr;
  llvm::StoreInst *cStore = nullptr;

  llvm::Value *aBase = nullptr, *bBase = nullptr, *cBase = nullptr;
  int aParam = -1, bParam = -1, cParam = -1;

  RuntimeDim M, Ncols, K, lda, ldb, ldc;
  // Layout in the convention __poseidon_ozaki_dgemm_ex takes: "colMajor" means
  // the operand's REDUCTION index (A, B) or ROW index (C) is the contiguous
  // one.
  bool aColMajor = false, bColMajor = true, cColMajor = true;
  double alpha = 1.0, beta = 0.0;

  // Profile-scale extents, reconstructed from the profile header's launch
  // geometry and the profiled execution counts.
  unsigned profM = 0, profN = 0, profK = 0;
  // Per-CTA tile of the product. The accuracy model simulates an M x N x K
  // product per sample, so it is handed the per-CTA tile; the GLOBAL shape
  // stays in profM/profN/profK and in AbstractMatmul's globalM/globalN/globalK.
  unsigned tileM = 0, tileN = 0;
  uint64_t profMacs = 0; // total profiled MACs over all members
  uint64_t profEpilogue = 0;
  unsigned launches = 0; // kernel launches observed during profiling
  unsigned ctaPerLaunch = 0;
};

struct AbstractMatmul {
  unsigned id = ~0u;

  unsigned M = ~0u, N = ~0u, K = ~0u;

  // Global launch geometry from the profiler for the occupancy / wave-fill
  // correction; M/N above are the per-CTA tile. 0 = unknown (neutral).
  unsigned gridCTAs = 0;             // CTAs the in-kernel launch issues
  unsigned globalM = 0, globalN = 0; // full GEMM output extent
  // Profile-scale reduction length recorded at profgen by
  // recordScalarLoopReductionTrips; with globalM/globalN it gives a
  // single-scale shape for the Ozaki-II padding-waste cost. 0 = unavailable
  // (the cost path falls back to the compile-time trip).
  unsigned globalK = 0;

  FPKind aType = FPKind::Invalid, bType = FPKind::Invalid;
  FPKind accType = FPKind::Invalid;
  FPKind dType = FPKind::Invalid;

  enum class Layout { Invalid, RowMajor, ColMajor };
  Layout aLayout = Layout::Invalid;
  Layout bLayout = Layout::Invalid;
  Layout dLayout = Layout::Invalid;

  enum class Origin {
    Invalid,
    ScalarLoopReduction,
    // A dense GEMM whose M / Ncols / K and leading dimensions are RUNTIME
    // expressions over the kernel's arguments, contracted across one or more
    // reduction loops (see RuntimeGemmHandle). Host-dispatch only: every
    // in-kernel raise needs compile-time tile extents, which this shape by
    // construction does not have.
    HostGemmLoopNest,
  };
  Origin origin = Origin::Invalid;

  ScalarLoopHandle scalarLoop;
  std::shared_ptr<RuntimeGemmHandle> hostGemm;

  llvm::SetVector<llvm::Instruction *> footprint;
  llvm::Value *outputValue = nullptr;
};

MatmulProfile
loadProfile(const AbstractMatmul &m,
            const std::unordered_map<size_t, ProfileInfo> &scalarProfile);

// Recognize dense GEMMs written as thread-parallel reduction NESTS with RUNTIME
// dimensions (Origin::HostGemmLoopNest). Parallel arm to findScalarLoopMatmuls:
// it looks at the loops the constant-trip recognizer rejects outright and
// appends only matmuls whose whole descriptor (extents, leading dimensions,
// layouts, alpha/beta and the profile-scale shape) could be reconstructed.
// Everything else is skipped with a printed reason; no dimension is guessed.
void findHostGemmLoopNests(llvm::Function &F, llvm::ScalarEvolution &SE,
                           llvm::LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           const std::unordered_map<size_t, ProfileInfo> &prof,
                           llvm::SmallVectorImpl<AbstractMatmul> &out);

class CandidateMatmul {
public:
  struct Option {
    unsigned tileM = ~0u, tileN = ~0u, tileK = ~0u;
    FPKind inputPrec = FPKind::Invalid;
    FPKind accPrec = FPKind::Invalid;

    enum class Strategy {
      Direct,
      OzakiI,
      OzakiII,
      // Host dispatch to the TCEC runtime (__poseidon_tcec_dgemm): the same
      // scheme as the in-kernel OzakiI/TCEC candidates, materialized as a
      // library call. strategyParam carries the runtime's compute mode.
      TcecDispatch,
      // Host dispatch to the direct reduced-precision GEMM runtime
      // (__poseidon_direct_dgemm): the same arithmetic as the in-kernel Direct
      // candidate of the same (inputPrec, accPrec), whose accuracy evaluation
      // it borrows. strategyParam is the operand format (1 = FP16, 2 = BF16,
      // 3 = TF32, 4 = FP32, the last one cuBLAS's SGEMM rather than a
      // tensor-core product).
      DirectDispatch,
    };
    Strategy strategy = Strategy::Direct;
    unsigned strategyParam = 1;

    unsigned mChain = ~0u, nChain = ~0u, kChain = ~0u;
    unsigned padM = 0, padN = 0, padK = 0;

    // Per-MAC reciprocal-throughput cost, kept in double through candidate
    // generation and rounded once in getCompCostDelta after the MAC-count
    // scale.
    double compCost = -1.0;
    double accuracyCost = 0.0;
    // Relative DOMAIN error (sensitivity-weighted, normalized) at the
    // confidence level the site was solved at: a real relative error a domain
    // tolerance can be compared to, unlike accuracyCost (an un-normalized SUM
    // used only for compute-budget ranking).
    double domainError = 0.0;
  };

  AbstractMatmul *matmul = nullptr;
  double initialCompCost = -1.0; // per-MAC cost of the baseline (naive) matmul
  double initialAccCost = 0.0;
  // Profiled total MAC count; deploy-size profiles exceed 2^32.
  uint64_t executions = 0;
  llvm::SmallVector<Option, 8> candidates;

  llvm::InstructionCost getCompCostDelta(size_t idx) const {
    double delta =
        (candidates[idx].compCost - initialCompCost) * (double)executions;
    return llvm::InstructionCost((int64_t)std::llround(delta));
  }
  double getAccCostDelta(size_t idx) const {
    return candidates[idx].accuracyCost - initialAccCost;
  }

  void apply(size_t candidateIndex);
};

std::string matmulOptionLabel(const CandidateMatmul::Option &opt);

// Static shared-memory preconditions for the in-kernel raises: ptxas caps a
// kernel's static shared allocation at 48 KB on sm_120 and rejects the whole
// module otherwise, so the proposer refuses a candidate whose scratch does not
// fit. The two entry points mirror, allocation for allocation, what the Direct
// and TCEC/Ozaki-I materializers create.
constexpr uint64_t kStaticShmemCap = 48u * 1024u;

// Bytes of addrspace(3) globals already live alongside a raise placed in `F`:
// the max over the PTX kernels that call F, falling back to the module-wide
// sum (the conservative direction) when no enclosing kernel is found.
uint64_t enclosingKernelSharedBytes(llvm::Function &F);

// Bytes of new static shared scratch an in-kernel raise of `opt` allocates
// (0 for host dispatch); `breakdown` receives the term list for the refusal.
uint64_t inKernelRaiseSharedBytes(llvm::Module &M,
                                  const CandidateMatmul::Option &opt,
                                  std::string *breakdown);

// `confidence`: the fraction of the sampled inputs each candidate's
// `domainError` bounds (the site's own value or -poseidon-confidence).
// `sampleLogBits`: 0 samples operands uniformly over the profiled [min, max];
// B > 0 log-uniformly in magnitude over [maxMag * 2^-B, maxMag].
void generateMatmulCandidates(
    llvm::ArrayRef<AbstractMatmul> matmuls,
    const std::unordered_map<size_t, ProfileInfo> &scalarProfile,
    double confidence, unsigned sampleLogBits,
    llvm::SmallVectorImpl<CandidateMatmul> &out);

} // namespace poseidon
#endif // POSEIDON_MATMUL_H
