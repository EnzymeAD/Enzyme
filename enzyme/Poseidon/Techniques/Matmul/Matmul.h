//===- Matmul.h - matrix products and their candidates --------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_MATMUL_H
#define POSEIDON_MATMUL_H

#include "MatmulScheme.h"
#include "Model/FPFormat.h"
#include "Profile/ProfileRead.h"
#include "Sites/LaunchDescriptors.h"
#include "Techniques/Unit.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/JSON.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class Function;
class Loop;
class SCEV;
class ScalarEvolution;
} // namespace llvm

namespace poseidon {

struct Options;
class Session;

enum class TidAxis : uint8_t {
  Unknown,
  TidX,
  TidY,
  TidZ,
  Other,
};

// A matrix row or column index of the thread: tid[fast], or, fused from two
// thread axes, tid[fast] + fuseMult * tid[slow], where fuseMult is the extent
// of the fast axis.
struct MatrixIndex {
  TidAxis fast = TidAxis::Unknown;
  TidAxis slow = TidAxis::Unknown;
  unsigned fuseMult = 0;
  bool fused() const { return slow != TidAxis::Unknown; }
};

// The index (0, 1, 2) of a thread axis; none for Unknown and Other.
inline std::optional<unsigned> tidAxisIndex(TidAxis a) {
  switch (a) {
  case TidAxis::TidX:
    return 0u;
  case TidAxis::TidY:
    return 1u;
  case TidAxis::TidZ:
    return 2u;
  case TidAxis::Unknown:
  case TidAxis::Other:
    break;
  }
  return std::nullopt;
}

struct ScalarLoopHandle {
  // Copied out of the LoopInfo of the recognition, which does not outlive it.
  llvm::BasicBlock *preheader = nullptr;
  llvm::BasicBlock *exitBB = nullptr;
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> blocks;
  llvm::Instruction *fma = nullptr;
  llvm::Instruction *fmul = nullptr;
  // The pointer operands of the A and B loads.
  llvm::Value *aPtr = nullptr;
  llvm::Value *bPtr = nullptr;

  // Per-thread addresses of A and B at the first iteration, including the
  // blockIdx * blockDim term.
  const llvm::SCEV *aStartSCEV = nullptr;
  const llvm::SCEV *bStartSCEV = nullptr;
  llvm::ScalarEvolution *SE = nullptr;

  // Set only when the profile header shows blockDim.z == 1.
  bool is2DBlock = false;
  // Whether the in-kernel raise can take the place of the loop: every thread
  // of the block runs the preheader, where it synchronizes the block, exactly
  // once, and the loop's exit takes only the product from it.
  bool inKernelRaisable = false;
  // The block shape the profile recorded, which the raise is laid out for.
  uint32_t profiledBlock[3] = {0, 0, 0};

  // The thread index of A's row and of B's column.
  MatrixIndex aRow, bCol;

  // Threads per CTA from the profile header (0 = unknown). A block whose size
  // is not a multiple of 32 has a partial last warp, in which the
  // warp-collective wmma ops are undefined.
  unsigned blockThreads = 0;

  int64_t aStrideByte = 0;
  int64_t bStrideByte = 0;
  int64_t aLeadingDimByte = 0;
  int64_t bLeadingDimByte = 0;
};

// One GEMM dimension (extent or leading dimension) as a monomial over launch
// values of the optimized body:
//     value = mul * factors[0] * factors[1] * ... + add
// (no factors => the constant `add`), the form the host-side stub rewrite
// evaluates from the launch arguments alone.
struct RuntimeDim {
  llvm::SmallVector<LaunchRef, 2> factors;
  int64_t mul = 1;
  int64_t add = 0;
  static RuntimeDim constant(int64_t value) {
    RuntimeDim d;
    d.add = value;
    return d;
  }
  bool isConst() const { return factors.empty(); }
};

std::string runtimeDimString(const RuntimeDim &d);

// A dense GEMM recognized from a thread-parallel reduction nest, whose extents
// and leading dimensions are runtime expressions.
struct RuntimeGemmHandle {
  // Chain members: the R reduction loops the contraction was split across by
  // full unrolling of the outer contraction level (R = 1 when there was only
  // one level). fmas[r] is member r's accumulating fma; fmuls[r] is its
  // separate fmul, or null for an llvm.fmuladd.
  llvm::SmallVector<llvm::Instruction *, 8> fmas;
  llvm::SmallVector<llvm::Instruction *, 8> fmuls;
  llvm::SmallVector<llvm::LoadInst *, 8> aLoads;

  // Epilogue: `C[..] (+)= acc`, null for a beta=0 store.
  llvm::Instruction *epilogue = nullptr;
  llvm::StoreInst *cStore = nullptr;

  LaunchRef aRef, bRef, cRef;

  RuntimeDim M, Ncols, K, lda, ldb, ldc;
  // Launch values the dense reading of the nest takes to be equal, with no way
  // to prove it here, e.g. a captured tensor's extent and the loop bound next
  // to it. The dispatch checks each pair at the launch and runs the kernel as
  // written when one fails.
  llvm::SmallVector<std::pair<LaunchRef, LaunchRef>, 2> assumedEqual;
  // Layout in the convention __poseidon_ozaki_dgemm takes: "colMajor" means
  // the operand's reduction index (A, B) or row index (C) is the contiguous
  // one.
  bool aColMajor = false, bColMajor = true, cColMajor = true;
  double beta = 0.0;

  // Profile-scale extents, reconstructed from the profile header's launch
  // geometry and the profiled execution counts.
  unsigned profM = 0, profN = 0, profK = 0;
  // Per-CTA tile of the product, the size the accuracy model simulates.
  unsigned tileM = 0, tileN = 0;
  uint64_t profMacs = 0; // total profiled MACs over all members
  unsigned launches = 0; // kernel launches observed during profiling
  unsigned ctaPerLaunch = 0;
};

struct AbstractMatmul {
  unsigned id = ~0u;

  unsigned M = ~0u, N = ~0u, K = ~0u;

  // Global launch geometry from the profile; M/N above are the per-CTA tile.
  // 0 = unknown.
  unsigned gridCTAs = 0;             // CTAs the in-kernel launch issues
  unsigned globalM = 0, globalN = 0; // full GEMM output extent
  // Profile-scale reduction length, so that with globalM/globalN the shape is
  // at one scale. 0 = unavailable, and the compile-time trip count K is used.
  unsigned globalK = 0;

  // The type of A, B, the accumulator and the result.
  FPKind elemType = FPKind::Invalid;

  enum class Origin {
    ScalarLoopReduction,
    // A dense GEMM whose M / Ncols / K and leading dimensions are runtime
    // expressions over the kernel's arguments, contracted across one or more
    // reduction loops (see RuntimeGemmHandle). Host dispatch only: an in-kernel
    // raise needs compile-time tile extents.
    HostGemmLoopNest,
  };
  Origin origin = Origin::ScalarLoopReduction;

  ScalarLoopHandle scalarLoop;
  std::shared_ptr<RuntimeGemmHandle> hostGemm;

  llvm::SetVector<llvm::Instruction *> footprint;
  llvm::Value *outputValue = nullptr;
};

// Recognize dense GEMMs written as thread-parallel reduction nests with runtime
// dimensions (Origin::HostGemmLoopNest) among the loops no matmul in `out`
// already claims. Appends only matmuls whose whole descriptor (extents,
// leading dimensions, layouts, beta and the profile-scale shape) could
// be reconstructed; every other nest is skipped with a printed reason.
void findHostGemmLoopNests(const Options &O, llvm::Function &F,
                           llvm::ScalarEvolution &SE, llvm::LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           const ProfileMap &prof,
                           llvm::SmallVectorImpl<AbstractMatmul> &out);

// Recognize the constant-trip reduction loops of F that compute a matrix
// product (Origin::ScalarLoopReduction), with the per-CTA extents and the
// launch geometry from `profileHeader`.
void findScalarLoopMatmuls(const Options &O, llvm::Function &F,
                           llvm::ScalarEvolution &SE, llvm::LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           llvm::SmallVectorImpl<AbstractMatmul> &out);

// Profile generation only: the reduction trip counts (profile-scale) of the
// scalar-loop matmuls of one profiled clone, as (fma probe index, trip) pairs
// for the clone's profile header, so the profile-use compile sees the shape
// at one scale. Must run on the clone without probes, which break the
// reduction shape.
void collectScalarLoopReductionTrips(
    const Options &O, llvm::Function &F,
    llvm::SmallVectorImpl<std::pair<size_t, unsigned>> &out);

// The dimensions of a GEMM, in the order the dispatch entries take them.
enum GemmDim : unsigned { DimM, DimNcols, DimK, DimLda, DimLdb, DimLdc };
constexpr unsigned kNumGemmDims = 6;

// What a host-dispatch descriptor holds: launch values per dimension, assumed
// equalities, and the magnitude of a dimension's constants and of its value,
// an int of the dispatch entries.
constexpr unsigned kMaxDimFactors = 8;
constexpr unsigned kMaxAssumedEqual = 16;
constexpr int64_t kMaxDimConstant = INT32_MAX;

// The host-dispatch descriptor of a GEMM in an optimized body. Operand roles
// and runtime-dimension factors name parameters of the body until the
// descriptor is written, and launch arguments of the wrapper kernel after.
struct GemmBodyNote {
  int cParam = -1, aParam = -1, bParam = -1;
  // Byte offsets of the three operand pointers inside their parameter, when
  // the parameter is a by-value aggregate; -1 when it is the pointer itself.
  int64_t cOffset = -1, aOffset = -1, bOffset = -1;
  // The selected variant's scheme, which names the host runtime entry, and
  // its descriptorModulusField.
  Scheme scheme = Scheme::OzakiII;
  unsigned numModuli = kOzakiIIMaxModuli;
  bool standalone = false; // body is a pure GEMM (only the C store)
  bool valid = false;
  // M x Ncols x K and the leading dimensions (in elements) at deploy scale,
  // indexed by GemmDim. The host stub evaluates a dimension with factors from
  // the launch arguments, so a build profiled at one problem size runs at
  // another.
  RuntimeDim dims[kNumGemmDims];
  bool aColMajor = false, bColMajor = false, cColMajor = false;
  // 1 accumulates into C.
  double beta = 0.0;

  // Deploy extents recovered from the launch geometry: a dimension that does
  // not track matrix order has no sound rescale from a surrogate profile. With
  // `launchCrop` set, dims[DimM] and dims[DimNcols] are the constants gating
  // the C store, and cropMAxis/cropNAxis name the thread axis each index runs
  // along so the host stub takes extent = min(crop, gridDim * blockDim).
  bool launchCrop = false;
  int cropMAxis = -1, cropNAxis = -1; // 0/1/2 = thread axis x/y/z

  // Launch values the recognition assumed equal (RuntimeGemmHandle); the stub
  // dispatches only when every pair agrees and launches the kernel otherwise.
  llvm::SmallVector<std::pair<LaunchRef, LaunchRef>, 2> assumedEqual;
};

struct PendingGemm {
  std::string wrapperName;
  std::vector<int> paramToArg; // body-param index -> wrapper launch-arg index
};

struct WmmaTarget {
  unsigned M, N, K;
  FPKind inputPrec;
  FPKind accPrec;
};

// What the matrix-product pipeline records over a compile.
struct MatmulState {
  // The GEMM note of each optimized body clone, keyed by the clone.
  std::map<const llvm::Function *, GemmBodyNote> GemmBodies;
  // In the order the sites were split, so that of two sites of one wrapper the
  // same one is recorded last in every compile.
  llvm::MapVector<const llvm::Function *, PendingGemm> PendingGemms;

  // The tensor-core shapes the cost model prices, and the warnings already
  // given.
  std::optional<llvm::SmallVector<WmmaTarget, 16>> AvailableWmmaTargets;
  // The cost-model rows a missing-row warning has named.
  llvm::StringSet<> ReportedRows;
};

class MatmulUnit : public SiteUnit {
public:
  struct Option {
    unsigned tileM = ~0u, tileN = ~0u, tileK = ~0u;
    SchemeVariant variant;
    Realization realization = Realization::InKernel;

    // Per-MAC reciprocal-throughput cost in cost-model units; the solver puts
    // the change over the profiled MACs on its axis.
    double compCost = -1.0;
    double accuracyCost = 0.0;
    // Relative domain error (sensitivity-weighted, normalized) at the
    // confidence percentile, comparable to a domain tolerance; accuracyCost
    // is an unnormalized sum that only ranks candidates.
    double domainError = 0.0;
  };

  MatmulUnit() : SiteUnit(UnitKind::M) {}

  AbstractMatmul *matmul = nullptr;
  // The descriptor of the product's library call, when it has one.
  std::optional<GemmBodyNote> library;
  double initialCompCost = -1.0; // per-MAC cost of the baseline (naive) matmul
  // Profiled total MAC count; deploy-size profiles exceed 2^32.
  uint64_t executions = 0;
  llvm::SmallVector<Option, 8> candidates;
  // Candidates this build cannot offer, by name, each with why.
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;

  double getCompCostDelta(size_t idx) const {
    return (candidates[idx].compCost - initialCompCost) * (double)executions;
  }
  double getAccCostDelta(size_t idx) const {
    return candidates[idx].accuracyCost;
  }

  void apply(Session &S, size_t candidateIndex);

  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::M);
  }
  size_t numCandidates() const override { return candidates.size(); }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep>) override {
    return getCompCostDelta(i);
  }
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep>) override {
    return getAccCostDelta(i);
  }
  // A raise replaces the loop that computes the product.
  bool consumesFootprint() const override { return true; }
  const llvm::SetVector<llvm::Instruction *> *footprint() const override {
    return matmul ? &matmul->footprint : nullptr;
  }
  std::string candidateName(size_t i) const override;
  llvm::ArrayRef<std::pair<std::string, std::string>>
  droppedCandidates() const override {
    return dropped;
  }
  std::string choiceText(size_t i) const override;
  void materialize(size_t i) override { apply(session(), i); }
  void scaleErrors(double factor) override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override {
    return "CM " + std::to_string(matmul->id);
  }
  std::string selectionText(size_t i) const override;
};

std::string matmulOptionLabel(const MatmulUnit::Option &opt);

// `sampleLogBits`: 0 samples operands uniformly over the profiled [min, max];
// B > 0 log-uniformly in magnitude over [maxMag * 2^-B, maxMag].
void generateMatmulCandidates(Session &S,
                              llvm::ArrayRef<AbstractMatmul> matmuls,
                              const ProfileMap &scalarProfile,
                              unsigned sampleLogBits,
                              llvm::SmallVectorImpl<MatmulUnit> &out);

// Per matmul, pick the cheapest candidate whose estimated relative domain error
// is <= `budget` and that beats the F64 baseline; scalar subgraphs are left at
// baseline in this mode.
llvm::SmallVector<SolutionStep>
errorBudgetSelector(const Options &O, llvm::SmallVectorImpl<MatmulUnit> &CMs,
                    double budget);

} // namespace poseidon
#endif // POSEIDON_MATMUL_H
