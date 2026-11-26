//===- Matmul.h - matrix products and their candidates --------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POSEIDON_TECHNIQUES_MATMUL_MATMUL_H
#define POSEIDON_TECHNIQUES_MATMUL_MATMUL_H

#include "Model/FPFormat.h"
#include "Profile/ProfileReader.h"
#include "Scheme.h"
#include "Sites/LaunchManifest.h"
#include "Techniques/SiteUnit.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/JSON.h"

#include <array>
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

/// The thread axis a matrix index runs along: x, y or z; Unknown when no index
/// is recorded, Other for an index of no thread axis, of three, or of two that
/// do not fuse.
enum class TidAxis : uint8_t {
  Unknown,
  TidX,
  TidY,
  TidZ,
  Other,
};

/// A matrix row or column index of the thread: tid[fast], or, fused from two
/// thread axes, tid[fast] + fuseMult * tid[slow], where fuseMult is the extent
/// of the fast axis.
struct MatrixIndex {
  /// The axis whose thread index steps the matrix index by one.
  TidAxis fast = TidAxis::Unknown;
  /// The second axis of a fused index; Unknown for an index of one axis.
  TidAxis slow = TidAxis::Unknown;
  /// What the slow axis's thread index is multiplied by, in units of the matrix
  /// index; 0 for an index of one axis.
  unsigned fuseMult = 0;
  /// Whether the index fuses two thread axes.
  bool fused() const { return slow != TidAxis::Unknown; }
};

/// Exclusive upper bounds on the global thread index of each axis,
/// blockIdx.a * blockDim.a + threadIdx.a; 0 leaves the axis unbounded.
struct ThreadIndexBounds {
  /// Per axis x, y, z, the exclusive bound in threads of the grid; 0 for none.
  std::array<uint64_t, 3> ub = {0, 0, 0};
  /// Lowers the bound of `axis` (0, 1 or 2) to `bound` unless it is already
  /// lower.
  void tighten(unsigned axis, uint64_t bound) {
    if (!ub[axis] || bound < ub[axis])
      ub[axis] = bound;
  }
};

/// The index (0, 1, 2) of a thread axis; none for Unknown and Other.
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

/// A constant-trip reduction loop that computes a matrix product. Its IR
/// pointers are into the site's function and stay valid while that function is
/// optimized.
struct ScalarLoopHandle {
  /// The loop's preheader, where the in-kernel raise is emitted. The blocks
  /// are copied because the recognition's LoopInfo does not outlive the handle.
  llvm::BasicBlock *preheader = nullptr;
  /// The loop's unique exit block, where the in-kernel raise reads the product
  /// back.
  llvm::BasicBlock *exitBB = nullptr;
  /// The blocks of the loop.
  llvm::SmallPtrSet<llvm::BasicBlock *, 4> blocks;
  /// The accumulating multiply-add, whose value after the latch is the product.
  llvm::Instruction *fma = nullptr;
  /// The contractable multiply under `fma`; null when `fma` is llvm.fma or
  /// llvm.fmuladd.
  llvm::Instruction *fmul = nullptr;
  /// The pointer operands of the A and B loads.
  llvm::Value *aPtr = nullptr;
  /// The pointer operand of the B load.
  llvm::Value *bPtr = nullptr;

  /// Per-thread addresses of A and B at the first iteration, including the
  /// blockIdx * blockDim term.
  const llvm::SCEV *aStartSCEV = nullptr;
  /// B's per-thread address at the first iteration.
  const llvm::SCEV *bStartSCEV = nullptr;
  /// The analysis the two start addresses belong to, alive while the site's
  /// function is optimized.
  llvm::ScalarEvolution *SE = nullptr;

  /// Whether the in-kernel raise can take the place of the loop: every thread
  /// of the block runs the preheader, where it synchronizes the block, exactly
  /// once, the loop's exit is entered only from its latch, and the loop does
  /// nothing but compute the product.
  bool inKernelRaisable = false;
  /// What the loop does besides computing the product, which a raise in its
  /// place would drop; empty when it does nothing else.
  std::string otherWork;
  /// The block shape the profile recorded, which the raise is laid out for.
  uint32_t profiledBlock[3] = {0, 0, 0};
  /// The guards a thread passes on its way to the loop: a block raises only
  /// when every one of its threads passes them.
  ThreadIndexBounds loopGuard;

  /// The thread index of A's row and of B's column.
  MatrixIndex aRow, bCol;

  /// Threads per CTA in the profiled block (0 = unknown). A block whose size
  /// is not a multiple of 32 has a partial last warp, in which the
  /// warp-collective wmma ops are undefined.
  unsigned blockThreads() const {
    return profiledBlock[0] * profiledBlock[1] * profiledBlock[2];
  }
  /// Whether the profiled block has blockDim.z == 1.
  bool is2DBlock() const { return profiledBlock[2] == 1; }

  /// A's byte step per reduction iteration: one element.
  int64_t aStrideByte = 0;
  /// B's byte step per reduction iteration: one element when B is K-contiguous,
  /// a positive multiple of it when B is K-strided.
  int64_t bStrideByte = 0;
  /// A's leading dimension in bytes: the byte step per unit of A's row index;
  /// without one, the element size of A's outermost GEP when that is above one
  /// element, else K elements.
  int64_t aLeadingDimByte = 0;
  /// B's leading dimension in bytes: bStrideByte when B is K-strided; the byte
  /// step per unit of B's column index when B is K-contiguous, without one as
  /// for A, from B's outermost GEP or K elements.
  int64_t bLeadingDimByte = 0;
};

/// One GEMM dimension (extent or leading dimension) as a monomial over launch
/// values of the optimized body:
///     value = mul * factors[0] * factors[1] * ... + add
/// (no factors => the constant `add`), the form the host-side stub rewrite
/// evaluates from the launch arguments alone.
struct RuntimeDim {
  /// The launch values multiplied together, in launch order; empty for a
  /// constant.
  llvm::SmallVector<LaunchRef, 2> factors;
  /// The coefficient of the product of `factors`.
  int64_t mul = 1;
  /// The constant term; the whole value when `factors` is empty.
  int64_t add = 0;
  /// The dimension that is `value` at every launch.
  static RuntimeDim constant(int64_t value) {
    RuntimeDim d;
    d.add = value;
    return d;
  }
  /// Whether the dimension depends on no launch value.
  bool isConst() const { return factors.empty(); }
};

/// `d` as diagnostics and printed descriptors show it, a factor named `arg<i>`
/// and a field of one `arg<i>[+<offset>]`.
std::string runtimeDimString(const RuntimeDim &d);

/// The dimensions of a GEMM, in the order the library-call entries take them.
enum GemmDim : unsigned { DimM, DimNcols, DimK, DimLda, DimLdb, DimLdc };
/// The number of GemmDim values.
constexpr unsigned kNumGemmDims = 6;

/// The most launch values one dimension of a descriptor multiplies.
constexpr unsigned kMaxDimFactors = 8;
/// The most pairs of assumed-equal launch values a descriptor holds.
constexpr unsigned kMaxAssumedEqual = 16;
/// The largest magnitude of a dimension's constants and of its value, the int a
/// library-call entry takes.
constexpr int64_t kMaxDimConstant = INT32_MAX;

/// The library-call descriptor of a GEMM in an optimized body. Operand roles
/// and runtime-dimension factors name parameters of the body until the
/// descriptor is written, and launch arguments of the wrapper kernel after.
struct GemmBodyNote {
  /// The parameters that hold C, A and B: of the optimized body until the
  /// descriptor is written, of the wrapper kernel's launch after; -1 when
  /// unknown.
  int cParam = -1, aParam = -1, bParam = -1;
  /// Byte offsets of the three operand pointers inside their parameter, when
  /// the parameter is a by-value aggregate; -1 when it is the pointer itself.
  int64_t cOffset = -1, aOffset = -1, bOffset = -1;
  /// The product's store is the body's one effect (hasEffect), so the library
  /// call takes the place of the launch; otherwise it precedes the launch.
  bool standalone = false;
  /// M x Ncols x K and the leading dimensions (in elements) at deploy scale,
  /// indexed by GemmDim. The host stub evaluates a dimension with factors from
  /// the launch arguments, so a build profiled at one problem size runs at
  /// another.
  RuntimeDim dims[kNumGemmDims];
  /// Storage orders as the library-call entries take them: A or B column-major
  /// has its reduction index contiguous, C column-major its row index.
  bool aColMajor = false, bColMajor = false, cColMajor = false;
  /// 1 accumulates into C.
  double beta = 0.0;

  /// Deploy extents recovered from the launch geometry: a dimension that does
  /// not track matrix order has no sound rescale from a surrogate profile. With
  /// `launchCrop` set, dims[DimM] and dims[DimNcols] are the constants gating
  /// the C store, and cropMAxis/cropNAxis name the thread axis each index runs
  /// along so the host stub takes extent = min(crop, gridDim * blockDim).
  bool launchCrop = false;
  int cropMAxis = -1, cropNAxis = -1; // 0/1/2 = thread axis x/y/z

  /// Launch values the runtime-shape recognition assumed equal; the stub
  /// dispatches only when every pair agrees and launches the kernel otherwise.
  llvm::SmallVector<std::pair<LaunchRef, LaunchRef>, 2> assumedEqual;
};

/// A dense GEMM recognized from a thread-parallel reduction nest, whose extents
/// and leading dimensions are runtime expressions.
struct RuntimeGemmHandle {
  /// Chain members: the R reduction loops the contraction was split across by
  /// full unrolling of the outer contraction level (R = 1 when there was only
  /// one level). fmas[r] is member r's accumulating fma.
  llvm::SmallVector<llvm::Instruction *, 8> fmas;
  /// Per member, the separate multiply of fmas[r]; null for llvm.fma or
  /// llvm.fmuladd.
  llvm::SmallVector<llvm::Instruction *, 8> fmuls;
  /// Per member, the load of the operand the descriptor takes as A.
  llvm::SmallVector<llvm::LoadInst *, 8> aLoads;

  /// Epilogue: `C[..] (+)= acc`, null for a beta=0 store.
  llvm::Instruction *epilogue = nullptr;
  /// The store of the product into C.
  llvm::StoreInst *cStore = nullptr;

  /// The library-call descriptor, every field derived from the nest.
  GemmBodyNote library;
};

/// A matrix product recognized in a site's function, with the handle of the
/// recognizer that found it. Its IR pointers are into that function and stay
/// valid while it is optimized.
struct RecognizedMatmul {
  /// The product's index among the products of its function, `matmul[id]` in
  /// diagnostics.
  unsigned id = ~0u;

  /// The extents in elements the accuracy model simulates and an in-kernel
  /// raise tiles: M x N is the output tile of one CTA at the profiled block
  /// shape; K is the compile-time trip count of a constant-shape product and
  /// the profile-scale reduction length of a runtime-shape one. ~0u until
  /// recognized.
  unsigned M = ~0u, N = ~0u, K = ~0u;

  /// Global launch geometry from the profile; M/N above are the per-CTA tile.
  /// 0 = unknown.
  unsigned gridCTAs = 0;             // CTAs the in-kernel launch issues
  unsigned globalM = 0, globalN = 0; // full GEMM output extent
  /// Profile-scale reduction length, so that with globalM/globalN the shape is
  /// at one scale. 0 = unavailable, and the compile-time trip count K is used.
  unsigned globalK = 0;

  /// The type of A, B, the accumulator and the result.
  FPKind elemType = FPKind::Invalid;

  /// Which recognizer found the product.
  enum class Origin {
    /// A constant-trip reduction loop whose output rows and columns are thread
    /// indices (see ScalarLoopHandle); either realization.
    ConstantShape,
    /// A dense GEMM whose M / Ncols / K and leading dimensions are runtime
    /// expressions over the kernel's arguments, contracted across one or more
    /// reduction loops (see RuntimeGemmHandle). Library call only: an
    /// in-kernel raise needs compile-time tile extents.
    RuntimeShape,
  };
  /// Which of `scalarLoop` and `runtimeGemm` describes the product.
  Origin origin = Origin::ConstantShape;

  /// The loop of a ConstantShape product; empty for the other origin.
  ScalarLoopHandle scalarLoop;
  /// The nest of a RuntimeShape product; null for the other origin.
  std::shared_ptr<RuntimeGemmHandle> runtimeGemm;

  /// The instructions a realization replaces: the blocks of the reduction loops
  /// and, for a runtime-shape product, its epilogue fadd.
  llvm::SetVector<llvm::Instruction *> footprint;
  /// The product's value: the last member's multiply-add, read after its loop.
  llvm::Value *outputValue = nullptr;
};

/// Recognize dense GEMMs written as thread-parallel reduction nests with
/// runtime dimensions (Origin::RuntimeShape) among the loops no matmul in
/// `out` already claims. Appends only matmuls whose whole descriptor (extents,
/// leading dimensions, layouts, beta and the profile-scale shape) could be
/// reconstructed; every other nest is skipped with a printed reason.
void findRuntimeShapeMatmuls(const Options &O, llvm::Function &F,
                             llvm::ScalarEvolution &SE, llvm::LoopInfo &LI,
                             const SiteProfileHeader &profileHeader,
                             const ProfileMap &prof,
                             llvm::SmallVectorImpl<RecognizedMatmul> &out);

/// Recognize the constant-trip reduction loops of F that compute a matrix
/// product (Origin::ConstantShape), with the per-CTA extents and the
/// launch geometry from `profileHeader`.
void findConstantShapeMatmuls(const Options &O, llvm::Function &F,
                              llvm::ScalarEvolution &SE, llvm::LoopInfo &LI,
                              const SiteProfileHeader &profileHeader,
                              llvm::SmallVectorImpl<RecognizedMatmul> &out);

/// Profile generation only: the reduction trip counts (profile-scale) of the
/// scalar-loop matmuls of one profiled clone, as (fma probe index, trip) pairs
/// for the clone's profile header, so the profile-use compile sees the shape
/// at one scale. Must run on the clone without probes, which break the
/// reduction shape.
void collectScalarLoopReductionTrips(
    const Options &O, llvm::Function &F,
    llvm::SmallVectorImpl<std::pair<size_t, unsigned>> &out);

/// The library call the solve chose for a body: the product's descriptor and
/// the variant whose scheme names the runtime entry and whose
/// descriptorModulusField is the entry's last argument.
struct ChosenLibraryCall {
  /// The descriptor the recognizer derived for the product.
  GemmBodyNote note;
  /// The variant the solve chose.
  SchemeVariant variant;
};

/// A site of a joint solve whose library-call descriptor is written after
/// materialization: the kernel it is launched through and how the body's
/// parameters map to that kernel's arguments.
struct PendingGemm {
  /// The name of the wrapper kernel in the module.
  std::string wrapperName;
  std::vector<int> paramToArg; // body-param index -> wrapper launch-arg index
};

/// A tensor-core instruction shape the cost model prices.
struct InKernelTile {
  /// The tile extents in elements.
  unsigned M, N, K;
  /// The operand format.
  FPKind operand;
  /// The accumulator format.
  FPKind accumulator;
};

/// What the matrix-product pipeline records over a compile.
struct MatmulState {
  /// The library call chosen for each optimized body clone, keyed by the
  /// clone.
  std::map<const llvm::Function *, ChosenLibraryCall> GemmBodies;
  /// In the order the sites were split, so that of two sites of one wrapper the
  /// same one is recorded last in every compile.
  llvm::MapVector<const llvm::Function *, PendingGemm> PendingGemms;

  /// The tensor-core shapes the cost model prices, read at their first use in
  /// a compile.
  std::optional<llvm::SmallVector<InKernelTile, 16>> AvailableWmmaTargets;
  /// The cost-model rows a missing-row warning has named.
  llvm::StringSet<> ReportedRows;
  /// Whether -poseidon-print has said that this build has no TCEC runtime.
  bool ReportedNoTcecRuntime = false;
};

/// The optimization unit of one matrix product: its candidates, each a scheme
/// realized in the kernel or as a library call, priced per MAC of the product.
class MatmulUnit : public SiteUnit {
public:
  /// One candidate: a scheme variant, its realization, its tile, its price and
  /// its accuracy.
  struct Option {
    /// The tensor-core tile in elements of an in-kernel raise; for a library
    /// call, the tile its runtime accumulates in, whose K the accuracy model
    /// may take as the order of the sum. ~0u when unset.
    unsigned tileM = ~0u, tileN = ~0u, tileK = ~0u;
    /// The scheme, its formats and its parameter.
    SchemeVariant variant;
    /// Where the scheme runs.
    Realization realization = Realization::InKernel;

    /// The candidate's cost per MAC of the product in cost-model units
    /// (reciprocal throughput), comparable with initialCompCost; -1 unpriced.
    double compCost = -1.0;
    /// The mean over the sampled operands of the absolute error of the
    /// simulated output tile, weighted by the profiled gradient; unnormalized,
    /// it only ranks candidates.
    double accuracyCost = 0.0;
    /// Relative domain error (sensitivity-weighted, normalized) at the
    /// confidence percentile, comparable to a domain tolerance; accuracyCost
    /// is an unnormalized sum that only ranks candidates.
    double domainError = 0.0;
  };

  /// An empty unit; `matmul` is set before it is solved.
  MatmulUnit() : SiteUnit(UnitKind::Matmul) {}

  /// The product; it belongs to its function's recognition results, which
  /// outlive the unit.
  RecognizedMatmul *matmul = nullptr;
  /// The descriptor of the product's library call, when it has one.
  std::optional<GemmBodyNote> library;
  /// The cost per MAC of the loop as written in cost-model units: one
  /// iteration's reciprocal throughputs (for a runtime-shape product, one
  /// member's iteration and the epilogue spread over K).
  double initialCompCost = -1.0;
  /// The product's MACs over the profiling run, at profile scale; a profile at
  /// deploy size exceeds 2^32.
  uint64_t executions = 0;
  /// The candidates, indexed as the solver indexes them.
  llvm::SmallVector<Option, 8> candidates;
  /// Candidates not offered, by name, each with why: this build cannot realize
  /// them, their formats cannot hold the profiled operands, or the in-kernel
  /// raise would drop what else the product's loop does.
  llvm::SmallVector<std::pair<std::string, std::string>, 1> dropped;
  /// Under an accuracy target, the site's accuracy scale
  /// (SiteState::accScale), which a candidate's domainError is multiplied by
  /// on the DP's accuracy axis; none in a budget solve.
  std::optional<double> targetScale;

  /// The cost of candidate `idx` over the profiling run, less the loop's as
  /// written: the change of its per-MAC cost times `executions`, in cost-model
  /// units.
  double getCompCostDelta(size_t idx) const {
    return (candidates[idx].compCost - initialCompCost) * (double)executions;
  }
  /// The error candidate `idx` adds on the DP's accuracy axis; the loop as
  /// written has none. Under an accuracy target it is domainError times
  /// targetScale, so that divided by the site's accuracy scale it is the
  /// product's relative output error; in a budget solve, accuracyCost.
  double getAccCostDelta(size_t idx) const {
    return targetScale ? candidates[idx].domainError * *targetScale
                       : candidates[idx].accuracyCost;
  }

  /// Realizes candidate `candidateIndex` in place of the product. An error is a
  /// user error the realization finds, such as a second library call for one
  /// body.
  llvm::Error apply(Session &S, size_t candidateIndex);

  /// Whether `U` is a MatmulUnit, for isa and dyn_cast.
  static bool classof(const OptimizationUnit *U) {
    return U->kind() == unsigned(UnitKind::Matmul);
  }
  size_t numCandidates() const override { return candidates.size(); }
  double costDelta(size_t i, llvm::ArrayRef<SolutionStep>) override {
    return getCompCostDelta(i);
  }
  double errorDelta(size_t i, llvm::ArrayRef<SolutionStep>) override {
    return getAccCostDelta(i);
  }
  /// The raise and dispatch prices are measured wall-clock ratios that repeat
  /// to about 0.6% run to run.
  double costTieBand(size_t i) const override {
    return 0.006 * std::fabs(candidates[i].compCost) * (double)executions;
  }
  /// A raise replaces the loop that computes the product.
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
  llvm::Error materialize(size_t i) override { return apply(session(), i); }
  void scaleErrors(double factor) override;
  void printPriced(llvm::raw_ostream &OS) override;
  std::string keyHeader() const override {
    return "M " + std::to_string(matmul->id);
  }
  std::string selectionText(size_t i) const override;
};

/// The label of `opt` in candidate names and diagnostics.
std::string matmulOptionLabel(const MatmulUnit::Option &opt);

/// The class whose inkernel_rel cost-model row prices an in-kernel raise of
/// `v`, such as "f16_f32" or "tcec_n2_f16_f32".
std::string inKernelCostClass(const SchemeVariant &v);

/// Appends to `out` a unit with priced candidates for each product of
/// `matmuls` the profiling run executed. `sampleLogBits`: 0 samples operands
/// uniformly over the profiled [min, max]; B > 0 log-uniformly in magnitude
/// over [maxMag * 2^-B, maxMag]. An error is a user error about the profile or
/// the cost model.
llvm::Error generateMatmulCandidates(Session &S,
                                     llvm::ArrayRef<RecognizedMatmul> matmuls,
                                     const ProfileMap &scalarProfile,
                                     unsigned sampleLogBits,
                                     llvm::SmallVectorImpl<MatmulUnit> &out);

} // namespace poseidon
#endif // POSEIDON_TECHNIQUES_MATMUL_MATMUL_H
