//===- MatmulCandidates.cpp - matmul candidate proposal -------------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Matmul candidate proposal: one proposer per (scheme, realization), run in a
// fixed order by generateMatmulCandidates, with the labels, prices and
// shared-memory preconditions of the candidates and the MatmulUnit methods.
//
//===----------------------------------------------------------------------===//

#include "IR/Utils.h"
#include "MatmulInternal.h"
#include "Model/CostModel.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "Targets/Target.h"
#include "runtime/ozaki/OzakiII.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <set>
#include <string>
#include <utility>

using namespace llvm;

namespace poseidon {

static std::string mmaShapeSuffix(unsigned M, unsigned N, unsigned K) {
  return ("m" + Twine(M) + "n" + Twine(N) + "k" + Twine(K)).str();
}

// Labels name candidates in logs, reports and DP cache keys, so their irregular
// forms are kept: the library split drops its slice count, and the
// cuBLAS-backed library calls drop their tile.
std::string matmulOptionLabel(const MatmulUnit::Option &opt) {
  const SchemeVariant &v = opt.variant;
  const std::string formats = schemeFormats(v, '/');
  const std::string wmma =
      "wmma " + mmaShapeSuffix(opt.tileM, opt.tileN, opt.tileK);
  switch (opt.realization) {
  case Realization::InKernel:
    switch (v.scheme) {
    case Scheme::Direct:
      return wmma + " " + formats;
    case Scheme::Tcec:
      return std::string(schemeTag(v)) + " " + schemeParameter(v) + " " + wmma +
             " " + formats;
    case Scheme::OzakiII:
    case Scheme::Native:
      break;
    }
    break;
  case Realization::LibraryCall:
    switch (v.scheme) {
    case Scheme::OzakiII:
      return std::string(schemeTag(v)) + " " + schemeParameter(v) + " " + wmma +
             " " + formats;
    case Scheme::Tcec:
      return std::string(schemeTag(v)) + "-dispatch " + wmma + " " + formats;
    case Scheme::Direct:
      return std::string(schemeTag(v)) + "-dispatch cublas " + formats;
    case Scheme::Native:
      return std::string(schemeTag(v)) + " cublas " + formats;
    }
    break;
  }
  llvm_unreachable("a matrix-product candidate whose scheme has no such "
                   "realization");
}

// The class an in-kernel candidate's measured row is keyed by, in the same unit
// as ozaki_dispatch_rel (candidate wall clock / scalar FP64 baseline wall
// clock):
//     wmma_inkernel_rel,<class>,<rel>
// with <class> naming the candidate as the solver's table does ("f16_f32",
// "tcec_n2_f16_f32").
static std::string inKernelCostClass(const SchemeVariant &v) {
  switch (v.scheme) {
  case Scheme::Direct:
    return schemeFormats(v, '_');
  case Scheme::Tcec:
    return std::string(schemeTag(v)) + "_n" + std::to_string(sliceCount(v)) +
           "_" + schemeFormats(v, '_');
  case Scheme::OzakiII:
  case Scheme::Native:
    break;
  }
  llvm_unreachable("no in-kernel realization of this scheme");
}

// The policy for a cost-model row a candidate's price needs and the model does
// not carry: the candidate is not proposed, its price is never estimated from
// another row, and the first miss of `row` in a compilation warns, naming the
// row and the command that measures it. Strict mode makes the miss an error.
static void reportMissingRow(Session &S, const AbstractMatmul &m,
                             const Twine &what, StringRef row,
                             StringRef measure) {
  if (S.Opts.Model.StrictMode)
    reportUserError(anchorOf(m), "strict mode: " + what + " has no measured '" +
                                     row + "' row in " +
                                     costModelPath(S.CostModel) + "; " +
                                     measure + " measures it.");
  if (!S.Matmul.ReportedRows.insert(row).second)
    return;
  llvm::errs() << "[poseidon] WARNING: matmul[" << m.id << "] " << m.M << 'x'
               << m.N << 'x' << m.K << ": " << what
               << " NOT PROPOSED -- this device's cost model carries no "
                  "measured '"
               << row << "' row; " << measure << " measures it.\n";
}

// The command that measures a device's cost model; a rectangular host
// dispatch's rows also take --ozaki-rect.
constexpr StringLiteral kCalibrate = "poseidon-calibrate --gpu <id>";

// Price one in-kernel raise from its measured wmma_inkernel_rel row. A price
// composed from per-MMA rows would omit the cooperative fill, the
// shared-memory staging and the accumulator traffic.
static std::optional<double>
priceInKernelRaiseFromMeasuredRow(Session &S, const AbstractMatmul &m,
                                  const std::string &cls, double baselinePerMac,
                                  StringRef what) {
  const Options &O = S.Opts;
  double rel = lookupCost(S.CostModel, "wmma_inkernel_rel", cls).value_or(-1.0);
  if (rel > 0.0)
    return baselinePerMac * rel;

  if (O.Matmul.InKernelCalibration) {
    if (O.Solver.ApplyRewrites.empty())
      reportUserError(anchorOf(m),
                      "-poseidon-inkernel-calibration requires "
                      "-poseidon-apply-rewrites. Uncalibrated in-kernel "
                      "classes are proposed at a placeholder price under "
                      "that flag, and a DP solve must never see one.");
    llvm::errs() << "[poseidon] CALIBRATION MODE: " << what
                 << " has no measured wmma_inkernel_rel," << cls
                 << " row; proposing it at rel=1 to be timed\n";
    return baselinePerMac;
  }
  reportMissingRow(S, m, what, "wmma_inkernel_rel," + cls, kCalibrate);
  return std::nullopt;
}

// The tiles poseidon-calibrate measures, so that a missing row is always one
// it can produce.
static constexpr WmmaTarget kExpectedWmmaTargets[] = {
    {16, 16, 16, FPKind::F16, FPKind::F16},
    {16, 16, 16, FPKind::F16, FPKind::F32},
    {16, 16, 16, FPKind::BF16, FPKind::F32},
    {16, 16, 8, FPKind::TF32, FPKind::F32},
    {8, 8, 4, FPKind::F64, FPKind::F64},
};

// The tensor-core tiles the cost model has an MMA row for; a tile without one
// is not proposed.
static const SmallVector<WmmaTarget, 16> &
getAvailableWmmaTargets(Session &S, const AbstractMatmul &m) {
  if (!S.Matmul.AvailableWmmaTargets) {
    SmallVector<WmmaTarget, 16> result;
    const auto &model = getCostModel(S.CostModel);
    for (const WmmaTarget &t : kExpectedWmmaTargets) {
      std::string opcode = "wmma_mma_" + mmaShapeSuffix(t.M, t.N, t.K);
      std::string prec =
          std::string(fpKindName(t.inputPrec)) + "_" + fpKindName(t.accPrec);
      if (model.find({opcode, prec}) != model.end())
        result.push_back(t);
      else
        reportMissingRow(S, m,
                         "the wmma " + mmaShapeSuffix(t.M, t.N, t.K) + " " +
                             prec + " tile",
                         opcode + "," + prec, kCalibrate);
    }
    S.Matmul.AvailableWmmaTargets = std::move(result);
  }
  return *S.Matmul.AvailableWmmaTargets;
}

// Static shared-memory preconditions at proposal time: a module whose kernel
// exceeds Target::maxStaticSharedBytes is rejected whole, so a candidate whose
// scratch does not fit is not proposed.
//
// Bytes of shared globals already live alongside a raise placed in `F`: the
// max over the kernels that call F, or the module-wide sum when no enclosing
// kernel is found.
static uint64_t enclosingKernelSharedBytes(const Session &S, Function &F) {
  Module *mod = F.getParent();
  const Target &T = Target::of(*mod);
  const DataLayout &DL = mod->getDataLayout();
  // Only the addrspace(3) globals co-resident with this raise count, so take
  // the max over the enclosing kernels rather than a module-wide sum: another
  // kernel of the module does not share this one's shared memory.
  // A shared global counts for a function when an instruction of it or of a
  // function it calls uses the global, directly or through constant
  // expressions.
  auto sharedBytesUsedBy = [&](const Function *root,
                               SmallPtrSetImpl<const GlobalVariable *> &out) {
    SmallVector<const Function *, 8> work{root};
    SmallPtrSet<const Function *, 8> reached{root};
    while (!work.empty()) {
      const Function *cur = work.pop_back_val();
      for (const Instruction &I : instructions(*cur))
        if (auto *CB = dyn_cast<CallBase>(&I))
          if (const Function *Callee = CB->getCalledFunction())
            if (!Callee->isDeclaration() && reached.insert(Callee).second)
              work.push_back(Callee);
    }
    for (const GlobalVariable &GV : mod->globals()) {
      if (!T.isShared(GV) || !GV.getValueType()->isSized())
        continue;
      SmallVector<const User *, 8> users(GV.users());
      SmallPtrSet<const User *, 8> seen;
      while (!users.empty()) {
        const User *U = users.pop_back_val();
        if (!seen.insert(U).second)
          continue;
        if (const auto *I = dyn_cast<Instruction>(U)) {
          if (reached.contains(I->getFunction())) {
            out.insert(&GV);
            break;
          }
          continue;
        }
        append_range(users, U->users());
      }
    }
  };
  auto sizeOfSet = [&](const SmallPtrSetImpl<const GlobalVariable *> &s) {
    uint64_t b = 0;
    for (const GlobalVariable *GV : s)
      b += DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
    return b;
  };
  uint64_t bytes = 0;
  SmallPtrSet<const GlobalVariable *, 16> own;
  sharedBytesUsedBy(&F, own);
  // The kernels call the body the site clone was cut from.
  const Function *origFn = siteOriginOf(S.Sites, F);
  if (!origFn)
    origFn = &F;
  bool sawKernel = false;
  for (const User *U : origFn->users())
    if (auto *CB = dyn_cast<CallBase>(U))
      if (const Function *K = CB->getFunction())
        if (T.isKernel(*K)) {
          SmallPtrSet<const GlobalVariable *, 16> s(own.begin(), own.end());
          sharedBytesUsedBy(K, s);
          bytes = std::max(bytes, sizeOfSet(s));
          sawKernel = true;
        }
  if (!sawKernel) {
    // Module-wide sum, the conservative direction, even though a tighter answer
    // is available when F is itself the kernel.
    for (GlobalVariable &G : mod->globals())
      if (T.isShared(G) && G.getValueType()->isSized())
        bytes += DL.getTypeAllocSize(G.getValueType()).getFixedValue();
  }
  return bytes;
}

// Bytes of new static shared scratch an in-kernel raise of `opt` allocates
// (0 for a library call), per inKernelLayout. `breakdown` receives the term
// list for the refusal.
static uint64_t inKernelRaiseSharedBytes(Module &M, const AbstractMatmul &m,
                                         const MatmulUnit::Option &opt,
                                         std::string *breakdown) {
  if (opt.realization == Realization::LibraryCall)
    return 0;
  const DataLayout &DL = M.getDataLayout();
  LLVMContext &ctx = M.getContext();
  auto bytesOf = [&](FPKind k) -> uint64_t {
    return DL.getTypeAllocSize(llvmTypeForFPKind(ctx, k)).getFixedValue();
  };
  const InKernelLayout L = inKernelLayout(m, opt);
  const SchemeVariant &v = opt.variant;
  const uint64_t a = L.aElts() * bytesOf(v.operand),
                 b = L.bElts() * bytesOf(v.operand),
                 d = L.dElts() * bytesOf(v.accumulator);
  if (breakdown) {
    const std::string tile = (Twine(L.mPad) + "x" + Twine(L.nPad)).str();
    if (v.scheme == Scheme::Tcec)
      *breakdown =
          (Twine(L.slices) + " slices x " + Twine(a + b + d) + " B (A " +
           Twine(a) + " + B " + Twine(b) + " + D " + fpKindName(v.accumulator) +
           " " + Twine(d) + ", " + tile + " output tile)")
              .str();
    else
      *breakdown =
          (Twine(d) + " B D tile (" + tile + " " + fpKindName(v.accumulator) +
           ") + " + Twine(a) + " B A tile + " + Twine(b) + " B B tile")
              .str();
  }
  return L.slices * (a + b + d);
}

// Shared refusal for every in-kernel class. Returns true when the raise fits.
static bool inKernelRaiseFits(const Options &O, Function &F,
                              const AbstractMatmul &m,
                              const MatmulUnit::Option &opt,
                              uint64_t existingBytes, StringRef className) {
  std::string breakdown;
  uint64_t raiseBytes =
      inKernelRaiseSharedBytes(*F.getParent(), m, opt, &breakdown);
  if (raiseBytes == 0)
    return true;
  uint64_t total = existingBytes + raiseBytes;
  const uint64_t cap = Target::of(F).maxStaticSharedBytes();
  if (total <= cap)
    return true;
  if (O.Diagnostics.Print)
    llvm::errs() << "  Matmul[" << m.id << "] " << m.M << 'x' << m.N << 'x'
                 << m.K << ": " << className << " NOT PROPOSED -- it needs "
                 << total << " B of static shared memory against a " << cap
                 << " B cap (" << existingBytes
                 << " B already live in the enclosing kernel + " << raiseBytes
                 << " B for the raise: " << breakdown << ").\n";
  return false;
}

namespace {
struct Product {
  Session &S;
  const AbstractMatmul &m;
  // The product's own per-MAC cost: the cost-model rows are ratios to it.
  double baselinePerMac;
};
} // namespace

// The accuracy model is asked what the scheme computes, whichever code
// realizes it; only the reorder charge reads the realized tile, and the split
// library call, which scales its operands before it splits them, has its own
// simulation.
static MatmulAccuracyQuery accuracyQuery(const AbstractMatmul &m,
                                         const MatmulUnit::Option &opt) {
  const SchemeVariant &v = opt.variant;
  MatmulAccuracyQuery q;
  switch (v.scheme) {
  case Scheme::Direct:
    // A product in the source formats is exact to the rounding model, so its
    // tile's accumulation order is charged through orderTileK; otherwise it
    // prices as zero error and enters the budget-0 solution.
    q.inputPrec = v.operand;
    q.accPrec = v.accumulator;
    if (v.operand == m.elemType && v.accumulator == m.elemType)
      q.orderTileK = opt.tileK;
    return q;
  case Scheme::Tcec:
    if (opt.realization == Realization::LibraryCall) {
      q.model = MatmulAccuracyQuery::Model::TcecLibrary;
      return q;
    }
    q.inputPrec = FPKind::F32;
    q.accPrec = FPKind::F32;
    q.exponentPrec = v.operand;
    q.inputMantBits = splitCapturedBits(v);
    return q;
  case Scheme::OzakiII:
    q.model = MatmulAccuracyQuery::Model::OzakiII;
    q.moduli = modulusCount(v);
    return q;
  case Scheme::Native:
    // Full FP64 but not the scalar chain's accumulation order, so the reorder
    // is priced through orderTileK instead of claiming exact zero.
    q.inputPrec = v.operand;
    q.accPrec = v.accumulator;
    q.orderTileK = opt.tileK;
    return q;
  }
  llvm_unreachable("unknown Scheme");
}

static bool wmmaTileAvailable(Session &S, const AbstractMatmul &m, unsigned M,
                              unsigned N, unsigned K, FPKind operand,
                              FPKind accumulator) {
  return llvm::any_of(getAvailableWmmaTargets(S, m), [&](const WmmaTarget &wt) {
    return wt.M == M && wt.N == N && wt.K == K && wt.inputPrec == operand &&
           wt.accPrec == accumulator;
  });
}

//===----------------------------------------------------------------------===//
// In-kernel realization. Only a constant-trip product has one: the raise needs
// compile-time tile extents. It allocates addrspace(3) scratch in the body, so
// the enclosing kernel's static shared budget is a precondition, and it is
// priced from the measured wmma_inkernel_rel row of its class, with no shape
// ratio on top.
//===----------------------------------------------------------------------===//

namespace {
struct InKernelSite {
  Function *raiseF = nullptr;
  uint64_t existingShmem = 0;
};
} // namespace

// Whether every thread of the block reaches the call of the site exactly once,
// as the block barriers the raise puts inside the site need: the call
// post-dominates its kernel's entry and lies in no loop.
static bool siteReachedByWholeBlock(const Session &S, const AbstractMatmul &m) {
  const Function &F = *m.scalarLoop.fma->getFunction();
  const CallBase *site = siteCallOf(S, F);
  if (site) {
    Function &kernel = *const_cast<Function *>(site->getFunction());
    DominatorTree DT(kernel);
    LoopInfo LI(DT);
    PostDominatorTree PDT(kernel);
    const BasicBlock *BB = site->getParent();
    if (!LI.getLoopFor(BB) && PDT.dominates(BB, &kernel.getEntryBlock()))
      return true;
  }
  if (S.Opts.Diagnostics.Print)
    llvm::errs() << "[raise-wmma] " << F.getName()
                 << ": not every thread of the block reaches the site in "
                 << (site ? site->getFunction()->getName() : "its kernel")
                 << " once, so it has no in-kernel raise\n";
  return false;
}

static InKernelSite inKernelSite(const Session &S, const AbstractMatmul &m) {
  InKernelSite K;
  K.raiseF = m.scalarLoop.fma->getFunction();
  K.existingShmem = enclosingKernelSharedBytes(S, *K.raiseF);
  return K;
}

static std::string inKernelDescription(const SchemeVariant &v, unsigned tileM,
                                       unsigned tileN, unsigned tileK) {
  const std::string tile =
      mmaShapeSuffix(tileM, tileN, tileK) + " " + schemeFormats(v, '/');
  switch (v.scheme) {
  case Scheme::Direct:
    return "in-kernel Direct raise " + tile;
  case Scheme::Tcec:
    return "in-kernel TCEC " + schemeParameter(v) + " " + tile;
  case Scheme::OzakiII:
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
  const std::string what = inKernelDescription(v, tileM, tileN, tileK);

  // Buildability before price: a shape whose tiles do not fit the static
  // shared cap is rejected by ptxas for the whole module.
  if (!inKernelRaiseFits(P.S.Opts, *K.raiseF, m, opt, K.existingShmem, what))
    return;
  // An uncalibrated class is refused, not estimated.
  std::optional<double> cost = priceInKernelRaiseFromMeasuredRow(
      P.S, m, inKernelCostClass(v), P.baselinePerMac, what);
  if (!cost)
    return;
  opt.compCost = *cost;
  cm.candidates.push_back(opt);
}

static void proposeInKernelDirect(const Product &P, const InKernelSite &K,
                                  MatmulUnit &cm) {
  for (const WmmaTarget &t : getAvailableWmmaTargets(P.S, P.m))
    proposeInKernel(P, K, SchemeVariant::direct(t.inputPrec, t.accPrec), t.M,
                    t.N, t.K, cm);
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
    {16, 16, 16, SchemeVariant::tcec(FPKind::F16, FPKind::F32, 2)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::F16, FPKind::F32, 3)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::F16, FPKind::F32, 5)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::BF16, FPKind::F32, 2)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::F32, 2)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::F32, 3)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::F32, 5)},
    // Integer slices (s8) are intentionally not proposed: in-kernel they have
    // the same structural slowdown as an in-kernel modular product, and the
    // modular library calls cover their range.
};

static void proposeInKernelSplit(const Product &P, const InKernelSite &K,
                                 MatmulUnit &cm) {
  for (const InKernelSplitVariant &s : kInKernelSplitVariants) {
    if (!wmmaTileAvailable(P.S, P.m, s.tileM, s.tileN, s.tileK,
                           s.variant.operand, s.variant.accumulator))
      continue;
    // No size gate: the padded fill zero-fills boundary tiles and every slice
    // outside the valid extent, so padding is numerically inert. The one
    // precondition is the static shared cap (N slice buffers per side plus N
    // F32 tiles live at once), a capability limit rather than a price.
    proposeInKernel(P, K, s.variant, s.tileM, s.tileN, s.tileK, cm);
  }
}

//===----------------------------------------------------------------------===//
// Library-call realization, proposed under -poseidon-ozaki-host-dispatch. The
// body must be a GEMM the runtime can take over (recognition already
// established it for a runtime-shape product). Each scheme is priced from
// `baselinePerMac` times its measured *_dispatch_rel row, times the shape rule
// the scheme's runtime needs to extrapolate that row to this shape.
//===----------------------------------------------------------------------===//

namespace {
struct LibrarySite {
  // The library call's descriptor; none when no library call can take the
  // place of the product.
  std::optional<GemmBodyNote> note;
  bool dispatchable = false;
  // The product's extents at profile scale. A square product is priced from
  // the modular row "ozaki_dispatch_rel,nmNN", measured on a square; any other
  // from "ozaki_dispatch_rect_rel,nmNN_m<M>k<K>".
  unsigned shapeM = 0, shapeN = 0, shapeK = 0;
  // The extents the dispatch is compiled with, when all three are constants of
  // this compile (a scalar-loop product's descriptor); 0 when one is a
  // run-time value.
  unsigned staticM = 0, staticN = 0, staticK = 0;
};
} // namespace

static LibrarySite librarySite(Session &S, const AbstractMatmul &m) {
  LibrarySite L;
  L.note = libraryCallNote(S, m);
  const std::optional<GemmBodyNote> &note = L.note;
  L.dispatchable = note.has_value();
  if (S.Opts.Diagnostics.Print && !L.dispatchable)
    llvm::errs() << "  Matmul[" << m.id << "] " << m.M << 'x' << m.N << 'x'
                 << m.K
                 << ": host-dispatched Ozaki-II NOT PROPOSED -- the body "
                    "is not a host-dispatchable GEMM.\n";
  if (note && m.origin == AbstractMatmul::Origin::ScalarLoopReduction) {
    L.staticM = (unsigned)note->dims[DimM].add;
    L.staticN = (unsigned)note->dims[DimNcols].add;
    L.staticK = (unsigned)note->dims[DimK].add;
  }
  L.shapeM = m.globalM ? m.globalM : m.M;
  L.shapeN = m.globalN ? m.globalN : m.N;
  L.shapeK = m.globalK ? m.globalK : m.K;
  return L;
}

static MatmulUnit::Option libraryOption(const SchemeVariant &v, unsigned tileM,
                                        unsigned tileN, unsigned tileK) {
  MatmulUnit::Option opt;
  opt.tileM = tileM;
  opt.tileN = tileN;
  opt.tileK = tileK;
  opt.variant = v;
  opt.realization = Realization::LibraryCall;
  return opt;
}

// A square product is priced from the row measured on a square; any other only
// from the row measured at its own (M, K), since the runtime's per-MAC cost
// depends on M and K but not on the column count. A modulus count without its
// row is not proposed.
static void proposeLibraryModular(const Product &P, const LibrarySite &L,
                                  MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  const AbstractMatmul &m = P.m;
  // The runtime's INT32 residue sums hold a reduction shorter than
  // kOzakiIIMaxK, at the scale the call runs at and at the one simulated.
  const unsigned reduction = std::max({L.staticK, L.shapeK, m.K});
  if (reduction >= (unsigned)kOzakiIIMaxK) {
    if (P.S.Opts.Diagnostics.Print)
      llvm::errs() << "  Matmul[" << m.id << "] ozaki-ii NOT PROPOSED -- its "
                   << "reduction of " << reduction
                   << " is not below the runtime's limit " << kOzakiIIMaxK
                   << ".\n";
    return;
  }
  const bool square = L.shapeM == L.shapeN && L.shapeN == L.shapeK;
  const std::string shapeKey =
      "m" + std::to_string(L.shapeM) + "k" + std::to_string(L.shapeK);
  const std::string family =
      square ? "ozaki_dispatch_rel" : "ozaki_dispatch_rect_rel";
  std::string missing;
  for (unsigned nm : kOzIIModuliCounts) {
    const SchemeVariant v = SchemeVariant::ozakiII(nm);
    const std::string key =
        "nm" + std::to_string(nm) + (square ? std::string() : "_" + shapeKey);
    const double rel = lookupCost(P.S.CostModel, family, key).value_or(-1.0);
    if (rel <= 0.0) {
      missing += (missing.empty() ? "" : ", ") + std::to_string(nm);
      continue;
    }
    MatmulUnit::Option opt =
        libraryOption(v, kOzakiIIPadTile, kOzakiIIPadTile, kOzakiIIPadTile);
    opt.compCost = P.baselinePerMac * rel;
    cm.candidates.push_back(opt);
  }
  // One row name for all the moduli of the shape.
  if (!missing.empty())
    reportMissingRow(
        P.S, m, "ozaki-ii host dispatch at nm=" + missing,
        family + ",nm<nm>" + (square ? std::string() : "_" + shapeKey),
        square ? std::string(kCalibrate)
               : std::string(kCalibrate) + " --ozaki-rect " +
                     std::to_string(L.shapeM) + "x" + std::to_string(L.shapeK));
}

// The split scheme as __poseidon_tcec_dgemm, which realizes a different rate
// than the in-kernel raise and so is a distinct candidate. Absent its measured
// tcec_dispatch_rel row it is not proposed, and in a build without cuMpSGEMM,
// which has no such runtime, a pin that names it is told why. The runtime
// repacks operands rather than padding to a square, so no shape ratio is
// charged.
static void proposeLibrarySplit(const Product &P, const LibrarySite &L,
                                MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  MatmulUnit::Option opt = libraryOption(
      SchemeVariant::tcec(FPKind::F16, FPKind::F32, 2), 16, 16, 16);
  if (!kHaveTcecDispatch) {
    if (P.S.Opts.Diagnostics.Print &&
        !std::exchange(P.S.Matmul.ReportedNoTcecRuntime, true))
      llvm::errs() << "[poseidon] WARNING: matmul[" << P.m.id << "] " << P.m.M
                   << 'x' << P.m.N << 'x' << P.m.K
                   << ": TCEC dispatch NOT PROPOSED -- this build has no TCEC "
                      "runtime; configure it with "
                      "-DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>.\n";
    cm.dropped.push_back(
        {matmulOptionLabel(opt),
         "this compiler was built without cuMpSGEMM, on which the "
         "error-corrected dispatch runs; configure it with "
         "-DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>"});
    return;
  }
  double rel =
      lookupCost(P.S.CostModel, "tcec_dispatch_rel", "fp16tcec").value_or(-1.0);
  if (rel <= 0.0) {
    reportMissingRow(P.S, P.m, "tcec host dispatch",
                     "tcec_dispatch_rel,fp16tcec", kCalibrate);
    return;
  }
  opt.compCost = P.baselinePerMac * rel;
  cm.candidates.push_back(opt);
}

// The direct scheme as __poseidon_direct_dgemm: each FP64 operand is rounded
// once to the library's input format and one library GEMM accumulates in FP32
// (runtime/direct/direct_rt.cu). Same numerics as the in-kernel raise up to
// accumulation order, but a distinct realized rate. Priced only from this
// device's measured direct_dispatch_rel,<class> row; absent the row it is not
// proposed. The runtime repacks operands rather than padding to a square, so
// no shape ratio is charged.
//
// f32/f32 is cuBLAS's SGEMM, reached by changing the compute type alone; it
// does not run on tensor cores and has no MMA tile, so its tile triple is 0.
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
        lookupCost(P.S.CostModel, "direct_dispatch_rel", cls).value_or(-1.0);
    if (rel <= 0.0) {
      reportMissingRow(
          P.S, m, Twine(schemeTag(d.variant)) + " " + cls + " host dispatch",
          "direct_dispatch_rel," + cls, kCalibrate);
      continue;
    }
    MatmulUnit::Option opt =
        libraryOption(d.variant, d.tileM, d.tileN, d.tileK);
    opt.compCost = P.baselinePerMac * rel;
    cm.candidates.push_back(opt);
  }
}

// The native product (__poseidon_native_dgemm), one cublasDgemm on the real
// M x Ncols x K for every layout combination. Its per-MAC rate depends on the
// shape, so a call whose extents are constants of this compile is priced from
// the row measured at those extents when the model has one. A run-time extent
// leaves only the square row: the surrogate's shape is not the one the call
// runs at.
static void proposeLibraryNative(const Product &P, const LibrarySite &L,
                                 MatmulUnit &cm) {
  if (!L.dispatchable)
    return;
  const AbstractMatmul &m = P.m;
  // Extents of this compile are priced from the row measured at them, a
  // square one also from the square row; run-time extents, whose surrogate
  // shape is not the one the call runs at, from the square row.
  const bool isStatic = L.staticM && L.staticN && L.staticK;
  const bool square = L.staticM == L.staticN && L.staticN == L.staticK;
  const std::string shape = std::to_string(L.staticM) + "x" +
                            std::to_string(L.staticN) + "x" +
                            std::to_string(L.staticK);
  const std::string rectKey = "dgemm_m" + std::to_string(L.staticM) + "n" +
                              std::to_string(L.staticN) + "k" +
                              std::to_string(L.staticK);
  double rel =
      isStatic ? lookupCost(P.S.CostModel, "ozaki_dispatch_rect_rel", rectKey)
                     .value_or(-1.0)
               : -1.0;
  if (rel <= 0.0 && isStatic && !square) {
    reportMissingRow(P.S, m, "native-dgemm host dispatch at " + shape,
                     "ozaki_dispatch_rect_rel," + rectKey,
                     std::string(kCalibrate) + " --ozaki-rect " + shape);
    return;
  }
  if (rel <= 0.0)
    rel =
        lookupCost(P.S.CostModel, "ozaki_dispatch_rel", "dgemm").value_or(-1.0);
  if (rel <= 0.0) {
    reportMissingRow(P.S, m, "native-dgemm host dispatch",
                     "ozaki_dispatch_rel,dgemm", kCalibrate);
    return;
  }
  MatmulUnit::Option opt = libraryOption(SchemeVariant::native(), 16, 16, 16);
  opt.compCost = P.baselinePerMac * rel;
  cm.candidates.push_back(opt);
}

void generateMatmulCandidates(Session &S, ArrayRef<AbstractMatmul> matmuls,
                              const ProfileMap &scalarProfile,
                              unsigned sampleLogBits,
                              SmallVectorImpl<MatmulUnit> &out) {
  for (const AbstractMatmul &m : matmuls) {
    std::optional<MatmulProfile> prof = loadProfile(m, scalarProfile);
    if (!prof) {
      llvm::errs() << "[poseidon] matmul[" << m.id << "] in '"
                   << anchorOf(m).getFunction()->getName()
                   << "': the profiling run never executed it; left as "
                      "written.\n";
      continue;
    }

    MatmulUnit cm;
    cm.matmul = const_cast<AbstractMatmul *>(&m);
    cm.executions = prof->totalCalls;

    double baselinePerMac = 0.0;
    switch (m.origin) {
    case AbstractMatmul::Origin::ScalarLoopReduction:
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
    }
    cm.initialCompCost = baselinePerMac;
    const Product P{S, m, baselinePerMac};

    // The candidate index is the position in this sequence.
    if (m.origin == AbstractMatmul::Origin::ScalarLoopReduction &&
        m.scalarLoop.inKernelRaisable && siteReachedByWholeBlock(S, m)) {
      const InKernelSite K = inKernelSite(S, m);
      proposeInKernelDirect(P, K, cm);
      proposeInKernelSplit(P, K, cm);
    }
    if (S.Opts.Matmul.OzakiHostDispatch) {
      const LibrarySite L = librarySite(S, m);
      cm.library = L.note;
      proposeLibraryModular(P, L, cm);
      proposeLibrarySplit(P, L, cm);
      proposeLibraryDirect(P, L, cm);
      proposeLibraryNative(P, L, cm);
    }
    SmallVector<MatmulAccuracyQuery> queries;
    for (const MatmulUnit::Option &opt : cm.candidates)
      queries.push_back(accuracyQuery(m, opt));
    SmallVector<MatmulAccuracy> accuracy(queries.size());
    evaluateMatmulAccuracy(S, m, *prof, sampleLogBits, queries, accuracy);
    for (auto [opt, a] : zip_equal(cm.candidates, accuracy)) {
      opt.accuracyCost = a.accuracyCost;
      opt.domainError = a.domainError;
    }

    if (m.origin == AbstractMatmul::Origin::HostGemmLoopNest &&
        S.Opts.Diagnostics.Print)
      llvm::errs() << "[hostgemm] matmul[" << m.id
                   << "] baseline: composed/MAC=" << baselinePerMac << '\n';

    // compCost is per-MAC reciprocal throughput; rel is it divided by the naive
    // per-MAC baseline, the unit the measured *_dispatch_rel rows use.
    if (S.Opts.Diagnostics.Print) {
      llvm::errs() << "[poseidon] matmul[" << cm.matmul->id
                   << "] pricing: baseline/MAC=" << cm.initialCompCost
                   << " executions=" << cm.executions << '\n';
      for (size_t ci = 0; ci < cm.candidates.size(); ++ci) {
        const auto &o = cm.candidates[ci];
        llvm::errs() << "[poseidon]   #" << ci << ' ' << matmulOptionLabel(o)
                     << "  compCost/MAC=" << o.compCost << "  rel="
                     << (cm.initialCompCost > 0
                             ? o.compCost / cm.initialCompCost
                             : -1.0)
                     << "  domainError=" << o.domainError << '\n';
      }
    }

    out.push_back(std::move(cm));
  }
}

void MatmulUnit::apply(Session &S, size_t candidateIndex) {
  if (candidateIndex >= candidates.size())
    report_fatal_error("MatmulUnit index out of range");
  const Option &opt = candidates[candidateIndex];

  if (opt.realization == Realization::LibraryCall) {
    assert(library && "a library call is proposed only with its descriptor");
    applyLibraryCall(S, *matmul, *library, opt.variant);
    return;
  }
  switch (matmul->origin) {
  case AbstractMatmul::Origin::HostGemmLoopNest:
    report_fatal_error("a runtime-shape host GEMM was selected with a "
                       "scheme that has no host-dispatch realization");
  case AbstractMatmul::Origin::ScalarLoopReduction:
    break;
  }
  switch (opt.variant.scheme) {
  case Scheme::Direct:
  case Scheme::Tcec: {
    materializeInKernelRaise(*matmul, opt);
    // Each raise was proposed against the shared memory its kernel used
    // before any raise; raises of one kernel that do not share their scratch
    // add up.
    Function &F = *matmul->scalarLoop.preheader->getParent();
    const uint64_t bytes = enclosingKernelSharedBytes(S, F);
    const uint64_t cap = Target::of(F).maxStaticSharedBytes();
    if (bytes > cap)
      reportUserError(F, F.getSubprogram(),
                      "the in-kernel raises selected for the matrix products "
                      "of " +
                          F.getName() + " need " + Twine(bytes) +
                          " B of static shared memory with what their kernel "
                          "uses, above the " +
                          Twine(cap) +
                          " B a kernel may hold; select fewer of them with "
                          "-poseidon-apply-rewrites.");
    return;
  }
  case Scheme::OzakiII:
  case Scheme::Native:
    break;
  }
  llvm_unreachable("no in-kernel realization of this scheme");
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

void MatmulUnit::scaleErrors(double factor) {
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
       << " ΔaccCost=" << getAccCostDelta(i) << '\n';
}

// Dispatch and raise prices are measured wall-clock ratios that repeat to about
// 0.6% run to run, so a smaller modelled cost gap is a tie.
static constexpr double kCostTieBandRel = 0.006;

SmallVector<SolutionStep> errorBudgetSelector(const Options &O,
                                              SmallVectorImpl<MatmulUnit> &CMs,
                                              double budget) {
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
                     << budget << " at confidence " << kDefaultConfidence
                     << " faster than F64)\n";
      continue;
    }
    if (O.Diagnostics.Print)
      llvm::errs() << "Poseidon error-budget: matmul -> "
                   << matmulOptionLabel(cm.candidates[best]) << " (domain err "
                   << cm.candidates[best].domainError << " <= " << budget
                   << " at confidence " << kDefaultConfidence << ")\n";
    steps.emplace_back(&cm, (size_t)best);
  }
  return steps;
}

} // namespace poseidon
