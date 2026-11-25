//===- Matmul.cpp - matmul candidate proposal -----------------------------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
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

#include "Internal.h"
#include "Model/CostModel.h"
#include "Pass/Diagnostics.h"
#include "Pass/Session.h"
#include "Sites/SiteIdentity.h"
#include "Support/Utils.h"
#include "Targets/Target.h"
#include "runtime/gemm/Schemes.h"

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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
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
// as library_ozaki_rel (candidate wall clock / scalar FP64 baseline wall
// clock):
//     inkernel_rel,<class>,<rel>
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

// The first miss of `row` in a compilation warns that `what` `consequence`,
// naming the row and the command that measures it. Strict mode makes the miss
// an error.
static Error reportRow(Session &S, const RecognizedMatmul &m, const Twine &what,
                       StringRef row, StringRef measure,
                       const Twine &consequence) {
  if (S.Opts.Model.StrictMode)
    return userError(anchorOf(m), "strict mode: " + what +
                                      " has no measured '" + row + "' row in " +
                                      costModelPath(S.CostModel) + "; " +
                                      measure + " measures it.");
  if (!S.Matmul.ReportedRows.insert(row).second)
    return Error::success();
  const Instruction &at = anchorOf(m);
  diagnose(*at.getFunction(), at.getDebugLoc(),
           "matrix product " + Twine(m.id) + ", " + Twine(m.M) + "x" +
               Twine(m.N) + "x" + Twine(m.K) + ": " + what + " " + consequence +
               ": this device's cost model carries no measured '" + row +
               "' row; " + measure + " measures it",
           DS_Warning);
  return Error::success();
}

// The policy for a cost-model row a candidate's price needs and the model does
// not carry: the candidate is not proposed, unless its pricing names an
// estimate (see reportEstimatedRow).
static Error reportMissingRow(Session &S, const RecognizedMatmul &m,
                              const Twine &what, StringRef row,
                              StringRef measure) {
  return reportRow(S, m, what, row, measure, "is not proposed");
}

// A candidate priced from another row than its own, as the estimate `how`
// says, until its row is measured.
static Error reportEstimatedRow(Session &S, const RecognizedMatmul &m,
                                const Twine &what, StringRef row,
                                StringRef measure, const Twine &how) {
  return reportRow(S, m, what, row, measure, "is priced " + how);
}

// The measured rows a matrix-product candidate is priced from, each the
// candidate's wall clock over the scalar FP64 loop's, and the shape each is
// read at:
//   inkernel_rel,<class>_m<M>n<N>k<K>_<layout>
//       An in-kernel raise of a product whose extents are constants of the
//       compile (inKernelShape), measured at them with B's layout along K:
//       bcolmajor contiguous, browmajor strided.
//   inkernel_rel,<class>
//       An in-kernel raise of a product whose output extents are run-time
//       values, and the estimate of a missing row of the form above. Measured
//       on a square product at the calibration's --n (2048 by default), B
//       strided along K, whose launch saturates the device; a product with
//       few output columns or with B contiguous along K does not run at the
//       rate it gives.
//   library_ozaki_rect_rel,nm<k>_m<M>n<N>k<K> and ,dgemm_m<M>n<N>k<K>
//       A library call of constant extents (LibrarySite::runShape), measured
//       at them.
//   library_ozaki_rect_rel,nm<k>_m<M>k<K>
//       (M, K) of LibrarySite::profileShape, for a library call whose extents
//       are run-time values; measured at many columns.
//   library_ozaki_rel,nm<k> and ,dgemm
//       A square, read as they are, and for a square runShape. The estimate of
//       a missing rectangular Ozaki-II row multiplies the square row by the
//       padding of profileShape (squarePaddingRatio), that of a missing
//       native DGEMM row is the square row.
//   library_tcec_rel,fp16tcec and library_direct_rel,<class>
//       Every shape.
// The command that measures a device's cost model; the rows of the extents of
// one product also take --product MxNxK[:layout], and the (M, K) rows
// --ozaki-rect MxK.
constexpr StringLiteral kCalibrate = "poseidon-calibrate --gpu <id>";

// Layouts of B along the reduction, as the exact in-kernel rows name them.
constexpr StringLiteral kBColMajor = "bcolmajor", kBRowMajor = "browmajor";

namespace {
// The extents an in-kernel raise runs at and the layout of its B, which key
// its exact rows.
struct InKernelShape {
  unsigned M, N, K;
  StringRef layout;
  // "m<M>n<N>k<K>_<layout>".
  std::string key() const {
    return ("m" + Twine(M) + "n" + Twine(N) + "k" + Twine(K) + "_" + layout)
        .str();
  }
  // The --product argument that measures its rows.
  std::string product() const {
    return (Twine(M) + "x" + Twine(N) + "x" + Twine(K) + ":" + layout).str();
  }
};
} // namespace

// The shape of the scalar-loop product `m` at the reduction length of the
// compile, its output extents as constantOutputExtents gives them from the
// guards before the loop; none when they are run-time values.
static std::optional<InKernelShape> inKernelShape(const RecognizedMatmul &m) {
  const ScalarLoopHandle &h = m.scalarLoop;
  std::optional<unsigned> axM = tidAxisIndex(h.aRow.fast),
                          axN = tidAxisIndex(h.bCol.fast);
  if (!axM || !axN || *axM == *axN)
    return std::nullopt;
  std::optional<OutputExtents> out =
      constantOutputExtents(m, h.loopGuard, *axM, *axN);
  if (!out)
    return std::nullopt;
  // A steps by one element along the reduction.
  const StringRef layout =
      h.bStrideByte == h.aStrideByte ? kBColMajor : kBRowMajor;
  return InKernelShape{out->M, out->Ncols, m.K, layout};
}

// Price one in-kernel raise of class `cls` from its measured row: the row of
// `shape` when the product has one, the class row otherwise. A missing row of
// `shape` is estimated by the class row, and `cls` added to `estimated`. A
// price composed from per-MMA rows would omit the cooperative fill, the
// shared-memory staging and the accumulator traffic.
static Expected<std::optional<double>> priceInKernelRaiseFromMeasuredRow(
    Session &S, const RecognizedMatmul &m, const std::string &cls,
    const std::optional<InKernelShape> &shape, double baselinePerMac,
    StringRef what, SmallVectorImpl<std::string> &estimated) {
  const Options &O = S.Opts;
  if (shape) {
    double rel =
        lookupCost(S.CostModel, "inkernel_rel", cls + "_" + shape->key())
            .value_or(-1.0);
    if (rel > 0.0)
      return std::optional<double>(baselinePerMac * rel);
  }
  double rel = lookupCost(S.CostModel, "inkernel_rel", cls).value_or(-1.0);
  if (rel > 0.0) {
    if (shape)
      estimated.push_back(cls);
    return std::optional<double>(baselinePerMac * rel);
  }

  if (O.Matmul.MatmulCalibration) {
    if (O.Solver.Pin.empty())
      return userError(anchorOf(m),
                       "-poseidon-matmul-calibration requires "
                       "-poseidon-pin. Uncalibrated in-kernel "
                       "classes are proposed at a placeholder price under "
                       "that flag, and a DP solve must never see one.");
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] calibration mode: " << what
                   << " has no measured inkernel_rel," << cls
                   << " row; proposing it at rel=1 to be timed\n";
    return std::optional<double>(baselinePerMac);
  }
  if (Error E = reportMissingRow(S, m, what, "inkernel_rel," + cls, kCalibrate))
    return std::move(E);
  return std::optional<double>();
}

// The tiles poseidon-calibrate measures, so that a missing row is always one
// it can produce.
static constexpr InKernelTile kInKernelTiles[] = {
    {16, 16, 16, FPKind::FP16, FPKind::FP16},
    {16, 16, 16, FPKind::FP16, FPKind::FP32},
    {16, 16, 16, FPKind::BF16, FPKind::FP32},
    {16, 16, 8, FPKind::TF32, FPKind::FP32},
    {8, 8, 4, FPKind::FP64, FPKind::FP64},
};

// The tensor-core tiles the cost model has an MMA row for; a tile without one
// is not proposed.
static Expected<const SmallVector<InKernelTile, 16> *>
getAvailableWmmaTargets(Session &S, const RecognizedMatmul &m) {
  if (!S.Matmul.AvailableWmmaTargets) {
    SmallVector<InKernelTile, 16> result;
    const auto &model = getCostModel(S.CostModel);
    for (const InKernelTile &t : kInKernelTiles) {
      std::string opcode = "wmma_mma_" + mmaShapeSuffix(t.M, t.N, t.K);
      std::string prec =
          std::string(fpKindName(t.operand)) + "_" + fpKindName(t.accumulator);
      if (model.find({opcode, prec}) != model.end())
        result.push_back(t);
      else if (Error E = reportMissingRow(S, m,
                                          "the wmma " +
                                              mmaShapeSuffix(t.M, t.N, t.K) +
                                              " " + prec + " tile",
                                          opcode + "," + prec, kCalibrate))
        return std::move(E);
    }
    S.Matmul.AvailableWmmaTargets = std::move(result);
  }
  return &*S.Matmul.AvailableWmmaTargets;
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
static uint64_t inKernelRaiseSharedBytes(Module &M, const RecognizedMatmul &m,
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
                              const RecognizedMatmul &m,
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
    llvm::errs() << "[poseidon] matmul[" << m.id << "] " << m.M << 'x' << m.N
                 << 'x' << m.K << ": " << className
                 << " not proposed: it needs " << total
                 << " B of static shared memory against a " << cap << " B cap ("
                 << existingBytes
                 << " B already live in the enclosing kernel + " << raiseBytes
                 << " B for the raise: " << breakdown << ")\n";
  return false;
}

namespace {
struct Product {
  Session &S;
  const RecognizedMatmul &m;
  // The product's own per-MAC cost: the cost-model rows are ratios to it.
  double baselinePerMac;
};
} // namespace

// The accuracy model is asked what the scheme computes, whichever code
// realizes it; only the reorder charge reads the realized tile, and the split
// library call, which scales its operands before it splits them, has its own
// simulation.
static MatmulAccuracyQuery accuracyQuery(const RecognizedMatmul &m,
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
    q.inputPrec = FPKind::FP32;
    q.accPrec = FPKind::FP32;
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

static Expected<bool> wmmaTileAvailable(Session &S, const RecognizedMatmul &m,
                                        unsigned M, unsigned N, unsigned K,
                                        FPKind operand, FPKind accumulator) {
  Expected<const SmallVector<InKernelTile, 16> *> targets =
      getAvailableWmmaTargets(S, m);
  if (!targets)
    return targets.takeError();
  return llvm::any_of(**targets, [&](const InKernelTile &wt) {
    return wt.M == M && wt.N == N && wt.K == K && wt.operand == operand &&
           wt.accumulator == accumulator;
  });
}

//===----------------------------------------------------------------------===//
// In-kernel realization. Only a constant-trip product has one: the raise needs
// compile-time tile extents. It allocates addrspace(3) scratch in the body, so
// the enclosing kernel's static shared budget is a precondition, and it is
// priced from the measured inkernel_rel row of its class at its shape,
// with no shape ratio on top.
//===----------------------------------------------------------------------===//

namespace {
struct InKernelSite {
  Function *raiseF = nullptr;
  uint64_t existingShmem = 0;
  std::optional<InKernelShape> shape;
};
} // namespace

// Whether every thread of the block reaches the call of the site exactly once,
// as the block barriers the raise puts inside the site need: the call
// post-dominates its kernel's entry and lies in no loop.
static bool siteReachedByWholeBlock(const Session &S,
                                    const RecognizedMatmul &m) {
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
    llvm::errs() << "[poseidon] " << F.getName()
                 << ": not every thread of the block reaches the site in "
                 << (site ? site->getFunction()->getName() : "its kernel")
                 << " once, so it has no in-kernel raise\n";
  return false;
}

static InKernelSite inKernelSite(const Session &S, const RecognizedMatmul &m) {
  InKernelSite K;
  K.raiseF = m.scalarLoop.fma->getFunction();
  K.existingShmem = enclosingKernelSharedBytes(S, *K.raiseF);
  K.shape = inKernelShape(m);
  return K;
}

static std::string inKernelDescription(const SchemeVariant &v, unsigned tileM,
                                       unsigned tileN, unsigned tileK) {
  const std::string tile =
      mmaShapeSuffix(tileM, tileN, tileK) + " " + schemeFormats(v, '/');
  switch (v.scheme) {
  case Scheme::Direct:
    return "in-kernel direct raise " + tile;
  case Scheme::Tcec:
    return "in-kernel TCEC " + schemeParameter(v) + " " + tile;
  case Scheme::OzakiII:
  case Scheme::Native:
    break;
  }
  llvm_unreachable("no in-kernel realization of this scheme");
}

static Error proposeInKernel(const Product &P, const InKernelSite &K,
                             const SchemeVariant &v, unsigned tileM,
                             unsigned tileN, unsigned tileK, MatmulUnit &cm,
                             SmallVectorImpl<std::string> &estimated) {
  const RecognizedMatmul &m = P.m;
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
    return Error::success();
  // A class without a row is refused, not estimated.
  Expected<std::optional<double>> cost = priceInKernelRaiseFromMeasuredRow(
      P.S, m, inKernelCostClass(v), K.shape, P.baselinePerMac, what, estimated);
  if (!cost)
    return cost.takeError();
  if (!*cost)
    return Error::success();
  opt.compCost = **cost;
  cm.candidates.push_back(opt);
  return Error::success();
}

static Error proposeInKernelDirect(const Product &P, const InKernelSite &K,
                                   MatmulUnit &cm,
                                   SmallVectorImpl<std::string> &estimated) {
  Expected<const SmallVector<InKernelTile, 16> *> targets =
      getAvailableWmmaTargets(P.S, P.m);
  if (!targets)
    return targets.takeError();
  for (const InKernelTile &t : **targets)
    if (Error E = proposeInKernel(
            P, K, SchemeVariant::direct(t.operand, t.accumulator), t.M, t.N,
            t.K, cm, estimated))
      return E;
  return Error::success();
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
    {16, 16, 16, SchemeVariant::tcec(FPKind::FP16, FPKind::FP32, 2)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::FP16, FPKind::FP32, 3)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::FP16, FPKind::FP32, 5)},
    {16, 16, 16, SchemeVariant::tcec(FPKind::BF16, FPKind::FP32, 2)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::FP32, 2)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::FP32, 3)},
    {16, 16, 8, SchemeVariant::tcec(FPKind::TF32, FPKind::FP32, 5)},
    // Integer slices (s8) are intentionally not proposed: in-kernel they have
    // the same structural slowdown as an in-kernel modular product, and the
    // modular library calls cover their range.
};

static Error proposeInKernelSplit(const Product &P, const InKernelSite &K,
                                  MatmulUnit &cm,
                                  SmallVectorImpl<std::string> &estimated) {
  for (const InKernelSplitVariant &s : kInKernelSplitVariants) {
    Expected<bool> available =
        wmmaTileAvailable(P.S, P.m, s.tileM, s.tileN, s.tileK,
                          s.variant.operand, s.variant.accumulator);
    if (!available)
      return available.takeError();
    if (!*available)
      continue;
    // No size gate: the padded fill zero-fills boundary tiles and every slice
    // outside the valid extent, so padding is numerically inert. The one
    // precondition is the static shared cap (N slice buffers per side plus N
    // F32 tiles live at once), a capability limit rather than a price.
    if (Error E = proposeInKernel(P, K, s.variant, s.tileM, s.tileN, s.tileK,
                                  cm, estimated))
      return E;
  }
  return Error::success();
}

// Every in-kernel class, with one warning for the classes whose row at the
// product's shape is estimated by the class row.
static Error proposeInKernelRaises(const Product &P, const InKernelSite &K,
                                   MatmulUnit &cm) {
  SmallVector<std::string, 16> estimated;
  if (Error E = proposeInKernelDirect(P, K, cm, estimated))
    return E;
  if (Error E = proposeInKernelSplit(P, K, cm, estimated))
    return E;
  if (estimated.empty())
    return Error::success();
  return reportEstimatedRow(
      P.S, P.m, "in-kernel raise as " + join(estimated, ", "),
      "inkernel_rel,<class>_" + K.shape->key(),
      std::string(kCalibrate) + " --product " + K.shape->product(),
      "from the row of its class");
}

//===----------------------------------------------------------------------===//
// Library-call realization, proposed under -poseidon-matmul-library-calls. The
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
  // The M x Ncols x K the call runs at, when all three are constants of this
  // compile (a scalar-loop product's descriptor). The rows each shape reads
  // are listed above kCalibrate.
  std::optional<std::array<unsigned, 3>> runShape;
  // The extents at profile scale: the profiled global extents of the product.
  std::array<unsigned, 3> profileShape = {0, 0, 0};
};
} // namespace

static Expected<LibrarySite> librarySite(Session &S,
                                         const RecognizedMatmul &m) {
  LibrarySite L;
  Expected<std::optional<GemmBodyNote>> found = libraryCallNote(S, m);
  if (!found)
    return found.takeError();
  L.note = std::move(*found);
  const std::optional<GemmBodyNote> &note = L.note;
  L.dispatchable = note.has_value();
  if (S.Opts.Diagnostics.Print && !L.dispatchable)
    llvm::errs() << "[poseidon] matmul[" << m.id << "] " << m.M << 'x' << m.N
                 << 'x' << m.K
                 << ": host-dispatched Ozaki-II not proposed: the body "
                    "is not a host-dispatchable GEMM\n";
  if (note && m.origin == RecognizedMatmul::Origin::ConstantShape)
    L.runShape = {(unsigned)note->dims[DimM].add,
                  (unsigned)note->dims[DimNcols].add,
                  (unsigned)note->dims[DimK].add};
  L.profileShape = {m.globalM ? m.globalM : m.M, m.globalN ? m.globalN : m.N,
                    m.globalK ? m.globalK : m.K};
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

// S^3 / (M*N*K) for S = max(M, N, K), at the profiled extents (1 when the
// profile has none): the MACs of the smallest square a product embeds in, per
// MAC of the product.
static double squarePaddingRatio(const RecognizedMatmul &m) {
  const double M = m.globalM, N = m.globalN, K = m.globalK ? m.globalK : m.K;
  if (!M || !N || !K)
    return 1.0;
  const double S = std::max({M, N, K});
  return (S * S * S) / (M * N * K);
}

// A rectangular product is priced from the rows measured at its shape: all
// three extents for a call of constant extents, (M, K) for one whose column
// count is a run-time extent; a square one from the square rows. The shape is
// the one LibrarySite gives Ozaki-II. A rectangular product whose row is
// missing is priced as the square product it embeds in: the square row times
// the padding ratio, an estimate meant to err dear, so that the missing row
// costs picks rather than admitting one the measured row would refuse. A
// modulus count with neither row is not proposed.
static Error proposeLibraryModular(const Product &P, const LibrarySite &L,
                                   MatmulUnit &cm) {
  if (!L.dispatchable)
    return Error::success();
  const RecognizedMatmul &m = P.m;
  // The runtime's INT32 residue sums hold a reduction shorter than
  // kOzakiIIMaxK, at the scale the call runs at and at the one simulated.
  const unsigned reduction =
      std::max({L.runShape ? (*L.runShape)[2] : 0u, L.profileShape[2], m.K});
  if (reduction >= (unsigned)kOzakiIIMaxK) {
    if (P.S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] matmul[" << m.id
                   << "] ozaki-ii not proposed: its "
                   << "reduction of " << reduction
                   << " is not below the runtime's limit " << kOzakiIIMaxK
                   << '\n';
    return Error::success();
  }
  const std::array<unsigned, 3> &shape =
      L.runShape ? *L.runShape : L.profileShape;
  const bool square = shape[0] == shape[1] && shape[1] == shape[2];
  // A call of constant extents is keyed by all three, as it is measured at
  // them; one whose column count is a run-time extent by (M, K), whose rows
  // are measured at many columns.
  const std::string shapeM = std::to_string(shape[0]),
                    shapeN = std::to_string(shape[1]),
                    shapeK = std::to_string(shape[2]);
  const std::string shapeKey = L.runShape
                                   ? "m" + shapeM + "n" + shapeN + "k" + shapeK
                                   : "m" + shapeM + "k" + shapeK;
  const std::string family =
      square ? "library_ozaki_rel" : "library_ozaki_rect_rel";
  const double padding = squarePaddingRatio(m);
  std::string missing, estimated;
  for (unsigned nm : kOzakiIIModuliCounts) {
    const SchemeVariant v = SchemeVariant::ozakiII(nm);
    const std::string squareKey = "nm" + std::to_string(nm);
    const std::string key =
        squareKey + (square ? std::string() : "_" + shapeKey);
    double rel = lookupCost(P.S.CostModel, family, key).value_or(-1.0);
    if (rel <= 0.0 && !square) {
      rel = lookupCost(P.S.CostModel, "library_ozaki_rel", squareKey)
                .value_or(-1.0) *
            padding;
      if (rel > 0.0)
        estimated += (estimated.empty() ? "" : ", ") + std::to_string(nm);
    }
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
  const std::string rows =
      family + ",nm<nm>" + (square ? std::string() : "_" + shapeKey);
  std::string measure(kCalibrate);
  if (L.runShape && !square)
    measure += " --product " + shapeM + "x" + shapeN + "x" + shapeK;
  else if (!square)
    measure += " --ozaki-rect " + shapeM + "x" + shapeK;
  if (!estimated.empty())
    if (Error E = reportEstimatedRow(
            P.S, m, "ozaki-ii host dispatch at nm=" + estimated, rows, measure,
            "as the square product it embeds in (the square row times " +
                formatv("{0:F2}", padding) + ")"))
      return E;
  if (!missing.empty())
    return reportMissingRow(P.S, m, "ozaki-ii host dispatch at nm=" + missing,
                            rows, measure);
  return Error::success();
}

// The split scheme as __poseidon_tcec_dgemm, which realizes a different rate
// than the in-kernel raise and so is a distinct candidate. Absent its measured
// library_tcec_rel row it is not proposed, and in a build without cuMpSGEMM,
// which has no such runtime, a pin that names it is told why. The runtime
// repacks operands rather than padding to a square, so no shape ratio is
// charged.
static Error proposeLibrarySplit(const Product &P, const LibrarySite &L,
                                 MatmulUnit &cm) {
  if (!L.dispatchable)
    return Error::success();
  MatmulUnit::Option opt = libraryOption(
      SchemeVariant::tcec(FPKind::FP16, FPKind::FP32, 2), 16, 16, 16);
  if (!kHaveTcecDispatch) {
    if (P.S.Opts.Diagnostics.Print &&
        !std::exchange(P.S.Matmul.ReportedNoTcecRuntime, true))
      llvm::errs() << "[poseidon] matmul[" << P.m.id << "] " << P.m.M << 'x'
                   << P.m.N << 'x' << P.m.K
                   << ": TCEC dispatch not proposed: this build has no TCEC "
                      "runtime; configure it with "
                      "-DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>\n";
    cm.dropped.push_back(
        {matmulOptionLabel(opt),
         "this compiler was built without cuMpSGEMM, on which the "
         "error-corrected dispatch runs; configure it with "
         "-DPOSEIDON_CUMPSGEMM_ROOT=<cuMpSGEMM checkout>"});
    return Error::success();
  }
  double rel =
      lookupCost(P.S.CostModel, "library_tcec_rel", "fp16tcec").value_or(-1.0);
  if (rel <= 0.0)
    return reportMissingRow(P.S, P.m, "tcec host dispatch",
                            "library_tcec_rel,fp16tcec", kCalibrate);
  opt.compCost = P.baselinePerMac * rel;
  cm.candidates.push_back(opt);
  return Error::success();
}

// The direct scheme as __poseidon_direct_dgemm: each FP64 operand is rounded
// once to the library's input format and one library GEMM accumulates in FP32
// (runtime/gemm/Direct.cu). Same numerics as the in-kernel raise up to
// accumulation order, but a distinct realized rate. Priced only from this
// device's measured library_direct_rel,<class> row; absent the row it is not
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
    {SchemeVariant::direct(FPKind::FP16, FPKind::FP32), 16, 16, 16},
    {SchemeVariant::direct(FPKind::BF16, FPKind::FP32), 16, 16, 16},
    {SchemeVariant::direct(FPKind::TF32, FPKind::FP32), 16, 16, 8},
    {SchemeVariant::direct(FPKind::FP32, FPKind::FP32), 0, 0, 0},
};

static Error proposeLibraryDirect(const Product &P, const LibrarySite &L,
                                  MatmulUnit &cm) {
  if (!L.dispatchable)
    return Error::success();
  const RecognizedMatmul &m = P.m;
  for (const LibraryDirectVariant &d : kLibraryDirectVariants) {
    const std::string cls = schemeFormats(d.variant, '_');
    double rel =
        lookupCost(P.S.CostModel, "library_direct_rel", cls).value_or(-1.0);
    if (rel <= 0.0) {
      if (Error E = reportMissingRow(P.S, m,
                                     Twine(schemeTag(d.variant)) + " " + cls +
                                         " host dispatch",
                                     "library_direct_rel," + cls, kCalibrate))
        return E;
      continue;
    }
    MatmulUnit::Option opt =
        libraryOption(d.variant, d.tileM, d.tileN, d.tileK);
    opt.compCost = P.baselinePerMac * rel;
    cm.candidates.push_back(opt);
  }
  return Error::success();
}

// The native product (__poseidon_native_dgemm), one cublasDgemm on the real
// M x Ncols x K for every layout combination. Its per-MAC rate depends on the
// shape, so a call whose extents are constants of this compile is priced from
// the row measured at those extents when the model has one; any other from
// the square row (see LibrarySite). The square row is not an upper bound for a
// rectangular call, whose per-MAC rate is usually worse, so the warning names
// the row that replaces the estimate.
static Error proposeLibraryNative(const Product &P, const LibrarySite &L,
                                  MatmulUnit &cm) {
  if (!L.dispatchable)
    return Error::success();
  const RecognizedMatmul &m = P.m;
  // Extents of this compile are priced from the row measured at them, a
  // square one also from the square row.
  const unsigned M = L.runShape ? (*L.runShape)[0] : 0,
                 N = L.runShape ? (*L.runShape)[1] : 0,
                 K = L.runShape ? (*L.runShape)[2] : 0;
  const std::string shape =
      std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
  const std::string rectKey = "dgemm_m" + std::to_string(M) + "n" +
                              std::to_string(N) + "k" + std::to_string(K);
  double rel =
      L.runShape ? lookupCost(P.S.CostModel, "library_ozaki_rect_rel", rectKey)
                       .value_or(-1.0)
                 : -1.0;
  const bool rectangular = L.runShape && !(M == N && N == K);
  const bool fromSquare = rel <= 0.0;
  if (fromSquare)
    rel =
        lookupCost(P.S.CostModel, "library_ozaki_rel", "dgemm").value_or(-1.0);
  if (rel <= 0.0)
    return reportMissingRow(P.S, m, "native-dgemm host dispatch",
                            "library_ozaki_rel,dgemm", kCalibrate);
  if (fromSquare && rectangular)
    if (Error E =
            reportEstimatedRow(P.S, m, "native-dgemm host dispatch at " + shape,
                               "library_ozaki_rect_rel," + rectKey,
                               std::string(kCalibrate) + " --product " + shape,
                               "from the square 'library_ozaki_rel,dgemm' row"))
      return E;
  MatmulUnit::Option opt = libraryOption(SchemeVariant::native(), 16, 16, 16);
  opt.compCost = P.baselinePerMac * rel;
  cm.candidates.push_back(opt);
  return Error::success();
}

Error generateMatmulCandidates(Session &S, ArrayRef<RecognizedMatmul> matmuls,
                               const ProfileMap &scalarProfile,
                               unsigned sampleLogBits,
                               SmallVectorImpl<MatmulUnit> &out) {
  for (const RecognizedMatmul &m : matmuls) {
    Expected<std::optional<MatmulProfile>> loaded =
        loadProfile(m, scalarProfile);
    if (!loaded)
      return loaded.takeError();
    std::optional<MatmulProfile> &prof = *loaded;
    if (!prof) {
      if (S.Opts.Diagnostics.Print)
        llvm::errs() << "[poseidon] matmul[" << m.id << "] in "
                     << anchorOf(m).getFunction()->getName()
                     << ": the profiling run never executed it, so it is left "
                        "as written\n";
      continue;
    }

    MatmulUnit cm;
    cm.matmul = const_cast<RecognizedMatmul *>(&m);
    cm.executions = prof->macs;

    double baselinePerMac = 0.0;
    switch (m.origin) {
    case RecognizedMatmul::Origin::ConstantShape:
      // The loop body is one MAC's worth of work and the cost-model rows are
      // reciprocal throughput per op, so this sum is the baseline's cost per
      // MAC (executions = profiled total MACs).
      for (Instruction *I : m.footprint) {
        Expected<double> c = instructionCost(S.CostModel, I);
        if (!c)
          return c.takeError();
        baselinePerMac += *c;
      }
      break;
    case RecognizedMatmul::Origin::RuntimeShape: {
      const RuntimeGemmHandle &g = *m.runtimeGemm;
      const unsigned R = g.fmas.size();
      // Composed in the unit every "*_dispatch_rel" row is measured in: one
      // MAC of this loop nest is one fma at the element type, plus the
      // epilogue's fadd amortized over the K-long reduction it closes.
      double bodySum = 0.0, epi = 0.0;
      for (Instruction *I : m.footprint) {
        Expected<double> c = instructionCost(S.CostModel, I);
        if (!c)
          return c.takeError();
        if (I == g.epilogue)
          epi += *c;
        else
          bodySum += *c;
      }
      baselinePerMac = bodySum / (double)R + (m.K ? epi / (double)m.K : 0.0);
      break;
    }
    }
    cm.initialCompCost = baselinePerMac;
    const Product P{S, m, baselinePerMac};

    // The candidate index is the position in this sequence.
    if (m.origin == RecognizedMatmul::Origin::ConstantShape &&
        m.scalarLoop.inKernelRaisable && siteReachedByWholeBlock(S, m)) {
      if (Error E = proposeInKernelRaises(P, inKernelSite(S, m), cm))
        return E;
    }
    if (S.Opts.Matmul.MatmulLibraryCalls) {
      Expected<LibrarySite> site = librarySite(S, m);
      if (!site)
        return site.takeError();
      const LibrarySite &L = *site;
      cm.library = L.note;
      if (Error E = proposeLibraryModular(P, L, cm))
        return E;
      if (Error E = proposeLibrarySplit(P, L, cm))
        return E;
      if (Error E = proposeLibraryDirect(P, L, cm))
        return E;
      if (Error E = proposeLibraryNative(P, L, cm))
        return E;
    }
    SmallVector<MatmulAccuracyQuery> queries;
    for (const MatmulUnit::Option &opt : cm.candidates)
      queries.push_back(accuracyQuery(m, opt));
    SmallVector<MatmulAccuracy> accuracy(queries.size());
    if (Error E = evaluateMatmulAccuracy(S, m, *prof, sampleLogBits, queries,
                                         accuracy))
      return E;
    for (auto [opt, a] : zip_equal(cm.candidates, accuracy)) {
      opt.accuracyCost = a.accuracyCost;
      opt.domainError = a.domainError;
    }

    if (m.origin == RecognizedMatmul::Origin::RuntimeShape &&
        S.Opts.Diagnostics.Print)
      llvm::errs() << "[poseidon] matmul[" << m.id
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
  return Error::success();
}

Error MatmulUnit::apply(Session &S, size_t candidateIndex) {
  assert(candidateIndex < candidates.size() &&
         "the solver names one of the candidates");
  const Option &opt = candidates[candidateIndex];

  if (opt.realization == Realization::LibraryCall) {
    assert(library && "a library call is proposed only with its descriptor");
    return applyLibraryCall(S, *matmul, *library, opt.variant);
  }
  assert(matmul->origin == RecognizedMatmul::Origin::ConstantShape &&
         "a runtime-shape product is proposed only as a library call");
  switch (opt.variant.scheme) {
  case Scheme::Direct:
  case Scheme::Tcec: {
    materializeInKernel(*matmul, opt);
    // Each raise was proposed against the shared memory its kernel used
    // before any raise; raises of one kernel that do not share their scratch
    // add up.
    Function &F = *matmul->scalarLoop.preheader->getParent();
    const uint64_t bytes = enclosingKernelSharedBytes(S, F);
    const uint64_t cap = Target::of(F).maxStaticSharedBytes();
    if (bytes > cap)
      return userError(F, F.getSubprogram(),
                       "the in-kernel raises selected for the matrix products "
                       "of " +
                           F.getName() + " need " + Twine(bytes) +
                           " B of static shared memory with what their kernel "
                           "uses, above the " +
                           Twine(cap) +
                           " B a kernel may hold; select fewer of them with "
                           "-poseidon-pin.");
    return Error::success();
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
  OS << "[poseidon] matmul[" << matmul->id
     << "] candidates (initial cost=" << initialCompCost
     << ", executions=" << executions << ", globalM=" << matmul->globalM
     << ", globalN=" << matmul->globalN << ", globalK=" << matmul->globalK
     << ", K=" << matmul->K << ", gridCTAs=" << matmul->gridCTAs << "):\n";
  if (candidates.empty()) {
    OS << "[poseidon]   (no candidates emitted)\n";
    return;
  }
  for (size_t i = 0; i < candidates.size(); ++i)
    OS << "[poseidon]   " << matmulOptionLabel(candidates[i])
       << ": dcost=" << getCompCostDelta(i) << " dacc=" << getAccCostDelta(i)
       << '\n';
}

// Dispatch and raise prices are measured wall-clock ratios that repeat to about
// 0.6% run to run, so a smaller modelled cost gap is a tie.
static constexpr double kCostTieBandRel = 0.006;

SmallVector<SolutionStep>
errorBudgetSelector(const Options &O, SmallVectorImpl<MatmulUnit> &matmulUnits,
                    double budget) {
  SmallVector<SolutionStep> steps;
  for (auto &cm : matmulUnits) {
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
        llvm::errs() << "[poseidon] error budget: cost tie inside "
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
        llvm::errs() << "[poseidon] error budget: matmul -> baseline F64 (no "
                        "rewrite clears "
                     << budget << " at confidence " << kDefaultConfidence
                     << " faster than F64)\n";
      continue;
    }
    if (O.Diagnostics.Print)
      llvm::errs() << "[poseidon] error budget: matmul -> "
                   << matmulOptionLabel(cm.candidates[best]) << " (domain err "
                   << cm.candidates[best].domainError << " <= " << budget
                   << " at confidence " << kDefaultConfidence << ")\n";
    steps.emplace_back(&cm, (size_t)best);
  }
  return steps;
}

} // namespace poseidon
