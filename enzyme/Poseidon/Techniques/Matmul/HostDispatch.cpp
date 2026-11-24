//===- HostDispatch.cpp - host-side GEMM library dispatch -----------------===//
//
// Part of the Poseidon Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// See HostDispatch.h. Suppressed during profile-gen and when no profile
// is given.
//
//===----------------------------------------------------------------------===//

#include "HostDispatch.h"
#include "IR/Utils.h"
#include "Matmul.h"
#include "Pass/Session.h"
#include "Profile/Instrument.h"
#include "Sites/Annotations.h"
#include "Sites/LaunchDescriptors.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <optional>
#include <string>

using namespace llvm;

namespace poseidon {

// Strip LCSSA / single-value phis and casts from a stored value, testing
// whether it is `target`.
static bool valueReaches(const Value *v, const Value *target) {
  for (int guard = 0; v && guard < 16; ++guard) {
    if (v == target)
      return true;
    if (auto *P = dyn_cast<PHINode>(v)) {
      const Value *same = nullptr;
      bool ok = true;
      for (const Value *in : P->incoming_values()) {
        if (in == P)
          continue;
        if (!same)
          same = in;
        else if (same != in) {
          ok = false;
          break;
        }
      }
      if (ok && same) {
        v = same;
        continue;
      }
      return false;
    }
    if (auto *C = dyn_cast<CastInst>(v)) {
      v = C->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

// `blockIdx.a * blockDim.a + threadIdx.a` on one axis, modulo casts and operand
// order; an intra-block index would bound the tile, not the matrix.
static bool isGlobalThreadIndex(const Value *v, int &axis) {
  while (const auto *CI = dyn_cast<CastInst>(v))
    v = CI->getOperand(0);
  const auto *add = dyn_cast<BinaryOperator>(v);
  if (!add || add->getOpcode() != Instruction::Add)
    return false;
  for (unsigned i = 0; i < 2; ++i) {
    LaunchGeometry t = launchGeometryOf(add->getOperand(i));
    if (t.Kind != LaunchValue::ThreadIndex)
      continue;
    const auto *mul = dyn_cast<BinaryOperator>(add->getOperand(1 - i));
    if (!mul || mul->getOpcode() != Instruction::Mul)
      continue;
    for (unsigned j = 0; j < 2; ++j) {
      LaunchGeometry c = launchGeometryOf(mul->getOperand(j));
      LaunchGeometry n = launchGeometryOf(mul->getOperand(1 - j));
      if (c.Kind == LaunchValue::BlockIndex &&
          n.Kind == LaunchValue::BlockDim && c.Axis == t.Axis &&
          n.Axis == t.Axis) {
        axis = t.Axis;
        return true;
      }
    }
  }
  return false;
}

// Fold a condition known to hold into exclusive upper bounds on the global
// thread index of each axis; false if any leaf is not such a bound, so a store
// gated by anything else is never mistaken for a dense rectangle.
static bool collectIndexBounds(const Value *cond, bool holds, uint64_t bound[3],
                               unsigned depth = 0) {
  if (depth > 8)
    return false;
  if (const auto *BO = dyn_cast<BinaryOperator>(cond)) {
    if ((BO->getOpcode() == Instruction::Or && !holds) ||
        (BO->getOpcode() == Instruction::And && holds))
      return collectIndexBounds(BO->getOperand(0), holds, bound, depth + 1) &&
             collectIndexBounds(BO->getOperand(1), holds, bound, depth + 1);
    return false;
  }
  const auto *IC = dyn_cast<ICmpInst>(cond);
  if (!IC)
    return false;
  CmpInst::Predicate P = holds ? IC->getPredicate() : IC->getInversePredicate();
  const Value *idx = IC->getOperand(0);
  const auto *cst = dyn_cast<ConstantInt>(IC->getOperand(1));
  if (!cst) {
    cst = dyn_cast<ConstantInt>(IC->getOperand(0));
    if (!cst)
      return false;
    idx = IC->getOperand(1);
    P = CmpInst::getSwappedPredicate(P);
  }
  int axis = -1;
  if (!isGlobalThreadIndex(idx, axis))
    return false;
  if (cst->getValue().isNegative() || cst->getValue().getActiveBits() > 31)
    return false;
  uint64_t ub;
  switch (P) {
  case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_ULT:
    ub = cst->getZExtValue();
    break;
  case CmpInst::ICMP_SLE:
  case CmpInst::ICMP_ULE:
    ub = cst->getZExtValue() + 1;
    break;
  default:
    return false;
  }
  if (ub == 0)
    return false;
  if (!bound[axis] || ub < bound[axis])
    bound[axis] = ub;
  return true;
}

// True deploy extents of the product this body writes, from the constants that
// gate its C store: every condition dominating the store (outside the reduction
// loop) must bound a global thread index, and both matrix axes must be bounded.
// The host stub bounds them again by the launch geometry, taking
// min(crop, gridDim[axis] * blockDim[axis]).
static bool recoverGuardExtents(const Options &O, Function &F,
                                const AbstractMatmul &m,
                                const StoreInst *cStore, unsigned &cropM,
                                unsigned &cropN, int &axM, int &axN) {
  const ScalarLoopHandle &h = m.scalarLoop;
  if (h.aRowSlowAxis != TidAxis::Unknown || h.bColSlowAxis != TidAxis::Unknown)
    return false; // fused index: its extent is a product of two axes
  std::optional<unsigned> ixM = tidAxisIndex(h.aRowAxis);
  std::optional<unsigned> ixN = tidAxisIndex(h.bColAxis);
  if (!ixM || !ixN || *ixM == *ixN)
    return false;
  axM = (int)*ixM;
  axN = (int)*ixN;

  DominatorTree DT(F);
  const BasicBlock *SB = cStore->getParent();
  uint64_t bound[3] = {0, 0, 0};
  for (BasicBlock &BB : F) {
    if (h.blocks.count(&BB))
      continue;
    Instruction *BI = BB.getTerminator();
    if (!isConditionalBranch(BI))
      continue;
    for (unsigned i = 0; i < 2; ++i) {
      // The edge query rejects a duplicated edge on its own, so a two-way
      // branch to one block dominates nothing and contributes no fact.
      if (!DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(i)), SB))
        continue;
      if (!collectIndexBounds(branchCondition(BI), /*holds=*/i == 0, bound)) {
        if (O.Diagnostics.Print)
          llvm::errs() << "[ozaki-host-dispatch] " << F.getName()
                       << ": a condition gating the GEMM store is not a thread-"
                          "index bound, so the written extent is not a "
                          "rectangle this dispatch can reproduce\n";
        return false;
      }
    }
  }
  if (!bound[axM] || !bound[axN])
    return false;
  cropM = (unsigned)bound[axM];
  cropN = (unsigned)bound[axN];
  return true;
}

// The store the host dispatch takes the place of: the first in F into an
// argument whose value is the product m computes. The dispatch writes the
// product where it stores, and fissionGemmForDispatch reads it back there for
// the stores and other uses that follow. Null if F has none.
static StoreInst *productStore(Function &F, const AbstractMatmul &m) {
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<StoreInst>(&I);
        S && tracePointerRoot(S->getPointerOperand()).param >= 0 &&
        valueReaches(S->getValueOperand(), m.outputValue))
      return S;
  return nullptr;
}

bool computeGemmBodyNote(const Options &O, Function &F, const AbstractMatmul &m,
                         GemmBodyNote &note) {
  note = GemmBodyNote{};
  if (m.origin != AbstractMatmul::Origin::ScalarLoopReduction)
    return false;

  // The raiser reports m.M/m.N as the per-block tile and m.K as the K-loop
  // trip, which for a full-reduction square GEMM is the global dim, so the
  // dispatch dimension is m.K. Require FP64, A contiguous along K, and B either
  // row-major K-strided by N or col-major (K-contiguous).
  const ScalarLoopHandle &h = m.scalarLoop;
  int64_t K = (int64_t)m.K;
  if (m.K == ~0u || K <= 0)
    return false;
  if (h.aStrideByte != 8)
    return false; // FP64 element + A contiguous along K (A row-major)
  bool bColMajor;
  if (h.bStrideByte == 8)
    bColMajor = true;
  else if (h.bStrideByte == K * 8)
    bColMajor = false;
  else
    return false;

  LaunchRef aRoot = tracePointerRoot(m.scalarLoop.aBase);
  LaunchRef bRoot = tracePointerRoot(m.scalarLoop.bBase);
  note.aParam = aRoot.param;
  note.aOffset = aRoot.offset;
  note.bParam = bRoot.param;
  note.bOffset = bRoot.offset;
  if (note.aParam < 0 || note.bParam < 0)
    return false;

  // The body is standalone iff it has exactly one argument-rooted store.
  int storeCount = 0;
  const StoreInst *lastArgStore = nullptr;
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<StoreInst>(&I);
        S && tracePointerRoot(S->getPointerOperand()).param >= 0) {
      storeCount++;
      lastArgStore = S;
    }
  const StoreInst *cStore = productStore(F, m);
  // A fused body's stored value reaches the matmul result only through the
  // loop's lcssa accumulator, which valueReaches may miss; with exactly one
  // argument-rooted store, that store is the C store.
  if (!cStore && storeCount == 1)
    cStore = lastArgStore;
  if (!cStore)
    return false;
  LaunchRef cRoot = tracePointerRoot(cStore->getPointerOperand());
  note.cParam = cRoot.param;
  note.cOffset = cRoot.offset;
  if (note.cParam < 0)
    return false;

  // Host dispatch supports only a pure GEMM body: a self-overwriting library
  // dispatch cannot represent a fused epilogue that may read its own output, so
  // the stored value (modulo lcssa) must be an instruction inside the loop.
  {
    const Value *sv = cStore->getValueOperand();
    while (const auto *phi = dyn_cast<PHINode>(sv)) {
      if (phi->getNumIncomingValues() != 1)
        break;
      sv = phi->getIncomingValue(0);
    }
    const auto *svI = dyn_cast<Instruction>(sv);
    if (!svI || !m.scalarLoop.blocks.count(svI->getParent()))
      return false;
  }

  note.N = (unsigned)K;
  // Every dim must be deploy scale: gK is, but globalM/globalN are
  // profile-scale and are rescaled by K/globalK. Fall back to the square dim K
  // when the global geometry is unavailable. Leading dims come from the handle
  // (bytes to elements); C mirrors B's major order.
  note.gK = (unsigned)K;
  if (m.globalM && m.globalN && m.globalK) {
    // The K/globalK rescale is only sound for a dimension that tracks matrix
    // order (one equal to globalK at profile scale); rescaling a
    // runtime-dependent dimension fabricates a deploy extent and the runtime
    // then reads past the operand. When such a dimension meets a changed
    // scale, no static value is correct.
    const bool scaled = (uint64_t)K != (uint64_t)m.globalK;
    bool recovered = false;
    if (scaled && (m.globalM != m.globalK || m.globalN != m.globalK)) {
      // Before refusing, recover the true deploy extents from the kernel's own
      // index guard (recoverGuardExtents): exact, already at deploy scale, and
      // intersected with the launch geometry by the host stub.
      unsigned cropM = 0, cropN = 0;
      int axM = -1, axN = -1;
      if (recoverGuardExtents(O, F, m, cStore, cropM, cropN, axM, axN)) {
        note.gM = cropM;
        note.gNcols = cropN;
        note.launchCrop = true;
        note.cropMAxis = axM;
        note.cropNAxis = axN;
        recovered = true;
        llvm::errs() << "[ozaki-host-dispatch] recovered deploy geometry for "
                     << F.getName() << ": M=" << cropM << " Ncols=" << cropN
                     << " K=" << K << " (profile M/N/K = " << m.globalM << '/'
                     << m.globalN << '/' << m.globalK
                     << " does not track matrix order; extents taken from the "
                        "kernel's own index guard on thread axes "
                     << axM << '/' << axN
                     << " and bounded again by the launch geometry at the "
                        "dispatch)\n";
      }
    }
    if (scaled && !recovered &&
        (m.globalM != m.globalK || m.globalN != m.globalK)) {
      // Never silent: refusing the note withdraws the host-dispatch candidate,
      // and the solver then picks among in-kernel candidates that may be
      // materially less accurate.
      llvm::errs()
          << "[ozaki-host-dispatch] REFUSING host dispatch for " << F.getName()
          << ": non-square geometry under a changed deploy scale (profile "
             "M/N/K = "
          << m.globalM << '/' << m.globalN << '/' << m.globalK
          << ", deploy K = " << K
          << "). A dimension that does not track matrix order cannot be "
             "rescaled by K/globalK, and its true deploy extent is not "
             "statically known here, so no value would be correct. The "
             "host-dispatch candidate is WITHDRAWN for this site; the solver "
             "will choose among the remaining in-kernel candidates, which may "
             "be MATERIALLY LESS ACCURATE than the withdrawn one. Re-check the "
             "selected scheme for this site before trusting the result.\n";
      if (O.Model.StrictMode)
        reportUserError(
            F, F.getSubprogram(),
            "strict mode: the host-dispatch geometry of " + F.getName() +
                " is unrecoverable for a non-square GEMM at a changed deploy "
                "scale, so its candidate set would be reduced; profile at the "
                "deploy shape, or leave out -poseidon-strict-mode.");
      note.valid = false;
      return false;
    }
    if (!recovered) {
      note.gM =
          (unsigned)((uint64_t)m.globalM * (uint64_t)K / (uint64_t)m.globalK);
      note.gNcols =
          (unsigned)((uint64_t)m.globalN * (uint64_t)K / (uint64_t)m.globalK);
    }
  } else {
    note.gM = m.globalM ? m.globalM : (unsigned)K;
    note.gNcols = m.globalN ? m.globalN : (unsigned)K;
  }
  note.aColMajor = (h.aStrideByte != 8); // always false here (A K-contiguous)
  note.bColMajor = bColMajor;
  note.cColMajor = bColMajor; // heuristic: output matches B's major order
  note.lda = h.aLeadingDimByte ? (unsigned)(h.aLeadingDimByte / 8) : note.gK;
  note.ldb = h.bLeadingDimByte ? (unsigned)(h.bLeadingDimByte / 8) : note.gK;
  note.ldc = note.cColMajor ? note.gM : note.gNcols;
  note.standalone = (storeCount == 1);
  note.valid = true;
  return true;
}

void fissionGemmForDispatch(Function &F, const AbstractMatmul &m) {
  StoreInst *cStore = productStore(F, m);
  assert(cStore &&
         "a body with a fused note stores the product to an argument");

  // Replace the GEMM result where the epilogue consumes it with a load from the
  // output buffer (filled by the prepended host dispatch); the reduction loop
  // becomes dead. In-loop (recurrence) uses stay.
  Value *resultVal = cStore->getValueOperand();
  Value *outAddr = cStore->getPointerOperand();
  IRBuilder<> B(cStore);
  LoadInst *loaded = B.CreateLoad(resultVal->getType(), outAddr, "ozdisp.gemm");
  SmallVector<Use *, 8> repl;
  for (Use &U : resultVal->uses()) {
    User *user = U.getUser();
    if (user == cStore)
      continue;
    if (auto *I = dyn_cast<Instruction>(user))
      if (m.footprint.count(I))
        continue; // in-loop recurrence, keep
    repl.push_back(&U);
  }
  for (Use *U : repl)
    U->set(loaded);
  cStore->eraseFromParent(); // dispatch already wrote the output buffer
  errs() << "[ozaki-host-dispatch] fissioned GEMM in " << F.getName()
         << " (epilogue reads dispatched output; reduction loop will DCE)\n";
}

void noteGemmBody(Session &S, const Function *body, const GemmBodyNote &n) {
  S.Matmul.GemmBodies.note(body, n);
}
bool getGemmBody(const Session &S, const Function *body, GemmBodyNote &out) {
  std::optional<GemmBodyNote> note = S.Matmul.GemmBodies.get(body);
  if (!note)
    return false;
  out = *note;
  return true;
}

static std::string launchArgString(int arg, int64_t offset) {
  std::string s = "arg" + std::to_string(arg);
  if (offset >= 0)
    s += "[+" + std::to_string(offset) + "]";
  return s;
}

static void emitDescriptor(Session &S, StringRef wrapperName, int cArg,
                           int aArg, int bArg, const GemmBodyNote &n) {
  const GemmBodyNote::RtDim *dims[6] = {&n.rM,   &n.rNcols, &n.rK,
                                        &n.rlda, &n.rldb,   &n.rldc};
  bool plainDims = llvm::all_of(
      dims, [](const GemmBodyNote::RtDim *x) { return x->isPlain(); });
  std::string message;
  raw_string_ostream msg(message);
  msg << "[ozaki-host-dispatch] wrote descriptor for " << wrapperName
      << ": C=" << launchArgString(cArg, n.cOffset)
      << " A=" << launchArgString(aArg, n.aOffset)
      << " B=" << launchArgString(bArg, n.bOffset) << ' ' << n.gM << 'x'
      << n.gNcols << 'x' << n.gK << " nm=" << n.numModuli
      << (n.standalone ? " (standalone)" : " (fused)");
  for (const auto &[x, y] : n.assumedEqual)
    msg << ", if " << launchArgString(x.param, x.offset)
        << " == " << launchArgString(y.param, y.offset);
  msg << '\n';
  // cArg aArg bArg N standalone numModuli gM gNcols gK lda ldb ldc aCol bCol
  // cCol scheme, then the keyword-tagged blocks.
  recordDescriptor(
      S, LaunchKind::HostDispatch, wrapperName,
      [&](raw_ostream &os) {
        os << ' ' << cArg << ' ' << aArg << ' ' << bArg << ' ' << n.N << ' '
           << (n.standalone ? 1 : 0) << ' ' << n.numModuli << ' ' << n.gM << ' '
           << n.gNcols << ' ' << n.gK << ' ' << n.lda << ' ' << n.ldb << ' '
           << n.ldc << ' ' << (n.aColMajor ? 1 : 0) << ' '
           << (n.bColMajor ? 1 : 0) << ' ' << (n.cColMajor ? 1 : 0) << ' '
           << (n.scheme == DispatchScheme::Tcec     ? 1
               : n.scheme == DispatchScheme::Direct ? 2
                                                    : 0);
        // Optional trailing runtime-geometry block (tokens 17..36).
        if (n.runtimeDims && plainDims) {
          os << " 1";
          for (const GemmBodyNote::RtDim *x : dims)
            os << ' ' << (x->isConst() ? -1 : x->factors[0].param) << ' '
               << x->mul << ' ' << x->add;
          os << ' ' << n.beta;
        }
        // The same geometry when a dimension is a field of an aggregate
        // argument or a product of launch values: per dimension the
        // factor count, each factor's argument and offset, mul and add.
        if (n.runtimeDims && !plainDims) {
          os << " rd";
          for (const GemmBodyNote::RtDim *x : dims) {
            os << ' ' << x->factors.size();
            for (const LaunchRef &f : x->factors)
              os << ' ' << f.param << ' ' << f.offset;
            os << ' ' << x->mul << ' ' << x->add;
          }
          os << ' ' << n.beta;
        }
        // Keyword-tagged rather than positional: the extents the kernel's
        // index guard gates its store with, intersected at the dispatch with
        // the launch geometry.
        if (n.launchCrop)
          os << " lc " << n.cropMAxis << ' ' << n.cropNAxis;
        if (n.cOffset >= 0 || n.aOffset >= 0 || n.bOffset >= 0)
          os << " pf " << n.cOffset << ' ' << n.aOffset << ' ' << n.bOffset;
        if (!n.assumedEqual.empty()) {
          os << " eq " << n.assumedEqual.size();
          for (const auto &[x, y] : n.assumedEqual)
            os << ' ' << x.param << ' ' << x.offset << ' ' << y.param << ' '
               << y.offset;
        }
      },
      std::move(message));
}

// Shared descriptor emit for the per-site path and the deferred joint-dp
// flush: map C/A/B via `mapParam`, skipping (with a message) if any fails.
static void emitMappedDescriptor(Session &S, StringRef wrapperName,
                                 const GemmBodyNote &n,
                                 function_ref<int(int)> mapParam) {
  int cArg = mapParam(n.cParam);
  int aArg = mapParam(n.aParam);
  int bArg = mapParam(n.bParam);
  if (cArg < 0 || aArg < 0 || bArg < 0) {
    errs() << "[ozaki-host-dispatch] could not map body params to wrapper "
              "launch args for "
           << wrapperName << "; skipping descriptor\n";
    return;
  }
  // Runtime dimensions name body parameters too, and the host side only sees
  // the wrapper's launch arguments, so they go through the same mapping.
  GemmBodyNote mapped = n;
  SmallVector<LaunchRef *, 16> refs;
  if (mapped.runtimeDims) {
    GemmBodyNote::RtDim *dd[6] = {&mapped.rM,   &mapped.rNcols, &mapped.rK,
                                  &mapped.rlda, &mapped.rldb,   &mapped.rldc};
    for (GemmBodyNote::RtDim *x : dd)
      for (LaunchRef &f : x->factors)
        refs.push_back(&f);
  }
  for (auto &[x, y] : mapped.assumedEqual) {
    refs.push_back(&x);
    refs.push_back(&y);
  }
  for (LaunchRef *f : refs) {
    int a = mapParam(f->param);
    if (a < 0) {
      errs() << "[ozaki-host-dispatch] could not map a runtime GEMM "
                "dimension's body param to a wrapper launch arg for "
             << wrapperName << "; skipping descriptor\n";
      return;
    }
    f->param = a;
  }
  emitDescriptor(S, wrapperName, cArg, aArg, bArg, mapped);
}

void writeGemmDescriptor(Session &S, Function &wrapper,
                         ArrayRef<Value *> primalArgs, const GemmBodyNote &n) {
  if (!n.valid)
    return;
  emitMappedDescriptor(S, wrapper.getName(), n, [&](int bp) -> int {
    if (bp < 0 || (size_t)bp >= primalArgs.size())
      return -1;
    return traceToArgIndex(primalArgs[bp]);
  });
}

void addPendingGemmDispatch(Session &S, const Function *body, Function &wrapper,
                            ArrayRef<Value *> primalArgs) {
  PendingGemm pg;
  pg.wrapperName = wrapper.getName().str();
  pg.paramToArg.resize(primalArgs.size());
  for (size_t i = 0; i < primalArgs.size(); ++i)
    pg.paramToArg[i] = traceToArgIndex(primalArgs[i]);
  S.Matmul.PendingGemms[body] = std::move(pg);
}

void flushPendingGemmDispatches(Session &S) {
  for (auto &kv : S.Matmul.PendingGemms) {
    GemmBodyNote n;
    if (!getGemmBody(S, kv.first, n))
      continue; // this site was not materialized as a host dispatch
    const PendingGemm &pg = kv.second;
    emitMappedDescriptor(S, pg.wrapperName, n, [&](int bp) -> int {
      if (bp < 0 || (size_t)bp >= pg.paramToArg.size())
        return -1;
      return pg.paramToArg[bp];
    });
  }
  S.Matmul.PendingGemms.clear();
}

namespace {
// A descriptor read back on the host side, its operand roles and dimension
// factors naming launch arguments of `kernel`.
struct LaunchDesc : GemmBodyNote {
  std::string kernel;
  // Whether anything is read from inside an aggregate argument, which only the
  // launch's own argument array holds in every host ABI.
  bool readsFields() const {
    if (cOffset >= 0 || aOffset >= 0 || bOffset >= 0 || !assumedEqual.empty())
      return true;
    for (const GemmBodyNote::RtDim *x :
         {&rM, &rNcols, &rK, &rlda, &rldb, &rldc})
      if (!x->isPlain())
        return true;
    return false;
  }
};
} // namespace

// All host runtimes share one calling convention, so the scheme picks only the
// symbol.
static const char *dispatchSym(DispatchScheme s, bool square) {
  if (s == DispatchScheme::Tcec)
    return square ? "__poseidon_tcec_dgemm" : "__poseidon_tcec_dgemm_ex";
  if (s == DispatchScheme::Direct)
    return square ? "__poseidon_direct_dgemm" : "__poseidon_direct_dgemm_ex";
  return square ? "__poseidon_ozaki_dgemm" : "__poseidon_ozaki_dgemm_ex";
}

static FunctionCallee getOzakiDgemm(Module &M, DispatchScheme scheme) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  // void(C,A,B,N,lda,ldb,ldc,transA,transB,alpha,beta,stream,num_moduli)
  Type *params[] = {ptr, ptr, ptr, i32, i32, i32, i32,
                    i32, i32, dbl, dbl, ptr, i32};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(C), params, false);
  return M.getOrInsertFunction(dispatchSym(scheme, /*square=*/true), FT);
}

static FunctionCallee getOzakiDgemmEx(Module &M, DispatchScheme scheme) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  // void(C,A,B, M,Ncols,K, lda,ldb,ldc, aColMajor,bColMajor,cColMajor,
  //      alpha,beta, stream, num_moduli)
  Type *params[] = {ptr, ptr, ptr, i32, i32, i32, i32, i32,
                    i32, i32, i32, i32, dbl, dbl, ptr, i32};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(C), params, false);
  return M.getOrInsertFunction(dispatchSym(scheme, /*square=*/false), FT);
}

namespace {
// The tokens of a descriptor after the kernel name, read in order. The first
// token that is missing or does not parse stops the reading: every later read
// gives a default value, and takeError says what was wrong.
class DescriptorCursor {
public:
  explicit DescriptorCursor(ArrayRef<StringRef> toks) : toks(toks) {}

  bool ok() const { return why.empty(); }
  bool atEnd() const { return !ok() || at == toks.size(); }
  // Takes the next token when it is `keyword`.
  bool consume(StringRef keyword) {
    if (atEnd() || toks[at] != keyword)
      return false;
    ++at;
    return true;
  }
  template <typename T> T integer(StringRef what) {
    T value{};
    if (StringRef tok = next(what); ok() && tok.getAsInteger(10, value))
      fail("its " + what + " '" + tok + "' is not an integer of its range");
    return value;
  }
  double real(StringRef what) {
    double value = 0.0;
    if (StringRef tok = next(what); ok() && tok.getAsDouble(value))
      fail("its " + what + " '" + tok + "' is not a number");
    return value;
  }
  StringRef peek() const { return atEnd() ? StringRef() : toks[at]; }
  void fail(const Twine &reason) {
    if (ok())
      why = reason.str();
  }
  Error takeError() {
    if (ok() && at != toks.size())
      fail("it has the unexpected token '" + toks[at] + "'");
    return ok() ? Error::success()
                : createStringError(inconvertibleErrorCode(), why);
  }

private:
  StringRef next(StringRef what) {
    if (!atEnd())
      return toks[at++];
    fail("it ends before its " + what);
    return StringRef();
  }

  ArrayRef<StringRef> toks;
  size_t at = 1;
  std::string why;
};
} // namespace

static bool
readOzDispatchDescriptors(const Options &O, Module &M,
                          std::map<std::string, LaunchDesc> &descs) {
  readDescriptors(
      O, M, LaunchKind::HostDispatch, [&](ArrayRef<StringRef> toks) -> Error {
        DescriptorCursor Cursor(toks);
        LaunchDesc d;
        d.kernel = toks[0].str();
        d.cParam = Cursor.integer<int>("C argument");
        d.aParam = Cursor.integer<int>("A argument");
        d.bParam = Cursor.integer<int>("B argument");
        d.N = Cursor.integer<unsigned>("N");
        d.standalone = Cursor.integer<int>("standalone flag") != 0;
        d.numModuli = Cursor.integer<unsigned>("modulus field");
        if (d.numModuli > kOzakiIIMaxModuli)
          Cursor.fail("its modulus field " + Twine(d.numModuli) + " is above " +
                      Twine(kOzakiIIMaxModuli));
        d.gM = Cursor.integer<unsigned>("M");
        d.gNcols = Cursor.integer<unsigned>("column count");
        d.gK = Cursor.integer<unsigned>("K");
        d.lda = Cursor.integer<unsigned>("lda");
        d.ldb = Cursor.integer<unsigned>("ldb");
        d.ldc = Cursor.integer<unsigned>("ldc");
        d.aColMajor = Cursor.integer<int>("A layout") != 0;
        d.bColMajor = Cursor.integer<int>("B layout") != 0;
        d.cColMajor = Cursor.integer<int>("C layout") != 0;
        switch (Cursor.integer<int>("scheme")) {
        case 0:
          d.scheme = DispatchScheme::OzakiII;
          break;
        case 1:
          d.scheme = DispatchScheme::Tcec;
          break;
        case 2:
          d.scheme = DispatchScheme::Direct;
          break;
        default:
          Cursor.fail("its scheme is not 0, 1 or 2");
        }
        GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                      &d.rlda, &d.rldb,   &d.rldc};
        // Positional runtime-geometry block: flag + 6 x (param mul add), then
        // beta, which a descriptor followed by a keyword block may lack.
        if (Cursor.consume("1")) {
          for (GemmBodyNote::RtDim *x : dd) {
            int param = Cursor.integer<int>("runtime dimension argument");
            if (param >= 0)
              x->factors.push_back({param, -1});
            x->mul = Cursor.integer<int64_t>("runtime dimension factor");
            x->add = Cursor.integer<int64_t>("runtime dimension term");
          }
          if (!Cursor.atEnd() &&
              !is_contained({"rd", "pf", "eq", "lc"}, Cursor.peek()))
            d.beta = Cursor.real("beta");
          d.runtimeDims = true;
        }
        // Keyword blocks of fields and products, and of the launch geometry
        // (see emitDescriptor).
        while (!Cursor.atEnd()) {
          if (Cursor.consume("rd")) {
            for (GemmBodyNote::RtDim *x : dd) {
              unsigned nf = Cursor.integer<unsigned>("factor count");
              if (nf > 8)
                Cursor.fail("its factor count " + Twine(nf) + " is above 8");
              for (unsigned f = 0; f < nf && Cursor.ok(); ++f) {
                LaunchRef r;
                r.param = Cursor.integer<int>("factor argument");
                r.offset = Cursor.integer<int64_t>("factor offset");
                x->factors.push_back(r);
              }
              x->mul = Cursor.integer<int64_t>("runtime dimension factor");
              x->add = Cursor.integer<int64_t>("runtime dimension term");
            }
            d.beta = Cursor.real("beta");
            d.runtimeDims = true;
          } else if (Cursor.consume("pf")) {
            d.cOffset = Cursor.integer<int64_t>("C offset");
            d.aOffset = Cursor.integer<int64_t>("A offset");
            d.bOffset = Cursor.integer<int64_t>("B offset");
          } else if (Cursor.consume("eq")) {
            unsigned ne = Cursor.integer<unsigned>("equality count");
            if (ne > 16)
              Cursor.fail("its equality count " + Twine(ne) + " is above 16");
            for (unsigned e = 0; e < ne && Cursor.ok(); ++e) {
              LaunchRef x, y;
              x.param = Cursor.integer<int>("equality argument");
              x.offset = Cursor.integer<int64_t>("equality offset");
              y.param = Cursor.integer<int>("equality argument");
              y.offset = Cursor.integer<int64_t>("equality offset");
              d.assumedEqual.push_back({x, y});
            }
          } else if (Cursor.consume("lc")) {
            d.cropMAxis = Cursor.integer<int>("M axis");
            d.cropNAxis = Cursor.integer<int>("column axis");
            if (d.cropMAxis < 0 || d.cropMAxis > 2 || d.cropNAxis < 0 ||
                d.cropNAxis > 2 || d.cropMAxis == d.cropNAxis)
              Cursor.fail(
                  "its launch axes are not two distinct axes of x, y, z");
            d.launchCrop = true;
          } else {
            break;
          }
        }
        if (Error E = Cursor.takeError())
          return E;
        descs[d.kernel] = d;
        return Error::success();
      });
  return !descs.empty();
}

// Launch geometry, host side: the configuration a <<<grid, block>>> site pushes
// is popped inside the stub, so the stub is where a launch-derived extent can
// be taken; at the call site the dim3 temporaries are not constants at
// PipelineStart.
//
// Fresh pop for the standalone rewrite, whose body (pop included) is replaced;
// it also keeps the runtime's configuration stack balanced.
static void emitPopConfig(IRBuilder<> &B, Module &M, Value *&gridOut,
                          Value *&blockOut) {
  LLVMContext &C = M.getContext();
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  Type *ptr = PointerType::get(C, 0);
  Type *dim3Ty = ArrayType::get(i32, 3);
  auto *grid = B.CreateAlloca(dim3Ty, nullptr, "ozdisp.grid");
  auto *block = B.CreateAlloca(dim3Ty, nullptr, "ozdisp.block");
  grid->setAlignment(Align(8));
  block->setAlignment(Align(8));
  auto *shmem = B.CreateAlloca(i64, nullptr, "ozdisp.shmem");
  auto *stream = B.CreateAlloca(ptr, nullptr, "ozdisp.stream");
  B.CreateCall(popCallConfiguration(M), {grid, block, shmem, stream});
  gridOut = grid;
  blockOut = block;
}

// The pop the fused stub already performs; its dim3 out-parameters are the
// same geometry, and a second pop would consume the next launch's.
static CallInst *findPopConfig(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *cal = CI->getCalledFunction())
        if (isPopCallConfiguration(cal->getName()) && CI->arg_size() >= 2)
          return CI;
  return nullptr;
}

// A square row-major NN GEMM takes the square entry; anything non-square,
// transposed or launch-cropped takes the layout-aware _ex entry. A
// runtime-shaped GEMM always takes _ex: its extents are not known here.
static bool takesSquareEntry(const LaunchDesc &d) {
  return !d.runtimeDims && !d.launchCrop &&
         (d.gM == d.gNcols && d.gNcols == d.gK && !d.aColMajor &&
          !d.bColMajor && !d.cColMajor);
}

// The launch a CUDA or HIP stub ends in.
static CallInst *findLaunchCall(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *cal = CI->getCalledFunction())
        if (isLaunchKernel(cal->getName()) && CI->arg_size() >= 4)
          return CI;
  return nullptr;
}

// Kernel argument `r.param` lives where entry r.param of the launch's argument
// array points, as the CUDA runtime reads it; a field of an aggregate argument
// is at its offset from there in every host ABI, which the stub's own
// parameters are not (a small aggregate arrives split into registers).
static Value *loadLaunchValue(IRBuilder<> &B, Value *argArray,
                              const LaunchRef &r, Type *T) {
  Type *ptr = B.getPtrTy();
  Value *at = B.CreateLoad(
      ptr, B.CreateConstInBoundsGEP1_32(ptr, argArray, (unsigned)r.param));
  if (r.offset > 0)
    at = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), at, (uint64_t)r.offset);
  return B.CreateLoad(T, at);
}

// The dispatch for a descriptor that reads inside an aggregate argument: built
// in front of the stub's launch, from the launch's own argument array. The
// launch is removed (standalone) or kept (fused); with assumed equalities it
// runs instead of the dispatch whenever one of them does not hold.
static bool dispatchFromLaunchArgs(Module &M, Function &F, const LaunchDesc &d,
                                   FunctionCallee callee, bool squareRM) {
  CallInst *launch = findLaunchCall(F);
  CallInst *pop = findPopConfig(F);
  // A fused body was already fissioned device-side and computes nothing
  // without the dispatch, so it can neither fall back nor be skipped.
  if (!d.standalone &&
      (!launch || (d.launchCrop && !pop) || !d.assumedEqual.empty()))
    report_fatal_error(
        Twine("Poseidon: the fused host-dispatched GEMM of ") + F.getName() +
        " reads its operands from the stub's launch and cannot fall back to "
        "it, and this stub has no launch to read them from or a condition to "
        "check first; the kernel was already fissioned device-side, so there "
        "is no correct code to emit here.");
  if (!launch || (d.launchCrop && !pop)) {
    errs() << "[ozaki-host-dispatch] " << F.getName()
           << ": the dispatch reads the kernel's arguments from its launch, "
              "and this stub has no launch to read them from; launching the "
              "kernel as written\n";
    return false;
  }
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  auto I32 = [&](int64_t v) { return ConstantInt::get(i32, (uint64_t)v); };
  Value *argArray = launchArguments(*launch);

  IRBuilder<> B(launch);
  auto value = [&](const LaunchRef &r, Type *T) {
    return loadLaunchValue(B, argArray, r, T);
  };
  auto dim = [&](const GemmBodyNote::RtDim &r, unsigned constFallback) {
    if (!d.runtimeDims)
      return (Value *)I32(constFallback);
    Value *v = nullptr;
    for (const LaunchRef &f : r.factors) {
      Value *x = value(f, i32);
      v = v ? B.CreateMul(v, x) : x;
    }
    if (!v)
      return (Value *)I32(r.add);
    if (r.mul != 1)
      v = B.CreateMul(v, I32(r.mul));
    if (r.add)
      v = B.CreateAdd(v, I32(r.add));
    return v;
  };
  auto cropDim = [&](int axis, unsigned cropConst) -> Value * {
    Value *g = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedGrid(*pop), axis));
    Value *b = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedBlock(*pop), axis));
    Value *ext = B.CreateMul(g, b);
    return B.CreateSelect(B.CreateICmpULT(ext, I32(cropConst)), ext,
                          I32(cropConst));
  };

  Value *Cp = value({d.cParam, d.cOffset}, ptr);
  Value *Ap = value({d.aParam, d.aOffset}, ptr);
  Value *Bp = value({d.bParam, d.bOffset}, ptr);
  Value *beta = ConstantFP::get(dbl, d.beta);
  Value *one = ConstantFP::get(dbl, 1.0);
  Value *stream = ConstantPointerNull::get(cast<PointerType>(ptr));
  Value *nm = I32(d.numModuli);
  SmallVector<Value *, 16> args;
  if (squareRM) {
    Value *N = I32(d.N);
    args = {Cp, Ap, Bp, N, N, N, N, I32(0), I32(0), one, beta, stream, nm};
  } else {
    args = {Cp,
            Ap,
            Bp,
            d.launchCrop ? cropDim(d.cropMAxis, d.gM) : dim(d.rM, d.gM),
            d.launchCrop ? cropDim(d.cropNAxis, d.gNcols)
                         : dim(d.rNcols, d.gNcols),
            dim(d.rK, d.gK),
            dim(d.rlda, d.lda),
            dim(d.rldb, d.ldb),
            dim(d.rldc, d.ldc),
            I32(d.aColMajor ? 1 : 0),
            I32(d.bColMajor ? 1 : 0),
            I32(d.cColMajor ? 1 : 0),
            one,
            beta,
            stream,
            nm};
  }
  Value *holds = nullptr;
  for (const auto &[x, y] : d.assumedEqual) {
    Value *lhs = value(x, i32);
    Value *rhs = value(y, i32);
    Value *eq = B.CreateICmpEQ(lhs, rhs);
    holds = holds ? B.CreateAnd(holds, eq) : eq;
  }

  Value *success = Constant::getNullValue(launch->getType());
  if (!holds) {
    B.CreateCall(callee, args);
    if (d.standalone) {
      launch->replaceAllUsesWith(success);
      launch->eraseFromParent();
    }
  } else {
    Instruction *thenTerm = nullptr, *elseTerm = nullptr;
    SplitBlockAndInsertIfThenElse(holds, launch->getIterator(), &thenTerm,
                                  &elseTerm);
    IRBuilder<>(thenTerm).CreateCall(callee, args);
    BasicBlock *join = launch->getParent();
    launch->moveBefore(elseTerm->getIterator());
    if (!launch->use_empty()) {
      PHINode *phi = PHINode::Create(launch->getType(), 2, "", join->begin());
      launch->replaceAllUsesWith(phi);
      phi->addIncoming(success, thenTerm->getParent());
      phi->addIncoming(launch, elseTerm->getParent());
    }
  }
  errs() << "[ozaki-host-dispatch] "
         << (d.standalone ? "replaced" : "prepended") << " the launch in stub "
         << F.getName() << " with " << dispatchSym(d.scheme, squareRM)
         << " (C=" << launchArgString(d.cParam, d.cOffset)
         << " A=" << launchArgString(d.aParam, d.aOffset)
         << " B=" << launchArgString(d.bParam, d.bOffset) << ')';
  for (const auto &[x, y] : d.assumedEqual)
    errs() << ", if " << launchArgString(x.param, x.offset)
           << " == " << launchArgString(y.param, y.offset);
  errs() << '\n';
  return true;
}

// Runs before inlining (PipelineStart), while `__device_stub__K` is still a
// standalone function whose parameters are the kernel's launch args. Standalone
// descriptors replace the whole body with the dispatch call; fused bodies get
// the dispatch prepended and keep the (device-side fissioned) launch.
bool rewriteGemmStubBodies(const Options &O, Module &M) {
  std::map<std::string, LaunchDesc> descs;
  if (!readOzDispatchDescriptors(O, M, descs))
    return false;

  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);

  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    StringRef cn = F.getName();
    auto dIt = llvm::find_if(
        descs, [&](const auto &kv) { return isLaunchStubFor(F, kv.first); });
    if (dIt == descs.end())
      continue;
    const LaunchDesc &d = dIt->second;
    if (d.readsFields()) {
      changed |= dispatchFromLaunchArgs(M, F, d,
                                        takesSquareEntry(d)
                                            ? getOzakiDgemm(M, d.scheme)
                                            : getOzakiDgemmEx(M, d.scheme),
                                        takesSquareEntry(d));
      continue;
    }
    unsigned need = (unsigned)std::max({d.cParam, d.aParam, d.bParam});
    if (d.runtimeDims) {
      const GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                          &d.rlda, &d.rldb,   &d.rldc};
      for (const GemmBodyNote::RtDim *x : dd)
        if (!x->isConst())
          need = std::max(need, (unsigned)x->factors[0].param);
    }
    if (F.arg_size() <= need)
      continue;
    if (d.runtimeDims) {
      const GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                          &d.rlda, &d.rldb,   &d.rldc};
      if (llvm::any_of(dd, [&](const GemmBodyNote::RtDim *x) {
            return !x->isConst() &&
                   !F.getArg(x->factors[0].param)->getType()->isIntegerTy();
          })) {
        errs() << "[ozaki-host-dispatch] " << cn
               << ": a runtime GEMM dimension references a non-integer launch "
                  "argument; skipping the dispatch rewrite\n";
        continue;
      }
    }
    Value *Cp = F.getArg(d.cParam);
    Value *Ap = F.getArg(d.aParam);
    Value *Bp = F.getArg(d.bParam);
    Value *one_d = ConstantFP::get(dbl, 1.0);
    Value *beta_d = ConstantFP::get(dbl, d.beta);
    Value *nullStream = ConstantPointerNull::get(cast<PointerType>(ptr));
    Value *nmV = ConstantInt::get(i32, d.numModuli);
    auto I32 = [&](int v) { return ConstantInt::get(i32, (unsigned)v); };

    // The launch configuration of this launch, when the descriptor's extents
    // are intersected with it.
    Value *gridCfg = nullptr, *blockCfg = nullptr;
    // extent = min(crop, gridDim[axis] * blockDim[axis]): neither bound alone
    // is the extent.
    auto cropDim = [&](IRBuilder<> &B, int axis,
                       unsigned cropConst) -> Value * {
      Value *g = B.CreateLoad(
          i32, B.CreateConstInBoundsGEP1_32(i32, gridCfg, (unsigned)axis));
      Value *b = B.CreateLoad(
          i32, B.CreateConstInBoundsGEP1_32(i32, blockCfg, (unsigned)axis));
      Value *ext = B.CreateMul(g, b);
      Value *cst = ConstantInt::get(i32, cropConst);
      return B.CreateSelect(B.CreateICmpULT(ext, cst), ext, cst);
    };

    bool squareRM = takesSquareEntry(d);
    // Resolved per descriptor: the scheme is a per-site solver decision.
    FunctionCallee callee =
        squareRM ? getOzakiDgemm(M, d.scheme) : getOzakiDgemmEx(M, d.scheme);
    // Runtime dimensions are materialized as `mul*arg + add` right before the
    // dispatch call, so a build profiled at one problem size deploys at
    // another.
    auto rtDim = [&](IRBuilder<> &B, const GemmBodyNote::RtDim &r,
                     unsigned constFallback) -> Value * {
      if (!d.runtimeDims)
        return I32((int)constFallback);
      if (r.isConst())
        return ConstantInt::get(i32, (uint64_t)r.add);
      Value *v = B.CreateZExtOrTrunc(F.getArg(r.factors[0].param), i32);
      if (r.mul != 1)
        v = B.CreateMul(v, ConstantInt::get(i32, (uint64_t)r.mul));
      if (r.add)
        v = B.CreateAdd(v, ConstantInt::get(i32, (uint64_t)r.add));
      return v;
    };
    auto buildArgs = [&](IRBuilder<> &B) -> SmallVector<Value *, 16> {
      if (squareRM) {
        Value *Nv = I32((int)d.N);
        return {Cp,     Ap,     Bp,    Nv,     Nv,         Nv, Nv,
                I32(0), I32(0), one_d, beta_d, nullStream, nmV};
      }
      return {Cp,
              Ap,
              Bp,
              d.launchCrop ? cropDim(B, d.cropMAxis, d.gM)
                           : rtDim(B, d.rM, d.gM),
              d.launchCrop ? cropDim(B, d.cropNAxis, d.gNcols)
                           : rtDim(B, d.rNcols, d.gNcols),
              rtDim(B, d.rK, d.gK),
              rtDim(B, d.rlda, d.lda),
              rtDim(B, d.rldb, d.ldb),
              rtDim(B, d.rldc, d.ldc),
              I32(d.aColMajor ? 1 : 0),
              I32(d.bColMajor ? 1 : 0),
              I32(d.cColMajor ? 1 : 0),
              one_d,
              beta_d,
              nullStream,
              nmV};
    };
    const char *calleeName = dispatchSym(d.scheme, squareRM);
    if (d.launchCrop)
      errs() << "[ozaki-host-dispatch] " << cn
             << ": deploy extents taken from this launch -- M=min(" << d.gM
             << ", gridDim*blockDim on axis " << d.cropMAxis << "), Ncols=min("
             << d.gNcols << ", gridDim*blockDim on axis " << d.cropNAxis
             << ")\n";
    if (d.standalone) {
      // Pure GEMM: replace the launch entirely.
      F.deleteBody();
      BasicBlock *bb = BasicBlock::Create(C, "entry", &F);
      IRBuilder<> B(bb);
      if (d.launchCrop)
        emitPopConfig(B, M, gridCfg, blockCfg);
      SmallVector<Value *, 16> args = buildArgs(B);
      B.CreateCall(callee, args);
      B.CreateRetVoid();
      errs() << "[ozaki-host-dispatch] replaced stub body " << cn << " -> "
             << calleeName << " (C=arg" << d.cParam << " A=arg" << d.aParam
             << " B=arg" << d.bParam << ' ' << d.gM << 'x' << d.gNcols << 'x'
             << d.gK << ")\n";
    } else {
      // Fused: prepend the dispatch (fills the output buffer) and keep the
      // launch, whose kernel was fissioned device-side into the epilogue.
      Instruction *ip = &*F.getEntryBlock().getFirstInsertionPt();
      if (d.launchCrop) {
        // After the stub's own pop (its out-parameters are this launch's
        // geometry) and still before the launch.
        CallInst *pop = findPopConfig(F);
        if (!pop)
          report_fatal_error(
              "a fused host-dispatched GEMM needs the launch geometry, but its "
              "stub performs no __cudaPopCallConfiguration; the kernel was "
              "already fissioned device-side, so there is no correct code to "
              "emit here.");
        gridCfg = poppedGrid(*pop);
        blockCfg = poppedBlock(*pop);
        ip = pop->getNextNode();
      }
      IRBuilder<> B(ip);
      SmallVector<Value *, 16> args = buildArgs(B);
      B.CreateCall(callee, args);
      errs() << "[ozaki-host-dispatch] prepended dispatch to stub " << cn
             << " -> " << calleeName << " (fused) (C=arg" << d.cParam
             << " A=arg" << d.aParam << " B=arg" << d.bParam << ' ' << d.gM
             << 'x' << d.gNcols << 'x' << d.gK << ")\n";
    }
    changed = true;
  }
  return changed;
}

namespace {
struct ProfDesc {
  std::string kernel;
  int siteId = 0;
  unsigned nargs = 0;
  SmallVector<int, 8> ptrIdx;
  SmallVector<int, 8> seedWidth;
  // Byte offset of each pointer inside its argument (0 for a pointer
  // argument).
  SmallVector<int, 8> ptrOff;
};
} // namespace

static bool readProfDescriptors(const Options &O, Module &M,
                                std::map<std::string, ProfDesc> &out) {
  readDescriptors(
      O, M, LaunchKind::Profiling, [&](ArrayRef<StringRef> toks) -> Error {
        DescriptorCursor Cursor(toks);
        ProfDesc d;
        d.kernel = toks[0].str();
        d.siteId = Cursor.integer<int>("site id");
        d.nargs = Cursor.integer<unsigned>("argument count");
        unsigned np = Cursor.integer<unsigned>("pointer count");
        for (unsigned i = 0; i < np && Cursor.ok(); ++i)
          d.ptrIdx.push_back(Cursor.integer<int>("pointer argument"));
        for (unsigned i = 0; i < np && Cursor.ok(); ++i)
          d.seedWidth.push_back(Cursor.integer<int>("seed width"));
        d.ptrOff.assign(d.ptrIdx.size(), 0);
        if (Cursor.consume("off"))
          for (int &off : d.ptrOff) {
            off = Cursor.integer<int>("pointer offset");
            if (off < 0)
              Cursor.fail("its pointer offset " + Twine(off) + " is negative");
          }
        if (Error E = Cursor.takeError())
          return E;
        out[d.kernel] = d;
        return Error::success();
      });
  return !out.empty();
}

static Constant *profIntArray(Module &M, const Twine &nameT,
                              ArrayRef<int> vals) {
  std::string name = nameT.str();
  Type *i32 = Type::getInt32Ty(M.getContext());
  SmallVector<Constant *, 8> elts;
  for (int v : vals)
    elts.push_back(ConstantInt::get(i32, (unsigned)v));
  ArrayType *AT = ArrayType::get(i32, elts.size());
  Constant *init = ConstantArray::get(AT, elts);
  if (auto *old = M.getNamedGlobal(name))
    if (old->getValueType() == AT && old->hasInitializer() &&
        old->getInitializer() == init)
      return old;
  auto *GV = new GlobalVariable(M, AT, /*isConstant=*/true,
                                GlobalValue::PrivateLinkage, init, name);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

bool rewriteProfileStubBodies(const Options &O, Module &M) {
  std::map<std::string, ProfDesc> descs;
  if (!readProfDescriptors(O, M, descs))
    return false;

  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  FunctionCallee helper = M.getOrInsertFunction(
      "__poseidon_launch_profiled",
      FunctionType::get(
          Type::getVoidTy(C),
          {ptr, i32, ptr, ptr, i64, ptr, i32, ptr, i32, ptr, ptr, ptr}, false));

  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    auto dIt = llvm::find_if(
        descs, [&](const auto &kv) { return isLaunchStubFor(F, kv.first); });
    if (dIt == descs.end())
      continue;
    const ProfDesc &d = dIt->second;
    // The pass reaches the host module through more than one extension point;
    // a stub that already calls the helper must not be rewritten again.
    if (llvm::any_of(instructions(F), [](Instruction &I) {
          auto *CI = dyn_cast<CallInst>(&I);
          const Function *cal = CI ? CI->getCalledFunction() : nullptr;
          return cal && cal->getName() == "__poseidon_launch_profiled";
        }))
      continue;
    // The launch's own argument array and configuration: what the stub's
    // parameters look like depends on the host ABI (an aggregate argument
    // arrives by value, by reference or split into registers), what the
    // launch is handed does not.
    CallInst *launch = findLaunchCall(F);
    CallInst *pop = findPopConfig(F);
    // Launched as written, the profiled kernel would read shadows nobody
    // passed it.
    if (!launch || !pop) {
      diagnoseSite(F,
                   "the launch stub " + F.getName() +
                       " does not end in __cuda/__hipPopCallConfiguration "
                       "and cuda/hipLaunchKernel, so the profiling run cannot "
                       "pass its kernel the shadows it is profiled with. "
                       "Compile against the CUDA or ROCm headers, which "
                       "declare that launch API.",
                   DS_Error);
      continue;
    }

    Constant *idxGV = profIntArray(M, "poseidon.ptr." + d.kernel, d.ptrIdx);
    Constant *offGV = profIntArray(M, "poseidon.off." + d.kernel, d.ptrOff);
    Constant *seedGV =
        profIntArray(M, "poseidon.seed." + d.kernel, d.seedWidth);

    IRBuilder<> B(launch);
    B.CreateCall(
        helper,
        {launchedKernel(*launch), ConstantInt::get(i32, (unsigned)d.siteId),
         poppedGrid(*pop), poppedBlock(*pop), launchSharedBytes(*launch),
         launchStream(*launch), ConstantInt::get(i32, d.nargs),
         launchArguments(*launch), ConstantInt::get(i32, d.ptrIdx.size()),
         idxGV, offGV, seedGV});
    launch->replaceAllUsesWith(Constant::getNullValue(launch->getType()));
    launch->eraseFromParent();
    if (O.Diagnostics.Print)
      errs() << "[poseidon-profgen] launch of " << d.kernel
             << " now allocates and seeds " << d.ptrIdx.size()
             << " shadow buffer(s)\n";
    changed = true;
  }
  return changed;
}

// The header of every kernel the device half profiled, registered with the
// host profiler as a host site's is: the device profiler writes a kernel's
// profile only once the kernel has run, and the host profiler writes one with
// no records for a registered site that nothing wrote.
static bool registerProfiledKernels(const Options &O, Module &M) {
  bool changed = false;
  readProfileHeaders(O, M, [&](StringRef profileName, StringRef header) {
    changed |= registerProfileStatic(M, profileName, header);
  });
  return changed;
}

llvm::PreservedAnalyses HostStubPass::run(llvm::Module &M,
                                          llvm::ModuleAnalysisManager &MAM) {
  const Options &O = sessionOf(M, MAM).Opts;
  if (Target::of(M).isDevice() || !handsOverLaunches(O))
    return llvm::PreservedAnalyses::all();
  // Profile generation: every launch of an annotated kernel goes through the
  // runtime helper that supplies its shadow buffers. Host dispatch
  // materializes only in the optimize phase: profile-gen builds instrument the
  // original kernel and do not link the runtime.
  bool changed = false;
  if (O.Profiling.ProfileGenerate) {
    changed |= rewriteProfileStubBodies(O, M);
    changed |= registerProfiledKernels(O, M);
  } else {
    changed = rewriteGemmStubBodies(O, M);
  }
  return changed ? llvm::PreservedAnalyses::none()
                 : llvm::PreservedAnalyses::all();
}

} // namespace poseidon
