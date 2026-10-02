//===- HostDispatch.cpp - host-side GEMM library dispatch ----------------===//
//
// See HostDispatch.h. Suppressed during profile-gen and when no profile
// is given.
//===---------------------------------------------------------------------===//
#include "HostDispatch.h"
#include "Evaluators.h"
#include "Flags.h"
#include "LaunchDescriptors.h"
#include "Optimize.h"
#include "Matmul.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <map>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

int traceToArgIndex(const Value *v) {
  while (v) {
    if (auto *A = dyn_cast<Argument>(v))
      return (int)A->getArgNo();
    if (auto *GEP = dyn_cast<GetElementPtrInst>(v)) {
      v = GEP->getPointerOperand();
      continue;
    }
    if (auto *BC = dyn_cast<BitCastInst>(v)) {
      v = BC->getOperand(0);
      continue;
    }
    if (auto *AC = dyn_cast<AddrSpaceCastInst>(v)) {
      v = AC->getOperand(0);
      continue;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(v)) {
      if (CE->getOpcode() == Instruction::GetElementPtr || CE->isCast()) {
        v = CE->getOperand(0);
        continue;
      }
    }
    break;
  }
  return -1;
}

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

// Deploy-extent recovery: the deploy extents are statically present in the
// kernel's own bounds check as the constants gating the C store, and the launch
// geometry bounds them a second time; extent = min(crop, gridDim[axis] *
// blockDim[axis]) is exact in both directions. The second half is taken in the
// host stub rewrite, where the launch configuration lives.
static int axisIndexOf(TidAxis a) {
  switch (a) {
  case TidAxis::TidX:
    return 0;
  case TidAxis::TidY:
    return 1;
  case TidAxis::TidZ:
    return 2;
  default:
    return -1;
  }
}

namespace {
enum class SregKind { None, Tid, Ctaid, Ntid };
} // namespace

static SregKind sregKindOf(const Value *v, int &axis) {
  const auto *II = dyn_cast<IntrinsicInst>(v);
  if (!II)
    return SregKind::None;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    axis = 0;
    return SregKind::Tid;
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    axis = 1;
    return SregKind::Tid;
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
    axis = 2;
    return SregKind::Tid;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    axis = 0;
    return SregKind::Ctaid;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    axis = 1;
    return SregKind::Ctaid;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
    axis = 2;
    return SregKind::Ctaid;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
    axis = 0;
    return SregKind::Ntid;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
    axis = 1;
    return SregKind::Ntid;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
    axis = 2;
    return SregKind::Ntid;
  default:
    return SregKind::None;
  }
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
    int ta = -1;
    if (sregKindOf(add->getOperand(i), ta) != SregKind::Tid)
      continue;
    const auto *mul = dyn_cast<BinaryOperator>(add->getOperand(1 - i));
    if (!mul || mul->getOpcode() != Instruction::Mul)
      continue;
    for (unsigned j = 0; j < 2; ++j) {
      int ca = -1, na = -1;
      if (sregKindOf(mul->getOperand(j), ca) == SregKind::Ctaid &&
          sregKindOf(mul->getOperand(1 - j), na) == SregKind::Ntid &&
          ca == ta && na == ta) {
        axis = ta;
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
static bool recoverGuardExtents(Function &F, const AbstractMatmul &m,
                                const StoreInst *cStore, unsigned &cropM,
                                unsigned &cropN, int &axM, int &axN) {
  const ScalarLoopHandle &h = m.scalarLoop;
  if (h.aRowSlowAxis != TidAxis::Unknown || h.bColSlowAxis != TidAxis::Unknown)
    return false; // fused index: its extent is a product of two axes
  axM = axisIndexOf(h.aRowAxis);
  axN = axisIndexOf(h.bColAxis);
  if (axM < 0 || axN < 0 || axM == axN)
    return false;

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
        if (flags::Print)
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

bool computeGemmBodyNote(Function &F, const AbstractMatmul &m,
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

  note.aParam = traceToArgIndex(m.scalarLoop.aBase);
  note.bParam = traceToArgIndex(m.scalarLoop.bBase);
  if (note.aParam < 0 || note.bParam < 0)
    return false;

  // Find the C store and count argument-rooted stores (standalone iff exactly
  // one).
  int storeCount = 0;
  const StoreInst *cStore = nullptr;
  const StoreInst *lastArgStore = nullptr;
  for (Instruction &I : instructions(F)) {
    auto *S = dyn_cast<StoreInst>(&I);
    if (!S)
      continue;
    int dst = traceToArgIndex(S->getPointerOperand());
    if (dst < 0)
      continue;
    storeCount++;
    lastArgStore = S;
    if (valueReaches(S->getValueOperand(), m.outputValue))
      cStore = S;
  }
  // A fused body's stored value reaches the matmul result only through the
  // loop's lcssa accumulator, which valueReaches may miss; with exactly one
  // argument-rooted store, that store is the C store.
  if (!cStore && storeCount == 1)
    cStore = lastArgStore;
  if (!cStore)
    return false;
  note.cParam = traceToArgIndex(cStore->getPointerOperand());
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
      if (recoverGuardExtents(F, m, cStore, cropM, cropN, axM, axN)) {
        note.gM = cropM;
        note.gNcols = cropN;
        note.launchCrop = true;
        note.cropMAxis = axM;
        note.cropNAxis = axN;
        recovered = true;
        llvm::errs() << "[ozaki-host-dispatch] recovered deploy geometry for "
                     << F.getName() << ": M=" << cropM << " Ncols=" << cropN
                     << " K=" << K << " (profile M/N/K = " << m.globalM << "/"
                     << m.globalN << "/" << m.globalK
                     << " does not track matrix order; extents taken from the "
                        "kernel's own index guard on thread axes "
                     << axM << "/" << axN
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
          << m.globalM << "/" << m.globalN << "/" << m.globalK
          << ", deploy K = " << K
          << "). A dimension that does not track matrix order cannot be "
             "rescaled by K/globalK, and its true deploy extent is not "
             "statically known here, so no value would be correct. The "
             "host-dispatch candidate is WITHDRAWN for this site; the solver "
             "will choose among the remaining in-kernel candidates, which may "
             "be MATERIALLY LESS ACCURATE than the withdrawn one. Re-check the "
             "selected scheme for this site before trusting the result.\n";
      if (flags::StrictMode)
        llvm::report_fatal_error(
            "host-dispatch geometry is unrecoverable for a non-square GEMM at "
            "a "
            "changed deploy scale, and -poseidon-strict-mode is set: refusing "
            "to "
            "continue with a silently reduced candidate set.");
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

bool fissionGemmForDispatch(Function &F, const AbstractMatmul &m) {
  StoreInst *cStore = nullptr;
  for (Instruction &I : instructions(F)) {
    auto *S = dyn_cast<StoreInst>(&I);
    if (!S)
      continue;
    if (traceToArgIndex(S->getPointerOperand()) >= 0 &&
        valueReaches(S->getValueOperand(), m.outputValue)) {
      cStore = S;
      break;
    }
  }
  if (!cStore)
    return false;

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
  return true;
}

static NoteMap<GemmBodyNote> &noteMap() {
  static NoteMap<GemmBodyNote> m;
  return m;
}
void noteGemmBody(const Function *body, const GemmBodyNote &n) {
  noteMap().note(body, n);
}
bool getGemmBody(const Function *body, GemmBodyNote &out) {
  return noteMap().get(body, out);
}

static void emitDescriptorFile(StringRef cacheDir, StringRef wrapperName,
                               int cArg, int aArg, int bArg,
                               const GemmBodyNote &n) {
  // cArg aArg bArg N standalone numModuli gM gNcols gK lda ldb ldc aCol bCol
  // cCol scheme, then the keyword-tagged launch-crop block.
  if (!writeDescriptor(
          cacheDir, wrapperName, kOzDispatchScheme, "[ozaki-host-dispatch]",
          [&](raw_ostream &os) {
            os << ' ' << cArg << ' ' << aArg << ' ' << bArg << ' ' << n.N << ' '
               << (n.standalone ? 1 : 0) << ' ' << n.numModuli << ' ' << n.gM
               << ' ' << n.gNcols << ' ' << n.gK << ' ' << n.lda << ' ' << n.ldb
               << ' ' << n.ldc << ' ' << (n.aColMajor ? 1 : 0) << ' '
               << (n.bColMajor ? 1 : 0) << ' ' << (n.cColMajor ? 1 : 0) << ' '
               << (n.scheme == DispatchScheme::Tcec     ? 1
                   : n.scheme == DispatchScheme::Direct ? 2
                                                        : 0);
            // Optional trailing runtime-geometry block (tokens 17..36).
            if (n.runtimeDims) {
              const GemmBodyNote::RtDim *d[6] = {&n.rM,   &n.rNcols, &n.rK,
                                                 &n.rlda, &n.rldb,   &n.rldc};
              os << " 1";
              for (const GemmBodyNote::RtDim *x : d)
                os << ' ' << x->param << ' ' << x->mul << ' ' << x->add;
              os << ' ' << n.beta;
            }
            // Keyword-tagged rather than positional: the extents
            // the kernel's index guard gates its store with,
            // intersected at the dispatch with the launch
            // geometry.
            if (n.launchCrop)
              os << " lc " << n.cropMAxis << ' ' << n.cropNAxis;
          }))
    return;
  errs() << "[ozaki-host-dispatch] wrote descriptor for " << wrapperName
         << ": C=arg" << cArg << " A=arg" << aArg << " B=arg" << bArg << " "
         << n.gM << "x" << n.gNcols << "x" << n.gK << " nm=" << n.numModuli
         << (n.standalone ? " (standalone)" : " (fused)") << "\n";
}

// Shared descriptor emit for the per-site path and the deferred joint-dp
// flush: map C/A/B via `mapParam`, skipping (with a message) if any fails.
static void emitMappedDescriptor(StringRef cacheDir, StringRef wrapperName,
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
  if (mapped.runtimeDims) {
    GemmBodyNote::RtDim *dd[6] = {&mapped.rM,   &mapped.rNcols, &mapped.rK,
                                  &mapped.rlda, &mapped.rldb,   &mapped.rldc};
    for (GemmBodyNote::RtDim *x : dd) {
      if (x->param < 0)
        continue;
      int a = mapParam(x->param);
      if (a < 0) {
        errs() << "[ozaki-host-dispatch] could not map a runtime GEMM "
                  "dimension's body param to a wrapper launch arg for "
               << wrapperName << "; skipping descriptor\n";
        return;
      }
      x->param = a;
    }
  }
  emitDescriptorFile(cacheDir, wrapperName, cArg, aArg, bArg, mapped);
}

void writeGemmDescriptor(Function &wrapper, ArrayRef<Value *> primalArgs,
                         const GemmBodyNote &n, StringRef cacheDir) {
  if (!n.valid)
    return;
  emitMappedDescriptor(cacheDir, wrapper.getName(), n, [&](int bp) -> int {
    if (bp < 0 || (size_t)bp >= primalArgs.size())
      return -1;
    return traceToArgIndex(primalArgs[bp]);
  });
}

namespace {
struct PendingGemm {
  std::string wrapperName;
  std::vector<int> paramToArg; // body-param index -> wrapper launch-arg index
  std::string cacheDir;
};
static std::map<const Function *, PendingGemm> &pendingMap() {
  static std::map<const Function *, PendingGemm> m;
  return m;
}
} // namespace

void addPendingGemmDispatch(const Function *body, Function &wrapper,
                            ArrayRef<Value *> primalArgs, StringRef cacheDir) {
  PendingGemm pg;
  pg.wrapperName = wrapper.getName().str();
  pg.cacheDir = cacheDir.str();
  pg.paramToArg.resize(primalArgs.size());
  for (size_t i = 0; i < primalArgs.size(); ++i)
    pg.paramToArg[i] = traceToArgIndex(primalArgs[i]);
  pendingMap()[body] = std::move(pg);
}

void flushPendingGemmDispatches() {
  for (auto &kv : pendingMap()) {
    GemmBodyNote n;
    if (!getGemmBody(kv.first, n))
      continue; // this site was not materialized as a host dispatch
    const PendingGemm &pg = kv.second;
    emitMappedDescriptor(pg.cacheDir, pg.wrapperName, n, [&](int bp) -> int {
      if (bp < 0 || (size_t)bp >= pg.paramToArg.size())
        return -1;
      return pg.paramToArg[bp];
    });
  }
  pendingMap().clear();
}

namespace {
struct LaunchDesc {
  std::string kernel;
  int cArg, aArg, bArg;
  unsigned N;
  bool standalone;
  unsigned numModuli;
  DispatchScheme scheme = DispatchScheme::OzakiII;
  unsigned gM = 0, gNcols = 0, gK = 0;
  unsigned lda = 0, ldb = 0, ldc = 0;
  bool aColMajor = false, bColMajor = false, cColMajor = false;
  // Runtime geometry (see GemmBodyNote::runtimeDims).
  bool runtimeDims = false;
  GemmBodyNote::RtDim rM, rNcols, rK, rlda, rldb, rldc;
  double beta = 0.0;
  // Launch-geometry intersection (see GemmBodyNote::launchCrop).
  bool launchCrop = false;
  int cropMAxis = -1, cropNAxis = -1;
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

static bool
readOzDispatchDescriptors(StringRef cacheDir,
                          std::map<std::string, LaunchDesc> &descs) {
  readDescriptors(cacheDir, kOzDispatchScheme, [&](ArrayRef<StringRef> toks) {
    if (toks.size() < 17)
      return;
    LaunchDesc d;
    d.kernel = toks[0].str();
    (void)toks[1].getAsInteger(10, d.cArg);
    (void)toks[2].getAsInteger(10, d.aArg);
    (void)toks[3].getAsInteger(10, d.bArg);
    (void)toks[4].getAsInteger(10, d.N);
    int sa = 0;
    (void)toks[5].getAsInteger(10, sa);
    d.standalone = sa != 0;
    d.numModuli = kOzakiIIMaxModuli;
    int nm = (int)kOzakiIIMaxModuli;
    (void)toks[6].getAsInteger(10, nm);
    if (nm >= 0 && nm <= (int)kOzakiIIMaxModuli)
      d.numModuli = (unsigned)nm;
    int sc = 0;
    (void)toks[16].getAsInteger(10, sc);
    d.scheme = sc == 1   ? DispatchScheme::Tcec
               : sc == 2 ? DispatchScheme::Direct
                         : DispatchScheme::OzakiII;
    (void)toks[7].getAsInteger(10, d.gM);
    (void)toks[8].getAsInteger(10, d.gNcols);
    (void)toks[9].getAsInteger(10, d.gK);
    (void)toks[10].getAsInteger(10, d.lda);
    (void)toks[11].getAsInteger(10, d.ldb);
    (void)toks[12].getAsInteger(10, d.ldc);
    int ac = 0, bc = 0, cc = 0;
    (void)toks[13].getAsInteger(10, ac);
    (void)toks[14].getAsInteger(10, bc);
    (void)toks[15].getAsInteger(10, cc);
    d.aColMajor = ac != 0;
    d.bColMajor = bc != 0;
    d.cColMajor = cc != 0;
    // Trailing runtime-geometry block: flag + 6 x (param mul add) + beta.
    if (toks.size() >= 36) {
      int rt = 0;
      (void)toks[17].getAsInteger(10, rt);
      if (rt == 1) {
        GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                      &d.rlda, &d.rldb,   &d.rldc};
        for (unsigned i = 0; i < 6; ++i) {
          (void)toks[18 + 3 * i + 0].getAsInteger(10, dd[i]->param);
          (void)toks[18 + 3 * i + 1].getAsInteger(10, dd[i]->mul);
          (void)toks[18 + 3 * i + 2].getAsInteger(10, dd[i]->add);
        }
        d.runtimeDims = true;
        if (toks.size() >= 37)
          (void)toks[36].getAsDouble(d.beta);
      }
    }
    // Keyword-tagged launch-geometry block (every other token is an integer, so
    // the keyword cannot collide with one).
    for (unsigned i = 0; i + 2 < toks.size(); ++i) {
      if (toks[i] != "lc")
        continue;
      int am = -1, an = -1;
      if (!toks[i + 1].getAsInteger(10, am) &&
          !toks[i + 2].getAsInteger(10, an) && am >= 0 && am < 3 && an >= 0 &&
          an < 3 && am != an) {
        d.launchCrop = true;
        d.cropMAxis = am;
        d.cropNAxis = an;
      }
      break;
    }
    descs[d.kernel] = d;
  });
  return !descs.empty();
}

// Launch geometry, host side: the configuration a <<<grid, block>>> site pushes
// is popped inside the stub, so the stub is where a launch-derived extent can
// be taken; at the call site the dim3 temporaries are not constants at
// PipelineStart.
static FunctionCallee getPopCallConfig(Module &M) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  FunctionType *FT =
      FunctionType::get(Type::getInt32Ty(C), {ptr, ptr, ptr, ptr}, false);
  return M.getOrInsertFunction("__cudaPopCallConfiguration", FT);
}

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
  B.CreateCall(getPopCallConfig(M), {grid, block, shmem, stream});
  gridOut = grid;
  blockOut = block;
}

// The pop the fused stub already performs; its dim3 out-parameters are the
// same geometry, and a second pop would consume the NEXT launch's.
static CallInst *findPopConfig(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *CI = dyn_cast<CallInst>(&I))
      if (const Function *cal = CI->getCalledFunction())
        if (cal->getName() == "__cudaPopCallConfiguration" &&
            CI->arg_size() >= 2)
          return CI;
  return nullptr;
}

// Runs before inlining (PipelineStart), while `__device_stub__K` is still a
// standalone function whose parameters are the kernel's launch args. Standalone
// descriptors replace the whole body with the dispatch call; fused bodies get
// the dispatch prepended and keep the (device-side fissioned) launch.
bool rewriteGemmStubBodies(Module &M, StringRef cacheDir) {
  std::map<std::string, LaunchDesc> descs;
  if (!readOzDispatchDescriptors(cacheDir, descs))
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
    const LaunchDesc *dp = nullptr;
    for (auto &kv : descs) {
      StringRef kname = kv.first;
      if (isLaunchStubFor(F, kname)) {
        dp = &kv.second;
        break;
      }
    }
    if (!dp)
      continue;
    const LaunchDesc &d = *dp;
    unsigned need = (unsigned)std::max({d.cArg, d.aArg, d.bArg});
    if (d.runtimeDims) {
      const GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                          &d.rlda, &d.rldb,   &d.rldc};
      for (const GemmBodyNote::RtDim *x : dd)
        if (x->param >= 0)
          need = std::max(need, (unsigned)x->param);
    }
    if (F.arg_size() <= need)
      continue;
    if (d.runtimeDims) {
      bool ok = true;
      const GemmBodyNote::RtDim *dd[6] = {&d.rM,   &d.rNcols, &d.rK,
                                          &d.rlda, &d.rldb,   &d.rldc};
      for (const GemmBodyNote::RtDim *x : dd)
        if (x->param >= 0 && !F.getArg(x->param)->getType()->isIntegerTy())
          ok = false;
      if (!ok) {
        errs() << "[ozaki-host-dispatch] " << cn
               << ": a runtime GEMM dimension references a non-integer launch "
                  "argument; skipping the dispatch rewrite\n";
        continue;
      }
    }
    Value *Cp = F.getArg(d.cArg);
    Value *Ap = F.getArg(d.aArg);
    Value *Bp = F.getArg(d.bArg);
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

    // A square row-major NN GEMM takes the square entry; anything non-square,
    // transposed or launch-cropped takes the layout-aware _ex entry.
    // A runtime-shaped GEMM always takes _ex: its extents are not known here.
    bool squareRM = !d.runtimeDims && !d.launchCrop &&
                    (d.gM == d.gNcols && d.gNcols == d.gK && !d.aColMajor &&
                     !d.bColMajor && !d.cColMajor);
    // Resolved per descriptor: the scheme is a per-site solver decision.
    FunctionCallee callee =
        squareRM ? getOzakiDgemm(M, d.scheme) : getOzakiDgemmEx(M, d.scheme);
    // Runtime dimensions are materialized as `mul*arg + add` right before the
    // dispatch call, so a build profiled at one mesh size deploys at another.
    auto rtDim = [&](IRBuilder<> &B, const GemmBodyNote::RtDim &r,
                     unsigned constFallback) -> Value * {
      if (!d.runtimeDims)
        return I32((int)constFallback);
      if (r.param < 0)
        return ConstantInt::get(i32, (uint64_t)r.add);
      Value *v = B.CreateZExtOrTrunc(F.getArg(r.param), i32);
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
             << calleeName << " (C=arg" << d.cArg << " A=arg" << d.aArg
             << " B=arg" << d.bArg << " " << d.gM << "x" << d.gNcols << "x"
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
        gridCfg = pop->getArgOperand(0);
        blockCfg = pop->getArgOperand(1);
        ip = pop->getNextNode();
      }
      IRBuilder<> B(ip);
      SmallVector<Value *, 16> args = buildArgs(B);
      B.CreateCall(callee, args);
      errs() << "[ozaki-host-dispatch] prepended dispatch to stub " << cn
             << " -> " << calleeName << " (fused) (C=arg" << d.cArg << " A=arg"
             << d.aArg << " B=arg" << d.bArg << " " << d.gM << "x" << d.gNcols
             << "x" << d.gK << ")\n";
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
};
} // namespace

static bool readProfDescriptors(StringRef cacheDir,
                                std::map<std::string, ProfDesc> &out) {
  readDescriptors(cacheDir, kProfGenScheme, [&](ArrayRef<StringRef> toks) {
    if (toks.size() < 4)
      return;
    ProfDesc d;
    d.kernel = toks[0].str();
    unsigned np = 0;
    if (toks[1].getAsInteger(10, d.siteId) ||
        toks[2].getAsInteger(10, d.nargs) || toks[3].getAsInteger(10, np))
      return;
    if (toks.size() < 4 + 2 * (size_t)np)
      return;
    for (unsigned i = 0; i < np; ++i) {
      int v = 0;
      if (toks[4 + i].getAsInteger(10, v))
        return;
      d.ptrIdx.push_back(v);
    }
    for (unsigned i = 0; i < np; ++i) {
      int v = 0;
      if (toks[4 + np + i].getAsInteger(10, v))
        return;
      d.seedWidth.push_back(v);
    }
    out[d.kernel] = d;
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

bool rewriteProfileStubBodies(Module &M, StringRef cacheDir) {
  std::map<std::string, ProfDesc> descs;
  if (!readProfDescriptors(cacheDir, descs))
    return false;

  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *i64 = Type::getInt64Ty(C);
  Type *dim3Ty = ArrayType::get(i32, 3);
  FunctionCallee helper = M.getOrInsertFunction(
      "__poseidon_launch_profiled",
      FunctionType::get(Type::getVoidTy(C),
                        {ptr, i32, ptr, ptr, i64, ptr, i32, ptr, i32, ptr, ptr},
                        false));

  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    const ProfDesc *dp = nullptr;
    for (auto &kv : descs)
      if (isLaunchStubFor(F, kv.first)) {
        dp = &kv.second;
        break;
      }
    if (!dp)
      continue;
    const ProfDesc &d = *dp;
    // The pass reaches the host module through more than one extension point;
    // a stub that already calls the helper must not be rebuilt.
    bool done = false;
    for (Instruction &I : instructions(F))
      if (auto *CI = dyn_cast<CallInst>(&I))
        if (const Function *cal = CI->getCalledFunction())
          done |= cal->getName() == "__poseidon_launch_profiled";
    if (done)
      continue;
    if (F.arg_size() != d.nargs) {
      errs() << "[poseidon-profgen] " << F.getName() << " takes "
             << F.arg_size() << " argument(s) but the device compilation "
             << "recorded " << d.nargs << "; not rewriting this launch\n";
      continue;
    }

    Constant *idxGV = profIntArray(M, "poseidon.ptr." + d.kernel, d.ptrIdx);
    Constant *seedGV =
        profIntArray(M, "poseidon.seed." + d.kernel, d.seedWidth);

    F.deleteBody();
    IRBuilder<> B(BasicBlock::Create(C, "entry", &F));
    auto *grid = B.CreateAlloca(dim3Ty, nullptr, "poseidon.grid");
    auto *block = B.CreateAlloca(dim3Ty, nullptr, "poseidon.block");
    grid->setAlignment(Align(8));
    block->setAlignment(Align(8));
    auto *shmem = B.CreateAlloca(i64, nullptr, "poseidon.shmem");
    auto *stream = B.CreateAlloca(ptr, nullptr, "poseidon.stream");
    B.CreateCall(getPopCallConfig(M), {grid, block, shmem, stream});

    ArrayType *argsTy = ArrayType::get(ptr, F.arg_size());
    Value *argbuf = B.CreateAlloca(argsTy, nullptr, "poseidon.args");
    for (unsigned i = 0; i < F.arg_size(); ++i) {
      Argument *A = F.getArg(i);
      Value *slot = B.CreateAlloca(A->getType());
      B.CreateStore(A, slot);
      B.CreateStore(slot, B.CreateConstInBoundsGEP2_32(argsTy, argbuf, 0, i));
    }
    B.CreateCall(helper,
                 {&F, ConstantInt::get(i32, (unsigned)d.siteId), grid, block,
                  B.CreateLoad(i64, shmem), B.CreateLoad(ptr, stream),
                  ConstantInt::get(i32, F.arg_size()), argbuf,
                  ConstantInt::get(i32, d.ptrIdx.size()), idxGV, seedGV});
    B.CreateRetVoid();
    errs() << "[poseidon-profgen] launch of " << d.kernel
           << " now allocates and seeds " << d.ptrIdx.size()
           << " shadow buffer(s)\n";
    changed = true;
  }
  return changed;
}

llvm::PreservedAnalyses HostStubPass::run(llvm::Module &M,
                                          llvm::ModuleAnalysisManager &) {
  applyFlagDefaults();
  // Profile generation: every launch of an annotated kernel goes through the
  // runtime helper that supplies its shadow buffers.
  if (flags::ProfileGenerate && !Triple(M.getTargetTriple()).isNVPTX())
    return rewriteProfileStubBodies(M, flags::Cache)
               ? llvm::PreservedAnalyses::none()
               : llvm::PreservedAnalyses::all();
  // Host dispatch materializes only in the optimize phase: profile-gen builds
  // instrument the original kernel and do not link the runtime, and with no
  // profile any cache/*.ozdispatch is a stale leftover.
  if (!flags::OzakiHostDispatch || Triple(M.getTargetTriple()).isNVPTX() ||
      flags::ProfileGenerate || flags::ProfileUse.empty())
    return llvm::PreservedAnalyses::all();
  return rewriteGemmStubBodies(M, flags::Cache)
             ? llvm::PreservedAnalyses::none()
             : llvm::PreservedAnalyses::all();
}

} // namespace poseidon
