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
#include "MatmulInternal.h"
#include "Pass/Session.h"
#include "Sites/Annotations.h"
#include "Sites/LaunchDescriptors.h"
#include "Sites/LaunchStubs.h"
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
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <map>
#include <optional>
#include <string>

using namespace llvm;
using namespace llvm::PatternMatch;

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
    Value *Cond;
    if (!match(BI, m_Br(m_Value(Cond), m_BasicBlock(), m_BasicBlock())))
      continue;
    for (unsigned i = 0; i < 2; ++i) {
      // The edge query rejects a duplicated edge on its own, so a two-way
      // branch to one block dominates nothing and contributes no fact.
      if (!DT.dominates(BasicBlockEdge(&BB, BI->getSuccessor(i)), SB))
        continue;
      if (!collectIndexBounds(Cond, /*holds=*/i == 0, bound)) {
        if (O.Diagnostics.Print)
          llvm::errs() << "[ozaki-host-dispatch] " << F.getName()
                       << ": a condition gating the GEMM store is not a "
                          "thread-index bound\n";
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

  LaunchRef aRoot = tracePointerRoot(m.scalarLoop.aPtr);
  LaunchRef bRoot = tracePointerRoot(m.scalarLoop.bPtr);
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

  // Every dim must be deploy scale: K is, but globalM/globalN are
  // profile-scale and are rescaled by K/globalK. Fall back to the square dim K
  // when the global geometry is unavailable. Leading dims come from the handle
  // (bytes to elements); C mirrors B's major order.
  unsigned gM = 0, gNcols = 0;
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
        gM = cropM;
        gNcols = cropN;
        note.launchCrop = true;
        note.cropMAxis = axM;
        note.cropNAxis = axN;
        recovered = true;
        llvm::errs() << "[ozaki-host-dispatch] recovered deploy geometry for "
                     << F.getName() << ": M=" << cropM << " Ncols=" << cropN
                     << " K=" << K << " from the index guard on thread axes "
                     << axM << '/' << axN << '\n';
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
          << ", deploy K = " << K << ")\n";
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
      gM = (unsigned)((uint64_t)m.globalM * (uint64_t)K / (uint64_t)m.globalK);
      gNcols =
          (unsigned)((uint64_t)m.globalN * (uint64_t)K / (uint64_t)m.globalK);
    }
  } else {
    gM = m.globalM ? m.globalM : (unsigned)K;
    gNcols = m.globalN ? m.globalN : (unsigned)K;
  }
  note.aColMajor = false;
  note.bColMajor = bColMajor;
  note.cColMajor = bColMajor; // heuristic: output matches B's major order
  note.dims[DimM] = RuntimeDim::constant(gM);
  note.dims[DimNcols] = RuntimeDim::constant(gNcols);
  note.dims[DimK] = RuntimeDim::constant(K);
  note.dims[DimLda] =
      RuntimeDim::constant(h.aLeadingDimByte ? h.aLeadingDimByte / 8 : K);
  note.dims[DimLdb] =
      RuntimeDim::constant(h.bLeadingDimByte ? h.bLeadingDimByte / 8 : K);
  note.dims[DimLdc] = RuntimeDim::constant(note.cColMajor ? gM : gNcols);
  note.standalone = (storeCount == 1);
  note.valid = true;
  return true;
}

// Replaces the GEMM result where the epilogue consumes it with a load from the
// output buffer that the prepended host dispatch fills, so the reduction loop
// dies. F stores a value that reaches the product to an argument, as it does
// when computeGemmBodyNote gave it a valid note that is not standalone.
static void fissionGemmForDispatch(Function &F, const AbstractMatmul &m) {
  StoreInst *cStore = productStore(F, m);
  assert(cStore &&
         "a body with a fused note stores the product to an argument");

  // Replace the GEMM result where the epilogue consumes it with a load from the
  // output buffer (filled by the prepended host dispatch); the reduction loop
  // becomes dead. In-loop (recurrence) uses stay.
  Value *resultVal = cStore->getValueOperand();
  Value *outAddr = cStore->getPointerOperand();
  IRBuilder<> B(cStore);
  LoadInst *loaded =
      B.CreateLoad(resultVal->getType(), outAddr, "dispatch.gemm");
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

static GemmBodyNote runtimeShapeNote(const AbstractMatmul &m) {
  const RuntimeGemmHandle &g = *m.hostGemm;
  GemmBodyNote note;
  note.cParam = g.cRef.param;
  note.aParam = g.aRef.param;
  note.bParam = g.bRef.param;
  note.cOffset = g.cRef.offset;
  note.aOffset = g.aRef.offset;
  note.bOffset = g.bRef.offset;
  note.assumedEqual = g.assumedEqual;
  note.standalone = true;
  note.valid = true;
  note.beta = g.beta;
  note.aColMajor = g.aColMajor;
  note.bColMajor = g.bColMajor;
  note.cColMajor = g.cColMajor;
  note.dims[DimM] = g.M;
  note.dims[DimNcols] = g.Ncols;
  note.dims[DimK] = g.K;
  note.dims[DimLda] = g.lda;
  note.dims[DimLdb] = g.ldb;
  note.dims[DimLdc] = g.ldc;
  return note;
}

void applyLibraryCall(Session &S, const AbstractMatmul &m,
                      const SchemeVariant &v) {
  assert(S.Opts.Matmul.OzakiHostDispatch &&
         "library calls are proposed only under -poseidon-ozaki-host-dispatch");
  Function *F = nullptr;
  GemmBodyNote note;
  switch (m.origin) {
  case AbstractMatmul::Origin::HostGemmLoopNest:
    F = m.hostGemm->cStore->getFunction();
    note = runtimeShapeNote(m);
    break;
  case AbstractMatmul::Origin::ScalarLoopReduction: {
    F = m.scalarLoop.fma->getFunction();
    bool built = computeGemmBodyNote(S.Opts, *F, m, note);
    assert(built && note.valid &&
           "a library call is proposed only for a body with a GEMM note");
    (void)built;
    break;
  }
  }
  note.scheme = descriptorScheme(v);
  note.numModuli = descriptorModulusField(v);
  S.Matmul.GemmBodies[F] = note;
  if (!note.standalone)
    fissionGemmForDispatch(*F, m);
}

static void emitDescriptor(Session &S, StringRef wrapperName, int cArg,
                           int aArg, int bArg, const GemmBodyNote &n) {
  std::string message;
  raw_string_ostream msg(message);
  msg << "[ozaki-host-dispatch] wrote descriptor for " << wrapperName
      << ": C=" << launchRefString({cArg, n.cOffset}, "arg")
      << " A=" << launchRefString({aArg, n.aOffset}, "arg")
      << " B=" << launchRefString({bArg, n.bOffset}, "arg") << ' '
      << runtimeDimString(n.dims[DimM]) << 'x'
      << runtimeDimString(n.dims[DimNcols]) << 'x'
      << runtimeDimString(n.dims[DimK]) << " nm=" << n.numModuli
      << (n.standalone ? " (standalone)" : " (fused)");
  for (const auto &[x, y] : n.assumedEqual)
    msg << ", if " << launchRefString(x, "arg")
        << " == " << launchRefString(y, "arg");
  msg << '\n';
  recordDescriptor(
      S, LaunchKind::HostDispatch, wrapperName,
      [&](raw_ostream &os) {
        os << ' ' << cArg << ' ' << aArg << ' ' << bArg << ' '
           << (n.standalone ? 1 : 0) << ' ' << n.numModuli << ' '
           << (n.aColMajor ? 1 : 0) << ' ' << (n.bColMajor ? 1 : 0) << ' '
           << (n.cColMajor ? 1 : 0) << ' ' << (unsigned)n.scheme;
        // Per dimension the factor count, each factor's argument and offset,
        // mul and add.
        os << " rd";
        for (const RuntimeDim &x : n.dims) {
          os << ' ' << x.factors.size();
          for (const LaunchRef &f : x.factors)
            os << ' ' << f.param << ' ' << f.offset;
          os << ' ' << x.mul << ' ' << x.add;
        }
        os << ' ' << n.beta;
        // The extents the kernel's index guard gates its store with,
        // intersected at the dispatch with the launch geometry.
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
// flush: map C/A/B through `paramToArg` (body parameter to wrapper launch
// argument, -1 when there is none), skipping (with a message) if any fails.
static void emitMappedDescriptor(Session &S, StringRef wrapperName,
                                 const GemmBodyNote &n,
                                 ArrayRef<int> paramToArg) {
  auto mapParam = [&](int bp) -> int {
    if (bp < 0 || (size_t)bp >= paramToArg.size())
      return -1;
    return paramToArg[bp];
  };
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
  for (RuntimeDim &x : mapped.dims)
    for (LaunchRef &f : x.factors)
      refs.push_back(&f);
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

static std::vector<int> launchArgsOf(ArrayRef<Value *> primalArgs) {
  std::vector<int> paramToArg(primalArgs.size());
  for (size_t i = 0; i < primalArgs.size(); ++i)
    paramToArg[i] = traceToArgIndex(primalArgs[i]);
  return paramToArg;
}

void writeGemmDescriptor(Session &S, Function &wrapper,
                         ArrayRef<Value *> primalArgs, const GemmBodyNote &n) {
  if (!n.valid)
    return;
  emitMappedDescriptor(S, wrapper.getName(), n, launchArgsOf(primalArgs));
}

void addPendingGemmDispatch(Session &S, const Function *body, Function &wrapper,
                            ArrayRef<Value *> primalArgs) {
  S.Matmul.PendingGemms[body] = {wrapper.getName().str(),
                                 launchArgsOf(primalArgs)};
}

void flushPendingGemmDispatches(Session &S) {
  for (auto &[body, pg] : S.Matmul.PendingGemms)
    // A site not materialized as a host dispatch has no note.
    if (auto n = S.Matmul.GemmBodies.find(body);
        n != S.Matmul.GemmBodies.end() && n->second.valid)
      emitMappedDescriptor(S, pg.wrapperName, n->second, pg.paramToArg);
  S.Matmul.PendingGemms.clear();
}

namespace {
// A descriptor read back on the host side, its operand roles and dimension
// factors naming launch arguments of `kernel`.
struct LaunchDesc : GemmBodyNote {
  std::string kernel;
};
} // namespace

// The entry points of runtime/GemmDispatch.h.
static const char *dispatchSym(DispatchScheme s) {
  switch (s) {
  case DispatchScheme::OzakiII:
    return "__poseidon_ozaki_dgemm";
  case DispatchScheme::Tcec:
    return "__poseidon_tcec_dgemm";
  case DispatchScheme::Direct:
    return "__poseidon_direct_dgemm";
  case DispatchScheme::Native:
    return "__poseidon_native_dgemm";
  }
  llvm_unreachable("unknown DispatchScheme");
}

// void(C,A,B, M,Ncols,K, lda,ldb,ldc, aColMajor,bColMajor,cColMajor,
//      alpha,beta, stream[, the descriptor's num_moduli field])
static FunctionCallee getDispatchCallee(Module &M, DispatchScheme scheme) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  SmallVector<Type *, 16> params = {ptr, ptr, ptr, i32, i32, i32, i32, i32,
                                    i32, i32, i32, i32, dbl, dbl, ptr};
  if (scheme != DispatchScheme::Native)
    params.push_back(i32);
  return M.getOrInsertFunction(
      dispatchSym(scheme),
      FunctionType::get(Type::getVoidTy(C), params, false));
}

static bool readDispatchDescriptors(const Options &O, Module &M,
                                    std::map<std::string, LaunchDesc> &descs) {
  readDescriptors(
      O, M, LaunchKind::HostDispatch, [&](ArrayRef<StringRef> toks) -> Error {
        DescriptorCursor Cursor(toks);
        LaunchDesc d;
        d.kernel = toks[0].str();
        d.cParam = Cursor.integer<int>("C argument");
        d.aParam = Cursor.integer<int>("A argument");
        d.bParam = Cursor.integer<int>("B argument");
        d.standalone = Cursor.integer<int>("standalone flag") != 0;
        d.numModuli = Cursor.integer<unsigned>("modulus field");
        if (d.numModuli > kOzakiIIMaxModuli)
          Cursor.fail("its modulus field " + Twine(d.numModuli) + " is above " +
                      Twine(kOzakiIIMaxModuli));
        d.aColMajor = Cursor.integer<int>("A layout") != 0;
        d.bColMajor = Cursor.integer<int>("B layout") != 0;
        d.cColMajor = Cursor.integer<int>("C layout") != 0;
        unsigned scheme = Cursor.integer<unsigned>("scheme");
        if (scheme > (unsigned)DispatchScheme::Native)
          Cursor.fail("its scheme " + Twine(scheme) + " is above " +
                      Twine((unsigned)DispatchScheme::Native));
        d.scheme = (DispatchScheme)scheme;
        // Keyword blocks (see emitDescriptor), of which rd is required.
        bool haveDims = false;
        while (!Cursor.atEnd()) {
          if (Cursor.consume("rd")) {
            for (RuntimeDim &x : d.dims) {
              unsigned nf = Cursor.integer<unsigned>("factor count");
              if (nf > 8)
                Cursor.fail("its factor count " + Twine(nf) + " is above 8");
              for (unsigned f = 0; f < nf && Cursor.ok(); ++f) {
                LaunchRef r;
                r.param = Cursor.integer<int>("factor argument");
                r.offset = Cursor.integer<int64_t>("factor offset");
                x.factors.push_back(r);
              }
              x.mul = Cursor.integer<int64_t>("dimension factor");
              x.add = Cursor.integer<int64_t>("dimension term");
            }
            d.beta = Cursor.real("beta");
            haveDims = true;
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
        if (!haveDims)
          Cursor.fail("it has no rd block");
        if (Error E = Cursor.takeError())
          return E;
        descs[d.kernel] = d;
        return Error::success();
      });
  return !descs.empty();
}

// Kernel argument `r.param` lives where entry r.param of the launch's argument
// array points, as the CUDA runtime reads it; a field of an aggregate argument
// is at its offset from there in every host ABI, which the stub's own
// parameters are not (a small aggregate arrives split into registers). An
// integer is read as `T` at its address: recognition admits no integer
// narrower than 32 bits, and on the little-endian hosts the runtimes support
// the first 32 bits of a wider one are its low 32 bits.
static Value *loadLaunchValue(IRBuilder<> &B, Value *argArray,
                              const LaunchRef &r, Type *T) {
  Type *ptr = B.getPtrTy();
  Value *at = B.CreateLoad(
      ptr, B.CreateConstInBoundsGEP1_32(ptr, argArray, (unsigned)r.param));
  if (r.offset > 0)
    at = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), at, (uint64_t)r.offset);
  return B.CreateLoad(T, at);
}

// Builds the dispatch in front of the stub's launch, from the launch's own
// argument array. The launch is removed (standalone) or kept (fused); with
// assumed equalities it runs instead of the dispatch whenever one of them does
// not hold.
static bool rewriteStub(Module &M, Function &F, const LaunchDesc &d) {
  CallInst *launch = findLaunchCall(F);
  CallInst *pop = findPopConfig(F);
  // A fused body was already fissioned device-side and computes nothing
  // without the dispatch, so it can neither fall back nor be skipped.
  if (!d.standalone &&
      (!launch || (d.launchCrop && !pop) || !d.assumedEqual.empty()))
    report_fatal_error(Twine("Poseidon: the stub ") + F.getName() +
                       " of a fused host-dispatched GEMM has no launch to "
                       "read the operands from, or a condition to check "
                       "first, and its kernel was fissioned device-side");
  if (!launch || (d.launchCrop && !pop)) {
    errs() << "[ozaki-host-dispatch] " << F.getName()
           << ": no launch to read the kernel's arguments from; launching the "
              "kernel as written\n";
    return false;
  }
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  auto I32 = [&](int64_t v) { return ConstantInt::get(i32, (uint64_t)v); };
  Value *argArray = launchArguments(*launch);
  // Resolved per descriptor: the scheme is a per-site solver decision.
  FunctionCallee callee = getDispatchCallee(M, d.scheme);

  IRBuilder<> B(launch);
  auto value = [&](const LaunchRef &r, Type *T) {
    return loadLaunchValue(B, argArray, r, T);
  };
  auto dim = [&](const RuntimeDim &r) {
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
  auto cropDim = [&](int axis, const RuntimeDim &crop) -> Value * {
    Value *g = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedGrid(*pop), axis));
    Value *b = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedBlock(*pop), axis));
    Value *ext = B.CreateMul(g, b);
    return B.CreateSelect(B.CreateICmpULT(ext, I32(crop.add)), ext,
                          I32(crop.add));
  };

  Value *Cp = value({d.cParam, d.cOffset}, ptr);
  Value *Ap = value({d.aParam, d.aOffset}, ptr);
  Value *Bp = value({d.bParam, d.bOffset}, ptr);
  Value *beta = ConstantFP::get(dbl, d.beta);
  Value *one = ConstantFP::get(dbl, 1.0);
  Value *stream = ConstantPointerNull::get(cast<PointerType>(ptr));
  SmallVector<Value *, 16> args = {
      Cp,
      Ap,
      Bp,
      d.launchCrop ? cropDim(d.cropMAxis, d.dims[DimM]) : dim(d.dims[DimM]),
      d.launchCrop ? cropDim(d.cropNAxis, d.dims[DimNcols])
                   : dim(d.dims[DimNcols]),
      dim(d.dims[DimK]),
      dim(d.dims[DimLda]),
      dim(d.dims[DimLdb]),
      dim(d.dims[DimLdc]),
      I32(d.aColMajor ? 1 : 0),
      I32(d.bColMajor ? 1 : 0),
      I32(d.cColMajor ? 1 : 0),
      one,
      beta,
      stream};
  if (d.scheme != DispatchScheme::Native)
    args.push_back(I32(d.numModuli));
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
         << F.getName() << " with " << dispatchSym(d.scheme)
         << " (C=" << launchRefString({d.cParam, d.cOffset}, "arg")
         << " A=" << launchRefString({d.aParam, d.aOffset}, "arg")
         << " B=" << launchRefString({d.bParam, d.bOffset}, "arg") << ')';
  for (const auto &[x, y] : d.assumedEqual)
    errs() << ", if " << launchRefString(x, "arg")
           << " == " << launchRefString(y, "arg");
  errs() << '\n';
  return true;
}

// Runs before inlining (PipelineStart), while `__device_stub__K` is still a
// function of its own that ends in the kernel's launch.
bool rewriteGemmStubBodies(const Options &O, Module &M) {
  std::map<std::string, LaunchDesc> descs;
  if (!readDispatchDescriptors(O, M, descs))
    return false;

  bool changed = false;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.getReturnType()->isVoidTy())
      continue;
    auto dIt = llvm::find_if(
        descs, [&](const auto &kv) { return isLaunchStubFor(F, kv.first); });
    if (dIt != descs.end())
      changed |= rewriteStub(M, F, dIt->second);
  }
  return changed;
}

} // namespace poseidon
