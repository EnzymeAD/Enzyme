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
#include "Sites/SiteIdentity.h"
#include "Targets/OffloadRuntime.h"
#include "Targets/Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <array>
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

// Tighten bound[axis] to `ub` when `idx` is the global thread index of `axis`.
static bool boundGlobalThreadIndex(const Value *idx, uint64_t ub,
                                   uint64_t bound[3]) {
  int axis = -1;
  if (!isGlobalThreadIndex(idx, axis))
    return false;
  if (!bound[axis] || ub < bound[axis])
    bound[axis] = ub;
  return true;
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
  // (a | b) u< 2^n is how (a u< 2^n) && (b u< 2^n) folds.
  if (const auto *Or = dyn_cast<BinaryOperator>(idx);
      Or && Or->getOpcode() == Instruction::Or && CmpInst::isUnsigned(P) &&
      isPowerOf2_64(ub))
    return boundGlobalThreadIndex(Or->getOperand(0), ub, bound) &&
           boundGlobalThreadIndex(Or->getOperand(1), ub, bound);
  return boundGlobalThreadIndex(idx, ub, bound);
}

// The exclusive upper bound that the conditions gating the C store (outside the
// reduction loop) put on the global thread index of each axis, 0 for an axis
// they leave unbounded; none when one of them is not such a bound.
static std::optional<std::array<uint64_t, 3>>
storeGuardBounds(Function &F, const AbstractMatmul &m,
                 const StoreInst *cStore) {
  DominatorTree DT(F);
  const BasicBlock *SB = cStore->getParent();
  std::array<uint64_t, 3> bound = {0, 0, 0};
  for (BasicBlock &BB : F) {
    if (m.scalarLoop.blocks.count(&BB))
      continue;
    Instruction *T = BB.getTerminator();
    if (T->getNumSuccessors() < 2)
      continue;
    for (unsigned i = 0, e = T->getNumSuccessors(); i < e; ++i) {
      // The edge query rejects a duplicated edge on its own, so a two-way
      // branch to one block dominates nothing and contributes no fact.
      if (!DT.dominates(BasicBlockEdge(&BB, T->getSuccessor(i)), SB))
        continue;
      Value *Cond = nullptr;
      if (!match(T, m_Br(m_Value(Cond), m_BasicBlock(), m_BasicBlock())) ||
          !collectIndexBounds(Cond, /*holds=*/i == 0, bound.data()))
        return std::nullopt;
    }
  }
  return bound;
}

// Whether `addr` is its base pointer when every thread and block index is 0.
static bool startsAtBase(ScalarEvolution &SE, const SCEV *addr) {
  const SCEV *offset = SE.getMinusSCEV(addr, SE.getPointerBase(addr));
  struct LaunchIndices {
    ScalarEvolution &SE;
    ValueToSCEVMapTy atZero;
    bool follow(const SCEV *S) {
      if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
        LaunchValue kind = launchGeometryOf(U->getValue()).Kind;
        if (kind == LaunchValue::ThreadIndex || kind == LaunchValue::BlockIndex)
          atZero[U->getValue()] = SE.getZero(U->getType());
      }
      return true;
    }
    bool isDone() const { return false; }
  } indices{SE, {}};
  visitAll(offset, indices);
  return SCEVParameterRewriter::rewrite(offset, SE, indices.atZero)->isZero();
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

static bool computeGemmBodyNote(const Options &O, Function &F,
                                const AbstractMatmul &m, GemmBodyNote &note) {
  note = GemmBodyNote{};
  if (m.origin != AbstractMatmul::Origin::ScalarLoopReduction)
    return false;
  auto refuse = [&](const Twine &why) {
    if (O.Diagnostics.Print)
      llvm::errs() << "[ozaki-host-dispatch] " << F.getName() << ": " << why
                   << '\n';
    return false;
  };

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
  std::optional<unsigned> axM = tidAxisIndex(h.aRowAxis);
  std::optional<unsigned> axN = tidAxisIndex(h.bColAxis);
  if (!axM || !axN || *axM == *axN)
    return refuse("the product's rows and columns do not run along two thread "
                  "axes");

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
  StoreInst *lastArgStore = nullptr;
  for (Instruction &I : instructions(F))
    if (auto *S = dyn_cast<StoreInst>(&I);
        S && tracePointerRoot(S->getPointerOperand()).param >= 0) {
      storeCount++;
      lastArgStore = S;
    }
  StoreInst *cStore = productStore(F, m);
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

  std::optional<std::array<uint64_t, 3>> bound = storeGuardBounds(F, m, cStore);
  if (!bound)
    return refuse("a condition gating the GEMM store is not a thread-index "
                  "bound");

  // Every dim must be deploy scale: K is, but globalM/globalN are
  // profile-scale and are rescaled by K/globalK. Leading dims come from the
  // handle (bytes to elements).
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
      // index guard: exact, already at deploy scale, and intersected with the
      // launch geometry by the host stub, min(crop, gridDim * blockDim).
      if ((*bound)[*axM] && (*bound)[*axN]) {
        gM = (unsigned)(*bound)[*axM];
        gNcols = (unsigned)(*bound)[*axN];
        note.launchCrop = true;
        note.cropMAxis = (int)*axM;
        note.cropNAxis = (int)*axN;
        recovered = true;
        llvm::errs() << "[ozaki-host-dispatch] recovered deploy geometry for "
                     << F.getName() << ": M=" << gM << " Ncols=" << gNcols
                     << " K=" << K << " from the index guard on thread axes "
                     << *axM << '/' << *axN << '\n';
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
    if (!m.globalM || !m.globalN)
      return refuse("the profile does not give the extent of the product's "
                    "rows and columns");
    gM = m.globalM;
    gNcols = m.globalN;
  }
  if (!note.launchCrop) {
    if ((*bound)[*axM] && (*bound)[*axM] < gM)
      return refuse("the GEMM store is gated by row < " +
                    Twine((*bound)[*axM]) + ", which drops rows of the " +
                    Twine(gM) + " x " + Twine(gNcols) +
                    " product the library call writes");
    if ((*bound)[*axN] && (*bound)[*axN] < gNcols)
      return refuse("the GEMM store is gated by column < " +
                    Twine((*bound)[*axN]) + ", which drops columns of the " +
                    Twine(gM) + " x " + Twine(gNcols) +
                    " product the library call writes");
  }

  // The call takes each operand from the start of its pointer parameter.
  const SCEV *cAddr = h.SE->getSCEV(cStore->getPointerOperand());
  if (!startsAtBase(*h.SE, h.aStartSCEV) ||
      !startsAtBase(*h.SE, h.bStartSCEV) || !startsAtBase(*h.SE, cAddr))
    return refuse("an operand of the product does not start where its "
                  "pointer parameter points");

  // C's layout and leading dimension from the byte step of its store address
  // along the row and column thread axes: one element along the column axis is
  // row-major, along the row axis column-major.
  const std::array<int64_t, 3> cStep = threadIndexCoefficients(cAddr);
  const unsigned axOther = 3 - *axM - *axN;
  constexpr int64_t elemBytes = 8;
  int64_t ldc = 0;
  if (cStep[axOther] == 0 && cStep[*axN] == elemBytes &&
      cStep[*axM] % elemBytes == 0 && cStep[*axM] / elemBytes >= gNcols) {
    note.cColMajor = false;
    ldc = cStep[*axM] / elemBytes;
  } else if (cStep[axOther] == 0 && cStep[*axM] == elemBytes &&
             cStep[*axN] % elemBytes == 0 && cStep[*axN] / elemBytes >= gM) {
    note.cColMajor = true;
    ldc = cStep[*axN] / elemBytes;
  } else {
    return refuse("the product's store address is not a row-major or "
                  "column-major element of C");
  }
  note.aColMajor = false;
  note.bColMajor = bColMajor;
  note.dims[DimM] = RuntimeDim::constant(gM);
  note.dims[DimNcols] = RuntimeDim::constant(gNcols);
  note.dims[DimK] = RuntimeDim::constant(K);
  note.dims[DimLda] =
      RuntimeDim::constant(h.aLeadingDimByte ? h.aLeadingDimByte / 8 : K);
  note.dims[DimLdb] =
      RuntimeDim::constant(h.bLeadingDimByte ? h.bLeadingDimByte / 8 : K);
  note.dims[DimLdc] = RuntimeDim::constant(ldc);
  if (llvm::any_of(note.dims,
                   [](const RuntimeDim &d) { return d.add > kMaxDimConstant; }))
    return refuse("a dimension of the product is out of the range of the "
                  "dispatch's int dimensions");
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

// The launch argument each caller value is, -1 for one that is not: the host
// reads the argument itself, so a pointer may differ from it only by casts.
static std::vector<int> launchArgsOf(ArrayRef<Value *> primalArgs) {
  std::vector<int> paramToArg(primalArgs.size(), -1);
  for (size_t i = 0; i < primalArgs.size(); ++i) {
    Value *v = primalArgs[i];
    if (v && v->getType()->isPointerTy())
      v = v->stripPointerCasts();
    if (auto *A = dyn_cast_or_null<Argument>(v))
      paramToArg[i] = (int)A->getArgNo();
  }
  return paramToArg;
}

// Whether `wrapper` does nothing with an effect but call its site, so that a
// library call can take the place of its whole launch.
static bool onlyCallsItsSite(const Function &wrapper) {
  unsigned effects = 0;
  for (const Instruction &I : instructions(wrapper)) {
    if (!I.mayHaveSideEffects())
      continue;
    if (const auto *II = dyn_cast<IntrinsicInst>(&I);
        II && II->isAssumeLikeIntrinsic())
      continue;
    ++effects;
  }
  return effects <= 1;
}

const CallBase *siteCallOf(const Session &S, const Function &F) {
  SmallVector<Value *, 8> callerArgs;
  const Function *kernel = nullptr;
  for (unsigned i = 0, e = F.arg_size(); i < e; ++i) {
    Value *v = siteArg(S, &F, i);
    callerArgs.push_back(v);
    if (auto *A = dyn_cast_or_null<Argument>(v))
      kernel = A->getParent();
    else if (auto *I = dyn_cast_or_null<Instruction>(v))
      kernel = I->getFunction();
  }
  if (!kernel)
    return nullptr;
  for (const Instruction &I : instructions(*kernel))
    if (const auto *CB = dyn_cast<CallBase>(&I);
        CB && llvm::all_of(callerArgs, [&](Value *v) {
          return !v || llvm::is_contained(CB->args(), v);
        }))
      return CB;
  return nullptr;
}

std::optional<GemmBodyNote> libraryCallNote(const Session &S,
                                            const AbstractMatmul &m) {
  Function *F = nullptr;
  GemmBodyNote note;
  switch (m.origin) {
  case AbstractMatmul::Origin::HostGemmLoopNest:
    F = m.hostGemm->cStore->getFunction();
    note = runtimeShapeNote(m);
    break;
  case AbstractMatmul::Origin::ScalarLoopReduction:
    F = m.scalarLoop.fma->getFunction();
    if (!computeGemmBodyNote(S.Opts, *F, m, note))
      return std::nullopt;
    break;
  }
  auto refuse = [&](const Twine &why) -> std::optional<GemmBodyNote> {
    if (S.Opts.Diagnostics.Print)
      llvm::errs() << "[ozaki-host-dispatch] " << F->getName() << ": " << why
                   << '\n';
    return std::nullopt;
  };
  // The host replaces or precedes the whole launch of the kernel with the
  // library call, reading every operand and dimension from that launch.
  const CallBase *site = siteCallOf(S, *F);
  if (!site)
    return refuse("the kernel the site is called from is not known");
  const Function *kernel = site->getFunction();
  if (!onlyCallsItsSite(*kernel))
    return refuse("its kernel " + kernel->getName() +
                  " does more than call this site, and the library call takes "
                  "the place of the whole launch");
  SmallVector<Value *, 8> callerArgs;
  for (unsigned i = 0, e = F->arg_size(); i < e; ++i)
    callerArgs.push_back(siteArg(S, F, i));
  const std::vector<int> paramToArg = launchArgsOf(callerArgs);
  auto mapped = [&](int param) {
    return param >= 0 && (size_t)param < paramToArg.size() &&
           paramToArg[param] >= 0;
  };
  SmallVector<int, 16> params = {note.cParam, note.aParam, note.bParam};
  for (const RuntimeDim &d : note.dims)
    for (const LaunchRef &f : d.factors)
      params.push_back(f.param);
  for (const auto &[x, y] : note.assumedEqual)
    params.append({x.param, y.param});
  if (!llvm::all_of(params, mapped))
    return refuse("an operand or dimension of the product is not an argument "
                  "of the kernel " +
                  kernel->getName() + ", from which the host reads it");

  return note;
}

void applyLibraryCall(Session &S, const AbstractMatmul &m, GemmBodyNote note,
                      const SchemeVariant &v) {
  assert(S.Opts.Matmul.OzakiHostDispatch &&
         "library calls are proposed only under -poseidon-ozaki-host-dispatch");
  Function *F = const_cast<Function *>(anchorOf(m).getFunction());
  note.scheme = v.scheme;
  note.numModuli = descriptorModulusField(v);
  // One descriptor per body: the host replaces or precedes its kernel's launch
  // with one library call, and the solve chooses for the body's products
  // independently.
  if (S.Matmul.GemmBodies.count(F))
    reportUserError(anchorOf(m),
                    "a second matrix product of " + F->getName() +
                        " was selected to run as a library call, and a "
                        "kernel's launch takes one; leave out "
                        "-poseidon-ozaki-host-dispatch for this body.");
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
// flush: map C/A/B and the dimensions through `paramToArg` (body parameter to
// wrapper launch argument), which libraryCallNote checked.
static void emitMappedDescriptor(Session &S, Function &wrapper,
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
  assert(cArg >= 0 && aArg >= 0 && bArg >= 0 &&
         "a library call is proposed only with its operands kernel arguments");
  // Runtime dimensions name body parameters too, and the host side only sees
  // the wrapper's launch arguments, so they go through the same mapping.
  GemmBodyNote mapped = n;
  for (RuntimeDim &x : mapped.dims)
    for (LaunchRef &f : x.factors)
      f.param = mapParam(f.param);
  for (auto &[x, y] : mapped.assumedEqual) {
    x.param = mapParam(x.param);
    y.param = mapParam(y.param);
  }
  assert(onlyCallsItsSite(wrapper) &&
         "a library call is proposed only for a kernel that does nothing else");
  emitDescriptor(S, wrapper.getName(), cArg, aArg, bArg, mapped);
}

void writeGemmDescriptor(Session &S, Function &wrapper,
                         ArrayRef<Value *> primalArgs, const GemmBodyNote &n) {
  if (!n.valid)
    return;
  emitMappedDescriptor(S, wrapper, n, launchArgsOf(primalArgs));
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
      emitMappedDescriptor(S, *body->getParent()->getFunction(pg.wrapperName),
                           n->second, pg.paramToArg);
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
static const char *dispatchSym(Scheme s) {
  switch (s) {
  case Scheme::OzakiII:
    return "__poseidon_ozaki_dgemm";
  case Scheme::Tcec:
    return "__poseidon_tcec_dgemm";
  case Scheme::Direct:
    return "__poseidon_direct_dgemm";
  case Scheme::Native:
    return "__poseidon_native_dgemm";
  }
  llvm_unreachable("unknown Scheme");
}

// void(C,A,B, M,Ncols,K, lda,ldb,ldc, aColMajor,bColMajor,cColMajor,
//      alpha,beta, stream[, the descriptor's num_moduli field])
static FunctionCallee getDispatchCallee(Module &M, Scheme scheme) {
  LLVMContext &C = M.getContext();
  Type *ptr = PointerType::get(C, 0);
  Type *i32 = Type::getInt32Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  SmallVector<Type *, 16> params = {ptr, ptr, ptr, i32, i32, i32, i32, i32,
                                    i32, i32, i32, i32, dbl, dbl, ptr};
  if (scheme != Scheme::Native)
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
        if (scheme > (unsigned)Scheme::Native)
          Cursor.fail("its scheme " + Twine(scheme) + " is above " +
                      Twine((unsigned)Scheme::Native));
        d.scheme = (Scheme)scheme;
        // Keyword blocks (see emitDescriptor), of which rd is required.
        bool haveDims = false;
        while (!Cursor.atEnd()) {
          if (Cursor.consume("rd")) {
            for (RuntimeDim &x : d.dims) {
              unsigned nf = Cursor.integer<unsigned>("factor count");
              if (nf > kMaxDimFactors)
                Cursor.fail("its factor count " + Twine(nf) + " is above " +
                            Twine(kMaxDimFactors));
              for (unsigned f = 0; f < nf && Cursor.ok(); ++f) {
                LaunchRef r;
                r.param = Cursor.integer<int>("factor argument");
                r.offset = Cursor.integer<int64_t>("factor offset");
                x.factors.push_back(r);
              }
              x.mul = Cursor.integer<int64_t>("dimension factor");
              x.add = Cursor.integer<int64_t>("dimension term");
              for (int64_t c : {x.mul, x.add})
                if (c < -kMaxDimConstant || c > kMaxDimConstant)
                  Cursor.fail("its dimension constant " + Twine(c) +
                              " is out of the int range");
              if (!x.isConst() && !d.standalone)
                Cursor.fail("it is fused and has a runtime dimension");
            }
            d.beta = Cursor.real("beta");
            haveDims = true;
          } else if (Cursor.consume("pf")) {
            d.cOffset = Cursor.integer<int64_t>("C offset");
            d.aOffset = Cursor.integer<int64_t>("A offset");
            d.bOffset = Cursor.integer<int64_t>("B offset");
          } else if (Cursor.consume("eq")) {
            unsigned ne = Cursor.integer<unsigned>("equality count");
            if (ne > kMaxAssumedEqual)
              Cursor.fail("its equality count " + Twine(ne) + " is above " +
                          Twine(kMaxAssumedEqual));
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
  Type *i64 = Type::getInt64Ty(C);
  Type *dbl = Type::getDoubleTy(C);
  auto I32 = [&](int64_t v) { return ConstantInt::get(i32, (uint64_t)v); };
  auto I64 = [&](int64_t v) { return ConstantInt::get(i64, (uint64_t)v); };
  Value *argArray = launchArguments(*launch);
  // Resolved per descriptor: the scheme is a per-site solver decision.
  FunctionCallee callee = getDispatchCallee(M, d.scheme);

  IRBuilder<> B(launch);
  auto value = [&](const LaunchRef &r, Type *T) {
    return loadLaunchValue(B, argArray, r, T);
  };
  // Whether every runtime dimension fits the int the dispatch takes it as.
  // The dimensions are evaluated in i64 and every intermediate value is
  // checked: while all checks pass, each operand is below 2^31 and nothing
  // wraps.
  Value *inRange = nullptr;
  auto checked = [&](Value *v) {
    Value *fits = B.CreateICmpULE(v, I64(kMaxDimConstant));
    inRange = inRange ? B.CreateAnd(inRange, fits) : fits;
    return v;
  };
  auto dim = [&](const RuntimeDim &r) -> Value * {
    if (r.isConst())
      return I32(r.add);
    Value *v = nullptr;
    for (const LaunchRef &f : r.factors) {
      Value *x = checked(B.CreateSExt(value(f, i32), i64));
      v = v ? checked(B.CreateMul(v, x)) : x;
    }
    if (r.mul != 1)
      v = checked(B.CreateMul(v, I64(r.mul)));
    if (r.add)
      v = checked(B.CreateAdd(v, I64(r.add)));
    return B.CreateTrunc(v, i32);
  };
  auto cropDim = [&](int axis, const RuntimeDim &crop) -> Value * {
    Value *g = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedGrid(*pop), axis));
    Value *b = B.CreateLoad(
        i32, B.CreateConstInBoundsGEP1_32(i32, poppedBlock(*pop), axis));
    Value *g64 = B.CreateZExt(g, i64);
    Value *b64 = B.CreateZExt(b, i64);
    Value *ext = B.CreateMul(g64, b64);
    Value *below = B.CreateICmpULT(ext, I64(crop.add));
    return B.CreateTrunc(B.CreateSelect(below, ext, I64(crop.add)), i32);
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
  if (d.scheme != Scheme::Native)
    args.push_back(I32(d.numModuli));
  Value *holds = nullptr;
  for (const auto &[x, y] : d.assumedEqual) {
    Value *lhs = value(x, i32);
    Value *rhs = value(y, i32);
    Value *eq = B.CreateICmpEQ(lhs, rhs);
    holds = holds ? B.CreateAnd(holds, eq) : eq;
  }
  assert((d.standalone || !inRange) &&
         "the reader admits no runtime dimension in a fused descriptor");
  if (inRange)
    holds = holds ? B.CreateAnd(holds, inRange) : inRange;

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
