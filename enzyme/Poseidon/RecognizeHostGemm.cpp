// Runtime-dimension dense GEMM recognition (Origin::HostGemmLoopNest).
//
// findScalarLoopMatmuls recognizes ONE innermost reduction loop whose trip
// count is a compile-time constant, whose leading dimensions are compile-time
// byte strides, and whose row/column indices are bare thread axes. A production
// finite-element apply is none of those: MFEM's elasticity partial-assembly
// reduce computes
//
//     y(i,q,e) += sum_{m<d} sum_{p<numPoints} Q(p,m,q,e) * G(p,m,i)
//
// which IS the dense product C = A^T B + C with M = nDofs, K = d*numPoints,
// Ncols = d*NE, all three RUNTIME values, contracted over a two-level (m,p)
// nest of which only the outer level is a compile-time constant (and is
// therefore unrolled away before Poseidon runs), with the row index behind a
// strided MFEM_FOREACH_THREAD loop and the column index threadIdx.x fused with
// the CTA index.
//
// The dimensions are therefore EXPRESSIONS, not integers. Showing that the
// (m,p) nest flattens needs (3i+1)*numPoints - 3i*numPoints == numPoints, and
// SCEV does not distribute a non-constant multiplier over a sum, so
// getMinusSCEV of those two never folds. Integer polynomials over SCEV atoms
// (RgPoly below) make the comparison exact and total, and every dimension is
// either derived or the whole site is refused.
#include "Flags.h"
#include "MatmulInternal.h"
#include "ProfileRead.h"
#include "Utils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;

namespace poseidon {

std::string runtimeDimString(const RuntimeDim &d) {
  if (d.isConst())
    return std::to_string(d.add);
  std::string s;
  if (d.mul != 1)
    s += std::to_string(d.mul) + "*";
  s += "arg" + std::to_string(d.param);
  if (d.add)
    s += (d.add > 0 ? "+" : "") + std::to_string(d.add);
  return s;
}

namespace {

// ---------------------------------------------------------------------------
// Integer polynomials over SCEV atoms.
// ---------------------------------------------------------------------------
using RgMono = std::vector<unsigned>; // sorted atom ids; empty == constant term

struct RgPoly {
  std::map<RgMono, int64_t> t;

  static RgPoly constant(int64_t c) {
    RgPoly p;
    if (c)
      p.t[RgMono{}] = c;
    return p;
  }
  static RgPoly atom(unsigned id) {
    RgPoly p;
    p.t[RgMono{id}] = 1;
    return p;
  }
  bool isZero() const { return t.empty(); }
  bool isConstant(int64_t *c = nullptr) const {
    if (t.empty()) {
      if (c)
        *c = 0;
      return true;
    }
    if (t.size() != 1 || !t.begin()->first.empty())
      return false;
    if (c)
      *c = t.begin()->second;
    return true;
  }
  bool contains(unsigned atom) const {
    for (auto &kv : t)
      if (llvm::is_contained(kv.first, atom))
        return true;
    return false;
  }
  void addTerm(const RgMono &m, int64_t c) {
    if (!c)
      return;
    auto it = t.find(m);
    if (it == t.end()) {
      t[m] = c;
      return;
    }
    it->second += c;
    if (it->second == 0)
      t.erase(it);
  }
  bool operator==(const RgPoly &o) const { return t == o.t; }
};

static RgPoly rgAdd(const RgPoly &a, const RgPoly &b) {
  RgPoly r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, kv.second);
  return r;
}
static RgPoly rgSub(const RgPoly &a, const RgPoly &b) {
  RgPoly r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, -kv.second);
  return r;
}
static RgPoly rgMul(const RgPoly &a, const RgPoly &b) {
  RgPoly r;
  for (auto &x : a.t)
    for (auto &y : b.t) {
      RgMono m = x.first;
      m.insert(m.end(), y.first.begin(), y.first.end());
      llvm::sort(m);
      r.addTerm(m, x.second * y.second);
    }
  return r;
}
// Exact division by a nonzero integer. A non-integral result is a recognition
// failure (a fractional leading dimension), never a rounded guess.
static bool rgDivExact(const RgPoly &a, int64_t d, RgPoly &out) {
  if (d == 0)
    return false;
  out = RgPoly{};
  for (auto &kv : a.t) {
    if (kv.second % d)
      return false;
    out.addTerm(kv.first, kv.second / d);
  }
  return true;
}
// Divide out one occurrence of `atom` from every monomial that contains it;
// the quotient is returned in `q` and the untouched terms in `rest`. Refuses a
// monomial carrying the atom to a power > 1 (a quadratic index is not a matrix
// subscript).
static bool rgDivideByAtom(const RgPoly &p, unsigned atom, RgPoly &q,
                           RgPoly &rest) {
  q = RgPoly{};
  rest = RgPoly{};
  for (auto &kv : p.t) {
    unsigned n = 0;
    for (unsigned a : kv.first)
      n += (a == atom);
    if (n == 0) {
      rest.addTerm(kv.first, kv.second);
      continue;
    }
    if (n > 1)
      return false;
    RgMono m;
    bool dropped = false;
    for (unsigned a : kv.first) {
      if (a == atom && !dropped) {
        dropped = true;
        continue;
      }
      m.push_back(a);
    }
    q.addTerm(m, kv.second);
  }
  return true;
}

// ---------------------------------------------------------------------------
// Atom table and SCEV -> polynomial expansion.
//
// Every AddRec is expanded as start + step * j<L>, with j<L> a synthetic atom
// standing for the loop's iteration counter. That is what makes a strided
// MFEM_FOREACH_THREAD index tractable: its contribution to an address becomes
// coef*(tid.k + ntid.k*j), and the two coefficients can be checked against
// each other instead of assuming the loop runs once.
// ---------------------------------------------------------------------------
struct RgAtoms {
  SmallVector<const SCEV *, 16> byId; // null for synthetic loop-counter atoms
  DenseMap<const SCEV *, unsigned> idOf;
  DenseMap<const Loop *, unsigned> counterOf;
  SmallVector<std::string, 16> name;

  unsigned id(const SCEV *S) {
    auto it = idOf.find(S);
    if (it != idOf.end())
      return it->second;
    unsigned n = byId.size();
    byId.push_back(S);
    idOf[S] = n;
    std::string s;
    raw_string_ostream os(s);
    S->print(os);
    name.push_back(os.str());
    return n;
  }
  unsigned counter(const Loop *L) {
    auto it = counterOf.find(L);
    if (it != counterOf.end())
      return it->second;
    unsigned n = byId.size();
    byId.push_back(nullptr);
    counterOf[L] = n;
    name.push_back(("j<" + L->getHeader()->getName() + ">").str());
    return n;
  }
  std::string str(const RgPoly &p) const {
    if (p.t.empty())
      return "0";
    std::string s;
    bool first = true;
    for (auto &kv : p.t) {
      if (!first)
        s += " + ";
      first = false;
      s += std::to_string(kv.second);
      for (unsigned a : kv.first)
        s += "*" + (a < name.size() ? name[a] : std::string("?"));
    }
    return s;
  }
};

struct RgExpand {
  ScalarEvolution &SE;
  RgAtoms &A;
  std::string err;

  bool expand(const SCEV *S, RgPoly &out) {
    if (auto *C = dyn_cast<SCEVConstant>(S)) {
      out = RgPoly::constant(C->getAPInt().getSExtValue());
      return true;
    }
    if (auto *Cast = dyn_cast<SCEVCastExpr>(S)) {
      // zext / sext / trunc are transparent. Each one here comes from index
      // arithmetic LLVM proved non-wrapping (the nsw/nneg flags on the source
      // adds and muls), so it is value preserving; refusing them would refuse
      // every 32-bit-indexed CUDA kernel there is.
      return expand(Cast->getOperand(), out);
    }
    if (auto *Add = dyn_cast<SCEVAddExpr>(S)) {
      out = RgPoly{};
      for (const SCEV *Op : Add->operands()) {
        RgPoly p;
        if (!expand(Op, p))
          return false;
        out = rgAdd(out, p);
      }
      return true;
    }
    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
      out = RgPoly::constant(1);
      for (const SCEV *Op : Mul->operands()) {
        RgPoly p;
        if (!expand(Op, p))
          return false;
        out = rgMul(out, p);
      }
      return true;
    }
    if (auto *Rec = dyn_cast<SCEVAddRecExpr>(S)) {
      if (!Rec->isAffine()) {
        err = "non-affine add-recurrence";
        return false;
      }
      RgPoly start;
      if (!expand(Rec->getStart(), start))
        return false;
      RgPoly step;
      if (!expand(Rec->getStepRecurrence(SE), step))
        return false;
      out = rgAdd(start, rgMul(step, RgPoly::atom(A.counter(Rec->getLoop()))));
      return true;
    }
    if (isa<SCEVUnknown>(S)) {
      out = RgPoly::atom(A.id(S));
      return true;
    }
    err = "unsupported SCEV node (udiv / min / max)";
    return false;
  }
};

// A GEMM dimension is usable only if it is affine in at most ONE parameter of
// the optimized body: mul*param + add. That is exactly what the host-side stub
// rewrite can evaluate from the launch arguments, and refusing anything wider
// is what keeps the emitted call honest rather than approximately right.
static bool rgToRuntimeDim(const RgPoly &p, const RgAtoms &A, RuntimeDim &out,
                           std::string &why) {
  out = RuntimeDim{};
  bool haveArg = false;
  for (auto &kv : p.t) {
    if (kv.first.empty()) {
      out.add = kv.second;
      continue;
    }
    if (kv.first.size() != 1) {
      why = "value is a product of two or more runtime terms";
      return false;
    }
    if (haveArg) {
      why = "value depends on more than one runtime term";
      return false;
    }
    const SCEV *S = A.byId[kv.first[0]];
    auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
    auto *Arg = U ? dyn_cast<Argument>(U->getValue()) : nullptr;
    if (!Arg) {
      why = "value depends on a term that is not a parameter of the optimized "
            "body";
      return false;
    }
    out.param = (int)Arg->getArgNo();
    out.mul = kv.second;
    haveArg = true;
  }
  return true;
}

// ---------------------------------------------------------------------------
// fma matching. A deliberate second copy of PoseidonRaiseWMMA's predicate: the
// two recognizers are parallel arms and this one must stay free to move
// without perturbing the other.
// ---------------------------------------------------------------------------
static bool rgMatchFMA(Value *V, Instruction *&fma, Instruction *&fmul,
                       Value *&a, Value *&b, Value *&acc) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    Intrinsic::ID id = II->getIntrinsicID();
    if (id == Intrinsic::fma || id == Intrinsic::fmuladd) {
      fma = II;
      fmul = nullptr;
      a = II->getArgOperand(0);
      b = II->getArgOperand(1);
      acc = II->getArgOperand(2);
      return true;
    }
  }
  auto *fadd = dyn_cast<BinaryOperator>(I);
  if (!fadd || fadd->getOpcode() != Instruction::FAdd)
    return false;
  if (!fadd->getFastMathFlags().allowContract())
    return false;
  for (unsigned side = 0; side < 2; ++side) {
    auto *fmulI = dyn_cast<BinaryOperator>(fadd->getOperand(side));
    if (!fmulI || fmulI->getOpcode() != Instruction::FMul)
      continue;
    if (!fmulI->getFastMathFlags().allowContract())
      continue;
    fma = fadd;
    fmul = fmulI;
    a = fmulI->getOperand(0);
    b = fmulI->getOperand(1);
    acc = fadd->getOperand(1 - side);
    return true;
  }
  return false;
}

static int rgTidAxis(const Value *v) {
  auto *II = dyn_cast<IntrinsicInst>(v);
  if (!II)
    return -1;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    return 0;
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    return 1;
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
    return 2;
  default:
    return -1;
  }
}
static int rgNtidAxis(const Value *v) {
  auto *II = dyn_cast<IntrinsicInst>(v);
  if (!II)
    return -1;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
    return 0;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
    return 1;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
    return 2;
  default:
    return -1;
  }
}
static int rgCtaidAxis(const Value *v) {
  auto *II = dyn_cast<IntrinsicInst>(v);
  if (!II)
    return -1;
  switch (II->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    return 0;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    return 1;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
    return 2;
  default:
    return -1;
  }
}
static bool rgMentionsLaunchReg(const SCEV *S) {
  return SCEVExprContains(S, [](const SCEV *X) {
    auto *U = dyn_cast<SCEVUnknown>(X);
    if (!U)
      return false;
    Value *v = U->getValue();
    return rgTidAxis(v) >= 0 || rgCtaidAxis(v) >= 0;
  });
}

// One recognized index source: a thread axis (optionally behind a strided
// MFEM_FOREACH_THREAD loop) or the CuKernel CTA linearization.
struct RgIndex {
  RgPoly form;            // the index's symbolic value
  unsigned sigAtom = ~0u; // atom appearing in `form` and in no other index
  // The rest of the monomial the signature atom sits in (coefficient 1). For a
  // thread index this is empty; for the CuKernel CTA linearization
  // ctaid.a*ntid.b + tid.b it is {ntid.b}.
  SmallVector<unsigned, 2> sigMono;
  SmallVector<unsigned, 4> ownAtoms; // atoms that must not survive extraction
  RgPoly extent;           // number of distinct values the index takes
  unsigned profExtent = 0; // profile-scale extent, 0 when not fixed by geometry
  std::string name;
};

// Peel the index `idx` out of an address polynomial:
//     p == coef * idx.form + rest,  with rest free of every atom of `idx`.
// Returns false when p is not of that form, which is the honest signal that
// the address is not a matrix subscript in this index.
static bool rgSplitIndex(const RgPoly &p, const RgIndex &idx, RgPoly &coef,
                         RgPoly &rest) {
  RgPoly q, r;
  if (!rgDivideByAtom(p, idx.sigAtom, q, r))
    return false;
  // Peel off the rest of the signature monomial (the ntid factor of a CTA
  // linearization); anything left over means the address does not carry the
  // index in the shape the index actually has.
  for (unsigned a : idx.sigMono) {
    RgPoly q2, r2;
    if (!rgDivideByAtom(q, a, q2, r2) || !r2.isZero())
      return false;
    q = q2;
  }
  coef = q;
  rest = rgSub(p, rgMul(coef, idx.form));
  for (unsigned a : idx.ownAtoms)
    if (rest.contains(a))
      return false;
  return true;
}

// One member of the contraction chain. The operand addresses are kept as the
// FULL SCEVs: a two-level index like G(p,m,i) reaches Poseidon as
// sext(i32 {{...}<i-loop>,+,1}<p-loop>), i.e. an integer add-recurrence under a
// cast, which is not a pointer add-recurrence of the reduction loop at all.
// The reduction byte step is recovered from the polynomial expansion instead
// (the coefficient of the reduction loop's iteration counter), which is exact
// for every cast/trunc shape the front end produces.
struct RgMember {
  Loop *L = nullptr;
  PHINode *phi = nullptr;
  Instruction *fma = nullptr;
  Instruction *fmul = nullptr;
  LoadInst *aLoad = nullptr;
  LoadInst *bLoad = nullptr;
  const SCEV *aAddr = nullptr, *bAddr = nullptr;
  int64_t aStep = 0, bStep = 0; // filled by rgAnalyzeChain
  const SCEV *trip = nullptr;
  Value *init = nullptr;
};

// Can `target` reach `v` through phis and casts only?
static bool rgReaches(Value *v, Value *target, unsigned depth = 0) {
  if (!v || depth > 12)
    return false;
  if (v == target)
    return true;
  if (auto *P = dyn_cast<PHINode>(v)) {
    for (Value *in : P->incoming_values())
      if (in != P && rgReaches(in, target, depth + 1))
        return true;
    return false;
  }
  if (auto *C = dyn_cast<CastInst>(v))
    return rgReaches(C->getOperand(0), target, depth + 1);
  return false;
}

static bool rgIsZeroSeed(Value *v, unsigned depth = 0) {
  if (!v || depth > 12)
    return false;
  if (auto *C = dyn_cast<ConstantFP>(v))
    return C->isZero() && !C->isNegative();
  if (auto *P = dyn_cast<PHINode>(v)) {
    for (Value *in : P->incoming_values())
      if (in != P && !rgIsZeroSeed(in, depth + 1))
        return false;
    return true;
  }
  return false;
}

// Walk GEP / cast chains back to a Function argument index (-1 if none).
static int rgArgIndex(const Value *v) {
  while (v) {
    if (auto *A = dyn_cast<Argument>(v))
      return (int)A->getArgNo();
    if (auto *G = dyn_cast<GEPOperator>(v)) {
      v = G->getPointerOperand();
      continue;
    }
    if (auto *O = dyn_cast<Operator>(v)) {
      if (O->getOpcode() == Instruction::BitCast ||
          O->getOpcode() == Instruction::AddrSpaceCast) {
        v = O->getOperand(0);
        continue;
      }
    }
    break;
  }
  return -1;
}

} // namespace

namespace {

static const SCEV *rgStripSCEVCasts(const SCEV *S) {
  while (auto *C = dyn_cast<SCEVCastExpr>(S))
    S = C->getOperand();
  return S;
}

// MFEM_FOREACH_THREAD(i,k,N) == for (int i = threadIdx.k; i < N; i +=
// blockDim.k) Returns the index value, its upper bound, and the thread axis.
static bool rgMatchStridedThreadLoop(Loop *L, ScalarEvolution &SE,
                                     Value *&idxVal, Value *&bound, int &axis) {
  BasicBlock *latch = L->getLoopLatch();
  if (!latch)
    return false;
  Instruction *br = latch->getTerminator();
  if (!isConditionalBranch(br) || br->getSuccessor(0) != L->getHeader())
    return false;
  auto *cmp = dyn_cast<ICmpInst>(branchCondition(br));
  if (!cmp)
    return false;
  auto pred = cmp->getPredicate();
  if (pred != ICmpInst::ICMP_SLT && pred != ICmpInst::ICMP_ULT)
    return false;
  auto *nextIdx = dyn_cast<BinaryOperator>(cmp->getOperand(0));
  if (!nextIdx || nextIdx->getOpcode() != Instruction::Add)
    return false;
  int a0 = rgNtidAxis(nextIdx->getOperand(0));
  int a1 = rgNtidAxis(nextIdx->getOperand(1));
  if ((a0 < 0) == (a1 < 0))
    return false; // need exactly one blockDim operand
  axis = a0 >= 0 ? a0 : a1;
  idxVal = a0 >= 0 ? nextIdx->getOperand(1) : nextIdx->getOperand(0);
  bound = cmp->getOperand(1);

  const SCEV *S = rgStripSCEVCasts(SE.getSCEV(idxVal));
  auto *Rec = dyn_cast<SCEVAddRecExpr>(S);
  if (!Rec || Rec->getLoop() != L || !Rec->isAffine())
    return false;
  auto *st = dyn_cast<SCEVUnknown>(rgStripSCEVCasts(Rec->getStart()));
  auto *sp =
      dyn_cast<SCEVUnknown>(rgStripSCEVCasts(Rec->getStepRecurrence(SE)));
  if (!st || !sp)
    return false;
  if (rgTidAxis(st->getValue()) != axis || rgNtidAxis(sp->getValue()) != axis)
    return false;
  return true;
}

// Fill in an RgIndex from its symbolic form: pick the signature atom (a CTA id
// if present, else a thread id), record the monomial it is multiplied by, and
// list the atoms that must not survive its extraction from an address.
static bool rgFinishIndex(RgIndex &idx, const RgAtoms &A) {
  int best = -1;
  bool bestIsCta = false;
  for (auto &kv : idx.form.t)
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
      if (!U)
        continue;
      bool isCta = rgCtaidAxis(U->getValue()) >= 0;
      bool isTid = rgTidAxis(U->getValue()) >= 0;
      if (!isCta && !isTid)
        continue;
      if (best < 0 || (isCta && !bestIsCta)) {
        best = (int)a;
        bestIsCta = isCta;
      }
    }
  if (best < 0)
    return false;
  idx.sigAtom = (unsigned)best;
  // The signature atom must appear in exactly one monomial, with coefficient 1.
  const RgMono *sigMonoFull = nullptr;
  for (auto &kv : idx.form.t) {
    if (!llvm::is_contained(kv.first, idx.sigAtom))
      continue;
    if (sigMonoFull || kv.second != 1)
      return false;
    sigMonoFull = &kv.first;
  }
  if (!sigMonoFull)
    return false;
  idx.sigMono.clear();
  bool dropped = false;
  for (unsigned a : *sigMonoFull) {
    if (a == idx.sigAtom && !dropped) {
      dropped = true;
      continue;
    }
    idx.sigMono.push_back(a);
  }
  idx.ownAtoms.clear();
  for (auto &kv : idx.form.t)
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      if (!S) { // synthetic loop counter
        idx.ownAtoms.push_back(a);
        continue;
      }
      auto *U = dyn_cast<SCEVUnknown>(S);
      if (U &&
          (rgTidAxis(U->getValue()) >= 0 || rgCtaidAxis(U->getValue()) >= 0))
        idx.ownAtoms.push_back(a);
    }
  llvm::sort(idx.ownAtoms);
  idx.ownAtoms.erase(std::unique(idx.ownAtoms.begin(), idx.ownAtoms.end()),
                     idx.ownAtoms.end());
  return true;
}

// Largest value the index can take at PROFILE scale, from the launch geometry
// in the profile header. Returns false when the form involves a quantity the
// header does not fix (a loop counter, a runtime bound).
static bool rgProfileExtent(const RgIndex &idx, const RgAtoms &A,
                            const FunctionProfileHeader &hdr, unsigned &out) {
  int64_t c = 0;
  if (idx.extent.isConstant(&c) && c > 0) {
    out = (unsigned)c; // compile-time bound: exact at every scale
    return true;
  }
  int64_t maxVal = 0;
  for (auto &kv : idx.form.t) {
    int64_t term = kv.second;
    if (term < 0)
      return false;
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
      if (!U)
        return false; // loop counter or opaque runtime term
      Value *v = U->getValue();
      int ax;
      if ((ax = rgCtaidAxis(v)) >= 0) {
        if (!hdr.maxGridDim[ax])
          return false;
        term *= (int64_t)hdr.maxGridDim[ax] - 1;
      } else if ((ax = rgTidAxis(v)) >= 0) {
        if (!hdr.maxBlockDim[ax])
          return false;
        term *= (int64_t)hdr.maxBlockDim[ax] - 1;
      } else if ((ax = rgNtidAxis(v)) >= 0) {
        if (!hdr.maxBlockDim[ax])
          return false;
        term *= (int64_t)hdr.maxBlockDim[ax];
      } else {
        return false;
      }
    }
    maxVal += term;
  }
  if (maxVal < 0)
    return false;
  out = (unsigned)(maxVal + 1);
  return true;
}

// Every fused-index reconstruction Poseidon can express: one index, or an
// ordered pair (fast, slow) with stride(slow) == stride(fast) * extent(fast).
// Anything wider is refused rather than guessed.
struct RgFusedIndex {
  SmallVector<unsigned, 2> order; // positions into the caller's index list,
                                  // fastest first
  RgPoly unitStride;              // byte stride of one step of the fused index
  RgPoly extent;                  // total extent
};

} // namespace

// Reconstruct one contraction chain into a RuntimeGemmHandle. Returns false
// with `why` set on any non-match; the caller prints it under -poseidon-print.
// Nothing is defaulted: every field below is either derived or the whole site
// is refused.
static bool rgAnalyzeChain(Function &F, ScalarEvolution &SE, LoopInfo &LI,
                           DominatorTree &DT, const FunctionProfileHeader &hdr,
                           const std::unordered_map<size_t, ProfileInfo> &prof,
                           ArrayRef<RgMember> chain, RuntimeGemmHandle &H,
                           std::string &why) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  const unsigned R = chain.size();
  Type *eltTy = chain[0].aLoad->getType();
  const int64_t elemSize = (int64_t)DL.getTypeAllocSize(eltTy).getFixedValue();

  // --- uniform member shape -------------------------------------------------
  for (const RgMember &m : chain) {
    if (m.aLoad->getType() != eltTy || m.bLoad->getType() != eltTy) {
      why = "chain members disagree on element type";
      return false;
    }
  }

  // --- index sources --------------------------------------------------------
  RgAtoms atoms;
  RgExpand EX{SE, atoms, {}};
  SmallVector<RgIndex, 4> indices;
  for (Loop *L = chain[0].L->getParentLoop(); L; L = L->getParentLoop()) {
    Value *idxVal = nullptr, *bound = nullptr;
    int axis = -1;
    if (!rgMatchStridedThreadLoop(L, SE, idxVal, bound, axis))
      continue;
    RgIndex idx;
    if (!EX.expand(SE.getSCEV(idxVal), idx.form) ||
        !EX.expand(SE.getSCEV(bound), idx.extent)) {
      why = "cannot expand a strided-thread-loop index or its bound (" +
            EX.err + ")";
      return false;
    }
    if (!rgFinishIndex(idx, atoms)) {
      why = "strided-thread-loop index has no usable signature term";
      return false;
    }
    idx.name = ("tid." + Twine("xyz"[axis]) + "-strided").str();
    indices.push_back(idx);
  }
  BasicBlock *anchorBB = chain.back().fma->getParent();
  for (BasicBlock &BB : F) {
    Instruction *br = BB.getTerminator();
    if (!isConditionalBranch(br))
      continue;
    auto *cmp = dyn_cast<ICmpInst>(branchCondition(br));
    if (!cmp)
      continue;
    auto pred = cmp->getPredicate();
    bool trueIsLess = pred == ICmpInst::ICMP_SLT || pred == ICmpInst::ICMP_ULT;
    bool falseIsLess = pred == ICmpInst::ICMP_SGE || pred == ICmpInst::ICMP_UGE;
    if (!trueIsLess && !falseIsLess)
      continue;
    BasicBlock *succ = br->getSuccessor(trueIsLess ? 0 : 1);
    if (succ == &BB || !DT.dominates(BasicBlockEdge(&BB, succ), anchorBB))
      continue;
    const SCEV *SX = SE.getSCEV(cmp->getOperand(0));
    if (!rgMentionsLaunchReg(SX))
      continue;
    RgIndex idx;
    if (!EX.expand(SX, idx.form))
      continue; // an unexpandable guard is simply not an index source
    bool dup = false;
    for (const RgIndex &prev : indices)
      if (prev.form == idx.form || idx.form == RgPoly::atom(prev.sigAtom))
        dup = true;
    if (dup)
      continue; // already covered by the strided-loop form it seeds
    if (!EX.expand(SE.getSCEV(cmp->getOperand(1)), idx.extent))
      continue;
    if (!rgFinishIndex(idx, atoms))
      continue;
    idx.name = "guarded";
    indices.push_back(idx);
  }
  if (indices.empty()) {
    why = "no thread/CTA index source dominates the contraction";
    return false;
  }
  // Signature atoms must be distinct, or peeling one index out of an address
  // would consume another's contribution.
  for (unsigned i = 0; i < indices.size(); ++i)
    for (unsigned j = i + 1; j < indices.size(); ++j)
      if (indices[i].sigAtom == indices[j].sigAtom) {
        why = "two index sources share a signature term";
        return false;
      }

  // --- operand address polynomials -----------------------------------------
  auto splitAll = [&](const RgPoly &p, SmallVectorImpl<RgPoly> &coefs,
                      RgPoly &base) -> bool {
    coefs.assign(indices.size(), RgPoly{});
    RgPoly cur = p;
    for (unsigned i = 0; i < indices.size(); ++i) {
      RgPoly c, rest;
      if (!rgSplitIndex(cur, indices[i], c, rest))
        return false;
      coefs[i] = c;
      cur = rest;
    }
    base = cur;
    return true;
  };
  auto isBasePointer = [&](const RgPoly &b, int &argIdx, Value *&val) -> bool {
    if (b.t.size() != 1)
      return false;
    auto &kv = *b.t.begin();
    if (kv.first.size() != 1 || kv.second != 1)
      return false;
    const SCEV *S = atoms.byId[kv.first[0]];
    auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
    if (!U || !U->getType()->isPointerTy())
      return false;
    val = U->getValue();
    argIdx = rgArgIndex(val);
    return argIdx >= 0;
  };

  SmallVector<RgPoly, 8> op0Poly(R), op1Poly(R);
  SmallVector<int64_t, 8> step0(R), step1(R);
  SmallVector<unsigned, 8> redCounter(R);
  for (unsigned r = 0; r < R; ++r)
    redCounter[r] = atoms.counter(chain[r].L);
  for (unsigned r = 0; r < R; ++r) {
    RgPoly full0, full1;
    if (!EX.expand(chain[r].aAddr, full0) ||
        !EX.expand(chain[r].bAddr, full1)) {
      why = "cannot expand an operand address (" + EX.err + ")";
      return false;
    }
    // The reduction byte step is the coefficient of this loop's iteration
    // counter; the remainder is the address at reduction index 0.
    RgPoly q0, q1;
    if (!rgDivideByAtom(full0, redCounter[r], q0, op0Poly[r]) ||
        !rgDivideByAtom(full1, redCounter[r], q1, op1Poly[r])) {
      why = "an operand address is not affine in the reduction index";
      return false;
    }
    if (!q0.isConstant(&step0[r]) || !q1.isConstant(&step1[r]) || !step0[r] ||
        !step1[r]) {
      why = "an operand's reduction-axis byte stride is zero or not a "
            "compile-time constant";
      return false;
    }
    if (step0[r] != step0[0] || step1[r] != step1[0]) {
      why = "chain members disagree on the reduction-axis byte stride";
      return false;
    }
    for (unsigned q = 0; q < R; ++q)
      if (q != r && (op0Poly[r].contains(redCounter[q]) ||
                     op1Poly[r].contains(redCounter[q]))) {
        why = "an operand address mixes two members' reduction indices";
        return false;
      }
  }

  RgPoly tripPoly;
  if (!EX.expand(chain[0].trip, tripPoly)) {
    why = "cannot expand the reduction trip count (" + EX.err + ")";
    return false;
  }
  for (unsigned r = 1; r < R; ++r) {
    RgPoly t;
    if (!EX.expand(chain[r].trip, t) || !(t == tripPoly)) {
      why = "chain members have different reduction lengths";
      return false;
    }
  }

  // --- flattening: member r must start exactly r*trip elements along ---------
  for (unsigned r = 1; r < R; ++r) {
    RgPoly wantA = rgMul(RgPoly::constant((int64_t)r * step0[0]), tripPoly);
    RgPoly wantB = rgMul(RgPoly::constant((int64_t)r * step1[0]), tripPoly);
    if (!(rgSub(op0Poly[r], op0Poly[0]) == wantA) ||
        !(rgSub(op1Poly[r], op1Poly[0]) == wantB)) {
      why = "chain member " + std::to_string(r) +
            " is not contiguous with member 0 along the reduction axis, so the "
            "two contraction levels do not flatten into one index";
      return false;
    }
  }

  // --- role assignment: row operand vs column operand -----------------------
  SmallVector<RgPoly, 4> c0, c1, cC;
  RgPoly b0, b1, bC;
  if (!splitAll(op0Poly[0], c0, b0) || !splitAll(op1Poly[0], c1, b1)) {
    why = "an operand address is not affine in the recognized index sources";
    return false;
  }
  if (!chain.back().fma)
    return false;

  // Consumer: the (accumulating) store of the final accumulator.
  Instruction *last = chain.back().fma;
  StoreInst *cStore = nullptr;
  Instruction *epilogue = nullptr;
  {
    // The chain's own phis/fmas are internal recurrences, not consumers.
    SmallPtrSet<const Value *, 24> internal;
    for (const RgMember &m : chain) {
      internal.insert(m.fma);
      internal.insert(m.phi);
      if (m.fmul)
        internal.insert(m.fmul);
    }
    SmallVector<Value *, 8> wl{last};
    SmallPtrSet<Value *, 16> seen;
    while (!wl.empty()) {
      Value *v = wl.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      for (User *U : v->users()) {
        if (internal.count(U))
          continue;
        if (auto *S = dyn_cast<StoreInst>(U)) {
          if (S->getValueOperand() != v) {
            why = "the accumulator is used as a store ADDRESS";
            return false;
          }
          if (cStore && cStore != S) {
            why = "the accumulator reaches more than one store";
            return false;
          }
          cStore = S;
          continue;
        }
        auto *I = dyn_cast<Instruction>(U);
        if (!I)
          continue;
        if (isa<PHINode>(I) || isa<CastInst>(I)) {
          wl.push_back(I);
          continue;
        }
        if (I->getOpcode() == Instruction::FAdd && !epilogue) {
          epilogue = I;
          wl.push_back(I);
          continue;
        }
        why = "the accumulator has a consumer that is not a store or a single "
              "accumulating fadd (fused epilogue: not a dispatchable GEMM)";
        return false;
      }
    }
  }
  if (!cStore) {
    why = "no store consumes the accumulator";
    return false;
  }
  H.beta = 0.0;
  H.cLoad = nullptr;
  if (epilogue) {
    Value *other = epilogue->getOperand(0);
    if (rgReaches(other, last))
      other = epilogue->getOperand(1);
    auto *ld = dyn_cast<LoadInst>(other);
    if (!ld || ld->getPointerOperand() != cStore->getPointerOperand()) {
      why = "the epilogue fadd does not add the output element itself "
            "(C = alpha*A*B + beta*C with beta != 1 is not representable)";
      return false;
    }
    H.cLoad = ld;
    H.beta = 1.0;
  }
  RgPoly cPoly;
  if (!EX.expand(SE.getSCEV(cStore->getPointerOperand()), cPoly)) {
    why = "cannot expand the output address (" + EX.err + ")";
    return false;
  }
  for (unsigned r = 0; r < R; ++r)
    if (cPoly.contains(redCounter[r])) {
      why = "the output address moves with the reduction index";
      return false;
    }
  if (!splitAll(cPoly, cC, bC)) {
    why = "the output address is not affine in the recognized index sources";
    return false;
  }

  // Partition the indices by which operand they move. Which group is "M" is
  // NOT decided here: both assignments describe the same product (one is the
  // transpose of the other), and picking by source order would be arbitrary.
  SmallVector<unsigned, 2> g0, g1;
  for (unsigned i = 0; i < indices.size(); ++i) {
    bool in0 = !c0[i].isZero(), in1 = !c1[i].isZero(), inC = !cC[i].isZero();
    if (!in0 && !in1 && !inC)
      continue;
    if (in0 && in1) {
      why = "an index moves BOTH operands: this is not a matrix product";
      return false;
    }
    if (!inC) {
      why = "an index moves an operand but not the output: the contraction is "
            "not fully reduced";
      return false;
    }
    (in0 ? g0 : g1).push_back(i);
  }
  if (g0.empty() || g1.empty()) {
    why = "could not separate the two operands' free indices";
    return false;
  }
  if (g0.size() > 2 || g1.size() > 2) {
    why = "more than two fused indices on one side (unsupported; refusing to "
          "guess the fusion order)";
    return false;
  }

  // Fuse a (possibly two-index) side into one matrix index: the slow index's
  // stride must be exactly the fast one's times the fast extent, which is what
  // makes the pair a dense numbering of one matrix dimension.
  auto fuse = [&](ArrayRef<unsigned> ids, ArrayRef<RgPoly> coef,
                  RgPoly &unitStride, RgPoly &extent,
                  SmallVectorImpl<unsigned> &order) -> bool {
    if (ids.size() == 1) {
      unitStride = coef[ids[0]];
      extent = indices[ids[0]].extent;
      order.assign(1, ids[0]);
      return true;
    }
    for (unsigned f = 0; f < 2; ++f) {
      unsigned fast = ids[f], slow = ids[1 - f];
      if (rgMul(coef[fast], indices[fast].extent) == coef[slow]) {
        unitStride = coef[fast];
        extent = rgMul(indices[fast].extent, indices[slow].extent);
        order.assign({fast, slow});
        return true;
      }
    }
    return false;
  };
  RgPoly s0, e0, s1, e1;
  SmallVector<unsigned, 2> ord0, ord1;
  if (!fuse(g0, c0, s0, e0, ord0) || !fuse(g1, c1, s1, e1, ord1)) {
    why = "an operand's two free indices are not a dense fused numbering "
          "(stride/extent mismatch)";
    return false;
  }
  RgPoly sc0, sc1, dummyExtent;
  SmallVector<unsigned, 2> ord;
  if (!fuse(ord0, cC, sc0, dummyExtent, ord) || ord != ord0 ||
      !fuse(ord1, cC, sc1, dummyExtent, ord) || ord != ord1) {
    why = "the output does not use the same fused numbering as the operands";
    return false;
  }

  // Canonical orientation: the index group that is CONTIGUOUS in the output is
  // M. That is exactly what makes the output column-major with ldc = M, the
  // form every host GEMM entry point here describes, and it makes the choice
  // deterministic instead of source-order dependent.
  int64_t k0 = 0, k1 = 0;
  bool op0IsRow = sc0.isConstant(&k0) && k0 == elemSize;
  bool op1IsRow = sc1.isConstant(&k1) && k1 == elemSize;
  if (op0IsRow == op1IsRow) {
    why = "no operand's free index is contiguous in the output (or both are), "
          "so the output is not a dense M x Ncols matrix";
    return false;
  }
  const bool aIsOp0 = op0IsRow;
  RgPoly rowStride = aIsOp0 ? s0 : s1;  // lda, in bytes
  RgPoly rowExtent = aIsOp0 ? e0 : e1;  // M
  RgPoly colStrideB = aIsOp0 ? s1 : s0; // ldb, in bytes
  RgPoly colExtent = aIsOp0 ? e1 : e0;  // Ncols
  RgPoly ldcBytes = aIsOp0 ? sc1 : sc0; // ldc, in bytes
  SmallVector<unsigned, 2> rowOrder = aIsOp0 ? ord0 : ord1;
  SmallVector<unsigned, 2> colOrder = aIsOp0 ? ord1 : ord0;
  RgPoly &aBasePoly = aIsOp0 ? b0 : b1;
  RgPoly &bBasePoly = aIsOp0 ? b1 : b0;
  const int64_t aStep = aIsOp0 ? step0[0] : step1[0];
  const int64_t bStep = aIsOp0 ? step1[0] : step0[0];

  // --- layouts --------------------------------------------------------------
  // A must be K-contiguous: the host runtimes read A as an M x K row-major
  // block (aColMajor == 0). If the OTHER operand were the K-contiguous one the
  // product would have to be emitted transposed, which this arm does not do.
  if (aStep != elemSize) {
    why = "the operand carrying the output's contiguous index is not "
          "contiguous along the reduction axis; emitting this product would "
          "require the transposed orientation, which is not implemented";
    return false;
  }
  H.aColMajor = false;
  H.cColMajor = true; // by construction of the orientation choice above
  H.bColMajor = (bStep == elemSize);
  if (!H.bColMajor) {
    // K-strided B: the reduction step IS ldb and the column step must be one
    // element.
    int64_t cs = 0;
    if (!colStrideB.isConstant(&cs) || cs != elemSize) {
      why = "B is K-strided but its column step is not one element";
      return false;
    }
    colStrideB = RgPoly::constant(bStep);
  }

  // --- dimensions in ELEMENTS ----------------------------------------------
  RgPoly kPoly = rgMul(RgPoly::constant((int64_t)R), tripPoly);
  RgPoly ldaElems, ldbElems, ldcElems;
  if (!rgDivExact(rowStride, elemSize, ldaElems) ||
      !rgDivExact(colStrideB, elemSize, ldbElems) ||
      !rgDivExact(ldcBytes, elemSize, ldcElems)) {
    why = "a leading dimension is not a whole number of elements";
    return false;
  }
  std::string dimWhy;
  if (!rgToRuntimeDim(rowExtent, atoms, H.M, dimWhy) ||
      !rgToRuntimeDim(colExtent, atoms, H.Ncols, dimWhy) ||
      !rgToRuntimeDim(kPoly, atoms, H.K, dimWhy) ||
      !rgToRuntimeDim(ldaElems, atoms, H.lda, dimWhy) ||
      !rgToRuntimeDim(ldbElems, atoms, H.ldb, dimWhy) ||
      !rgToRuntimeDim(ldcElems, atoms, H.ldc, dimWhy)) {
    why = "a GEMM dimension is not expressible as mul*arg+add: " + dimWhy;
    return false;
  }

  // --- operand roles as body parameters ------------------------------------
  Value *aBaseVal = nullptr, *bBaseVal = nullptr, *cBaseVal = nullptr;
  if (!isBasePointer(aBasePoly, H.aParam, aBaseVal) ||
      !isBasePointer(bBasePoly, H.bParam, bBaseVal) ||
      !isBasePointer(bC, H.cParam, cBaseVal)) {
    why = "an operand does not root in a parameter of the optimized body";
    return false;
  }
  H.aBase = aBaseVal;
  H.bBase = bBaseVal;
  H.cBase = cBaseVal;
  H.cStore = cStore;
  H.epilogue = epilogue;
  H.alpha = 1.0;
  for (const RgMember &m : chain) {
    H.fmas.push_back(m.fma);
    H.fmuls.push_back(m.fmul);
    H.aLoads.push_back(aIsOp0 ? m.aLoad : m.bLoad);
    H.bLoads.push_back(aIsOp0 ? m.bLoad : m.aLoad);
  }

  // --- profile-scale shape --------------------------------------------------
  // Column extents must be fixed by the launch geometry (a CTA span, or a
  // compile-time bound); the row extent then follows EXACTLY from the profiled
  // execution counts. Nothing here is estimated: a missing count aborts.
  unsigned ctaPerLaunch = std::max(1u, hdr.maxGridDim[0]) *
                          std::max(1u, hdr.maxGridDim[1]) *
                          std::max(1u, hdr.maxGridDim[2]);
  if (!hdr.launchCount || hdr.launchCount % ctaPerLaunch) {
    why = "profile header launch count is not a whole number of grids";
    return false;
  }
  H.ctaPerLaunch = ctaPerLaunch;
  H.launches = (unsigned)(hdr.launchCount / ctaPerLaunch);

  uint64_t profNcols = 1;
  for (unsigned i : colOrder) {
    unsigned e = 0;
    if (!rgProfileExtent(indices[i], atoms, hdr, e) || e == 0) {
      why = "the profile does not fix the extent of a column index (launch "
            "geometry gives no bound for " +
            indices[i].name + ")";
      return false;
    }
    profNcols *= e;
  }

  uint64_t macs = 0;
  for (const RgMember &m : chain) {
    size_t idx;
    if (!tryReadProfIdxMetadata(m.fma, idx)) {
      why = "a chain member's fma carries no profile index";
      return false;
    }
    auto it = prof.find(idx);
    if (it == prof.end() || it->second.exec == 0) {
      why = "a chain member's fma has no profiled execution count";
      return false;
    }
    macs += it->second.exec;
  }
  H.profMacs = macs;

  uint64_t rowsTimesCols = 0;
  if (epilogue) {
    size_t idx;
    auto it = prof.end();
    if (tryReadProfIdxMetadata(epilogue, idx))
      it = prof.find(idx);
    if (it == prof.end() || it->second.exec == 0) {
      why = "the accumulating epilogue has no profiled execution count, and it "
            "is the only exact source for the row extent";
      return false;
    }
    H.profEpilogue = it->second.exec;
    if (H.profEpilogue % H.launches) {
      why = "profiled epilogue count is not a whole number of launches";
      return false;
    }
    rowsTimesCols = H.profEpilogue / H.launches;
  } else {
    // beta == 0: no epilogue op to count. Fall back to a compile-time trip.
    int64_t tripConst = 0;
    if (!tripPoly.isConstant(&tripConst) || tripConst <= 0) {
      why = "no accumulating epilogue and a runtime reduction length: the "
            "profile cannot fix M and K separately";
      return false;
    }
    uint64_t k = (uint64_t)R * (uint64_t)tripConst;
    if (macs % ((uint64_t)H.launches * k)) {
      why = "profiled MAC count is not divisible by launches*K";
      return false;
    }
    rowsTimesCols = macs / ((uint64_t)H.launches * k);
  }
  if (rowsTimesCols == 0 || rowsTimesCols % profNcols) {
    why = "profiled output-element count " + std::to_string(rowsTimesCols) +
          " is not a multiple of the reconstructed column count " +
          std::to_string(profNcols);
    return false;
  }
  H.profN = (unsigned)profNcols;
  H.profM = (unsigned)(rowsTimesCols / profNcols);
  uint64_t denom = (uint64_t)H.launches * rowsTimesCols;
  if (!denom || macs % denom) {
    why = "profiled MAC count is not divisible by launches*M*Ncols";
    return false;
  }
  H.profK = (unsigned)(macs / denom);
  if (!H.profM || !H.profN || !H.profK) {
    why = "a reconstructed profile-scale extent is zero";
    return false;
  }
  // Per-CTA tile: drop the index factors that are spread ACROSS blocks.
  {
    auto hasCta = [&](const RgIndex &ix) {
      for (auto &kv : ix.form.t)
        for (unsigned a : kv.first) {
          const SCEV *S = atoms.byId[a];
          auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
          if (U && rgCtaidAxis(U->getValue()) >= 0)
            return true;
        }
      return false;
    };
    uint64_t colSpread = 1, rowSpread = 1;
    for (unsigned i : colOrder)
      if (hasCta(indices[i])) {
        unsigned e = 0;
        if (!rgProfileExtent(indices[i], atoms, hdr, e) || !e) {
          why = "cannot size the per-CTA column tile";
          return false;
        }
        colSpread *= e;
      }
    for (unsigned i : rowOrder)
      if (hasCta(indices[i])) {
        unsigned e = 0;
        if (!rgProfileExtent(indices[i], atoms, hdr, e) || !e) {
          why = "cannot size the per-CTA row tile";
          return false;
        }
        rowSpread *= e;
      }
    if (!colSpread || !rowSpread || H.profN % colSpread ||
        H.profM % rowSpread) {
      why = "the product does not divide evenly into per-CTA tiles";
      return false;
    }
    H.tileN = (unsigned)(H.profN / colSpread);
    H.tileM = (unsigned)(H.profM / rowSpread);
    if (!H.tileM || !H.tileN) {
      why = "a per-CTA tile extent is zero";
      return false;
    }
  }
  // Cross-check the symbolic K against the profile when K is compile-time.
  {
    int64_t kc = 0;
    if (kPoly.isConstant(&kc) && kc > 0 && (unsigned)kc != H.profK) {
      why = "compile-time K (" + std::to_string(kc) +
            ") disagrees with the profiled reduction length (" +
            std::to_string(H.profK) + ")";
      return false;
    }
  }

  return true;
}

namespace {
// One reduction loop of a contraction chain: a closed fma/phi cycle over two
// loads whose addresses are affine in the loop with CONSTANT byte steps. The
// trip count is deliberately NOT required to be a constant; that is the whole
// point of this arm.
#define RG_REJECT(msg)                                                         \
  do {                                                                         \
    if (flags::Print && phi.getType()->isFloatingPointTy())                    \
      llvm::errs() << "[hostgemm]   reject phi in "                            \
                   << L->getHeader()->getName() << ": " << (msg) << "\n";      \
    return false;                                                              \
  } while (0)

static bool rgMatchReductionPhi(Loop *L, PHINode &phi, ScalarEvolution &SE,
                                BasicBlock *latch, RgMember &out) {
  if (!phi.getType()->isFloatingPointTy())
    return false;
  Value *be = phi.getIncomingValueForBlock(latch);
  if (!be)
    RG_REJECT("no backedge value");
  Instruction *fma = nullptr, *fmul = nullptr;
  Value *a = nullptr, *b = nullptr, *acc = nullptr;
  if (!rgMatchFMA(be, fma, fmul, a, b, acc) || acc != &phi)
    RG_REJECT("backedge value is not an accumulating fma");
  auto onlyInLoopUser = [&](Value *v, Instruction *expected) {
    for (User *U : v->users()) {
      auto *I = dyn_cast<Instruction>(U);
      if (!I || !L->contains(I))
        continue;
      if (I != expected)
        return false;
    }
    return true;
  };
  if (!onlyInLoopUser(fma, &phi) || !onlyInLoopUser(&phi, fma))
    RG_REJECT("fma/phi cycle is not closed");
  auto *aL = dyn_cast<LoadInst>(a);
  auto *bL = dyn_cast<LoadInst>(b);
  if (!aL || !bL)
    RG_REJECT("an fma multiplicand is not a load");
  if (aL->getType() != phi.getType() || bL->getType() != phi.getType())
    RG_REJECT("mixed-precision operands");
  const SCEV *btc = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(btc))
    RG_REJECT("backedge-taken count is not computable");
  BasicBlock *pre = L->getLoopPreheader();
  if (!pre)
    RG_REJECT("loop has no unique preheader");
  out.L = L;
  out.phi = &phi;
  out.fma = fma;
  out.fmul = fmul;
  out.aLoad = aL;
  out.bLoad = bL;
  out.aAddr = SE.getSCEV(aL->getPointerOperand());
  out.bAddr = SE.getSCEV(bL->getPointerOperand());
  out.trip = SE.getAddExpr(btc, SE.getOne(btc->getType()));
  out.init = phi.getIncomingValueForBlock(pre);
  return out.init != nullptr;
}
#undef RG_REJECT
} // namespace

void findHostGemmLoopNests(Function &F, ScalarEvolution &SE, LoopInfo &LI,
                           const FunctionProfileHeader &profileHeader,
                           const std::unordered_map<size_t, ProfileInfo> &prof,
                           SmallVectorImpl<AbstractMatmul> &out) {
  Module *M = F.getParent();
  if (!Triple(M->getTargetTriple()).isNVPTX())
    return;
  if (profileHeader.launchCount == 0) {
    if (flags::Print)
      llvm::errs() << "[hostgemm] " << F.getName()
                   << ": no profile header (launchCount=0); a runtime-shape "
                      "GEMM cannot be priced without one. Skipping.\n";
    return;
  }

  // Loops findScalarLoopMatmuls already claimed keep their existing candidate
  // set; this arm only looks at what that one left behind.
  SmallPtrSet<const Instruction *, 8> claimed;
  for (const AbstractMatmul &am : out)
    if (am.origin == AbstractMatmul::Origin::ScalarLoopReduction &&
        am.scalarLoop.fma)
      claimed.insert(am.scalarLoop.fma);

  DominatorTree DT(F);

  SmallVector<RgMember, 8> members;
  {
    SmallVector<Loop *, 8> wl(LI.begin(), LI.end());
    while (!wl.empty()) {
      Loop *L = wl.pop_back_val();
      for (Loop *S : L->getSubLoops())
        wl.push_back(S);
      if (!L->getSubLoops().empty())
        continue;
      BasicBlock *latch = L->getLoopLatch();
      if (!latch)
        continue;
      for (PHINode &phi : L->getHeader()->phis()) {
        RgMember m;
        if (rgMatchReductionPhi(L, phi, SE, latch, m))
          members.push_back(m);
      }
    }
  }
  if (flags::Print)
    llvm::errs() << "[hostgemm] " << F.getName() << ": " << members.size()
                 << " candidate reduction loop(s)\n";
  if (members.empty())
    return;

  // Chain the members. Member j's initial accumulator reaches the fma of EVERY
  // earlier member (the guard phis the unroller leaves make the whole prefix
  // reachable), so the immediate predecessor is the reachable member with the
  // longest prefix of its own; requiring that prefix to be exactly one shorter
  // is what makes the order a total chain rather than a guess.
  const unsigned n = members.size();
  SmallVector<SmallVector<unsigned, 4>, 8> reach(n);
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < n; ++i)
      if (i != j && rgReaches(members[j].init, members[i].fma))
        reach[j].push_back(i);
  SmallVector<int, 8> next(n, -1), prevOf(n, -1);
  SmallVector<bool, 8> ambiguous(n, false);
  for (unsigned j = 0; j < n; ++j) {
    if (reach[j].empty())
      continue;
    int best = -1;
    size_t bestSz = 0;
    unsigned ties = 0;
    for (unsigned i : reach[j]) {
      size_t sz = reach[i].size();
      if (best < 0 || sz > bestSz) {
        best = (int)i;
        bestSz = sz;
        ties = 1;
      } else if (sz == bestSz)
        ++ties;
    }
    if (ties != 1 || bestSz + 1 != reach[j].size() || next[best] >= 0) {
      ambiguous[j] = true;
      if (best >= 0)
        ambiguous[best] = true;
      continue;
    }
    next[best] = (int)j;
    prevOf[j] = best;
  }

  unsigned nextId = static_cast<unsigned>(out.size());
  for (unsigned h = 0; h < n; ++h) {
    if (prevOf[h] >= 0 || ambiguous[h]) {
      if (flags::Print && prevOf[h] < 0)
        llvm::errs() << "[hostgemm]   loop " << h
                     << " is not a chain head: ambiguous accumulator "
                        "dataflow\n";
      continue;
    }
    if (!rgIsZeroSeed(members[h].init)) {
      if (flags::Print)
        llvm::errs() << "[hostgemm]   loop " << h
                     << " is a chain head whose accumulator does not start at "
                        "+0.0; not a fresh contraction\n";
      continue;
    }
    SmallVector<RgMember, 8> chain;
    bool bad = false;
    for (int c = (int)h; c >= 0; c = next[c]) {
      if (ambiguous[c]) {
        bad = true;
        break;
      }
      chain.push_back(members[c]);
      if (chain.size() > 64) {
        bad = true;
        break;
      }
    }
    if (bad)
      continue;
    bool anyClaimed = false;
    for (const RgMember &m : chain)
      if (claimed.count(m.fma))
        anyClaimed = true;
    if (anyClaimed)
      continue;

    RuntimeGemmHandle H;
    std::string why;
    if (!rgAnalyzeChain(F, SE, LI, DT, profileHeader, prof, chain, H, why)) {
      if (flags::Print)
        llvm::errs() << "[hostgemm] " << F.getName() << ": reduction chain of "
                     << chain.size()
                     << " loop(s) is not a dispatchable runtime-shape GEMM: "
                     << why << "\n";
      continue;
    }

    AbstractMatmul am;
    am.id = nextId++;
    // M/N are the PER-CTA tile (accuracy-model simulation size); the full
    // product lives in globalM/globalN/globalK.
    am.M = H.tileM;
    am.N = H.tileN;
    am.K = H.profK;
    am.globalM = H.profM;
    am.globalN = H.profN;
    am.globalK = H.profK;
    am.gridCTAs = H.ctaPerLaunch;
    FPKind k = fpKindFromType(chain[0].aLoad->getType());
    am.aType = am.bType = am.accType = am.dType = k;
    am.aLayout = H.aColMajor ? AbstractMatmul::Layout::ColMajor
                             : AbstractMatmul::Layout::RowMajor;
    am.bLayout = H.bColMajor ? AbstractMatmul::Layout::ColMajor
                             : AbstractMatmul::Layout::RowMajor;
    am.dLayout = H.cColMajor ? AbstractMatmul::Layout::ColMajor
                             : AbstractMatmul::Layout::RowMajor;
    am.origin = AbstractMatmul::Origin::HostGemmLoopNest;
    am.hostGemm = std::make_shared<RuntimeGemmHandle>(std::move(H));
    am.outputValue = chain.back().fma;
    for (const RgMember &m : chain)
      for (BasicBlock *BB : m.L->blocks())
        for (Instruction &I : *BB)
          am.footprint.insert(&I);
    if (am.hostGemm->epilogue)
      am.footprint.insert(am.hostGemm->epilogue);

    if (flags::Print) {
      const RuntimeGemmHandle &g = *am.hostGemm;
      llvm::errs() << "[hostgemm] " << F.getName() << ": recognized GEMM from "
                   << chain.size() << " reduction loop(s)\n"
                   << "[hostgemm]   C = alpha*A^T*B + beta*C  alpha=" << g.alpha
                   << " beta=" << g.beta << "\n"
                   << "[hostgemm]   M=" << runtimeDimString(g.M)
                   << "  Ncols=" << runtimeDimString(g.Ncols)
                   << "  K=" << runtimeDimString(g.K) << "\n"
                   << "[hostgemm]   lda=" << runtimeDimString(g.lda)
                   << "  ldb=" << runtimeDimString(g.ldb)
                   << "  ldc=" << runtimeDimString(g.ldc) << "\n"
                   << "[hostgemm]   aColMajor=" << (g.aColMajor ? 1 : 0)
                   << " bColMajor=" << (g.bColMajor ? 1 : 0)
                   << " cColMajor=" << (g.cColMajor ? 1 : 0) << "  C=param"
                   << g.cParam << " A=param" << g.aParam << " B=param"
                   << g.bParam << "\n"
                   << "[hostgemm]   profile shape M=" << g.profM
                   << " Ncols=" << g.profN << " K=" << g.profK
                   << " (per-CTA tile " << g.tileM << "x" << g.tileN
                   << ", launches=" << g.launches
                   << " CTAs/launch=" << g.ctaPerLaunch
                   << " MACs=" << g.profMacs << ")\n";
    }
    out.push_back(std::move(am));
  }
}

} // namespace poseidon
