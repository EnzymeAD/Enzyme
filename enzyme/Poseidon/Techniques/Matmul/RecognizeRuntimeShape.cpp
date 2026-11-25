//===- RecognizeRuntimeShape.cpp - runtime-shape product recognition ------===//
//
// Part of the Enzyme Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Recognition of dense GEMMs with runtime dimensions
// (Origin::RuntimeShape): thread-parallel reduction nests whose extents
// and leading dimensions are expressions over the kernel's launch values,
// whose contraction may be split across several reduction loops (an unrolled
// constant-trip outer level), and whose row and column indices may run in
// block-strided thread loops or combine a thread index with the block index.
//
// SCEV does not distribute a non-constant multiplier over a sum, so a
// difference such as (3i+1)*n - 3i*n does not fold. Addresses are expanded
// into integer polynomials over SCEV atoms (Polynomial) instead, on which such
// comparisons are exact; every dimension is either derived or the site is
// refused.
//
//===----------------------------------------------------------------------===//

#include "Internal.h"
#include "Pass/Options.h"
#include "Profile/ProfileReader.h"
#include "Sites/Canonicalize.h"
#include "Support/Utils.h"
#include "Targets/Target.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::PatternMatch;

namespace poseidon {

/// Why a reduction chain is not a dispatchable runtime-shape product.
static Error refuse(const Twine &why) {
  return createStringError(inconvertibleErrorCode(), why);
}

namespace {

// ---------------------------------------------------------------------------
// Integer polynomials over SCEV atoms.
// ---------------------------------------------------------------------------
using Monomial =
    std::vector<unsigned>; // sorted atom ids; empty == constant term

struct Polynomial {
  std::map<Monomial, int64_t> t;

  static Polynomial constant(int64_t c) {
    Polynomial p;
    if (c)
      p.t[Monomial{}] = c;
    return p;
  }
  static Polynomial atom(unsigned id) {
    Polynomial p;
    p.t[Monomial{id}] = 1;
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
  void addTerm(const Monomial &m, int64_t c) {
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
  bool operator==(const Polynomial &o) const { return t == o.t; }
};

} // namespace

static Polynomial operator+(const Polynomial &a, const Polynomial &b) {
  Polynomial r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, kv.second);
  return r;
}
static Polynomial operator-(const Polynomial &a, const Polynomial &b) {
  Polynomial r = a;
  for (auto &kv : b.t)
    r.addTerm(kv.first, -kv.second);
  return r;
}
static Polynomial operator*(const Polynomial &a, const Polynomial &b) {
  Polynomial r;
  for (auto &x : a.t)
    for (auto &y : b.t) {
      Monomial m = x.first;
      m.insert(m.end(), y.first.begin(), y.first.end());
      llvm::sort(m);
      r.addTerm(m, x.second * y.second);
    }
  return r;
}
/// `a` divided by the nonzero integer `d`; none when a coefficient is not a
/// multiple of `d` (a fractional leading dimension), never a rounded guess.
static std::optional<Polynomial> divideExactly(const Polynomial &a, int64_t d) {
  if (d == 0)
    return std::nullopt;
  Polynomial out;
  for (auto &kv : a.t) {
    if (kv.second % d)
      return std::nullopt;
    out.addTerm(kv.first, kv.second / d);
  }
  return out;
}

namespace {
/// `p` as quotient * atom + rest, with `rest` free of the atom.
struct AtomSplit {
  Polynomial quotient, rest;
};
} // namespace

/// Divides one occurrence of `atom` out of every monomial of `p` that has it.
/// None when a monomial has the atom to a power above 1: a quadratic index is
/// not a matrix subscript.
static std::optional<AtomSplit> divideByAtom(const Polynomial &p,
                                             unsigned atom) {
  AtomSplit s;
  for (auto &kv : p.t) {
    unsigned n = 0;
    for (unsigned a : kv.first)
      n += (a == atom);
    if (n == 0) {
      s.rest.addTerm(kv.first, kv.second);
      continue;
    }
    if (n > 1)
      return std::nullopt;
    Monomial m;
    bool dropped = false;
    for (unsigned a : kv.first) {
      if (a == atom && !dropped) {
        dropped = true;
        continue;
      }
      m.push_back(a);
    }
    s.quotient.addTerm(m, kv.second);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Atom table and SCEV -> polynomial expansion.
//
// Every AddRec is expanded as start + step * j<L>, with j<L> a synthetic atom
// standing for the loop's iteration counter. The contribution of a
// block-strided thread loop index to an address then becomes
// coef*(tid.k + ntid.k*j), and the two coefficients can be checked against
// each other instead of assuming the loop runs once.
// ---------------------------------------------------------------------------
// A parameter of the optimized body, or a field of a by-value aggregate
// parameter: a value the host can read at the launch.
static LaunchRef launchValueOf(const Value *v) {
  if (auto *Arg = dyn_cast<Argument>(v))
    return {(int)Arg->getArgNo(), -1};
  return launchFieldOf(v);
}

/// Why the host cannot read the launch value `v` (at `r`) as a dimension or an
/// assumed-equal operand, which it loads as an i32: a narrower integer would be
/// read past its end, and a wider one would lose the bits it reads past 32,
/// which the dispatch's int dimensions could not take anyway. None when it can.
static std::optional<std::string> hostReadRefusal(const Value *v,
                                                  const LaunchRef &r) {
  auto *T = dyn_cast<IntegerType>(v->getType());
  if (T && T->getBitWidth() == 32)
    return std::nullopt;
  std::string what = T ? ", an i" + std::to_string(T->getBitWidth()) + ","
                       : ", which is not an integer,";
  return launchRefString(r, "param") + what +
         " but the host reads each launch value as an i32";
}

namespace {

struct AtomTable {
  SmallVector<const SCEV *, 16> byId; // null for synthetic loop-counter atoms
  DenseMap<const SCEV *, unsigned> idOf;
  // Two loads of one field are one atom: the value is the launch's either way.
  std::map<LaunchRef, unsigned> idOfField;
  DenseMap<const Loop *, unsigned> counterOf;

  unsigned id(const SCEV *S) {
    auto it = idOf.find(S);
    if (it != idOf.end())
      return it->second;
    LaunchRef field;
    if (auto *U = dyn_cast<SCEVUnknown>(S))
      if (!isa<Argument>(U->getValue()))
        field = launchFieldOf(U->getValue());
    if (field.param >= 0) {
      auto fit = idOfField.find(field);
      if (fit != idOfField.end()) {
        idOf[S] = fit->second;
        return fit->second;
      }
    }
    unsigned n = byId.size();
    byId.push_back(S);
    idOf[S] = n;
    if (field.param >= 0)
      idOfField[field] = n;
    return n;
  }
  unsigned counter(const Loop *L) {
    auto it = counterOf.find(L);
    if (it != counterOf.end())
      return it->second;
    unsigned n = byId.size();
    byId.push_back(nullptr);
    counterOf[L] = n;
    return n;
  }

  /// `S` as a polynomial over this table's atoms, adding the atoms it meets.
  Expected<Polynomial> expand(ScalarEvolution &SE, const SCEV *S) {
    if (auto *C = dyn_cast<SCEVConstant>(S))
      return Polynomial::constant(C->getAPInt().getSExtValue());
    if (auto *Cast = dyn_cast<SCEVCastExpr>(S)) {
      // zext / sext / trunc are transparent: each one here comes from index
      // arithmetic LLVM proved non-wrapping (the nsw/nneg flags on the source
      // adds and muls), so it preserves the value.
      return expand(SE, Cast->getOperand());
    }
    if (auto *Add = dyn_cast<SCEVAddExpr>(S)) {
      Polynomial out;
      for (const SCEV *Op : Add->operands()) {
        Expected<Polynomial> p = expand(SE, Op);
        if (!p)
          return p.takeError();
        out = out + *p;
      }
      return out;
    }
    if (auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
      Polynomial out = Polynomial::constant(1);
      for (const SCEV *Op : Mul->operands()) {
        Expected<Polynomial> p = expand(SE, Op);
        if (!p)
          return p.takeError();
        out = out * *p;
      }
      return out;
    }
    if (auto *Rec = dyn_cast<SCEVAddRecExpr>(S)) {
      if (!Rec->isAffine())
        return refuse("non-affine add-recurrence");
      Expected<Polynomial> start = expand(SE, Rec->getStart());
      if (!start)
        return start.takeError();
      Expected<Polynomial> step = expand(SE, Rec->getStepRecurrence(SE));
      if (!step)
        return step.takeError();
      return *start + *step * Polynomial::atom(counter(Rec->getLoop()));
    }
    if (isa<SCEVUnknown>(S))
      return Polynomial::atom(id(S));
    return refuse("unsupported SCEV node (udiv / min / max)");
  }
};

} // namespace

// A GEMM dimension is usable only if it is one monomial over launch values of
// the optimized body plus a constant, mul*v0*v1*... + add, which the host-side
// stub rewrite evaluates from the launch arguments.
static Expected<RuntimeDim> runtimeDimOf(const Polynomial &p,
                                         const AtomTable &A) {
  RuntimeDim out;
  bool haveTerm = false;
  for (auto &kv : p.t) {
    if (kv.second < -kMaxDimConstant || kv.second > kMaxDimConstant)
      return refuse("a GEMM dimension has the constant " +
                    std::to_string(kv.second) +
                    ", out of the range a dispatch dimension takes");
    if (kv.first.empty()) {
      out.add = kv.second;
      continue;
    }
    if (kv.first.size() > kMaxDimFactors)
      return refuse("a GEMM dimension is a product of " +
                    std::to_string(kv.first.size()) +
                    " launch values, more than the " +
                    std::to_string(kMaxDimFactors) + " a descriptor holds");
    if (haveTerm)
      return refuse("a GEMM dimension is not expressible as mul*arg+add: value "
                    "depends on more than one runtime term");
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
      LaunchRef r = U ? launchValueOf(U->getValue()) : LaunchRef{};
      if (r.param < 0)
        return refuse("a GEMM dimension is not expressible as mul*arg+add: "
                      "value depends on a term that is not a parameter of the "
                      "optimized body or a field of one passed by value");
      if (std::optional<std::string> refusal =
              hostReadRefusal(U->getValue(), r))
        return refuse("a GEMM dimension depends on " + *refusal);
      out.factors.push_back(r);
    }
    // In launch order rather than in the order the atoms were met.
    llvm::sort(out.factors);
    out.mul = kv.second;
    haveTerm = true;
  }
  return out;
}

/// The launch value `p` is, when it is exactly one (coefficient 1, no
/// constant); none when it is not one; an error when the host cannot read it.
static Expected<std::optional<LaunchRef>>
singleLaunchValue(const Polynomial &p, const AtomTable &A) {
  if (p.t.size() != 1 || p.t.begin()->first.size() != 1 ||
      p.t.begin()->second != 1)
    return std::nullopt;
  const SCEV *S = A.byId[p.t.begin()->first[0]];
  auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
  LaunchRef r = U ? launchValueOf(U->getValue()) : LaunchRef{};
  if (r.param < 0 || !U->getType()->isIntegerTy())
    return std::nullopt;
  if (std::optional<std::string> refusal = hostReadRefusal(U->getValue(), r))
    return refuse(*refusal);
  return r;
}

static int threadIndexAxis(const Value *v) {
  LaunchGeometry g = launchGeometryOf(v);
  return g.Kind == LaunchValue::ThreadIndex ? g.Axis : -1;
}
// A block dimension, zero-extended where the index arithmetic uses it.
static int blockDimAxis(const Value *v) {
  if (auto *Z = dyn_cast<ZExtInst>(v))
    v = Z->getOperand(0);
  LaunchGeometry g = launchGeometryOf(v);
  return g.Kind == LaunchValue::BlockDim ? g.Axis : -1;
}
static int blockIndexAxis(const Value *v) {
  LaunchGeometry g = launchGeometryOf(v);
  return g.Kind == LaunchValue::BlockIndex ? g.Axis : -1;
}
static bool mentionsLaunchIndex(const SCEV *S) {
  return SCEVExprContains(S, [](const SCEV *X) {
    auto *U = dyn_cast<SCEVUnknown>(X);
    if (!U)
      return false;
    Value *v = U->getValue();
    return threadIndexAxis(v) >= 0 || blockIndexAxis(v) >= 0;
  });
}

namespace {

// One recognized index source: a thread axis (optionally behind a
// block-strided thread loop) or a CTA linearization.
struct IndexSource {
  Polynomial form;        // the index's symbolic value
  unsigned sigAtom = ~0u; // atom appearing in `form` and in no other index
  // The rest of the monomial the signature atom sits in (coefficient 1). For a
  // thread index this is empty; for a CTA linearization
  // ctaid.a*ntid.b + tid.b it is {ntid.b}.
  SmallVector<unsigned, 2> sigMono;
  SmallVector<unsigned, 4> ownAtoms; // atoms that must not survive extraction
  Polynomial extent; // number of distinct values the index takes
  std::string name;
};

/// `p` as coef * idx.form + rest, with `rest` free of every atom of `idx`.
struct IndexSplit {
  Polynomial coef, rest;
};

} // namespace

/// Peels the index `idx` out of an address polynomial; none when `p` is not
/// coef * idx.form + rest: the address is not a matrix subscript in `idx`.
static std::optional<IndexSplit> splitIndex(const Polynomial &p,
                                            const IndexSource &idx) {
  std::optional<AtomSplit> s = divideByAtom(p, idx.sigAtom);
  if (!s)
    return std::nullopt;
  Polynomial q = s->quotient;
  // Peel off the rest of the signature monomial (the ntid factor of a CTA
  // linearization); anything left over means the address does not carry the
  // index in the shape the index actually has.
  for (unsigned a : idx.sigMono) {
    std::optional<AtomSplit> s2 = divideByAtom(q, a);
    if (!s2 || !s2->rest.isZero())
      return std::nullopt;
    q = s2->quotient;
  }
  IndexSplit out{q, p - q * idx.form};
  for (unsigned a : idx.ownAtoms)
    if (out.rest.contains(a))
      return std::nullopt;
  return out;
}

namespace {

// One member of the contraction chain. The operand addresses are kept as full
// SCEVs: a two-level index arrives as an integer add-recurrence under a cast,
// sext(i32 {{...}<i-loop>,+,1}<p-loop>), not as a pointer add-recurrence of
// the reduction loop. The reduction byte step is recovered from the
// polynomial expansion instead (the coefficient of the reduction loop's
// iteration counter), which is exact for every cast shape.
struct ChainMember {
  Loop *L = nullptr;
  PHINode *phi = nullptr;
  Instruction *fma = nullptr;
  Instruction *fmul = nullptr;
  LoadInst *aLoad = nullptr;
  LoadInst *bLoad = nullptr;
  const SCEV *aAddr = nullptr, *bAddr = nullptr;
  const SCEV *trip = nullptr;
  Value *init = nullptr;
};

} // namespace

// Can `target` reach `v` through phis and casts only?
static bool reachesThroughPhis(Value *v, Value *target, unsigned depth = 0) {
  if (!v || depth > 12)
    return false;
  if (v == target)
    return true;
  if (auto *P = dyn_cast<PHINode>(v)) {
    for (Value *in : P->incoming_values())
      if (in != P && reachesThroughPhis(in, target, depth + 1))
        return true;
    return false;
  }
  if (auto *C = dyn_cast<CastInst>(v))
    return reachesThroughPhis(C->getOperand(0), target, depth + 1);
  return false;
}

static const SCEV *stripCasts(const SCEV *S) {
  while (auto *C = dyn_cast<SCEVCastExpr>(S))
    S = C->getOperand();
  return S;
}

namespace {
/// A block-strided thread loop, for (i = threadIdx.k; i < bound; i +=
/// blockDim.k).
struct StridedThreadLoop {
  Value *index;
  Value *bound;
  int axis;
};
} // namespace

/// `L` as a block-strided thread loop; none when it is not one.
static std::optional<StridedThreadLoop>
matchStridedThreadLoop(Loop *L, ScalarEvolution &SE) {
  BasicBlock *latch = L->getLoopLatch();
  if (!latch)
    return std::nullopt;
  Instruction *br = latch->getTerminator();
  Value *cond;
  if (!match(br,
             m_Br(m_Value(cond), m_Specific(L->getHeader()), m_BasicBlock())))
    return std::nullopt;
  auto *cmp = dyn_cast<ICmpInst>(cond);
  if (!cmp)
    return std::nullopt;
  auto pred = cmp->getPredicate();
  if (pred != ICmpInst::ICMP_SLT && pred != ICmpInst::ICMP_ULT)
    return std::nullopt;
  auto *nextIdx = dyn_cast<BinaryOperator>(cmp->getOperand(0));
  if (!nextIdx || nextIdx->getOpcode() != Instruction::Add)
    return std::nullopt;
  int a0 = blockDimAxis(nextIdx->getOperand(0));
  int a1 = blockDimAxis(nextIdx->getOperand(1));
  if ((a0 < 0) == (a1 < 0))
    return std::nullopt; // need exactly one blockDim operand
  StridedThreadLoop S{a0 >= 0 ? nextIdx->getOperand(1) : nextIdx->getOperand(0),
                      cmp->getOperand(1), a0 >= 0 ? a0 : a1};

  auto *Rec = dyn_cast<SCEVAddRecExpr>(stripCasts(SE.getSCEV(S.index)));
  if (!Rec || Rec->getLoop() != L || !Rec->isAffine())
    return std::nullopt;
  auto *st = dyn_cast<SCEVUnknown>(stripCasts(Rec->getStart()));
  auto *sp = dyn_cast<SCEVUnknown>(stripCasts(Rec->getStepRecurrence(SE)));
  if (!st || !sp || threadIndexAxis(st->getValue()) != S.axis ||
      blockDimAxis(sp->getValue()) != S.axis)
    return std::nullopt;
  return S;
}

// Fill in an IndexSource from its symbolic form: pick the signature atom (a CTA
// id if present, else a thread id), record the monomial it is multiplied by,
// and list the atoms that must not survive its extraction from an address.
static bool finishIndex(IndexSource &idx, const AtomTable &A) {
  int best = -1;
  bool bestIsCta = false;
  for (auto &kv : idx.form.t)
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
      if (!U)
        continue;
      bool isCta = blockIndexAxis(U->getValue()) >= 0;
      bool isTid = threadIndexAxis(U->getValue()) >= 0;
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
  const Monomial *sigMonoFull = nullptr;
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
      if (U && (threadIndexAxis(U->getValue()) >= 0 ||
                blockIndexAxis(U->getValue()) >= 0))
        idx.ownAtoms.push_back(a);
    }
  llvm::sort(idx.ownAtoms);
  idx.ownAtoms.erase(std::unique(idx.ownAtoms.begin(), idx.ownAtoms.end()),
                     idx.ownAtoms.end());
  return true;
}

// Largest value the index can take at profile scale, from the launch geometry
// in the profile header. None when the form involves a quantity the header
// does not fix (a loop counter, a runtime bound).
static std::optional<unsigned> profileExtent(const IndexSource &idx,
                                             const AtomTable &A,
                                             const SiteProfileHeader &hdr) {
  int64_t c = 0;
  if (idx.extent.isConstant(&c) && c > 0)
    return (unsigned)c; // compile-time bound: exact at every scale
  int64_t maxVal = 0;
  for (auto &kv : idx.form.t) {
    int64_t term = kv.second;
    if (term < 0)
      return std::nullopt;
    for (unsigned a : kv.first) {
      const SCEV *S = A.byId[a];
      auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
      if (!U)
        return std::nullopt; // loop counter or opaque runtime term
      Value *v = U->getValue();
      int ax;
      if ((ax = blockIndexAxis(v)) >= 0) {
        if (!hdr.maxGridDim[ax])
          return std::nullopt;
        term *= (int64_t)hdr.maxGridDim[ax] - 1;
      } else if ((ax = threadIndexAxis(v)) >= 0) {
        if (!hdr.maxBlockDim[ax])
          return std::nullopt;
        term *= (int64_t)hdr.maxBlockDim[ax] - 1;
      } else if ((ax = blockDimAxis(v)) >= 0) {
        if (!hdr.maxBlockDim[ax])
          return std::nullopt;
        term *= (int64_t)hdr.maxBlockDim[ax];
      } else {
        return std::nullopt;
      }
    }
    maxVal += term;
  }
  if (maxVal < 0)
    return std::nullopt;
  return (unsigned)(maxVal + 1);
}

namespace {

/// What the steps recognizing one contraction chain share: the chain, the
/// atoms its addresses are expanded over, and the index sources the product's
/// rows and columns run along. Atoms are numbered in the order the steps meet
/// them, which orders every polynomial, so the steps run in a fixed order.
struct ChainAnalysis {
  ChainAnalysis(ScalarEvolution &SE, ArrayRef<ChainMember> chain)
      : SE(SE), chain(chain) {}
  ScalarEvolution &SE;
  ArrayRef<ChainMember> chain;
  AtomTable atoms;
  SmallVector<IndexSource, 4> indices;
  /// Per member, the atom of its reduction loop's iteration counter.
  SmallVector<unsigned, 8> redCounter;

  Expected<Polynomial> expand(const SCEV *S) { return atoms.expand(SE, S); }
};

/// The address polynomial of each member's two operands at reduction index 0,
/// and the operands' byte steps along the reduction, which all members share.
struct OperandAddresses {
  SmallVector<Polynomial, 8> at[2];
  int64_t step[2] = {0, 0};
};

/// An address split over every index source: a coefficient per source and
/// the base the sources leave.
struct AddressSplit {
  SmallVector<Polynomial, 4> coef;
  Polynomial base;
};

/// The store of the product and the fadd that accumulates into it, if any.
struct OutputStore {
  StoreInst *store = nullptr;
  Instruction *epilogue = nullptr;
};

/// One side's index sources fused into one matrix index.
struct FusedIndex {
  Polynomial unitStride, extent;
  SmallVector<unsigned, 2> order;
};

/// Which operand is A and how the product is laid out, in bytes.
struct Orientation {
  bool aIsOp0;
  Polynomial lda, rows, ldb, cols, ldc;
  SmallVector<unsigned, 2> rowOrder, colOrder;
  bool bColMajor;
};

/// The product's shape at profile scale, and the profile counts it is
/// reconstructed from.
struct ProfileShape {
  unsigned M = 0, N = 0, K = 0;
  /// Per-CTA tile of the product, the size the accuracy model simulates.
  unsigned tileM = 0, tileN = 0;
  unsigned launches = 0, ctaPerLaunch = 0;
  uint64_t macs = 0;
};

/// A recognized chain: its library-call handle and its profile-scale shape.
struct RecognizedChain {
  RuntimeGemmHandle H;
  ProfileShape profile;
};

} // namespace

/// Finds the index sources dominating the contraction: block-strided thread
/// loops around it and guards of a launch index against a bound. Returns the
/// guard branches the sources come from.
static Expected<SmallPtrSet<const Instruction *, 8>>
collectIndexSources(ChainAnalysis &C, Function &F, DominatorTree &DT) {
  ScalarEvolution &SE = C.SE;
  for (Loop *L = C.chain[0].L->getParentLoop(); L; L = L->getParentLoop()) {
    std::optional<StridedThreadLoop> S = matchStridedThreadLoop(L, SE);
    if (!S)
      continue;
    IndexSource idx;
    Expected<Polynomial> form = C.expand(SE.getSCEV(S->index));
    if (!form)
      return refuse("cannot expand a strided-thread-loop index or its bound (" +
                    toString(form.takeError()) + ")");
    Expected<Polynomial> extent = C.expand(SE.getSCEV(S->bound));
    if (!extent)
      return refuse("cannot expand a strided-thread-loop index or its bound (" +
                    toString(extent.takeError()) + ")");
    idx.form = std::move(*form);
    idx.extent = std::move(*extent);
    if (!finishIndex(idx, C.atoms))
      return refuse("strided-thread-loop index has no usable signature term");
    idx.name = ("tid." + Twine("xyz"[S->axis]) + "-strided").str();
    C.indices.push_back(idx);
  }
  BasicBlock *anchorBB = C.chain.back().fma->getParent();
  SmallPtrSet<const Instruction *, 8> indexBranches;
  for (BasicBlock &BB : F) {
    Instruction *br = BB.getTerminator();
    Value *cond;
    if (!match(br, m_Br(m_Value(cond), m_BasicBlock(), m_BasicBlock())))
      continue;
    auto *cmp = dyn_cast<ICmpInst>(cond);
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
    if (!mentionsLaunchIndex(SX))
      continue;
    // An unexpandable guard is simply not an index source.
    IndexSource idx;
    Expected<Polynomial> form = C.expand(SX);
    if (!form) {
      consumeError(form.takeError());
      continue;
    }
    Expected<Polynomial> extent = C.expand(SE.getSCEV(cmp->getOperand(1)));
    if (!extent) {
      consumeError(extent.takeError());
      continue;
    }
    idx.form = std::move(*form);
    idx.extent = std::move(*extent);
    // The guard a rotated strided loop is entered through, on the loop's own
    // bound. A guard of the same index with another bound is a second source
    // of that index, which the signature check below refuses.
    if (llvm::any_of(C.indices, [&](const IndexSource &prev) {
          return (prev.form == idx.form ||
                  idx.form == Polynomial::atom(prev.sigAtom)) &&
                 prev.extent == idx.extent;
        })) {
      indexBranches.insert(br);
      continue;
    }
    if (!finishIndex(idx, C.atoms))
      continue;
    idx.name = "guarded";
    C.indices.push_back(idx);
    indexBranches.insert(br);
  }
  if (C.indices.empty())
    return refuse("no thread/CTA index source dominates the contraction");
  // Signature atoms must be distinct, or peeling one index out of an address
  // would consume another's contribution.
  for (unsigned i = 0; i < C.indices.size(); ++i)
    for (unsigned j = i + 1; j < C.indices.size(); ++j)
      if (C.indices[i].sigAtom == C.indices[j].sigAtom)
        return refuse("two index sources share a signature term");
  return indexBranches;
}

/// Each member's operand addresses at reduction index 0 and their byte steps
/// along the reduction, the coefficient of the member's iteration counter.
static Expected<OperandAddresses> operandAddresses(ChainAnalysis &C) {
  const unsigned R = C.chain.size();
  for (unsigned r = 0; r < R; ++r)
    C.redCounter.push_back(C.atoms.counter(C.chain[r].L));
  OperandAddresses ops;
  for (unsigned r = 0; r < R; ++r) {
    int64_t step[2];
    const SCEV *addr[2] = {C.chain[r].aAddr, C.chain[r].bAddr};
    Polynomial full[2];
    for (unsigned o = 0; o < 2; ++o) {
      Expected<Polynomial> p = C.expand(addr[o]);
      if (!p)
        return refuse("cannot expand an operand address (" +
                      toString(p.takeError()) + ")");
      full[o] = std::move(*p);
    }
    std::optional<AtomSplit> split[2];
    for (unsigned o = 0; o < 2; ++o)
      if (!(split[o] = divideByAtom(full[o], C.redCounter[r])))
        return refuse("an operand address is not affine in the reduction "
                      "index");
    for (unsigned o = 0; o < 2; ++o)
      if (!split[o]->quotient.isConstant(&step[o]) || !step[o])
        return refuse("an operand's reduction-axis byte stride is zero or not "
                      "a compile-time constant");
    if (r == 0) {
      ops.step[0] = step[0];
      ops.step[1] = step[1];
    }
    if (step[0] != ops.step[0] || step[1] != ops.step[1])
      return refuse("chain members disagree on the reduction-axis byte stride");
    for (unsigned o = 0; o < 2; ++o)
      ops.at[o].push_back(split[o]->rest);
    for (unsigned q = 0; q < R; ++q)
      if (q != r && (ops.at[0][r].contains(C.redCounter[q]) ||
                     ops.at[1][r].contains(C.redCounter[q])))
        return refuse("an operand address mixes two members' reduction "
                      "indices");
  }
  return ops;
}

/// The reduction length every member runs.
static Expected<Polynomial> reductionLength(ChainAnalysis &C) {
  Expected<Polynomial> trip = C.expand(C.chain[0].trip);
  if (!trip)
    return refuse("cannot expand the reduction trip count (" +
                  toString(trip.takeError()) + ")");
  for (unsigned r = 1; r < C.chain.size(); ++r) {
    Expected<Polynomial> t = C.expand(C.chain[r].trip);
    if (!t || !(*t == *trip)) {
      if (!t)
        consumeError(t.takeError());
      return refuse("chain members have different reduction lengths");
    }
  }
  return trip;
}

/// Whether `a` and `b` are equal, or two launch values the dense reading
/// needs equal and nothing here can prove so (e.g. a captured tensor's extent
/// and the loop bound next to it), which joins `assumed` for the dispatch to
/// check at the launch. An error when the host cannot read one of them.
static Expected<bool>
sameOrAssumed(const Polynomial &a, const Polynomial &b, const AtomTable &A,
              SmallVectorImpl<std::pair<LaunchRef, LaunchRef>> &assumed) {
  if (a == b)
    return true;
  Expected<std::optional<LaunchRef>> x = singleLaunchValue(a, A);
  if (!x)
    return x.takeError();
  if (!*x)
    return false;
  Expected<std::optional<LaunchRef>> y = singleLaunchValue(b, A);
  if (!y)
    return y.takeError();
  if (!*y)
    return false;
  std::pair<LaunchRef, LaunchRef> eq = std::minmax(**x, **y);
  if (!llvm::is_contained(assumed, eq))
    assumed.push_back(eq);
  return true;
}

/// Checks that the members flatten into one reduction index: member r starts
/// exactly r * trip elements after member 0 in both operands.
static Error
flattenMembers(const ChainAnalysis &C, const OperandAddresses &ops,
               const Polynomial &trip,
               SmallVectorImpl<std::pair<LaunchRef, LaunchRef>> &assumed) {
  for (unsigned r = 1; r < C.chain.size(); ++r) {
    auto notContiguous = [&] {
      return refuse("chain member " + std::to_string(r) +
                    " is not contiguous with member 0 along the reduction "
                    "axis, so the two contraction levels do not flatten into "
                    "one index");
    };
    std::optional<Polynomial> along[2];
    for (unsigned o = 0; o < 2; ++o)
      if (!(along[o] = divideExactly(ops.at[o][r] - ops.at[o][0],
                                     (int64_t)r * ops.step[o])))
        return notContiguous();
    for (unsigned o = 0; o < 2; ++o) {
      Expected<bool> same = sameOrAssumed(*along[o], trip, C.atoms, assumed);
      if (!same)
        return refuse("chain member " + std::to_string(r) +
                      " is contiguous with member 0 only under an equality "
                      "the host cannot check: " +
                      toString(same.takeError()));
      if (!*same)
        return notContiguous();
    }
  }
  if (assumed.size() > kMaxAssumedEqual)
    return refuse("the dense reading takes " + std::to_string(assumed.size()) +
                  " pairs of launch values to be equal, more than the " +
                  std::to_string(kMaxAssumedEqual) + " a descriptor holds");
  return Error::success();
}

/// `p` split over every index source in turn; none when one does not split.
static std::optional<AddressSplit> splitAll(const Polynomial &p,
                                            ArrayRef<IndexSource> indices) {
  AddressSplit out;
  Polynomial cur = p;
  for (const IndexSource &idx : indices) {
    std::optional<IndexSplit> s = splitIndex(cur, idx);
    if (!s)
      return std::nullopt;
    out.coef.push_back(s->coef);
    cur = s->rest;
  }
  out.base = cur;
  return out;
}

/// The store the final accumulator reaches, through phis, casts and at most
/// one accumulating fadd.
static Expected<OutputStore> findOutputStore(ArrayRef<ChainMember> chain) {
  // The chain's own phis/fmas are internal recurrences, not consumers.
  SmallPtrSet<const Value *, 24> internal;
  for (const ChainMember &m : chain) {
    internal.insert(m.fma);
    internal.insert(m.phi);
    if (m.fmul)
      internal.insert(m.fmul);
  }
  OutputStore out;
  SmallVector<Value *, 8> wl{chain.back().fma};
  SmallPtrSet<Value *, 16> seen;
  while (!wl.empty()) {
    Value *v = wl.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    for (User *U : v->users()) {
      if (internal.count(U))
        continue;
      if (auto *S = dyn_cast<StoreInst>(U)) {
        if (S->getValueOperand() != v)
          return refuse("the accumulator is used as a store ADDRESS");
        if (out.store && out.store != S)
          return refuse("the accumulator reaches more than one store");
        out.store = S;
        continue;
      }
      auto *I = dyn_cast<Instruction>(U);
      if (!I)
        continue;
      if (isa<PHINode>(I) || isa<CastInst>(I)) {
        wl.push_back(I);
        continue;
      }
      if (I->getOpcode() == Instruction::FAdd && !out.epilogue) {
        out.epilogue = I;
        wl.push_back(I);
        continue;
      }
      return refuse("the accumulator has a consumer that is not a store or a "
                    "single accumulating fadd (fused epilogue: not a "
                    "dispatchable GEMM)");
    }
  }
  if (!out.store)
    return refuse("no store consumes the accumulator");
  return out;
}

/// The library call writes every element of the product, so the output store
/// may be gated only by the bounds of the index sources (`indexBranches`) and
/// by the exits of the contraction's own loops.
static Error
checkOutputGuards(Function &F, DominatorTree &DT, ArrayRef<ChainMember> chain,
                  const SmallPtrSetImpl<const Instruction *> &indexBranches,
                  const StoreInst *store) {
  SmallPtrSet<const Instruction *, 8> loopExits;
  for (const ChainMember &m : chain)
    loopExits.insert(m.L->getLoopLatch()->getTerminator());
  const BasicBlock *storeBB = store->getParent();
  for (BasicBlock &BB : F) {
    const Instruction *T = BB.getTerminator();
    if (T->getNumSuccessors() < 2 || indexBranches.contains(T) ||
        loopExits.contains(T))
      continue;
    if (llvm::any_of(successors(&BB), [&](const BasicBlock *S) {
          return DT.dominates(BasicBlockEdge(&BB, S), storeBB);
        }))
      return refuse("the output store is gated by a condition that is not a "
                    "bound of the product's indices");
  }
  return Error::success();
}

/// beta of C = A*B + beta*C: 1 when the epilogue adds the output element
/// itself, 0 without an epilogue.
static Expected<double> betaOf(ArrayRef<ChainMember> chain,
                               const OutputStore &out) {
  if (!out.epilogue)
    return 0.0;
  Value *other = out.epilogue->getOperand(0);
  if (reachesThroughPhis(other, chain.back().fma))
    other = out.epilogue->getOperand(1);
  auto *ld = dyn_cast<LoadInst>(other);
  if (!ld || ld->getPointerOperand() != out.store->getPointerOperand())
    return refuse("the epilogue fadd does not add the output element itself "
                  "(C = alpha*A*B + beta*C with beta != 1 is not "
                  "representable)");
  return 1.0;
}

/// The output address split over the index sources.
static Expected<AddressSplit> outputAddress(ChainAnalysis &C,
                                            StoreInst *store) {
  Expected<Polynomial> cPoly =
      C.expand(C.SE.getSCEV(store->getPointerOperand()));
  if (!cPoly)
    return refuse("cannot expand the output address (" +
                  toString(cPoly.takeError()) + ")");
  for (unsigned counter : C.redCounter)
    if (cPoly->contains(counter))
      return refuse("the output address moves with the reduction index");
  std::optional<AddressSplit> split = splitAll(*cPoly, C.indices);
  if (!split)
    return refuse("the output address is not affine in the recognized index "
                  "sources");
  return std::move(*split);
}

/// Fuses a side's (one or two) index sources into one matrix index: the slow
/// index's stride must be exactly the fast one's times the fast extent, which
/// is what makes the pair a dense numbering of one matrix dimension.
static std::optional<FusedIndex> fuseIndices(ArrayRef<unsigned> ids,
                                             ArrayRef<Polynomial> coef,
                                             ArrayRef<IndexSource> indices) {
  if (ids.size() == 1)
    return FusedIndex{coef[ids[0]], indices[ids[0]].extent, {ids[0]}};
  for (unsigned f = 0; f < 2; ++f) {
    unsigned fast = ids[f], slow = ids[1 - f];
    if (coef[fast] * indices[fast].extent == coef[slow])
      return FusedIndex{coef[fast],
                        indices[fast].extent * indices[slow].extent,
                        {fast, slow}};
  }
  return std::nullopt;
}

/// Partitions the index sources by the operand they move, fuses each side into
/// one matrix index, and orients the product so that the output is
/// column-major.
static Expected<Orientation>
orient(ArrayRef<IndexSource> indices, const AddressSplit &op0,
       const AddressSplit &op1, const AddressSplit &out,
       const OperandAddresses &ops, int64_t elemSize) {
  // Which group is "M" is not decided here: both assignments describe the
  // same product (one is the transpose of the other).
  SmallVector<unsigned, 2> g0, g1;
  for (unsigned i = 0; i < indices.size(); ++i) {
    bool in0 = !op0.coef[i].isZero(), in1 = !op1.coef[i].isZero(),
         inC = !out.coef[i].isZero();
    if (!in0 && !in1 && !inC)
      continue;
    if (in0 && in1)
      return refuse("an index moves BOTH operands: this is not a matrix "
                    "product");
    if (!inC)
      return refuse("an index moves an operand but not the output: the "
                    "contraction is not fully reduced");
    (in0 ? g0 : g1).push_back(i);
  }
  if (g0.empty() || g1.empty())
    return refuse("could not separate the two operands' free indices");
  if (g0.size() > 2 || g1.size() > 2)
    return refuse("more than two fused indices on one side (unsupported; "
                  "refusing to guess the fusion order)");

  std::optional<FusedIndex> f0 = fuseIndices(g0, op0.coef, indices);
  std::optional<FusedIndex> f1 =
      f0 ? fuseIndices(g1, op1.coef, indices) : std::nullopt;
  if (!f0 || !f1)
    return refuse("an operand's two free indices are not a dense fused "
                  "numbering (stride/extent mismatch)");
  std::optional<FusedIndex> c0 = fuseIndices(f0->order, out.coef, indices);
  std::optional<FusedIndex> c1;
  if (c0 && c0->order == f0->order)
    c1 = fuseIndices(f1->order, out.coef, indices);
  if (!c1 || c1->order != f1->order)
    return refuse("the output does not use the same fused numbering as the "
                  "operands");

  // Canonical orientation: the index group that is contiguous in the output is
  // M, which makes the output column-major with ldc = M, the form every host
  // GEMM entry point takes, and makes the choice independent of source order.
  int64_t k0 = 0, k1 = 0;
  bool op0IsRow = c0->unitStride.isConstant(&k0) && k0 == elemSize;
  bool op1IsRow = c1->unitStride.isConstant(&k1) && k1 == elemSize;
  if (op0IsRow == op1IsRow)
    return refuse("no operand's free index is contiguous in the output (or "
                  "both are), so the output is not a dense M x Ncols matrix");
  const bool aIsOp0 = op0IsRow;
  const FusedIndex &row = aIsOp0 ? *f0 : *f1, &col = aIsOp0 ? *f1 : *f0;
  Orientation O{aIsOp0,         row.unitStride, row.extent,
                col.unitStride, col.extent,     (aIsOp0 ? c1 : c0)->unitStride,
                row.order,      col.order,      false};
  const int64_t aStep = ops.step[aIsOp0 ? 0 : 1];
  const int64_t bStep = ops.step[aIsOp0 ? 1 : 0];

  // A must be K-contiguous: the host runtimes read A as an M x K row-major
  // block (aColMajor == 0). If the other operand were the K-contiguous one the
  // product would have to be emitted transposed, which is not implemented.
  if (aStep != elemSize)
    return refuse("the operand carrying the output's contiguous index is not "
                  "contiguous along the reduction axis; emitting this product "
                  "would require the transposed orientation, which is not "
                  "implemented");
  O.bColMajor = (bStep == elemSize);
  if (!O.bColMajor) {
    // K-strided B: the reduction step is ldb and the column step must be one
    // element.
    int64_t cs = 0;
    if (!O.ldb.isConstant(&cs) || cs != elemSize)
      return refuse("B is K-strided but its column step is not one element");
    O.ldb = Polynomial::constant(bStep);
  }
  return O;
}

/// Fills the extents and leading dimensions of `note`, in elements.
static Error runtimeDims(const ChainAnalysis &C, const Orientation &O,
                         const Polynomial &kPoly, int64_t elemSize,
                         GemmBodyNote &note) {
  std::optional<Polynomial> lda = divideExactly(O.lda, elemSize);
  std::optional<Polynomial> ldb =
      lda ? divideExactly(O.ldb, elemSize) : std::nullopt;
  std::optional<Polynomial> ldc =
      ldb ? divideExactly(O.ldc, elemSize) : std::nullopt;
  if (!ldc)
    return refuse("a leading dimension is not a whole number of elements");
  const Polynomial *dims[kNumGemmDims];
  dims[DimM] = &O.rows;
  dims[DimNcols] = &O.cols;
  dims[DimK] = &kPoly;
  dims[DimLda] = &*lda;
  dims[DimLdb] = &*ldb;
  dims[DimLdc] = &*ldc;
  for (unsigned i = 0; i < kNumGemmDims; ++i) {
    Expected<RuntimeDim> d = runtimeDimOf(*dims[i], C.atoms);
    if (!d)
      return d.takeError();
    note.dims[i] = std::move(*d);
  }
  return Error::success();
}

/// The launch value an operand's base polynomial is, when it is one pointer
/// that roots in a parameter of the optimized body.
static std::optional<LaunchRef> basePointer(const Polynomial &b,
                                            const AtomTable &A) {
  if (b.t.size() != 1)
    return std::nullopt;
  auto &kv = *b.t.begin();
  if (kv.first.size() != 1 || kv.second != 1)
    return std::nullopt;
  const SCEV *S = A.byId[kv.first[0]];
  auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
  if (!U || !U->getType()->isPointerTy())
    return std::nullopt;
  LaunchRef ref = tracePointerRoot(U->getValue());
  if (ref.param < 0)
    return std::nullopt;
  return ref;
}

/// The product's shape at profile scale. Column extents must be fixed by the
/// launch geometry (a CTA span, or a compile-time bound); the row extent then
/// follows exactly from the profiled execution counts. A missing count refuses
/// the site.
static Expected<ProfileShape>
profileShape(const ChainAnalysis &C, const SiteProfileHeader &hdr,
             const ProfileMap &prof, const Orientation &O,
             const Instruction *epilogue, const Polynomial &trip,
             const Polynomial &kPoly) {
  ProfileShape P;
  P.ctaPerLaunch = std::max(1u, hdr.maxGridDim[0]) *
                   std::max(1u, hdr.maxGridDim[1]) *
                   std::max(1u, hdr.maxGridDim[2]);
  if (!hdr.launchCount || hdr.launchCount % P.ctaPerLaunch)
    return refuse("profile header launch count is not a whole number of "
                  "grids");
  P.launches = (unsigned)(hdr.launchCount / P.ctaPerLaunch);

  uint64_t profNcols = 1;
  for (unsigned i : O.colOrder) {
    std::optional<unsigned> e = profileExtent(C.indices[i], C.atoms, hdr);
    if (!e || *e == 0)
      return refuse("the profile does not fix the extent of a column index "
                    "(launch geometry gives no bound for " +
                    C.indices[i].name + ")");
    profNcols *= *e;
  }

  for (const ChainMember &m : C.chain) {
    std::optional<size_t> slot = slotOf(m.fma);
    if (!slot)
      return refuse("a chain member's fma carries no profile index");
    auto it = prof.find(*slot);
    if (it == prof.end() || it->second.exec == 0)
      return refuse("a chain member's fma has no profiled execution count");
    P.macs += it->second.exec;
  }

  uint64_t rowsTimesCols = 0;
  if (epilogue) {
    auto it = prof.end();
    if (std::optional<size_t> slot = slotOf(epilogue))
      it = prof.find(*slot);
    if (it == prof.end() || it->second.exec == 0)
      return refuse("the accumulating epilogue has no profiled execution "
                    "count, and it is the only exact source for the row "
                    "extent");
    const uint64_t profEpilogue = it->second.exec;
    if (profEpilogue % P.launches)
      return refuse("profiled epilogue count is not a whole number of "
                    "launches");
    rowsTimesCols = profEpilogue / P.launches;
  } else {
    // beta == 0: no epilogue to count, so K must be a compile-time trip count.
    int64_t tripConst = 0;
    if (!trip.isConstant(&tripConst) || tripConst <= 0)
      return refuse("no accumulating epilogue and a runtime reduction length: "
                    "the profile cannot fix M and K separately");
    uint64_t k = (uint64_t)C.chain.size() * (uint64_t)tripConst;
    if (P.macs % ((uint64_t)P.launches * k))
      return refuse("profiled MAC count is not divisible by launches*K");
    rowsTimesCols = P.macs / ((uint64_t)P.launches * k);
  }
  if (rowsTimesCols == 0 || rowsTimesCols % profNcols)
    return refuse("profiled output-element count " +
                  std::to_string(rowsTimesCols) +
                  " is not a multiple of the reconstructed column count " +
                  std::to_string(profNcols));
  P.N = (unsigned)profNcols;
  P.M = (unsigned)(rowsTimesCols / profNcols);
  uint64_t denom = (uint64_t)P.launches * rowsTimesCols;
  if (!denom || P.macs % denom)
    return refuse("profiled MAC count is not divisible by launches*M*Ncols");
  P.K = (unsigned)(P.macs / denom);
  if (!P.M || !P.N || !P.K)
    return refuse("a reconstructed profile-scale extent is zero");

  // Per-CTA tile: drop the index factors that are spread across blocks.
  auto hasCta = [&](const IndexSource &ix) {
    for (auto &kv : ix.form.t)
      for (unsigned a : kv.first) {
        const SCEV *S = C.atoms.byId[a];
        auto *U = S ? dyn_cast<SCEVUnknown>(S) : nullptr;
        if (U && blockIndexAxis(U->getValue()) >= 0)
          return true;
      }
    return false;
  };
  uint64_t colSpread = 1, rowSpread = 1;
  for (unsigned i : O.colOrder)
    if (hasCta(C.indices[i])) {
      std::optional<unsigned> e = profileExtent(C.indices[i], C.atoms, hdr);
      if (!e || !*e)
        return refuse("cannot size the per-CTA column tile");
      colSpread *= *e;
    }
  for (unsigned i : O.rowOrder)
    if (hasCta(C.indices[i])) {
      std::optional<unsigned> e = profileExtent(C.indices[i], C.atoms, hdr);
      if (!e || !*e)
        return refuse("cannot size the per-CTA row tile");
      rowSpread *= *e;
    }
  if (!colSpread || !rowSpread || P.N % colSpread || P.M % rowSpread)
    return refuse("the product does not divide evenly into per-CTA tiles");
  P.tileN = (unsigned)(P.N / colSpread);
  P.tileM = (unsigned)(P.M / rowSpread);
  if (!P.tileM || !P.tileN)
    return refuse("a per-CTA tile extent is zero");
  // Cross-check the symbolic K against the profile when K is compile-time.
  int64_t kc = 0;
  if (kPoly.isConstant(&kc) && kc > 0 && (unsigned)kc != P.K)
    return refuse("compile-time K (" + std::to_string(kc) +
                  ") disagrees with the profiled reduction length (" +
                  std::to_string(P.K) + ")");
  return P;
}

/// Reconstructs one contraction chain into a library-call handle and its
/// profile-scale shape. Nothing is defaulted: every field is either derived or
/// the chain is refused, with the reason as the error.
static Expected<RecognizedChain> analyzeChain(Function &F, ScalarEvolution &SE,
                                              DominatorTree &DT,
                                              const SiteProfileHeader &hdr,
                                              const ProfileMap &prof,
                                              ArrayRef<ChainMember> chain) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  Type *eltTy = chain[0].aLoad->getType();
  const int64_t elemSize = (int64_t)DL.getTypeAllocSize(eltTy).getFixedValue();
  for (const ChainMember &m : chain)
    if (m.aLoad->getType() != eltTy || m.bLoad->getType() != eltTy)
      return refuse("chain members disagree on element type");

  ChainAnalysis C(SE, chain);
  RecognizedChain out;
  RuntimeGemmHandle &H = out.H;
  GemmBodyNote &note = H.library;
  Expected<SmallPtrSet<const Instruction *, 8>> indexBranches =
      collectIndexSources(C, F, DT);
  if (!indexBranches)
    return indexBranches.takeError();
  Expected<OperandAddresses> ops = operandAddresses(C);
  if (!ops)
    return ops.takeError();
  Expected<Polynomial> trip = reductionLength(C);
  if (!trip)
    return trip.takeError();
  if (Error E = flattenMembers(C, *ops, *trip, note.assumedEqual))
    return std::move(E);
  std::optional<AddressSplit> op0 = splitAll(ops->at[0][0], C.indices);
  std::optional<AddressSplit> op1 =
      op0 ? splitAll(ops->at[1][0], C.indices) : std::nullopt;
  if (!op1)
    return refuse("an operand address is not affine in the recognized index "
                  "sources");

  Expected<OutputStore> store = findOutputStore(chain);
  if (!store)
    return store.takeError();
  if (Error E = checkOutputGuards(F, DT, chain, *indexBranches, store->store))
    return std::move(E);
  Expected<double> beta = betaOf(chain, *store);
  if (!beta)
    return beta.takeError();
  note.beta = *beta;
  Expected<AddressSplit> cSplit = outputAddress(C, store->store);
  if (!cSplit)
    return cSplit.takeError();

  Expected<Orientation> O =
      orient(C.indices, *op0, *op1, *cSplit, *ops, elemSize);
  if (!O)
    return O.takeError();
  note.aColMajor = false;
  note.cColMajor = true; // by construction of the orientation
  note.bColMajor = O->bColMajor;
  const Polynomial kPoly = Polynomial::constant((int64_t)chain.size()) * *trip;
  if (Error E = runtimeDims(C, *O, kPoly, elemSize, note))
    return std::move(E);

  std::optional<LaunchRef> aRef =
      basePointer(O->aIsOp0 ? op0->base : op1->base, C.atoms);
  std::optional<LaunchRef> bRef =
      basePointer(O->aIsOp0 ? op1->base : op0->base, C.atoms);
  std::optional<LaunchRef> cRef = basePointer(cSplit->base, C.atoms);
  if (!aRef || !bRef || !cRef)
    return refuse("an operand does not root in a parameter of the optimized "
                  "body");
  note.aParam = aRef->param;
  note.aOffset = aRef->offset;
  note.bParam = bRef->param;
  note.bOffset = bRef->offset;
  note.cParam = cRef->param;
  note.cOffset = cRef->offset;
  // A recognized nest is the whole body: the caller refuses a body that
  // stores anything else into its arguments.
  note.standalone = true;
  H.cStore = store->store;
  H.epilogue = store->epilogue;
  for (const ChainMember &m : chain) {
    H.fmas.push_back(m.fma);
    H.fmuls.push_back(m.fmul);
    H.aLoads.push_back(O->aIsOp0 ? m.aLoad : m.bLoad);
  }

  Expected<ProfileShape> P =
      profileShape(C, hdr, prof, *O, store->epilogue, *trip, kPoly);
  if (!P)
    return P.takeError();
  out.profile = *P;
  return std::move(out);
}

/// One reduction loop of a contraction chain: a closed fma/phi cycle over two
/// loads whose addresses are affine in the loop with constant byte steps. The
/// trip count may be a runtime value.
static Expected<ChainMember> matchChainMember(Loop *L, PHINode &phi,
                                              ScalarEvolution &SE,
                                              BasicBlock *latch) {
  Expected<ReductionCycle> rc = matchReductionCycle(L, phi, latch);
  if (!rc)
    return rc.takeError();
  if (std::optional<StringRef> shape = reductionLoopShapeRefusal(*L))
    return refuse(*shape);
  const SCEV *btc = SE.getBackedgeTakenCount(L);
  if (isa<SCEVCouldNotCompute>(btc))
    return refuse("backedge-taken count is not computable");
  ChainMember out;
  out.L = L;
  out.phi = &phi;
  out.fma = rc->fma;
  out.fmul = rc->fmul;
  out.aLoad = rc->aLoad;
  out.bLoad = rc->bLoad;
  out.aAddr = SE.getSCEV(rc->aLoad->getPointerOperand());
  out.bAddr = SE.getSCEV(rc->bLoad->getPointerOperand());
  out.trip = SE.getAddExpr(btc, SE.getOne(btc->getType()));
  out.init = phi.getIncomingValueForBlock(L->getLoopPreheader());
  return out;
}

void findRuntimeShapeMatmuls(const Options &O, Function &F, ScalarEvolution &SE,
                             LoopInfo &LI,
                             const SiteProfileHeader &profileHeader,
                             const ProfileMap &prof,
                             SmallVectorImpl<RecognizedMatmul> &out) {
  assert(Target::of(F).isDevice() && "a runtime-shape product is device code");
  if (profileHeader.launchCount == 0) {
    if (O.Diagnostics.Print)
      llvm::errs()
          << "[poseidon] " << F.getName()
          << ": no profile header (launchCount=0); a runtime-shape "
             "product cannot be priced without one, so it is skipped\n";
    return;
  }

  // Loops a scalar-loop matmul in `out` already claims are left to it.
  SmallPtrSet<const Instruction *, 8> claimed;
  for (const RecognizedMatmul &am : out)
    if (am.origin == RecognizedMatmul::Origin::ConstantShape &&
        am.scalarLoop.fma)
      claimed.insert(am.scalarLoop.fma);

  // A library dispatch replaces the whole launch, so the body may store into
  // its arguments only the product.
  unsigned argStores = 0;
  for (Instruction &I : instructions(F))
    if (auto *St = dyn_cast<StoreInst>(&I))
      if (tracePointerRoot(St->getPointerOperand()).param >= 0)
        ++argStores;

  DominatorTree DT(F);

  SmallVector<ChainMember, 8> members;
  forEachInnermostLoop(LI, [&](Loop *L) {
    BasicBlock *latch = L->getLoopLatch();
    if (!latch)
      return;
    for (PHINode &phi : L->getHeader()->phis()) {
      if (!phi.getType()->isFloatingPointTy())
        continue;
      Expected<ChainMember> m = matchChainMember(L, phi, SE, latch);
      if (m) {
        members.push_back(*m);
        continue;
      }
      std::string why = toString(m.takeError());
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon]   reject phi in "
                     << L->getHeader()->getName() << ": " << why << '\n';
    }
  });
  if (O.Diagnostics.Print)
    llvm::errs() << "[poseidon] " << F.getName() << ": " << members.size()
                 << " candidate reduction loop(s)\n";
  if (members.empty())
    return;

  // Chain the members. Member j's initial accumulator reaches the fma of every
  // earlier member (the guard phis the unroller leaves make the whole prefix
  // reachable), so the immediate predecessor is the reachable member with the
  // longest prefix of its own; requiring that prefix to be exactly one shorter
  // is what makes the order a total chain rather than a guess.
  const unsigned n = members.size();
  SmallVector<SmallVector<unsigned, 4>, 8> reach(n);
  for (unsigned j = 0; j < n; ++j)
    for (unsigned i = 0; i < n; ++i)
      if (i != j && reachesThroughPhis(members[j].init, members[i].fma))
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
      if (O.Diagnostics.Print && prevOf[h] < 0)
        llvm::errs() << "[poseidon]   loop " << h
                     << " is not a chain head: ambiguous accumulator "
                        "dataflow\n";
      continue;
    }
    if (!isPositiveZeroSeed(members[h].init)) {
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon]   loop " << h
                     << " is a chain head whose accumulator does not start at "
                        "+0.0; not a fresh contraction\n";
      continue;
    }
    SmallVector<ChainMember, 8> chain;
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
    if (llvm::any_of(chain, [&](const ChainMember &m) {
          return claimed.contains(m.fma);
        }))
      continue;

    Expected<RecognizedChain> R =
        analyzeChain(F, SE, DT, profileHeader, prof, chain);
    if (R && argStores != 1)
      R = refuse("the body performs " + std::to_string(argStores) +
                 " argument-rooted stores, and only a pure GEMM body can be "
                 "replaced by a library dispatch");
    if (!R) {
      std::string why = toString(R.takeError());
      if (O.Diagnostics.Print)
        llvm::errs() << "[poseidon] " << F.getName() << ": reduction chain of "
                     << chain.size()
                     << " loop(s) is not a dispatchable runtime-shape product: "
                     << why << '\n';
      continue;
    }
    const ProfileShape &P = R->profile;

    RecognizedMatmul am;
    am.id = nextId++;
    // M/N are the per-CTA tile the accuracy model simulates; the full product
    // is globalM/globalN/globalK.
    am.M = P.tileM;
    am.N = P.tileN;
    am.K = P.K;
    am.globalM = P.M;
    am.globalN = P.N;
    am.globalK = P.K;
    am.gridCTAs = P.ctaPerLaunch;
    am.elemType = fpKindFromType(chain[0].aLoad->getType());
    am.origin = RecognizedMatmul::Origin::RuntimeShape;
    am.runtimeGemm = std::make_shared<RuntimeGemmHandle>(std::move(R->H));
    am.outputValue = chain.back().fma;
    for (const ChainMember &m : chain)
      for (BasicBlock *BB : m.L->blocks())
        for (Instruction &I : *BB)
          am.footprint.insert(&I);
    if (am.runtimeGemm->epilogue)
      am.footprint.insert(am.runtimeGemm->epilogue);

    if (O.Diagnostics.Print) {
      const GemmBodyNote &g = am.runtimeGemm->library;
      llvm::errs() << "[poseidon] " << F.getName() << ": recognized GEMM from "
                   << chain.size() << " reduction loop(s)\n"
                   << "[poseidon]   C = A^T*B + beta*C  beta=" << g.beta << '\n'
                   << "[poseidon]   M=" << runtimeDimString(g.dims[DimM])
                   << "  Ncols=" << runtimeDimString(g.dims[DimNcols])
                   << "  K=" << runtimeDimString(g.dims[DimK]) << '\n'
                   << "[poseidon]   lda=" << runtimeDimString(g.dims[DimLda])
                   << "  ldb=" << runtimeDimString(g.dims[DimLdb])
                   << "  ldc=" << runtimeDimString(g.dims[DimLdc]) << '\n'
                   << "[poseidon]   aColMajor=" << (g.aColMajor ? 1 : 0)
                   << " bColMajor=" << (g.bColMajor ? 1 : 0)
                   << " cColMajor=" << (g.cColMajor ? 1 : 0)
                   << "  C=" << launchRefString({g.cParam, g.cOffset}, "param")
                   << " A=" << launchRefString({g.aParam, g.aOffset}, "param")
                   << " B=" << launchRefString({g.bParam, g.bOffset}, "param")
                   << '\n';
      for (const auto &[x, y] : g.assumedEqual)
        llvm::errs() << "[poseidon]   dense only if "
                     << launchRefString(x, "param")
                     << " == " << launchRefString(y, "param")
                     << " (checked at the launch)\n";
      llvm::errs() << "[poseidon]   profile shape M=" << P.M << " Ncols=" << P.N
                   << " K=" << P.K << " (per-CTA tile " << P.tileM << 'x'
                   << P.tileN << ", launches=" << P.launches
                   << " CTAs/launch=" << P.ctaPerLaunch << " MACs=" << P.macs
                   << ")\n";
    }
    out.push_back(std::move(am));
  }
}

} // namespace poseidon
